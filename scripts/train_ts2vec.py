#!/usr/bin/env python3
"""Train TS2Vec representations on the NAB time-series dataset.

Changes vs the original script:
- safer default training prefix: 15% instead of the full series;
- normalization statistics are explicitly stored per series and should be reused at eval time;
- metadata includes enough information to build a normal-prefix reference set downstream;
- optional warning when training on the full series, because this can expose TS2Vec to anomalies.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from anomaly_detection.data import DATASET_DIR, load_dataset  # noqa: E402

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "training" / "ts2vec"
DEFAULT_TRAIN_FRACTION = 0.15
DEFAULT_MAX_TRAIN_LENGTH = 512
OOM_HINT_THRESHOLD_BYTES = 4 * 1024**3
DEFAULT_TS2VEC_PATHS = (
    PROJECT_ROOT / "src" / "anomaly_detection" / "vendor" / "ts2vec",
    PROJECT_ROOT / "vendor" / "ts2vec",
    PROJECT_ROOT / "ts2vec",
    PROJECT_ROOT.parent / "ts2vec",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the official zhihanyue/ts2vec implementation on NAB series loaded "
            "through anomaly_detection.data. By default, only the initial prefix of each "
            "series is used for unsupervised training, which is usually safer for NAB-style "
            "anomaly detection than training on the full series."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DATASET_DIR,
        help=f"NAB dataset directory. Defaults to {DATASET_DIR}.",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Optional NAB category names to train on, e.g. realKnownCause realAWSCloudwatch.",
    )
    parser.add_argument(
        "--limit-series",
        type=int,
        default=None,
        help="Optionally train on only the first N loaded series for smoke tests.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=DEFAULT_TRAIN_FRACTION,
        help=(
            "Prefix fraction of each series used for unsupervised training and for "
            "normalization statistics. Defaults to 0.15 to avoid fitting on the full "
            "anomalous series. Use 1.0 only intentionally."
        ),
    )
    parser.add_argument(
        "--normalization",
        choices=("per-series", "global", "none"),
        default="per-series",
        help=(
            "Value normalization before padding and training. For per-series/global, "
            "statistics are computed only on the selected training prefixes."
        ),
    )
    parser.add_argument(
        "--ts2vec-dir",
        type=Path,
        default=None,
        help=(
            "Path to a local checkout of https://github.com/zhihanyue/ts2vec. "
            "If omitted, common project-local locations are searched."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for artifacts.")
    parser.add_argument("--output-dims", type=int, default=320, help="TS2Vec representation size.")
    parser.add_argument("--hidden-dims", type=int, default=64, help="TS2Vec encoder hidden size.")
    parser.add_argument("--depth", type=int, default=10, help="Number of encoder residual blocks.")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs to train.")
    parser.add_argument("--iters", type=int, default=2000, help="Number of training iterations.")
    parser.add_argument(
        "--max-train-length",
        type=int,
        default=DEFAULT_MAX_TRAIN_LENGTH,
        help=(
            "Maximum sequence length seen by the TS2Vec loss. Longer series are split/cropped "
            "by the official trainer. Lower this if CUDA memory is tight; use 0 to disable. "
            f"Defaults to {DEFAULT_MAX_TRAIN_LENGTH}."
        ),
    )
    parser.add_argument("--temporal-unit", type=int, default=0, help="TS2Vec temporal unit.")
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device. Defaults to cuda:0 when available, otherwise cpu.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--save-representations",
        action="store_true",
        help="Also encode and save full training-prefix representations for all training series.",
    )
    parser.add_argument(
        "--encoding-window",
        default="full_series",
        help="Encoding window passed to TS2Vec when --save-representations is set.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip the conservative CUDA memory preflight check.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print TS2Vec epoch losses.")
    return parser.parse_args()


def import_ts2vec(ts2vec_dir: Path | None) -> tuple[type, Path | None]:
    candidates = [ts2vec_dir.expanduser().resolve()] if ts2vec_dir is not None else []
    candidates.extend(path for path in DEFAULT_TS2VEC_PATHS if path not in candidates)

    for candidate in candidates:
        if (candidate / "ts2vec.py").is_file():
            sys.path.insert(0, str(candidate))
            from ts2vec import TS2Vec

            return TS2Vec, candidate

    try:
        from ts2vec import TS2Vec

        return TS2Vec, None
    except ImportError as exc:
        searched = "\n  - ".join(str(path) for path in candidates)
        raise SystemExit(
            "Could not import the official TS2Vec implementation.\n"
            "Clone it and pass the path, for example:\n"
            "  git clone https://github.com/zhihanyue/ts2vec.git vendor/ts2vec\n"
            "  uv run scripts/train_ts2vec.py --ts2vec-dir vendor/ts2vec\n\n"
            f"Searched:\n  - {searched}"
        ) from exc


def resolve_device(device: str | None) -> str:
    if device is not None:
        return device
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def validate_args(args: argparse.Namespace) -> None:
    if args.train_fraction <= 0 or args.train_fraction > 1:
        raise SystemExit("--train-fraction must be in the interval (0, 1].")
    if args.limit_series is not None and args.limit_series <= 0:
        raise SystemExit("--limit-series must be positive.")
    if args.epochs is None and args.iters is None:
        raise SystemExit("At least one of --epochs or --iters must be set.")
    if args.max_train_length is not None and args.max_train_length < 0:
        raise SystemExit("--max-train-length must be non-negative.")


def normalize_max_train_length(value: int | None) -> int | None:
    if value == 0:
        return None
    return value


def make_output_dir(output_dir: Path | None) -> Path:
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = DEFAULT_OUTPUT_ROOT / timestamp
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def compute_prefix_length(series_length: int, train_fraction: float) -> int:
    return max(2, int(np.ceil(series_length * train_fraction)))


def series_to_arrays(
    dataset: dict[str, Any],
    train_fraction: float,
    normalization: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    names = list(dataset)
    values: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []

    for name in names:
        frame = dataset[name]
        full_series = frame["value"].to_numpy(dtype=np.float32)
        train_len = compute_prefix_length(len(full_series), train_fraction)
        series = full_series[:train_len]
        if len(series) < 2:
            continue
        values.append(series)
        metadata.append(
            {
                "name": name,
                "original_length": int(len(frame)),
                "train_length": int(len(series)),
                "train_fraction": float(train_fraction),
                "start_timestamp": frame["timestamp"].iloc[0].isoformat(),
                "end_timestamp": frame["timestamp"].iloc[len(series) - 1].isoformat(),
            }
        )

    if not values:
        raise SystemExit("No trainable series were loaded.")

    if normalization == "global":
        joined = np.concatenate(values)
        mean = float(joined.mean())
        std = float(joined.std())
        std = 1.0 if std == 0 else std
        values = [(series - mean) / std for series in values]
        for item in metadata:
            item["normalization"] = {"mean": mean, "std": std, "scope": "global_train_prefix"}
    elif normalization == "per-series":
        normalized = []
        for series, item in zip(values, metadata, strict=True):
            mean = float(series.mean())
            std = float(series.std())
            std = 1.0 if std == 0 else std
            normalized.append((series - mean) / std)
            item["normalization"] = {"mean": mean, "std": std, "scope": "series_train_prefix"}
        values = normalized
    else:
        for item in metadata:
            item["normalization"] = {"scope": "none"}

    max_len = max(len(series) for series in values)
    data = np.full((len(values), max_len, 1), np.nan, dtype=np.float32)
    for index, series in enumerate(values):
        data[index, : len(series), 0] = series

    return data, metadata


def estimate_temporal_loss_bytes(batch_size: int, sequence_length: int) -> int:
    # TS2Vec temporal contrast computes sim = matmul(z, z.transpose(1, 2)), shaped B x 2T x 2T.
    # This lower-bound estimate only counts that float32 similarity matrix;
    # activations and gradients add more.
    return batch_size * (2 * sequence_length) ** 2 * np.dtype(np.float32).itemsize


def format_gib(num_bytes: int | float) -> str:
    return f"{num_bytes / 1024**3:.2f} GiB"


def preflight_memory_check(
    train_data: np.ndarray,
    batch_size: int,
    max_train_length: int | None,
    device: str,
    force: bool,
) -> None:
    effective_length = train_data.shape[1]
    if max_train_length is not None:
        effective_length = min(effective_length, max_train_length)

    estimated_bytes = estimate_temporal_loss_bytes(batch_size, effective_length)
    print(
        "TS2Vec loss preflight: "
        f"batch_size={batch_size}, effective_length={effective_length}, "
        f"similarity_matrix~{format_gib(estimated_bytes)}."
    )

    if not device.startswith("cuda") or force:
        return

    free_bytes = None
    try:
        free_bytes = torch.cuda.mem_get_info(torch.device(device))[0]
    except (RuntimeError, ValueError):
        pass

    if free_bytes is not None and estimated_bytes > free_bytes * 0.5:
        raise SystemExit(
            "The requested TS2Vec training shape is likely to run out of CUDA memory. "
            "The temporal contrast similarity matrix alone is about "
            f"{format_gib(estimated_bytes)}, while CUDA reports "
            f"{format_gib(free_bytes)} free. Try one or more of:\n"
            "  --max-train-length 256\n"
            "  --batch-size 4\n"
            "  --output-dims 160\n"
            "  --hidden-dims 32\n"
            "Use --force to skip this check."
        )

    if free_bytes is None and estimated_bytes > OOM_HINT_THRESHOLD_BYTES:
        raise SystemExit(
            "The requested TS2Vec training shape may be too large for CUDA. "
            "The temporal contrast similarity matrix alone is about "
            f"{format_gib(estimated_bytes)}. Try --max-train-length 256 "
            "and/or --batch-size 4, or use --force to skip this check."
        )


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def jsonable_args(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    values = vars(args).copy()
    for key, value in list(values.items()):
        if isinstance(value, Path):
            values[key] = str(value)
    values["output_dir"] = str(output_dir)
    return values


def main() -> None:
    args = parse_args()
    validate_args(args)

    if args.train_fraction >= 0.999:
        print(
            "WARNING: --train-fraction is 1.0, so TS2Vec will train on the full series, "
            "including any anomalies present in NAB. This is usually not recommended for AD."
        )

    device = resolve_device(args.device)
    max_train_length = normalize_max_train_length(args.max_train_length)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)

    TS2Vec, ts2vec_source = import_ts2vec(args.ts2vec_dir)
    output_dir = make_output_dir(args.output_dir)

    dataset = load_dataset(args.dataset_dir, args.categories)
    if args.limit_series is not None:
        dataset = dict(list(dataset.items())[: args.limit_series])
    train_data, series_metadata = series_to_arrays(
        dataset,
        train_fraction=args.train_fraction,
        normalization=args.normalization,
    )

    print(f"Loaded {train_data.shape[0]} series with padded training-prefix shape {train_data.shape}.")
    preflight_memory_check(
        train_data=train_data,
        batch_size=args.batch_size,
        max_train_length=max_train_length,
        device=device,
        force=args.force,
    )
    print(f"Training TS2Vec on {device}; artifacts will be written to {output_dir}.")

    model = TS2Vec(
        input_dims=train_data.shape[-1],
        output_dims=args.output_dims,
        hidden_dims=args.hidden_dims,
        depth=args.depth,
        device=device,
        lr=args.lr,
        batch_size=args.batch_size,
        max_train_length=max_train_length,
        temporal_unit=args.temporal_unit,
    )
    loss_log = model.fit(
        train_data,
        n_epochs=args.epochs,
        n_iters=args.iters,
        verbose=args.verbose,
    )

    model_path = output_dir / "ts2vec_model.pt"
    model.save(str(model_path))
    np.save(output_dir / "loss_log.npy", np.asarray(loss_log, dtype=np.float32))
    save_json(output_dir / "series.json", series_metadata)
    save_json(
        output_dir / "metadata.json",
        {
            "dataset_dir": str(args.dataset_dir.expanduser().resolve()),
            "categories": args.categories,
            "ts2vec_source": None if ts2vec_source is None else str(ts2vec_source),
            "device": device,
            "train_shape": list(train_data.shape),
            "normalization": args.normalization,
            "train_fraction": args.train_fraction,
            "effective_max_train_length": max_train_length,
            "model_path": str(model_path),
            "loss_log_path": str(output_dir / "loss_log.npy"),
            "loss_log": [float(loss) for loss in loss_log],
            "args": jsonable_args(args, output_dir),
        },
    )

    if args.save_representations:
        representations = model.encode(train_data, encoding_window=args.encoding_window)
        np.save(output_dir / "representations.npy", representations)
        print(f"Saved representations with shape {representations.shape}.")

    print(f"Saved model to {model_path}.")


if __name__ == "__main__":
    main()
