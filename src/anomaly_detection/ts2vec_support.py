from __future__ import annotations

import sys
from pathlib import Path

import torch

from anomaly_detection.data import PROJECT_ROOT

DEFAULT_TS2VEC_PATHS = (
    PROJECT_ROOT / "vendor" / "ts2vec",
    PROJECT_ROOT / "ts2vec",
    PROJECT_ROOT.parent / "ts2vec",
)


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
            "Could not import TS2Vec. Clone https://github.com/zhihanyue/ts2vec "
            "and pass --ts2vec-dir.\n"
            f"Searched:\n  - {searched}"
        ) from exc


def resolve_torch_device(device: str | None) -> str:
    if device is not None:
        return device
    return "cuda:0" if torch.cuda.is_available() else "cpu"
