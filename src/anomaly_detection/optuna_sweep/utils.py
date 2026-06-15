from __future__ import annotations

import json
import math
import pickle
import subprocess
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

from anomaly_detection.optuna_sweep.config import PROJECT_ROOT


def set_reproducible_seed(seed: int, device: str | torch.device | None = None) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device is not None and str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2) + "\n", encoding="utf-8")


def jsonable(value: Any) -> Any:
    if isinstance(value, Path | torch.device):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def flatten_dict(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            output.update(flatten_dict(value, name))
        elif not isinstance(value, list | tuple):
            output[name] = value
    return output


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def safe_series_name(name: str) -> str:
    return name.replace("/", "__").replace("\\", "__").replace(".csv", "")


def resolved_models_dir(args: Any, output_root: Path) -> Path:
    if args.models_dir is not None:
        return args.models_dir.expanduser().resolve()
    return output_root / "models" / args.model


def model_payload_path(models_dir: Path, series_name: str, suffix: str = ".pkl") -> Path:
    return models_dir / f"{safe_series_name(series_name)}{suffix}"


def global_model_path(models_dir: Path, suffix: str = ".pkl") -> Path:
    return models_dir / f"global_model{suffix}"


def ts2vec_checkpoint_path(models_dir: Path, series_name: str | None = None) -> Path:
    if series_name is not None:
        return model_payload_path(models_dir, series_name, suffix=".pt")
    direct = models_dir / "ts2vec_model.pt"
    if direct.is_file():
        return direct
    nested = models_dir / "artifacts" / "ts2vec_model.pt"
    if nested.is_file():
        return nested
    return direct


def save_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def parameter_count_from_model(model: Any) -> int:
    network = (
        getattr(model, "model_", None)
        or getattr(model, "_net", None)
        or getattr(model, "net", None)
    )
    if network is None:
        return 0
    if hasattr(network, "module"):
        network = network.module
    if hasattr(network, "parameters"):
        return int(sum(parameter.numel() for parameter in network.parameters()))
    return 0


def cuda_device_index(device: str | torch.device | None) -> int | None:
    if device is None or not torch.cuda.is_available():
        return None
    try:
        torch_device = torch.device(device)
    except (RuntimeError, TypeError, ValueError) as exc:
        warnings.warn(
            f"Invalid torch device {device!r}; CUDA metrics disabled: {exc}", stacklevel=2
        )
        return None
    if torch_device.type != "cuda":
        return None
    device_count = torch.cuda.device_count()
    if device_count <= 0:
        return None
    index = 0 if torch_device.index is None else int(torch_device.index)
    if index < 0 or index >= device_count:
        warnings.warn(
            f"CUDA device cuda:{index} is unavailable; device_count={device_count}. "
            "CUDA metrics disabled.",
            stacklevel=2,
        )
        return None
    return index


def reset_cuda_peak_memory_stats(device: str | torch.device | None) -> None:
    index = cuda_device_index(device)
    if index is None:
        return
    try:
        torch.cuda.set_device(index)
        torch.empty(1, device=f"cuda:{index}")
        torch.cuda.synchronize(index)
        torch.cuda.reset_peak_memory_stats()
    except Exception as exc:
        warnings.warn(
            f"Could not reset CUDA peak memory stats for device={device!r} "
            f"(resolved cuda:{index}): {exc}",
            stacklevel=2,
        )


def peak_gpu_memory(device: str | torch.device | None) -> int | None:
    index = cuda_device_index(device)
    if index is None:
        return None
    try:
        torch.cuda.set_device(index)
        torch.cuda.synchronize(index)
        return int(torch.cuda.max_memory_allocated())
    except Exception as exc:
        warnings.warn(
            f"Could not read CUDA peak memory for device={device!r} (resolved cuda:{index}): {exc}",
            stacklevel=2,
        )
        return None
