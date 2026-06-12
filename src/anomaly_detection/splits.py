from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SeriesSplit:
    validation: list[str]
    test: list[str]
    seed: int
    validation_fraction: float

    def to_dict(self) -> dict[str, object]:
        return {
            "validation": self.validation,
            "test": self.test,
            "seed": self.seed,
            "validation_fraction": self.validation_fraction,
        }


def split_series_names(
    names: list[str],
    validation_fraction: float,
    seed: int,
) -> SeriesSplit:
    if not names:
        raise ValueError("Cannot split an empty list of series.")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in the interval (0, 1).")

    rng = np.random.default_rng(seed)
    shuffled = list(names)
    rng.shuffle(shuffled)

    if len(shuffled) == 1:
        validation = shuffled
        test = shuffled
    else:
        n_validation = int(round(len(shuffled) * validation_fraction))
        n_validation = min(max(1, n_validation), len(shuffled) - 1)
        validation = sorted(shuffled[:n_validation])
        test = sorted(shuffled[n_validation:])

    return SeriesSplit(
        validation=validation,
        test=test,
        seed=seed,
        validation_fraction=validation_fraction,
    )


def save_split(path: Path, split: SeriesSplit) -> None:
    path.write_text(json.dumps(split.to_dict(), indent=2) + "\n", encoding="utf-8")
