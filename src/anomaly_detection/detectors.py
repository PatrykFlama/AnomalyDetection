from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from torch import nn

from anomaly_detection.preprocessing import (
    EPS,
    ScaleStats,
    apply_scale,
    finite_array,
    point_features,
    prediction_errors_to_points,
    robust_scale_stats,
    rolling_mad,
    shifted_rolling,
    sliding_windows,
    window_scores_to_points,
)


class Detector:
    name: str

    def fit(self, values: np.ndarray, train_end: int) -> Detector:
        raise NotImplementedError

    def score(self, values: np.ndarray) -> np.ndarray:
        raise NotImplementedError


@dataclass
class RobustZScoreDetector(Detector):
    name: str = "robust-zscore"

    def fit(self, values: np.ndarray, train_end: int) -> RobustZScoreDetector:
        self.stats_ = robust_scale_stats(values[:train_end])
        return self

    def score(self, values: np.ndarray) -> np.ndarray:
        scaled = apply_scale(values, self.stats_)
        return finite_array(np.abs(scaled))


@dataclass
class RollingResidualDetector(Detector):
    window: int = 64
    name: str = "rolling-residual"

    def fit(self, values: np.ndarray, train_end: int) -> RollingResidualDetector:
        self.fallback_stats_ = robust_scale_stats(values[:train_end])
        return self

    def score(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        baseline = shifted_rolling(values, self.window, "median")
        scale = rolling_mad(values, self.window)
        scale = np.where(scale < EPS, self.fallback_stats_.scale, scale)
        scores = np.abs(values - baseline) / scale
        return finite_array(scores)


@dataclass
class IsolationForestDetector(Detector):
    rolling_window: int = 64
    n_estimators: int = 200
    contamination: str | float = "auto"
    seed: int = 42
    name: str = "isolation-forest"

    def fit(self, values: np.ndarray, train_end: int) -> IsolationForestDetector:
        features = point_features(values, self.rolling_window)
        self.scaler_ = RobustScaler()
        train_features = self.scaler_.fit_transform(features[:train_end])
        self.model_ = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.seed,
            n_jobs=-1,
        )
        self.model_.fit(train_features)
        return self

    def score(self, values: np.ndarray) -> np.ndarray:
        features = self.scaler_.transform(point_features(values, self.rolling_window))
        return finite_array(-self.model_.score_samples(features))


class _MLPAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, bottleneck_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class _RNNPredictor(nn.Module):
    def __init__(self, kind: Literal["lstm", "gru"], hidden_dim: int) -> None:
        super().__init__()
        recurrent = nn.LSTM if kind == "lstm" else nn.GRU
        self.rnn = recurrent(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        output, _ = self.rnn(values)
        return self.head(output[:, -1, :]).squeeze(-1)


def resolve_torch_device(device: str | None) -> str:
    if device is not None:
        return device
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def fit_torch_model(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> None:
    torch.manual_seed(seed)
    if inputs.device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    n_samples = inputs.shape[0]
    for _ in range(epochs):
        order = torch.randperm(n_samples, device=inputs.device)
        for start in range(0, n_samples, batch_size):
            batch_index = order[start : start + batch_size]
            prediction = model(inputs[batch_index])
            loss = loss_fn(prediction, targets[batch_index])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


@dataclass
class MLPAutoencoderDetector(Detector):
    window: int = 64
    hidden_dim: int = 64
    bottleneck_dim: int = 16
    epochs: int = 20
    batch_size: int = 128
    lr: float = 1e-3
    device: str | None = None
    seed: int = 42
    name: str = "mlp-autoencoder"

    def fit(self, values: np.ndarray, train_end: int) -> MLPAutoencoderDetector:
        self.stats_: ScaleStats = robust_scale_stats(values[:train_end])
        scaled = apply_scale(values[:train_end], self.stats_)
        self.window_ = min(self.window, len(scaled))
        if self.window_ < 2:
            self.model_ = None
            return self

        train_windows = sliding_windows(scaled, self.window_)
        self.device_ = resolve_torch_device(self.device)
        torch.manual_seed(self.seed)
        if self.device_.startswith("cuda"):
            torch.cuda.manual_seed_all(self.seed)
        self.model_ = _MLPAutoencoder(self.window_, self.hidden_dim, self.bottleneck_dim).to(
            self.device_
        )
        inputs = torch.as_tensor(train_windows, dtype=torch.float32, device=self.device_)
        fit_torch_model(
            self.model_,
            inputs,
            inputs,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            seed=self.seed,
        )
        return self

    def score(self, values: np.ndarray) -> np.ndarray:
        scaled = apply_scale(values, self.stats_)
        if self.model_ is None or len(scaled) < self.window_:
            return finite_array(np.abs(scaled))
        windows = sliding_windows(scaled, self.window_)
        inputs = torch.as_tensor(windows, dtype=torch.float32, device=self.device_)
        self.model_.eval()
        errors = []
        with torch.no_grad():
            for start in range(0, len(inputs), self.batch_size):
                batch = inputs[start : start + self.batch_size]
                reconstructed = self.model_(batch)
                errors.append(torch.mean((reconstructed - batch) ** 2, dim=1).cpu().numpy())
        return window_scores_to_points(len(values), self.window_, np.concatenate(errors))


@dataclass
class RNNPredictorDetector(Detector):
    kind: Literal["lstm", "gru"]
    window: int = 64
    hidden_dim: int = 32
    epochs: int = 15
    batch_size: int = 128
    lr: float = 1e-3
    device: str | None = None
    seed: int = 42

    @property
    def name(self) -> str:
        return self.kind

    def fit(self, values: np.ndarray, train_end: int) -> RNNPredictorDetector:
        self.stats_: ScaleStats = robust_scale_stats(values[:train_end])
        scaled = apply_scale(values[:train_end], self.stats_)
        self.window_ = min(self.window, max(1, len(scaled) - 1))
        if len(scaled) <= self.window_:
            self.model_ = None
            return self

        windows = sliding_windows(scaled[:-1], self.window_)
        targets = scaled[self.window_ :]
        self.device_ = resolve_torch_device(self.device)
        torch.manual_seed(self.seed)
        if self.device_.startswith("cuda"):
            torch.cuda.manual_seed_all(self.seed)
        self.model_ = _RNNPredictor(self.kind, self.hidden_dim).to(self.device_)
        inputs = torch.as_tensor(windows[:, :, None], dtype=torch.float32, device=self.device_)
        target_tensor = torch.as_tensor(targets, dtype=torch.float32, device=self.device_)
        fit_torch_model(
            self.model_,
            inputs,
            target_tensor,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            seed=self.seed,
        )
        return self

    def score(self, values: np.ndarray) -> np.ndarray:
        scaled = apply_scale(values, self.stats_)
        if self.model_ is None or len(scaled) <= self.window_:
            return finite_array(np.abs(scaled))
        windows = sliding_windows(scaled[:-1], self.window_)
        targets = scaled[self.window_ :]
        inputs = torch.as_tensor(windows[:, :, None], dtype=torch.float32, device=self.device_)
        self.model_.eval()
        errors = []
        with torch.no_grad():
            for start in range(0, len(inputs), self.batch_size):
                batch = inputs[start : start + self.batch_size]
                prediction = self.model_(batch).cpu().numpy()
                target = targets[start : start + len(prediction)]
                errors.append((prediction - target) ** 2)
        return prediction_errors_to_points(len(values), self.window_, np.concatenate(errors))


def build_detector(
    method: str,
    rolling_window: int,
    window: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str | None,
    seed: int,
) -> Detector:
    if method == "robust-zscore":
        return RobustZScoreDetector()
    if method == "rolling-residual":
        return RollingResidualDetector(window=rolling_window)
    if method == "isolation-forest":
        return IsolationForestDetector(
            rolling_window=rolling_window,
            seed=seed,
        )
    if method == "mlp-autoencoder":
        return MLPAutoencoderDetector(
            window=window,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
            seed=seed,
        )
    if method in {"lstm", "gru"}:
        return RNNPredictorDetector(
            kind=method,
            window=window,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
            seed=seed,
        )
    raise ValueError(f"Unknown method: {method}")
