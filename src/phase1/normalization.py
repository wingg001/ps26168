"""
Per-channel z-score normalization fit on the TRAIN split only, then applied
to both train and val windows. Prevents statistics leaking from validation
data into the normalization used for training.
"""
from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass
class NormalizationStats:
    mean: np.ndarray  # (C,)
    std: np.ndarray   # (C,)
    channel_names: list[str]
    fit_n_windows: int

    def to_dict(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "channel_names": self.channel_names,
            "fit_n_windows": self.fit_n_windows,
        }


def fit_normalizer(train_windows: np.ndarray, channel_names: list[str], eps: float = 1e-6) -> NormalizationStats:
    """train_windows: (N, T, C). Stats computed over N and T jointly, per channel C."""
    flat = train_windows.reshape(-1, train_windows.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < eps, eps, std)
    return NormalizationStats(mean=mean, std=std, channel_names=channel_names, fit_n_windows=train_windows.shape[0])


def apply_normalizer(windows: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    return (windows - stats.mean) / stats.std
