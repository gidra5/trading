from __future__ import annotations

import torch
from torch import Tensor, nn

from next_return_dataset import HISTORY_RETURN_COUNT
from normalized_glu_next_return import NormalizedGluNextReturn


ARCHITECTURE_CONTRACT = (
    "autoregressive-shared-two-second-glu-to-single-minute-return-v2"
)


def autoregressive_return_path(
    predictor: nn.Module,
    features: Tensor,
    *,
    horizon_seconds: int,
    chunk_seconds: int,
) -> Tensor:
    """Roll a return predictor forward while retaining the complete graph."""
    if features.ndim != 2 or features.shape[1] != HISTORY_RETURN_COUNT:
        raise ValueError("autoregressive input must have shape [example, 120]")
    if chunk_seconds < 1 or horizon_seconds < chunk_seconds \
            or horizon_seconds % chunk_seconds:
        raise ValueError("rollout horizon must contain complete prediction chunks")
    history = features
    chunks: list[Tensor] = []
    for _ in range(horizon_seconds // chunk_seconds):
        prediction = predictor(history)
        if prediction.shape != (features.shape[0], chunk_seconds):
            raise ValueError("chunk predictor returned an invalid shape")
        chunks.append(prediction)
        history = torch.cat((history[:, chunk_seconds:], prediction), dim=1)
    return torch.cat(chunks, dim=1)


class AutoregressiveMinuteReturn(nn.Module):
    """Generate sixty one-second returns but expose their single 1m sum."""

    architecture_contract = ARCHITECTURE_CONTRACT

    def __init__(
        self,
        feature_mean: Tensor,
        feature_std: Tensor,
        chunk_target_mean: Tensor,
        chunk_target_std: Tensor,
        *,
        horizon_seconds: int = 60,
        chunk_seconds: int = 2,
        widths: tuple[int, ...] = (512,),
        dropout: float = 0.05,
        dropout_rate: float = 0.5,
        initial_radius: float = 0.0031622776601683794,
        minimum_radius: float = 1e-4,
    ) -> None:
        super().__init__()
        if chunk_target_mean.shape != (chunk_seconds,) \
                or chunk_target_std.shape != chunk_target_mean.shape:
            raise ValueError("chunk normalization must match the chunk horizon")
        if horizon_seconds != 60:
            raise ValueError("this model contract predicts one 60-second return")
        self.horizon_seconds = int(horizon_seconds)
        self.chunk_seconds = int(chunk_seconds)
        self.core = NormalizedGluNextReturn(
            feature_mean,
            feature_std,
            chunk_target_mean,
            chunk_target_std,
            widths=widths,
            dropout=dropout,
            dropout_rate=dropout_rate,
            initial_radius=initial_radius,
            minimum_radius=minimum_radius,
        )

    def forward_path(self, features: Tensor) -> Tensor:
        return autoregressive_return_path(
            self.core,
            features,
            horizon_seconds=self.horizon_seconds,
            chunk_seconds=self.chunk_seconds,
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.forward_path(features).sum(dim=1)
