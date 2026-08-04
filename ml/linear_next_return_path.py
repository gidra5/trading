from __future__ import annotations

import torch
from torch import Tensor, nn

from next_return_dataset import HISTORY_RETURN_COUNT


ARCHITECTURE_CONTRACT = "train-normalized-linear-return-path-v1"


class LinearNextReturnPath(nn.Module):
    """A train-normalized linear map from 120 returns to the next T returns."""

    architecture_contract = ARCHITECTURE_CONTRACT

    def __init__(
        self,
        feature_mean: Tensor,
        feature_std: Tensor,
        target_mean: Tensor,
        target_std: Tensor,
    ) -> None:
        super().__init__()
        if feature_mean.shape != (HISTORY_RETURN_COUNT,) \
                or feature_std.shape != feature_mean.shape \
                or not bool(torch.isfinite(feature_mean).all()) \
                or not bool(torch.isfinite(feature_std).all()) \
                or bool((feature_std <= 0).any()):
            raise ValueError("linear path feature normalization is invalid")
        if target_mean.ndim > 1 or target_std.shape != target_mean.shape \
                or target_mean.numel() < 1 \
                or not bool(torch.isfinite(target_mean).all()) \
                or not bool(torch.isfinite(target_std).all()) \
                or bool((target_std <= 0).any()):
            raise ValueError("linear path target normalization is invalid")
        self.horizon_return_count = int(target_mean.numel())
        target_shape = () if self.horizon_return_count == 1 else (-1,)
        self.register_buffer("feature_mean", feature_mean.float().clone())
        self.register_buffer("feature_std", feature_std.float().clone())
        self.register_buffer(
            "target_mean", target_mean.float().reshape(target_shape).clone()
        )
        self.register_buffer(
            "target_std", target_std.float().reshape(target_shape).clone()
        )
        self.output = nn.Linear(HISTORY_RETURN_COUNT, self.horizon_return_count)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # The raw model starts at the train-only per-lead mean baseline.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward_standardized(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[1] != HISTORY_RETURN_COUNT:
            raise ValueError("linear path model expects [example, 120] returns")
        standardized_features = (
            features.float() - self.feature_mean
        ) / self.feature_std
        result = self.output(standardized_features)
        return result.squeeze(-1) \
            if self.horizon_return_count == 1 else result

    def forward(self, features: Tensor) -> Tensor:
        return self.forward_standardized(features) * self.target_std \
            + self.target_mean


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
