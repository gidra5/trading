from __future__ import annotations

from collections.abc import Mapping
import math

import torch
from torch import Tensor, nn


SWA_START_FRACTIONS = (0.5, 0.75, 0.9)
SWA_CONTRACT = "epoch-swa-sweep-v1"


def swa_candidate_id(start_fraction: float) -> str:
    return f"swa-start-{round(start_fraction * 100):02d}-percent"


def swa_config() -> dict:
    return {
        "contract": SWA_CONTRACT,
        "updateInterval": "epoch-end",
        "startFractions": list(SWA_START_FRACTIONS),
    }


def validate_swa_config(value: object) -> None:
    if value != swa_config():
        raise ValueError("SWA sweep contract is invalid")


def _clone_state(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }


class EpochSwaSweep:
    """Track several stochastic-weight averages from one optimizer path."""

    def __init__(self, model: nn.Module, *, maximum_epochs: int) -> None:
        if maximum_epochs < 1:
            raise ValueError("maximum epochs must be positive")
        self.maximum_epochs = int(maximum_epochs)
        initial = _clone_state(model)
        self.candidates = {
            swa_candidate_id(fraction): {
                name: value.clone() for name, value in initial.items()
            }
            for fraction in SWA_START_FRACTIONS
        }
        self.counts = {name: 0 for name in self.candidates}
        self.completed_epochs = 0

    def update(self, model: nn.Module, *, epoch: int) -> None:
        if epoch != self.completed_epochs:
            raise ValueError("SWA epochs must be contiguous")
        current = model.state_dict()
        completed = epoch + 1
        with torch.no_grad():
            for fraction in SWA_START_FRACTIONS:
                if completed < math.ceil(self.maximum_epochs * fraction):
                    continue
                candidate_id = swa_candidate_id(fraction)
                candidate = self.candidates[candidate_id]
                count = self.counts[candidate_id]
                for name, value in current.items():
                    if value.is_floating_point() or value.is_complex():
                        candidate[name].lerp_(value.detach(), 1 / (count + 1))
                    else:
                        candidate[name].copy_(value.detach())
                self.counts[candidate_id] = count + 1
        self.completed_epochs = completed

    def candidate_states(self) -> dict[str, dict[str, Tensor]]:
        missing = [name for name, count in self.counts.items() if count < 1]
        if missing:
            raise RuntimeError(f"SWA candidates have no observations: {missing}")
        return self.candidates

    def state_dict(self) -> dict:
        return {
            "contract": SWA_CONTRACT,
            "maximumEpochs": self.maximum_epochs,
            "completedEpochs": self.completed_epochs,
            "candidates": self.candidates,
            "counts": self.counts,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if state.get("contract") != SWA_CONTRACT \
                or int(state.get("maximumEpochs", 0)) != self.maximum_epochs:
            raise ValueError("SWA checkpoint contract changed")
        candidates = state.get("candidates")
        counts = state.get("counts")
        if not isinstance(candidates, dict) \
                or candidates.keys() != self.candidates.keys() \
                or not isinstance(counts, dict) \
                or counts.keys() != self.candidates.keys():
            raise ValueError("SWA checkpoint candidates changed")
        for candidate_id, destination in self.candidates.items():
            source = candidates[candidate_id]
            if not isinstance(source, dict) or source.keys() != destination.keys():
                raise ValueError("SWA model state changed")
            for name, value in destination.items():
                candidate_value = source[name]
                if not isinstance(candidate_value, Tensor) \
                        or candidate_value.shape != value.shape:
                    raise ValueError("SWA tensor state changed")
                value.copy_(candidate_value.to(device=value.device))
        self.counts = {name: int(counts[name]) for name in self.candidates}
        self.completed_epochs = int(state.get("completedEpochs", -1))
        if not 0 <= self.completed_epochs <= self.maximum_epochs \
                or any(count < 0 for count in self.counts.values()):
            raise ValueError("SWA checkpoint progress is invalid")
