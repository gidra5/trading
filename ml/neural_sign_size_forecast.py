"""Historical structured backbones specialized for sign and conditional size.

The new readouts predict horizon-specific tasks from two observed production59
states. The sign backbone keeps the old base17 dimensions, but its intermediate
outputs are latent coordinates: this wrapper does not supervise or reconstruct
the old primitive feature graph. Size outputs are positive normalized means;
the trainer owns their sign-conditional target scales and probability mixing.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from structured_feature_process import StructuredSharedIoFeatureProcess

CONTRACT = "structured-specialist-sign-size-v1"
HORIZONS = (1, 900, 1800, 7200)
SIGN_SOURCE_PLAN = "structured-shared-io-feature-process-k1-2-k2-2-i59-obase17-derived59-w256-m128-p128-f128-pf256-nf256-layer8-selfattn256-productiondense-train256k-batch10240-mse-ema4-v1"
SIZE_SOURCE_PLAN = "structured-production59-direct59-layer4branches5w256-selfattn256-k1-2-k2-2-train256k-b10240-ema4-v1"


def architecture_config(task: str) -> dict:
    """Return the embedded historical dimensions and declared specialist change."""
    if task not in ("sign", "size"):
        raise ValueError("task must be sign or size")
    size = task == "size"
    config = {
        "contract": CONTRACT, "task": task,
        "sourcePlanId": SIZE_SOURCE_PLAN if size else SIGN_SOURCE_PLAN,
        "backboneContract": "structured-shared-io-feature-process-v1",
        "inputSteps": 2, "outputSteps": 2, "inputFeatures": 59,
        "outputFeatures": 59 if size else 17,
        "featureWidth": 256, "marketWidth": 128,
        "prefixWidth": 256 if size else 128,
        "featureDistributionWidth": 256 if size else 128,
        "extendedPrefixWidth": 256, "nextFeatureDistributionWidth": 256,
        "initialRadius": math.sqrt(1e-5), "minimumRadius": 1e-4,
        "learnableCentering": False,
        "layer8Attention": {"type": "causal-self-attention-v1", "queryWidth": 256,
            "keyWidth": 256, "valueWidth": 256, "outputWidth": 256, "heads": 1,
            "sequence": "generated-next-feature-distribution-prefix"},
        "backboneOutputStatistics": "zero mean and unit standard deviation",
        "readout": "linear on flattened two-step standardized backbone outputs",
        "taskOutput": "positive normalized conditional mean sizes [horizon,up/down] via softplus/log(2)"
            if size else "unconstrained sign logits [horizon]",
        "specialization": "new task readout and objective; no historical primitive reconstruction loss",
    }
    if size:
        config["layer4ParallelBranches"] = {
            "type": "independent-current-feature-gnglu-v1", "count": 5, "width": 256,
            "combination": "concatenate-into-shared-layer-4",
            "parameterSharing": "each added branch owns independent parameters reused across input and forecast transitions",
            "source": "current observed W embedding for input transitions and preceding generated NF state for forecast transitions",
        }
    return config


class StructuredSpecialist(nn.Module):
    """Task readout over the preserved historical structured architecture."""

    def __init__(self, task: str, input_mean: Tensor, input_std: Tensor,
                 horizons: Sequence[int] = HORIZONS) -> None:
        super().__init__()
        config = architecture_config(task)
        if input_mean.shape != (59,) or input_std.shape != (59,):
            raise ValueError("specialist input statistics must have shape [59]")
        if not torch.isfinite(input_mean).all() or not torch.isfinite(input_std).all() \
                or bool((input_std <= 0).any()):
            raise ValueError("specialist statistics must be finite with positive standard deviations")
        horizons = tuple(horizons)
        if not horizons or any(isinstance(h, bool) or not isinstance(h, int) or h <= 0 for h in horizons) \
                or len(set(horizons)) != len(horizons):
            raise ValueError("horizons must be distinct positive integer seconds")
        self.task = task
        self.horizons = horizons
        width = config["outputFeatures"]
        self.backbone = StructuredSharedIoFeatureProcess(
            input_mean, input_std, torch.zeros(width), torch.ones(width),
            input_steps=2, output_steps=2, feature_width=config["featureWidth"],
            market_width=config["marketWidth"], prefix_width=config["prefixWidth"],
            feature_distribution_width=config["featureDistributionWidth"],
            extended_prefix_width=config["extendedPrefixWidth"],
            next_feature_distribution_width=config["nextFeatureDistributionWidth"],
            initial_radius=config["initialRadius"], minimum_radius=config["minimumRadius"],
            learnable_centering=config["learnableCentering"],
            layer4_parallel_branches=config.get("layer4ParallelBranches"),
            layer8_attention=config["layer8Attention"],
        )
        self.readout = nn.Linear(2*width, len(horizons)*(2 if task == "size" else 1))
        nn.init.zeros_(self.readout.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 3 or inputs.shape[1:] != (2, 59):
            raise ValueError("specialist inputs must have shape [batch,2,59]")
        prediction = self.readout(self.backbone(inputs).flatten(1))
        if self.task == "sign":
            return prediction
        # A zero readout represents the training-conditional mean (scale one).
        positive = (F.softplus(prediction) / math.log(2)).clamp_min(torch.finfo(prediction.dtype).tiny)
        return positive.reshape(-1, len(self.horizons), 2)

    def muon_parameters(self) -> tuple[Tensor, ...]:
        """The backbone's matrix partition plus the new task-readout matrix."""
        return (*self.backbone.muon_parameters(), self.readout.weight)
