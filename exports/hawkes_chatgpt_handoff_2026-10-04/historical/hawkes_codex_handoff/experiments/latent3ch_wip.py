#!/usr/bin/env python3
"""WIP scaffold for the model that should replace the residual-proxy latent scripts.

Do NOT treat this file as implemented inference. It documents the intended
binned generative model and leaves TODOs for a variational/SMC/EM smoother.
"""
from dataclasses import dataclass
import numpy as np

@dataclass
class Latent3CHConfig:
    bin_seconds: float = 1.0
    stability_max_rho: float = 0.95
    max_modes_per_edge: int = 6

class LatentThreeChannelHawkes:
    """Binned Poisson approximation to B/S/L multivariate Hawkes.

    Observed: Y_B[t], Y_S[t]
    Latent:   Y_L[t]

    For each i,j in {B,S,L},
        lambda_i[t] = mu_i + sum_k alpha[i,j,k] * x[j,k,t]
        x[j,k,t+1] = rho[j,k] * x[j,k,t] + normalized_jump * Y_j[t]

    The final implementation should support edge-specific mode counts/scales,
    or a more general source-specific shared trace bank with sparse edge loadings.
    """
    def __init__(self, cfg: Latent3CHConfig):
        self.cfg = cfg
        raise NotImplementedError(
            "Implement proper latent-count inference: variational smoother, SMC, "
            "or EM with an approximate smoother. Do not infer L directly from residuals."
        )

# Required experiments:
#   M0  = observed B/S only, optimized K_ij and continuous tau
#   M2  = M0 + slow-state modulation of event excitation amplitude
#   ML  = true latent B/S/L Hawkes
#   M2L = M2 + true latent L
#
# Required diagnostics:
#   posterior E[N_L,t | B,S]
#   B->L, S->L, L->L, L->B, L->S integrated masses
#   initialization sensitivity
#   mode-collapse checks
#   branching spectral radius for additive linear variants
#   held-out validation/test likelihood and horizon forecasts
