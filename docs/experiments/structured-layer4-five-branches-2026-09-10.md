# Structured layer (4): five branches

Run: `structured-production59-direct59-layer4branches5w256-selfattn256-k1-2-k2-2-train256k-b10240-ema4-v1`.

The experiment returns to the direct production59 structured model with layer-(8)
causal self-attention. It uses two observed seconds and predicts two future seconds.
There is no hindsight input or residual register stream.

## Architecture

Layer (4) receives five concatenated 256-wide vectors (1,280 inputs): the existing
P and F states, plus three independently parameterized GNGLUs. The new branches
consume the current W feature embedding during observed-input transitions and the
preceding NF embedding during forecast transitions. Each added branch is reused
across transitions; the three branches do not share parameters with each other.

W/P/F/PF/NF and attention Q/K/V/O are 256; M remains 128. P and F therefore widen
from the historical control's 128. Total trainable parameters: **9,067,378**.

This implements the user's branches 3, 4, and 5 as three additions to the existing
two inputs to layer (4). It is one five-branch experiment.

## Training and comparison

256,000 training examples, 65,534 usable validation and test examples, equal train
and evaluation batch 10,240, 512 epochs, learning rate 0.0001, Muon/AdamW, EMA 4.
The objective is standardized MSE across all 59 directly predicted features and
both forecast steps. No SAM, dropout, adversarial inputs, or weight decay.

The original compact production dataset was rebuilt from the existing historical
source with the original 512,000/65,536/65,536 export configuration; the structured
loader selects the original 256,000-example training subset and constructible
two-step targets.

Headline/dashboard metrics use the next completed 1s return. Checkpoint selection
continues to use the complete feature-state validation MSE/correlation, as in the
control. The comparison table also includes both individual return-step rows.

Control:
`structured-shared-io-feature-process-k1-2-k2-2-i59-o59-w256-m128-p128-f128-pf256-nf256-layer8-selfattn256-productiondense-train256k-batch10240-mse-ema4-v1`.

## Verification and launch

All 51 structured-model tests passed, including branch independence, sharing
across transitions, gradient flow, and optimizer coverage. A CUDA smoke completed
with the full 10,240 training/evaluation batch. The full compiled run started on
2026-09-10; compilation disables CUDA Graphs. After warmup, epoch increments were
about 7–8 seconds (the log's `seconds` field is cumulative elapsed time).

Last, best feature-validation MSE, and best feature-validation correlation
checkpoints were loaded successfully and checked for finite model tensors.

The frontend is at http://localhost:5173/#/mlp-training and the backend listens on
3001. The metrics reader exposes live logs and persisted evaluation policies;
partial trailing log records are deferred until complete. Pages load on demand
so unrelated pages affected by the current branch's module deletions do not block
the training dashboard. Training and launcher logs remain in the run directory.

## Final evaluation

Stopped and finalized at the user's request after 95 completed epochs. All three
durable checkpoint policies were evaluated on the full train/validation/test
splits using CUDA with batch 10,240. The policies below select by complete
feature-state validation metrics; the reported metrics are next-1s return only.

| Checkpoint | Completed epoch | Validation MSE skill | Validation correlation |
| --- | ---: | ---: | ---: |
| Best feature validation MSE | 33 | 6.3869% | 0.252962 |
| Best feature validation correlation | 32 | 6.4254% | 0.253718 |
| Last | 95 | -25.9294% | 0.099753 |

The last checkpoint deteriorated substantially; the selected checkpoints retain
the earlier performance. Persisted evaluation is in the run's
`state/stopped-evaluation.json` and its completed-evaluation benchmark artifact.
Cleanup removed the three checkpoint pointers and three now-unreferenced
immutable objects. Logs, plan snapshot, status, result, and evaluation were
verified unchanged by cleanup.
