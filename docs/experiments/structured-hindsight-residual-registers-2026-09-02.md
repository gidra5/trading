# Residual hindsight stream with reset learned registers

Plan:
`structured-production59-w256-selfattn256-k1-2-k2-2-hindsight-residual-r128-s1-layer2h512-no13-nocarry-corrgate-train256k-b10240-ema4-v5`.

The user confirmed S=1 means one complete 59-feature sample per predicted
second, not one scalar return. Registers do not carry between predicted
seconds. Layer (13) is explicitly omitted.

## Output transform

For every output step s, independently:

```
r0 = layer12                         # one shared learned R128 vector
z0 = concat(r0, noisy_hindsight[s])  # 128 + 1*59 = 187 coordinates
delta = layer2(concat(W[s], z0, v))  # 256 + 187 + 1 -> hidden512 -> 187
z1 = z0 + delta
prediction[s] = z1[128:]            # emit only the full 59-feature sample
```

The scalar noise variance v still conditions only layer (2). Layer (12) is
initialized to zeros but is an unconstrained learned parameter. The residual
addition has fixed coefficient one; the previous learned bypass/residual
gain vectors are removed. To retain near-passthrough initialization, layer
(2)'s final projection starts as a rectangular identity multiplied by 0.001,
with zero output bias. This scaling is only an initialization, not a permanent
multiplier on gradients or residual outputs.

No layer (13) is constructed. No register/sample state or noisy hindsight is
passed to the structured recurrence or attention. Each step resets r0 from
the same learned parameter rather than reading z1 from another step.

Consequently the register UPDATE coordinates are currently unused by the
loss and have zero gradients in the corresponding output-projection rows.
The INITIAL learned registers receive gradients through their influence on
the sample residual. Full updated streams are available in model traces;
no auxiliary register loss or hidden cross-step connection is invented.

## Unchanged setup

K1=K2=2; production-59 direct feature output and standardized full-feature MSE;
W/PF/NF=256, M/P/F=128; layer-8 causal self-attention Q/K/V/O=256; 256,000
training examples, equal train/evaluation batch 10,240; seed 1337, original
optimizer, EMA half-life 4 and 512 epochs. No SAM, dropout or adversarial
regularization. This is a fresh run.

The correlation gate is unchanged: start at variance zero and increase by
0.01 only after raw-weight training reconstruction next-1s correlation reaches
2^(-1/3600). Pure-noise forecasting metrics and checkpoint selection remain
separate from the target-assisted training/validation reconstruction probes.

Total parameters: **5,098,167**, including 128 learned initial registers.
Layer (2) has 1,076,925 parameters. The old learned-gain H512 model had
4,901,421 total parameters.

## Predecessor completion

The H512 learned-gain predecessor completed 512 epochs naturally. Its
best-validation-MSE, best-validation-correlation, and last policies all chose
zero-based epoch 511. Pure-noise validation next-1s correlation was
0.0044199581, with MSE skill -23.769497% for all three policies.

At training variance 0.01, the last raw checkpoint's separate reconstruction
probe had correlation 0.9990015057 on training and 0.9993449324 on validation.
These target-assisted values are not forecast accuracy. The diagnostic was
persisted before cleanup and exactly reproduced the last logged probe.

After verification, the three checkpoint pointers and three unreferenced
immutable objects were deleted. The training-log SHA-256 remained unchanged;
all training/smoke/launcher logs, results, evaluations, diagnostic, status,
and saved plan remain. The superseded learned-gain readout implementation
was replaced, not retained as a second runtime path.

## Verification

All 37 structured-model tests passed. Coverage includes exact parameter
counts; literal residual addition; S=1 full-feature shape; learned-register
gradients and intentionally unused register-update rows; reset per output;
absence of layer (13); future/noisy-token isolation; near-identity
initialization; optimizer coverage; checkpoint reload; and separation of
current-noise diagnostics from target-independent forecast evaluation.

The compiled full-batch CUDA smoke completed one optimizer step with the
intended 5,098,167 parameters and no regularization. Its online standardized
feature MSE was 3.11026e-7. At variance zero, the raw-weight next-1s probe
correlations after that update were 0.9999725298 (training) and 0.9999965520
(held-out reconstruction). The training-only gate passed and selected 0.01
for the next epoch. The only stderr output was PyTorch's benign small-GPU
autotuning warning.

Full training was verified through six completed epochs, starting fresh at
epoch 0 with equal train/evaluation batches of 10,240. The first full epoch
reached training reconstruction correlation 0.9999994520 at variance zero;
the second used variance 0.01 and correctly held below the gate. Initial
warm epochs took 6.50-6.70 s including diagnostics. Last/best-MSE/best-correlation
checkpoint pointers are durable, the exact launcher/workers are alive, full
stderr is empty, and the dashboard API reports this run as training.
