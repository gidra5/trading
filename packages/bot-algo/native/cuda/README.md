# VW-KAMA CUDA screening helper

This optional helper batches independent genetic-optimizer candidates on NVIDIA CUDA. It uses
32-bit screening arithmetic and the optimizer rescans selected candidates with the existing
64-bit TypeScript evaluator before writing fit, validation, test, or preset results.

On WSL, install the CUDA **toolkit** from NVIDIA's WSL repository; do not install a Linux NVIDIA
display driver. The Windows driver supplies CUDA through WSL.

```sh
npm install
npm run build:cuda
npm run search:kama -- --accelerator auto
```

`--accelerator cuda` requires the helper and a working device. `--accelerator cpu` disables it.
`auto` uses CUDA for batches of at least 32 candidates and otherwise retains CPU workers.

On native Windows, `npm run mlp:bootstrap` downloads the redistributable CUDA
compiler/runtime into `.tools` and builds `build/vw_kama_cuda.dll` with Visual
Studio 2022 C++ Build Tools. Linux/WSL uses the system CUDA toolkit and builds
`build/libvw_kama_cuda.so`. Both outputs are machine-local and ignored by Git. The JavaScript
wrapper loads its stable C ABI through Koffi, so it does not depend on a particular Node native
addon ABI. The binary contains RTX 30-series (`sm_86`) code plus forward-compatible `compute_86`
PTX.

The exposure-value distribution path factors each target into a precomputed
mandatory initial hold and a fee-separable continuation. The initial target is
held for `H` candles; continuation then advances one candle per Bellman layer
until the total `T`-candle horizon, followed by an exact closeout to cash.
Buy/sell prefix and suffix scans make each continuation layer linear in the
action-grid size and query each passively drifted exposure exactly, without
interpolating a sampled continuation row. For the production rolling horizon,
complete rows stay in warp-local memory and the compact path allocates only its
Float32 holding and endpoint rows; the outer forced-action values are
normalized directly from registers into the probability output. Infeasible
`-Infinity` values receive exactly zero probability. The unused general
Float64 Bellman tables and intermediate forced-action table are omitted.
Reused double-buffered pinned host slots stage prices into CUDA and
probabilities back out while the dataset worker pipeline streams the preceding
output into compression.
