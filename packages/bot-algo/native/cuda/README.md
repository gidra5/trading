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

The generated `build/libvw_kama_cuda.so` is machine-local and ignored by Git. The JavaScript
wrapper loads its stable C ABI through Koffi, so it does not depend on a particular Node native
addon ABI. The binary contains RTX 30-series (`sm_86`) code plus forward-compatible `compute_86`
PTX.
