import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const python = path.join(
  repoRoot,
  ".venv-ml",
  process.platform === "win32" ? "Scripts/python.exe" : "bin/python",
);
const args = process.argv.slice(2);
const overrides = {};
while (args[0] === "--env") {
  args.shift();
  const assignment = args.shift();
  const equals = assignment?.indexOf("=") ?? -1;
  if (!assignment || equals <= 0) {
    throw new Error("--env requires NAME=VALUE.");
  }
  overrides[assignment.slice(0, equals)] = assignment.slice(equals + 1);
}
if (args.length === 0) throw new Error("Python arguments are required.");

const runtimeRoot = path.join(repoRoot, "data", "runtime-cache");
const temporaryDirectory = path.join(runtimeRoot, "tmp");
const tritonCacheDirectory = path.join(runtimeRoot, "triton");
const torchInductorCacheDirectory = path.join(runtimeRoot, "torchinductor");
const cudaCacheDirectory = path.join(runtimeRoot, "cuda");
for (const directory of [
  temporaryDirectory,
  tritonCacheDirectory,
  torchInductorCacheDirectory,
  cudaCacheDirectory,
]) {
  fs.mkdirSync(directory, { recursive: true });
}

const child = spawn(python, args, {
  cwd: repoRoot,
  env: {
    ...process.env,
    ...overrides,
    PYTHONUTF8: process.env.PYTHONUTF8 || "1",
    PYTHONIOENCODING: process.env.PYTHONIOENCODING || "utf-8",
    PYTHONPATH: [path.join(repoRoot, "ml"), process.env.PYTHONPATH]
      .filter(Boolean)
      .join(path.delimiter),
    TMPDIR: process.env.TRADING_ML_TMP_DIR || temporaryDirectory,
    TEMP: process.env.TRADING_ML_TMP_DIR || temporaryDirectory,
    TMP: process.env.TRADING_ML_TMP_DIR || temporaryDirectory,
    TRITON_CACHE_DIR: process.env.TRITON_CACHE_DIR || tritonCacheDirectory,
    TORCHINDUCTOR_CACHE_DIR:
      process.env.TORCHINDUCTOR_CACHE_DIR || torchInductorCacheDirectory,
    CUDA_CACHE_PATH: process.env.CUDA_CACHE_PATH || cudaCacheDirectory,
  },
  stdio: "inherit",
});
for (const signal of ["SIGINT", "SIGTERM"]) {
  process.once(signal, () => child.kill(signal));
}
child.once("error", (error) => {
  console.error(error);
  process.exitCode = 1;
});
child.once("exit", (code) => {
  process.exitCode = code ?? 1;
});
