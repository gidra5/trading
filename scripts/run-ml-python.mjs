import { spawn, spawnSync } from "node:child_process";
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
const toolchainEnvironment =
  process.platform === "win32" ? visualStudioEnvironment() : {};

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
    ...toolchainEnvironment,
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

function visualStudioEnvironment() {
  const roots = [
    process.env.VSINSTALLDIR,
    "C:\\Program Files (x86)\\Microsoft Visual Studio\\2022\\BuildTools",
    "C:\\Program Files\\Microsoft Visual Studio\\2022\\BuildTools",
  ].filter(Boolean);
  const setup = roots
    .map((root) => path.join(root, "VC", "Auxiliary", "Build", "vcvars64.bat"))
    .find((candidate) => fs.existsSync(candidate));
  if (!setup) return {};
  const result = spawnSync(
    process.env.ComSpec ?? "cmd.exe",
    ["/d", "/s", "/c", `""${setup}" >nul && set"`],
    { encoding: "utf8", windowsVerbatimArguments: true },
  );
  if (result.status !== 0) {
    throw new Error(
      `Failed to load the Visual Studio build environment: ${result.stderr}`,
    );
  }
  return Object.fromEntries(
    result.stdout
      .split(/\r?\n/)
      .flatMap((line) => {
        const equals = line.indexOf("=");
        return equals > 0
          ? [[line.slice(0, equals), line.slice(equals + 1)]]
          : [];
      }),
  );
}
