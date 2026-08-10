import { spawn, spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { prepareTrainingCache } from "./training-storage.mjs";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const python = path.join(
  repoRoot,
  ".venv-forecast",
  process.platform === "win32" ? "Scripts/python.exe" : "bin/python",
);
const cache = prepareTrainingCache(repoRoot);
const toolchainEnvironment = process.platform === "win32" ? visualStudioEnvironment() : {};
const toolchainPathKey = Object.keys(toolchainEnvironment).find(
  (key) => key.toLowerCase() === "path",
);
const toolchainPath = toolchainPathKey ? toolchainEnvironment[toolchainPathKey] : undefined;
if (toolchainPathKey) delete toolchainEnvironment[toolchainPathKey];
if (process.platform === "win32") {
  toolchainEnvironment.CL = [toolchainEnvironment.CL, "/FImalloc.h", "/Dalloca=_alloca"]
    .filter(Boolean)
    .join(" ");
}
const child = spawn(python, process.argv.slice(2), {
  cwd: repoRoot,
  env: {
    ...process.env,
    ...toolchainEnvironment,
    PATH: [path.dirname(python), toolchainPath, process.env.PATH]
      .filter(Boolean)
      .join(path.delimiter),
    PYTHONUTF8: process.env.PYTHONUTF8 || "1",
    PYTHONIOENCODING: process.env.PYTHONIOENCODING || "utf-8",
    PYTHONPATH: [path.join(repoRoot, "ml"), process.env.PYTHONPATH]
      .filter(Boolean)
      .join(path.delimiter),
    TMPDIR: process.env.TRADING_ML_TMP_DIR || cache.temporary,
    TEMP: process.env.TRADING_ML_TMP_DIR || cache.temporary,
    TMP: process.env.TRADING_ML_TMP_DIR || cache.temporary,
    TRITON_CACHE_DIR: process.env.TRITON_CACHE_DIR || cache.triton,
    TORCHINDUCTOR_CACHE_DIR: process.env.TORCHINDUCTOR_CACHE_DIR || cache.torchinductor,
    CUDA_CACHE_PATH: process.env.CUDA_CACHE_PATH || cache.cuda,
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
    throw new Error(`Failed to load Visual Studio environment: ${result.stderr}`);
  }
  return Object.fromEntries(
    result.stdout.split(/\r?\n/).flatMap((line) => {
      const equals = line.indexOf("=");
      return equals > 0 ? [[line.slice(0, equals), line.slice(equals + 1)]] : [];
    }),
  );
}
