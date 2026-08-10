import { existsSync } from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
process.env.HF_HUB_DISABLE_XET ||= "1";
const windows = process.platform === "win32";
const basePython = path.join(repoRoot, ".venv-ml", windows ? "Scripts/python.exe" : "bin/python");
const forecastRoot = path.join(repoRoot, ".venv-forecast");
const forecastPython = path.join(forecastRoot, windows ? "Scripts/python.exe" : "bin/python");
const uv = path.join(repoRoot, ".tools", windows ? "uv/uv.exe" : "uv-linux/uv");
const requestedArgs = process.argv.slice(2);

if (!existsSync(basePython) || !existsSync(uv)) {
  run(process.execPath, [path.join(repoRoot, "scripts", "bootstrap-ml.mjs")]);
}

run(basePython, [
  path.join(repoRoot, "ml", "setup_forecast_models.py"),
  "--sources-only",
  ...requestedArgs,
]);

if (!existsSync(forecastPython)) {
  run(uv, ["venv", "--python", "3.12", forecastRoot]);
}

const indexArguments = windows
  ? [
      "--extra-index-url",
      "https://download.pytorch.org/whl/cu130",
      "--index-strategy",
      "unsafe-best-match",
    ]
  : [];
run(uv, [
  "pip",
  "install",
  "--python",
  forecastPython,
  "--link-mode",
  "copy",
  ...indexArguments,
  "-r",
  path.join(repoRoot, "ml", "forecast-models-requirements.txt"),
  "-e",
  path.join(repoRoot, ".tools", "FinCast"),
  "-e",
  path.join(repoRoot, ".tools", "TiRex-2"),
  "-e",
  path.join(repoRoot, ".tools", "chronos-forecasting"),
]);
run(uv, ["pip", "check", "--python", forecastPython]);
run(forecastPython, [
  path.join(repoRoot, "ml", "setup_forecast_models.py"),
  ...requestedArgs,
]);
run(forecastPython, [path.join(repoRoot, "ml", "smoke_forecast_models.py"), "--models", "all"]);

function run(command, args) {
  const result = spawnSync(command, args, {
    cwd: repoRoot,
    env: { ...process.env },
    stdio: "inherit",
  });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    throw new Error(`${command} exited with code ${result.status ?? 1}`);
  }
}
