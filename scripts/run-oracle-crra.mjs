import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2);
function option(key, fallback) { const i = args.indexOf(key); return i < 0 ? fallback : args[i + 1]; }
const count = option("--count", "3600");
const gamma = option("--gamma", "0");
if (gamma !== "0" && gamma !== "2") throw new Error("--gamma must be 0 or 2");
const output = path.resolve(root, option("--output", "data/backtests/stochastic-crra/reproduced"));
const runDirectory = path.resolve(root, "data/training/runs/stochastic-crra-production178-zeroatom-k256-v1");
const python = path.join(root, ".venv-ml", process.platform === "win32" ? "Scripts/python.exe" : "bin/python");
if (!existsSync(python)) throw new Error("Install the ML environment with node scripts/bootstrap-ml.mjs first");
function run(command, params) {
  const result = spawnSync(command, params, { cwd: root, stdio: "inherit", windowsHide: true });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`${command} failed with exit ${result.status}`);
}
const node = (...params) => run(process.execPath, params);
if (!existsSync(path.join(runDirectory, "best.pt"))) run(python, ["ml/train_stochastic_price_forecast.py"]);
// Compare on validation before export, honoring whichever distribution wins.
run(python, ["ml/compare_stochastic_price_forecasts.py"]);
const selection = JSON.parse(readFileSync(path.join(runDirectory, "model-selection.json"), "utf8"));
const checkpoint = path.resolve(root, selection.selectedCheckpoint.replaceAll("\\", "/"));
mkdirSync(output, { recursive: true });
const forecasts = path.join(output, "forecasts.jsonl");
run(python, ["ml/export_stochastic_price_forecast.py", "--checkpoint", checkpoint, "--output", forecasts, "--count", count]);
node("node_modules/typescript/bin/tsc", "-p", "packages/bot-algo/tsconfig.oracle.json");
node("--conditions=development", "--import", "tsx", "scripts/backtest-oracle-crra.ts", "--forecasts", forecasts,
  "--output", output, "--count", count, "--certificates", "3", "--gamma", gamma);
const fixtures = path.join(output, "oracle-conformance.jsonl");
node("--import", "tsx", "packages/bot-algo/test/oracle-conformance-fixtures.ts", fixtures);
const projectCache = path.join(root, "docs/theory/lean/.lake/build/lib/lean/TradingTheory/ExactCRRAKernel.olean");
node("scripts/theory-lean.mjs", existsSync(projectCache) ? "check" : "setup");
node("scripts/test-oracle-risk-certificates.mjs");
node("scripts/verify-oracle-crra.mjs", "--fixtures", fixtures, "--certificates", path.join(output, "decisions.jsonl"),
  "--output", path.join(output, "lean-verification.json"));
