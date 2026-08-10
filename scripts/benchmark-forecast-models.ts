import path from "node:path";
import { spawn } from "node:child_process";
import { KamaInspector } from "../apps/server/src/kama-inspector.js";

const repoRoot = path.resolve(import.meta.dirname, "..");
const inspector = new KamaInspector(path.join(repoRoot, "data"));
const windows = inspector.catalog().windows.filter((window) =>
  window.id !== "latest" && !window.id.startsWith("fit-"));
if (windows.length !== 28) {
  throw new Error(`Expected 28 non-fit static inspector windows; found ${windows.length}.`);
}
const child = spawn(
  process.execPath,
  [
    path.join(repoRoot, "scripts", "run-forecast-python.mjs"),
    "ml/benchmark_forecast_models.py",
    "--windows-json",
    JSON.stringify(windows),
    ...process.argv.slice(2),
  ],
  { cwd: repoRoot, env: { ...process.env }, stdio: "inherit" },
);
for (const signal of ["SIGINT", "SIGTERM"] as const) {
  process.once(signal, () => child.kill(signal));
}
child.once("error", (error) => {
  console.error(error);
  process.exitCode = 1;
});
child.once("exit", (code) => {
  process.exitCode = code ?? 1;
});
