/** Exercise the real persistent Python worker through the server paper adapter. */
import { readFileSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import path from "node:path";
import { performance } from "node:perf_hooks";
import { DenseOracleController, type DenseInitialization, type DenseObservation } from "../src/oracle-dense-controller.js";

const [python, config, inputPath, outputPath, stateDirectory] = process.argv.slice(2);
if (!python || !config || !inputPath || !outputPath)
  throw new Error("Usage: check-oracle-dense-worker.ts PYTHON CONFIG INPUT OUTPUT");
const input = JSON.parse(readFileSync(inputPath, "utf8")) as {
  readonly initial: DenseInitialization;
  readonly ticks: readonly DenseObservation[];
};
const options = { pythonExecutable: path.resolve(python), configPath: path.resolve(config), cwd: process.cwd(),
  ...(stateDirectory ? { stateDirectory: path.resolve(stateDirectory) } : {}) };
let controller = new DenseOracleController(options);
const started = performance.now();
try {
  const decisions = [await controller.initialize(input.initial)];
  let restartChecked = false;
  for (const tick of input.ticks) {
    const decision = await controller.onTick(tick);
    decisions.push(decision);
    if (stateDirectory && !restartChecked && decision.phase === "active") {
      await controller.checkpoint();
      await controller.close();
      controller = new DenseOracleController(options);
      const restored = await controller.restore();
      if (restored.update !== decision.update || restored.timestamp !== decision.timestamp ||
          JSON.stringify(restored.account) !== JSON.stringify(decision.account))
        throw new Error("Restored worker differs from the completed decision boundary");
      restartChecked = true;
    }
  }
  if (stateDirectory && !restartChecked) throw new Error("No nonterminal tick available for restart check");
  const last = decisions.at(-1)!;
  if (last.phase !== "complete" || last.account.asset !== 0 || last.account.orders.length !== 0 ||
      last.terminalWealth === null || last.terminalWealth !== last.account.quote)
    throw new Error("Smoke episode did not finish with an executed flat account");
  const files = ["ml/serve_oracle_dense.py", "ml/oracle_dense_controller.py",
    "apps/server/src/oracle-dense-controller.ts", config, inputPath];
  const result = { passed: true, restartChecked, elapsedSeconds: (performance.now() - started) / 1000,
    decisions, sourceHashes: Object.fromEntries(files.map(file =>
      [file, createHash("sha256").update(readFileSync(file)).digest("hex")])),
    scope: "Real Python NSF worker and server paper adapter through a short supplied observation sequence; not a profitability evaluation." };
  writeFileSync(outputPath, JSON.stringify(result, null, 2));
  console.log(JSON.stringify({ passed: true, elapsedSeconds: result.elapsedSeconds,
    updates: last.update, terminalWealth: last.terminalWealth, restartChecked, outputPath }));
} finally { await controller.close(); }
