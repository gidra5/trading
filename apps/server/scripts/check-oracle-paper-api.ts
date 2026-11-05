/** Real HTTP-route/NSF-worker smoke episode, including a complete server restart. */
import { readFileSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import path from "node:path";
import { performance } from "node:perf_hooks";
import { createServer } from "../src/index.js";

const [python, config, inputPath, stateDirectory, outputPath] = process.argv.slice(2);
if (!python || !config || !inputPath || !stateDirectory || !outputPath)
  throw new Error("Usage: check-oracle-paper-api.ts PYTHON CONFIG INPUT STATE_DIRECTORY OUTPUT");
const input = JSON.parse(readFileSync(inputPath, "utf8"));
if (input.ticks.length !== 2) throw new Error("API smoke requires two observed ticks");
const options = { databasePath: ":memory:", oraclePaper: { symbol: "BTCUSDT", checkpointEvery: 1,
  worker: { pythonExecutable: path.resolve(python), configPath: path.resolve(config), cwd: process.cwd(),
    stateDirectory: path.resolve(stateDirectory) } } };
let server = await createServer(options);
const post = async (suffix: string, payload?: unknown) => {
  const response = await server.inject({ method: "POST", url: "/bot/oracle-paper/" + suffix, payload });
  if (response.statusCode !== 200) throw new Error(`${suffix}: ${response.statusCode} ${response.body}`);
  return response.json();
};
const started = performance.now();
try {
  const initial = await post("start", input.initial);
  const first = await post("tick", input.ticks[0]);
  await post("stop");
  await server.close();
  server = await createServer(options);
  const restored = await post("resume");
  if (restored.update !== first.update || restored.timestamp !== first.timestamp ||
      JSON.stringify(restored.account) !== JSON.stringify(first.account))
    throw new Error("Server restart changed the saved account boundary");
  const terminal = await post("tick", input.ticks[1]);
  if (terminal.phase !== "complete" || terminal.account.asset !== 0 || terminal.account.orders.length ||
      terminal.terminalWealth !== terminal.account.quote) throw new Error("Invalid terminal paper account");
  const status = (await server.inject({ method: "GET", url: "/bot/oracle-paper" })).json();
  if (status.running || !status.completed) throw new Error("Completed paper account still running");
  const files = ["apps/server/src/index.ts", "apps/server/src/oracle-paper-service.ts",
    "apps/server/src/oracle-paper-routes.ts", "apps/server/src/oracle-dense-controller.ts",
    "ml/oracle_dense_controller.py", "ml/oracle_dense_metadata.py", "ml/oracle_dense_observe_batch.py",
    "ml/oracle_dense_worker_checkpoint.py", "ml/serve_oracle_dense.py", config, inputPath];
  const result = { passed: true, elapsedSeconds: (performance.now() - started) / 1000, initial, first, restored, terminal, status,
    sourceHashes: Object.fromEntries(files.map(file => [file, createHash("sha256").update(readFileSync(file)).digest("hex")])),
    scope: "Real Fastify paper routes and CPU NSF worker, with stop/server restart/restore/final close. Two-tick synthetic continuation; not a market backtest." };
  writeFileSync(outputPath, JSON.stringify(result, null, 2));
  console.log(JSON.stringify({ passed: true, elapsedSeconds: result.elapsedSeconds, terminalWealth: terminal.terminalWealth, outputPath }));
} finally { await server.close(); }
