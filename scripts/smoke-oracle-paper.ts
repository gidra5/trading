/** Real HTTP -> persistent Python worker -> checkpoint/restart integration check.
 * Run with an explicit worker config and a short causal observation fixture.
 * This uses paper execution only and never instantiates an exchange client.
 */
import assert from "node:assert/strict";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { parseArgs } from "node:util";
import Fastify from "fastify";
import { serializerCompiler, validatorCompiler } from "fastify-type-provider-zod";
import { OraclePaperService } from "../apps/server/src/oracle-paper-service.js";
import { oraclePaperRoutes } from "../apps/server/src/oracle-paper-routes.js";
import type { DenseDecision, DenseInitialization, DenseObservation } from "../apps/server/src/oracle-dense-controller.js";

async function main() {
const { values } = parseArgs({ options: {
  config: { type: "string" }, input: { type: "string" }, output: { type: "string" }, python: { type: "string" },
} });
assert(values.config && values.input && values.output, "Required: --config --input --output (new directory)");
const output = path.resolve(values.output);
const configPath = path.resolve(values.config);
const input = JSON.parse(await readFile(values.input, "utf8")) as {
  readonly initial: DenseInitialization; readonly ticks: readonly DenseObservation[];
};
assert(input.ticks.length >= 2, "Need at least two ticks to exercise restart");
assert.equal(input.initial.terminalSeconds, input.ticks.at(-1)!.timestamps[1]);
await mkdir(path.dirname(output), { recursive: true });
await mkdir(output); // Refuse to reuse or overwrite a previous experiment.
const stateDirectory = path.join(output, "state");
const options = { symbol: "BTCUSDT", checkpointEvery: 1, worker: {
  pythonExecutable: path.resolve(values.python ?? ".venv-ml/Scripts/python.exe"),
  configPath, cwd: process.cwd(), stateDirectory, initializationTimeoutMs: 300000, tickTimeoutMs: 300000,
} };
const records: { readonly operation: string; readonly seconds: number; readonly response: unknown }[] = [];
const started = performance.now();
async function server() {
  const service = new OraclePaperService(options);
  const app = Fastify();
  app.setValidatorCompiler(validatorCompiler); app.setSerializerCompiler(serializerCompiler);
  app.addHook("onClose", async () => service.close());
  await app.register(oraclePaperRoutes, { service });
  const url = await app.listen({ host: "127.0.0.1", port: 0 });
  return { app, url };
}
let active = await server();
async function request(operation: string, body?: unknown): Promise<DenseDecision> {
  const before = performance.now();
  const response = await fetch(`${active.url}/bot/oracle-paper/${operation}`, {
    method: "POST", headers: body === undefined ? undefined : { "content-type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body), signal: AbortSignal.timeout(360000),
  });
  const result = await response.json();
  records.push({ operation, seconds: (performance.now() - before) / 1000, response: result });
  assert.equal(response.status, 200, JSON.stringify(result));
  return result as DenseDecision;
}
try {
  await request("start", input.initial);
  const boundary = await request("tick", input.ticks[0]);
  assert.equal(boundary.phase, "active");
  await active.app.close();
  active = await server();
  const restored = await request("resume");
  assert.equal(restored.update, boundary.update);
  assert.equal(restored.timestamp, boundary.timestamp);
  assert.deepEqual(restored.account, boundary.account);
  let final = restored;
  for (const tick of input.ticks.slice(1)) final = await request("tick", tick);
  assert.equal(final.phase, "complete");
  assert.equal(final.timestamp, input.initial.terminalSeconds);
  assert.equal(final.account.asset, 0);
  assert.deepEqual(final.account.orders, []);
  assert.equal(final.terminalWealth, final.account.quote);
  await writeFile(path.join(output, "results.json"), JSON.stringify({
    seconds: (performance.now() - started) / 1000, configPath, records,
    scope: "Real local HTTP, NSF worker, saved account recovery and terminal flatness; not forecasting or profitability accuracy.",
  }, null, 2));
  console.log(JSON.stringify({ output, seconds: (performance.now() - started) / 1000, terminalWealth: final.terminalWealth }));
} catch (error) {
  await writeFile(path.join(output, "failure.json"), JSON.stringify({ error: String(error), records }, null, 2));
  throw error;
} finally { await active.app.close(); }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
