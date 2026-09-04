/** Check the new H1 continuation upper against all source-bound saved H1 optima. */
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { prepareEventExecutionUpper } from "../packages/bot-algo/src/event-execution-upper.js";
const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
const root = path.resolve(__dirname, ".."), directory = (name: string) => path.resolve(root, "data/benchmarks", name);
const source = directory(arg("source")), output = directory(arg("output"));
assert.ok(arg("source") && arg("output") && !fs.existsSync(output));
const read = (file: string) => JSON.parse(fs.readFileSync(file, "utf8"));
const hash = (file: string) => createHash("sha256").update(fs.readFileSync(file)).digest("hex");
const coverage = read(path.join(source, "summary.json")); assert.equal(coverage.contract, "native-execution-h1-coverage-v1");
fs.mkdirSync(output, { recursive: true });
const results = [], started = performance.now();
for (const row of coverage.results) {
  const config = read(path.join(row.replay, "config.json"));
  assert.equal(config.modelHash, row.modelHash); assert.equal(config.lawHash, row.lawHash);
  assert.equal(hash(path.join(config.source, "law.json")), row.lawHash);
  const file = path.join(row.replay, "execution-trades.json"); assert.equal(hash(file), row.traceHash);
  const trace = read(file), law = read(path.join(config.source, "law.json")), begin = performance.now();
  const bounds = law.kernels.map((k: any[]) => prepareEventExecutionUpper(k.map(a => ({ probability: a.probability, path: law.paths[a.path] }))));
  const preparationSeconds = (performance.now() - begin) / 1000, queried = performance.now();
  let minimumSlackBps = Infinity, maximumSlackBps = 0, totalSlack = 0;
  const values = trace.map((decision: any) => {
    assert.ok(decision.order.complete && decision.order.feasible && Number.isFinite(decision.order.value));
    const upper = bounds[decision.leaf]({ equity: decision.equityBefore, price: decision.order.price, exposure: decision.exposureBefore });
    const slackBps = (upper - decision.order.value) * 10000;
    assert.ok(Number.isFinite(upper) && slackBps >= -1e-6, `${row.id} at ${decision.time}: ${slackBps}`);
    minimumSlackBps = Math.min(minimumSlackBps, slackBps); maximumSlackBps = Math.max(maximumSlackBps, slackBps); totalSlack += slackBps;
    return { time: decision.time, leaf: decision.leaf, value: decision.order.value, upper, slackBps };
  });
  assert.equal(trace.length, row.decisions);
  results.push({ id: row.id, decisions: trace.length, modelHash: row.modelHash, lawHash: row.lawHash, traceHash: row.traceHash,
    preparationSeconds, querySeconds: (performance.now() - queried) / 1000, minimumSlackBps, maximumSlackBps,
    meanSlackBps: totalSlack / trace.length, groups: bounds.map((b: any) => b.groups), values });
}
const summary = { contract: "native-execution-upper-coverage-v1", source, sourceHash: hash(path.join(source, "summary.json")),
  windows: results.length, decisions: results.reduce((s, r) => s + r.decisions, 0),
  minimumSlackBps: Math.min(...results.map(r => r.minimumSlackBps)), maximumSlackBps: Math.max(...results.map(r => r.maximumSlackBps)),
  preparationSeconds: results.reduce((s, r) => s + r.preparationSeconds, 0), querySeconds: results.reduce((s, r) => s + r.querySeconds, 0),
  elapsedSeconds: (performance.now() - started) / 1000, results,
  scope: "The analytical relaxation dominates every saved globally optimized marked H1 value. This checks 27 strict-source windows and the declared March scenario without new policy replays or forecast fitting. The upper is optimistic information and execution, not an executable strategy or a stationary Bellman certificate." };
assert.equal(summary.windows, coverage.windows); assert.equal(summary.decisions, coverage.decisions);
fs.writeFileSync(path.join(output, "sources.json"), JSON.stringify(Object.fromEntries([
  "scripts/audit-native-execution-upper.ts", "packages/bot-algo/src/event-execution-upper.ts",
  "packages/bot-algo/src/event-execution-one-step.ts"].map(file => [file, fs.readFileSync(path.join(root, file), "utf8")])), null, 2));
fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify(summary, null, 2));
console.log(JSON.stringify({ ...summary, results: undefined }));
