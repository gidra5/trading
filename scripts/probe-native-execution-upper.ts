/** Compare an execution-consistent H1 upper bound to saved exact Bellman continuations. */
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { prepareEventExecutionUpper } from "../packages/bot-algo/src/event-execution-upper.js";
const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
const root = path.resolve(__dirname, ".."), directory = (name: string) => path.resolve(root, "data/benchmarks", name);
const source = directory(arg("source")), output = directory(arg("output"));
assert.ok(arg("source") && arg("output") && !fs.existsSync(path.join(output, "summary.json")));
const read = (file: string) => JSON.parse(fs.readFileSync(file, "utf8"));
const hash = (file: string) => createHash("sha256").update(fs.readFileSync(file)).digest("hex");
const config = read(path.join(source, "config.json")), saved = read(path.join(source, "summary.json"));
assert.equal(config.contract, "native-execution-bellman-candidates-v1");
assert.equal(hash(path.join(config.source, "law.json")), config.lawHash);
const law = read(path.join(config.source, "law.json"));
const started = performance.now(), bounds = law.kernels.map((k: any[]) => prepareEventExecutionUpper(k.map(a => ({
  probability: a.probability, path: law.paths[a.path] }))));
const preparationSeconds = (performance.now() - started) / 1000;
const incumbent = Math.max(...saved.results.filter((r: any) => r.complete && r.finite).map((r: any) => r.value));
const results = [], before = performance.now();
for (const candidate of saved.results) {
  assert.ok(candidate.complete && candidate.finite && !candidate.terminalRuin);
  let value = 0, meanSlackBps = 0, maxSlackBps = 0, minimumSlackBps = Infinity;
  const rows = [], begin = performance.now();
  for (const row of candidate.branches) {
    const upper = bounds[row.next](row.account), slackBps = (upper - row.continuation) * 10000;
    assert.ok(upper >= row.continuation - 1e-10, `Invalid upper at request ${candidate.request}, branch ${row.index}`);
    value += row.probability * (row.logGrowth + upper); meanSlackBps += row.probability * slackBps;
    maxSlackBps = Math.max(maxSlackBps, slackBps); minimumSlackBps = Math.min(minimumSlackBps, slackBps);
    rows.push({ index: row.index, next: row.next, exact: row.continuation, upper, slackBps });
  }
  results.push({ request: candidate.request, exactValue: candidate.value, upperValue: value, meanSlackBps,
    maxSlackBps, minimumSlackBps, prunableAgainstIncumbent: value < incumbent,
    seconds: (performance.now() - begin) / 1000, exactSeconds: candidate.seconds, rows });
}
fs.mkdirSync(output, { recursive: true });
fs.writeFileSync(path.join(output, "sources.json"), JSON.stringify(Object.fromEntries([
  "scripts/probe-native-execution-upper.ts", "packages/bot-algo/src/event-execution-upper.ts",
  "packages/bot-algo/src/event-execution-one-step.ts", "packages/bot-algo/src/event-execution-path.ts",
  "packages/bot-algo/src/event-log-policy.ts"].map(file => [file, fs.readFileSync(path.join(root, file), "utf8")])), null, 2));
const summary = { contract: "execution-opening-upper-probe-v1", source, sourceSummaryHash: hash(path.join(source, "summary.json")),
  modelHash: config.modelHash, lawHash: config.lawHash, preparationSeconds,
  querySeconds: (performance.now() - before) / 1000, groups: bounds.map((b: any) => b.groups),
  maxOptimizationGap: Math.max(...bounds.map((b: any) => b.maxOptimizationGap)), incumbent, results,
  scope: "Upper relaxes knowledge of the next opening price, continuous order size and intrabar risk while preserving conditional paths, exact borrowing, fees and opening solvency. It covers above-cap holds/recovery and rounding through an explicit cash credit. Compared with saved full fixed-action H2 values; no root search or future market labels are added." };
fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify(summary, null, 2));
console.log(JSON.stringify({ ...summary, results: results.map(r => ({ ...r, rows: undefined })) }));
