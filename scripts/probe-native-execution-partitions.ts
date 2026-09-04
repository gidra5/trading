/** Profile nested information relaxations only on predeclared saved calibration accounts. */
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { prepareEventExecutionPartitions } from "../packages/bot-algo/src/event-execution-partitions.js";
import { prepareEventExecutionUpper } from "../packages/bot-algo/src/event-execution-upper.js";
const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
const root = path.resolve(__dirname, ".."), directory = (name: string) => path.resolve(root, "data/benchmarks", name);
const source = directory(arg("source")), output = directory(arg("output"));
assert.ok(arg("source") && arg("output") && !fs.existsSync(path.join(output, "summary.json")));
const read = (file: string) => JSON.parse(fs.readFileSync(file, "utf8"));
const hash = (file: string) => createHash("sha256").update(fs.readFileSync(file)).digest("hex");
const config = read(path.join(source, "config.json")), saved = read(path.join(source, "summary.json"));
assert.equal(config.contract, "native-execution-bellman-candidates-v1"); assert.equal(hash(path.join(config.source, "law.json")), config.lawHash);
const law = read(path.join(config.source, "law.json")), kernels = law.kernels.map((k: any[]) => k.map(a => ({ probability: a.probability, path: law.paths[a.path] })));
const cases = new Map<string, any>(), add = (leaf: number, account: any, value: number) => {
  const bucket = !account.exposure ? "cash" : Math.abs(account.exposure) > law.costs.maxLeverage ? "above-cap" : "invested";
  const key = `${leaf}:${bucket}`;
  if (!cases.has(key)) cases.set(key, { key, leaf, account, exactValue: value });
};
add(config.probe.leaf, config.probe.account, config.oneStep.value);
for (const r of saved.results) for (const b of r.branches) add(b.next, b.account, b.continuation);
const started = performance.now(), solvers = kernels.map((k: any[]) => prepareEventExecutionPartitions(k, 4));
const analytical = kernels.map((k: any[]) => prepareEventExecutionUpper(k));
const preparationSeconds = (performance.now() - started) / 1000, results = [];
for (const [index, row] of [...cases.values()].entries()) {
  const depths: any[] = [];
  for (const depth of index % 2 ? [0, 1, 2, 3, 4] : [4, 3, 2, 1, 0]) {
    const timings = [], begin = performance.now(); let result;
    for (let repeat = 0; repeat < 3; repeat++) {
      const start = performance.now(); result = solvers[row.leaf](row.account, depth); timings.push(performance.now() - start);
    }
    assert.ok(result!.lowerValue <= row.exactValue + 1e-10 && result!.upperValue >= row.exactValue - 1e-10);
    if (!depth) assert.ok(Math.abs(result!.lowerValue - row.exactValue) < 1e-12);
    depths.push({ depth, groups: result!.groups.length, candidates: result!.candidates.length, complete: result!.complete,
      lowerRegretBps: (row.exactValue - result!.lowerValue) * 10000, upperSlackBps: (result!.upperValue - row.exactValue) * 10000,
      gapBps: result!.gap * 10000, quantity: result!.quantity, evaluatedOrders: result!.groups.reduce((s, g) => s + g.search.evaluatedOrders, 0),
      medianMs: timings.sort((a, b) => a - b)[1], elapsedMs: performance.now() - begin });
  }
  depths.sort((a, b) => a.depth - b.depth);
  for (let i = 1; i < depths.length; i++) assert.ok(depths[i].upperSlackBps >= depths[i - 1].upperSlackBps - 1e-6);
  results.push({ ...row, analyticalSlackBps: (analytical[row.leaf](row.account) - row.exactValue) * 10000, depths });
  console.log(JSON.stringify({ key: row.key, depths }));
}
fs.mkdirSync(output, { recursive: true });
fs.writeFileSync(path.join(output, "sources.json"), JSON.stringify(Object.fromEntries([
  "scripts/probe-native-execution-partitions.ts", "packages/bot-algo/src/event-execution-partitions.ts",
  "packages/bot-algo/src/event-execution-one-step.ts", "packages/bot-algo/src/event-execution-path.ts"]
  .map(file => [file, fs.readFileSync(path.join(root, file), "utf8")])), null, 2));
const summary = { contract: "native-execution-partition-probe-v1", source, sourceSummaryHash: hash(path.join(source, "summary.json")),
  modelHash: config.modelHash, lawHash: config.lawHash, selection: "First saved calibration occurrence of each leaf and cash/invested/above-cap bucket, including the root. No performance selection.",
  preparationSeconds, results, elapsedSeconds: (performance.now() - started) / 1000 };
fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify(summary, null, 2));
