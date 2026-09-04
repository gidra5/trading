/** Check partition bounds/proposals on every predeclared calibration probe in the frozen-law coverage. */
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { prepareEventExecutionPartitions } from "../packages/bot-algo/src/event-execution-partitions.js";
const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
const root = path.resolve(__dirname, ".."), directory = (name: string) => path.resolve(root, "data/benchmarks", name);
const source = directory(arg("source")), output = directory(arg("output"));
assert.ok(arg("source") && arg("output") && !fs.existsSync(output));
const read = (file: string) => JSON.parse(fs.readFileSync(file, "utf8"));
const hash = (file: string) => createHash("sha256").update(fs.readFileSync(file)).digest("hex");
const coverage = read(path.join(source, "summary.json")); assert.equal(coverage.contract, "native-execution-h1-coverage-v1");
const results = [], started = performance.now();
for (const window of coverage.results) {
  const replay = read(path.join(window.replay, "config.json")), compiled = read(path.join(replay.source, "summary.json"));
  assert.equal(hash(path.join(replay.source, "law.json")), window.lawHash);
  assert.equal(hash(path.join(replay.baseSource, "model.json")), window.modelHash);
  assert.equal(hash(path.join(replay.baseSource, "calibration-trades.json")), compiled.calibrationTraceHash);
  const law = read(path.join(replay.source, "law.json")), before = performance.now();
  const prepared = law.kernels.map((k: any[]) => prepareEventExecutionPartitions(k.map(a => ({ probability: a.probability, path: law.paths[a.path] })), 2));
  const preparationSeconds = (performance.now() - before) / 1000, probes = [];
  for (const [index, probe] of compiled.probes.entries()) {
    const rows: any[] = [];
    for (const depth of index % 2 ? [0, 1, 2] : [2, 1, 0]) {
      const begin = performance.now(), result = prepared[probe.leaf](probe.account, depth);
      assert.ok(Number.isFinite(result.lowerValue) && Number.isFinite(result.upperValue));
      rows.push({ depth, lowerValue: result.lowerValue, upperValue: result.upperValue, quantity: result.quantity,
        milliseconds: performance.now() - begin, groups: result.groups.length });
    }
    rows.sort((a, b) => a.depth - b.depth);
    for (const row of rows) {
      assert.ok(row.lowerValue <= rows[0].lowerValue + 1e-10 && row.upperValue >= rows[0].lowerValue - 1e-10);
      row.regretBps = (rows[0].lowerValue - row.lowerValue) * 10000;
      row.slackBps = (row.upperValue - rows[0].lowerValue) * 10000;
      row.gapBps = (row.upperValue - row.lowerValue) * 10000;
    }
    assert.ok(rows[2].upperValue >= rows[1].upperValue - 1e-10);
    probes.push({ index, time: probe.time, leaf: probe.leaf, account: probe.account, rows });
  }
  results.push({ id: window.id, modelHash: window.modelHash, lawHash: window.lawHash,
    compiledSummaryHash: hash(path.join(replay.source, "summary.json")), preparationSeconds, probes });
}
const probes = results.flatMap(r => r.probes), byDepth = [0, 1, 2].map(depth => {
  const rows = probes.map(p => p.rows[depth]);
  return { depth, milliseconds: rows.reduce((s, r) => s + r.milliseconds, 0),
    meanSlackBps: rows.reduce((s, r) => s + r.slackBps, 0) / rows.length, maxSlackBps: Math.max(...rows.map(r => r.slackBps)),
    maxRegretBps: Math.max(...rows.map(r => r.regretBps)), exactValueProposals: rows.filter(r => Math.abs(r.regretBps) < 1e-8).length,
    certifiedAt001Bps: rows.filter(r => r.gapBps <= .001).length };
});
const summary = { contract: "native-execution-partition-calibration-coverage-v1", source, sourceHash: hash(path.join(source, "summary.json")),
  windows: results.length, probes: probes.length, byDepth, preparationSeconds: results.reduce((s, r) => s + r.preparationSeconds, 0),
  elapsedSeconds: (performance.now() - started) / 1000, results,
  scope: "All predeclared first-leaf/held calibration probes for the 28 unchanged laws. Exact H1 reference versus two/four conditional opening groups, with executable full-law candidate scores. This is a computational/value-bound comparison; test-window returns and forecasting models are unchanged." };
fs.mkdirSync(output, { recursive: true });
fs.writeFileSync(path.join(output, "sources.json"), JSON.stringify(Object.fromEntries([
  "scripts/audit-native-execution-partitions.ts", "packages/bot-algo/src/event-execution-partitions.ts",
  "packages/bot-algo/src/event-execution-one-step.ts", "packages/bot-algo/src/event-execution-path.ts"]
  .map(file => [file, fs.readFileSync(path.join(root, file), "utf8")])), null, 2));
fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify(summary, null, 2));
console.log(JSON.stringify({ ...summary, results: undefined }));
