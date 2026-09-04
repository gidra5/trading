/** Eliminate complete guarded regions whose first requests all leave the same account as hold. */
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { evaluateEventExecutionPath } from "../packages/bot-algo/src/event-execution-path.js";
const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
const root = path.resolve(__dirname, ".."), directory = (name: string) => path.resolve(root, "data/benchmarks", name);
const source = directory(arg("source")), output = directory(arg("output"));
assert.ok(arg("source") && arg("output") && !fs.existsSync(output));
const read = (file: string) => JSON.parse(fs.readFileSync(file, "utf8"));
const hash = (file: string) => createHash("sha256").update(fs.readFileSync(file)).digest("hex");
const captured = read(path.join(source, "summary.json")); assert.equal(captured.contract, "native-execution-root-region-probe-v1");
const results = [], started = performance.now();
for (const row of captured.results) {
  const config = read(path.join(row.source, "config.json")), saved = read(path.join(row.source, "summary.json"));
  assert.equal(hash(path.join(row.source, "summary.json")), row.sourceSummaryHash);
  assert.equal(hash(path.join(config.source, "law.json")), row.lawHash);
  const law = read(path.join(config.source, "law.json")), kernel = law.kernels[config.probe.leaf].map((a: any) => ({
    ...a, path: law.paths[a.path] })), account = config.probe.account, step = law.costs.quantityStep;
  const hold = saved.results.find((r: any) => r.request === 0); assert.ok(hold.complete && hold.finite);
  const incumbent = Math.max(...saved.results.map((r: any) => r.value));
  const upper = hold.value + 1e-10; assert.ok(upper < incumbent);
  const reference = kernel.map((a: any) => evaluateEventExecutionPath(a.path, account, 0));
  const pruned: Array<readonly [number, number]> = [], remaining: Array<readonly [number, number]> = [];
  let compared = 0;
  for (const region of row.cover.requestRegions as Array<[number, number]>) {
    const [lo, hi] = region;
    let equivalent = true;
    for (const lot of new Set([lo, Math.floor((lo + hi) / 2), hi])) {
      for (let i = 0; i < kernel.length; i++) {
        const next = evaluateEventExecutionPath(kernel[i].path, account, lot * step), zero = reference[i]; compared++;
        if (next.filledQuantity || next.equity !== zero.equity || next.price !== zero.price || next.exposure !== zero.exposure) {
          equivalent = false; break;
        }
      }
      if (!equivalent) break;
    }
    (equivalent ? pruned : remaining).push(region);
  }
  const lots = (regions: Array<readonly [number, number]>) => regions.reduce((s, [lo, hi]) => s + hi - lo + 1, 0);
  results.push({ source: row.source, sourceSummaryHash: row.sourceSummaryHash, lawHash: row.lawHash,
    holdValue: hold.value, upperValue: upper, incumbent, scope: "Hold-equivalent guarded H2 root regions only",
    maximumLots: row.cover.search.maximumLots, totalRegions: row.cover.requestRegions.length,
    pruned, remaining, prunedLots: lots(pruned), remainingLots: lots(remaining), comparedTransitions: compared,
    outsideMaximum: "Every request beyond the finite maximum fails the size cap in all outcomes and shares this same hold value." });
  console.log(JSON.stringify({ source: path.basename(row.source), totalRegions: row.cover.requestRegions.length,
    prunedRegions: pruned.length, remainingRegions: remaining.length, prunedLots: lots(pruned), remainingLots: lots(remaining),
    upperValue: upper, incumbent }));
}
fs.mkdirSync(output, { recursive: true });
fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify({ contract: "native-execution-hold-region-pruning-v1", source,
  sourceSummaryHash: hash(path.join(source, "summary.json")), results, elapsedSeconds: (performance.now() - started) / 1000,
  proof: "The captured non-singleton regions have constant acceptance and funding branches. Unfilled requests at all checked guards/endpoints/midpoints leave exactly the hold successor account for every outcome; thus their complete H2 values equal the previously fully evaluated hold request. These regions and requests outside the maximum cannot beat the verified incumbent. Other regions remain unsearched." }, null, 2));
fs.writeFileSync(path.join(output, "sources.json"), JSON.stringify(Object.fromEntries([
  "scripts/prune-native-execution-hold-regions.ts", "packages/bot-algo/src/event-execution-one-step.ts", "packages/bot-algo/src/event-execution-path.ts"]
  .map(file => [file, fs.readFileSync(path.join(root, file), "utf8")])), null, 2));
