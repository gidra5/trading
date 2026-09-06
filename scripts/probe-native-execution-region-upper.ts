/** Diagnose a common-root H2 upper at fixed requests over unresolved regions. */
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { prepareEventExecutionUpper } from "../packages/bot-algo/src/event-execution-upper.js";
import { evaluateEventExecutionPath } from "../packages/bot-algo/src/event-execution-path.js";

const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
const root = path.resolve(__dirname, ".."), directory = (name: string) => path.resolve(root, "data/benchmarks", name);
const samples = directory(arg("samples")), output = directory(arg("output"));
const points = Number(arg("points") || "33");
assert.ok(arg("samples") && arg("output") && !fs.existsSync(output));
assert.ok(Number.isInteger(points) && points >= 3 && points <= 1001);
const read = (file: string) => JSON.parse(fs.readFileSync(file, "utf8"));
const hash = (file: string) => createHash("sha256").update(fs.readFileSync(file)).digest("hex");
const sampled = read(path.join(samples, "summary.json"));
assert.equal(sampled.contract, "native-execution-root-region-samples-v1"); assert.equal(sampled.completed, true);
const source = sampled.source, config = read(path.join(source, "config.json"));
assert.equal(hash(path.join(config.source, "law.json")), sampled.lawHash);
const law = read(path.join(config.source, "law.json")), step = law.costs.quantityStep;
const kernels = law.kernels.map((kernel: any[]) => kernel.map(atom => ({
  probability: atom.probability, next: atom.next, path: law.paths[atom.path],
})));
const bounds = kernels.map((kernel: any[]) => prepareEventExecutionUpper(kernel));
assert.ok(bounds.every(bound => bound.supported));
fs.mkdirSync(output, { recursive: true });
const save = (name: string, value: unknown) => fs.writeFileSync(path.join(output, name), JSON.stringify(value,
  (_, v) => typeof v === "number" && !Number.isFinite(v) ? String(v) : v, 2));
save("sources.json", Object.fromEntries([
  "scripts/probe-native-execution-region-upper.ts", "packages/bot-algo/src/event-execution-upper.ts",
  "packages/bot-algo/src/event-execution-one-step.ts", "packages/bot-algo/src/event-execution-path.ts",
  "packages/bot-algo/src/event-log-policy.ts",
].map(file => [file, fs.readFileSync(path.join(root, file), "utf8")])));
const started = performance.now(), fixedUpper = (lot: number) => {
  let value = 0;
  for (const atom of kernels[config.probe.leaf]) {
    const next = evaluateEventExecutionPath(atom.path, config.probe.account, lot * step);
    if (!Number.isFinite(next.logGrowth)) return -Infinity;
    const continuation = bounds[atom.next]({ equity: next.equity, price: next.price, exposure: next.exposure });
    if (continuation === Infinity) return Infinity;
    if (continuation === -Infinity) return -Infinity;
    value += atom.probability * (next.logGrowth + continuation);
  }
  return value;
};
const exact = new Map(sampled.results.map((r: any) => [r.lot, r.value]));
const results = [];
for (const [lo, hi] of sampled.regions as Array<[number, number]>) {
  const lots = [...new Set(Array.from({ length: points }, (_, i) => Math.round(lo + (hi - lo) * i / (points - 1)))
    .concat(sampled.results.filter((r: any) => r.lot >= lo && r.lot <= hi).map((r: any) => r.lot)))].sort((a, b) => a - b);
  const rows = lots.map(lot => {
    const upperValue = fixedUpper(lot), exactValue = exact.get(lot) as number | undefined;
    if (exactValue !== undefined) assert.ok(upperValue >= exactValue - 1e-10);
    return { lot, request: lot * step, upperValue, exactValue: exactValue ?? null,
      exactSlackBps: exactValue === undefined ? null : (upperValue - exactValue) * 10000 };
  });
  const best = rows.reduce((a, b) => b.upperValue > a.upperValue ? b : a);
  const result = { region: [lo, hi], sampledLots: rows.length, maximum: best,
    belowIncumbentAtEverySample: best.upperValue <= sampled.incumbent, rows };
  results.push(result); console.log(JSON.stringify({ ...result, rows: undefined }));
}
save("summary.json", {
  contract: "native-execution-root-region-upper-samples-v1", samples, samplesHash: hash(path.join(samples, "summary.json")),
  source, sourceSummaryHash: sampled.sourceSummaryHash, lawHash: sampled.lawHash, modelHash: sampled.modelHash,
  scope: "Fixed-root H2 information upper sampled on an evenly spaced grid. The child controller may observe its next opening before choosing a continuous action. Each point is an upper for that fixed executable root request, but interpolation between points is not certified and this artifact does not prove a region or global optimum.",
  account: config.probe.account, leaf: config.probe.leaf, step, incumbent: sampled.incumbent,
  incumbentRequest: sampled.incumbentRequest, points, results, elapsedSeconds: (performance.now() - started) / 1000,
});
