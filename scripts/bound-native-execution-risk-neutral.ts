/** Bound guarded H2 root regions with a convex risk-neutral/Jensen relaxation. */
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { prepareEventExecutionRiskNeutralUpper } from "../packages/bot-algo/src/event-execution-risk-upper.js";
import { evaluateEventExecutionPath } from "../packages/bot-algo/src/event-execution-path.js";

const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
const root = path.resolve(__dirname, ".."), directory = (name: string) => path.resolve(root, "data/benchmarks", name);
const pruning = directory(arg("pruning")), singleton = directory(arg("singletons")), output = directory(arg("output"));
const requestBins = Number(arg("bins") || "1");
assert.ok(arg("pruning") && arg("singletons") && arg("output") && !fs.existsSync(output));
assert.ok(Number.isInteger(requestBins) && requestBins >= 1 && requestBins <= 129);
const read = (file: string) => JSON.parse(fs.readFileSync(file, "utf8"));
const hash = (file: string) => createHash("sha256").update(fs.readFileSync(file)).digest("hex");
const pruned = read(path.join(pruning, "summary.json")), classified = read(path.join(singleton, "summary.json"));
assert.equal(pruned.contract, "native-execution-hold-region-pruning-v1");
assert.equal(classified.contract, "native-execution-singleton-h2-probe-v1");
assert.equal(classified.pruningHash, hash(path.join(pruning, "summary.json")));
const row = pruned.results.find((candidate: any) => candidate.source === classified.source); assert.ok(row);
const source = row.source, config = read(path.join(source, "config.json"));
const lawFile = path.join(config.source, "law.json"); assert.equal(hash(lawFile), row.lawHash);
const law = read(lawFile), step = law.costs.quantityStep;
const kernels = law.kernels.map((kernel: any[]) => kernel.map(atom => ({
  probability: atom.probability, next: atom.next, path: law.paths[atom.path],
})));
const bounds = kernels.map((kernel: any[]) => prepareEventExecutionRiskNeutralUpper(kernel, { requestBins }));
const regions = (row.remaining as Array<[number, number]>).filter(([lo, hi]) => lo < hi);
assert.deepEqual(regions, classified.nonSingletonRegions);
fs.mkdirSync(output, { recursive: true });
const save = (name: string, value: unknown) => fs.writeFileSync(path.join(output, name), JSON.stringify(value,
  (_, value) => typeof value === "number" && !Number.isFinite(value) ? String(value) : value, 2));
save("sources.json", Object.fromEntries([
  "scripts/bound-native-execution-risk-neutral.ts", "packages/bot-algo/src/event-execution-risk-upper.ts",
  "packages/bot-algo/src/event-execution-one-step.ts", "packages/bot-algo/src/event-execution-path.ts",
  "packages/bot-algo/src/event-log-policy.ts",
].map(file => [file, fs.readFileSync(path.join(root, file), "utf8")])));
const started = performance.now(), results = [];
for (const region of regions) {
  const begin = performance.now(), endpoints = [];
  for (const lot of region) {
    const request = lot * step;
    let expectedTerminalEquity = 0, finite = true;
    for (const atom of kernels[config.probe.leaf]) {
      if (!atom.probability) continue;
      const next = evaluateEventExecutionPath(atom.path, config.probe.account, request);
      if (!Number.isFinite(next.logGrowth)) { finite = false; break; }
      const upper = bounds[atom.next]({ equity: next.equity, price: next.price, exposure: next.exposure });
      if (!Number.isFinite(upper)) { finite = false; break; }
      expectedTerminalEquity += atom.probability * upper;
    }
    const upperValue = finite ? Math.log(expectedTerminalEquity / config.probe.account.equity) + 2e-10 : Infinity;
    endpoints.push({ lot, request, expectedTerminalEquity, upperValue });
  }
  // On a guarded root region every first transition is affine in the root
  // request. The compiled child bound is a maximum of affine wealth functions,
  // hence convex along that segment. Its probability-weighted sum is convex,
  // so the region maximum is attained at one of these endpoints. Jensen then
  // upper-bounds the complete expected-log objective.
  const upperValue = Math.max(...endpoints.map(endpoint => endpoint.upperValue));
  const result = { region, requests: region.map(lot => lot * step), endpoints, upperValue,
    upperBps: upperValue * 10000, pruned: upperValue <= classified.incumbent,
    seconds: (performance.now() - begin) / 1000 };
  results.push(result); console.log(JSON.stringify(result));
}
save("summary.json", { contract: "native-execution-risk-neutral-root-bound-v1", pruning,
  pruningHash: hash(path.join(pruning, "summary.json")), singleton,
  singletonHash: hash(path.join(singleton, "summary.json")), source,
  sourceSummaryHash: row.sourceSummaryHash, lawHash: row.lawHash, modelHash: classified.modelHash,
  account: config.probe.account, leaf: config.probe.leaf, step, incumbent: classified.incumbent,
  incumbentRequest: classified.incumbentRequest, requestBins, regions,
  scope: "Guarded non-singleton H2 root regions under the unchanged native one-second law. One common child request is retained. Child acceptance, funding, size, cap, availability and maintenance are relaxed only upward in arithmetic terminal equity. Convexity places the common-root maximum at a region endpoint; Jensen converts full-tree expected equity to an expected-log upper.",
  results, completed: results.length === regions.length, elapsedSeconds: (performance.now() - started) / 1000 });
