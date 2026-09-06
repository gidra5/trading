/** Bound H2 root regions with linked mean-equity continuation segments. */
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { prepareEventExecutionMeanSegmentUpper } from "../packages/bot-algo/src/event-execution-mean-segment.js";
import { evaluateEventExecutionPath } from "../packages/bot-algo/src/event-execution-path.js";

const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
const root = path.resolve(__dirname, ".."), directory = (name: string) => path.resolve(root, "data/benchmarks", name);
const pruning = directory(arg("pruning")), singleton = directory(arg("singletons")), output = directory(arg("output"));
const offset = Number(arg("offset") || "0"), count = Number(arg("count") || "999");
assert.ok(arg("pruning") && arg("singletons") && arg("output") && !fs.existsSync(output));
assert.ok(Number.isInteger(offset) && offset >= 0 && Number.isInteger(count) && count > 0);
const read = (file: string) => JSON.parse(fs.readFileSync(file, "utf8"));
const hash = (file: string) => createHash("sha256").update(fs.readFileSync(file)).digest("hex");
const pruned = read(path.join(pruning, "summary.json")), classified = read(path.join(singleton, "summary.json"));
assert.equal(pruned.contract, "native-execution-hold-region-pruning-v1");
assert.equal(classified.contract, "native-execution-singleton-h2-probe-v1");
assert.equal(classified.pruningHash, hash(path.join(pruning, "summary.json")));
const row = pruned.results.find((candidate: any) => candidate.source === classified.source); assert.ok(row);
const source = row.source, config = read(path.join(source, "config.json")), lawFile = path.join(config.source, "law.json");
assert.equal(hash(lawFile), row.lawHash);
const law = read(lawFile), step = law.costs.quantityStep;
const kernels = law.kernels.map((kernel: any[]) => kernel.map(atom => ({
  probability: atom.probability, next: atom.next, path: law.paths[atom.path],
})));
const bounds = kernels.map((kernel: any[]) => prepareEventExecutionMeanSegmentUpper(kernel));
const allRegions = (row.remaining as Array<[number, number]>).filter(([lo, hi]) => lo < hi);
assert.deepEqual(allRegions, classified.nonSingletonRegions);
const requestedRegions = arg("regions") ? arg("regions").split(",").map(value => value.split(":").map(Number) as [number, number]) : [];
assert.ok(requestedRegions.every(([lo, hi]) => Number.isSafeInteger(lo) && Number.isSafeInteger(hi) && lo <= hi
  && allRegions.some(([outerLo, outerHi]) => lo >= outerLo && hi <= outerHi)));
const regions = requestedRegions.length ? requestedRegions : allRegions.slice(offset, offset + count);
fs.mkdirSync(output, { recursive: true });
const save = (name: string, value: unknown) => fs.writeFileSync(path.join(output, name), JSON.stringify(value,
  (_, value) => typeof value === "number" && !Number.isFinite(value) ? String(value) : value, 2));
save("sources.json", Object.fromEntries([
  "scripts/bound-native-execution-mean-segments.ts", "packages/bot-algo/src/event-execution-mean-segment.ts",
  "packages/bot-algo/src/event-execution-acceptance.ts", "packages/bot-algo/src/event-execution-one-step.ts",
  "packages/bot-algo/src/event-execution-path.ts", "packages/bot-algo/src/event-log-policy.ts",
].map(file => [file, fs.readFileSync(path.join(root, file), "utf8")])));
const started = performance.now(), results: any[] = [];
const checkpoint = () => save("summary.json", { contract: "native-execution-mean-segment-root-bound-v1",
  pruning, pruningHash: hash(path.join(pruning, "summary.json")), singleton,
  singletonHash: hash(path.join(singleton, "summary.json")), source,
  sourceSummaryHash: row.sourceSummaryHash, lawHash: row.lawHash, modelHash: classified.modelHash,
  account: config.probe.account, leaf: config.probe.leaf, step, incumbent: classified.incumbent,
  incumbentRequest: classified.incumbentRequest, allRegions, offset, count, requestedRegions, regions,
  scope: "Guarded H2 root regions under the unchanged native one-second law. Each successor uses one child request and segment-wide possible/certain ordered acceptance. Arithmetic terminal-equity upper functions retain the same root endpoint across every first outcome; convexity places the segment maximum at an endpoint. One full-tree Jensen step produces the expected-log upper.",
  results, completed: results.length === regions.length, elapsedSeconds: (performance.now() - started) / 1000 });
for (const region of regions) {
  const begin = performance.now(), grouped = new Map<string, any>();
  for (const atom of kernels[config.probe.leaf]) {
    const transitions = region.map(lot => evaluateEventExecutionPath(atom.path, config.probe.account, lot * step));
    assert.ok(transitions.every(next => Number.isFinite(next.logGrowth)));
    const accounts = transitions.map(next => ({ equity: next.equity, price: next.price, exposure: next.exposure }));
    const key = JSON.stringify([atom.next, accounts]);
    const existing = grouped.get(key);
    if (existing) existing.probability += atom.probability;
    else grouped.set(key, { probability: atom.probability, next: atom.next, accounts });
  }
  const endpointExpectedEquities = [0, 0], branches = []; let evaluatedOrders = 0, intervals = 0;
  for (const branch of grouped.values()) {
    assert.equal(branch.accounts[0].price, branch.accounts[1].price);
    const vertices = branch.accounts.map((account: any) => [account.equity * (1 - account.exposure),
      account.exposure * account.equity / account.price]) as [[number, number], [number, number]];
    const bounded = bounds[branch.next]({ price: branch.accounts[0].price, vertices });
    assert.equal(bounded.complete, true); evaluatedOrders += bounded.search.evaluatedOrders; intervals += bounded.search.intervals;
    for (let endpoint = 0; endpoint < 2; endpoint++)
      endpointExpectedEquities[endpoint] += branch.probability * bounded.endpointUpperTerminalEquities[endpoint];
    branches.push({ probability: branch.probability, next: branch.next, vertices,
      endpointUpperTerminalEquities: bounded.endpointUpperTerminalEquities,
      optimisticRequests: bounded.optimisticRequests, search: bounded.search });
  }
  const endpointUpperValues = endpointExpectedEquities.map(equity => Math.log(equity / config.probe.account.equity) + 2e-10);
  const upperValue = Math.max(...endpointUpperValues), result = { region, requests: region.map(lot => lot * step),
    groups: grouped.size, endpointExpectedEquities, endpointUpperValues, upperValue, upperBps: upperValue * 10000,
    pruned: upperValue <= classified.incumbent, evaluatedOrders, intervals,
    seconds: (performance.now() - begin) / 1000, branches };
  results.push(result); checkpoint(); console.log(JSON.stringify({ ...result, branches: undefined }));
}
checkpoint();
