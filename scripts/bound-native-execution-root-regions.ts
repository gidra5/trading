/** Bound every remaining non-singleton H2 root region with linked child-account segments. */
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { prepareEventExecutionBoxUpper } from "../packages/bot-algo/src/event-execution-box.js";
import { prepareEventExecutionOneStep } from "../packages/bot-algo/src/event-execution-one-step.js";
import { evaluateEventExecutionPath } from "../packages/bot-algo/src/event-execution-path.js";

const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
const root = path.resolve(__dirname, ".."), directory = (name: string) => path.resolve(root, "data/benchmarks", name);
const pruning = directory(arg("pruning")), singleton = directory(arg("singletons")), output = directory(arg("output"));
const refinements = Number(arg("refinements") || "10000"), toleranceBps = Number(arg("tolerance-bps") || ".001");
assert.ok(arg("pruning") && arg("singletons") && arg("output") && !fs.existsSync(output));
assert.ok(Number.isInteger(refinements) && refinements >= 0 && Number.isFinite(toleranceBps) && toleranceBps >= 0);
const read = (file: string) => JSON.parse(fs.readFileSync(file, "utf8"));
const hash = (file: string) => createHash("sha256").update(fs.readFileSync(file)).digest("hex");
const pruned = read(path.join(pruning, "summary.json")), classified = read(path.join(singleton, "summary.json"));
assert.equal(pruned.contract, "native-execution-hold-region-pruning-v1");
assert.equal(classified.contract, "native-execution-singleton-h2-probe-v1");
assert.equal(classified.pruningHash, hash(path.join(pruning, "summary.json")));
assert.equal(classified.completed, true);
const row = pruned.results.find((r: any) => r.source === classified.source); assert.ok(row);
assert.equal(row.lawHash, classified.lawHash); assert.equal(row.sourceSummaryHash, classified.sourceSummaryHash);
const source = row.source, config = read(path.join(source, "config.json")), saved = read(path.join(source, "summary.json"));
assert.equal(hash(path.join(config.source, "law.json")), row.lawHash);
const law = read(path.join(config.source, "law.json")), step = law.costs.quantityStep;
const kernels = law.kernels.map((kernel: any[]) => kernel.map(atom => ({
  probability: atom.probability, next: atom.next, path: law.paths[atom.path],
})));
const exact = kernels.map((kernel: any[]) => prepareEventExecutionOneStep(kernel));
const bounds = kernels.map((kernel: any[]) => prepareEventExecutionBoxUpper(kernel, {
  ordered: true, coupledWealth: true, maxRefinements: refinements, valueTolerance: toleranceBps / 10000,
}));
const stateKey = (leaf: number, account: any) => JSON.stringify([leaf, account.equity, account.price, account.exposure]);
const exactCache = new Map<string, { value: number; quantity: number }>();
for (const result of saved.results) for (const branch of result.branches) {
  assert.ok(branch.complete); exactCache.set(stateKey(branch.next, branch.account),
    { value: branch.continuation, quantity: branch.continuationRequest });
}
const nonSingletons = (row.remaining as Array<[number, number]>).filter(([lo, hi]) => lo < hi);
assert.deepEqual(nonSingletons, classified.nonSingletonRegions);
fs.mkdirSync(output, { recursive: true });
const save = (name: string, value: unknown) => fs.writeFileSync(path.join(output, name), JSON.stringify(value,
  (_, v) => typeof v === "number" && !Number.isFinite(v) ? String(v) : v, 2));
save("sources.json", Object.fromEntries([
  "scripts/bound-native-execution-root-regions.ts", "packages/bot-algo/src/event-execution-box.ts",
  "packages/bot-algo/src/event-execution-acceptance.ts", "packages/bot-algo/src/event-affine-segment.ts",
  "packages/bot-algo/src/event-execution-one-step.ts", "packages/bot-algo/src/event-execution-path.ts",
  "packages/bot-algo/src/event-log-policy.ts",
].map(file => [file, fs.readFileSync(path.join(root, file), "utf8")])));
const started = performance.now(), results: any[] = [];
const checkpoint = () => save("summary.json", {
  contract: "native-execution-root-region-bounds-v1", pruning, pruningHash: hash(path.join(pruning, "summary.json")),
  singleton, singletonHash: hash(path.join(singleton, "summary.json")), source,
  sourceSummaryHash: row.sourceSummaryHash, lawHash: row.lawHash, modelHash: classified.modelHash,
  scope: "Complete non-singleton guarded H1 root regions only. Each root outcome receives an independent maximum over its linked child-account segment, so the sum is an H2 upper relaxation. Constant successor states reuse exact saved continuations. A region is pruned only when this upper cannot beat the achievable singleton incumbent.",
  account: config.probe.account, leaf: config.probe.leaf, step, incumbent: classified.incumbent,
  incumbentRequest: classified.incumbentRequest, refinements, toleranceBps,
  nonSingletons, results, completed: results.length === nonSingletons.length,
  elapsedSeconds: (performance.now() - started) / 1000,
});
for (const [lo, hi] of nonSingletons) {
  const begin = performance.now(), grouped = new Map<string, any>();
  for (const atom of kernels[config.probe.leaf]) {
    const first = evaluateEventExecutionPath(atom.path, config.probe.account, lo * step);
    const last = evaluateEventExecutionPath(atom.path, config.probe.account, hi * step);
    assert.ok(Number.isFinite(first.logGrowth) && Number.isFinite(last.logGrowth));
    const firstState = { equity: first.equity, price: first.price, exposure: first.exposure };
    const lastState = { equity: last.equity, price: last.price, exposure: last.exposure };
    const key = JSON.stringify([atom.next, firstState, lastState]);
    const existing = grouped.get(key);
    if (existing) existing.probability += atom.probability;
    else grouped.set(key, { probability: atom.probability, next: atom.next, first: firstState, last: lastState });
  }
  const variableGroups = [...grouped.values()].filter(branch =>
    stateKey(branch.next, branch.first) !== stateKey(branch.next, branch.last)).length;
  console.log(JSON.stringify({ region: [lo, hi], phase: "grouped", groups: grouped.size, variableGroups }));
  let upperAbsolute = 0, exactStates = 0, exactSolves = 0, boxes = 0, incompleteBoxes = 0;
  let maximumBoxGap = 0, boxRefinements = 0;
  const branches = [];
  for (const branch of grouped.values()) {
    const same = stateKey(branch.next, branch.first) === stateKey(branch.next, branch.last);
    let upperLogEquity: number, kind: "exact" | "box", detail: any;
    if (same) {
      const key = stateKey(branch.next, branch.first);
      let continuation = exactCache.get(key);
      if (!continuation) {
        const solved = exact[branch.next](branch.first); assert.ok(solved.complete);
        continuation = { value: solved.value, quantity: solved.quantity }; exactCache.set(key, continuation); exactSolves++;
      }
      upperLogEquity = Math.log(branch.first.equity) + continuation.value; exactStates++; kind = "exact";
      detail = { continuationRequest: continuation.quantity, continuationValue: continuation.value };
    } else {
      assert.equal(branch.first.price, branch.last.price);
      const C0 = branch.first.equity - branch.first.exposure * branch.first.equity;
      const Q0 = branch.first.exposure * branch.first.equity / branch.first.price;
      const C1 = branch.last.equity - branch.last.exposure * branch.last.equity;
      const Q1 = branch.last.exposure * branch.last.equity / branch.last.price;
      const bounded = bounds[branch.next]({ price: branch.first.price,
        cash: [Math.min(C0, C1), Math.max(C0, C1)], quantity: [Math.min(Q0, Q1), Math.max(Q0, Q1)],
        balanceVertices: [[C0, Q0], [C1, Q1]], quantityLattice: true });
      upperLogEquity = bounded.upperLogEquity; boxes++; incompleteBoxes += Number(!bounded.complete);
      maximumBoxGap = Math.max(maximumBoxGap, bounded.relaxedGap ?? 0); boxRefinements += bounded.search.refinements;
      kind = "box"; detail = bounded;
      if (boxes % 25 === 0) console.log(JSON.stringify({ region: [lo, hi], phase: "boxes", boxes,
        groups: grouped.size, seconds: (performance.now() - begin) / 1000 }));
    }
    upperAbsolute += branch.probability * upperLogEquity;
    branches.push({ probability: branch.probability, next: branch.next, kind, upperLogEquity,
      first: branch.first, last: branch.last, detail });
  }
  const upperValue = upperAbsolute - Math.log(config.probe.account.equity);
  const result = { region: [lo, hi], requests: [lo * step, hi * step], groups: grouped.size,
    exactStates, exactSolves, boxes, incompleteBoxes, maximumBoxGap, boxRefinements,
    upperValue, pruned: upperValue <= classified.incumbent, seconds: (performance.now() - begin) / 1000, branches };
  results.push(result); checkpoint(); console.log(JSON.stringify({ ...result, branches: undefined }));
}
checkpoint();
