/** Certify an epsilon-global H2 root request over exact guarded request regions. */
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { prepareEventExecutionBackup } from "../packages/bot-algo/src/event-execution-backup.js";
import { prepareEventExecutionBoxUpper } from "../packages/bot-algo/src/event-execution-box.js";
import { prepareEventExecutionOneStep } from "../packages/bot-algo/src/event-execution-one-step.js";
import { evaluateEventExecutionPath } from "../packages/bot-algo/src/event-execution-path.js";

const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
const root = path.resolve(__dirname, ".."), directory = (name: string) => path.resolve(root, "data/benchmarks", name);
const pruning = directory(arg("pruning")), singleton = directory(arg("singletons")), output = directory(arg("output"));
const toleranceBps = Number(arg("tolerance-bps") || ".001"), maxSeconds = Number(arg("max-seconds") || "180");
const refinements = Number(arg("refinements") || "1000");
assert.ok(arg("pruning") && arg("singletons") && arg("output") && !fs.existsSync(output));
assert.ok(Number.isFinite(toleranceBps) && toleranceBps >= 0 && Number.isFinite(maxSeconds) && maxSeconds > 0
  && Number.isInteger(refinements) && refinements >= 0);
const read = (file: string) => JSON.parse(fs.readFileSync(file, "utf8"));
const hash = (file: string) => createHash("sha256").update(fs.readFileSync(file)).digest("hex");
const pruned = read(path.join(pruning, "summary.json")), classified = read(path.join(singleton, "summary.json"));
assert.equal(pruned.contract, "native-execution-hold-region-pruning-v1");
assert.equal(classified.contract, "native-execution-singleton-h2-probe-v1");
assert.equal(classified.pruningHash, hash(path.join(pruning, "summary.json"))); assert.equal(classified.completed, true);
const captured = pruned.results.find((row: any) => row.source === classified.source); assert.ok(captured);
assert.equal(captured.lawHash, classified.lawHash); assert.deepEqual(classified.nonSingletonRegions,
  (captured.remaining as Array<[number, number]>).filter(([lo, hi]) => lo < hi));
const source = captured.source, config = read(path.join(source, "config.json")), saved = read(path.join(source, "summary.json"));
assert.equal(hash(path.join(config.source, "law.json")), captured.lawHash);
const law = read(path.join(config.source, "law.json")), step = law.costs.quantityStep;
const kernels = law.kernels.map((kernel: any[]) => kernel.map(atom => ({
  probability: atom.probability, next: atom.next, path: law.paths[atom.path],
})));
const exact = kernels.map((kernel: any[]) => prepareEventExecutionOneStep(kernel));
const backup = prepareEventExecutionBackup(kernels);
const bounds = kernels.map((kernel: any[]) => prepareEventExecutionBoxUpper(kernel, {
  ordered: true, coupledWealth: true, maxRefinements: refinements, valueTolerance: toleranceBps / 10000,
}));
const stateKey = (leaf: number, account: any) => JSON.stringify([leaf, account.equity, account.price, account.exposure]);
const childCache = new Map<string, { value: number; quantity: number }>();
for (const result of saved.results) for (const branch of result.branches) if (branch.complete)
  childCache.set(stateKey(branch.next, branch.account), { value: branch.continuation, quantity: branch.continuationRequest });

type Segment = { lo: number; hi: number; upperValue: number | null; parent: readonly [number, number] | null };
const queue: Segment[] = classified.nonSingletonRegions.map(([lo, hi]: [number, number]) => ({ lo, hi, upperValue: null, parent: null }));
let incumbent = classified.incumbent as number, incumbentRequest = classified.incumbentRequest as number;
const initialIncumbent = incumbent, initialIncumbentRequest = incumbentRequest;
const evaluated: any[] = [], bounded: any[] = [], prunedSegments: any[] = [], started = performance.now();
let exactRootEvaluations = 0, exactChildSolves = 0, boxes = 0, boxRefinements = 0, maximumBoxGap = 0;
fs.mkdirSync(output, { recursive: true });
const save = (name: string, value: unknown) => fs.writeFileSync(path.join(output, name), JSON.stringify(value,
  (_, v) => typeof v === "number" && !Number.isFinite(v) ? String(v) : v, 2));
save("sources.json", Object.fromEntries([
  "scripts/optimize-native-execution-h2-regions.ts", "packages/bot-algo/src/event-execution-backup.ts",
  "packages/bot-algo/src/event-execution-box.ts", "packages/bot-algo/src/event-execution-acceptance.ts",
  "packages/bot-algo/src/event-affine-segment.ts", "packages/bot-algo/src/event-execution-one-step.ts",
  "packages/bot-algo/src/event-execution-path.ts", "packages/bot-algo/src/event-log-policy.ts",
].map(file => [file, fs.readFileSync(path.join(root, file), "utf8")])));

const elapsed = () => (performance.now() - started) / 1000;
const checkpoint = (complete = false) => {
  const unresolvedUpper = queue.reduce((best, segment) => Math.max(best, segment.upperValue ?? Infinity), -Infinity);
  save("summary.json", {
    contract: "native-execution-h2-region-optimizer-v1", pruning, pruningHash: hash(path.join(pruning, "summary.json")),
    singleton, singletonHash: hash(path.join(singleton, "summary.json")), source,
    sourceSummaryHash: captured.sourceSummaryHash, lawHash: captured.lawHash, modelHash: classified.modelHash,
    scope: "Every feasible non-hold root request is covered by exact guarded lattice regions. Singleton guards are inherited from an exact/pruned fixed-root audit. Each remaining interval is recursively split; linked child-account box maxima provide valid interval upper bounds, while exact midpoint Bellman backups update an achievable incumbent. Completion certifies the incumbent within the declared log-value tolerance under this execution law.",
    account: config.probe.account, leaf: config.probe.leaf, step, toleranceBps, maxSeconds, refinements,
    initialIncumbent, initialIncumbentRequest, incumbent, incumbentRequest,
    certificateGapBps: complete ? Math.max(0, unresolvedUpper - incumbent) * 10000 : null,
    complete, queue, evaluated, bounded, prunedSegments, exactRootEvaluations, exactChildSolves,
    boxes, boxRefinements, maximumBoxGap, elapsedSeconds: elapsed(),
  });
};

const boundSegment = (segment: Segment) => {
  const begin = performance.now(), grouped = new Map<string, any>();
  for (const atom of kernels[config.probe.leaf]) {
    const first = evaluateEventExecutionPath(atom.path, config.probe.account, segment.lo * step);
    const last = evaluateEventExecutionPath(atom.path, config.probe.account, segment.hi * step);
    assert.ok(Number.isFinite(first.logGrowth) && Number.isFinite(last.logGrowth));
    const firstState = { equity: first.equity, price: first.price, exposure: first.exposure };
    const lastState = { equity: last.equity, price: last.price, exposure: last.exposure };
    const key = JSON.stringify([atom.next, firstState, lastState]);
    const row = grouped.get(key);
    if (row) row.probability += atom.probability;
    else grouped.set(key, { probability: atom.probability, next: atom.next, first: firstState, last: lastState });
  }
  let upperAbsolute = 0, localBoxes = 0, localExact = 0;
  for (const branch of grouped.values()) {
    let upperLogEquity: number;
    if (stateKey(branch.next, branch.first) === stateKey(branch.next, branch.last)) {
      const key = stateKey(branch.next, branch.first);
      let continuation = childCache.get(key);
      if (!continuation) {
        const solved = exact[branch.next](branch.first); assert.ok(solved.complete);
        continuation = { value: solved.value, quantity: solved.quantity }; childCache.set(key, continuation);
        exactChildSolves++; localExact++;
      }
      upperLogEquity = Math.log(branch.first.equity) + continuation.value;
    } else {
      assert.equal(branch.first.price, branch.last.price);
      const C0 = branch.first.equity * (1 - branch.first.exposure);
      const Q0 = branch.first.exposure * branch.first.equity / branch.first.price;
      const C1 = branch.last.equity * (1 - branch.last.exposure);
      const Q1 = branch.last.exposure * branch.last.equity / branch.last.price;
      const result = bounds[branch.next]({ price: branch.first.price,
        cash: [Math.min(C0, C1), Math.max(C0, C1)], quantity: [Math.min(Q0, Q1), Math.max(Q0, Q1)],
        balanceVertices: [[C0, Q0], [C1, Q1]], quantityLattice: true });
      upperLogEquity = result.upperLogEquity; boxes++; localBoxes++; boxRefinements += result.search.refinements;
      maximumBoxGap = Math.max(maximumBoxGap, result.relaxedGap ?? 0);
    }
    upperAbsolute += branch.probability * upperLogEquity;
  }
  segment.upperValue = upperAbsolute - Math.log(config.probe.account.equity) + 1e-10;
  bounded.push({ region: [segment.lo, segment.hi], parent: segment.parent, upperValue: segment.upperValue,
    groups: grouped.size, boxes: localBoxes, exactStates: localExact, seconds: (performance.now() - begin) / 1000 });
};

while (queue.length && elapsed() < maxSeconds) {
  for (const segment of queue) if (segment.upperValue === null) {
    boundSegment(segment);
    if (elapsed() >= maxSeconds) break;
  }
  queue.sort((a, b) => (b.upperValue ?? Infinity) - (a.upperValue ?? Infinity)
    || (b.hi - b.lo) - (a.hi - a.lo));
  const segment = queue.shift()!;
  if (segment.upperValue! <= incumbent + toleranceBps / 10000) {
    prunedSegments.push({ region: [segment.lo, segment.hi], upperValue: segment.upperValue, incumbent });
    if ((bounded.length + evaluated.length) % 25 === 0) checkpoint();
    continue;
  }
  const lot = Math.floor((segment.lo + segment.hi) / 2), request = lot * step;
  const result = backup(config.probe.leaf, config.probe.account, request);
  assert.ok(result.complete && result.value !== null); exactRootEvaluations++;
  const prior = incumbent;
  if (result.value > incumbent) { incumbent = result.value; incumbentRequest = request; }
  evaluated.push({ lot, request, value: result.value, priorIncumbent: prior, improved: result.value > prior,
    continuationStates: result.continuationStates, continuationSolves: result.continuationSolves, seconds: result.seconds,
    sourceRegion: [segment.lo, segment.hi], sourceUpperValue: segment.upperValue });
  if (segment.lo <= lot - 1) queue.push({ lo: segment.lo, hi: lot - 1, upperValue: null, parent: [segment.lo, segment.hi] });
  if (lot + 1 <= segment.hi) queue.push({ lo: lot + 1, hi: segment.hi, upperValue: null, parent: [segment.lo, segment.hi] });
  if ((bounded.length + evaluated.length) % 25 === 0) checkpoint();
  if (evaluated.length % 10 === 0) console.log(JSON.stringify({ evaluated: evaluated.length, bounded: bounded.length,
    queue: queue.length, incumbent, incumbentRequest, elapsedSeconds: elapsed() }));
}
for (const segment of queue) if (segment.upperValue === null && elapsed() < maxSeconds) boundSegment(segment);
const complete = queue.every(segment => segment.upperValue !== null
  && segment.upperValue <= incumbent + toleranceBps / 10000);
checkpoint(complete);
console.log(JSON.stringify({ complete, incumbent, incumbentRequest, initialIncumbent, initialIncumbentRequest,
  toleranceBps, evaluated: evaluated.length, bounded: bounded.length, unresolved: queue.length,
  maximumUnresolvedUpper: queue.reduce((best, segment) => Math.max(best, segment.upperValue ?? Infinity), -Infinity),
  boxes, exactRootEvaluations, elapsedSeconds: elapsed() }));
if (!complete) process.exitCode = 1;
