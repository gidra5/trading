/** Compare exact H2 continuation requests with their target-inventory coordinates. */
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { prepareEventExecutionBackup } from "../packages/bot-algo/src/event-execution-backup.js";
import { evaluateEventExecutionPath } from "../packages/bot-algo/src/event-execution-path.js";

const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
const root = path.resolve(__dirname, ".."), directory = (name: string) => path.resolve(root, "data/benchmarks", name);
const samples = directory(arg("samples")), output = directory(arg("output"));
const perRequestSeconds = Number(arg("seconds") || "20"), regionIndex = Number(arg("region") || "0");
assert.ok(arg("samples") && arg("output") && !fs.existsSync(output));
assert.ok(Number.isFinite(perRequestSeconds) && perRequestSeconds > 0 && Number.isInteger(regionIndex) && regionIndex >= 0);
const read = (file: string) => JSON.parse(fs.readFileSync(file, "utf8"));
const hash = (file: string) => createHash("sha256").update(fs.readFileSync(file)).digest("hex");
const sampled = read(path.join(samples, "summary.json"));
assert.equal(sampled.contract, "native-execution-root-region-samples-v1"); assert.equal(sampled.completed, true);
const region = sampled.regions[regionIndex] as [number, number]; assert.ok(region);
const source = sampled.source, config = read(path.join(source, "config.json")), saved = read(path.join(source, "summary.json"));
assert.equal(hash(path.join(config.source, "law.json")), sampled.lawHash);
const law = read(path.join(config.source, "law.json")), step = law.costs.quantityStep;
const fee = (law.costs.feeBps + law.costs.slippageBps) / 10000;
const kernels = law.kernels.map((kernel: any[]) => kernel.map(atom => ({
  probability: atom.probability, next: atom.next, path: law.paths[atom.path],
})));
const backup = prepareEventExecutionBackup(kernels), started = performance.now();
fs.mkdirSync(output, { recursive: true });
const save = (name: string, value: unknown) => fs.writeFileSync(path.join(output, name), JSON.stringify(value,
  (_, v) => typeof v === "number" && !Number.isFinite(v) ? String(v) : v, 2));
save("sources.json", Object.fromEntries([
  "scripts/probe-native-execution-target-stability.ts", "packages/bot-algo/src/event-execution-backup.ts",
  "packages/bot-algo/src/event-execution-upper.ts", "packages/bot-algo/src/event-execution-partitions.ts",
  "packages/bot-algo/src/event-execution-one-step.ts", "packages/bot-algo/src/event-execution-path.ts",
  "packages/bot-algo/src/event-log-policy.ts",
].map(file => [file, fs.readFileSync(path.join(root, file), "utf8")])));
const warmups = [];
for (const request of [...new Set([0, sampled.incumbentRequest])]) {
  const result = backup(config.probe.leaf, config.probe.account, request, { maxSeconds: perRequestSeconds });
  const reference = saved.results.find((r: any) => r.request === request);
  assert.ok(result.complete && reference?.complete && Math.abs(result.value! - reference.value) < 1e-12);
  warmups.push({ request, value: result.value, seconds: result.seconds, continuationSolves: result.continuationSolves,
    continuationStates: result.continuationStates, cacheSize: result.cacheSize });
}
const [lo, hi] = region, lots = [...new Set([lo, Math.floor((lo + hi) / 2), hi])];
const policies = [];
for (const lot of lots) {
  let result = backup(config.probe.leaf, config.probe.account, lot * step, { maxSeconds: perRequestSeconds });
  if (result.status === "budget") result = backup(config.probe.leaf, config.probe.account, lot * step,
    { maxSeconds: perRequestSeconds });
  assert.ok(result.complete);
  const key = (leaf: number, account: any) => JSON.stringify([leaf, account.equity, account.price, account.exposure]);
  const actions = new Map<string, number>(result.continuationPolicy.map(r => [key(r.leaf, r.account), r.quantity]));
  const rows = kernels[config.probe.leaf].map((atom: any, index: number) => {
    const next = evaluateEventExecutionPath(atom.path, config.probe.account, lot * step);
    const childRequest = actions.get(key(atom.next, next)); assert.notEqual(childRequest, undefined);
    const inventoryLot = Math.round(next.quantity / step), requestLot = Math.round(childRequest! / step);
    assert.ok(Math.abs(next.quantity - inventoryLot * step) < 1e-10);
    assert.ok(Math.abs(childRequest! - requestLot * step) < 1e-10);
    const intendedEquity = next.equity - Math.abs(childRequest!) * next.price * fee;
    const targetExposure = intendedEquity > 0 ? (next.quantity + childRequest!) * next.price / intendedEquity : Infinity;
    return { index, probability: atom.probability, next: atom.next, rootFilled: next.filledQuantity !== 0,
      inventoryLot, requestLot, targetLot: inventoryLot + requestLot, targetExposure };
  });
  policies.push({ lot, request: lot * step, value: result.value, seconds: result.seconds,
    continuationSolves: result.continuationSolves, cacheSize: result.cacheSize, rows });
  console.log(JSON.stringify({ lot, value: result.value, seconds: result.seconds,
    continuationSolves: result.continuationSolves, cacheSize: result.cacheSize }));
}
let sameRequestCount = 0, sameRequestMass = 0, sameTargetCount = 0, sameTargetMass = 0;
let sameTargetWhenRootFilledCount = 0, sameTargetWhenRootFilledMass = 0, rootFilledMass = 0, rootFilledCount = 0;
const exposureTolerance = [1e-9, 1e-6, 1e-3, 1e-2];
const stableExposure = exposureTolerance.map(tolerance => ({ tolerance, count: 0, mass: 0 }));
let maximumTargetExposureRange = 0;
for (let i = 0; i < kernels[config.probe.leaf].length; i++) {
  const rows = policies.map(policy => policy.rows[i]), probability = rows[0].probability;
  assert.ok(rows.every(row => row.probability === probability && row.next === rows[0].next));
  const sameRequest = rows.every(row => row.requestLot === rows[0].requestLot);
  const sameTarget = rows.every(row => row.targetLot === rows[0].targetLot);
  const targetExposureRange = Math.max(...rows.map(row => row.targetExposure)) - Math.min(...rows.map(row => row.targetExposure));
  maximumTargetExposureRange = Math.max(maximumTargetExposureRange, targetExposureRange);
  for (const item of stableExposure) if (targetExposureRange <= item.tolerance) { item.count++; item.mass += probability; }
  sameRequestCount += Number(sameRequest); sameRequestMass += sameRequest ? probability : 0;
  sameTargetCount += Number(sameTarget); sameTargetMass += sameTarget ? probability : 0;
  if (rows.every(row => row.rootFilled)) {
    rootFilledCount++; rootFilledMass += probability;
    sameTargetWhenRootFilledCount += Number(sameTarget); sameTargetWhenRootFilledMass += sameTarget ? probability : 0;
  }
}
const comparison = { outcomes: kernels[config.probe.leaf].length,
  sameRequestCount, sameRequestMass, sameTargetCount, sameTargetMass,
  stableExposure, maximumTargetExposureRange,
  rootFilledCount, rootFilledMass, sameTargetWhenRootFilledCount, sameTargetWhenRootFilledMass };
save("summary.json", {
  contract: "native-execution-target-stability-v1", samples, samplesHash: hash(path.join(samples, "summary.json")),
  source, sourceSummaryHash: sampled.sourceSummaryHash, lawHash: sampled.lawHash, modelHash: sampled.modelHash,
  scope: "Exact H2 continuation actions at the endpoint and midpoint of one unresolved guarded root interval. Target lot equals incoming inventory lot plus child request lot; intended target exposure uses the child decision price and immediate fee. Stability is descriptive across these three roots and does not prove stability between them or certify the interval optimum.",
  account: config.probe.account, leaf: config.probe.leaf, step, regionIndex, region, lots,
  incumbent: sampled.incumbent, incumbentRequest: sampled.incumbentRequest, warmups, comparison, policies,
  elapsedSeconds: (performance.now() - started) / 1000,
});
console.log(JSON.stringify(comparison));
