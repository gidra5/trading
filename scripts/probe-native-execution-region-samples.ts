/** Exact H2 samples from unresolved guarded root regions; diagnostic, not a region certificate. */
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { prepareEventExecutionBackup } from "../packages/bot-algo/src/event-execution-backup.js";

const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
const root = path.resolve(__dirname, ".."), directory = (name: string) => path.resolve(root, "data/benchmarks", name);
const singleton = directory(arg("singletons")), output = directory(arg("output"));
const perRequestSeconds = Number(arg("seconds") || "20");
assert.ok(arg("singletons") && arg("output") && !fs.existsSync(output));
assert.ok(Number.isFinite(perRequestSeconds) && perRequestSeconds > 0);
const read = (file: string) => JSON.parse(fs.readFileSync(file, "utf8"));
const hash = (file: string) => createHash("sha256").update(fs.readFileSync(file)).digest("hex");
const classified = read(path.join(singleton, "summary.json"));
assert.equal(classified.contract, "native-execution-singleton-h2-probe-v1"); assert.equal(classified.completed, true);
const source = classified.source, config = read(path.join(source, "config.json")), saved = read(path.join(source, "summary.json"));
assert.equal(hash(path.join(config.source, "law.json")), classified.lawHash);
const law = read(path.join(config.source, "law.json")), step = law.costs.quantityStep;
const kernels = law.kernels.map((kernel: any[]) => kernel.map(atom => ({
  probability: atom.probability, next: atom.next, path: law.paths[atom.path],
})));
const backup = prepareEventExecutionBackup(kernels), started = performance.now();
fs.mkdirSync(output, { recursive: true });
const save = (name: string, value: unknown) => fs.writeFileSync(path.join(output, name), JSON.stringify(value,
  (_, v) => typeof v === "number" && !Number.isFinite(v) ? String(v) : v, 2));
save("sources.json", Object.fromEntries([
  "scripts/probe-native-execution-region-samples.ts", "packages/bot-algo/src/event-execution-backup.ts",
  "packages/bot-algo/src/event-execution-upper.ts", "packages/bot-algo/src/event-execution-partitions.ts",
  "packages/bot-algo/src/event-execution-one-step.ts", "packages/bot-algo/src/event-execution-path.ts",
  "packages/bot-algo/src/event-log-policy.ts",
].map(file => [file, fs.readFileSync(path.join(root, file), "utf8")])));

const warmups = [];
for (const request of [...new Set([0, classified.incumbentRequest])]) {
  const result = backup(config.probe.leaf, config.probe.account, request, { maxSeconds: perRequestSeconds });
  const reference = saved.results.find((r: any) => r.request === request);
  assert.ok(result.complete && reference?.complete && Math.abs(result.value! - reference.value) < 1e-12);
  warmups.push({ request, value: result.value, seconds: result.seconds, continuationSolves: result.continuationSolves,
    continuationStates: result.continuationStates, cacheSize: result.cacheSize });
}
const sampleLots = (classified.nonSingletonRegions as Array<[number, number]>).flatMap(([lo, hi]) =>
  [...new Set([lo, Math.floor((lo + hi) / 2), hi])]);
const results: any[] = [];
const checkpoint = () => save("summary.json", {
  contract: "native-execution-root-region-samples-v1", singleton,
  singletonHash: hash(path.join(singleton, "summary.json")), source,
  sourceSummaryHash: classified.sourceSummaryHash, lawHash: classified.lawHash, modelHash: classified.modelHash,
  scope: "Three exact H2 samples per unresolved non-singleton guarded root region. These samples diagnose value shape and cannot bound unsampled requests or certify a global optimum.",
  account: config.probe.account, leaf: config.probe.leaf, step, incumbent: classified.incumbent,
  incumbentRequest: classified.incumbentRequest, regions: classified.nonSingletonRegions,
  warmups, sampleLots, results, completed: results.length === sampleLots.length,
  elapsedSeconds: (performance.now() - started) / 1000,
});
for (const lot of sampleLots) {
  const request = lot * step;
  let result = backup(config.probe.leaf, config.probe.account, request, { maxSeconds: perRequestSeconds });
  if (result.status === "budget") result = backup(config.probe.leaf, config.probe.account, request,
    { maxSeconds: perRequestSeconds });
  const item = { lot, request, status: result.status, value: result.value,
    lowerValue: result.lowerValue, upperValue: result.upperValue,
    continuationSolves: result.continuationSolves, partitionSolves: result.partitionSolves,
    continuationStates: result.continuationStates, cacheSize: result.cacheSize, seconds: result.seconds };
  results.push(item); checkpoint(); console.log(JSON.stringify(item));
}
checkpoint();
if (results.some(r => r.status === "budget")) process.exitCode = 1;
