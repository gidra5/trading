/** Evaluate the remaining singleton root regions in an exact H2 request cover. */
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { prepareEventExecutionBackup } from "../packages/bot-algo/src/event-execution-backup.js";
import { evaluateEventExecutionPath } from "../packages/bot-algo/src/event-execution-path.js";

const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
const root = path.resolve(__dirname, ".."), directory = (name: string) => path.resolve(root, "data/benchmarks", name);
const pruning = directory(arg("pruning")), output = directory(arg("output"));
const perRequestSeconds = Number(arg("seconds") || "20");
assert.ok(arg("pruning") && arg("output") && !fs.existsSync(output));
assert.ok(Number.isFinite(perRequestSeconds) && perRequestSeconds > 0);
const read = (file: string) => JSON.parse(fs.readFileSync(file, "utf8"));
const hash = (file: string) => createHash("sha256").update(fs.readFileSync(file)).digest("hex");
const captured = read(path.join(pruning, "summary.json"));
assert.equal(captured.contract, "native-execution-hold-region-pruning-v1");
const sourceName = arg("source");
const row = captured.results.find((r: any) => !sourceName || path.basename(r.source) === sourceName);
assert.ok(row, "Requested source is absent from the pruning artifact");
const source = row.source, config = read(path.join(source, "config.json")), saved = read(path.join(source, "summary.json"));
assert.equal(config.contract, "native-execution-bellman-candidates-v1");
assert.equal(hash(path.join(config.source, "law.json")), row.lawHash);
assert.equal(hash(path.join(source, "summary.json")), row.sourceSummaryHash);
const law = read(path.join(config.source, "law.json")), step = law.costs.quantityStep;
const kernels = law.kernels.map((kernel: any[]) => kernel.map(atom => ({
  probability: atom.probability, next: atom.next, path: law.paths[atom.path],
})));
const backup = prepareEventExecutionBackup(kernels), started = performance.now();
fs.mkdirSync(output, { recursive: true });
const save = (name: string, value: unknown) => fs.writeFileSync(path.join(output, name), JSON.stringify(value,
  (_, v) => typeof v === "number" && !Number.isFinite(v) ? String(v) : v, 2));
save("sources.json", Object.fromEntries([
  "scripts/probe-native-execution-singleton-regions.ts", "packages/bot-algo/src/event-execution-backup.ts",
  "packages/bot-algo/src/event-execution-upper.ts", "packages/bot-algo/src/event-execution-partitions.ts",
  "packages/bot-algo/src/event-execution-one-step.ts", "packages/bot-algo/src/event-execution-path.ts",
  "packages/bot-algo/src/event-log-policy.ts",
].map(file => [file, fs.readFileSync(path.join(root, file), "utf8")])));

let incumbent = Math.max(...saved.results.filter((r: any) => r.complete && r.finite).map((r: any) => r.value));
const incumbentRow = saved.results.find((r: any) => r.complete && r.finite && r.value === incumbent);
assert.ok(incumbentRow);
const warmRequests = [...new Set([0, incumbentRow.request])];
const warmups = [];
for (const request of warmRequests) {
  const result = backup(config.probe.leaf, config.probe.account, request, { maxSeconds: perRequestSeconds });
  const reference = saved.results.find((r: any) => r.request === request);
  assert.ok(result.complete && reference?.complete && Math.abs(result.value! - reference.value) < 1e-12);
  warmups.push({ request, value: result.value, seconds: result.seconds, continuationSolves: result.continuationSolves,
    continuationStates: result.continuationStates, cacheSize: result.cacheSize });
}

const singletonLots = (row.remaining as Array<[number, number]>).filter(([lo, hi]) => lo === hi).map(([lo]) => lo);
const results: any[] = [], scorePolicy = (request: number, result: any) => {
  const key = (leaf: number, account: any) => JSON.stringify([leaf, account.equity, account.price, account.exposure]);
  const policy = new Map<string, number>(result.continuationPolicy.map((r: any) => [key(r.leaf, r.account), r.quantity]));
  let value = 0, paths = 0;
  for (const atom of kernels[config.probe.leaf]) {
    const next = evaluateEventExecutionPath(atom.path, config.probe.account, request);
    const childRequest = policy.get(key(atom.next, next)); assert.notEqual(childRequest, undefined);
    for (const child of kernels[atom.next]) {
      const terminal = evaluateEventExecutionPath(child.path, next, childRequest!);
      value += atom.probability * child.probability * Math.log(terminal.equity / config.probe.account.equity); paths++;
    }
  }
  return { value, paths };
};
const checkpoint = () => save("summary.json", {
  contract: "native-execution-singleton-h2-probe-v1", pruning, pruningHash: hash(path.join(pruning, "summary.json")),
  source, sourceSummaryHash: row.sourceSummaryHash, lawHash: row.lawHash, modelHash: config.modelHash,
  scope: "Only singleton regions remaining after exact H1 cover and hold-equivalent pruning. Each root is either exactly evaluated or certified unable to beat an achievable incumbent. The four non-singleton regions remain unsearched.",
  account: config.probe.account, leaf: config.probe.leaf, step, warmups, initialIncumbent: incumbentRow.value,
  incumbent, incumbentRequest: results.find(r => r.exactValue === incumbent)?.request ?? incumbentRow.request,
  singletonLots, results, completed: results.length === singletonLots.length,
  nonSingletonRegions: (row.remaining as Array<[number, number]>).filter(([lo, hi]) => lo < hi),
  elapsedSeconds: (performance.now() - started) / 1000,
});
for (const lot of singletonLots) {
  const request = lot * step;
  const priorIncumbent = incumbent;
  let result = backup(config.probe.leaf, config.probe.account, request, { incumbent, maxSeconds: perRequestSeconds });
  // A time budget keeps all interval semantics honest. Retry once in-process so
  // tightened continuation states and the shared exact cache are retained.
  if (result.status === "budget") result = backup(config.probe.leaf, config.probe.account, request,
    { incumbent, maxSeconds: perRequestSeconds });
  assert.notEqual(result.status, "certified");
  let audit = null;
  if (result.lowerValue > priorIncumbent) {
    const checked = scorePolicy(request, result); assert.ok(Math.abs(checked.value - result.lowerValue) < 1e-12);
    audit = { independentlyScoredLowerValue: checked.value, paths: checked.paths };
  }
  if (result.complete && Number.isFinite(result.value) && result.value! > incumbent) incumbent = result.value!;
  const item = { lot, request, status: result.status, exactValue: result.value, lowerValue: result.lowerValue,
    upperValue: result.upperValue, continuationSolves: result.continuationSolves,
    partitionSolves: result.partitionSolves, continuationStates: result.continuationStates,
    cacheSize: result.cacheSize, seconds: result.seconds, audit };
  results.push(item); checkpoint(); console.log(JSON.stringify(item));
}
checkpoint();
if (results.some(r => r.status === "budget")) process.exitCode = 1;
