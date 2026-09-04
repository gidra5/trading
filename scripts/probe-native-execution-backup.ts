/** Validate adaptive Bellman request bounds against source-bound exact candidate values. */
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { prepareEventExecutionBackup } from "../packages/bot-algo/src/event-execution-backup.js";
import { evaluateEventExecutionPath } from "../packages/bot-algo/src/event-execution-path.js";
const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
const root = path.resolve(__dirname, ".."), directory = (name: string) => path.resolve(root, "data/benchmarks", name);
const source = directory(arg("source")), output = directory(arg("output"));
const toleranceBps = Number(arg("tolerance-bps") || "0");
assert.ok(Number.isFinite(toleranceBps) && toleranceBps >= 0);
assert.ok(arg("source") && arg("output") && !fs.existsSync(output));
const read = (file: string) => JSON.parse(fs.readFileSync(file, "utf8"));
const hash = (file: string) => createHash("sha256").update(fs.readFileSync(file)).digest("hex");
const config = read(path.join(source, "config.json")), saved = read(path.join(source, "summary.json"));
assert.equal(config.contract, "native-execution-bellman-candidates-v1");
assert.equal(hash(path.join(config.source, "law.json")), config.lawHash);
const law = read(path.join(config.source, "law.json")), kernels = law.kernels.map((k: any[]) => k.map(a => ({
  probability: a.probability, next: a.next, path: law.paths[a.path] })));
const started = performance.now(), backup = prepareEventExecutionBackup(kernels), preparationSeconds = (performance.now() - started) / 1000;
const incumbent = Math.max(...saved.results.filter((r: any) => r.complete && r.finite).map((r: any) => r.value));
fs.mkdirSync(output, { recursive: true });
const save = (name: string, data: unknown) => fs.writeFileSync(path.join(output, name), JSON.stringify(data,
  (_, v) => typeof v === "number" && !Number.isFinite(v) ? String(v) : v, 2));
save("config.json", { contract: "execution-adaptive-backup-probe-v1", source, sourceSummaryHash: hash(path.join(source, "summary.json")),
  modelHash: config.modelHash, lawHash: config.lawHash, account: config.probe.account, leaf: config.probe.leaf, incumbent, toleranceBps,
  method: "Use a previously fully evaluated fixed-root H2 candidate as an achievable incumbent. Bound each predefined request with the opening-information relaxation and refine continuations in descending probability-weighted uncertainty. With a positive tolerance, use nested opening-information partitions on large kernels before exact H1 fallback. Stop at complete evaluation, a fixed-request value interval within the declared tolerance, certified inferiority, or an explicit time limit. An interval certificate retains a null exact value and a concrete lower-bound contingent policy, which is rescored using the complete joint law. No future market path is used." });
save("sources.json", Object.fromEntries(["scripts/probe-native-execution-backup.ts", "packages/bot-algo/src/event-execution-backup.ts",
  "packages/bot-algo/src/event-execution-upper.ts", "packages/bot-algo/src/event-execution-one-step.ts",
  "packages/bot-algo/src/event-execution-path.ts", "packages/bot-algo/src/event-log-policy.ts"]
  .map(file => [file, fs.readFileSync(path.join(root, file), "utf8")])));
const results = [];
for (const candidate of saved.results) {
  assert.ok(candidate.complete && candidate.finite);
  const result = backup(config.probe.leaf, config.probe.account, candidate.request, { incumbent, maxSeconds: 30, valueTolerance: toleranceBps / 10000 });
  assert.ok(result.lowerValue <= candidate.value + 1e-10 && result.upperValue >= candidate.value - 1e-10);
  if (result.status === "pruned") assert.ok(result.upperValue <= incumbent);
  if (result.complete) assert.ok(Math.abs(result.value! - candidate.value) < 1e-12);
  if (result.status === "certified") assert.ok(result.upperValue - result.lowerValue <= toleranceBps / 10000);
  const key = (leaf: number, a: any) => JSON.stringify([leaf, a.equity, a.price, a.exposure]);
  const policy = new Map(result.continuationPolicy.map(r => [key(r.leaf, r.account), r.quantity]));
  let lowerValue = 0, paths = 0;
  for (const atom of kernels[config.probe.leaf]) {
    const next = evaluateEventExecutionPath(atom.path, config.probe.account, candidate.request);
    const request = policy.get(key(atom.next, next)); assert.notEqual(request, undefined);
    for (const child of kernels[atom.next]) {
      const terminal = evaluateEventExecutionPath(child.path, next, request!);
      lowerValue += atom.probability * child.probability * Math.log(terminal.equity / config.probe.account.equity); paths++;
    }
  }
  assert.ok(Math.abs(lowerValue - result.lowerValue) < 1e-12);
  results.push({ ...result, savedExactValue: candidate.value, independentlyScoredLowerValue: lowerValue, auditedPaths: paths,
    originalSeconds: candidate.seconds });
  save("summary.json", { preparationSeconds, incumbent, results, elapsedSeconds: (performance.now() - started) / 1000 });
  console.log(JSON.stringify({ ...results.at(-1), continuationPolicy: undefined }));
}
if (results.some(r => r.status === "budget")) process.exitCode = 1;
