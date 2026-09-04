/** Test predefined minimal inventory seeds as H2 requests under the frozen law. */
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { prepareEventExecutionBackup } from "../packages/bot-algo/src/event-execution-backup.js";
import { evaluateEventExecutionPath } from "../packages/bot-algo/src/event-execution-path.js";
const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
const root = path.resolve(__dirname, ".."), directory = (name: string) => path.resolve(root, "data/benchmarks", name);
const source = directory(arg("source")), output = directory(arg("output"));
assert.ok(arg("source") && arg("output") && !fs.existsSync(output));
const read = (file: string) => JSON.parse(fs.readFileSync(file, "utf8"));
const hash = (file: string) => createHash("sha256").update(fs.readFileSync(file)).digest("hex");
const config = read(path.join(source, "config.json")), saved = read(path.join(source, "summary.json"));
assert.equal(config.contract, "native-execution-bellman-candidates-v1"); assert.equal(hash(path.join(config.source, "law.json")), config.lawHash);
const law = read(path.join(config.source, "law.json")), kernels = law.kernels.map((k: any[]) => k.map(a => ({
  probability: a.probability, next: a.next, path: law.paths[a.path] })));
const account = config.probe.account, leaf = config.probe.leaf, costs = law.costs;
assert.equal(account.exposure, 0);
const lowestOpen = Math.min(...kernels[leaf].filter((a: any) => a.path.openingAvailable).map((a: any) => account.price * a.path.openRatio));
const lots = Math.ceil(Math.max(costs.minQuantity, costs.minNotional / lowestOpen) / costs.quantityStep);
const requests = [-1, 1, -2, 2].map(multiplier => multiplier * lots * costs.quantityStep);
const incumbent = Math.max(...saved.results.map((r: any) => r.value));
const start = performance.now(), solve = prepareEventExecutionBackup(kernels), results = [];
fs.mkdirSync(output, { recursive: true });
const save = (name: string, value: unknown) => fs.writeFileSync(path.join(output, name), JSON.stringify(value,
  (_, v) => typeof v === "number" && !Number.isFinite(v) ? String(v) : v, 2));
save("config.json", { source, sourceSummaryHash: hash(path.join(source, "summary.json")), modelHash: config.modelHash, lawHash: config.lawHash,
  account, leaf, requests, lots, incumbent, method: "Both signs of the smallest request meeting size minima at every available forecast opening, and twice that size. Predefined fixed-root H2 values test whether cheap initial inventory changes improve future integer-order acceptance. Same model, exact executor and costs. Stop individual queries at 15 seconds; no policy or global optimality claim." });
save("sources.json", Object.fromEntries(["scripts/probe-native-execution-seeds.ts", "packages/bot-algo/src/event-execution-backup.ts",
  "packages/bot-algo/src/event-execution-one-step.ts", "packages/bot-algo/src/event-execution-path.ts"]
  .map(file => [file, fs.readFileSync(path.join(root, file), "utf8")])));
for (const request of requests) {
  const result = solve(leaf, account, request, { incumbent, valueTolerance: 1e-7, maxSeconds: 15 });
  const key = (leaf: number, a: any) => JSON.stringify([leaf, a.equity, a.price, a.exposure]);
  const policy = new Map(result.continuationPolicy.map(r => [key(r.leaf, r.account), r.quantity]));
  let lower = 0, paths = 0, immediate = 0, rejectedMass = 0;
  for (const atom of kernels[leaf]) {
    const first = evaluateEventExecutionPath(atom.path, account, request);
    const nextRequest = policy.get(key(atom.next, first)); assert.notEqual(nextRequest, undefined);
    immediate += atom.probability * first.logGrowth;
    if (first.canceled) rejectedMass += atom.probability;
    for (const child of kernels[atom.next]) {
      const final = evaluateEventExecutionPath(child.path, first, nextRequest!);
      lower += atom.probability * child.probability * Math.log(final.equity / account.equity); paths++;
    }
  }
  assert.ok(Math.abs(lower - result.lowerValue) < 1e-12);
  results.push({ ...result, immediate, rejectedMass, independentLower: lower, auditedPaths: paths });
  save("summary.json", { results, incumbent, elapsedSeconds: (performance.now() - start) / 1000 });
  console.log(JSON.stringify({ ...results.at(-1), continuationPolicy: undefined }));
}
