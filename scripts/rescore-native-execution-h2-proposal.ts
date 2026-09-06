/** Bound a compact-law H2 root proposal under the original execution law. */
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { prepareEventExecutionBackup } from "../packages/bot-algo/src/event-execution-backup.js";

const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
const root = path.resolve(__dirname, ".."), directory = (name: string) => path.resolve(root, "data/benchmarks", name);
const source = directory(arg("source")), output = directory(arg("output"));
const requests = arg("requests").split(",").filter(Boolean).map(Number);
const seconds = Number(arg("seconds") || "45"), toleranceBps = Number(arg("tolerance-bps") || ".001");
const incumbent = Number(arg("incumbent") || "0");
assert.ok(arg("source") && arg("output") && requests.length && !fs.existsSync(output));
assert.ok(requests.every(Number.isFinite) && Number.isFinite(seconds) && seconds > 0
  && Number.isFinite(toleranceBps) && toleranceBps >= 0 && Number.isFinite(incumbent));
const read = (file: string) => JSON.parse(fs.readFileSync(file, "utf8"));
const hash = (file: string) => createHash("sha256").update(fs.readFileSync(file)).digest("hex");
const config = read(path.join(source, "config.json"));
assert.equal(config.contract, "native-execution-bellman-candidates-v1");
assert.equal(hash(path.join(config.source, "law.json")), config.lawHash);
const law = read(path.join(config.source, "law.json")), step = law.costs.quantityStep;
assert.ok(requests.every(request => Math.abs(request / step - Math.round(request / step)) < 1e-7));
const kernels = law.kernels.map((kernel: any[]) => kernel.map(atom => ({
  probability: atom.probability, next: atom.next, path: law.paths[atom.path],
})));
const solve = prepareEventExecutionBackup(kernels), results: any[] = [], started = performance.now();
fs.mkdirSync(output, { recursive: true });
const save = (name: string, value: unknown) => fs.writeFileSync(path.join(output, name), JSON.stringify(value,
  (_, v) => typeof v === "number" && !Number.isFinite(v) ? String(v) : v, 2));
save("config.json", { contract: "native-execution-h2-proposal-rescore-v1", source,
  sourceConfigHash: hash(path.join(source, "config.json")), lawHash: config.lawHash, modelHash: config.modelHash,
  account: config.probe.account, leaf: config.probe.leaf, requests, incumbent, seconds, toleranceBps,
  method: "Evaluate compact-law root proposals under the original uncompressed execution law. The lower endpoint is an executable continuation policy; the upper endpoint is an opening-information bound. A pruned result proves the request cannot beat the declared achievable incumbent. A certified interval bounds this fixed request only and does not establish a global H2 optimum." });
save("sources.json", Object.fromEntries([
  "scripts/rescore-native-execution-h2-proposal.ts", "packages/bot-algo/src/event-execution-backup.ts",
  "packages/bot-algo/src/event-execution-upper.ts", "packages/bot-algo/src/event-execution-partitions.ts",
  "packages/bot-algo/src/event-execution-one-step.ts", "packages/bot-algo/src/event-execution-path.ts",
  "packages/bot-algo/src/event-log-policy.ts",
].map(file => [file, fs.readFileSync(path.join(root, file), "utf8")])));
for (const request of requests) {
  const result = solve(config.probe.leaf, config.probe.account, request, {
    incumbent, maxSeconds: seconds, valueTolerance: toleranceBps / 10000,
  });
  results.push(result);
  save("summary.json", { contract: "native-execution-h2-proposal-rescore-v1", results,
    elapsedSeconds: (performance.now() - started) / 1000 });
  console.log(JSON.stringify({ ...result, continuationPolicy: undefined }));
}
