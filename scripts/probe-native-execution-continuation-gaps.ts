/** Decompose exact fixed-root H2 values and continuation upper-bound slack. */
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { prepareEventExecutionBackup } from "../packages/bot-algo/src/event-execution-backup.js";
import { prepareEventExecutionUpper } from "../packages/bot-algo/src/event-execution-upper.js";
import { prepareEventExecutionOneStep } from "../packages/bot-algo/src/event-execution-one-step.js";
import { evaluateEventExecutionPath } from "../packages/bot-algo/src/event-execution-path.js";

const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
const root = path.resolve(__dirname, ".."), directory = (name: string) => path.resolve(root, "data/benchmarks", name);
const source = directory(arg("source")), output = directory(arg("output"));
const lots = arg("lots").split(",").filter(Boolean).map(Number);
assert.ok(arg("source") && arg("output") && lots.length && lots.every(Number.isSafeInteger) && !fs.existsSync(output));
const read = (file: string) => JSON.parse(fs.readFileSync(file, "utf8"));
const hash = (file: string) => createHash("sha256").update(fs.readFileSync(file)).digest("hex");
const sourceConfig = read(path.join(source, "config.json")), sourceSummary = read(path.join(source, "summary.json"));
const lawFile = path.join(sourceConfig.source, "law.json"), law = read(lawFile), step = law.costs.quantityStep;
const kernels = law.kernels.map((kernel: any[]) => kernel.map(atom => ({
  probability: atom.probability, next: atom.next, path: law.paths[atom.path],
})));
const account = sourceConfig.probe.account, leaf = sourceConfig.probe.leaf;
const backup = prepareEventExecutionBackup(kernels), uppers = kernels.map((kernel: any[]) => prepareEventExecutionUpper(kernel));
const means = kernels.map((kernel: any[]) => prepareEventExecutionOneStep(kernel, "marked", { objective: "mean" }));
fs.mkdirSync(output, { recursive: true });
const save = (name: string, value: unknown) => fs.writeFileSync(path.join(output, name), JSON.stringify(value,
  (_, v) => typeof v === "number" && !Number.isFinite(v) ? String(v) : v, 2));
save("sources.json", Object.fromEntries([
  "scripts/probe-native-execution-continuation-gaps.ts", "packages/bot-algo/src/event-execution-backup.ts",
  "packages/bot-algo/src/event-execution-upper.ts", "packages/bot-algo/src/event-execution-one-step.ts",
  "packages/bot-algo/src/event-execution-path.ts", "packages/bot-algo/src/event-log-policy.ts",
].map(file => [file, fs.readFileSync(path.join(root, file), "utf8")])));
const started = performance.now(), results = [];
for (const lot of lots) {
  const quantity = lot * step, begin = performance.now();
  const solved = backup(leaf, account, quantity, { maxSolves: Infinity, maxSeconds: Infinity });
  assert.equal(solved.status, "complete"); assert.equal(solved.value, solved.lowerValue);
  const policies = new Map(solved.continuationPolicy.map(row => [JSON.stringify([row.leaf, row.account.equity,
    row.account.price, row.account.exposure]), row]));
  const branches = new Map<string, any>();
  for (const atom of kernels[leaf]) {
    if (!atom.probability) continue;
    const next = evaluateEventExecutionPath(atom.path, account, quantity);
    assert.ok(Number.isFinite(next.logGrowth));
    const nextAccount = { equity: next.equity, price: next.price, exposure: next.exposure };
    const key = JSON.stringify([atom.next, nextAccount.equity, nextAccount.price, nextAccount.exposure]);
    const policy: any = policies.get(key); assert.ok(policy);
    const upper = uppers[atom.next](nextAccount), mean = means[atom.next](nextAccount);
    assert.equal(mean.objective, "mean-terminal-equity-ratio");
    const existing = branches.get(key);
    if (existing) existing.probability += atom.probability;
    else branches.set(key, { probability: atom.probability, next: atom.next, account: nextAccount,
      immediate: next.logGrowth, continuation: policy.lower, continuationRequest: policy.quantity, upper,
      upperGap: upper - policy.lower, meanTerminalEquity: next.equity * mean.value, meanRequest: mean.quantity });
  }
  const rows = [...branches.values()];
  const weighted = (field: string) => rows.reduce((sum, row) => sum + row.probability * row[field], 0);
  const byLeaf = [...new Set(rows.map(row => row.next))].map(next => {
    const subset = rows.filter(row => row.next === next), mass = subset.reduce((sum, row) => sum + row.probability, 0);
    return { next, states: subset.length, mass,
      continuation: subset.reduce((sum, row) => sum + row.probability * row.continuation, 0) / mass,
      upper: subset.reduce((sum, row) => sum + row.probability * row.upper, 0) / mass,
      upperGap: subset.reduce((sum, row) => sum + row.probability * row.upperGap, 0) / mass,
      minimumContinuation: Math.min(...subset.map(row => row.continuation)),
      maximumContinuation: Math.max(...subset.map(row => row.continuation)),
      maximumUpperGap: Math.max(...subset.map(row => row.upperGap)),
      holdingMass: subset.filter(row => row.continuationRequest === 0).reduce((sum, row) => sum + row.probability, 0),
      uniqueRequests: new Set(subset.map(row => row.continuationRequest)).size,
      minimumRequest: Math.min(...subset.map(row => row.continuationRequest)),
      maximumRequest: Math.max(...subset.map(row => row.continuationRequest)) };
  }).sort((a, b) => b.mass - a.mass);
  const ordered = [...rows].sort((a, b) => b.probability - a.probability);
  const result = { lot, quantity, value: solved.value, states: rows.length,
    immediate: weighted("immediate"), continuation: weighted("continuation"),
    continuationUpper: weighted("upper"), continuationUpperGap: weighted("upperGap"),
    exactMeanTerminalEquity: weighted("meanTerminalEquity"),
    exactMeanLogUpper: Math.log(weighted("meanTerminalEquity") / account.equity),
    minimumContinuation: Math.min(...rows.map(row => row.continuation)),
    maximumContinuation: Math.max(...rows.map(row => row.continuation)),
    maximumUpperGap: Math.max(...rows.map(row => row.upperGap)), byLeaf,
    probabilityQuantiles: [.5, .9, .99, .999].map(mass => {
      let cumulative = 0, count = 0, weightedGap = 0;
      for (const row of ordered) { cumulative += row.probability; weightedGap += row.probability * row.upperGap; count++;
        if (cumulative >= mass) break; }
      return { targetMass: mass, count, mass: cumulative, weightedUpperGap: weightedGap };
    }), seconds: (performance.now() - begin) / 1000 };
  assert.ok(Math.abs(result.immediate + result.continuation - result.value) <= 2e-12);
  results.push(result); console.log(JSON.stringify({ ...result, byLeaf: undefined, probabilityQuantiles: undefined }));
}
save("summary.json", { contract: "native-execution-continuation-gap-probe-v1", source,
  sourceConfigHash: hash(path.join(source, "config.json")), sourceSummaryHash: hash(path.join(source, "summary.json")),
  lawHash: hash(lawFile), modelHash: sourceSummary.modelHash, account, leaf, step, lots,
  scope: "Exact fixed-root H2 decomposition under the unchanged native one-second law. The continuation upper observes its next opening before choosing a continuous order, so its gap is diagnostic only and never an executable value.",
  results, elapsedSeconds: (performance.now() - started) / 1000 });
