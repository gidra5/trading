/** Positive moment compression of an execution law before deeper Bellman work. */
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { compressEventKernel, eventQuadratureFeatures } from "../packages/bot-algo/src/event-quadrature.js";
import type { MoveAtom } from "../packages/bot-algo/src/event-distribution.js";
import { prepareEventExecutionOneStep } from "../packages/bot-algo/src/event-execution-one-step.js";

const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
const root = path.resolve(__dirname, ".."), directory = (name: string) => path.resolve(root, "data/benchmarks", name);
const source = directory(arg("source")), output = directory(arg("output"));
assert.ok(arg("source") && arg("output") && !fs.existsSync(output));
const read = (file: string) => JSON.parse(fs.readFileSync(file, "utf8"));
const hash = (file: string) => createHash("sha256").update(fs.readFileSync(file)).digest("hex");
const config = read(path.join(source, "config.json")), compiled = read(path.join(source, "summary.json"));
const law = read(path.join(source, "law.json"));
assert.equal(config.contract, "native-event-execution-law-v1");
assert.equal(hash(path.join(source, "law.json")), compiled.lawHash);
assert.equal(hash(path.join(config.source, "model.json")), law.modelHash);
for (const ref of config.sourceReferences) assert.equal(hash(ref.file), ref.sha256);

type IndexedAtom = MoveAtom & { path: number };
const expanded = law.kernels.map((kernel: Array<{ probability: number; next: number; path: number }>) =>
  kernel.map(atom => { const execution = law.paths[atom.path]; return { probability: atom.probability, next: atom.next,
    return: execution.closeRatio - 1, duration: execution.seconds / 60,
    low: Math.min(1, execution.lowRatio) - 1, high: Math.max(1, execution.highRatio) - 1, path: atom.path } as IndexedAtom; }));
const compact = expanded.map((kernel: IndexedAtom[]) => compressEventKernel(kernel, config.clock) as IndexedAtom[]);
assert.ok(compact.every(kernel => kernel.length && kernel.every(atom => Number.isSafeInteger(atom.path) && law.paths[atom.path])));

const moments = expanded.map((kernel: IndexedAtom[], leaf: number) => {
  const before = eventQuadratureFeatures(kernel[0]).map((_, i) => kernel.reduce((sum, atom) =>
    sum + atom.probability * eventQuadratureFeatures(atom)[i], 0));
  const after = eventQuadratureFeatures(compact[leaf][0]).map((_, i) => compact[leaf].reduce((sum, atom) =>
    sum + atom.probability * eventQuadratureFeatures(atom)[i], 0));
  return { leaf, before: kernel.length, after: compact[leaf].length,
    maximumMomentError: Math.max(...before.map((value, i) => Math.abs(value - after[i]))) };
});
assert.ok(moments.every(row => row.maximumMomentError <= 1e-9));

const execution = (kernels: IndexedAtom[][]) => kernels.map(kernel => kernel.map(atom => ({
  probability: atom.probability, next: atom.next, path: law.paths[atom.path],
})));
const originalSolvers = execution(expanded).map(kernel => prepareEventExecutionOneStep(kernel, "marked"));
const compactSolvers = execution(compact).map(kernel => prepareEventExecutionOneStep(kernel, "marked"));
const probes = compiled.probes.map((probe: any) => {
  const uncompressed = originalSolvers[probe.leaf](probe.account), compressed = compactSolvers[probe.leaf](probe.account);
  assert.ok(uncompressed.complete && compressed.complete);
  return { time: probe.time, leaf: probe.leaf, account: probe.account, original: compressed, uncompressed,
    valueErrorBps: (compressed.value - uncompressed.value) * 10_000,
    quantityDifference: compressed.quantity - uncompressed.quantity };
});

fs.mkdirSync(output, { recursive: true });
const save = (file: string, value: unknown) => fs.writeFileSync(path.join(output, file), JSON.stringify(value,
  (_, item) => typeof item === "number" && !Number.isFinite(item) ? String(item) : item, 2));
save("config.json", { ...config, compressionSource: source, compressionLawHash: compiled.lawHash,
  method: "Apply the existing positive Tchakaloff-style reduction independently within actual/reciprocal event class and successor strata. Preserve mass, arithmetic and reciprocal return, log return, second return moment, duration and log-duration moments; retain adverse excursion/duration Pareto frontiers. Every compact atom remains an observed execution path. This defines a smaller approximate forecasting law and does not claim exact Bellman equivalence." });
save("sources.json", Object.fromEntries(["scripts/compress-native-event-execution-law.ts",
  "packages/bot-algo/src/event-quadrature.ts", "packages/bot-algo/src/event-execution-one-step.ts"]
  .map(file => [file, fs.readFileSync(path.join(root, file), "utf8")])));
save("law.json", { ...law, kernels: compact.map(kernel => kernel.map(atom => ({
  probability: atom.probability, next: atom.next, path: atom.path }))) });
const lawHash = hash(path.join(output, "law.json"));
save("summary.json", { window: compiled.window, compressionSource: source,
  paths: law.paths.length, atomsBefore: expanded.reduce((sum, kernel) => sum + kernel.length, 0),
  atomsAfter: compact.reduce((sum, kernel) => sum + kernel.length, 0), moments, probes,
  exactOldJointProjection: false, positiveObservedSupport: true, lawHash });
console.log(JSON.stringify({ atomsBefore: expanded.map(kernel => kernel.length), atomsAfter: compact.map(kernel => kernel.length),
  moments, probes: probes.map((probe: any) => ({ leaf: probe.leaf, valueErrorBps: probe.valueErrorBps,
    uncompressedQuantity: probe.uncompressed.quantity, compressedQuantity: probe.original.quantity })) }));
