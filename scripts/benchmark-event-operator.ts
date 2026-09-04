import fs from "node:fs";
import path from "node:path";
import Module from "node:module";
import { createHash } from "node:crypto";
import ts from "typescript";
import { buildEventPolicy, type EventPolicy, type SerializedEventPolicy } from "../packages/bot-algo/src/event-log-policy.js";

const root = path.resolve(__dirname, ".."), at = (key: string) => process.argv[process.argv.indexOf(`--${key}`) + 1];
if (!["source", "window", "output"].every(key => process.argv.includes(`--${key}`))) throw new Error("Specify --source, --window and --output");
const source = path.join(root, "data/benchmarks", at("source")), output = path.join(root, "data/benchmarks", at("output"));
if (fs.existsSync(output)) throw new Error("Choose a new benchmark output");
const policyFile = "packages/bot-algo/src/event-log-policy.ts";
const sources = JSON.parse(fs.readFileSync(path.join(source, "sources.json"), "utf8"));
const referenceSource = sources[policyFile] as string;
if (!referenceSource) throw new Error("Source artifact has no policy implementation snapshot");
// Load the authoritative saved implementation in memory; no legacy source copy
// is added to the workspace. Relative imports retain their package context.
const filename = path.join(root, "packages/bot-algo/src/event-policy-benchmark-reference.ts");
const referenceModule = new Module(filename, module) as Module & { _compile(source: string, filename: string): void };
referenceModule.filename = filename;
referenceModule.paths = module.paths;
referenceModule._compile(ts.transpileModule(referenceSource, { compilerOptions: { module: ts.ModuleKind.CommonJS,
  target: ts.ScriptTarget.ES2022 } }).outputText, filename);
const referenceBuild = referenceModule.exports.buildEventPolicy as typeof buildEventPolicy;
const saved = JSON.parse(fs.readFileSync(path.join(source, `${at("window")}-model.json`), "utf8")).policy as SerializedEventPolicy;
const options = { depths: saved.tables.length, referenceEquity: saved.equities[2], referencePrice: saved.prices[1], actionSteps: (saved.targets.length - 1) / 2 };
const compare = (a: EventPolicy, b: EventPolicy) => {
  let maxError = 0, infinityMismatch = 0;
  for (let d = 0; d < a.tables.length; d++) for (let i = 0; i < a.tables[d].holdValues.length; i++) {
    const x = a.tables[d].holdValues[i], y = b.tables[d].holdValues[i];
    if (Number.isFinite(x) !== Number.isFinite(y)) infinityMismatch++;
    if (Number.isFinite(x) && Number.isFinite(y)) maxError = Math.max(maxError, Math.abs(x - y));
  }
  return { maxError, infinityMismatch };
};
const measure = (build: typeof buildEventPolicy) => {
  const start = performance.now(), policy = build(saved.model, saved.costs, options);
  return { policy, seconds: (performance.now() - start) / 1000 };
};
const trials = [];
for (let trial = 0; trial < 3; trial++) {
  const first = measure(trial % 2 ? buildEventPolicy : referenceBuild);
  const second = measure(trial % 2 ? referenceBuild : buildEventPolicy);
  const reference = trial % 2 ? second : first, current = trial % 2 ? first : second;
  trials.push({ trial, referenceSeconds: reference.seconds, currentSeconds: current.seconds, ...compare(reference.policy, current.policy) });
  console.log(JSON.stringify(trials.at(-1)));
}
const median = (values: number[]) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];
const result = { source, window: at("window"), options, atoms: saved.model.kernels.flat().length, states: saved.model.kernels.length,
  referenceHash: createHash("sha256").update(referenceSource).digest("hex"),
  currentHash: createHash("sha256").update(fs.readFileSync(path.join(root, policyFile))).digest("hex"), trials,
  medianReferenceSeconds: median(trials.map(t => t.referenceSeconds)), medianCurrentSeconds: median(trials.map(t => t.currentSeconds)) };
fs.writeFileSync(output, JSON.stringify(result, null, 2));
if (trials.some(t => t.infinityMismatch || t.maxError > 1e-10)) throw new Error("Bellman operator changed reference values");
