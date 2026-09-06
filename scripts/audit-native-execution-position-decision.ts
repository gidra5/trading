/** Validate the bounded position-coordinate and remaining-root decision artifacts. */
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";

const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
const root = path.resolve(__dirname, ".."), directory = (name: string) => path.resolve(root, "data/benchmarks", name);
const output = directory(arg("output")); assert.ok(arg("output") && !fs.existsSync(output));
const names = {
  singleton: "event-native-execution-singletons-v514",
  intervalBound: "event-native-execution-root-region-bounds-v515",
  exactSamples: "event-native-execution-root-region-samples-v516",
  fixedUpper: "event-native-execution-root-region-upper-v517",
  target: "event-native-execution-target-exposure-stability-v519",
  concavity: "event-native-execution-h2-concavity-v520",
};
const read = (file: string) => JSON.parse(fs.readFileSync(file, "utf8"));
const hash = (file: string) => createHash("sha256").update(fs.readFileSync(file)).digest("hex");
const artifacts = Object.fromEntries(Object.entries(names).map(([key, name]) => {
  const folder = directory(name), summaryFile = path.join(folder, "summary.json"), sourcesFile = path.join(folder, "sources.json");
  const sources = read(sourcesFile);
  for (const [file, text] of Object.entries(sources)) assert.equal(fs.readFileSync(path.join(root, file), "utf8"), text,
    `${name} source changed: ${file}`);
  return [key, { name, folder, summary: read(summaryFile), summaryHash: hash(summaryFile), sourcesHash: hash(sourcesFile) }];
})) as Record<keyof typeof names, { name: string; folder: string; summary: any; summaryHash: string; sourcesHash: string }>;
const singleton = artifacts.singleton.summary, interval = artifacts.intervalBound.summary;
const samples = artifacts.exactSamples.summary, fixed = artifacts.fixedUpper.summary;
const target = artifacts.target.summary, concavity = artifacts.concavity.summary;
assert.equal(singleton.contract, "native-execution-singleton-h2-probe-v1"); assert.equal(singleton.completed, true);
assert.equal(singleton.results.length, 22); assert.equal(singleton.results.filter((r: any) => r.status === "complete").length, 2);
assert.equal(singleton.results.filter((r: any) => r.status === "pruned").length, 20);
assert.ok(Math.abs(singleton.incumbentRequest + .72268) < 1e-12);
assert.ok(Math.abs(singleton.incumbent - 0.000023490190236269185) < 1e-15);
assert.equal(interval.contract, "native-execution-root-region-bounds-v1"); assert.equal(interval.completed, false);
assert.equal(interval.results.length, 1); assert.equal(interval.results[0].boxes, 1222);
assert.ok(interval.results[0].upperValue > singleton.incumbent && interval.results[0].seconds > 100);
assert.equal(samples.contract, "native-execution-root-region-samples-v1"); assert.equal(samples.completed, true);
assert.equal(samples.results.length, 12); assert.ok(samples.results.every((r: any) => r.status === "complete"));
assert.ok(Math.max(...samples.results.map((r: any) => r.value)) < singleton.incumbent);
assert.equal(fixed.contract, "native-execution-root-region-upper-samples-v1"); assert.equal(fixed.samplesHash, artifacts.exactSamples.summaryHash);
assert.ok(fixed.results.every((r: any) => !r.belowIncumbentAtEverySample));
for (const region of fixed.results) for (const row of region.rows) if (row.exactValue !== null)
  assert.ok(row.upperValue >= row.exactValue - 1e-10);
assert.equal(target.contract, "native-execution-target-stability-v1"); assert.equal(target.samplesHash, artifacts.exactSamples.summaryHash);
assert.equal(target.comparison.sameTargetCount, 0); assert.equal(target.comparison.stableExposure[0].count, 0);
assert.ok(target.comparison.stableExposure.find((r: any) => r.tolerance === .001).mass < .2);
assert.equal(concavity.contract, "native-execution-h2-concavity-probe-v1"); assert.ok(concavity.violatedTriples > 0);
const result = {
  contract: "native-execution-position-decision-audit-v1",
  artifacts: Object.fromEntries(Object.entries(artifacts).map(([key, value]) => [key,
    { name: value.name, summaryHash: value.summaryHash, sourcesHash: value.sourcesHash }])),
  singleton: { total: singleton.results.length, exact: singleton.results.filter((r: any) => r.status === "complete").length,
    pruned: singleton.results.filter((r: any) => r.status === "pruned").length,
    incumbent: singleton.incumbent, incumbentRequest: singleton.incumbentRequest },
  intervalBound: { regionsCompleted: interval.results.length, boxes: interval.results[0].boxes,
    upperValue: interval.results[0].upperValue, seconds: interval.results[0].seconds, useful: false },
  exactSamples: { count: samples.results.length, maximum: Math.max(...samples.results.map((r: any) => r.value)) },
  fixedUpper: { minimumExactSlackBps: Math.min(...fixed.results.flatMap((r: any) => r.rows)
    .filter((r: any) => r.exactSlackBps !== null).map((r: any) => r.exactSlackBps)), useful: false },
  targetStability: target.comparison,
  concavity: { cases: concavity.cases, triples: concavity.finiteTriples,
    violations: concavity.violatedTriples, maximumSecondDifference: concavity.maximumSecondDifference },
  decision: "Retain lifecycle position decomposition for exact accounting and signal classification. Keep the single-asset action optimizer at account level with the full request/value curve. A full target-exposure curve plus global coordinator is an equivalent reparameterization, while local optima or stable-target restrictions discard observed action dependence. The next exact H2 method must preserve root nonanticipativity and mixed-integer recourse explicitly.",
  references: [
    "https://optimization-online.org/2014/08/4474/",
    "https://optimization-online.org/2016/03/5391/",
    "https://optimization-online.org/2024/11/on-the-relu-lagrangian-cuts-for-stochastic-mixed-integer-programming/",
  ],
};
fs.mkdirSync(output, { recursive: true });
fs.writeFileSync(path.join(output, "source.ts"), fs.readFileSync(__filename, "utf8"));
fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify(result, null, 2));
console.log(JSON.stringify(result));
