/** Profile complete one-event request searches at saved calibration accounts. */
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { prepareEventExecutionOneStep } from "../packages/bot-algo/src/event-execution-one-step.js";
import { evaluateEventExecutionPath } from "../packages/bot-algo/src/event-execution-path.js";
const arg = (key: string, fallback = "") => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? fallback : process.argv[i + 1]; };
const root = path.resolve(__dirname, ".."), directory = (name: string) => path.resolve(root, "data/benchmarks", name);
const source = directory(arg("source")), output = directory(arg("output"));
assert.ok(arg("source") && arg("output") && !fs.existsSync(output));
const read = (file: string) => JSON.parse(fs.readFileSync(path.join(source, file), "utf8"));
const hash = (file: string) => createHash("sha256").update(fs.readFileSync(file)).digest("hex");
const config = read("config.json"), summary = read("summary.json"), law = read("law.json");
assert.equal(config.contract, "native-event-execution-law-v1"); assert.equal(hash(path.join(source, "law.json")), summary.lawHash);
const offset = Number(arg("offset", "0")), count = arg("count") === "all" ? summary.probes.length - offset : Number(arg("count", "1"));
assert.ok(Number.isInteger(count) && count > 0 && Number.isInteger(offset) && offset >= 0 && offset + count <= summary.probes.length);
fs.mkdirSync(output, { recursive: true });
const save = (name: string, data: unknown) => fs.writeFileSync(path.join(output, name), JSON.stringify(data,
  (_, v) => typeof v === "number" && !Number.isFinite(v) ? String(v) : v, 2));
save("config.json", { contract: "native-event-execution-one-step-probe-v1", source, lawHash: summary.lawHash, modelHash: law.modelHash,
  count, offset, terminal: "marked", probeHash: hash(path.join(source, "summary.json")),
  method: "Complete outcome-dependent acceptance partition and concave integer searches. Requests beyond the maximum potentially accepted notional are equivalent to no trade. Account states come from predeclared calibration probes. All request values use the fixed training-path mixture; realized calibration futures are never scored by the optimizer. This is H1 only, not deeper Bellman or stationary optimization." });
save("sources.json", Object.fromEntries(["scripts/probe-native-event-execution.ts", "packages/bot-algo/src/event-execution-one-step.ts",
  "packages/bot-algo/src/event-execution-path.ts", "packages/bot-algo/src/event-log-policy.ts"]
  .map(file => [file, fs.readFileSync(path.join(root, file), "utf8")])));
const started = performance.now(), results = [];
const prepared = new Map<number, ReturnType<typeof prepareEventExecutionOneStep>>();
for (const probe of summary.probes.slice(offset, offset + count)) {
  const kernel = law.kernels[probe.leaf].map((a: any) => ({ probability: a.probability, path: law.paths[a.path] }));
  const begin = performance.now();
  let solve = prepared.get(probe.leaf);
  if (!solve) { solve = prepareEventExecutionOneStep(kernel); prepared.set(probe.leaf, solve); }
  const preparedAt = performance.now(), result = solve(probe.account), solvedAt = performance.now();
  let value = 0, rejectedMass = 0;
  for (const atom of kernel) {
    const evaluated = evaluateEventExecutionPath(atom.path, probe.account, result.quantity);
    value += atom.probability * evaluated.logGrowth; rejectedMass += atom.probability * Number(evaluated.canceled);
  }
  assert.equal(value, result.value); assert.ok(result.value >= probe.best.value - 1e-12);
  const row = { time: probe.time, leaf: probe.leaf, account: probe.account, atoms: kernel.length, result, rejectedMass,
    original: probe.original, candidateBest: probe.best, improvementBps: (result.value - probe.original.value) * 10000,
    preparationSeconds: (preparedAt - begin) / 1000, solveSeconds: (solvedAt - preparedAt) / 1000 };
  results.push(row); save("summary.json", { results, elapsedSeconds: (performance.now() - started) / 1000 });
  console.log(JSON.stringify(row));
}
