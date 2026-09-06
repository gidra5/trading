/** Exact H2 values at a small predeclared root candidate set; not global H2 search. */
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { prepareEventExecutionOneStep } from "../packages/bot-algo/src/event-execution-one-step.js";
import { evaluateEventExecutionPath } from "../packages/bot-algo/src/event-execution-path.js";
const arg = (key: string, fallback = "") => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? fallback : process.argv[i + 1]; };
const root = path.resolve(__dirname, ".."), directory = (name: string) => path.resolve(root, "data/benchmarks", name);
const read = (file: string) => JSON.parse(fs.readFileSync(file, "utf8"));
const hash = (file: string) => createHash("sha256").update(fs.readFileSync(file)).digest("hex");
const source = directory(arg("source")), output = directory(arg("output"));
const probeIndex = Number(arg("probe", "0")), seconds = Number(arg("seconds-per-root", "45"));
const requested = arg("requests").split(",").filter(Boolean).map(Number);
assert.ok(arg("source") && arg("output") && !fs.existsSync(output));
assert.ok(Number.isSafeInteger(probeIndex) && probeIndex >= 0 && Number.isFinite(seconds) && seconds > 0 && seconds <= 60);
assert.ok(requested.every(Number.isFinite));
const config = read(path.join(source, "config.json")), compiled = read(path.join(source, "summary.json"));
assert.equal(config.contract, "native-event-execution-law-v1");
assert.equal(hash(path.join(source, "law.json")), compiled.lawHash);
const law = read(path.join(source, "law.json")), probe = compiled.probes[probeIndex]; assert.ok(probe);
assert.equal(hash(path.join(config.source, "model.json")), law.modelHash);
for (const ref of config.sourceReferences) assert.equal(hash(ref.file), ref.sha256);
const kernels = law.kernels.map((k: any[]) => k.map(a => ({ ...a, path: law.paths[a.path] })));
assert.ok(kernels.every((k: any[]) => k.every(a => Number.isInteger(a.next) && a.next >= 0 && a.next < kernels.length)));
const solvers = kernels.map((k: any[]) => prepareEventExecutionOneStep(k, "marked"));
const oneStep = solvers[probe.leaf](probe.account);
const requests = [...new Set([0, probe.original.quantity, oneStep.quantity, ...requested])] as number[];
assert.ok(requests.every(request => Math.abs(request / law.costs.quantityStep
  - Math.round(request / law.costs.quantityStep)) < 1e-7));
fs.mkdirSync(output, { recursive: true });
const save = (name: string, data: unknown) => fs.writeFileSync(path.join(output, name), JSON.stringify(data,
  (_, v) => typeof v === "number" && !Number.isFinite(v) ? String(v) : v, 2));
save("config.json", { contract: "native-execution-bellman-candidates-v1", source, lawHash: compiled.lawHash,
  modelHash: law.modelHash, probeIndex, probe, oneStep, requests, secondsPerRoot: seconds,
  method: "Before reading any future evaluation path, compare zero, the saved old H1 request, the global execution-H1 request, and any explicitly predeclared request list. For each root outcome apply the exact execution transition, then globally optimize execution H1 at the resulting account and successor leaf. Sum probability times immediate log growth plus continuation. This is exact H2 value for each fully evaluated fixed request, hence a candidate lower bound on global H2. Partial time-capped sums are not reported as values. Neither candidate ranking nor H1 completeness proves a global H2 optimum or deeper convergence." });
save("sources.json", Object.fromEntries(["scripts/probe-native-execution-bellman.ts",
  "packages/bot-algo/src/event-execution-one-step.ts", "packages/bot-algo/src/event-execution-path.ts",
  "packages/bot-algo/src/event-log-policy.ts"].map(file => [file, fs.readFileSync(path.join(root, file), "utf8")])));
const cache = new Map<string, ReturnType<ReturnType<typeof prepareEventExecutionOneStep>>>();
const results: any[] = [], started = performance.now();
for (const request of requests) {
  const begin = performance.now(), branches = [], kernel = kernels[probe.leaf];
  let value = 0, immediate = 0, continuation = 0, processedMass = 0, calls = 0, hits = 0, evaluatedOrders = 0;
  let complete = true, terminalRuin = false, modeledRejectionMass = 0;
  for (let index = 0; index < kernel.length; index++) {
    const atom = kernel[index]; if (!atom.probability) continue;
    if (performance.now() - begin > seconds * 1000) { complete = false; break; }
    const next = evaluateEventExecutionPath(atom.path, probe.account, request);
    if (next.canceled) modeledRejectionMass += atom.probability;
    if (!Number.isFinite(next.logGrowth)) {
      branches.push({ index, probability: atom.probability, next: atom.next, rootRuin: true });
      value = -Infinity; terminalRuin = true; break;
    }
    const account = { equity: next.equity, price: next.price, exposure: next.exposure };
    const key = JSON.stringify([atom.next, account.equity, account.price, account.exposure]);
    let child = cache.get(key);
    if (child) hits++;
    else { child = solvers[atom.next](account); cache.set(key, child); calls++; evaluatedOrders += child.search.evaluatedOrders; }
    assert.ok(child.complete);
    branches.push({ index, probability: atom.probability, next: atom.next, account, rootFilled: next.filledQuantity,
      rootCanceled: next.canceled, logGrowth: next.logGrowth, continuation: child.value,
      continuationRequest: child.quantity, complete: child.complete });
    processedMass += atom.probability; immediate += atom.probability * next.logGrowth;
    continuation += atom.probability * child.value;
    value += atom.probability * (next.logGrowth + child.value);
    if (!Number.isFinite(child.value)) { terminalRuin = true; value = -Infinity; break; }
    if (index % 100 === 0) save("progress.json", { request, index, total: kernel.length, calls, hits,
      elapsedSeconds: (performance.now() - begin) / 1000 });
  }
  const result = { request, complete, finite: complete && Number.isFinite(value), terminalRuin,
    value: complete ? value : null, immediate: complete && !terminalRuin ? immediate : null,
    continuation: complete && !terminalRuin ? continuation : null,
    modeledRejectionMass: complete && !terminalRuin ? modeledRejectionMass : null,
    processedMass, processedOutcomes: branches.length, totalOutcomes: kernel.length,
    continuationCalls: calls, cacheHits: hits, evaluatedOrders, seconds: (performance.now() - begin) / 1000, branches };
  results.push(result); console.log(JSON.stringify({ ...result, branches: undefined }));
  save("summary.json", { results, elapsedSeconds: (performance.now() - started) / 1000,
    completeCandidateValues: results.filter(r => r.complete).length, globallySearchedRoot: false, cacheSize: cache.size });
}
save("progress.json", { phase: "finished", completeCandidateValues: results.filter(r => r.complete).length, requested: requests.length });
if (results.some(r => !r.complete)) process.exitCode = 1;
