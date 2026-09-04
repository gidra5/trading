/** Fixed-law three-event values for predeclared root orders; never global H3 optimality. */
import fs from "node:fs";
import path from "node:path";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { restoreEventPolicy } from "../packages/bot-algo/src/event-log-policy.js";
import { decideEventOneStep } from "../packages/bot-algo/src/event-one-step.js";
import { prepareEventThreeStepActions } from "../packages/bot-algo/src/event-three-step.js";
const arg = (k: string) => { const at = process.argv.indexOf(`--${k}`); return at < 0 ? "" : process.argv[at + 1]; };
if (!arg("source") || !arg("output") || !arg("window")) throw new Error("Specify merged suite --source, --window and new --output");
const root = path.resolve(__dirname, ".."), directory = (s: string) => path.join(root, "data/benchmarks", s);
const source = directory(arg("source")), output = directory(arg("output"));
if (fs.existsSync(output)) throw new Error("Choose a new output directory");
const read = (d: string, f: string) => JSON.parse(fs.readFileSync(path.join(d, f), "utf8"));
assert.equal(read(source, "config.json").contract, "event-fixed-law-suite-merge-v1");
const window = read(source, "summary.json").results.find((r: any) => r.window.id === arg("window"));
assert.ok(window, "Unknown scored window");
const forecastSource = read(window.h1Source, "config.json").source, modelFile = `${window.window.id}-model.json`;
const modelHash = createHash("sha256").update(fs.readFileSync(path.join(forecastSource, modelFile))).digest("hex");
assert.equal(modelHash, window.modelHash);
const policy = restoreEventPolicy(read(forecastSource, modelFile).policy);
const index = Number(arg("index") || 0), budget = Number(arg("budget") || 64), limitOutcomes = arg("limit") ? Number(arg("limit")) : undefined;
const offsetOutcomes = Number(arg("offset") || 0);
const tolerance = Number(arg("tolerance") || 1e-7);
const warmStart = !process.argv.includes("--cold");
const globalUpper = process.argv.includes("--global-upper"), shadowPoints = Number(arg("points") || 129);
const trace = read(window.h2Source, `${window.window.id}-trades.json`);
assert.ok(Number.isInteger(index) && trace[index] && Number.isInteger(budget) && budget >= 2);
const state = trace[index], account = { equity: state.equityBefore, price: state.order.price, exposure: state.exposureBefore };
const h1 = decideEventOneStep(policy.model.kernels[state.leaf], account, policy.costs, "marked");
const quantities = [...new Set(arg("quantities") ? arg("quantities").split(",").map(Number) : [0, h1.quantity, state.order.quantity])];
assert.ok(quantities.length > 0 && quantities.every(Number.isFinite));
fs.mkdirSync(output, { recursive: true });
const save = (name: string, value: unknown) => fs.writeFileSync(path.join(output, name), JSON.stringify(value,
  (_, v) => typeof v === "number" && !Number.isFinite(v) ? String(v) : v, 2));
save("config.json", { contract: "event-three-step-action-probe-v1", source, forecastSource, modelHash, window: window.window,
  index, time: state.time, leaf: state.leaf, account, h1Quantity: h1.quantity, h2Quantity: state.order.quantity,
  quantities, budget, tolerance, limitOutcomes, offsetOutcomes, warmStart, globalUpper, shadowPoints, terminal: "marked", rootCandidatesOnly: true,
  method: "Keep the saved forecast and observed account fixed. Bound the H3 value of each specified root order by summing every forecast successor's bounded H2 continuation; no realized future path is read by the evaluator. Identical successor accounts are coalesced only after checking every original outcome for liquidation. A limit is a computation profile, not a sampled score. A dominant candidate is proved only among the listed quantities, never the full root lot lattice." });
save("sources.json", Object.fromEntries(["scripts/probe-event-three-step.ts", "packages/bot-algo/src/event-three-step.ts",
  "packages/bot-algo/src/event-two-step.ts", "packages/bot-algo/src/event-one-step.ts", "packages/bot-algo/src/event-one-step-prepared.ts",
  "packages/bot-algo/src/event-one-step-upper.ts", "packages/bot-algo/src/event-holding-law.ts", "packages/bot-algo/src/event-multi-step-upper.ts", "packages/bot-algo/src/event-log-policy.ts"]
  .map(f => [f, fs.readFileSync(path.join(root, f), "utf8")])));
const started = performance.now(), evaluate = prepareEventThreeStepActions(policy.model, policy.costs, "marked", { globalUpper, shadowPoints });
const preparationSec = (performance.now() - started) / 1000, results: any[] = [];
for (const quantity of quantities) {
  const begin = performance.now(), branches: any[] = [];
  const result = evaluate(state.leaf, account, quantity, { maxEvaluations: budget, tolerance, limitOutcomes, offsetOutcomes, warmStart,
    onBranch: (r, i, total) => {
      branches.push({ next: r.next, probability: r.probability, account: r.account, logHolding: r.logHolding,
        quantity: r.result.quantity, lowerValue: r.result.lowerValue, upperValue: r.result.upperValue,
        gap: r.result.gap, converged: r.result.converged, feasible: r.result.feasible, search: r.result.search });
      if ((i + 1) % 100 === 0) {
        const progress = { quantity, outcomes: i + 1, total, elapsedSec: (performance.now() - begin) / 1000 };
        save("progress.json", progress); console.log(JSON.stringify(progress));
      }
    } });
  const row = { result, elapsedSec: (performance.now() - begin) / 1000 };
  save(`action-${results.length}-branches.json`, branches); results.push(row);
  save("summary.json", { preparationSec, results, elapsedSec: (performance.now() - started) / 1000 }); console.log(JSON.stringify(row,
    (_, v) => typeof v === "number" && !Number.isFinite(v) ? String(v) : v));
}
const best = [...results].sort((a, b) => b.result.lowerValue - a.result.lowerValue)[0];
const dominant = results.every(r => r.result.complete) && Number.isFinite(best.result.lowerValue)
  && results.every(r => r === best || best.result.lowerValue > r.result.upperValue);
save("summary.json", { preparationSec, results, dominantCandidate: dominant ? best.result.quantity : null, rootCandidatesOnly: true,
  elapsedSec: (performance.now() - started) / 1000 });
