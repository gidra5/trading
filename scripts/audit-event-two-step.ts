/** Bounded two-event searches against an independently enumerated Bellman law. */
import fs from "node:fs";
import path from "node:path";
import assert from "node:assert/strict";
import { decideEventTwoStep, prepareEventTwoStep } from "../packages/bot-algo/src/event-two-step.js";
import { eventBellmanReference } from "../packages/bot-algo/test/event-bellman-reference.js";
import { DEFAULT_EVENT_COSTS } from "../packages/bot-algo/src/event-log-policy.js";
import { EVENT_FEATURES, type EventDistribution } from "../packages/bot-algo/src/event-distribution.js";

const arg = (k: string) => { const at = process.argv.indexOf(`--${k}`); return at < 0 ? "" : process.argv[at + 1]; };
if (!arg("reference") || !arg("output")) throw new Error("Specify --reference and new --output");
const root = path.resolve(__dirname, ".."), directory = (s: string) => path.join(root, "data/benchmarks", s);
const source = directory(arg("reference")), output = directory(arg("output"));
if (fs.existsSync(output)) throw new Error("Choose a new output directory");
const config = JSON.parse(fs.readFileSync(path.join(source, "config.json"), "utf8"));
assert.equal(config.contract, "event-fixed-model-optimality-audit-v1");
const seed = 20260905, randomCases = Number(arg("cases") || 300), tolerance = 1e-7;
const copies = Number(arg("copies") || 1);
const globalUpper = process.argv.includes("--global-upper"), shadowPoints = Number(arg("points") || 9);
assert.ok(Number.isInteger(copies) && copies >= 1 && copies <= 64);
assert.ok(Number.isInteger(randomCases) && randomCases > 0);
let state = seed;
const random = () => { state = Math.imul(state, 1664525) + 1013904223 | 0; return (state >>> 0) / 2 ** 32; };
const pick = <T>(values: T[]) => values[Math.floor(random() * values.length)];
const cases = config.cases.filter((r: any) => r.depths.includes(2)).flatMap((r: any) => r.exposures.flatMap((exposure: number) =>
  r.model.kernels.map((_: any, leaf: number) => ({ name: r.name, model: r.model as EventDistribution, costs: r.costs,
    account: { equity: r.equity, price: r.price, exposure }, leaf }))));
for (let i = 0; i < randomCases; i++) {
  const equity = 10 + random() * 1000, price = 5 + random() * 200, step = equity / (10 * price), leverage = pick([1, 2, 5]);
  const costs = { ...DEFAULT_EVENT_COSTS, maxLeverage: leverage, feeBps: pick([0, 10, 100]), slippageBps: pick([0, 2, 50]),
    minNotional: pick([0, .03, .12]) * equity, maxNotional: pick([.2, .8, 2, 4]) * equity,
    minQuantity: pick([0, 1, 3]) * step, quantityStep: step, maintenanceMargin: pick([.005, .1, .3]),
    longBorrowBpsPerDay: pick([0, 1, 1000]), shortBorrowBpsPerDay: pick([0, 1, 1000]) };
  const kernels = [0, 1].map(() => {
    const masses = Array.from({ length: pick([2, 3]) }, () => .01 + random()), total = masses.reduce((s, p) => s + p, 0);
    return masses.map(probability => {
      const r = (random() - .5) * .7;
      return { probability: probability / total, return: r, low: Math.max(-.99, Math.min(0, r) - random() * .3),
        high: Math.max(0, r) + random() * .3, duration: pick([1, 60, 1440]), next: pick([0, 1]) };
    });
  });
  const model = { version: 1, clock: { thresholdBps: 20, maxCandles: 3 }, featureNames: EVENT_FEATURES,
    nodes: [{ feature: -1, cut: 0, left: -1, right: -1, leaf: 0 }], trainingSamples: 2, counts: [1, 1],
    priorClasses: [], classProbabilities: [], kernels } as EventDistribution;
  cases.push({ name: `seeded-${i}`, model, costs, account: { equity, price,
    exposure: pick([0, .2, .95, 1.01, 1.5, -1.4, 100]) * leverage }, leaf: pick([0, 1]) });
}
const budgets = [2, 16, 512];
fs.mkdirSync(output, { recursive: true });
const save = (name: string, value: unknown) => fs.writeFileSync(path.join(output, name), JSON.stringify(value,
  (_, v) => typeof v === "number" && !Number.isFinite(v) ? String(v) : v, 2));
save("config.json", { contract: "event-two-step-bound-audit-v1", source, seed, randomCases, copies, globalUpper, shadowPoints, tolerance, budgets,
  method: "Freeze each transition law. Compare feasible lower values and interval upper bounds with independent exhaustive two-event Bellman enumeration. If copies>1, split each production atom's mass equally among identical copies to exercise dense-law evaluation without changing the reference distribution. Budget truncation must retain every unresolved interval; no hindsight path maximization.", cases });
save("sources.json", Object.fromEntries(["scripts/audit-event-two-step.ts", "packages/bot-algo/src/event-two-step.ts", "packages/bot-algo/src/event-one-step-upper.ts", "packages/bot-algo/src/event-one-step-prepared.ts",
  "packages/bot-algo/src/event-holding-law.ts", "packages/bot-algo/src/event-multi-step-upper.ts", "packages/bot-algo/src/event-one-step.ts", "packages/bot-algo/src/event-log-policy.ts", "packages/bot-algo/test/event-bellman-reference.ts"]
  .map(f => [f, fs.readFileSync(path.join(root, f), "utf8")])));
const results = [], started = performance.now();
for (const c of cases) for (const terminal of ["friction", "marked"] as const) {
  const expanded = copies === 1 ? c.model : { ...c.model, kernels: c.model.kernels.map(k =>
    k.flatMap(a => Array.from({ length: copies }, () => ({ ...a, probability: a.probability / copies })))) };
  const reference = eventBellmanReference(c.model, c.costs, { terminal });
  const prepared = globalUpper ? prepareEventTwoStep(expanded, c.costs, terminal, { globalUpper, shadowPoints }) : undefined;
  const start = performance.now(), optimal = reference.decide(c.leaf, c.account, 2), referenceMs = performance.now() - start;
  let previousLower = -Infinity, previousUpper = Infinity;
  for (const maxEvaluations of budgets) {
    const begin = performance.now(), fast = prepared ? prepared(c.leaf, c.account, { tolerance, maxEvaluations })
      : decideEventTwoStep(expanded, c.leaf, c.account, c.costs, terminal, { tolerance, maxEvaluations });
    const fastMs = performance.now() - begin, actual = reference.actionValue(c.leaf, c.account, 2, fast.quantity);
    const equal = (a: number, b: number) => a === b || Math.abs(a - b) <= 1e-10;
    const matches = equal(actual, fast.lowerValue) && (fast.lowerValue <= optimal.value + 1e-10 || equal(fast.lowerValue, optimal.value))
      && (fast.upperValue >= optimal.value - 1e-10 || equal(fast.upperValue, optimal.value))
      && (fast.lowerValue >= previousLower - 1e-10 || equal(fast.lowerValue, previousLower))
      && (fast.upperValue <= previousUpper + 1e-10 || equal(fast.upperValue, previousUpper))
      && (!fast.converged || fast.gap <= tolerance) && (maxEvaluations < 512 || fast.converged);
    const row = { name: c.name, exposure: c.account.exposure, terminal, maxEvaluations, optimal, fast, exactChosenValue: actual,
      regretBps: Number.isFinite(optimal.value) ? (optimal.value - actual) * 10000 : null,
      referenceMs: maxEvaluations === budgets[0] ? referenceMs : 0, fastMs, matches };
    results.push(row);
    if (!matches) { save("failure.json", { case: c, result: row, previousLower, previousUpper }); throw new Error(`Two-event bound audit failed: ${c.name} ${terminal} budget ${maxEvaluations}`); }
    previousLower = fast.lowerValue; previousUpper = fast.upperValue;
  }
}
const byBudget = budgets.map(budget => {
  const rows = results.filter(r => r.maxEvaluations === budget);
  return { budget, count: rows.length, converged: rows.filter(r => r.fast.converged).length,
    maximumRegretBps: Math.max(...rows.map(r => r.regretBps ?? 0)), maximumGapBps: Math.max(...rows.map(r => r.fast.gap * 10000)),
    fastMs: rows.reduce((s, r) => s + r.fastMs, 0), evaluations: rows.reduce((s, r) => s + r.fast.search.evaluations, 0) };
});
const summary = { count: results.length, byBudget, referenceMs: results.reduce((s, r) => s + r.referenceMs, 0),
  elapsedSec: (performance.now() - started) / 1000, results };
save("summary.json", summary); console.log(JSON.stringify({ ...summary, results: undefined }));
