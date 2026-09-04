/** Check recursive global upper bounds against finite-lattice Bellman optima. */
import fs from "node:fs";
import path from "node:path";
import assert from "node:assert/strict";
import { EVENT_FEATURES, type EventDistribution } from "../packages/bot-algo/src/event-distribution.js";
import { DEFAULT_EVENT_COSTS } from "../packages/bot-algo/src/event-log-policy.js";
import { prepareEventMultiStepUpper } from "../packages/bot-algo/src/event-multi-step-upper.js";
import { eventBellmanReference } from "../packages/bot-algo/test/event-bellman-reference.js";
const arg = (k: string) => { const i = process.argv.indexOf(`--${k}`); return i < 0 ? "" : process.argv[i + 1]; };
if (!arg("output")) throw new Error("Specify new --output");
const root = path.resolve(__dirname, ".."), output = path.join(root, "data/benchmarks", arg("output"));
if (fs.existsSync(output)) throw new Error("Choose a new output directory");
const seed = 20260906, count = Number(arg("cases") || 30); let state = seed;
const method = (arg("method") || "uniform") as "uniform" | "marginal";
const directions = process.argv.includes("--directions");
const random = () => { state = Math.imul(state, 1664525) + 1013904223 | 0; return (state >>> 0) / 2 ** 32; };
const pick = <T>(values: T[]) => values[Math.floor(random() * values.length)];
const cases = Array.from({ length: count }, (_, i) => {
  const kernels = [0, 1].map(() => {
    const p = .2 + .6 * random();
    return [p, 1 - p].map(probability => {
      const r = (random() - .5) * .16;
      return { probability, return: r, low: Math.min(0, r) - random() * .15, high: Math.max(0, r) + random() * .15,
        duration: pick([1, 30, 1440]), next: pick([0, 1]) };
    });
  });
  const model: EventDistribution = { version: 1, clock: { thresholdBps: 20, maxCandles: 3 }, featureNames: [...EVENT_FEATURES],
    nodes: [{ feature: -1, cut: 0, left: -1, right: -1, leaf: 0 }], trainingSamples: 1, counts: [1, 1], priorClasses: [], classProbabilities: [], kernels };
  const costs = { ...DEFAULT_EVENT_COSTS, maxLeverage: pick([1, 2]), quantityStep: 2, minQuantity: pick([0, 2, 6]),
    minNotional: pick([0, 20, 45]), maxNotional: 800, feeBps: pick([0, 12, 100]), slippageBps: 0,
    longBorrowBpsPerDay: pick([0, 1, 1000]), shortBorrowBpsPerDay: pick([0, 1, 1000]) };
  return { id: i, model, costs };
});
fs.mkdirSync(output, { recursive: true });
const save = (f: string, x: unknown) => fs.writeFileSync(path.join(output, f), JSON.stringify(x,
  (_, v) => typeof v === "number" && !Number.isFinite(v) ? String(v) : v, 2));
save("config.json", { contract: "event-recursive-upper-audit-v1", seed, cases, method, directions,
  method: "Independent exhaustive H1/H2/H3 optima on seeded two-state laws, varied fees/borrowing/order minima/leverage and starting long/short/cash inventory. Use large enough maximum orders to exercise the finite bound after its recovery-exclusion proof, rather than passing vacuously with infinity." });
save("sources.json", Object.fromEntries(["scripts/audit-event-multi-step-upper.ts", "packages/bot-algo/src/event-multi-step-upper.ts",
  "packages/bot-algo/test/event-bellman-reference.ts"].map(f => [f, fs.readFileSync(path.join(root, f), "utf8")])));
const results = [], started = performance.now();
for (const c of cases) for (const terminal of ["marked", "friction"] as const) {
  const upper = prepareEventMultiStepUpper(c.model, c.costs, terminal, { depth: 3, shadowPoints: 9, method });
  for (const exposure of [0, -1.9, 1.9]) for (const depth of [1, 2, 3]) {
    const account = { equity: 100, price: 10, exposure }, bound = upper.query(0, account, depth);
    const reference = eventBellmanReference(c.model, c.costs, { terminal, maxNodes: 1000000 });
    const exact = reference.decide(0, account, depth);
    const valid = bound.recoveryExcluded && Number.isFinite(bound.upperValue) && bound.upperValue >= exact.value - 1e-9;
    const row = { id: c.id, terminal, account, depth, bound, exact, valid }; results.push(row);
    if (!valid) { save("failure.json", { case: c, row }); throw new Error(`Upper audit failed: ${c.id} ${terminal} ${depth}`); }
    if (directions) for (const side of [-1, 1] as const) {
      const directional = upper.query(0, account, depth, side);
      const lots = Math.floor(c.costs.maxNotional / account.price / c.costs.quantityStep + 1e-8);
      const values = Array.from({ length: lots + 1 }, (_, i) => ({ quantity: side * i * c.costs.quantityStep,
        value: reference.actionValue(0, account, depth, side * i * c.costs.quantityStep) }));
      const expected = values.reduce((a, b) => a.value >= b.value ? a : b);
      const matches = directional.recoveryExcluded && directional.upperValue !== Infinity
        && (directional.upperValue >= expected.value - 1e-9 || directional.upperValue === expected.value);
      const directionalRow = { id: c.id, terminal, account, depth, side, bound: directional, exact: expected, valid: matches };
      results.push(directionalRow);
      if (!matches) { save("failure.json", { case: c, row: directionalRow }); throw new Error(`Directional upper failed: ${c.id} ${terminal} ${depth} ${side}`); }
    }
  }
}
const summary = { cases: count, queries: results.length, finite: results.filter(r => Number.isFinite(r.bound.upperValue)).length,
  provenInfeasible: results.filter(r => r.bound.upperValue === -Infinity).length,
  maximumViolationBps: Math.max(0, ...results.map(r => r.exact.value === r.bound.upperValue ? 0 : (r.exact.value - r.bound.upperValue) * 10000)),
  elapsedSec: (performance.now() - started) / 1000, results };
save("summary.json", summary); console.log(JSON.stringify({ ...summary, results: undefined }));
