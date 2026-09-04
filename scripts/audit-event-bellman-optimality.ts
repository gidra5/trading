/** Separate finite-model action search from value interpolation on small laws. */
import fs from "node:fs";
import path from "node:path";
import assert from "node:assert/strict";
import { buildEventPolicy, chooseEventTrade, decideEvent, DEFAULT_EVENT_COSTS } from "../packages/bot-algo/src/event-log-policy.js";
import { EVENT_FEATURES, type EventDistribution } from "../packages/bot-algo/src/event-distribution.js";
import { eventBellmanReference } from "../packages/bot-algo/test/event-bellman-reference.js";

const arg = process.argv.indexOf("--output");
if (arg < 0 || !process.argv[arg + 1]) throw new Error("Specify a new --output directory");
const output = path.resolve(__dirname, "../data/benchmarks", process.argv[arg + 1]);
if (fs.existsSync(output)) throw new Error("Choose a new output directory");
const atom = (probability: number, r: number, next: number) => ({ probability, return: r, low: Math.min(0, r), high: Math.max(0, r), duration: 1, next });
const law = (kernels: EventDistribution["kernels"]): EventDistribution => ({ version: 1, clock: { thresholdBps: 20, maxCandles: 3 },
  featureNames: EVENT_FEATURES, nodes: [{ feature: -1, cut: 0, left: -1, right: -1, leaf: 0 }],
  trainingSamples: 1, counts: kernels.map(() => 1), priorClasses: [], classProbabilities: [], kernels });
const cases = [
  { name: "binary-interior-size", model: law([[atom(.6, .1, 0), atom(.4, -.1, 0)]]),
    costs: { ...DEFAULT_EVENT_COSTS, maxLeverage: 5, minNotional: 1, maxNotional: 500, quantityStep: .1, minQuantity: .1 },
    price: 10, equity: 100, steps: 5, depths: [1], exposures: [0, 1, -1] },
  ...[10, 20].map(steps => ({ name: `binary-interior-size-grid-${steps}`, model: law([[atom(.6, .1, 0), atom(.4, -.1, 0)]]),
    costs: { ...DEFAULT_EVENT_COSTS, maxLeverage: 5, minNotional: 1, maxNotional: 500, quantityStep: .1, minQuantity: .1 },
    price: 10, equity: 100, steps, depths: [1], exposures: [0, 1, -1] })),
  { name: "two-state-reversal", model: law([[atom(.7, .03, 1), atom(.3, -.02, 0)], [atom(.35, .025, 1), atom(.65, -.035, 0)]]),
    costs: { ...DEFAULT_EVENT_COSTS, maxLeverage: 1, minNotional: 10, maxNotional: 50, quantityStep: .1, minQuantity: .1 },
    price: 100, equity: 100, steps: 2, depths: [1, 2, 3], exposures: [0, .5, -.5] },
  { name: "terminal-dust", model: law([[atom(.5, .02, 0), atom(.5, -.02, 0)]]),
    costs: { ...DEFAULT_EVENT_COSTS, maxLeverage: 1, minNotional: 25, maxNotional: 50, quantityStep: .1, minQuantity: .1 },
    price: 100, equity: 100, steps: 2, depths: [1, 2], exposures: [0, .2, -.2] },
];
fs.mkdirSync(output, { recursive: true });
fs.writeFileSync(path.join(output, "config.json"), JSON.stringify({ contract: "event-fixed-model-optimality-audit-v1", cases,
  method: "Freeze synthetic stochastic event kernels. Enumerate every feasible base-order lot on exact accounts at every Bellman node. Compare production grid decisions, exact optimization over the production candidate set, and exhaustive lot decisions under the same forecast. Average future outcomes before maximizing; no realized market path. Bound work at 100000 nodes and 1000 order lots per side.",
  terminal: "Primary comparison uses the planner's proportional terminal friction for every remaining unit. Also report the exact optimum with replay-style sub-minimum terminal dust retained.",
  caveat: "Exact up to floating point within these finite synthetic laws and event-close execution. This does not certify current fitted controllers, real-market profitability, next-open fills, infinite-horizon convergence, or the full inspector suite." }, null, 2));
const files = ["scripts/audit-event-bellman-optimality.ts", "packages/bot-algo/test/event-bellman-reference.ts", "packages/bot-algo/src/event-log-policy.ts"];
fs.writeFileSync(path.join(output, "sources.json"), JSON.stringify(Object.fromEntries(files.map(f => [f, fs.readFileSync(path.resolve(__dirname, "..", f), "utf8")]))));
const results = [], started = performance.now();
for (const fixture of cases) {
  const p = buildEventPolicy(fixture.model, fixture.costs, { depths: Math.max(...fixture.depths),
    referenceEquity: fixture.equity, referencePrice: fixture.price, actionSteps: fixture.steps });
  const reference = eventBellmanReference(fixture.model, fixture.costs), market = eventBellmanReference(fixture.model, fixture.costs, { terminal: "market" });
  for (const depth of fixture.depths) for (const exposure of fixture.exposures) for (let leaf = 0; leaf < fixture.model.kernels.length; leaf++) {
    const account = { equity: fixture.equity, price: fixture.price, exposure };
    const optimal = reference.decide(leaf, account, depth), current = decideEvent(p, leaf, account, depth);
    const candidate = chooseEventTrade(p, account, a => reference.holdValue(leaf, a, depth));
    const rootValue = reference.actionValue(leaf, account, depth, current.quantity);
    const policyValue = reference.evaluatePolicy(leaf, account, depth, (l, a, d) => decideEvent(p, l, a, d).quantity);
    assert.ok(optimal.value >= candidate.value - 1e-10 && candidate.value >= rootValue - 1e-10 && optimal.value >= policyValue - 1e-10);
    const row = { name: fixture.name, leaf, depth, exposure, optimal, current: { quantity: current.quantity, value: current.value },
      exactCandidate: { quantity: candidate.quantity, value: candidate.value }, exactCurrentRootValue: rootValue, exactCurrentPolicyValue: policyValue,
      actionSetRegretBps: (optimal.value - candidate.value) * 10000, interpolationRootRegretBps: (candidate.value - rootValue) * 10000,
      totalPolicyRegretBps: (optimal.value - policyValue) * 10000, valueEstimateErrorBps: (current.value - policyValue) * 10000,
      marketTerminalOptimum: market.decide(leaf, account, depth) };
    results.push(row);
  }
  console.log(JSON.stringify({ name: fixture.name, stats: reference.stats, marketStats: market.stats,
    maximumPolicyRegretBps: Math.max(...results.filter(r => r.name === fixture.name).map(r => r.totalPolicyRegretBps)) }));
}
const summary = { results, elapsedSec: (performance.now() - started) / 1000 };
fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify(summary, null, 2));
console.log(JSON.stringify({ rows: results.length, elapsedSec: summary.elapsedSec }));
