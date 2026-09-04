/** Test the one-event optimizer against all feasible lots with fixed laws. */
import fs from "node:fs";
import path from "node:path";
import assert from "node:assert/strict";
import { decideEventOneStep } from "../packages/bot-algo/src/event-one-step.js";
import { prepareEventOneStep } from "../packages/bot-algo/src/event-one-step-prepared.js";
import { eventBellmanReference } from "../packages/bot-algo/test/event-bellman-reference.js";
import { DEFAULT_EVENT_COSTS } from "../packages/bot-algo/src/event-log-policy.js";
import { EVENT_FEATURES } from "../packages/bot-algo/src/event-distribution.js";

const arg = (k: string) => { const at = process.argv.indexOf(`--${k}`); return at < 0 ? "" : process.argv[at + 1]; };
if (!arg("reference") || !arg("output")) throw new Error("Specify --reference and new --output");
const root = path.resolve(__dirname, ".."), directory = (s: string) => path.join(root, "data/benchmarks", s);
const source = directory(arg("reference")), output = directory(arg("output"));
if (fs.existsSync(output)) throw new Error("Choose a new output directory");
const config = JSON.parse(fs.readFileSync(path.join(source, "config.json"), "utf8"));
assert.equal(config.contract, "event-fixed-model-optimality-audit-v1");
const seed = 20260904, randomCases = 1800;
const prepared = process.argv.includes("--prepared");
let state = seed;
const random = () => { state = Math.imul(state, 1664525) + 1013904223 | 0; return (state >>> 0) / 2 ** 32; };
const pick = <T>(values: T[]) => values[Math.floor(random() * values.length)];
const cases = config.cases.flatMap((r: any) => r.exposures.flatMap((exposure: number) => r.model.kernels.map((_: any, leaf: number) => ({
  name: r.name, model: r.model, costs: r.costs, account: { equity: r.equity, price: r.price, exposure }, leaf }))));
for (let i = 0; i < randomCases; i++) {
  const equity = 10 + random() * 1000, price = 5 + random() * 200, step = equity / (20 * price), leverage = pick([1, 2, 5]);
  const costs = { ...DEFAULT_EVENT_COSTS, maxLeverage: leverage, feeBps: pick([0, 10, 100]), slippageBps: pick([0, 2, 50]),
    minNotional: pick([0, .03, .12]) * equity, maxNotional: pick([.2, .8, 2, 4]) * equity,
    minQuantity: pick([0, 1, 3]) * step, quantityStep: step, maintenanceMargin: pick([.005, .1, .3]),
    longBorrowBpsPerDay: pick([0, 1, 1000]), shortBorrowBpsPerDay: pick([0, 1, 1000]) };
  const masses = Array.from({ length: 2 + Math.floor(random() * 4) }, () => .01 + random());
  const total = masses.reduce((s, p) => s + p, 0);
  const kernel = masses.map(probability => {
    const r = (random() - .5) * .7;
    return { probability: probability / total, return: r, low: Math.max(-.99, Math.min(0, r) - random() * .3),
      high: Math.max(0, r) + random() * .3, duration: pick([1, 60, 1440]), next: 0 };
  });
  const model = { version: 1, clock: { thresholdBps: 20, maxCandles: 3 }, featureNames: EVENT_FEATURES,
    nodes: [{ feature: -1, cut: 0, left: -1, right: -1, leaf: 0 }], trainingSamples: 1, counts: [1], priorClasses: [], classProbabilities: [], kernels: [kernel] };
  cases.push({ name: `seeded-${i}`, model, costs, account: { equity, price, exposure: pick([0, .2, .95, 1.01, 1.5, -1.4]) * leverage }, leaf: 0 });
}
fs.mkdirSync(output, { recursive: true });
fs.writeFileSync(path.join(output, "config.json"), JSON.stringify({ contract: "event-one-step-optimizer-audit-v1", source, seed, randomCases, prepared,
  method: "Compare marked-wealth, proportional-friction and minimum-order-aware market terminal conventions against independent exhaustive feasible-lot enumeration. Include saved fixed-law fixtures and seeded variations in fees, leverage, borrowing, extrema, order minima/maxima, lot sizes, cash, longs, shorts and above-cap recovery. Check exact expected values, not hindsight returns.",
  tolerance: 1e-10, cases }, null, 2));
const files = ["scripts/audit-event-one-step.ts", "packages/bot-algo/src/event-one-step.ts", "packages/bot-algo/src/event-one-step-prepared.ts", "packages/bot-algo/src/event-holding-law.ts", "packages/bot-algo/src/event-log-policy.ts", "packages/bot-algo/test/event-bellman-reference.ts"];
fs.writeFileSync(path.join(output, "sources.json"), JSON.stringify(Object.fromEntries(files.map(f => [f, fs.readFileSync(path.join(root, f), "utf8")]))));
const results = [], started = performance.now();
for (const c of cases) for (const terminal of ["friction", "market", "marked"] as const) {
  if (prepared && terminal === "market") continue;
  const reference = eventBellmanReference(c.model, c.costs, { terminal });
  const before = performance.now(), exact = reference.decide(c.leaf, c.account, 1), referenceMs = performance.now() - before;
  const beforePrepare = performance.now(), compiled = prepared ? prepareEventOneStep(c.model.kernels[c.leaf], c.costs, terminal as "marked" | "friction") : undefined;
  const preparationMs = performance.now() - beforePrepare;
  const begin = performance.now(), fast = compiled ? compiled(c.account) : decideEventOneStep(c.model.kernels[c.leaf], c.account, c.costs, terminal), fastMs = performance.now() - begin;
  const actual = reference.actionValue(c.leaf, c.account, 1, fast.quantity);
  const matches = Number.isFinite(exact.value) === Number.isFinite(fast.value)
    && (!Number.isFinite(exact.value) || (Math.abs(exact.value - fast.value) <= 1e-10 && Math.abs(actual - exact.value) <= 1e-10));
  const row = { name: c.name, exposure: c.account.exposure, terminal, optimal: exact, fast,
    exactChosenValue: actual, regretBps: Number.isFinite(exact.value) ? (exact.value - actual) * 10000 : null,
    referenceOrders: reference.stats.actions + 1, referenceMs, fastMs, preparationMs, matches };
  results.push(row);
  if (!matches) {
    fs.writeFileSync(path.join(output, "failure.json"), JSON.stringify({ case: c, result: row }, null, 2));
    throw new Error(`One-event optimizer differs from enumeration: ${c.name} ${terminal}`);
  }
}
const summary = { results, count: results.length, maximumRegretBps: Math.max(...results.map(r => r.regretBps ?? 0)),
  evaluatedOrders: results.reduce((s, r) => s + r.fast.search.evaluatedOrders, 0), referenceOrders: results.reduce((s, r) => s + r.referenceOrders, 0),
  referenceMs: results.reduce((s, r) => s + r.referenceMs, 0), fastMs: results.reduce((s, r) => s + r.fastMs, 0),
  preparationMs: results.reduce((s, r) => s + r.preparationMs, 0), elapsedSec: (performance.now() - started) / 1000 };
fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify(summary, null, 2));
console.log(JSON.stringify({ ...summary, results: undefined }));
