/** Paired one-event optimizer replay with saved, unchanged stochastic forecasts. */
import fs from "node:fs";
import path from "node:path";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { restoreEventPolicy, decideEvent, type EventAccount } from "../packages/bot-algo/src/event-log-policy.js";
import { decideEventOneStep } from "../packages/bot-algo/src/event-one-step.js";
import { eventBellmanReference } from "../packages/bot-algo/test/event-bellman-reference.js";
import { eventOriginScore, eventRefitOrigins } from "./research-event-refits.js";
import { loadEventCandles, replayEventPolicy } from "./research-event-policy.js";

const root = path.resolve(__dirname, ".."), DAY = 86400000;
const arg = (k: string) => { const at = process.argv.indexOf(`--${k}`); return at < 0 ? "" : process.argv[at + 1]; };
if (!arg("source") || !arg("output")) throw new Error("Specify saved joint-law source and new output");
const source = path.join(root, "data/benchmarks", arg("source")), output = path.join(root, "data/benchmarks", arg("output"));
if (fs.existsSync(output)) throw new Error("Choose a new output directory");
const read = (d: string, f: string) => JSON.parse(fs.readFileSync(path.join(d, f), "utf8"));
const config = read(source, "config.json"), old = read(source, "summary.json");
assert.equal(config.contract, "event-volatility-law-policy-v1");
const oc = read(config.source, "config.json"), sc = read(oc.source, "config.json"), window = config.window;
assert.ok(!window.id.startsWith("fit-"));
const phases = [...eventRefitOrigins(window.startTime, oc.foldCount, oc.foldDays), { ...window, id: "final" }];
const variants = ["grid", "friction", "market", "marked"] as const;
const reference = arg("reference") ? path.join(root, "data/benchmarks", arg("reference")) : undefined;
if (reference) {
  const rc = read(reference, "config.json"); assert.equal(rc.contract, "event-one-step-screen-v1");
  assert.equal(rc.source, source); assert.deepEqual(rc.phases, phases);
}
fs.mkdirSync(output, { recursive: true });
fs.writeFileSync(path.join(output, "config.json"), JSON.stringify({ contract: "event-one-step-screen-v1", source, reference, phases, variants,
  method: "Use the identical saved joint event kernel at depth one. Compare the current interpolated grid policy with exact lot optimization under proportional-friction, market-dust and marked-wealth terminal rules. Marked wealth pays current trade and holding costs but assumes no forced sale at the next event; the backtest still settles actual terminal inventory. Reproduce saved grid economics where a depth-one result exists. Freeze all variants before replay; record prior ranking before loading final candles. Final results are unconditional diagnostics, not a new selected incumbent.",
  probes: "At every grid-visited state, compare fixed-model values on that same observed account. Also probe full long/short at the first state of each prior origin. Independently enumerate every feasible lot for the three first-origin accounts; later probes use the verified concave-interval solver.",
  caveat: "The solver is exact up to numerical tolerance for one event under the supplied decision-price kernel. The simulator fills next open and charges minute borrowing; those execution differences and repeated horizon resets remain. No claim of multi-event Bellman optimality, a new independent holdout, or profitability. All models are reused without retraining." }, null, 2));
const files = ["scripts/screen-event-one-step.ts", "scripts/research-event-policy.ts", "packages/bot-algo/src/event-one-step.ts",
  "packages/bot-algo/src/event-log-policy.ts", "packages/bot-algo/test/event-bellman-reference.ts"];
fs.writeFileSync(path.join(output, "sources.json"), JSON.stringify(Object.fromEntries(files.map(f => [f, fs.readFileSync(path.join(root, f), "utf8")]))));
const results: any[] = [], started = performance.now();
for (const [phaseIndex, phase] of phases.entries()) {
  if (phase.id === "final") {
    const ranking = variants.map(choice => ({ choice, ...eventOriginScore(results.map(r => r.metrics[choice]), sc.riskPenalty) })).sort((a, b) => b.score - a.score);
    fs.writeFileSync(path.join(output, "prior-ranking.json"), JSON.stringify({ ranking, diagnosticOnly: true, finalCandlesNotLoaded: true }, null, 2));
  }
  const begin = performance.now(), file = `${phase.id}-policy.json`, raw = read(source, file), p = restoreEventPolicy(raw);
  const fit = read(config.source, `${window.id}-${phase.id}-model.json`); assert.ok(fit.trainEnd <= phase.startTime);
  const c = loadEventCandles(phase.startTime - 2 * DAY, phase.endTime), byTime = new Map(c.map(r => [r.openTime + 60000, r.close]));
  const metrics: Record<string, any> = {}, traces: Record<string, any[]> = {};
  for (const variant of variants) {
    const { trace, ...m } = replayEventPolicy(c, p, phase.startTime, phase.endTime, 1,
      { ...(variant === "grid" ? {} : { oneStepTerminal: variant }), trace: true });
    metrics[variant] = m; traces[variant] = trace;
    fs.writeFileSync(path.join(output, `${phase.id}-${variant}-trades.json`), JSON.stringify(trace));
  }
  const expected = phase.id !== "final" ? read(source, `${phase.id}-scores.json`).rows.find((r: any) => r.choice === "joint-volatility" && r.depth === 1)
    : old.diagnosticChoice.depth === 1 ? old.diagnostic : undefined;
  if (expected) for (const key of ["returnPct", "logGrowth", "finalEquity", "maxDrawdownPct", "closeDrawdownPct", "fees", "borrow", "trades", "liquidations", "canceledOrders", "decisions"])
    assert.ok(Math.abs(expected[key] - metrics.grid[key]) < 1e-9, `Saved grid result changed: ${phase.id} ${key}`);
  for (const variant of variants) assert.deepEqual(traces[variant].map(r => [r.time, r.leaf]), traces.grid.map(r => [r.time, r.leaf]));
  if (reference) for (const variant of ["grid", "friction", "market"]) {
    assert.deepEqual(metrics[variant], read(reference, "summary.json").results[phaseIndex].metrics[variant]);
    assert.equal(JSON.stringify(traces[variant]), JSON.stringify(read(reference, `${phase.id}-${variant}-trades.json`)));
  }
  const probes: any[] = [];
  const probe = (r: any, account: EventAccount, kind: string, exhaustive: boolean) => {
    const grid = decideEvent(p, r.leaf, account, 1), exact = decideEventOneStep(p.model.kernels[r.leaf], account, p.costs, "friction");
    const reference = eventBellmanReference(p.model, p.costs, { maxOrderLots: 1000000 });
    const gridValue = reference.actionValue(r.leaf, account, 1, grid.quantity), exactValue = reference.actionValue(r.leaf, account, 1, exact.quantity);
    assert.ok(Math.abs(exact.value - exactValue) < 1e-10 || exact.value === exactValue);
    assert.ok(exactValue >= gridValue - 1e-10);
    let allLots;
    if (exhaustive) {
      const start = performance.now(), optimum = reference.decide(r.leaf, account, 1);
      assert.ok(Math.abs(optimum.value - exactValue) < 1e-10 || optimum.value === exactValue);
      allLots = { optimum, stats: reference.stats, elapsedSec: (performance.now() - start) / 1000 };
    }
    probes.push({ time: r.time, leaf: r.leaf, kind, account, gridQuantity: grid.quantity, exactQuantity: exact.quantity,
      gridValue, exactValue, regretBps: Number.isFinite(gridValue) && Number.isFinite(exactValue) ? (exactValue - gridValue) * 10000 : null,
      search: exact.search, allLots });
  };
  for (const [i, r] of traces.grid.entries()) probe(r, { equity: r.equityBefore, price: byTime.get(r.time)!, exposure: r.exposureBefore }, "visited", !reference && !phaseIndex && !i);
  if (phase.id !== "final") for (const exposure of [-p.costs.maxLeverage, p.costs.maxLeverage]) {
    const r = traces.grid[0]; probe(r, { equity: r.equityBefore, price: byTime.get(r.time)!, exposure }, "initial-inventory", !reference && !phaseIndex);
  }
  fs.writeFileSync(path.join(output, `${phase.id}-probes.json`), JSON.stringify(probes));
  const count = (kind: string) => {
    const rows = probes.filter(r => r.kind === kind), finite = rows.filter(r => r.regretBps !== null);
    return { count: rows.length, finite: finite.length, changedOrders: rows.filter(r => Math.abs(r.gridQuantity - r.exactQuantity) > p.costs.quantityStep / 2).length,
      meanRegretBps: finite.length ? finite.reduce((s, r) => s + r.regretBps, 0) / finite.length : null,
      maxRegretBps: finite.length ? Math.max(...finite.map(r => r.regretBps)) : null };
  };
  const result = { phase, modelHash: createHash("sha256").update(fs.readFileSync(path.join(source, file))).digest("hex"), trainEnd: fit.trainEnd,
    gridEconomicsReproduced: Boolean(expected), referenceTracesAndMetricsExact: reference ? true : undefined,
    metrics, probes: { visited: count("visited"), initialInventory: count("initial-inventory") },
    exhaustiveChecks: probes.filter(r => r.allLots).map(r => ({ kind: r.kind, exposure: r.account.exposure, ...r.allLots })),
    elapsedSec: (performance.now() - begin) / 1000 };
  results.push(result);
  fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify({ results, elapsedSec: (performance.now() - started) / 1000 }, null, 2));
  console.log(JSON.stringify({ phase: phase.id, metrics: Object.fromEntries(variants.map(v => [v, { returnPct: metrics[v].returnPct,
    trades: metrics[v].trades, drawdown: metrics[v].maxDrawdownPct }])), probes: result.probes, elapsedSec: result.elapsedSec }));
}
