/** Reconcile actual cash-to-cash episodes and classify missed original entries. */
import fs from "node:fs";
import path from "node:path";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";

const root = path.resolve(__dirname, ".."), arg = (k: string) => { const i = process.argv.indexOf(`--${k}`); return i < 0 ? "" : process.argv[i + 1]; };
if (!arg("source") || !arg("output")) throw new Error("Specify composed controller screen and new output");
const source = path.resolve(root, "data/benchmarks", arg("source")), output = path.resolve(root, "data/benchmarks", arg("output"));
const read = (dir: string, file: string) => JSON.parse(fs.readFileSync(path.join(dir, file), "utf8"));
if (fs.existsSync(output)) throw new Error("Choose a new output directory");
const config = read(source, "config.json"), summary = read(source, "summary.json");
const original = config.replanCash || config.quoteEntries ? config.reference : config.source;
if (config.contract !== "event-controller-composition-screen-v1") throw new Error("Requires composed controller screen");
assert.deepEqual(summary.results.map((r: any) => r.phase), config.phases);
const near = (a: number, b: number, epsilon: number) => assert.ok(Math.abs(a - b) < epsilon, `${a} != ${b}`);
const episodes = (trace: any[], metrics: any, end: number) => {
  const cycles: any[] = []; let active: any;
  assert.equal(metrics.liquidations, 0, "Liquidated episodes need a separate attribution");
  for (const r of trace) {
    if (!r.orderQuantity) continue;
    if (Math.abs(r.previousQuantity) < 1e-9) {
      assert.ok(!active);
      active = { entry: new Date(r.time).toISOString(), equityBefore: r.equityBefore,
        initialSide: Math.sign(r.orderQuantity), orders: [], reversals: 0 };
    }
    assert.ok(active);
    const quantity = r.previousQuantity + r.orderQuantity, reversal = r.previousQuantity * quantity < -1e-12;
    active.reversals += Number(reversal);
    active.orders.push({ date: new Date(r.time).toISOString(), quantity: r.orderQuantity, resultingQuantity: quantity, reversal,
      controller: r.optionController, intermediate: r.optionHolding, remaining: r.optionRemaining });
    if (Math.abs(quantity) < 1e-9) {
      cycles.push({ ...active, exit: new Date(r.time).toISOString(), equityAfter: r.equityAfter,
        returnPct: 100 * (r.equityAfter / active.equityBefore - 1), pnl: r.equityAfter - active.equityBefore });
      active = undefined;
    }
  }
  if (active) cycles.push({ ...active, exit: new Date(end).toISOString(), terminalSettlement: true,
    equityAfter: metrics.finalEquity, returnPct: 100 * (metrics.finalEquity / active.equityBefore - 1), pnl: metrics.finalEquity - active.equityBefore });
  const pnl = cycles.reduce((s, c) => s + c.pnl, 0), logGrowth = cycles.reduce((s, c) => s + Math.log(c.equityAfter / c.equityBefore), 0);
  near(pnl, metrics.finalEquity - 10000, 1e-7); near(logGrowth, metrics.logGrowth, 1e-12);
  near(pnl, metrics.longPnl + metrics.shortPnl - metrics.fees - metrics.borrow, 1e-7);
  near(cycles.reduce((s, c) => s + c.reversals, 0), metrics.reversals, 1e-9);
  return { pnl, logGrowth, fees: metrics.fees, borrow: metrics.borrow, cycles };
};
fs.mkdirSync(output, { recursive: true });
fs.writeFileSync(path.join(output, "config.json"), JSON.stringify({ contract: "event-controller-composition-audit-v1", source, original,
  sourceHash: createHash("sha256").update(fs.readFileSync(path.join(source, "config.json"))).update(fs.readFileSync(path.join(source, "summary.json"))).digest("hex"),
  method: "Use executed orderQuantity, not planned order.quantity, to reconstruct complete cash-to-cash episodes. Reconcile PnL, log wealth, fees, borrowing and reversals. At each original executed flat entry, inspect the composed controller's state and actual order at that time.",
  caveat: "Hindsight attribution of previously declared replays. A skipped entry during a committed H1 step is not evidence that the H2 entry-value comparison rejected it." }, null, 2));
fs.writeFileSync(path.join(output, "sources.json"), JSON.stringify({ "scripts/audit-event-controller-composition.ts": fs.readFileSync(__filename, "utf8") }));
const results = [];
for (const r of summary.results) {
  const trace = read(source, `${r.phase.id}-trades.json`), control = read(original, `${r.phase.id}-trades.json`);
  const byTime = new Map<number, any>(trace.map((t: any) => [t.time, t]));
  const originalEntries = control.filter((t: any) => Math.abs(t.previousQuantity) < 1e-9 && t.orderQuantity).map((t: any) => {
    const next = byTime.get(t.time); assert.ok(next);
    return { date: new Date(t.time).toISOString(), originalSide: Math.sign(t.orderQuantity), originalOrder: t.order,
      candidateWasFlat: Math.abs(next.previousQuantity) < 1e-9, candidateOrder: next.order,
      executedQuantity: next.orderQuantity, intermediate: next.optionHolding, controller: next.optionController,
      sameEntry: Math.abs(next.previousQuantity) < 1e-9 && Math.sign(t.orderQuantity) === Math.sign(next.orderQuantity),
      skippedWhileAdvancing: Math.abs(next.previousQuantity) < 1e-9 && !next.orderQuantity && next.optionHolding };
  });
  const result = { phase: r.phase, reconciled: true, original: episodes(control,
    config.quoteEntries ? r.executionControl : config.replanCash ? r.committedControl : r.control, r.phase.endTime),
    candidate: episodes(trace, r.candidate, r.phase.endTime),
    fixedSampled: r.fixedSampledControl ? episodes(read(source, `${r.phase.id}-sampled-fixed-trades.json`), r.fixedSampledControl, r.phase.endTime) : undefined,
    originalEntries };
  results.push(result);
  console.log(JSON.stringify({ phase: r.phase.id, originalEntries: originalEntries.length,
    sameEntry: originalEntries.filter((e: any) => e.sameEntry).length,
    skippedWhileAdvancing: originalEntries.filter((e: any) => e.skippedWhileAdvancing).map((e: any) => e.date),
    cycles: result.candidate.cycles.map(c => ({ entry: c.entry, exit: c.exit, initialSide: c.initialSide, reversals: c.reversals, returnPct: c.returnPct })) }));
}
fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify({ results }, null, 2));
