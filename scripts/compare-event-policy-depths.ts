/** Describe matched fixed-forecast replays without selecting a winning policy. */
import fs from "node:fs";
import path from "node:path";
import assert from "node:assert/strict";
const arg = (k: string) => { const i = process.argv.indexOf(`--${k}`); return i < 0 ? "" : process.argv[i + 1]; };
if (!["lower", "higher", "bounds", "output"].every(k => arg(k))) throw new Error("Specify --lower, --higher, --bounds and new --output");
const root = path.resolve(__dirname, ".."), directory = (s: string) => path.join(root, "data/benchmarks", s);
const lower = directory(arg("lower")), higher = directory(arg("higher")), bounds = directory(arg("bounds")), output = directory(arg("output"));
if (fs.existsSync(output)) throw new Error("Choose a new output directory");
const read = (d: string, f: string) => JSON.parse(fs.readFileSync(path.join(d, f), "utf8"));
const a = read(lower, "summary.json"), b = read(higher, "summary.json"), certified = read(bounds, "summary.json");
assert.equal(read(bounds, "config.json").source, higher);
fs.mkdirSync(output, { recursive: true });
const save = (f: string, x: unknown) => fs.writeFileSync(path.join(output, f), JSON.stringify(x, null, 2));
save("config.json", { contract: "event-policy-depth-comparison-v1", lower, higher, bounds,
  method: "Match complete fixed-forecast windows and event times. Reconcile realized wealth changes into gross long/short P&L, fees and borrowing. Report executed holdings separately from requested orders and aggregate consecutive intervals by each policy's holding sign. This describes two whole replays; it is not a same-account action-value comparison or an isolated causal attribution. Forecast residuals exclude the event ending at the window boundary. No retraining, return-based selection or new replay." });
save("sources.json", { "scripts/compare-event-policy-depths.ts": fs.readFileSync(__filename, "utf8") });
const results = [];
for (const high of b.results) {
  const low = a.results.find((r: any) => r.window.id === high.window.id), certification = certified.results.find((r: any) => r.window.id === high.window.id);
  assert.ok(low && certification && low.fullWindow && high.fullWindow && certification.fullWindow);
  assert.equal(low.modelHash, high.modelHash); assert.equal(certification.modelHash, high.modelHash);
  const name = `${high.window.id}-trades.json`, oldTrace = read(lower, name), newTrace = read(higher, name);
  assert.equal(oldTrace.length, newTrace.length);
  const rows = newTrace.map((r: any, i: number) => {
    const p = oldTrace[i]; assert.equal(p.time, r.time); assert.equal(p.endTime, r.endTime);
    assert.equal(p.leaf, r.leaf); assert.equal(p.expectedReturnBps, r.expectedReturnBps);
    const oldPosition = p.previousQuantity + p.orderQuantity, newPosition = r.previousQuantity + r.orderQuantity;
    return { decision: i + 1, time: r.time, endTime: r.endTime, leaf: r.leaf,
      forecastReturnBps: r.expectedReturnBps, realizedReturnBps: r.realizedReturnBps,
      lowerRequested: p.order.quantity, higherRequested: r.order.quantity,
      lowerExecuted: p.orderQuantity, higherExecuted: r.orderQuantity, lowerPosition: oldPosition, higherPosition: newPosition,
      lowerEquityChange: p.equityAfter - p.equityBefore, higherEquityChange: r.equityAfter - r.equityBefore,
      difference: (r.equityAfter - r.equityBefore) - (p.equityAfter - p.equityBefore) };
  });
  const segments: any[] = [];
  for (const r of rows) {
    const key = `${Math.sign(r.lowerPosition)}:${Math.sign(r.higherPosition)}`;
    let segment = segments.at(-1);
    if (!segment || segment.key !== key) {
      segment = { key, firstDecision: r.decision, lastDecision: r.decision, startTime: r.time, endTime: r.endTime,
        lowerEquityChange: 0, higherEquityChange: 0, difference: 0 }; segments.push(segment);
    }
    segment.lastDecision = r.decision; segment.endTime = r.endTime;
    segment.lowerEquityChange += r.lowerEquityChange; segment.higherEquityChange += r.higherEquityChange; segment.difference += r.difference;
  }
  const leafResiduals: any[] = [];
  for (const leaf of [...new Set(rows.map((r: any) => r.leaf))]) {
    const group = rows.filter((r: any) => r.leaf === leaf && r.endTime < high.end);
    if (!group.length) continue;
    leafResiduals.push({ leaf, count: group.length, forecastMeanBps: group[0].forecastReturnBps,
      realizedMeanBps: group.reduce((s: number, r: any) => s + r.realizedReturnBps, 0) / group.length,
      signMatches: group.filter((r: any) => Math.sign(r.forecastReturnBps) === Math.sign(r.realizedReturnBps)).length });
  }
  const gross = (m: any) => m.longPnl + m.shortPnl;
  const grossDifference = gross(high.metrics) - gross(low.metrics), feeDifference = high.metrics.fees - low.metrics.fees,
    borrowDifference = high.metrics.borrow - low.metrics.borrow, equityDifference = high.metrics.finalEquity - low.metrics.finalEquity;
  assert.ok(Math.abs(equityDifference - (grossDifference - feeDifference - borrowDifference)) < 1e-7);
  const traceDifference = rows.reduce((s: number, r: any) => s + r.difference, 0);
  const result = { window: high.window, modelHash: high.modelHash, certification,
    lowerMetrics: low.metrics, higherMetrics: high.metrics,
    attribution: { equityDifference, grossDifference, feeDifference, borrowDifference, traceDifference,
      terminalSettlementDifference: equityDifference - traceDifference }, segments, leafResiduals,
    requestedOrdersCanceled: {
      lower: rows.filter((r: any) => r.lowerRequested && !r.lowerExecuted).map((r: any) => r.decision),
      higher: rows.filter((r: any) => r.higherRequested && !r.higherExecuted).map((r: any) => r.decision) } };
  save(`${high.window.id}-comparison.json`, { ...result, rows }); results.push(result);
  console.log(JSON.stringify({ window: high.window.id, attribution: result.attribution, leafResiduals }));
}
save("summary.json", { results });
