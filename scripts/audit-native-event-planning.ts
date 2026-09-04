/** Join saved replay certificates and diagnose the unchanged forecast. */
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { eventMarketUnavailable, validateEventMarketClosures } from "../packages/bot-algo/src/event-market-availability.js";
const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
const directory = (name: string) => path.resolve(__dirname, "../data/benchmarks", name), output = directory(arg("output"));
assert.ok(arg("replays") && arg("output") && !fs.existsSync(output));
const read = (name: string, file: string) => JSON.parse(fs.readFileSync(path.join(directory(name), file), "utf8"));
const hash = (data: Buffer) => createHash("sha256").update(data).digest("hex");
const refinements = arg("refinements").split(",").filter(Boolean).flatMap(name => {
  const config = read(name, "config.json");
  assert.equal(config.contract, "native-event-planning-refinement-v1");
  return read(name, "summary.json").results.map((row: any) => ({ name, config, row }));
});
const used = new Set<string>(), results = arg("replays").split(",").map(name => {
  const config = read(name, "config.json"), summary = read(name, "summary.json");
  const traceBytes = fs.readFileSync(path.join(directory(name), "h2-trades.json")), trace = JSON.parse(traceBytes.toString());
  const modelBytes = fs.readFileSync(path.join(config.source, "model.json")), { model, costs } = JSON.parse(modelBytes.toString());
  assert.equal(config.contract, "native-event-planning-screen-v1"); assert.equal(config.mode, "replay");
  assert.equal(hash(modelBytes), config.modelHash);
  for (const ref of config.replaySourceReferences ?? []) assert.equal(hash(fs.readFileSync(ref.file)), ref.sha256);
  let availabilityAudit;
  if (config.marketAvailability) {
    const availability = config.marketAvailability;
    assert.equal(hash(fs.readFileSync(availability.file)), availability.sha256);
    const manifest = JSON.parse(fs.readFileSync(availability.file, "utf8"));
    const { file, sha256, ...recorded } = availability;
    assert.deepEqual(recorded, manifest); validateEventMarketClosures(availability.closures);
    for (const ref of availability.sourceReferences) assert.equal(hash(fs.readFileSync(ref.file)), ref.sha256);
    const checks = [];
    for (const [prefix, metrics] of [["h1", summary.h1], ["h2", summary.results[0].metrics]] as const) {
      const rows = prefix === "h2" ? trace : read(name, "h1-trades.json");
      const a = metrics.marketAvailability;
      assert.ok(a && a.scoredUntil >= config.start && a.scoredUntil <= config.end);
      if (!metrics.liquidations) assert.equal(a.scoredUntil, config.end);
      const unavailable = (time: number) => eventMarketUnavailable(availability.closures, time);
      const expectedSeconds = availability.closures.reduce((s: number, c: any) => s
        + Math.max(0, Math.min(c.end, a.scoredUntil) - Math.max(c.start, config.start)) / 1000, 0);
      assert.equal(a.unavailableSeconds, expectedSeconds);
      for (const row of rows) {
        assert.equal(unavailable(row.time - 1000), false, "A controller decision used an unavailable completed bar");
        if (unavailable(row.time)) assert.equal(row.orderQuantity, 0, "A requested trade filled during closure");
      }
      for (const wait of a.forcedWaits) {
        assert.ok(unavailable(wait.time - 1000)); assert.equal("order" in wait, false);
        if (wait.quantityBefore !== wait.quantityAfter) { assert.ok(metrics.liquidations); assert.equal(wait.quantityAfter, 0); }
      }
      const intervals = [...rows, ...a.forcedWaits].sort((a, b) => a.time - b.time);
      let time = config.start;
      for (const interval of intervals) { assert.equal(interval.time, time); assert.ok(interval.endTime > time); time = interval.endTime; }
      assert.equal(time, a.scoredUntil);
      let fills = 0;
      for (const row of [...rows, ...a.forcedWaits]) for (const change of row.positionChanges ?? []) if (change.reason === "fill") {
        assert.equal(unavailable(change.time), false, "Position ledger records a fill during closure"); fills++;
      }
      assert.equal(a.terminalUnavailable, unavailable(config.end) || unavailable(config.end - 1000));
      if (a.terminalUnavailable) assert.deepEqual(metrics.terminalPositionChanges, []);
      assert.ok(Math.abs(metrics.positionSummary.equityChange - (metrics.finalEquity - 10000)) < 1e-7);
      checks.push({ policy: prefix, unavailableSeconds: a.unavailableSeconds, forcedWaitIntervals: a.forcedWaits.length,
        unavailableCanceledOrders: a.unavailableCanceledOrders, availableFills: fills, scoredUntil: a.scoredUntil });
    }
    availabilityAudit = { manifestHash: availability.sha256, assumptions: availability.assumptions, checks };
  }
  assert.equal(summary.results[0].decisions, trace.length);
  assert.equal(summary.results[0].converged, trace.filter((r: any) => r.order.converged).length);
  let maxGap = 0, newlyCertified = 0;
  for (const row of trace) {
    let gap = row.order.gap;
    assert.ok(row.order.feasible && Number.isFinite(row.order.value));
    assert.ok(Math.abs(row.order.value - row.order.lowerValue) < 1e-12);
    if (row.order.converged) assert.ok(Number.isFinite(row.order.upperValue)
      && row.order.upperValue >= row.order.value - 1e-12
      && Math.abs(Math.max(0, row.order.upperValue - row.order.value) - gap) < 1e-12);
    if (!row.order.converged) {
      const candidates = refinements.filter(r => r.row.replay === name && r.row.time === row.time);
      assert.equal(candidates.length, 1, `Missing or duplicate refinement for ${name} ${row.time}`);
      const r = candidates[0], ref = r.config.replays.find((s: any) => s.name === name);
      assert.equal(ref.traceHash, hash(traceBytes)); assert.equal(ref.modelHash, config.modelHash);
      assert.equal(r.row.leaf, row.leaf); assert.equal(r.row.quantity, row.order.quantity);
      assert.deepEqual(r.row.account, { equity: row.equityBefore, price: row.order.price, exposure: row.exposureBefore });
      assert.ok(Math.abs(r.row.originalValue - row.order.value) <= 1e-10);
      gap = Math.max(0, r.row.refined.upperValue - r.row.originalValue);
      assert.ok(r.row.originalCertified && Math.abs(gap - r.row.originalGap) <= 1e-12);
      used.add(`${name}:${row.time}`); newlyCertified++;
    }
    assert.ok(Number.isFinite(gap) && gap <= config.tolerance);
    maxGap = Math.max(maxGap, gap);
  }
  const means = model.kernels.map((kernel: any[]) => kernel.reduce((sum, a) => sum + a.probability * a.return, 0));
  const twoMeans = model.kernels.map((kernel: any[]) => kernel.reduce((sum, a) => sum + a.probability * ((1 + a.return) * (1 + means[a.next]) - 1), 0));
  const pairs = trace.slice(0, -1).flatMap((a: any, i: number) => {
    const b = trace[i + 1];
    // Omit the boundary-censored tail instead of scoring it as a full event.
    if (b.endTime >= config.end || a.endTime !== b.time || a.interruptedByUnavailable || b.interruptedByUnavailable) return [];
    return [{ leaf: a.leaf, realized: ((1 + a.realizedReturnBps / 10000) * (1 + b.realizedReturnBps / 10000) - 1) * 10000 }];
  });
  const forecast = means.map((mean: number, leaf: number) => {
    const observations = pairs.filter((p: any) => p.leaf === leaf);
    return { leaf, oneEventForecastMeanBps: mean * 10000, twoEventForecastMeanBps: twoMeans[leaf] * 10000,
      completedTwoEventPairs: observations.length,
      realizedTwoEventMeanBps: observations.length ? observations.reduce((s: number, p: any) => s + p.realized, 0) / observations.length : null };
  });
  const h2 = summary.results[0].metrics;
  assert.ok(Math.abs(h2.finalEquity - (10000 + h2.longPnl + h2.shortPnl - h2.fees - h2.borrow)) < 1e-7);
  return { name, phase: config.phase, modelHash: config.modelHash, traceHash: hash(traceBytes), start: config.start, end: config.end, fullWindow: config.fullWindow,
    decisions: trace.length, originallyCertified: trace.length - newlyCertified, newlyCertified, certified: trace.length,
    maxGapBps: maxGap * 10000, h1: summary.h1, h2, forecast, costs, ...(availabilityAudit ? { availabilityAudit } : {}) };
});
assert.equal(used.size, refinements.length, "Unused or duplicate refinement rows");
assert.ok(results.every(r => r.modelHash === results[0].modelHash));
fs.mkdirSync(output, { recursive: true });
const save = (file: string, data: unknown) => fs.writeFileSync(path.join(output, file), JSON.stringify(data, null, 2));
fs.writeFileSync(path.join(output, "source.ts"), fs.readFileSync(__filename, "utf8"));
save("summary.json", { method: "Full original H2 replay traces, supplemented only by value certificates at identical saved accounts. Every original optimized action has a verified upper/lower gap <= 0.001 bp. This proves numerical H2 conditional optimality under the frozen decision-price marked-terminal model on these saved intervals; it does not certify stationary control, next-open execution optimality, or all inspector windows. Declared unavailable intervals have separate forced-wait/account-coverage checks and are not assigned fabricated absolute values or H2 certificates. Two-event diagnostics use contiguous completed pairs, omit censored/interrupted events and do not provide independent sample counts.",
  decisions: results.reduce((s, r) => s + r.decisions, 0), certified: results.reduce((s, r) => s + r.certified, 0),
  maxGapBps: Math.max(...results.map(r => r.maxGapBps)), results });
console.log(JSON.stringify(results.map(r => ({ name: r.name, certified: r.certified, maxGapBps: r.maxGapBps,
  h1ReturnPct: r.h1.returnPct, h2ReturnPct: r.h2.returnPct, forecast: r.forecast }))));
