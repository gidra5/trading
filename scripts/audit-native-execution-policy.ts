/** Verify frozen-law action values and realized transitions for execution H1 replays. */
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { decideEventOneStep } from "../packages/bot-algo/src/event-one-step.js";
import { evaluateEventExecutionPath, summarizeEventExecutionPath } from "../packages/bot-algo/src/event-execution-path.js";
import { eventMarketUnavailable } from "../packages/bot-algo/src/event-market-availability.js";
import { loadEventCandles } from "./research-event-policy.js";
const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
const root = path.resolve(__dirname, ".."), directory = (name: string) => path.resolve(root, "data/benchmarks", name);
const output = directory(arg("output"));
assert.ok(arg("replays") && arg("output") && !fs.existsSync(output));
const read = (file: string) => JSON.parse(fs.readFileSync(file, "utf8"));
const hash = (file: string) => createHash("sha256").update(fs.readFileSync(file)).digest("hex");
const results = [], started = performance.now();
for (const name of arg("replays").split(",")) {
  const replay = directory(name), config = read(path.join(replay, "config.json")), summary = read(path.join(replay, "summary.json"));
  assert.equal(config.contract, "native-event-execution-one-step-replay-v1");
  assert.equal(hash(path.join(config.source, "law.json")), config.lawHash);
  assert.equal(hash(path.join(config.baseSource, "model.json")), config.modelHash);
  for (const ref of config.replaySourceReferences) assert.equal(hash(ref.file), ref.sha256);
  if (config.marketAvailability) assert.equal(hash(config.marketAvailability.file), config.marketAvailability.sha256);
  const law = read(path.join(config.source, "law.json")), policy = read(path.join(config.baseSource, "model.json"));
  const traceFile = path.join(replay, "execution-trades.json"), trace = read(traceFile), metrics = summary.execution;
  const closures = config.marketAvailability?.closures ?? [], sourceStart = config.start - 1000;
  const candles = loadEventCandles(sourceStart, config.end, 1000, closures.length ? closures : undefined);
  const valueRows = [], transitions = [], leaves = new Map<number, any>();
  let maxValueError = 0, maxEquityError = 0;
  for (const row of trace) {
    assert.equal(row.optimizer, "execution-one-event-lots"); assert.ok(row.order.complete && row.order.feasible);
    assert.equal(row.order.accountConvention, "before-order");
    assert.equal(row.order.equity, row.equityBefore); assert.equal(row.order.exposure, row.exposureBefore);
    const account = { equity: row.equityBefore, price: row.order.price, exposure: row.exposureBefore };
    const kernel = law.kernels[row.leaf];
    const score = (quantity: number) => {
      let value = 0, rejectedMass = 0, acceptedReturn = 0, acceptedMass = 0;
      for (const atom of kernel) {
        const path = law.paths[atom.path], result = evaluateEventExecutionPath(path, account, quantity);
        value += atom.probability * result.logGrowth;
        if (result.canceled) rejectedMass += atom.probability;
        if (result.filledQuantity) { acceptedMass += atom.probability; acceptedReturn += atom.probability * (path.closeRatio - 1) * 10000; }
      }
      return { value, rejectedMass, acceptedMass, acceptedReturnMeanBps: acceptedMass ? acceptedReturn / acceptedMass : null };
    };
    const chosen = score(row.order.quantity), hold = score(0);
    const oldRequest = decideEventOneStep(policy.model.kernels[row.leaf], account, policy.costs, "marked").quantity;
    const old = score(oldRequest), error = Math.abs(chosen.value - row.order.value);
    maxValueError = Math.max(maxValueError, error);
    assert.ok(error < 1e-10); assert.ok(chosen.value >= old.value - 1e-10); assert.ok(chosen.value >= hold.value - 1e-10);
    const item = { time: row.time, leaf: row.leaf, request: row.order.quantity, filledQuantity: row.orderQuantity,
      exposureBefore: row.exposureBefore, oldRequest, expectedLogValue: chosen.value,
      advantageOverOldBps: (chosen.value - old.value) * 10000, advantageOverHoldBps: (chosen.value - hold.value) * 10000,
      rejectedMass: chosen.rejectedMass, acceptedReturnMeanBps: chosen.acceptedReturnMeanBps };
    valueRows.push(item);
    const leaf = leaves.get(row.leaf) ?? { leaf: row.leaf, decisions: 0, requests: 0, fills: 0, mostlyRejectedRequests: 0,
      meanValueGainOverOldBps: 0, realizedEventLogGrowth: 0 };
    leaf.decisions++; leaf.requests += Number(row.order.quantity !== 0); leaf.fills += Number(row.orderQuantity !== 0);
    leaf.mostlyRejectedRequests += Number(chosen.rejectedMass >= .9 && row.order.quantity !== 0);
    leaf.meanValueGainOverOldBps += item.advantageOverOldBps;
    leaf.realizedEventLogGrowth += Math.log(row.equityAfter / row.equityBefore); leaves.set(row.leaf, leaf);
  }
  const records = [...trace.map((r: any) => ({ ...r, forcedWait: false })),
    ...(metrics.marketAvailability?.forcedWaits ?? []).map((r: any) => ({ ...r, forcedWait: true }))].sort((a, b) => a.time - b.time);
  let time = config.start, equity = 10000, quantity = 0, fills = 0, cancellations = 0, terminalEquity = NaN;
  for (const row of records) {
    assert.equal(row.time, time); assert.ok(Math.abs(row.equityBefore - equity) < 1e-7);
    const beforeQuantity = row.forcedWait ? row.quantityBefore : row.previousQuantity;
    assert.ok(Math.abs(beforeQuantity - quantity) < 1e-10);
    const origin = (row.time - sourceStart) / 1000 - 1, end = (row.endTime - sourceStart) / 1000 - 1;
    const compressed = summarizeEventExecutionPath(candles, origin, end, policy.costs,
      !candles[end].carriedMark && !eventMarketUnavailable(closures, row.endTime));
    const price = candles[origin].close, account = { equity, price, exposure: quantity * price / equity };
    const request = row.forcedWait ? 0 : row.order.quantity, result = evaluateEventExecutionPath(compressed, account, request);
    maxEquityError = Math.max(maxEquityError, Math.abs(result.equity - row.equityAfter));
    assert.ok(Math.abs(result.equity - row.equityAfter) < 1e-6);
    assert.ok(Math.abs(result.filledQuantity - (row.forcedWait ? 0 : row.orderQuantity)) < 1e-10);
    if (result.filledQuantity) assert.equal(eventMarketUnavailable(closures, row.time), false);
    fills += Number(result.filledQuantity !== 0); cancellations += Number(result.canceled);
    transitions.push({ time: row.time, endTime: row.endTime, forcedWait: row.forcedWait, quantityBefore: quantity,
      quantityAfter: result.quantity, filledQuantity: result.filledQuantity, equityError: result.equity - row.equityAfter });
    quantity = result.quantity; equity = row.equityAfter; time = row.endTime;
    if (time === config.end) {
      const settled = evaluateEventExecutionPath(compressed, account, request, "market"); terminalEquity = settled.equity;
      assert.ok(Math.abs(terminalEquity - metrics.finalEquity) < 1e-6); assert.equal(fills + settled.terminalOrders, metrics.trades);
    }
  }
  assert.equal(time, config.end); assert.equal(cancellations, metrics.canceledOrders);
  assert.ok(Math.abs(metrics.positionSummary.equityChange - (metrics.finalEquity - 10000)) < 1e-7);
  assert.ok(Math.abs(10000 + metrics.longPnl + metrics.shortPnl - metrics.fees - metrics.borrow - metrics.finalEquity) < 1e-7);
  assert.equal(trace.length, summary.completeSearches); assert.equal(trace.length, summary.finiteValueDecisions);
  const requested = valueRows.filter(r => r.request !== 0);
  results.push({ name, window: config.window, phase: config.phase, modelHash: config.modelHash, lawHash: config.lawHash,
    traceHash: hash(traceFile), decisions: trace.length, maxValueError, maxEquityError,
    meanGainOverOldBps: valueRows.reduce((s, r) => s + r.advantageOverOldBps, 0) / valueRows.length,
    requests: requested.length, mostlyRejectedRequests: requested.filter(r => r.rejectedMass >= .9).length,
    meanRejectionMass: requested.reduce((s, r) => s + r.rejectedMass, 0) / (requested.length || 1),
    control: summary.control, execution: metrics, executionSeconds: summary.executionSeconds,
    leafBehavior: [...leaves.values()].map(r => ({ ...r, meanValueGainOverOldBps: r.meanValueGainOverOldBps / r.decisions })),
    valueRows, transitions });
  console.log(JSON.stringify({ name, decisions: trace.length, maxValueError, maxEquityError }));
}
fs.mkdirSync(output, { recursive: true });
fs.writeFileSync(path.join(output, "source.ts"), fs.readFileSync(__filename, "utf8"));
fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify({ method: "Verify source-bound complete H1 searches, recompute every chosen expected value from the full fixed law, compare the old request rule and hold at identical accounts, and reconcile realized requests, cancellations, equity and position accounting using the independently checked compressed transition. Complete globality relies on the acceptance partition/concavity construction and exhaustive solver tests; this audit does not infer globality merely from beating two alternatives. Historical performance is descriptive on previously inspected windows, with March's declared availability scenario retained.",
  decisions: results.reduce((s, r) => s + r.decisions, 0), elapsedSeconds: (performance.now() - started) / 1000, results },
  (_, v) => typeof v === "number" && !Number.isFinite(v) ? String(v) : v, 2));
