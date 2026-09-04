/** Realized-transition audit only. No future path from this script enters a policy. */
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { loadEventCandles } from "./research-event-policy.js";
import { eventHolding } from "../packages/bot-algo/src/event-log-policy.js";
import { eventMarketUnavailable } from "../packages/bot-algo/src/event-market-availability.js";
import { summarizeEventExecutionPath, evaluateEventExecutionPath, type EventExecutionPath } from "../packages/bot-algo/src/event-execution-path.js";
const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
const root = path.resolve(__dirname, ".."), directory = (name: string) => path.resolve(root, "data/benchmarks", name);
const source = directory(arg("source")), output = directory(arg("output"));
assert.ok(arg("source") && arg("output") && !fs.existsSync(output));
const hash = (file: string) => createHash("sha256").update(fs.readFileSync(file)).digest("hex");
const read = (file: string) => JSON.parse(fs.readFileSync(file, "utf8"));
const coverage = read(path.join(source, "summary.json"));
assert.equal(coverage.contract, "native-event-availability-coverage-v1");
const requested = arg("windows").split(",").filter(Boolean);
assert.ok(requested.every(id => coverage.results.some((r: any) => r.id === id)));
fs.mkdirSync(output, { recursive: true });
const save = (name: string, value: unknown) => fs.writeFileSync(path.join(output, name), JSON.stringify(value, null, 2));
save("config.json", { contract: "native-event-execution-path-audit-v1", source, coverageHash: hash(path.join(source, "summary.json")),
  requested, method: "Evaluate the saved committed base-quantity request on its realized path using a compressed account transition, and compare with the unchanged per-second replay. This uses future data solely as a diagnostic, never as a forecast or an input to order selection. One aggregate physical account, shared costs and independent optional lifecycle attribution. No new optimization or profit claim." });
save("sources.json", Object.fromEntries(["scripts/audit-native-event-execution.ts", "packages/bot-algo/src/event-execution-path.ts",
  "scripts/research-event-policy.ts", "packages/bot-algo/src/event-log-policy.ts"].map(f => [f, fs.readFileSync(path.join(root, f), "utf8")])));
const results: any[] = [], started = performance.now();
const distribution = (values: number[]) => ({ count: values.length, mean: values.length ? values.reduce((s, x) => s + x, 0) / values.length : 0,
  meanAbsolute: values.length ? values.reduce((s, x) => s + Math.abs(x), 0) / values.length : 0,
  maxAbsolute: Math.max(0, ...values.map(Math.abs)) });
for (const entry of coverage.results.filter((r: any) => !requested.length || requested.includes(r.id))) {
  const replay = directory(entry.replay), config = read(path.join(replay, "config.json"));
  assert.equal(hash(path.join(directory(entry.audit), "summary.json")), entry.auditHash);
  const modelFile = path.join(config.source, "model.json"); assert.equal(hash(modelFile), entry.modelHash);
  const { costs } = read(modelFile), closures = config.marketAvailability?.closures ?? [];
  if (config.marketAvailability) assert.equal(hash(config.marketAvailability.file), config.marketAvailability.sha256);
  for (const ref of config.replaySourceReferences) assert.equal(hash(ref.file), ref.sha256);
  const sourceStart = config.start - 1000;
  const candles = loadEventCandles(sourceStart, config.end, 1000, closures.length ? closures : undefined);
  const summary = read(path.join(replay, "summary.json")), paths = new Map<string, EventExecutionPath>();
  for (const prefix of ["h1", "h2"]) {
    const metrics = prefix === "h1" ? summary.h1 : summary.results[0].metrics;
    const traceFile = path.join(replay, `${prefix}-trades.json`), trace = read(traceFile);
    const records = [...trace.map((r: any) => ({ ...r, forcedWait: false })),
      ...(metrics.marketAvailability?.forcedWaits ?? []).map((r: any) => ({ ...r, forcedWait: true }))].sort((a, b) => a.time - b.time);
    let maximumEquityError = 0, maximumLogErrorBps = 0, cancellations = 0, fills = 0, evaluationMs = 0;
    const oldErrors: number[] = [], borrowingErrors: number[] = [], worst: any[] = [];
    let terminalEquity = 0;
    for (const row of records) {
      const origin = (row.time - sourceStart) / 1000 - 1, end = (row.endTime - sourceStart) / 1000 - 1;
      assert.ok(Number.isInteger(origin) && Number.isInteger(end));
      const key = `${origin}:${end}`;
      let compressed = paths.get(key);
      if (!compressed) {
        compressed = summarizeEventExecutionPath(candles, origin, end, costs,
          !candles[end].carriedMark && !eventMarketUnavailable(closures, row.endTime)); paths.set(key, compressed);
      }
      const quantity = row.forcedWait ? row.quantityBefore : row.previousQuantity;
      const account = { equity: row.equityBefore, price: candles[origin].close,
        exposure: row.forcedWait ? quantity * candles[origin].close / row.equityBefore : row.exposureBefore };
      const request = row.forcedWait ? 0 : row.order.quantity;
      const begin = performance.now(), evaluated = evaluateEventExecutionPath(compressed, account, request);
      evaluationMs += performance.now() - begin;
      assert.ok(Math.abs(evaluated.filledQuantity - (row.forcedWait ? 0 : row.orderQuantity)) < 1e-10);
      assert.ok(Math.abs(evaluated.quantity - (row.forcedWait ? row.quantityAfter : row.positionQuantity)) < 1e-10);
      const error = Math.abs(evaluated.equity - row.equityAfter);
      maximumEquityError = Math.max(maximumEquityError, error);
      assert.ok(error < 1e-6, `${entry.id}/${prefix}/${row.time} equity error ${error}`);
      if (row.equityAfter > 0) maximumLogErrorBps = Math.max(maximumLogErrorBps,
        Math.abs(evaluated.logGrowth - Math.log(row.equityAfter / row.equityBefore)) * 10000);
      cancellations += Number(evaluated.canceled); fills += Number(evaluated.filledQuantity !== 0);
      if (!row.forcedWait && !evaluated.liquidated) {
        const atom = { return: compressed.closeRatio - 1, low: Math.min(1, compressed.lowRatio) - 1,
          high: Math.max(1, compressed.highRatio) - 1, duration: compressed.seconds / 60 };
        const old = eventHolding(row.order.exposure, atom, costs);
        if (!old.liquidated && old.factor > 0) {
          const errorBps = Math.log(row.order.equity * old.factor / evaluated.equity) * 10000;
          oldErrors.push(errorBps);
          worst.push({ time: row.time, requestedQuantity: request, filledQuantity: evaluated.filledQuantity,
            openingGapBps: (compressed.openRatio - 1) * 10000, originalTransitionErrorBps: errorBps });
        }
        const open = account.price * compressed.openRatio, postEquity = account.equity + quantity * (open - account.price) - evaluated.fee;
        const postExposure = evaluated.quantity * open / postEquity;
        const simple = eventHolding(postExposure, { return: compressed.closeRatio / compressed.openRatio - 1,
          low: Math.min(compressed.openRatio, compressed.lowRatio) / compressed.openRatio - 1,
          high: Math.max(compressed.openRatio, compressed.highRatio) / compressed.openRatio - 1,
          duration: compressed.seconds / 60 }, costs);
        if (!simple.liquidated && simple.factor > 0) borrowingErrors.push(Math.log(postEquity * simple.factor / evaluated.equity) * 10000);
      }
      if (row.endTime === config.end) {
        const settled = evaluateEventExecutionPath(compressed, account, request, "market");
        terminalEquity = settled.equity;
        assert.ok(Math.abs(terminalEquity - metrics.finalEquity) < 1e-6);
        assert.equal(fills + settled.terminalOrders, metrics.trades);
      }
    }
    assert.equal(cancellations, metrics.canceledOrders);
    const result = { id: entry.id, policy: prefix, modelHash: entry.modelHash, traceHash: hash(traceFile),
      decisions: trace.length, forcedWaitSegments: records.length - trace.length, maximumEquityError, maximumLogErrorBps,
      fills, cancellations, terminalEquity, evaluationMs,
      originalTransitionErrorBps: distribution(oldErrors), postOpenSimpleBorrowingErrorBps: distribution(borrowingErrors),
      worstOriginalTransitionErrors: worst.sort((a, b) => Math.abs(b.originalTransitionErrorBps) - Math.abs(a.originalTransitionErrorBps)).slice(0, 5) };
    results.push(result);
  }
  save("progress.json", { window: entry.id, processedWindows: results.length / 2, elapsedSeconds: (performance.now() - started) / 1000 });
  console.log(JSON.stringify({ window: entry.id, policies: results.slice(-2).map(r => ({ policy: r.policy, decisions: r.decisions,
    maximumEquityError: r.maximumEquityError, largestOldTransitionErrorBps: r.originalTransitionErrorBps.maxAbsolute })) }));
}
save("summary.json", { method: "Compressed next-open execution and constant-inventory holding reproduce the saved research simulator. Funding, maintenance, quantities and order acceptance remain that simulator's assumptions, not exact historical exchange reconstruction. Errors of the old transition use the same realized path, not out-of-sample forecasting errors. No policy is fitted or changed.",
  windows: results.length / 2, decisions: results.reduce((s, r) => s + r.decisions, 0),
  maximumEquityError: Math.max(...results.map(r => r.maximumEquityError)), maximumLogErrorBps: Math.max(...results.map(r => r.maximumLogErrorBps)),
  elapsedSeconds: (performance.now() - started) / 1000, results });
