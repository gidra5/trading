import assert from "node:assert/strict";
import test from "node:test";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { EventSecondBasis } from "./event-second-basis.js";
import { eventCandleShapes, eventFuturesFeatures, eventFuturesBasisDeviations, eventFittedFuturesInputs,
  eventNativeSecondFuturesFeatures } from "./event-futures-basis.js";
import { eventSecondDynamics, eventSecondDynamicsAt } from "./event-second-dynamics.js";
import { IndicatorEngine, buildSignalDefinitions } from "./analyze-technical-indicator-predictiveness.js";
import { eventMartingaleCandles } from "./screen-event-martingale.js";
import { eventControllerHolding, eventPathHolding, eventSignHorizonPaths, eventPolicyEvaluationFolds } from "./event-paths.js";
import { eventRidgeInfluence } from "./event-ridge-influence.js";
import { eventLocalValuePredictor } from "./event-local-value.js";
import type { SequentialDerivativesKlineRow } from "../packages/storage/src/derivatives-klines.js";
import { invertEventCandles, makeSamples, overlaps, replayEventPolicy, rollingEventScale } from "./research-event-policy.js";
import { buildEventOutcomeLookahead, buildEventPolicy, buildEventSignLookahead, DEFAULT_EVENT_COSTS } from "../packages/bot-algo/src/event-log-policy.js";
import { eventFeatures, eventFeatureWarmup, eventMoveLabel, observeMove, validateEventDistribution, reestimateEventTreeWithSources,
  EVENT_FEATURES, EVENT_SECOND_INPUTS, type EventCandle, type EventDistribution } from "../packages/bot-algo/src/event-distribution.js";
import { nativeSecondEventFeatures, NATIVE_SECOND_EVENT_FEATURES, NATIVE_SECOND_CONTEXT_FEATURES,
  NATIVE_SECOND_CONTEXT_LAGS, nativeSecondContextFeatures, NATIVE_SECOND_FLOW_CONTEXT_FEATURES,
  nativeSecondFlowContextFeatures, NATIVE_SECOND_FLOW_VWAP_CONTEXT_FEATURES,
  nativeSecondFlowVwapContextFeatures, NATIVE_SECOND_SELECTED_SIGN_FEATURES,
  nativeSecondSelectedSignFeatures, NATIVE_SECOND_DAY_CONTEXT_FEATURES,
  NATIVE_SECOND_DAY_CONTEXT_LAGS, nativeSecondDayContextFeatures,
  NATIVE_SECOND_VOLATILITY_CONTEXT_FEATURES, nativeSecondVolatilityContextFeatures } from "../packages/bot-algo/src/event-second-features.js";
import type { EventSignHead } from "../packages/bot-algo/src/event-sign.js";
import { eventSizeSignGroup } from "../packages/bot-algo/src/event-size-sign.js";
import { decideFittedEvent, trainEventFittedValue } from "../packages/bot-algo/src/event-fitted-value.js";
import { eventOriginScore, eventRefitOrigins } from "./research-event-refits.js";
import { eventFitPeriods, eventFittingExclusions, eventSourceDays, mergeEventSourceRanges } from "./event-fit-periods.js";
import { decideEventOneStep } from "../packages/bot-algo/src/event-one-step.js";
import { markUnavailableEventSeconds, validateEventSourceSecond, validateEventMarketClosures,
  type EventMarketClosure } from "../packages/bot-algo/src/event-market-availability.js";
import { summarizeEventExecutionPath, evaluateEventExecutionPath } from "../packages/bot-algo/src/event-execution-path.js";
import { prepareEventExecutionOneStep } from "../packages/bot-algo/src/event-execution-one-step.js";
import { prepareEventExecutionUpper } from "../packages/bot-algo/src/event-execution-upper.js";
import { prepareEventExecutionBackup } from "../packages/bot-algo/src/event-execution-backup.js";
import { prepareEventExecutionPartitions } from "../packages/bot-algo/src/event-execution-partitions.js";
import { prepareEventExecutionBoxUpper } from "../packages/bot-algo/src/event-execution-box.js";
import { prepareEventExecutionRiskNeutralUpper } from "../packages/bot-algo/src/event-execution-risk-upper.js";
import { prepareEventExecutionMeanSegmentUpper } from "../packages/bot-algo/src/event-execution-mean-segment.js";
import { maximizeEventAcceptanceSequence } from "../packages/bot-algo/src/event-execution-acceptance.js";
import { eventAffineSegmentSupports } from "../packages/bot-algo/src/event-affine-segment.js";

const candles = (): EventCandle[] => Array.from({ length: 1500 }, (_, i) => ({ openTime: i * 60_000,
  open: 100, high: 100, low: 100, close: 100, volume: 1 }));
const model: EventDistribution = { version: 1, clock: { thresholdBps: 20, maxCandles: 3 },
  featureNames: EVENT_FEATURES, nodes: [{ feature: -1, cut: 0, left: -1, right: -1, leaf: 0 }],
  trainingSamples: 1, counts: [1], priorClasses: [], classProbabilities: [],
  kernels: [[{ probability: 1, return: 0.01, low: 0, high: 0.01, duration: 3, next: 0 }]] };

const seconds = (): EventCandle[] => candles().slice(0, 200).map((c, i) => ({ ...c, openTime: i * 1000 }));
const secondModel = (): EventDistribution => ({ ...structuredClone(model),
  clock: { candleIntervalMs: 1000, thresholdBps: 20, maxCandles: 3 }, featureNames: NATIVE_SECOND_EVENT_FEATURES,
  kernels: [[{ probability: 1, return: .01, low: 0, high: .01, duration: 3 / 60, next: 0 }]] });

const sourceSeconds = () => seconds().map(c => ({ ...c, closeTime: c.openTime + 999, closed: true }));
const closure = (start = 64000, end = 70000): EventMarketClosure[] => [{ start, end, reason: "synthetic venue halt" }];
const closureSeconds = (closed = closure(), end = 200000) => {
  const rows = sourceSeconds().filter(c => c.openTime < end);
  for (const c of rows) if (closed.some(v => c.openTime >= v.start && c.openTime < v.end)) c.volume = 0;
  return markUnavailableEventSeconds(rows, 0, end, closed);
};

test("declared closures repair only replay marks, reject contradictions and preserve strict source validation", () => {
  const rows = sourceSeconds().filter(c => c.openTime !== 66000), before = structuredClone(rows);
  for (const c of rows) if (c.openTime >= 64000 && c.openTime < 70000) c.volume = 0;
  const partial = rows.find(c => c.openTime === 65000)!; partial.closeTime -= 500;
  const snapshot = structuredClone(rows), marked = markUnavailableEventSeconds(rows, 0, 80000, closure());
  assert.equal(marked.length, 80); assert.equal(marked.filter(c => c.carriedMark).length, 6);
  assert.ok(marked.slice(64, 70).every(c => c.close === 100 && c.volume === 0));
  assert.deepEqual(rows, snapshot); assert.equal(before.length, rows.length);
  assert.throws(() => validateEventSourceSecond(partial), /Malformed/);
  assert.throws(() => markUnavailableEventSeconds(rows, 0, 80000, []), /Malformed/);
  assert.throws(() => markUnavailableEventSeconds(rows.filter(c => c.openTime !== 71000), 0, 80000, closure()), /Gap outside/);
  partial.volume = 1; assert.throws(() => markUnavailableEventSeconds(rows, 0, 80000, closure()), /contradicts/);
  partial.volume = 0; partial.high = 101; assert.throws(() => markUnavailableEventSeconds(rows, 0, 80000, closure()), /contradicts/);
  assert.throws(() => validateEventMarketClosures([...closure(), ...closure()]), /overlapping/);
  assert.throws(() => makeSamples(marked, secondModel().clock, 64000, 80000, [], 1, "stride", NATIVE_SECOND_EVENT_FEATURES), /Replay-only/);
});

test("a replay starting inside a closure requires a real pre-closure anchor", () => {
  const rows = sourceSeconds(); for (const c of rows) if (c.openTime >= 64000 && c.openTime < 70000) c.volume = 0;
  assert.equal(markUnavailableEventSeconds(rows, 67000, 72000, closure())[0].close, 100);
  assert.throws(() => markUnavailableEventSeconds(rows.filter(c => c.openTime >= 64000), 67000, 72000, closure()), /Gap outside/);
  const adjacent = [...closure(64000, 67000), ...closure(67000, 70000)];
  assert.equal(markUnavailableEventSeconds(rows, 68000, 72000, adjacent)[0].close, 100);
  rows[70].volume = 0;
  assert.equal(markUnavailableEventSeconds(rows, 0, 72000, closure())[70].carriedMark, undefined);
});

test("unavailable next-open orders cancel once, never defer, and do not create value certificates for waits", () => {
  const c = closureSeconds(), p = buildEventPolicy(secondModel(), { ...DEFAULT_EVENT_COSTS, maxLeverage: 1 },
    { depths: 1, referenceEquity: 10000, referencePrice: 100 });
  const run = replayEventPolicy(c, p, 64000, 74000, 1, { trace: true, oneStepTerminal: "marked", marketClosures: closure() });
  assert.equal(run.canceledOrders, 1); assert.equal(run.marketAvailability!.unavailableCanceledOrders, 1);
  assert.equal(run.marketAvailability!.unavailableSeconds, 6);
  assert.deepEqual(run.trace.map(r => r.time), [64000, 71000]);
  assert.equal(run.trace[0].orderQuantity, 0); assert.equal(run.trace[0].interruptedByUnavailable, true);
  assert.ok(Number((run.trace[0].order as { quantity: number }).quantity) > 0);
  assert.equal(run.marketAvailability!.forcedWaits.length, 1);
  const wait = run.marketAvailability!.forcedWaits[0];
  assert.equal(wait.time, 65000); assert.equal(wait.endTime, 71000); assert.equal(wait.equityBefore, wait.equityAfter);
  assert.equal("order" in wait, false);
  assert.ok(run.positions!.every(p => p.entryTime >= 71000));
  assert.throws(() => replayEventPolicy(c, p, 64000, 74000, 1, { oneStepTerminal: "marked" }), /marks do not match/);
});

test("closure holding retains borrowing and reopening gap PnL in the account and position ledger", () => {
  const c = closureSeconds();
  for (let i = 70; i < c.length; i++) Object.assign(c[i], { open: 110, high: 110, low: 110, close: 110 });
  const p = buildEventPolicy(secondModel(), { ...DEFAULT_EVENT_COSTS, feeBps: 0, slippageBps: 0,
    maxLeverage: 2, shortBorrowBpsPerDay: 1440 }, { depths: 1, referenceEquity: 10000, referencePrice: 100 });
  const r = replayEventPolicy(c, p, 64000, 74000, 1,
    { trace: true, oneStepTerminal: "marked", initialQuantity: -10, cash: true, marketClosures: closure() });
  assert.ok(Math.abs(r.marketAvailability!.unavailableBorrow - .01) < 1e-8);
  assert.ok(Math.abs(r.borrow - (6 * 1000 + 4 * 1100) * .144 / 86400) < 1e-8);
  assert.equal(r.shortPnl, -100); assert.ok(Math.abs(r.finalEquity - (9900 - r.borrow)) < 1e-8);
  assert.ok(Math.abs(r.positionSummary!.equityChange - (r.finalEquity - 10000)) < 1e-8);
});

test("closure terminal inventory is unsettled rather than dust or an invented exit", () => {
  const c = closureSeconds(), p = buildEventPolicy(secondModel(), DEFAULT_EVENT_COSTS,
    { depths: 1, referenceEquity: 10000, referencePrice: 100 });
  for (const end of [64000, 68000, 70000]) {
    const r = replayEventPolicy(c, p, 64000, end, 1,
      { trace: true, oneStepTerminal: "marked", initialQuantity: -10, cash: true, marketClosures: closure() });
    assert.equal(r.trades, 0); assert.equal(r.fees, 0); assert.equal(r.terminalDust, 0);
    assert.equal(r.marketAvailability!.terminalUnavailable, true); assert.equal(r.marketAvailability!.unsettledNotional, 1000);
    assert.equal(r.marketAvailability!.terminalQuantity, -10); assert.equal(r.positionSummary!.active, 1);
  }
});

test("halt end and future prices cannot alter a committed action or the account prefix", () => {
  const p = buildEventPolicy(secondModel(), DEFAULT_EVENT_COSTS, { depths: 1, referenceEquity: 10000, referencePrice: 100 });
  const a = replayEventPolicy(closureSeconds(), p, 64000, 68000, 1,
    { trace: true, oneStepTerminal: "marked", marketClosures: closure(), initialQuantity: -10 });
  const later = closure(64000, 79000), c = closureSeconds(later);
  c[79] = { ...c[79], open: 50, low: 50, close: 50 };
  const b = replayEventPolicy(c, p, 64000, 68000, 1,
    { trace: true, oneStepTerminal: "marked", marketClosures: later, initialQuantity: -10 });
  assert.deepEqual(a, b);
  const plain = replayEventPolicy(seconds(), p, 64000, 80000, 1, { trace: true, oneStepTerminal: "marked" });
  const explicit = replayEventPolicy(seconds(), p, 64000, 80000, 1, { trace: true, oneStepTerminal: "marked", marketClosures: [] });
  assert.deepEqual(plain, explicit);
});

test("H2 availability replay certifies only optimized actions and keeps waits outside utility totals", () => {
  const p = buildEventPolicy(secondModel(), { ...DEFAULT_EVENT_COSTS, maxLeverage: 1 },
    { depths: 1, referenceEquity: 10000, referencePrice: 100 });
  const r = replayEventPolicy(closureSeconds(), p, 64000, 74000, 2, { trace: true, marketClosures: closure(),
    twoStep: { terminal: "marked", maxEvaluations: 64, globalUpper: true } });
  assert.deepEqual(r.trace.map(row => row.time), [64000, 71000]);
  assert.equal(r.decisions, 2); assert.equal(r.marketAvailability!.forcedWaits.length, 1);
  assert.ok(r.trace.every(row => (row.order as { converged: boolean }).converged));
  assert.equal(r.predictedGain, r.trace.reduce((sum, row) => sum + (row.order as { value: number }).value, 0));
});

test("a reopening gap can liquidate held inventory without creating a closing fill", () => {
  const c = closureSeconds();
  for (let i = 70; i < c.length; i++) Object.assign(c[i], { open: 140, high: 140, low: 140, close: 140 });
  const p = buildEventPolicy(secondModel(), { ...DEFAULT_EVENT_COSTS, feeBps: 0, slippageBps: 0 },
    { depths: 1, referenceEquity: 10000, referencePrice: 100 });
  const r = replayEventPolicy(c, p, 64000, 74000, 1,
    { trace: true, oneStepTerminal: "marked", marketClosures: closure(), initialQuantity: -300, cash: true });
  assert.equal(r.liquidations, 1); assert.equal(r.finalEquity, 0); assert.equal(r.trades, 0);
  assert.equal(r.positionSummary!.active, 0); assert.equal(r.positions![0].closeReason, "liquidation");
});

test("compressed execution matches an independent cash ledger across paths, funding regimes, orders and liquidation", () => {
  let seed = 92147;
  const random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 2 ** 32; };
  const costs = { ...DEFAULT_EVENT_COSTS, maxLeverage: 2, maintenanceMargin: .05,
    minNotional: 5, maxNotional: 1000, minQuantity: .01, quantityStep: .01,
    longBorrowBpsPerDay: 10000, shortBorrowBpsPerDay: 21000 };
  for (let sample = 0; sample < 24; sample++) {
    const c = seconds().slice(0, 31);
    for (let i = 1; i < c.length; i++) {
      const open = c[i - 1].close * (1 + (random() - .5) * .12), close = open * (1 + (random() - .5) * .12);
      c[i] = { ...c[i], open, close, low: Math.min(open, close) * (1 - random() * .04),
        high: Math.max(open, close) * (1 + random() * .04) };
    }
    if (sample % 3 === 0) c[1].carriedMark = true;
    const path = summarizeEventExecutionPath(c, 0, 30, costs);
    for (const initialQuantity of [-45, -20, -10, 0, 5, 10, 20, 45]) for (const request of [-20, -10, -.05, -.01, 0, .01, .05, 10, 20]) {
      const account = { equity: 1000, price: 100, exposure: initialQuantity / 10 };
      const result = evaluateEventExecutionPath(path, account, request);
      let q = initialQuantity, cash = account.equity - q * account.price, totalBorrow = 0, fee = 0, fill = 0, liquidated = false;
      for (let i = 1; i < c.length; i++) {
        const bar = c[i], marked = cash + q * bar.open;
        if (marked <= costs.maintenanceMargin * Math.abs(q) * bar.open) { liquidated = true; break; }
        if (i === 1 && request && !bar.carriedMark) {
          const notional = Math.abs(request) * bar.open, cost = notional * .0012;
          const before = q * bar.open / marked, after = (q + request) * bar.open / (marked - cost);
          const cap = Math.abs(after) <= costs.maxLeverage + 1e-8 || (Math.abs(before) > costs.maxLeverage
            && Math.abs(after) < Math.abs(before) - 1e-9 && notional >= costs.maxNotional - costs.quantityStep * bar.open - 1e-8);
          if (Math.abs(request) >= costs.minQuantity - 1e-12 && notional >= costs.minNotional - 1e-8
            && notional <= costs.maxNotional + 1e-8 && marked > cost && cap) {
            q = Math.round((q + request) / costs.quantityStep) * costs.quantityStep;
            cash = marked - cost - q * bar.open; fee = cost; fill = request;
          }
        }
        const interest = q > 0 ? Math.max(0, -cash) * costs.longBorrowBpsPerDay / 10000 / 86400
          : -q * bar.open * costs.shortBorrowBpsPerDay / 10000 / 86400;
        cash -= interest; totalBorrow += interest;
        const worst = q >= 0 ? bar.low : bar.high;
        if (cash + q * worst <= costs.maintenanceMargin * Math.abs(q) * worst || cash + q * bar.close <= 0) {
          liquidated = true; break;
        }
      }
      assert.equal(result.liquidated, liquidated, `liquidation sample ${sample}, quantity ${initialQuantity}, request ${request}`);
      assert.equal(result.filledQuantity, fill); assert.ok(Math.abs(result.fee - fee) < 1e-10);
      if (liquidated) { assert.equal(result.equity, 0); assert.equal(result.logGrowth, -Infinity); }
      else {
        assert.ok(Math.abs(result.equity - (cash + q * c[30].close)) < 1e-8);
        assert.ok(Math.abs(result.borrowing! - totalBorrow) < 1e-8); assert.equal(result.quantity, q);
      }
    }
  }
});

test("equal return-duration-extrema summaries can have different short borrowing and terminal equity", () => {
  const paths = [[100, 120, 120, 100], [100, 80, 80, 100]].map(opens => [seconds()[0],
    ...opens.map((open, i) => ({ openTime: (i + 1) * 1000, open, close: 100, low: 80, high: 120, volume: 1 }))]);
  const costs = { ...DEFAULT_EVENT_COSTS, shortBorrowBpsPerDay: 8640 };
  const summaries = paths.map(p => summarizeEventExecutionPath(p, 0, 4, costs));
  const atom = (p: typeof summaries[number]) => [p.openRatio, p.closeRatio, p.lowRatio, p.highRatio, p.seconds];
  assert.deepEqual(atom(summaries[0]), atom(summaries[1]));
  const account = { equity: 1000, price: 100, exposure: -.1 };
  const result = summaries.map(p => evaluateEventExecutionPath(p, account, 0));
  assert.ok(Math.abs(result[0].borrowing! - .0044) < 1e-12);
  assert.ok(Math.abs(result[1].borrowing! - .0036) < 1e-12);
  assert.ok(result[0].equity < result[1].equity);
});

test("empirical mixture provenance retains distinct paths even when old return summaries coincide", () => {
  const m = secondModel(), samples = Array.from({ length: 130 }, (_, i) => ({ start: i * 10, end: i * 10 + 3,
    features: m.featureNames.map(() => 0), nextFeatures: m.featureNames.map(() => 0),
    return: .01, low: 0, high: .01, duration: 3 / 60, label: eventMoveLabel(.01, 3 / 60, m.clock) }));
  const weights = samples.map((_, i) => i % 2 ? 1 : 10);
  const fitted = reestimateEventTreeWithSources(m, samples, 32, weights);
  assert.deepEqual(fitted.sources[0].slice(0, 130), samples.map((_, i) => i));
  assert.deepEqual(fitted.sources[0].slice(130), samples.filter((_, i) => i % 2 === 0).map((_, i) => i * 2));
  assert.equal(fitted.sources[0].length, fitted.model.kernels[0].length);
  assert.ok(Math.abs(fitted.model.kernels[0][0].probability / fitted.model.kernels[0][1].probability - 10) < 1e-12);
});

test("compressed debt growth, open rejection, terminal dust and closure settlement follow account semantics", () => {
  const c = seconds().slice(0, 5), costs = { ...DEFAULT_EVENT_COSTS, longBorrowBpsPerDay: 864000 };
  const path = summarizeEventExecutionPath(c, 0, 4, costs);
  const held = evaluateEventExecutionPath(path, { equity: 100, price: 100, exposure: 2 }, 0);
  assert.ok(Math.abs(held.borrowing! - 100 * (1.001 ** 4 - 1)) < 1e-10);
  const cash = { equity: 10000, price: 100, exposure: 0 };
  const rejected = evaluateEventExecutionPath({ ...path, openRatio: 1.1 }, cash, 499);
  assert.equal(rejected.canceled, true); assert.equal(rejected.equity, cash.equity); assert.equal(rejected.fee, 0);
  const dust = evaluateEventExecutionPath(path, { ...cash, exposure: .00001 }, 0, "market");
  assert.equal(dust.terminalOrders, 0); assert.ok(Math.abs(dust.terminalDust - .1) < 1e-12);
  const unsettled = evaluateEventExecutionPath({ ...path, terminalAvailable: false }, { ...cash, exposure: .1 }, 0, "market");
  assert.equal(unsettled.terminalOrders, 0); assert.equal(unsettled.terminalDust, 0); assert.equal(unsettled.unsettledNotional, 1000);
  const jump = seconds().slice(0, 2); Object.assign(jump[1], { open: 200, high: 200, low: 200, close: 200 });
  const preOrderRuin = evaluateEventExecutionPath(summarizeEventExecutionPath(jump, 0, 1, costs),
    { equity: 100, price: 100, exposure: -1 }, 1);
  assert.equal(preOrderRuin.liquidationPhase, "opening"); assert.equal(preOrderRuin.filledQuantity, 0);
  assert.equal(preOrderRuin.canceled, false); assert.equal(preOrderRuin.fee, 0);
});

test("execution request search matches exhaustive lattices with outcome-dependent fills, borrowing and dust", () => {
  let seed = 481903;
  const random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 2 ** 32; };
  let checked = 0;
  for (const terminal of ["marked", "market"] as const) for (const leverage of [1, 3]) for (let sample = 0; sample < 30; sample++) {
    const costs = { ...DEFAULT_EVENT_COSTS, feeBps: sample % 3 === 0 ? 0 : 11, slippageBps: 0,
      maxLeverage: leverage, maintenanceMargin: .04, minQuantity: .25, quantityStep: .25,
      minNotional: sample % 2 ? 5 : 0, maxNotional: [8, 25, 70][sample % 3],
      longBorrowBpsPerDay: 14400, shortBorrowBpsPerDay: 86400 };
    const kernel = [.15, .35, .5].map((probability, j) => {
      const bars: EventCandle[] = [{ openTime: 0, open: 10, high: 10, low: 10, close: 10, volume: 1 }];
      for (let i = 1; i <= 5; i++) {
        const open = bars[i - 1].close * (.8 + .4 * random()), close = open * (.8 + .4 * random());
        bars.push({ openTime: i * 1000, open, close, low: Math.min(open, close) * .97,
          high: Math.max(open, close) * 1.03, volume: 1 });
      }
      if (sample % 4 === 0 && j === 1) bars[1].carriedMark = true;
      return { probability, path: summarizeEventExecutionPath(bars, 0, 5, costs, sample % 5 !== 0) };
    });
    const solve = prepareEventExecutionOneStep(kernel, terminal, { captureRegions: true }), upper = prepareEventExecutionUpper(kernel);
    const mean = prepareEventExecutionOneStep(kernel, terminal, { objective: "mean" });
    const alternativeProbabilities = kernel.map((_, index) => kernel[kernel.length - 1 - index].probability);
    const robust = prepareEventExecutionOneStep(kernel, terminal, { alternativeProbabilities: [alternativeProbabilities] });
    for (const quantity of [-10, -4, -.125, 0, .125, 2, 6, 12]) {
      const account = { equity: 40, price: 10, exposure: quantity / 4 }, result = solve(account);
      const maximum = Math.ceil(costs.maxNotional / Math.min(...kernel.map(a => a.path.openRatio * account.price)) / costs.quantityStep) + 4;
      const value = (k: number) => kernel.reduce((s, a) => s + a.probability
        * evaluateEventExecutionPath(a.path, account, k * costs.quantityStep, terminal).logGrowth, 0);
      let best = -Infinity;
      let bestMean = -Infinity;
      assert.ok("requestRegions" in result);
      const regions = result.requestRegions;
      for (let k = -maximum; k <= maximum; k++) {
        const score = value(k); best = Math.max(best, score);
        if (Number.isFinite(score)) bestMean = Math.max(bestMean, kernel.reduce((sum, atom) => sum + atom.probability
          * evaluateEventExecutionPath(atom.path, account, k * costs.quantityStep, terminal).equity / account.equity, 0));
        if (Math.abs(k) <= result.search.maximumLots) assert.equal(regions.some(([lo, hi]) => k >= lo && k <= hi), Number.isFinite(score),
          `region coverage ${terminal}, L=${leverage}, sample=${sample}, Q=${quantity}, lot=${k}`);
      }
      for (let i = 0; i < regions.length; i++) {
        const [lo, hi] = regions[i]; if (i) assert.ok(lo > regions[i - 1][1]);
        if (lo === hi) continue;
        const mid = Math.floor((lo + hi) / 2), fraction = (mid - lo) / (hi - lo);
        for (const atom of kernel) {
          const points = [lo, mid, hi].map(k => evaluateEventExecutionPath(atom.path, account, k * costs.quantityStep, terminal));
          assert.ok(points.every(p => !p.liquidated && p.canceled === points[0].canceled
            && (p.filledQuantity !== 0) === (points[0].filledQuantity !== 0)));
          assert.ok(Math.abs(points[1].equity - (points[0].equity + fraction * (points[2].equity - points[0].equity))) < 1e-9);
          assert.ok(Math.abs(points[1].quantity - (points[0].quantity + fraction * (points[2].quantity - points[0].quantity))) < 1e-9);
        }
      }
      assert.equal(result.complete, true);
      if (Number.isFinite(best)) assert.ok(Math.abs(result.value - best) < 1e-11,
        `execution optimum ${terminal}, L=${leverage}, sample=${sample}, Q=${quantity}: ${result.value} vs ${best}`);
      else assert.equal(result.value, best);
      assert.ok(upper(account) >= best - 1e-10, `opening-information upper ${terminal}, L=${leverage}, sample=${sample}, Q=${quantity}`);
      assert.equal(result.value, value(Math.round(result.quantity / costs.quantityStep))); checked++;
      const meanResult = mean(account); assert.equal(meanResult.objective, "mean-terminal-equity-ratio");
      assert.ok(meanResult.value === bestMean || Math.abs(meanResult.value - bestMean) < 1e-11,
        `execution mean optimum ${terminal}, L=${leverage}, sample=${sample}, Q=${quantity}: ${meanResult.value} vs ${bestMean}`);
      const minimumLiquidatableEquity = account.equity * (sample % 2 ? .98 : .9);
      const constrained = solve(account, { minimumLiquidatableEquity });
      let constrainedBest = -Infinity;
      for (let k = -maximum; k <= maximum; k++) {
        let score = 0;
        for (const atom of kernel) {
          const next = evaluateEventExecutionPath(atom.path, account, k * costs.quantityStep, terminal);
          const closeRate = (costs.feeBps + costs.slippageBps) / 10_000;
          const liquidatableEquity = next.equity - Math.abs(next.quantity) * next.price * closeRate;
          if (!Number.isFinite(next.logGrowth) || liquidatableEquity < minimumLiquidatableEquity - 1e-8) {
            score = -Infinity; break;
          }
          score += atom.probability * next.logGrowth;
        }
        constrainedBest = Math.max(constrainedBest, score);
      }
      assert.ok(constrained.value === constrainedBest || Math.abs(constrained.value - constrainedBest) < 1e-11,
        `risk-constrained optimum ${terminal}, L=${leverage}, sample=${sample}, Q=${quantity}: ${constrained.value} vs ${constrainedBest}`);
      const selectedConstrained = kernel.map(atom => evaluateEventExecutionPath(atom.path, account, constrained.quantity, terminal));
      assert.ok(constrained.value === -Infinity || selectedConstrained.every(next =>
        next.equity - Math.abs(next.quantity) * next.price * (costs.feeBps + costs.slippageBps) / 10_000
          >= minimumLiquidatableEquity - 1e-8));
      const robustResult = robust(account, { minimumLiquidatableEquity });
      let robustBest = -Infinity;
      for (let k = -maximum; k <= maximum; k++) {
        const outcomes = kernel.map(atom => evaluateEventExecutionPath(atom.path, account, k * costs.quantityStep, terminal));
        const closeRate = (costs.feeBps + costs.slippageBps) / 10_000;
        if (outcomes.some(next => !Number.isFinite(next.logGrowth)
          || next.equity - Math.abs(next.quantity) * next.price * closeRate < minimumLiquidatableEquity - 1e-8)) continue;
        const base = outcomes.reduce((sum, next, index) => sum + kernel[index].probability * next.logGrowth, 0);
        const alternative = outcomes.reduce((sum, next, index) => sum + alternativeProbabilities[index] * next.logGrowth, 0);
        robustBest = Math.max(robustBest, Math.min(base, alternative));
      }
      assert.ok(robustResult.value === robustBest || Math.abs(robustResult.value - robustBest) < 1e-11,
        `robust risk-constrained optimum ${terminal}, L=${leverage}, sample=${sample}, Q=${quantity}: ${robustResult.value} vs ${robustBest}`);
    }
  }
  assert.equal(checked, 960);
  assert.throws(() => prepareEventExecutionOneStep([{ probability: 1, path: summarizeEventExecutionPath(
    seconds().slice(0, 2), 0, 1, DEFAULT_EVENT_COSTS) }])({ equity: 100, price: 100, exposure: 0 },
    { minimumLiquidatableEquity: -1 }), /minimum liquidatable equity/);
  const valid = [{ probability: 1, path: summarizeEventExecutionPath(seconds().slice(0, 2), 0, 1, DEFAULT_EVENT_COSTS) }];
  assert.throws(() => prepareEventExecutionOneStep(valid, "marked", { alternativeProbabilities: [[.5]] }),
    /alternative execution probabilities/);
  assert.throws(() => prepareEventExecutionOneStep(valid, "marked", { objective: "mean", alternativeProbabilities: [[1]] }),
    /Robust mean/);
});

test("opening-information bound retains joint outcomes, fees and above-cap holds", () => {
  const costs = { ...DEFAULT_EVENT_COSTS, maxLeverage: 1, minNotional: 0, maxNotional: 20,
    quantityStep: .1, minQuantity: 0, longBorrowBpsPerDay: 0, shortBorrowBpsPerDay: 0 };
  const make = (open: number, close: number) => summarizeEventExecutionPath([
    { openTime: 0, open: 10, close: 10, low: 10, high: 10, volume: 1 },
    { openTime: 1000, open, close, low: Math.min(open, close), high: Math.max(open, close), volume: 1 },
  ], 0, 1, costs);
  const kernel = [{ probability: .5, path: make(10, 11) }, { probability: .5, path: make(10, 9) }];
  const bound = prepareEventExecutionUpper(kernel), flat = { equity: 100, price: 10, exposure: 0 };
  assert.equal(bound.groups, 1); assert.ok(bound(flat) < 1e-8, "No revelation of future return when opens coincide");
  const distinct = prepareEventExecutionUpper([{ probability: .5, path: make(10.1, 11) }, { probability: .5, path: make(9.9, 9) }]);
  assert.equal(distinct.groups, 2); assert.ok(distinct(flat) > .07, "Opening-price information is a deliberate relaxation");
  const positive = [{ probability: 1, path: make(10, 11) }];
  const high = { ...flat, exposure: 2 };
  assert.ok(prepareEventExecutionUpper(positive)(high) >= Math.log(1.2));
  const noBound = prepareEventExecutionUpper([{ probability: 1, path: { ...positive[0].path,
    costs: { ...costs, maintenanceMargin: 0 } } }]);
  assert.equal(noBound.supported, false); assert.equal(noBound(flat), Infinity);
  assert.equal(bound({ equity: 1e308, price: 1e-308, exposure: 2 }), Infinity);
  const doomed = { ...flat, exposure: 201 };
  assert.equal(prepareEventExecutionUpper(positive)(doomed), -Infinity);
  assert.throws(() => prepareEventExecutionUpper([{ probability: 1 + 1e-9, path: positive[0].path }]), /normalized/);
});

test("risk-neutral execution relaxation bounds every common request and is convex in balances", () => {
  let seed = 739120;
  const random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 2 ** 32; };
  for (let sample = 0; sample < 40; sample++) {
    const costs = { ...DEFAULT_EVENT_COSTS, feeBps: sample % 3 ? 12 : 0, slippageBps: 0,
      maxLeverage: 3, maintenanceMargin: .04, minQuantity: .25, quantityStep: .25,
      minNotional: sample % 2 ? 5 : 0, maxNotional: 25,
      longBorrowBpsPerDay: 7200, shortBorrowBpsPerDay: 14400 };
    const kernel = [.2, .3, .5].map((probability, j) => {
      const bars: EventCandle[] = [{ openTime: 0, open: 10, high: 10, low: 10, close: 10, volume: 1 }];
      for (let i = 1; i <= 4; i++) {
        const open = bars[i - 1].close * (.82 + .36 * random()), close = open * (.82 + .36 * random());
        bars.push({ openTime: i * 1000, open, close, low: Math.min(open, close) * .96,
          high: Math.max(open, close) * 1.04, volume: 1 });
      }
      if (sample % 5 === 0 && j === 1) bars[1].carriedMark = true;
      return { probability, path: summarizeEventExecutionPath(bars, 0, 4, costs) };
    });
    const upper = prepareEventExecutionRiskNeutralUpper(kernel), price = 10;
    const endpoints = [[30 + 40 * random(), -3 + 6 * random()], [30 + 40 * random(), -3 + 6 * random()]];
    const account = ([cash, quantity]: number[]) => {
      const equity = cash + quantity * price; return { equity, price, exposure: quantity * price / equity };
    };
    if (endpoints.some(([cash, quantity]) => cash + quantity * price <= 5)) { sample--; continue; }
    for (const balance of endpoints) {
      const a = account(balance), bound = upper(a);
      for (let lots = -14; lots <= 14; lots++) {
        const expected = kernel.reduce((sum, atom) => sum + atom.probability
          * evaluateEventExecutionPath(atom.path, a, lots * costs.quantityStep).equity, 0);
        assert.ok(bound >= expected - 1e-8, `risk upper sample=${sample}, lots=${lots}: ${bound} < ${expected}`);
      }
    }
    const middle = endpoints[0].map((value, i) => (value + endpoints[1][i]) / 2);
    assert.ok(upper(account(middle)) <= (upper(account(endpoints[0])) + upper(account(endpoints[1]))) / 2 + 1e-8,
      `risk upper convexity sample=${sample}`);
  }
});

test("execution Bellman backup bounds complete policies and prunes only against achievable values", () => {
  const costs = { ...DEFAULT_EVENT_COSTS, maxLeverage: 1, minNotional: 0, maxNotional: 100,
    quantityStep: 1, minQuantity: 1, longBorrowBpsPerDay: 0, shortBorrowBpsPerDay: 0 };
  const path = summarizeEventExecutionPath([
    { openTime: 0, open: 10, close: 10, low: 10, high: 10, volume: 1 },
    { openTime: 1000, open: 10, close: 11, low: 10, high: 11, volume: 1 },
  ], 0, 1, costs);
  const kernel = [{ probability: 1, next: 0, path }], account = { equity: 100, price: 10, exposure: 0 };
  const backup = prepareEventExecutionBackup([kernel]);
  const partial = backup(0, account, 9, { maxSolves: 0 });
  assert.equal(partial.status, "budget"); assert.equal(partial.value, null);
  const first = evaluateEventExecutionPath(path, account, 9);
  let exact = -Infinity;
  for (let quantity = -12; quantity <= 12; quantity++) {
    const second = evaluateEventExecutionPath(path, first, quantity);
    exact = Math.max(exact, Math.log(second.equity / account.equity));
  }
  assert.ok(partial.lowerValue <= exact + 1e-12 && partial.upperValue >= exact - 1e-12);
  const certified = prepareEventExecutionBackup([kernel])(0, account, 9, { valueTolerance: .01, maxSolves: 0 });
  assert.equal(certified.status, "certified"); assert.equal(certified.complete, false); assert.equal(certified.value, null);
  assert.equal(certified.certified, true); assert.equal(certified.continuationSolves, 0);
  assert.ok(certified.lowerValue <= exact + 1e-12 && certified.upperValue >= exact - 1e-12);
  assert.ok(certified.upperValue - certified.lowerValue <= certified.valueTolerance);
  const complete = backup(0, account, 9);
  assert.equal(complete.status, "complete"); assert.ok(Math.abs(complete.value! - exact) < 1e-12);
  const cached = backup(0, account, 9); assert.equal(cached.continuationSolves, 0); assert.deepEqual(cached.continuationPolicy, complete.continuationPolicy);
  const pruned = backup(0, account, 0, { incumbent: complete.value! });
  assert.equal(pruned.status, "pruned"); assert.equal(pruned.value, null); assert.equal(pruned.continuationSolves, 0);
  assert.ok(pruned.upperValue <= complete.value!);
  const wrong = [{ probability: 1, next: 0, path: { ...path, costs: { ...costs, feeBps: 2 } } }];
  assert.throws(() => prepareEventExecutionBackup([kernel, wrong]), /identical costs/);
});

test("coarsening execution-information partitions tightens valid bounds to the original request optimum", () => {
  const costs = { ...DEFAULT_EVENT_COSTS, maxLeverage: 2, maxNotional: 100, minNotional: 3,
    quantityStep: .5, minQuantity: .5, longBorrowBpsPerDay: 1000, shortBorrowBpsPerDay: 5000 };
  const kernel = [.08, .12, .15, .2, .25, .2].map((probability, i) => {
    const open = [9.5, 10, 10, 10, 10.4, 11][i], close = open * [.97, 1.09, .95, 1.02, .93, 1.07][i];
    return { probability, path: summarizeEventExecutionPath([
      { openTime: 0, open: 10, close: 10, low: 10, high: 10, volume: 1 },
      { openTime: 1000, open, close, low: Math.min(open, close), high: Math.max(open, close), volume: 1 },
    ], 0, 1, costs) };
  });
  const original = structuredClone(kernel), partitions = prepareEventExecutionPartitions(kernel, 3);
  assert.deepEqual(partitions.groupCounts, [1, 2, 4, 4]);
  for (const exposure of [-3, -1, 0, .5, 1, 2.2, 5]) {
    const account = { equity: 40, price: 10, exposure };
    const score = (quantity: number) => kernel.reduce((s, a) => s + a.probability * evaluateEventExecutionPath(a.path, account, quantity).logGrowth, 0);
    let exact = -Infinity;
    for (let k = -24; k <= 24; k++) exact = Math.max(exact, score(k * costs.quantityStep));
    let previous = -Infinity;
    for (let depth = 0; depth <= 3; depth++) {
      const result = partitions(account, depth);
      assert.ok(result.lowerValue <= exact + 1e-10 && result.upperValue >= exact - 1e-10);
      assert.ok(result.upperValue >= previous - 1e-10); previous = result.upperValue;
      assert.equal(result.lowerValue, score(result.quantity));
      if (!depth) { assert.equal(result.complete, true); assert.ok(Math.abs(result.lowerValue - exact) < 1e-10); }
    }
  }
  assert.deepEqual(kernel, original);
  const sameOpen = prepareEventExecutionPartitions(kernel.slice(1, 4).map(a => ({ ...a, probability: a.probability / .47 })), 3);
  assert.deepEqual(sameOpen.groupCounts, [1, 1, 1, 1]);
  assert.equal(sameOpen({ equity: 40, price: 10, exposure: 0 }, 3).complete, true);
});

test("balance-box upper dominates exhaustive common-request policies across inventory and acceptance boundaries", () => {
  let seed = 501903, checked = 0;
  const random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 2 ** 32; };
  for (let sample = 0; sample < 120; sample++) {
    const costs = { ...DEFAULT_EVENT_COSTS, feeBps: sample % 3 ? 13 : 0, slippageBps: 0,
      maxLeverage: [1, 3, 5][sample % 3], maintenanceMargin: sample % 4 ? .04 : 0,
      minQuantity: .25, quantityStep: .25, minNotional: sample % 2 ? 5 : 0,
      maxNotional: [8, 25, 70][sample % 3], longBorrowBpsPerDay: 14400, shortBorrowBpsPerDay: 86400 };
    const sharedOpen = 10 * (.9 + .2 * random());
    const kernel = [.15, .35, .5].map((probability, j) => {
      const bars: EventCandle[] = [{ openTime: 0, open: 10, high: 10, low: 10, close: 10, volume: 1 }];
      for (let i = 1; i <= 5; i++) {
        const open = i === 1 && (sample % 2 === 0 || j < 2) ? sharedOpen : bars[i - 1].close * (.85 + .3 * random());
        const close = open * (.85 + .3 * random());
        bars.push({ openTime: i * 1000, open, close, low: Math.min(open, close) * .97,
          high: Math.max(open, close) * 1.03, volume: 1 });
      }
      if (sample % 4 === 0 && j === 1) bars[1].carriedMark = true;
      return { probability, path: summarizeEventExecutionPath(bars, 0, 5, costs) };
    });
    const Q = [-10, -4, -.125, 0, .125, 2, 6, 12][sample % 8], C = 40 - Q * 10;
    const width = [0, .5, 2][Math.floor(sample / 8) % 3], lattice = Number.isInteger(Q / .25);
    const box = { price: 10, cash: [C - width * 5, C + width * 5] as const,
      quantity: [Q - width, Q + width] as const, quantityLattice: lattice };
    const upper = prepareEventExecutionBoxUpper(kernel)(box);
    assert.equal(upper.complete, true);
    const ordered = prepareEventExecutionBoxUpper(kernel, { ordered: true, valueTolerance: 0 })(box);
    assert.equal(ordered.complete, true);
    assert.ok(ordered.upperLogEquity <= upper.upperLogEquity + 1e-10);
    const linked = prepareEventExecutionBoxUpper(kernel)({ ...box,
      balanceVertices: [[box.cash[0], box.quantity[1]], [box.cash[1], box.quantity[0]]] });
    assert.equal(linked.complete, true);
    assert.ok(linked.upperLogEquity <= upper.upperLogEquity + 1e-10);
    const coupled = prepareEventExecutionBoxUpper(kernel, { ordered: true, coupledWealth: true, valueTolerance: 0 })({ ...box,
      balanceVertices: [[box.cash[0], box.quantity[1]], [box.cash[1], box.quantity[0]]] });
    assert.equal(coupled.complete, true);
    assert.ok(coupled.upperLogEquity <= linked.upperLogEquity + 1e-10);
    const maximum = Math.ceil(costs.maxNotional / Math.min(...kernel.map(a => a.path.openRatio * 10)) / .25) + 4;
    for (let i = 0; i <= 4; i++) for (let j = 0; j <= 4; j++) {
      const cash = C + (i / 2 - 1) * width * 5;
      const quantity = Q + (j / 2 - 1) * width;
      if (lattice && !Number.isInteger(quantity / .25)) continue;
      const equity = cash + quantity * 10; if (!(equity > 0)) continue;
      const account = { equity, price: 10, exposure: quantity * 10 / equity };
      let best = -Infinity;
      for (let k = -maximum; k <= maximum; k++) best = Math.max(best, kernel.reduce((s, a) =>
        s + a.probability * Math.log(evaluateEventExecutionPath(a.path, account, k * .25).equity), 0));
      assert.ok(upper.upperLogEquity >= best - 1e-10,
        `sample=${sample}, cash=${cash}, quantity=${quantity}: upper=${upper.upperLogEquity}, exact=${best}`);
      assert.ok(ordered.upperLogEquity >= best - 1e-10,
        `ordered sample=${sample}, cash=${cash}, quantity=${quantity}: upper=${ordered.upperLogEquity}, exact=${best}`);
      if (i + j === 4) assert.ok(linked.upperLogEquity >= best - 1e-10,
        `linked sample=${sample}, cash=${cash}, quantity=${quantity}: upper=${linked.upperLogEquity}, exact=${best}`);
      if (i + j === 4) assert.ok(coupled.upperLogEquity >= best - 1e-10,
        `coupled sample=${sample}, cash=${cash}, quantity=${quantity}: upper=${coupled.upperLogEquity}, exact=${best}`);
      if (!width && lattice && Number.isFinite(best)) assert.ok(upper.upperLogEquity - best < 1e-7);
      checked++;
    }
  }
  assert.equal(checked, 3000);
});

test("mean-equity segment bound preserves one account coordinate and covers exact request policies", () => {
  let seed = 904731;
  const random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 2 ** 32; };
  for (let sample = 0; sample < 60; sample++) {
    const costs = { ...DEFAULT_EVENT_COSTS, feeBps: sample % 3 ? 13 : 0, slippageBps: 0,
      maxLeverage: [1, 3, 5][sample % 3], maintenanceMargin: .04,
      minQuantity: .25, quantityStep: .25, minNotional: sample % 2 ? 5 : 0,
      maxNotional: [8, 25, 70][sample % 3], longBorrowBpsPerDay: 14400, shortBorrowBpsPerDay: 86400 };
    const kernel = [.2, .3, .5].map((probability, j) => {
      const bars: EventCandle[] = [{ openTime: 0, open: 10, high: 10, low: 10, close: 10, volume: 1 }];
      for (let i = 1; i <= 4; i++) {
        const open = bars[i - 1].close * (.85 + .3 * random()), close = open * (.85 + .3 * random());
        bars.push({ openTime: i * 1000, open, close, low: Math.min(open, close) * .97,
          high: Math.max(open, close) * 1.03, volume: 1 });
      }
      if (sample % 5 === 0 && j === 1) bars[1].carriedMark = true;
      return { probability, path: summarizeEventExecutionPath(bars, 0, 4, costs) };
    });
    const vertices = [[25 + 35 * random(), -2 + 4 * random()], [25 + 35 * random(), -2 + 4 * random()]] as const;
    if (vertices.some(([cash, quantity]) => cash + 10 * quantity <= 3)) { sample--; continue; }
    const bound = prepareEventExecutionMeanSegmentUpper(kernel)({ price: 10, vertices });
    assert.equal(bound.complete, true);
    for (const t of [0, .25, .5, .75, 1]) {
      const cash = vertices[0][0] + t * (vertices[1][0] - vertices[0][0]);
      const quantity = vertices[0][1] + t * (vertices[1][1] - vertices[0][1]);
      const equity = cash + quantity * 10, account = { equity, price: 10, exposure: quantity * 10 / equity };
      let exact = -Infinity;
      for (let lots = -40; lots <= 40; lots++) {
        const outcomes = kernel.map(atom => evaluateEventExecutionPath(atom.path, account, lots * costs.quantityStep));
        if (outcomes.some(row => !Number.isFinite(row.logGrowth))) continue;
        exact = Math.max(exact, outcomes.reduce((sum, row, i) => sum + kernel[i].probability * row.equity, 0));
      }
      assert.ok(bound.upperTerminalEquity >= exact - 1e-8,
        `mean segment sample=${sample}, t=${t}: ${bound.upperTerminalEquity} < ${exact}`);
      if (t === 0 || t === 1) assert.ok(bound.endpointUpperTerminalEquities[t] >= exact - 1e-8);
    }
  }
});

test("affine balance-segment supports match exhaustive primal breakpoints", () => {
  for (let sample = 0; sample < 100; sample++) {
    const facets = Array.from({ length: 4 }, (_, i) => ({ a: 10 + Math.sin(sample * 3 + i) * 3,
      b: Math.cos(sample + i * 3), slope: sample % 4 ? Math.sin(sample * 7 + i * 5) * 4 : 0 }));
    const supports = eventAffineSegmentSupports(facets);
    for (let k = -10; k <= 10; k++) {
      const candidates = [0, 1];
      for (const a of facets) for (const b of facets) {
        const t = (b.a - a.a + (b.b - a.b) * k) / (a.slope - b.slope);
        if (t > 0 && t < 1) candidates.push(t);
      }
      const exact = Math.max(...candidates.map(t => Math.min(...facets.map(f => f.a + f.b * k + f.slope * t))));
      const upper = Math.min(...supports.map(s => s.a + s.b * k));
      assert.ok(Math.abs(upper - exact) < 1e-11);
    }
  }
});

test("ordered acceptance scan matches exhaustive prefix, suffix and interval choices with exceptions", () => {
  for (let sample = 0; sample < 60; sample++) {
    const rows = Array.from({ length: 7 }, (_, i) => ({ accepted: (sample + i) % 13 ? Math.sin(sample * 7 + i * 3) : -Infinity,
      rejected: (sample + 2 * i) % 17 ? Math.cos(sample * 2 + i * 5) : -Infinity, free: (sample + i) % 4 === 0 }));
    for (const direction of ["prefix", "suffix", "interval"] as const) {
      let exact = -Infinity;
      for (let mask = 0; mask < 1 << rows.length; mask++) {
        const bits = rows.flatMap((row, i) => row.free ? [] : [mask >> i & 1]).join("");
        const pattern = direction === "prefix" ? /^1*0*$/ : direction === "suffix" ? /^0*1*$/ : /^0*1*0*$/;
        if (!pattern.test(bits)) continue;
        const value = rows.reduce((s, row, i) => s + ((mask >> i & 1) ? row.accepted : row.rejected), 0);
        exact = Math.max(exact, value);
      }
      const actual = maximizeEventAcceptanceSequence(rows, direction);
      if (Number.isFinite(exact)) assert.ok(Math.abs(actual - exact) < 1e-12);
      else assert.equal(actual, exact);
    }
  }
});

test("balance-box groups cannot use future returns to split identical available openings", () => {
  const costs = { ...DEFAULT_EVENT_COSTS, feeBps: 0, slippageBps: 0, maxLeverage: 1,
    quantityStep: .01, minQuantity: .01, minNotional: 0, maxNotional: 200,
    longBorrowBpsPerDay: 0, shortBorrowBpsPerDay: 0 };
  const kernel = [11, 9].map(close => ({ probability: .5, path: summarizeEventExecutionPath([
    { openTime: 0, open: 10, close: 10, low: 10, high: 10, volume: 1 },
    { openTime: 1000, open: 10, close, low: Math.min(10, close), high: Math.max(10, close), volume: 1 },
  ], 0, 1, costs) }));
  const box = { price: 10, cash: [99.9, 100.1] as const, quantity: [-.01, .01] as const, quantityLattice: true,
    balanceVertices: [[99.9, .01], [100.1, -.01]] as const };
  const bound = prepareEventExecutionBoxUpper(kernel)(box);
  assert.ok(bound.upperLogEquity - Math.log(100) < .003, "Identical opens cannot accept only the profitable outcome");
  const closed = structuredClone(kernel); closed[1].path.openingAvailable = false;
  const asymmetric = prepareEventExecutionBoxUpper(closed)(box);
  assert.ok(asymmetric.upperLogEquity - Math.log(100) > .04, "Known availability remains part of the execution law");
});

test("balance-box grouped search matches exhaustive relaxed acceptance choices on each request lot", () => {
  // At fixed positive cash both cap inequalities increase with inventory.
  // Endpoints therefore give exact possible/certain acceptance here. The upper
  // inventory is also the financial dominator, so the original path evaluator
  // independently scores both permitted group branches without copying the
  // optimizer's affine geometry or crossover search.
  for (let sample = 0; sample < 18; sample++) {
    const costs = { ...DEFAULT_EVENT_COSTS, feeBps: 13, slippageBps: 0, maxLeverage: 2,
      quantityStep: .25, minQuantity: .25, minNotional: sample % 2 ? 5 : 0, maxNotional: 70,
      longBorrowBpsPerDay: 14400, shortBorrowBpsPerDay: 86400 };
    const cash = [25, 47, 100][sample % 3];
    const lowAccount = { equity: cash - 10, price: 10, exposure: -10 / (cash - 10) };
    const highAccount = { equity: cash + 30, price: 10, exposure: 30 / (cash + 30) };
    const groups = [9.5, 10, 10.5].map((open, i) => [.4, .6].map((p, j) => {
      const close = open * (1 + Math.sin(sample * 2 + i * 3 + j * 5) * .18);
      return { probability: p / 3, path: summarizeEventExecutionPath([
        { openTime: 0, open: 10, close: 10, low: 10, high: 10, volume: 1 },
        { openTime: 1000, open, close, low: Math.min(open, close), high: Math.max(open, close), volume: 1 },
      ], 0, 1, costs) };
    }));
    let exact = -Infinity, orderedExact = -Infinity;
    for (let lot = -34; lot <= 34; lot++) {
      const request = lot * .25;
      let value = 0;
      const choices = [];
      for (const group of groups) {
        const possible = evaluateEventExecutionPath(group[0].path, highAccount, request).filledQuantity !== 0;
        const certain = evaluateEventExecutionPath(group[0].path, lowAccount, request).filledQuantity !== 0;
        const trade = possible ? group.reduce((s, a) => s + a.probability
          * Math.log(evaluateEventExecutionPath(a.path, highAccount, request).equity), 0) : -Infinity;
        const hold = certain ? -Infinity : group.reduce((s, a) => s + a.probability
          * Math.log(evaluateEventExecutionPath(a.path, highAccount, 0).equity), 0);
        value += Math.max(trade, hold);
        const turnover = Math.abs(request) * 10 * group[0].path.openRatio;
        const eligible = !!request && Math.abs(request) >= costs.minQuantity - 1e-12
          && turnover >= costs.minNotional - 1e-8 && turnover <= costs.maxNotional + 1e-8;
        choices.push({ trade, hold, eligible });
      }
      exact = Math.max(exact, value);
      const eligible = choices.filter(r => r.eligible), fixed = choices.filter(r => !r.eligible).reduce((s, r) => s + r.hold, 0);
      for (let cutoff = 0; cutoff <= eligible.length; cutoff++) orderedExact = Math.max(orderedExact,
        fixed + eligible.reduce((s, r, i) => s + (i < cutoff ? r.trade : r.hold), 0));
    }
    const bound = prepareEventExecutionBoxUpper(groups.flat())({ price: 10, cash: [cash, cash], quantity: [-1, 3], quantityLattice: true });
    assert.ok(bound.upperLogEquity >= exact - 1e-10 && bound.upperLogEquity < exact + 1e-7,
      `sample=${sample}, upper=${bound.upperLogEquity}, exact relaxed=${exact}`);
    const ordered = prepareEventExecutionBoxUpper(groups.flat(), { ordered: true, valueTolerance: 0 })({
      price: 10, cash: [cash, cash], quantity: [-1, 3], quantityLattice: true });
    assert.equal(ordered.complete, true);
    assert.ok(ordered.upperLogEquity >= orderedExact - 1e-10 && ordered.upperLogEquity < orderedExact + 1e-7,
      `sample=${sample}, upper=${ordered.upperLogEquity}, exact ordered=${orderedExact}`);
  }
});

test("ordered box subdivision budgets preserve upper bounds without claiming an exact maximum", () => {
  const costs = { ...DEFAULT_EVENT_COSTS, feeBps: 0, slippageBps: 0, maxLeverage: 3,
    quantityStep: .1, minQuantity: .1, minNotional: 0, maxNotional: 500,
    longBorrowBpsPerDay: 0, shortBorrowBpsPerDay: 0 };
  const kernel = [11, 9].map((close, i) => ({ probability: i ? .4 : .6, path: summarizeEventExecutionPath([
    { openTime: 0, open: 10, close: 10, low: 10, high: 10, volume: 1 },
    { openTime: 1000, open: 10, close, low: Math.min(10, close), high: Math.max(10, close), volume: 1 },
  ], 0, 1, costs) }));
  const box = { price: 10, cash: [100, 100] as const, quantity: [0, 0] as const, quantityLattice: true };
  const partial = prepareEventExecutionBoxUpper(kernel, { ordered: true, valueTolerance: 0, maxRefinements: 0 })(box);
  const full = prepareEventExecutionBoxUpper(kernel, { ordered: true, valueTolerance: 0 })(box);
  assert.equal(partial.complete, false); assert.equal(partial.certified, false);
  assert.equal(full.complete, true); assert.ok(full.search.refinements > 0);
  assert.ok(partial.upperLogEquity >= full.upperLogEquity - 1e-10);
  assert.ok(partial.relaxedLowerLogEquity! <= full.relaxedLowerLogEquity! + 1e-10);
  assert.ok(partial.relaxedGap! > 0);
});

test("balance-box recovery bounds cover nonmonotone acceptance and unavailable openings", () => {
  const costs = { ...DEFAULT_EVENT_COSTS, feeBps: 0, slippageBps: 0, maxLeverage: 5,
    quantityStep: .5, minQuantity: .5, minNotional: 0, maxNotional: 36,
    longBorrowBpsPerDay: 0, shortBorrowBpsPerDay: 0 };
  const p = summarizeEventExecutionPath([
    { openTime: 0, open: 1, close: 1, low: 1, high: 1, volume: 1 },
    { openTime: 1000, open: 1, close: 1.02, low: 1, high: 1.02, volume: 1 },
  ], 0, 1, costs), kernel = [{ probability: 1, path: p }];
  const before = { equity: 1, price: 1, exposure: -20 }, richer = { equity: 3.5, price: 1, exposure: -17.5 / 3.5 };
  assert.equal(evaluateEventExecutionPath(p, before, 36).filledQuantity, 36);
  assert.equal(evaluateEventExecutionPath(p, richer, 36).filledQuantity, 0);
  const solve = prepareEventExecutionBoxUpper(kernel), box = { price: 1, cash: [21, 21] as const,
    quantity: [-20, -17.5] as const, quantityLattice: true };
  for (let q = -20; q <= -17.5; q += .5) {
    const account = { equity: 21 + q, price: 1, exposure: q / (21 + q) };
    const value = prepareEventExecutionOneStep(kernel)(account).value + Math.log(account.equity);
    assert.ok(solve(box).upperLogEquity >= value - 1e-10);
  }
  const closed = prepareEventExecutionBoxUpper([{ probability: 1, path: { ...p, openingAvailable: false } }])(box);
  assert.equal(closed.search.maximumLots, 0);
  assert.ok(Math.abs(closed.upperLogEquity - Math.log(21 - 17.5 * 1.02)) < 1e-8);
  assert.throws(() => solve({ ...box, quantity: [-20.1, -17.5] }), /lattice/);
  assert.throws(() => solve({ ...box, cash: [22, 21] }), /Invalid/);
  assert.throws(() => solve({ ...box, balanceVertices: [[0, 0]] }), /vertices/);
  assert.equal(solve({ price: 1, cash: [-21, -21], quantity: [20, 20] }).upperLogEquity, -Infinity);
  const overflow = solve({ price: 1e308, cash: [1e308, 1e308], quantity: [2, 2] });
  assert.equal(overflow.complete, false); assert.equal(overflow.upperLogEquity, Infinity);
});

test("execution search retains no-trade above the entry cap and rejects inconsistent path costs", () => {
  const costs = { ...DEFAULT_EVENT_COSTS, maxLeverage: 1, maxNotional: 5, minNotional: 5,
    quantityStep: .1, minQuantity: .1, longBorrowBpsPerDay: 0 };
  const bars = [{ openTime: 0, open: 10, high: 10, low: 10, close: 10, volume: 1 },
    { openTime: 1000, open: 10, high: 11, low: 10, close: 11, volume: 1 }];
  const path = summarizeEventExecutionPath(bars, 0, 1, costs), kernel = [{ probability: 1, path }];
  const account = { equity: 100, price: 10, exposure: 1.1 };
  const result = prepareEventExecutionOneStep(kernel)(account);
  assert.equal(result.quantity, 0); assert.ok(result.value > 0);
  const closed = prepareEventExecutionOneStep([{ probability: 1, path: { ...path, openingAvailable: false } }])(account);
  assert.equal(closed.quantity, 0); assert.equal(closed.value, result.value); assert.equal(closed.search.evaluatedOrders, 1);
  assert.throws(() => prepareEventExecutionOneStep([{ probability: .5, path }, { probability: .5,
    path: { ...path, costs: { ...costs, feeBps: 9 } } }]), /inconsistent/);
});

test("execution-law replay freezes the request before the next open and verifies its original forecast", () => {
  const costs = { ...DEFAULT_EVENT_COSTS, maxLeverage: 1 }, training = seconds().slice(0, 4);
  training[3] = { ...training[3], close: 101, high: 101 };
  const path = summarizeEventExecutionPath(training, 0, 3, costs), m = secondModel();
  m.kernels = [[{ probability: 1, next: 0, return: path.closeRatio - 1, low: Math.min(1, path.lowRatio) - 1,
    high: Math.max(1, path.highRatio) - 1, duration: path.seconds / 60 }]];
  const policy = buildEventPolicy(m, costs, { depths: 1, referenceEquity: 10000, referencePrice: 100 });
  const executionOneStep = [[{ probability: 1, next: 0, path }]], c = seconds();
  const original = replayEventPolicy(c, policy, 64000, 67000, 1, { trace: true, oneStepTerminal: "marked", executionOneStep });
  c[64] = { ...c[64], open: 110, high: 110 };
  const changed = replayEventPolicy(c, policy, 64000, 67000, 1, { trace: true, oneStepTerminal: "marked", executionOneStep });
  assert.deepEqual(original.trace[0].order, changed.trace[0].order);
  assert.equal(original.trace[0].optimizer, "execution-one-event-lots");
  assert.equal(original.canceledOrders, 0); assert.equal(changed.canceledOrders, 1);
  assert.equal(changed.finalEquity, 10000); assert.equal(changed.positionSummary!.created, 0);
  assert.throws(() => replayEventPolicy(c, policy, 64000, 67000, 1, { oneStepTerminal: "marked",
    executionOneStep: [[{ probability: 1, next: 0, path: { ...path, closeRatio: 1.02 } }]] }), /changed the frozen/);
  assert.throws(() => replayEventPolicy(c, policy, 64000, 67000, 1,
    { oneStepTerminal: "marked", executionOneStep, quoteEntries: true }), /base requests/);
});

test("fit periods use the latest admissible whole-day block including feature history", () => {
  const day = 86400000, start = Date.parse("2022-06-12T00:00:00Z"), history = 14401000;
  const conflict = { id: "earlier-inspector", startTime: start - 5 * day, endTime: start + 2 * day };
  const future = { id: "target", startTime: start, endTime: start + 7 * day };
  const plain = eventFitPeriods(start, 4, history, [future]);
  assert.equal(plain.calibrationEnd, start); assert.equal(plain.fitStart, start - 5 * day); assert.deepEqual(plain.shifts, []);
  const shifted = eventFitPeriods(start, 4, history, [conflict, future]);
  assert.equal(shifted.calibrationEnd, conflict.startTime); assert.equal(shifted.testGapDays, 5);
  assert.equal(shifted.fitEnd - shifted.fitStart, 4 * day);
  assert.equal(shifted.calibrationEnd - shifted.calibrationStart, day);
  assert.equal(shifted.sourceStart, shifted.fitStart - history);
  assert.ok([conflict, future].every(r => r.startTime >= shifted.calibrationEnd || r.endTime <= shifted.sourceStart));
  // Even an exclusion confined to warmup must move the block.
  const historyOnly = { id: "old-feature-only", startTime: plain.sourceStart + 1000, endTime: plain.fitStart - 1000 };
  const older = eventFitPeriods(start, 4, history, [future, historyOnly]);
  assert.equal(older.calibrationEnd, Math.floor(historyOnly.startTime / day) * day);
  assert.equal(older.shifts[0].excludedIds[0], historyOnly.id);
  // A non-midnight exclusion moves to the preceding whole-day boundary.
  const halfDay = { id: "partial-day", startTime: start - day / 2, endTime: start + day };
  assert.equal(eventFitPeriods(start, 4, history, [halfDay]).calibrationEnd, start - day);
  const sevenDayCalibration = eventFitPeriods(start, 4, history, [future], 7);
  assert.equal(sevenDayCalibration.fitEnd, start - 7 * day);
  assert.equal(sevenDayCalibration.calibrationStart, start - 7 * day);
  assert.equal(sevenDayCalibration.calibrationEnd, start);
  assert.throws(() => eventFitPeriods(start, 4, history, [], 0), /periods/);
  assert.throws(() => eventFitPeriods(start + 1, 4, history, []), /periods/);
});

test("native fitting can retain unscored fit windows while excluding every scored window", () => {
  const catalog = [
    { id: "fit-full", startTime: 0, endTime: 10 },
    { id: "sideways-test", startTime: 10, endTime: 20 },
    { id: "latest", startTime: 20, endTime: 30 },
  ];
  assert.deepEqual(eventFittingExclusions(catalog, "all-catalog").map(row => row.id), ["fit-full", "sideways-test"]);
  assert.deepEqual(eventFittingExclusions(catalog, "non-fit").map(row => row.id), ["sideways-test"]);
  assert.throws(() => eventFittingExclusions(catalog, "bad" as any), /exclusion mode/);
});

test("native source blocks merge overlaps while omitted gaps cannot become returns or features", () => {
  const day = 86400000, ranges = [{ start: day * 5, end: day * 6 }, { start: 0, end: day }, { start: day, end: 2 * day }];
  assert.deepEqual(mergeEventSourceRanges(ranges), [{ start: 0, end: 2 * day }, { start: 5 * day, end: 6 * day }]);
  assert.deepEqual(eventSourceDays(ranges), [0, day, 5 * day]);
  assert.equal(ranges[0].start, 5 * day);
  const clock = { ...secondModel().clock, maxCandles: 1 }, original = seconds();
  const later = original.slice(100).map(c => ({ ...c, openTime: c.openTime + day }));
  const joined = [...original.slice(0, 100), ...later];
  assert.equal(observeMove(joined, 99, clock, NATIVE_SECOND_EVENT_FEATURES), null);
  assert.throws(() => nativeSecondEventFeatures(joined, 100), /contiguous/);
  const start = later[63].openTime + 1000, end = later.at(-1)!.openTime + 1000;
  const physical = (c: EventCandle[]) => makeSamples(c, clock, start, end, [], 1, "chain", NATIVE_SECOND_EVENT_FEATURES)
    .map(s => [c[s.start].openTime, c[s.end].openTime, s.features, s.nextFeatures, s.return, s.duration]);
  assert.deepEqual(physical(joined), physical(later));
  assert.ok(physical(joined).length > 0);
});

test("native slow context uses exact historical endpoints and purges its full support", () => {
  const c = Array.from({ length: 14420 }, (_, i) => ({ openTime: i * 1000,
    open: 100 + i / 1000, high: 100 + i / 1000, low: 100 + i / 1000, close: 100 + i / 1000, volume: 1 }));
  const clock = { ...secondModel().clock, maxCandles: 1 }, i = 14400;
  const x = eventFeatures(c, i, NATIVE_SECOND_CONTEXT_FEATURES, clock);
  assert.deepEqual(x.slice(0, 12), nativeSecondEventFeatures(c, i));
  assert.deepEqual(x.slice(12), NATIVE_SECOND_CONTEXT_LAGS.map(lag => Math.log(c[i].close / c[i - lag].close) * 10000));
  c[i + 1].close = 1000;
  assert.deepEqual(nativeSecondContextFeatures(c, i), x);
  c[i + 1].close = c[i + 1].open;
  assert.throws(() => eventFeatures(c, i, NATIVE_SECOND_CONTEXT_FEATURES, model.clock), /interval/);
  assert.equal(makeSamples(c, clock, 14401000, 14403000, [], 1, "stride", NATIVE_SECOND_CONTEXT_FEATURES).length, 1);
  assert.equal(makeSamples(c, clock, 14401000, 14403000,
    [{ id: "old-feature-input", startTime: 1000, endTime: 2000 }], 1, "stride", NATIVE_SECOND_CONTEXT_FEATURES).length, 0);
  c[0].openTime = 1;
  assert.throws(() => nativeSecondContextFeatures(c, i), /endpoint/);
});

test("native multiscale volatility context is causal and uses exact rolling return variances", () => {
  let price = 100;
  const c = Array.from({ length: 14420 }, (_, i) => {
    if (i) price *= Math.exp(((i % 11) - 5) * 0.000002);
    return { openTime: i * 1000, open: price, high: price, low: price, close: price, volume: 1 };
  });
  const clock = { ...secondModel().clock, maxCandles: 1 }, i = 14400;
  const values = eventFeatures(c, i, NATIVE_SECOND_VOLATILITY_CONTEXT_FEATURES, clock);
  assert.equal(eventFeatureWarmup(clock, NATIVE_SECOND_VOLATILITY_CONTEXT_FEATURES), 14400);
  assert.deepEqual(values.slice(0, NATIVE_SECOND_CONTEXT_FEATURES.length), nativeSecondContextFeatures(c, i));
  const level = (window: number) => {
    let square = 0;
    for (let j = i - window + 1; j <= i; j++) square += Math.log(c[j].close / c[j - 1].close) ** 2;
    return Math.log1p(Math.sqrt(square / window) * 10000);
  };
  const anchor = level(3600);
  const expected = [anchor, level(900) - anchor, level(1800) - anchor, level(14400) - anchor];
  values.slice(-4).forEach((value, index) => assert.ok(Math.abs(value - expected[index]) < 1e-12));
  c[i + 1].close *= 2;
  assert.deepEqual(nativeSecondVolatilityContextFeatures(c, i), values);
  const broken = c.map(row => ({ ...row })); broken[i - 100].openTime++;
  assert.throws(() => nativeSecondVolatilityContextFeatures(broken, i), /missing seconds/);
});

test("native second features require their declared clock and cannot read future or cross missing seconds", () => {
  const c = seconds(), m = secondModel();
  const before = eventFeatures(c, 63, m.featureNames, m.clock);
  assert.equal(before.length, NATIVE_SECOND_EVENT_FEATURES.length);
  assert.deepEqual(before.slice(0, 4), [0, 0, 0, 0]);
  assert.equal(before[10], Math.log1p(63));
  c[64].close = 123;
  assert.deepEqual(eventFeatures(c, 63, m.featureNames, m.clock), before);
  assert.throws(() => eventFeatures(c, 63, EVENT_FEATURES, m.clock), /interval/);
  assert.throws(() => eventFeatures(c, 63, m.featureNames), /explicit second clock/);
  assert.throws(() => validateEventDistribution({ ...m, clock: model.clock }), /interval/);
  c[12].openTime++;
  assert.throws(() => nativeSecondEventFeatures(c, 63), /contiguous/);
});

test("native trade-flow context uses only the completed matching second", () => {
  const c = Array.from({ length: 14420 }, (_, i) => ({ openTime: i * 1000,
    open: 100 + i / 1000, high: 100 + i / 1000, low: 100 + i / 1000, close: 100 + i / 1000, volume: 1,
    nativeTradeFlow: { availableAt: i * 1000 + 1000, aggregateCountImbalance: i === 14400 ? -0.25 : 0,
      lastAggressorSide: i === 14400 ? -1 : 0, buyerSellerVwapGap: i === 14400 ? 0.0002 : 0 } }));
  const clock = { ...secondModel().clock, maxCandles: 1 }, i = 14400;
  const values = eventFeatures(c, i, NATIVE_SECOND_FLOW_CONTEXT_FEATURES, clock);
  assert.equal(eventFeatureWarmup(clock, NATIVE_SECOND_FLOW_CONTEXT_FEATURES), 14400);
  assert.deepEqual(values.slice(0, NATIVE_SECOND_CONTEXT_FEATURES.length), nativeSecondContextFeatures(c, i));
  assert.deepEqual(values.slice(-2), [-0.25, -1]);
  const vwapValues = eventFeatures(c, i, NATIVE_SECOND_FLOW_VWAP_CONTEXT_FEATURES, clock);
  assert.deepEqual(vwapValues.slice(-3), [-0.25, -1, 0.0002]);
  c[i + 1].nativeTradeFlow.aggregateCountImbalance = 1;
  assert.deepEqual(nativeSecondFlowContextFeatures(c, i), values);
  c[i].nativeTradeFlow.availableAt++;
  assert.throws(() => nativeSecondFlowContextFeatures(c, i), /trade flow/);
  c[i].nativeTradeFlow.availableAt--;
  c[i].nativeTradeFlow.aggregateCountImbalance = 2;
  assert.throws(() => eventFeatures(c, i, NATIVE_SECOND_FLOW_CONTEXT_FEATURES, clock), /trade flow/);
  c[i].nativeTradeFlow.aggregateCountImbalance = -0.25;
  c[i].nativeTradeFlow.buyerSellerVwapGap = 3;
  assert.throws(() => nativeSecondFlowVwapContextFeatures(c, i), /VWAP/);
  c[i].nativeTradeFlow.buyerSellerVwapGap = 0.0002;
  delete (c[i] as Partial<typeof c[number]>).nativeTradeFlow;
  assert.throws(() => eventFeatures(c, i, NATIVE_SECOND_FLOW_CONTEXT_FEATURES, clock), /trade flow/);
  assert.throws(() => eventFeatures(c, i, NATIVE_SECOND_FLOW_CONTEXT_FEATURES, model.clock), /interval/);
});

test("selected native sign features are causal and reproduce audited price and rich-flow transforms", () => {
  let price = 100;
  const c = Array.from({ length: 14420 }, (_, i) => {
    const logReturn = i ? ((i % 5) - 2) * 0.00001 : 0;
    price *= Math.exp(logReturn);
    return { openTime: i * 1000, open: price, high: price, low: price, close: price, volume: 1,
      nativeTradeFlow: { availableAt: i * 1000 + 1000, aggregateCountImbalance: 1 / 3,
        lastAggressorSide: 1, buyerSellerVwapGap: 0,
        aggressiveBuyQuoteVolume: 2, aggressiveSellQuoteVolume: 1,
        aggressiveBuyMaxAggregateQuantity: 2, aggressiveSellMaxAggregateQuantity: 1 } };
  });
  const clock = { ...secondModel().clock, maxCandles: 1 }, i = 14400;
  Object.assign(c[i - 1].nativeTradeFlow!, { lastAggressorSide: -1 });
  Object.assign(c[i].nativeTradeFlow!, { aggregateCountImbalance: -0.25, lastAggressorSide: 1,
    aggressiveBuyQuoteVolume: 6, aggressiveSellQuoteVolume: 2,
    aggressiveBuyMaxAggregateQuantity: 5, aggressiveSellMaxAggregateQuantity: 1 });
  const values = eventFeatures(c, i, NATIVE_SECOND_SELECTED_SIGN_FEATURES, clock);
  assert.equal(eventFeatureWarmup(clock, NATIVE_SECOND_SELECTED_SIGN_FEATURES), 14400);
  assert.deepEqual(values.slice(0, NATIVE_SECOND_CONTEXT_FEATURES.length), nativeSecondContextFeatures(c, i));
  const returns = Array.from({ length: 16 }, (_, offset) => Math.log(c[i - 15 + offset].close / c[i - 16 + offset].close));
  const expectedHaar = (returns[14] - returns[15])
    / (Math.SQRT2 * Math.sqrt(returns.reduce((sum, value) => sum + value * value, 0)));
  let buyEma = 2, sellEma = 1;
  buyEma += 2 / 3 * (6 - buyEma); sellEma += 2 / 3 * (2 - sellEma);
  assert.ok(Math.abs(values.at(-8)! - returns[14] * 10000) < 1e-10);
  assert.ok(Math.abs(values.at(-7)! - expectedHaar) < 1e-12);
  assert.deepEqual(values.slice(-6, -3), [-0.25, 1, -1]);
  assert.ok(Math.abs(values.at(-3)! - 0.5) < 1e-12);
  assert.ok(Math.abs(values.at(-1)! - (5 - 1) / (5 + 1)) < 1e-12);
  assert.ok(Math.abs(values.at(-2)! - (buyEma - sellEma) / (buyEma + sellEma)) < 1e-12);
  c[i + 1].close *= 2;
  assert.deepEqual(nativeSecondSelectedSignFeatures(c, i), values);
  delete c[i].nativeTradeFlow!.aggressiveBuyQuoteVolume;
  assert.throws(() => nativeSecondSelectedSignFeatures(c, i), /rich native trade flow/);
  c[i].nativeTradeFlow!.aggressiveBuyQuoteVolume = 6;
  c[i - 1].nativeTradeFlow!.availableAt++;
  assert.throws(() => nativeSecondSelectedSignFeatures(c, i), /trade flow/);
});

test("native day context preserves shorter features, rejects bad endpoints and purges the whole day", () => {
  const c = Array.from({ length: 86420 }, (_, i) => ({ openTime: i * 1000,
    open: 100 + i / 1000, high: 100 + i / 1000, low: 100 + i / 1000, close: 100 + i / 1000, volume: 1 }));
  const clock = { ...secondModel().clock, maxCandles: 1 }, i = 86400, names = NATIVE_SECOND_DAY_CONTEXT_FEATURES;
  const x = eventFeatures(c, i, names, clock);
  assert.equal(eventFeatureWarmup(clock, names), 86400);
  assert.deepEqual(x.slice(0, 19), nativeSecondContextFeatures(c, i));
  assert.deepEqual(x.slice(12), NATIVE_SECOND_DAY_CONTEXT_LAGS.map(lag => Math.log(c[i].close / c[i - lag].close) * 10000));
  c[i + 1].close = 1000;
  assert.deepEqual(nativeSecondDayContextFeatures(c, i), x);
  c[i + 1].close = c[i + 1].open;
  assert.throws(() => nativeSecondDayContextFeatures(c, i - 1), /one day/);
  assert.throws(() => eventFeatures(c, i, names, model.clock), /interval/);
  assert.equal(makeSamples(c, clock, 86401000, 86403000, [], 1, "stride", names).length, 1);
  assert.equal(makeSamples(c, clock, 86401000, 86403000,
    [{ id: "day-old-input", startTime: 1000, endTime: 2000 }], 1, "stride", names).length, 0);
  const saved = JSON.parse(JSON.stringify({ ...secondModel(), featureNames: names }));
  validateEventDistribution(saved);
  assert.deepEqual(eventFeatures(c, i, saved.featureNames, saved.clock), x);
  c[0].openTime = 1;
  assert.throws(() => nativeSecondDayContextFeatures(c, i), /endpoint/);
  c[0].openTime = 0; c[43200].close = NaN;
  assert.throws(() => nativeSecondDayContextFeatures(c, i), /endpoint/);
});

test("native run boundaries include the revealing candle and keep physical durations and censored tails", () => {
  const c = seconds(), clock = { ...secondModel().clock, maxCandles: 100, runClock: true };
  c[66] = { ...c[66], high: 101, close: 101 };
  const move = observeMove(c, 63, clock, NATIVE_SECOND_EVENT_FEATURES)!;
  assert.equal(move.end, 66); assert.equal(move.duration, 3 / 60);
  assert.ok(Math.abs(move.return - .01) < 1e-14);
  const relabeled = observeMove(c, 63, { ...clock, durationBinsMinutes: [1 / 60, 2 / 60] }, NATIVE_SECOND_EVENT_FEATURES)!;
  assert.equal(relabeled.label, 14);
  assert.deepEqual({ ...relabeled, label: move.label }, move);
  assert.equal(eventMoveLabel(0, 6 / 60, { thresholdBps: 20, maxCandles: 100, candleIntervalMs: 1000 }), 7);
  assert.equal(eventMoveLabel(0, 21 / 60, { thresholdBps: 20, maxCandles: 100, candleIntervalMs: 1000 }), 8);
  assert.equal(eventMoveLabel(0, 6 / 60, { thresholdBps: 20, maxCandles: 100 }), 6);
  assert.equal(observeMove(c.slice(0, 66), 63, clock, NATIVE_SECOND_EVENT_FEATURES), null);
  const samples = makeSamples(c, clock, 64000, 70000, [], 1, "chain", NATIVE_SECOND_EVENT_FEATURES);
  assert.equal(samples[0].end, 66);
  assert.ok(samples.every(s => c[s.end].openTime + 1000 < 70000));
  const excluded = [{ id: "support", startTime: 1000, endTime: 2000 }];
  assert.equal(makeSamples(c, clock, 64000, 68000, [], 1, "chain", NATIVE_SECOND_EVENT_FEATURES).length, 1);
  assert.equal(makeSamples(c, clock, 64000, 68000, excluded, 1, "chain", NATIVE_SECOND_EVENT_FEATURES).length, 0);
  c[65].openTime++;
  assert.equal(observeMove(c, 63, clock, NATIVE_SECOND_EVENT_FEATURES), null);
});

test("native replay charges one minute of borrowing over 60 seconds and reconciles initial positions", () => {
  const c = seconds(), m = secondModel(), costs = { ...DEFAULT_EVENT_COSTS, maxLeverage: 2,
    feeBps: 0, slippageBps: 0, shortBorrowBpsPerDay: 1440 };
  const p = buildEventPolicy(m, costs, { depths: 1, referenceEquity: 10000, referencePrice: 100 });
  const r = replayEventPolicy(c, p, 64000, 124000, 1,
    { trace: true, oneStepTerminal: "marked", initialQuantity: -10, cash: true });
  assert.ok(Math.abs(r.borrow - .1) < 1e-8);
  assert.ok(Math.abs(r.finalEquity - 9999.9) < 1e-8);
  assert.ok(Math.abs(r.shortMinutes - 1) < 1e-12);
  assert.equal(r.trace[0].time, 64000); assert.equal(r.trace[0].endTime, 67000);
  assert.equal(r.trace.at(-1)!.endTime, 124000);
  assert.equal(r.positionSummary!.active, 0);
});

test("native next-open fills cannot alter the preceding close's decision or forecast", () => {
  const c = seconds(), m = secondModel(), p = buildEventPolicy(m,
    { ...DEFAULT_EVENT_COSTS, maxLeverage: 1 }, { depths: 1, referenceEquity: 10000, referencePrice: 100 });
  const original = replayEventPolicy(c, p, 64000, 67000, 1, { trace: true, oneStepTerminal: "marked" });
  c[64] = { ...c[64], open: 110, high: 110, low: 100 };
  const changed = replayEventPolicy(c, p, 64000, 67000, 1, { trace: true, oneStepTerminal: "marked" });
  assert.deepEqual(original.trace[0].order, changed.trace[0].order);
  assert.equal(original.canceledOrders, 0); assert.equal(changed.canceledOrders, 1);
  assert.equal(changed.positionSummary!.created, 0);
  assert.equal(changed.finalEquity, 10000);
});

test("paired controller payoff charges one settlement and chooses before the second return", () => {
  const base = buildEventPolicy(model, { ...DEFAULT_EVENT_COSTS, maxLeverage: 1, feeBps: 10, slippageBps: 0 },
    { depths: 1, referenceEquity: 10000, referencePrice: 100 });
  const step = { features: [0], nextFeatures: [0], leaf: 0, nextLeaf: 0, move: model.kernels[0][0] };
  const p = trainEventFittedValue(base, [step, step], .1, 1), account = { equity: 10000, price: 100, exposure: 1 };
  const up = { return: .1, low: 0, high: .1, duration: 1 }, down = { return: -.1, low: -.1, high: 0, duration: 1 };
  const grid = p.tables[0].coefficients.map((_, i) => -.01 * Math.abs(p.exposures[i % p.exposures.length]));
  const hold = eventControllerHolding(p, account, up, down, 0), sampled = eventControllerHolding(p, account, up, down, 0, grid);
  assert.equal(hold.trade!.quantity, 0);
  assert.ok(Math.abs(hold.value - Math.log(.99 * .999)) < 1e-12);
  assert.ok(Math.abs(hold.value - eventPathHolding(1, [up, down], p.costs)[1].value) < 1e-12);
  assert.ok(Math.abs(sampled.trade!.exposure) < 1e-12); assert.ok(Math.abs(sampled.value - Math.log(1.1 * .999)) < 1e-12);
  assert.deepEqual(sampled.trade, eventControllerHolding(p, account, up, up, 0, grid).trade);
  const ruin = eventControllerHolding(p, { ...account, exposure: -1 }, { ...up, return: 2, high: 2 }, down, 0, grid);
  assert.equal(ruin.value, -Infinity); assert.equal(ruin.liquidated, true);
});

test("local value averages share positive target-independent weights and bound extrapolation", () => {
  const design = [[0], [1], [10]], targets = [[-1, 4], [2, 1], [100, -7]];
  const predict = eventLocalValuePredictor(design, targets), result = predict([0.2], [2, 1]);
  assert.deepEqual(result[0].indices, [0, 1]); assert.deepEqual(result[0].values, [0.5, 2.5]);
  assert.deepEqual(result[1].values, [-1, 4]);
  assert.deepEqual(predict([100], [2])[0].values, [51, -3]);
  const changed = eventLocalValuePredictor(design, [[999, 0], [-100, 0], [1, 0]]);
  assert.deepEqual(changed([0.2], [2])[0].indices, result[0].indices);
  assert.deepEqual(predict([0.5], [1])[0].indices, [0]);
  targets[0][0] = 500; design[0][0] = 50;
  assert.deepEqual(predict([0.2], [2])[0].values, [0.5, 2.5]);
  assert.throws(() => predict([0.2], [0]), /query/);
  assert.throws(() => predict([0.2], [2, 2]), /query/);
  assert.throws(() => eventLocalValuePredictor([[0]], [[NaN]]), /training/);
});

test("policy-evaluation folds purge long event tails and input history, with strictly earlier past fits", () => {
  const training = Array.from({ length: 38 }, (_, i) => ({ start: i + 1, end: i + 2 }));
  const evaluation = [{ start: 6, end: 9 }, { start: 11, end: 15 }, { start: 19, end: 29 }, { start: 22, end: 25 }, { start: 31, end: 35 }];
  const folds = eventPolicyEvaluationFolds(training, evaluation, 0, 40, 4, 2);
  assert.deepEqual(folds.flatMap(f => f.test).sort((a, b) => a - b), [0, 1, 2, 3, 4]);
  assert.equal(folds[1].supportEnd, 29);
  assert.deepEqual(folds[0].past, []);
  assert.ok(folds[1].complement.includes(31));
  assert.ok(!folds[1].complement.includes(30)); // start 31 has history starting at held-out tail 29.
  assert.ok(!folds[1].complement.includes(27)); // nominal next block still contains the long test event.
  assert.ok(!folds[1].past.includes(6)); // label ending at the purged history boundary 8.
  for (const fold of folds) {
    for (const i of fold.complement) assert.ok(training[i].end < fold.supportStart || training[i].start - 2 > fold.supportEnd);
    for (const i of fold.past) assert.ok(training[i].end < fold.supportStart && fold.complement.includes(i));
  }
  assert.throws(() => eventPolicyEvaluationFolds(training, [{ start: 39, end: 40 }], 0, 40, 4, 2), /support/);
  assert.throws(() => eventPolicyEvaluationFolds(training, evaluation, 0, 40, 1, 2), /support/);
});

test("centered fitted basis removes absolute spread while preserving established input layouts", () => {
  const e = { price: [100, 2, 3], flow: [4, 5, 6] }, d = [7, 8, 9];
  assert.deepEqual(eventFittedFuturesInputs("spot", e), []);
  assert.deepEqual(eventFittedFuturesInputs("price", e), [100, 2, 3]);
  assert.deepEqual(eventFittedFuturesInputs("all", e), [100, 2, 3, 4, 5, 6]);
  assert.deepEqual(eventFittedFuturesInputs("deviation", e, d), [100, 2, 3, 7, 8, 9]);
  assert.deepEqual(eventFittedFuturesInputs("centered", e, d), [2, 3, 7, 8, 9]);
  assert.deepEqual(eventFittedFuturesInputs("centered", { ...e, price: [-100, 2, 3] }, d), [2, 3, 7, 8, 9]);
  assert.throws(() => eventFittedFuturesInputs("centered", e), /layout/);
});

test("ridge prediction attribution preserves the intercept and signed extrapolation weights", () => {
  const design = [[1, -1], [1, 0], [1, 1]], penalty = 0.1;
  const influence = eventRidgeInfluence(design, penalty), weights = influence([1, 0.5]);
  weights.forEach((v, i) => assert.ok(Math.abs(v - (1 / 3 + 0.5 * design[i][1] / (3 * (2 / 3 + penalty)))) < 1e-12));
  assert.ok(Math.abs(weights.reduce((s, v) => s + v, 0) - 1) < 1e-12);
  const extrapolated = influence([1, 5]); assert.ok(extrapolated.some(v => v < 0));
  assert.throws(() => influence([1]), /query/);
});

test("fixed-inventory paths mark exposure, settle once, and report infeasible continuation", () => {
  const costs = { ...DEFAULT_EVENT_COSTS, maxLeverage: 2, feeBps: 10, slippageBps: 2, longBorrowBpsPerDay: 0, shortBorrowBpsPerDay: 0 };
  const steps = [{ return: .1, low: 0, high: .1, duration: 10 }, { return: -.1, low: -.1, high: 0, duration: 20 }];
  const result = eventPathHolding(1.5, steps, costs);
  const factor = 1 + 1.5 * (.99 - 1), exposure = 1.5 * .99 / factor;
  assert.ok(Math.abs(result[1].factor - factor) < 1e-12);
  assert.ok(Math.abs(result[1].value - Math.log(factor * (1 - exposure * .0012))) < 1e-12);
  assert.deepEqual(eventPathHolding(1.5, steps.slice(0, 1), costs), result.slice(0, 1));
  const capped = eventPathHolding(-1, steps, { ...costs, maxLeverage: 1 });
  assert.equal(capped[0].capBreaches, 0); assert.equal(capped[1].capBreaches, 1);
  const borrow = eventPathHolding(-1, [{ return: 0, low: 0, high: 0, duration: 720 }, { return: 0, low: 0, high: 0, duration: 720 }],
    { ...costs, feeBps: 0, slippageBps: 0, shortBorrowBpsPerDay: 100 });
  assert.ok(Math.abs(borrow[1].factor - .99) < 1e-12);
  const ruin = eventPathHolding(2, [{ return: -.6, low: -.6, high: 0, duration: 1 }, steps[0]], costs);
  assert.ok(ruin.every(r => r.liquidated && r.value === -Infinity));
  assert.throws(() => eventPathHolding(1, [{ ...steps[0], low: .1 }], costs), /Invalid/);
});

test("multi-event targets compound only complete paths and cannot bridge omitted events", () => {
  const moves = [
    { start: 0, end: 2, return: 0.1, duration: 2 },
    { start: 2, end: 5, return: -0.1, duration: 3 },
    { start: 5, end: 6, return: 0.02, duration: 1 },
    { start: 6, end: 8, return: -0.03, duration: 2 },
  ];
  const paths = eventSignHorizonPaths(moves, 3);
  assert.equal(paths.length, 2);
  assert.deepEqual(paths[0].cumulativeMinutes, [2, 5, 6]);
  assert.ok(Math.abs(paths[0].cumulativeReturns[1] - (-0.01)) < 1e-12);
  assert.ok(Math.abs(paths[0].cumulativeReturns[2] - 0.0098) < 1e-12);
  assert.equal(paths[0].end, 6);
  assert.deepEqual(eventSignHorizonPaths(moves.slice(0, 3), 3), paths.slice(0, 1));
  assert.deepEqual(eventSignHorizonPaths([moves[0], moves[2], moves[3]], 3), []);
  assert.deepEqual(eventSignHorizonPaths(moves.slice(0, 2), 3), []);
  assert.throws(() => eventSignHorizonPaths(moves, 0), /Invalid/);
});

test("exact-second dynamics reproduce the documented indicators with fixed causal support", () => {
  const c = Array.from({ length: 80 }, (_, i) => ({ openTime: i * 1000, close: 100 + Math.sin(i) + i / 20 }));
  const definitions = buildSignalDefinitions().filter(d => ["ema-acceleration-2-1", "ema-slope-2-2", "rsi-2"].includes(d.id));
  const engine = new IndicatorEngine(definitions, c[0].close);
  for (let i = 1; i <= 63; i++) engine.update(c[i].close);
  const expected = engine.values(new Float64Array(definitions.length)), values = eventSecondDynamics(c, 63)!;
  const get = (id: string) => expected[definitions.findIndex(d => d.id === id)];
  assert.ok(Math.abs(values[1] - get("ema-acceleration-2-1")) < 1e-9);
  assert.ok(Math.abs(values[2] - get("ema-slope-2-2")) < 1e-9);
  assert.ok(Math.abs(values[3] - (get("rsi-2") / 50 - 1)) < 1e-12);
  c[64].close = 1000; assert.deepEqual(eventSecondDynamics(c, 63), values);
  assert.deepEqual(eventSecondDynamics(c, 79), eventSecondDynamics(c.slice(16), 63));
  const flat = c.map(r => ({ ...r, close: 100 })); assert.deepEqual(eventSecondDynamics(flat, 63), [0, 0, 0, 0]);
  c[20].openTime++; assert.equal(eventSecondDynamics(c, 63), null);
  assert.equal(eventSecondDynamics(c, 62), null);
  const minute = { openTime: 0, close: 100 }, rows = new Map([[60000, { availableAt: 60000, close: 100, values: [0, 0, 0, 0] }]]);
  assert.deepEqual(eventSecondDynamicsAt(rows, minute), [0, 0, 0, 0]);
  rows.get(60000)!.availableAt++; assert.throws(() => eventSecondDynamicsAt(rows, minute), /misaligned/);
});

test("futures event inputs use completed contiguous minutes and distinguish missing from zero flow", () => {
  const c = candles().slice(0, 10);
  const rows = new Map<number, SequentialDerivativesKlineRow>(c.map(c => [c.openTime, {
    openTime: c.openTime, open: 101, high: 101, low: 101, close: 101,
    baseVolume: 10, quoteVolume: 1000, tradeCount: 100, takerBuyBaseVolume: 7, takerBuyQuoteVolume: 700,
  }]));
  const result = eventFuturesFeatures(c, 5, t => rows.get(t))!;
  assert.ok(Math.abs(result.price[0] - Math.log(1.01) * 1e4) < 1e-12);
  assert.equal(result.price[1], 0); assert.equal(result.price[2], 0);
  assert.equal(result.flow[0], Math.log1p(100)); assert.ok(Math.abs(result.flow[1] - 0.4) < 1e-12);
  rows.get(c[6].openTime)!.close = 200;
  rows.get(c[6].openTime)!.takerBuyQuoteVolume = 0;
  assert.deepEqual(eventFuturesFeatures(c, 5, t => rows.get(t)), result);
  const missingTime = c[3].openTime, saved = rows.get(missingTime)!;
  rows.delete(missingTime); assert.equal(eventFuturesFeatures(c, 5, t => rows.get(t)), null);
  rows.set(missingTime, { ...saved, close: null }); assert.equal(eventFuturesFeatures(c, 5, t => rows.get(t)), null);
  rows.set(missingTime, saved);
  for (const r of rows.values()) { r.quoteVolume = 0; r.takerBuyQuoteVolume = 0; r.tradeCount = 0; }
  assert.deepEqual(eventFuturesFeatures(c, 5, t => rows.get(t))!.flow, [0, 0, 0]);
});

test("native-second futures inputs use only fully completed paired minutes", () => {
  const count = 241 * 60 + 31;
  const c: EventCandle[] = Array.from({ length: count }, (_, i) => ({ openTime: i * 1_000,
    open: 100, high: 100, low: 100, close: 100, volume: 1 }));
  const rows = new Map<number, SequentialDerivativesKlineRow>();
  for (let minute = 0; minute < 242; minute++) rows.set(minute * 60_000, {
    openTime: minute * 60_000, open: 101, high: 102, low: 100, close: 101,
    baseVolume: 10, quoteVolume: 1_000, tradeCount: 100,
    takerBuyBaseVolume: 6, takerBuyQuoteVolume: 600,
  });
  const index = count - 2, result = eventNativeSecondFuturesFeatures(c, index, time => rows.get(time))!;
  assert.equal(result.completedMinuteOpenTime, 240 * 60_000);
  assert.ok(Math.abs(result.price[0] - Math.log(1.01) * 1e4) < 1e-12);
  assert.deepEqual(result.price.slice(1), [0, 0]);
  assert.equal(result.flow[0], Math.log1p(100));
  assert.ok(Math.abs(result.flow[1] - .2) < 1e-12 && Math.abs(result.flow[2] - .2) < 1e-12);
  assert.ok(Math.abs(result.rangeBps - Math.log(1.02) * 1e4) < 1e-12);
  result.deviations.forEach(value => assert.ok(Math.abs(value) < 1e-12));
  rows.get(241 * 60_000)!.close = 999;
  c[index + 1].close = 999;
  assert.deepEqual(eventNativeSecondFuturesFeatures(c, index, time => rows.get(time)), result);
  const missing = rows.get(100 * 60_000)!;
  rows.delete(100 * 60_000);
  assert.equal(eventNativeSecondFuturesFeatures(c, index, time => rows.get(time)), null);
  rows.set(100 * 60_000, missing);
  c[100 * 60 + 59].openTime++;
  assert.equal(eventNativeSecondFuturesFeatures(c, index, time => rows.get(time)), null);
});

test("candle close locations use aligned completed OHLC and distinguish flat from invalid sources", () => {
  const c = candles().slice(0, 2);
  c[0] = { ...c[0], open: 100, high: 104, low: 96, close: 102 };
  const future = { ...c[0], close: 96 } as SequentialDerivativesKlineRow;
  const visited: number[] = [];
  assert.deepEqual(eventCandleShapes(c, 0, t => { visited.push(t); return future; }), [0.5, -1]);
  assert.deepEqual(visited, [c[0].openTime]);
  c[1] = { ...c[1], close: NaN, high: NaN };
  assert.deepEqual(eventCandleShapes(c, 0, () => future), [0.5, -1]);
  const scale = <T extends EventCandle>(r: T): T => ({ ...r, open: r.open * 10, high: r.high * 10, low: r.low * 10, close: r.close * 10 });
  assert.deepEqual(eventCandleShapes([scale(c[0])], 0, () => scale(future as EventCandle) as SequentialDerivativesKlineRow), [0.5, -1]);
  const flat = candles()[0];
  assert.deepEqual(eventCandleShapes([flat], 0, () => flat as SequentialDerivativesKlineRow), [0, 0]);
  assert.equal(eventCandleShapes(c, 0, () => undefined), null);
  assert.equal(eventCandleShapes(c, 0, () => ({ ...future, openTime: 60000 })), null);
  for (const change of [{ low: null }, { open: NaN }, { high: Infinity }, { low: 0 }, { low: 105 }, { high: 99 }, { close: 105 }, { open: 105 }])
    assert.equal(eventCandleShapes(c, 0, () => ({ ...future, ...change })), null);
  assert.equal(eventCandleShapes([{ ...c[0], low: 103 }], 0, () => future), null);
});

test("basis deviations use fixed completed history, remove constant offsets, and reject gaps", () => {
  const c = candles().slice(0, 300), rows = new Map(c.map(r => [r.openTime, { openTime: r.openTime, close: r.close } as SequentialDerivativesKlineRow]));
  assert.deepEqual(eventFuturesBasisDeviations(c, 239, t => rows.get(t)), [0, 0, 0]);
  rows.get(c[239].openTime)!.close = 100 * Math.exp(10 / 10000);
  const expected = [5, 15, 60].map(span => 10 * (1 - 2 / (span + 1))), values = eventFuturesBasisDeviations(c, 239, t => rows.get(t))!;
  values.forEach((v, i) => assert.ok(Math.abs(v - expected[i]) < 1e-9));
  rows.get(c[240].openTime)!.close = 1000;
  assert.deepEqual(eventFuturesBasisDeviations(c, 239, t => rows.get(t)), values);
  for (const r of rows.values()) r.close! *= Math.exp(50 / 10000);
  eventFuturesBasisDeviations(c, 239, t => rows.get(t))!.forEach((v, i) => assert.ok(Math.abs(v - expected[i]) < 1e-9));
  const full = eventFuturesBasisDeviations(c, 250, t => rows.get(t));
  assert.deepEqual(eventFuturesBasisDeviations(c.slice(10), 240, t => rows.get(t)), full);
  rows.delete(c[100].openTime); assert.equal(eventFuturesBasisDeviations(c, 239, t => rows.get(t)), null);
  assert.equal(eventFuturesBasisDeviations(c, 238, t => rows.get(t)), null);
});

test("martingale null uses symmetric arithmetic factors without Jensen drift or future price dependence", () => {
  const c = candles().slice(0, 3);
  c[1] = { ...c[1], open: 100, close: 200, low: 90, high: 220 };
  c[2] = { ...c[2], open: 200, close: 150, low: 140, high: 210 };
  const down = eventMartingaleCandles(c, 11), up = eventMartingaleCandles(c, 123456789);
  assert.ok(down[1].close < 100 && up[1].close > 100);
  assert.ok(Math.abs((down[1].close + up[1].close) / 2 - 100) < 1e-12);
  assert.ok(Math.abs(up[1].close / 100 - 1 - Math.tanh(Math.log(2))) < 1e-12);
  assert.deepEqual(eventMartingaleCandles(c, 11), down);
  for (const row of [...up, ...down]) assert.ok(row.low <= Math.min(row.open, row.close) && row.high >= Math.max(row.open, row.close));
  c[2].close = 300; c[2].high = 330;
  assert.deepEqual(eventMartingaleCandles(c, 11).slice(0, 2), down.slice(0, 2));
  assert.throws(() => eventMartingaleCandles(c, 0), /seed/);
  c[2].openTime += 60000; assert.throws(() => eventMartingaleCandles(c, 11), /Gap/);
});

for (const kind of ["sign", "sizeSign"] as const) test(`external ${kind} observations reproduce aligned inference and reject missing, future or stale values`, () => {
  const c = candles(), joint = { ...model, kernels: [[-0.02, -0.005, 0.005, 0.02].map(value => ({
    probability: 0.25, return: value, low: Math.min(0, value), high: Math.max(0, value), duration: 3, next: 0 }))] };
  const p = buildEventPolicy(joint, { ...DEFAULT_EVENT_COSTS, maxLeverage: 1 }, { depths: 2, referenceEquity: 10000, referencePrice: 100 });
  const head: EventSignHead = { means: new Array(12).fill(0), scales: new Array(12).fill(1), coefficients: new Array(12).fill(0),
    intercept: 1, penalty: 0.1, samples: 100, iterations: 1, loss: 0.5 };
  const sign = { head, blend: 1, lookahead: buildEventSignLookahead(p) }, start = c[1441].openTime, end = c[1450].openTime;
  if (kind === "sign") assert.throws(() => replayEventPolicy(c, p, start, end, 2,
    { sign: { ...sign, head: { ...head, objective: "weighted-sign" } } }), /inversion/);
  const sizeSign = { head: { gate: head, ordinarySign: head, largeSign: head, thresholdLogBps: 100, quantile: 0.75, samples: 100 },
    blend: 1, lookahead: buildEventOutcomeLookahead(p, 5, a => eventSizeSignGroup(a.return, 100)) };
  const standard = replayEventPolicy(c, p, start, end, 2, { ...(kind === "sign" ? { sign } : { sizeSign }), trace: true });
  const observations = new Map(standard.trace.map((r: any) => [r.time, { availableAt: r.time,
    values: eventFeatures(c, c.findIndex(v => v.openTime + 60000 === r.time), p.model.featureNames, p.model.clock) }]));
  const options = () => kind === "sign" ? { sign: { ...sign, observations } } : { sizeSign: { ...sizeSign, observations } };
  assert.deepEqual(replayEventPolicy(c, p, start, end, 2, { ...options(), trace: true }), standard);
  const first = observations.keys().next().value!, original = observations.get(first)!;
  observations.set(first, { ...original, availableAt: first + 1 });
  assert.throws(() => replayEventPolicy(c, p, start, end, 2, options()), /misaligned/);
  observations.set(first, { ...original, availableAt: first - 60000 });
  assert.throws(() => replayEventPolicy(c, p, start, end, 2, options()), /misaligned/);
  observations.delete(first);
  assert.throws(() => replayEventPolicy(c, p, start, end, 2, options()), /Missing/);
});

test("fitted replay uses aligned observations and its first order cannot see future candles", () => {
  const c = candles(), p = buildEventPolicy(model, { ...DEFAULT_EVENT_COSTS, maxLeverage: 1 },
    { depths: 2, referenceEquity: 10000, referencePrice: 100 });
  const policy = trainEventFittedValue(p, [0, 1].map(value => ({ features: [value], nextFeatures: [value],
    leaf: 0, nextLeaf: 0, move: model.kernels[0][0] })), 0.1, 2);
  const start = c[1441].openTime, end = c[1450].openTime;
  const observations = new Map(c.filter(r => r.openTime + 60000 >= start && r.openTime + 60000 < end)
    .map(r => [r.openTime + 60000, { availableAt: r.openTime + 60000, values: [0] }]));
  const replay = () => replayEventPolicy(c, p, start, end, 2, { fitted: { policy, observations }, trace: true });
  const standard = replay(), first = observations.keys().next().value!, original = observations.get(first)!;
  assert.deepEqual(standard.trace[0].order, decideFittedEvent(policy, [0], { equity: 10000, price: 100, exposure: 0 }, 2, 0));
  assert.ok(standard.trace[0].order.longEntry);
  for (let i = 1442; i < c.length; i++) c[i] = { ...c[i], open: 90, close: 90, low: 89, high: 91 };
  assert.deepEqual(replay().trace[0].order, standard.trace[0].order);
  observations.set(first, { ...original, availableAt: first + 1 });
  assert.throws(replay, /misaligned/);
  observations.set(first, { ...original, availableAt: first - 60000 });
  assert.throws(replay, /misaligned/);
  observations.delete(first); assert.throws(replay, /Missing/);
});

test("holding-option replay commits for completed events and replans without future observations", () => {
  const c = candles(), p = buildEventPolicy(model, { ...DEFAULT_EVENT_COSTS, maxLeverage: 1, feeBps: 1, slippageBps: 0 },
    { depths: 2, referenceEquity: 10000, referencePrice: 100 });
  const rows = [-1, 1].map(x => ({ features: [x], nextFeatures: [x], leaf: 0, nextLeaf: 0, move: model.kernels[0][0] }));
  const policy = trainEventFittedValue(p, rows, .1, 2, undefined, undefined,
    { following: rows.map(r => [r]), minimumTurnover: true });
  policy.tables[1].coefficients = policy.tables[1].coefficients.map((_, i) => [0, .02 * policy.exposures[i % policy.exposures.length]]);
  const start = c[1441].openTime, end = c[1460].openTime;
  const observations = new Map(c.filter(r => r.openTime + 60000 >= start && r.openTime + 60000 < end)
    .map(r => [r.openTime + 60000, { availableAt: r.openTime + 60000, values: [r.openTime + 60000 === start ? 1 : -1] }]));
  const run = () => replayEventPolicy(c, p, start, end, 2, { fitted: { policy, observations, replanEvery: 2 }, trace: true });
  const result = run();
  assert.ok(result.trace[0].order.longEntry); assert.equal(result.trace[0].optionRemaining, 2);
  assert.equal(result.trace[1].optionHolding, true); assert.equal(result.trace[1].order.quantity, 0);
  assert.equal(result.trace[2].optionHolding, false); assert.ok(result.trace[2].order.shortEntry);
  const resume = result.trace[1];
  const continued = replayEventPolicy(c, p, resume.time, end, 2, { equity: resume.equityBefore, initialQuantity: resume.previousQuantity,
    fitted: { policy, observations, replanEvery: 2, initialState: { remaining: resume.optionRemaining, controller: 0 } }, trace: true });
  assert.equal(JSON.stringify(continued.trace), JSON.stringify(result.trace.slice(1)));
  observations.get(result.trace[1].time)!.values = [5];
  assert.deepEqual(run().trace.slice(0, 2), result.trace.slice(0, 2));
  assert.throws(() => replayEventPolicy(c, p, start, end, 2, { fitted: { policy, observations, replanEvery: 3 } }), /holding-option/);
});

test("a sampled cash option advances to H1 and resumes with its original account and phase", () => {
  const c = candles(), p = buildEventPolicy(model, { ...DEFAULT_EVENT_COSTS, maxLeverage: 1, feeBps: 1, slippageBps: 0 },
    { depths: 2, referenceEquity: 10000, referencePrice: 100 });
  const step = { features: [0], nextFeatures: [0], leaf: 0, nextLeaf: 0, move: model.kernels[0][0] };
  const policy = trainEventFittedValue(p, [step, step], .1, 2, undefined, undefined,
    { following: [[step], [step]], minimumTurnover: true });
  const alternative = structuredClone(policy); alternative.targetMode = "sampled-path";
  policy.tables[1].coefficients = policy.tables[1].coefficients.map(() => [0, 0]);
  alternative.tables[1].coefficients = alternative.tables[1].coefficients.map((_, i) => [policy.exposures[i % policy.exposures.length] === 0 ? 1 : -1, 0]);
  const start = c[1441].openTime, end = c[1460].openTime;
  const observations = new Map(c.filter(r => r.openTime + 60000 >= start && r.openTime + 60000 < end)
    .map(r => [r.openTime + 60000, { availableAt: r.openTime + 60000, values: [0] }]));
  const result = replayEventPolicy(c, p, start, end, 2, { fitted: { policy, observations, replanEvery: 2, sampledAlternative: alternative }, trace: true });
  assert.equal(result.trace[0].optionController, "sampled"); assert.equal(result.trace[0].order.quantity, 0);
  assert.equal(result.trace[1].optionHolding, true); assert.equal(result.trace[1].optionRemaining, 1);
  assert.ok(result.trace[1].order.longEntry);
  assert.equal(result.trace[2].optionHolding, false);
  for (const offset of [1, 2]) {
    const first = result.trace[offset];
    const resumed = replayEventPolicy(c, p, first.time, end, 2, { equity: first.equityBefore, initialQuantity: first.previousQuantity,
      fitted: { policy, observations, replanEvery: 2, sampledAlternative: alternative,
        initialState: { remaining: first.optionHolding ? first.optionRemaining : 0, controller: first.optionController === "sampled" ? 1 : 0 } }, trace: true });
    assert.equal(JSON.stringify(resumed.trace), JSON.stringify(result.trace.slice(offset)));
  }
  const sampledOnly = replayEventPolicy(c, p, start, end, 2, { fitted: { policy: alternative, observations, replanEvery: 2 }, trace: true });
  assert.equal(sampledOnly.trace[0].order.quantity, 0); assert.equal(sampledOnly.trace[0].optionController, "sampled");
  assert.equal(sampledOnly.trace[1].optionHolding, true); assert.ok(sampledOnly.trace[1].order.longEntry);
  assert.equal(sampledOnly.trace[2].optionHolding, false);
  assert.throws(() => replayEventPolicy(c, p, start, end, 3, { fitted: { policy: alternative, observations, replanEvery: 3 } }), /holding-option/);
  const cashReplan = replayEventPolicy(c, p, start, end, 2,
    { fitted: { policy, observations, replanEvery: 2, sampledAlternative: alternative, replanCash: true }, trace: true });
  assert.equal(cashReplan.trace[1].optionHolding, false); assert.equal(cashReplan.trace[1].order.quantity, 0);
  assert.throws(() => replayEventPolicy(c, p, start, end, 2, { fitted: { policy, observations, replanCash: true } }), /Cash replanning/);
  assert.throws(() => replayEventPolicy(c, p, start, end, 2,
    { fitted: { policy, observations, replanEvery: 2, initialState: { remaining: 1, controller: 1 } } }), /initial option state/);
  assert.throws(() => replayEventPolicy(c, p, start, end, 2, { initialQuantity: NaN }), /initial replay account/);
});

test("rolling-origin selection covers disjoint earlier periods and cannot replace them with the best origin", () => {
  const testStart = Date.UTC(2023, 2, 18), origins = eventRefitOrigins(testStart, 3, 21);
  assert.equal(origins[0].startTime, Date.UTC(2023, 0, 14));
  assert.equal(origins.at(-1)!.endTime, testStart);
  assert.ok(origins.every((o, i) => o.startTime < o.endTime && o.endTime <= testStart && (!i || origins[i - 1].endTime === o.startTime)));
  const unstable = eventOriginScore([{ logGrowth: 0.2, maxDrawdownPct: 0 }, { logGrowth: -0.1, maxDrawdownPct: 10 }, { logGrowth: -0.1, maxDrawdownPct: 10 }], 0.1);
  const stable = eventOriginScore(origins.map(() => ({ logGrowth: 0.01, maxDrawdownPct: 1 })), 0.1);
  assert.ok(unstable.score < 0 && stable.score > 0);
  assert.equal(unstable.positiveOrigins, 1); assert.equal(stable.positiveOrigins, 3);
  assert.throws(() => eventRefitOrigins(testStart, 0, 21), /Invalid/);
  assert.throws(() => eventRefitOrigins(testStart, 3, -1), /Invalid/);
  assert.throws(() => eventOriginScore([], 0.1), /Invalid/);
});

test("scheduled joint-law policies cannot act before availability or change the account contract", () => {
  const c = candles(), costs = { ...DEFAULT_EVENT_COSTS, maxLeverage: 1, feeBps: 1, slippageBps: 0 };
  const options = { depths: 1, referenceEquity: 10_000, referencePrice: 100 };
  const up = buildEventPolicy(model, costs, options);
  const down = buildEventPolicy({ ...model, kernels: [[{ probability: 1, return: -0.01, low: -0.01, high: 0, duration: 3, next: 0 }]] }, costs, options);
  const start = c[1441].openTime, end = c[1450].openTime, at = start + 180_000;
  const fixed = replayEventPolicy(c, up, start, end, 1, { trace: true });
  const updated = replayEventPolicy(c, up, start, end, 1, { trace: true, updates: [{ at, policy: down }] });
  assert.deepEqual(updated.trace[0].order, fixed.trace[0].order);
  assert.equal(updated.trace[0].policyUpdatedAt, undefined);
  assert.equal(updated.trace[1].time, at); assert.equal(updated.trace[1].policyUpdatedAt, at);
  assert.equal(updated.trace[1].expectedReturnBps, -100);
  assert.equal((updated.trace[1].order as { shortEntry: boolean }).shortEntry, true);
  assert.throws(() => replayEventPolicy(c, up, start, end, 1,
    { updates: [{ at, policy: { ...down, costs: { ...costs, feeBps: 0 } } }] }), /contracts/);
  assert.throws(() => replayEventPolicy(c, up, start, end, 1,
    { updates: [{ at, policy: down }, { at: at - 1, policy: up }] }), /ordered/);
});

test("one-second cache joins by close availability and never forwards stale observations", () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "event-second-basis-"));
  try {
    fs.writeFileSync(path.join(directory, "manifest.json"), JSON.stringify({ datasets: [{ id: "1s", rows: 3,
      featureCount: 5, features: EVENT_SECOND_INPUTS.map(id => ({ id })), files: { times: "times.f64", features: "features.f32" } }] }));
    fs.writeFileSync(path.join(directory, "times.f64"), Buffer.from(new Float64Array([59_000, 60_000, 179_000]).buffer));
    fs.writeFileSync(path.join(directory, "features.f32"), Buffer.from(new Float32Array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]).buffer));
    const c = candles().slice(0, 4);
    new EventSecondBasis(directory).attach(c);
    assert.equal(c[0].secondBasis!.availableAt, 60_000);
    assert.deepEqual(c[0].secondBasis!.values, [1, 1, 1, 1, 1]);
    assert.equal(c[1].secondBasis!.availableAt, 61_000);
    assert.deepEqual(c[2].secondBasis!.values, [3, 3, 3, 3, 3]);
    assert.equal(c[3].secondBasis, undefined);
  } finally { fs.rmSync(directory, { recursive: true, force: true }); }
});

test("rolling event scale uses only completed pairs and reverses repeatedly wrong directional means", () => {
  const completed: Array<[number, number]> = [[0.01, -0.02], [0.01, -0.01], [0.01, -0.03], [0.01, -0.02]];
  assert.equal(rollingEventScale(completed.slice(0, 3), 8), 1);
  assert.equal(rollingEventScale(completed, 8), -2);
  assert.equal(rollingEventScale(completed.map(([x]) => [x, 0.02]), 8), 2);
  assert.equal(rollingEventScale([[0, 1], [0, -1], [0, 2], [0, -2]], 8), 0);
});

test("a boundary-truncated event is not inserted as a completed calibration outcome", () => {
  const c = candles();
  const p = buildEventPolicy(model, DEFAULT_EVENT_COSTS, { depths: 1, referenceEquity: 10_000, referencePrice: 100 });
  const result = replayEventPolicy(c, p, c[1441].openTime, c[1442].openTime, 1,
    { adaptive: { policies: [p], scales: [1], window: 8 } });
  assert.deepEqual(result.adaptationPairs, []);
});

test("terminal event replay needs no post-window candles and still rejects missing scored history", () => {
  const c = candles(), start = c[1441].openTime, end = c[1442].openTime;
  const p = buildEventPolicy(model, DEFAULT_EVENT_COSTS, { depths: 1, referenceEquity: 10_000, referencePrice: 100 });
  const options = { trace: true, adaptive: { policies: [p], scales: [1], window: 8 } };
  const full = replayEventPolicy(c, p, start, end, 1, options);
  assert.ok(full.trades > 0);
  assert.deepEqual(full.adaptationPairs, []);
  assert.deepEqual(replayEventPolicy(c.slice(0, 1442), p, start, end, 1, options), full);
  for (let i = 1442; i < c.length; i++) c[i] = { ...c[i], open: 10, high: 1000, low: 1, close: 500 };
  assert.deepEqual(replayEventPolicy(c, p, start, end, 1, options), full);
  assert.throws(() => replayEventPolicy(c.slice(0, 1441), p, start, end, 1, options), /Incomplete replay history/);
  const gap = candles().filter((_, i) => i !== 1442);
  assert.throws(() => replayEventPolicy(gap, p, start, c[1444].openTime, 1, options), /Gap in scored history/);
});

test("a censored sample chain cannot restart inside its unresolved final event", () => {
  const c = candles(), clock = { thresholdBps: 20, maxCandles: 10 };
  for (const [i, close] of [[1441, 100.18], [1442, 99.96], [1443, 100.1]])
    c[i] = { ...c[i], close, high: Math.max(100, close), low: Math.min(100, close) };
  const start = c[1441].openTime, end = c[1444].openTime;
  const full = makeSamples(c, clock, start, end, [], 1, "chain");
  assert.deepEqual(full, []);
  assert.deepEqual(makeSamples(c.slice(0, 1444), clock, start, end, [], 1, "chain"), full);
});

test("hidden replay updates only from observed events and its first order cannot see the future", () => {
  const laws = [[0.9, 0.1], [0.1, 0.9]];
  const emissions = [Array.from({ length: 15 }, (_, i) => Number(i === 0)), Array.from({ length: 15 }, (_, i) => Number(i === 12))];
  const hiddenModel: EventDistribution = { ...model, nodes: [], counts: [100, 100],
    classProbabilities: laws.map(row => emissions[0].map((_, i) => i === 0 ? row[0] : i === 12 ? row[1] : 0)),
    hidden: { transition: laws, emission: emissions, beliefs: [[1, 0], [0, 1]], initial: 0,
      nextByClass: laws.map(() => Array.from({ length: 15 }, (_, i) => Number(i === 12))),
      iterations: 1, sequences: 1, logLikelihood: 0, resolution: 1 },
    kernels: laws.map(row => row.map((probability, i) => ({ probability, return: i ? 0.01 : -0.01,
      low: i ? 0 : -0.01, high: i ? 0.01 : 0, duration: 1, next: i }))) };
  const p = buildEventPolicy(hiddenModel, { ...DEFAULT_EVENT_COSTS, maxLeverage: 1 },
    { depths: 2, referenceEquity: 10000, referencePrice: 100 });
  const replay = (change: number) => {
    const c = candles();
    for (let i = 1441; i < c.length; i++) c[i] = { ...c[i], open: i === 1441 ? 100 : 100 + change,
      high: Math.max(100, 100 + change), low: Math.min(100, 100 + change), close: 100 + change };
    return replayEventPolicy(c, p, c[1441].openTime, c[1444].openTime, 2, { trace: true });
  };
  const up = replay(1), down = replay(-1);
  assert.deepEqual(up.trace[0].order, down.trace[0].order);
  assert.equal(up.trace[0].leaf, 0); assert.equal(down.trace[0].leaf, 0);
  assert.equal(up.trace[1].leaf, 1); assert.equal(down.trace[1].leaf, 0);
});

test("inverted-chart augmentation rebuilds valid OHLC before making event labels", () => {
  const c = candles(); c[1441] = { ...c[1441], open: 100, high: 102, low: 99, close: 101 };
  const inverted = invertEventCandles(c);
  assert.equal(inverted[1441].high, 1 / 99); assert.equal(inverted[1441].low, 1 / 102);
  assert.equal(inverted[1441].close, 1 / 101);
  assert.ok(Math.abs(inverted[1441].close / inverted[1440].close - 1 + 0.01 / 1.01) < 1e-12);
});

test("training labels and input history are purged across evaluation boundaries", () => {
  const c = candles(), clock = { thresholdBps: 20, maxCandles: 3 };
  const start = 1440 * 60_000, end = 1490 * 60_000;
  const ordinary = makeSamples(c, clock, start, end, [], 1);
  assert.ok(ordinary.length > 0);
  assert.ok(ordinary.every(r => c[r.end].openTime + 60_000 < end));
  assert.equal(makeSamples(c, clock, start, end,
    [{ id: "holdout", startTime: 1400 * 60_000, endTime: 1430 * 60_000 }], 1).length, 0);
  assert.equal(overlaps(10, 20, [{ id: "range", startTime: 20, endTime: 30 }]), true);
});

test("a future opening gap is neither earned before entry nor used to resize the signal order", () => {
  const c = candles();
  for (let i = 1441; i < c.length; i++) c[i] = { ...c[i], open: 110, high: 110, low: 110, close: 110 };
  const p = buildEventPolicy(model, DEFAULT_EVENT_COSTS, { depths: 1, referenceEquity: 10_000, referencePrice: 100 });
  const result = replayEventPolicy(c, p, c[1441].openTime, c[1442].openTime, 1);
  assert.equal(result.canceledOrders, 1); assert.equal(result.trades, 0); assert.equal(result.returnPct, 0);
});

test("quote entries precommit turnover and earn only post-fill movement on either side", () => {
  for (const side of [1, -1]) {
    const law = { ...model, kernels: [[{ probability: 1, return: side * .01, low: Math.min(0, side * .01),
      high: Math.max(0, side * .01), duration: 3, next: 0 }]] };
    const costs = { ...DEFAULT_EVENT_COSTS, maxLeverage: 1, longBorrowBpsPerDay: 0, shortBorrowBpsPerDay: 0 };
    const p = buildEventPolicy(law, costs, { depths: 1, referenceEquity: 10_000, referencePrice: 100 });
    const runs = [90, 110].map(open => {
      const c = candles(); c[1441] = { ...c[1441], open, high: open + 1, low: open, close: open + 1 };
      const result = replayEventPolicy(c, p, c[1441].openTime, c[1442].openTime, 1, { quoteEntries: true, trace: true });
      const row = result.trace[0] as any, dq = row.orderQuantity;
      assert.equal(Math.sign(dq), side); assert.equal(result.canceledOrders, 0); assert.equal(result.trades, 2);
      assert.equal(row.quoteOrderQty, row.order.turnover);
      assert.ok(Math.abs(dq) * open <= row.quoteOrderQty + 1e-8);
      assert.ok(row.quoteOrderQty - Math.abs(dq) * open < costs.quantityStep * open + 1e-8);
      const feeRate = (costs.feeBps + costs.slippageBps) / 10000;
      assert.ok(Math.abs(dq * open / (10_000 - Math.abs(dq) * open * feeRate)) <= 1 + 1e-8);
      assert.ok(Math.abs(result.longPnl + result.shortPnl - dq) < 1e-8);
      assert.ok(Math.abs(result.finalEquity - (10_000 + dq - Math.abs(dq) * (2 * open + 1) * feeRate)) < 1e-8);
      const invested = { initialQuantity: side * 50, trace: true };
      assert.equal(JSON.stringify(replayEventPolicy(c, p, c[1441].openTime, c[1442].openTime, 1, invested)),
        JSON.stringify(replayEventPolicy(c, p, c[1441].openTime, c[1442].openTime, 1, { ...invested, quoteEntries: true })));
      return row;
    });
    assert.deepEqual(runs[0].order, runs[1].order); assert.equal(runs[0].quoteOrderQty, runs[1].quoteOrderQty);
    assert.notEqual(runs[0].orderQuantity, runs[1].orderQuantity);
  }
});

test("quote entry lot rounding still enforces minimum order size", () => {
  const c = candles(); c[1441] = { ...c[1441], open: 1e12, high: 1e12, low: 1e12, close: 1e12 };
  const p = buildEventPolicy(model, DEFAULT_EVENT_COSTS, { depths: 1, referenceEquity: 10_000, referencePrice: 100 });
  const result = replayEventPolicy(c, p, c[1441].openTime, c[1442].openTime, 1, { quoteEntries: true, trace: true });
  assert.equal(result.canceledOrders, 1); assert.equal(result.trades, 0); assert.equal(result.returnPct, 0);
  assert.ok(Number(result.trace[0].quoteOrderQty) > 0); assert.equal(result.trace[0].orderQuantity, 0);
});

test("one-event lot replay fixes the first order before future candles and rejects mixed horizons", () => {
  const first = candles(), second = candles();
  for (let i = 1441; i < second.length; i++) second[i] = { ...second[i], open: 110, high: 115, low: 105, close: 109 };
  const p = buildEventPolicy(model, DEFAULT_EVENT_COSTS, { depths: 2, referenceEquity: 10000, referencePrice: 100 });
  const start = first[1441].openTime, end = first[1443].openTime, options = { oneStepTerminal: "market" as const, trace: true };
  const a = replayEventPolicy(first, p, start, end, 1, options), b = replayEventPolicy(second, p, start, end, 1, options);
  assert.deepEqual(a.trace[0].order, b.trace[0].order); assert.equal(a.trace[0].optimizer, "one-event-lots");
  assert.throws(() => replayEventPolicy(first, p, start, end, 2, options), /Exact one-event/);
});

test("two-event replay fixes its first order and bound before observing future prices", () => {
  const first = candles(), second = candles();
  for (let i = 1441; i < second.length; i++) second[i] = { ...second[i], open: 110, high: 115, low: 105, close: 109 };
  const p = buildEventPolicy(model, DEFAULT_EVENT_COSTS, { depths: 2, referenceEquity: 10000, referencePrice: 100 });
  const start = first[1441].openTime, end = first[1443].openTime;
  const options = { twoStep: { terminal: "marked" as const, maxEvaluations: 16, globalUpper: true as const }, trace: true };
  const a = replayEventPolicy(first, p, start, end, 2, options), b = replayEventPolicy(second, p, start, end, 2, options);
  assert.deepEqual(a.trace[0].order, b.trace[0].order); assert.equal(a.trace[0].optimizer, "two-event-bounds");
  assert.equal(typeof (a.trace[0].order as { globalUpperValue: number }).globalUpperValue, "number");
  assert.throws(() => replayEventPolicy(first, p, start, end, 1, options), /Bounded two-event/);
  assert.throws(() => replayEventPolicy(first, p, start, end, 2, { ...options, oneStepTerminal: "marked" }), /Exact one-event/);
});

test("two-event countdown reacts to observed states, advances through cash and resumes its depth", () => {
  const c = candles(), p = buildEventPolicy(model, DEFAULT_EVENT_COSTS, { depths: 2, referenceEquity: 10000, referencePrice: 100 });
  const start = c[1441].openTime, end = c[1453].openTime;
  const twoStep = { terminal: "marked" as const, maxEvaluations: 16, countdown: true as const };
  const run = replayEventPolicy(c, p, start, end, 2, { twoStep, trace: true });
  assert.deepEqual(run.trace.map(r => r.decisionDepth), [2, 1, 2, 1]);
  for (const r of run.trace.filter(r => r.decisionDepth === 1)) {
    const order = r.order as { price: number; quantity: number };
    const expected = decideEventOneStep(p.model.kernels[r.leaf as number],
      { equity: r.equityBefore as number, price: order.price, exposure: r.exposureBefore as number }, p.costs, "marked");
    assert.deepEqual(r.order, expected);
  }
  const second = run.trace[1], resumed = replayEventPolicy(c, p, second.time as number, end, 2, { trace: true,
    equity: second.equityBefore as number, initialQuantity: second.previousQuantity as number, twoStep: { ...twoStep, initialDepth: 1 } });
  assert.deepEqual(resumed.trace, run.trace.slice(1));
  const quiet = structuredClone(p); quiet.model.kernels[0][0] = { ...quiet.model.kernels[0][0], return: 0, high: 0 };
  const cash = replayEventPolicy(c, quiet, start, end, 2, { trace: true, twoStep });
  assert.ok(cash.trace.every(r => (r.order as { quantity: number }).quantity === 0));
  assert.deepEqual(cash.trace.map(r => r.decisionDepth), [2, 1, 2, 1]);
  assert.throws(() => replayEventPolicy(c, p, start, end, 2, { twoStep: { terminal: "marked", maxEvaluations: 16, initialDepth: 1 } }), /countdown controller/);
});

test("three-event replay fixes its global bounds before future candles and rejects forecast mixing", () => {
  const first = candles(), second = candles();
  for (let i = 1441; i < second.length; i++) second[i] = { ...second[i], open: 110, high: 115, low: 105, close: 109 };
  const p = buildEventPolicy(model, { ...DEFAULT_EVENT_COSTS, maxLeverage: 1 }, { depths: 3, referenceEquity: 10000, referencePrice: 100 });
  const start = first[1441].openTime, end = first[1443].openTime;
  const options = { threeStep: { terminal: "marked" as const, maxEvaluations: 16, maxRootEvaluations: 1, shadowPoints: 9 }, trace: true };
  const a = replayEventPolicy(first, p, start, end, 3, options), b = replayEventPolicy(second, p, start, end, 3, options);
  assert.deepEqual(a.trace[0].order, b.trace[0].order); assert.equal(a.trace[0].optimizer, "three-event-bounds");
  assert.throws(() => replayEventPolicy(first, p, start, end, 2, options), /Bounded three-event/);
  assert.throws(() => replayEventPolicy(first, p, start, end, 3, { ...options, updates: [] }), /Bounded three-event/);
  assert.throws(() => replayEventPolicy(first, p, start, end, 3, { ...options, twoStep: { terminal: "marked", maxEvaluations: 16 } }), /Bounded two-event/);
});

test("simulator reconciles terminal cash, all fees, and long/short PnL", () => {
  const c = candles();
  for (let i = 1441; i < c.length; i++) {
    const open = 100 * 1.001 ** (i - 1441), close = open * 1.001;
    c[i] = { ...c[i], open, low: open, high: close, close };
  }
  const p = buildEventPolicy(model, DEFAULT_EVENT_COSTS, { depths: 2, referenceEquity: 10_000, referencePrice: 100 });
  const result = replayEventPolicy(c, p, c[1441].openTime, c[1480].openTime, 2);
  assert.ok(result.trades >= 2); assert.ok(result.fees > 0);
  assert.ok(Math.abs(result.finalEquity - (10_000 + result.longPnl + result.shortPnl - result.fees - result.borrow)) < 1e-7);
  assert.ok(Math.abs(result.daily.reduce((s, d) => s + d.logReturn, 0) - result.logGrowth) < 1e-9);
});

test("drawdown retains an intrabar equity peak and separately reports close-only drawdown", () => {
  const c = candles();
  c[1441] = { ...c[1441], open: 100, high: 120, low: 100, close: 110 };
  c[1442] = { ...c[1442], open: 110, high: 110, low: 100, close: 105 };
  const costs = { ...DEFAULT_EVENT_COSTS, maxLeverage: 1, feeBps: 0, slippageBps: 0, longBorrowBpsPerDay: 0, shortBorrowBpsPerDay: 0 };
  const p = buildEventPolicy(model, costs, { depths: 1, referenceEquity: 10_000, referencePrice: 100 });
  const result = replayEventPolicy(c, p, c[1441].openTime, c[1443].openTime, 1);
  assert.ok(Math.abs(result.maxDrawdownPct - (1 - 100 / 120) * 100) < 1e-6);
  assert.ok(Math.abs(result.closeDrawdownPct - (1 - 105 / 110) * 100) < 1e-6);
});
