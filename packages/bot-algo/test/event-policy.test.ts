import assert from "node:assert/strict";
import test from "node:test";
import { calibrateEventMean, canonicalEventRunFeatures, eventMoveLabel, EVENT_FEATURES, EVENT_SECOND_FEATURES, EVENT_PATH_FEATURES, EVENT_RUN_FEATURES, eventFeatures, eventRunState, eventLeaf, eventHiddenNext, distributionMetrics, observeMove, trainEventDistribution, trainEventForest, trainEventProjection, eventProjectionScore, trainEventBoost, eventBoostScore, type EventCandle,
  type EventDistribution, type MoveSample } from "../src/event-distribution.js";
import { buildEventPolicy, buildEventKernelLookahead, buildEventOutcomeLookahead, buildEventSignLookahead, chooseEventTrade, decideEvent, decideEventKernel, decideEventOutcomes, decideEventSign, DEFAULT_EVENT_COSTS, eventActionValues, eventHolding, eventTrade, serializeEventPolicy, restoreEventPolicy } from "../src/event-log-policy.js";
import { trainEventHidden } from "../src/event-hidden.js";
import { trainEventRunDistribution } from "../src/event-run-model.js";
import { EventRecentLaw, type CompletedEventMove } from "../src/event-recent-law.js";
import { eventProbabilityFromReturnWeight, eventSignMass, predictEventSign, reweightEventSigns, trainEventSign } from "../src/event-sign.js";
import { EventSizeGateCalibration, eventFastVolatilityFeatures, eventSizeSignGroup, eventSizeSignMass, mixEventSizeSigns, predictEventSizeSigns, projectEventSizeSigns, reweightEventSizeSigns, withEventSizeGate,
  trainEventSizeGate, trainEventSizeSign, selectEventSizeSignComponents } from "../src/event-size-sign.js";
import { EventCompletedHistory } from "../src/event-completed-history.js";
import { eventValueHeadMse, trainEventValueHead } from "../src/event-value-head.js";
import { eventSeverityMeans, predictEventSeverity, tiltEventSeverity, trainEventSeverity } from "../src/event-severity.js";
import { retainQuietEventLaw, trainEventVolatilityLaw } from "../src/event-volatility-law.js";
import { EVENT_RUN_VOLATILITY_FEATURES } from "../src/event-distribution.js";
import { compressEventKernel, eventQuadratureFeatures } from "../src/event-quadrature.js";
import { decideFittedEvent, decideFittedEventControllers, fittedEventHolding, predictEventHoldingGrid, trainEventFittedValue } from "../src/event-fitted-value.js";
import { predictEventValueBoost, trainEventValueBoost } from "../src/event-value-boost.js";
import { eventBellmanReference } from "./event-bellman-reference.js";
import { decideEventOneStep } from "../src/event-one-step.js";
import { decideEventTwoStep, prepareEventTwoStep } from "../src/event-two-step.js";
import { eventOneStepUpper } from "../src/event-one-step-upper.js";
import { prepareEventOneStep } from "../src/event-one-step-prepared.js";
import { prepareEventHoldingLaw } from "../src/event-holding-law.js";
import { prepareEventThreeStep, prepareEventThreeStepActions } from "../src/event-three-step.js";
import { EventPositionLedger } from "../src/event-positions.js";
import { eventCashHorizon } from "../src/event-cash-horizon.js";
import { eventMarginalCrps, prepareEventCrps } from "../src/event-crps.js";
import { eventAverageUniqueness } from "../src/event-sampling.js";
import { eventConformalResidualRadius, eventIntervalConfidence, eventRelativeUncertaintyConfidence,
  eventUncertaintyRiskFloor } from "../src/event-uncertainty.js";
import { reestimateEventTree, scaleEventMean, validateEventClock, type EventClock } from "../src/event-distribution.js";

test("event conformal residual radius uses the finite-sample corrected order statistic", () => {
  const rows = [5, 1, 3, 2, 8, 4].map(realized => ({ predicted: 0, realized }));
  assert.equal(eventConformalResidualRadius(rows, .5), 4);
  assert.equal(eventConformalResidualRadius(rows, .75), 8);
  assert.equal(eventConformalResidualRadius(rows, .9), Infinity);
  assert.throws(() => eventConformalResidualRadius([], .75), /nonempty/);
  assert.throws(() => eventConformalResidualRadius(rows, 1), /coverage/);
  assert.throws(() => eventConformalResidualRadius([{ predicted: 0, realized: NaN }], .5), /finite/);
});

test("forecast uncertainty continuously withdraws risk and protects accumulated profit", () => {
  assert.equal(eventIntervalConfidence(40, -10, 90), 0);
  assert.equal(eventIntervalConfidence(-40, -90, 10), 0);
  near(eventIntervalConfidence(40, 20, 60), .5);
  near(eventIntervalConfidence(.7, .6, .8, .5), .5);
  assert.equal(eventIntervalConfidence(40, 40, 40), 1);
  assert.throws(() => eventIntervalConfidence(1, 2, 3), /interval/);

  near(eventRelativeUncertaintyConfidence(40, -10, 90), 40 / 90);
  near(eventRelativeUncertaintyConfidence(-40, -90, 10), 40 / 90);
  near(eventRelativeUncertaintyConfidence(.7, .6, .8, .5), 2 / 3);
  assert.equal(eventRelativeUncertaintyConfidence(40, 40, 40), 1);
  assert.equal(eventRelativeUncertaintyConfidence(0, -10, 10), 0);
  assert.throws(() => eventRelativeUncertaintyConfidence(1, 2, 3), /interval/);

  const uncertain = eventUncertaintyRiskFloor({ initialEquity: 10_000,
    liquidatableHighWater: 10_200, maximumInitialRiskBps: 10,
    minimumProtectedProfitFraction: .5, confidence: 0 });
  assert.equal(uncertain.floor, 10_200);
  assert.equal(uncertain.effectiveInitialRiskBps, 0);
  assert.equal(uncertain.protectedProfitFraction, 1);
  const partial = eventUncertaintyRiskFloor({ initialEquity: 10_000,
    liquidatableHighWater: 10_200, maximumInitialRiskBps: 10,
    minimumProtectedProfitFraction: .5, confidence: .4 });
  near(partial.floor, 10_160);
  near(partial.protectedProfitFraction, .8);
  const certain = eventUncertaintyRiskFloor({ initialEquity: 10_000,
    liquidatableHighWater: 10_200, maximumInitialRiskBps: 10,
    minimumProtectedProfitFraction: .5, confidence: 1 });
  assert.equal(certain.floor, 10_100);
  const seed = eventUncertaintyRiskFloor({ initialEquity: 10_000,
    liquidatableHighWater: 10_000, maximumInitialRiskBps: 10,
    minimumProtectedProfitFraction: .5, confidence: .4 });
  assert.equal(seed.floor, 9_996);
  assert.throws(() => eventUncertaintyRiskFloor({ initialEquity: 10_000,
    liquidatableHighWater: 10_000, maximumInitialRiskBps: 10,
    minimumProtectedProfitFraction: .5, confidence: 2 }), /risk state/);
});

test("event uniqueness agrees with per-return concurrency and excludes shared endpoints", () => {
  const rows = [{ start: 0, end: 3 }, { start: 2, end: 4 }, { start: 4, end: 5 }, { start: 1000000000, end: 1000000001 }];
  const weights = eventAverageUniqueness(rows);
  near(weights[0], 5 / 6); near(weights[1], 3 / 4); near(weights[2], 1); near(weights[3], 1);
  assert.deepEqual(eventAverageUniqueness([{ start: 0, end: 5 }, { start: 0, end: 5 }]), [.5, .5]);
  const varied = Array.from({ length: 50 }, (_, i) => ({ start: (i * 7) % 40, end: (i * 7) % 40 + 1 + i % 9 }));
  const direct = varied.map(row => {
    let sum = 0;
    for (let t = row.start; t < row.end; t++) sum += 1 / varied.filter(other => other.start <= t && t < other.end).length;
    return sum / (row.end - row.start);
  });
  eventAverageUniqueness(varied).forEach((value, i) => near(value, direct[i]));
  assert.deepEqual(eventAverageUniqueness([]), []);
  assert.throws(() => eventAverageUniqueness([{ start: 1, end: 1 }]), /interval/);
  assert.throws(() => eventAverageUniqueness([{ start: .5, end: 2 }]), /interval/);
});

test("weighted joint estimation matches replicated observations and preserves unit-weight controls", () => {
  const base = constantModel([-.02, .03]), features = EVENT_FEATURES.map(() => 0);
  const rows: MoveSample[] = [-.02, -.004, .005, .03].map((r, i) => ({ start: i, end: i + 1,
    features, nextFeatures: features, return: r, low: Math.min(0, r) - .001, high: Math.max(0, r) + .001,
    duration: i + 1, label: eventMoveLabel(r, i + 1, base.clock) }));
  const weights = [1, 3, 2, 4], snapshot = JSON.stringify(base);
  const weighted = reestimateEventTree(base, rows, 2, weights);
  const replicated = reestimateEventTree(base, rows.flatMap((row, i) => Array.from({ length: weights[i] }, () => row)), 2);
  const atoms = (model: EventDistribution) => {
    const grouped = new Map<string, number>();
    for (const { probability, ...a } of model.kernels[0]) {
      const key = JSON.stringify(a); grouped.set(key, (grouped.get(key) ?? 0) + probability);
    }
    return grouped;
  };
  const expected = atoms(replicated);
  for (const [key, probability] of atoms(weighted)) near(probability, expected.get(key)!);
  weighted.priorClasses.forEach((p, i) => near(p, replicated.priorClasses[i]));
  weighted.classProbabilities[0].forEach((p, i) => near(p, replicated.classProbabilities[0][i]));
  assert.equal(JSON.stringify(base), snapshot);
  const many = Array.from({ length: 301 }, (_, i) => ({ ...rows[i % rows.length], return: -.01 + i / 10000,
    low: -.02, high: .04, label: eventMoveLabel(-.01 + i / 10000, rows[i % rows.length].duration, base.clock) }));
  assert.deepEqual(reestimateEventTree(base, many, 32, many.map(() => 1)), reestimateEventTree(base, many, 32));
  assert.throws(() => reestimateEventTree(base, rows, 2, [1]), /weights/);
  assert.throws(() => reestimateEventTree(base, rows, 2, [1, 1, 0, 1]), /weights/);
  assert.throws(() => reestimateEventTree(base, rows, 2, [1, 1, NaN, 1]), /weights/);
});

test("weighted event CRPS agrees with direct pairwise scoring and physical units", () => {
  const values = [7, -3, 7, 0, 100], weights = [2, 3, 1, 4, 0], total = 10;
  const score = prepareEventCrps(values, weights);
  for (const actual of [-10, -3, 0, 2, 7, 15, 100]) {
    let expected = 0;
    for (let i = 0; i < values.length; i++) {
      expected += weights[i] / total * Math.abs(values[i] - actual);
      for (let j = 0; j < values.length; j++)
        expected -= .5 * weights[i] * weights[j] / total ** 2 * Math.abs(values[i] - values[j]);
    }
    near(score(actual), expected, 1e-12);
    near(prepareEventCrps(values.map(v => 2 * v + 5), weights.map(w => w * 3))(2 * actual + 5), 2 * expected, 1e-12);
  }
  assert.equal(prepareEventCrps([3], [5])(-2), 5);
  assert.equal(prepareEventCrps([3, 3], [1, 2])(3), 0);
  assert.throws(() => prepareEventCrps([1], [0]), /mass/);
  assert.throws(() => prepareEventCrps([1], [-1]), /distribution/);
  assert.throws(() => score(NaN), /observation/);
  const model = constantModel([.001]); model.kernels[0][0].duration = 2;
  const row: MoveSample = { start: 0, end: 1, features: new Array<number>(EVENT_FEATURES.length).fill(0), nextFeatures: new Array<number>(EVENT_FEATURES.length).fill(0),
    return: -.001, duration: 1, low: -.001, high: 0, label: 0 };
  assert.deepEqual(eventMarginalCrps(model, [row]), { returnCrpsBps: 20, durationCrpsSeconds: 60 });
});

test("explicit duration bins survive policy serialization and mean recalibration", () => {
  const clock: EventClock = { thresholdBps: 60, maxCandles: 3600, candleIntervalMs: 1000, durationBinsMinutes: [5, 30] };
  validateEventClock(clock);
  for (const [duration, label] of [[5, 6], [5 + 1 / 60, 7], [30, 7], [30 + 1 / 60, 8]]) {
    assert.equal(eventMoveLabel(0, duration, clock), label);
    assert.equal(eventMoveLabel(0, duration, { ...clock, candleIntervalMs: 60000 }), label);
  }
  for (const bins of [[0, 30], [30, 5], [5, 5], [5, Infinity], [5]])
    assert.throws(() => validateEventClock({ ...clock, durationBinsMinutes: bins as [number, number] }), /duration bins/);
  const model = constantModel([-.001, .001, 0]);
  model.clock = { ...model.clock, thresholdBps: 60, durationBinsMinutes: [5, 30] };
  model.kernels[0].forEach((a, i) => a.duration = [5, 30, 31][i]);
  const calibrated = scaleEventMean(model, 1), n = model.counts[0];
  const expected = new Array<number>(15).fill(.5 / (n + 7.5));
  for (let i = 0; i < 3; i++) expected[6 + i] += calibrated.kernels[0][i].probability * n / (n + 7.5);
  calibrated.classProbabilities[0].forEach((p, i) => near(p, expected[i], 1e-12));
  const policy = buildEventPolicy(calibrated, DEFAULT_EVENT_COSTS, { depths: 0, referenceEquity: 10000, referencePrice: 100 });
  const restored = restoreEventPolicy(JSON.parse(JSON.stringify(serializeEventPolicy(policy))));
  assert.deepEqual(restored.model.clock.durationBinsMinutes, [5, 30]);
  assert.deepEqual(restored.model.kernels, policy.model.kernels);
});

test("quadrature preserves custom duration class mass and successor coupling", () => {
  const clock: EventClock = { thresholdBps: 60, maxCandles: 3600, candleIntervalMs: 1000, durationBinsMinutes: [5, 30] };
  const kernel = Array.from({ length: 900 }, (_, i) => ({ return: .007 + (i % 17) / 10000,
    low: -.01, high: .02, duration: [1, 10, 40][i % 3] + (i % 19) / 60, next: i % 2, probability: 1 / 900 }));
  const reduced = compressEventKernel(kernel, clock);
  assert.ok(reduced.length < kernel.length);
  for (let next = 0; next < 2; next++) for (let bin = 0; bin < 3; bin++) {
    const mass = (atoms: typeof kernel) => atoms.reduce((sum, a) => sum + (a.next === next
      && (a.duration <= 5 ? 0 : a.duration <= 30 ? 1 : 2) === bin ? a.probability : 0), 0);
    near(mass(reduced), mass(kernel), 1e-12);
  }
});

test("cash horizon bounds agree with exhaustive Bellman values and stop before a profitable longer hold", () => {
  const costs = { ...DEFAULT_EVENT_COSTS, feeBps: 200, slippageBps: 0, maxLeverage: 1,
    minNotional: 0, maxNotional: 2, minQuantity: 1, quantityStep: 1, longBorrowBpsPerDay: 0, shortBorrowBpsPerDay: 0 };
  const account = { equity: 2, price: 1, exposure: 0 };
  const drift = constantModel([.01]), check = eventCashHorizon(drift, costs, 20);
  assert.equal(check.verifiedDepth, 1); assert.equal(check.rows.length, 2);
  const oracle = eventBellmanReference(drift, costs, { terminal: "marked" });
  assert.equal(oracle.decide(0, account, 1).value, 0);
  assert.ok(oracle.decide(0, account, 2).value > 0);
  const fair = constantModel([-.01, .01]), fairCheck = eventCashHorizon(fair, costs, 1000);
  assert.equal(fairCheck.verifiedDepth, 1000);
  const fairOracle = eventBellmanReference(fair, costs, { terminal: "marked" });
  for (let depth = 1; depth <= 3; depth++) assert.equal(fairOracle.decide(0, account, depth).value, 0);
});

test("cash horizon retains return-successor dependence and does not reuse a current-state-only drift bound", () => {
  const m = constantModel([.02, 0]);
  m.nodes = [{ feature: 0, cut: 0, leaf: -1, left: 1, right: 2 },
    { feature: -1, cut: 0, leaf: 0, left: -1, right: -1 }, { feature: -1, cut: 0, leaf: 1, left: -1, right: -1 }];
  m.kernels[0][1].next = 1;
  m.kernels.push([{ probability: .5, return: -.02, low: -.02, high: 0, duration: 1, next: 1 },
    { probability: .5, return: 0, low: 0, high: 0, duration: 1, next: 0 }]);
  m.counts = [2, 2];
  const result = eventCashHorizon(m, { ...DEFAULT_EVENT_COSTS, feeBps: 300 }, 2);
  assert.equal(result.verifiedDepth, 2);
  assert.ok(Math.abs(result.rows[1].maximumShadowBps - 101) < 1e-9);
  const unsupported = structuredClone(m); unsupported.kernels[0][0].probability += 1e-6;
  assert.throws(() => eventCashHorizon(unsupported, DEFAULT_EVENT_COSTS, 2), /probabilit|normalized/i);
});

test("lifecycle positions open once, reduce pro rata and cross zero without charging each virtual lot", () => {
  const ledger = new EventPositionLedger();
  ledger.fill({ time: 0, price: 100, quantityAfter: 2, cost: .2 });
  ledger.fill({ time: 1, price: 110, quantityAfter: 3, cost: .11 });
  const changed = ledger.fill({ time: 2, price: 120, quantityAfter: 2.25, cost: .09 });
  assert.deepEqual(changed.map(p => p.quantity), [.5, .25]);
  assert.deepEqual(ledger.positions().map(p => p.remainingQuantity), [1.5, .75]);
  assert.equal(ledger.summary(120).realizedPnl, 12.5);
  assert.equal(ledger.closeDomains().length, 2);
  ledger.charge(3, 3);
  const reversed = ledger.fill({ time: 4, price: 90, quantityAfter: -.75, cost: .27 });
  assert.deepEqual(reversed.map(p => p.operation), ["close", "close", "open"]);
  assert.equal(ledger.quantity(), -.75);
  assert.deepEqual(ledger.positions().map(p => p.initialQuantity), [2, 1, .75]);
  assert.deepEqual(ledger.positions().map(p => p.remainingQuantity), [0, 0, .75]);
  assert.ok(Math.abs(ledger.summary(90).costs - .67) < 1e-12);
  assert.ok(Math.abs(ledger.positions().reduce((s, p) => s + p.borrowing, 0) - 3) < 1e-12);
  assert.throws(() => ledger.fill({ time: 3, price: 90, quantityAfter: .25, cost: 0 }), /Unordered/);
  assert.throws(() => ledger.fill({ time: 5, price: 90, quantityAfter: -.75, cost: 1 }), /Invalid/);
});

test("lifecycle decomposition conserves cash-ledger wealth through repeated fills, funding and settlement", () => {
  const ledger = new EventPositionLedger(false), initial = 10000;
  let cash = initial, quantity = 0;
  for (let i = 1; i <= 1000; i++) {
    const price = 100 + ((i * 37) % 59), target = (((i * 43) % 101) - 50) / 10, delta = target - quantity;
    const cost = Math.abs(delta) * price * .0012;
    ledger.fill({ time: i * 2, price, quantityAfter: target, cost });
    cash -= delta * price + cost; quantity = target;
    const funding = quantity ? (i % 5 === 0 ? -.01 : .03) : 0;
    if (funding) ledger.charge(i * 2 + 1, funding);
    cash -= funding;
    const s = ledger.summary(price);
    assert.ok(Math.abs(s.collapsedQuantity - quantity) < 1e-11);
    assert.ok(Math.abs(initial + s.equityChange - cash - quantity * price) < 1e-8);
    assert.ok(ledger.positions().every(p => p.remainingQuantity <= p.initialQuantity && p.remainingQuantity > 0));
  }
  const cost = Math.abs(quantity) * 120 * .0012;
  ledger.fill({ time: 2002, price: 120, quantityAfter: 0, cost, reason: "terminal" });
  cash += quantity * 120 - cost;
  assert.equal(ledger.quantity(), 0);
  assert.equal(ledger.summary(120).active, 0);
  assert.ok(Math.abs(initial + ledger.summary(120).equityChange - cash) < 1e-8);
});

test("lifecycle initial marks and liquidation writeoffs reconcile without inventing an exchange fill", () => {
  const ledger = new EventPositionLedger();
  ledger.fill({ time: 0, price: 100, quantityAfter: -2, cost: 0, origin: "initial-mark" });
  ledger.charge(1, .4);
  ledger.liquidate(2, 160, 150 - 120 - .4);
  const s = ledger.summary(160);
  assert.equal(s.quantity, 0);
  assert.ok(Math.abs(150 + s.equityChange) < 1e-12);
  assert.equal(ledger.positions()[0].closeReason, "liquidation");
  assert.equal(ledger.positions()[0].origin, "initial-mark");
});
import { prepareEventMultiStepUpper } from "../src/event-multi-step-upper.js";

const costs = { ...DEFAULT_EVENT_COSTS, feeBps: 5, slippageBps: 0,
  minNotional: 0, minQuantity: 0, maxNotional: 1e12, quantityStep: 1e-10,
  longBorrowBpsPerDay: 0, shortBorrowBpsPerDay: 0 };
const near = (actual: number, expected: number, tolerance = 1e-8) => assert.ok(Math.abs(actual - expected) < tolerance, `${actual} != ${expected}`);

test("compiled holding laws preserve dense empirical values, derivatives and rare ruin", () => {
  const kernel = Array.from({ length: 128 }, (_, i) => {
    const r = .035 * Math.sin(i * 7);
    return { probability: 1 / 128, return: r, low: Math.min(0, r) - .01, high: Math.max(0, r) + .01, duration: 1 + i * 10, next: 0 };
  });
  const c = { ...DEFAULT_EVENT_COSTS, maxLeverage: 5, longBorrowBpsPerDay: 500, shortBorrowBpsPerDay: 300 };
  for (const terminal of ["marked", "friction"] as const) {
    const law = prepareEventHoldingLaw(kernel, c, terminal), f = terminal === "friction" ? (c.feeBps + c.slippageBps) / 10000 : 0;
    let series = 0, direct = 0;
    for (const x of [-30, -5, -1.01, -1, -.5, 0, .2, 1, 1.0001, 2, 5, 30]) {
      let value = 0, derivative = 0;
      for (const a of kernel) {
        const beta = (x < 0 ? c.shortBorrowBpsPerDay : c.longBorrowBpsPerDay) / 10000 * a.duration / 1440;
        const wealth = eventHolding(x, a, c).factor - f * Math.abs(x) * (1 + a.return);
        if (wealth <= 0) { value = -Infinity; break; }
        value += a.probability * Math.log(wealth);
        derivative += a.probability * (a.return + (x < 0 ? beta : x > 1 ? -beta : 0)
          - (x < 0 ? -1 : 1) * f * (1 + a.return)) / wealth;
      }
      const actual = law.score(x); actual.series ? series++ : direct++;
      assert.equal(law.survives(x), kernel.every(a => !eventHolding(x, a, c).liquidated));
      if (Number.isFinite(value)) { near(actual.value, value, 1e-13); near(actual.derivative, derivative, 1e-13); }
      else assert.equal(actual.value, value);
    }
    assert.ok(series > 0 && direct > 0);
    const rare = [...kernel, { probability: 1e-300, return: 0, low: -.999, high: 200, duration: 1, next: 0 }];
    const rareLaw = prepareEventHoldingLaw(rare, c, terminal);
    assert.equal(rareLaw.survives(-1), false); assert.equal(rareLaw.survives(2), false);
  }
});

test("prepared two-event search reuses fixed laws without leaking accounts or later model mutations", () => {
  const m = constantModel([.02, -.03]), c = { ...DEFAULT_EVENT_COSTS, quantityStep: .1, minQuantity: .1, maxNotional: 300 };
  const snapshot = structuredClone(m), fixedCosts = { ...c }, solve = prepareEventTwoStep(m, c, "marked");
  const accounts = [{ equity: 100, price: 10, exposure: 0 }, { equity: 200, price: 15, exposure: -1.1 },
    { equity: 10, price: 30, exposure: .7 }, { equity: 100, price: 10, exposure: 0 }];
  m.kernels[0][0].return = .1; m.kernels[0][0].high = .1; c.feeBps = 500;
  for (const a of accounts) assert.deepEqual(solve(0, a, { maxEvaluations: 128 }),
    decideEventTwoStep(snapshot, 0, a, fixedCosts, "marked", { maxEvaluations: 128 }));
  assert.throws(() => solve(999, accounts[0]), /Invalid two-event/);
  assert.throws(() => solve(0, accounts[0], { maxEvaluations: 0 }), /Invalid two-event/);
});

test("three-event action probes bound stochastic H2 continuation and never certify a sampled prefix", () => {
  const model = constantModel([.08, -.06]), c = { ...DEFAULT_EVENT_COSTS, maxLeverage: 2, quantityStep: 1, minQuantity: 1,
    minNotional: 5, maxNotional: 20 }, account = { equity: 100, price: 10, exposure: .2 };
  for (const terminal of ["marked", "friction"] as const) {
    const reference = eventBellmanReference(model, c, { terminal }), evaluate = prepareEventThreeStepActions(model, c, terminal);
    for (const q of [-2, 0, 2]) {
      const result = evaluate(0, account, q, { maxEvaluations: 128 }), exact = reference.actionValue(0, account, 3, q);
      assert.ok(result.complete && result.feasible); near(result.evaluatedProbability, 1);
      assert.ok(result.lowerValue <= exact + 1e-10 && result.upperValue >= exact - 1e-10);
      assert.ok(result.gap <= 1e-7); near(result.lowerValue, exact, 1e-10);
    }
    const partial = evaluate(0, account, 0, { limitOutcomes: 1 });
    assert.equal(partial.complete, false); assert.equal(partial.lowerValue, -Infinity); assert.equal(partial.upperValue, Infinity);
    assert.equal(partial.evaluatedOutcomes, 1); assert.equal(evaluate(0, account, .5).feasible, false);
    const tail = evaluate(0, account, 0, { offsetOutcomes: 1 });
    assert.equal(tail.complete, false); assert.equal(tail.lowerValue, -Infinity); assert.equal(tail.upperValue, Infinity);
    assert.equal(tail.evaluatedOutcomes, 1);
    assert.throws(() => evaluate(0, account, 0, { offsetOutcomes: 2 }), /offset exceeds/);
    assert.throws(() => evaluate(0, account, 0, { maxEvaluations: 1 }), /Invalid three-event/);
  }
  const dense = { ...model, kernels: model.kernels.map(k => k.flatMap(a => Array.from({ length: 40 }, () => ({ ...a, probability: a.probability / 40 })))) };
  const coalesced = prepareEventThreeStepActions(dense, c, "marked")(0, account, 0);
  assert.equal(coalesced.totalOutcomes, 2); near(coalesced.lowerValue, eventBellmanReference(model, c, { terminal: "marked" }).actionValue(0, account, 3, 0), 1e-10);
  const rare = { ...model, kernels: [[...model.kernels[0], { probability: 1e-300, return: 0, low: -.999, high: 0, duration: 1, next: 0 }]] };
  const doomed = prepareEventThreeStepActions(rare, c, "marked")(0, { ...account, exposure: 1.8 }, 0);
  assert.equal(doomed.complete, true); assert.equal(doomed.upperValue, -Infinity);
});

test("three-event policy certifies against a global bound and leaves unsupported recovery states unresolved", () => {
  const model = constantModel([.001]), c = { ...DEFAULT_EVENT_COSTS, maxLeverage: 1, maxNotional: 400,
    feeBps: 12, slippageBps: 0, quantityStep: 1, minQuantity: 1, minNotional: 5,
    longBorrowBpsPerDay: 0, shortBorrowBpsPerDay: 0 }, account = { equity: 100.12, price: 10, exposure: 0 };
  const actual = prepareEventThreeStep(model, c, "marked", { shadowPoints: 33 })(0, account, { maxRootEvaluations: 2 });
  const expected = eventBellmanReference(model, c, { terminal: "marked", maxNodes: 1000000 }).decide(0, account, 3);
  assert.ok(actual.converged && actual.quantity > 0 && actual.rootEvaluations <= 2);
  assert.ok(actual.lowerValue <= expected.value + 1e-10 && actual.upperValue >= expected.value - 1e-10);
  assert.ok(expected.value - actual.lowerValue <= actual.tolerance + 1e-10);
  // Small future short top-ups allowed by the continuous relaxation are below
  // the lot minimum here. A good candidate alone must not close that gap.
  const discrete = prepareEventThreeStep(constantModel([-.001]), c, "marked")(0, account);
  assert.ok(discrete.quantity < 0 && Number.isFinite(discrete.globalUpperValue));
  assert.ok(!discrete.converged && discrete.gap > discrete.tolerance);
  for (const terminal of ["marked", "friction"] as const) {
    const quiet = prepareEventThreeStep(constantModel([0]), c, terminal)(0, account);
    assert.ok(quiet.converged && quiet.cashLowerPolicy); assert.equal(quiet.quantity, 0); assert.equal(quiet.rootEvaluations, 0);
    const limitedCosts = { ...c, maxNotional: 20 }, limitedAccount = { ...account, exposure: -3 };
    const limitedModel = constantModel([-.01, .01]);
    const limited = prepareEventThreeStep(limitedModel, limitedCosts, terminal)(0, limitedAccount, { maxRootEvaluations: 1 });
    const exact = eventBellmanReference(limitedModel, limitedCosts, { terminal }).decide(0, limitedAccount, 3);
    assert.equal(limited.converged, false); assert.equal(limited.globalUpperValue, Infinity);
    assert.ok(limited.lowerValue <= exact.value + 1e-10 || limited.lowerValue === exact.value);
    assert.ok(limited.rootEvaluations <= 1);
  }
});

test("complete holding bounds close future-lot gaps near the leverage cap", () => {
  const model = constantModel([.001]), c = { ...DEFAULT_EVENT_COSTS, maxLeverage: 1, maxNotional: 400,
    feeBps: 12, slippageBps: 0, quantityStep: 1, minQuantity: 1, minNotional: 5,
    longBorrowBpsPerDay: 0, shortBorrowBpsPerDay: 0 }, account = { equity: 100.12, price: 10, exposure: .9997 };
  const actual = prepareEventThreeStep(model, c, "marked")(0, account, { maxRootEvaluations: 2 });
  const exact = eventBellmanReference(model, c, { terminal: "marked", maxNodes: 1000000 }).decide(0, account, 3);
  assert.equal(actual.quantity, 0); assert.ok(actual.converged);
  assert.ok(actual.recursiveUpperValue - actual.lowerValue > actual.tolerance);
  assert.ok(actual.lowerValue <= exact.value + 1e-10 && actual.upperValue >= exact.value - 1e-10);
  assert.ok(actual.nonzeroUpperValues.every(v => v! < actual.lowerValue));
});

test("recursive shadow-price bounds cover global finite-horizon optima and reject reachable recovery exceptions", () => {
  const model = constantModel([.03, .06]); model.kernels[0].forEach(a => a.next = 1);
  model.kernels.push(constantModel([-.07, -.01]).kernels[0]); model.counts.push(1);
  const c = { ...DEFAULT_EVENT_COSTS, maxLeverage: 1, maxNotional: 400, minNotional: 12,
    quantityStep: 1, minQuantity: 1, feeBps: 25, longBorrowBpsPerDay: 40, shortBorrowBpsPerDay: 20 };
  for (const terminal of ["marked", "friction"] as const) {
    const upper = prepareEventMultiStepUpper(model, c, terminal, { depth: 3, shadowPoints: 9 });
    for (const exposure of [0, -.9]) for (const h of [1, 2, 3]) {
      const account = { equity: 100, price: 10, exposure }, bound = upper.query(0, account, h);
      const exact = eventBellmanReference(model, c, { terminal, maxNodes: 1000000 }).decide(0, account, h);
      assert.equal(bound.recoveryExcluded, true); assert.ok(Number.isFinite(bound.upperValue));
      assert.ok(bound.upperValue >= exact.value - 1e-9, `${h}: ${bound.upperValue} < ${exact.value}`);
      if (h === 3) for (const side of [-1, 1] as const) {
        const directional = upper.query(0, account, h, side);
        const reference = eventBellmanReference(model, c, { terminal, maxNodes: 1000000 });
        const value = Math.max(...Array.from({ length: 41 }, (_, i) => reference.actionValue(0, account, h, side * i)));
        assert.ok(directional.upperValue >= value - 1e-9);
      }
    }
  }
  const limited = prepareEventMultiStepUpper(model, { ...c, maxNotional: 20 }, "marked", { depth: 3 });
  const rejected = limited.query(0, { equity: 100, price: 10, exposure: -3 });
  assert.equal(rejected.recoveryExcluded, false); assert.equal(rejected.upperValue, Infinity);
});

test("recovery bounds include maximum clips that restore the ordinary cap", () => {
  const model = constantModel([-.02, .03]), account = { equity: 10, price: 1, exposure: 0 };
  const costs = { ...DEFAULT_EVENT_COSTS, maxLeverage: 2, minNotional: 0, maxNotional: 12,
    quantityStep: 1, minQuantity: 1, longBorrowBpsPerDay: 20, shortBorrowBpsPerDay: 35 };
  // 12 is smaller than twice the reachable position; the former sufficient
  // check rejected this case. Such clips still restore leverage below two.
  for (const terminal of ["marked", "friction"] as const) {
    const upper = prepareEventMultiStepUpper(model, costs, terminal, { depth: 2, shadowPoints: 17 });
    for (const exposure of [-2.1, -1.9, 0, 1.9, 2.1]) {
      const state = { ...account, exposure }, bound = upper.query(0, state);
      const exact = eventBellmanReference(model, costs, { terminal, maxNodes: 1000000 }).decide(0, state, 2);
      assert.ok(bound.recoveryExcluded && Number.isFinite(bound.upperValue));
      assert.ok(bound.upperValue >= exact.value - 1e-9, `${terminal} ${exposure}: ${bound.upperValue} < ${exact.value}`);
    }
  }
  const insufficient = { ...costs, maxNotional: 1 };
  const rejected = prepareEventMultiStepUpper(model, insufficient, "marked", { depth: 2 }).query(0, { ...account, exposure: 4 });
  assert.equal(rejected.recoveryExcluded, false); assert.equal(rejected.upperValue, Infinity);
  const highCosts = { ...costs, feeBps: 3000, slippageBps: 0, maxNotional: 40 };
  const severe = constantModel([-.9, .9]);
  assert.equal(prepareEventMultiStepUpper(severe, highCosts, "marked", { depth: 2 }).query(0, account).recoveryExcluded, false);
});

test("directional order bounds retain tolerance-edge orders and disconnected holding", () => {
  const c = { ...DEFAULT_EVENT_COSTS, maxLeverage: 1, feeBps: 0, slippageBps: 0,
    quantityStep: .1, minQuantity: 1, minNotional: 0, maxNotional: .000099995,
    longBorrowBpsPerDay: 0, shortBorrowBpsPerDay: 0 };
  const account = { equity: .001, price: .0001, exposure: 0 };
  for (const side of [-1, 1] as const) {
    const model = constantModel([side * .1]);
    const upper = prepareEventMultiStepUpper(model, c, "marked", { depth: 1 });
    const bound = upper.query(0, account, 1, side, "exchange-relaxation");
    // The dollar tolerance admits this order although rounding maxNotional
    // down to the nearest lot would exclude it.
    const exact = eventBellmanReference(model, c, { terminal: "marked" }).actionValue(0, account, 1, side);
    assert.ok(Number.isFinite(exact) && exact > 0);
    assert.ok(Number.isFinite(bound.upperValue) && bound.upperValue >= exact - 1e-10);
    const opposite = upper.query(0, account, 1, -side as -1 | 1, "exchange-relaxation");
    assert.ok(opposite.upperValue >= 0 && opposite.upperValue < 1e-8, "holding remains available separately");
  }
});

test("root lot hull bounds charge mandatory trims and retain discrete cap endpoints", () => {
  const c = { ...DEFAULT_EVENT_COSTS, maxLeverage: 1, feeBps: 12, slippageBps: 0,
    quantityStep: 1, minQuantity: 1, minNotional: 5, maxNotional: 400,
    longBorrowBpsPerDay: 0, shortBorrowBpsPerDay: 0 };
  for (const direction of [-1, 1]) for (const exposure of [0, direction * 1.001]) {
    const model = constantModel([direction * .01]), account = { equity: 100, price: 10, exposure };
    const upper = prepareEventMultiStepUpper(model, c, "marked", { depth: 1 });
    const exact = eventBellmanReference(model, c, { terminal: "marked" }).decide(0, account, 1);
    const bound = Math.max(upper.query(0, account, 1, -1, "exchange-relaxation").upperValue,
      upper.query(0, account, 1, 1, "exchange-relaxation").upperValue);
    assert.ok(Number.isFinite(bound) && bound >= exact.value - 1e-10 && bound - exact.value <= 1e-7);
    if (exposure) assert.ok(exact.quantity * direction < 0, "holding above the cap is infeasible");
  }
});

test("a compiled global H2 bound can certify a feasible seed without interval search", () => {
  const model = constantModel([.03, -.03]), c = { ...DEFAULT_EVENT_COSTS, maxLeverage: 1, maxNotional: 400,
    quantityStep: .1, minQuantity: .1, minNotional: 5 }, account = { equity: 100, price: 10, exposure: 0 };
  const solve = prepareEventTwoStep(model, c, "marked", { globalUpper: true, shadowPoints: 33 });
  const result = solve(0, account, { maxEvaluations: 4 });
  assert.ok(result.converged && Number.isFinite(result.globalUpperValue));
  assert.equal(result.quantity, 0); assert.equal(result.search.boundEvaluations, 0);
  const exact = eventBellmanReference(model, c, { terminal: "marked" }).decide(0, account, 2);
  near(result.value, exact.value, 1e-10); assert.ok(result.upperValue >= exact.value - 1e-10);
  const limited = prepareEventTwoStep(model, { ...c, maxNotional: 20 }, "marked", { globalUpper: true });
  const fallback = limited(0, { ...account, exposure: -3 }, { maxEvaluations: 512 });
  assert.equal(fallback.globalUpperValue, Infinity); assert.ok(fallback.converged && fallback.search.boundEvaluations > 0);
});

test("recursive target seeds find a fee-paying H2 entry while H1 stays cash", () => {
  const model = constantModel([-.001]), c = { ...DEFAULT_EVENT_COSTS, maxLeverage: 1, maxNotional: 400,
    feeBps: 12, slippageBps: 0, quantityStep: 1, minQuantity: 1, minNotional: 5,
    longBorrowBpsPerDay: 0, shortBorrowBpsPerDay: 0 }, account = { equity: 100.12, price: 10, exposure: 0 };
  assert.equal(decideEventOneStep(model.kernels[0], account, c, "marked").quantity, 0);
  const actual = prepareEventTwoStep(model, c, "marked", { globalUpper: true, shadowPoints: 33 })(0, account, { maxEvaluations: 4 });
  const expected = eventBellmanReference(model, c, { terminal: "marked" }).decide(0, account, 2);
  assert.ok(actual.converged && actual.quantity < 0 && actual.search.boundEvaluations === 0);
  near(actual.value, expected.value, 1e-10); assert.ok(actual.upperValue >= expected.value - 1e-10);
});

test("two-event bounds retain a pruned numerical gap and do not anticipate a fair sign", () => {
  const c = { ...DEFAULT_EVENT_COSTS, feeBps: 0, slippageBps: 0, maxLeverage: 1,
    minNotional: 1, maxNotional: 50, quantityStep: 1, minQuantity: 1, longBorrowBpsPerDay: 0, shortBorrowBpsPerDay: 0 };
  const m = constantModel([.1, -.1]), account = { equity: 100, price: 10, exposure: 0 };
  const result = decideEventTwoStep(m, 0, account, c, "marked", { maxEvaluations: 16 });
  near(result.value, 0); assert.equal(result.quantity, 0); assert.ok(result.converged);
  assert.ok(result.gap > 0 && result.gap <= result.tolerance && result.search.prunedIntervals > 0);
  assert.equal(result.upperValue - result.lowerValue, result.gap);
  assert.throws(() => decideEventTwoStep(m, 0, account, c, "market" as never), /Invalid two-event/);
});

test("two-event search respects a budget and finds a constrained anticipatory reversal", () => {
  const c = { ...DEFAULT_EVENT_COSTS, feeBps: 100, slippageBps: 2, maxLeverage: 2,
    minNotional: 12, maxNotional: 80, quantityStep: 1, minQuantity: 0, longBorrowBpsPerDay: 1000, shortBorrowBpsPerDay: 0 };
  const m = constantModel([.03, .05]);
  m.kernels[0].forEach(a => { a.next = 1; a.low = -.3; a.high = .24; });
  m.kernels.push(constantModel([-.25, -.13]).kernels[0]);
  m.kernels[1][0].probability = .3; m.kernels[1][1].probability = .7;
  m.counts.push(1);
  const account = { equity: 100, price: 10, exposure: 0 };
  const reference = eventBellmanReference(m, c, { terminal: "marked" }), optimal = reference.decide(0, account, 2);
  const small = decideEventTwoStep(m, 0, account, c, "marked", { maxEvaluations: 2 });
  const full = decideEventTwoStep(m, 0, account, c, "marked", { maxEvaluations: 128 });
  assert.equal(small.converged, false); assert.ok(small.search.evaluations <= 2);
  assert.ok(small.lowerValue < optimal.value && small.upperValue >= optimal.value);
  near(full.value, optimal.value, 1e-10); near(full.quantity, optimal.quantity);
  assert.ok(full.shortEntry && full.converged); assert.ok(decideEventOneStep(m.kernels[0], account, c, "marked").quantity > 0);
  near(reference.actionValue(0, account, 2, full.quantity), full.lowerValue, 1e-10);
});

test("two-event upper bounds include maximum-order recovery outside the leverage cap", () => {
  const c = { ...DEFAULT_EVENT_COSTS, maxLeverage: 1, minNotional: 1, maxNotional: 20, quantityStep: .1, minQuantity: .1 };
  const m = constantModel([.02, -.03]), account = { equity: 100, price: 10, exposure: -3 };
  for (const terminal of ["marked", "friction"] as const) {
    const reference = eventBellmanReference(m, c, { terminal }).decide(0, account, 2);
    const result = decideEventTwoStep(m, 0, account, c, terminal, { maxEvaluations: 128 });
    assert.ok(result.converged && result.quantity > 0 && result.exposure < -1);
    near(result.value, reference.value, 1e-10); assert.ok(result.upperValue >= reference.value - 1e-10);
    assert.ok(result.search.exceptionEvaluations > 0);
  }
});

test("recovery bounds cover cap-restoring and still-over-cap clips across borrowing regimes", () => {
  const m = constantModel([-.07, .05]);
  m.kernels[0] = [{ probability: .6, return: -.07, low: -.12, high: .03, duration: 60, next: 1 },
    { probability: .4, return: .05, low: -.01, high: .08, duration: 120, next: 0 }];
  m.kernels.push([{ probability: .3, return: .06, low: -.02, high: .1, duration: 30, next: 1 },
    { probability: .7, return: -.03, low: -.08, high: .04, duration: 10, next: 0 }]);
  m.counts.push(1);
  for (const terminal of ["marked", "friction"] as const) for (const leverage of [1, 5]) for (const maximum of [20, 60, 200]) {
    const c = { ...DEFAULT_EVENT_COSTS, maxLeverage: leverage, maxNotional: maximum, quantityStep: 1, minQuantity: 1,
      longBorrowBpsPerDay: 50, shortBorrowBpsPerDay: 90 };
    const exact = eventBellmanReference(m, c, { terminal }), solve = prepareEventTwoStep(m, c, terminal, { globalUpper: true });
    for (const exposure of [-6, -3, 0, .5, 1, 3, 6]) {
      const account = { equity: 100, price: 10, exposure }, reference = exact.decide(0, account, 2);
      const result = solve(0, account, { maxEvaluations: 256 });
      assert.ok(result.converged && result.upperValue >= reference.value - 1e-10);
      assert.ok(Math.abs(result.value - reference.value) < 1e-8 || result.value === reference.value);
    }
  }
});

test("two-event continuation preserves positive-probability ruin, however rare", () => {
  const c = { ...DEFAULT_EVENT_COSTS, maxLeverage: 5, minNotional: 1, maxNotional: 100, quantityStep: 1, minQuantity: 1 };
  const m = constantModel([.1, -.99]); m.kernels[0][0].probability = 1; m.kernels[0][1].probability = 1e-300;
  const account = { equity: 100, price: 10, exposure: 0 };
  const reference = eventBellmanReference(m, c, { terminal: "marked", maxOrderLots: 10000 }).decide(0, account, 2);
  const result = decideEventTwoStep(m, 0, account, c, "marked", { maxEvaluations: 128 });
  assert.ok(result.converged && Number.isFinite(result.value)); near(result.value, reference.value, 1e-10);
  assert.equal(eventHolding(result.exposure, m.kernels[0][1], c).liquidated, false);
});

test("continuous shadow-price tangents bound integer orders across different portfolios", () => {
  const c = { ...DEFAULT_EVENT_COSTS, maxLeverage: 2, feeBps: 100, slippageBps: 50, minNotional: 12,
    maxNotional: 500, quantityStep: .1, minQuantity: .1, longBorrowBpsPerDay: 100, shortBorrowBpsPerDay: 50 };
  const m = constantModel([.1, -.08]); m.kernels[0][0].probability = .6; m.kernels[0][1].probability = .4;
  m.kernels[0].forEach(a => { a.duration = 1440; });
  for (const terminal of ["marked", "friction"] as const) {
    const relax = eventOneStepUpper(m.kernels[0], c, terminal);
    for (const exposure of [-2, 0, .5, 1, 2]) {
      const at = { cash: 100 * (1 - exposure), quantity: exposure * 10 }, tangent = relax(at.cash, at.quantity, 10)!;
      for (const equity of [50, 100, 200]) for (const x of [-2, -.5, 0, .5, 1, 2]) {
        const cash = equity * (1 - x), quantity = equity * x / 10;
        const upper = tangent.value + tangent.dCash * (cash - at.cash) + tangent.dQuantity * (quantity - at.quantity);
        const exact = eventBellmanReference(m, c, { terminal }).decide(0, { equity, price: 10, exposure: x }, 1);
        assert.ok(upper >= Math.log(equity) + exact.value - 1e-10);
      }
    }
  }
});

test("capacity tangents stay global and close the gap when the next order is size-limited", () => {
  const c = { ...DEFAULT_EVENT_COSTS, maxLeverage: 5, feeBps: 12, slippageBps: 3,
    minNotional: 1, maxNotional: 20, quantityStep: .1, minQuantity: .1,
    longBorrowBpsPerDay: 30, shortBorrowBpsPerDay: 70 };
  for (const sign of [-1, 1]) for (const terminal of ["marked", "friction"] as const) for (const copies of [1, 36]) {
    const m = constantModel([sign * .03, sign * .04]);
    m.kernels[0] = Array.from({ length: copies }, () => m.kernels[0].map(a =>
      ({ ...a, probability: a.probability / copies, duration: 1440 }))).flat();
    const relax = eventOneStepUpper(m.kernels[0], c, terminal);
    const reference = eventBellmanReference(m, c, { terminal });
    for (const atExposure of [-3, 0, 3]) {
      const at = { cash: 100 * (1 - atExposure), quantity: atExposure * 10 };
      const tangent = relax(at.cash, at.quantity, 10)!;
      const optimum = reference.decide(0, { equity: 100, price: 10, exposure: atExposure }, 1);
      near(optimum.quantity, sign * 2, 1e-12);
      near(tangent.value, Math.log(100) + optimum.value, 1e-9);
      for (const equity of [30, 100, 300]) for (const exposure of [-5, -2, 0, 1, 3, 5]) {
        const cash = equity * (1 - exposure), quantity = equity * exposure / 10;
        const upper = tangent.value + tangent.dCash * (cash - at.cash) + tangent.dQuantity * (quantity - at.quantity);
        const actual = reference.decide(0, { equity, price: 10, exposure }, 1);
        assert.ok(upper >= Math.log(equity) + actual.value - 1e-10);
      }
    }
  }
});

test("exhaustive event Bellman reference matches binary Kelly and never observes the future sign", () => {
  const c = { ...DEFAULT_EVENT_COSTS, maxLeverage: 5, feeBps: 0, slippageBps: 0,
    longBorrowBpsPerDay: 0, shortBorrowBpsPerDay: 0, minNotional: 1, maxNotional: 500, quantityStep: .1, minQuantity: .1 };
  const m = constantModel([.1, -.1]); m.kernels[0][0].probability = .6; m.kernels[0][1].probability = .4;
  const account = { equity: 100, price: 10, exposure: 0 }, reference = eventBellmanReference(m, c);
  const best = reference.decide(0, account, 1);
  near(best.quantity, 20); near(best.value, .6 * Math.log(1.2) + .4 * Math.log(.8));
  near(reference.evaluatePolicy(0, account, 1, () => 20), best.value);
  m.kernels[0][0].probability = .5; m.kernels[0][1].probability = .5;
  const fair = eventBellmanReference(m, { ...c, maxNotional: 50, quantityStep: 1, minQuantity: 1 });
  near(fair.decide(0, account, 2).value, 0); near(fair.decide(0, account, 2).quantity, 0);
  m.kernels[0][1] = { ...m.kernels[0][1], return: -.99, low: -.99 };
  assert.equal(eventBellmanReference(m, c).actionValue(0, account, 1, 20), -Infinity);
  assert.throws(() => eventBellmanReference(m, c, { maxOrderLots: 1 }).decide(0, account, 1), /lot budget/);
  assert.throws(() => eventBellmanReference(m, c, { maxNodes: 0 }).decide(0, account, 1), /node budget/);
});

test("exhaustive event reference distinguishes proportional settlement from untradeable dust", () => {
  const c = { ...DEFAULT_EVENT_COSTS, maxLeverage: 1, feeBps: 10, slippageBps: 0,
    minNotional: 25, maxNotional: 50, quantityStep: .1, minQuantity: .1 };
  const m = constantModel([0]), account = { equity: 100, price: 100, exposure: .2 };
  const friction = eventBellmanReference(m, c).decide(0, account, 1);
  const market = eventBellmanReference(m, c, { terminal: "market" }).decide(0, account, 1);
  near(friction.value, Math.log(1 - .2 * .001)); near(market.value, 0);
  assert.equal(friction.quantity, 0); assert.equal(market.quantity, 0);
});

test("prepared one-event targets preserve lot, minimum-order and recovery optima across accounts", () => {
  const c = { ...DEFAULT_EVENT_COSTS, maxLeverage: 2, minNotional: 12, maxNotional: 80, quantityStep: .1, minQuantity: .3,
    longBorrowBpsPerDay: 100, shortBorrowBpsPerDay: 25 };
  const m = constantModel([.08, -.06]); m.kernels[0][0].probability = .6; m.kernels[0][1].probability = .4;
  m.kernels[0].forEach(a => { a.duration = 1440; a.low = -.25; a.high = .3; });
  for (const terminal of ["marked", "friction"] as const) {
    const compiled = prepareEventOneStep(m.kernels[0], c, terminal);
    for (const equity of [50, 100, 250]) for (const price of [7, 30]) for (const exposure of [-3, -1, 0, .5, 1, 1.8, 3]) {
      const account = { equity, price, exposure }, expected = eventBellmanReference(m, c, { terminal }).decide(0, account, 1);
      const actual = compiled(account);
      assert.ok(actual.value === expected.value || Math.abs(actual.value - expected.value) < 1e-10);
      assert.equal(compiled.value(account), actual.value);
      if (Number.isFinite(expected.value)) near(actual.quantity, expected.quantity, 1e-8);
    }
  }
});

test("prepared one-event optimization resolves billions of lots and retains rare survival constraints", () => {
  const c = { ...DEFAULT_EVENT_COSTS, maxLeverage: 5, longBorrowBpsPerDay: 0, shortBorrowBpsPerDay: 0,
    minNotional: 1, maxNotional: 500, quantityStep: 1e-8, minQuantity: 1e-8 };
  const m = constantModel([.1, -.1]); m.kernels[0][0].probability = .6; m.kernels[0][1].probability = .4;
  const account = { equity: 100, price: 10, exposure: 0 };
  const expected = decideEventOneStep(m.kernels[0], account, c, "friction"), actual = prepareEventOneStep(m.kernels[0], c, "friction")(account);
  near(actual.value, expected.value, 1e-12); assert.ok(actual.search.evaluatedOrders < 50);
  const rare = constantModel([.1, -.99]); rare.kernels[0][0].probability = 1; rare.kernels[0][1].probability = 1e-300;
  const coarse = { ...c, quantityStep: 1, minQuantity: 1 };
  const limited = prepareEventOneStep(rare.kernels[0], coarse, "marked")(account);
  near(limited.value, eventBellmanReference(rare, coarse, { terminal: "marked" }).decide(0, account, 1).value, 1e-10);
  assert.equal(eventHolding(limited.exposure, rare.kernels[0][1], coarse).liquidated, false);
  assert.throws(() => prepareEventOneStep(m.kernels[0], c, "market" as never), /Invalid prepared/);
});

test("prepared scalar continuation keeps no-trade, forced recovery and ruin semantics", () => {
  const c = { ...DEFAULT_EVENT_COSTS, maxLeverage: 1, quantityStep: .1, minQuantity: .1,
    minNotional: 5, maxNotional: 40, longBorrowBpsPerDay: 0, shortBorrowBpsPerDay: 0 };
  const model = constantModel([-.005, .005]);
  for (const copies of [1, 64]) for (const terminal of ["marked", "friction"] as const) {
    const kernel = model.kernels[0].flatMap(a => Array.from({ length: copies }, () => ({ ...a, probability: a.probability / copies })));
    const prepared = prepareEventOneStep(kernel, c, terminal);
    for (const exposure of [-10, -1, -.2, 0, .2, 1, 1.000001, 10]) {
      const account = { equity: 100, price: 10, exposure };
      const expected = eventBellmanReference(model, c, { terminal }).decide(0, account, 1);
      const actual = prepared.value(account);
      assert.ok(actual === expected.value || Math.abs(actual - expected.value) < 1e-10);
    }
    assert.equal(prepared({ equity: 100, price: 10, exposure: 0 }).quantity, 0);
  }
  const singular = { ...c, feeBps: 9900, slippageBps: 0 }, account = { equity: 100, price: 10, exposure: 0 };
  assert.equal(prepareEventOneStep(model.kernels[0], singular, "marked").value(account),
    decideEventOneStep(model.kernels[0], account, singular, "marked").value);
});

test("one-event hold shortcut respects the borrowing kink and preserves the exact optimum", () => {
  const c = { ...DEFAULT_EVENT_COSTS, maxLeverage: 2, minNotional: 1, maxNotional: 100, quantityStep: .1,
    minQuantity: .1, longBorrowBpsPerDay: 1000, shortBorrowBpsPerDay: 0 };
  const m = constantModel([0]); m.kernels[0][0].duration = 1440;
  for (const exposure of [-1, 0, .5, 1, 1.2]) {
    const account = { equity: 100, price: 10, exposure };
    const exact = eventBellmanReference(m, c, { terminal: "marked" }).decide(0, account, 1);
    const result = decideEventOneStep(m.kernels[0], account, c, "marked");
    near(result.value, exact.value, 1e-10); near(result.quantity, exact.quantity);
    if (exposure <= 1) { assert.equal(result.quantity, 0); assert.equal(result.search.evaluatedOrders, 1); }
    else assert.ok(result.quantity < 0);
  }
});

test("one-event lot search matches analytical fee-aware sizing on a five-billion-lot lattice", () => {
  const c = { ...DEFAULT_EVENT_COSTS, maxLeverage: 5, longBorrowBpsPerDay: 0, shortBorrowBpsPerDay: 0,
    minNotional: 1, maxNotional: 500, quantityStep: 1e-8, minQuantity: 1e-8 };
  const m = constantModel([.1, -.1]); m.kernels[0][0].probability = .6; m.kernels[0][1].probability = .4;
  const account = { equity: 100, price: 10, exposure: 0 }, f = (c.feeBps + c.slippageBps) / 10000;
  const up = .1 - f * 2.1, down = -.1 - f * 1.9, optimum = -(.6 * up + .4 * down) / (up * down);
  const quantity = optimum * account.equity / account.price;
  const expected = (q: number) => .6 * Math.log1p(up * q / 10) + .4 * Math.log1p(down * q / 10);
  const lower = Math.floor(quantity / c.quantityStep) * c.quantityStep;
  const result = decideEventOneStep(m.kernels[0], account, c, "friction");
  near(result.value, Math.max(expected(lower), expected(lower + c.quantityStep)), 1e-12);
  assert.ok(result.feasible && result.longEntry && !result.shortEntry);
  assert.ok(result.search.evaluatedOrders < 200 && result.search.derivativeEvaluations < 200);
  const coarse = { ...c, quantityStep: .1, minQuantity: .1 };
  const enumerated = eventBellmanReference(m, coarse).decide(0, account, 1);
  const actual = decideEventOneStep(m.kernels[0], account, coarse, "friction");
  near(actual.quantity, enumerated.quantity); near(actual.value, enumerated.value);
});

test("one-event market settlement charges exact minimum lots and retains rare ruin support", () => {
  const c = { ...DEFAULT_EVENT_COSTS, maxLeverage: 5, feeBps: 100, slippageBps: 50,
    minNotional: 48.05675793327391, maxNotional: 60.0709474165924,
    minQuantity: .9921691650956175, quantityStep: .3307230550318725 };
  const m = constantModel([.1, -.1]); m.kernels[0][0].probability = .8; m.kernels[0][1].probability = .2;
  const account = { equity: 400.4729827772826, price: 60.545065831393, exposure: 0 };
  const reference = eventBellmanReference(m, c, { terminal: "market" }).decide(0, account, 1);
  const result = decideEventOneStep(m.kernels[0], account, c, "market");
  near(result.quantity, c.minQuantity); near(result.value, reference.value, 1e-12);
  const extreme = constantModel([.1, -.99]); extreme.kernels[0][0].probability = 1; extreme.kernels[0][1].probability = 1e-300;
  const bounded = decideEventOneStep(extreme.kernels[0], { equity: 100, price: 10, exposure: 0 },
    { ...c, maxNotional: 500, minNotional: 1, minQuantity: .1, quantityStep: .1 }, "friction");
  assert.ok(Number.isFinite(bounded.value));
  assert.equal(eventHolding(bounded.exposure, extreme.kernels[0][1], c).liquidated, false);
});

test("marked one-step wealth pays current fees without assuming a second transaction", () => {
  const c = { ...DEFAULT_EVENT_COSTS, maxLeverage: 1, minNotional: 1, maxNotional: 100,
    minQuantity: .001, quantityStep: .001, longBorrowBpsPerDay: 0, shortBorrowBpsPerDay: 0 };
  const m = constantModel([.0018]), account = { equity: 100, price: 10, exposure: 0 };
  const marked = decideEventOneStep(m.kernels[0], account, c, "marked"), friction = decideEventOneStep(m.kernels[0], account, c, "friction");
  assert.ok(marked.quantity > 0 && marked.value > 0); assert.equal(friction.quantity, 0); assert.equal(friction.value, 0);
  const fee = marked.turnover * (c.feeBps + c.slippageBps) / 10000;
  near(marked.value, Math.log((100 - fee + marked.quantity * 10 * .0018) / 100));
  const reference = eventBellmanReference(m, c, { terminal: "marked", maxOrderLots: 20000 }).decide(0, account, 1);
  near(reference.value, marked.value); near(reference.quantity, marked.quantity);
});

test("return-weighted sign fitting learns economic balance and inverts to the original sign law", () => {
  const rows = Array.from({ length: 100 }, (_, i) => ({ features: [0], return: i < 80 ? 0.01 : -0.08 }));
  const ordinary = trainEventSign(rows, 0.1), weights = rows.map(r => Math.abs(r.return));
  const weighted = trainEventSign(rows, 0.1, weights), q = predictEventSign(weighted, [0]);
  near(predictEventSign(ordinary, [0]), 0.8, 1e-6);
  near(q, 1 / 3, 1e-6); assert.equal(weighted.objective, "weighted-sign");
  near(eventProbabilityFromReturnWeight(q, 0.01, -0.08), 0.8, 1e-6);
  near(predictEventSign(trainEventSign(rows, 0.1, weights.map(w => w * 1000)), [0]), q, 1e-10);
  near(eventProbabilityFromReturnWeight(0.5, 0.01, -0.08), 8 / 9);
  near(eventProbabilityFromReturnWeight(0.25, 0.02, -0.02), 0.25);
  assert.throws(() => trainEventSign(rows, 0.1, [1]), /weights/);
  assert.throws(() => trainEventSign(rows, 0.1, weights.map(() => 0)), /weight total/);
  assert.throws(() => eventProbabilityFromReturnWeight(q, 0, -0.08), /inversion/);
});
function constantModel(returns: number[]): EventDistribution {
  return { version: 1, clock: { thresholdBps: 20, maxCandles: 60 }, featureNames: EVENT_FEATURES,
    nodes: [{ feature: -1, cut: 0, left: -1, right: -1, leaf: 0 }], counts: [returns.length], trainingSamples: returns.length,
    classProbabilities: [], priorClasses: [], kernels: [returns.map(r => ({ return: r, low: Math.min(0, r),
      high: Math.max(0, r), duration: 10, probability: 1 / returns.length, next: 0 }))] };
}

test("vector value boosting learns conditional interactions, preserves tied groups and serializes", () => {
  const features = Array.from({ length: 400 }, (_, i) => [i % 4 < 2 ? -1 : 1, i % 2 ? 1 : -1]);
  const targets = features.map(([x, y]) => { const v = x > 0 && y > 0 ? 0.02 : -0.01; return [v, -v, 0]; });
  const options = { trees: 40, depth: 2, minLeaf: 50, rate: 0.2 };
  const model = trainEventValueBoost(features, targets, options);
  const restored = JSON.parse(JSON.stringify(model));
  for (let i = 0; i < 4; i++) {
    const prediction = predictEventValueBoost(model, features[i]);
    prediction.forEach((v, j) => near(v, targets[i][j], 1e-5));
    assert.deepEqual(predictEventValueBoost(restored, features[i]), prediction);
  }
  assert.ok(model.trees.every(tree => tree.filter(n => n.feature < 0).every(n => n.samples >= options.minLeaf)));
  assert.throws(() => trainEventValueBoost(features, targets, { ...options, minLeaf: 0 }), /Invalid/);
  assert.throws(() => predictEventValueBoost(model, [NaN, 0]), /Invalid/);
});

test("boosted holding values retain cash and ruin constraints through a second Bellman backup", () => {
  const model = constantModel([-0.01, 0.02]), base = buildEventPolicy(model, { ...costs, maxLeverage: 1 },
    { depths: 2, referenceEquity: 1000, referencePrice: 100, actionSteps: 2 });
  const rows = Array.from({ length: 200 }, (_, i) => {
    const features = [i % 4 < 2 ? -1 : 1, i % 2 ? 1 : -1], move = model.kernels[0][features.every(v => v > 0) ? 1 : 0];
    return { features, nextFeatures: [0, 0], leaf: 0, nextLeaf: 0, move };
  });
  const options = { trees: 32, depth: 2, minLeaf: 25, rate: 0.2 };
  const fitted = trainEventFittedValue(base, rows, 1, 2, options), account = { equity: 1000, price: 100, exposure: 0 };
  assert.ok(decideFittedEvent(fitted, [1, 1], account, 1, 0).longEntry);
  assert.ok(decideFittedEvent(fitted, [-1, 1], account, 1, 0).shortEntry);
  assert.equal(fitted.tables[0].correction!.outputs, base.exposures.length);
  assert.equal(fitted.tables[1].correction!.outputs, base.exposures.length * base.prices.length * base.equities.length);
  assert.deepEqual(decideFittedEvent(JSON.parse(JSON.stringify(fitted)), [1, 1], account, 2, 0), decideFittedEvent(fitted, [1, 1], account, 2, 0));
  const rare = constantModel([-0.01, 10]); rare.kernels[0][1].probability = 1e-300; rare.kernels[0][0].probability = 1;
  const guarded = trainEventFittedValue({ ...base, model: rare }, rows, 1, 1, options);
  assert.equal(fittedEventHolding(guarded, predictEventHoldingGrid(guarded, [-1, 1], 1), { ...account, exposure: -1 }, 0), -Infinity);
});

test("fitted Bellman values average outcomes before selecting actions and retain the complete account grid", () => {
  const model = constantModel([-0.01, 0.01]), p = buildEventPolicy(model, { ...costs, maxLeverage: 1 },
    { depths: 2, referenceEquity: 1000, referencePrice: 100, actionSteps: 2 });
  const rows = model.kernels[0].map(move => ({ features: [0], nextFeatures: [0], leaf: 0, nextLeaf: 0, move }));
  const fitted = trainEventFittedValue(p, rows, 0.1, 2), grid = predictEventHoldingGrid(fitted, [0], 1);
  assert.equal(grid.length, p.equities.length * p.prices.length * p.exposures.length);
  for (let i = 0; i < grid.length; i++) {
    const exposure = p.exposures[i % p.exposures.length];
    const expected = rows.reduce((s, row) => {
      const h = eventHolding(exposure, row.move, p.costs);
      return s + (Math.log(h.factor) + Math.log(1 - Math.abs(h.exposure) * (p.costs.feeBps + p.costs.slippageBps) / 1e4)) / rows.length;
    }, 0);
    near(grid[i], expected, 1e-12);
  }
  const account = { equity: 1000, price: 100, exposure: 0 };
  assert.equal(decideFittedEvent(fitted, [0], account, 1, 0).quantity, 0);
  assert.equal(decideFittedEvent(fitted, [0], account, 2, 0).quantity, 0);
  const second = predictEventHoldingGrid(fitted, [0], 2);
  for (const exposure of [-1, 0, 1]) {
    const direct = rows.reduce((s, row) => {
      const h = eventHolding(exposure, row.move, p.costs);
      return s + (Math.log(h.factor) + decideFittedEvent(fitted, [0], { equity: account.equity * h.factor,
        price: account.price * (1 + row.move.return), exposure: h.exposure }, 1, 0).value) / rows.length;
    }, 0);
    near(fittedEventHolding(fitted, second, { ...account, exposure }, 0), direct, 1e-12);
  }
  assert.deepEqual(decideFittedEvent(JSON.parse(JSON.stringify(fitted)), [0], account, 2, 0), decideFittedEvent(fitted, [0], account, 2, 0));
});

test("sampled Bellman targets evaluate a frozen action before observing its next return", () => {
  const model = constantModel([0.02]), base = buildEventPolicy(model, { ...costs, maxLeverage: 1 },
    { depths: 3, referenceEquity: 1000, referencePrice: 100, actionSteps: 2 });
  const transition = (r: number) => ({ features: [0], nextFeatures: [0], leaf: 0, nextLeaf: 0,
    move: { return: r, low: Math.min(0, r), high: Math.max(0, r), duration: 10 } });
  const rows = [transition(0.02), transition(0.02)];
  const following = [[transition(-0.1), transition(0.03)], [transition(0.1), transition(-0.03)]];
  const one = trainEventFittedValue(base, rows, 0.1, 1);
  const fitted = trainEventFittedValue(base, rows, 0.1, 2, undefined, undefined, { following });
  assert.deepEqual(fitted.tables[0], one.tables[0]);
  const account = { equity: 1000, price: 102, exposure: 0 };
  const action = decideFittedEvent(one, [0], account, 1, 0);
  assert.ok(action.exposure > 0.9);
  const expected = following.reduce((sum, tail) => {
    const held = eventHolding(action.exposure, tail[0].move, one.costs);
    return sum + Math.log(action.equity / account.equity) + Math.log(held.factor)
      + Math.log(1 - Math.abs(held.exposure) * (costs.feeBps + costs.slippageBps) / 10000);
  }, 0) / following.length;
  const cashIndex = ((2 * base.prices.length + 1) * base.exposures.length) + base.exposures.indexOf(0);
  near(fitted.tables[1].coefficients[cashIndex]![0], expected, 1e-9);
  assert.ok(expected < 0);
  near(fittedEventHolding(fitted, predictEventHoldingGrid(fitted, [0], 2), account, 0), 0);
  const resumed = trainEventFittedValue(base, rows, 0.1, 3, undefined, fitted, { following });
  assert.deepEqual(resumed, trainEventFittedValue(base, rows, 0.1, 3, undefined, undefined, { following }));
  const changed = structuredClone(following); changed[0][1].move.return = 0.025;
  assert.throws(() => trainEventFittedValue(base, rows, 0.1, 3, undefined, fitted, { following: changed }), /checkpoint/);
  assert.throws(() => trainEventFittedValue(base, rows, 0.1, 3, undefined, fitted), /checkpoint/);
  assert.throws(() => trainEventFittedValue(base, rows, 0.1, 3, undefined, undefined, { following: following.map(p => p.slice(0, 1)) }), /continuation/);
});

test("controller comparison maximizes after account interpolation and keeps deterministic ties", () => {
  const base = buildEventPolicy(constantModel([-.01, .01]), { ...costs, maxLeverage: 1, feeBps: 0 },
    { depths: 1, referenceEquity: 1000, referencePrice: 100, actionSteps: 2 });
  const row = { features: [0], nextFeatures: [0], leaf: 0, nextLeaf: 0, move: { return: .01, low: 0, high: .01, duration: 1 } };
  const a = trainEventFittedValue(base, [row, row], .1, 1), b = structuredClone(a), stride = a.prices.length * a.exposures.length;
  a.tables[0].coefficients = a.tables[0].coefficients.map((_, i) => [Number(Math.floor(i / stride) === 0), 0]);
  b.tables[0].coefficients = b.tables[0].coefficients.map((_, i) => [Number(Math.floor(i / stride) !== 0), 0]);
  const account = { equity: (a.equities[0] + a.equities[1]) / 2, price: 100, exposure: 0 };
  const result = decideFittedEventControllers([a, b], [0], account, 1, 0);
  near(result.decision.value, .5); assert.equal(result.controller, 0);
  assert.deepEqual(result.holdingValues, [.5, .5]);
  assert.throws(() => decideFittedEventControllers([a, { ...b, costs: { ...b.costs, feeBps: 5 } }], [0], account, 1, 0), /controller accounts/);
});

test("holding-option targets preserve cash and charge mandatory cap reductions before the next move", () => {
  const base = buildEventPolicy(constantModel([-.1, .1]), { ...costs, maxLeverage: 1 },
    { depths: 2, referenceEquity: 1000, referencePrice: 100, actionSteps: 2 });
  const step = { features: [0], nextFeatures: [0], leaf: 0, nextLeaf: 0, move: { return: .1, low: 0, high: .1, duration: 10 } };
  const rows = [step, step], following = [[step], [step]], rollout = { following, minimumTurnover: true as const };
  const fitted = trainEventFittedValue(base, rows, .1, 2, undefined, undefined, rollout);
  assert.deepEqual(fitted.tables[0], trainEventFittedValue(base, rows, .1, 1).tables[0]);
  const account = { equity: base.equities[2], price: base.prices[1], exposure: -1 };
  const first = eventHolding(-1, step.move, base.costs);
  const next = { equity: account.equity * first.factor, price: account.price * 1.1, exposure: first.exposure };
  const trade = chooseEventTrade(base, next, () => 0);
  assert.ok(trade.cost > 0 && trade.shortExit && Math.abs(trade.exposure) <= 1 + 1e-8);
  const second = eventHolding(trade.exposure, step.move, base.costs);
  const expected = Math.log(first.factor) + Math.log(trade.equity / next.equity) + Math.log(second.factor)
    + Math.log(1 - Math.abs(second.exposure) * (costs.feeBps + costs.slippageBps) / 10000);
  const index = ((2 * base.prices.length + 1) * base.exposures.length) + base.exposures.indexOf(-1);
  near(fitted.tables[1].coefficients[index]![0], expected, 1e-10);
  near(fittedEventHolding(fitted, predictEventHoldingGrid(fitted, [0], 2), { ...account, exposure: 0 }, 0), 0);
  assert.throws(() => trainEventFittedValue(base, rows, .1, 2, undefined, fitted, { following }), /checkpoint/);
  assert.throws(() => trainEventFittedValue(base, rows, .1, 2, undefined, undefined, { ...rollout, policies: [fitted, fitted] }), /continuation heads/);
});

test("held-out continuation heads keep H1 fixed, evaluate the supplied row policy and bind checkpoints", () => {
  const base = buildEventPolicy(constantModel([-0.02, 0.02]), { ...costs, maxLeverage: 1 },
    { depths: 2, referenceEquity: 1000, referencePrice: 100, actionSteps: 2 });
  const transition = (r: number) => ({ features: [0], nextFeatures: [0], leaf: 0, nextLeaf: 0,
    move: { return: r, low: Math.min(0, r), high: Math.max(0, r), duration: 10 } });
  const rows = [transition(0.02), transition(0.02)], following = [[transition(-0.1)], [transition(0.1)]];
  const up = trainEventFittedValue(base, rows, 0.1, 1), down = trainEventFittedValue(base, [transition(-0.02), transition(-0.02)], 0.1, 1);
  const rollout = { following, policies: [up, down] };
  const first = trainEventFittedValue(base, rows, 0.1, 1, undefined, undefined, rollout);
  const fitted = trainEventFittedValue(base, rows, 0.1, 2, undefined, first, rollout);
  assert.deepEqual(fitted, trainEventFittedValue(base, rows, 0.1, 2, undefined, undefined, rollout));
  assert.deepEqual(fitted.tables[0], up.tables[0]);
  const account = { equity: 1000, price: 102, exposure: 0 };
  const expected = rollout.policies.reduce((sum, p, i) => {
    const action = decideFittedEvent(p, [0], account, 1, 0);
    assert.ok(i ? action.exposure < -0.9 : action.exposure > 0.9);
    const held = eventHolding(action.exposure, following[i][0].move, p.costs);
    return sum + Math.log(action.equity / account.equity) + Math.log(held.factor)
      + Math.log(1 - Math.abs(held.exposure) * (costs.feeBps + costs.slippageBps) / 10000);
  }, 0) / rows.length;
  const cash = ((2 * base.prices.length + 1) * base.exposures.length) + base.exposures.indexOf(0);
  near(fitted.tables[1].coefficients[cash]![0], expected, 1e-9);
  assert.ok(expected < -0.09); // Both frozen policies choose the wrong side; no hindsight maximum.
  assert.throws(() => trainEventFittedValue(base, rows, 0.1, 2, undefined, first, { following, policies: [down, up] }), /checkpoint/);
  const changed = structuredClone(up); changed.tables[0].coefficients[0]![0] += 0.001;
  assert.throws(() => trainEventFittedValue(base, rows, 0.1, 2, undefined, first, { following, policies: [changed, down] }), /checkpoint/);
  assert.throws(() => trainEventFittedValue(base, rows, 0.1, 2, undefined, first, { following }), /checkpoint/);
  assert.throws(() => trainEventFittedValue(base, rows, 0.1, 2, undefined, undefined, { following, policies: [up] }), /count/);
  const bad = { ...up, costs: { ...up.costs, feeBps: up.costs.feeBps + 1 } };
  assert.throws(() => trainEventFittedValue(base, rows, 0.1, 2, undefined, undefined, { following, policies: [bad, down] }), /Incompatible/);
});

test("fitted depth checkpoints reproduce uninterrupted training and reject changed transitions", () => {
  const model = constantModel([-0.01, 0.02]), base = buildEventPolicy(model, { ...costs, maxLeverage: 1 },
    { depths: 1, referenceEquity: 1000, referencePrice: 100, actionSteps: 2 });
  const rows = model.kernels[0].map((move, i) => ({ features: [i], nextFeatures: [1 - i], leaf: 0, nextLeaf: 0, move }));
  const checkpoint = trainEventFittedValue(base, rows, 0.1, 1), snapshot = JSON.stringify(checkpoint);
  const resumed = trainEventFittedValue(base, rows, 0.1, 3, undefined, JSON.parse(snapshot));
  assert.deepEqual(resumed, trainEventFittedValue(base, rows, 0.1, 3));
  assert.equal(JSON.stringify(checkpoint), snapshot);
  const changed = rows.map(r => ({ ...r, move: { ...r.move, duration: r.move.duration + 1 } }));
  assert.throws(() => trainEventFittedValue(base, changed, 0.1, 3, undefined, checkpoint), /checkpoint/);
  assert.throws(() => trainEventFittedValue(base, rows, 1, 3, undefined, checkpoint), /checkpoint/);
  assert.throws(() => trainEventFittedValue(base, rows, 0.1, 3, undefined, { ...checkpoint, trainingSignature: undefined }), /checkpoint/);
});

test("fitted cash continuation retains its feasible zero lower bound without erasing positive option value", () => {
  const model = constantModel([-0.01, 0.01]), base = buildEventPolicy(model, { ...costs, maxLeverage: 1 },
    { depths: 1, referenceEquity: 1000, referencePrice: 100 });
  const p = trainEventFittedValue(base, model.kernels[0].map(move => ({ features: [0], nextFeatures: [0], leaf: 0, nextLeaf: 0, move })), 0.1, 1);
  // Simulate an unconstrained regressor's negative/positive extrapolation.
  p.tables[0].coefficients = p.tables[0].coefficients.map(() => [0, 0.01]);
  const negative = predictEventHoldingGrid(p, [-1], 1), positive = predictEventHoldingGrid(p, [1], 1);
  for (let i = 0; i < negative.length; i++) {
    if (p.exposures[i % p.exposures.length] === 0) assert.equal(negative[i], 0);
    else assert.ok(negative[i] < 0);
    assert.ok(positive[i] > 0);
  }
  assert.equal(decideFittedEvent(p, [-1], { equity: 1000, price: 100, exposure: 0 }, 1, 0).quantity, 0);
});

test("fitted values use observed features but cannot remove a positive-probability liquidation path", () => {
  const model = constantModel([-0.02, 0.02]), p = buildEventPolicy(model, { ...costs, maxLeverage: 1 },
    { depths: 1, referenceEquity: 1000, referencePrice: 100, actionSteps: 5 });
  const rows = Array.from({ length: 100 }, (_, i) => { const move = model.kernels[0][i % 2];
    return { features: [Math.sign(move.return)], nextFeatures: [0], leaf: 0, nextLeaf: 0, move }; });
  const fitted = trainEventFittedValue(p, rows, 0.01, 1), account = { equity: 1000, price: 100, exposure: 0 };
  assert.ok(decideFittedEvent(fitted, [1], account, 1, 0).longEntry);
  assert.ok(decideFittedEvent(fitted, [-1], account, 1, 0).shortEntry);
  const rare = constantModel([-0.02, 10]); rare.kernels[0][1].probability = 1e-300; rare.kernels[0][0].probability = 1;
  const guarded = trainEventFittedValue({ ...p, model: rare }, rows, 0.01, 1), limit = guarded.limits[0].short!;
  const grid = predictEventHoldingGrid(guarded, [-1], 1);
  assert.equal(fittedEventHolding(guarded, grid, { ...account, exposure: -1 }, 0), -Infinity);
  assert.ok(!eventHolding(-limit * (1 - 1e-8), rare.kernels[0][1], p.costs).liquidated);
  assert.ok(eventHolding(-limit * (1 + 1e-8), rare.kernels[0][1], p.costs).liquidated);
  assert.ok(decideFittedEvent(guarded, [-1], account, 1, 0).exposure > -limit);
});

test("positive quadrature preserves conditional moments, observed paths and rare ruin excursions", () => {
  const kernel = Array.from({ length: 600 }, (_, i) => {
    const r = 0.0065 + ((i * 73) % 199) / 10000, duration = 21 + ((i * 11) % 101);
    return { return: r, low: -0.002 - ((i * 29) % 77) / 10000, high: r + 0.002, duration, next: i % 3, probability: 1 / 600 };
  });
  kernel[0].low = -0.3; kernel[0].probability = 1e-300;
  kernel[1].probability += 1 / 600 - 1e-300;
  const snapshot = JSON.stringify(kernel), reduced = compressEventKernel(kernel, { thresholdBps: 60, maxCandles: 100 });
  assert.equal(JSON.stringify(kernel), snapshot);
  assert.ok(reduced.length < kernel.length / 3, `${reduced.length} paths were retained`);
  assert.ok(reduced.some(a => a.low === -0.3 && a.probability === 1e-300));
  assert.ok(reduced.every(a => a.probability > 0 && kernel.some(b => a.return === b.return && a.low === b.low && a.high === b.high && a.duration === b.duration && a.next === b.next)));
  for (const next of [0, 1, 2]) {
    const a = kernel.filter(a => a.next === next), b = reduced.filter(a => a.next === next);
    for (let j = 0; j < eventQuadratureFeatures(a[0]).length; j++) {
      const expected = a.reduce((s, p) => s + p.probability * eventQuadratureFeatures(p)[j], 0);
      const actual = b.reduce((s, p) => s + p.probability * eventQuadratureFeatures(p)[j], 0);
      near(actual, expected, 1e-9 * Math.max(1, Math.abs(expected)));
    }
    near(Math.min(...a.map(a => a.low)), Math.min(...b.map(a => a.low)), 1e-15);
    near(Math.max(...a.map(a => a.high)), Math.max(...b.map(a => a.high)), 1e-15);
  }
  const allSame = Array.from({ length: 100 }, () => ({ ...kernel[5], probability: 0.01 }));
  const sameReduced = compressEventKernel(allSame, { thresholdBps: 60, maxCandles: 100 });
  near(sameReduced.reduce((s, a) => s + a.probability, 0), 1, 1e-12);
  assert.ok(sameReduced.length <= 9);
  assert.throws(() => compressEventKernel([{ ...kernel[0], probability: -1 }], { thresholdBps: 60, maxCandles: 100 }));
  const borrowing = Array.from({ length: 300 }, (_, i) => ({ return: 0.01 + i / 1e6, low: -0.01, high: 0.02,
    duration: 30 + i % 30, next: 0, probability: 1 / 300 }));
  borrowing[0].low = -0.15; borrowing[0].duration = 21;
  const rare = { return: 0.011, low: -0.12, high: 0.02, duration: 1440, next: 0, probability: 1e-200 };
  const withRare = [...borrowing, rare], safe = compressEventKernel(withRare, { thresholdBps: 60, maxCandles: 100 });
  assert.ok(safe.some(a => a.low === rare.low && a.duration === rare.duration && a.probability === rare.probability));
  const borrowingCosts = { ...costs, longBorrowBpsPerDay: 1250 };
  assert.equal(eventHolding(5, borrowing[0], borrowingCosts).liquidated, false);
  assert.equal(eventHolding(5, rare, borrowingCosts).liquidated, true);
  assert.ok(safe.some(a => a.probability > 0 && eventHolding(5, a, borrowingCosts).liquidated));
});

test("pooled volatility laws carry observed successor bands and exclude older quiet paths", () => {
  const basic = constantModel([-0.01, 0.01]);
  const base: EventDistribution = { ...basic, clock: { thresholdBps: 60, maxCandles: 1440, reversalClock: true }, featureNames: EVENT_RUN_FEATURES,
    runConditioned: { directionPrior: 100 }, kernels: [basic.kernels[0], basic.kernels[0], basic.kernels[0]], counts: [2, 2, 2] };
  const row = (i: number, high: boolean, value: number): MoveSample => {
    const features = new Array<number>(21).fill(0), nextFeatures = [...features];
    features[12] = i % 2 ? -1 : 1; nextFeatures[12] = -features[12];
    features[20] = high ? 3 : 1; nextFeatures[20] = i % 2 ? 3 : 1;
    return { start: i * 10, end: i * 10 + 1, features, nextFeatures, return: value, low: Math.min(0, value), high: Math.max(0, value), duration: high ? 1 : 30, label: eventMoveLabel(value, high ? 1 : 30, { thresholdBps: 60, maxCandles: 100 }) };
  };
  const recent = Array.from({ length: 12 }, (_, i) => row(i, i % 3 === 0, i % 2 ? -0.01 : 0.01));
  const older = [...Array.from({ length: 6 }, (_, i) => row(i + 12, true, -0.02)), row(18, false, 0.8)];
  const model = trainEventVolatilityLaw(base, recent, older, { quantile: 0.5, prior: 10 });
  assert.equal(model.kernels.length, 6); assert.equal(model.trainingSamples, 18);
  assert.equal(model.runVolatility!.pooledHighSamples, 6);
  assert.ok(model.kernels.every(kernel => kernel.every(a => a.return !== 0.8)));
  for (const sample of [...recent, ...older.slice(0, 6)]) {
    const leaf = eventLeaf(model, sample.features), next = eventLeaf(model, sample.nextFeatures);
    assert.equal(leaf % 2, Number(sample.features[20] > model.runVolatility!.cut));
    assert.ok(model.kernels[leaf].some(a => a.return === sample.return && a.duration === sample.duration && a.next === next && a.probability > 0));
  }
  const policy = buildEventPolicy(model, costs, { depths: 2, referencePrice: 100, referenceEquity: 1000, actionSteps: 2 });
  const restored = restoreEventPolicy(serializeEventPolicy(policy)), features = recent[0].features;
  assert.deepEqual(decideEvent(policy, eventLeaf(model, features), { price: 100, equity: 1000, exposure: 0 }, 2),
    decideEvent(restored, eventLeaf(restored.model, features), { price: 100, equity: 1000, exposure: 0 }, 2));
  const rare = { ...row(420, false, 0.0001), duration: 2, label: eventMoveLabel(0.0001, 2, { thresholdBps: 60, maxCandles: 100 }) };
  const crowded = trainEventVolatilityLaw(base, [...recent, ...Array.from({ length: 400 }, (_, i) => row(i + 20, false, 0.01)), rare], [], { quantile: 0.9, prior: 100 });
  for (let leaf = 0; leaf < crowded.kernels.length; leaf += 2) {
    assert.ok(crowded.kernels[leaf].some(a => eventMoveLabel(a.return, a.duration, { thresholdBps: 60, maxCandles: 100 }) === rare.label && a.next === eventLeaf(crowded, rare.nextFeatures) && a.probability > 0));
  }
  const source = trainEventRunDistribution(recent.map(r => ({ ...r, features: r.features.slice(0, 20), nextFeatures: r.nextFeatures.slice(0, 20) })),
    base.clock, { maxDepth: 0, minLeaf: 2, prior: 2, directionPrior: 2 });
  const hybrid = retainQuietEventLaw(source, trainEventVolatilityLaw(source, recent, older, { quantile: 0.5, prior: 10 }), recent);
  const rounded = structuredClone(source), originalHigh = rounded.kernels[0][0].high;
  rounded.kernels[0][0].high += Math.max(1e-8, Math.abs(originalHigh)) * Number.EPSILON;
  assert.doesNotThrow(() => retainQuietEventLaw(rounded, hybrid, recent));
  rounded.kernels[0][0].high += 1e-6;
  assert.throws(() => retainQuietEventLaw(rounded, hybrid, recent), /no observed/);
  for (let leaf = 0; leaf < source.kernels.length; leaf++) {
    const moment = (kernel: typeof source.kernels[number], f: (a: typeof kernel[number]) => number) => kernel.reduce((s, a) => s + a.probability * f(a), 0);
    for (const f of [(a: typeof source.kernels[number][number]) => a.return, a => a.low, a => a.high, a => a.duration])
      near(moment(source.kernels[leaf], f), moment(hybrid.kernels[2 * leaf], f), 1e-12);
    source.kernels[leaf].forEach(a => assert.ok(hybrid.kernels[2 * leaf].some(b => b.return === a.return && b.duration === a.duration && Math.floor(b.next / 2) === a.next)));
  }
  const options = { depths: 1, referencePrice: 100, referenceEquity: 1000, actionSteps: 2 };
  const originalPolicy = buildEventPolicy(source, costs, options), hybridPolicy = buildEventPolicy(hybrid, costs, options);
  const account = { price: 100, equity: 1000, exposure: 0.4 };
  near(decideEvent(originalPolicy, 0, account, 1).value, decideEvent(hybridPolicy, 0, account, 1).value, 1e-12);
});

test("size/sign component selection retains the untouched conditionals and zero mass", () => {
  const base = [0.12, 0.48, 0.24, 0.06, 0.1], predicted = [0.3, 0.1, 0.15, 0.45];
  const gateOnly = selectEventSizeSignComponents(base, predicted, [1, 0, 0]);
  near(gateOnly[2] + gateOnly[3], 0.6);
  near(gateOnly[1] / (gateOnly[0] + gateOnly[1]), 0.8);
  near(gateOnly[3] / (gateOnly[2] + gateOnly[3]), 0.2);
  const signOnly = selectEventSizeSignComponents(base, predicted, [0, 1, 1]);
  near(signOnly[2] + signOnly[3], 1 / 3);
  near(signOnly[1] / (signOnly[0] + signOnly[1]), 0.25);
  near(signOnly[3] / (signOnly[2] + signOnly[3]), 0.75);
  assert.deepEqual(selectEventSizeSignComponents(base, predicted, [1, 1, 1]), predicted);
  const unchanged = mixEventSizeSigns(base, selectEventSizeSignComponents(base, predicted, [0, 0, 0]), 1);
  unchanged.forEach((v, i) => near(v, base[i], 1e-15));
  const mixed = mixEventSizeSigns(base, gateOnly, 1);
  assert.equal(mixed[4], base[4]); near(mixed.reduce((s, p) => s + p, 0), 1);
  const missing = [0, 0.6, 0.3, 0, 0.1];
  assert.deepEqual(mixEventSizeSigns(missing, selectEventSizeSignComponents(missing, predicted, [1, 0, 0]), 1), missing);
  assert.throws(() => selectEventSizeSignComponents(base, predicted, [1, 0, 2]));
});

test("conditional severity learns a positive volatility response and predicts both signs without seeing the next sign", () => {
  const threshold = 60;
  const samples = Array.from({ length: 120 }, (_, i) => {
    const rv5Bps = [5, 10, 20, 50, 100, 200][Math.floor(i / 2) % 6], sign = i % 2 ? 1 : -1;
    const magnitude = threshold * (1 + 0.1 * (1 + rv5Bps) ** 0.4 * Math.exp(sign * 0.15));
    return { rv5Bps, return: Math.expm1(sign * magnitude / 1e4) };
  });
  const head = trainEventSeverity(samples, threshold, 0.0001);
  for (const rv of [5, 20, 100]) {
    const prediction = predictEventSeverity(head, rv);
    for (const [i, sign] of [-1, 1].entries()) {
      const target = threshold * (1 + 0.1 * (1 + rv) ** 0.4 * Math.exp(sign * 0.15));
      assert.ok(Math.abs(prediction[i] / target - 1) < 0.005);
    }
    assert.ok(prediction[1] > prediction[0]);
  }
  assert.ok(predictEventSeverity(head, 100)[0] > predictEventSeverity(head, 10)[0]);
  assert.deepEqual(head, trainEventSeverity([...samples, { rv5Bps: 10000, return: 0.0001 }], threshold, 0.0001));
  assert.throws(() => predictEventSeverity(head, -1), /Invalid/);
});

test("lazy kernel backups reproduce compiled values, change forecast weights without stale cache, and reject changed paths", () => {
  const p = buildEventPolicy(constantModel([-0.03, -0.01, 0, 0.01, 0.02]), { ...costs, maxLeverage: 1 },
    { depths: 3, referenceEquity: 1000, referencePrice: 100, actionSteps: 5 });
  const lookahead = buildEventKernelLookahead(p), signs = buildEventSignLookahead(p);
  for (const depth of [1, 3]) for (const exposure of [-0.3, 0, 0.7, 1.2]) {
    const account = { equity: 1375, price: 113, exposure };
    const before = decideEvent(p, 0, account, depth), after = decideEventKernel(lookahead, 0, account, depth, p.model.kernels[0]);
    near(before.value, after.value, 1e-10); near(before.quantity, after.quantity, 1e-10);
    for (const probability of [0.2, 0.8]) {
      const predicted = decideEventKernel(lookahead, 0, account, depth, reweightEventSigns(p.model.kernels[0], probability));
      const expected = decideEventSign(signs, 0, account, depth, probability);
      near(predicted.value, expected.value, 1e-10); near(predicted.quantity, expected.quantity, 1e-10);
    }
  }
  assert.ok(lookahead.cells.size > 0);
  assert.throws(() => decideEventKernel(lookahead, 0, { equity: 1000, price: 100, exposure: 0 }, 1,
    p.model.kernels[0].map(a => ({ ...a, duration: a.duration + 1 }))), /joint atoms/);
});

test("severity tilting preserves group probabilities, joint outcomes, zero blend, and even underflowed ruin tails", () => {
  const kernel = constantModel([-0.9, -0.01, -0.001, 0, 0.001, 0.01, 0.02, 0.9]).kernels[0];
  kernel.forEach((a, i) => { a.duration = i + 1; a.next = i; });
  const changed = tiltEventSeverity(kernel, 75, [300, 500], 1);
  const before = eventSizeSignMass(kernel, 75), after = eventSizeSignMass(changed, 75);
  before.forEach((v, i) => near(v, after[i], 1e-12));
  eventSeverityMeans(changed, 75).forEach((v, i) => near(v!, [300, 500][i], 1e-6));
  for (let i = 0; i < changed.length; i++) {
    const { probability: _, ...a } = changed[i], { probability: __, ...b } = kernel[i];
    assert.deepEqual(a, b); assert.ok(changed[i].probability > 0);
    if (eventSizeSignGroup(kernel[i].return, 75) < 2 || kernel[i].return === 0) assert.equal(changed[i].probability, kernel[i].probability);
  }
  assert.deepEqual(tiltEventSeverity(kernel, 75, [300, 500], 0), kernel);
  const minimal = tiltEventSeverity(kernel, 75, [75, 75], 1);
  assert.ok(minimal[0].probability > 0 && eventHolding(2, minimal[0], costs).liquidated);
  assert.throws(() => tiltEventSeverity(kernel, 75, [74, 100], 1), /Invalid/);
});

test("action-value audit agrees with the closed-form binary log bet and excludes an over-cap hold", () => {
  const p = buildEventPolicy(constantModel([-0.1, 0.1]), { ...costs, feeBps: 0, maxLeverage: 1 },
    { depths: 1, referenceEquity: 1000, referencePrice: 100, actionSteps: 5 });
  const lookahead = buildEventSignLookahead(p), account = { equity: 1000, price: 100, exposure: 0 };
  const rows = eventActionValues(p, 0, account, 1, { lookahead, mass: [0.2, 0, 0.8] });
  near(rows.find(a => a.target === 1)!.value, 0.2 * Math.log(0.9) + 0.8 * Math.log(1.1));
  near(rows.find(a => a.target === 0)!.value, 0);
  const best = rows.reduce((a, b) => a.value > b.value ? a : b);
  near(best.value, decideEventSign(lookahead, 0, account, 1, 0.8).value);
  near(best.exposure, 1);
  assert.ok(eventActionValues(p, 0, { ...account, exposure: 1.2 }, 1).every(a => a.quantity !== 0));
  assert.throws(() => eventActionValues(p, 0, account, 1, { lookahead, mass: [0, 0, 1] }), /tail/);
});

test("value-aware fitting learns signed utility without removing outcome support or changing its incumbent", () => {
  const component = { means: [0], scales: [1], coefficients: [0], intercept: 0, penalty: 0.1, samples: 100, iterations: 0, loss: Math.log(2) };
  const base = { quantile: 0.75, thresholdLogBps: 100, samples: 100, gate: component, ordinarySign: component, largeSign: component };
  const saved = JSON.stringify(base);
  const rows = Array.from({ length: 80 }, (_, i) => ({ features: [i % 2 ? 1 : -1], offsets: [0, 0],
    coefficients: [[-0.01, 0.01, -0.04, 0.04], [0.01, -0.01, 0.04, -0.04]],
    targets: i % 2 ? [0.015, -0.015] : [-0.015, 0.015] }));
  const head = trainEventValueHead(base, rows, 0.001);
  assert.equal(JSON.stringify(base), saved);
  assert.ok(eventValueHeadMse(head, rows) < eventValueHeadMse(base, rows) * 0.01);
  assert.ok(head.valueFit.loss < head.valueFit.initialLoss);
  for (const x of [-1, 1]) {
    const p = predictEventSizeSigns(head, [x]);
    near(p.reduce((s, v) => s + v, 0), 1);
    assert.ok(p.every(v => v > 0 && v < 1));
    assert.ok(x * (p[1] + p[3] - p[0] - p[2]) > 0);
  }
  assert.throws(() => trainEventValueHead(base, [{ ...rows[0], targets: [Infinity, 0] }, rows[1]], 0.1), /Invalid/);
});

test("separate sign fitting ignores magnitudes and keeps uncertain probabilities", () => {
  const rows = Array.from({ length: 120 }, (_, i) => ({ features: [i % 2 ? 1 : -1], return: i % 2 ? 0.01 : -0.01 }));
  const head = trainEventSign(rows, 0.01);
  assert.ok(predictEventSign(head, [1]) > 0.9);
  assert.ok(predictEventSign(head, [-1]) < 0.1);
  assert.deepEqual(head, trainEventSign(rows.map((r, i) => ({ ...r, return: r.return * (i + 1) })), 0.01));
  assert.deepEqual(head, trainEventSign([...rows, { features: [100], return: 0 }], 0.01));
  const constant = trainEventSign(rows.map(r => ({ ...r, return: 1 })), 0.01);
  assert.ok(predictEventSign(constant, [1]) < 1);
  assert.throws(() => trainEventSign(rows, 0), /Invalid/);
});

test("sign replacement preserves the joint law within each sign, zero mass and tail support", () => {
  const kernel = constantModel([-0.9, -0.01, 0, 0.01, 0.03]).kernels[0];
  kernel.forEach((a, i) => { a.duration = i + 1; a.next = i; });
  const changed = reweightEventSigns(kernel, 0.8), before = eventSignMass(kernel), after = eventSignMass(changed);
  near(after.probability, 0.8); near(after.zero, before.zero);
  near(changed.reduce((s, a) => s + a.probability, 0), 1);
  near(changed[0].probability / changed[1].probability, kernel[0].probability / kernel[1].probability);
  for (let i = 0; i < changed.length; i++) {
    assert.ok(changed[i].probability > 0);
    const { probability: _, ...a } = changed[i], { probability: __, ...b } = kernel[i];
    assert.deepEqual(a, b);
  }
  const unsupported = constantModel([0, 0.1]).kernels[0];
  assert.deepEqual(reweightEventSigns(unsupported, 0.2), unsupported);
  assert.throws(() => reweightEventSigns(kernel, 1), /strictly/);
});

test("sign lookahead reproduces unchanged Bellman values and changes the fee-aware first action", () => {
  const base = constantModel([-0.02, 0, 0.02]);
  const p = buildEventPolicy(base, costs, { depths: 3, referenceEquity: 1000, referencePrice: 100, actionSteps: 5 });
  const lookahead = buildEventSignLookahead(p);
  for (const exposure of [-1.2, 0, 0.33, 1]) for (const depth of [1, 2, 3]) {
    const account = { equity: 1200, price: 105, exposure };
    const a = decideEvent(p, 0, account, depth), b = decideEventSign(lookahead, 0, account, depth, 0.5);
    near(a.value, b.value, 1e-10); near(a.quantity, b.quantity, 1e-10);
  }
  const account = { equity: 1000, price: 100, exposure: 0 };
  assert.ok(decideEventSign(lookahead, 0, account, 1, 0.8).longEntry);
  assert.ok(decideEventSign(lookahead, 0, account, 1, 0.2).shortEntry);
  assert.throws(() => decideEventSign(lookahead, 0, account, 1, 1), /tail/);
  const risky = buildEventPolicy(constantModel([-0.9, 0.02]), costs,
    { depths: 2, referenceEquity: 1000, referencePrice: 100, actionSteps: 5 });
  const highConfidence = decideEventSign(buildEventSignLookahead(risky), 0, account, 2, 1 - 1e-6);
  assert.ok(risky.model.kernels[0].every(a => !eventHolding(highConfidence.exposure, a, costs).liquidated));
});

test("size-conditioned signs learn opposite directional patterns without observing future size", () => {
  const samples = Array.from({ length: 240 }, (_, i) => {
    const x = i % 2 ? 1 : -1, large = i % 4 >= 2;
    return { features: [x], return: (large ? -x * 0.03 : x * 0.002) };
  });
  const h = trainEventSizeSign(samples, 0.01, 0.5), p = predictEventSizeSigns(h, [1]);
  near(p.reduce((s, v) => s + v, 0), 1);
  assert.ok(p[1] > 0.4 && p[2] > 0.4);
  assert.ok(p[0] < 0.1 && p[3] < 0.1);
  const kernel = constantModel([-0.04, -0.002, 0, 0.002, 0.04]).kernels[0];
  const base = eventSizeSignMass(kernel, h.thresholdLogBps), mixed = mixEventSizeSigns(base, p, 1);
  const changed = reweightEventSizeSigns(kernel, h.thresholdLogBps, p, 1);
  near(changed.reduce((s, a) => s + a.probability, 0), 1); near(mixed[4], base[4]);
  for (let i = 0; i < changed.length; i++) {
    const { probability, ...a } = changed[i], { probability: _, ...b } = kernel[i];
    assert.deepEqual(a, b); assert.ok(probability > 0);
  }
  assert.deepEqual(mixEventSizeSigns([0.5, 0, 0, 0.5, 0], p, 1), [0.5, 0, 0, 0.5, 0]);
  assert.equal(eventSizeSignGroup(0, h.thresholdLogBps), 4);
  const gateOnly = trainEventSizeGate(samples.map((s, i) => ({ ...s, features: [...s.features, i % 4 >= 2 ? 1 : -1] })), h, 0.01);
  assert.equal(gateOnly.largeSign, h.largeSign); assert.equal(gateOnly.ordinarySign, h.ordinarySign);
  const gateProbabilities = predictEventSizeSigns(gateOnly, [1, 1]);
  near(gateProbabilities[0] / gateProbabilities[1], p[0] / p[1]); near(gateProbabilities[2] / gateProbabilities[3], p[2] / p[3]);
  assert.ok(gateProbabilities[2] + gateProbabilities[3] > 0.9);
  const policy = buildEventPolicy(constantModel([-0.04, -0.002, 0, 0.002, 0.04]), costs,
    { depths: 2, referenceEquity: 1000, referencePrice: 100, actionSteps: 5 });
  const lookup = buildEventOutcomeLookahead(policy, 5, a => eventSizeSignGroup(a.return, h.thresholdLogBps));
  const account = { equity: 1200, price: 105, exposure: 0.4 };
  const ordinary = decideEvent(policy, 0, account, 2), grouped = decideEventOutcomes(lookup, 0, account, 2, lookup.masses[0]);
  near(ordinary.value, grouped.value); near(ordinary.quantity, grouped.quantity);
});

test("fast volatility heads only observe completed candle history", () => {
  const candles = Array.from({ length: 40 }, (_, i) => ({ openTime: i * 60_000,
    open: 100, close: 100 + i / 100, high: 101, low: 99, volume: 10 }));
  const a = eventFastVolatilityFeatures(candles, 20);
  const changed = candles.map((c, i) => i > 20 ? { ...c, close: 200, high: 250, low: 50 } : c);
  assert.deepEqual(eventFastVolatilityFeatures(changed, 20), a);
  assert.deepEqual(eventFastVolatilityFeatures(candles.slice(0, 21), 20), a);
  changed[20] = { ...changed[20], high: 150 };
  assert.ok(eventFastVolatilityFeatures(changed, 20)[0] > a[0]);
  changed[10] = { ...changed[10], openTime: changed[10].openTime + 1 };
  assert.throws(() => eventFastVolatilityFeatures(changed, 20), /Gap/);
});

test("run-volatility features preserve the original run basis and use completed RV5", () => {
  const candles = Array.from({ length: 1500 }, (_, i) => ({ openTime: i * 60_000,
    open: 100, close: 100 + Math.sin(i / 3), high: 102, low: 98, volume: 10 }));
  const clock = { thresholdBps: 60, maxCandles: 1440, reversalClock: true };
  const features = eventFeatures(candles, 1460, EVENT_RUN_VOLATILITY_FEATURES, clock);
  assert.deepEqual(features.slice(0, 20), eventFeatures(candles, 1460, EVENT_RUN_FEATURES, clock));
  near(features[20], eventFastVolatilityFeatures(candles, 1460)[1], 1e-14);
  assert.deepEqual(features, eventFeatures(candles.slice(0, 1461), 1460, EVENT_RUN_VOLATILITY_FEATURES, clock));
});

test("rolling size-gate calibration cannot consume unfinished or overlapping events", () => {
  const gate = new EventSizeGateCalibration({ window: 8, strength: 4 }, 100);
  near(gate.forecast(0.2, 100).probability, 0.2);
  const event = { originTime: 100, availableAt: 200, rawProbability: 0.2, large: true };
  assert.throws(() => gate.observe(event, 199), /completed/);
  near(gate.forecast(0.2, 100).probability, 0.2);
  gate.observe(event, 200);
  assert.throws(() => gate.forecast(0.2, 199), /availability/);
  const updated = gate.forecast(0.2, 200);
  assert.ok(updated.probability > 0.2 && updated.probability < 1); assert.equal(updated.count, 1);
  assert.throws(() => gate.observe(event, 200), /ordered/);
  const prior = [0.2, 0.6, 0.15, 0.05], probabilities = withEventSizeGate(prior, updated.probability);
  near(probabilities[0] / probabilities[1], prior[0] / prior[1]);
  near(probabilities[2] / probabilities[3], prior[2] / prior[3]);
  near(probabilities[2] + probabilities[3], updated.probability);
});

test("completed-event features exclude the current target, gaps and out-of-lookback origins", () => {
  const h = new EventCompletedHistory(0), empty = h.features(0);
  assert.ok(empty.every(v => v === 0));
  const e = { originTime: 0, availableAt: 60_000, return: -0.01, duration: 1 };
  assert.throws(() => h.observe(e, 59_999), /completed/); assert.deepEqual(h.features(0), empty);
  h.observe(e, 60_000);
  const f = h.features(60_000); near(f[0], 1 / 16); assert.equal(f[1], -1); near(f[6], -1);
  assert.throws(() => h.features(59_999), /available/);
  assert.throws(() => h.observe(e, 60_000), /ordered/);
  assert.deepEqual(h.features(86_400_001), empty);
  const gap = { originTime: 86_460_000, availableAt: 86_520_000, return: 0.02, duration: 1 };
  h.observe(gap, gap.availableAt); const g = h.features(gap.availableAt);
  near(g[0], 1 / 16); assert.equal(g[1], 1);
  const native = new EventCompletedHistory(0);
  native.observe({ originTime: 0, availableAt: 1_000, return: .001, duration: 1 / 60 }, 1_000);
  const nativeFeatures = native.features(1_000);
  near(nativeFeatures[0], 1 / 16); assert.equal(nativeFeatures[1], 1);
});

test("projected continuation preserves conditional paths and rebuilds deeper values under the new law", () => {
  const model = constantModel([-0.02, -0.001, 0, 0.001, 0.02]);
  const before = structuredClone(model), features = EVENT_FEATURES.map(() => 0);
  const projected = projectEventSizeSigns(model, 50, [{ features, probabilities: [0.05, 0.15, 0.1, 0.7] }], 1);
  assert.deepEqual(model, before);
  const mass = eventSizeSignMass(projected.kernels[0], 50);
  near(mass[3], 0.8 * 0.7); near(mass[4], 0.2);
  for (let i = 0; i < model.kernels[0].length; i++) {
    const { probability: _, ...a } = model.kernels[0][i], { probability: __, ...b } = projected.kernels[0][i];
    assert.deepEqual(a, b);
  }
  const options = { depths: 3, referenceEquity: 1000, referencePrice: 100, actionSteps: 5 };
  const originalPolicy = buildEventPolicy(model, costs, options), changed = buildEventPolicy(projected, costs, options);
  assert.notDeepEqual(changed.tables[2].holdValues, originalPolicy.tables[2].holdValues);
  const control = projectEventSizeSigns(model, 50, [{ features, probabilities: [0.05, 0.15, 0.1, 0.7] }], 0);
  assert.deepEqual(control.kernels, model.kernels);
});

test("recent joint laws use only completed post-fit events and preserve tail support", () => {
  const base = constantModel([-0.2, 0.02]), after = 1_000_000, features = EVENT_FEATURES.map(() => 0);
  const online = new EventRecentLaw(base, { window: 2, prior: 2, after });
  const first = online.forecast(features, after);
  near(first.mean, -0.09); assert.equal(first.recentCount, 0);
  const move: CompletedEventMove = { originTime: after, availableAt: after + 600_000,
    features, nextFeatures: features, duration: 10, return: 0.03, low: -0.01, high: 0.05 };
  assert.throws(() => online.observe(move, move.availableAt - 1), /completed/);
  assert.deepEqual(online.forecast(features, after), first);
  assert.throws(() => online.observe({ ...move, originTime: after - 600_000, availableAt: after }, after), /post-fit/);
  online.observe(move, move.availableAt);
  const next = online.forecast(features, move.availableAt), model = online.snapshot(move.availableAt);
  near(next.mean, (-0.18 + 0.03) / 3); assert.equal(next.recentCount, 1);
  near(next.classes.reduce((s, p) => s + p, 0), 1);
  near(model.kernels[0].reduce((s, a) => s + a.probability * a.return, 0), next.mean);
  assert.deepEqual(model.kernels[0][2], { probability: 1 / 3, return: 0.03, low: -0.01, high: 0.05, duration: 10, next: 0 });
  assert.ok(model.kernels[0][0].probability > 0); assert.equal(base.kernels[0][0].probability, 0.5);
  assert.throws(() => online.observe(move, move.availableAt), /nonoverlapping/);
  assert.throws(() => online.forecast(features, after), /backwards/);
  for (let j = 1; j <= 2; j++) online.observe({ ...move, originTime: after + j * 600_000,
    availableAt: after + (j + 1) * 600_000, return: -0.01 }, after + (j + 1) * 600_000);
  const last = online.forecast(features, after + 1_800_000);
  assert.equal(last.recentCount, 2); near(last.mean, (-0.18 - 0.02) / 4);
  assert.throws(() => new EventRecentLaw(base, { window: 2, prior: 0, after }), /options/);
  const rare = constantModel([-0.6, 0.02]);
  rare.kernels[0][0].probability = Number.MIN_VALUE; rare.kernels[0][1].probability = 1;
  const tails = new EventRecentLaw(rare, { window: 2, prior: 0.1, after });
  tails.observe(move, move.availableAt);
  assert.equal(tails.snapshot(move.availableAt).kernels[0][0].probability, Number.MIN_VALUE);
});

test("recent law updates retain physical state and successor conditioning", () => {
  const base = constantModel([-0.02, 0.02]);
  base.nodes = [{ feature: 0, cut: 0, left: 1, right: 2, leaf: -1 },
    { feature: -1, cut: 0, left: -1, right: -1, leaf: 0 }, { feature: -1, cut: 0, left: -1, right: -1, leaf: 1 }];
  base.kernels.push(base.kernels[0].map(a => ({ ...a, next: 1 }))); base.counts.push(2);
  const down = EVENT_FEATURES.map(() => 0), up = down.map((x, i) => i === 0 ? 1 : x);
  const online = new EventRecentLaw(base, { window: 1, prior: 1, after: 0 });
  online.observe({ originTime: 0, availableAt: 60_000, duration: 1, features: down, nextFeatures: up,
    return: 0.01, low: -0.01, high: 0.02 }, 60_000);
  assert.equal(online.forecast(up, 60_000).recentCount, 0);
  assert.equal(online.forecast(down, 60_000).recentCount, 1);
  assert.equal(online.snapshot(60_000).kernels[0].at(-1)!.next, 1);
  online.observe({ originTime: 60_000, availableAt: 120_000, duration: 1, features: up, nextFeatures: down,
    return: -0.01, low: -0.02, high: 0.01 }, 120_000);
  assert.equal(online.forecast(down, 120_000).recentCount, 0);
  assert.equal(online.snapshot(120_000).kernels[1].at(-1)!.next, 0);
});

test("shared recent laws borrow only allowed ancestor or physical-direction observations", () => {
  const base = constantModel([-0.02, 0.02]);
  base.featureNames = EVENT_RUN_FEATURES; base.clock.reversalClock = true; base.runConditioned = { directionPrior: 1 };
  base.nodes = [{ feature: 13, cut: 1.5, left: 1, right: 2, leaf: -1 },
    { feature: -1, cut: 0, left: -1, right: -1, leaf: 0 },
    { feature: 15, cut: 0.5, left: 3, right: 4, leaf: -1 },
    { feature: -1, cut: 0, left: -1, right: -1, leaf: 1 },
    { feature: -1, cut: 0, left: -1, right: -1, leaf: 2 }];
  base.kernels = Array.from({ length: 9 }, (_, next) => base.kernels[0].map(a => ({ ...a, next })));
  base.counts = Array.from({ length: 9 }, () => 2); base.trainingSamples = 18;
  const features = (leaf: number, side = 1) => EVENT_RUN_FEATURES.map((_, i) =>
    i === 12 ? side : i === 13 ? (leaf ? 2 : 1) * side : i === 15 && leaf === 2 ? side : 0);
  const parent = new EventRecentLaw(base, { window: 4, prior: 2, after: 0, scope: "parent" });
  const direction = new EventRecentLaw(base, { window: 4, prior: 2, after: 0, scope: "direction" });
  for (const updater of [parent, direction]) {
    updater.observe({ originTime: 0, availableAt: 60_000, features: features(2), nextFeatures: features(0, -1),
      duration: 1, return: 0.03, low: -0.01, high: 0.04 }, 60_000);
    updater.observe({ originTime: 60_000, availableAt: 120_000, features: features(0), nextFeatures: features(2),
      duration: 1, return: -0.02, low: -0.03, high: 0.01 }, 120_000);
  }
  const p = parent.forecast(features(1), 120_000), d = direction.forecast(features(1), 120_000);
  assert.equal(p.localCount, 0); assert.equal(p.recentCount, 1); near(p.mean, 0.01);
  assert.equal(d.localCount, 0); assert.equal(d.recentCount, 2); near(d.mean, 0.0025);
  assert.equal(parent.forecast(features(1, -1), 120_000).recentCount, 0);
  const snapshot = parent.snapshot(120_000), leaf = eventLeaf(base, features(1));
  assert.equal(snapshot.kernels[leaf].at(-1)!.next, eventLeaf(base, features(0, -1)));
  near(snapshot.kernels[leaf].reduce((s, a) => s + a.probability * a.return, 0), p.mean);
  assert.equal(snapshot.trainingSamples, 20);
  assert.throws(() => new EventRecentLaw(constantModel([0]), { window: 4, prior: 1, after: 0, scope: "direction" }), /canonical run states/);
});

test("Bellman diagnostics recognize the zero-return zero-friction fixed policy", () => {
  const p = buildEventPolicy(constantModel([0]), { ...costs, feeBps: 0, maxLeverage: 1 },
    { depths: 4, referenceEquity: 10000, referencePrice: 100 });
  for (const t of p.tables.slice(1)) {
    assert.equal(t.convergence!.changedActionFraction, 0);
    assert.equal(t.convergence!.maxActionChange, 0);
    assert.equal(t.convergence!.valueIncrementSpan, 0);
  }
  assert.deepEqual(restoreEventPolicy(serializeEventPolicy(p)).tables.map(t => t.convergence), p.tables.map(t => t.convergence));
});

test("Bellman grid can use the same marked terminal boundary as the exact finite-horizon solvers", () => {
  const model = constantModel([0]), options = { depths: 1, referenceEquity: 10000, referencePrice: 100 };
  const friction = buildEventPolicy(model, { ...costs, maxLeverage: 1 }, options);
  const marked = buildEventPolicy(model, { ...costs, maxLeverage: 1 }, { ...options, terminal: "marked" });
  const account = { equity: 10000, price: 100, exposure: 1 };
  assert.ok(decideEvent(friction, 0, account, 1).value < 0);
  near(decideEvent(marked, 0, account, 1).value, 0);
});

test("compiled Bellman operator retains even subnormal positive-probability ruin", () => {
  const model = constantModel([-0.6, 0.02]);
  model.kernels[0][0].probability = Number.MIN_VALUE; model.kernels[0][1].probability = 1;
  const p = buildEventPolicy(model, { ...costs, feeBps: 0 }, { depths: 2, referenceEquity: 10000, referencePrice: 100 });
  const decision = decideEvent(p, 0, { equity: 10000, price: 100, exposure: 0 }, 1);
  assert.ok(decision.exposure > 0 && decision.exposure <= 1.5 + 1e-8);
  const x = p.exposures.indexOf(5), index = ((2 * p.prices.length + 1) * p.exposures.length + x);
  assert.equal(p.tables[0].holdValues[index], -Infinity);
});

function persistentEvents(count = 1200): MoveSample[] {
  return Array.from({ length: count }, (_, i) => {
    const r = Math.floor(i / 40) % 2 ? 0.013 : -0.013;
    return { start: i * 30, end: (i + 1) * 30, duration: 30, return: r, low: Math.min(0, r), high: Math.max(0, r),
      label: r < 0 ? 2 : 14, features: EVENT_FEATURES.map(() => 0), nextFeatures: EVENT_FEATURES.map(() => 0) };
  });
}

test("run reflection pools patterns while retaining physical returns, tails, neutral states and successor direction", () => {
  const features = (orientation: number, extension: number) => EVENT_RUN_FEATURES.map((_, i) =>
    i === 12 ? orientation : i === 13 ? orientation * (1 + extension) : i === 15 ? orientation * extension : 0);
  const clock = { thresholdBps: 60, progressBps: 60, reversalClock: true, maxCandles: 1440 };
  const samples: MoveSample[] = Array.from({ length: 400 }, (_, i) => {
    const orientation = i % 2 ? -1 : 1, extended = i % 4 >= 2;
    const canonicalReturn = extended ? -0.008 : 0.01, value = orientation > 0 ? canonicalReturn : -canonicalReturn / (1 + canonicalReturn);
    const low = orientation > 0 ? -0.015 : -0.02 / 1.02, high = orientation > 0 ? 0.02 : 0.015 / 0.985;
    return { start: i * 10, end: (i + 1) * 10, return: value, low, high, duration: 10,
      label: eventMoveLabel(value, 10, { thresholdBps: 60, maxCandles: 100 }), features: features(orientation, extended ? 2 : 0),
      nextFeatures: features(extended ? -orientation : orientation, extended ? 0 : 2) };
  });
  const model = trainEventRunDistribution(samples, clock, { maxDepth: 1, minLeaf: 20, prior: 5, criterion: "mean" });
  assert.equal(model.kernels.length, 6); assert.equal(model.trainingSamples, samples.length);
  const up = eventLeaf(model, features(1, 0)), down = eventLeaf(model, features(-1, 0));
  assert.equal(down, up + 1);
  for (let i = 0; i < model.kernels[up].length; i++) {
    const a = model.kernels[up][i], b = model.kernels[down][i];
    near(b.return, -a.return / (1 + a.return)); near(b.low, -a.high / (1 + a.high)); near(b.high, -a.low / (1 + a.low));
    near(b.probability, a.probability); assert.equal(b.duration, a.duration);
    assert.equal(b.next, a.next - a.next % 3 + (a.next % 3 === 2 ? 2 : 1 - a.next % 3));
  }
  assert.equal(model.kernels[up][0].next, eventLeaf(model, samples[0].nextFeatures));
  assert.equal(eventLeaf(model, features(0, 0)) % 3, 2);
  for (const kernel of model.kernels) near(kernel.reduce((s, a) => s + a.probability, 0), 1);
  const p = buildEventPolicy(model, { ...costs, maxLeverage: 1 }, { depths: 2, referenceEquity: 10000, referencePrice: 100 });
  const account = { equity: 10000, price: 100, exposure: 0 };
  assert.ok(decideEvent(p, up, account, 1).longEntry);
  assert.ok(decideEvent(p, down, account, 1).shortEntry);
  assert.ok(decideEvent(p, eventLeaf(model, features(1, 2)), { ...account, exposure: 1 }, 1).longExit);
  assert.deepEqual(decideEvent(restoreEventPolicy(serializeEventPolicy(p)), down, account, 2), decideEvent(p, down, account, 2));
  const broken = structuredClone(model); broken.kernels[down][0].return = 0;
  assert.throws(() => buildEventPolicy(broken, costs, { depths: 1, referenceEquity: 10000, referencePrice: 100 }), /reflection law/);
  const brokenNeutral = structuredClone(model); brokenNeutral.kernels[up + 2][0].return = 0;
  assert.throws(() => buildEventPolicy(brokenNeutral, costs, { depths: 1, referenceEquity: 10000, referencePrice: 100 }), /neutral run-reflection/);
  assert.throws(() => calibrateEventMean(model, samples), /independent mean tilts/);
});

test("direction-conditioned run laws retain physical drift and shrink complete joint outcomes", () => {
  const clock = { thresholdBps: 60, progressBps: 60, reversalClock: true, maxCandles: 1440 };
  const features = (side: number) => EVENT_RUN_FEATURES.map((_, i) => i === 12 ? side : 0);
  const samples: MoveSample[] = Array.from({ length: 200 }, (_, i) => ({ start: i * 10, end: (i + 1) * 10,
    return: 0.01, low: -0.005, high: 0.02, duration: 10, label: 13,
    features: features(i % 2 ? -1 : 1), nextFeatures: features(i % 2 ? 1 : -1) }));
  const options = { maxDepth: 0, minLeaf: 20, prior: 0, criterion: "mean" as const };
  const pooled = trainEventRunDistribution(samples, clock, options);
  const model = trainEventRunDistribution(samples, clock, { ...options, directionPrior: 25 });
  assert.equal(model.runSymmetry, undefined); assert.deepEqual(model.runConditioned, { directionPrior: 25 });
  assert.deepEqual(model.counts, [100, 100, 0]); assert.equal(model.trainingSamples, 200);
  for (const side of [1, -1]) {
    const leaf = eventLeaf(model, features(side)), kernel = model.kernels[leaf];
    near(kernel.reduce((s, a) => s + a.probability, 0), 1);
    const mean = (m: EventDistribution) => m.kernels[leaf].reduce((s, a) => s + a.probability * a.return, 0);
    near(mean(model), 0.8 * 0.01 + 0.2 * mean(pooled));
    near(kernel.slice(0, 100).reduce((s, a) => s + a.probability, 0), 0.8);
    for (const a of kernel.slice(0, 100)) {
      assert.equal(a.return, 0.01); assert.equal(a.low, -0.005); assert.equal(a.high, 0.02);
      assert.equal(a.duration, 10); assert.equal(a.next, eventLeaf(model, features(-side)));
    }
  }
  assert.deepEqual(model.kernels[2], pooled.kernels[2]);
  const policy = buildEventPolicy(model, { ...costs, maxLeverage: 1 }, { depths: 2, referenceEquity: 10000, referencePrice: 100 });
  for (const side of [1, -1]) {
    const account = { equity: 10000, price: 100, exposure: 0 }, leaf = eventLeaf(model, features(side));
    assert.ok(decideEvent(policy, leaf, account, 1).longEntry);
    assert.deepEqual(decideEvent(restoreEventPolicy(serializeEventPolicy(policy)), leaf, account, 2), decideEvent(policy, leaf, account, 2));
  }
  const laterLoss = samples.map((s, i) => i < 100 ? s : { ...s, return: -0.01, low: -0.02, label: 1 });
  const honest = trainEventRunDistribution(laterLoss, clock, { ...options, directionPrior: 0, honestyFraction: 0.5 });
  assert.deepEqual(honest.counts, [50, 50, 0]); assert.equal(honest.trainingSamples, 100);
  for (const kernel of honest.kernels.slice(0, 2)) for (const a of kernel) assert.equal(a.return, -0.01);
  assert.throws(() => trainEventRunDistribution(samples, clock, { ...options, directionPrior: -1 }), /direction prior/);
  assert.throws(() => buildEventPolicy({ ...model, runSymmetry: true }, costs,
    { depths: 1, referenceEquity: 10000, referencePrice: 100 }), /canonical run distribution/);
});

test("hidden event belief learns persistence without revealing the latent regime to Bellman", () => {
  const rows = persistentEvents(), model = trainEventHidden(rows, { thresholdBps: 120, maxCandles: 1440 },
    { states: 2, resolution: 8, iterations: 30, smoothing: 1 });
  const hidden = model.hidden!;
  let up = hidden.initial, down = hidden.initial;
  for (let i = 0; i < 5; i++) { up = eventHiddenNext(model, up, 14); down = eventHiddenNext(model, down, 2); }
  assert.ok(model.classProbabilities[up][14] > 0.8);
  assert.ok(model.classProbabilities[down][2] > 0.8);
  assert.ok(distributionMetrics(model, rows).mseSkill > 0.7);
  for (let state = 0; state < model.kernels.length; state++) {
    near(model.kernels[state].reduce((s, a) => s + a.probability, 0), 1);
    for (const atom of model.kernels[state]) assert.equal(atom.next, eventHiddenNext(model, state, atom.return < 0 ? 2 : 14));
  }
  assert.throws(() => eventLeaf(model, rows[0].features), /belief state/);
  const policy = buildEventPolicy(model, { ...costs, maxLeverage: 1 }, { depths: 2, referenceEquity: 10000, referencePrice: 100 });
  const account = { equity: 10000, price: 100, exposure: 0 };
  assert.ok(decideEvent(policy, up, account, 2).longEntry);
  assert.ok(decideEvent(policy, down, account, 2).shortEntry);
  assert.deepEqual(decideEvent(restoreEventPolicy(serializeEventPolicy(policy)), up, account, 2), decideEvent(policy, up, account, 2));
});

test("hidden event fit resets across gaps and rejects overlapping event labels", () => {
  const rows = persistentEvents(100).map((s, i) => ({ ...s, start: i * 60, end: i * 60 + 30 }));
  const model = trainEventHidden(rows, { thresholdBps: 120, maxCandles: 1440 },
    { states: 2, resolution: 4, iterations: 10, smoothing: 1 });
  assert.equal(model.hidden!.sequences, rows.length);
  for (const row of model.hidden!.transition) for (const p of row) near(p, 0.5);
  rows[1].start = 20;
  assert.throws(() => trainEventHidden(rows, model.clock, { states: 2, resolution: 4, iterations: 10, smoothing: 1 }), /non-overlapping/);
  const continuous = persistentEvents(100);
  const mirrored = trainEventHidden([...continuous, ...continuous.map(s => ({ ...s, series: 1 }))], model.clock,
    { states: 2, resolution: 4, iterations: 10, smoothing: 1 });
  assert.equal(mirrored.hidden!.sequences, 2);
});

test("causal features and observed barrier termination cannot see later prices", () => {
  const candles: EventCandle[] = Array.from({ length: 1460 }, (_, i) => ({ openTime: i * 60_000,
    open: 100, close: 100, low: 100, high: 100, volume: 1 }));
  candles[1443] = { ...candles[1443], close: 100.3, high: 100.3 };
  const features = eventFeatures(candles, 1440);
  const result = observeMove(candles, 1440, { thresholdBps: 20, maxCandles: 10 })!;
  assert.equal(result.end, 1443); assert.equal(result.duration, 3);
  candles[1450].close = 1e6;
  assert.deepEqual(eventFeatures(candles, 1440), features);
  assert.equal(observeMove(candles, 1440, { thresholdBps: 20, maxCandles: 10 })!.return, result.return);
  candles[1442].openTime++;
  assert.equal(observeMove(candles, 1440, { thresholdBps: 20, maxCandles: 10 }), null);
  assert.equal(observeMove(candles.slice(0, 1442), 1440, { thresholdBps: 20, maxCandles: 10 }), null);
});

test("feature contracts reject discarded models and future or stale second observations", () => {
  const model = constantModel([0.01]);
  const config = { depths: 1, referenceEquity: 10_000, referencePrice: 100 };
  assert.throws(() => buildEventPolicy({ ...model, featureNames: [...EVENT_FEATURES, "discarded"] }, costs, config), /feature contract/);
  assert.throws(() => buildEventPolicy({ ...model, kernels: [[{ ...model.kernels[0][0], next: 1 }]] }, costs, config), /transition/);
  const candles: EventCandle[] = Array.from({ length: 1441 }, (_, i) => ({ openTime: i * 60_000,
    open: 100, close: 100, low: 100, high: 100, volume: 1 }));
  const last = candles[1440], decisionTime = last.openTime + 60_000;
  last.secondBasis = { availableAt: decisionTime, values: [1, 2, 3, 4, 5] };
  assert.deepEqual(eventFeatures(candles, 1440, EVENT_SECOND_FEATURES).slice(-5), [1, 2, 3, 4, 5]);
  last.secondBasis.availableAt++;
  assert.throws(() => eventFeatures(candles, 1440, EVENT_SECOND_FEATURES), /future/);
  last.secondBasis.availableAt = decisionTime - 60_000;
  assert.throws(() => eventFeatures(candles, 1440, EVENT_SECOND_FEATURES), /stale/);
});

test("run and flat events end on the first observed change without backdating extrema", () => {
  const c: EventCandle[] = Array.from({ length: 1460 }, (_, i) => ({ openTime: i * 60_000,
    open: 100, close: 100, low: 100, high: 100, volume: 1 }));
  for (const [i, price] of [[1440, 101], [1441, 102], [1442, 103], [1443, 102]]) {
    c[i] = { ...c[i], open: price, close: price, low: price, high: price };
  }
  const clock = { thresholdBps: 20, maxCandles: 10, runClock: true };
  const run = observeMove(c, 1440, clock)!;
  assert.equal(run.end, 1443); near(run.return, 102 / 101 - 1);
  c[1440] = { ...c[1440], open: 100, close: 100, low: 100, high: 100 };
  c[1441] = { ...c[1441], open: 100, close: 100, low: 100, high: 100 };
  assert.equal(observeMove(c, 1440, clock)!.end, 1442);
});

test("directional-change events include continuation and end at observed reversal confirmation", () => {
  const c: EventCandle[] = Array.from({ length: 1460 }, (_, i) => ({ openTime: i * 60_000,
    open: 100, close: 100, low: 100, high: 100, volume: 1 }));
  [101.1, 102, 104, 103.5, 102.8].forEach((close, n) => {
    const i = 1439 + n, open = c[i - 1].close;
    c[i] = { ...c[i], open, close, low: Math.min(open, close), high: Math.max(open, close) };
  });
  const clock = { thresholdBps: 100, maxCandles: 20, reversalClock: true };
  const features = eventFeatures(c, 1440, EVENT_RUN_FEATURES, clock);
  assert.equal(eventRunState(c, 1440, 100).direction, 1);
  const result = observeMove(c, 1440, clock, EVENT_RUN_FEATURES)!;
  assert.equal(result.end, 1443); assert.equal(result.duration, 3);
  near(result.return, 102.8 / 102 - 1);
  assert.ok(result.return > 0, "next confirmed reversal can still finish above the decision price");
  assert.equal(eventRunState(c, result.end, 100).direction, -1);
  const inverse = c.map(v => ({ ...v, open: 1 / v.open, close: 1 / v.close, high: 1 / v.low, low: 1 / v.high }));
  const mirrored = observeMove(inverse, 1440, clock, EVENT_RUN_FEATURES)!;
  assert.equal(mirrored.end, result.end); near(mirrored.return, -result.return / (1 + result.return));
  const normalFeatures = canonicalEventRunFeatures(result.features), inverseFeatures = canonicalEventRunFeatures(mirrored.features);
  normalFeatures.forEach((v, i) => near(v, inverseFeatures[i]));
  const progressClock = { ...clock, progressBps: 100 };
  const extension = observeMove(c, 1440, progressClock, EVENT_RUN_FEATURES)!;
  assert.equal(extension.end, 1441);
  assert.equal(eventRunState(c, extension.end, 100).direction, 1);
  assert.equal(observeMove(inverse, 1440, progressClock, EVENT_RUN_FEATURES)!.end, extension.end);
  const afterExtension = observeMove(c, extension.end, progressClock, EVENT_RUN_FEATURES)!;
  assert.equal(afterExtension.end, 1443);
  assert.equal(afterExtension.features[EVENT_FEATURES.length + 3] > 0, true, "mid-run origins retain observed overshoot");
  assert.throws(() => observeMove(c, 1440, { ...progressClock, reversalClock: false }), /event clock/);
  assert.throws(() => buildEventPolicy({ ...constantModel([0.01]), clock: { ...clock, thresholdBps: Infinity } }, costs,
    { depths: 1, referenceEquity: 10000, referencePrice: 100 }), /event clock/);
  c[1450].close = 1e9;
  assert.deepEqual(eventFeatures(c, 1440, EVENT_RUN_FEATURES, clock), features);
  assert.equal(observeMove(c, 1440, clock, EVENT_RUN_FEATURES)!.end, result.end);
  assert.throws(() => eventFeatures(c, 1440, EVENT_RUN_FEATURES), /model clock/);
});

test("path features locate a completed pullback without seeing a future peak", () => {
  const c: EventCandle[] = Array.from({ length: 1450 }, (_, i) => ({ openTime: i * 60_000,
    open: 100, close: 100, high: 100, low: 100, volume: 1 }));
  c[1435].high = 110; c[1436].low = 95;
  const before = eventFeatures(c, 1440, EVENT_PATH_FEATURES);
  assert.equal(before.length, EVENT_PATH_FEATURES.length);
  assert.ok(before[13] < 0 && before[14] > 0);
  near(before[19], Math.log1p(5) / Math.log1p(1440));
  near(before[20], Math.log1p(4) / Math.log1p(1440));
  c[1442].high = 1e6; c[1442].close = 1e6;
  assert.deepEqual(eventFeatures(c, 1440, EVENT_PATH_FEATURES), before);
});

test("post-fee target conserves equity, charges round trips and handles both sides", () => {
  const account = { equity: 10_000, price: 100, exposure: 0 };
  const long = eventTrade(account, 5, costs)!;
  near(long.equity + long.cost, account.equity);
  near(long.exposure, 5); near(long.equity, account.equity / (1 + 5 * 0.0005));
  const short = eventTrade(account, -5, costs)!;
  near(short.equity, long.equity); near(short.quantity, -long.quantity);
  const close = eventTrade(long, 0, costs)!;
  near(close.exposure, 0); assert.ok(close.equity < long.equity);
  const small = { ...costs, minNotional: 100, maxNotional: 200 };
  assert.equal(eventTrade(account, 0.001, small), null);
  assert.equal(eventTrade(account, 1, small), null);
});

test("ruin and intra-event liquidation cannot be hidden by endpoint recovery", () => {
  assert.equal(eventHolding(5, { return: 0.1, low: -0.3, high: 0.1, duration: 1 }, costs).liquidated, true);
  assert.equal(eventHolding(-5, { return: -0.1, low: -0.1, high: 0.3, duration: 1 }, costs).liquidated, true);
  assert.equal(eventHolding(0, { return: 1, low: -0.9, high: 3, duration: 1 }, costs).factor, 1);
});

test("Bellman averages outcomes before choosing current action; uncertainty is not hindsight", () => {
  const p = buildEventPolicy(constantModel([-0.02, 0.02]), costs,
    { depths: 2, referenceEquity: 10_000, referencePrice: 100, actionSteps: 5 });
  for (const depth of [1, 2]) assert.equal(decideEvent(p, 0, { equity: 10_000, price: 100, exposure: 0 }, depth).quantity, 0);
});

test("one-event long and short decisions reflect arithmetic returns and close existing risk", () => {
  const make = (r: number) => buildEventPolicy(constantModel([r]), costs,
    { depths: 1, referenceEquity: 10_000, referencePrice: 100, actionSteps: 5 });
  const account = { equity: 10_000, price: 100, exposure: 0 };
  const long = decideEvent(make(0.01), 0, account, 1);
  const short = decideEvent(make(-0.01), 0, account, 1);
  near(long.exposure, -short.exposure); assert.ok(long.longEntry && short.shortEntry);
  const exit = decideEvent(make(-0.01), 0, { ...account, exposure: 5 }, 1);
  assert.ok(exit.longExit && exit.shortEntry);
});

test("actual margin borrowing prevents blind long-short entry reflection", () => {
  // Equal quoted borrow rates still apply to different principals: a fully
  // funded long borrows nothing, while a margin short borrows the whole asset.
  const account = { equity: 10000, price: 100, exposure: 0 }, r = .001202;
  const kernel = (value: number) => [{ return: value, low: Math.min(0, value), high: Math.max(0, value),
    duration: 60, next: 0, probability: 1 }];
  const long = decideEventOneStep(kernel(r), account, DEFAULT_EVENT_COSTS, "marked");
  const short = decideEventOneStep(kernel(-r), account, DEFAULT_EVENT_COSTS, "marked");
  assert.ok(long.longEntry && long.quantity > 0 && long.value > 0);
  near(long.exposure, 1, 1e-6);
  assert.equal(short.quantity, 0); assert.equal(short.value, 0);
  const reflectedQuantity = -long.quantity, fee = Math.abs(reflectedQuantity) * account.price * .0012;
  const reflectedExposure = reflectedQuantity * account.price / (account.equity - fee);
  const held = eventHolding(reflectedExposure, kernel(-r)[0], DEFAULT_EVENT_COSTS);
  const reflectedValue = Math.log((account.equity - fee) / account.equity) + Math.log(held.factor);
  assert.ok(reflectedValue < 0);
  const symmetricCosts = { ...DEFAULT_EVENT_COSTS, longBorrowBpsPerDay: 0, shortBorrowBpsPerDay: 0 };
  const a = decideEventOneStep(kernel(r), account, symmetricCosts, "marked");
  const b = decideEventOneStep(kernel(-r), account, symmetricCosts, "marked");
  near(a.quantity, -b.quantity, 1e-10); near(a.value, b.value, 1e-12);
});

test("conditional tree preserves normalized joint outcomes and terminal states", () => {
  const samples: MoveSample[] = Array.from({ length: 400 }, (_, i) => {
    const sign = i % 2 ? 1 : -1;
    return { start: i, end: i + 1, features: [sign, ...new Array(EVENT_FEATURES.length - 1).fill(0)],
      nextFeatures: [-sign, ...new Array(EVENT_FEATURES.length - 1).fill(0)], return: sign * 0.01,
      low: Math.min(0, sign * 0.01), high: Math.max(0, sign * 0.01), duration: 1, label: sign > 0 ? 12 : 0 };
  });
  const model = trainEventDistribution(samples, { thresholdBps: 20, maxCandles: 10 }, { maxDepth: 2, minLeaf: 20, prior: 10 });
  assert.equal(model.kernels.length, 2);
  for (let leaf = 0; leaf < 2; leaf++) {
    near(model.kernels[leaf].reduce((s, a) => s + a.probability, 0), 1);
    near(model.classProbabilities[leaf].reduce((s, p) => s + p, 0), 1);
    assert.ok(model.kernels[leaf].every(a => a.next >= 0 && a.next < 2));
  }
});

test("marked exposure above the target cap must reduce risk", () => {
  const p = buildEventPolicy(constantModel([0.1]), costs,
    { depths: 1, referenceEquity: 10_000, referencePrice: 100, actionSteps: 5 });
  const decision = decideEvent(p, 0, { equity: 10_000, price: 100, exposure: 5.5 }, 1);
  assert.ok(Math.abs(decision.exposure) <= 5 + 1e-8);
  assert.ok(decision.quantity < 0);
});

test("max-order clips reduce excessive exposure when one order cannot restore the cap", () => {
  const capped = { ...costs, maxNotional: 50_000 };
  const account = { equity: 10_000, price: 100, exposure: 20 };
  const target = (account.equity * account.exposure - capped.maxNotional) / (account.equity - capped.maxNotional * capped.feeBps / 1e4);
  const trade = eventTrade(account, target, capped)!;
  assert.ok(trade && trade.exposure < 20 && trade.exposure > 5);
  near(trade.turnover, 50_000);
  assert.equal(eventTrade(account, 19.9, capped), null);
});

test("honest estimation does not reuse partition returns as estimated edge", () => {
  const rows: MoveSample[] = Array.from({ length: 400 }, (_, i) => {
    const direction = i % 2 ? 1 : -1, r = (i < 200 ? direction : -direction) * 0.01;
    return { start: i, end: i + 1, features: [direction, ...new Array(EVENT_FEATURES.length - 1).fill(0)],
      nextFeatures: [direction, ...new Array(EVENT_FEATURES.length - 1).fill(0)], return: r,
      low: Math.min(0, r), high: Math.max(0, r), duration: 1, label: r < 0 ? 0 : 12 };
  });
  const model = trainEventDistribution(rows, { thresholdBps: 20, maxCandles: 10 },
    { maxDepth: 1, minLeaf: 20, prior: 0, criterion: "mean", honestyFraction: 0.5 });
  const leaf = eventLeaf(model, rows[1].features);
  near(model.kernels[leaf].reduce((s, a) => s + a.probability * a.return, 0), -0.01);
  assert.equal(model.trainingSamples, 200);
  const explicit = trainEventDistribution(rows.slice(0, 200), { thresholdBps: 20, maxCandles: 10 },
    { maxDepth: 1, minLeaf: 20, prior: 0, criterion: "mean", estimationSamples: rows.slice(200) });
  assert.deepEqual(explicit, model);
  assert.deepEqual(reestimateEventTree(model, rows.slice(200), 0), explicit);
  const opposite = rows.slice(200).map(r => ({ ...r, return: -r.return,
    high: -r.low, low: -r.high, label: r.return > 0 ? 0 : 12 }));
  const changed = trainEventDistribution(rows.slice(0, 200), model.clock,
    { maxDepth: 1, minLeaf: 20, prior: 0, criterion: "mean", estimationSamples: opposite });
  assert.deepEqual(changed.nodes, model.nodes);
  near(changed.kernels[leaf].reduce((s, a) => s + a.probability * a.return, 0), 0.01);
  const snapshot = JSON.stringify(model), refitted = reestimateEventTree(model, opposite, 0);
  assert.deepEqual(refitted, changed); assert.equal(JSON.stringify(model), snapshot);
  const oneLeaf = reestimateEventTree(model, opposite.filter(s => s.features[0] > 0), 0);
  assert.equal(oneLeaf.counts.filter(n => n === 0).length, 1);
  for (const kernel of oneLeaf.kernels) {
    near(kernel.reduce((sum, a) => sum + a.probability, 0), 1);
    assert.ok(kernel.every(a => a.return > 0 && a.next === leaf));
  }
  assert.throws(() => reestimateEventTree(model, [], 32), /population/);
  assert.throws(() => reestimateEventTree(model, [{ ...opposite[0], label: 7 }], 32), /population/);
  assert.throws(() => trainEventDistribution(rows, model.clock,
    { maxDepth: 1, minLeaf: 20, prior: 0, estimationSamples: [] }), /nonempty/);
});

test("honest tree rejects partitions without separate estimation covariate support", () => {
  const row = (i: number, feature: number, value: number): MoveSample => ({
    start: i, end: i + 1,
    features: [feature, ...new Array(EVENT_FEATURES.length - 1).fill(0)],
    nextFeatures: [feature, ...new Array(EVENT_FEATURES.length - 1).fill(0)],
    return: value, low: Math.min(0, value), high: Math.max(0, value), duration: 1,
    label: value < 0 ? 0 : 12,
  });
  const partition = Array.from({ length: 200 }, (_, i) => row(i, i < 100 ? -1 : 1, i < 100 ? -.01 : .01));
  const estimation = Array.from({ length: 100 }, (_, i) => row(1000 + i, 1, i % 2 ? -.01 : .01));
  const ordinary = trainEventDistribution(partition, { thresholdBps: 20, maxCandles: 10 },
    { maxDepth: 1, minLeaf: 20, prior: 0, criterion: "mean", estimationSamples: estimation });
  assert.equal(ordinary.kernels.length, 2);
  assert.deepEqual(ordinary.counts, [0, 100]);
  const supported = trainEventDistribution(partition, ordinary.clock,
    { maxDepth: 1, minLeaf: 20, prior: 0, criterion: "mean", estimationSamples: estimation,
      minimumEstimationLeaf: 20 });
  assert.equal(supported.kernels.length, 1);
  assert.deepEqual(supported.counts, [100]);
  const changedOutcomes = estimation.map(sample => ({ ...sample, return: -sample.return,
    low: -sample.high, high: -sample.low, label: sample.return > 0 ? 0 : 12 }));
  const changed = trainEventDistribution(partition, ordinary.clock,
    { maxDepth: 1, minLeaf: 20, prior: 0, criterion: "mean", estimationSamples: changedOutcomes,
      minimumEstimationLeaf: 20 });
  assert.deepEqual(changed.nodes, supported.nodes);
  assert.throws(() => trainEventDistribution(partition, ordinary.clock,
    { maxDepth: 1, minLeaf: 20, prior: 0, minimumEstimationLeaf: 20 }), /explicit estimation/);
  assert.throws(() => trainEventDistribution(partition, ordinary.clock,
    { maxDepth: 1, minLeaf: 20, prior: 0, estimationSamples: estimation,
      minimumEstimationLeaf: 0 }), /positive integer/);
});

test("joint forest states retain normalized transition and class probabilities", () => {
  const rows: MoveSample[] = Array.from({ length: 400 }, (_, i) => {
    const direction = i % 2 ? 1 : -1, r = direction * 0.01;
    return { start: i * 1440, end: i * 1440 + 1, features: [direction, ...new Array(EVENT_FEATURES.length - 1).fill(0)],
      nextFeatures: [-direction, ...new Array(EVENT_FEATURES.length - 1).fill(0)], return: r,
      low: Math.min(0, r), high: Math.max(0, r), duration: 1, label: r < 0 ? 0 : 12 };
  });
  const model = trainEventForest(rows, { thresholdBps: 20, maxCandles: 10 },
    { trees: 3, maxDepth: 1, minLeaf: 20, prior: 10, seed: 7 });
  assert.equal(model.kernels.length, 8);
  for (let i = 0; i < model.kernels.length; i++) {
    near(model.kernels[i].reduce((s, a) => s + a.probability, 0), 1);
    near(model.classProbabilities[i].reduce((s, v) => s + v, 0), 1);
    assert.ok(model.kernels[i].every(a => a.next >= 0 && a.next < model.kernels.length));
  }
  const positive = model.kernels[eventLeaf(model, rows[1].features)];
  assert.ok(positive.reduce((s, a) => s + a.probability * a.return, 0) > 0.009);
});

test("ridge projection learns joint directional evidence and serializes its state mapping", () => {
  const rows: MoveSample[] = Array.from({ length: 400 }, (_, i) => {
    const x = ((i * 37) % 101) / 50 - 1, y = ((i * 53) % 97) / 48 - 1, r = 0.005 * (x + y);
    const features = [x, y, ...new Array(EVENT_FEATURES.length - 2).fill(0)];
    return { start: i, end: i + 1, features, nextFeatures: features.map(v => -v), return: r,
      low: Math.min(0, r), high: Math.max(0, r), duration: 1, label: r < 0 ? 0 : 12 };
  });
  const model = trainEventProjection(rows, { thresholdBps: 20, maxCandles: 10 },
    { penalty: 0.001, cells: 8, minLeaf: 20, prior: 10 });
  assert.equal(model.kernels.length, 8);
  assert.ok(model.projection!.coefficients[0] > 0 && model.projection!.coefficients[1] > 0);
  const error = rows.reduce((sum, row) => sum + (eventProjectionScore(model.projection!, row.features) - row.return) ** 2, 0);
  assert.ok(error / rows.length < 1e-8);
  for (const kernel of model.kernels) near(kernel.reduce((s, a) => s + a.probability, 0), 1);
  const restored = JSON.parse(JSON.stringify(model));
  assert.deepEqual(rows.map(row => eventLeaf(restored, row.features)), rows.map(row => eventLeaf(model, row.features)));
  const heavy = trainEventProjection(rows, model.clock, { penalty: 100, cells: 8, minLeaf: 20, prior: 10 });
  assert.ok(Math.max(...heavy.kernels.map(k => Math.abs(k.reduce((s, a) => s + a.probability * a.return, 0)))) < 0.001);
});

test("boosted event forecast learns a nonlinear interaction with finite joint states", () => {
  const rows: MoveSample[] = Array.from({ length: 400 }, (_, i) => {
    const x = i % 4 < 2 ? -1 : 1, y = i % 2 ? 1 : -1, r = (x > 0 && y > 0 ? 0.02 : -0.01);
    const features = [x, y, ...new Array(EVENT_FEATURES.length - 2).fill(0)];
    return { start: i, end: i + 1, features, nextFeatures: features.map(v => -v), return: r,
      low: Math.min(0, r), high: Math.max(0, r), duration: 1, label: r < 0 ? 0 : 12 };
  });
  const model = trainEventBoost(rows, { thresholdBps: 20, maxCandles: 10 },
    { iterations: 32, rate: 0.2, cells: 4, minLeaf: 20, prior: 10 });
  assert.ok(Math.abs(eventBoostScore(model.boost!, rows[3].features) - 0.02) < 1e-4);
  assert.ok(Math.abs(eventBoostScore(model.boost!, rows[0].features) + 0.01) < 1e-4);
  assert.ok(model.kernels.length >= 2, "Tied forecast scores must retain both directional states");
  for (const kernel of model.kernels) near(kernel.reduce((s, a) => s + a.probability, 0), 1);
  const p = buildEventPolicy(model, costs, { depths: 1, referenceEquity: 10_000, referencePrice: 100 });
  assert.ok(decideEvent(p, eventLeaf(model, rows[3].features), { equity: 10_000, price: 100, exposure: 0 }, 1).longEntry);
  assert.ok(decideEvent(p, eventLeaf(model, rows[0].features), { equity: 10_000, price: 100, exposure: 0 }, 1).shortEntry);
});

test("two Bellman events compound log wealth and serialized inference reproduces the action", () => {
  const free = { ...costs, feeBps: 0 };
  const p = buildEventPolicy(constantModel([0.01]), free,
    { depths: 2, referenceEquity: 10_000, referencePrice: 100, actionSteps: 5 });
  const account = { equity: 10_000, price: 100, exposure: 0 };
  const first = decideEvent(p, 0, account, 1), second = decideEvent(p, 0, account, 2);
  near(first.value, Math.log(1.05)); near(second.value, 2 * Math.log(1.05));
  const restored = restoreEventPolicy(JSON.parse(JSON.stringify(serializeEventPolicy(p))));
  assert.deepEqual(decideEvent(restored, 0, account, 2), second);
});

test("mean calibration tilts probabilities without changing joint outcome support", () => {
  const raw = constantModel([-0.01, 0.01]);
  const row = (r: number): MoveSample => ({ start: 0, end: 1, features: new Array(EVENT_FEATURES.length).fill(0),
    nextFeatures: new Array(EVENT_FEATURES.length).fill(0), return: r, low: Math.min(r, 0), high: Math.max(r, 0), duration: 10, label: r < 0 ? 0 : 12 });
  const calibrated = calibrateEventMean(raw, [row(0.01), row(0.01), row(-0.01)]);
  assert.ok(calibrated.meanCalibration!.scale >= 0 && calibrated.meanCalibration!.scale <= 2);
  assert.deepEqual(calibrated.kernels[0].map(a => [a.return, a.duration, a.next]), raw.kernels[0].map(a => [a.return, a.duration, a.next]));
  near(calibrated.kernels[0].reduce((s, a) => s + a.probability, 0), 1);
  near(calibrated.classProbabilities[0].reduce((s, v) => s + v, 0), 1);
});
