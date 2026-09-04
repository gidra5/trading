/** Exact fixed-design attribution of depth-two target changes, including crossfit actions. */
import fs from "node:fs";
import path from "node:path";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { eventFeatures, eventLeaf, type MoveSample } from "../packages/bot-algo/src/event-distribution.js";
import { chooseEventTrade, eventHolding, eventTrade, restoreEventPolicy, type EventAccount } from "../packages/bot-algo/src/event-log-policy.js";
import { decideFittedEvent, fittedEventHolding, predictEventHoldingGrid, trainEventFittedValue, type FittedEventValue } from "../packages/bot-algo/src/event-fitted-value.js";
import { eventFastVolatilityFeatures, EVENT_FAST_VOLATILITY_INPUTS } from "../packages/bot-algo/src/event-size-sign.js";
import { eventCandleShapes, EVENT_CANDLE_SHAPE_INPUTS, eventFuturesFeatures, eventFuturesBasisDeviations, eventFittedFuturesInputs, loadEventFuturesRows, EVENT_FUTURES_PRICE_INPUTS, EVENT_FUTURES_FLOW_INPUTS, EVENT_FUTURES_DEVIATION_INPUTS } from "./event-futures-basis.js";
import { eventFittedSettingName } from "./event-fitted-settings.js";
import { eventSignHorizonPaths } from "./event-paths.js";
import { fitEventCrossfitContinuations } from "./event-fitted-crossfit.js";
import { eventRidgeInfluence } from "./event-ridge-influence.js";
import { loadEventCandles, makeSamples } from "./research-event-policy.js";

const root = path.resolve(__dirname, ".."), DAY = 86400000;
const arg = (k: string) => { const i = process.argv.indexOf(`--${k}`); return i < 0 ? "" : process.argv[i + 1]; };
if (!arg("source") || !arg("output")) throw new Error("Specify paired sampled-path screen and new output");
const source = path.resolve(root, "data/benchmarks", arg("source")), output = path.resolve(root, "data/benchmarks", arg("output"));
const crossfitSource = arg("crossfit-source") ? path.resolve(root, "data/benchmarks", arg("crossfit-source")) : undefined;
const read = (dir: string, file: string) => JSON.parse(fs.readFileSync(path.join(dir, file), "utf8"));
if (fs.existsSync(output)) throw new Error("Choose a new output directory");
const config = read(source, "config.json"), setting = config.settings.find((s: any) => s.sampledPath);
if (config.contract !== "event-fitted-value-screen-v1" || config.settings.length !== 2 || !setting || setting.boost || setting.secondDynamics || setting.continuationFolds
  || !setting.pathHorizon || config.depths < 2) throw new Error("Requires paired plain ridge sampled paths");
const ordinarySetting = { ...setting }; delete ordinarySetting.sampledPath;
assert.ok(config.settings.some((s: any) => eventFittedSettingName(s) === eventFittedSettingName(ordinarySetting)));
const crossfitConfig = crossfitSource ? read(crossfitSource, "config.json") : undefined;
const crossfitSetting = crossfitConfig?.settings[0];
if (crossfitConfig) {
  assert.equal(crossfitConfig.contract, config.contract); assert.equal(crossfitConfig.source, config.source);
  assert.deepEqual(crossfitConfig.phases, config.phases); assert.equal(crossfitConfig.settings.length, 1);
  assert.equal(crossfitConfig.depths, 2); assert.ok([6, 20].includes(crossfitSetting.continuationFolds));
  assert.deepEqual(crossfitSetting, { ...setting, continuationFolds: crossfitSetting.continuationFolds });
}
const jc = read(config.source, "config.json"), oc = read(jc.source, "config.json"), sc = read(oc.source, "config.json");
const fits = config.phases.map((p: any) => read(jc.source, `${jc.window.id}-${p.id}-model.json`));
const c = loadEventCandles(fits[0].trainStart - DAY, config.phases.at(-1).endTime), external = loadEventFuturesRows(fits[0].trainStart - DAY, config.phases.at(-1).endTime);
const byTime = new Map(c.map((r, i) => [r.openTime + 60000, i])), cache = new Map<number, number[] | null>();
const names = [...sc.featureNames, ...EVENT_FAST_VOLATILITY_INPUTS,
  ...eventFittedFuturesInputs(setting.basis, { price: EVENT_FUTURES_PRICE_INPUTS, flow: EVENT_FUTURES_FLOW_INPUTS }, EVENT_FUTURES_DEVIATION_INPUTS),
  ...(setting.candleShape ? EVENT_CANDLE_SHAPE_INPUTS : [])];
const inputs = (i: number) => {
  if (cache.has(i)) return cache.get(i)!;
  const e = eventFuturesFeatures(c, i, t => external.rows.get(t));
  const d = setting.historyMinutes ? eventFuturesBasisDeviations(c, i, t => external.rows.get(t)) : [];
  const shape = setting.candleShape ? eventCandleShapes(c, i, t => external.rows.get(t)) : [];
  const values = e && d && shape ? [...eventFeatures(c, i, sc.featureNames, sc.clock), ...eventFastVolatilityFeatures(c, i),
    ...eventFittedFuturesInputs(setting.basis, e, d), ...shape] : null;
  cache.set(i, values); return values;
};
const coord = (p: FittedEventValue, values: readonly number[]) => [1, ...values.map((v, i) => Math.max(-5, Math.min(5, (v - p.means[i]) / p.scales[i])))];
const dot = (a: readonly number[], b: readonly number[]) => a.reduce((s, v, i) => s + v * b[i], 0);
const near = (a: number, b: number, tolerance = 1e-9) => assert.ok(Math.abs(a - b) < tolerance, `${a} != ${b}`);
function interpolation(p: FittedEventValue, account: EventAccount) {
  const bracket = (axis: readonly number[], value: number): Array<[number, number]> => {
    if (value <= axis[0]) return [[0, 1]]; if (value >= axis.at(-1)!) return [[axis.length - 1, 1]];
    const high = axis.findIndex(v => v >= value), low = high - 1, t = (value - axis[low]) / (axis[high] - axis[low]);
    return [[low, 1 - t], [high, t]];
  };
  const cells = new Map<number, number>();
  for (const [w, a] of bracket(p.equities, account.equity)) for (const [price, b] of bracket(p.prices, account.price)) for (const [x, d] of bracket(p.exposures, account.exposure))
    if (a * b * d) cells.set((w * p.prices.length + price) * p.exposures.length + x, a * b * d);
  return cells;
}
fs.mkdirSync(output, { recursive: true });
fs.writeFileSync(path.join(output, "config.json"), JSON.stringify({ source, crossfitSource, phases: config.phases, setting,
  sourceHash: createHash("sha256").update(fs.readFileSync(path.join(source, "config.json"))).update(crossfitSource ? fs.readFileSync(path.join(crossfitSource, "config.json")) : "").update(external.fingerprint).digest("hex"),
  reference: crossfitSource ? "Full-data H1 sampled continuation" : "Bootstrapped continuation",
  candidate: crossfitSource ? "Block-held-out H1 sampled continuation" : "Full-data H1 sampled continuation",
  method: crossfitSource
    ? "Exact fixed-design ridge weights attribute the held-out-minus-full-data sampled target change. Each H1 policy chooses its own next action, then the same observed next return and settlement evaluate it. Original sampled orders are compared with cash and the crossfit choice on the same actual account. Block and day contributions sum to the raw change; cash-floor correction is separate."
    : "Fixed-design ridge prediction weights attribute the sampled-minus-bootstrap action contrast. Shared H1 chooses the same next action; the target difference is observed holding/settlement minus fitted holding value. Account interpolation and cash-floor correction are explicit.",
  caveat: "Post-selection prior-origin diagnosis at every executed original sampled depth-two order. No final-window outcomes or candidate selection. Contributions hold fitted H1 policies, design, normalization and penalty fixed; they are not leave-one-out refits or independent causal market effects." }, null, 2));
const files = ["scripts/audit-event-fitted-influence.ts", "scripts/event-ridge-influence.ts", "scripts/event-paths.ts", "scripts/event-fitted-crossfit.ts", "scripts/research-event-policy.ts", "scripts/event-futures-basis.ts", "packages/bot-algo/src/event-fitted-value.ts", "packages/bot-algo/src/event-log-policy.ts"];
fs.writeFileSync(path.join(output, "sources.json"), JSON.stringify(Object.fromEntries(files.map(f => [f, fs.readFileSync(path.join(root, f), "utf8")]))));
const results = [], started = performance.now();
for (const [phaseIndex, phase] of config.phases.entries()) {
  const original: FittedEventValue = read(source, `${phase.id}-${eventFittedSettingName(setting)}-policy.json`);
  const reference: FittedEventValue = crossfitSource ? original : read(source, `${phase.id}-${eventFittedSettingName(ordinarySetting)}-policy.json`);
  const candidate: FittedEventValue = crossfitSource ? read(crossfitSource, `${phase.id}-${eventFittedSettingName(crossfitSetting)}-policy.json`) : original;
  const base = restoreEventPolicy(read(config.source, `${phase.id}-policy.json`));
  assert.deepEqual(reference.tables[0], candidate.tables[0]); assert.deepEqual(reference.means, candidate.means); assert.deepEqual(reference.scales, candidate.scales);
  const state = (i: number) => eventLeaf(base.model, eventFeatures(c, i, base.model.featureNames, base.model.clock));
  const transition = (r: MoveSample) => ({ features: inputs(r.start)!, nextFeatures: inputs(r.end)!, leaf: state(r.start), nextLeaf: state(r.end),
    move: { return: r.return, low: r.low, high: r.high, duration: r.duration } });
  const single = makeSamples(c, sc.clock, fits[phaseIndex].trainStart, fits[phaseIndex].trainEnd, [jc.window], sc.stride, "chain", sc.featureNames).filter(r => inputs(r.start) && inputs(r.end));
  const paths = eventSignHorizonPaths(single, setting.pathHorizon), rows = paths.map(p => transition(p.steps[0])), following = paths.map(p => p.steps.slice(1).map(transition));
  const crossfit = crossfitSource ? fitEventCrossfitContinuations(base, rows,
    paths.map(p => ({ start: c[p.start].openTime + 60000, end: c[p.steps[0].end].openTime + 60000 })),
    paths.map(p => ({ start: c[p.steps[1].start].openTime + 60000, end: c[p.steps[1].end].openTime + 60000 })),
    fits[phaseIndex].trainStart, fits[phaseIndex].trainEnd, setting.penalty, crossfitSetting.continuationFolds) : undefined;
  if (crossfitSource) assert.deepEqual(read(crossfitSource, `${phase.id}-${eventFittedSettingName(crossfitSetting)}-crossfit.json`), { folds: crossfit!.folds, models: crossfit!.models });
  trainEventFittedValue(base, rows, setting.penalty, 1, undefined, { ...reference, tables: reference.tables.slice(0, 1) }, crossfit ? { following } : undefined);
  trainEventFittedValue(base, rows, setting.penalty, 1, undefined, { ...candidate, tables: candidate.tables.slice(0, 1) }, { following, ...(crossfit ? { policies: crossfit.policies } : {}) });
  const foldIds: number[] = []; for (const fold of crossfit?.folds ?? []) for (const i of fold.test) foldIds[i] = fold.index;
  const design = rows.map(r => coord(reference, r.features)), weightFor = eventRidgeInfluence(design, reference.penalty);
  const futureGrids = rows.map(r => predictEventHoldingGrid(reference, r.nextFeatures, 1));
  const crossfitGrids = crossfit?.policies.map((p, i) => predictEventHoldingGrid(p, rows[i].nextFeatures, 1));
  const settle = (exposure: number) => Math.log(1 - Math.abs(exposure) * (reference.costs.feeBps + reference.costs.slippageBps) / 10000);
  const targetCache = new Map<number, number[]>();
  const deltaTargets = (cell: number) => {
    if (targetCache.has(cell)) return targetCache.get(cell)!;
    const xi = cell % reference.exposures.length, price = Math.floor(cell / reference.exposures.length) % reference.prices.length;
    const wi = Math.floor(cell / (reference.exposures.length * reference.prices.length));
    const values = rows.map((r, i) => {
      const h = eventHolding(reference.exposures[xi], r.move, reference.costs);
      const a = { equity: reference.equities[wi] * h.factor, price: reference.prices[price] * (1 + r.move.return), exposure: h.exposure };
      const trade = chooseEventTrade(reference, a, after => fittedEventHolding(reference, futureGrids[i], after, r.nextLeaf));
      const next = eventHolding(trade.exposure, following[i][0].move, reference.costs);
      assert.ok(!h.liquidated && !next.liquidated && Number.isFinite(trade.value));
      if (crossfit) {
        const p = crossfit.policies[i], other = chooseEventTrade(p, a, after => fittedEventHolding(p, crossfitGrids![i], after, r.nextLeaf));
        const nextOther = eventHolding(other.exposure, following[i][0].move, reference.costs);
        assert.ok(!nextOther.liquidated && Number.isFinite(other.value));
        return Math.log(other.equity / trade.equity) + Math.log(nextOther.factor / next.factor) + settle(nextOther.exposure) - settle(next.exposure);
      }
      return Math.log(next.factor) + settle(next.exposure) - fittedEventHolding(reference, futureGrids[i], trade, r.nextLeaf);
    });
    targetCache.set(cell, values); return values;
  };
  const trace = read(source, `${phase.id}-${eventFittedSettingName(setting)}-d2-trades.json`), decisions = [];
  for (const raw of trace.filter((r: any) => r.orderQuantity)) {
    const i = byTime.get(raw.time)!, features = inputs(i)!, query = coord(reference, features), leaf = state(i);
    const account = { equity: raw.equityBefore, price: c[i].close, exposure: raw.exposureBefore };
    const choice = decideFittedEvent(original, features, account, 2, leaf), alternative = decideFittedEvent(crossfit ? candidate : reference, features, account, 2, leaf);
    near(choice.quantity, raw.order.quantity, 1e-8);
    const exit = eventTrade(account, 0, reference.costs)!;
    const comparisons = [{ name: crossfit ? "crossfit-choice" : "ordinary-choice", account: alternative }, { name: "exit", account: exit }];
    const weights = weightFor(query), candidateGrid = predictEventHoldingGrid(candidate, features, 2), referenceGrid = predictEventHoldingGrid(reference, features, 2);
    const rawGrid = (p: FittedEventValue) => p.tables[1].coefficients.map(b => b ? dot(b, query) : -Infinity);
    const oldRaw = rawGrid(reference), newRaw = rawGrid(candidate);
    const details = comparisons.map(comparison => {
      const cells = interpolation(reference, choice);
      for (const [cell, w] of interpolation(reference, comparison.account)) cells.set(cell, (cells.get(cell) ?? 0) - w);
      for (const [cell, w] of cells) if (Math.abs(w) < 1e-16) cells.delete(cell);
      const contrasts = rows.map(() => 0);
      for (const [cell, w] of cells) {
        assert.ok(reference.tables[1].coefficients[cell] && candidate.tables[1].coefficients[cell]);
        const delta = deltaTargets(cell); delta.forEach((v, j) => contrasts[j] += w * v);
      }
      const contributions = weights.map((w, j) => w * contrasts[j] * 10000);
      const rawDifference = [...cells].reduce((s, [cell, w]) => s + w * (newRaw[cell] - oldRaw[cell]), 0) * 10000;
      near(contributions.reduce((s, v) => s + v, 0), rawDifference, 1e-7);
      const value = (p: FittedEventValue, grid: number[], after: EventAccount) => Math.log(after.equity / account.equity) + fittedEventHolding(p, grid, after, leaf);
      const referenceAdvantage = (value(reference, referenceGrid, choice) - value(reference, referenceGrid, comparison.account)) * 10000;
      const candidateAdvantage = (value(candidate, candidateGrid, choice) - value(candidate, candidateGrid, comparison.account)) * 10000;
      const clampCorrection = [...cells].reduce((s, [cell, w]) => s + w * ((candidateGrid[cell] - newRaw[cell]) - (referenceGrid[cell] - oldRaw[cell])), 0) * 10000;
      near(candidateAdvantage - referenceAdvantage, rawDifference + clampCorrection, 1e-7);
      const attributions = paths.map((p, j) => ({ start: c[p.start].openTime + 60000, firstEnd: c[p.steps[0].end].openTime + 60000,
        nextEnd: c[p.steps[1].end].openTime + 60000, firstReturnBps: p.steps[0].return * 10000, nextReturnBps: p.steps[1].return * 10000,
        ...(crossfit ? { fold: foldIds[j] } : {}), weight: weights[j], deltaTargetContrastBps: contrasts[j] * 10000, contributionBps: contributions[j] }));
      const sorted = [...attributions].sort((a, b) => Math.abs(b.contributionBps) - Math.abs(a.contributionBps));
      const abs = contributions.reduce((s, v) => s + Math.abs(v), 0), days = new Map<string, { sum: number; abs: number }>();
      for (const r of attributions) { const day = new Date(r.start).toISOString().slice(0, 10), row = days.get(day) ?? { sum: 0, abs: 0 }; row.sum += r.contributionBps; row.abs += Math.abs(r.contributionBps); days.set(day, row); }
      const featureContributions = query.map((v, j) => ({ name: j ? names[j - 1] : "intercept", z: v,
        bps: v * [...cells].reduce((s, [cell, w]) => s + w * (candidate.tables[1].coefficients[cell]![j] - reference.tables[1].coefficients[cell]![j]), 0) * 10000 }));
      fs.writeFileSync(path.join(output, `${phase.id}-${raw.time}-${comparison.name}-paths.json`), JSON.stringify(attributions));
      return { comparison: comparison.name, alternativeExposure: comparison.account.exposure, referenceAdvantageBps: referenceAdvantage, candidateAdvantageBps: candidateAdvantage,
        rawDifferenceBps: rawDifference, cashClampCorrectionBps: clampCorrection, absoluteContributionBps: abs,
        top1AbsoluteShare: abs ? Math.abs(sorted[0].contributionBps) / abs : 0, top10AbsoluteShare: abs ? sorted.slice(0, 10).reduce((s, r) => s + Math.abs(r.contributionBps), 0) / abs : 0,
        topPaths: sorted.slice(0, 10), topDays: [...days].map(([date, r]) => ({ date, ...r })).sort((a, b) => Math.abs(b.sum) - Math.abs(a.sum)).slice(0, 10),
        ...(crossfit ? { foldContributions: crossfit.folds.map(fold => ({ fold: fold.index, from: fold.from, to: fold.to, rows: fold.test.length,
          changedPaths: fold.test.filter(i => Math.abs(contrasts[i]) > 1e-14).length,
          contributionBps: fold.test.reduce((s, i) => s + contributions[i], 0), absoluteBps: fold.test.reduce((s, i) => s + Math.abs(contributions[i]), 0) })) } : {}),
        featureContributions: featureContributions.sort((a, b) => Math.abs(b.bps) - Math.abs(a.bps)) };
    });
    decisions.push({ time: raw.time, account, selectedExposure: choice.exposure, nextReturnBps: raw.realizedReturnBps,
      weightSum: weights.reduce((s, v) => s + v, 0), negativeWeightMass: -weights.filter(v => v < 0).reduce((s, v) => s + v, 0),
      inverseSquaredWeightSum: 1 / weights.reduce((s, v) => s + v * v, 0), clippedInputs: features.map((v, j) => Math.abs((v - reference.means[j]) / reference.scales[j]) > 5), details });
  }
  const result = { phase, trainingPaths: paths.length, firstDepthAndPathSignaturesExact: true, attributionReconciled: true, decisions };
  results.push(result); fs.writeFileSync(path.join(output, `${phase.id}-summary.json`), JSON.stringify(result, null, 2));
  console.log(JSON.stringify({ phase: phase.id, trainingPaths: paths.length, decisions: decisions.length, evaluatedAccountCells: targetCache.size }));
}
fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify({ results, elapsedSec: (performance.now() - started) / 1000 }, null, 2));
