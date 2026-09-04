/** Measure same-data H1 selection effects before changing sampled-path backups. */
import fs from "node:fs";
import path from "node:path";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { eventFeatures, eventLeaf, type MoveSample } from "../packages/bot-algo/src/event-distribution.js";
import { chooseEventTrade, eventHolding, restoreEventPolicy } from "../packages/bot-algo/src/event-log-policy.js";
import { fittedEventHolding, predictEventHoldingGrid, trainEventFittedValue, type FittedEventValue } from "../packages/bot-algo/src/event-fitted-value.js";
import { eventFastVolatilityFeatures } from "../packages/bot-algo/src/event-size-sign.js";
import { eventCandleShapes, eventFuturesFeatures, eventFuturesBasisDeviations, eventFittedFuturesInputs, loadEventFuturesRows } from "./event-futures-basis.js";
import { eventFittedSettingName } from "./event-fitted-settings.js";
import { eventSignHorizonPaths, eventPolicyEvaluationFolds } from "./event-paths.js";
import { loadEventCandles, makeSamples } from "./research-event-policy.js";

const root = path.resolve(__dirname, ".."), DAY = 86400000;
const arg = (k: string) => { const i = process.argv.indexOf(`--${k}`); return i < 0 ? "" : process.argv[i + 1]; };
if (!arg("source") || !arg("output")) throw new Error("Specify sampled-path source and new output");
const source = path.resolve(root, "data/benchmarks", arg("source")), output = path.resolve(root, "data/benchmarks", arg("output"));
const read = (dir: string, file: string) => JSON.parse(fs.readFileSync(path.join(dir, file), "utf8"));
const parts = Number(arg("folds") || 6);
if (parts !== 6 && parts !== 20) throw new Error("Invalid diagnostic block count");
if (fs.existsSync(output)) throw new Error("Choose a new output directory");
const config = read(source, "config.json"), selected = config.settings.filter((s: any) => s.sampledPath);
if (config.contract !== "event-fitted-value-screen-v1" || selected.length !== 1 || selected[0].boost || selected[0].secondDynamics || selected[0].continuationFolds
  || !(selected[0].pathHorizon >= 2)) throw new Error("Requires one plain ridge sampled-path setting");
const setting = selected[0], jc = read(config.source, "config.json"), oc = read(jc.source, "config.json"), sc = read(oc.source, "config.json");
const fits = config.phases.map((p: any) => read(jc.source, `${jc.window.id}-${p.id}-model.json`));
const c = loadEventCandles(fits[0].trainStart - DAY, config.phases.at(-1).endTime);
const external = loadEventFuturesRows(fits[0].trainStart - DAY, config.phases.at(-1).endTime), cache = new Map<number, number[] | null>();
const inputs = (i: number) => {
  if (cache.has(i)) return cache.get(i)!;
  const e = eventFuturesFeatures(c, i, t => external.rows.get(t));
  const d = setting.historyMinutes ? eventFuturesBasisDeviations(c, i, t => external.rows.get(t)) : [];
  const shape = setting.candleShape ? eventCandleShapes(c, i, t => external.rows.get(t)) : [];
  const values = e && d && shape ? [...eventFeatures(c, i, sc.featureNames, sc.clock), ...eventFastVolatilityFeatures(c, i),
    ...eventFittedFuturesInputs(setting.basis, e, d), ...shape] : null;
  cache.set(i, values); return values;
};
const available = (i: number) => c[i].openTime + 60000;
fs.mkdirSync(output, { recursive: true });
fs.writeFileSync(path.join(output, "config.json"), JSON.stringify({ source, phases: config.phases, setting, parts, historyMs: DAY,
  sourceHash: createHash("sha256").update(fs.readFileSync(path.join(source, "config.json"))).update(external.fingerprint).digest("hex"),
  method: "Compare saved H1 with block-held-out and past-only H1 on the same second-event outcomes. The declared fixed calendar blocks span the training interval. Purge overlapping full one-day input/label support, including events ending beyond the nominal block. Refit normalization and all H1 heads. Past-only comparison starts after at least one third of the blocks as warmup. Probe initial exposures -max, cash, +max on the original account grid center, mark the first event, select an action using H1, then evaluate its realized second event and terminal settlement.",
  caveat: "Training-target diagnosis only; no policy selection or final inspector outcomes. Complement fits can use later pre-origin blocks. Common account axes and leaf liquidation guards remain the original full-training ones; only H1 heads and normalization are held out. Past-only is causal for these heads, not a fully refitted causal market model. Changed train size and market regime confound any inferred overfitting effect. Repeated paths/account probes are dependent." }, null, 2));
const files = ["scripts/audit-event-continuation-crossfit.ts", "scripts/event-paths.ts", "scripts/event-fitted-settings.ts", "scripts/event-futures-basis.ts",
  "scripts/research-event-policy.ts", "packages/bot-algo/src/event-fitted-value.ts", "packages/bot-algo/src/event-log-policy.ts"];
fs.writeFileSync(path.join(output, "sources.json"), JSON.stringify(Object.fromEntries(files.map(f => [f, fs.readFileSync(path.join(root, f), "utf8")]))));
type Probe = { time: number; end: number; fold: number; mode: string; initialExposure: number; selectedExposure: number; baselineExposure: number;
  selectedQuantity: number; baselineQuantity: number; predictedBps: number; actualBps: number; baselinePredictedBps: number; baselineActualBps: number;
  mse: number; baselineMse: number; correct: number; baselineCorrect: number };
function summarize(rows: Probe[]) {
  const mean = (fn: (r: Probe) => number) => rows.reduce((s, r) => s + fn(r), 0) / rows.length;
  return { count: rows.length, changedActionFraction: mean(r => Number(Math.abs(r.selectedQuantity - r.baselineQuantity) > 1e-8)),
    selectedActiveFraction: mean(r => Number(Math.abs(r.selectedExposure) > 1e-8)), baselineActiveFraction: mean(r => Number(Math.abs(r.baselineExposure) > 1e-8)),
    actualBps: mean(r => r.actualBps), baselineActualBps: mean(r => r.baselineActualBps), actualDifferenceBps: mean(r => r.actualBps - r.baselineActualBps),
    predictionErrorBps: mean(r => r.predictedBps - r.actualBps), baselinePredictionErrorBps: mean(r => r.baselinePredictedBps - r.baselineActualBps),
    mse: mean(r => r.mse), baselineMse: mean(r => r.baselineMse), accuracy: mean(r => r.correct), baselineAccuracy: mean(r => r.baselineCorrect) };
}
const results = [], started = performance.now();
for (const [index, phase] of config.phases.entries()) {
  const begin = performance.now(), fit = fits[index], base = restoreEventPolicy(read(config.source, `${phase.id}-policy.json`));
  const sampled: FittedEventValue = read(source, `${phase.id}-${eventFittedSettingName(setting)}-policy.json`);
  const state = (i: number) => eventLeaf(base.model, eventFeatures(c, i, base.model.featureNames, sc.clock));
  const transition = (r: MoveSample) => ({ features: inputs(r.start)!, nextFeatures: inputs(r.end)!, leaf: state(r.start), nextLeaf: state(r.end),
    move: { return: r.return, low: r.low, high: r.high, duration: r.duration } });
  const single = makeSamples(c, sc.clock, fit.trainStart, fit.trainEnd, [jc.window], sc.stride, "chain", sc.featureNames).filter(r => inputs(r.start) && inputs(r.end));
  const paths = eventSignHorizonPaths(single, setting.pathHorizon), rows = paths.map(p => transition(p.steps[0]));
  const following = paths.map(p => p.steps.slice(1).map(transition));
  trainEventFittedValue(base, rows, setting.penalty, 1, undefined, { ...sampled, tables: sampled.tables.slice(0, 1) }, { following });
  const full = trainEventFittedValue(base, rows, setting.penalty, 1);
  assert.deepEqual(full.tables[0], sampled.tables[0]); assert.deepEqual(full.means, sampled.means); assert.deepEqual(full.scales, sampled.scales);
  const folds = eventPolicyEvaluationFolds(paths.map(p => ({ start: available(p.start), end: available(p.steps[0].end) })),
    paths.map(p => ({ start: available(p.steps[1].start), end: available(p.steps[1].end) })), fit.trainStart, fit.trainEnd, parts, DAY);
  assert.equal(folds.reduce((sum, f) => sum + f.test.length, 0), rows.length);
  const probes: Probe[] = [], foldStats = [];
  const settlement = (x: number) => Math.log(1 - Math.abs(x) * (base.costs.feeBps + base.costs.slippageBps) / 10000);
  const metrics = (p: FittedEventValue, i: number, initialExposure: number) => {
    const r = rows[i], step = following[i][0], held = eventHolding(initialExposure, r.move, p.costs);
    assert.ok(!held.liquidated);
    const grid = predictEventHoldingGrid(p, step.features, 1);
    const account = { equity: p.equities[2] * held.factor, price: p.prices[1] * (1 + r.move.return), exposure: held.exposure };
    const trade = chooseEventTrade(p, account, after => fittedEventHolding(p, grid, after, step.leaf));
    const after = eventHolding(trade.exposure, step.move, p.costs);
    assert.ok(!after.liquidated && Number.isFinite(trade.value));
    const actualBps = (Math.log(trade.equity / account.equity) + Math.log(after.factor) + settlement(after.exposure)) * 10000;
    const predictions = [-p.costs.maxLeverage, p.costs.maxLeverage].map(exposure => {
      const h = eventHolding(exposure, step.move, p.costs); assert.ok(!h.liquidated);
      return { predicted: fittedEventHolding(p, grid, { ...account, exposure }, step.leaf), actual: Math.log(h.factor) + settlement(h.exposure) };
    });
    assert.ok(predictions.every(r => Number.isFinite(r.predicted)));
    return { selectedExposure: trade.exposure, selectedQuantity: trade.quantity, predictedBps: trade.value * 10000, actualBps,
      mse: predictions.reduce((s, r) => s + (r.predicted - r.actual) ** 2, 0) / 2,
      correct: Number(Math.sign(predictions[1].predicted - predictions[0].predicted) === Math.sign(predictions[1].actual - predictions[0].actual)) };
  };
  const exposures = [-base.costs.maxLeverage, 0, base.costs.maxLeverage];
  for (const fold of folds) {
    if (!fold.test.length) continue;
    const models: Array<{ mode: string; ids: number[] }> = [{ mode: "complement", ids: fold.complement }];
    if (fold.index >= Math.ceil(parts / 3)) models.push({ mode: "past", ids: fold.past });
    const baseline = new Map(fold.test.map(i => [i, exposures.map(x => metrics(full, i, x))]));
    for (const { mode, ids } of models) {
      assert.ok(ids.length >= 100, "Insufficient held-out training rows");
      const p = trainEventFittedValue(base, ids.map(i => rows[i]), setting.penalty, 1);
      fs.writeFileSync(path.join(output, `${phase.id}-fold${fold.index}-${mode}-policy.json`), JSON.stringify(p));
      const selectedRows: Probe[] = [];
      for (const i of fold.test) for (const [j, initialExposure] of exposures.entries()) {
        const original = baseline.get(i)![j], value = metrics(p, i, initialExposure);
        const probe = { time: available(paths[i].steps[1].start), end: available(paths[i].steps[1].end), fold: fold.index, mode, initialExposure, ...value,
          baselineExposure: original.selectedExposure, baselineQuantity: original.selectedQuantity, baselinePredictedBps: original.predictedBps,
          baselineActualBps: original.actualBps, baselineMse: original.mse, baselineCorrect: original.correct };
        probes.push(probe); selectedRows.push(probe);
      }
      foldStats.push({ fold: fold.index, mode, from: fold.from, to: fold.to, supportStart: fold.supportStart, supportEnd: fold.supportEnd,
        training: ids.length, evaluation: fold.test.length, ...summarize(selectedRows) });
    }
  }
  const aggregate = ["complement", "past"].map(mode => ({ mode, ...summarize(probes.filter(r => r.mode === mode)),
    byExposure: exposures.map(initialExposure => ({ initialExposure, ...summarize(probes.filter(r => r.mode === mode && r.initialExposure === initialExposure)) })) }));
  const result = { phase, trainingPaths: paths.length, baselineExact: true, foldStats, aggregate, elapsedSec: (performance.now() - begin) / 1000 };
  results.push(result);
  fs.writeFileSync(path.join(output, `${phase.id}-probes.json`), JSON.stringify(probes));
  fs.writeFileSync(path.join(output, `${phase.id}-folds.json`), JSON.stringify(folds));
  fs.writeFileSync(path.join(output, `${phase.id}-summary.json`), JSON.stringify(result, null, 2));
  console.log(JSON.stringify({ phase: phase.id, trainingPaths: paths.length, aggregate: aggregate.map(({ byExposure, ...r }) => r), elapsedSec: result.elapsedSec }));
}
fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify({ results, elapsedSec: (performance.now() - started) / 1000 }, null, 2));
