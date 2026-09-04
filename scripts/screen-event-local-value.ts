/** Matched one-event forecast gate for a positive empirical neighbor law. */
import fs from "node:fs";
import path from "node:path";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { eventFeatures, eventLeaf, type MoveSample } from "../packages/bot-algo/src/event-distribution.js";
import { eventHolding, restoreEventPolicy } from "../packages/bot-algo/src/event-log-policy.js";
import { fittedEventHolding, predictEventHoldingGrid, trainEventFittedValue, type FittedEventValue } from "../packages/bot-algo/src/event-fitted-value.js";
import { eventFastVolatilityFeatures } from "../packages/bot-algo/src/event-size-sign.js";
import { eventCandleShapes, eventFuturesFeatures, eventFuturesBasisDeviations, eventFittedFuturesInputs, loadEventFuturesRows } from "./event-futures-basis.js";
import { eventFittedSettingName } from "./event-fitted-settings.js";
import { eventSignHorizonPaths } from "./event-paths.js";
import { eventLocalValuePredictor } from "./event-local-value.js";
import { loadEventCandles, makeSamples } from "./research-event-policy.js";

const root = path.resolve(__dirname, ".."), DAY = 86400000;
const arg = (k: string) => { const i = process.argv.indexOf(`--${k}`); return i < 0 ? "" : process.argv[i + 1]; };
if (!arg("source") || !arg("output")) throw new Error("Specify sampled screen and new output");
const source = path.resolve(root, "data/benchmarks", arg("source")), output = path.resolve(root, "data/benchmarks", arg("output"));
const read = (dir: string, file: string) => JSON.parse(fs.readFileSync(path.join(dir, file), "utf8"));
if (fs.existsSync(output)) throw new Error("Choose a new output directory");
const config = read(source, "config.json"), choices = config.settings.filter((s: any) => s.sampledPath);
if (config.contract !== "event-fitted-value-screen-v1" || choices.length !== 1 || choices[0].boost || choices[0].secondDynamics
  || choices[0].continuationFolds || !choices[0].pathHorizon) throw new Error("Requires one plain sampled setting");
const setting = choices[0], jc = read(config.source, "config.json"), oc = read(jc.source, "config.json"), sc = read(oc.source, "config.json");
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
const counts = [32, 128, 512];
fs.mkdirSync(output, { recursive: true });
fs.writeFileSync(path.join(output, "config.json"), JSON.stringify({ source, phases: config.phases, setting, counts,
  sourceHash: createHash("sha256").update(fs.readFileSync(path.join(source, "config.json"))).update(external.fingerprint).digest("hex"),
  method: "Uniform averages of the 32/128/512 nearest matched historical events in the reference H1 training-standardized, clipped feature space. Shared nonnegative weights predict long/short holding log values, arithmetic return, duration and sign probability. The matched complete-path cohort, features, coordinates and one-event targets exactly match the reference ridge screen. Only prior-origin forecasts are scored; no Bellman expansion or trading-policy selection.",
  caveat: "Research origins reused. Small neighborhoods may have high variance and overlapping observations; positive weights alone do not establish predictive edge. Full-exposure holding-value forecasts include terminal settlement, not entry execution or sequential backtest returns. Sign Brier/accuracy excludes exactly flat returns; reported up probability is conditional on nonflat neighbors." }, null, 2));
const files = ["scripts/screen-event-local-value.ts", "scripts/event-local-value.ts", "scripts/event-paths.ts", "scripts/event-futures-basis.ts", "scripts/event-fitted-settings.ts",
  "scripts/research-event-policy.ts", "packages/bot-algo/src/event-fitted-value.ts", "packages/bot-algo/src/event-log-policy.ts"];
fs.writeFileSync(path.join(output, "sources.json"), JSON.stringify(Object.fromEntries(files.map(f => [f, fs.readFileSync(path.join(root, f), "utf8")]))));
const results = [], started = performance.now();
for (const [index, phase] of config.phases.entries()) {
  const begin = performance.now(), fit = fits[index], base = restoreEventPolicy(read(config.source, `${phase.id}-policy.json`));
  const reference: FittedEventValue = read(source, `${phase.id}-${eventFittedSettingName(setting)}-policy.json`);
  const state = (i: number) => eventLeaf(base.model, eventFeatures(c, i, base.model.featureNames, sc.clock));
  const transition = (r: MoveSample) => ({ features: inputs(r.start)!, nextFeatures: inputs(r.end)!, leaf: state(r.start), nextLeaf: state(r.end),
    move: { return: r.return, low: r.low, high: r.high, duration: r.duration } });
  const training = makeSamples(c, sc.clock, fit.trainStart, fit.trainEnd, [jc.window], sc.stride, "chain", sc.featureNames).filter(r => inputs(r.start) && inputs(r.end));
  const validation = makeSamples(c, sc.clock, phase.startTime, phase.endTime, [], sc.stride, "chain", sc.featureNames).filter(r => inputs(r.start) && inputs(r.end));
  const trainPaths = eventSignHorizonPaths(training, setting.pathHorizon), testPaths = eventSignHorizonPaths(validation, setting.pathHorizon);
  const train = trainPaths.map(p => p.steps[0]), test = testPaths.map(p => p.steps[0]);
  trainEventFittedValue(base, train.map(transition), setting.penalty, 1, undefined, { ...reference, tables: reference.tables.slice(0, 1) },
    { following: trainPaths.map(p => p.steps.slice(1).map(transition)) });
  const coordinates = (i: number) => inputs(i)!.map((v, f) => Math.max(-5, Math.min(5, (v - reference.means[f]) / reference.scales[f])));
  const value = (r: MoveSample, exposure: number) => {
    const h = eventHolding(exposure, r, base.costs); assert.ok(!h.liquidated);
    return Math.log(h.factor) + Math.log(1 - Math.abs(h.exposure) * (base.costs.feeBps + base.costs.slippageBps) / 10000);
  };
  const targets = train.map(r => [value(r, -base.costs.maxLeverage), value(r, base.costs.maxLeverage), Number(r.return > 0), Number(r.return !== 0), r.return, r.duration]);
  const design = train.map(r => coordinates(r.start)), predict = eventLocalValuePredictor(design, targets);
  fs.writeFileSync(path.join(output, `${phase.id}-model.json`), JSON.stringify({ means: reference.means, scales: reference.scales, counts, design, targets,
    events: train.map(r => ({ start: c[r.start].openTime + 60000, end: c[r.end].openTime + 60000, return: r.return, low: r.low, high: r.high, duration: r.duration })) }));
  const predictions = test.map(r => {
    const actual = [-base.costs.maxLeverage, base.costs.maxLeverage].map(x => value(r, x)), leaf = state(r.start), grid = predictEventHoldingGrid(reference, inputs(r.start)!, 1);
    const ridge = [-base.costs.maxLeverage, base.costs.maxLeverage].map(exposure => fittedEventHolding(reference, grid, { equity: base.equities[2], price: c[r.start].close, exposure }, leaf));
    const local = predict(coordinates(r.start), counts).map(({ count, values, radius, indices }) => ({ count, holding: values.slice(0, 2),
      upProbability: values[3] ? values[2] / values[3] : 0.5, flatProbability: 1 - values[3], meanReturn: values[4], meanDuration: values[5], radius,
      distinctTrainingDays: new Set(indices.map(i => Math.floor(c[train[i].start].openTime / DAY))).size }));
    return { time: c[r.start].openTime + 60000, availableAt: c[r.end].openTime + 60000, actual, return: r.return, duration: r.duration, ridge, local };
  });
  const mse = (predicted: number[], actual: number[]) => predicted.reduce((s, v, i) => s + (v - actual[i]) ** 2, 0) / actual.length;
  const mean = (values: number[]) => values.reduce((s, v) => s + v, 0) / values.length;
  const ridgeMse = mean(predictions.map(r => mse(r.ridge, r.actual))), zero = Math.log(1 - base.costs.maxLeverage * (base.costs.feeBps + base.costs.slippageBps) / 10000);
  const zeroMse = mean(predictions.map(r => mse([zero, zero], r.actual)));
  const old = read(source, `${phase.id}-scores.json`).rows.find((r: any) => r.sampledPath);
  assert.equal(train.length, old.samples); assert.equal(test.length, old.validation); assert.ok(Math.abs(ridgeMse - old.mse) < 1e-16);
  const rows = counts.map((count, i) => {
    const valueMse = mean(predictions.map(r => mse(r.local[i].holding, r.actual))), directional = predictions.filter(r => r.return !== 0);
    return { count, mse: valueMse, skill: 1 - valueMse / zeroMse, differenceFromRidge: valueMse - ridgeMse,
      holdingDirectionAccuracy: mean(predictions.map(r => Number(Math.sign(r.local[i].holding[1] - r.local[i].holding[0]) === Math.sign(r.actual[1] - r.actual[0])))),
      signAccuracy: mean(directional.map(r => Number((r.local[i].upProbability > 0.5) === (r.return > 0)))),
      brier: mean(directional.map(r => (r.local[i].upProbability - Number(r.return > 0)) ** 2)),
      meanRadius: mean(predictions.map(r => r.local[i].radius)), meanDistinctDays: mean(predictions.map(r => r.local[i].distinctTrainingDays)) };
  });
  const result = { phase, training: train.length, validation: test.length, ridgeMse, zeroMse, referenceReproduced: true, rows, elapsedSec: (performance.now() - begin) / 1000 };
  results.push(result); fs.writeFileSync(path.join(output, `${phase.id}-predictions.json`), JSON.stringify(predictions));
  console.log(JSON.stringify(result));
}
const ranking = counts.map(count => ({ count, meanMse: results.reduce((s, r) => s + r.rows.find(v => v.count === count)!.mse, 0) / results.length,
  meanSkill: results.reduce((s, r) => s + r.rows.find(v => v.count === count)!.skill, 0) / results.length,
  meanDifferenceFromRidge: results.reduce((s, r) => s + r.rows.find(v => v.count === count)!.differenceFromRidge, 0) / results.length })).sort((a, b) => a.meanMse - b.meanMse);
fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify({ results, ranking, elapsedSec: (performance.now() - started) / 1000 }, null, 2));
console.log(JSON.stringify({ ranking, elapsedSec: (performance.now() - started) / 1000 }));
