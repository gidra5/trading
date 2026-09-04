/** Direct multi-event holding-value forecast gate, without choosing future orders. */
import fs from "node:fs";
import path from "node:path";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { eventFeatures, eventLeaf, type MoveSample } from "../packages/bot-algo/src/event-distribution.js";
import { restoreEventPolicy } from "../packages/bot-algo/src/event-log-policy.js";
import { fittedEventHolding, predictEventHoldingGrid, trainEventFittedValue, type FittedEventValue } from "../packages/bot-algo/src/event-fitted-value.js";
import { eventFastVolatilityFeatures } from "../packages/bot-algo/src/event-size-sign.js";
import { eventFuturesFeatures, eventFuturesBasisDeviations, eventFittedFuturesInputs, loadEventFuturesRows } from "./event-futures-basis.js";
import { eventFittedSettingName } from "./event-fitted-settings.js";
import { eventPathHolding, eventSignHorizonPaths } from "./event-paths.js";
import { eventRidgeInfluence } from "./event-ridge-influence.js";
import { loadEventCandles, makeSamples } from "./research-event-policy.js";

const root = path.resolve(__dirname, ".."), DAY = 86400000;
const arg = (k: string) => { const i = process.argv.indexOf(`--${k}`); return i < 0 ? "" : process.argv[i + 1]; };
if (!arg("source") || !arg("output")) throw new Error("Specify sampled-path screen and new output");
const source = path.resolve(root, "data/benchmarks", arg("source")), output = path.resolve(root, "data/benchmarks", arg("output"));
const read = (dir: string, file: string) => JSON.parse(fs.readFileSync(path.join(dir, file), "utf8"));
if (fs.existsSync(output)) throw new Error("Choose a new output directory");
const config = read(source, "config.json"), choices = config.settings.filter((s: any) => s.sampledPath);
if (config.contract !== "event-fitted-value-screen-v1" || choices.length !== 1 || choices[0].boost || choices[0].secondDynamics
  || choices[0].candleShape || choices[0].continuationFolds || choices[0].pathHorizon !== 3) throw new Error("Requires one plain three-event sampled setting");
const setting = choices[0], jc = read(config.source, "config.json"), oc = read(jc.source, "config.json"), sc = read(oc.source, "config.json");
const fits = config.phases.map((p: any) => read(jc.source, `${jc.window.id}-${p.id}-model.json`));
const c = loadEventCandles(fits[0].trainStart - DAY, config.phases.at(-1).endTime);
const external = loadEventFuturesRows(fits[0].trainStart - DAY, config.phases.at(-1).endTime), cache = new Map<number, number[] | null>();
const inputs = (i: number) => {
  if (cache.has(i)) return cache.get(i)!;
  const e = eventFuturesFeatures(c, i, t => external.rows.get(t));
  const d = setting.historyMinutes ? eventFuturesBasisDeviations(c, i, t => external.rows.get(t)) : [];
  const values = e && d ? [...eventFeatures(c, i, sc.featureNames, sc.clock), ...eventFastVolatilityFeatures(c, i),
    ...eventFittedFuturesInputs(setting.basis, e, d)] : null;
  cache.set(i, values); return values;
};
const dcIndex = sc.featureNames.indexOf("dcDirection");
if (dcIndex < 0) throw new Error("Clock-state baseline needs DC direction");
const mean = (v: number[]) => v.reduce((s, x) => s + x, 0) / v.length;
const mse = (predicted: number[], actual: number[]) => mean(predicted.map((v, i) => (v - actual[i]) ** 2));
fs.mkdirSync(output, { recursive: true });
fs.writeFileSync(path.join(output, "config.json"), JSON.stringify({ contract: "event-hold-path-screen-v1", source, phases: config.phases, setting,
  sourceHash: createHash("sha256").update(fs.readFileSync(path.join(source, "config.json"))).update(JSON.stringify(c)).update(external.fingerprint).digest("hex"),
  method: "Direct fixed-inventory holding log values through one/two/three complete events. Use the reference H1 features, normalization, clipped coordinates and ridge penalty on the identical three-event cohort. Mark exposure and borrowing at each event, charge settlement once at each prefix. Separate holding and post-first-event increment forecast errors. Compare training-only DC-state means and a zero-return/zero-borrow terminal-fee baseline.",
  caveat: "Forecast diagnostic only. No entry/intermediate orders, no policy improvement assertion, no final outcomes or policy selection. Report target-cap breaches before continuation; an unrebalanced path that breaches the cap is not a feasible option under the current event policy. Full-position terminal settlement follows the existing forecast convention; quantity/minimum/maximum order constraints would need an executable option model before backtesting. Reused, overlapping research paths." }, null, 2));
const files = ["scripts/screen-event-hold-paths.ts", "scripts/event-paths.ts", "scripts/event-ridge-influence.ts", "scripts/event-fitted-settings.ts",
  "scripts/event-futures-basis.ts", "scripts/research-event-policy.ts", "packages/bot-algo/src/event-fitted-value.ts", "packages/bot-algo/src/event-log-policy.ts"];
fs.writeFileSync(path.join(output, "sources.json"), JSON.stringify(Object.fromEntries(files.map(f => [f, fs.readFileSync(path.join(root, f), "utf8")]))));
fs.writeFileSync(path.join(output, "external-sources.json"), JSON.stringify({ ...external, rows: undefined }));
const results: any[] = [], started = performance.now();
for (const [index, phase] of config.phases.entries()) {
  const begin = performance.now(), fit = fits[index], base = restoreEventPolicy(read(config.source, `${phase.id}-policy.json`));
  const reference: FittedEventValue = read(source, `${phase.id}-${eventFittedSettingName(setting)}-policy.json`);
  const state = (i: number) => eventLeaf(base.model, eventFeatures(c, i, base.model.featureNames, sc.clock));
  const transition = (r: MoveSample) => ({ features: inputs(r.start)!, nextFeatures: inputs(r.end)!, leaf: state(r.start), nextLeaf: state(r.end),
    move: { return: r.return, low: r.low, high: r.high, duration: r.duration } });
  const training = makeSamples(c, sc.clock, fit.trainStart, fit.trainEnd, [jc.window], sc.stride, "chain", sc.featureNames).filter(r => inputs(r.start) && inputs(r.end));
  const validation = makeSamples(c, sc.clock, phase.startTime, phase.endTime, [], sc.stride, "chain", sc.featureNames).filter(r => inputs(r.start) && inputs(r.end));
  const train = eventSignHorizonPaths(training, 3), test = eventSignHorizonPaths(validation, 3);
  assert.ok(train.every(p => c[p.end].openTime + 60000 < phase.startTime));
  assert.ok(test.every(p => c[p.end].openTime + 60000 < phase.endTime));
  trainEventFittedValue(base, train.map(p => transition(p.steps[0])), setting.penalty, 1, undefined, { ...reference, tables: reference.tables.slice(0, 1) },
    { following: train.map(p => p.steps.slice(1).map(transition)) });
  const old = read(source, `${phase.id}-scores.json`).rows.find((r: any) => r.sampledPath);
  assert.equal(train.length, old.samples); assert.equal(test.length, old.validation);
  const coordinates = (i: number) => [1, ...inputs(i)!.map((v, f) => Math.max(-5, Math.min(5, (v - reference.means[f]) / reference.scales[f])))];
  const values = (steps: MoveSample[]) => {
    const paths = [-base.costs.maxLeverage, base.costs.maxLeverage].map(x => eventPathHolding(x, steps, base.costs));
    if (paths.some(p => p.some(r => r.liquidated || !Number.isFinite(r.value)))) throw new Error("Nonfinite holding target; do not drop ruin paths");
    return [0, 1, 2].map(h => paths.map(p => p[h]));
  };
  const observed = train.map(p => values(p.steps)), targets = observed.map(p => p.flatMap(h => h.map(r => r.value)));
  const design = train.map(p => coordinates(p.start)), weights = eventRidgeInfluence(design, setting.penalty);
  const groups = [-1, 0, 1].map(direction => {
    const indices = train.flatMap((p, i) => inputs(p.start)![dcIndex] === direction ? [i] : []);
    const selected = indices.length ? indices : train.map((_, i) => i);
    return { direction, count: indices.length, values: targets[0].map((_, f) => mean(selected.map(i => targets[i][f]))) };
  });
  fs.writeFileSync(path.join(output, `${phase.id}-model.json`), JSON.stringify({ means: reference.means, scales: reference.scales,
    penalty: setting.penalty, design, targets, groups, referenceTrainingSignature: reference.trainingSignature,
    events: train.map(p => ({ start: c[p.start].openTime + 60000, end: c[p.end].openTime + 60000 })) }));
  let maximumH1Error = 0;
  const predictions = test.map(p => {
    const w = weights(coordinates(p.start));
    const predicted = targets[0].map((_, f) => w.reduce((s, value, i) => s + value * targets[i][f], 0));
    const grid = predictEventHoldingGrid(reference, inputs(p.start)!, 1), leaf = state(p.start);
    const original = [-base.costs.maxLeverage, base.costs.maxLeverage].map(exposure => fittedEventHolding(reference, grid,
      { equity: base.equities[2], price: c[p.start].close, exposure }, leaf));
    for (let i = 0; i < 2; i++) maximumH1Error = Math.max(maximumH1Error, Math.abs(original[i] - predicted[i]));
    const actual = values(p.steps), clock = groups.find(g => g.direction === inputs(p.start)![dcIndex])!.values;
    return { time: c[p.start].openTime + 60000, horizons: actual.map((h, i) => ({ horizon: i + 1,
      availableAt: c[p.steps[i].end].openTime + 60000, duration: p.cumulativeMinutes[i], actual: h.map(r => r.value),
      predicted: predicted.slice(i * 2, i * 2 + 2), clock: clock.slice(i * 2, i * 2 + 2), capBreaches: h.map(r => r.capBreaches) })) };
  });
  assert.ok(maximumH1Error < 1e-12, `H1 ridge reconstruction error ${maximumH1Error}`);
  const zero = Math.log(1 - base.costs.maxLeverage * (base.costs.feeBps + base.costs.slippageBps) / 10000);
  const rows = [0, 1, 2].map(h => {
    const selected = predictions.map(p => p.horizons[h]), valueMse = mean(selected.map(r => mse(r.predicted, r.actual)));
    const zeroMse = mean(selected.map(r => mse([zero, zero], r.actual))), clockMse = mean(selected.map(r => mse(r.clock, r.actual)));
    const increment = h ? predictions.map(p => ({ actual: p.horizons[h].actual.map((v, i) => v - p.horizons[0].actual[i]),
      predicted: p.horizons[h].predicted.map((v, i) => v - p.horizons[0].predicted[i]), clock: p.horizons[h].clock.map((v, i) => v - p.horizons[0].clock[i]) })) : [];
    return { horizon: h + 1, mse: valueMse, zeroMse, clockMse, skill: 1 - valueMse / zeroMse, clockSkill: 1 - valueMse / clockMse,
      meanDuration: mean(selected.map(r => r.duration)), capBreachFraction: [0, 1].map(i => mean(selected.map(r => Number(r.capBreaches[i] > 0)))),
      increment: h ? { mse: mean(increment.map(r => mse(r.predicted, r.actual))), zeroMse: mean(increment.map(r => mse([0, 0], r.actual))),
        clockMse: mean(increment.map(r => mse(r.clock, r.actual))) } : null };
  });
  assert.ok(Math.abs(rows[0].mse - old.mse) < 1e-16);
  const result = { phase, training: train.length, validation: test.length, referenceReproduced: true, maximumH1Error,
    rows, elapsedSec: (performance.now() - begin) / 1000 };
  results.push(result); fs.writeFileSync(path.join(output, `${phase.id}-predictions.json`), JSON.stringify(predictions));
  console.log(JSON.stringify(result));
}
const horizons = [1, 2, 3].map(horizon => { const rows = results.map(r => r.rows[horizon - 1]);
  return { horizon, meanSkill: mean(rows.map(r => r.skill)), meanClockSkill: mean(rows.map(r => r.clockSkill)),
    meanMse: mean(rows.map(r => r.mse)), meanClockMse: mean(rows.map(r => r.clockMse)),
    meanIncrementSkill: horizon > 1 ? mean(rows.map(r => 1 - r.increment.mse / r.increment.zeroMse)) : null,
    meanIncrementClockSkill: horizon > 1 ? mean(rows.map(r => 1 - r.increment.mse / r.increment.clockMse)) : null };
});
fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify({ results, horizons, elapsedSec: (performance.now() - started) / 1000 }, null, 2));
console.log(JSON.stringify({ horizons, elapsedSec: (performance.now() - started) / 1000 }));
