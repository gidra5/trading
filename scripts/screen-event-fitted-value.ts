/** Prior-origin fitted Bellman-value screen; no final-window evaluation. */
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { eventFeatures, eventLeaf, type MoveSample } from "../packages/bot-algo/src/event-distribution.js";
import { eventHolding, restoreEventPolicy } from "../packages/bot-algo/src/event-log-policy.js";
import { fittedEventHolding, predictEventHoldingGrid, trainEventFittedValue, type EventValueTransition } from "../packages/bot-algo/src/event-fitted-value.js";
import { eventFastVolatilityFeatures } from "../packages/bot-algo/src/event-size-sign.js";
import { eventCandleShapes, eventFuturesFeatures, eventFuturesBasisDeviations, eventFittedFuturesInputs, loadEventFuturesRows } from "./event-futures-basis.js";
import { loadEventCandles, makeSamples, replayEventPolicy } from "./research-event-policy.js";
import { eventOriginScore, eventRefitOrigins } from "./research-event-refits.js";
import { eventFittedSettingName as key, type EventFittedSetting } from "./event-fitted-settings.js";
import { eventSecondDynamicsAt, loadEventSecondDynamics } from "./event-second-dynamics.js";
import { eventSignHorizonPaths } from "./event-paths.js";
import { fitEventCrossfitContinuations } from "./event-fitted-crossfit.js";

const root = path.resolve(__dirname, ".."), DAY = 86_400_000;
const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
if (!arg("source") || !arg("output")) throw new Error("Specify joint-policy source and new output");
const source = path.resolve(root, "data/benchmarks", arg("source")), output = path.resolve(root, "data/benchmarks", arg("output"));
const read = (dir: string, file: string) => JSON.parse(fs.readFileSync(path.join(dir, file), "utf8"));
if (fs.existsSync(output)) throw new Error("Choose a new output directory");
const jc = read(source, "config.json"), oc = read(jc.source, "config.json"), sc = read(oc.source, "config.json");
if (jc.contract !== "event-volatility-law-policy-v1") throw new Error("Requires joint event source");
const phases = eventRefitOrigins(jc.window.startTime, oc.foldCount, oc.foldDays), depths = Number(arg("depths") || 1);
if (!Number.isInteger(depths) || depths < 1 || depths > 8) throw new Error("Invalid fitted depth");
const fits = phases.map(p => read(jc.source, `${jc.window.id}-${p.id}-model.json`));
const selectedSource = arg("settings-source") ? path.resolve(root, "data/benchmarks", arg("settings-source")) : undefined;
const selected = selectedSource ? read(selectedSource, "summary.json").forecastRanking[0] : undefined;
if (selectedSource && read(selectedSource, "config.json").source !== source) throw new Error("Head selection source mismatch");
const baseSettings: EventFittedSetting[] = selected ? (arg("basis") ? arg("basis").split(",") : [selected.basis]).map(basis => ({ basis, penalty: selected.penalty,
  ...(selected.boost ? { boost: selected.boost } : {}), ...(selected.historyMinutes ? { historyMinutes: selected.historyMinutes } : {}),
  ...(selected.secondDynamics ? { secondDynamics: true } : {}), ...(selected.candleShape ? { candleShape: true } : {}),
  ...(selected.pathHorizon ? { pathHorizon: selected.pathHorizon } : {}),
  ...(selected.sampledPath ? { sampledPath: true } : {}), ...(selected.continuationFolds ? { continuationFolds: selected.continuationFolds } : {}) }))
  : (arg("basis") ? arg("basis").split(",") : ["spot", "price", "all"]).flatMap(basis => [0.01, 0.1, 1].map(penalty => ({ basis, penalty })));
if (baseSettings.some(s => !["spot", "price", "all", "deviation", "centered"].includes(s.basis))) throw new Error("Invalid fitted feature basis");
const treeCounts = arg("boost-trees") ? arg("boost-trees").split(",").map(Number) : undefined;
if (treeCounts?.some(v => !Number.isInteger(v) || v < 0 || v > 256)) throw new Error("Invalid residual tree counts");
let settings: EventFittedSetting[] = treeCounts ? baseSettings.flatMap(s => treeCounts.map(trees => ({ ...s, boost: undefined,
  ...(trees ? { boost: { trees, depth: 2, minLeaf: Number(arg("boost-min-leaf") || 128), rate: 0.05 } } : {}) }))) : baseSettings;
if (settings.some(s => ["deviation", "centered"].includes(s.basis) || s.historyMinutes === 240)) for (const s of settings) s.historyMinutes = 240;
if (arg("second-dynamics")) {
  const choices = arg("second-dynamics").split(",").map(Number);
  if (choices.some(v => v !== 0 && v !== 1) || new Set(choices).size !== choices.length) throw new Error("Invalid second-dynamics choices");
  settings = settings.flatMap(s => choices.map(v => ({ ...s, secondDynamics: v ? true : undefined })));
}
if (arg("candle-shape")) {
  const choices = arg("candle-shape").split(",").map(Number);
  if (choices.some(v => v !== 0 && v !== 1) || new Set(choices).size !== choices.length) throw new Error("Invalid candle-shape choices");
  settings = settings.flatMap(s => choices.map(v => ({ ...s, candleShape: v ? true : undefined })));
}
if (arg("path-horizon")) {
  const horizon = Number(arg("path-horizon"));
  if (!Number.isInteger(horizon) || horizon < depths || horizon > 8) throw new Error("Invalid path horizon");
  settings = settings.map(s => ({ ...s, pathHorizon: horizon }));
}
if (arg("sampled-path")) {
  const choices = arg("sampled-path").split(",").map(Number);
  if (choices.some(v => v !== 0 && v !== 1) || new Set(choices).size !== choices.length
    || settings.some(s => !s.pathHorizon || s.pathHorizon < depths)) throw new Error("Invalid sampled-path choices/horizon");
  settings = settings.flatMap(s => choices.map(v => ({ ...s, sampledPath: v ? true : undefined })));
}
if (settings.some(s => s.sampledPath && (!s.pathHorizon || s.pathHorizon < depths))) throw new Error("Missing completed path horizon");
if (arg("continuation-folds")) {
  const parts = Number(arg("continuation-folds"));
  if (parts !== 6 && parts !== 20) throw new Error("Only the fixed six/twenty-block diagnostics are supported");
  settings = settings.map(s => ({ ...s, continuationFolds: parts }));
}
if (settings.some(s => s.continuationFolds && (!s.sampledPath || depths > 2))) throw new Error("Crossfit continuation requires sampled depth one/two");
const resumeSource = arg("resume-source") ? path.resolve(root, "data/benchmarks", arg("resume-source")) : undefined;
if (resumeSource) {
  const previous = read(resumeSource, "config.json");
  if (previous.contract !== "event-fitted-value-screen-v1" || previous.source !== source || previous.depths >= depths
    || JSON.stringify(previous.phases) !== JSON.stringify(phases)
    || settings.some(s => !previous.settings.some((p: EventFittedSetting) => JSON.stringify(p) === JSON.stringify(s))))
    throw new Error("Incompatible fitted screen checkpoint");
}
const c = loadEventCandles(fits[0].trainStart - DAY, phases.at(-1)!.endTime);
const external = loadEventFuturesRows(fits[0].trainStart - DAY, phases.at(-1)!.endTime);
const seconds = settings.some(s => s.secondDynamics) ? loadEventSecondDynamics(fits[0].trainStart - DAY, phases.at(-1)!.endTime) : undefined;
if (seconds) console.log(JSON.stringify({ secondDynamics: { built: seconds.built, reused: seconds.reused, observations: seconds.rows.size, elapsedSec: seconds.elapsedSec } }));
const byTime = new Map(c.map((r, i) => [r.openTime + 60000, i])), cache = new Map<number, ReturnType<typeof eventFuturesFeatures>>();
const extras = (i: number) => { if (!cache.has(i)) cache.set(i, eventFuturesFeatures(c, i, t => external.rows.get(t))); return cache.get(i)!; };
const deviationCache = new Map<number, number[] | null>(), needsDeviation = settings.some(s => s.historyMinutes === 240);
const deviation = (i: number) => { if (!deviationCache.has(i)) deviationCache.set(i, eventFuturesBasisDeviations(c, i, t => external.rows.get(t))); return deviationCache.get(i)!; };
const shapes = (i: number) => eventCandleShapes(c, i, t => external.rows.get(t));
const needsShape = settings.some(s => s.candleShape);
const matched = (i: number) => extras(i) && (!needsDeviation || deviation(i)) && (!needsShape || shapes(i));
const inputs = (i: number, setting: EventFittedSetting) => {
  const { basis } = setting;
  const e = extras(i); if (!e) throw new Error("Missing matched external observation");
  const d = ["deviation", "centered"].includes(basis) ? deviation(i) : []; if (!d) throw new Error("Missing completed basis deviation");
  const shape = setting.candleShape ? shapes(i) : []; if (!shape) throw new Error("Missing completed candle shape");
  return [...eventFeatures(c, i, sc.featureNames, sc.clock), ...eventFastVolatilityFeatures(c, i),
    ...eventFittedFuturesInputs(basis, e, d), ...(setting.secondDynamics ? eventSecondDynamicsAt(seconds!.rows, c[i]) : []), ...shape];
};
fs.mkdirSync(output, { recursive: true });
fs.writeFileSync(path.join(output, "config.json"), JSON.stringify({ contract: "event-fitted-value-screen-v1", source, selectedSource, resumeSource, phases, depths, settings,
  sourceHash: createHash("sha256").update(fs.readFileSync(path.join(source, "config.json"))).update(external.fingerprint).update(seconds?.fingerprint ?? "").digest("hex"),
  target: settings.some(s => s.continuationFolds)
    ? "Observed two-event continuation returns under block-held-out H1 actions; the declared fixed blocks purge overlapping one-day feature/label support. Final H1 uses all matched rows. Common account axes and liquidation guards remain fixed."
    : settings.some(s => s.sampledPath)
    ? "Compare bootstrapped continuation with observed-path returns under earlier fitted remaining-horizon policies; both settle terminal inventory and use the declared matched path cohort."
    : "Holding log return plus the previous fitted value at observed next features and the marked account; terminal cash settlement at depth one.",
  selection: "Mean prior-origin depth-one squared holding-value error chooses feature/penalty settings for a later depth test. Policy results report every depth and mean log growth minus the existing drawdown penalty.",
  caveat: "Research origins reused. No final inspector outcome is read or used. Fitted function approximation and account-grid interpolation are not optimality guarantees; liquidation support remains a separate hard guard. Optional residual trees share partitions across holding-value outputs." }, null, 2));
const files = ["scripts/screen-event-fitted-value.ts", "scripts/research-event-policy.ts", "scripts/event-paths.ts", "scripts/event-fitted-crossfit.ts", "scripts/event-fitted-settings.ts", "scripts/event-futures-basis.ts", "scripts/event-second-dynamics.ts", "packages/bot-algo/src/event-fitted-value.ts", "packages/bot-algo/src/event-value-boost.ts", "packages/bot-algo/src/event-log-policy.ts"];
fs.writeFileSync(path.join(output, "sources.json"), JSON.stringify(Object.fromEntries(files.map(f => [f, fs.readFileSync(path.join(root, f), "utf8")]))));
if (seconds) fs.writeFileSync(path.join(output, "second-dynamics-sources.json"), JSON.stringify({ ...seconds, rows: undefined }, null, 2));
const results: any[] = [], started = performance.now();
for (const [index, phase] of phases.entries()) {
  const begin = performance.now(), base = restoreEventPolicy(read(source, `${phase.id}-policy.json`));
  const allTrain = makeSamples(c, sc.clock, fits[index].trainStart, fits[index].trainEnd, [jc.window], sc.stride, "chain", sc.featureNames);
  const allTest = makeSamples(c, sc.clock, phase.startTime, phase.endTime, [], sc.stride, "chain", sc.featureNames);
  const train = allTrain.filter(r => matched(r.start) && matched(r.end)), test = allTest.filter(r => matched(r.start) && matched(r.end));
  if (train.length / allTrain.length < 0.98 || test.length !== allTest.length || train.some(r => c[r.end].openTime + 60000 >= phase.startTime)) throw new Error("Invalid matched chronology/coverage");
  const control = replayEventPolicy(c, base, phase.startTime, phase.endTime, 1, { trace: true });
  const old = read(source, `${phase.id}-scores.json`).rows.find((r: any) => r.choice === "joint-volatility" && r.depth === 1);
  if (control.returnPct !== old.returnPct || control.trades !== old.trades || control.maxDrawdownPct !== old.maxDrawdownPct) throw new Error("Original policy changed");
  const state = (i: number) => eventLeaf(base.model, eventFeatures(c, i, base.model.featureNames, sc.clock));
  const baselineValues = base.model.kernels.map(kernel => [-base.costs.maxLeverage, base.costs.maxLeverage].map(exposure => kernel.reduce((s, a) => {
    const held = eventHolding(exposure, a, base.costs), f = (base.costs.feeBps + base.costs.slippageBps) / 1e4;
    return s + a.probability * (Math.log(held.factor) + Math.log(1 - Math.abs(held.exposure) * f));
  }, 0)));
  const transitions = (samples: MoveSample[], setting: EventFittedSetting): EventValueTransition[] => samples.map(r => ({
    features: inputs(r.start, setting), nextFeatures: inputs(r.end, setting), leaf: state(r.start), nextLeaf: state(r.end),
    move: { return: r.return, low: r.low, high: r.high, duration: r.duration } }));
  const phaseRows = [];
  for (const setting of settings) {
    const fitStart = performance.now();
    const trainPaths = setting.pathHorizon ? eventSignHorizonPaths(train, setting.pathHorizon) : undefined;
    const testPaths = setting.pathHorizon ? eventSignHorizonPaths(test, setting.pathHorizon) : undefined;
    const selectedTrain = trainPaths ? trainPaths.map(p => p.steps[0]) : train, selectedTest = testPaths ? testPaths.map(p => p.steps[0]) : test;
    const rows = transitions(selectedTrain, setting), evaluation = transitions(selectedTest, setting);
    const crossfit = setting.continuationFolds ? fitEventCrossfitContinuations(base, rows,
      trainPaths!.map(p => ({ start: c[p.start].openTime + 60000, end: c[p.steps[0].end].openTime + 60000 })),
      trainPaths!.map(p => ({ start: c[p.steps[1].start].openTime + 60000, end: c[p.steps[1].end].openTime + 60000 })),
      fits[index].trainStart, fits[index].trainEnd, setting.penalty, setting.continuationFolds) : undefined;
    const rollout = setting.sampledPath ? { following: trainPaths!.map(p => transitions(p.steps.slice(1), setting)), ...(crossfit ? { policies: crossfit.policies } : {}) } : undefined;
    const name = key(setting), checkpoint = resumeSource ? read(resumeSource, `${phase.id}-${name}-policy.json`) : undefined;
    const policy = trainEventFittedValue(base, rows, setting.penalty, depths, setting.boost, checkpoint, rollout);
    if (crossfit) fs.writeFileSync(path.join(output, `${phase.id}-${name}-crossfit.json`), JSON.stringify({ folds: crossfit.folds, models: crossfit.models }));
    fs.writeFileSync(path.join(output, `${phase.id}-${name}-policy.json`), JSON.stringify(policy));
    const observations = new Map(control.trace.map((r: any) => [r.time, { availableAt: r.time, values: inputs(byTime.get(r.time)!, setting) }]));
    let mse = 0, zeroReturnMse = 0, baselineMse = 0, count = 0;
    const predictions = evaluation.map((r, i) => {
      const grid = predictEventHoldingGrid(policy, r.features, 1), probes = [-base.costs.maxLeverage, base.costs.maxLeverage].map(exposure => {
        const account = { equity: base.equities[2], price: c[selectedTest[i].start].close, exposure }, h = eventHolding(exposure, r.move, base.costs);
        if (h.liquidated) throw new Error("One-event validation probe liquidates");
        const f = (base.costs.feeBps + base.costs.slippageBps) / 1e4;
        const actual = Math.log(h.factor) + Math.log(1 - Math.abs(h.exposure) * f);
        const predicted = fittedEventHolding(policy, grid, account, r.leaf), zero = Math.log(1 - Math.abs(exposure) * f);
        const baseline = baselineValues[r.leaf][Number(exposure > 0)];
        if (![actual, predicted, zero, baseline].every(Number.isFinite)) throw new Error("Non-finite value probe");
        mse += (actual - predicted) ** 2; zeroReturnMse += (actual - zero) ** 2; baselineMse += (actual - baseline) ** 2; count++;
        return { exposure, actual, predicted, zero, baseline };
      });
      return { time: c[selectedTest[i].start].openTime + 60000, availableAt: c[selectedTest[i].end].openTime + 60000, probes };
    });
    fs.writeFileSync(path.join(output, `${phase.id}-${name}-predictions.json`), JSON.stringify(predictions));
    const policyRows = Array.from({ length: depths }, (_, d) => {
      const { trace, ...metrics } = replayEventPolicy(c, base, phase.startTime, phase.endTime, d + 1, { fitted: { policy, observations }, trace: true });
      fs.writeFileSync(path.join(output, `${phase.id}-${name}-d${d + 1}-trades.json`), JSON.stringify(trace));
      return { depth: d + 1, ...metrics };
    });
    const result = { ...setting, samples: rows.length, validation: evaluation.length, mse: mse / count, zeroReturnMse: zeroReturnMse / count,
      baselineMse: baselineMse / count, meanSkill: 1 - mse / zeroReturnMse, trainingMse: policy.tables.map(t => t.trainingMse), policyRows, elapsedSec: (performance.now() - fitStart) / 1000 };
    phaseRows.push(result);
  }
  results.push({ phase, rows: phaseRows });
  fs.writeFileSync(path.join(output, `${phase.id}-scores.json`), JSON.stringify({ phase, rows: phaseRows }));
  console.log(JSON.stringify({ phase: phase.id, rows: phaseRows.map(r => ({ ...r, policyRows: r.policyRows.map(p => ({ depth: p.depth, returnPct: p.returnPct, drawdown: p.maxDrawdownPct, trades: p.trades })) })), elapsedSec: (performance.now() - begin) / 1000 }));
}
const forecastRanking = settings.map(s => { const rows = results.map(r => r.rows.find((v: any) => key(v) === key(s)));
  return { ...s, meanMse: rows.reduce((v, r) => v + r.mse, 0) / rows.length, meanSkill: rows.reduce((v, r) => v + r.meanSkill, 0) / rows.length }; }).sort((a, b) => a.meanMse - b.meanMse);
const policyRanking = settings.flatMap(s => Array.from({ length: depths }, (_, d) => {
  const rows = results.map(r => r.rows.find((v: any) => key(v) === key(s)).policyRows[d]);
  return { ...s, depth: d + 1, ...eventOriginScore(rows, sc.riskPenalty), trades: rows.reduce((v, r) => v + r.trades, 0) };
})).sort((a, b) => b.score - a.score);
fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify({ forecastRanking, policyRanking, results, elapsedSec: (performance.now() - started) / 1000 }, null, 2));
console.log(JSON.stringify({ forecastBest: forecastRanking[0], policyBest: policyRanking[0], elapsedSec: (performance.now() - started) / 1000 }));
