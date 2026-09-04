/** Select from every retained prior-origin candidate before final fitted-value replay. */
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { eventFeatures, eventLeaf, type MoveSample } from "../packages/bot-algo/src/event-distribution.js";
import { restoreEventPolicy } from "../packages/bot-algo/src/event-log-policy.js";
import { trainEventFittedValue } from "../packages/bot-algo/src/event-fitted-value.js";
import { eventFittedSettingName, type EventFittedSetting } from "./event-fitted-settings.js";
import { eventSecondDynamicsAt, loadEventSecondDynamics } from "./event-second-dynamics.js";
import { eventFastVolatilityFeatures } from "../packages/bot-algo/src/event-size-sign.js";
import { eventCandleShapes, eventFuturesFeatures, eventFuturesBasisDeviations, eventFittedFuturesInputs, loadEventFuturesRows } from "./event-futures-basis.js";
import { eventOriginScore, eventRefitOrigins } from "./research-event-refits.js";
import { loadEventCandles, makeSamples, replayEventPolicy } from "./research-event-policy.js";
import { eventSignHorizonPaths } from "./event-paths.js";
import { fitEventCrossfitContinuations } from "./event-fitted-crossfit.js";

const root = path.resolve(__dirname, ".."), DAY = 86_400_000;
const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
if (!["source", "screens", "incumbent", "output"].every(key => arg(key))) throw new Error("Specify joint source, retained screens, incumbent and new output");
const directory = (id: string) => path.resolve(root, "data/benchmarks", id);
const source = directory(arg("source")), screens = arg("screens").split(",").map(directory), incumbentSource = directory(arg("incumbent")), output = directory(arg("output"));
const read = (dir: string, file: string) => JSON.parse(fs.readFileSync(path.join(dir, file), "utf8"));
if (fs.existsSync(output)) throw new Error("Choose a new output directory");
const jc = read(source, "config.json"), oc = read(jc.source, "config.json"), sc = read(oc.source, "config.json"), window = jc.window;
if (jc.contract !== "event-volatility-law-policy-v1"
  || (incumbentSource !== source && read(incumbentSource, "config.json").source !== source) || window.id.startsWith("fit-"))
  throw new Error("Incompatible joint source/incumbent/window");
const phases = eventRefitOrigins(window.startTime, oc.foldCount, oc.foldDays);
const hash = createHash("sha256").update(fs.readFileSync(path.join(source, "config.json")));
const folds = phases.map(phase => ({ phase, rows: read(incumbentSource, `${phase.id}-scores.json`).rows }));
const incumbentKeys = new Set(folds[0].rows.map((r: any) => `${r.choice}:${r.depth}`));
const retainedIncumbent = read(incumbentSource, "selection.json");
hash.update(fs.readFileSync(path.join(incumbentSource, "selection.json")));
for (const phase of phases) hash.update(fs.readFileSync(path.join(incumbentSource, `${phase.id}-scores.json`)));
const settings = new Map<string, EventFittedSetting>();
const duplicateChecks: Array<{ source: string; phase: string; choice: string; depth: number }> = [];
for (const dir of screens) {
  const config = read(dir, "config.json");
  if (config.contract !== "event-fitted-value-screen-v1" || config.source !== source || JSON.stringify(config.phases) !== JSON.stringify(phases))
    throw new Error("Fitted screen source/origin mismatch");
  if (config.settings.some((s: EventFittedSetting) => ["deviation", "centered"].includes(s.basis)) && config.settings.some((s: EventFittedSetting) => s.historyMinutes !== 240))
    throw new Error("Deviation cohort must be explicitly identified; rerun the screen");
  hash.update(fs.readFileSync(path.join(dir, "config.json")));
  for (const fold of folds) {
    const file = `${fold.phase.id}-scores.json`, saved = read(dir, file); hash.update(fs.readFileSync(path.join(dir, file)));
    for (const row of saved.rows) {
      const choice = `fitted-value-${eventFittedSettingName(row)}`;
      settings.set(choice, { basis: row.basis, penalty: row.penalty, ...(row.boost ? { boost: row.boost } : {}), ...(row.historyMinutes ? { historyMinutes: row.historyMinutes } : {}),
        ...(row.secondDynamics ? { secondDynamics: true } : {}), ...(row.candleShape ? { candleShape: true } : {}),
        ...(row.pathHorizon ? { pathHorizon: row.pathHorizon } : {}),
        ...(row.sampledPath ? { sampledPath: true } : {}), ...(row.continuationFolds ? { continuationFolds: row.continuationFolds } : {}) });
      for (const metrics of row.policyRows) {
        const previous = fold.rows.find((r: any) => r.choice === choice && r.depth === metrics.depth);
        if (previous) {
          // A structural cash bound can change reported values without changing
          // forecasts, decisions or execution. All economic fields must match.
          if (Object.keys(metrics).some(k => k !== "predictedGain" && JSON.stringify(previous[k]) !== JSON.stringify(metrics[k])))
            throw new Error("Repeated fitted candidate changed economic results");
          duplicateChecks.push({ source: dir, phase: fold.phase.id, choice, depth: metrics.depth });
        } else fold.rows.push({ choice, ...metrics });
      }
    }
  }
}
const ranking = folds[0].rows.map((r: any) => {
  const rows = folds.map(f => f.rows.find((v: any) => v.choice === r.choice && v.depth === r.depth));
  if (rows.some(v => !v)) throw new Error("Incomplete retained policy grid");
  return { choice: r.choice, depth: r.depth, ...eventOriginScore(rows, sc.riskPenalty), trades: rows.reduce((n, v) => n + v.trades, 0) };
}).sort((a: any, b: any) => b.score - a.score);
const incumbentRanking = ranking.filter((r: any) => incumbentKeys.has(`${r.choice}:${r.depth}`));
if (JSON.stringify(incumbentRanking) !== JSON.stringify(retainedIncumbent.ranking)) throw new Error("Retained incumbent ranking changed");
const chosen = ranking[0], cash = chosen.score <= 0, diagnosticChoice = ranking.find((r: any) => settings.has(r.choice));
if (!diagnosticChoice) throw new Error("No fitted candidate");
fs.mkdirSync(output, { recursive: true });
fs.writeFileSync(path.join(output, "selection.json"), JSON.stringify({ chosen, cash, diagnosticChoice, ranking }, null, 2));
for (const fold of folds) fs.writeFileSync(path.join(output, `${fold.phase.id}-scores.json`), JSON.stringify(fold));
fs.writeFileSync(path.join(output, "reproduction-check.json"), JSON.stringify({ incumbentRankingExact: true, incumbentCandidates: incumbentRanking.length, totalCandidates: ranking.length, duplicateChecks }, null, 2));
fs.writeFileSync(path.join(output, "config.json"), JSON.stringify({ contract: "event-fitted-value-policy-v1", source, screens, incumbentSource, window, settings: Object.fromEntries(settings), sourceHash: hash.digest("hex"),
  selection: "All retained prior-origin candidates compete on mean log growth minus drawdown penalty; incumbents win exact ties. Selection is written before loading final outcomes. Only the chosen fitted candidate is refitted for final replay.",
  caveat: "Repeatedly inspected research windows, not a sealed test or an optimality guarantee. Finite-horizon fitted holding values with hard liquidation support and a cash lower bound." }, null, 2));
const files = ["scripts/replay-event-fitted-value.ts", "scripts/research-event-policy.ts", "scripts/event-paths.ts", "scripts/event-fitted-crossfit.ts", "scripts/event-fitted-settings.ts", "scripts/event-futures-basis.ts", "scripts/event-second-dynamics.ts", "packages/bot-algo/src/event-fitted-value.ts", "packages/bot-algo/src/event-value-boost.ts", "packages/bot-algo/src/event-log-policy.ts"];
fs.writeFileSync(path.join(output, "sources.json"), JSON.stringify(Object.fromEntries(files.map(f => [f, fs.readFileSync(path.join(root, f), "utf8")]))));
console.log(JSON.stringify({ chosen, diagnosticChoice, retainedCandidates: ranking.length, selectionSaved: true }));

const started = performance.now(), fit = read(jc.source, `${window.id}-final-model.json`), base = restoreEventPolicy(read(source, "final-policy.json"));
const c = loadEventCandles(fit.trainStart - DAY, window.endTime), external = loadEventFuturesRows(fit.trainStart - DAY, window.endTime);
const byTime = new Map(c.map((r, i) => [r.openTime + 60000, i])), cache = new Map<number, ReturnType<typeof eventFuturesFeatures>>();
const extras = (i: number) => { if (!cache.has(i)) cache.set(i, eventFuturesFeatures(c, i, t => external.rows.get(t))); return cache.get(i)!; };
const setting = settings.get(diagnosticChoice.choice)!;
const seconds = setting.secondDynamics ? loadEventSecondDynamics(fit.trainStart - DAY, window.endTime) : undefined;
const deviations = new Map<number, number[] | null>();
const deviation = (i: number) => { if (!deviations.has(i)) deviations.set(i, eventFuturesBasisDeviations(c, i, t => external.rows.get(t))); return deviations.get(i)!; };
const shapes = (i: number) => eventCandleShapes(c, i, t => external.rows.get(t));
const inputs = (i: number) => {
  const e = extras(i); if (!e) throw new Error("Missing completed fitted-value observation");
  const d = ["deviation", "centered"].includes(setting.basis) ? deviation(i) : []; if (!d) throw new Error("Missing completed basis deviation");
  const shape = setting.candleShape ? shapes(i) : []; if (!shape) throw new Error("Missing completed candle shape");
  return [...eventFeatures(c, i, sc.featureNames, sc.clock), ...eventFastVolatilityFeatures(c, i),
    ...eventFittedFuturesInputs(setting.basis, e, d), ...(setting.secondDynamics ? eventSecondDynamicsAt(seconds!.rows, c[i]) : []), ...shape];
};
const all = makeSamples(c, sc.clock, fit.trainStart, fit.trainEnd, [window], sc.stride, "chain", sc.featureNames), train = all.filter(r => extras(r.start) && extras(r.end)
  && (setting.historyMinutes !== 240 || deviation(r.start) && deviation(r.end)) && (!setting.candleShape || shapes(r.start) && shapes(r.end)));
if (train.length / all.length < 0.98 || train.some(r => c[r.end].openTime + 60000 >= window.startTime)) throw new Error("Final training coverage/chronology mismatch");
const state = (i: number) => eventLeaf(base.model, eventFeatures(c, i, base.model.featureNames, sc.clock));
if (setting.sampledPath && (!setting.pathHorizon || setting.pathHorizon < diagnosticChoice.depth)) throw new Error("Invalid final sampled path horizon");
const trainPaths = setting.pathHorizon ? eventSignHorizonPaths(train, setting.pathHorizon) : undefined;
const selectedTrain = trainPaths ? trainPaths.map(p => p.steps[0]) : train;
const transition = (r: MoveSample) => ({ features: inputs(r.start), nextFeatures: inputs(r.end), leaf: state(r.start), nextLeaf: state(r.end),
  move: { return: r.return, low: r.low, high: r.high, duration: r.duration } });
if (setting.continuationFolds && (![6, 20].includes(setting.continuationFolds) || !setting.sampledPath || diagnosticChoice.depth > 2)) throw new Error("Invalid final crossfit setting");
const rows = selectedTrain.map(transition);
const crossfit = setting.continuationFolds ? fitEventCrossfitContinuations(base, rows,
  trainPaths!.map(p => ({ start: c[p.start].openTime + 60000, end: c[p.steps[0].end].openTime + 60000 })),
  trainPaths!.map(p => ({ start: c[p.steps[1].start].openTime + 60000, end: c[p.steps[1].end].openTime + 60000 })),
  fit.trainStart, fit.trainEnd, setting.penalty, setting.continuationFolds) : undefined;
const rollout = setting.sampledPath ? { following: trainPaths!.map(p => p.steps.slice(1).map(transition)), ...(crossfit ? { policies: crossfit.policies } : {}) } : undefined;
const policy = trainEventFittedValue(base, rows, setting.penalty, diagnosticChoice.depth, setting.boost, undefined, rollout);
if (crossfit) fs.writeFileSync(path.join(output, "final-crossfit.json"), JSON.stringify({ folds: crossfit.folds, models: crossfit.models }));
fs.writeFileSync(path.join(output, "final-fitted-policy.json"), JSON.stringify(policy));
fs.writeFileSync(path.join(output, "external-sources.json"), JSON.stringify({ ...external, rows: undefined }, null, 2));
if (seconds) fs.writeFileSync(path.join(output, "second-dynamics-sources.json"), JSON.stringify({ ...seconds, rows: undefined }, null, 2));
const control = replayEventPolicy(c, base, window.startTime, window.endTime, 1, { trace: true }), originalControl = read(source, "joint-law-diagnostic-trades.json");
if (JSON.stringify(control.trace) !== JSON.stringify(originalControl)) throw new Error("Final original-policy trace changed");
const observations = new Map(control.trace.map((r: any) => [r.time, { availableAt: r.time, values: inputs(byTime.get(r.time)!) }]));
const { trace, ...diagnostic } = replayEventPolicy(c, base, window.startTime, window.endTime, diagnosticChoice.depth,
  { fitted: { policy, observations }, cash: diagnosticChoice.score <= 0, trace: true });
fs.writeFileSync(path.join(output, "fitted-diagnostic-trades.json"), JSON.stringify(trace));
let test: typeof diagnostic;
if (settings.has(chosen.choice)) {
  test = diagnostic; fs.writeFileSync(path.join(output, "trades.json"), JSON.stringify(trace));
} else {
  const incumbent = read(incumbentSource, "summary.json");
  if (chosen.choice !== incumbent.chosen.choice || chosen.depth !== incumbent.chosen.depth || cash !== incumbent.cash) throw new Error("Incumbent final selection changed");
  test = incumbent.test; fs.copyFileSync(path.join(incumbentSource, "trades.json"), path.join(output, "trades.json"));
}
const result = { window, chosen, cash, test, diagnosticChoice, diagnostic, training: { all: all.length, matched: train.length,
  ...(trainPaths ? { completePaths: trainPaths.length } : {}) }, elapsedSec: (performance.now() - started) / 1000 };
fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify(result, null, 2));
console.log(JSON.stringify({ ...result, test: { ...test, daily: undefined }, diagnostic: { ...diagnostic, daily: undefined } }));
