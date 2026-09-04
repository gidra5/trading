/** Retain every incumbent, freeze the new option comparison, then replay its final window. */
import fs from "node:fs";
import path from "node:path";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { eventFeatures, eventLeaf, type MoveSample } from "../packages/bot-algo/src/event-distribution.js";
import { restoreEventPolicy } from "../packages/bot-algo/src/event-log-policy.js";
import { trainEventFittedValue } from "../packages/bot-algo/src/event-fitted-value.js";
import { eventFastVolatilityFeatures } from "../packages/bot-algo/src/event-size-sign.js";
import { eventFuturesFeatures, eventFuturesBasisDeviations, eventFittedFuturesInputs, loadEventFuturesRows } from "./event-futures-basis.js";
import { eventFittedSettingName } from "./event-fitted-settings.js";
import { eventSignHorizonPaths } from "./event-paths.js";
import { eventOriginScore } from "./research-event-refits.js";
import { loadEventCandles, makeSamples, replayEventPolicy } from "./research-event-policy.js";

const root = path.resolve(__dirname, ".."), DAY = 86400000;
const arg = (k: string) => { const i = process.argv.indexOf(`--${k}`); return i < 0 ? "" : process.argv[i + 1]; };
if (!["source", "incumbent", "output"].every(k => arg(k))) throw new Error("Specify option screen, incumbent and new output");
const source = path.resolve(root, "data/benchmarks", arg("source")), incumbent = path.resolve(root, "data/benchmarks", arg("incumbent")), output = path.resolve(root, "data/benchmarks", arg("output"));
const read = (dir: string, file: string) => JSON.parse(fs.readFileSync(path.join(dir, file), "utf8"));
if (fs.existsSync(output)) throw new Error("Choose a new output directory");
const config = read(source, "config.json"), parent = read(config.source, "config.json"), joint = parent.source;
const jc = read(joint, "config.json"), oc = read(jc.source, "config.json"), sc = read(oc.source, "config.json"), window = jc.window;
if (config.contract !== "event-hold-option-screen-v1" || read(incumbent, "config.json").source !== joint || window.id.startsWith("fit-")) throw new Error("Incompatible option comparison");
const setting = config.setting, summary = read(source, "summary.json"), previous = read(incumbent, "selection.json");
assert.deepEqual(summary.results.map((r: any) => r.phase), config.phases);
const choice = `hold-option-${eventFittedSettingName(setting)}`, candidate = { choice, depth: 2,
  ...eventOriginScore(summary.results.map((r: any) => r.candidate), sc.riskPenalty), trades: summary.results.reduce((s: number, r: any) => s + r.candidate.trades, 0) };
assert.ok(!previous.ranking.some((r: any) => r.choice === choice));
const ranking = [...previous.ranking, candidate].sort((a, b) => b.score - a.score), chosen = ranking[0], cash = chosen.score <= 0;
assert.deepEqual(ranking.filter(r => r.choice !== choice), previous.ranking);
fs.mkdirSync(output, { recursive: true });
fs.writeFileSync(path.join(output, "selection.json"), JSON.stringify({ chosen, cash, diagnosticChoice: candidate, ranking }, null, 2));
for (const [i, phase] of config.phases.entries()) {
  const saved = read(incumbent, `${phase.id}-scores.json`);
  fs.writeFileSync(path.join(output, `${phase.id}-scores.json`), JSON.stringify({ ...saved, rows: [...saved.rows, { choice, depth: 2, ...summary.results[i].candidate }] }));
}
fs.writeFileSync(path.join(output, "config.json"), JSON.stringify({ contract: "event-hold-option-policy-v1", source: joint, screen: source, incumbent, window, setting,
  sourceHash: createHash("sha256").update(fs.readFileSync(path.join(source, "config.json"))).update(fs.readFileSync(path.join(source, "summary.json")))
    .update(fs.readFileSync(path.join(incumbent, "selection.json"))).digest("hex"),
  selection: "Retain the exact incumbent ranking and append one declared H2 option candidate from prior-origin metrics. Save selection before loading final outcomes. Refit and replay the new candidate as a diagnostic, cash-gated only by its prior score. Keep the incumbent's final result if it remains selected.",
  caveat: "Repeated research windows, not a sealed test. Restricted option value regression is not Bellman optimality or a profitability guarantee." }, null, 2));
const files = ["scripts/replay-event-hold-option.ts", "scripts/screen-event-hold-options.ts", "scripts/event-paths.ts", "scripts/event-fitted-settings.ts", "scripts/event-futures-basis.ts",
  "scripts/research-event-policy.ts", "scripts/research-event-refits.ts", "packages/bot-algo/src/event-fitted-value.ts", "packages/bot-algo/src/event-log-policy.ts"];
fs.writeFileSync(path.join(output, "sources.json"), JSON.stringify(Object.fromEntries(files.map(f => [f, fs.readFileSync(path.join(root, f), "utf8")]))));
console.log(JSON.stringify({ chosen, diagnosticChoice: candidate, selectionSaved: true, retainedCandidates: ranking.length }));
const started = performance.now(), fit = read(jc.source, `${window.id}-final-model.json`), base = restoreEventPolicy(read(joint, "final-policy.json"));
const c = loadEventCandles(fit.trainStart - DAY, window.endTime), external = loadEventFuturesRows(fit.trainStart - DAY, window.endTime), cache = new Map<number, number[] | null>();
const inputs = (i: number) => {
  if (cache.has(i)) return cache.get(i)!;
  const e = eventFuturesFeatures(c, i, t => external.rows.get(t)), d = setting.historyMinutes ? eventFuturesBasisDeviations(c, i, t => external.rows.get(t)) : [];
  const values = e && d ? [...eventFeatures(c, i, sc.featureNames, sc.clock), ...eventFastVolatilityFeatures(c, i), ...eventFittedFuturesInputs(setting.basis, e, d)] : null;
  cache.set(i, values); return values;
};
const all = makeSamples(c, sc.clock, fit.trainStart, fit.trainEnd, [window], sc.stride, "chain", sc.featureNames), matched = all.filter(r => inputs(r.start) && inputs(r.end));
if (matched.length / all.length < .98) throw new Error("Incomplete final option training coverage");
const paths = eventSignHorizonPaths(matched, 3);
assert.ok(paths.every(p => c[p.end].openTime + 60000 < window.startTime));
const state = (i: number) => eventLeaf(base.model, eventFeatures(c, i, base.model.featureNames, sc.clock));
const transition = (r: MoveSample) => ({ features: inputs(r.start)!, nextFeatures: inputs(r.end)!, leaf: state(r.start), nextLeaf: state(r.end),
  move: { return: r.return, low: r.low, high: r.high, duration: r.duration } });
const policy = trainEventFittedValue(base, paths.map(p => transition(p.steps[0])), setting.penalty, 2, undefined, undefined,
  { following: paths.map(p => p.steps.slice(1).map(transition)), minimumTurnover: true });
fs.writeFileSync(path.join(output, "final-fitted-policy.json"), JSON.stringify(policy));
fs.writeFileSync(path.join(output, "external-sources.json"), JSON.stringify({ ...external, rows: undefined }));
const control = replayEventPolicy(c, base, window.startTime, window.endTime, 1, { trace: true });
assert.equal(JSON.stringify(control.trace), JSON.stringify(read(joint, "joint-law-diagnostic-trades.json")), "Original final replay changed");
const byTime = new Map(c.map((r, i) => [r.openTime + 60000, i]));
const observations = new Map<number, { availableAt: number; values: number[] }>(control.trace.map((r: any) => {
  const values = inputs(byTime.get(r.time)!); if (!values) throw new Error("Missing final option observation");
  return [r.time, { availableAt: r.time, values }];
}));
const { trace, ...diagnostic } = replayEventPolicy(c, base, window.startTime, window.endTime, 2,
  { fitted: { policy, observations, replanEvery: 2 }, cash: candidate.score <= 0, trace: true });
fs.writeFileSync(path.join(output, "fitted-diagnostic-trades.json"), JSON.stringify(trace));
let test = diagnostic;
if (chosen.choice === choice) fs.writeFileSync(path.join(output, "trades.json"), JSON.stringify(trace));
else {
  const old = read(incumbent, "summary.json"); assert.deepEqual(chosen, old.chosen); assert.equal(cash, old.cash);
  test = old.test; fs.copyFileSync(path.join(incumbent, "trades.json"), path.join(output, "trades.json"));
}
fs.writeFileSync(path.join(output, "reproduction-check.json"), JSON.stringify({ incumbentRankingExact: true, originalFinalReplayExact: true,
  incumbentCandidates: previous.ranking.length, totalCandidates: ranking.length }, null, 2));
const result = { window, chosen, cash, diagnosticChoice: candidate, test, diagnostic,
  training: { all: all.length, matched: matched.length, completePaths: paths.length }, elapsedSec: (performance.now() - started) / 1000 };
fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify(result, null, 2));
console.log(JSON.stringify({ ...result, test: { ...test, daily: undefined }, diagnostic: { ...diagnostic, daily: undefined } }));
