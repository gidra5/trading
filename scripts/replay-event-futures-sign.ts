/** Fee-aware sign-head comparison after three completed forecast origins. */
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { eventFeatures, type MoveSample } from "../packages/bot-algo/src/event-distribution.js";
import { buildEventOutcomeLookahead, buildEventPolicy, buildEventSignLookahead, restoreEventPolicy, serializeEventPolicy, type EventPolicy } from "../packages/bot-algo/src/event-log-policy.js";
import { trainEventSign, type EventSignHead } from "../packages/bot-algo/src/event-sign.js";
import { eventFastVolatilityFeatures, eventSizeSignGroup, predictEventSizeSigns, projectEventSizeSigns, trainEventSizeSign, type EventSizeSignHead } from "../packages/bot-algo/src/event-size-sign.js";
import { eventOriginScore, eventRefitOrigins } from "./research-event-refits.js";
import { eventFuturesFeatures, loadEventFuturesRows } from "./event-futures-basis.js";
import { loadEventCandles, makeSamples, replayEventPolicy } from "./research-event-policy.js";
const root = path.resolve(__dirname, ".."), DAY = 86_400_000;
const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
const projectContinuation = process.argv.includes("--project-continuation");
if (!arg("source") || !arg("forecasts") || !arg("output")) throw new Error("Specify joint policy, three forecast directories and new output");
const source = path.resolve(root, "data/benchmarks", arg("source")), output = path.resolve(root, "data/benchmarks", arg("output"));
const incumbentSource = arg("incumbent") ? path.resolve(root, "data/benchmarks", arg("incumbent")) : source;
const forecasts = arg("forecasts").split(",").map(id => path.resolve(root, "data/benchmarks", id));
const read = (dir: string, file: string) => JSON.parse(fs.readFileSync(path.join(dir, file), "utf8"));
if (fs.existsSync(output)) throw new Error("Choose a new output directory");
const config = read(source, "config.json"), original = read(config.source, "config.json"), sc = read(original.source, "config.json"), window = config.window;
if (config.contract !== "event-volatility-law-policy-v1") throw new Error("Requires joint-policy source");
if (incumbentSource !== source && read(incumbentSource, "config.json").source !== source) throw new Error("Incumbent must use the same joint source");
const origins = eventRefitOrigins(window.startTime, original.foldCount, original.foldDays), phases = [...origins, { ...window, id: "final" }];
if (forecasts.length !== origins.length) throw new Error("Need every prior forecast origin");
const forecastContract = read(forecasts[0], "config.json").contract, sizeRegimes = forecastContract === "event-futures-size-sign-screen-v1";
if (!sizeRegimes && forecastContract !== "event-futures-sign-screen-v1") throw new Error("Unknown forecast head contract");
if (projectContinuation && (!sizeRegimes || incumbentSource === source)) throw new Error("Projection requires size heads and their existing policy comparison");
const suffix = projectContinuation ? "-projected" : "";
const spotName = (sizeRegimes ? "spot-size-sign" : "spot-sign") + suffix, futuresName = (sizeRegimes ? "futures-size-sign" : "futures-sign") + suffix;
const summaries = forecasts.map((dir, i) => {
  const f = read(dir, "config.json");
  if (f.contract !== forecastContract || f.source !== source || f.phase.id !== origins[i].id || (sizeRegimes && f.quantile !== 0.75)) throw new Error("Forecast origin mismatch");
  return read(dir, "summary.json");
});
const forecastRanking = summaries[0].results.filter((r: any) => r.basis !== "joint-law").map((setting: any) => {
  const rows = summaries.map(s => s.results.find((r: any) => r.basis === setting.basis && r.penalty === setting.penalty && r.blend === setting.blend));
  if (rows.some(r => !r)) throw new Error("Incomplete forecast grid");
  return { basis: setting.basis, penalty: setting.penalty, blend: setting.blend,
    meanSignLoss: rows.reduce((s, r) => s + r.signLoss, 0) / rows.length,
    ...(sizeRegimes ? { meanSizeSignLoss: rows.reduce((s, r) => s + r.sizeSignLoss, 0) / rows.length } : {}) };
}).sort((a: any, b: any) => sizeRegimes ? a.meanSizeSignLoss - b.meanSizeSignLoss : a.meanSignLoss - b.meanSignLoss);
const settings: Record<string, { basis: string; penalty: number; blend: number }> = {
  [spotName]: forecastRanking.find((r: any) => r.basis === "spot"),
  [futuresName]: forecastRanking.find((r: any) => r.basis !== "spot"),
};
if (projectContinuation && Object.values(settings).some(s => s.blend !== 1)) throw new Error("Continuation isolation requires full current-head replacement");
fs.mkdirSync(output, { recursive: true });
const hash = createHash("sha256").update(fs.readFileSync(path.join(source, "config.json")));
for (const dir of forecasts) for (const file of ["config.json", "summary.json", "heads.json"]) hash.update(fs.readFileSync(path.join(dir, file)));
for (const phase of phases) hash.update(fs.readFileSync(path.join(source, `${phase.id}-policy.json`)));
hash.update(fs.readFileSync(path.join(incumbentSource, "selection.json")));
fs.writeFileSync(path.join(output, "config.json"), JSON.stringify({ contract: "event-futures-sign-policy-v1", source, incumbentSource, forecasts, sourceHash: hash.digest("hex"), settings, window, sizeRegimes, projectContinuation,
  selection: `Choose spot and external feature/penalty/blend by mean prior-origin ${sizeRegimes ? "four-group size/sign" : "sign"} loss, then compare their depths 1-8 with every incumbent on mean log growth minus drawdown penalty. Cash eligible. Save selections before final evaluation.`,
  caveat: `Research windows reused. ${projectContinuation ? "Training-feature head probabilities are averaged in each existing market state and the Bellman tables rebuilt; current conditional path laws remain unchanged." : "Sign probabilities improve the first Bellman backup with the frozen joint-law continuation."} Future external features are not generated recursively. No claimed global policy optimum.` }, null, 2));
fs.writeFileSync(path.join(output, "forecast-selection.json"), JSON.stringify({ settings, ranking: forecastRanking }, null, 2));
const files = ["scripts/replay-event-futures-sign.ts", "scripts/event-futures-basis.ts", "scripts/research-event-policy.ts",
  "packages/bot-algo/src/event-sign.ts", "packages/bot-algo/src/event-size-sign.ts", "packages/bot-algo/src/event-log-policy.ts", "packages/bot-algo/src/event-distribution.ts"];
fs.writeFileSync(path.join(output, "sources.json"), JSON.stringify(Object.fromEntries(files.map(f => [f, fs.readFileSync(path.join(root, f), "utf8")]))));
const firstFit = read(config.source, `${window.id}-${origins[0].id}-model.json`);
const c = loadEventCandles(firstFit.trainStart - DAY, window.endTime + DAY);
const external = loadEventFuturesRows(firstFit.trainStart - DAY, window.endTime), byTime = new Map(c.map((r, i) => [r.openTime + 60_000, i]));
fs.writeFileSync(path.join(output, "external-sources.json"), JSON.stringify({ ...external, rows: undefined }, null, 2));
const extraCache = new Map<number, ReturnType<typeof eventFuturesFeatures>>();
const extras = (i: number) => { if (!extraCache.has(i)) extraCache.set(i, eventFuturesFeatures(c, i, t => external.rows.get(t))); return extraCache.get(i)!; };
const input = (features: readonly number[], i: number, basis: string) => {
  const e = extras(i); if (!e) throw new Error(`Missing completed external observation ${new Date(c[i].openTime).toISOString()}`);
  return [...features, ...eventFastVolatilityFeatures(c, i), ...(["price", "all"].includes(basis) ? e.price : []), ...(["flow", "all"].includes(basis) ? e.flow : [])];
};
const folds: any[] = [], started = performance.now();
let chosen: any, diagnosticChoice: any, cash = false;
for (const [index, phase] of phases.entries()) {
  const begin = performance.now(), p = restoreEventPolicy(read(source, `${phase.id}-policy.json`));
  if (phase.id === "final") {
    const names = [...new Set(folds[0].rows.map((r: any) => r.choice))] as string[];
    const ranking = names.flatMap(choice => Array.from({ length: sc.maxDepth }, (_, d) => {
      const rows = folds.map(f => f.rows.find((r: any) => r.choice === choice && r.depth === d + 1));
      if (rows.some(r => !r)) throw new Error("Incomplete policy grid");
      return { choice, depth: d + 1, ...eventOriginScore(rows, sc.riskPenalty), trades: rows.reduce((s, r) => s + r.trades, 0) };
    })).sort((a, b) => b.score - a.score);
    chosen = ranking[0]; cash = chosen.score <= 0; diagnosticChoice = ranking.find(r => r.choice === futuresName);
    fs.writeFileSync(path.join(output, "selection.json"), JSON.stringify({ chosen, cash, diagnosticChoice, ranking }, null, 2));
  }
  const fit = read(config.source, `${window.id}-${phase.id}-model.json`);
  const control = replayEventPolicy(c, p, phase.startTime, phase.endTime, sc.maxDepth, { trace: true });
  const heads: Record<string, EventSignHead | EventSizeSignHead> = {};
  const observations: Record<string, Map<number, { availableAt: number; values: number[] }>> = {};
  for (const [name, setting] of Object.entries(settings)) {
    if (phase.id === "final") {
      const allRows = makeSamples(c, sc.clock, fit.trainStart, fit.trainEnd, [window], sc.stride, "chain", sc.featureNames);
      const rows = allRows.filter((r: MoveSample) => extras(r.start));
      if (rows.length !== fit.support.samples || rows.some(r => c[r.end].openTime + 60_000 >= phase.startTime)) throw new Error("Final head training coverage/chronology mismatch");
      const inputs = rows.map(r => ({ features: input(r.features, r.start, setting.basis), return: r.return }));
      heads[name] = sizeRegimes ? trainEventSizeSign(inputs, setting.penalty, 0.75) : trainEventSign(inputs, setting.penalty);
    } else {
      const saved = read(forecasts[index], "heads.json").find((r: any) => r.basis === setting.basis && r.penalty === setting.penalty);
      if (!saved) throw new Error("Missing selected forecast head"); heads[name] = saved.head;
    }
    observations[name] = new Map(control.trace.map((r: any) => {
      const i = byTime.get(r.time)!;
      return [r.time, { availableAt: r.time, values: input(eventFeatures(c, i, sc.featureNames, sc.clock), i, setting.basis) }];
    }));
  }
  const threshold = sizeRegimes ? (heads[spotName] as EventSizeSignHead).thresholdLogBps : 0;
  if (sizeRegimes && Object.values(heads).some(h => (h as EventSizeSignHead).thresholdLogBps !== threshold)) throw new Error("Heads disagree on size groups");
  const policies: Record<string, EventPolicy> = Object.fromEntries(Object.keys(settings).map(name => [name, p]));
  if (projectContinuation) {
    const training = makeSamples(c, sc.clock, fit.trainStart, fit.trainEnd, [window], sc.stride, "chain", sc.featureNames).filter(r => extras(r.start));
    const expectedRows = phase.id === "final" ? fit.support.samples : read(forecasts[index], "config.json").coverage.train;
    if (training.length !== expectedRows || training.some(r => c[r.end].openTime + 60_000 >= phase.startTime)) throw new Error("Projection training chronology/population mismatch");
    for (const [name, setting] of Object.entries(settings)) {
      const probabilities = training.map(r => ({ features: eventFeatures(c, r.start, p.model.featureNames, sc.clock),
        probabilities: predictEventSizeSigns(heads[name] as EventSizeSignHead, input(r.features, r.start, setting.basis)) }));
      const model = projectEventSizeSigns(p.model, threshold, probabilities, 1);
      policies[name] = buildEventPolicy(model, p.costs, { depths: p.tables.length, referenceEquity: p.equities[2], referencePrice: p.prices[1], actionSteps: (p.targets.length - 1) / 2 });
      fs.writeFileSync(path.join(output, `${phase.id}-${name}-policy.json`), JSON.stringify(serializeEventPolicy(policies[name])));
    }
  }
  const sharedLookahead = projectContinuation ? undefined : sizeRegimes
    ? buildEventOutcomeLookahead(p, 5, a => eventSizeSignGroup(a.return, threshold)) : buildEventSignLookahead(p);
  const lookaheads = Object.fromEntries(Object.keys(settings).map(name => [name, sharedLookahead
    ?? buildEventOutcomeLookahead(policies[name], 5, a => eventSizeSignGroup(a.return, threshold))]));
  fs.writeFileSync(path.join(output, `${phase.id}-heads.json`), JSON.stringify({ phase, settings, heads }));
  const options = (name: string) => sizeRegimes
    ? { sizeSign: { head: heads[name] as EventSizeSignHead, blend: settings[name].blend, lookahead: lookaheads[name], observations: observations[name] } }
    : { sign: { head: heads[name] as EventSignHead, blend: settings[name].blend, lookahead: lookaheads[name], observations: observations[name] } };
  if (phase.id !== "final") {
    const old = read(incumbentSource, `${phase.id}-scores.json`), expected = old.rows.find((r: any) => r.choice === "joint-volatility" && r.depth === sc.maxDepth);
    if (old.rows.some((r: any) => settings[r.choice])) throw new Error("New candidate name collides with incumbent");
    if (expected.returnPct !== control.returnPct || expected.maxDrawdownPct !== control.maxDrawdownPct || expected.trades !== control.trades) throw new Error("Joint-law control changed");
    const forecastReproduction: Record<string, { events: number; maximumProbabilityError: number; maximumMeanErrorBps: number }> = {};
    const extra = Object.keys(settings).flatMap(name => Array.from({ length: sc.maxDepth }, (_, d) => {
      const { trace, ...metrics } = replayEventPolicy(c, policies[name], phase.startTime, phase.endTime, d + 1, { ...options(name), trace: d === 0 });
      if (d === 0) {
        if (projectContinuation) {
          const baseline = old.rows.find((r: any) => r.choice === name.slice(0, -suffix.length) && r.depth === 1);
          if (!baseline || ["returnPct", "maxDrawdownPct", "trades", "fees", "borrow", "exposedMinutes"].some(k => Math.abs((metrics as any)[k] - baseline[k]) > 1e-10))
            throw new Error("Projection changed the one-event policy");
        }
        const s = settings[name], expected = read(forecasts[index], `${s.basis}-${s.penalty}-${s.blend}-predictions.json`);
        const actual = new Map(trace.map((r: any) => [r.time, r]));
        let maximumProbabilityError = 0, maximumMeanErrorBps = 0;
        for (const r of expected) {
          const match = actual.get(r.time); if (!match) throw new Error("Forecast event is missing from replay");
          maximumProbabilityError = Math.max(maximumProbabilityError, sizeRegimes
            ? Math.max(...r.probabilities.map((v: number, i: number) => Math.abs(v - match.sizeSignProbabilities[i])))
            : Math.abs(r.probability - match.signProbability));
          maximumMeanErrorBps = Math.max(maximumMeanErrorBps, Math.abs(r.mean * 1e4 - match.expectedReturnBps));
        }
        if (maximumProbabilityError > 1e-12 || maximumMeanErrorBps > 1e-9) throw new Error("Forecast/replay inputs or weighting differ");
        forecastReproduction[name] = { events: expected.length, maximumProbabilityError, maximumMeanErrorBps };
      }
      return { choice: name, depth: d + 1, ...metrics };
    }));
    const fold = { phase, forecastReproduction, rows: [...old.rows, ...extra] }; folds.push(fold);
    fs.writeFileSync(path.join(output, `${phase.id}-scores.json`), JSON.stringify(fold));
    console.log(JSON.stringify({ phase: phase.id, controlExact: true, bestNew: extra.sort((a, b) => eventOriginScore([b], sc.riskPenalty).score - eventOriginScore([a], sc.riskPenalty).score).slice(0, 2).map(r => ({ choice: r.choice, depth: r.depth, returnPct: r.returnPct, drawdown: r.maxDrawdownPct, trades: r.trades })), elapsedSec: (performance.now() - begin) / 1000 }));
  } else {
    let test: any;
    if (settings[chosen.choice]) {
      const replay = replayEventPolicy(c, policies[chosen.choice], phase.startTime, phase.endTime, chosen.depth, { ...options(chosen.choice), cash, trace: true });
      const { trace, ...metrics } = replay; test = metrics; fs.writeFileSync(path.join(output, "trades.json"), JSON.stringify(trace));
    } else {
      const incumbent = read(incumbentSource, "summary.json");
      if (chosen.choice !== incumbent.chosen.choice || chosen.depth !== incumbent.chosen.depth || cash !== incumbent.cash) throw new Error("Incumbent selection changed unexpectedly");
      test = incumbent.test; fs.copyFileSync(path.join(incumbentSource, "trades.json"), path.join(output, "trades.json"));
    }
    const diagnostic = replayEventPolicy(c, policies[futuresName], phase.startTime, phase.endTime, diagnosticChoice.depth, { ...options(futuresName), cash: diagnosticChoice.score <= 0, trace: true });
    const { trace, ...metrics } = diagnostic;
    fs.writeFileSync(path.join(output, "futures-diagnostic-trades.json"), JSON.stringify(trace));
    const result = { window, settings, chosen, cash, test, diagnosticChoice, diagnostic: metrics, elapsedSec: (performance.now() - started) / 1000 };
    fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify(result, null, 2));
    console.log(JSON.stringify({ ...result, test: { ...test, daily: undefined, adaptationPairs: undefined }, diagnostic: { ...metrics, daily: undefined, adaptationPairs: undefined } }));
  }
}
