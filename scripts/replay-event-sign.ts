/** One Bellman policy-improvement backup with the forecast-selected sign head. */
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { buildEventPolicy, buildEventOutcomeLookahead, buildEventSignLookahead, restoreEventPolicy, serializeEventPolicy, type EventPolicy, type SerializedEventPolicy } from "../packages/bot-algo/src/event-log-policy.js";
import type { EventSignHead } from "../packages/bot-algo/src/event-sign.js";
import { eventFastVolatilityFeatures, eventSizeSignGroup, predictEventSizeSigns, projectEventSizeSigns, trainEventSizeSign, type EventSizeSignHead } from "../packages/bot-algo/src/event-size-sign.js";
import { EventCompletedHistory } from "../packages/bot-algo/src/event-completed-history.js";
import { eventCalibrationRanges, loadEventCandles, makeSamples, replayEventPolicy } from "./research-event-policy.js";

const root = path.resolve(__dirname, ".."), DAY = 86_400_000;
const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
const forecastId = arg("forecast"), outputId = arg("output");
const control = process.argv.includes("--control");
const projectContinuation = process.argv.includes("--project-continuation");
const compareBase = process.argv.includes("--compare-base");
const compareContinuations = process.argv.includes("--compare-continuations");
const compareHeadLag = process.argv.includes("--compare-head-lag");
if (compareContinuations && (!projectContinuation || !compareBase)) throw new Error("Continuation comparison requires projection and base comparison");
if (compareHeadLag && !compareContinuations) throw new Error("Head-age comparison requires all original continuation candidates");
if (!forecastId || !outputId) throw new Error("Specify forecast source and new output directory");
const forecast = path.resolve(root, "data/benchmarks", forecastId), output = path.resolve(root, "data/benchmarks", outputId);
if (fs.existsSync(output)) throw new Error("Choose a new output directory");
const forecastConfig = JSON.parse(fs.readFileSync(path.join(forecast, "config.json"), "utf8"));
if (!["event-sign-forecast-screen-v1", "event-size-sign-forecast-screen-v1"].includes(forecastConfig.contract)) throw new Error("Incompatible sign forecast artifact");
const source: string = forecastConfig.source;
const sourceBytes = fs.readFileSync(path.join(source, "config.json")), sourceConfig = JSON.parse(sourceBytes.toString());
const summaries = JSON.parse(fs.readFileSync(path.join(source, "summary.json"), "utf8")) as Array<{
  window: { id: string; startTime: number; endTime: number }; test: Record<string, unknown>;
  fitSamples: number; selectionFitSamples: number;
}>;
const hash = createHash("sha256").update(sourceBytes);
const saved = summaries.map(s => {
  const bytes = fs.readFileSync(path.join(source, `${s.window.id}-model.json`)); hash.update(bytes);
  return JSON.parse(bytes.toString()) as { policy: SerializedEventPolicy; selectionPolicy: SerializedEventPolicy;
    trainStart: number; trainEnd: number; selectionTrainingEnd: number; policyCalibrationStart: number; calibrationEnd: number;
    selectionExcludedWindows: Array<{ id: string; startTime: number; endTime: number }> };
});
if (hash.digest("hex") !== forecastConfig.sourceHash) throw new Error("Original source changed since sign selection");
const ids = arg("windows").split(",").filter(Boolean);
if (ids.some(id => !summaries.some(s => s.window.id === id))) throw new Error("Unknown requested window");
fs.mkdirSync(output, { recursive: true });
fs.writeFileSync(path.join(output, "config.json"), JSON.stringify({ contract: "event-sign-lookahead-replay-v1", forecast, source, control, projectContinuation, compareBase, compareContinuations, compareHeadLag,
  headLagDays: compareHeadLag ? sourceConfig.calibrationDays + 1 : undefined,
  headLagRule: compareHeadLag ? "Lag head training cutoff by calibrationDays + 1 relative to each evaluation start; use trainDays - 1 of history; retain current continuation models. Same rule in calibration and test; all incumbent candidates remain eligible." : undefined,
  windows: ids.length ? ids : summaries.map(s => s.window.id),
  selection: compareContinuations ? "Unchanged original, head with frozen continuation, and head with projected continuation compete at every available depth on preceding calibration log growth minus configured drawdown penalty; cash eligible"
    : compareBase ? "Forecast-selected head versus unchanged original policy; policy and depth maximize preceding calibration log growth minus configured drawdown penalty; cash eligible"
    : "Head and blend frozen by preceding forecast screen; depth and cash reselected on preceding policy calibration",
  approximation: projectContinuation ? "Average head probabilities over training features in each original finite state; rebuild Bellman values at every depth under that projected law; current backup uses observed fine features; hypothetical event memory remains projected"
    : "New sign law in current Bellman backup; frozen base value for all hypothetical later moves; recompute using observed features at each real event",
  caveat: "This is one policy-improvement experiment, not recursively fitted sign-state Bellman convergence or an untouched final holdout" }, null, 2));
const files = ["scripts/replay-event-sign.ts", "scripts/research-event-policy.ts", "packages/bot-algo/src/event-distribution.ts",
  "packages/bot-algo/src/event-sign.ts", "packages/bot-algo/src/event-size-sign.ts", "packages/bot-algo/src/event-completed-history.ts", "packages/bot-algo/src/event-log-policy.ts"];
fs.writeFileSync(path.join(output, "sources.json"), JSON.stringify(Object.fromEntries(files.map(f => [f, fs.readFileSync(path.join(root, f), "utf8")]))));
const results = [];
for (let i = 0; i < summaries.length; i++) {
  const { window } = summaries[i], s = saved[i];
  if (ids.length && !ids.includes(window.id)) continue;
  const started = performance.now();
  const headBytes = fs.readFileSync(path.join(forecast, `${window.id}-model.json`));
  const h = JSON.parse(headBytes.toString()) as { selected: { penalty: number; blend: number; quantile?: number; fastVolatility?: boolean; eventHistory?: boolean; historyGateOnly?: boolean } | null;
    selectionHead?: EventSignHead | EventSizeSignHead; head?: EventSignHead | EventSizeSignHead; trainEnd: number; selectionTrainingEnd: number };
  if (h.trainEnd !== s.trainEnd || h.selectionTrainingEnd !== s.selectionTrainingEnd || s.trainEnd > window.startTime
    || s.calibrationEnd > window.startTime || window.id.startsWith("fit-")) throw new Error("Invalid sign fit boundaries");
  if (h.selected && (!h.head || !h.selectionHead)) throw new Error("Missing selected sign model");
  if (h.selected && "gate" in h.selected && h.selected.gate) throw new Error("Rolling size-gate artifacts are forecast-only; replay updating has not been implemented");
  fs.writeFileSync(path.join(output, `${window.id}-sign-model.json`), headBytes);
  let selectionPolicy = restoreEventPolicy(s.selectionPolicy), policy = restoreEventPolicy(s.policy);
  const originalSelectionPolicy = selectionPolicy, originalPolicy = policy;
  const selectionTrainStart = window.startTime - (sourceConfig.calibrationDays + sourceConfig.trainDays) * DAY;
  const lagDays = sourceConfig.calibrationDays + 1;
  const lagSelectionEnd = s.policyCalibrationStart - lagDays * DAY, lagSelectionStart = lagSelectionEnd - (sourceConfig.trainDays - 1) * DAY;
  const c = loadEventCandles(Math.min(projectContinuation ? selectionTrainStart : s.policyCalibrationStart,
    compareHeadLag ? lagSelectionStart : Infinity) - 2 * DAY, window.endTime + DAY);
  const ranges = eventCalibrationRanges(s.policyCalibrationStart, s.calibrationEnd, s.selectionExcludedWindows);
  const appliedBlend = control ? 0 : h.selected?.blend;
  let lagSelectionHead: EventSizeSignHead | undefined, lagHead: EventSizeSignHead | undefined;
  if (compareHeadLag) {
    if (!h.selected || h.selected.eventHistory || !h.head || !("gate" in h.head) || "valueFit" in h.head
      || !h.selectionHead || !("gate" in h.selectionHead)) throw new Error("Head-age screen currently requires ordinary fixed size/sign heads without history or value correction");
    if (window.startTime - lagDays * DAY !== s.selectionTrainingEnd
      || s.selectionTrainingEnd - (sourceConfig.trainDays - 1) * DAY !== selectionTrainStart) throw new Error("Lagged final head cannot reuse the saved training interval");
    const excluded = sourceConfig.trainingIsolation === "causal" ? [window] : sourceConfig.excludedWindows;
    const lagFit = makeSamples(c, policy.model.clock, lagSelectionStart, lagSelectionEnd, excluded, sourceConfig.stride, "chain", policy.model.featureNames);
    lagSelectionHead = trainEventSizeSign(lagFit.map(sample => ({ ...sample,
      features: [...sample.features, ...(h.selected!.fastVolatility ? eventFastVolatilityFeatures(c, sample.start) : [])] })), h.selected.penalty, h.selected.quantile!);
    lagHead = h.selectionHead;
    fs.writeFileSync(path.join(output, `${window.id}-lag-head.json`), JSON.stringify({ selectionHead: lagSelectionHead, head: lagHead,
      lagDays, lagSelectionStart, lagSelectionEnd, selectionSamples: lagFit.length, finalTrainStart: selectionTrainStart, finalTrainEnd: s.selectionTrainingEnd }, null, 2));
  }
  if (projectContinuation && h.selected) {
    if (!h.head || !("gate" in h.head) || !h.selectionHead || !("gate" in h.selectionHead)) throw new Error("Continuation projection requires size/sign heads");
    const project = (p: EventPolicy, head: EventSizeSignHead, from: number, to: number, expectedSamples: number) => {
      const excluded = sourceConfig.trainingIsolation === "causal" ? [window] : sourceConfig.excludedWindows;
      const fit = makeSamples(c, p.model.clock, from, to, excluded, sourceConfig.stride, "chain", p.model.featureNames);
      if (fit.length !== expectedSamples) throw new Error("Continuation projection training population changed");
      let memory: EventCompletedHistory | undefined, previousEnd = -1;
      const probabilities = fit.map(sample => {
        const at = c[sample.start].openTime + 60_000;
        if (h.selected!.eventHistory && sample.start !== previousEnd) memory = new EventCompletedHistory(at);
        const inputs = [...sample.features, ...(h.selected!.fastVolatility ? eventFastVolatilityFeatures(c, sample.start) : []),
          ...(memory?.features(at) ?? [])];
        const probabilities = predictEventSizeSigns(head, inputs);
        memory?.observe({ ...sample, originTime: at, availableAt: c[sample.end].openTime + 60_000 }, c[sample.end].openTime + 60_000);
        previousEnd = sample.end;
        return { features: sample.features, probabilities };
      });
      const model = projectEventSizeSigns(p.model, head.thresholdLogBps, probabilities, appliedBlend!);
      return buildEventPolicy(model, p.costs, { depths: p.tables.length, referenceEquity: p.equities[2],
        referencePrice: p.prices[1], actionSteps: (p.targets.length - 1) / 2 });
    };
    selectionPolicy = project(selectionPolicy, h.selectionHead, selectionTrainStart, s.selectionTrainingEnd, summaries[i].selectionFitSamples);
    policy = project(policy, h.head, s.trainStart, s.trainEnd, summaries[i].fitSamples);
    fs.writeFileSync(path.join(output, `${window.id}-continuation.json`), JSON.stringify({
      selectionPolicy: serializeEventPolicy(selectionPolicy), policy: serializeEventPolicy(policy),
      selectionTrainStart, selectionTrainingEnd: s.selectionTrainingEnd, trainStart: s.trainStart, trainEnd: s.trainEnd,
      selectionSamples: summaries[i].selectionFitSamples, finalSamples: summaries[i].fitSamples }));
  }
  const prepare = (p: EventPolicy, head: typeof h.head): Pick<NonNullable<Parameters<typeof replayEventPolicy>[5]>, "sign" | "sizeSign"> => {
    if (!h.selected || !head) return {};
    if ("gate" in head) return { sizeSign: { head, blend: appliedBlend!, fastVolatility: h.selected.fastVolatility, eventHistory: h.selected.eventHistory,
      lookahead: buildEventOutcomeLookahead(p, 5, atom => eventSizeSignGroup(atom.return, head.thresholdLogBps)) } };
    return { sign: { head, blend: appliedBlend!, lookahead: buildEventSignLookahead(p) } };
  };
  const calOptions = prepare(selectionPolicy, h.selectionHead);
  const frozenCalOptions = compareContinuations ? prepare(originalSelectionPolicy, h.selectionHead) : undefined;
  const lagCalOptions = compareHeadLag ? prepare(selectionPolicy, lagSelectionHead) : undefined;
  const lagFrozenCalOptions = compareHeadLag ? prepare(originalSelectionPolicy, lagSelectionHead) : undefined;
  const policyChoices = compareHeadLag ? ["base", "head-frozen", "head", "head-lag-frozen", "head-lag"] as const
    : compareContinuations ? ["base", "head-frozen", "head"] as const : compareBase ? ["base", "head"] as const : ["head"] as const;
  const calibration = policyChoices.flatMap(choice => selectionPolicy.tables.map(({ depth }) => {
    const p = choice === "head" || choice === "head-lag" ? selectionPolicy : originalSelectionPolicy;
    const options = choice === "base" ? {} : choice === "head-frozen" ? frozenCalOptions
      : choice === "head-lag" ? lagCalOptions : choice === "head-lag-frozen" ? lagFrozenCalOptions : calOptions;
    const metrics = ranges.map(r => replayEventPolicy(c, p, r.startTime, r.endTime, depth, options));
    const logGrowth = metrics.reduce((n, m) => n + m.logGrowth, 0), drawdown = Math.max(0, ...metrics.map(m => m.maxDrawdownPct)) / 100;
    return { choice, depth, logGrowth, drawdown, score: logGrowth - sourceConfig.riskPenalty * drawdown,
      trades: metrics.reduce((n, m) => n + m.trades, 0) };
  })).sort((a, b) => b.score - a.score);
  const chosen = calibration[0], cash = chosen.score <= 0;
  fs.writeFileSync(path.join(output, `${window.id}-selection.json`), JSON.stringify({ window, selected: h.selected, appliedBlend, chosen, cash, calibration }, null, 2));
  const replayPolicy = chosen.choice === "head" || chosen.choice === "head-lag" ? policy : originalPolicy;
  const replayHead = chosen.choice === "head-lag" || chosen.choice === "head-lag-frozen" ? lagHead : h.head;
  const replay = replayEventPolicy(c, replayPolicy, window.startTime, window.endTime, chosen.depth,
    { ...(chosen.choice === "base" ? {} : prepare(replayPolicy, replayHead)), cash, trace: true });
  fs.writeFileSync(path.join(output, `${window.id}-trades.json`), JSON.stringify(replay.trace));
  const { trace: _, ...test } = replay;
  const result = { window, selected: h.selected, appliedBlend, chosen, cash, test, elapsedSec: (performance.now() - started) / 1000 };
  results.push(result);
  fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify(results, null, 2));
  console.log(JSON.stringify({ event: "sign-lookahead-replay", ...result, test: { ...test, daily: undefined, adaptationPairs: undefined } }));
}
