/** Economic check of the forecast-selected pooled head, preserving all saved
 * rolling-origin incumbent policy choices and their original selection rule. */
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { EventCompletedHistory } from "../packages/bot-algo/src/event-completed-history.js";
import { buildEventPolicy, buildEventOutcomeLookahead, restoreEventPolicy, serializeEventPolicy, type EventPolicy } from "../packages/bot-algo/src/event-log-policy.js";
import { eventFastVolatilityFeatures, eventSizeSignGroup, predictEventSizeSigns, projectEventSizeSigns, type EventSizeSignHead } from "../packages/bot-algo/src/event-size-sign.js";
import { eventOriginScore, eventRefitOrigins } from "./research-event-refits.js";
import { loadEventCandles, makeSamples, replayEventPolicy } from "./research-event-policy.js";

const root = path.resolve(__dirname, ".."), DAY = 86_400_000;
const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
const forecastId = arg("forecast"), outputId = arg("output");
if (!forecastId || !outputId) throw new Error("Specify pooled forecast and new output");
const forecast = path.resolve(root, "data/benchmarks", forecastId), output = path.resolve(root, "data/benchmarks", outputId);
const read = (dir: string, file: string) => JSON.parse(fs.readFileSync(path.join(dir, file), "utf8"));
if (fs.existsSync(output)) throw new Error("Choose a new output directory");
const fc = read(forecast, "config.json"), forecastSelection = read(forecast, "selection.json"), source = fc.source;
const config = read(source, "config.json"), sc = read(config.source, "config.json"), window = fc.window;
if (fc.contract !== "event-pooling-forecast-v1" || config.contract !== "rolling-event-refit-v1" || !fc.choices.includes(forecastSelection.selected))
  throw new Error("Incompatible pooling source");
const origins = eventRefitOrigins(window.startTime, config.foldCount, config.foldDays), phases = [...origins, { ...window, id: "final" }];
const hash = createHash("sha256").update(fs.readFileSync(path.join(source, "config.json")));
for (const p of phases) hash.update(fs.readFileSync(path.join(source, `${window.id}-${p.id}-model.json`)));
if (hash.digest("hex") !== fc.sourceHash) throw new Error("Pooled forecast source models changed");
const headHash = createHash("sha256").update(fs.readFileSync(path.join(forecast, "config.json"))).update(fs.readFileSync(path.join(forecast, "selection.json")));
for (const p of phases) headHash.update(fs.readFileSync(path.join(forecast, `${p.id}-heads.json`)));
fs.mkdirSync(output, { recursive: true });
fs.writeFileSync(path.join(output, "config.json"), JSON.stringify({ contract: "event-pooling-policy-v1", forecast, source, headHash: headHash.digest("hex"), window,
  forecastChoice: forecastSelection.selected,
  selection: "Retain all original rolling-origin choices; add forecast-selected pooled head with original and projected continuation at every original depth; mean origin log growth minus drawdown penalty; cash eligible",
  caveat: "Reuses the forecast-selection origins for economic selection, so not a nested untouched validation. Coarse-state continuation averages current-head probabilities over recent training features." }, null, 2));
const files = ["scripts/replay-event-pooling.ts", "scripts/research-event-policy.ts", "scripts/research-event-refits.ts", "packages/bot-algo/src/event-log-policy.ts", "packages/bot-algo/src/event-size-sign.ts"];
fs.writeFileSync(path.join(output, "sources.json"), JSON.stringify(Object.fromEntries(files.map(f => [f, fs.readFileSync(path.join(root, f), "utf8")]))));
const started = performance.now(), first = read(source, `${window.id}-${origins[0].id}-model.json`);
const c = loadEventCandles(first.trainStart - 2 * DAY, window.endTime + DAY), folds: any[] = [];
const candidates = [...config.choices, "pooled-base", "pooled-projected"];
let chosen: any, cash = false;
for (const phase of phases) {
  const start = performance.now();
  if (phase.id === "final") {
    const ranking = candidates.flatMap(choice => Array.from({ length: sc.maxDepth }, (_, d) => {
      const depth = d + 1, rows = folds.map(f => f.rows.find((r: any) => r.choice === choice && r.depth === depth));
      return { choice, depth, ...eventOriginScore(rows, sc.riskPenalty), trades: rows.reduce((s, r) => s + r.trades, 0) };
    })).sort((a, b) => b.score - a.score);
    chosen = ranking[0]; cash = chosen.score <= 0;
    fs.writeFileSync(path.join(output, "selection.json"), JSON.stringify({ chosen, cash, ranking, origins }, null, 2));
  }
  const saved = read(source, `${window.id}-${phase.id}-model.json`), heads = read(forecast, `${phase.id}-heads.json`);
  const pooled: EventSizeSignHead = heads.heads[forecastSelection.selected], setting = heads.headSetting;
  const p = restoreEventPolicy(saved.base), projected = restoreEventPolicy(saved.projected);
  if (saved.origin !== phase.startTime || saved.support.lastTarget >= phase.startTime
    || pooled.thresholdLogBps !== saved.head.thresholdLogBps) throw new Error("Invalid pooled phase chronology or threshold");
  const rows = makeSamples(c, sc.clock, saved.trainStart, saved.trainEnd, [window], sc.stride, "chain", sc.featureNames);
  if (rows.length !== saved.support.samples) throw new Error("Continuation projection training population changed");
  let previousEnd = -1, memory: EventCompletedHistory | undefined;
  const probabilities = rows.map(row => {
    const time = c[row.start].openTime + 60_000, availableAt = c[row.end].openTime + 60_000;
    if (setting.eventHistory && row.start !== previousEnd) memory = new EventCompletedHistory(time);
    const features = [...row.features, ...(setting.fastVolatility ? eventFastVolatilityFeatures(c, row.start) : []), ...(memory?.features(time) ?? [])];
    const probabilities = predictEventSizeSigns(pooled, features);
    memory?.observe({ ...row, originTime: time, availableAt }, availableAt); previousEnd = row.end;
    return { features: row.features, probabilities };
  });
  const pooledProjected = buildEventPolicy(projectEventSizeSigns(p.model, pooled.thresholdLogBps, probabilities, setting.blend), p.costs,
    { depths: p.tables.length, referenceEquity: p.equities[2], referencePrice: p.prices[1], actionSteps: sc.actionSteps });
  fs.writeFileSync(path.join(output, `${phase.id}-model.json`), JSON.stringify({ head: pooled, headSetting: setting, projected: serializeEventPolicy(pooledProjected) }));
  const prepare = (policy: EventPolicy, head: EventSizeSignHead) => ({ sizeSign: { head, blend: setting.blend,
    fastVolatility: setting.fastVolatility, eventHistory: setting.eventHistory,
    lookahead: buildEventOutcomeLookahead(policy, 5, a => eventSizeSignGroup(a.return, head.thresholdLogBps)) } });
  if (phase.id !== "final") {
    const old = read(source, `${window.id}-${phase.id}-scores.json`), controlOptions = prepare(p, saved.head);
    const check = replayEventPolicy(c, p, phase.startTime, phase.endTime, sc.maxDepth, controlOptions);
    const expected = old.rows.find((r: any) => r.choice === "head-base" && r.depth === sc.maxDepth);
    const error = Math.max(Math.abs(check.returnPct - expected.returnPct), Math.abs(check.maxDrawdownPct - expected.maxDrawdownPct));
    if (error > 1e-10 || check.trades !== expected.trades) throw new Error("Incumbent rolling replay does not reproduce");
    const extra = ["pooled-base", "pooled-projected"].flatMap(choice => {
      const policy = choice === "pooled-base" ? p : pooledProjected, options = prepare(policy, pooled);
      return Array.from({ length: sc.maxDepth }, (_, d) => {
        const depth = d + 1, { trace: _trace, ...metrics } = replayEventPolicy(c, policy, phase.startTime, phase.endTime, depth, options);
        return { choice, depth, ...metrics };
      });
    });
    const fold = { phase, rows: [...old.rows, ...extra], controlError: error };
    folds.push(fold); fs.writeFileSync(path.join(output, `${phase.id}-scores.json`), JSON.stringify(fold));
    console.log(JSON.stringify({ event: "pooled-policy-origin", phase: phase.id, controlError: error, elapsedSec: (performance.now() - start) / 1000 }));
  } else {
    const policy = chosen.choice === "pooled-projected" ? pooledProjected : chosen.choice.endsWith("projected") ? projected : p;
    const head = chosen.choice.startsWith("pooled") ? pooled : chosen.choice.startsWith("lag") ? saved.lagHead : saved.head;
    const replay = replayEventPolicy(c, policy, phase.startTime, phase.endTime, chosen.depth, { ...(chosen.choice === "base" ? {} : prepare(policy, head)), cash, trace: true });
    fs.writeFileSync(path.join(output, "trades.json"), JSON.stringify(replay.trace));
    const { trace: _, ...test } = replay;
    const result = { window, chosen, cash, test, elapsedSec: (performance.now() - started) / 1000 };
    fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify(result, null, 2));
    console.log(JSON.stringify({ ...result, test: { ...test, daily: undefined, adaptationPairs: undefined } }));
  }
}
