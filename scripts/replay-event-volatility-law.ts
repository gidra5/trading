/** Fee-aware comparison of the forecast-selected joint volatility law. */
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { buildEventPolicy, buildEventOutcomeLookahead, restoreEventPolicy, serializeEventPolicy } from "../packages/bot-algo/src/event-log-policy.js";
import { eventSizeSignGroup } from "../packages/bot-algo/src/event-size-sign.js";
import { eventOriginScore, eventRefitOrigins } from "./research-event-refits.js";
import { loadEventCandles, replayEventPolicy } from "./research-event-policy.js";

const root = path.resolve(__dirname, ".."), DAY = 86_400_000;
const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
const forecastId = arg("forecast"), outputId = arg("output");
if (!forecastId || !outputId) throw new Error("Specify joint-law forecast and new output");
const forecast = path.resolve(root, "data/benchmarks", forecastId), output = path.resolve(root, "data/benchmarks", outputId);
const read = (dir: string, file: string) => JSON.parse(fs.readFileSync(path.join(dir, file), "utf8"));
if (fs.existsSync(output)) throw new Error("Choose a new output directory");
const fc = read(forecast, "config.json"), selection = read(forecast, "selection.json"), source = fc.source;
const config = read(source, "config.json"), sc = read(config.source, "config.json"), window = fc.window;
if (fc.contract !== "event-volatility-law-screen-v1" || config.contract !== "rolling-event-refit-v1" || !["local", "pooled", "hybrid", "quadrature"].includes(selection.selected))
  throw new Error("No selected new joint volatility law to replay");
const origins = eventRefitOrigins(window.startTime, config.foldCount, config.foldDays), phases = [...origins, { ...window, id: "final" }];
const hash = createHash("sha256").update(fs.readFileSync(path.join(source, "config.json")));
for (const p of phases) hash.update(fs.readFileSync(path.join(source, `${window.id}-${p.id}-model.json`)));
if (hash.digest("hex") !== fc.sourceHash) throw new Error("Origin source changed");
const modelHash = createHash("sha256").update(fs.readFileSync(path.join(forecast, "config.json"))).update(fs.readFileSync(path.join(forecast, "selection.json")));
for (const p of phases) modelHash.update(fs.readFileSync(path.join(forecast, `${p.id}-models.json`)));
fs.mkdirSync(output, { recursive: true });
fs.writeFileSync(path.join(output, "config.json"), JSON.stringify({ contract: "event-volatility-law-policy-v1", source, forecast, modelHash: modelHash.digest("hex"), window,
  selectedLaw: selection.selected, selection: "All incumbent rolling policy choices plus the forecast-selected law at depths 1-8; mean origin log growth minus configured drawdown penalty; cash eligible",
  caveat: "The same prior origins screen forecasts and policy utility. The final research window is diagnostic, not an untouched outer holdout. New-law diagnostic uses its best calibration depth even if an incumbent wins." }, null, 2));
const files = ["scripts/replay-event-volatility-law.ts", "scripts/research-event-policy.ts", "packages/bot-algo/src/event-distribution.ts", "packages/bot-algo/src/event-volatility-law.ts", "packages/bot-algo/src/event-log-policy.ts"];
fs.writeFileSync(path.join(output, "sources.json"), JSON.stringify(Object.fromEntries(files.map(f => [f, fs.readFileSync(path.join(root, f), "utf8")]))));
const started = performance.now(), c = loadEventCandles(origins[0].startTime - 2 * DAY, window.endTime + DAY), folds: any[] = [];
const setting = config.settings.find((s: any) => s.window.id === window.id).head;
let chosen: any, diagnosticChoice: any, cash = false;
for (const phase of phases) {
  const start = performance.now();
  if (phase.id === "final") {
    const ranking = [...config.choices, "joint-volatility"].flatMap(choice => Array.from({ length: sc.maxDepth }, (_, d) => {
      const depth = d + 1, rows = folds.map(f => f.rows.find((r: any) => r.choice === choice && r.depth === depth));
      return { choice, depth, ...eventOriginScore(rows, sc.riskPenalty), trades: rows.reduce((s, r) => s + r.trades, 0) };
    })).sort((a, b) => b.score - a.score);
    chosen = ranking[0]; cash = chosen.score <= 0; diagnosticChoice = ranking.find(r => r.choice === "joint-volatility");
    fs.writeFileSync(path.join(output, "selection.json"), JSON.stringify({ chosen, cash, diagnosticChoice, ranking }, null, 2));
  }
  const saved = read(source, `${window.id}-${phase.id}-model.json`), base = restoreEventPolicy(saved.base);
  const model = read(forecast, `${phase.id}-models.json`).models[selection.selected];
  if (saved.origin !== phase.startTime || saved.support.lastTarget >= phase.startTime) throw new Error("Future origin model");
  const policy = buildEventPolicy(model, base.costs, { depths: sc.maxDepth, referenceEquity: base.equities[2], referencePrice: base.prices[1], actionSteps: sc.actionSteps });
  fs.writeFileSync(path.join(output, `${phase.id}-policy.json`), JSON.stringify(serializeEventPolicy(policy)));
  const headOptions = (p: typeof base, head: typeof saved.head) => ({ sizeSign: { head, blend: setting.blend,
    fastVolatility: setting.fastVolatility, eventHistory: setting.eventHistory,
    lookahead: buildEventOutcomeLookahead(p, 5, a => eventSizeSignGroup(a.return, head.thresholdLogBps)) } });
  if (phase.id !== "final") {
    const old = read(source, `${window.id}-${phase.id}-scores.json`);
    const control = replayEventPolicy(c, base, phase.startTime, phase.endTime, sc.maxDepth, headOptions(base, saved.head));
    const expected = old.rows.find((r: any) => r.choice === "head-base" && r.depth === sc.maxDepth);
    const controlError = Math.max(Math.abs(control.returnPct - expected.returnPct), Math.abs(control.maxDrawdownPct - expected.maxDrawdownPct));
    if (controlError > 1e-10 || control.trades !== expected.trades) throw new Error("Incumbent control failed");
    const extra = Array.from({ length: sc.maxDepth }, (_, d) => {
      const depth = d + 1, { trace, ...metrics } = replayEventPolicy(c, policy, phase.startTime, phase.endTime, depth);
      return { choice: "joint-volatility", depth, ...metrics };
    });
    const fold = { phase, controlError, rows: [...old.rows, ...extra] };
    folds.push(fold); fs.writeFileSync(path.join(output, `${phase.id}-scores.json`), JSON.stringify(fold));
    console.log(JSON.stringify({ event: "joint-policy-origin", phase: phase.id, controlError, states: model.kernels.length, elapsedSec: (performance.now() - start) / 1000 }));
  } else {
    const incumbent = chosen.choice.endsWith("projected") ? restoreEventPolicy(saved.projected) : base;
    const chosenPolicy = chosen.choice === "joint-volatility" ? policy : incumbent;
    const options = chosen.choice === "joint-volatility" || chosen.choice === "base" ? {} : headOptions(incumbent, chosen.choice.startsWith("lag") ? saved.lagHead : saved.head);
    const replay = replayEventPolicy(c, chosenPolicy, phase.startTime, phase.endTime, chosen.depth, { ...options, cash, trace: true });
    fs.writeFileSync(path.join(output, "trades.json"), JSON.stringify(replay.trace));
    const { trace, ...test } = replay;
    const diagnostic = chosen.choice === "joint-volatility" ? replay : replayEventPolicy(c, policy, phase.startTime, phase.endTime, diagnosticChoice.depth, { cash: diagnosticChoice.score <= 0, trace: true });
    fs.writeFileSync(path.join(output, "joint-law-diagnostic-trades.json"), JSON.stringify(diagnostic.trace));
    const { trace: diagnosticTrace, ...diagnosticMetrics } = diagnostic;
    const result = { window, chosen, cash, test, diagnosticChoice, diagnostic: diagnosticMetrics, elapsedSec: (performance.now() - started) / 1000 };
    fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify(result, null, 2));
    console.log(JSON.stringify({ ...result, test: { ...test, daily: undefined, adaptationPairs: undefined }, diagnostic: { ...diagnosticMetrics, daily: undefined, adaptationPairs: undefined } }));
  }
}
