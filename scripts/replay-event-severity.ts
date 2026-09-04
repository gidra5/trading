/** Utility comparison for the forecast-selected severity head. Existing policy
 * families stay eligible; only pre-window calibration selects the winner. */
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { buildEventKernelLookahead, buildEventOutcomeLookahead, restoreEventPolicy, type EventPolicy } from "../packages/bot-algo/src/event-log-policy.js";
import { eventSizeSignGroup, type EventSizeSignHead } from "../packages/bot-algo/src/event-size-sign.js";
import type { EventSeverityHead } from "../packages/bot-algo/src/event-severity.js";
import { eventCalibrationRanges, loadEventCandles, replayEventPolicy } from "./research-event-policy.js";

const root = path.resolve(__dirname, ".."), DAY = 86_400_000;
const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
const forecastId = arg("forecast"), incumbentId = arg("incumbent"), outputId = arg("output"), control = process.argv.includes("--control");
if (!forecastId || !incumbentId || !outputId) throw new Error("Specify severity forecast, incumbent comparison and new output");
const forecast = path.resolve(root, "data/benchmarks", forecastId), incumbent = path.resolve(root, "data/benchmarks", incumbentId), output = path.resolve(root, "data/benchmarks", outputId);
const read = (dir: string, file: string) => JSON.parse(fs.readFileSync(path.join(dir, file), "utf8"));
if (fs.existsSync(output)) throw new Error("Choose a new output directory");
const fc = read(forecast, "config.json"), ic = read(incumbent, "config.json"), source: string = fc.source, sc = read(source, "config.json");
if (fc.contract !== "event-severity-screen-v1" || ic.contract !== "event-sign-lookahead-replay-v1" || !ic.compareContinuations
  || !ic.projectContinuation || ic.compareHeadLag || ic.control || ic.forecast !== fc.forecast || ic.source !== source) throw new Error("Incompatible severity/incumbent artifacts");
const summary = read(source, "summary.json");
const requested = arg("windows").split(",").filter(Boolean), ids: string[] = requested.length ? requested : fc.windows;
if (ids.some(id => !fc.windows.includes(id) || !summary.some((s: any) => s.window.id === id))) throw new Error("Unknown severity window");
const hash = createHash("sha256").update(fs.readFileSync(path.join(source, "config.json")));
for (const s of summary) hash.update(fs.readFileSync(path.join(source, `${s.window.id}-model.json`)));
if (hash.digest("hex") !== fc.sourceHash) throw new Error("Severity base source changed");
const headHash = createHash("sha256").update(fs.readFileSync(path.join(fc.forecast, "config.json")));
for (const s of summary) headHash.update(fs.readFileSync(path.join(fc.forecast, `${s.window.id}-model.json`)));
if (headHash.digest("hex") !== fc.headHash) throw new Error("Frozen size/sign heads changed");
fs.mkdirSync(output, { recursive: true });
fs.writeFileSync(path.join(output, "config.json"), JSON.stringify({ contract: "event-severity-policy-replay-v1", forecast, incumbent, source, control, windows: ids,
  selection: "All incumbent policy/depth choices plus severity with original or projected continuation; preceding calibration log growth minus configured drawdown penalty; cash eligible",
  approximation: "Exact probability reweighting of the current joint event, with fixed incumbent continuation at later events; no severity interpolation or synthetic paths",
  caveat: "One policy-improvement backup with observed features, not fully recursive severity-state learning or untouched holdout profitability" }, null, 2));
const files = ["scripts/replay-event-severity.ts", "scripts/research-event-policy.ts", "packages/bot-algo/src/event-severity.ts",
  "packages/bot-algo/src/event-size-sign.ts", "packages/bot-algo/src/event-completed-history.ts", "packages/bot-algo/src/event-log-policy.ts"];
fs.writeFileSync(path.join(output, "sources.json"), JSON.stringify(Object.fromEntries(files.map(f => [f, fs.readFileSync(path.join(root, f), "utf8")]))));
const results = [];
for (const row of summary) {
  const { window } = row, id = window.id;
  if (!ids.includes(id)) continue;
  const started = performance.now(), s = read(source, `${id}-model.json`), h = read(forecast, `${id}-model.json`), projected = read(incumbent, `${id}-continuation.json`);
  if (h.trainEnd !== s.trainEnd || h.selectionTrainingEnd !== s.selectionTrainingEnd || s.trainEnd > window.startTime
    || s.calibrationEnd > window.startTime || id.startsWith("fit-") || (h.selected && (!h.severityHead || !h.selectionSeverityHead))) throw new Error("Invalid severity fit chronology");
  fs.copyFileSync(path.join(forecast, `${id}-model.json`), path.join(output, `${id}-severity-model.json`));
  const p = restoreEventPolicy(s.selectionPolicy), pp = restoreEventPolicy(projected.selectionPolicy);
  const c = loadEventCandles(s.policyCalibrationStart - 2 * DAY, window.endTime + DAY);
  const ranges = eventCalibrationRanges(s.policyCalibrationStart, s.calibrationEnd, s.selectionExcludedWindows);
  const prepare = (p: EventPolicy, head: EventSizeSignHead, severity?: EventSeverityHead) => ({ sizeSign: { head, blend: h.headSetting.blend,
    fastVolatility: h.headSetting.fastVolatility, eventHistory: h.headSetting.eventHistory,
    lookahead: buildEventOutcomeLookahead(p, 5, a => eventSizeSignGroup(a.return, head.thresholdLogBps)) },
    ...(severity ? { severity: { head: severity, blend: control ? 0 : h.selected.blend, lookahead: buildEventKernelLookahead(p) } } : {}) });
  const options = prepare(p, h.selectionHead), projectedOptions = prepare(pp, h.selectionHead);
  const severityOptions = h.selected ? prepare(p, h.selectionHead, h.selectionSeverityHead) : undefined;
  const projectedSeverityOptions = h.selected ? prepare(pp, h.selectionHead, h.selectionSeverityHead) : undefined;
  const variants = [{ choice: "base", p, options: {} }, { choice: "head-frozen", p, options }, { choice: "head", p: pp, options: projectedOptions },
    ...(h.selected ? [{ choice: "severity-frozen", p, options: severityOptions! }, { choice: "severity", p: pp, options: projectedSeverityOptions! }] : [])];
  const calibration = variants.flatMap(variant => {
    const start = performance.now();
    const scores = variant.p.tables.map(({ depth }) => {
      const metrics = ranges.map(r => replayEventPolicy(c, variant.p, r.startTime, r.endTime, depth, variant.options));
      const logGrowth = metrics.reduce((n, m) => n + m.logGrowth, 0), drawdown = Math.max(0, ...metrics.map(m => m.maxDrawdownPct)) / 100;
      return { choice: variant.choice, depth, logGrowth, drawdown, score: logGrowth - sc.riskPenalty * drawdown,
        trades: metrics.reduce((n, m) => n + m.trades, 0) };
    });
    console.log(JSON.stringify({ event: "severity-calibration", window: id, choice: variant.choice, elapsedSec: (performance.now() - start) / 1000 }));
    return scores;
  }).sort((a, b) => b.score - a.score);
  const old = read(incumbent, `${id}-selection.json`);
  const maxIncumbentScoreDifference = Math.max(...old.calibration.map((r: any) => Math.abs(r.score - calibration.find(v => v.choice === r.choice && v.depth === r.depth)!.score)));
  if (maxIncumbentScoreDifference > 1e-12) throw new Error("Existing policy calibration does not reproduce");
  const chosen = calibration[0], cash = chosen.score <= 0;
  fs.writeFileSync(path.join(output, `${id}-selection.json`), JSON.stringify({ window, selected: h.selected, chosen, cash, calibration, maxIncumbentScoreDifference }, null, 2));
  const finalPolicy = restoreEventPolicy(["head", "severity"].includes(chosen.choice) ? projected.policy : s.policy);
  const replay = replayEventPolicy(c, finalPolicy, window.startTime, window.endTime, chosen.depth,
    { ...(chosen.choice === "base" ? {} : prepare(finalPolicy, h.head, chosen.choice.startsWith("severity") ? h.severityHead : undefined)), cash, trace: true });
  fs.writeFileSync(path.join(output, `${id}-trades.json`), JSON.stringify(replay.trace));
  const { trace: _, ...test } = replay;
  const result = { window, selected: h.selected, chosen, cash, maxIncumbentScoreDifference, test,
    kernelCacheCells: [severityOptions, projectedSeverityOptions].map(o => o?.severity?.lookahead.cells.size ?? 0), elapsedSec: (performance.now() - started) / 1000 };
  results.push(result); fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify(results, null, 2));
  console.log(JSON.stringify({ event: "severity-policy-result", ...result, test: { ...test, daily: undefined, adaptationPairs: undefined } }));
}
