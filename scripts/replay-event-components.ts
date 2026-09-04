/** Retain useful state-law conditionals while testing the learned size gate
 * and learned sign heads separately. No additional model fitting. */
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { buildEventOutcomeLookahead, restoreEventPolicy, type EventPolicy } from "../packages/bot-algo/src/event-log-policy.js";
import { eventSizeSignGroup, type EventSizeSignHead } from "../packages/bot-algo/src/event-size-sign.js";
import { eventCalibrationRanges, loadEventCandles, replayEventPolicy } from "./research-event-policy.js";

const root = path.resolve(__dirname, ".."), DAY = 86_400_000;
const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
const sourceId = arg("source"), outputId = arg("output");
if (!sourceId || !outputId) throw new Error("Specify incumbent comparison and new output");
const source = path.resolve(root, "data/benchmarks", sourceId), output = path.resolve(root, "data/benchmarks", outputId);
const read = (dir: string, file: string) => JSON.parse(fs.readFileSync(path.join(dir, file), "utf8"));
const config = read(source, "config.json"), summary = read(source, "summary.json"), baseConfig = read(config.source, "config.json");
if (config.contract !== "event-sign-lookahead-replay-v1" || !config.compareContinuations || !config.projectContinuation || config.control || config.compareHeadLag)
  throw new Error("Requires original, frozen-head and projected-head incumbent comparison");
const requested = arg("windows").split(",").filter(Boolean), ids = requested.length ? requested : config.windows;
if (ids.some((id: string) => !summary.some((s: any) => s.window.id === id))) throw new Error("Unknown component window");
if (fs.existsSync(output)) throw new Error("Choose a new output directory");
const fingerprint = createHash("sha256").update(fs.readFileSync(path.join(source, "config.json")));
for (const s of summary) for (const file of [`${s.window.id}-sign-model.json`, `${s.window.id}-continuation.json`, `${s.window.id}-selection.json`])
  fingerprint.update(fs.readFileSync(path.join(source, file)));
fs.mkdirSync(output, { recursive: true });
fs.writeFileSync(path.join(output, "config.json"), JSON.stringify({ contract: "event-component-policy-replay-v1", source, sourceHash: fingerprint.digest("hex"), windows: ids,
  selection: "All 24 incumbent choices plus learned gate only and learned signs only with original continuation, each at depths 1 through 8; preceding calibration utility and cash gate",
  caveat: "Frozen previously selected head settings; two component ablations, not new independent validation or full recursive fine-feature learning" }, null, 2));
const files = ["scripts/replay-event-components.ts", "scripts/research-event-policy.ts", "packages/bot-algo/src/event-size-sign.ts", "packages/bot-algo/src/event-log-policy.ts"];
fs.writeFileSync(path.join(output, "sources.json"), JSON.stringify(Object.fromEntries(files.map(f => [f, fs.readFileSync(path.join(root, f), "utf8")]))));
const results = [];
for (const s of summary) {
  const { window } = s, id = window.id;
  if (!ids.includes(id)) continue;
  const started = performance.now(), model = read(config.source, `${id}-model.json`), heads = read(source, `${id}-sign-model.json`);
  const projected = read(source, `${id}-continuation.json`), old = read(source, `${id}-selection.json`);
  if (id.startsWith("fit-") || model.trainEnd > window.startTime || model.calibrationEnd > window.startTime || heads.trainEnd !== model.trainEnd
    || heads.selectionTrainingEnd !== model.selectionTrainingEnd || !heads.selected) throw new Error("Invalid component fit chronology");
  fs.writeFileSync(path.join(output, `${id}-head.json`), JSON.stringify(heads));
  const c = loadEventCandles(model.policyCalibrationStart - 2 * DAY, window.endTime + DAY);
  const ranges = eventCalibrationRanges(model.policyCalibrationStart, model.calibrationEnd, model.selectionExcludedWindows);
  const weights: Record<string, number[]> = { "gate-only": [1, 0, 0], "signs-only": [0, 1, 1] };
  const prepare = (p: EventPolicy, head: EventSizeSignHead, choice: string) => choice === "base" ? {} : { sizeSign: { head,
    blend: heads.selected.blend, fastVolatility: heads.selected.fastVolatility, eventHistory: heads.selected.eventHistory, components: weights[choice],
    lookahead: buildEventOutcomeLookahead(p, 5, a => eventSizeSignGroup(a.return, head.thresholdLogBps)) } };
  const p = restoreEventPolicy(model.selectionPolicy), pp = restoreEventPolicy(projected.selectionPolicy);
  const choices = ["base", "head-frozen", "head", "gate-only", "signs-only"];
  const calibration = choices.flatMap(choice => {
    const policy = choice === "head" ? pp : p, options = prepare(policy, heads.selectionHead, choice);
    return policy.tables.map(({ depth }) => {
      const metrics = ranges.map(r => replayEventPolicy(c, policy, r.startTime, r.endTime, depth, options));
      const logGrowth = metrics.reduce((s, m) => s + m.logGrowth, 0), drawdown = Math.max(0, ...metrics.map(m => m.maxDrawdownPct)) / 100;
      return { choice, depth, logGrowth, drawdown, score: logGrowth - baseConfig.riskPenalty * drawdown, trades: metrics.reduce((s, m) => s + m.trades, 0) };
    });
  }).sort((a, b) => b.score - a.score);
  const maxIncumbentScoreDifference = Math.max(...old.calibration.map((r: any) => Math.abs(r.score - calibration.find(v => v.choice === r.choice && v.depth === r.depth)!.score)));
  if (maxIncumbentScoreDifference > 1e-12) throw new Error("Existing policy calibration does not reproduce");
  const chosen = calibration[0], cash = chosen.score <= 0;
  fs.writeFileSync(path.join(output, `${id}-selection.json`), JSON.stringify({ window, chosen, cash, calibration, maxIncumbentScoreDifference }, null, 2));
  const finalPolicy = restoreEventPolicy(chosen.choice === "head" ? projected.policy : model.policy);
  const replay = replayEventPolicy(c, finalPolicy, window.startTime, window.endTime, chosen.depth,
    { ...prepare(finalPolicy, heads.head, chosen.choice), cash, trace: true });
  fs.writeFileSync(path.join(output, `${id}-trades.json`), JSON.stringify(replay.trace));
  const { trace: _, ...test } = replay;
  const result = { window, chosen, cash, maxIncumbentScoreDifference, test, elapsedSec: (performance.now() - started) / 1000 };
  results.push(result); fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify(results, null, 2));
  console.log(JSON.stringify({ ...result, test: { ...test, daily: undefined, adaptationPairs: undefined } }));
}
