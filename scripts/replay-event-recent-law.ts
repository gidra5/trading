/** Frozen forecast-selected updates; depth/cash reselected only on preceding calibration. */
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { EventRecentLaw, type RecentEventOptions } from "../packages/bot-algo/src/event-recent-law.js";
import { buildEventPolicy, restoreEventPolicy, type EventPolicy, type SerializedEventPolicy } from "../packages/bot-algo/src/event-log-policy.js";
import { type EventCandle } from "../packages/bot-algo/src/event-distribution.js";
import { eventCalibrationRanges, loadEventCandles, makeSamples, replayEventPolicy, type EventPolicyUpdate } from "./research-event-policy.js";
import { EventSecondBasis } from "./event-second-basis.js";

const root = path.resolve(__dirname, ".."), DAY = 86_400_000;
const arg = (key: string, fallback = "") => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? fallback : process.argv[i + 1]; };
if (!arg("forecast") || !arg("output")) throw new Error("Specify forecast screen and new output directory");
const forecast = path.resolve(root, "data/benchmarks", arg("forecast")), output = path.resolve(root, "data/benchmarks", arg("output"));
if (fs.existsSync(output)) throw new Error("Choose a new output directory");
const forecastConfig = JSON.parse(fs.readFileSync(path.join(forecast, "config.json"), "utf8"));
if (forecastConfig.contract !== "event-recent-law-forecast-screen-v1") throw new Error("Invalid forecast screen");
const source: string = forecastConfig.source;
const sourceConfig = JSON.parse(fs.readFileSync(path.join(source, "config.json"), "utf8"));
const sourceRows = JSON.parse(fs.readFileSync(path.join(source, "summary.json"), "utf8")) as Array<{
  window: { id: string; startTime: number; endTime: number }; insufficientCalibration?: boolean;
}>;
const hash = createHash("sha256").update(fs.readFileSync(path.join(source, "config.json")));
for (const row of sourceRows) hash.update(fs.readFileSync(path.join(source, `${row.window.id}-model.json`)));
if (hash.digest("hex") !== forecastConfig.sourceHash) throw new Error("Source models changed since forecast selection");
const requested = arg("windows", "all").split(","), refresh = Number(arg("refresh-events", "32"));
if (!Number.isInteger(refresh) || refresh < 1 || (requested[0] !== "all" && requested.some(id => !sourceRows.some(r => r.window.id === id))))
  throw new Error("Invalid refresh interval or source window");
const rows = sourceRows.filter(r => requested[0] === "all" || requested.includes(r.window.id));
const second = sourceConfig.secondBasisFingerprint ? new EventSecondBasis(path.join(root, "data/runtime-cache/global-feature-basis")) : undefined;
if (second && second.fingerprint !== sourceConfig.secondBasisFingerprint) throw new Error("Feature source changed");
fs.mkdirSync(output, { recursive: true });
fs.writeFileSync(path.join(output, "config.json"), JSON.stringify({ forecast, source, refresh, sourceHash: forecastConfig.sourceHash,
  contract: "event-recent-law-policy-replay-v1", windows: rows.map(r => r.window.id),
  selection: "Forecast update setting inherited from preceding calibration; depth/cash reselected on the same preceding calibration with scheduled updates",
  approximation: "Each Bellman solve freezes the current joint law; future belief updates are not part of its state",
  caveat: "Repeatedly inspected research suite; not an untouched holdout" }, null, 2));
const files = ["scripts/replay-event-recent-law.ts", "scripts/research-event-policy.ts", "scripts/event-second-basis.ts",
  "packages/bot-algo/src/event-recent-law.ts", "packages/bot-algo/src/event-log-policy.ts", "packages/bot-algo/src/event-distribution.ts"];
fs.writeFileSync(path.join(output, "sources.json"), JSON.stringify(Object.fromEntries(files.map(f => [f, fs.readFileSync(path.join(root, f), "utf8")]))));

function schedule(c: EventCandle[], base: EventPolicy, range: { startTime: number; endTime: number }, after: number,
  setting: Omit<RecentEventOptions, "after"> | null, label: string) {
  const updates: EventPolicyUpdate[] = [], audit: unknown[] = [];
  if (!setting) return { updates, audit };
  const online = new EventRecentLaw(base.model, { ...setting, after });
  const samples = makeSamples(c, base.model.clock, range.startTime, range.endTime, [], 1, "chain", base.model.featureNames);
  let completed = 0;
  for (const s of samples) {
    const originTime = c[s.start].openTime + 60_000, availableAt = c[s.end].openTime + 60_000;
    online.observe({ ...s, originTime, availableAt }, availableAt); completed++;
    if (completed % refresh) continue;
    const model = online.snapshot(availableAt), started = performance.now();
    const policy = buildEventPolicy(model, base.costs, { depths: base.tables.length, referenceEquity: base.equities[2],
      referencePrice: base.prices[1], actionSteps: (base.targets.length - 1) / 2 });
    updates.push({ at: availableAt, policy });
    audit.push({ at: availableAt, completed, lastCompletedAt: availableAt,
      meansBps: model.kernels.map(k => k.reduce((v, a) => v + a.probability * a.return, 0) * 1e4),
      buildSeconds: (performance.now() - started) / 1000 });
    if (updates.length % 4 === 0) console.log(JSON.stringify({ event: "scheduled", label, completed, updates: updates.length }));
  }
  return { updates, audit };
}

const results = [];
for (const row of rows) {
  const window = row.window, started = performance.now();
  const saved = JSON.parse(fs.readFileSync(path.join(source, `${window.id}-model.json`), "utf8")) as {
    policy: SerializedEventPolicy; selectionPolicy?: SerializedEventPolicy; trainEnd: number; selectionTrainingEnd: number;
    policyCalibrationStart: number; calibrationEnd: number;
    selectionExcludedWindows: Array<{ id: string; startTime: number; endTime: number }>; };
  if (window.id.startsWith("fit-") || !saved.selectionPolicy || saved.calibrationEnd > window.startTime || saved.trainEnd > window.startTime
    || saved.selectionTrainingEnd > saved.policyCalibrationStart) throw new Error("Invalid source selection boundary");
  const selected = JSON.parse(fs.readFileSync(path.join(forecast, `${window.id}-selection.json`), "utf8")).selected as Omit<RecentEventOptions, "after"> | null;
  const c = loadEventCandles(saved.policyCalibrationStart - 2 * DAY, window.endTime + DAY); second?.attach(c);
  const prior = restoreEventPolicy(saved.selectionPolicy), final = restoreEventPolicy(saved.policy);
  const ranges = eventCalibrationRanges(saved.policyCalibrationStart, saved.calibrationEnd, saved.selectionExcludedWindows);
  const calibrationPlans = ranges.map((range, i) => schedule(c, prior, range, saved.selectionTrainingEnd, selected, `${window.id}/calibration/${i}`));
  const calibration = prior.tables.map(t => {
    const metrics = ranges.map((range, i) => replayEventPolicy(c, prior, range.startTime, range.endTime, t.depth,
      { updates: calibrationPlans[i].updates }));
    const logGrowth = metrics.reduce((s, m) => s + m.logGrowth, 0), drawdown = Math.max(0, ...metrics.map(m => m.maxDrawdownPct)) / 100;
    return { depth: t.depth, logGrowth, drawdown, score: logGrowth - sourceConfig.riskPenalty * drawdown,
      trades: metrics.reduce((s, m) => s + m.trades, 0) };
  }).sort((a, b) => b.score - a.score);
  const chosen = calibration[0], cash = !!row.insufficientCalibration || !ranges.length || chosen.score <= 0;
  fs.writeFileSync(path.join(output, `${window.id}-selection.json`), JSON.stringify({ selected, chosenDepth: chosen.depth, cash, calibration,
    scheduledUpdates: calibrationPlans.map(p => p.audit) }, null, 2));
  const plan = schedule(c, final, window, saved.trainEnd, selected, `${window.id}/test`);
  const { trace, ...test } = replayEventPolicy(c, final, window.startTime, window.endTime, chosen.depth, { updates: plan.updates, cash, trace: true });
  fs.writeFileSync(path.join(output, `${window.id}-trades.json`), JSON.stringify(trace));
  fs.writeFileSync(path.join(output, `${window.id}-updates.json`), JSON.stringify(plan.audit, null, 2));
  const result = { window, selected, refresh, chosenDepth: chosen.depth, cash, calibration, test,
    updates: plan.updates.length, elapsedSec: (performance.now() - started) / 1000 };
  results.push(result); fs.writeFileSync(path.join(output, `${window.id}.json`), JSON.stringify(result, null, 2));
  fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify(results, null, 2));
  console.log(JSON.stringify({ event: "updated-policy", ...result, calibration: undefined }));
}
