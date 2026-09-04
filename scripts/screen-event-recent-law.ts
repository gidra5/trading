/** Prequential forecast screen. No trading-performance claims or test-set tuning. */
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { EventRecentLaw, type CompletedEventMove, type RecentEventOptions } from "../packages/bot-algo/src/event-recent-law.js";
import { eventLeaf, eventMoveLabel, type EventDistribution, type EventCandle } from "../packages/bot-algo/src/event-distribution.js";
import { type SerializedEventPolicy } from "../packages/bot-algo/src/event-log-policy.js";
import { eventCalibrationRanges, loadEventCandles, makeSamples } from "./research-event-policy.js";
import { EventSecondBasis } from "./event-second-basis.js";

const root = path.resolve(__dirname, ".."), DAY = 86_400_000;
const arg = (key: string, fallback = "") => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? fallback : process.argv[i + 1]; };
const sourceId = arg("source"), outputId = arg("output");
if (!sourceId || !outputId) throw new Error("Specify source and new output directory");
const source = path.resolve(root, "data/benchmarks", sourceId), output = path.resolve(root, "data/benchmarks", outputId);
if (fs.existsSync(output)) throw new Error("Choose a new output directory");
const sourceConfig = JSON.parse(fs.readFileSync(path.join(source, "config.json"), "utf8"));
if (sourceConfig.contract !== "causal-event-tree-bellman-v1" || sourceConfig.sampling !== "chain")
  throw new Error("Recent-law screen requires an endpoint-chain event research source");
const rows = JSON.parse(fs.readFileSync(path.join(source, "summary.json"), "utf8")) as Array<{
  window: { id: string; startTime: number; endTime: number };
}>;
if (sourceConfig.windows.some((id: string) => !rows.some(r => r.window.id === id))) throw new Error("Source is incomplete");
const second = sourceConfig.secondBasisFingerprint ? new EventSecondBasis(path.join(root, "data/runtime-cache/global-feature-basis")) : undefined;
if (second && second.fingerprint !== sourceConfig.secondBasisFingerprint) throw new Error("Feature source changed");
const shared = process.argv.includes("--shared-states");
const scopes: NonNullable<RecentEventOptions["scope"]>[] = shared ? ["leaf", "parent", "direction"] : ["leaf"];
const settings = [null, ...scopes.flatMap(scope => [64, 256].flatMap(window => [8, 32].map(prior => ({ window, prior, scope }))))];
const hash = createHash("sha256").update(fs.readFileSync(path.join(source, "config.json")));
const saved = rows.map(r => {
  const bytes = fs.readFileSync(path.join(source, `${r.window.id}-model.json`)); hash.update(bytes);
  return JSON.parse(bytes.toString()) as { policy: SerializedEventPolicy; selectionPolicy?: SerializedEventPolicy;
    trainEnd: number; selectionTrainingEnd: number; policyCalibrationStart: number; calibrationEnd: number;
    selectionExcludedWindows: Array<{ id: string; startTime: number; endTime: number }> };
});
fs.mkdirSync(output, { recursive: true });
fs.writeFileSync(path.join(output, "config.json"), JSON.stringify({ source, sourceHash: hash.digest("hex"), settings,
  selection: "minimum prequential mean MSE on preceding calibration, including unchanged base",
  contract: "event-recent-law-forecast-screen-v1", caveat: "Repeatedly inspected research windows. Forecast skill is not trading profit.",
  updating: "Only completed post-fit endpoint events; history resets across excluded gaps and at final refit; whole physical joint-law mixture",
  windows: rows.map(r => r.window.id) }, null, 2));
const files = ["scripts/screen-event-recent-law.ts", "scripts/research-event-policy.ts", "scripts/event-second-basis.ts",
  "packages/bot-algo/src/event-recent-law.ts", "packages/bot-algo/src/event-distribution.ts", "packages/bot-algo/src/event-run-model.ts"];
fs.writeFileSync(path.join(output, "sources.json"), JSON.stringify(Object.fromEntries(files.map(f => [f, fs.readFileSync(path.join(root, f), "utf8")]))));

function evaluate(c: EventCandle[], model: EventDistribution, ranges: Array<{ startTime: number; endTime: number }>,
  after: number, setting: (typeof settings)[number], trace = false) {
  let count = 0, squared = 0, baseSquared = 0, nll = 0, baseNll = 0, recentCount = 0, localCount = 0;
  const predictions: unknown[] = [];
  const means = model.kernels.map(k => k.reduce((s, a) => s + a.probability * a.return, 0));
  const classes = model.kernels.map(k => {
    const p = new Array<number>(15).fill(0);
    for (const a of k) p[eventMoveLabel(a.return, a.duration, model.clock)] += a.probability;
    return p;
  });
  for (const range of ranges) {
    const online = setting ? new EventRecentLaw(model, { ...setting, after }) : undefined;
    const samples = makeSamples(c, model.clock, range.startTime, range.endTime, [], 1, "chain", model.featureNames);
    for (const s of samples) {
      const originTime = c[s.start].openTime + 60_000, availableAt = c[s.end].openTime + 60_000;
      const leaf = eventLeaf(model, s.features), prediction = online?.forecast(s.features, originTime);
      const mean = prediction?.mean ?? means[leaf], p = prediction?.classes ?? classes[leaf];
      squared += (s.return - mean) ** 2; baseSquared += (s.return - means[leaf]) ** 2;
      nll -= Math.log(Math.max(1e-12, p[s.label])); baseNll -= Math.log(Math.max(1e-12, classes[leaf][s.label]));
      recentCount += prediction?.recentCount ?? 0; localCount += prediction?.localCount ?? 0; count++;
      if (trace) predictions.push({ originTime, availableAt, leaf, meanBps: mean * 1e4, baseMeanBps: means[leaf] * 1e4,
        returnBps: s.return * 1e4, recentCount: prediction?.recentCount ?? 0, localCount: prediction?.localCount ?? 0 });
      const completed: CompletedEventMove = { ...s, originTime, availableAt };
      online?.observe(completed, availableAt);
    }
  }
  if (!count) throw new Error("No complete events in forecast screen");
  return { setting, count, mse: squared / count, baselineMse: baseSquared / count,
    mseSkillVsBase: baseSquared ? 1 - squared / baseSquared : 0,
    nll: nll / count, baselineNll: baseNll / count, meanRecentCount: recentCount / count, meanLocalCount: localCount / count, predictions };
}

const results = [];
for (let i = 0; i < rows.length; i++) {
  const window = rows[i].window, s = saved[i], started = performance.now();
  if (window.id.startsWith("fit-") || !s.selectionPolicy || s.calibrationEnd > window.startTime || s.trainEnd > window.startTime
    || s.selectionTrainingEnd > s.policyCalibrationStart) throw new Error("Invalid source boundaries or missing selection model");
  const c = loadEventCandles(s.policyCalibrationStart - 2 * DAY, window.endTime + DAY); second?.attach(c);
  const ranges = eventCalibrationRanges(s.policyCalibrationStart, s.calibrationEnd, s.selectionExcludedWindows);
  const calibration = settings.map(setting => evaluate(c, s.selectionPolicy!.model, ranges, s.selectionTrainingEnd, setting))
    .sort((a, b) => a.mse - b.mse);
  const selected = calibration[0].setting;
  fs.writeFileSync(path.join(output, `${window.id}-selection.json`), JSON.stringify({ window, selected, calibration }, null, 2));
  const test = evaluate(c, s.policy.model, [window], s.trainEnd, selected, true);
  fs.writeFileSync(path.join(output, `${window.id}-predictions.json`), JSON.stringify(test.predictions));
  const { predictions: _, ...metrics } = test;
  const result = { window, selected, calibration, test: metrics, elapsedSec: (performance.now() - started) / 1000 };
  results.push(result); fs.writeFileSync(path.join(output, `${window.id}.json`), JSON.stringify(result, null, 2));
  fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify(results, null, 2));
  console.log(JSON.stringify({ event: "forecast-screen", ...result, calibration: undefined }));
}
