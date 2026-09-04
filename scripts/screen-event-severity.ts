/** Fit conditional large-move magnitude, retaining the incumbent sign/size
 * probabilities and within-group joint path support. No policy selection here. */
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { eventLeaf, eventMoveLabel, type EventDistribution, type MoveSample } from "../packages/bot-algo/src/event-distribution.js";
import { EventCompletedHistory } from "../packages/bot-algo/src/event-completed-history.js";
import { eventFastVolatilityFeatures, eventSizeSignGroup, eventSizeSignMass, predictEventSizeSigns, reweightEventSizeSigns, type EventSizeSignHead } from "../packages/bot-algo/src/event-size-sign.js";
import { eventSeverityMeans, predictEventSeverity, tiltEventSeverity, trainEventSeverity, type EventSeverityHead } from "../packages/bot-algo/src/event-severity.js";
import { loadEventCandles, makeSamples } from "./research-event-policy.js";

const root = path.resolve(__dirname, ".."), DAY = 86_400_000;
const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
const forecastId = arg("forecast"), outputId = arg("output");
if (!forecastId || !outputId) throw new Error("Specify frozen forecast and new severity screen output");
const forecast = path.resolve(root, "data/benchmarks", forecastId), output = path.resolve(root, "data/benchmarks", outputId);
const read = (dir: string, file: string) => JSON.parse(fs.readFileSync(path.join(dir, file), "utf8"));
if (fs.existsSync(output)) throw new Error("Choose a new output directory");
const fc = read(forecast, "config.json"), source: string = fc.source, sc = read(source, "config.json"), summary = read(source, "summary.json");
if (fc.contract !== "event-size-sign-forecast-screen-v1" || sc.sampling !== "chain") throw new Error("Requires a fixed chained size/sign forecast");
const ids = arg("windows").split(",").filter(Boolean);
if (ids.some(id => !summary.some((s: any) => s.window.id === id))) throw new Error("Unknown window");
const hash = createHash("sha256").update(fs.readFileSync(path.join(source, "config.json")));
for (const s of summary) hash.update(fs.readFileSync(path.join(source, `${s.window.id}-model.json`)));
if (hash.digest("hex") !== fc.sourceHash) throw new Error("Base source changed after head selection");
const headHash = createHash("sha256").update(fs.readFileSync(path.join(forecast, "config.json")));
for (const s of summary) headHash.update(fs.readFileSync(path.join(forecast, `${s.window.id}-model.json`)));
type Setting = { penalty: number; blend: number } | null;
const settings: Setting[] = [null, ...[0.01, 0.1, 1].flatMap(penalty => [0.5, 1].map(blend => ({ penalty, blend })))];
fs.mkdirSync(output, { recursive: true });
fs.writeFileSync(path.join(output, "config.json"), JSON.stringify({ contract: "event-severity-screen-v1", forecast, source,
  sourceHash: fc.sourceHash, headHash: headHash.digest("hex"), settings, windows: ids.length ? ids : summary.map((s: any) => s.window.id),
  model: "Positive excess-magnitude mean, shared log-RV5 slope and candidate-sign contrast; three parameters; within-group probability tilts preserve joint atoms",
  selection: "Preceding calibration conditional large-move magnitude MSE; unchanged incumbent wins ties. Signed-mean MSE, joint NLL and extreme-event Brier are reported separately.",
  caveat: "Forecast screen, not demonstrated trading utility. A later policy comparison must retain existing policies. Realized groups are evaluation labels, never inference inputs." }, null, 2));
const files = ["scripts/screen-event-severity.ts", "scripts/research-event-policy.ts", "packages/bot-algo/src/event-severity.ts",
  "packages/bot-algo/src/event-size-sign.ts", "packages/bot-algo/src/event-completed-history.ts"];
fs.writeFileSync(path.join(output, "sources.json"), JSON.stringify(Object.fromEntries(files.map(f => [f, fs.readFileSync(path.join(root, f), "utf8")]))));
const results = [];
for (const entry of summary) {
  const { window } = entry, id = window.id;
  if (ids.length && !ids.includes(id)) continue;
  const start = performance.now(), s = read(source, `${id}-model.json`), h = read(forecast, `${id}-model.json`);
  if (!h.selected || h.selected.gate || !h.head?.gate || h.trainEnd !== s.trainEnd || h.selectionTrainingEnd !== s.selectionTrainingEnd
    || s.trainEnd > window.startTime || s.calibrationEnd > window.startTime || id.startsWith("fit-")) throw new Error("Invalid severity source or chronology");
  const from = window.startTime - (sc.calibrationDays + sc.trainDays) * DAY;
  const c = loadEventCandles(from - 2 * DAY, window.endTime + DAY), excluded = sc.trainingIsolation === "causal" ? [window] : sc.excludedWindows;
  const samples = (start: number, end: number, exclusions = excluded) => makeSamples(c, sc.clock, start, end, exclusions, sc.stride, "chain", sc.featureNames);
  const prepare = (rows: MoveSample[], head: EventSizeSignHead) => {
    let memory: EventCompletedHistory | undefined, previousEnd = -1;
    return rows.map(row => {
      const at = c[row.start].openTime + 60_000, fast = eventFastVolatilityFeatures(c, row.start);
      if (h.selected.eventHistory && row.start !== previousEnd) memory = new EventCompletedHistory(at);
      const features = [...row.features, ...(h.selected.fastVolatility ? fast : []), ...(memory?.features(at) ?? [])];
      const probabilities = predictEventSizeSigns(head, features);
      memory?.observe({ ...row, originTime: at, availableAt: c[row.end].openTime + 60_000 }, c[row.end].openTime + 60_000);
      previousEnd = row.end;
      return { row, at, probabilities, rv5Bps: Math.expm1(fast[1]), return: row.return };
    });
  };
  const evaluate = (rows: ReturnType<typeof prepare>, model: EventDistribution, head: EventSizeSignHead,
    severity: EventSeverityHead | undefined, setting: Setting, highVolCut: number, trace = false) => {
    let squared = 0, nll = 0, severitySquared = 0, large = 0, highSquared = 0, highLarge = 0, brier = 0, massError = 0, clipped = 0;
    const predictions = [];
    for (const input of rows) {
      const { row, at, rv5Bps } = input, leaf = eventLeaf(model, row.features);
      const original = reweightEventSizeSigns(model.kernels[leaf], head.thresholdLogBps, input.probabilities, h.selected.blend);
      const targets = severity ? predictEventSeverity(severity, rv5Bps) : undefined;
      const kernel = targets && setting ? tiltEventSeverity(original, head.thresholdLogBps, targets, setting.blend) : original;
      const means = eventSeverityMeans(kernel, head.thresholdLogBps), originalMeans = eventSeverityMeans(original, head.thresholdLogBps);
      if (targets && setting) for (let g = 0; g < 2; g++) if (means[g] !== null && originalMeans[g] !== null
        && Math.abs(means[g]! - ((1 - setting.blend) * originalMeans[g]! + setting.blend * targets[g])) > 1e-5) clipped++;
      const mass = eventSizeSignMass(kernel, head.thresholdLogBps), originalMass = eventSizeSignMass(original, head.thresholdLogBps);
      massError = Math.max(massError, ...mass.map((v, g) => Math.abs(v - originalMass[g])));
      const expected = kernel.reduce((sum, a) => sum + a.probability * a.return, 0);
      squared += (row.return - expected) ** 2;
      nll -= Math.log(Math.max(1e-12, kernel.reduce((sum, a) => sum + a.probability * Number(eventMoveLabel(a.return, a.duration, model.clock) === row.label), 0)));
      const extreme = Number(Math.abs(Math.log1p(row.return)) * 1e4 >= head.thresholdLogBps * 2);
      const extremeProbability = kernel.reduce((sum, a) => sum + a.probability * Number(Math.abs(Math.log1p(a.return)) * 1e4 >= head.thresholdLogBps * 2), 0);
      brier += (extreme - extremeProbability) ** 2;
      const group = eventSizeSignGroup(row.return, head.thresholdLogBps);
      if (group === 2 || group === 3) {
        if (means[group - 2] === null) throw new Error("Unobserved severity group cannot be scored as zero error");
        const error = (Math.abs(Math.log1p(row.return)) * 1e4 - means[group - 2]!) ** 2;
        severitySquared += error; large++;
        if (rv5Bps > highVolCut) { highSquared += error; highLarge++; }
      }
      if (trace) predictions.push({ time: at, availableAt: c[row.end].openTime + 60_000, leaf, rv5Bps, group,
        realizedReturnBps: row.return * 1e4, expectedReturnBps: expected * 1e4, targets, achievedSeverity: means, extremeProbability, sizeSignProbabilities: input.probabilities });
    }
    if (!large || !rows.length) throw new Error("No severity evaluation targets");
    return { setting, samples: rows.length, large, severityMse: severitySquared / large,
      highVolLarge: highLarge, highVolSeverityMse: highLarge ? highSquared / highLarge : null,
      signedMeanMse: squared / rows.length, nll: nll / rows.length, extremeBrier: brier / rows.length, maximumGroupMassError: massError, clipped, predictions };
  };
  const trainingSamples = samples(from, s.selectionTrainingEnd), finalSamples = samples(s.trainStart, s.trainEnd);
  if (trainingSamples.length !== entry.selectionFitSamples || finalSamples.length !== entry.fitSamples) throw new Error("Severity fit population differs from incumbent");
  const fit = prepare(trainingSamples, h.selectionHead), validation = prepare(samples(s.policyCalibrationStart, s.calibrationEnd), h.selectionHead);
  const quantile90 = (rows: typeof fit) => rows.map(r => r.rv5Bps).sort((a, b) => a - b)[Math.floor(rows.length * 0.9)];
  const selectionHeads = new Map([0.01, 0.1, 1].map(penalty => [penalty, trainEventSeverity(fit, h.selectionHead.thresholdLogBps, penalty)]));
  const scores = settings.map(setting => {
    const { predictions: _, ...metrics } = evaluate(validation, s.selectionPolicy.model, h.selectionHead,
      setting ? selectionHeads.get(setting.penalty) : undefined, setting, quantile90(fit));
    return metrics;
  }).sort((a, b) => a.severityMse - b.severityMse);
  const selected = scores[0].setting;
  fs.writeFileSync(path.join(output, `${id}-selection.json`), JSON.stringify({ window, selected, scores }, null, 2));
  const final = prepare(finalSamples, h.head);
  const severityHead = selected ? trainEventSeverity(final, h.head.thresholdLogBps, selected.penalty) : undefined;
  fs.writeFileSync(path.join(output, `${id}-model.json`), JSON.stringify({ selected, headSetting: h.selected, head: h.head, selectionHead: h.selectionHead,
    severityHead, selectionSeverityHead: selected ? selectionHeads.get(selected.penalty) : undefined,
    trainEnd: s.trainEnd, selectionTrainingEnd: s.selectionTrainingEnd }, null, 2));
  const testRows = prepare(samples(window.startTime, window.endTime, []), h.head);
  const test = evaluate(testRows, s.policy.model, h.head, severityHead, selected, quantile90(final), true);
  const { predictions: _baseTrace, ...baseline } = evaluate(testRows, s.policy.model, h.head, undefined, null, quantile90(final));
  fs.writeFileSync(path.join(output, `${id}-predictions.json`), JSON.stringify(test.predictions));
  const { predictions: _, ...metrics } = test;
  const result = { window, selected, calibration: scores, test: metrics, baseline,
    severitySkill: 1 - test.severityMse / baseline.severityMse, signedMeanSkill: 1 - test.signedMeanMse / baseline.signedMeanMse,
    elapsedSec: (performance.now() - start) / 1000 };
  results.push(result); fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify(results, null, 2));
  console.log(JSON.stringify({ ...result, calibration: undefined }));
}
