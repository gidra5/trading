/** Cheap sign-only comparison on the same purged event targets as the policy. */
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { eventLeaf, eventMoveLabel, type EventDistribution, type MoveSample } from "../packages/bot-algo/src/event-distribution.js";
import { eventSignMass, predictEventSign, trainEventSign, type EventSignHead } from "../packages/bot-algo/src/event-sign.js";
import { EVENT_FAST_VOLATILITY_INPUTS, EventSizeGateCalibration, eventFastVolatilityFeatures, eventSizeSignGroup, mixEventSizeSigns, predictEventSizeSigns, trainEventSizeGate, trainEventSizeSign,
  withEventSizeGate, type EventSizeGateOptions, type EventSizeSignHead } from "../packages/bot-algo/src/event-size-sign.js";
import type { SerializedEventPolicy } from "../packages/bot-algo/src/event-log-policy.js";
import { EVENT_COMPLETED_HISTORY_INPUTS, EventCompletedHistory } from "../packages/bot-algo/src/event-completed-history.js";
import { loadEventCandles, makeSamples } from "./research-event-policy.js";

const root = path.resolve(__dirname, ".."), DAY = 86_400_000;
const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
const sourceId = arg("source"), outputId = arg("output");
if (!sourceId || !outputId) throw new Error("Specify source and new output directory");
const source = path.resolve(root, "data/benchmarks", sourceId), output = path.resolve(root, "data/benchmarks", outputId);
if (fs.existsSync(output)) throw new Error("Choose a new output directory");
const sourceBytes = fs.readFileSync(path.join(source, "config.json")), config = JSON.parse(sourceBytes.toString());
if (config.contract !== "causal-event-tree-bellman-v1" || config.sampling !== "chain" || !config.refitLatest
  || config.invertAugment || config.calibrateMean || config.onlineScale || config.secondBasisFingerprint
  || config.honestyFraction || config.learner !== "tree") throw new Error("Unsupported source contract for sign screen");
type Range = { id: string; startTime: number; endTime: number };
const rows = JSON.parse(fs.readFileSync(path.join(source, "summary.json"), "utf8")) as Array<{
  window: Range; selectionFitSamples: number; fitSamples: number;
}>;
if (config.windows.some((id: string) => !rows.some(r => r.window.id === id))) throw new Error("Source is incomplete");
const hash = createHash("sha256").update(sourceBytes);
const saved = rows.map(r => {
  const bytes = fs.readFileSync(path.join(source, `${r.window.id}-model.json`)); hash.update(bytes);
  return JSON.parse(bytes.toString()) as { policy: SerializedEventPolicy; selectionPolicy: SerializedEventPolicy;
    trainStart: number; trainEnd: number; selectionTrainingEnd: number; policyCalibrationStart: number; calibrationEnd: number };
});
type Setting = { penalty: number; blend: number; quantile?: number; fastVolatility?: boolean; eventHistory?: boolean; historyGateOnly?: boolean; gate?: EventSizeGateOptions } | null;
type Head = EventSignHead | EventSizeSignHead;
const sizeRegimes = process.argv.includes("--size-regimes");
const fastVolatility = process.argv.includes("--fast-volatility-head");
const gateHeadId = arg("gate-head-source"), historyHeadId = arg("history-head-source"), sourceHash = hash.digest("hex");
const includeGateHistory = process.argv.includes("--include-gate-history");
if (includeGateHistory && !historyHeadId) throw new Error("Gate-history comparison requires a history-head source");
if (gateHeadId && historyHeadId) throw new Error("Use one head-source comparison at a time");
const gateOptions = [8, 32].flatMap(window => [4, 16].map(strength => ({ window, strength })));
let frozenHeadSource: string | undefined, frozenHeadHash: string | undefined;
let frozenHeads: Array<{ selected: Setting; selectionHead: Head; head: Head; trainEnd: number; selectionTrainingEnd: number }> | undefined;
if (gateHeadId || historyHeadId) {
  if (!sizeRegimes) throw new Error("Head-source comparison requires size regimes");
  frozenHeadSource = path.resolve(root, "data/benchmarks", gateHeadId || historyHeadId);
  const bytes = fs.readFileSync(path.join(frozenHeadSource, "config.json")), parent = JSON.parse(bytes.toString());
  if (parent.contract !== "event-size-sign-forecast-screen-v1" || parent.sourceHash !== sourceHash) throw new Error("Gate head source does not match base models");
  const headHash = createHash("sha256").update(bytes);
  frozenHeads = rows.map(r => {
    const bytes = fs.readFileSync(path.join(frozenHeadSource!, `${r.window.id}-model.json`)); headHash.update(bytes);
    const h = JSON.parse(bytes.toString());
    if (h.selected?.gate || h.selected?.eventHistory) throw new Error("Head-source comparison requires a fixed head without event history or online calibration");
    return h;
  });
  frozenHeadHash = headHash.digest("hex");
}
if (fastVolatility && !sizeRegimes) throw new Error("Fast volatility screen currently requires size regimes");
const settings: Setting[] = [null, ...[0.01, 0.1, 1].flatMap(penalty => [0.5, 1].flatMap(blend =>
  sizeRegimes ? [0.5, 0.75].flatMap(quantile => fastVolatility
    ? [{ penalty, blend, quantile }, { penalty, blend, quantile, fastVolatility: true }] : [{ penalty, blend, quantile }])
    : [{ penalty, blend }]))];
const key = (s: NonNullable<Setting>) => `${s.penalty}:${s.quantile ?? "sign"}:${!!s.fastVolatility}:${!!s.eventHistory}:${!!s.historyGateOnly}`;
type ExtraFeatures = (sample: MoveSample, setting: NonNullable<Setting>) => number[];
const train = (samples: MoveSample[], s: NonNullable<Setting>, extras: ExtraFeatures, base?: Head): Head => {
  const input = s.fastVolatility || s.eventHistory ? samples.map(row => ({ ...row, features: [...row.features, ...extras(row, s)] })) : samples;
  if (s.historyGateOnly) {
    if (!base || !("gate" in base)) throw new Error("Gate-only history requires frozen size/sign heads");
    return trainEventSizeGate(input, base, s.penalty);
  }
  return s.quantile === undefined ? trainEventSign(input, s.penalty) : trainEventSizeSign(input, s.penalty, s.quantile);
};
fs.mkdirSync(output, { recursive: true });
fs.writeFileSync(path.join(output, "config.json"), JSON.stringify({ contract: sizeRegimes ? "event-size-sign-forecast-screen-v1" : "event-sign-forecast-screen-v1", source,
  sourceHash, settings: frozenHeads ? undefined : settings, frozenHeadSource, frozenHeadHash,
  headReuse: gateHeadId ? "gate-calibration" : historyHeadId ? "event-history-comparison" : undefined,
  gateOptions: gateHeadId ? gateOptions : undefined, windows: rows.map(r => r.window.id),
  headExtraInputs: fastVolatility || frozenHeads?.some(h => h.selected?.fastVolatility) ? EVENT_FAST_VOLATILITY_INPUTS : [],
  eventHistoryInputs: historyHeadId ? EVENT_COMPLETED_HISTORY_INPUTS : undefined,
  includeGateHistory,
  eventHistory: historyHeadId ? "Last 16 completed endpoint events whose whole target is inside the preceding day; reset at each fit/calibration/test episode or gap; empty initial history; penalties .01/.1/1" : undefined,
  gateUpdating: gateHeadId ? "Rolling penalized log-odds intercept; only completed post-fit event sizes; reset across excluded gaps and final refit; raw forecasts paired with labels" : undefined,
  selection: sizeRegimes ? "minimum joint 15-class NLL on preceding calibration, including unchanged base" : "minimum active-sign cross entropy on preceding calibration, including unchanged base",
  model: sizeRegimes ? "L2 logistic size gate and sign heads conditional on size regime; train-only magnitude quantiles; physical event features"
    : "L2 logistic active-sign head; train-only normalization; physical event features; same event target and fit boundaries as source",
  combination: sizeRegimes ? "P(size | features) P(sign | size, features) times base joint law conditional on sign, size and base state; zero mass unchanged"
    : "P(sign | features) times base joint law conditional on sign and base state; zero mass unchanged",
  caveat: "Forecast experiment on repeatedly inspected research windows; no trading-profit claim" }, null, 2));
const files = ["scripts/screen-event-sign.ts", "scripts/research-event-policy.ts", "packages/bot-algo/src/event-distribution.ts",
  "packages/bot-algo/src/event-sign.ts", "packages/bot-algo/src/event-size-sign.ts", "packages/bot-algo/src/event-completed-history.ts"];
fs.writeFileSync(path.join(output, "sources.json"), JSON.stringify(Object.fromEntries(files.map(f => [f, fs.readFileSync(path.join(root, f), "utf8")]))));

function evaluate(samples: MoveSample[], model: EventDistribution, head: Head | undefined, setting: Setting, extras: ExtraFeatures, times: number[], trace = false) {
  const sizeHead = head && "gate" in head ? head : undefined;
  const group = (r: number) => sizeHead ? eventSizeSignGroup(r, sizeHead.thresholdLogBps) : Math.sign(r) + 1;
  const laws = model.kernels.map(kernel => {
    const mass = eventSignMass(kernel), moments = Array.from({ length: sizeHead ? 5 : 3 }, () =>
      ({ mass: 0, mean: 0, largeDown: 0, classes: new Array<number>(15).fill(0) }));
    for (const a of kernel) {
      const m = moments[group(a.return)]; m.mass += a.probability; m.mean += a.probability * a.return;
      m.classes[eventMoveLabel(a.return, a.duration, model.clock)] += a.probability;
      if (Math.log1p(a.return) * 1e4 <= -model.clock.thresholdBps) m.largeDown += a.probability;
    }
    return { ...mass, moments, mean: moments.reduce((s, m) => s + m.mean, 0) };
  });
  let active = 0, loss = 0, baseLoss = 0, correct = 0, baseCorrect = 0, positive = 0, brier = 0, baseBrier = 0;
  let squared = 0, baseSquared = 0, nll = 0, baseNll = 0;
  let largeDownBrier = 0, baseLargeDownBrier = 0, largeDownCount = 0, unsupported = 0;
  let gate: EventSizeGateCalibration | undefined, previousEnd = -1;
  const predictions: unknown[] = [];
  for (const s of samples) {
    if (setting?.gate && s.start !== previousEnd) gate = new EventSizeGateCalibration(setting.gate, times[s.start] - 1);
    const leaf = eventLeaf(model, s.features), b = laws[leaf];
    const headFeatures = setting && (setting.fastVolatility || setting.eventHistory) ? [...s.features, ...extras(s, setting)] : s.features;
    const baseMass = b.moments.map(m => m.mass), activeMass = b.negative + b.positive;
    let masses = baseMass, headPPositive = b.probability, sizeProbabilities: number[] | undefined;
    let rawGateProbability = 0, gateOffset: number | undefined, gateCount: number | undefined;
    if (sizeHead) {
      sizeProbabilities = predictEventSizeSigns(sizeHead, headFeatures);
      rawGateProbability = sizeProbabilities[2] + sizeProbabilities[3];
      if (gate) {
        const updated = gate.forecast(rawGateProbability, times[s.start]);
        sizeProbabilities = withEventSizeGate(sizeProbabilities, updated.probability);
        gateOffset = updated.offset; gateCount = updated.count;
      }
      headPPositive = sizeProbabilities[1] + sizeProbabilities[3];
      masses = mixEventSizeSigns(baseMass, sizeProbabilities, setting!.blend);
      if (baseMass.slice(0, 4).some(v => !v)) unsupported++;
    } else if (head && setting && b.negative && b.positive) {
      headPPositive = predictEventSign(head as EventSignHead, headFeatures);
      const up = (1 - setting.blend) * b.probability + setting.blend * headPPositive;
      masses = [activeMass * (1 - up), b.zero, activeMass * up];
    }
    const p = activeMass ? (sizeHead ? masses[1] + masses[3] : masses[2]) / activeMass : 0.5;
    const ratios = masses.map((m, i) => baseMass[i] ? m / baseMass[i] : 1);
    const mean = b.moments.reduce((sum, m, i) => sum + m.mean * ratios[i], 0);
    squared += (s.return - mean) ** 2; baseSquared += (s.return - b.mean) ** 2;
    nll -= Math.log(Math.max(1e-12, b.moments.reduce((sum, m, i) => sum + m.classes[s.label] * ratios[i], 0)));
    baseNll -= Math.log(Math.max(1e-12, b.moments.reduce((sum, m) => sum + m.classes[s.label], 0)));
    const largeDown = Number(Math.log1p(s.return) * 1e4 <= -model.clock.thresholdBps);
    const largeDownProbability = b.moments.reduce((sum, m, i) => sum + m.largeDown * ratios[i], 0);
    const baseLargeDownProbability = b.moments.reduce((sum, m) => sum + m.largeDown, 0);
    largeDownBrier += (largeDown - largeDownProbability) ** 2;
    baseLargeDownBrier += (largeDown - baseLargeDownProbability) ** 2; largeDownCount += largeDown;
    if (s.return !== 0) {
      const y = Number(s.return > 0); positive += y; active++;
      loss -= Math.log(Math.max(1e-12, y ? p : 1 - p));
      baseLoss -= Math.log(Math.max(1e-12, y ? b.probability : 1 - b.probability));
      brier += (p - y) ** 2; baseBrier += (b.probability - y) ** 2;
      correct += Number((p >= 0.5) === !!y); baseCorrect += Number((b.probability >= 0.5) === !!y);
    }
    if (trace) predictions.push({ originTime: times[s.start], availableAt: times[s.end], leaf, pPositive: p,
      basePPositive: b.probability, headPPositive, sizeProbabilities, largeDownProbability, baseLargeDownProbability,
      rawGateProbability: sizeHead ? rawGateProbability : undefined, gateOffset, gateCount,
      returnBps: s.return * 1e4, meanBps: mean * 1e4, baseMeanBps: b.mean * 1e4, duration: s.duration });
    if (gate && sizeHead && s.return !== 0) gate.observe({ originTime: times[s.start], availableAt: times[s.end],
      rawProbability: rawGateProbability, large: Math.abs(Math.log1p(s.return)) * 1e4 >= sizeHead.thresholdLogBps }, times[s.end]);
    previousEnd = s.end;
  }
  if (!active) throw new Error("No active targets in sign evaluation");
  return { setting, count: samples.length, active, positiveRate: positive / active,
    signLoss: loss / active, baseSignLoss: baseLoss / active, signGainBits: (baseLoss - loss) / active / Math.LN2,
    accuracy: correct / active, baseAccuracy: baseCorrect / active, brier: brier / active, baseBrier: baseBrier / active,
    mse: squared / samples.length, baseMse: baseSquared / samples.length, mseSkill: 1 - squared / baseSquared,
    nll: nll / samples.length, baseNll: baseNll / samples.length, unsupported,
    largeDownCount, largeDownBrier: largeDownBrier / samples.length, baseLargeDownBrier: baseLargeDownBrier / samples.length, predictions };
}

const results = [];
for (let i = 0; i < rows.length; i++) {
  const { window } = rows[i], s = saved[i], started = performance.now();
  if (window.id.startsWith("fit-") || !s.selectionPolicy || s.trainEnd > window.startTime || s.calibrationEnd > window.startTime
    || s.selectionTrainingEnd >= s.policyCalibrationStart) throw new Error("Invalid source boundaries");
  const calibrationStart = window.startTime - config.calibrationDays * DAY, trainStart = calibrationStart - config.trainDays * DAY;
  if (calibrationStart !== s.policyCalibrationStart) throw new Error("Sign screen requires whole source calibration");
  const candles = loadEventCandles(trainStart - 2 * DAY, window.endTime + DAY), times = candles.map(c => c.openTime + 60_000);
  const extraCache = new Map<number, number[]>();
  const extrasFor = (samples: MoveSample[]): ExtraFeatures => {
    const histories = new Map<number, number[]>();
    if (historyHeadId) {
      let previousEnd = -1, memory: EventCompletedHistory | undefined;
      for (const sample of samples) {
        if (sample.start !== previousEnd) memory = new EventCompletedHistory(times[sample.start]);
        histories.set(sample.start, memory!.features(times[sample.start]));
        memory!.observe({ ...sample, originTime: times[sample.start], availableAt: times[sample.end] }, times[sample.end]);
        previousEnd = sample.end;
      }
    }
    return (sample, setting) => {
      const out: number[] = [];
      if (setting.fastVolatility) {
        if (!extraCache.has(sample.start)) extraCache.set(sample.start, eventFastVolatilityFeatures(candles, sample.start));
        out.push(...extraCache.get(sample.start)!);
      }
      if (setting.eventHistory) {
        const features = histories.get(sample.start);
        if (!features) throw new Error("Missing completed-event feature state");
        out.push(...features);
      }
      return out;
    };
  };
  const excluded: Range[] = config.trainingIsolation === "causal" ? [window] : config.excludedWindows;
  const samples = (start: number, end: number, purge: Range[]) => makeSamples(candles, config.clock, start, end, purge, config.stride, "chain", config.featureNames);
  const fit = samples(trainStart, s.selectionTrainingEnd, excluded), calibration = samples(calibrationStart, s.calibrationEnd, excluded);
  const fitExtras = extrasFor(fit), calibrationExtras = extrasFor(calibration);
  if (fit.length !== rows[i].selectionFitSamples) throw new Error("Original sign-training sample population changed");
  const frozen = frozenHeads?.[i];
  if (frozen && (frozen.trainEnd !== s.trainEnd || frozen.selectionTrainingEnd !== s.selectionTrainingEnd)) throw new Error("Frozen head boundary mismatch");
  const windowSettings: Setting[] = frozen ? frozen.selected
    ? [null, frozen.selected, ...(gateHeadId ? gateOptions.map(gate => ({ ...frozen.selected!, gate }))
      : [0.01, 0.1, 1].flatMap(penalty => includeGateHistory
        ? [{ ...frozen.selected!, penalty, eventHistory: true }, { ...frozen.selected!, penalty, eventHistory: true, historyGateOnly: true }]
        : [{ ...frozen.selected!, penalty, eventHistory: true }]))] : [null] : settings;
  const heads = new Map<string, Head>();
  for (const setting of windowSettings) if (setting && !heads.has(key(setting))) heads.set(key(setting),
    frozen && !setting.eventHistory ? frozen.selectionHead : train(fit, setting, fitExtras, frozen?.selectionHead));
  const scores = windowSettings.map(setting => {
    const { predictions: _, ...metrics } = evaluate(calibration, s.selectionPolicy.model, setting ? heads.get(key(setting)) : undefined, setting, calibrationExtras, times);
    return metrics;
  }).sort((a, b) => sizeRegimes ? a.nll - b.nll : a.signLoss - b.signLoss);
  const selected = scores[0].setting;
  // Freeze choice before reading any scored-window target into evaluation.
  fs.writeFileSync(path.join(output, `${window.id}-selection.json`), JSON.stringify({ window, selected, scores }, null, 2));
  const finalFit = samples(s.trainStart, s.trainEnd, excluded);
  if (finalFit.length !== rows[i].fitSamples) throw new Error("Final sign-training sample population changed");
  const finalHead = selected ? frozen && !selected.eventHistory ? frozen.head : train(finalFit, selected, extrasFor(finalFit), frozen?.head) : undefined;
  fs.writeFileSync(path.join(output, `${window.id}-model.json`), JSON.stringify({ selected,
    selectionHead: selected ? heads.get(key(selected)) : undefined, head: finalHead,
    selectionTrainingEnd: s.selectionTrainingEnd, trainEnd: s.trainEnd, finalTargetEnd: times[finalFit.at(-1)!.end] }, null, 2));
  const testSamples = samples(window.startTime, window.endTime, []);
  const test = evaluate(testSamples, s.policy.model, finalHead, selected, extrasFor(testSamples), times, true);
  fs.writeFileSync(path.join(output, `${window.id}-predictions.json`), JSON.stringify(test.predictions));
  const { predictions: _, ...metrics } = test;
  const result = { window, selected, calibration: scores, test: metrics, elapsedSec: (performance.now() - started) / 1000 };
  results.push(result);
  fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify(results, null, 2));
  console.log(JSON.stringify({ event: "sign-screen", ...result, calibration: undefined }));
}
