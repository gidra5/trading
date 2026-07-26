import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { gunzipSync } from "node:zlib";
import {
  conditionalFourSegmentExposureProbabilities,
} from "../packages/bot-algo/src/conditional-exposure-distribution.js";
import {
  DEFAULT_DIRECT_INDICATOR_PARAMETERS,
  DIRECT_INDICATOR_PARAMETER_BOUNDS,
  decodeDirectIndicatorConditionalParameters,
  type DirectIndicatorPredictorParameters,
} from "../packages/bot-algo/src/direct-indicator-conditional-predictor.js";
import {
  predictHandcraftedIndicatorRegret,
  prepareHandcraftedIndicatorStates,
  type HandcraftedIndicatorPredictorParameters,
} from "../packages/bot-algo/src/handcrafted-indicator-predictor.js";
import {
  conditionalExposureProbabilities,
  exposureValueOracleProbabilities,
  prepareExposureValueOracle,
} from "../packages/bot-algo/src/exposure-value-distillation.js";
import { HANDCRAFTED_PREDICTOR_PRESETS } from "../apps/server/src/handcrafted-predictor-presets.js";

const DAY_MS = 86_400_000;
const INTERVAL_MS = 60_000;
const HOLDING_STEPS = 1;
const HORIZON_STEPS = 60;
const GRID_SIZE = 31;
const TEMPERATURE = 0.01;
const FRICTION = 0.00175;
const SAMPLES_PER_WINDOW = 8;
const RANDOM_CANDIDATES = 144;
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const historyRoot = path.join(repoRoot, "data/historical/spot-btcusdt/btcusdt/1m");
const actionGrid = Float64Array.from(
  { length: GRID_SIZE },
  (_, index) => -100 + index * 200 / (GRID_SIZE - 1),
);
const execution = {
  friction: FRICTION,
  minExposure: -100,
  maxExposure: 100,
  maxEffectiveExposure: 250,
  quoteBorrowRate: 0,
  assetBorrowRate: 0,
};
const predictionOptions = {
  intervalMs: INTERVAL_MS,
  holdingPeriodSteps: HOLDING_STEPS,
  valueHorizonSteps: HORIZON_STEPS,
  temperature: TEMPERATURE,
  execution,
};

interface CalibrationSample {
  baseOracleProbabilities: Float64Array;
  zeroCurrentOracleProbabilities: Float64Array;
}

interface CalibrationWindow {
  id: string;
  prices: Float64Array;
  sampleIndices: number[];
  currentGrid: Float64Array;
  samples: CalibrationSample[];
}

interface ScoredParameters {
  parameters: DirectIndicatorPredictorParameters;
  loss: number;
  crossEntropy: number;
}

const windows = loadCalibrationWindows();
const sampleCount = windows.reduce((sum, item) => sum + item.sampleIndices.length, 0);
process.stdout.write(`Loaded ${windows.length} inspector windows and ${sampleCount} causal samples.\n`);
const baseline = score(DEFAULT_DIRECT_INDICATOR_PARAMETERS);
process.stdout.write(`direct default regret ${baseline.loss.toFixed(8)} · conditional CE ${baseline.crossEntropy.toFixed(8)}\n`);
const candidates = initialCandidates();
let best: ScoredParameters | undefined;
for (let index = 0; index < candidates.length; index += 1) {
  const scored = score(candidates[index]!);
  if (!best || scored.loss < best.loss) best = scored;
  if ((index + 1) % 12 === 0 || index === candidates.length - 1) {
    process.stdout.write(`random ${index + 1}/${candidates.length} · best regret ${best.loss.toFixed(8)}\n`);
  }
}
for (let round = 0; round < 4; round += 1) {
  for (const candidate of coordinateRefinements(best!.parameters, round)) {
    const scored = score(candidate);
    if (scored.loss < best!.loss) best = scored;
  }
  process.stdout.write(`refine ${round + 1}/4 · best regret ${best!.loss.toFixed(8)}\n`);
}
const localFits = windows.map((calibrationWindow, windowIndex) => {
  let localBest = score(best!.parameters, [calibrationWindow]);
  for (const candidate of candidates) {
    const scored = score(candidate, [calibrationWindow]);
    if (scored.loss < localBest.loss) localBest = scored;
  }
  for (let round = 0; round < 4; round += 1) {
    for (const candidate of coordinateRefinements(localBest.parameters, round)) {
      const scored = score(candidate, [calibrationWindow]);
      if (scored.loss < localBest.loss) localBest = scored;
    }
  }
  process.stdout.write(
    `local ${windowIndex + 1}/${windows.length} ${calibrationWindow.id} · regret ${localBest.loss.toFixed(8)} · CE ${localBest.crossEntropy.toFixed(8)}\n`,
  );
  return { windowId: calibrationWindow.id, ...localBest };
});
const handcraftedGlobal = HANDCRAFTED_PREDICTOR_PRESETS.find((preset) =>
  preset.model === "handcrafted" && preset.scope === "global");
const handcraftedComparison = handcraftedGlobal
  ? scoreHandcrafted(handcraftedGlobal.parameters as HandcraftedIndicatorPredictorParameters)
  : null;
const generatedAt = new Date().toISOString();
const artifact = {
  generatedAt,
  objective: "mean oracle conditional regret of predicted optimal exposure at current exposure zero",
  diagnostic: "uniform-current conditional cross-entropy",
  intervalMs: INTERVAL_MS,
  holdingPeriodMs: HOLDING_STEPS * INTERVAL_MS,
  valueHorizonMs: HORIZON_STEPS * INTERVAL_MS,
  gridSize: GRID_SIZE,
  windowCount: windows.length,
  sampleCount,
  global: {
    loss: best!.loss,
    diagnosticCrossEntropy: best!.crossEntropy,
    parameters: best!.parameters,
  },
  comparison: {
    directDefault: { loss: baseline.loss, diagnosticCrossEntropy: baseline.crossEntropy },
    previousHandcraftedGlobal: handcraftedComparison,
  },
  windows: localFits.map((fit) => ({
    windowId: fit.windowId,
    loss: fit.loss,
    diagnosticCrossEntropy: fit.crossEntropy,
    parameters: fit.parameters,
  })),
};
process.stdout.write(`${JSON.stringify(artifact, null, 2)}\n`);
if (process.argv.includes("--write-presets")) writePresets(best!, localFits, generatedAt);

function score(
  parameters: DirectIndicatorPredictorParameters,
  targetWindows: CalibrationWindow[] = windows,
): ScoredParameters {
  let loss = 0;
  let crossEntropy = 0;
  let count = 0;
  for (const calibrationWindow of targetWindows) {
    const states = prepareHandcraftedIndicatorStates(
      calibrationWindow.prices,
      INTERVAL_MS,
      parameters,
    );
    for (let sample = 0; sample < calibrationWindow.sampleIndices.length; sample += 1) {
      const decoded = decodeDirectIndicatorConditionalParameters(
        actionGrid,
        calibrationWindow.currentGrid,
        states[calibrationWindow.sampleIndices[sample]!]!,
        parameters,
        predictionOptions,
      );
      const zeroPrediction = conditionalFourSegmentExposureProbabilities(
        actionGrid,
        0,
        decoded.parameters,
      );
      const target = calibrationWindow.samples[sample]!.zeroCurrentOracleProbabilities;
      loss += conditionalDecisionRegret(zeroPrediction, target);
      crossEntropy += conditionalCrossEntropy(
        decoded.parameters,
        calibrationWindow.currentGrid,
        calibrationWindow.samples[sample]!.baseOracleProbabilities,
      );
      count += 1;
    }
  }
  return {
    parameters,
    loss: loss / Math.max(1, count),
    crossEntropy: crossEntropy / Math.max(1, count),
  };
}

function scoreHandcrafted(parameters: HandcraftedIndicatorPredictorParameters): {
  loss: number;
  diagnosticCrossEntropy: number;
} {
  let loss = 0;
  let crossEntropy = 0;
  let count = 0;
  const targetRow = new Float64Array(actionGrid.length);
  const predictedRow = new Float64Array(actionGrid.length);
  for (const calibrationWindow of windows) {
    const states = prepareHandcraftedIndicatorStates(
      calibrationWindow.prices,
      INTERVAL_MS,
      parameters,
    );
    for (let sample = 0; sample < calibrationWindow.sampleIndices.length; sample += 1) {
      const prediction = predictHandcraftedIndicatorRegret(
        actionGrid,
        states[calibrationWindow.sampleIndices[sample]!]!,
        parameters,
        predictionOptions,
      );
      const target = calibrationWindow.samples[sample]!;
      conditionalExposureProbabilities(
        prediction.probabilities,
        actionGrid,
        0,
        FRICTION,
        1 / TEMPERATURE,
        predictedRow,
      );
      loss += conditionalDecisionRegret(predictedRow, target.zeroCurrentOracleProbabilities);
      for (const current of calibrationWindow.currentGrid) {
        conditionalExposureProbabilities(
          target.baseOracleProbabilities,
          actionGrid,
          current,
          FRICTION,
          1 / TEMPERATURE,
          targetRow,
        );
        conditionalExposureProbabilities(
          prediction.probabilities,
          actionGrid,
          current,
          FRICTION,
          1 / TEMPERATURE,
          predictedRow,
        );
        crossEntropy += rowCrossEntropy(targetRow, predictedRow)
          / calibrationWindow.currentGrid.length;
      }
      count += 1;
    }
  }
  return {
    loss: loss / Math.max(1, count),
    diagnosticCrossEntropy: crossEntropy / Math.max(1, count),
  };
}

function conditionalCrossEntropy(
  parameters: ReturnType<typeof decodeDirectIndicatorConditionalParameters>["parameters"],
  currentGrid: Float64Array,
  baseOracleProbabilities: Float64Array,
): number {
  const target = new Float64Array(actionGrid.length);
  const predicted = new Float64Array(actionGrid.length);
  let result = 0;
  for (const current of currentGrid) {
    conditionalExposureProbabilities(
      baseOracleProbabilities,
      actionGrid,
      current,
      FRICTION,
      1 / TEMPERATURE,
      target,
    );
    conditionalFourSegmentExposureProbabilities(actionGrid, current, parameters, predicted);
    result += rowCrossEntropy(target, predicted) / currentGrid.length;
  }
  return result;
}

function rowCrossEntropy(target: ArrayLike<number>, predicted: ArrayLike<number>): number {
  let result = 0;
  for (let index = 0; index < target.length; index += 1) {
    if (target[index]! > 0) {
      result -= target[index]! * Math.log(Math.max(Number.MIN_VALUE, predicted[index]!));
    }
  }
  return result;
}

function conditionalDecisionRegret(
  prediction: ArrayLike<number>,
  target: ArrayLike<number>,
): number {
  let predictedIndex = 0;
  let maximumTarget = 0;
  for (let index = 0; index < target.length; index += 1) {
    if (prediction[index]! > prediction[predictedIndex]!) predictedIndex = index;
    maximumTarget = Math.max(maximumTarget, target[index]!);
  }
  return -TEMPERATURE * Math.log(
    Math.max(Number.MIN_VALUE, target[predictedIndex]!)
      / Math.max(Number.MIN_VALUE, maximumTarget),
  );
}

function loadCalibrationWindows(): CalibrationWindow[] {
  const source = fs.readFileSync(path.join(repoRoot, "apps/server/src/kama-inspector.ts"), "utf8");
  const matches = [...source.matchAll(/window\("([^"]+)",\s*"[^"]+",\s*"[^"]+",\s*"(\d{4}-\d{2}-\d{2})",\s*"(\d{4}-\d{2}-\d{2})"/g)];
  const dailyCache = new Map<number, Array<{ openTime: number; close: number }>>();
  return matches.map((match) => {
    const startTime = Date.parse(`${match[2]}T00:00:00.000Z`);
    const endTime = Date.parse(`${match[3]}T00:00:00.000Z`) + DAY_MS;
    const loadStart = startTime - 3 * DAY_MS;
    const loadEnd = endTime + HORIZON_STEPS * INTERVAL_MS;
    const candles: Array<{ openTime: number; close: number }> = [];
    for (let day = utcDay(loadStart); day < loadEnd; day += DAY_MS) {
      let values = dailyCache.get(day);
      if (!values) {
        values = readDay(day);
        dailyCache.set(day, values);
      }
      candles.push(...values);
    }
    candles.sort((left, right) => left.openTime - right.openTime);
    const scoreStartIndex = lowerBound(candles, startTime);
    const scoreEndIndex = lowerBound(candles, endTime);
    const usableEnd = Math.max(scoreStartIndex, scoreEndIndex - HORIZON_STEPS - 1);
    const sampleIndices = evenIndices(scoreStartIndex, usableEnd, SAMPLES_PER_WINDOW);
    const prices = Float64Array.from(candles, (candle) => candle.close);
    let currentGrid: Float64Array | undefined;
    const samples = sampleIndices.map((index) => {
      const future = prices.slice(index, index + HORIZON_STEPS + 1);
      const oracle = prepareExposureValueOracle(future, {
        scoreStartIndex: 0,
        holdingPeriodSteps: HOLDING_STEPS,
        valueHorizonSteps: HORIZON_STEPS,
        friction: FRICTION,
        gridSize: GRID_SIZE,
        minExposure: -100,
        maxExposure: 100,
        maxEffectiveExposure: 250,
        temperature: TEMPERATURE,
        includeProbabilities: true,
      });
      currentGrid ??= Float64Array.from(oracle.currentGrid);
      const baseOracleProbabilities = Float64Array.from(exposureValueOracleProbabilities(oracle, 0));
      const zeroCurrentOracleProbabilities = conditionalExposureProbabilities(
        baseOracleProbabilities,
        oracle.grid,
        0,
        FRICTION,
        1 / TEMPERATURE,
      );
      return { baseOracleProbabilities, zeroCurrentOracleProbabilities };
    });
    if (!currentGrid) throw new Error(`Calibration window ${match[1]} has no causal samples.`);
    return { id: match[1]!, prices, sampleIndices, currentGrid, samples };
  });
}

function initialCandidates(): DirectIndicatorPredictorParameters[] {
  const result = [{ ...DEFAULT_DIRECT_INDICATOR_PARAMETERS }];
  const random = mulberry32(0x17c0de);
  while (result.length < RANDOM_CANDIDATES) {
    result.push({
      driftEstimateHalfLifeMs: logUniform(random, 60_000, 6 * 3_600_000),
      driftForecastHalfLifeMs: logUniform(random, 60_000, 6 * 3_600_000),
      driftScale: logUniform(random, 0.002, 1.5),
      varianceEstimateHalfLifeMs: logUniform(random, 5 * 60_000, 24 * 3_600_000),
      longRunVarianceHalfLifeMs: logUniform(random, 3 * 3_600_000, 72 * 3_600_000),
      varianceForecastHalfLifeMs: logUniform(random, 10 * 60_000, 48 * 3_600_000),
      transitionWidthGridCells: logUniform(random, 0.5, 8),
    });
  }
  return result;
}

function coordinateRefinements(
  center: DirectIndicatorPredictorParameters,
  round: number,
): DirectIndicatorPredictorParameters[] {
  const multiplier = 1 + 0.5 / (round + 1);
  const keys = Object.keys(center) as Array<keyof DirectIndicatorPredictorParameters>;
  return keys.flatMap((key) => [1 / multiplier, multiplier].map((scale) => ({
    ...center,
    [key]: clamp(center[key] * scale, ...DIRECT_INDICATOR_PARAMETER_BOUNDS[key]),
  })));
}

function writePresets(
  global: ScoredParameters,
  locals: Array<{ windowId: string } & ScoredParameters>,
  generatedAt: string,
): void {
  const presets = [{
    id: "direct-indicator-global-2026-07-20",
    label: "Direct indicator → 6 quadratic parameters · global fit",
    model: "direct-indicator",
    scope: "global",
    windowId: null,
    intervalMs: null,
    parameters: global.parameters,
    loss: global.loss,
    diagnosticCrossEntropy: global.crossEntropy,
    source: "Equal-window direct-decoder fit across all 33 static inspector windows",
    generatedAt,
  }, ...locals.map((fit) => ({
    id: `direct-indicator-local-${fit.windowId}-1m-2026-07-20`,
    label: `Direct indicator → 6 quadratic parameters · local fit · ${fit.windowId} · 1m`,
    model: "direct-indicator",
    scope: "window",
    windowId: fit.windowId,
    intervalMs: INTERVAL_MS,
    parameters: fit.parameters,
    loss: fit.loss,
    diagnosticCrossEntropy: fit.crossEntropy,
    source: `Hindsight direct-decoder fit on eight causal samples from ${fit.windowId}`,
    generatedAt,
  }))];
  const target = path.join(repoRoot, "apps/server/src/direct-indicator-predictor-presets.ts");
  fs.writeFileSync(
    target,
    `import type { VwKamaPredictorPreset } from "@trading/bot-algo";\n\n`
      + `export const DIRECT_INDICATOR_PREDICTOR_PRESETS = ${JSON.stringify(presets, null, 2)}`
      + ` satisfies VwKamaPredictorPreset[];\n`,
  );
  process.stdout.write(`Wrote ${presets.length} direct predictor presets to ${path.relative(repoRoot, target)}.\n`);
}

function readDay(day: number): Array<{ openTime: number; close: number }> {
  const date = new Date(day).toISOString().slice(0, 10);
  const plain = path.join(historyRoot, `${date}.jsonl`);
  const compressed = `${plain}.gz`;
  if (!fs.existsSync(plain) && !fs.existsSync(compressed)) {
    throw new Error(`Missing calibration history ${date}; fetch the BTCUSDT 1m daily shard and retry.`);
  }
  const content = fs.existsSync(plain)
    ? fs.readFileSync(plain, "utf8")
    : gunzipSync(fs.readFileSync(compressed)).toString("utf8");
  return content.trim().split("\n").filter(Boolean).map((line) => {
    const candle = JSON.parse(line) as { openTime: number; close: number };
    return { openTime: candle.openTime, close: candle.close };
  });
}

function lowerBound(values: Array<{ openTime: number }>, time: number): number {
  let low = 0;
  let high = values.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (values[middle]!.openTime < time) low = middle + 1;
    else high = middle;
  }
  return low;
}

function evenIndices(start: number, end: number, count: number): number[] {
  if (end < start) return [];
  if (count <= 1 || end === start) return [start];
  return Array.from({ length: count }, (_, index) =>
    Math.round(start + index * (end - start) / (count - 1)));
}

function utcDay(time: number): number {
  const date = new Date(time);
  return Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate());
}

function logUniform(random: () => number, minimum: number, maximum: number): number {
  return Math.exp(Math.log(minimum) + random() * (Math.log(maximum) - Math.log(minimum)));
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

function mulberry32(seed: number): () => number {
  return () => {
    seed |= 0;
    seed = seed + 0x6D2B79F5 | 0;
    let value = Math.imul(seed ^ seed >>> 15, 1 | seed);
    value = value + Math.imul(value ^ value >>> 7, 61 | value) ^ value;
    return ((value ^ value >>> 14) >>> 0) / 4_294_967_296;
  };
}
