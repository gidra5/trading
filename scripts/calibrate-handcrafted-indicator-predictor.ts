import fs from "node:fs";
import path from "node:path";
import { readCandleShardReferenceSync } from "@trading/storage";
import { fileURLToPath } from "node:url";
import {
  DEFAULT_HANDCRAFTED_INDICATOR_PARAMETERS,
  HANDCRAFTED_INDICATOR_PARAMETER_BOUNDS,
  predictHandcraftedIndicatorRegret,
  prepareHandcraftedIndicatorStates,
  type HandcraftedIndicatorPredictorParameters,
} from "../packages/bot-algo/src/handcrafted-indicator-predictor.js";
import {
  exposureValueOracleProbabilities,
  prepareExposureValueOracle,
} from "../packages/bot-algo/src/exposure-value-distillation.js";

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
const historyRoot = path.join(repoRoot, "data/market/immutable/refs/candles/spot-btcusdt/btcusdt/1m");
const grid = Float64Array.from(
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

interface CalibrationWindow {
  id: string;
  startTime: number;
  endTime: number;
  prices: Float64Array;
  scoreStartIndex: number;
  sampleIndices: number[];
  oracleProbabilities: Float64Array[];
}

interface ScoredParameters {
  parameters: HandcraftedIndicatorPredictorParameters;
  loss: number;
  crossEntropy: number;
}

const windows = loadCalibrationWindows();
process.stdout.write(`Loaded ${windows.length} inspector windows and ${windows.reduce((sum, item) => sum + item.sampleIndices.length, 0)} causal samples.\n`);
const baseline = score(DEFAULT_HANDCRAFTED_INDICATOR_PARAMETERS);
process.stdout.write(`default regret ${baseline.loss.toFixed(8)} · CE ${baseline.crossEntropy.toFixed(8)}\n`);
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
  const refinements = coordinateRefinements(best!.parameters, round);
  for (const candidate of refinements) {
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
    `local ${windowIndex + 1}/${windows.length} ${calibrationWindow.id} · regret ${localBest.loss.toFixed(8)}\n`,
  );
  return { windowId: calibrationWindow.id, ...localBest };
});
const generatedAt = new Date().toISOString();
const artifact = {
  generatedAt,
  objective: "mean oracle regret of predicted optimal exposure",
  intervalMs: INTERVAL_MS,
  holdingPeriodMs: HOLDING_STEPS * INTERVAL_MS,
  valueHorizonMs: HORIZON_STEPS * INTERVAL_MS,
  gridSize: GRID_SIZE,
  windowCount: windows.length,
  sampleCount: windows.reduce((sum, item) => sum + item.sampleIndices.length, 0),
  global: {
    loss: best!.loss,
    diagnosticCrossEntropy: best!.crossEntropy,
    parameters: best!.parameters,
  },
  windows: localFits.map((fit) => ({
    windowId: fit.windowId,
    loss: fit.loss,
    diagnosticCrossEntropy: fit.crossEntropy,
    parameters: fit.parameters,
  })),
};
process.stdout.write(`${JSON.stringify(artifact, null, 2)}\n`);
if (process.argv.includes("--write-presets")) {
  const presets = [{
    id: "handcrafted-global-2026-07-20",
    label: "Handcrafted · global fit",
    model: "handcrafted",
    scope: "global",
    windowId: null,
    intervalMs: null,
    parameters: best!.parameters,
    loss: best!.loss,
    diagnosticCrossEntropy: best!.crossEntropy,
    source: "Equal-window fit across all 33 static inspector windows",
    generatedAt,
  }, ...localFits.map((fit) => ({
    id: `handcrafted-local-${fit.windowId}-1m-2026-07-20`,
    label: `Handcrafted · local fit · ${fit.windowId} · 1m`,
    model: "handcrafted",
    scope: "window",
    windowId: fit.windowId,
    intervalMs: INTERVAL_MS,
    parameters: fit.parameters,
    loss: fit.loss,
    diagnosticCrossEntropy: fit.crossEntropy,
    source: `Hindsight fit on eight causal samples from ${fit.windowId}`,
    generatedAt,
  }))];
  const target = path.join(repoRoot, "apps/server/src/handcrafted-predictor-presets.ts");
  fs.writeFileSync(
    target,
    `import type { VwKamaPredictorPreset } from "@trading/bot-algo";\n\n`
      + `export const HANDCRAFTED_PREDICTOR_PRESETS = ${JSON.stringify(presets, null, 2)}`
      + ` satisfies VwKamaPredictorPreset[];\n`,
  );
  process.stdout.write(`Wrote ${presets.length} predictor presets to ${path.relative(repoRoot, target)}.\n`);
}

function score(
  parameters: HandcraftedIndicatorPredictorParameters,
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
      const index = calibrationWindow.sampleIndices[sample]!;
      const prediction = predictHandcraftedIndicatorRegret(
        grid,
        states[index]!,
        parameters,
        {
          intervalMs: INTERVAL_MS,
          holdingPeriodSteps: HOLDING_STEPS,
          valueHorizonSteps: HORIZON_STEPS,
          temperature: TEMPERATURE,
          execution,
        },
      );
      const target = calibrationWindow.oracleProbabilities[sample]!;
      let sampleCrossEntropy = 0;
      for (let action = 0; action < grid.length; action += 1) {
        if (target[action]! > 0) {
          sampleCrossEntropy -= target[action]!
            * Math.log(Math.max(Number.MIN_VALUE, prediction.probabilities[action]!));
        }
      }
      let predictedIndex = 0;
      for (let action = 1; action < grid.length; action += 1) {
        if (prediction.probabilities[action]! > prediction.probabilities[predictedIndex]!) {
          predictedIndex = action;
        }
      }
      const maximumTarget = Math.max(...target);
      loss += -TEMPERATURE * Math.log(
        Math.max(Number.MIN_VALUE, target[predictedIndex]!)
          / Math.max(Number.MIN_VALUE, maximumTarget),
      );
      crossEntropy += sampleCrossEntropy;
      count += 1;
    }
  }
  return {
    parameters,
    loss: loss / Math.max(1, count),
    crossEntropy: crossEntropy / Math.max(1, count),
  };
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
    const oracleProbabilities = sampleIndices.map((index) => {
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
      return Float64Array.from(exposureValueOracleProbabilities(oracle, 0));
    });
    return {
      id: match[1]!,
      startTime,
      endTime,
      prices,
      scoreStartIndex,
      sampleIndices,
      oracleProbabilities,
    };
  });
}

function initialCandidates(): HandcraftedIndicatorPredictorParameters[] {
  const result = [{ ...DEFAULT_HANDCRAFTED_INDICATOR_PARAMETERS }];
  const random = mulberry32(0x17c0de);
  while (result.length < RANDOM_CANDIDATES) {
    result.push({
      driftEstimateHalfLifeMs: logUniform(random, 60_000, 6 * 3_600_000),
      driftForecastHalfLifeMs: logUniform(random, 60_000, 6 * 3_600_000),
      driftScale: logUniform(random, 0.002, 1.5),
      varianceEstimateHalfLifeMs: logUniform(random, 5 * 60_000, 24 * 3_600_000),
      longRunVarianceHalfLifeMs: logUniform(random, 3 * 3_600_000, 72 * 3_600_000),
      varianceForecastHalfLifeMs: logUniform(random, 10 * 60_000, 48 * 3_600_000),
    });
  }
  return result;
}

function coordinateRefinements(
  center: HandcraftedIndicatorPredictorParameters,
  round: number,
): HandcraftedIndicatorPredictorParameters[] {
  const multiplier = 1 + 0.5 / (round + 1);
  const keys = Object.keys(center) as Array<keyof HandcraftedIndicatorPredictorParameters>;
  return keys.flatMap((key) => [1 / multiplier, multiplier].map((scale) => ({
    ...center,
    [key]: clamp(center[key] * scale, ...HANDCRAFTED_INDICATOR_PARAMETER_BOUNDS[key]),
  })));
}

function readDay(day: number): Array<{ openTime: number; close: number }> {
  const date = new Date(day).toISOString().slice(0, 10);
  return readCandleShardReferenceSync(
    path.join(historyRoot, `${date}.json`),
  ).map((candle) => {
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
