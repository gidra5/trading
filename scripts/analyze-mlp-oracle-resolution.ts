import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { gunzipSync } from "node:zlib";
import {
  prepareExposureValueOracle,
  type Candle,
} from "@trading/bot-algo";

const MINUTE_MS = 60_000;
const FLOAT32_TINY_LOG = Math.log(1.1754943508222875e-38);

interface DatasetShard {
  split: string;
  date: string;
  count: number;
  oracleRowOffset: number;
  oracleRowStride: number;
  oracleTargetTimeStart: number;
  rawOracleProbabilities: string;
  timeWeights: string;
}

interface DatasetManifest {
  actionCount: number;
  grid: number[];
  samplingIntervalMs: number;
  shards: DatasetShard[];
  policySupport: {
    visible_lower: number;
    visible_upper: number;
    friction: number;
    temperature: number;
  };
  execution: {
    feeBps: number;
    minimumEffectiveExposure: number;
    maximumEffectiveExposure: number;
    maintenanceBpsHour: {
      quoteLend: number;
      quoteBorrow: number;
      assetBorrow: number;
    };
    gridSize: number;
    temperature: number;
  };
}

interface ComparisonExample {
  featureDate: string;
  targetDate: string;
  targetMinute: number;
  fineRow: number;
  fineFile: string;
  weightFile: string;
  weightRow: number;
}

interface Observation {
  featureDate: string;
  targetDate: string;
  weight: number;
  kl: number;
  jsd: number;
  meanActionError: number;
  modalActionError: number;
}

void main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : error);
  process.exitCode = 1;
});

async function main(): Promise<void> {
  const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
  const datasetRoot = path.resolve(
    repoRoot,
    argument("dataset")
      ?? "data/ml-datasets/mlp-conservative-quadratic-cutoff-temporal-matmul-cuda-v11-study-subset",
  );
  const manifest = JSON.parse(
    fs.readFileSync(path.join(datasetRoot, "dataset.json"), "utf8"),
  ) as DatasetManifest;
  const requestedDates = new Set(
    (argument("dates") ?? "")
      .split(",")
      .map((value) => value.trim())
      .filter(Boolean),
  );
  const split = argument("split") ?? "validation";
  const includeCrossDay = hasFlag("include-cross-day");
  const examples = selectMinuteAlignedExamples(
    manifest,
    split,
    requestedDates,
    includeCrossDay,
  );
  if (examples.length === 0) {
    throw new Error("No minute-aligned dataset examples matched the requested split and dates.");
  }

  const historyRoot = path.resolve(
    repoRoot,
    argument("history")
      ?? "data/historical/spot-btcusdt/btcusdt/1m",
  );
  const byTargetDate = groupBy(examples, (example) => example.targetDate);
  const observations: Observation[] = [];
  const actionGrid = manifest.grid;
  const visibleIndexes = actionGrid
    .map((value, index) => ({ value, index }))
    .filter(({ value }) =>
      value >= manifest.policySupport.visible_lower
      && value <= manifest.policySupport.visible_upper);
  const visibleActions = visibleIndexes.map(({ value }) => value);
  const currentStates = linspace(
    manifest.policySupport.visible_lower,
    manifest.policySupport.visible_upper,
    31,
  );

  for (const [targetDate, dateExamples] of [...byTargetDate].sort(([left], [right]) =>
    left.localeCompare(right))) {
    const minuteCandles = [
      ...readCandleDay(historyRoot, targetDate),
      ...readCandleDay(historyRoot, offsetDate(targetDate, 1)).slice(0, 60),
    ];
    const prices = Float64Array.from(minuteCandles, (candle) => candle.close);
    const execution = manifest.execution;
    const coarseOracle = prepareExposureValueOracle(prices, {
      scoreStartIndex: 0,
      holdingPeriodSteps: 1,
      valueHorizonSteps: 60,
      friction: execution.feeBps / 10_000,
      gridSize: execution.gridSize,
      minExposure: execution.minimumEffectiveExposure,
      maxExposure: execution.maximumEffectiveExposure,
      maxEffectiveExposure: Math.max(
        Math.abs(execution.minimumEffectiveExposure),
        Math.abs(execution.maximumEffectiveExposure),
      ),
      terminalIndex: 1_439,
      temperature: execution.temperature,
      opportunityEpsilon: 0,
      quoteLendRate: bpsHourToPerSteps(
        execution.maintenanceBpsHour.quoteLend,
        60,
      ),
      quoteBorrowRate: bpsHourToPerSteps(
        execution.maintenanceBpsHour.quoteBorrow,
        60,
      ),
      assetBorrowRate: bpsHourToPerSteps(
        execution.maintenanceBpsHour.assetBorrow,
        60,
      ),
      includeActionValues: false,
      includeProbabilities: true,
    });
    if (!coarseOracle.probabilities) {
      throw new Error(`One-minute oracle did not retain probabilities for ${targetDate}.`);
    }
    assertSameGrid(actionGrid, coarseOracle.grid);

    const fineBuffers = new Map<string, Float32Array>();
    const weightBuffers = new Map<string, Float32Array>();
    for (const example of dateExamples) {
      const fine = cachedFloat32File(
        fineBuffers,
        path.join(datasetRoot, example.fineFile),
      );
      const weights = cachedFloat32File(
        weightBuffers,
        path.join(datasetRoot, example.weightFile),
      );
      const fineStart = example.fineRow * manifest.actionCount;
      const coarseStart = example.targetMinute * manifest.actionCount;
      const fineBase = fine.subarray(fineStart, fineStart + manifest.actionCount);
      const coarseBase = coarseOracle.probabilities.subarray(
        coarseStart,
        coarseStart + manifest.actionCount,
      );
      if (fineBase.length !== manifest.actionCount
        || coarseBase.length !== manifest.actionCount) {
        throw new Error(`Oracle row is outside its probability file for ${targetDate}.`);
      }
      const comparison = compareConditionalSurfaces(
        fineBase,
        coarseBase,
        visibleIndexes.map(({ index }) => index),
        visibleActions,
        currentStates,
        manifest.policySupport.friction,
        manifest.policySupport.temperature,
      );
      observations.push({
        featureDate: example.featureDate,
        targetDate,
        weight: weights[example.weightRow]!,
        ...comparison,
      });
    }
    process.stdout.write(`${JSON.stringify({
      event: "oracle-resolution-date",
      targetDate,
      examples: dateExamples.length,
      completedExamples: observations.length,
    })}\n`);
  }

  const result = {
    version: 1,
    comparison: "stored 1-second oracle versus close-only 1-minute oracle",
    split,
    selectedFeatureDates: [...new Set(observations.map((item) => item.featureDate))].sort(),
    includeCrossDay,
    examples: observations.length,
    currentStates: currentStates.length,
    visibleActions: visibleActions.length,
    kl: summarize(observations, (item) => item.kl),
    jsd: summarize(observations, (item) => item.jsd),
    meanActionAbsoluteError: summarize(
      observations,
      (item) => item.meanActionError,
    ),
    modalActionAbsoluteError: summarize(
      observations,
      (item) => item.modalActionError,
    ),
    byFeatureDate: [...groupBy(observations, (item) => item.featureDate)]
      .map(([date, items]) => ({
        date,
        examples: items.length,
        kl: summarize(items, (item) => item.kl),
        jsd: summarize(items, (item) => item.jsd),
      }))
      .sort((left, right) => right.kl.weightedMean - left.kl.weightedMean),
  };
  const output = argument("output");
  if (output) {
    const outputFile = path.resolve(repoRoot, output);
    fs.mkdirSync(path.dirname(outputFile), { recursive: true });
    fs.writeFileSync(outputFile, `${JSON.stringify(result, null, 2)}\n`);
  }
  process.stdout.write(`${JSON.stringify({
    event: "oracle-resolution-complete",
    ...result,
    ...(output ? { output: path.resolve(repoRoot, output) } : {}),
  })}\n`);
}

function selectMinuteAlignedExamples(
  manifest: DatasetManifest,
  split: string,
  requestedDates: Set<string>,
  includeCrossDay: boolean,
): ComparisonExample[] {
  const result: ComparisonExample[] = [];
  for (const shard of manifest.shards) {
    if (shard.split !== split
      || (requestedDates.size > 0 && !requestedDates.has(shard.date))) continue;
    for (let row = 0; row < shard.count; row += 1) {
      const targetTime = shard.oracleTargetTimeStart
        + row * manifest.samplingIntervalMs;
      if (positiveModulo(targetTime, MINUTE_MS) !== MINUTE_MS - 1) continue;
      const targetDate = new Date(targetTime).toISOString().slice(0, 10);
      if (!includeCrossDay && targetDate !== shard.date) continue;
      const dayStart = Date.parse(`${targetDate}T00:00:00.000Z`);
      const targetMinute = Math.floor((targetTime - dayStart) / MINUTE_MS);
      result.push({
        featureDate: shard.date,
        targetDate,
        targetMinute,
        fineRow: shard.oracleRowOffset + row * shard.oracleRowStride,
        fineFile: shard.rawOracleProbabilities,
        weightFile: shard.timeWeights,
        weightRow: row,
      });
    }
  }
  return result;
}

function compareConditionalSurfaces(
  fineBase: Float32Array,
  coarseBase: Float32Array,
  visibleIndexes: number[],
  visibleActions: number[],
  currentStates: number[],
  friction: number,
  temperature: number,
): Pick<Observation, "kl" | "jsd" | "meanActionError" | "modalActionError"> {
  let kl = 0;
  let jsd = 0;
  let meanActionError = 0;
  let modalActionError = 0;
  for (const current of currentStates) {
    const fine = conditionalProbabilities(
      fineBase,
      visibleIndexes,
      visibleActions,
      current,
      friction,
      temperature,
    );
    const coarse = conditionalProbabilities(
      coarseBase,
      visibleIndexes,
      visibleActions,
      current,
      friction,
      temperature,
    );
    let fineMean = 0;
    let coarseMean = 0;
    let fineMode = 0;
    let coarseMode = 0;
    for (let index = 0; index < fine.length; index += 1) {
      const p = fine[index]!;
      const q = coarse[index]!;
      const mixture = (p + q) / 2;
      if (p > 0) {
        kl += p * (Math.log(p) - Math.log(Math.max(Number.MIN_VALUE, q)));
        jsd += 0.5 * p * (Math.log(p) - Math.log(mixture));
      }
      if (q > 0) jsd += 0.5 * q * (Math.log(q) - Math.log(mixture));
      fineMean += p * visibleActions[index]!;
      coarseMean += q * visibleActions[index]!;
      if (p > fine[fineMode]!) fineMode = index;
      if (q > coarse[coarseMode]!) coarseMode = index;
    }
    meanActionError += Math.abs(fineMean - coarseMean);
    modalActionError += Math.abs(
      visibleActions[fineMode]! - visibleActions[coarseMode]!,
    );
  }
  const denominator = currentStates.length;
  return {
    kl: kl / denominator,
    jsd: jsd / denominator,
    meanActionError: meanActionError / denominator,
    modalActionError: modalActionError / denominator,
  };
}

function conditionalProbabilities(
  base: Float32Array,
  visibleIndexes: number[],
  visibleActions: number[],
  current: number,
  friction: number,
  temperature: number,
): Float64Array {
  const logits = new Float64Array(visibleIndexes.length);
  let maximum = Number.NEGATIVE_INFINITY;
  for (let output = 0; output < visibleIndexes.length; output += 1) {
    const action = visibleActions[output]!;
    const difference = action - current;
    const buyFactor = 1 - friction * difference
      / (1 - friction + friction * action);
    const sellFactor = 1 - friction * (-difference)
      / (1 - friction * action);
    const factor = difference > 0 ? buyFactor : difference < 0 ? sellFactor : 1;
    const probability = base[visibleIndexes[output]!]!;
    const logBase = probability > 0 ? Math.log(probability) : FLOAT32_TINY_LOG;
    const logit = logBase + Math.log(Math.max(Number.MIN_VALUE, factor)) / temperature;
    logits[output] = logit;
    maximum = Math.max(maximum, logit);
  }
  let total = 0;
  for (let index = 0; index < logits.length; index += 1) {
    logits[index] = Math.exp(logits[index]! - maximum);
    total += logits[index]!;
  }
  for (let index = 0; index < logits.length; index += 1) logits[index] /= total;
  return logits;
}

function summarize<T>(
  observations: T[],
  value: (item: T) => number,
): {
  weightedMean: number;
  weightedStdDev: number;
  mean: number;
  stdDev: number;
  p50: number;
  p90: number;
  p95: number;
  p99: number;
  maximum: number;
} {
  const weighted = observations as Array<T & { weight: number }>;
  const values = observations.map(value);
  const weightSum = weighted.reduce((sum, item) => sum + item.weight, 0);
  const weightedMean = weighted.reduce(
    (sum, item) => sum + item.weight * value(item),
    0,
  ) / weightSum;
  const mean = values.reduce((sum, item) => sum + item, 0) / values.length;
  const sorted = [...values].sort((left, right) => left - right);
  return {
    weightedMean,
    weightedStdDev: Math.sqrt(weighted.reduce(
      (sum, item) => sum + item.weight * (value(item) - weightedMean) ** 2,
      0,
    ) / weightSum),
    mean,
    stdDev: Math.sqrt(
      values.reduce((sum, item) => sum + (item - mean) ** 2, 0) / values.length,
    ),
    p50: quantile(sorted, 0.5),
    p90: quantile(sorted, 0.9),
    p95: quantile(sorted, 0.95),
    p99: quantile(sorted, 0.99),
    maximum: sorted.at(-1)!,
  };
}

function quantile(sorted: number[], fraction: number): number {
  const position = (sorted.length - 1) * fraction;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  const weight = position - lower;
  return sorted[lower]! * (1 - weight) + sorted[upper]! * weight;
}

function cachedFloat32File(
  cache: Map<string, Float32Array>,
  file: string,
): Float32Array {
  const existing = cache.get(file);
  if (existing) return existing;
  const buffer = fs.readFileSync(file);
  const values = new Float32Array(
    buffer.buffer,
    buffer.byteOffset,
    buffer.byteLength / Float32Array.BYTES_PER_ELEMENT,
  );
  cache.set(file, values);
  return values;
}

function readCandleDay(root: string, date: string): Candle[] {
  const plain = path.join(root, `${date}.jsonl`);
  const content = fs.existsSync(plain)
    ? fs.readFileSync(plain, "utf8")
    : gunzipSync(fs.readFileSync(`${plain}.gz`)).toString("utf8");
  const candles = content.split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line) as Candle);
  if (candles.length !== 1_440) {
    throw new Error(`${date} has ${candles.length}/1440 one-minute candles.`);
  }
  return candles;
}

function bpsHourToPerSteps(bps: number, stepsPerHour: number): number {
  return Math.expm1(Math.log1p(bps / 10_000) / stepsPerHour);
}

function linspace(lower: number, upper: number, count: number): number[] {
  return Array.from(
    { length: count },
    (_, index) => lower + index / (count - 1) * (upper - lower),
  );
}

function assertSameGrid(expected: number[], actual: ArrayLike<number>): void {
  if (expected.length !== actual.length
    || expected.some((value, index) => Math.abs(value - actual[index]!) > 1e-9)) {
    throw new Error("One-minute and stored one-second oracle grids differ.");
  }
}

function groupBy<T>(
  values: T[],
  key: (value: T) => string,
): Map<string, T[]> {
  const result = new Map<string, T[]>();
  for (const value of values) {
    const group = key(value);
    const items = result.get(group) ?? [];
    items.push(value);
    result.set(group, items);
  }
  return result;
}

function offsetDate(date: string, days: number): string {
  return new Date(Date.parse(`${date}T00:00:00.000Z`) + days * 86_400_000)
    .toISOString()
    .slice(0, 10);
}

function positiveModulo(value: number, divisor: number): number {
  return ((value % divisor) + divisor) % divisor;
}

function argument(name: string): string | undefined {
  const index = process.argv.indexOf(`--${name}`);
  return index >= 0 ? process.argv[index + 1] : undefined;
}

function hasFlag(name: string): boolean {
  return process.argv.includes(`--${name}`);
}
