import fs from "node:fs";
import path from "node:path";
import {
  readCandleShardReferenceSync,
  readReferencedPayloadSync,
} from "@trading/storage";
import type {
  Candle,
  ExposureValueOracleActionDistribution,
} from "@trading/bot-algo";
import { appConfig } from "../apps/server/src/config.js";
import {
  hindsightOracleTargetDecision,
  runBotBacktestFromCandles,
} from "../apps/server/src/bot-backtest.js";
import {
  JointPriceOracleRuntime,
  isJointPriceOracleDecisionTime,
} from "../apps/server/src/joint-price-oracle-runtime.js";

const DAY_MS = 86_400_000;
const HISTORY_ROOT = path.resolve(
  "data/market/immutable/refs/candles/spot-btcusdt/btcusdt/1s",
);
const TARGET_ROOT = path.resolve(
  "data/training/immutable/refs/oracle/1s/hindsight-bot-71391c44b323e044e6ab",
);
const DATA_DIR = path.resolve("data");
const TEST_DAYS = 30;

void main();

async function main(): Promise<void> {
const days = positiveIntegerArgument("days", 30);
if (days > TEST_DAYS) {
  throw new Error(`--days cannot exceed the ${TEST_DAYS}-day split boundary.`);
}
const temperature = optionalPositiveNumberArgument("temperature");
const maximumLeverage = optionalPositiveNumberArgument("max-leverage") ?? 1;
const requestedModelId = optionalStringArgument("model");
const split = choiceArgument("split", ["validation", "test"] as const, "validation");
if (split === "test" && !process.argv.includes("--allow-test")) {
  throw new Error("Backtesting the locked test split requires explicit --allow-test.");
}
const allTargetFiles = fs.readdirSync(TARGET_ROOT)
  .filter((file) => /^\d{4}-\d{2}-\d{2}\.json$/.test(file))
  .sort();
const splitFiles = split === "test"
  ? allTargetFiles.slice(-TEST_DAYS)
  : allTargetFiles.slice(-(TEST_DAYS * 2), -TEST_DAYS);
if (splitFiles.length !== TEST_DAYS) {
  throw new Error(`Verified oracle cache lacks the complete ${split} split.`);
}
const targetFiles = splitFiles.slice(-days);
if (targetFiles.length !== days) {
  throw new Error(`Verified oracle cache has ${targetFiles.length}/${days} requested ${split} days.`);
}
const dates = targetFiles.map((file) => file.slice(0, -5));
const firstDay = Date.parse(`${dates[0]}T00:00:00Z`);
const warmupDate = new Date(firstDay - DAY_MS).toISOString().slice(0, 10);
const warmupDay = loadCandles(warmupDate);
const warmup = warmupDay.slice(-3_600);
const candles = dates.flatMap(loadCandles);
if (warmup.length !== 3_600 || candles.length !== days * 86_400) {
  throw new Error("Held-out candle history is incomplete.");
}

console.log(`INFERENCE days=${days} decisions=${days * 1_440}`);
const rawDecisionTimes = candles
  .map((candle) => candle.closeTime)
  .filter(isJointPriceOracleDecisionTime);
const splitFirstDecision = Date.parse(`${splitFiles[0]!.slice(0, -5)}T00:00:00Z`) + 999;
const splitLastDecision = Date.parse(
  `${splitFiles.at(-1)!.slice(0, -5)}T00:00:00Z`,
) + 999 + 1_439 * 60_000;
// Match the trainer's chronological sealing exactly.  Validation drops one
// hour at its train-facing boundary; both splits drop the final 59 target rows
// whose oracle horizons cross the next boundary.
const firstUsableDecision = split === "validation"
  ? splitFirstDecision + 60 * 60_000
  : splitFirstDecision;
const lastUsableDecision = splitLastDecision - 59 * 60_000;
const decisionTimes = rawDecisionTimes.filter((timestamp) =>
  timestamp >= firstUsableDecision && timestamp <= lastUsableDecision);
const expectedDecisionCount = days * 1_440 - 59
  - (split === "validation" && targetFiles[0] === splitFiles[0] ? 60 : 0);
if (decisionTimes.length !== expectedDecisionCount) {
  throw new Error(
    `Requested ${split} window has ${decisionTimes.length}/${expectedDecisionCount} `
    + "split-isolated decisions.",
  );
}
const runtime = new JointPriceOracleRuntime(
  DATA_DIR,
  requestedModelId,
  temperature,
);
const manifest = requestedModelId
  ? runtime.models().find(({ id }) => id === requestedModelId)
  : runtime.models()[0];
if (!manifest) throw new Error("No joint price-oracle artifact is available.");
const learnedDistributions = await runtime.predictDistributions(
  [...warmup, ...candles],
  decisionTimes,
);
const learnedByTime = new Map(
  decisionTimes.map((time, index) => [time, learnedDistributions[index]!] as const),
);

console.log("BACKTEST learned-oracle-1s");
const learned = await runBotBacktestFromCandles(candles, {
  config: appConfig.strategy,
  strategy: "learned-oracle-1s",
  warmup,
  learnedOracleDistributionAt: (timestamp) => learnedByTime.get(timestamp) ?? null,
  learnedOracleMaximumLeverage: maximumLeverage,
  summaryOnly: true,
  onProgress: progressReporter("learned"),
});

console.log("BACKTEST hindsight-oracle-1s");
const storedHindsightAt = storedOracleProvider(dates);
const usableDecisionTimes = new Set(decisionTimes);
const hindsightAt = (timestamp: number) => usableDecisionTimes.has(timestamp)
  ? storedHindsightAt(timestamp)
  : null;
const hindsightDistributions = decisionTimes.map((timestamp) => {
  const distribution = hindsightAt(timestamp);
  if (!distribution) {
    throw new Error(`Missing verified oracle target at ${timestamp}.`);
  }
  return distribution;
});
const hindsight = await runBotBacktestFromCandles(candles, {
  config: appConfig.strategy,
  strategy: "hindsight-oracle-1s",
  warmup,
  hindsightOracleDistributionAt: hindsightAt,
  hindsightOracleMaximumLeverage: maximumLeverage,
  summaryOnly: true,
  onProgress: progressReporter("hindsight"),
});

const report = {
    version: 2,
  createdAt: new Date().toISOString(),
  modelId: manifest.id,
  oracle: {
    valueHorizonSteps: 3_600,
    decisionDelaySteps: 60,
    holdingPeriodSteps: 60,
  },
  split: {
    kind: split === "test"
      ? "chronological-held-out-test"
      : "chronological-validation",
    dates,
    decisions: decisionTimes.length,
  },
  logitTemperature: temperature ?? manifest.calibration.logitTemperature,
  maximumLeverage,
  learned: learned.summary,
  hindsight: hindsight.summary,
  policyDiagnostics: {
    learned: distributionDiagnostics(learnedDistributions),
    hindsight: distributionDiagnostics(hindsightDistributions),
  },
  comparison: {
    finalEquityRatio: safeRatio(
      learned.summary.finalEquity,
      hindsight.summary.finalEquity,
    ),
    returnPctRatio: safeRatio(
      learned.summary.returnPct,
      hindsight.summary.returnPct,
    ),
    drawdownDifferencePct:
      learned.summary.maxDrawdownPct - hindsight.summary.maxDrawdownPct,
  },
};
const outputFile = path.resolve(
  `data/benchmarks/${manifest.id}-${split}-${days}d-`
  + `leverage-${fileNumber(maximumLeverage)}-`
  + `temperature-${fileNumber(temperature ?? manifest.calibration.logitTemperature)}.json`,
);
fs.mkdirSync(path.dirname(outputFile), { recursive: true });
fs.writeFileSync(outputFile, `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify({
  output: outputFile,
  learned: report.learned,
  hindsight: report.hindsight,
  comparison: report.comparison,
}));
}

function fileNumber(value: number): string {
  return String(value).replaceAll("-", "minus-").replaceAll(".", "_");
}

function loadCandles(date: string): Candle[] {
  const file = path.join(HISTORY_ROOT, `${date}.json`);
  if (!fs.existsSync(file)) throw new Error(`Missing one-second candle day ${date}.`);
  return readCandleShardReferenceSync(file);
}

function storedOracleProvider(
  targetDates: readonly string[],
): (timestamp: number) => ExposureValueOracleActionDistribution | null {
  const rows = new Map<number, {
    values: Float32Array;
    offset: number;
    grid: number[];
  }>();
  for (const date of targetDates) {
    const file = path.join(TARGET_ROOT, `${date}.json`);
    const { reference, payload } = readReferencedPayloadSync(file);
    if (reference.kind !== "trading-sequential-shard") {
      throw new Error(`Oracle target is not sequential: ${file}`);
    }
    const columns = Number(reference.layout.columns);
    const count = reference.sequence.count;
    const copied = payload.buffer.slice(
      payload.byteOffset,
      payload.byteOffset + payload.byteLength,
    );
    const values = new Float32Array(copied);
    const grid = (reference.metadata as { contract?: { usableGrid?: number[] } } | undefined)
      ?.contract?.usableGrid;
    if (columns !== 101 || values.length !== count * columns || grid?.length !== columns) {
      throw new Error(`Oracle target layout is incompatible: ${file}`);
    }
    for (let row = 0; row < count; row += 1) {
      rows.set(reference.sequence.start + row * reference.sequence.step, {
        values,
        offset: row * columns,
        grid,
      });
    }
  }
  return (timestamp) => {
    const row = rows.get(timestamp);
    return row ? probabilityDistribution(
      row.values.subarray(row.offset, row.offset + row.grid.length),
      row.grid,
    ) : null;
  };
}

function probabilityDistribution(
  source: ArrayLike<number>,
  sourceGrid: readonly number[],
): ExposureValueOracleActionDistribution {
  const probabilities = Float32Array.from(source);
  const grid = Float64Array.from(sourceGrid);
  let total = 0;
  for (const probability of probabilities) total += probability;
  let modalIndex = 0;
  let mean = 0;
  let secondMoment = 0;
  let entropy = 0;
  let feasibleActionCount = 0;
  for (let index = 0; index < probabilities.length; index += 1) {
    const probability = probabilities[index]! / total;
    probabilities[index] = probability;
    if (probability > probabilities[modalIndex]!) modalIndex = index;
    mean += probability * grid[index]!;
    secondMoment += probability * grid[index]! ** 2;
    if (probability > 0) {
      entropy -= probability * Math.log(probability);
      feasibleActionCount += 1;
    }
  }
  return {
    grid,
    probabilities,
    mean,
    secondMoment,
    modalExposure: grid[modalIndex]!,
    entropy,
    opportunity: 0,
    feasibleActionCount,
  };
}

function positiveIntegerArgument(name: string, fallback: number): number {
  const prefix = `--${name}=`;
  const value = Number(
    process.argv.find((argument) => argument.startsWith(prefix))?.slice(prefix.length)
    ?? fallback,
  );
  if (!Number.isInteger(value) || value < 1) {
    throw new Error(`--${name} must be a positive integer.`);
  }
  return value;
}

function optionalPositiveNumberArgument(name: string): number | undefined {
  const prefix = `--${name}=`;
  const raw = process.argv.find((argument) => argument.startsWith(prefix))
    ?.slice(prefix.length);
  if (raw === undefined) return undefined;
  const value = Number(raw);
  if (!(value > 0) || !Number.isFinite(value)) {
    throw new Error(`--${name} must be finite and positive.`);
  }
  return value;
}

function optionalStringArgument(name: string): string | undefined {
  const prefix = `--${name}=`;
  const value = process.argv.find((argument) => argument.startsWith(prefix))
    ?.slice(prefix.length).trim();
  return value || undefined;
}

function choiceArgument<const T extends readonly string[]>(
  name: string,
  choices: T,
  fallback: T[number],
): T[number] {
  const prefix = `--${name}=`;
  const value = process.argv.find((argument) => argument.startsWith(prefix))
    ?.slice(prefix.length) ?? fallback;
  if (!choices.includes(value)) {
    throw new Error(`--${name} must be one of ${choices.join(", ")}.`);
  }
  return value as T[number];
}

function safeRatio(numerator: number, denominator: number): number | null {
  return Number.isFinite(numerator) && Number.isFinite(denominator) && denominator !== 0
    ? numerator / denominator
    : null;
}

function progressReporter(label: string): NonNullable<
  Parameters<typeof runBotBacktestFromCandles>[1]["onProgress"]
> {
  let reportedBucket = -1;
  return ({ candlesProcessed, totalCandles }) => {
    const bucket = Math.floor(candlesProcessed / totalCandles * 10);
    if (bucket <= reportedBucket) return;
    reportedBucket = bucket;
    console.log(
      `BACKTEST ${label} ${Math.min(100, bucket * 10)}% (${candlesProcessed}/${totalCandles})`,
    );
  };
}

function distributionDiagnostics(
  distributions: readonly ExposureValueOracleActionDistribution[],
): {
  decisions: number;
  meanRawEntropy: number;
  meanConditionalEntropy: number;
  meanConfidence: number;
  maximumConfidence: number;
  confidenceGatePasses: number;
  nonzeroRawModalDecisions: number;
  nonzeroModalDecisions: number;
  meanAbsoluteRawModalExposure: number;
  meanAbsoluteModalExposure: number;
} {
  const friction = (
    appConfig.strategy.feeBps
    + appConfig.strategy.positionRisk.marketSlippageBps
  ) / 10_000;
  let rawEntropy = 0;
  let conditionalEntropy = 0;
  let confidence = 0;
  let maximumConfidence = 0;
  let confidenceGatePasses = 0;
  let nonzeroRawModalDecisions = 0;
  let nonzeroModalDecisions = 0;
  let absoluteRawModalExposure = 0;
  let absoluteModalExposure = 0;
  for (const distribution of distributions) {
    const decision = hindsightOracleTargetDecision(distribution, 0, friction);
    rawEntropy += distribution.entropy;
    conditionalEntropy += decision.entropy;
    confidence += decision.confidence;
    maximumConfidence = Math.max(maximumConfidence, decision.confidence);
    if (decision.confidence >= 0.05) confidenceGatePasses += 1;
    if (distribution.modalExposure !== 0) nonzeroRawModalDecisions += 1;
    if (decision.targetExposure !== 0) nonzeroModalDecisions += 1;
    absoluteRawModalExposure += Math.abs(distribution.modalExposure);
    absoluteModalExposure += Math.abs(decision.targetExposure);
  }
  const count = distributions.length;
  return {
    decisions: count,
    meanRawEntropy: rawEntropy / count,
    meanConditionalEntropy: conditionalEntropy / count,
    meanConfidence: confidence / count,
    maximumConfidence,
    confidenceGatePasses,
    nonzeroRawModalDecisions,
    nonzeroModalDecisions,
    meanAbsoluteRawModalExposure: absoluteRawModalExposure / count,
    meanAbsoluteModalExposure: absoluteModalExposure / count,
  };
}
