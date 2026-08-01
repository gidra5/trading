import fs from "node:fs";
import path from "node:path";
import {
  readCandleShardReferenceSync,
  readReferencedPayload,
} from "@trading/storage";
import type {
  Candle,
  ExposureValueOracleActionDistribution,
} from "@trading/bot-algo";
import { appConfig } from "../apps/server/src/config.js";
import {
  confidenceConditionedHindsightOracleExposure,
  HINDSIGHT_ORACLE_CONFIDENCE_EXPOSURE_POWER,
  HINDSIGHT_ORACLE_CONFIDENCE_LEVERAGE_FLOOR,
  hindsightOracleTargetDecision,
  runBotBacktestFromCandles,
} from "../apps/server/src/bot-backtest.js";
import { historicalWarmupSamples } from "../apps/server/src/historical-backtest.js";

const DAY_MS = 86_400_000;
const INTERVAL_MS = 1_000;
const ORACLE_FUTURE_MS = 60 * 60_000;
const HISTORY_ROOT = path.resolve(
  "data/market/immutable/refs/candles/spot-btcusdt/btcusdt/1s",
);
const CACHE_ROOT = path.resolve("data/training/immutable/refs/oracle/1s");
const DEFAULT_OUTPUT_ROOT = path.resolve(
  "data/benchmarks/hindsight-oracle-decision-delay-v1/drawdown-analysis",
);

interface CachedOracleShard {
  start: number;
  step: number;
  count: number;
  columns: number;
  probabilities: Float32Array;
}

interface DrawdownPoint {
  time: number;
  equity: number;
  price: number;
}

interface DrawdownEpisode {
  peak: DrawdownPoint;
  trough: DrawdownPoint;
  drawdownPct: number;
  durationMs: number;
  priceChangePct: number;
}

async function main(): Promise<void> {
  const id = argument("id") ?? "regime-flat-2026-04";
  const start = parseDay(requiredArgument("start", "2026-04-22"));
  const end = parseDay(requiredArgument("end", "2026-04-28")) + DAY_MS;
  const maximumExposure = finiteArgument("max-exposure", 100);
  const confidenceExposurePower = finiteArgument(
    "confidence-power",
    HINDSIGHT_ORACLE_CONFIDENCE_EXPOSURE_POWER,
  );
  const confidenceLeverageFloor = finiteArgument(
    "confidence-floor",
    HINDSIGHT_ORACLE_CONFIDENCE_LEVERAGE_FLOOR,
  );
  if (!(maximumExposure > 0 && maximumExposure <= 100)) {
    throw new Error("--max-exposure must be in (0, 100].");
  }
  if (!(confidenceExposurePower >= 0)) {
    throw new Error("--confidence-power must be non-negative.");
  }
  if (!(confidenceLeverageFloor >= 0 && confidenceLeverageFloor <= 1)) {
    throw new Error("--confidence-floor must be in [0, 1].");
  }
  const warmupMs = historicalWarmupSamples(appConfig.strategy, INTERVAL_MS) * INTERVAL_MS;
  const loaded = loadWindow(start, end, warmupMs);
  const cache = await loadOracleCache(start, end, maximumExposure);
  const targetStats = summarizeTargets(
    cache.distributionAt,
    loaded.candles,
    maximumExposure,
    confidenceExposurePower,
    confidenceLeverageFloor,
  );
  const result = await runBotBacktestFromCandles(loaded.candles, {
    config: appConfig.strategy,
    strategy: "hindsight-oracle-1s",
    warmup: loaded.warmup,
    oracleFuture: loaded.oracleFuture,
    maxEquityPoints: loaded.candles.length,
    maxChartCandles: 2_000,
    summaryOnly: true,
    hindsightOracleDistributionAt: cache.distributionAt,
    hindsightOracleConfidenceExposurePower: confidenceExposurePower,
    hindsightOracleConfidenceLeverageFloor: confidenceLeverageFloor,
  });
  const closeDrawdown = maximumDrawdown(result.equityCurve);
  const largestOneSecondLosses = result.equityCurve
    .slice(1)
    .map((point, index) => {
      const previous = result.equityCurve[index]!;
      return {
        time: point.time,
        equityBefore: previous.equity,
        equityAfter: point.equity,
        equityChangePct: previous.equity > 0
          ? (point.equity / previous.equity - 1) * 100
          : 0,
        priceChangePct: previous.price > 0
          ? (point.price / previous.price - 1) * 100
          : 0,
      };
    })
    .sort((left, right) => left.equityChangePct - right.equityChangePct)
    .slice(0, 20);
  const report = {
    id,
    startTime: start,
    endTime: end,
    maximumExposure,
    confidenceExposurePower,
    confidenceLeverageFloor,
    cacheNamespace: cache.namespace,
    generatedAt: new Date().toISOString(),
    summary: result.summary,
    allTickDrawdownPct: result.summary.maxDrawdownPct,
    closeDrawdown,
    intrasecondDrawdownGapPct: result.summary.maxDrawdownPct - closeDrawdown.drawdownPct,
    targetStats,
    largestOneSecondLosses,
  };
  fs.mkdirSync(DEFAULT_OUTPUT_ROOT, { recursive: true });
  const output = path.resolve(
    argument("output")
      ?? path.join(
        DEFAULT_OUTPUT_ROOT,
        `${id}-max-${maximumExposure}-confidence-${confidenceExposurePower}`
          + `-floor-${confidenceLeverageFloor}.json`,
      ),
  );
  fs.writeFileSync(output, `${JSON.stringify(report, null, 2)}\n`);
  console.log(JSON.stringify(report, null, 2));
  console.log(`REPORT ${output}`);
}

async function loadOracleCache(
  start: number,
  end: number,
  maximumExposure: number,
): Promise<{
  namespace: string;
  distributionAt: (timestamp: number) => ExposureValueOracleActionDistribution | null;
}> {
  const namespaceDirectory = findCacheNamespace(isoDay(start));
  const references = new Map<string, CachedOracleShard>();
  let fullGrid: number[] | undefined;
  for (let day = utcDay(start); day < end; day += DAY_MS) {
    const date = isoDay(day);
    const referenceFile = path.join(namespaceDirectory, `${date}.json`);
    if (!fs.existsSync(referenceFile)) throw new Error(`Missing cached oracle day ${date}.`);
    const parsed = JSON.parse(fs.readFileSync(referenceFile, "utf8"));
    const grid = parsed.metadata?.contract?.usableGrid as number[] | undefined;
    if (!grid?.length) throw new Error(`Cached oracle ${date} has no usable grid metadata.`);
    fullGrid ??= grid;
    if (grid.length !== fullGrid.length || grid.some((value, index) => value !== fullGrid![index])) {
      throw new Error(`Cached oracle grid changed on ${date}.`);
    }
    const { payload } = await readReferencedPayload(referenceFile, false);
    const probabilities = payload.byteOffset % Float32Array.BYTES_PER_ELEMENT === 0
      ? new Float32Array(
          payload.buffer,
          payload.byteOffset,
          payload.byteLength / Float32Array.BYTES_PER_ELEMENT,
        )
      : new Float32Array(payload.buffer.slice(
          payload.byteOffset,
          payload.byteOffset + payload.byteLength,
        ));
    references.set(date, {
      start: parsed.sequence.start,
      step: parsed.sequence.step,
      count: parsed.sequence.count,
      columns: parsed.layout.columns,
      probabilities,
    });
  }
  const indexes = fullGrid!
    .map((exposure, index) => ({ exposure, index }))
    .filter(({ exposure }) => Math.abs(exposure) <= maximumExposure + 1e-12);
  if (indexes.length < 2) throw new Error("Exposure cap leaves fewer than two oracle cells.");
  const grid = Float64Array.from(indexes, ({ exposure }) => exposure);
  return {
    namespace: path.relative(CACHE_ROOT, namespaceDirectory).replaceAll("\\", "/"),
    distributionAt(timestamp: number): ExposureValueOracleActionDistribution | null {
      const shard = references.get(isoDay(timestamp));
      if (!shard) return null;
      const elapsed = timestamp - shard.start;
      if (elapsed < 0 || elapsed % shard.step !== 0) return null;
      const row = elapsed / shard.step;
      if (row < 0 || row >= shard.count) return null;
      const probabilities = Float64Array.from(
        indexes,
        ({ index }) => shard.probabilities[row * shard.columns + index]!,
      );
      const total = probabilities.reduce((sum, probability) => sum + probability, 0);
      if (!(total > 0)) return null;
      let mean = 0;
      let secondMoment = 0;
      let entropy = 0;
      let modalIndex = 0;
      let feasibleActionCount = 0;
      for (let index = 0; index < probabilities.length; index += 1) {
        probabilities[index] /= total;
        const probability = probabilities[index]!;
        if (probability > probabilities[modalIndex]!) modalIndex = index;
        if (probability > 0) {
          feasibleActionCount += 1;
          entropy -= probability * Math.log(probability);
        }
        mean += probability * grid[index]!;
        secondMoment += probability * grid[index]! ** 2;
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
    },
  };
}

function findCacheNamespace(date: string): string {
  const candidates = fs.readdirSync(CACHE_ROOT, { withFileTypes: true })
    .filter((entry) => entry.isDirectory() && entry.name.startsWith("hindsight-bot-"))
    .map((entry) => path.join(CACHE_ROOT, entry.name))
    .filter((directory) => fs.existsSync(path.join(directory, `${date}.json`)));
  const match = candidates.find((directory) => {
    const reference = JSON.parse(fs.readFileSync(path.join(directory, `${date}.json`), "utf8"));
    return reference.metadata?.contract?.options?.holdingPeriodSteps === 60
      && reference.metadata?.contract?.options?.decisionDelaySteps === 60
      && reference.metadata?.contract?.options?.valueHorizonSteps === 3600;
  });
  if (!match) throw new Error(`No compatible hindsight oracle cache contains ${date}.`);
  return match;
}

function summarizeTargets(
  distributionAt: (timestamp: number) => ExposureValueOracleActionDistribution | null,
  candles: readonly Candle[],
  maximumExposure: number,
  confidenceExposurePower: number,
  confidenceLeverageFloor: number,
): Record<string, number> {
  let currentExposure = 0;
  let decisions = 0;
  let reversals = 0;
  let totalAbsoluteExposure = 0;
  let totalAbsoluteDelta = 0;
  let totalConfidence = 0;
  let aboveHalfCap = 0;
  let aboveEightyPercentCap = 0;
  for (const candle of candles) {
    const distribution = distributionAt(candle.closeTime);
    if (!distribution) continue;
    const decision = hindsightOracleTargetDecision(
      distribution,
      currentExposure,
      (appConfig.strategy.feeBps + appConfig.strategy.positionRisk.marketSlippageBps) / 10_000,
    );
    const target = confidenceConditionedHindsightOracleExposure(
      decision.targetExposure,
      decision.confidence,
      maximumExposure,
      confidenceExposurePower,
      confidenceLeverageFloor,
    );
    if (currentExposure !== 0 && target !== 0 && Math.sign(currentExposure) !== Math.sign(target)) {
      reversals += 1;
    }
    const absolute = Math.abs(target);
    totalAbsoluteExposure += absolute;
    totalAbsoluteDelta += Math.abs(target - currentExposure);
    totalConfidence += decision.confidence;
    if (absolute >= maximumExposure * 0.5) aboveHalfCap += 1;
    if (absolute >= maximumExposure * 0.8) aboveEightyPercentCap += 1;
    currentExposure = target;
    decisions += 1;
  }
  return {
    decisions,
    reversals,
    reversalsPerDay: reversals / (candles.length * INTERVAL_MS / DAY_MS),
    meanAbsoluteExposure: decisions > 0 ? totalAbsoluteExposure / decisions : 0,
    meanAbsoluteDelta: decisions > 0 ? totalAbsoluteDelta / decisions : 0,
    meanConfidence: decisions > 0 ? totalConfidence / decisions : 0,
    fractionAboveHalfCap: decisions > 0 ? aboveHalfCap / decisions : 0,
    fractionAboveEightyPercentCap: decisions > 0 ? aboveEightyPercentCap / decisions : 0,
  };
}

function maximumDrawdown(points: readonly DrawdownPoint[]): DrawdownEpisode {
  let peak = points[0]!;
  let bestPeak = peak;
  let trough = peak;
  let drawdownPct = 0;
  for (const point of points) {
    if (point.equity > peak.equity) peak = point;
    const drawdown = peak.equity > 0 ? (peak.equity - point.equity) / peak.equity * 100 : 0;
    if (drawdown > drawdownPct) {
      drawdownPct = drawdown;
      bestPeak = peak;
      trough = point;
    }
  }
  return {
    peak: bestPeak,
    trough,
    drawdownPct,
    durationMs: trough.time - bestPeak.time,
    priceChangePct: bestPeak.price > 0 ? (trough.price / bestPeak.price - 1) * 100 : 0,
  };
}

function loadWindow(start: number, end: number, warmupMs: number): {
  warmup: Candle[];
  candles: Candle[];
  oracleFuture: Candle[];
} {
  const warmup: Candle[] = [];
  const candles: Candle[] = [];
  const oracleFuture: Candle[] = [];
  const loadStart = start - warmupMs;
  const loadEnd = end + ORACLE_FUTURE_MS;
  for (let day = utcDay(loadStart); day < loadEnd; day += DAY_MS) {
    const file = path.join(HISTORY_ROOT, `${isoDay(day)}.json`);
    if (!fs.existsSync(file)) continue;
    for (const candle of readCandleShardReferenceSync(file)) {
      if (candle.openTime < loadStart || candle.openTime >= loadEnd) continue;
      if (candle.openTime < start) warmup.push(candle);
      else if (candle.openTime < end) candles.push(candle);
      else oracleFuture.push(candle);
    }
  }
  if (candles.length !== (end - start) / INTERVAL_MS) {
    throw new Error(`Expected ${(end - start) / INTERVAL_MS} measured candles, got ${candles.length}.`);
  }
  return { warmup, candles, oracleFuture };
}

function argument(name: string): string | undefined {
  const prefix = `--${name}=`;
  return process.argv.find((value) => value.startsWith(prefix))?.slice(prefix.length);
}

function requiredArgument(name: string, fallback: string): string {
  return argument(name) ?? fallback;
}

function finiteArgument(name: string, fallback: number): number {
  const value = Number(argument(name) ?? fallback);
  if (!Number.isFinite(value)) throw new Error(`--${name} must be finite.`);
  return value;
}

function parseDay(value: string): number {
  const parsed = Date.parse(`${value}T00:00:00.000Z`);
  if (!Number.isFinite(parsed)) throw new Error(`Invalid UTC day: ${value}.`);
  return parsed;
}

function utcDay(timestamp: number): number {
  const date = new Date(timestamp);
  return Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate());
}

function isoDay(timestamp: number): string {
  return new Date(timestamp).toISOString().slice(0, 10);
}

void main().catch((error: unknown) => {
  console.error(error);
  process.exitCode = 1;
});
