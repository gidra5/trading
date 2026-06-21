import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  calculateRiskAdjustedMetrics,
  runBacktestFromCandles,
  type BacktestResult,
  type Candle,
  type EquityPoint,
  type StrategyAlgorithm,
  type StrategyConfig,
} from "../packages/bot-algo/src/index.js";

type BenchmarkMode = "days" | "year" | "random-lengths" | "grid-search" | "portfolio";

interface BenchmarkArgs {
  mode: BenchmarkMode;
  symbol: string;
  interval: string;
  days: number;
  startingQuote: number;
  leverage: number;
  cooldownSec: number;
  randomSampleCount: number;
  randomMinWindowDays: number;
  randomMaxWindowDays: number;
  randomLookbackDays: number;
  gridFolds: number;
  gridLimit: number;
  portfolioLookbackCandles: number;
  portfolioRebalanceCandles: number;
  portfolioGrossLeverage: number;
  portfolioTargetVolPct: number;
  seed: number;
  includeRandomSign: boolean;
  only?: string;
  symbols?: string[];
}

interface BenchmarkCase {
  label: string;
  algorithm: StrategyAlgorithm;
  config?: Partial<StrategyConfig>;
}

interface CandleWindow {
  index: number;
  label: string;
  candles: Candle[];
}

interface BenchmarkMetrics {
  returnPct: number;
  netPnl: number;
  maxDrawdownPct: number;
  tradeCount: number;
  winRate: number;
  riskAdjustedReturn: number | undefined;
  sharpeRatio: number | undefined;
  perfectCapturePct: number | undefined;
}

interface SingleBenchmarkRow extends BenchmarkMetrics {
  strategy: string;
}

interface RandomBenchmarkRow {
  strategy: string;
  sampleCount: number;
  profitableSamples: number;
  avgReturnPct: number;
  avgNetPnl: number;
  avgNetPnlPerDay: number;
  avgMaxDrawdownPct: number;
  avgTradeCount: number;
  avgRiskAdjustedReturn: number | undefined;
  avgSharpeRatio: number | undefined;
  avgPerfectCapturePct: number | undefined;
  bestReturnPct: number;
  worstReturnPct: number;
}

interface GridCandidate {
  label: string;
  algorithm: StrategyAlgorithm;
  config: Partial<StrategyConfig>;
}

interface GridSearchRow {
  rank: number;
  candidate: string;
  algorithm: StrategyAlgorithm;
  folds: number;
  profitableFolds: number;
  avgReturnPct: number;
  avgRiskAdjustedReturn: number | undefined;
  avgSharpeRatio: number | undefined;
  avgMaxDrawdownPct: number;
  avgTradeCount: number;
  worstReturnPct: number;
  config: string;
}

interface PortfolioSeries {
  label: string;
  points: EquityPoint[];
}

interface ReturnSeries {
  label: string;
  times: number[];
  returns: number[];
}

interface PortfolioMetrics {
  returnPct: number;
  netPnl: number;
  maxDrawdownPct: number;
  riskAdjustedReturn: number | undefined;
  sharpeRatio: number | undefined;
  turnover: number;
  periods: number;
}

interface PortfolioRow extends PortfolioMetrics {
  portfolio: string;
  subject: string;
}

const DAY_MS = 24 * 60 * 60 * 1000;
const YEAR_DAYS = 365;
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const args = parseArgs(process.argv.slice(2));
const files = historicalCandleFiles(args);

if (files.length === 0) {
  throw new Error(
    `No candles found under data/historical/${args.symbol.toLowerCase()}/${args.interval}.`,
  );
}

const cases = selectBenchmarkCases(args.only, args.includeRandomSign);

if (args.mode === "random-lengths") {
  runRandomLengthMode(args, files, cases);
} else if (args.mode === "grid-search") {
  runGridSearchMode(args, files);
} else if (args.mode === "portfolio") {
  runPortfolioMode(args, files, cases);
} else {
  runSingleWindowMode(args, files, cases);
}

function runSingleWindowMode(
  options: BenchmarkArgs,
  allFiles: string[],
  cases: BenchmarkCase[],
): void {
  const selectedFiles =
    options.mode === "year"
      ? allFiles.slice(-YEAR_DAYS)
      : allFiles.slice(-options.days);
  const candles = loadHistoricalCandles(options, selectedFiles);
  assertCandles(candles);

  const results = cases.map((benchmark) => runBenchmark(benchmark, candles, options));
  const rows = results.map((result) => ({
    strategy: result.label,
    ...metricsFromResult(result.result),
  }));

  console.log(singleWindowHeader(options, candles, selectedFiles.length));
  console.log("");
  console.log(singleBenchmarkTable(rows));
}

function runRandomLengthMode(
  options: BenchmarkArgs,
  allFiles: string[],
  cases: BenchmarkCase[],
): void {
  const lookbackFiles = allFiles.slice(
    -Math.ceil(options.randomLookbackDays + options.randomMaxWindowDays + 2),
  );
  const candles = loadHistoricalCandles(options, lookbackFiles);
  assertCandles(candles);

  const windows = createRandomLengthWindows(candles, options);
  const rows = cases.map((benchmark) => runRandomLengthBenchmark(benchmark, windows, options));

  console.log(randomLengthHeader(options, candles, windows));
  console.log("");
  console.log(randomBenchmarkTable(rows));
}

function runGridSearchMode(options: BenchmarkArgs, allFiles: string[]): void {
  const selectedFiles = allFiles.slice(-options.days);
  const candles = loadHistoricalCandles(options, selectedFiles);
  assertCandles(candles);

  const folds = splitCandlesIntoFolds(candles, options.gridFolds);
  const candidates = createGridCandidates();
  const rows = candidates
    .map((candidate) => runGridCandidate(candidate, folds, options))
    .sort(compareGridRows)
    .map((row, index) => ({ ...row, rank: index + 1 }));

  console.log(gridSearchHeader(options, candles, folds, candidates.length));
  console.log("");
  console.log(gridSearchTable(rows.slice(0, options.gridLimit)));
}

function runGridCandidate(
  candidate: GridCandidate,
  folds: Candle[][],
  options: BenchmarkArgs,
): GridSearchRow {
  console.error(`Grid candidate ${candidate.label} on ${folds.length} folds...`);
  const results = folds.map((fold) =>
    runBenchmark(
      {
        label: candidate.label,
        algorithm: candidate.algorithm,
        config: candidate.config,
      },
      fold,
      options,
      false,
    ),
  );
  const metrics = results.map((result) => metricsFromResult(result.result));

  return {
    rank: 0,
    candidate: candidate.label,
    algorithm: candidate.algorithm,
    folds: folds.length,
    profitableFolds: metrics.filter((metric) => metric.netPnl > 0).length,
    avgReturnPct: average(metrics.map((metric) => metric.returnPct)),
    avgRiskAdjustedReturn: averageDefined(
      metrics.map((metric) => metric.riskAdjustedReturn),
    ),
    avgSharpeRatio: averageDefined(metrics.map((metric) => metric.sharpeRatio)),
    avgMaxDrawdownPct: average(metrics.map((metric) => metric.maxDrawdownPct)),
    avgTradeCount: average(metrics.map((metric) => metric.tradeCount)),
    worstReturnPct: Math.min(...metrics.map((metric) => metric.returnPct)),
    config: formatGridConfig(candidate.config),
  };
}

function createGridCandidates(): GridCandidate[] {
  const candidates: GridCandidate[] = [];

  for (const fastWindow of [12, 24]) {
    for (const slowWindow of [72, 144, 288]) {
      for (const entryThresholdBps of [12, 24]) {
        for (const targetExposurePct of [0.2, 0.35]) {
          candidates.push({
            label: `Trend f${fastWindow}/s${slowWindow}/e${entryThresholdBps}/x${targetExposurePct}`,
            algorithm: "trend-following",
            config: {
              trendFollowing: {
                fastWindow,
                slowWindow,
                volatilityWindow: Math.max(48, Math.floor(slowWindow / 2)),
                entryThresholdBps,
                exitThresholdBps: Math.max(4, Math.floor(entryThresholdBps / 3)),
                targetExposurePct,
              },
            },
          });
        }
      }
    }
  }

  for (const lookbackWindow of [96, 288]) {
    for (const breakoutThresholdBps of [12, 24]) {
      for (const targetExposurePct of [0.2, 0.35]) {
        candidates.push({
          label: `Breakout l${lookbackWindow}/b${breakoutThresholdBps}/x${targetExposurePct}`,
          algorithm: "volatility-breakout",
          config: {
            volatilityBreakout: {
              lookbackWindow,
              breakoutThresholdBps,
              exitThresholdBps: Math.max(4, Math.floor(breakoutThresholdBps / 3)),
              targetExposurePct,
            },
          },
        });
      }
    }
  }

  for (const window of [48, 96]) {
    for (const entryZScore of [1.8, 2.4]) {
      for (const maxTrendBps of [30, 60]) {
        for (const targetExposurePct of [0.15, 0.25]) {
          candidates.push({
            label: `Reversion w${window}/z${entryZScore}/t${maxTrendBps}/x${targetExposurePct}`,
            algorithm: "mean-reversion",
            config: {
              meanReversion: {
                window,
                trendWindow: window * 3,
                entryZScore,
                exitZScore: 0.25,
                maxTrendBps,
                targetExposurePct,
              },
            },
          });
        }
      }
    }
  }

  return candidates;
}

function splitCandlesIntoFolds(candles: Candle[], requestedFolds: number): Candle[][] {
  const foldCount = clampInt(requestedFolds, 2, Math.min(10, candles.length));
  const foldSize = Math.floor(candles.length / foldCount);
  const folds: Candle[][] = [];
  for (let index = 0; index < foldCount; index += 1) {
    const start = index * foldSize;
    const end = index === foldCount - 1 ? candles.length : start + foldSize;
    const fold = candles.slice(start, end);
    if (fold.length > 0) {
      folds.push(fold);
    }
  }
  return folds;
}

function runPortfolioMode(
  options: BenchmarkArgs,
  allFiles: string[],
  cases: BenchmarkCase[],
): void {
  const selectedFiles = allFiles.slice(-options.days);
  const candles = loadHistoricalCandles(options, selectedFiles);
  assertCandles(candles);

  const strategyRows = runStrategyPortfolioExperiments(options, candles, cases);
  console.log(portfolioHeader(options, candles, "strategy ensemble"));
  console.log("");
  console.log(portfolioTable(strategyRows));

  const symbolRows = runSymbolPortfolioExperiments(options);
  console.log("");
  if (symbolRows.length === 0) {
    const symbols = discoverHistoricalSymbols(options.interval);
    console.log(
      `Multi-symbol portfolio skipped: need at least 2 cached ${options.interval} symbols, found ${symbols.length} (${symbols.join(", ") || "none"}).`,
    );
  } else {
    console.log(portfolioHeader(options, candles, "multi-symbol allocation"));
    console.log("");
    console.log(portfolioTable(symbolRows));
  }
}

function runStrategyPortfolioExperiments(
  options: BenchmarkArgs,
  candles: Candle[],
  cases: BenchmarkCase[],
): PortfolioRow[] {
  const series: PortfolioSeries[] = cases.map((benchmark) => {
    const result = runBenchmark(benchmark, candles, options, true, candles.length).result;
    return {
      label: benchmark.label,
      points: result.equityCurve,
    };
  });
  const returns = portfolioSeriesToReturns(series);
  if (returns.length === 0) {
    return [];
  }

  return [
    simulatePortfolio("Equal strategy mix", "strategies", returns, equalLongOnlyWeights, options, 0),
    simulatePortfolio("Inverse-vol strategy mix", "strategies", returns, inverseVolWeights, options, 0),
    simulatePortfolio("Rolling winner strategy mix", "strategies", returns, rollingWinnerWeights, options, 0),
    simulatePortfolio("Drawdown-guard strategy mix", "strategies", returns, drawdownGuardWeights, options, 0),
  ];
}

function runSymbolPortfolioExperiments(options: BenchmarkArgs): PortfolioRow[] {
  const symbols = (options.symbols?.length ? options.symbols : discoverHistoricalSymbols(options.interval))
    .map((symbol) => symbol.toUpperCase());
  const uniqueSymbols = [...new Set(symbols)];
  if (uniqueSymbols.length < 2) {
    return [];
  }

  const loaded = uniqueSymbols
    .map((symbol) => {
      const symbolOptions = {
        ...options,
        symbol,
      };
      const files = historicalCandleFiles(symbolOptions).slice(-options.days);
      if (files.length === 0) {
        console.error(`Skipping ${symbol}: no cached ${options.interval} candles.`);
      }
      return {
        symbol,
        candles: loadHistoricalCandles(symbolOptions, files),
      };
    })
    .filter((entry) => entry.candles.length > 1);

  if (loaded.length < 2) {
    return [];
  }

  const returns = alignedSymbolReturns(loaded);
  if (returns.length < 2 || returns[0].returns.length === 0) {
    return [];
  }

  const feeAndSlippageRate = (7.5 + 5) / 10_000;
  return [
    simulatePortfolio("Equal symbol basket", "symbols", returns, equalLongOnlyWeights, options, feeAndSlippageRate),
    simulatePortfolio("Inverse-vol symbol basket", "symbols", returns, inverseVolWeights, options, feeAndSlippageRate),
    simulatePortfolio("Momentum long/short basket", "symbols", returns, momentumLongShortWeights, options, feeAndSlippageRate),
    simulatePortfolio("Vol-target equal basket", "symbols", returns, volTargetEqualWeights, options, feeAndSlippageRate),
  ];
}

function portfolioSeriesToReturns(series: PortfolioSeries[]): ReturnSeries[] {
  return series
    .map((entry) => {
      const returns: number[] = [];
      const times: number[] = [];
      for (let index = 1; index < entry.points.length; index += 1) {
        const previous = entry.points[index - 1];
        const current = entry.points[index];
        if (previous.equity <= 0) {
          returns.push(-1);
        } else {
          returns.push((current.equity - previous.equity) / previous.equity);
        }
        times.push(current.time);
      }
      return {
        label: entry.label,
        times,
        returns,
      };
    })
    .filter((entry) => entry.returns.length > 0);
}

function alignedSymbolReturns(
  entries: Array<{ symbol: string; candles: Candle[] }>,
): ReturnSeries[] {
  const closeMaps = entries.map((entry) => ({
    symbol: entry.symbol,
    closes: new Map(entry.candles.map((candle) => [candle.closeTime, candle.close])),
  }));
  const commonTimes = [...closeMaps[0].closes.keys()]
    .filter((time) => closeMaps.every((entry) => entry.closes.has(time)))
    .sort((left, right) => left - right);

  if (commonTimes.length < 2) {
    return [];
  }

  return closeMaps.map((entry) => {
    const returns: number[] = [];
    const times: number[] = [];
    for (let index = 1; index < commonTimes.length; index += 1) {
      const previous = entry.closes.get(commonTimes[index - 1]) ?? 0;
      const current = entry.closes.get(commonTimes[index]) ?? 0;
      returns.push(previous > 0 ? (current - previous) / previous : 0);
      times.push(commonTimes[index]);
    }
    return {
      label: entry.symbol,
      times,
      returns,
    };
  });
}

type PortfolioAllocator = (
  periodIndex: number,
  series: ReturnSeries[],
  previousWeights: number[],
  options: BenchmarkArgs,
) => number[];

function simulatePortfolio(
  name: string,
  subject: string,
  series: ReturnSeries[],
  allocator: PortfolioAllocator,
  options: BenchmarkArgs,
  rebalanceCostRate: number,
): PortfolioRow {
  const periods = Math.min(...series.map((entry) => entry.returns.length));
  const equityCurve: EquityPoint[] = [
    {
      time: series[0].times[0] ?? Date.now(),
      equity: options.startingQuote,
      price: 1,
    },
  ];
  let equity = options.startingQuote;
  let weights = new Array(series.length).fill(0);
  let turnover = 0;

  for (let periodIndex = 0; periodIndex < periods; periodIndex += 1) {
    if (periodIndex % options.portfolioRebalanceCandles === 0) {
      const nextWeights = allocator(periodIndex, series, weights, options);
      const normalized = normalizeWeights(nextWeights, options.portfolioGrossLeverage);
      const periodTurnover = normalized.reduce(
        (total, weight, index) => total + Math.abs(weight - (weights[index] ?? 0)),
        0,
      );
      turnover += periodTurnover;
      if (rebalanceCostRate > 0) {
        equity -= equity * periodTurnover * rebalanceCostRate;
      }
      weights = normalized;
    }

    const portfolioReturn = series.reduce(
      (total, entry, index) => total + (weights[index] ?? 0) * entry.returns[periodIndex],
      0,
    );
    equity *= 1 + portfolioReturn;
    equityCurve.push({
      time: series[0].times[periodIndex],
      equity,
      price: 1,
    });
  }

  const maxDrawdownPct = calculateMaxDrawdownPct(equityCurve);
  const netPnl = equity - options.startingQuote;
  const returnPct = (netPnl / options.startingQuote) * 100;
  const riskMetrics = calculateRiskAdjustedMetrics(equityCurve, returnPct, maxDrawdownPct);

  return {
    portfolio: name,
    subject,
    returnPct,
    netPnl,
    maxDrawdownPct,
    riskAdjustedReturn: riskMetrics.riskAdjustedReturn,
    sharpeRatio: riskMetrics.sharpeRatio,
    turnover,
    periods,
  };
}

function equalLongOnlyWeights(
  _periodIndex: number,
  series: ReturnSeries[],
): number[] {
  return new Array(series.length).fill(1 / series.length);
}

function inverseVolWeights(
  periodIndex: number,
  series: ReturnSeries[],
  _previousWeights: number[],
  options: BenchmarkArgs,
): number[] {
  const lookback = recentLookback(periodIndex, options);
  const inverseVols = series.map((entry) => {
    const values = entry.returns.slice(Math.max(0, periodIndex - lookback), periodIndex);
    const volatility = sampleStdDev(values);
    return volatility > 0 ? 1 / volatility : 1;
  });
  return normalizeLongOnly(inverseVols);
}

function rollingWinnerWeights(
  periodIndex: number,
  series: ReturnSeries[],
  _previousWeights: number[],
  options: BenchmarkArgs,
): number[] {
  const lookback = recentLookback(periodIndex, options);
  const scores = series.map((entry) =>
    Math.max(
      0,
      entry.returns
        .slice(Math.max(0, periodIndex - lookback), periodIndex)
        .reduce((total, value) => total + value, 0),
    ),
  );
  return scores.some((score) => score > 0) ? normalizeLongOnly(scores) : equalLongOnlyWeights(periodIndex, series);
}

function drawdownGuardWeights(
  periodIndex: number,
  series: ReturnSeries[],
  _previousWeights: number[],
  options: BenchmarkArgs,
): number[] {
  const lookback = recentLookback(periodIndex, options);
  const eligible = series.map((entry) => {
    const values = entry.returns.slice(Math.max(0, periodIndex - lookback), periodIndex);
    const cumulative = values.reduce((total, value) => total * (1 + value), 1) - 1;
    return cumulative >= 0 ? 1 : 0;
  });
  return eligible.some((value) => value > 0) ? normalizeLongOnly(eligible) : new Array(series.length).fill(0);
}

function momentumLongShortWeights(
  periodIndex: number,
  series: ReturnSeries[],
  _previousWeights: number[],
  options: BenchmarkArgs,
): number[] {
  const lookback = recentLookback(periodIndex, options);
  const scores = series.map((entry, index) => ({
    index,
    score: entry.returns
      .slice(Math.max(0, periodIndex - lookback), periodIndex)
      .reduce((total, value) => total + value, 0),
  }));
  const sorted = [...scores].sort((left, right) => right.score - left.score);
  const half = Math.max(1, Math.floor(series.length / 2));
  const weights = new Array(series.length).fill(0);
  for (const entry of sorted.slice(0, half)) {
    weights[entry.index] = 1 / half;
  }
  for (const entry of sorted.slice(-half)) {
    weights[entry.index] -= 1 / half;
  }
  return weights;
}

function volTargetEqualWeights(
  periodIndex: number,
  series: ReturnSeries[],
  _previousWeights: number[],
  options: BenchmarkArgs,
): number[] {
  const baseWeights = equalLongOnlyWeights(periodIndex, series);
  const lookback = recentLookback(periodIndex, options);
  const portfolioReturns: number[] = [];
  for (let index = Math.max(0, periodIndex - lookback); index < periodIndex; index += 1) {
    portfolioReturns.push(
      series.reduce((total, entry, seriesIndex) => total + baseWeights[seriesIndex] * entry.returns[index], 0),
    );
  }
  const realizedVol = annualizedVolPct(portfolioReturns, series[0].times);
  const scale =
    realizedVol > 0 ? Math.min(options.portfolioGrossLeverage, options.portfolioTargetVolPct / realizedVol) : 1;
  return baseWeights.map((weight) => weight * scale);
}

function runBenchmark(
  benchmark: BenchmarkCase,
  sourceCandles: Candle[],
  options: BenchmarkArgs,
  log = true,
  maxEquityPoints?: number,
): { label: string; result: BacktestResult; elapsedMs: number } {
  const maxPositionQuote = options.startingQuote * options.leverage * 0.85;
  const startedAt = Date.now();
  if (log) {
    console.error(`Running ${benchmark.label} on ${sourceCandles.length.toLocaleString()} candles...`);
  }
  const result = runBacktestFromCandles(sourceCandles, {
    config: {
      symbol: options.symbol.toUpperCase(),
      algorithm: benchmark.algorithm,
      startingQuote: options.startingQuote,
      maxLeverage: options.leverage,
      maxPositionQuote,
      cooldownMs: options.cooldownSec * 1000,
      benchmarkRandomSeed: options.seed,
      maxOpenOrders: 3,
      ...benchmark.config,
    },
    ...(maxEquityPoints ? { maxEquityPoints } : {}),
    maxReturnedOrders: 0,
    maxReturnedFills: 0,
  });

  const elapsedMs = Date.now() - startedAt;
  if (log) {
    console.error(`Finished ${benchmark.label} in ${elapsedMs.toLocaleString()}ms.`);
  }

  return {
    label: benchmark.label,
    result,
    elapsedMs,
  };
}

function runRandomLengthBenchmark(
  benchmark: BenchmarkCase,
  windows: CandleWindow[],
  options: BenchmarkArgs,
): RandomBenchmarkRow {
  const startedAt = Date.now();
  console.error(`Running ${benchmark.label} on ${windows.length.toLocaleString()} random windows...`);
  const samples = windows.map((window) => {
    const result = runBenchmark(benchmark, window.candles, options, false).result;
    const durationDays = Math.max(
      1 / 24 / 60,
      (window.candles[window.candles.length - 1].closeTime - window.candles[0].openTime) /
        DAY_MS,
    );
    return {
      result,
      durationDays,
    };
  });
  const metrics = samples.map((sample) => metricsFromResult(sample.result));
  const elapsedMs = Date.now() - startedAt;
  console.error(`Finished ${benchmark.label} random aggregate in ${elapsedMs.toLocaleString()}ms.`);

  return {
    strategy: benchmark.label,
    sampleCount: samples.length,
    profitableSamples: samples.filter((sample) => sample.result.summary.netPnl > 0).length,
    avgReturnPct: average(metrics.map((metric) => metric.returnPct)),
    avgNetPnl: average(metrics.map((metric) => metric.netPnl)),
    avgNetPnlPerDay: average(
      samples.map((sample) => sample.result.summary.netPnl / sample.durationDays),
    ),
    avgMaxDrawdownPct: average(metrics.map((metric) => metric.maxDrawdownPct)),
    avgTradeCount: average(metrics.map((metric) => metric.tradeCount)),
    avgRiskAdjustedReturn: averageDefined(
      metrics.map((metric) => metric.riskAdjustedReturn),
    ),
    avgSharpeRatio: averageDefined(metrics.map((metric) => metric.sharpeRatio)),
    avgPerfectCapturePct: averageDefined(
      metrics.map((metric) => metric.perfectCapturePct),
    ),
    bestReturnPct: Math.max(...metrics.map((metric) => metric.returnPct)),
    worstReturnPct: Math.min(...metrics.map((metric) => metric.returnPct)),
  };
}

function metricsFromResult(result: BacktestResult): BenchmarkMetrics {
  const summary = result.summary;
  return {
    returnPct: summary.returnPct,
    netPnl: summary.netPnl,
    maxDrawdownPct: summary.maxDrawdownPct,
    tradeCount: summary.tradeCount,
    winRate: summary.winRate,
    riskAdjustedReturn: summary.riskAdjustedReturn,
    sharpeRatio: summary.sharpeRatio,
    perfectCapturePct: summary.perfectMarginCapturePct,
  };
}

function createRandomLengthWindows(
  candles: Candle[],
  options: BenchmarkArgs,
): CandleWindow[] {
  const rng = mulberry32(options.seed);
  const firstTime = candles[0].openTime;
  const latestTime = candles[candles.length - 1].closeTime;
  const lookbackStart = Math.max(firstTime, latestTime - options.randomLookbackDays * DAY_MS);
  const minWindowMs = options.randomMinWindowDays * DAY_MS;
  const maxWindowMs = options.randomMaxWindowDays * DAY_MS;
  const maxStart = latestTime - minWindowMs;

  if (maxStart <= lookbackStart) {
    throw new Error("Not enough local candle history for random-length benchmark windows.");
  }

  const windows: CandleWindow[] = [];
  for (let index = 0; index < options.randomSampleCount; index += 1) {
    const durationMs = randomInt(rng, minWindowMs, maxWindowMs);
    const startUpperBound = Math.max(lookbackStart, latestTime - durationMs);
    const startTime = randomInt(rng, lookbackStart, startUpperBound);
    const endTime = startTime + durationMs;
    const startIndex = lowerBoundCandleOpenTime(candles, startTime);
    const endIndex = upperBoundCandleOpenTime(candles, endTime);
    const windowCandles = candles.slice(startIndex, endIndex);
    if (windowCandles.length === 0) {
      index -= 1;
      continue;
    }
    windows.push({
      index,
      label: `${formatDate(windowCandles[0].openTime)} to ${formatDate(
        windowCandles[windowCandles.length - 1].closeTime,
      )}`,
      candles: windowCandles,
    });
  }

  return windows;
}

function historicalCandleFiles(options: BenchmarkArgs): string[] {
  const dir = historicalCandleDir(options);
  if (!fs.existsSync(dir)) {
    return [];
  }
  return fs
    .readdirSync(dir, { withFileTypes: true })
    .filter((entry) => entry.isFile() && entry.name.endsWith(".jsonl"))
    .map((entry) => entry.name)
    .sort();
}

function loadHistoricalCandles(options: BenchmarkArgs, files: string[]): Candle[] {
  const dir = historicalCandleDir(options);
  const candles: Candle[] = [];
  for (const file of files) {
    const content = fs.readFileSync(path.join(dir, file), "utf8");
    for (const line of content.split("\n")) {
      const trimmed = line.trim();
      if (!trimmed) {
        continue;
      }
      candles.push(JSON.parse(trimmed) as Candle);
    }
  }

  return candles.sort((left, right) => left.openTime - right.openTime);
}

function historicalCandleDir(options: Pick<BenchmarkArgs, "symbol" | "interval">): string {
  return path.join(
    repoRoot,
    "data",
    "historical",
    options.symbol.toLowerCase(),
    options.interval,
  );
}

function parseArgs(argv: string[]): BenchmarkArgs {
  const values = new Map<string, string>();
  for (let index = 0; index < argv.length; index += 1) {
    const key = argv[index];
    if (!key.startsWith("--")) {
      continue;
    }
    const next = argv[index + 1];
    if (next && !next.startsWith("--")) {
      values.set(key.slice(2), next);
      index += 1;
    } else {
      values.set(key.slice(2), "true");
    }
  }

  const mode = parseMode(values.get("mode"));
  return {
    mode,
    symbol: values.get("symbol") ?? "BTCUSDT",
    interval: values.get("interval") ?? "1m",
    days: parsePositiveInt(values.get("days"), 30),
    startingQuote: parsePositiveNumber(values.get("starting-quote"), 10_000),
    leverage: parsePositiveNumber(values.get("leverage"), 3),
    cooldownSec: parsePositiveNumber(values.get("cooldown-sec"), 300),
    randomSampleCount: parsePositiveInt(values.get("samples"), 40),
    randomMinWindowDays: parsePositiveNumber(values.get("min-window-days"), 1),
    randomMaxWindowDays: parsePositiveNumber(values.get("max-window-days"), 30),
    randomLookbackDays: parsePositiveNumber(values.get("lookback-days"), YEAR_DAYS),
    gridFolds: parsePositiveInt(values.get("grid-folds"), 3),
    gridLimit: parsePositiveInt(values.get("grid-limit"), 12),
    portfolioLookbackCandles: parsePositiveInt(values.get("portfolio-lookback-candles"), 720),
    portfolioRebalanceCandles: parsePositiveInt(values.get("portfolio-rebalance-candles"), 60),
    portfolioGrossLeverage: parsePositiveNumber(values.get("portfolio-gross-leverage"), 1),
    portfolioTargetVolPct: parsePositiveNumber(values.get("portfolio-target-vol-pct"), 35),
    seed: parsePositiveInt(values.get("seed"), 1337),
    includeRandomSign: values.has("include-random-sign"),
    only: values.get("only"),
    symbols: parseSymbols(values.get("symbols")),
  };
}

function parseMode(value: string | undefined): BenchmarkMode {
  if (
    value === "year" ||
    value === "random-lengths" ||
    value === "grid-search" ||
    value === "portfolio"
  ) {
    return value;
  }
  return "days";
}

function selectBenchmarkCases(
  only: string | undefined,
  includeRandomSign: boolean,
): BenchmarkCase[] {
  const cases: BenchmarkCase[] = [
    {
      label: "Baseline always flat",
      algorithm: "benchmark-always-flat",
    },
    {
      label: "Baseline always long",
      algorithm: "benchmark-always-long",
    },
    {
      label: "Baseline always short",
      algorithm: "benchmark-always-short",
    },
    {
      label: "Moving average",
      algorithm: "moving-average",
    },
    {
      label: "Legacy valley/peak",
      algorithm: "legacy-valley-peak",
    },
    {
      label: "Trend following L/S",
      algorithm: "trend-following",
    },
    {
      label: "Vol breakout L/S",
      algorithm: "volatility-breakout",
    },
    {
      label: "Mean reversion L/S",
      algorithm: "mean-reversion",
    },
  ];

  if (includeRandomSign || only?.toLowerCase().includes("random")) {
    cases.splice(3, 0, {
      label: "Baseline random sign",
      algorithm: "benchmark-random-sign",
    });
  }

  const selected = only
    ? cases.filter(
        (benchmark) =>
          benchmark.algorithm === only ||
          benchmark.label.toLowerCase().includes(only.toLowerCase()),
      )
    : cases;
  if (selected.length === 0) {
    throw new Error(`No benchmark case matched --only ${only}.`);
  }

  return selected;
}

function singleWindowHeader(
  options: BenchmarkArgs,
  candles: Candle[],
  fileCount: number,
): string {
  const modeLabel = options.mode === "year" ? "last-year" : `last-${options.days}-days`;
  return [
    `Strategy benchmark: ${modeLabel}`,
    `${options.symbol.toUpperCase()} ${options.interval}`,
    `${candles.length.toLocaleString()} candles`,
    `${fileCount.toLocaleString()} day files`,
    `${formatDate(candles[0].openTime)} to ${formatDate(candles[candles.length - 1].closeTime)}`,
    `${options.leverage}x max leverage`,
    benchmarkCapSummary(options),
    `${options.cooldownSec}s cooldown`,
  ].join(", ");
}

function randomLengthHeader(
  options: BenchmarkArgs,
  candles: Candle[],
  windows: CandleWindow[],
): string {
  const firstWindow = windows[0];
  const lastWindow = windows[windows.length - 1];
  return [
    "Strategy benchmark: random-lengths",
    `${options.symbol.toUpperCase()} ${options.interval}`,
    `${windows.length.toLocaleString()} samples`,
    `${options.randomMinWindowDays}-${options.randomMaxWindowDays} day windows`,
    `${options.randomLookbackDays} day lookback`,
    `seed ${options.seed}`,
    `${formatDate(candles[0].openTime)} to ${formatDate(candles[candles.length - 1].closeTime)} cache span`,
    `${firstWindow.label} first sample`,
    `${lastWindow.label} last sample`,
    `${options.leverage}x max leverage`,
    benchmarkCapSummary(options),
    `${options.cooldownSec}s cooldown`,
  ].join(", ");
}

function gridSearchHeader(
  options: BenchmarkArgs,
  candles: Candle[],
  folds: Candle[][],
  candidateCount: number,
): string {
  return [
    "Strategy benchmark: grid-search",
    `${options.symbol.toUpperCase()} ${options.interval}`,
    `${candidateCount.toLocaleString()} candidates`,
    `${folds.length.toLocaleString()} folds`,
    `${candles.length.toLocaleString()} candles`,
    `${formatDate(candles[0].openTime)} to ${formatDate(candles[candles.length - 1].closeTime)}`,
    `${options.leverage}x max leverage`,
    benchmarkCapSummary(options),
    `${options.cooldownSec}s cooldown`,
  ].join(", ");
}

function portfolioHeader(
  options: BenchmarkArgs,
  candles: Candle[],
  subject: string,
): string {
  return [
    `Strategy benchmark: portfolio ${subject}`,
    `${options.symbol.toUpperCase()} ${options.interval}`,
    `${candles.length.toLocaleString()} primary candles`,
    `${formatDate(candles[0].openTime)} to ${formatDate(candles[candles.length - 1].closeTime)}`,
    `${options.portfolioGrossLeverage}x portfolio gross leverage`,
    `${options.portfolioRebalanceCandles.toLocaleString()} candle rebalance`,
    `${options.portfolioLookbackCandles.toLocaleString()} candle lookback`,
  ].join(", ");
}

function singleBenchmarkTable(rows: SingleBenchmarkRow[]): string {
  const header = [
    "Strategy",
    "Return",
    "Net PnL",
    "Max DD",
    "Risk Ret",
    "Sharpe",
    "Trades",
    "Win Rate",
    "Oracle Capture",
  ];
  const tableRows = rows.map((row) => [
    row.strategy,
    `${formatNumber(row.returnPct, 2)}%`,
    formatMoney(row.netPnl),
    `${formatNumber(row.maxDrawdownPct, 2)}%`,
    formatOptionalNumber(row.riskAdjustedReturn, 3),
    formatOptionalNumber(row.sharpeRatio, 3),
    row.tradeCount.toLocaleString(),
    `${formatNumber(row.winRate, 1)}%`,
    row.perfectCapturePct === undefined ? "-" : `${formatNumber(row.perfectCapturePct, 3)}%`,
  ]);

  return markdownTable(header, tableRows);
}

function randomBenchmarkTable(rows: RandomBenchmarkRow[]): string {
  const header = [
    "Strategy",
    "Samples",
    "Profitable",
    "Avg Return",
    "Avg Net PnL",
    "Avg PnL/day",
    "Avg Max DD",
    "Avg Risk Ret",
    "Avg Sharpe",
    "Avg Trades",
    "Avg Capture",
    "Best",
    "Worst",
  ];
  const tableRows = rows.map((row) => [
    row.strategy,
    row.sampleCount.toLocaleString(),
    `${row.profitableSamples}/${row.sampleCount}`,
    `${formatNumber(row.avgReturnPct, 2)}%`,
    formatMoney(row.avgNetPnl),
    formatMoney(row.avgNetPnlPerDay),
    `${formatNumber(row.avgMaxDrawdownPct, 2)}%`,
    formatOptionalNumber(row.avgRiskAdjustedReturn, 3),
    formatOptionalNumber(row.avgSharpeRatio, 3),
    formatNumber(row.avgTradeCount, 1),
    row.avgPerfectCapturePct === undefined ? "-" : `${formatNumber(row.avgPerfectCapturePct, 3)}%`,
    `${formatNumber(row.bestReturnPct, 2)}%`,
    `${formatNumber(row.worstReturnPct, 2)}%`,
  ]);

  return markdownTable(header, tableRows);
}

function gridSearchTable(rows: GridSearchRow[]): string {
  const header = [
    "Rank",
    "Candidate",
    "Algo",
    "Folds",
    "Profitable",
    "Avg Return",
    "Avg Risk Ret",
    "Avg Sharpe",
    "Avg Max DD",
    "Avg Trades",
    "Worst",
    "Config",
  ];
  const tableRows = rows.map((row) => [
    row.rank.toLocaleString(),
    row.candidate,
    row.algorithm,
    row.folds.toLocaleString(),
    `${row.profitableFolds}/${row.folds}`,
    `${formatNumber(row.avgReturnPct, 2)}%`,
    formatOptionalNumber(row.avgRiskAdjustedReturn, 3),
    formatOptionalNumber(row.avgSharpeRatio, 3),
    `${formatNumber(row.avgMaxDrawdownPct, 2)}%`,
    formatNumber(row.avgTradeCount, 1),
    `${formatNumber(row.worstReturnPct, 2)}%`,
    row.config,
  ]);

  return markdownTable(header, tableRows);
}

function portfolioTable(rows: PortfolioRow[]): string {
  const header = [
    "Portfolio",
    "Subject",
    "Return",
    "Net PnL",
    "Max DD",
    "Risk Ret",
    "Sharpe",
    "Turnover",
    "Periods",
  ];
  const tableRows = rows.map((row) => [
    row.portfolio,
    row.subject,
    `${formatNumber(row.returnPct, 2)}%`,
    formatMoney(row.netPnl),
    `${formatNumber(row.maxDrawdownPct, 2)}%`,
    formatOptionalNumber(row.riskAdjustedReturn, 3),
    formatOptionalNumber(row.sharpeRatio, 3),
    formatNumber(row.turnover, 2),
    row.periods.toLocaleString(),
  ]);

  return markdownTable(header, tableRows);
}

function markdownTable(header: string[], rows: string[][]): string {
  return [
    `| ${header.join(" | ")} |`,
    `| ${header.map(() => "---").join(" | ")} |`,
    ...rows.map((row) => `| ${row.join(" | ")} |`),
  ].join("\n");
}

function lowerBoundCandleOpenTime(candles: Candle[], time: number): number {
  let low = 0;
  let high = candles.length;
  while (low < high) {
    const mid = Math.floor((low + high) / 2);
    if (candles[mid].openTime < time) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  return low;
}

function upperBoundCandleOpenTime(candles: Candle[], time: number): number {
  let low = 0;
  let high = candles.length;
  while (low < high) {
    const mid = Math.floor((low + high) / 2);
    if (candles[mid].openTime <= time) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  return low;
}

function randomInt(rng: () => number, min: number, max: number): number {
  if (max <= min) {
    return Math.round(min);
  }
  return Math.round(min + rng() * (max - min));
}

function mulberry32(seed: number): () => number {
  let value = seed >>> 0;
  return () => {
    value += 0x6d2b79f5;
    let t = value;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function average(values: number[]): number {
  if (values.length === 0) {
    return 0;
  }
  return values.reduce((total, value) => total + value, 0) / values.length;
}

function averageDefined(values: Array<number | undefined>): number | undefined {
  const defined = values.filter((value): value is number => Number.isFinite(value));
  return defined.length > 0 ? average(defined) : undefined;
}

function assertCandles(candles: Candle[]): asserts candles is [Candle, ...Candle[]] {
  if (candles.length === 0) {
    throw new Error("Benchmark requires at least one candle.");
  }
}

function parsePositiveInt(value: string | undefined, fallback: number): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? Math.round(parsed) : fallback;
}

function parsePositiveNumber(value: string | undefined, fallback: number): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function parseSymbols(value: string | undefined): string[] | undefined {
  if (!value) {
    return undefined;
  }
  const symbols = value
    .split(",")
    .map((symbol) => symbol.trim().toUpperCase())
    .filter(Boolean);
  return symbols.length > 0 ? symbols : undefined;
}

function clampInt(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, Math.round(value)));
}

function formatDate(value: number): string {
  return new Date(value).toISOString().slice(0, 10);
}

function formatMoney(value: number): string {
  return `$${formatNumber(value, 2)}`;
}

function formatNumber(value: number, digits: number): string {
  return (Number.isFinite(value) ? value : 0).toFixed(digits);
}

function formatOptionalNumber(value: number | undefined, digits: number): string {
  return Number.isFinite(value) ? (value as number).toFixed(digits) : "-";
}

function benchmarkCapSummary(options: BenchmarkArgs): string {
  const configuredCap = options.startingQuote * options.leverage * 0.85;
  const initialShortDebtCap = options.startingQuote * Math.max(0, options.leverage - 1) * 0.98;
  return `target cap ${formatMoney(configuredCap)}, initial short debt cap ${formatMoney(
    Math.min(configuredCap, initialShortDebtCap),
  )}`;
}

function compareGridRows(left: GridSearchRow, right: GridSearchRow): number {
  return (
    compareOptionalDesc(left.avgRiskAdjustedReturn, right.avgRiskAdjustedReturn) ||
    compareOptionalDesc(left.avgSharpeRatio, right.avgSharpeRatio) ||
    right.avgReturnPct - left.avgReturnPct ||
    right.worstReturnPct - left.worstReturnPct
  );
}

function compareOptionalDesc(
  left: number | undefined,
  right: number | undefined,
): number {
  const leftValue = Number.isFinite(left) ? (left as number) : -Infinity;
  const rightValue = Number.isFinite(right) ? (right as number) : -Infinity;
  return rightValue - leftValue;
}

function formatGridConfig(config: Partial<StrategyConfig>): string {
  if (config.trendFollowing) {
    const value = config.trendFollowing;
    return `fast=${value.fastWindow}, slow=${value.slowWindow}, vol=${value.volatilityWindow}, entry=${value.entryThresholdBps}, exit=${value.exitThresholdBps}, exposure=${value.targetExposurePct}`;
  }
  if (config.volatilityBreakout) {
    const value = config.volatilityBreakout;
    return `lookback=${value.lookbackWindow}, breakout=${value.breakoutThresholdBps}, exit=${value.exitThresholdBps}, exposure=${value.targetExposurePct}`;
  }
  if (config.meanReversion) {
    const value = config.meanReversion;
    return `window=${value.window}, trend=${value.trendWindow}, z=${value.entryZScore}, exit=${value.exitZScore}, maxTrend=${value.maxTrendBps}, exposure=${value.targetExposurePct}`;
  }
  return "-";
}

function discoverHistoricalSymbols(interval: string): string[] {
  const root = path.join(repoRoot, "data", "historical");
  if (!fs.existsSync(root)) {
    return [];
  }

  return fs
    .readdirSync(root, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name.toUpperCase())
    .filter((symbol) =>
      fs.existsSync(path.join(root, symbol.toLowerCase(), interval)),
    )
    .sort();
}

function normalizeWeights(weights: number[], grossLimit: number): number[] {
  const gross = weights.reduce((total, weight) => total + Math.abs(weight), 0);
  if (gross <= 0) {
    return weights.map(() => 0);
  }
  const scale = Math.min(Math.max(0, grossLimit), gross) / gross;
  return weights.map((weight) => weight * scale);
}

function normalizeLongOnly(values: number[]): number[] {
  const cleanValues = values.map((value) => (Number.isFinite(value) && value > 0 ? value : 0));
  const total = cleanValues.reduce((sum, value) => sum + value, 0);
  if (total <= 0) {
    return new Array(values.length).fill(1 / Math.max(1, values.length));
  }
  return cleanValues.map((value) => value / total);
}

function recentLookback(periodIndex: number, options: BenchmarkArgs): number {
  return Math.max(1, Math.min(periodIndex, options.portfolioLookbackCandles));
}

function sampleStdDev(values: number[]): number {
  if (values.length < 2) {
    return 0;
  }
  const mean = average(values);
  const variance =
    values.reduce((total, value) => total + (value - mean) ** 2, 0) / (values.length - 1);
  return Math.sqrt(variance);
}

function annualizedVolPct(values: number[], times: number[]): number {
  const deviation = sampleStdDev(values);
  if (deviation <= 0 || values.length < 2) {
    return 0;
  }
  const firstTime = times[0] ?? 0;
  const lastTime = times[Math.min(times.length - 1, values.length - 1)] ?? firstTime;
  const years = lastTime > firstTime ? (lastTime - firstTime) / (YEAR_DAYS * DAY_MS) : 0;
  const periodsPerYear = years > 0 ? values.length / years : values.length;
  return deviation * Math.sqrt(periodsPerYear) * 100;
}

function calculateMaxDrawdownPct(equityCurve: EquityPoint[]): number {
  let peak = equityCurve[0]?.equity ?? 0;
  let maxDrawdown = 0;
  for (const point of equityCurve) {
    peak = Math.max(peak, point.equity);
    if (peak > 0) {
      maxDrawdown = Math.max(maxDrawdown, ((peak - point.equity) / peak) * 100);
    }
  }
  return maxDrawdown;
}
