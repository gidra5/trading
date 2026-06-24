import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  calculateRiskAdjustedMetrics,
  legacyValleyPeakAsymmetricShortFavoringConfig,
  legacyValleyPeakStrictSymmetricConfig,
  runBacktestFromCandles,
  type BacktestResult,
  type Candle,
  type EquityPoint,
  type PartialStrategyConfig,
  type ShortMarginModel,
  type StrategyAlgorithm,
} from "../packages/bot-algo/src/index.js";

type BenchmarkMode =
  | "days"
  | "year"
  | "random-lengths"
  | "grid-search"
  | "portfolio"
  | "synthetic";

interface BenchmarkArgs {
  mode: BenchmarkMode;
  marketKey?: string;
  symbol: string;
  interval: string;
  days: number;
  startingQuote: number;
  leverage: number;
  shortMarginModel: ShortMarginModel;
  longBorrowDepth: number;
  shortBorrowDepth: number;
  lockBorrowedLenderCollateral: boolean;
  borrowerProfitShareToLender: number;
  borrowDepthMatrix: boolean;
  maxOpenOrders: number;
  cooldownSec: number;
  resampleMinutes: number;
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
  syntheticCandles: number;
  syntheticStartPrice: number;
  syntheticFrequency: number;
  syntheticAmplitude: number;
  syntheticTrend: number;
  syntheticNoise: number;
  seed: number;
  only?: string;
  symbols?: string[];
}

const MAX_POSITION_QUOTE_MULTIPLIER = 1;

interface BenchmarkCase {
  label: string;
  algorithm: StrategyAlgorithm;
  config?: PartialStrategyConfig;
}

interface CandleWindow {
  index: number;
  label: string;
  startIndex: number;
  endIndex: number;
  durationDays: number;
}

interface CandleReplayRange {
  startIndex: number;
  endIndex: number;
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
  perfectCompoundedReturnPct: number | undefined;
  perfectCompoundedCapturePct: number | undefined;
  closedPositionCount: number;
  profitableClosedPositionCount: number;
  profitableClosedPositionRate: number;
  liquidatedPositionCount: number;
}

interface SingleBenchmarkRow extends BenchmarkMetrics {
  strategy: string;
}

interface RandomBenchmarkRow {
  strategy: string;
  sampleCount: number;
  profitableSamples: number;
  avgReturnPct: number;
  medianReturnPct: number;
  p10ReturnPct: number;
  avgNetPnl: number;
  avgNetPnlPerDay: number;
  avgMaxDrawdownPct: number;
  avgTradeCount: number;
  avgRiskAdjustedReturn: number | undefined;
  avgSharpeRatio: number | undefined;
  avgPerfectCapturePct: number | undefined;
  avgPerfectCompoundedCapturePct: number | undefined;
  avgClosedPositionCount: number;
  avgProfitableClosedPositionCount: number;
  avgProfitableClosedPositionRate: number;
  avgLiquidatedPositionCount: number;
  bestReturnPct: number;
  worstReturnPct: number;
}

interface GridCandidate {
  label: string;
  algorithm: StrategyAlgorithm;
  config: PartialStrategyConfig;
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
  avgPerfectCapturePct: number | undefined;
  avgPerfectCompoundedCapturePct: number | undefined;
  avgClosedPositionCount: number;
  avgProfitableClosedPositionCount: number;
  avgProfitableClosedPositionRate: number;
  avgLiquidatedPositionCount: number;
  avgMaxDrawdownPct: number;
  avgTradeCount: number;
  bestReturnPct: number;
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
const FULL_BTC_CYCLE_DAYS = YEAR_DAYS * 5;
const BORROW_DEPTH_MATRIX: Array<[number, number]> = [
  [0, 0],
  [1, 0],
  [0, 1],
  [1, 1],
  [1, 2],
  [2, 1],
  [2, 2],
];
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const args = parseArgs(process.argv.slice(2));
if (args.mode === "synthetic") {
  const cases = selectBenchmarkCases(args.only, args.borrowDepthMatrix);
  runSyntheticMode(args, cases);
} else {
  const files = historicalCandleFiles(args);

  if (files.length === 0) {
    const searched = historicalCandleDirCandidates(args)
      .map((dir) => path.relative(repoRoot, dir))
      .join(", ");
    throw new Error(
      `No candles found for ${args.symbol.toUpperCase()} ${args.interval}. Searched: ${searched}.`,
    );
  }

  if (args.mode === "random-lengths") {
    const cases = selectBenchmarkCases(args.only, args.borrowDepthMatrix);
    runRandomLengthMode(args, files, cases);
  } else if (args.mode === "grid-search") {
    runGridSearchMode(args, files);
  } else if (args.mode === "portfolio") {
    const cases = selectBenchmarkCases(args.only, args.borrowDepthMatrix);
    runPortfolioMode(args, files, cases);
  } else {
    const cases = selectBenchmarkCases(args.only, args.borrowDepthMatrix);
    runSingleWindowMode(args, files, cases);
  }
}

function runSyntheticMode(options: BenchmarkArgs, cases: BenchmarkCase[]): void {
  const candles = createSyntheticCandles(options);
  assertCandles(candles);

  const results = cases.map((benchmark) => runBenchmark(benchmark, candles, options));
  const rows = results.map((result) => ({
    strategy: result.label,
    ...metricsFromResult(result.result),
  }));

  console.log(syntheticHeader(options, candles));
  console.log("");
  console.log(singleBenchmarkTable(rows));
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
  const rows = cases.map((benchmark) =>
    runRandomLengthBenchmark(benchmark, candles, windows, options),
  );

  console.log(randomLengthHeader(options, candles, windows));
  console.log("");
  console.log(randomBenchmarkTable(rows));
}

function runGridSearchMode(options: BenchmarkArgs, allFiles: string[]): void {
  const selectedFiles = allFiles.slice(-options.days);
  const candles = loadHistoricalCandles(options, selectedFiles);
  assertCandles(candles);

  const folds = splitCandlesIntoFolds(candles, options.gridFolds);
  const candidates = selectGridCandidates(options.only);
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
    avgPerfectCapturePct: averageDefined(
      metrics.map((metric) => metric.perfectCapturePct),
    ),
    avgPerfectCompoundedCapturePct: averageDefined(
      metrics.map((metric) => metric.perfectCompoundedCapturePct),
    ),
    avgClosedPositionCount: average(metrics.map((metric) => metric.closedPositionCount)),
    avgProfitableClosedPositionCount: average(
      metrics.map((metric) => metric.profitableClosedPositionCount),
    ),
    avgProfitableClosedPositionRate: average(
      metrics.map((metric) => metric.profitableClosedPositionRate),
    ),
    avgLiquidatedPositionCount: average(
      metrics.map((metric) => metric.liquidatedPositionCount),
    ),
    avgMaxDrawdownPct: average(metrics.map((metric) => metric.maxDrawdownPct)),
    avgTradeCount: average(metrics.map((metric) => metric.tradeCount)),
    bestReturnPct: Math.max(...metrics.map((metric) => metric.returnPct)),
    worstReturnPct: Math.min(...metrics.map((metric) => metric.returnPct)),
    config: formatGridConfig(candidate.config),
  };
}

function createGridCandidates(): GridCandidate[] {
  return [
    {
      label: "Legacy Valley/Peak default long/short",
      algorithm: "legacy-valley-peak",
      config: {},
    },
    {
      label: "Legacy Valley/Peak default long only",
      algorithm: "legacy-valley-peak",
      config: { legacyValleyPeak: { shortSideEnabled: false } },
    },
    {
      label: "Legacy Valley/Peak default short only",
      algorithm: "legacy-valley-peak",
      config: { legacyValleyPeak: { longSideEnabled: false, shortSideEnabled: true } },
    },
    {
      label: "Legacy Valley/Peak strict-symmetric reference",
      algorithm: "legacy-valley-peak",
      config: { legacyValleyPeak: legacyValleyPeakStrictSymmetricConfig },
    },
    {
      label: "Legacy Valley/Peak asymmetric short-favoring reference",
      algorithm: "legacy-valley-peak",
      config: { legacyValleyPeak: legacyValleyPeakAsymmetricShortFavoringConfig },
    },
    {
      label: "Legacy peak exit grid aggregate long/short",
      algorithm: "legacy-valley-peak",
      config: peakExitGridConfig("aggregate"),
    },
    {
      label: "Legacy peak exit grid per-lot strict",
      algorithm: "legacy-valley-peak",
      config: peakExitGridConfig("per-lot-strict"),
    },
    {
      label: "Legacy peak exit grid per-lot relaxed",
      algorithm: "legacy-valley-peak",
      config: peakExitGridConfig("per-lot-relaxed"),
    },
    {
      label: "Legacy peak exit grid per-lot relaxed short only",
      algorithm: "legacy-valley-peak",
      config: peakExitGridConfig("per-lot-relaxed", "short-only"),
    },
    {
      label: "Legacy smaller clips",
      algorithm: "legacy-valley-peak",
      config: { legacyValleyPeak: { maxTradeQuote: 500 } },
    },
    {
      label: "Legacy 1250 clips",
      algorithm: "legacy-valley-peak",
      config: { legacyValleyPeak: { maxTradeQuote: 1_250 } },
    },
    {
      label: "Legacy 5000 clips",
      algorithm: "legacy-valley-peak",
      config: { legacyValleyPeak: { maxTradeQuote: 5_000 } },
    },
    {
      label: "Legacy longer warmup",
      algorithm: "legacy-valley-peak",
      config: { legacyValleyPeak: { saturationSec: 7_200 } },
    },
    {
      label: "Legacy tighter sigma",
      algorithm: "legacy-valley-peak",
      config: { legacyValleyPeak: { buySigma: 0.06, sellSigma: 0.06 } },
    },
    {
      label: "Legacy wider sigma",
      algorithm: "legacy-valley-peak",
      config: { legacyValleyPeak: { buySigma: 0.16, sellSigma: 0.16 } },
    },
    {
      label: "Legacy conservative clips",
      algorithm: "legacy-valley-peak",
      config: {
        legacyValleyPeak: {
          buySpendRate: 0.6,
          sellAmountRate: 0.8,
          maxTradeQuote: 500,
          buySigma: 0.08,
          sellSigma: 0.12,
          buyConfirmationOffsets: [8],
        },
      },
    },
  ];
}

function selectGridCandidates(only: string | undefined): GridCandidate[] {
  const candidates = createGridCandidates();
  if (!only) {
    return candidates;
  }

  const normalizedOnly = only.toLowerCase();
  const selected = candidates.filter(
    (candidate) =>
      candidate.algorithm === only ||
      candidate.algorithm.toLowerCase().includes(normalizedOnly) ||
      candidate.label.toLowerCase().includes(normalizedOnly),
  );

  if (selected.length === 0) {
    throw new Error(`No grid candidate matched --only ${only}.`);
  }

  return selected;
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
  candleRange?: CandleReplayRange,
): { label: string; result: BacktestResult; elapsedMs: number } {
  const maxPositionQuote = options.startingQuote * MAX_POSITION_QUOTE_MULTIPLIER;
  const startedAt = Date.now();
  const candleCount = candleRange
    ? candleRange.endIndex - candleRange.startIndex
    : sourceCandles.length;
  if (log) {
    console.error(`Running ${benchmark.label} on ${candleCount.toLocaleString()} candles...`);
  }
  const result = runBacktestFromCandles(sourceCandles, {
    config: {
      symbol: options.symbol.toUpperCase(),
      algorithm: benchmark.algorithm,
      startingQuote: options.startingQuote,
      maxLeverage: options.leverage,
      shortMarginModel: options.shortMarginModel,
      longBorrowDepth: options.longBorrowDepth,
      shortBorrowDepth: options.shortBorrowDepth,
      lockBorrowedLenderCollateral: options.lockBorrowedLenderCollateral,
      borrowerProfitShareToLender: options.borrowerProfitShareToLender,
      maxPositionQuote,
      maxOpenOrders: options.maxOpenOrders,
      cooldownMs: options.cooldownSec * 1000,
      ...benchmark.config,
    },
    ...(maxEquityPoints ? { maxEquityPoints } : {}),
    ...(candleRange
      ? { startIndex: candleRange.startIndex, endIndex: candleRange.endIndex }
      : {}),
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
  candles: Candle[],
  windows: CandleWindow[],
  options: BenchmarkArgs,
): RandomBenchmarkRow {
  const startedAt = Date.now();
  console.error(`Running ${benchmark.label} on ${windows.length.toLocaleString()} random windows...`);
  let profitableSamples = 0;
  let returnPct = 0;
  let netPnl = 0;
  let netPnlPerDay = 0;
  let maxDrawdownPct = 0;
  let tradeCount = 0;
  let riskAdjustedReturn = 0;
  let riskAdjustedReturnCount = 0;
  let sharpeRatio = 0;
  let sharpeRatioCount = 0;
  let perfectCapturePct = 0;
  let perfectCaptureCount = 0;
  let perfectCompoundedCapturePct = 0;
  let perfectCompoundedCaptureCount = 0;
  let closedPositionCount = 0;
  let profitableClosedPositionCount = 0;
  let profitableClosedPositionRate = 0;
  let liquidatedPositionCount = 0;
  let bestReturnPct = Number.NEGATIVE_INFINITY;
  let worstReturnPct = Number.POSITIVE_INFINITY;
  const returnSamples: number[] = [];

  for (const window of windows) {
    const result = runBenchmark(
      benchmark,
      candles,
      options,
      false,
      undefined,
      { startIndex: window.startIndex, endIndex: window.endIndex },
    ).result;
    const metrics = metricsFromResult(result);
    if (result.summary.netPnl > 0) {
      profitableSamples += 1;
    }

    returnPct += metrics.returnPct;
    returnSamples.push(metrics.returnPct);
    netPnl += metrics.netPnl;
    netPnlPerDay += result.summary.netPnl / window.durationDays;
    maxDrawdownPct += metrics.maxDrawdownPct;
    tradeCount += metrics.tradeCount;
    closedPositionCount += metrics.closedPositionCount;
    profitableClosedPositionCount += metrics.profitableClosedPositionCount;
    profitableClosedPositionRate += metrics.profitableClosedPositionRate;
    liquidatedPositionCount += metrics.liquidatedPositionCount;
    bestReturnPct = Math.max(bestReturnPct, metrics.returnPct);
    worstReturnPct = Math.min(worstReturnPct, metrics.returnPct);
    if (metrics.riskAdjustedReturn !== undefined) {
      riskAdjustedReturn += metrics.riskAdjustedReturn;
      riskAdjustedReturnCount += 1;
    }
    if (metrics.sharpeRatio !== undefined) {
      sharpeRatio += metrics.sharpeRatio;
      sharpeRatioCount += 1;
    }
    if (metrics.perfectCapturePct !== undefined) {
      perfectCapturePct += metrics.perfectCapturePct;
      perfectCaptureCount += 1;
    }
    if (metrics.perfectCompoundedCapturePct !== undefined) {
      perfectCompoundedCapturePct += metrics.perfectCompoundedCapturePct;
      perfectCompoundedCaptureCount += 1;
    }
  }

  const sampleCount = windows.length;
  const averageOptional = (total: number, count: number) =>
    count > 0 ? total / count : undefined;
  const averageRequired = (total: number) => total / sampleCount;
  const elapsedMs = Date.now() - startedAt;
  console.error(`Finished ${benchmark.label} random aggregate in ${elapsedMs.toLocaleString()}ms.`);

  return {
    strategy: benchmark.label,
    sampleCount,
    profitableSamples,
    avgReturnPct: averageRequired(returnPct),
    medianReturnPct: percentile(returnSamples, 0.5),
    p10ReturnPct: percentile(returnSamples, 0.1),
    avgNetPnl: averageRequired(netPnl),
    avgNetPnlPerDay: averageRequired(netPnlPerDay),
    avgMaxDrawdownPct: averageRequired(maxDrawdownPct),
    avgTradeCount: averageRequired(tradeCount),
    avgRiskAdjustedReturn: averageOptional(riskAdjustedReturn, riskAdjustedReturnCount),
    avgSharpeRatio: averageOptional(sharpeRatio, sharpeRatioCount),
    avgPerfectCapturePct: averageOptional(perfectCapturePct, perfectCaptureCount),
    avgPerfectCompoundedCapturePct: averageOptional(
      perfectCompoundedCapturePct,
      perfectCompoundedCaptureCount,
    ),
    avgClosedPositionCount: averageRequired(closedPositionCount),
    avgProfitableClosedPositionCount: averageRequired(profitableClosedPositionCount),
    avgProfitableClosedPositionRate: averageRequired(profitableClosedPositionRate),
    avgLiquidatedPositionCount: averageRequired(liquidatedPositionCount),
    bestReturnPct,
    worstReturnPct,
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
    perfectCompoundedReturnPct: summary.perfectMarginCompoundedReturnPct,
    perfectCompoundedCapturePct: summary.perfectMarginCompoundedCapturePct,
    closedPositionCount: summary.closedPositionCount,
    profitableClosedPositionCount: summary.profitableClosedPositionCount,
    profitableClosedPositionRate: summary.profitableClosedPositionRate,
    liquidatedPositionCount: summary.liquidatedPositionCount,
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
    if (endIndex <= startIndex) {
      index -= 1;
      continue;
    }
    const firstCandle = candles[startIndex];
    const lastCandle = candles[endIndex - 1];
    windows.push({
      index,
      label: `${formatDate(firstCandle.openTime)} to ${formatDate(lastCandle.closeTime)}`,
      startIndex,
      endIndex,
      durationDays: Math.max(
        1 / 24 / 60,
        (lastCandle.closeTime - firstCandle.openTime) / DAY_MS,
      ),
    });
  }

  return windows;
}

function historicalCandleFiles(options: BenchmarkArgs): string[] {
  return historicalCandleSource(options).files;
}

function loadHistoricalCandles(options: BenchmarkArgs, files: string[]): Candle[] {
  const dir = historicalCandleSource(options).dir;
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

  const sorted = candles.sort((left, right) => left.openTime - right.openTime);
  return resampleCandles(sorted, options.resampleMinutes);
}

function createSyntheticCandles(options: BenchmarkArgs): Candle[] {
  const rng = mulberry32(options.seed);
  const intervalMs = intervalToMs(options.interval);
  const intervalDays = intervalMs / DAY_MS;
  const halfStepDeviation = options.syntheticNoise * Math.sqrt(intervalDays / 2);
  const startTime = Date.UTC(2021, 0, 1);
  const symbol = options.symbol.toUpperCase();
  const priceFloor = options.syntheticStartPrice * 0.0001;
  const candles: Candle[] = [];
  let brownian = 0;

  const priceAt = (elapsedDays: number, brownianValue: number): number => {
    const cycle = Math.sin(2 * Math.PI * options.syntheticFrequency * elapsedDays);
    const relative =
      1 +
      options.syntheticAmplitude * cycle +
      options.syntheticTrend * elapsedDays +
      brownianValue;
    return Math.max(priceFloor, options.syntheticStartPrice * relative);
  };

  for (let index = 0; index < options.syntheticCandles; index += 1) {
    const candleStartTime = startTime + index * intervalMs;
    const elapsedDays = index * intervalDays;
    const midpointDays = elapsedDays + intervalDays / 2;
    const closeDays = elapsedDays + intervalDays;

    if (index > 0) {
      brownian += normalRandom(rng) * halfStepDeviation;
    }
    const open = priceAt(elapsedDays, brownian);
    brownian += normalRandom(rng) * halfStepDeviation;
    const midpointPrice = priceAt(midpointDays, brownian);
    brownian += normalRandom(rng) * halfStepDeviation;
    const close = priceAt(closeDays, brownian);
    const high = Math.max(open, midpointPrice, close);
    const low = Math.min(open, midpointPrice, close);

    candles.push({
      symbol,
      interval: options.interval,
      openTime: candleStartTime,
      closeTime: candleStartTime + intervalMs - 1,
      open,
      high,
      low,
      close,
      volume: 1_000,
      closed: true,
    });
  }

  return candles;
}

function resampleCandles(candles: Candle[], minutes: number): Candle[] {
  if (minutes <= 1 || candles.length <= 1) {
    return candles;
  }

  const bucketMs = minutes * 60_000;
  const resampled: Candle[] = [];
  let current: Candle | undefined;
  let currentBucket = Number.NaN;
  let volume = 0;

  const flush = () => {
    if (!current) {
      return;
    }
    resampled.push({
      ...current,
      volume,
      closed: true,
    });
  };

  for (const candle of candles) {
    const bucket = Math.floor(candle.openTime / bucketMs) * bucketMs;
    if (!current || bucket !== currentBucket) {
      flush();
      currentBucket = bucket;
      volume = candle.volume;
      current = {
        ...candle,
        interval: `${minutes}m`,
        openTime: bucket,
        closeTime: bucket + bucketMs - 1,
        open: candle.open,
        high: candle.high,
        low: candle.low,
        close: candle.close,
      };
      continue;
    }

    current.high = Math.max(current.high, candle.high);
    current.low = Math.min(current.low, candle.low);
    current.close = candle.close;
    current.closeTime = Math.max(current.closeTime, candle.closeTime);
    volume += candle.volume;
  }

  flush();
  return resampled;
}

function historicalCandleDir(options: Pick<BenchmarkArgs, "symbol" | "interval">): string {
  return historicalCandleSource(options).dir;
}

function historicalCandleSource(
  options: Pick<BenchmarkArgs, "marketKey" | "symbol" | "interval">,
): { dir: string; files: string[] } {
  const candidates = historicalCandleDirCandidates(options)
    .map((dir) => ({
      dir,
      files: listHistoricalCandleFiles(dir),
    }))
    .sort(
      (left, right) =>
        right.files.length - left.files.length ||
        left.dir.localeCompare(right.dir),
    );

  return candidates[0] ?? {
    dir: fallbackHistoricalCandleDir(options),
    files: [],
  };
}

function historicalCandleDirCandidates(
  options: Pick<BenchmarkArgs, "marketKey" | "symbol" | "interval">,
): string[] {
  const root = path.join(repoRoot, "data", "historical");
  const symbol = safePathPart(options.symbol);
  const interval = safePathPart(options.interval);
  const dirs = new Set<string>();

  if (options.marketKey) {
    dirs.add(path.join(root, safePathPart(options.marketKey), symbol, interval));
  }

  dirs.add(fallbackHistoricalCandleDir(options));

  if (!fs.existsSync(root)) {
    return [...dirs];
  }

  for (const entry of fs.readdirSync(root, { withFileTypes: true })) {
    if (!entry.isDirectory()) {
      continue;
    }
    dirs.add(path.join(root, entry.name, symbol, interval));
  }

  return [...dirs];
}

function fallbackHistoricalCandleDir(options: Pick<BenchmarkArgs, "symbol" | "interval">): string {
  return path.join(
    repoRoot,
    "data",
    "historical",
    safePathPart(options.symbol),
    safePathPart(options.interval),
  );
}

function listHistoricalCandleFiles(dir: string): string[] {
  if (!fs.existsSync(dir)) {
    return [];
  }
  return fs
    .readdirSync(dir, { withFileTypes: true })
    .filter((entry) => entry.isFile() && entry.name.endsWith(".jsonl"))
    .map((entry) => entry.name)
    .sort();
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
  const interval = values.get("interval") ?? "1m";
  const days = parsePositiveInt(values.get("days"), FULL_BTC_CYCLE_DAYS);
  const defaultSyntheticCandles = Math.max(1, Math.ceil((days * DAY_MS) / intervalToMs(interval)));
  return {
    mode,
    marketKey: values.get("market-key"),
    symbol: values.get("symbol") ?? "BTCUSDT",
    interval,
    days,
    startingQuote: parsePositiveNumber(values.get("starting-quote"), 10_000),
    leverage: parsePositiveNumber(values.get("leverage"), 1),
    shortMarginModel: parseShortMarginModel(
      values.get("short-margin") ?? values.get("short-margin-model"),
    ),
    longBorrowDepth: parseNonNegativeInt(values.get("long-borrow-depth"), 999),
    shortBorrowDepth: parseNonNegativeInt(values.get("short-borrow-depth"), 999),
    lockBorrowedLenderCollateral:
      values.get("lock-borrowed-lender-collateral") === "true" ||
      values.get("lock-borrowed-collateral") === "true",
    borrowerProfitShareToLender: clamp(
      parseFiniteNumber(values.get("borrower-profit-share-to-lender"), 1),
      0,
      1,
    ),
    borrowDepthMatrix:
      values.get("borrow-depth-matrix") === "true" ||
      values.get("depth-matrix") === "true",
    maxOpenOrders: parsePositiveInt(values.get("max-open-orders"), 1024),
    cooldownSec: parsePositiveNumber(values.get("cooldown-sec"), 300),
    resampleMinutes: parsePositiveInt(values.get("resample-minutes"), 1),
    randomSampleCount: parsePositiveInt(values.get("samples"), 48),
    randomMinWindowDays: parsePositiveNumber(values.get("min-window-days"), 7),
    randomMaxWindowDays: parsePositiveNumber(values.get("max-window-days"), 120),
    randomLookbackDays: parsePositiveNumber(values.get("lookback-days"), FULL_BTC_CYCLE_DAYS),
    gridFolds: parsePositiveInt(values.get("grid-folds"), 3),
    gridLimit: parsePositiveInt(values.get("grid-limit"), 12),
    portfolioLookbackCandles: parsePositiveInt(values.get("portfolio-lookback-candles"), 720),
    portfolioRebalanceCandles: parsePositiveInt(values.get("portfolio-rebalance-candles"), 60),
    portfolioGrossLeverage: parsePositiveNumber(values.get("portfolio-gross-leverage"), 1),
    portfolioTargetVolPct: parsePositiveNumber(values.get("portfolio-target-vol-pct"), 35),
    syntheticCandles: parsePositiveInt(
      values.get("synthetic-candles"),
      defaultSyntheticCandles,
    ),
    syntheticStartPrice: parsePositiveNumber(values.get("synthetic-start-price"), 100_000),
    syntheticFrequency: parseNonNegativeNumber(values.get("synthetic-frequency"), 1),
    syntheticAmplitude: parseNonNegativeNumber(values.get("synthetic-amplitude"), 0.1),
    syntheticTrend: parseFiniteNumber(values.get("synthetic-trend"), 0),
    syntheticNoise: parseNonNegativeNumber(values.get("synthetic-noise"), 0.02),
    seed: parsePositiveInt(values.get("seed"), 1337),
    only: values.get("only"),
    symbols: parseSymbols(values.get("symbols")),
  };
}

function parseMode(value: string | undefined): BenchmarkMode {
  if (
    value === "days" ||
    value === "year" ||
    value === "random-lengths" ||
    value === "grid-search" ||
    value === "portfolio" ||
    value === "synthetic"
  ) {
    return value;
  }
  return "random-lengths";
}

function parseShortMarginModel(value: string | undefined): ShortMarginModel {
  return value === "spot-borrow" ? "spot-borrow" : "futures-margin";
}

function selectBenchmarkCases(
  only: string | undefined,
  borrowDepthMatrix: boolean,
): BenchmarkCase[] {
  const cases: BenchmarkCase[] = [
    {
      label: "Legacy Valley/Peak Long/Short",
      algorithm: "legacy-valley-peak",
    },
    {
      label: "Legacy Valley/Peak Long Only",
      algorithm: "legacy-valley-peak",
      config: { legacyValleyPeak: { shortSideEnabled: false } },
    },
    {
      label: "Legacy Valley/Peak Short Only",
      algorithm: "legacy-valley-peak",
      config: { legacyValleyPeak: { longSideEnabled: false, shortSideEnabled: true } },
    },
    {
      label: "Legacy Valley/Peak Strict-Symmetric Reference",
      algorithm: "legacy-valley-peak",
      config: { legacyValleyPeak: legacyValleyPeakStrictSymmetricConfig },
    },
    {
      label: "Legacy Valley/Peak Asymmetric Short-Favoring Reference",
      algorithm: "legacy-valley-peak",
      config: { legacyValleyPeak: legacyValleyPeakAsymmetricShortFavoringConfig },
    },
    {
      label: "Legacy Peak Exit Grid Aggregate Long/Short",
      algorithm: "legacy-valley-peak",
      config: peakExitGridConfig("aggregate"),
    },
    {
      label: "Legacy Peak Exit Grid Per-Lot Strict Long/Short",
      algorithm: "legacy-valley-peak",
      config: peakExitGridConfig("per-lot-strict"),
    },
    {
      label: "Legacy Peak Exit Grid Per-Lot Relaxed Long/Short",
      algorithm: "legacy-valley-peak",
      config: peakExitGridConfig("per-lot-relaxed"),
    },
    {
      label: "Legacy Peak Exit Grid Per-Lot Relaxed Long Only",
      algorithm: "legacy-valley-peak",
      config: peakExitGridConfig("per-lot-relaxed", "long-only"),
    },
    {
      label: "Legacy Peak Exit Grid Per-Lot Relaxed Short Only",
      algorithm: "legacy-valley-peak",
      config: peakExitGridConfig("per-lot-relaxed", "short-only"),
    },
  ];

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

  return borrowDepthMatrix ? expandBorrowDepthMatrix(selected) : selected;
}

function expandBorrowDepthMatrix(cases: BenchmarkCase[]): BenchmarkCase[] {
  return cases.flatMap((benchmark) =>
    BORROW_DEPTH_MATRIX.map(([longBorrowDepth, shortBorrowDepth]) => ({
      ...benchmark,
      label: `${benchmark.label} L${longBorrowDepth}/S${shortBorrowDepth}`,
      config: {
        ...(benchmark.config ?? {}),
        longBorrowDepth,
        shortBorrowDepth,
      },
    })),
  );
}

type PeakExitGridVariant = "aggregate" | "per-lot-strict" | "per-lot-relaxed";
type StrategySideMode = "long-short" | "long-only" | "short-only";

function peakExitGridConfig(
  variant: PeakExitGridVariant,
  sideMode: StrategySideMode = "long-short",
): PartialStrategyConfig {
  const isAggregate = variant === "aggregate";
  return {
    staleOrderMs: 30 * DAY_MS,
    legacyValleyPeak: {
      buySigma: 0.3,
      longSideEnabled: sideMode !== "short-only",
      shortSideEnabled: sideMode !== "long-only",
      exitGridEnabled: true,
      exitGridMarketEntry: true,
      exitGridOrderCount: 6,
      exitGridPriceDistribution: "uniform",
      exitGridSizeDistribution: "geometric",
      exitGridSellFraction: 0.35,
      exitGridMinProfitBps: 20,
      exitGridResetBps: 10,
      exitGridPositionMode: isAggregate ? "aggregate" : "per-lot",
      exitGridResetMode:
        variant === "per-lot-relaxed" ? "filled-grid" : "higher-peak",
    },
  };
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
    cacheSourceSummary(options),
    ...optionalHeaderPart(resampleSummary(options)),
    `${formatDate(candles[0].openTime)} to ${formatDate(candles[candles.length - 1].closeTime)}`,
    `${options.leverage}x max leverage`,
    `${formatShortMarginModel(options.shortMarginModel)} short margin`,
    borrowDepthSummary(options),
    borrowPolicySummary(options),
    maxOpenOrdersSummary(options),
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
    cacheSourceSummary(options),
    ...optionalHeaderPart(resampleSummary(options)),
    `${formatDate(candles[0].openTime)} to ${formatDate(candles[candles.length - 1].closeTime)} cache span`,
    `${firstWindow.label} first sample`,
    `${lastWindow.label} last sample`,
    `${options.leverage}x max leverage`,
    `${formatShortMarginModel(options.shortMarginModel)} short margin`,
    borrowDepthSummary(options),
    borrowPolicySummary(options),
    maxOpenOrdersSummary(options),
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
    cacheSourceSummary(options),
    ...optionalHeaderPart(resampleSummary(options)),
    `${formatDate(candles[0].openTime)} to ${formatDate(candles[candles.length - 1].closeTime)}`,
    `${options.leverage}x max leverage`,
    `${formatShortMarginModel(options.shortMarginModel)} short margin`,
    borrowDepthSummary(options),
    borrowPolicySummary(options),
    maxOpenOrdersSummary(options),
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
    cacheSourceSummary(options),
    ...optionalHeaderPart(resampleSummary(options)),
    `${formatDate(candles[0].openTime)} to ${formatDate(candles[candles.length - 1].closeTime)}`,
    `${options.leverage}x strategy max leverage`,
    `${formatShortMarginModel(options.shortMarginModel)} short margin`,
    borrowDepthSummary(options),
    borrowPolicySummary(options),
    maxOpenOrdersSummary(options),
    `${options.portfolioGrossLeverage}x portfolio gross leverage`,
    `${options.portfolioRebalanceCandles.toLocaleString()} candle rebalance`,
    `${options.portfolioLookbackCandles.toLocaleString()} candle lookback`,
  ].join(", ");
}

function syntheticHeader(options: BenchmarkArgs, candles: Candle[]): string {
  return [
    "Strategy benchmark: synthetic",
    `${options.symbol.toUpperCase()} ${options.interval}`,
    `${candles.length.toLocaleString()} candles`,
    `${formatDate(candles[0].openTime)} to ${formatDate(candles[candles.length - 1].closeTime)}`,
    `freq ${options.syntheticFrequency}/day`,
    `amp ${formatNumber(options.syntheticAmplitude * 100, 2)}%`,
    `trend ${formatNumber(options.syntheticTrend * 100, 2)}%/day`,
    `brownian ${formatNumber(options.syntheticNoise * 100, 2)}% daily vol`,
    `seed ${options.seed}`,
    `${options.leverage}x max leverage`,
    `${formatShortMarginModel(options.shortMarginModel)} short margin`,
    borrowDepthSummary(options),
    borrowPolicySummary(options),
    maxOpenOrdersSummary(options),
    benchmarkCapSummary(options),
    `${options.cooldownSec}s cooldown`,
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
    "Prof Pos",
    "Liq Pos",
    "Oracle Capture",
    "Reinvest Ret",
    "Reinvest Capture",
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
    `${row.profitableClosedPositionCount.toLocaleString()}/${row.closedPositionCount.toLocaleString()} (${formatNumber(row.profitableClosedPositionRate, 1)}%)`,
    formatNumber(row.liquidatedPositionCount, 0),
    row.perfectCapturePct === undefined ? "-" : `${formatNumber(row.perfectCapturePct, 3)}%`,
    formatOptionalPercent(row.perfectCompoundedReturnPct, 2, true),
    formatOptionalPercent(row.perfectCompoundedCapturePct, 6, true),
  ]);

  return markdownTable(header, tableRows);
}

function randomBenchmarkTable(rows: RandomBenchmarkRow[]): string {
  const header = [
    "Strategy",
    "Samples",
    "Profitable",
    "Avg Return",
    "Median",
    "P10",
    "Avg Net PnL",
    "Avg PnL/day",
    "Avg Max DD",
    "Avg Risk Ret",
    "Avg Sharpe",
    "Avg Trades",
    "Avg Prof Pos",
    "Avg Liq Pos",
    "Avg Capture",
    "Avg Reinvest Cap",
    "Best Return",
    "Worst Return",
  ];
  const tableRows = rows.map((row) => [
    row.strategy,
    row.sampleCount.toLocaleString(),
    `${row.profitableSamples}/${row.sampleCount}`,
    `${formatNumber(row.avgReturnPct, 2)}%`,
    `${formatNumber(row.medianReturnPct, 2)}%`,
    `${formatNumber(row.p10ReturnPct, 2)}%`,
    formatMoney(row.avgNetPnl),
    formatMoney(row.avgNetPnlPerDay),
    `${formatNumber(row.avgMaxDrawdownPct, 2)}%`,
    formatOptionalNumber(row.avgRiskAdjustedReturn, 3),
    formatOptionalNumber(row.avgSharpeRatio, 3),
    formatNumber(row.avgTradeCount, 1),
    `${formatNumber(row.avgProfitableClosedPositionCount, 1)}/${formatNumber(row.avgClosedPositionCount, 1)} (${formatNumber(row.avgProfitableClosedPositionRate, 1)}%)`,
    formatNumber(row.avgLiquidatedPositionCount, 1),
    row.avgPerfectCapturePct === undefined ? "-" : `${formatNumber(row.avgPerfectCapturePct, 3)}%`,
    formatOptionalPercent(row.avgPerfectCompoundedCapturePct, 6, true),
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
    "Avg Capture",
    "Avg Reinvest Cap",
    "Avg Prof Pos",
    "Avg Liq Pos",
    "Avg Max DD",
    "Avg Trades",
    "Best Return",
    "Worst Return",
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
    row.avgPerfectCapturePct === undefined ? "-" : `${formatNumber(row.avgPerfectCapturePct, 3)}%`,
    formatOptionalPercent(row.avgPerfectCompoundedCapturePct, 6, true),
    `${formatNumber(row.avgProfitableClosedPositionCount, 1)}/${formatNumber(row.avgClosedPositionCount, 1)} (${formatNumber(row.avgProfitableClosedPositionRate, 1)}%)`,
    formatNumber(row.avgLiquidatedPositionCount, 1),
    `${formatNumber(row.avgMaxDrawdownPct, 2)}%`,
    formatNumber(row.avgTradeCount, 1),
    `${formatNumber(row.bestReturnPct, 2)}%`,
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

function normalRandom(rng: () => number): number {
  const u1 = Math.max(Number.EPSILON, rng());
  const u2 = Math.max(Number.EPSILON, rng());
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
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

function percentile(values: number[], percentileRank: number): number {
  const finite = values
    .filter((value) => Number.isFinite(value))
    .sort((left, right) => left - right);
  if (finite.length === 0) {
    return 0;
  }

  const rank = clamp(percentileRank, 0, 1) * (finite.length - 1);
  const lower = Math.floor(rank);
  const upper = Math.ceil(rank);
  if (lower === upper) {
    return finite[lower];
  }

  return finite[lower] + (finite[upper] - finite[lower]) * (rank - lower);
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

function parseNonNegativeInt(value: string | undefined, fallback: number): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed >= 0 ? Math.round(parsed) : fallback;
}

function parsePositiveNumber(value: string | undefined, fallback: number): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function parseNonNegativeNumber(value: string | undefined, fallback: number): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : fallback;
}

function parseFiniteNumber(value: string | undefined, fallback: number): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
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

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function intervalToMs(interval: string): number {
  const match = /^(\d+)([mhdw])$/.exec(interval);
  if (!match) {
    return 60_000;
  }

  const value = Number(match[1]);
  const unit = match[2];
  const multipliers: Record<string, number> = {
    m: 60_000,
    h: 60 * 60_000,
    d: DAY_MS,
    w: 7 * DAY_MS,
  };

  return value * multipliers[unit];
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

function formatOptionalPercent(
  value: number | undefined,
  digits: number,
  scientificForLarge = false,
): string {
  if (!Number.isFinite(value)) {
    return "-";
  }

  const finite = value as number;
  if (scientificForLarge && Math.abs(finite) >= 1_000_000) {
    return `${finite.toExponential(Math.min(6, Math.max(0, digits)))}%`;
  }
  return `${finite.toFixed(digits)}%`;
}

function benchmarkCapSummary(options: BenchmarkArgs): string {
  const configuredCap = options.startingQuote * MAX_POSITION_QUOTE_MULTIPLIER;
  const grossMarginCap = options.startingQuote * options.leverage;
  if (options.shortMarginModel === "futures-margin") {
    return `target cap ${formatMoney(configuredCap)}, gross margin cap ${formatMoney(grossMarginCap)}`;
  }

  const initialShortDebtCap = options.startingQuote * Math.max(0, options.leverage - 1) * 0.98;
  return `target cap ${formatMoney(configuredCap)}, initial debt cap ${formatMoney(
    Math.min(configuredCap, initialShortDebtCap),
  )}`;
}

function borrowDepthSummary(options: BenchmarkArgs): string {
  if (options.borrowDepthMatrix) {
    return `borrow depth matrix ${BORROW_DEPTH_MATRIX.map(([long, short]) => `L${long}/S${short}`).join(",")}`;
  }
  return `borrow depth L${options.longBorrowDepth}/S${options.shortBorrowDepth}`;
}

function borrowPolicySummary(options: BenchmarkArgs): string {
  return `borrow lock ${options.lockBorrowedLenderCollateral ? "on" : "off"}, lender profit share ${formatNumber(
    options.borrowerProfitShareToLender,
    2,
  )}`;
}

function resampleSummary(options: BenchmarkArgs): string | undefined {
  return options.resampleMinutes > 1 ? `resampled ${options.resampleMinutes}m OHLC` : undefined;
}

function optionalHeaderPart(value: string | undefined): string[] {
  return value ? [value] : [];
}

function maxOpenOrdersSummary(options: BenchmarkArgs): string {
  return `open orders cap ${options.maxOpenOrders.toLocaleString()}`;
}

function formatShortMarginModel(value: ShortMarginModel): string {
  return value === "futures-margin" ? "futures-margin" : "spot-borrow";
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

function formatGridConfig(config: PartialStrategyConfig): string {
  const parts: string[] = [];
  if (config.shortMarginModel) {
    parts.push(`shortMargin=${config.shortMarginModel}`);
  }
  if (config.longBorrowDepth !== undefined) {
    parts.push(`longBorrowDepth=${config.longBorrowDepth}`);
  }
  if (config.shortBorrowDepth !== undefined) {
    parts.push(`shortBorrowDepth=${config.shortBorrowDepth}`);
  }
  if (config.lockBorrowedLenderCollateral !== undefined) {
    parts.push(`lockBorrowed=${config.lockBorrowedLenderCollateral}`);
  }
  if (config.borrowerProfitShareToLender !== undefined) {
    parts.push(`profitShare=${config.borrowerProfitShareToLender}`);
  }
  if (config.maxOpenOrders !== undefined) {
    parts.push(`maxOpenOrders=${config.maxOpenOrders}`);
  }

  if (config.legacyValleyPeak) {
    const value = config.legacyValleyPeak;
    parts.push(
      `buyRate=${value.buySpendRate ?? "-"}`,
      `sellRate=${value.sellAmountRate ?? "-"}`,
      `maxTrade=${value.maxTradeQuote ?? "-"}`,
      `buySigma=${value.buySigma ?? "-"}`,
      `sellSigma=${value.sellSigma ?? "-"}`,
      `longs=${value.longSideEnabled ?? "-"}`,
      `shorts=${value.shortSideEnabled ?? "-"}`,
      `buyConfirms=${value.buyConfirmationOffsets?.join("/") ?? "-"}`,
      `sellConfirms=${value.sellConfirmationOffsets?.join("/") ?? "-"}`,
      `sellIndex=${value.sellDataIndex ?? "-"}`,
      `warmup=${value.saturationSec ?? "-"}`,
      `exitGrid=${value.exitGridEnabled ?? "-"}`,
      `gridMode=${value.exitGridPositionMode ?? "-"}`,
      `resetMode=${value.exitGridResetMode ?? "-"}`,
      `gridOrders=${value.exitGridOrderCount ?? "-"}`,
      `gridPrice=${value.exitGridPriceDistribution ?? "-"}`,
      `gridSize=${value.exitGridSizeDistribution ?? "-"}`,
      `gridFraction=${value.exitGridSellFraction ?? "-"}`,
    );
  }
  return parts.length > 0 ? parts.join(", ") : "-";
}

function discoverHistoricalSymbols(interval: string): string[] {
  const root = path.join(repoRoot, "data", "historical");
  if (!fs.existsSync(root)) {
    return [];
  }

  const symbols = new Set<string>();
  const safeInterval = safePathPart(interval);

  for (const entry of fs.readdirSync(root, { withFileTypes: true })) {
    if (!entry.isDirectory()) {
      continue;
    }

    if (listHistoricalCandleFiles(path.join(root, entry.name, safeInterval)).length > 0) {
      symbols.add(entry.name.toUpperCase());
    }

    const marketDir = path.join(root, entry.name);
    for (const symbolEntry of fs.readdirSync(marketDir, { withFileTypes: true })) {
      if (!symbolEntry.isDirectory()) {
        continue;
      }
      if (
        listHistoricalCandleFiles(path.join(marketDir, symbolEntry.name, safeInterval)).length > 0
      ) {
        symbols.add(symbolEntry.name.toUpperCase());
      }
    }
  }

  return [...symbols].sort();
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

function cacheSourceSummary(options: Pick<BenchmarkArgs, "marketKey" | "symbol" | "interval">): string {
  const source = historicalCandleSource(options);
  return `cache ${path.relative(repoRoot, source.dir) || "."}`;
}

function safePathPart(value: string): string {
  return value.replace(/[^a-z0-9_-]+/gi, "-").replace(/^-+|-+$/g, "").toLowerCase();
}
