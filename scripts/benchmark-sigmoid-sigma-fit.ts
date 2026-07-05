import fs from "node:fs";
import path from "node:path";
import {
  defaultLegacyValleyPeakConfig,
  runBacktestFromCandles,
  type Candle,
  type PartialStrategyConfig,
} from "../packages/bot-algo/src/index.js";

interface BenchmarkCase {
  label: string;
  startDate: string;
  endDate: string;
}

interface Args {
  caseIndex?: number;
  top: number;
  sigmaLow: number;
  sigmaHigh: number;
  windowsSec: number[];
  slopeAValues: number[];
  slopeBValues: number[];
}

interface RunResult {
  caseIndex: number;
  case: string;
  interval: string;
  trendWindowSec: number;
  slopeA: number;
  slopeB: number;
  sigmoidSigmaLow: number;
  sigmoidSigmaHigh: number;
  returnPct: number;
  netPnl: number;
  maxDrawdownPct: number;
  perfectMarginCapturePct?: number;
  trades: number;
  winRate: number;
  stopReason: string;
}

interface CandidateSummary {
  trendWindowSec: number;
  slopeA: number;
  slopeB: number;
  sigmoidSigmaLow: number;
  sigmoidSigmaHigh: number;
  avgReturnPct: number;
  totalReturnPct: number;
  profitableCases: number;
  worstReturnPct: number;
  bestReturnPct: number;
  avgCapturePct: number;
  results: RunResult[];
}

const cases: BenchmarkCase[] = [
  { label: "uptrend-low-churn", startDate: "2024-02-24", endDate: "2024-02-26" },
  { label: "uptrend-high-churn", startDate: "2022-06-19", endDate: "2022-06-21" },
  { label: "downtrend-low-churn", startDate: "2023-06-03", endDate: "2023-06-05" },
  { label: "downtrend-high-churn", startDate: "2022-06-13", endDate: "2022-06-15" },
  { label: "sideways-high-bias-high-churn", startDate: "2021-10-19", endDate: "2021-10-21" },
  { label: "sideways-high-bias-low-churn", startDate: "2025-02-14", endDate: "2025-02-16" },
  { label: "sideways-low-bias-high-churn", startDate: "2024-07-07", endDate: "2024-07-09" },
  { label: "sideways-low-bias-low-churn", startDate: "2025-07-04", endDate: "2025-07-06" },
  { label: "sideways-mid-bias-high-churn", startDate: "2024-01-02", endDate: "2024-01-04" },
  { label: "sideways-mid-bias-low-churn", startDate: "2023-09-15", endDate: "2023-09-17" },
];

const defaultWindowsSec = [
  30 * 60,
  60 * 60,
  2 * 60 * 60,
  4 * 60 * 60,
  6 * 60 * 60,
  12 * 60 * 60,
];
const defaultSlopeAValues = [0, 10, 25, 50, 100, 200, 400, 800];
const defaultSlopeBValues = [0, 25, 50, 100, 200, 400, 800, 1600];

const args = parseArgs(process.argv.slice(2));
const selectedCases =
  args.caseIndex === undefined ? cases : [caseByIndex(args.caseIndex)];
const candleSets = selectedCases.map((testCase) => loadCandles(testCase));
const summaries: CandidateSummary[] = [];

for (const trendWindowSec of args.windowsSec) {
  for (const slopeA of args.slopeAValues) {
    for (const slopeB of args.slopeBValues) {
      const results = selectedCases.map((testCase, index) =>
        runCase({
          testCase,
          caseIndex: cases.indexOf(testCase),
          candles: candleSets[index] ?? [],
          trendWindowSec,
          slopeA,
          slopeB,
          sigmaLow: args.sigmaLow,
          sigmaHigh: args.sigmaHigh,
        }),
      );
      const summary = summarizeCandidate({
        trendWindowSec,
        slopeA,
        slopeB,
        sigmaLow: args.sigmaLow,
        sigmaHigh: args.sigmaHigh,
        results,
      });
      summaries.push(summary);
    }
  }
}

summaries.sort(
  (left, right) =>
    right.avgReturnPct - left.avgReturnPct ||
    right.profitableCases - left.profitableCases ||
    right.worstReturnPct - left.worstReturnPct,
);

for (const summary of summaries.slice(0, args.top)) {
  console.log(JSON.stringify(summary));
}

function runCase(input: {
  testCase: BenchmarkCase;
  caseIndex: number;
  candles: Candle[];
  trendWindowSec: number;
  slopeA: number;
  slopeB: number;
  sigmaLow: number;
  sigmaHigh: number;
}): RunResult {
  const result = runBacktestFromCandles(input.candles, {
    config: {
      symbol: "BTCUSDT",
      algorithm: "legacy-valley-peak",
      startingQuote: 10_000,
      maxLeverage: 1,
      shortMarginModel: "futures-margin",
      longBorrowDepth: 999,
      shortBorrowDepth: 999,
      internalBorrowAccounting: "inactive",
      legacyValleyPeak: {
        sigmaMode: "sigmoid-trend",
        trendSigmaWindowSec: input.trendWindowSec,
        trendSigmaSellB1: input.slopeA,
        trendSigmaBuyB2: input.slopeB,
        sigmoidSigmaLow: input.sigmaLow,
        sigmoidSigmaHigh: input.sigmaHigh,
        ...trendWindowProfile(input.trendWindowSec),
      },
    },
    maxReturnedOrders: 0,
    maxReturnedFills: 0,
    maxEquityPoints: 0,
    maxChartCandles: 1,
  });
  const summary = result.summary;
  return {
    caseIndex: input.caseIndex,
    case: input.testCase.label,
    interval: `${input.testCase.startDate}..${input.testCase.endDate}`,
    trendWindowSec: input.trendWindowSec,
    slopeA: input.slopeA,
    slopeB: input.slopeB,
    sigmoidSigmaLow: input.sigmaLow,
    sigmoidSigmaHigh: input.sigmaHigh,
    returnPct: round(summary.returnPct, 4),
    netPnl: round(summary.netPnl, 2),
    maxDrawdownPct: round(summary.maxDrawdownPct, 4),
    perfectMarginCapturePct:
      summary.perfectMarginCapturePct === undefined
        ? undefined
        : round(summary.perfectMarginCapturePct, 4),
    trades: summary.tradeCount,
    winRate: round(summary.winRate, 2),
    stopReason: summary.stopReason,
  };
}

function summarizeCandidate(input: {
  trendWindowSec: number;
  slopeA: number;
  slopeB: number;
  sigmaLow: number;
  sigmaHigh: number;
  results: RunResult[];
}): CandidateSummary {
  const returns = input.results.map((result) => result.returnPct);
  const captures = input.results
    .map((result) => result.perfectMarginCapturePct)
    .filter((value): value is number => value !== undefined);
  const totalReturnPct = returns.reduce((total, value) => total + value, 0);
  return {
    trendWindowSec: input.trendWindowSec,
    slopeA: input.slopeA,
    slopeB: input.slopeB,
    sigmoidSigmaLow: input.sigmaLow,
    sigmoidSigmaHigh: input.sigmaHigh,
    avgReturnPct: round(totalReturnPct / Math.max(1, returns.length), 4),
    totalReturnPct: round(totalReturnPct, 4),
    profitableCases: returns.filter((value) => value > 0).length,
    worstReturnPct: round(Math.min(...returns), 4),
    bestReturnPct: round(Math.max(...returns), 4),
    avgCapturePct: round(
      captures.reduce((total, value) => total + value, 0) / Math.max(1, captures.length),
      4,
    ),
    results: input.results,
  };
}

function trendWindowProfile(
  trendWindowSec: number,
): PartialStrategyConfig["legacyValleyPeak"] {
  const base = [1, 60, 600, 1800, 3600, 4 * 3600, 12 * 3600, trendWindowSec];
  const averagingRangesSec = [...new Set(base)].sort((left, right) => left - right);
  return {
    averagingRangesSec,
    rateRatios: valuesForWindows(averagingRangesSec, defaultLegacyValleyPeakConfig.rateRatios),
    rateThresholdsLow: valuesForWindows(
      averagingRangesSec,
      defaultLegacyValleyPeakConfig.rateThresholdsLow,
    ),
    rateThresholdsHigh: valuesForWindows(
      averagingRangesSec,
      defaultLegacyValleyPeakConfig.rateThresholdsHigh,
    ),
  };
}

function valuesForWindows(windowsSec: number[], values: number[]): number[] {
  return windowsSec.map((windowSec) => {
    const exact = defaultLegacyValleyPeakConfig.averagingRangesSec.indexOf(windowSec);
    if (exact >= 0) {
      return values[exact] ?? values.at(-1) ?? 0;
    }

    let closest = 0;
    let closestDistance = Number.POSITIVE_INFINITY;
    for (
      let index = 0;
      index < defaultLegacyValleyPeakConfig.averagingRangesSec.length;
      index += 1
    ) {
      const distance = Math.abs(
        (defaultLegacyValleyPeakConfig.averagingRangesSec[index] ?? 0) - windowSec,
      );
      if (distance < closestDistance) {
        closest = index;
        closestDistance = distance;
      }
    }
    return values[closest] ?? values.at(-1) ?? 0;
  });
}

function parseArgs(argv: string[]): Args {
  const values = new Map<string, string>();
  for (let index = 0; index < argv.length; index += 1) {
    const key = argv[index];
    if (!key?.startsWith("--")) {
      continue;
    }
    const value = argv[index + 1];
    if (!value || value.startsWith("--")) {
      values.set(key.slice(2), "true");
    } else {
      values.set(key.slice(2), value);
      index += 1;
    }
  }

  return {
    caseIndex: values.has("case-index")
      ? nonNegativeInteger(values.get("case-index") ?? "0", "case-index")
      : undefined,
    top: positiveInteger(values.get("top") ?? "10", "top"),
    sigmaLow: positiveNumber(values.get("sigma-low") ?? "0.05", "sigma-low"),
    sigmaHigh: positiveNumber(values.get("sigma-high") ?? "0.3", "sigma-high"),
    windowsSec: parseNumberList(values.get("windows-sec"), defaultWindowsSec),
    slopeAValues: parseNumberList(values.get("slope-a"), defaultSlopeAValues),
    slopeBValues: parseNumberList(values.get("slope-b"), defaultSlopeBValues),
  };
}

function parseNumberList(value: string | undefined, fallback: number[]): number[] {
  if (value === undefined) {
    return fallback;
  }
  const parsed = value
    .split(",")
    .map((item) => Number(item.trim()))
    .filter((item) => Number.isFinite(item) && item >= 0);
  if (parsed.length === 0) {
    throw new Error("Number lists must contain at least one non-negative number.");
  }
  return [...new Set(parsed)];
}

function caseByIndex(index: number): BenchmarkCase {
  const testCase = cases[index];
  if (!testCase) {
    throw new Error(`--case-index must be between 0 and ${cases.length - 1}.`);
  }
  return testCase;
}

function nonNegativeInteger(value: string, label: string): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < 0) {
    throw new Error(`--${label} must be non-negative.`);
  }
  return Math.round(parsed);
}

function positiveInteger(value: string, label: string): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(`--${label} must be positive.`);
  }
  return Math.round(parsed);
}

function positiveNumber(value: string, label: string): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(`--${label} must be positive.`);
  }
  return parsed;
}

function loadCandles(testCase: BenchmarkCase): Candle[] {
  const dir = "data/historical/spot-btcusdt/btcusdt/1m";
  const candles: Candle[] = [];
  for (
    let timestamp = Date.parse(`${testCase.startDate}T00:00:00Z`);
    timestamp <= Date.parse(`${testCase.endDate}T00:00:00Z`);
    timestamp += 24 * 60 * 60 * 1000
  ) {
    const file = `${new Date(timestamp).toISOString().slice(0, 10)}.jsonl`;
    const content = fs.readFileSync(path.join(dir, file), "utf8");
    for (const line of content.split("\n")) {
      if (line.trim()) {
        candles.push(JSON.parse(line) as Candle);
      }
    }
  }
  return candles.sort((left, right) => left.openTime - right.openTime);
}

function round(value: number, digits: number): number {
  return Number(value.toFixed(digits));
}
