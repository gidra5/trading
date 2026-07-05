import fs from "node:fs";
import path from "node:path";
import {
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

const sigmaPairs = [
  { buySigma: 0.1, sellSigma: 0.1 },
  { buySigma: 0.1, sellSigma: 0.3 },
  { buySigma: 0.3, sellSigma: 0.1 },
  { buySigma: 0.3, sellSigma: 0.3 },
];

const speedProfiles: Array<{
  label: "default-sma" | "all-sma-2x";
  legacyValleyPeak: PartialStrategyConfig["legacyValleyPeak"];
}> = [
  {
    label: "default-sma",
    legacyValleyPeak: {},
  },
  {
    label: "all-sma-2x",
    legacyValleyPeak: {
      averagingRangesSec: [1, 30, 300, 900, 1800, 7200, 21600],
    },
  },
];

const args = parseArgs(process.argv.slice(2));
const selectedCases =
  args.caseIndex === undefined ? cases : [caseByIndex(args.caseIndex)];

for (const testCase of selectedCases) {
  const candles = loadCandles(testCase.startDate, testCase.endDate);
  const market = summarizeMarket(candles);

  for (const speedProfile of speedProfiles) {
    for (const pair of sigmaPairs) {
      const started = Date.now();
      const result = runBacktestFromCandles(candles, {
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
            sigmaMode: "static",
            buySigma: pair.buySigma,
            sellSigma: pair.sellSigma,
            ...speedProfile.legacyValleyPeak,
          },
        },
        maxReturnedOrders: 0,
        maxReturnedFills: 0,
        maxEquityPoints: 16,
        maxChartCandles: 1,
      });
      const summary = result.summary;
      const metrics = result.finalState.metrics;
      console.log(
        JSON.stringify({
          caseIndex: cases.indexOf(testCase),
          case: testCase.label,
          interval: `${testCase.startDate}..${testCase.endDate}`,
          marketReturnPct: round(market.returnPct, 4),
          marketSpanPct: round(market.spanPct, 4),
          marketLow: market.low,
          marketHigh: market.high,
          speedProfile: speedProfile.label,
          sigmaMode: "static",
          buySigma: pair.buySigma,
          sellSigma: pair.sellSigma,
          returnPct: round(summary.returnPct, 4),
          netPnl: round(summary.netPnl, 2),
          maxDrawdownPct: round(summary.maxDrawdownPct, 4),
          perfectMarginReturnPct: round(summary.perfectMarginReturnPct ?? 0, 4),
          perfectMarginNetPnl: round(summary.perfectMarginNetPnl ?? 0, 2),
          perfectMarginCapturePct:
            summary.perfectMarginCapturePct === undefined
              ? undefined
              : round(summary.perfectMarginCapturePct, 4),
          perfectMarginCompoundedReturnPct: round(
            summary.perfectMarginCompoundedReturnPct ?? 0,
            4,
          ),
          perfectMarginCompoundedNetPnl: round(
            summary.perfectMarginCompoundedNetPnl ?? 0,
            2,
          ),
          perfectMarginCompoundedCapturePct:
            summary.perfectMarginCompoundedCapturePct === undefined
              ? undefined
              : round(summary.perfectMarginCompoundedCapturePct, 4),
          realizedPnl: round(metrics.realizedPnl, 2),
          unrealizedPnl: round(metrics.unrealizedPnl, 2),
          feesPaid: round(metrics.feesPaid, 2),
          trades: summary.tradeCount,
          winRate: round(summary.winRate, 2),
          profitableClosedPositionCount: summary.profitableClosedPositionCount,
          closedPositionCount: summary.closedPositionCount,
          liquidatedPositionCount: summary.liquidatedPositionCount,
          stopReason: summary.stopReason,
          elapsedMs: Date.now() - started,
        }),
      );
    }
  }
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
  };
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

function loadCandles(startDate: string, endDate: string): Candle[] {
  const dir = "data/historical/spot-btcusdt/btcusdt/1m";
  const candles: Candle[] = [];
  for (
    let timestamp = Date.parse(`${startDate}T00:00:00Z`);
    timestamp <= Date.parse(`${endDate}T00:00:00Z`);
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

function summarizeMarket(candles: Candle[]): {
  returnPct: number;
  spanPct: number;
  low: number;
  high: number;
} {
  if (candles.length === 0) {
    throw new Error("No candles loaded.");
  }
  const open = candles[0]?.open ?? 0;
  const close = candles.at(-1)?.close ?? open;
  const low = Math.min(...candles.map((candle) => candle.low));
  const high = Math.max(...candles.map((candle) => candle.high));
  return {
    returnPct: ((close - open) / open) * 100,
    spanPct: ((high - low) / open) * 100,
    low,
    high,
  };
}

function round(value: number, digits: number): number {
  return Number(value.toFixed(digits));
}
