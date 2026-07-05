import fs from "node:fs";
import path from "node:path";
import {
  runBacktestFromCandles,
  type Candle,
  type InternalBorrowAccounting,
  type LegacySigmaMode,
  type PartialStrategyConfig,
} from "../packages/bot-algo/src/index.js";

interface Args {
  sigmaMode: LegacySigmaMode;
  buySigma: number;
  sellSigma: number;
  trendSigmaA: number;
  trendSigmaSellB1: number;
  trendSigmaBuyB2: number;
  trendSigmaWindowSec: number;
  sigmoidSigmaLow: number;
  sigmoidSigmaHigh: number;
  days: number;
  startDate?: string;
  endDate?: string;
  signalProfile: SignalProfile;
  mode?: SideMode;
  longBorrowDepth?: number;
  shortBorrowDepth?: number;
  internalBorrowAccounting: InternalBorrowAccounting;
}

type SideMode = "both" | "long-only" | "short-only";
type SignalProfile =
  | "default"
  | "half-source-only"
  | "half-confirm-windows"
  | "half-windows"
  | "fast-confirm-only";

const args = parseArgs(process.argv.slice(2));
const candles = loadCandles(args);
const market = summarizeMarket(candles);
const cases = selectCases(args);

for (const testCase of cases) {
  const started = Date.now();
  const result = runBacktestFromCandles(candles, {
    config: {
      symbol: "BTCUSDT",
      algorithm: "legacy-valley-peak",
      startingQuote: 10_000,
      maxLeverage: 1,
      shortMarginModel: "futures-margin",
      longBorrowDepth: testCase.longBorrowDepth,
      shortBorrowDepth: testCase.shortBorrowDepth,
      internalBorrowAccounting: args.internalBorrowAccounting,
      legacyValleyPeak: legacyConfig(args, testCase.mode),
    },
    maxReturnedOrders: 0,
    maxReturnedFills: 0,
    maxEquityPoints: 32,
    maxChartCandles: 1,
  });
  const summary = result.summary;
  console.log(
    JSON.stringify({
      sigmaMode: args.sigmaMode,
      buySigma: args.buySigma,
      sellSigma: args.sellSigma,
      trendSigmaA: args.trendSigmaA,
      trendSigmaSellB1: args.trendSigmaSellB1,
      trendSigmaBuyB2: args.trendSigmaBuyB2,
      trendSigmaWindowSec: args.trendSigmaWindowSec,
      sigmoidSigmaLow: args.sigmoidSigmaLow,
      sigmoidSigmaHigh: args.sigmoidSigmaHigh,
      days: args.days,
      startDate: args.startDate,
      endDate: args.endDate,
      signalProfile: args.signalProfile,
      candles: candles.length,
      marketReturnPct: round(market.returnPct, 4),
      marketSpanPct: round(market.spanPct, 4),
      marketLow: market.low,
      marketHigh: market.high,
      mode: testCase.mode,
      longBorrowDepth: testCase.longBorrowDepth,
      shortBorrowDepth: testCase.shortBorrowDepth,
      internalBorrowAccounting: args.internalBorrowAccounting,
      returnPct: round(summary.returnPct, 4),
      netPnl: round(summary.netPnl, 2),
      maxDrawdownPct: round(summary.maxDrawdownPct, 4),
      maxEffectiveLeverage: round(summary.maxEffectiveLeverage ?? 0, 4),
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

function selectCases(args: Args): Array<{
  mode: SideMode;
  longBorrowDepth: number;
  shortBorrowDepth: number;
}> {
  const cases = [
    ...[0, 999].flatMap((longBorrowDepth) =>
      [0, 999].map((shortBorrowDepth) => ({
        mode: "both" as const,
        longBorrowDepth,
        shortBorrowDepth,
      })),
    ),
    { mode: "long-only" as const, longBorrowDepth: 0, shortBorrowDepth: 0 },
    { mode: "short-only" as const, longBorrowDepth: 0, shortBorrowDepth: 0 },
  ].filter((testCase) => {
    if (args.mode && testCase.mode !== args.mode) {
      return false;
    }
    if (
      args.longBorrowDepth !== undefined &&
      testCase.longBorrowDepth !== args.longBorrowDepth
    ) {
      return false;
    }
    if (
      args.shortBorrowDepth !== undefined &&
      testCase.shortBorrowDepth !== args.shortBorrowDepth
    ) {
      return false;
    }
    return true;
  });

  if (cases.length === 0) {
    throw new Error("No borrow-matrix case matched the selected filters.");
  }

  return cases;
}

function legacyConfig(args: Args, mode: SideMode): PartialStrategyConfig["legacyValleyPeak"] {
  return {
    sigmaMode: args.sigmaMode,
    buySigma: args.buySigma,
    sellSigma: args.sellSigma,
    trendSigmaA: args.trendSigmaA,
    trendSigmaSellB1: args.trendSigmaSellB1,
    trendSigmaBuyB2: args.trendSigmaBuyB2,
    trendSigmaWindowSec: args.trendSigmaWindowSec,
    sigmoidSigmaLow: args.sigmoidSigmaLow,
    sigmoidSigmaHigh: args.sigmoidSigmaHigh,
    ...signalProfileConfig(args.signalProfile),
    longSideEnabled: mode !== "short-only",
    shortSideEnabled: mode !== "long-only",
  };
}

function signalProfileConfig(
  profile: SignalProfile,
): PartialStrategyConfig["legacyValleyPeak"] {
  if (profile === "half-windows") {
    return {
      averagingRangesSec: [1, 30, 300, 900, 1800, 7200, 21600],
    };
  }
  if (profile === "half-source-only") {
    return {
      averagingRangesSec: [1, 30, 600, 1800, 3600, 14400, 43200],
    };
  }
  if (profile === "half-confirm-windows") {
    return {
      averagingRangesSec: [1, 60, 300, 900, 3600, 14400, 43200],
    };
  }
  if (profile === "fast-confirm-only") {
    return {
      buyConfirmationOffsets: [1],
      sellConfirmationOffsets: [1],
    };
  }
  return {};
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
    sigmaMode: parseSigmaMode(values.get("sigma-mode")),
    buySigma: positiveNumber(values.get("buy-sigma") ?? "0.3", "buy-sigma"),
    sellSigma: positiveNumber(values.get("sell-sigma") ?? "0.1", "sell-sigma"),
    trendSigmaA: positiveNumber(values.get("sigma-a") ?? "1", "sigma-a"),
    trendSigmaSellB1: Number(values.get("sell-b1") ?? "1"),
    trendSigmaBuyB2: Number(values.get("buy-b2") ?? "1"),
    trendSigmaWindowSec: positiveNumber(
      values.get("trend-window-sec") ?? "3600",
      "trend-window-sec",
    ),
    sigmoidSigmaLow: positiveNumber(values.get("sigmoid-low") ?? "0.05", "sigmoid-low"),
    sigmoidSigmaHigh: positiveNumber(values.get("sigmoid-high") ?? "0.3", "sigmoid-high"),
    days: positiveInteger(values.get("days") ?? "30", "days"),
    startDate: parseDate(values.get("start-date"), "start-date"),
    endDate: parseDate(values.get("end-date"), "end-date"),
    signalProfile: parseSignalProfile(values.get("signal-profile")),
    mode: parseMode(values.get("mode")),
    longBorrowDepth: values.has("long-borrow-depth")
      ? nonNegativeInteger(values.get("long-borrow-depth") ?? "0", "long-borrow-depth")
      : undefined,
    shortBorrowDepth: values.has("short-borrow-depth")
      ? nonNegativeInteger(values.get("short-borrow-depth") ?? "0", "short-borrow-depth")
      : undefined,
    internalBorrowAccounting: parseInternalBorrowAccounting(values.get("internal-borrow-accounting")),
  };
}

function parseSignalProfile(value: string | undefined): SignalProfile {
  if (value === undefined || value === "default") {
    return "default";
  }
  if (
    value === "half-source-only" ||
    value === "half-confirm-windows" ||
    value === "half-windows" ||
    value === "fast-confirm-only"
  ) {
    return value;
  }
  throw new Error(
    "--signal-profile must be default, half-source-only, half-confirm-windows, half-windows, or fast-confirm-only.",
  );
}

function parseDate(value: string | undefined, label: string): string | undefined {
  if (value === undefined) {
    return undefined;
  }
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value) || Number.isNaN(Date.parse(`${value}T00:00:00Z`))) {
    throw new Error(`--${label} must use YYYY-MM-DD.`);
  }
  return value;
}

function parseSigmaMode(value: string | undefined): LegacySigmaMode {
  if (value === undefined || value === "trend") {
    return "trend";
  }
  if (value === "static" || value === "sigmoid-trend") {
    return value;
  }
  throw new Error("--sigma-mode must be trend, static, or sigmoid-trend.");
}

function parseInternalBorrowAccounting(
  value: string | undefined,
): InternalBorrowAccounting {
  if (value === undefined || value === "inactive") {
    return "inactive";
  }
  if (value === "active") {
    return "active";
  }
  throw new Error("--internal-borrow-accounting must be active or inactive.");
}

function parseMode(value: string | undefined): SideMode | undefined {
  if (value === undefined) {
    return undefined;
  }
  if (value === "both" || value === "long-only" || value === "short-only") {
    return value;
  }
  throw new Error("--mode must be both, long-only, or short-only.");
}

function positiveNumber(value: string | undefined, label: string): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(`--${label} must be positive.`);
  }
  return parsed;
}

function nonNegativeNumber(value: string, label: string): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < 0) {
    throw new Error(`--${label} must be non-negative.`);
  }
  return parsed;
}

function positiveInteger(value: string, label: string): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(`--${label} must be positive.`);
  }
  return Math.round(parsed);
}

function nonNegativeInteger(value: string, label: string): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < 0) {
    throw new Error(`--${label} must be non-negative.`);
  }
  return Math.round(parsed);
}

function loadCandles(args: Args): Candle[] {
  const dir = "data/historical/spot-btcusdt/btcusdt/1m";
  const files =
    args.startDate || args.endDate
      ? historicalFilesForDateRange(dir, args.startDate, args.endDate)
      : fs.readdirSync(dir).filter((file) => file.endsWith(".jsonl")).sort().slice(-args.days);
  const candles: Candle[] = [];
  for (const file of files) {
    const content = fs.readFileSync(path.join(dir, file), "utf8");
    for (const line of content.split("\n")) {
      if (line.trim()) {
        candles.push(JSON.parse(line) as Candle);
      }
    }
  }
  return candles.sort((left, right) => left.openTime - right.openTime);
}

function historicalFilesForDateRange(
  dir: string,
  startDate: string | undefined,
  endDate: string | undefined,
): string[] {
  if (!startDate || !endDate) {
    throw new Error("--start-date and --end-date must be provided together.");
  }
  if (Date.parse(`${endDate}T00:00:00Z`) < Date.parse(`${startDate}T00:00:00Z`)) {
    throw new Error("--end-date must be on or after --start-date.");
  }

  const files: string[] = [];
  for (
    let timestamp = Date.parse(`${startDate}T00:00:00Z`);
    timestamp <= Date.parse(`${endDate}T00:00:00Z`);
    timestamp += 24 * 60 * 60 * 1000
  ) {
    const file = `${new Date(timestamp).toISOString().slice(0, 10)}.jsonl`;
    const fullPath = path.join(dir, file);
    if (!fs.existsSync(fullPath)) {
      throw new Error(`Missing historical candle file: ${fullPath}`);
    }
    files.push(file);
  }
  return files;
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
