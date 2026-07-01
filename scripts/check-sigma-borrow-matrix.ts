import fs from "node:fs";
import path from "node:path";
import {
  runBacktestFromCandles,
  type Candle,
  type InternalBorrowAccounting,
  type PartialStrategyConfig,
} from "../packages/bot-algo/src/index.js";

interface Args {
  buySigma: number;
  sellSigma: number;
  days: number;
  mode?: SideMode;
  longBorrowDepth?: number;
  shortBorrowDepth?: number;
  internalBorrowAccounting: InternalBorrowAccounting;
}

type SideMode = "both" | "long-only" | "short-only";

const args = parseArgs(process.argv.slice(2));
const candles = loadCandles(args.days);
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
      buySigma: args.buySigma,
      sellSigma: args.sellSigma,
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
    buySigma: args.buySigma,
    sellSigma: args.sellSigma,
    longSideEnabled: mode !== "short-only",
    shortSideEnabled: mode !== "long-only",
  };
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
    buySigma: positiveNumber(values.get("buy-sigma"), "buy-sigma"),
    sellSigma: positiveNumber(values.get("sell-sigma"), "sell-sigma"),
    days: positiveInteger(values.get("days") ?? "30", "days"),
    mode: parseMode(values.get("mode")),
    longBorrowDepth: values.has("long-borrow-depth")
      ? nonNegativeInteger(values.get("long-borrow-depth") ?? "0", "long-borrow-depth")
      : undefined,
    shortBorrowDepth: values.has("short-borrow-depth")
      ? nonNegativeInteger(values.get("short-borrow-depth") ?? "0", "short-borrow-depth")
      : undefined,
    internalBorrowAccounting: parseInternalBorrowAccounting(
      values.get("internal-borrow-accounting"),
    ),
  };
}

function parseInternalBorrowAccounting(
  value: string | undefined,
): InternalBorrowAccounting {
  if (value === undefined || value === "active") {
    return "active";
  }
  if (value === "inactive") {
    return "inactive";
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

function loadCandles(days: number): Candle[] {
  const dir = "data/historical/spot-btcusdt/btcusdt/1m";
  const files = fs.readdirSync(dir).filter((file) => file.endsWith(".jsonl")).sort().slice(-days);
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

function round(value: number, digits: number): number {
  return Number(value.toFixed(digits));
}
