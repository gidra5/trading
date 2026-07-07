import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  runBacktestFromCandles,
  type BacktestResult,
  type Candle,
  type LegacyDerivativeSource,
  type LegacyMovingAverageType,
  type PartialStrategyConfig,
} from "../packages/bot-algo/src/index.js";

interface BenchmarkCase {
  group: string;
  label: string;
  startDate: string;
  endDate: string;
}

interface Args {
  outputPath?: string;
  reportPath?: string;
  label: string;
  buySigma: number;
  sellSigma: number;
  derivativeSource: LegacyDerivativeSource;
  movingAverageTypes: LegacyMovingAverageType[];
}

interface ResultRow {
  type: "result";
  group: string;
  label: string;
  interval: string;
  runLabel: string;
  derivativeSource: LegacyDerivativeSource;
  movingAverageType: LegacyMovingAverageType;
  buySigma: number;
  sellSigma: number;
  marketReturnPct: number;
  marketSpanPct: number;
  marketLow: number;
  marketHigh: number;
  returnPct: number;
  netPnl: number;
  maxDrawdownPct: number;
  tradeCount: number;
  winRate: number;
  closedPositionCount: number;
  profitableClosedPositionCount: number;
  liquidatedPositionCount: number;
  stopReason?: string;
  elapsedMs: number;
}

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const candleDir = path.join(repoRoot, "data/historical/spot-btcusdt/btcusdt/1m");
const cases: BenchmarkCase[] = [
  { group: "choppy-week", label: "highest OHLC churn", startDate: "2022-07-28", endDate: "2022-08-03" },
  { group: "choppy-week", label: "highest close churn", startDate: "2022-05-14", endDate: "2022-05-20" },
  { group: "choppy-week", label: "very choppy 2021-12", startDate: "2021-12-14", endDate: "2021-12-20" },
  { group: "choppy-week", label: "very choppy 2021-09", startDate: "2021-09-08", endDate: "2021-09-14" },
  { group: "choppy-week", label: "near-flat close", startDate: "2023-03-18", endDate: "2023-03-24" },
  { group: "regime-week", label: "uptrend", startDate: "2023-03-11", endDate: "2023-03-17" },
  { group: "regime-week", label: "sideways", startDate: "2026-04-22", endDate: "2026-04-28" },
  { group: "regime-week", label: "downtrend", startDate: "2022-06-12", endDate: "2022-06-18" },
  { group: "3d-regime", label: "uptrend low churn", startDate: "2024-02-24", endDate: "2024-02-26" },
  { group: "3d-regime", label: "uptrend high churn", startDate: "2022-06-19", endDate: "2022-06-21" },
  { group: "3d-regime", label: "downtrend low churn", startDate: "2023-06-03", endDate: "2023-06-05" },
  { group: "3d-regime", label: "downtrend high churn", startDate: "2022-06-13", endDate: "2022-06-15" },
  { group: "3d-regime", label: "sideways high bias churn", startDate: "2021-10-19", endDate: "2021-10-21" },
  { group: "3d-regime", label: "sideways high bias low churn", startDate: "2025-02-14", endDate: "2025-02-16" },
  { group: "3d-regime", label: "sideways low bias churn", startDate: "2024-07-07", endDate: "2024-07-09" },
  { group: "3d-regime", label: "sideways low bias low churn", startDate: "2025-07-04", endDate: "2025-07-06" },
  { group: "3d-regime", label: "sideways mid bias churn", startDate: "2024-01-02", endDate: "2024-01-04" },
  { group: "3d-regime", label: "sideways mid bias low churn", startDate: "2023-09-15", endDate: "2023-09-17" },
  { group: "trend-sharpe", label: "3d up 2024-11", startDate: "2024-11-09", endDate: "2024-11-11" },
  { group: "trend-sharpe", label: "3d up 2023-12", startDate: "2023-12-03", endDate: "2023-12-05" },
  { group: "trend-sharpe", label: "3d down 2026-06", startDate: "2026-06-01", endDate: "2026-06-03" },
  { group: "trend-sharpe", label: "3d down 2023-03", startDate: "2023-03-07", endDate: "2023-03-09" },
  { group: "trend-sharpe", label: "7d up 2023-12", startDate: "2023-11-29", endDate: "2023-12-05" },
  { group: "trend-sharpe", label: "7d up 2024-11", startDate: "2024-11-05", endDate: "2024-11-11" },
  { group: "trend-sharpe", label: "7d down 2023-03", startDate: "2023-03-03", endDate: "2023-03-09" },
  { group: "trend-sharpe", label: "7d down 2026-06", startDate: "2026-05-27", endDate: "2026-06-02" },
  { group: "stress", label: "3d down 2022-06", startDate: "2022-06-11", endDate: "2022-06-13" },
  { group: "stress", label: "7d down 2022-06", startDate: "2022-06-07", endDate: "2022-06-13" },
];

const args = parseArgs(process.argv.slice(2));
const stamp = new Date().toISOString().replace(/[:.]/g, "").replace("T", "-").slice(0, 17);
const outputPath = path.resolve(
  repoRoot,
  args.outputPath ?? `data/benchmarks/extrema-signal-timing-${stamp}.jsonl`,
);
const reportPath = path.resolve(
  repoRoot,
  args.reportPath ?? `docs/extrema-signal-timing-${stamp}.md`,
);

runBenchmark(args, outputPath, reportPath);

function runBenchmark(configArgs: Args, targetOutputPath: string, targetReportPath: string): void {
  const startedAt = Date.now();
  const rows: ResultRow[] = [];
  fs.mkdirSync(path.dirname(targetOutputPath), { recursive: true });
  fs.writeFileSync(targetOutputPath, "");

  appendJson(targetOutputPath, {
    type: "meta",
    generatedAt: new Date(startedAt).toISOString(),
    candleDir: path.relative(repoRoot, candleDir),
    cases,
    config: {
      label: configArgs.label,
      sigmaMode: "static",
      buySigma: configArgs.buySigma,
      sellSigma: configArgs.sellSigma,
      derivativeSource: configArgs.derivativeSource,
      movingAverageTypes: configArgs.movingAverageTypes,
      derivativeClampMode: "deadband",
    },
  });

  for (const movingAverageType of configArgs.movingAverageTypes) {
    for (const testCase of cases) {
      const candles = loadCandles(testCase);
      const market = summarizeMarket(candles);
      const started = Date.now();
      const result = runBacktestFromCandles(candles, {
        config: benchmarkConfig(configArgs, movingAverageType),
        maxReturnedOrders: 0,
        maxReturnedFills: 0,
        maxEquityPoints: 16,
        maxChartCandles: 1,
      });
      const row = resultRow(
        testCase,
        market,
        result,
        configArgs,
        movingAverageType,
        Date.now() - started,
      );
      rows.push(row);
      appendJson(targetOutputPath, row);
      console.error(
        `${movingAverageType.toUpperCase()} ${testCase.startDate}..${testCase.endDate}: ${formatSignedPct(row.returnPct)}, DD ${formatPct(row.maxDrawdownPct)}, trades ${row.tradeCount}`,
      );
    }
  }

  appendJson(targetOutputPath, {
    type: "status",
    status: "completed",
    completedAt: new Date().toISOString(),
    elapsedMs: Date.now() - startedAt,
  });
  renderReport(rows, configArgs, targetOutputPath, targetReportPath);
  console.error(`Report written to ${path.relative(repoRoot, targetReportPath)}`);
}

function benchmarkConfig(
  configArgs: Args,
  movingAverageType: LegacyMovingAverageType,
): PartialStrategyConfig {
  return {
    symbol: "BTCUSDT",
    algorithm: "legacy-valley-peak",
    startingQuote: 10_000,
    maxLeverage: 1,
    shortMarginModel: "futures-margin",
    longBorrowDepth: 999,
    shortBorrowDepth: 999,
    internalBorrowAccounting: "inactive",
    legacyValleyPeak: {
      movingAverageType,
      sigmaMode: "static",
      buySigma: configArgs.buySigma,
      sellSigma: configArgs.sellSigma,
      derivativeSource: configArgs.derivativeSource,
      derivativeClampMode: "deadband",
    },
  };
}

function resultRow(
  testCase: BenchmarkCase,
  market: ReturnType<typeof summarizeMarket>,
  result: BacktestResult,
  configArgs: Args,
  movingAverageType: LegacyMovingAverageType,
  elapsedMs: number,
): ResultRow {
  const summary = result.summary;
  return {
    type: "result",
    group: testCase.group,
    label: testCase.label,
    interval: `${testCase.startDate}..${testCase.endDate}`,
    runLabel: configArgs.label,
    derivativeSource: configArgs.derivativeSource,
    movingAverageType,
    buySigma: configArgs.buySigma,
    sellSigma: configArgs.sellSigma,
    marketReturnPct: round(market.returnPct, 4),
    marketSpanPct: round(market.spanPct, 4),
    marketLow: market.low,
    marketHigh: market.high,
    returnPct: round(summary.returnPct, 4),
    netPnl: round(summary.netPnl, 2),
    maxDrawdownPct: round(summary.maxDrawdownPct, 4),
    tradeCount: summary.tradeCount,
    winRate: round(summary.winRate, 2),
    closedPositionCount: summary.closedPositionCount,
    profitableClosedPositionCount: summary.profitableClosedPositionCount,
    liquidatedPositionCount: summary.liquidatedPositionCount,
    stopReason: summary.stopReason,
    elapsedMs,
  };
}

function renderReport(
  rows: ResultRow[],
  configArgs: Args,
  rawPath: string,
  targetReportPath: string,
): void {
  const lines: string[] = [];
  lines.push(`# Extrema Signal Timing Benchmark: ${configArgs.label}`);
  lines.push("");
  lines.push(`Generated: ${formatDateTime(Date.now())}`);
  lines.push(`Raw results: \`${path.relative(repoRoot, rawPath)}\``);
  lines.push("");
  lines.push("## Scope");
  lines.push("");
  lines.push("- Market: BTCUSDT 1m spot candles, UTC day-inclusive intervals from `tasks.md`.");
  lines.push(`- Strategy: static sigma \`buySigma=${configArgs.buySigma}\`, \`sellSigma=${configArgs.sellSigma}\`, mode both, futures-margin shorts, borrow depths \`999/999\`, max leverage \`1x\`.`);
  lines.push(`- Derivative source: \`${configArgs.derivativeSource}\`; derivative clamp mode: \`deadband\`.`);
  lines.push(`- Moving average type(s): \`${configArgs.movingAverageTypes.join("`, `")}\`. EMA uses a continuous-time alpha with \`tau = window / 2\`, matching the SMA window's average sample age.`);
  lines.push("");
  lines.push("## Summary");
  lines.push("");
  lines.push("| Average | Avg return | Median return | Positive | Avg DD | Max DD | Avg trades |");
  lines.push("| --- | ---: | ---: | ---: | ---: | ---: | ---: |");
  for (const movingAverageType of configArgs.movingAverageTypes) {
    const typeRows = rows.filter((row) => row.movingAverageType === movingAverageType);
    lines.push(
      [
        `| ${movingAverageType.toUpperCase()}`,
        formatSignedPct(mean(typeRows.map((row) => row.returnPct))),
        formatSignedPct(median(typeRows.map((row) => row.returnPct))),
        `${typeRows.filter((row) => row.returnPct > 0).length}/${typeRows.length}`,
        formatPct(mean(typeRows.map((row) => row.maxDrawdownPct))),
        formatPct(Math.max(...typeRows.map((row) => row.maxDrawdownPct))),
        `${round(mean(typeRows.map((row) => row.tradeCount)), 1).toLocaleString()} |`,
      ].join(" | "),
    );
  }
  if (configArgs.movingAverageTypes.includes("sma") && configArgs.movingAverageTypes.includes("ema")) {
    lines.push("");
    lines.push("## EMA vs SMA");
    lines.push("");
    lines.push("| Group | Case | Interval | Market | SMA return | EMA return | Delta | SMA DD | EMA DD | Delta DD | SMA trades | EMA trades |");
    lines.push("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |");
    for (const comparison of compareAverageRows(rows)) {
      lines.push(
        [
          `| ${comparison.sma.group}`,
          comparison.sma.label,
          `\`${comparison.sma.interval}\``,
          formatSignedPct(comparison.sma.marketReturnPct),
          formatSignedPct(comparison.sma.returnPct),
          formatSignedPct(comparison.ema.returnPct),
          formatSignedPct(comparison.ema.returnPct - comparison.sma.returnPct),
          formatPct(comparison.sma.maxDrawdownPct),
          formatPct(comparison.ema.maxDrawdownPct),
          formatSignedPct(comparison.ema.maxDrawdownPct - comparison.sma.maxDrawdownPct),
          comparison.sma.tradeCount.toLocaleString(),
          `${comparison.ema.tradeCount.toLocaleString()} |`,
        ].join(" | "),
      );
    }
  }
  lines.push("");
  lines.push("## Intervals");
  lines.push("");
  lines.push("| Average | Group | Case | Interval | Market | Return | Max DD | Trades | Win rate | Stop |");
  lines.push("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |");
  for (const row of rows) {
    lines.push(
      [
        `| ${row.movingAverageType.toUpperCase()}`,
        row.group,
        row.label,
        `\`${row.interval}\``,
        formatSignedPct(row.marketReturnPct),
        formatSignedPct(row.returnPct),
        formatPct(row.maxDrawdownPct),
        row.tradeCount.toLocaleString(),
        formatPct(row.winRate),
        `${row.stopReason ?? "-"} |`,
      ].join(" | "),
    );
  }
  lines.push("");
  fs.mkdirSync(path.dirname(targetReportPath), { recursive: true });
  fs.writeFileSync(targetReportPath, `${lines.join("\n")}\n`);
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
    outputPath: values.get("output"),
    reportPath: values.get("report"),
    label: values.get("label") ?? "current",
    buySigma: positiveNumber(values.get("buy-sigma") ?? "0.1", "buy-sigma"),
    sellSigma: positiveNumber(values.get("sell-sigma") ?? "0.1", "sell-sigma"),
    derivativeSource: parseDerivativeSource(values.get("derivative-source")),
    movingAverageTypes: parseMovingAverageTypes(values.get("moving-average-type")),
  };
}

function compareAverageRows(rows: ResultRow[]): Array<{ sma: ResultRow; ema: ResultRow }> {
  const byKey = new Map<string, Partial<Record<LegacyMovingAverageType, ResultRow>>>();
  for (const row of rows) {
    const key = `${row.group}|${row.label}|${row.interval}`;
    const existing = byKey.get(key) ?? {};
    existing[row.movingAverageType] = row;
    byKey.set(key, existing);
  }
  return [...byKey.values()].flatMap((entry) =>
    entry.sma && entry.ema ? [{ sma: entry.sma, ema: entry.ema }] : [],
  );
}

function parseDerivativeSource(value: string | undefined): LegacyDerivativeSource {
  if (value === undefined || value === "price") {
    return "price";
  }
  if (value === "kama") {
    return "kama";
  }
  throw new Error("--derivative-source must be price or kama.");
}

function parseMovingAverageTypes(value: string | undefined): LegacyMovingAverageType[] {
  if (value === undefined) {
    return ["sma"];
  }
  if (value === "both") {
    return ["sma", "ema"];
  }

  const parsed = value
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean);
  const types = parsed.map((part) => {
    if (part === "sma" || part === "ema") {
      return part;
    }
    throw new Error("--moving-average-type must be sma, ema, both, or a comma list of sma/ema.");
  });
  return [...new Set(types)];
}

function positiveNumber(value: string, label: string): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(`--${label} must be positive.`);
  }
  return parsed;
}

function loadCandles(testCase: BenchmarkCase): Candle[] {
  const candles: Candle[] = [];
  for (
    let timestamp = Date.parse(`${testCase.startDate}T00:00:00Z`);
    timestamp <= Date.parse(`${testCase.endDate}T00:00:00Z`);
    timestamp += 24 * 60 * 60 * 1000
  ) {
    const file = `${new Date(timestamp).toISOString().slice(0, 10)}.jsonl`;
    const filePath = path.join(candleDir, file);
    if (!fs.existsSync(filePath)) {
      throw new Error(`Missing candle file: ${path.relative(repoRoot, filePath)}`);
    }
    const content = fs.readFileSync(filePath, "utf8");
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

function appendJson(filePath: string, value: unknown): void {
  fs.appendFileSync(filePath, `${JSON.stringify(value)}\n`);
}

function mean(values: number[]): number {
  return values.length === 0
    ? 0
    : values.reduce((total, value) => total + value, 0) / values.length;
}

function median(values: number[]): number {
  if (values.length === 0) {
    return 0;
  }
  const sorted = values.slice().sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? ((sorted[middle - 1] ?? 0) + (sorted[middle] ?? 0)) / 2
    : sorted[middle] ?? 0;
}

function round(value: number, digits: number): number {
  return Number(value.toFixed(digits));
}

function formatDateTime(timestamp: number): string {
  return new Date(timestamp).toISOString().replace("T", " ").slice(0, 16);
}

function formatSignedPct(value: number): string {
  return `${value >= 0 ? "+" : ""}${value.toFixed(3)}%`;
}

function formatPct(value: number): string {
  return `${value.toFixed(3)}%`;
}
