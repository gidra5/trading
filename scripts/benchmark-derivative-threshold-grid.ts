import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  defaultLegacyValleyPeakConfig,
  runBacktestFromCandles,
  type BacktestResult,
  type Candle,
  type LegacyDerivativeClampMode,
  type LegacyDerivativeSource,
  type PartialStrategyConfig,
} from "../packages/bot-algo/src/index.js";

type SourceArg = LegacyDerivativeSource | "both";

interface BenchmarkCase {
  group: string;
  label: string;
  startDate: string;
  endDate: string;
}

interface Args {
  outputPath?: string;
  reportPath?: string;
  derivativeSource: SourceArg;
  thresholdMultipliers: number[];
  innerRatios: number[];
  buySigma: number;
  sellSigma: number;
  kamaErLen: number;
  kamaFastLen: number;
  kamaSlowLen: number;
  kamaPower: number;
}

interface Combo {
  derivativeSource: LegacyDerivativeSource;
  thresholdMultiplier: number;
  clampMode: LegacyDerivativeClampMode;
  innerRatio: number;
}

interface ResultRow extends Combo {
  type: "result";
  group: string;
  label: string;
  interval: string;
  marketReturnPct: number;
  marketSpanPct: number;
  returnPct: number;
  maxDrawdownPct: number;
  tradeCount: number;
  winRate: number;
  elapsedMs: number;
}

interface ComboSummary extends Combo {
  cases: number;
  avgReturnPct: number;
  medianReturnPct: number;
  positiveCount: number;
  avgMaxDrawdownPct: number;
  maxDrawdownPct: number;
  avgTrades: number;
  totalTrades: number;
  avgElapsedMs: number;
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
  args.outputPath ?? `data/benchmarks/derivative-threshold-grid-${stamp}.jsonl`,
);
const reportPath = path.resolve(
  repoRoot,
  args.reportPath ?? `docs/derivative-threshold-grid-${stamp}.md`,
);

runBenchmark(args, outputPath, reportPath);

function runBenchmark(configArgs: Args, targetOutputPath: string, targetReportPath: string): void {
  const startedAt = Date.now();
  const rows: ResultRow[] = [];
  const candlesByInterval = new Map<string, Candle[]>();
  fs.mkdirSync(path.dirname(targetOutputPath), { recursive: true });
  fs.writeFileSync(targetOutputPath, "");

  const combos = buildCombos(configArgs);
  appendJson(targetOutputPath, {
    type: "meta",
    generatedAt: new Date(startedAt).toISOString(),
    candleDir: path.relative(repoRoot, candleDir),
    cases,
    config: {
      sigmaMode: "static",
      buySigma: configArgs.buySigma,
      sellSigma: configArgs.sellSigma,
      derivativeSource: configArgs.derivativeSource,
      thresholdMultipliers: configArgs.thresholdMultipliers,
      innerRatios: configArgs.innerRatios,
      kamaErLen: configArgs.kamaErLen,
      kamaFastLen: configArgs.kamaFastLen,
      kamaSlowLen: configArgs.kamaSlowLen,
      kamaPower: configArgs.kamaPower,
    },
  });

  let completed = 0;
  const total = cases.length * combos.length;
  for (const combo of combos) {
    for (const testCase of cases) {
      const interval = intervalLabel(testCase);
      let candles = candlesByInterval.get(interval);
      if (!candles) {
        candles = loadCandles(testCase);
        candlesByInterval.set(interval, candles);
      }
      const market = summarizeMarket(candles);
      const started = Date.now();
      const result = runBacktestFromCandles(candles, {
        config: benchmarkConfig(configArgs, combo),
        maxReturnedOrders: 0,
        maxReturnedFills: 0,
        maxEquityPoints: 16,
        maxChartCandles: 1,
      });
      const row = resultRow(testCase, market, result, combo, Date.now() - started);
      rows.push(row);
      appendJson(targetOutputPath, row);
      completed += 1;
      console.error(
        `${completed}/${total} ${comboLabel(combo)} ${interval}: ${formatSignedPct(row.returnPct)}, DD ${formatPct(row.maxDrawdownPct)}, trades ${row.tradeCount}`,
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

function buildCombos(configArgs: Args): Combo[] {
  const sources = configArgs.derivativeSource === "both"
    ? ["price", "kama"] as const
    : [configArgs.derivativeSource];
  return sources.flatMap((derivativeSource) =>
    configArgs.thresholdMultipliers.flatMap((thresholdMultiplier) => [
      {
        derivativeSource,
        thresholdMultiplier,
        clampMode: "deadband" as const,
        innerRatio: 0,
      },
      ...configArgs.innerRatios.map((innerRatio) => ({
        derivativeSource,
        thresholdMultiplier,
        clampMode: "hysteresis" as const,
        innerRatio,
      })),
    ]),
  );
}

function benchmarkConfig(configArgs: Args, combo: Combo): PartialStrategyConfig {
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
      sigmaMode: "static",
      buySigma: configArgs.buySigma,
      sellSigma: configArgs.sellSigma,
      derivativeSource: combo.derivativeSource,
      derivativeClampMode: combo.clampMode,
      derivativeClampInnerThresholdRatio: combo.innerRatio,
      kamaErLen: configArgs.kamaErLen,
      kamaFastLen: configArgs.kamaFastLen,
      kamaSlowLen: configArgs.kamaSlowLen,
      kamaPower: configArgs.kamaPower,
      rateThresholdsLow: scaleValues(
        defaultLegacyValleyPeakConfig.rateThresholdsLow,
        combo.thresholdMultiplier,
      ),
      rateThresholdsHigh: scaleValues(
        defaultLegacyValleyPeakConfig.rateThresholdsHigh,
        combo.thresholdMultiplier,
      ),
    },
  };
}

function resultRow(
  testCase: BenchmarkCase,
  market: ReturnType<typeof summarizeMarket>,
  result: BacktestResult,
  combo: Combo,
  elapsedMs: number,
): ResultRow {
  const summary = result.summary;
  return {
    type: "result",
    group: testCase.group,
    label: testCase.label,
    interval: intervalLabel(testCase),
    ...combo,
    marketReturnPct: round(market.returnPct, 4),
    marketSpanPct: round(market.spanPct, 4),
    returnPct: round(summary.returnPct, 4),
    maxDrawdownPct: round(summary.maxDrawdownPct, 4),
    tradeCount: summary.tradeCount,
    winRate: round(summary.winRate, 2),
    elapsedMs,
  };
}

function renderReport(
  rows: ResultRow[],
  configArgs: Args,
  rawPath: string,
  targetReportPath: string,
): void {
  const summaries = summarizeCombos(rows)
    .sort((left, right) =>
      right.avgReturnPct - left.avgReturnPct ||
      left.avgMaxDrawdownPct - right.avgMaxDrawdownPct ||
      left.avgTrades - right.avgTrades,
    );
  const baseline = summaries.find(
    (summary) =>
      summary.derivativeSource === "kama" &&
      summary.thresholdMultiplier === 1 &&
      summary.clampMode === "deadband",
  ) ?? summaries.find(
    (summary) =>
      summary.derivativeSource === "price" &&
      summary.thresholdMultiplier === 1 &&
      summary.clampMode === "deadband",
  );

  const lines: string[] = [];
  lines.push("# Derivative Threshold Grid Benchmark");
  lines.push("");
  lines.push(`Generated: ${formatDateTime(Date.now())}`);
  lines.push(`Raw results: \`${path.relative(repoRoot, rawPath)}\``);
  lines.push("");
  lines.push("## Scope");
  lines.push("");
  lines.push("- Market: BTCUSDT 1m spot candles, UTC day-inclusive intervals from `tasks.md`.");
  lines.push(`- Strategy: static sigma \`buySigma=${configArgs.buySigma}\`, \`sellSigma=${configArgs.sellSigma}\`, mode both, futures-margin shorts, borrow depths \`999/999\`, max leverage \`1x\`.`);
  lines.push(`- Sources: \`${configArgs.derivativeSource}\`.`);
  lines.push(`- Threshold multipliers: \`${configArgs.thresholdMultipliers.join(", ")}\`; default 60s threshold is \`0.25\`, so \`4x\` makes it \`1.0\`.`);
  lines.push(`- Hysteresis inner ratios: \`${configArgs.innerRatios.join(", ")}\`; ratio \`0\` is zero-cross exit, ratio \`1\` exits at the outer threshold.`);
  lines.push(`- KAMA: \`erLen=${configArgs.kamaErLen}\`, \`fastLen=${configArgs.kamaFastLen}\`, \`slowLen=${configArgs.kamaSlowLen}\`, \`power=${configArgs.kamaPower}\`.`);
  lines.push("");
  lines.push("## Top Combinations");
  lines.push("");
  lines.push(summaryTableHeader());
  for (const summary of summaries.slice(0, 20)) {
    lines.push(summaryLine(summary, baseline));
  }
  lines.push("");
  lines.push("## All Combinations");
  lines.push("");
  lines.push(summaryTableHeader());
  for (const summary of summaries) {
    lines.push(summaryLine(summary, baseline));
  }
  lines.push("");
  fs.mkdirSync(path.dirname(targetReportPath), { recursive: true });
  fs.writeFileSync(targetReportPath, `${lines.join("\n")}\n`);
}

function summarizeCombos(rows: ResultRow[]): ComboSummary[] {
  const groups = new Map<string, ResultRow[]>();
  for (const row of rows) {
    const key = comboKey(row);
    const group = groups.get(key);
    if (group) {
      group.push(row);
    } else {
      groups.set(key, [row]);
    }
  }
  return [...groups.values()].map((group) => {
    const first = group[0];
    if (!first) {
      throw new Error("Cannot summarize an empty group.");
    }
    return {
      derivativeSource: first.derivativeSource,
      thresholdMultiplier: first.thresholdMultiplier,
      clampMode: first.clampMode,
      innerRatio: first.innerRatio,
      cases: group.length,
      avgReturnPct: round(mean(group.map((row) => row.returnPct)), 4),
      medianReturnPct: round(median(group.map((row) => row.returnPct)), 4),
      positiveCount: group.filter((row) => row.returnPct > 0).length,
      avgMaxDrawdownPct: round(mean(group.map((row) => row.maxDrawdownPct)), 4),
      maxDrawdownPct: round(Math.max(...group.map((row) => row.maxDrawdownPct)), 4),
      avgTrades: round(mean(group.map((row) => row.tradeCount)), 1),
      totalTrades: sum(group.map((row) => row.tradeCount)),
      avgElapsedMs: round(mean(group.map((row) => row.elapsedMs)), 1),
    };
  });
}

function summaryTableHeader(): string {
  return [
    "| Source | Outer | Mode | Inner | Avg return | Vs baseline | Median | Positive | Avg DD | Max DD | Avg trades |",
    "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
  ].join("\n");
}

function summaryLine(summary: ComboSummary, baseline: ComboSummary | undefined): string {
  const baselineDelta = baseline
    ? summary.avgReturnPct - baseline.avgReturnPct
    : 0;
  return [
    `| \`${summary.derivativeSource}\``,
    `${summary.thresholdMultiplier}x`,
    `\`${summary.clampMode}\``,
    summary.clampMode === "hysteresis" ? summary.innerRatio.toFixed(2) : "",
    formatSignedPct(summary.avgReturnPct),
    formatSignedPct(baselineDelta),
    formatSignedPct(summary.medianReturnPct),
    `${summary.positiveCount}/${summary.cases}`,
    formatPct(summary.avgMaxDrawdownPct),
    formatPct(summary.maxDrawdownPct),
    `${summary.avgTrades.toLocaleString()} |`,
  ].join(" | ");
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
    derivativeSource: parseSource(values.get("derivative-source")),
    thresholdMultipliers: parseNumberList(
      values.get("threshold-multipliers") ?? "1,2,4",
      "threshold-multipliers",
      positiveNumber,
    ),
    innerRatios: parseNumberList(
      values.get("inner-ratios") ?? "0,0.25,0.5,0.75,1",
      "inner-ratios",
      ratioNumber,
    ),
    buySigma: positiveNumber(values.get("buy-sigma") ?? "0.1", "buy-sigma"),
    sellSigma: positiveNumber(values.get("sell-sigma") ?? "0.1", "sell-sigma"),
    kamaErLen: positiveInteger(values.get("kama-er-len") ?? "20", "kama-er-len"),
    kamaFastLen: positiveInteger(values.get("kama-fast-len") ?? "20", "kama-fast-len"),
    kamaSlowLen: positiveInteger(values.get("kama-slow-len") ?? "200", "kama-slow-len"),
    kamaPower: positiveNumber(values.get("kama-power") ?? "1", "kama-power"),
  };
}

function parseSource(value: string | undefined): SourceArg {
  if (value === undefined || value === "kama") {
    return "kama";
  }
  if (value === "price" || value === "both") {
    return value;
  }
  throw new Error("--derivative-source must be price, kama, or both.");
}

function parseNumberList(
  value: string,
  label: string,
  parseItem: (value: string, label: string) => number,
): number[] {
  const parsed = value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean)
    .map((item) => parseItem(item, label));
  const unique = [...new Set(parsed)];
  if (unique.length === 0) {
    throw new Error(`--${label} must include at least one value.`);
  }
  return unique;
}

function positiveNumber(value: string, label: string): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(`--${label} must be positive.`);
  }
  return parsed;
}

function ratioNumber(value: string, label: string): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < 0 || parsed > 1) {
    throw new Error(`--${label} must be between 0 and 1.`);
  }
  return parsed;
}

function positiveInteger(value: string, label: string): number {
  return Math.round(positiveNumber(value, label));
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
  };
}

function intervalLabel(testCase: BenchmarkCase): string {
  return `${testCase.startDate}..${testCase.endDate}`;
}

function comboLabel(combo: Combo): string {
  return `${combo.derivativeSource} ${combo.thresholdMultiplier}x ${combo.clampMode} inner ${combo.innerRatio}`;
}

function comboKey(combo: Combo): string {
  return [
    combo.derivativeSource,
    combo.thresholdMultiplier,
    combo.clampMode,
    combo.innerRatio,
  ].join(":");
}

function scaleValues(values: number[], multiplier: number): number[] {
  return values.map((value) => value * multiplier);
}

function appendJson(filePath: string, value: unknown): void {
  fs.appendFileSync(filePath, `${JSON.stringify(value)}\n`);
}

function sum(values: number[]): number {
  return values.reduce((total, value) => total + value, 0);
}

function mean(values: number[]): number {
  return values.length === 0 ? 0 : sum(values) / values.length;
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
