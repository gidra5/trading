import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  defaultLegacyValleyPeakConfig,
  runBacktestFromCandles,
  type BacktestResult,
  type Candle,
  type PartialStrategyConfig,
} from "../packages/bot-algo/src/index.js";

interface Args {
  outputPath?: string;
  reportPath?: string;
  thresholdMultipliers: number[];
  windowsDays: number[];
  leverage: number;
  buySigma: number;
  sellSigma: number;
  kamaErLen: number;
  kamaFastLen: number;
  kamaSlowLen: number;
  kamaPower: number;
  forceMaxEntryLeverage: boolean;
}

interface WindowSpec {
  label: string;
  requestedDays: number;
  startIndex: number;
  endIndex: number;
  start: string;
  end: string;
  actualDays: number;
  candleCount: number;
}

interface ResultRow {
  type: "result";
  window: string;
  requestedDays: number;
  start: string;
  end: string;
  actualDays: number;
  candleCount: number;
  thresholdMultiplier: number;
  marketReturnPct: number;
  marketSpanPct: number;
  returnPct: number;
  netPnl: number;
  finalEquity: number;
  maxDrawdownPct: number;
  perfectMarginReturnPct?: number;
  perfectMarginCapturePct?: number;
  tradeCount: number;
  winRate: number;
  closedPositionCount: number;
  profitableClosedPositionCount: number;
  liquidatedPositionCount: number;
  maxEntryLeverage?: number;
  maxEffectiveLeverage?: number;
  stopReason?: string;
  elapsedMs: number;
}

interface ThresholdSummary {
  thresholdMultiplier: number;
  windows: number;
  avgReturnPct: number;
  medianReturnPct: number;
  positiveCount: number;
  worstReturnPct: number;
  avgMaxDrawdownPct: number;
  maxDrawdownPct: number;
  totalLiquidations: number;
  avgTrades: number;
  maxEffectiveLeverage: number;
}

const DAY_MS = 24 * 60 * 60 * 1000;
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const candleDir = path.join(repoRoot, "data/historical/spot-btcusdt/btcusdt/1m");

const args = parseArgs(process.argv.slice(2));
const stamp = new Date().toISOString().replace(/[:.]/g, "").replace("T", "-").slice(0, 17);
const outputPath = path.resolve(
  repoRoot,
  args.outputPath ?? `data/benchmarks/recent-kama-thresholds-${stamp}.jsonl`,
);
const reportPath = path.resolve(
  repoRoot,
  args.reportPath ?? `docs/recent-kama-thresholds-${stamp}.md`,
);

runBenchmark(args, outputPath, reportPath);

function runBenchmark(configArgs: Args, targetOutputPath: string, targetReportPath: string): void {
  const startedAt = Date.now();
  const files = listCandleFiles();
  const maxDays = Math.max(...configArgs.windowsDays);
  const selectedFiles = files.slice(-Math.min(maxDays, files.length));
  if (selectedFiles.length === 0) {
    throw new Error(`No candle files found in ${path.relative(repoRoot, candleDir)}.`);
  }

  const candles = loadCandles(selectedFiles);
  assertCandles(candles);
  const windows = buildWindows(candles, selectedFiles, configArgs.windowsDays);
  const rows: ResultRow[] = [];

  fs.mkdirSync(path.dirname(targetOutputPath), { recursive: true });
  fs.writeFileSync(targetOutputPath, "");
  appendJson(targetOutputPath, {
    type: "meta",
    generatedAt: new Date(startedAt).toISOString(),
    candleDir: path.relative(repoRoot, candleDir),
    cacheStart: formatDateTime(candles[0]?.openTime ?? 0),
    cacheEnd: formatDateTime(candles.at(-1)?.closeTime ?? 0),
    selectedDayFiles: selectedFiles.length,
    config: {
      derivativeSource: "kama",
      derivativeClampMode: "deadband",
      sigmaMode: "static",
      buySigma: configArgs.buySigma,
      sellSigma: configArgs.sellSigma,
      thresholdMultipliers: configArgs.thresholdMultipliers,
      windowsDays: configArgs.windowsDays,
      leverage: configArgs.leverage,
      kamaErLen: configArgs.kamaErLen,
      kamaFastLen: configArgs.kamaFastLen,
      kamaSlowLen: configArgs.kamaSlowLen,
      kamaPower: configArgs.kamaPower,
      forceMaxEntryLeverage: configArgs.forceMaxEntryLeverage,
    },
  });

  let completed = 0;
  const total = windows.length * configArgs.thresholdMultipliers.length;
  for (const multiplier of configArgs.thresholdMultipliers) {
    for (const window of windows) {
      const market = summarizeMarket(candles, window.startIndex, window.endIndex);
      const started = Date.now();
      const result = runBacktestFromCandles(candles, {
        config: benchmarkConfig(configArgs, multiplier),
        startIndex: window.startIndex,
        endIndex: window.endIndex,
        maxReturnedOrders: 0,
        maxReturnedFills: 0,
        maxEquityPoints: 64,
        maxChartCandles: 1,
      });
      const row = resultRow(window, market, result, multiplier, Date.now() - started);
      rows.push(row);
      appendJson(targetOutputPath, row);
      completed += 1;
      console.error(
        `${completed}/${total} ${window.label} ${multiplier}x: ${formatSignedPct(row.returnPct)}, DD ${formatPct(row.maxDrawdownPct)}, trades ${row.tradeCount}, liq ${row.liquidatedPositionCount}${row.stopReason ? `, ${row.stopReason}` : ""}`,
      );
    }
  }

  appendJson(targetOutputPath, {
    type: "status",
    status: "completed",
    completedAt: new Date().toISOString(),
    elapsedMs: Date.now() - startedAt,
  });

  renderReport(rows, windows, configArgs, targetOutputPath, targetReportPath);
  console.error(`Report written to ${path.relative(repoRoot, targetReportPath)}`);
}

function benchmarkConfig(configArgs: Args, thresholdMultiplier: number): PartialStrategyConfig {
  return {
    symbol: "BTCUSDT",
    algorithm: "legacy-valley-peak",
    startingQuote: 10_000,
    maxLeverage: configArgs.leverage,
    shortMarginModel: "futures-margin",
    longBorrowDepth: 999,
    shortBorrowDepth: 999,
    internalBorrowAccounting: "inactive",
    legacyValleyPeak: {
      sigmaMode: "static",
      buySigma: configArgs.buySigma,
      sellSigma: configArgs.sellSigma,
      derivativeSource: "kama",
      derivativeClampMode: "deadband",
      derivativeClampInnerThresholdRatio: 0,
      kamaErLen: configArgs.kamaErLen,
      kamaFastLen: configArgs.kamaFastLen,
      kamaSlowLen: configArgs.kamaSlowLen,
      kamaPower: configArgs.kamaPower,
      rangeLeverageEnabled: !configArgs.forceMaxEntryLeverage,
      rateThresholdsLow: scaleValues(
        defaultLegacyValleyPeakConfig.rateThresholdsLow,
        thresholdMultiplier,
      ),
      rateThresholdsHigh: scaleValues(
        defaultLegacyValleyPeakConfig.rateThresholdsHigh,
        thresholdMultiplier,
      ),
    },
  };
}

function resultRow(
  window: WindowSpec,
  market: ReturnType<typeof summarizeMarket>,
  result: BacktestResult,
  thresholdMultiplier: number,
  elapsedMs: number,
): ResultRow {
  const summary = result.summary;
  return {
    type: "result",
    window: window.label,
    requestedDays: window.requestedDays,
    start: window.start,
    end: window.end,
    actualDays: round(window.actualDays, 4),
    candleCount: window.candleCount,
    thresholdMultiplier,
    marketReturnPct: round(market.returnPct, 4),
    marketSpanPct: round(market.spanPct, 4),
    returnPct: round(summary.returnPct, 4),
    netPnl: round(summary.netPnl, 2),
    finalEquity: round(summary.finalEquity, 2),
    maxDrawdownPct: round(summary.maxDrawdownPct, 4),
    perfectMarginReturnPct: optionalRound(summary.perfectMarginReturnPct, 4),
    perfectMarginCapturePct: optionalRound(summary.perfectMarginCapturePct, 4),
    tradeCount: summary.tradeCount,
    winRate: round(summary.winRate, 2),
    closedPositionCount: summary.closedPositionCount,
    profitableClosedPositionCount: summary.profitableClosedPositionCount,
    liquidatedPositionCount: summary.liquidatedPositionCount,
    maxEntryLeverage: optionalRound(summary.maxEntryLeverage, 4),
    maxEffectiveLeverage: optionalRound(summary.maxEffectiveLeverage, 4),
    stopReason: summary.stopReason,
    elapsedMs,
  };
}

function renderReport(
  rows: ResultRow[],
  windows: WindowSpec[],
  configArgs: Args,
  rawPath: string,
  targetReportPath: string,
): void {
  const summaries = summarizeThresholds(rows).sort((left, right) =>
    right.avgReturnPct - left.avgReturnPct ||
    left.maxDrawdownPct - right.maxDrawdownPct ||
    left.avgTrades - right.avgTrades,
  );
  const baseline =
    summaries.find((summary) => summary.thresholdMultiplier === 1) ?? summaries[0];

  const lines: string[] = [];
  lines.push("# Recent KAMA Threshold Leverage Benchmark");
  lines.push("");
  lines.push(`Generated: ${formatDateTime(Date.now())}`);
  lines.push(`Raw results: \`${path.relative(repoRoot, rawPath)}\``);
  lines.push("");
  lines.push("## Scope");
  lines.push("");
  lines.push(`- Market: BTCUSDT 1m spot candles from \`${path.relative(repoRoot, candleDir)}\`.`);
  lines.push(`- Cache span in this run: ${windows[windows.length - 1]?.start ?? "-"} to ${windows[0]?.end ?? "-"}.`);
  lines.push(`- Windows: ${configArgs.windowsDays.map((days) => `${days}d`).join(", ")} latest available UTC day-file suffixes.`);
  lines.push(`- Strategy: KAMA primary derivative source, deadband clamp, static sigma \`buySigma=${configArgs.buySigma}\`, \`sellSigma=${configArgs.sellSigma}\`, mode both.`);
  lines.push(`- Account model: 10000 USDT starting quote, ${configArgs.leverage}x max leverage, futures-margin shorts, borrow depths \`999/999\`, range-leverage selector ${configArgs.forceMaxEntryLeverage ? "disabled to force max entry leverage" : "enabled"}.`);
  lines.push(`- Threshold multipliers: \`${configArgs.thresholdMultipliers.join(", ")}\`; default 60s threshold is \`0.25\`, so \`2x\` makes it \`0.5\`.`);
  lines.push(`- KAMA: \`erLen=${configArgs.kamaErLen}\`, \`fastLen=${configArgs.kamaFastLen}\`, \`slowLen=${configArgs.kamaSlowLen}\`, \`power=${configArgs.kamaPower}\`.`);
  lines.push("");
  lines.push("## Threshold Summary");
  lines.push("");
  lines.push(summaryHeader());
  for (const summary of summaries) {
    lines.push(summaryLine(summary, baseline));
  }
  lines.push("");
  lines.push("## Window Details");
  lines.push("");
  lines.push(detailHeader());
  for (const row of rows.slice().sort(compareDetailRows)) {
    lines.push(detailLine(row));
  }
  lines.push("");

  fs.mkdirSync(path.dirname(targetReportPath), { recursive: true });
  fs.writeFileSync(targetReportPath, `${lines.join("\n")}\n`);
}

function summarizeThresholds(rows: ResultRow[]): ThresholdSummary[] {
  const groups = new Map<number, ResultRow[]>();
  for (const row of rows) {
    const group = groups.get(row.thresholdMultiplier);
    if (group) {
      group.push(row);
    } else {
      groups.set(row.thresholdMultiplier, [row]);
    }
  }

  return [...groups.entries()].map(([thresholdMultiplier, group]) => ({
    thresholdMultiplier,
    windows: group.length,
    avgReturnPct: round(mean(group.map((row) => row.returnPct)), 4),
    medianReturnPct: round(median(group.map((row) => row.returnPct)), 4),
    positiveCount: group.filter((row) => row.returnPct > 0).length,
    worstReturnPct: round(Math.min(...group.map((row) => row.returnPct)), 4),
    avgMaxDrawdownPct: round(mean(group.map((row) => row.maxDrawdownPct)), 4),
    maxDrawdownPct: round(Math.max(...group.map((row) => row.maxDrawdownPct)), 4),
    totalLiquidations: sum(group.map((row) => row.liquidatedPositionCount)),
    avgTrades: round(mean(group.map((row) => row.tradeCount)), 1),
    maxEffectiveLeverage: round(
      Math.max(...group.map((row) => row.maxEffectiveLeverage ?? 0)),
      4,
    ),
  }));
}

function buildWindows(
  candles: Candle[],
  files: string[],
  windowsDays: number[],
): WindowSpec[] {
  const latest = candles.at(-1);
  if (!latest) {
    throw new Error("No candles loaded.");
  }

  return windowsDays.map((requestedDays) => {
    const selected = files.slice(-Math.min(requestedDays, files.length));
    const firstFile = selected[0];
    if (!firstFile) {
      throw new Error(`No files available for ${requestedDays}d.`);
    }
    const startTime = Date.parse(`${firstFile.slice(0, 10)}T00:00:00Z`);
    const startIndex = lowerBoundOpenTime(candles, startTime);
    const endIndex = candles.length;
    const first = candles[startIndex];
    const last = candles[endIndex - 1];
    if (!first || !last || endIndex <= startIndex) {
      throw new Error(`No candles available for ${requestedDays}d.`);
    }
    return {
      label: `${requestedDays}d`,
      requestedDays,
      startIndex,
      endIndex,
      start: formatDateTime(first.openTime),
      end: formatDateTime(last.closeTime),
      actualDays: (last.closeTime - first.openTime + 1) / DAY_MS,
      candleCount: endIndex - startIndex,
    };
  });
}

function summarizeMarket(
  candles: Candle[],
  startIndex: number,
  endIndex: number,
): {
  returnPct: number;
  spanPct: number;
} {
  const first = candles[startIndex];
  const last = candles[endIndex - 1];
  if (!first || !last || endIndex <= startIndex) {
    throw new Error("Cannot summarize empty candle window.");
  }
  let low = Number.POSITIVE_INFINITY;
  let high = 0;
  for (let index = startIndex; index < endIndex; index += 1) {
    const candle = candles[index];
    low = Math.min(low, candle.low);
    high = Math.max(high, candle.high);
  }
  return {
    returnPct: ((last.close - first.open) / first.open) * 100,
    spanPct: ((high - low) / first.open) * 100,
  };
}

function listCandleFiles(): string[] {
  if (!fs.existsSync(candleDir)) {
    return [];
  }
  return fs
    .readdirSync(candleDir, { withFileTypes: true })
    .filter((entry) => entry.isFile() && entry.name.endsWith(".jsonl"))
    .map((entry) => entry.name)
    .sort();
}

function loadCandles(files: string[]): Candle[] {
  const candles: Candle[] = [];
  for (const file of files) {
    const content = fs.readFileSync(path.join(candleDir, file), "utf8");
    for (const line of content.split("\n")) {
      if (line.trim()) {
        candles.push(JSON.parse(line) as Candle);
      }
    }
  }
  return candles.sort((left, right) => left.openTime - right.openTime);
}

function assertCandles(candles: Candle[]): void {
  if (candles.length === 0) {
    throw new Error("No candles loaded.");
  }
  for (let index = 1; index < candles.length; index += 1) {
    const previous = candles[index - 1];
    const current = candles[index];
    if (!previous || !current || current.openTime < previous.openTime) {
      throw new Error(`Candles are not sorted at index ${index}.`);
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
    outputPath: values.get("output"),
    reportPath: values.get("report"),
    thresholdMultipliers: parseNumberList(
      values.get("threshold-multipliers") ?? "1,1.25,1.5,1.75,2,2.25,2.5,3",
      "threshold-multipliers",
      positiveNumber,
    ),
    windowsDays: parseNumberList(
      values.get("windows-days") ?? "3,7,30,90",
      "windows-days",
      positiveInteger,
    ),
    leverage: positiveNumber(values.get("leverage") ?? "100", "leverage"),
    buySigma: positiveNumber(values.get("buy-sigma") ?? "0.1", "buy-sigma"),
    sellSigma: positiveNumber(values.get("sell-sigma") ?? "0.1", "sell-sigma"),
    kamaErLen: positiveInteger(values.get("kama-er-len") ?? "20", "kama-er-len"),
    kamaFastLen: positiveInteger(values.get("kama-fast-len") ?? "20", "kama-fast-len"),
    kamaSlowLen: positiveInteger(values.get("kama-slow-len") ?? "200", "kama-slow-len"),
    kamaPower: positiveNumber(values.get("kama-power") ?? "1", "kama-power"),
    forceMaxEntryLeverage: values.get("force-max-entry-leverage") === "true",
  };
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

function positiveInteger(value: string, label: string): number {
  return Math.round(positiveNumber(value, label));
}

function lowerBoundOpenTime(candles: Candle[], timestamp: number): number {
  let low = 0;
  let high = candles.length;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    const candle = candles[middle];
    if (candle && candle.openTime < timestamp) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  return low;
}

function scaleValues(values: number[], multiplier: number): number[] {
  return values.map((value) => value * multiplier);
}

function appendJson(filePath: string, value: unknown): void {
  fs.appendFileSync(filePath, `${JSON.stringify(value)}\n`);
}

function compareDetailRows(left: ResultRow, right: ResultRow): number {
  return (
    left.requestedDays - right.requestedDays ||
    left.thresholdMultiplier - right.thresholdMultiplier
  );
}

function summaryHeader(): string {
  return [
    "| Threshold | Avg return | Vs 1x | Median | Positive | Worst | Avg DD | Max DD | Liq | Max eff lev | Avg trades |",
    "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
  ].join("\n");
}

function summaryLine(
  summary: ThresholdSummary,
  baseline: ThresholdSummary | undefined,
): string {
  return [
    `| ${formatMultiplier(summary.thresholdMultiplier)}`,
    formatSignedPct(summary.avgReturnPct),
    formatSignedPct(summary.avgReturnPct - (baseline?.avgReturnPct ?? 0)),
    formatSignedPct(summary.medianReturnPct),
    `${summary.positiveCount}/${summary.windows}`,
    formatSignedPct(summary.worstReturnPct),
    formatPct(summary.avgMaxDrawdownPct),
    formatPct(summary.maxDrawdownPct),
    summary.totalLiquidations.toLocaleString(),
    formatLeverage(summary.maxEffectiveLeverage),
    `${summary.avgTrades.toLocaleString()} |`,
  ].join(" | ");
}

function detailHeader(): string {
  return [
    "| Window | Dates | Actual | Threshold | Market | Bot | Net PnL | Max DD | Perfect capture | Trades | Closed win | Liq | Max eff lev | Stop |",
    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
  ].join("\n");
}

function detailLine(row: ResultRow): string {
  return [
    `| ${row.window}`,
    `${row.start}..${row.end}`,
    `${row.actualDays.toFixed(3)}d`,
    formatMultiplier(row.thresholdMultiplier),
    formatSignedPct(row.marketReturnPct),
    formatSignedPct(row.returnPct),
    formatCurrency(row.netPnl),
    formatPct(row.maxDrawdownPct),
    row.perfectMarginCapturePct === undefined ? "-" : formatSignedPct(row.perfectMarginCapturePct),
    row.tradeCount.toLocaleString(),
    `${row.profitableClosedPositionCount.toLocaleString()}/${row.closedPositionCount.toLocaleString()}`,
    row.liquidatedPositionCount.toLocaleString(),
    formatLeverage(row.maxEffectiveLeverage ?? 0),
    `${row.stopReason ?? "-"} |`,
  ].join(" | ");
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

function optionalRound(value: number | undefined, digits: number): number | undefined {
  return value === undefined || !Number.isFinite(value) ? undefined : round(value, digits);
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

function formatCurrency(value: number): string {
  return `${value >= 0 ? "+" : "-"}${Math.abs(value).toLocaleString("en-US", {
    maximumFractionDigits: 2,
    minimumFractionDigits: 2,
  })}`;
}

function formatMultiplier(value: number): string {
  return `${Number.isInteger(value) ? value.toFixed(0) : value.toFixed(2)}x`;
}

function formatLeverage(value: number): string {
  return `${value.toFixed(3)}x`;
}
