import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  runBacktestFromCandles,
  type BacktestResult,
  type Candle,
  type PartialStrategyConfig,
  type ShortMarginModel,
} from "../packages/bot-algo/src/index.js";

interface Args {
  marketKey?: string;
  symbol: string;
  interval: string;
  startingQuote: number;
  minOrderQuote: number;
  leverage: number;
  shortMarginModel: ShortMarginModel;
  longBorrowDepth: number;
  shortBorrowDepth: number;
  maxOpenOrders: number;
  cooldownSec: number;
  samples: number;
  seed: number;
  lookbackDays: number;
  randomWindowDays: number;
  grids: number[];
  maxStepPct: number;
  output?: string;
}

interface GridCandidate {
  label: string;
  value: number;
}

interface CandleWindow {
  label: string;
  startIndex: number;
  endIndex: number;
  durationDays: number;
}

interface Metrics {
  returnPct: number;
  netPnl: number;
  feesPaid: number;
  feePctOfStart: number;
  feePctOfAbsNetPnl: number | undefined;
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
  stopReason: string | undefined;
  candlesProcessed: number | undefined;
  avgExitGridSpan: number;
  avgExitGridOrderCount: number;
  exitGridSpanCount: number;
}

interface FixedRow extends Metrics {
  window: string;
  range: string;
  candidate: string;
}

interface RandomRow {
  candidate: string;
  sampleCount: number;
  profitableSamples: number;
  avgReturnPct: number;
  medianReturnPct: number;
  p10ReturnPct: number;
  worstReturnPct: number;
  bestReturnPct: number;
  avgNetPnl: number;
  avgFeesPaid: number;
  avgFeePctOfStart: number;
  feePctOfAbsAvgNetPnl: number | undefined;
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
  avgExitGridSpan: number;
  avgExitGridOrderCount: number;
  avgExitGridSpanCount: number;
}

const DAY_MS = 24 * 60 * 60 * 1000;
const DEFAULT_GRIDS = [1, 2, 5, 10, 20, 50, 100, 200, 500];
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const args = parseArgs(process.argv.slice(2));
const source = historicalCandleSource(args);
if (source.files.length === 0) {
  const searched = historicalCandleDirCandidates(args)
    .map((dir) => path.relative(repoRoot, dir))
    .join(", ");
  throw new Error(
    `No candles found for ${args.symbol.toUpperCase()} ${args.interval}. Searched: ${searched}.`,
  );
}

const fixedWindows = [
  createFixedFileWindow("last 30d", source, args, 30),
  createFixedFileWindow("last 90d", source, args, 90),
];
const randomCandles = loadHistoricalCandles(
  source.dir,
  source.files.slice(
    -Math.ceil(args.lookbackDays + args.randomWindowDays + 2),
  ),
);
assertCandles(randomCandles);
const randomWindows = createRandomWindows(randomCandles, args);
const fixedRows: FixedRow[] = [];
const randomRows: RandomRow[] = [];
const candidates = gridCandidates(args);

for (const candidate of candidates) {
  for (const window of fixedWindows) {
    console.error(
      `Running ${window.label}, ${candidate.label} on ${window.candles.length.toLocaleString()} candles...`,
    );
    const result = runGridBacktest(window.candles, args, candidate);
    fixedRows.push({
      window: window.label,
      range: rangeLabel(window.candles),
      candidate: candidate.label,
      ...metricsFromResult(result, args),
    });
  }

  console.error(
    `Running ${args.samples.toLocaleString()} random ${args.randomWindowDays}d samples, ${candidate.label}...`,
  );
  const sampleMetrics = randomWindows.map((window) => {
    const result = runGridBacktest(randomCandles, args, candidate, window);
    return metricsFromResult(result, args);
  });
  randomRows.push(aggregateRandomRow(candidate.label, sampleMetrics));
}

const report = buildReport(args, source, fixedWindows, randomCandles, randomWindows, fixedRows, randomRows);
if (args.output) {
  const outputPath = path.resolve(repoRoot, args.output);
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, `${report}\n`, "utf8");
  console.error(`Wrote ${path.relative(repoRoot, outputPath)}`);
}
console.log(report);

function createFixedFileWindow(
  label: string,
  source: { dir: string; files: string[] },
  options: Args,
  days: number,
): { label: string; candles: Candle[] } {
  const files = source.files.slice(-days);
  const candles = loadHistoricalCandles(source.dir, files);
  assertCandles(candles);
  return { label, candles };
}

function runGridBacktest(
  candles: Candle[],
  options: Args,
  candidate: GridCandidate,
  window?: CandleWindow,
): BacktestResult {
  return runBacktestFromCandles(candles, {
    config: {
      symbol: options.symbol.toUpperCase(),
      startingQuote: options.startingQuote,
      minOrderQuote: options.minOrderQuote,
      maxLeverage: options.leverage,
      shortMarginModel: options.shortMarginModel,
      longBorrowDepth: options.longBorrowDepth,
      shortBorrowDepth: options.shortBorrowDepth,
      maxPositionQuote: options.startingQuote,
      maxOpenOrders: options.maxOpenOrders,
      cooldownMs: options.cooldownSec * 1000,
      ...exitGridConfig(options, candidate),
    },
    ...(window ? { startIndex: window.startIndex, endIndex: window.endIndex } : {}),
    maxReturnedOrders: 0,
    maxReturnedFills: 0,
  });
}

function exitGridConfig(options: Args, candidate: GridCandidate): PartialStrategyConfig {
  return {
    staleOrderMs: 30 * DAY_MS,
    legacyValleyPeak: {
      buySigma: 0.3,
      longSideEnabled: true,
      shortSideEnabled: true,
      exitGridEnabled: true,
      exitGridMarketEntry: true,
      exitGridOrderCount: candidate.value,
      exitGridMaxStepPct: options.maxStepPct,
      exitGridPriceDistribution: "uniform",
      exitGridSizeDistribution: "geometric",
      exitGridSellFraction: 0.35,
      exitGridMinProfitBps: 20,
      exitGridResetBps: 10,
      exitGridPositionMode: "per-lot",
      exitGridResetMode: "filled-grid",
    },
  };
}

function metricsFromResult(result: BacktestResult, options: Args): Metrics {
  const summary = result.summary;
  const feesPaid = result.finalState.metrics.feesPaid;
  return {
    returnPct: summary.returnPct,
    netPnl: summary.netPnl,
    feesPaid,
    feePctOfStart: (feesPaid / options.startingQuote) * 100,
    feePctOfAbsNetPnl:
      summary.netPnl !== 0 ? (feesPaid / Math.abs(summary.netPnl)) * 100 : undefined,
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
    stopReason: summary.stopReason,
    candlesProcessed: summary.candlesProcessed,
    avgExitGridSpan: result.finalState.metrics.avgExitGridSpan,
    avgExitGridOrderCount: result.finalState.metrics.avgExitGridOrderCount,
    exitGridSpanCount: result.finalState.metrics.exitGridSpanCount,
  };
}

function aggregateRandomRow(candidate: string, samples: Metrics[]): RandomRow {
  const returns = samples.map((sample) => sample.returnPct);
  const avgNetPnl = average(samples.map((sample) => sample.netPnl));
  const avgFeesPaid = average(samples.map((sample) => sample.feesPaid));
  return {
    candidate,
    sampleCount: samples.length,
    profitableSamples: samples.filter((sample) => sample.netPnl > 0).length,
    avgReturnPct: average(returns),
    medianReturnPct: percentile(returns, 0.5),
    p10ReturnPct: percentile(returns, 0.1),
    worstReturnPct: Math.min(...returns),
    bestReturnPct: Math.max(...returns),
    avgNetPnl,
    avgFeesPaid,
    avgFeePctOfStart: average(samples.map((sample) => sample.feePctOfStart)),
    feePctOfAbsAvgNetPnl:
      avgNetPnl !== 0 ? (avgFeesPaid / Math.abs(avgNetPnl)) * 100 : undefined,
    avgMaxDrawdownPct: average(samples.map((sample) => sample.maxDrawdownPct)),
    avgTradeCount: average(samples.map((sample) => sample.tradeCount)),
    avgRiskAdjustedReturn: averageDefined(
      samples.map((sample) => sample.riskAdjustedReturn),
    ),
    avgSharpeRatio: averageDefined(samples.map((sample) => sample.sharpeRatio)),
    avgPerfectCapturePct: averageDefined(
      samples.map((sample) => sample.perfectCapturePct),
    ),
    avgPerfectCompoundedCapturePct: averageDefined(
      samples.map((sample) => sample.perfectCompoundedCapturePct),
    ),
    avgClosedPositionCount: average(samples.map((sample) => sample.closedPositionCount)),
    avgProfitableClosedPositionCount: average(
      samples.map((sample) => sample.profitableClosedPositionCount),
    ),
    avgProfitableClosedPositionRate: average(
      samples.map((sample) => sample.profitableClosedPositionRate),
    ),
    avgLiquidatedPositionCount: average(
      samples.map((sample) => sample.liquidatedPositionCount),
    ),
    avgExitGridSpan: weightedAverage(
      samples.map((sample) => ({
        value: sample.avgExitGridSpan,
        weight: sample.exitGridSpanCount,
      })),
    ),
    avgExitGridOrderCount: weightedAverage(
      samples.map((sample) => ({
        value: sample.avgExitGridOrderCount,
        weight: sample.exitGridSpanCount,
      })),
    ),
    avgExitGridSpanCount: average(samples.map((sample) => sample.exitGridSpanCount)),
  };
}

function buildReport(
  options: Args,
  source: { dir: string; files: string[] },
  fixedWindows: Array<{ label: string; candles: Candle[] }>,
  randomCandles: Candle[],
  randomWindows: CandleWindow[],
  fixedRows: FixedRow[],
  randomRows: RandomRow[],
): string {
  const command = [
    "npx tsx scripts/benchmark-exit-grid-points.ts",
    `--grids ${options.grids.join(",")}`,
    `--max-step-pct ${options.maxStepPct}`,
    `--min-order-quote ${options.minOrderQuote}`,
    `--max-open-orders ${options.maxOpenOrders}`,
    `--samples ${options.samples}`,
    `--seed ${options.seed}`,
    `--lookback-days ${options.lookbackDays}`,
    `--random-window-days ${options.randomWindowDays}`,
    `--output ${options.output ?? "-"}`,
  ].join(" ");
  const lines = [
    "# Exit Grid Point Sweep",
    "",
    `Command: \`${command}\``,
    "",
    [
      `${options.symbol.toUpperCase()} ${options.interval}`,
      `cache ${path.relative(repoRoot, source.dir)}`,
      `${source.files.length.toLocaleString()} day files`,
      `${formatDate(randomCandles[0].openTime)} to ${formatDate(randomCandles[randomCandles.length - 1].closeTime)} random cache span`,
      `${options.startingQuote.toLocaleString()} starting quote`,
      `$${formatStep(options.minOrderQuote)} min order`,
      `${options.leverage}x max leverage`,
      `${options.shortMarginModel} short margin`,
      `borrow depth L${options.longBorrowDepth}/S${options.shortBorrowDepth}`,
      `open orders cap ${options.maxOpenOrders.toLocaleString()}`,
      `${options.cooldownSec}s cooldown`,
      `max grid step ${formatStep(options.maxStepPct)}%`,
      `grid count caps ${options.grids.map((grid) => grid.toLocaleString()).join(",")}`,
    ].join(", "),
    "",
    "Fee columns use simulated fees paid by filled orders. `Fees/Start` is fees divided by starting quote. `Fees/|Net|` is fees divided by absolute net PnL. `Avg Span` is the average absolute quote-price range between each created grid's break-even edge and peak/trough edge.",
    "",
    "Fixed windows:",
    ...fixedWindows.map(
      (window) =>
        `- ${window.label}: ${rangeLabel(window.candles)}, ${window.candles.length.toLocaleString()} candles`,
    ),
    `- random ${options.randomWindowDays}d samples: ${randomWindows.length.toLocaleString()} samples, lookback ${options.lookbackDays.toLocaleString()}d, seed ${options.seed}`,
    "",
    "## Fixed Windows",
    "",
    fixedWindowTable(fixedRows),
    "",
    `## Random ${options.randomWindowDays}d Samples`,
    "",
    randomWindowTable(randomRows),
  ];
  return lines.join("\n");
}

function fixedWindowTable(rows: FixedRow[]): string {
  return markdownTable(
    [
      "Window",
      "Grid",
      "Return",
      "Net PnL",
      "Fees",
      "Fees/Start",
      "Fees/|Net|",
      "Avg Span",
      "Avg Orders/Grid",
      "Grid Builds",
      "Max DD",
      "Risk Ret",
      "Sharpe",
      "Trades",
      "Win Rate",
      "Prof Pos",
      "Liq Pos",
      "Oracle Cap",
      "Reinvest Cap",
    ],
    rows.map((row) => [
      row.window,
      row.candidate,
      percent(row.returnPct, 2),
      money(row.netPnl),
      money(row.feesPaid),
      percent(row.feePctOfStart, 2),
      optionalPercent(row.feePctOfAbsNetPnl, 1),
      money(row.avgExitGridSpan),
      number(row.avgExitGridOrderCount, 1),
      row.exitGridSpanCount.toLocaleString(),
      percent(row.maxDrawdownPct, 2),
      optionalNumber(row.riskAdjustedReturn, 3),
      optionalNumber(row.sharpeRatio, 3),
      row.tradeCount.toLocaleString(),
      percent(row.winRate, 1),
      `${row.profitableClosedPositionCount.toLocaleString()}/${row.closedPositionCount.toLocaleString()} (${number(row.profitableClosedPositionRate, 1)}%)`,
      number(row.liquidatedPositionCount, 0),
      optionalPercent(row.perfectCapturePct, 3),
      optionalPercent(row.perfectCompoundedCapturePct, 6, true),
    ]),
  );
}

function randomWindowTable(rows: RandomRow[]): string {
  return markdownTable(
    [
      "Grid",
      "Prof Samples",
      "Avg Return",
      "Median",
      "P10",
      "Worst",
      "Best",
      "Avg Net PnL",
      "Avg Fees",
      "Avg Fees/Start",
      "Fees/|Avg Net|",
      "Avg Span",
      "Avg Orders/Grid",
      "Avg Grid Builds",
      "Avg Max DD",
      "Risk Ret",
      "Sharpe",
      "Avg Trades",
      "Avg Prof Pos",
      "Avg Liq Pos",
      "Avg Oracle Cap",
      "Avg Reinvest Cap",
    ],
    rows.map((row) => [
      row.candidate,
      `${row.profitableSamples.toLocaleString()}/${row.sampleCount.toLocaleString()}`,
      percent(row.avgReturnPct, 2),
      percent(row.medianReturnPct, 2),
      percent(row.p10ReturnPct, 2),
      percent(row.worstReturnPct, 2),
      percent(row.bestReturnPct, 2),
      money(row.avgNetPnl),
      money(row.avgFeesPaid),
      percent(row.avgFeePctOfStart, 2),
      optionalPercent(row.feePctOfAbsAvgNetPnl, 1),
      money(row.avgExitGridSpan),
      number(row.avgExitGridOrderCount, 1),
      number(row.avgExitGridSpanCount, 1),
      percent(row.avgMaxDrawdownPct, 2),
      optionalNumber(row.avgRiskAdjustedReturn, 3),
      optionalNumber(row.avgSharpeRatio, 3),
      number(row.avgTradeCount, 1),
      `${number(row.avgProfitableClosedPositionCount, 1)}/${number(row.avgClosedPositionCount, 1)} (${number(row.avgProfitableClosedPositionRate, 1)}%)`,
      number(row.avgLiquidatedPositionCount, 1),
      optionalPercent(row.avgPerfectCapturePct, 3),
      optionalPercent(row.avgPerfectCompoundedCapturePct, 6, true),
    ]),
  );
}

function gridCandidates(options: Args): GridCandidate[] {
  return options.grids.map((value) => ({
    label: value.toLocaleString(),
    value,
  }));
}

function historicalCandleSource(
  options: Pick<Args, "marketKey" | "symbol" | "interval">,
): { dir: string; files: string[] } {
  const candidates = historicalCandleDirCandidates(options)
    .map((dir) => ({ dir, files: listHistoricalCandleFiles(dir) }))
    .sort(
      (left, right) =>
        right.files.length - left.files.length || left.dir.localeCompare(right.dir),
    );

  return candidates[0] ?? {
    dir: fallbackHistoricalCandleDir(options),
    files: [],
  };
}

function historicalCandleDirCandidates(
  options: Pick<Args, "marketKey" | "symbol" | "interval">,
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
    if (entry.isDirectory()) {
      dirs.add(path.join(root, entry.name, symbol, interval));
    }
  }
  return [...dirs];
}

function fallbackHistoricalCandleDir(
  options: Pick<Args, "symbol" | "interval">,
): string {
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

function loadHistoricalCandles(dir: string, files: string[]): Candle[] {
  const candles: Candle[] = [];
  for (const file of files) {
    const content = fs.readFileSync(path.join(dir, file), "utf8");
    for (const line of content.split("\n")) {
      const trimmed = line.trim();
      if (trimmed) {
        candles.push(JSON.parse(trimmed) as Candle);
      }
    }
  }
  return candles.sort((left, right) => left.openTime - right.openTime);
}

function createRandomWindows(candles: Candle[], options: Args): CandleWindow[] {
  const rng = mulberry32(options.seed);
  const firstTime = candles[0].openTime;
  const latestTime = candles[candles.length - 1].closeTime;
  const lookbackStart = Math.max(firstTime, latestTime - options.lookbackDays * DAY_MS);
  const durationMs = options.randomWindowDays * DAY_MS;
  const maxStart = latestTime - durationMs;
  if (maxStart <= lookbackStart) {
    throw new Error("Not enough local candle history for random windows.");
  }

  const windows: CandleWindow[] = [];
  for (let index = 0; index < options.samples; index += 1) {
    const startTime = randomInt(rng, lookbackStart, maxStart);
    const endTime = startTime + durationMs;
    const startIndex = lowerBoundCandleOpenTime(candles, startTime);
    const endIndex = upperBoundCandleOpenTime(candles, endTime);
    if (endIndex <= startIndex) {
      index -= 1;
      continue;
    }
    const first = candles[startIndex];
    const last = candles[endIndex - 1];
    windows.push({
      label: `${formatDate(first.openTime)} to ${formatDate(last.closeTime)}`,
      startIndex,
      endIndex,
      durationDays: Math.max(1 / 24 / 60, (last.closeTime - first.openTime) / DAY_MS),
    });
  }
  return windows;
}

function lowerBoundCandleOpenTime(candles: Candle[], target: number): number {
  let low = 0;
  let high = candles.length;
  while (low < high) {
    const mid = Math.floor((low + high) / 2);
    if (candles[mid].openTime < target) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  return low;
}

function upperBoundCandleOpenTime(candles: Candle[], target: number): number {
  let low = 0;
  let high = candles.length;
  while (low < high) {
    const mid = Math.floor((low + high) / 2);
    if (candles[mid].openTime <= target) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  return low;
}

function parseArgs(argv: string[]): Args {
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

  return {
    marketKey: values.get("market-key"),
    symbol: values.get("symbol") ?? "BTCUSDT",
    interval: values.get("interval") ?? "1m",
    startingQuote: positiveNumber(values.get("starting-quote"), 10_000),
    minOrderQuote: positiveNumber(values.get("min-order-quote"), 25),
    leverage: positiveNumber(values.get("leverage"), 1),
    shortMarginModel:
      values.get("short-margin") === "spot-borrow" ? "spot-borrow" : "futures-margin",
    longBorrowDepth: nonNegativeInt(values.get("long-borrow-depth"), 999),
    shortBorrowDepth: nonNegativeInt(values.get("short-borrow-depth"), 999),
    maxOpenOrders: positiveInt(values.get("max-open-orders"), 1024),
    cooldownSec: positiveNumber(values.get("cooldown-sec"), 300),
    samples: positiveInt(values.get("samples"), 24),
    seed: positiveInt(values.get("seed"), 1337),
    lookbackDays: positiveNumber(values.get("lookback-days"), 365 * 5),
    randomWindowDays: positiveNumber(values.get("random-window-days"), 30),
    grids: parseGridList(values.get("grids")),
    maxStepPct: nonNegativeNumber(values.get("max-step-pct"), 0),
    output: values.get("output"),
  };
}

function parseGridList(value: string | undefined): number[] {
  if (!value) {
    return DEFAULT_GRIDS;
  }
  return parseNumberList(value, DEFAULT_GRIDS).map((part) => Math.round(part));
}

function parseNumberList(value: string | undefined, fallback: number[]): number[] {
  if (!value) {
    return fallback;
  }
  const grids = value
    .split(",")
    .map((part) => Number(part.trim()))
    .filter((part) => Number.isFinite(part) && part > 0);
  return grids.length > 0 ? [...new Set(grids)] : fallback;
}

function assertCandles(candles: Candle[]): asserts candles is [Candle, ...Candle[]] {
  if (candles.length === 0) {
    throw new Error("Backtest requires at least one candle.");
  }
}

function positiveInt(value: string | undefined, fallback: number): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? Math.round(parsed) : fallback;
}

function nonNegativeInt(value: string | undefined, fallback: number): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed >= 0 ? Math.round(parsed) : fallback;
}

function positiveNumber(value: string | undefined, fallback: number): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function nonNegativeNumber(value: string | undefined, fallback: number): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : fallback;
}

function average(values: number[]): number {
  return values.reduce((total, value) => total + value, 0) / values.length;
}

function weightedAverage(values: Array<{ value: number; weight: number }>): number {
  const totalWeight = values.reduce((total, item) => total + item.weight, 0);
  if (totalWeight <= 0) {
    return 0;
  }
  return (
    values.reduce((total, item) => total + item.value * item.weight, 0) /
    totalWeight
  );
}

function averageDefined(values: Array<number | undefined>): number | undefined {
  const finite = values.filter((value): value is number => Number.isFinite(value));
  return finite.length > 0 ? average(finite) : undefined;
}

function percentile(values: number[], ratio: number): number {
  if (values.length === 0) {
    return 0;
  }
  const sorted = [...values].sort((left, right) => left - right);
  const rank = Math.max(0, Math.min(sorted.length - 1, ratio * (sorted.length - 1)));
  const lower = Math.floor(rank);
  const upper = Math.ceil(rank);
  if (lower === upper) {
    return sorted[lower];
  }
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (rank - lower);
}

function mulberry32(seed: number): () => number {
  let value = seed >>> 0;
  return () => {
    value += 0x6d2b79f5;
    let mixed = value;
    mixed = Math.imul(mixed ^ (mixed >>> 15), mixed | 1);
    mixed ^= mixed + Math.imul(mixed ^ (mixed >>> 7), mixed | 61);
    return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296;
  };
}

function randomInt(rng: () => number, min: number, max: number): number {
  return Math.floor(min + rng() * (max - min + 1));
}

function markdownTable(header: string[], rows: string[][]): string {
  return [
    `| ${header.join(" | ")} |`,
    `| ${header.map(() => "---").join(" | ")} |`,
    ...rows.map((row) => `| ${row.join(" | ")} |`),
  ].join("\n");
}

function rangeLabel(candles: Candle[]): string {
  return `${formatDate(candles[0].openTime)} to ${formatDate(candles[candles.length - 1].closeTime)}`;
}

function formatDate(value: number): string {
  return new Date(value).toISOString().slice(0, 10);
}

function money(value: number): string {
  return `$${number(value, 2)}`;
}

function formatStep(value: number): string {
  if (Math.abs(value) >= 1) {
    return number(value, 0);
  }
  return value.toFixed(6).replace(/0+$/, "").replace(/\.$/, "");
}

function percent(value: number, digits: number): string {
  return `${number(value, digits)}%`;
}

function optionalPercent(
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
  return `${number(finite, digits)}%`;
}

function optionalNumber(value: number | undefined, digits: number): string {
  return Number.isFinite(value) ? number(value as number, digits) : "-";
}

function number(value: number, digits: number): string {
  return (Number.isFinite(value) ? value : 0).toFixed(digits);
}

function safePathPart(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9._-]+/g, "-");
}
