import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  defaultStrategyConfig,
  runBacktestFromCandles,
  type BacktestResult,
  type Candle,
  type PartialStrategyConfig,
} from "../packages/bot-algo/src/index.js";

type BenchmarkKind = "fixed" | "random-sample";

interface Args {
  outputPath?: string;
  reportPath?: string;
  inputPath?: string;
  renderOnly: boolean;
  seed: number;
}

interface FixedSpec {
  label: string;
  requestedDays: number;
}

interface RandomSpec {
  label: string;
  windowDays: number;
  samples: number;
}

interface JsonMetaRow {
  type: "meta";
  runId: string;
  generatedAt: string;
  symbol: string;
  interval: string;
  marketKey: string;
  candleDir: string;
  dailyFiles: number;
  cacheStart: string;
  cacheEnd: string;
  seed: number;
  fixedSpecs: FixedSpec[];
  randomSpecs: RandomSpec[];
  config: Record<string, unknown>;
}

interface JsonStatusRow {
  type: "status";
  status: "completed";
  completedAt: string;
  elapsedMs: number;
}

interface JsonResultRow {
  type: "result";
  kind: BenchmarkKind;
  group: string;
  label: string;
  requestedDays: number;
  sampleIndex?: number;
  sampleCount?: number;
  startTime: number;
  endTime: number;
  start: string;
  end: string;
  actualDays: number;
  candleCount: number;
  marketReturnPct: number;
  marketSpanPct: number;
  marketLow: number;
  marketHigh: number;
  returnPct: number;
  returnPctPerDay: number;
  netPnl: number;
  netPnlPerDay: number;
  finalEquity: number;
  maxDrawdownPct: number;
  perfectMarginReturnPct?: number;
  perfectMarginCapturePct?: number;
  perfectMarginCompoundedReturnPct?: number;
  perfectMarginCompoundedCapturePct?: number;
  realizedPnl: number;
  unrealizedPnl: number;
  feesPaid: number;
  tradeCount: number;
  winRate: number;
  closedPositionCount: number;
  profitableClosedPositionCount: number;
  profitableClosedPositionRate: number;
  liquidatedPositionCount: number;
  maxEntryLeverage?: number;
  maxEffectiveLeverage?: number;
  stopReason?: string;
  elapsedMs: number;
}

type JsonRow = JsonMetaRow | JsonStatusRow | JsonResultRow;

interface WindowSpec {
  kind: BenchmarkKind;
  group: string;
  label: string;
  requestedDays: number;
  startIndex: number;
  endIndex: number;
  sampleIndex?: number;
  sampleCount?: number;
}

const DAY_MS = 24 * 60 * 60 * 1000;
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const symbol = "BTCUSDT";
const interval = "1m";
const marketKey = "spot-btcusdt";
const candleDir = path.join(
  repoRoot,
  "data",
  "historical",
  marketKey,
  symbol.toLowerCase(),
  interval,
);
const fixedSpecs: FixedSpec[] = [
  { label: "1w", requestedDays: 7 },
  { label: "1m", requestedDays: 30 },
  { label: "3m", requestedDays: 90 },
  { label: "6m", requestedDays: 180 },
  { label: "1y", requestedDays: 365 },
  { label: "5y", requestedDays: 365 * 5 },
];
const randomSpecs: RandomSpec[] = [
  { label: "30d random", windowDays: 30, samples: 20 },
  { label: "1w random", windowDays: 7, samples: 50 },
];

const args = parseArgs(process.argv.slice(2));
const stamp = new Date().toISOString().replace(/[:.]/g, "").replace("T", "-").slice(0, 17);
const outputPath = path.resolve(
  repoRoot,
  args.outputPath ?? `data/benchmarks/static-sigma-0.1-0.1-current-${stamp}.jsonl`,
);
const reportPath = path.resolve(
  repoRoot,
  args.reportPath ?? `docs/static-sigma-0.1-0.1-current-benchmark-${stamp}.md`,
);

if (args.renderOnly) {
  const inputPath = args.inputPath ?? args.outputPath;
  if (!inputPath) {
    throw new Error("--render-only requires --input or --output.");
  }
  renderReport(path.resolve(repoRoot, inputPath), reportPath);
} else {
  runBenchmark(args.seed, outputPath, reportPath);
}

function runBenchmark(seed: number, targetOutputPath: string, targetReportPath: string): void {
  const startedAt = Date.now();
  fs.mkdirSync(path.dirname(targetOutputPath), { recursive: true });
  fs.writeFileSync(targetOutputPath, "");

  const files = listCandleFiles(candleDir);
  if (files.length === 0) {
    throw new Error(`No ${symbol} ${interval} candle files found in ${candleDir}.`);
  }
  const candles = loadCandles(candleDir, files);
  assertCandles(candles);
  const config = benchmarkConfig();

  appendJson(targetOutputPath, {
    type: "meta",
    runId: stamp,
    generatedAt: new Date(startedAt).toISOString(),
    symbol,
    interval,
    marketKey,
    candleDir: path.relative(repoRoot, candleDir),
    dailyFiles: files.length,
    cacheStart: formatDateTime(candles[0].openTime),
    cacheEnd: formatDateTime(candles[candles.length - 1].closeTime),
    seed,
    fixedSpecs,
    randomSpecs,
    config: {
      algorithm: defaultStrategyConfig.algorithm,
      startingQuote: defaultStrategyConfig.startingQuote,
      maxLeverage: defaultStrategyConfig.maxLeverage,
      shortMarginModel: defaultStrategyConfig.shortMarginModel,
      longBorrowDepth: defaultStrategyConfig.longBorrowDepth,
      shortBorrowDepth: defaultStrategyConfig.shortBorrowDepth,
      internalBorrowAccounting: defaultStrategyConfig.internalBorrowAccounting,
      maxPositionQuote: "Infinity",
      maxOpenOrders: defaultStrategyConfig.maxOpenOrders,
      cooldownMs: defaultStrategyConfig.cooldownMs,
      staleOrderMs: defaultStrategyConfig.staleOrderMs,
      minOrderQuote: defaultStrategyConfig.minOrderQuote,
      legacyValleyPeak: config.legacyValleyPeak,
    },
  });

  const windows = [
    ...fixedWindows(candles, files),
    ...randomWindows(candles, seed),
  ];

  console.error(`Writing incremental results to ${path.relative(repoRoot, targetOutputPath)}`);
  for (const window of windows) {
    const candleCount = window.endIndex - window.startIndex;
    const progress =
      window.kind === "random-sample"
        ? `${window.group} sample ${window.sampleIndex}/${window.sampleCount}`
        : window.group;
    console.error(`Running ${progress} on ${candleCount.toLocaleString()} candles...`);
    const startedWindow = Date.now();
    const result = runBacktestFromCandles(candles, {
      config,
      startIndex: window.startIndex,
      endIndex: window.endIndex,
      maxReturnedOrders: 0,
      maxReturnedFills: 0,
      maxEquityPoints: 64,
      maxChartCandles: 1,
    });
    const elapsedMs = Date.now() - startedWindow;
    const row = resultRow(window, candles, result, elapsedMs);
    appendJson(targetOutputPath, row);
    console.error(
      `Finished ${progress}: ${formatPct(row.returnPct)}, max DD ${formatPct(row.maxDrawdownPct)}, ${elapsedMs.toLocaleString()}ms.`,
    );
  }

  appendJson(targetOutputPath, {
    type: "status",
    status: "completed",
    completedAt: new Date().toISOString(),
    elapsedMs: Date.now() - startedAt,
  });

  renderReport(targetOutputPath, targetReportPath);
  console.error(`Report written to ${path.relative(repoRoot, targetReportPath)}`);
}

function benchmarkConfig(): PartialStrategyConfig {
  return {
    symbol,
    algorithm: "legacy-valley-peak",
    startingQuote: defaultStrategyConfig.startingQuote,
    maxLeverage: defaultStrategyConfig.maxLeverage,
    shortMarginModel: defaultStrategyConfig.shortMarginModel,
    longBorrowDepth: defaultStrategyConfig.longBorrowDepth,
    shortBorrowDepth: defaultStrategyConfig.shortBorrowDepth,
    internalBorrowAccounting: defaultStrategyConfig.internalBorrowAccounting,
    borrowerProfitShareToLender: defaultStrategyConfig.borrowerProfitShareToLender,
    maxPositionQuote: defaultStrategyConfig.maxPositionQuote,
    minOrderQuote: defaultStrategyConfig.minOrderQuote,
    maxOpenOrders: defaultStrategyConfig.maxOpenOrders,
    cooldownMs: defaultStrategyConfig.cooldownMs,
    staleOrderMs: defaultStrategyConfig.staleOrderMs,
    legacyValleyPeak: {
      sigmaMode: "static",
      buySigma: 0.1,
      sellSigma: 0.1,
    },
  };
}

function fixedWindows(candles: Candle[], files: string[]): WindowSpec[] {
  return fixedSpecs.map((spec) => {
    const selected = files.slice(-Math.min(spec.requestedDays, files.length));
    const firstFile = selected[0];
    if (!firstFile) {
      throw new Error(`No files available for ${spec.label}.`);
    }
    const startTime = Date.parse(`${firstFile.slice(0, 10)}T00:00:00Z`);
    return {
      kind: "fixed",
      group: spec.label,
      label: `${spec.label} latest`,
      requestedDays: spec.requestedDays,
      startIndex: lowerBoundOpenTime(candles, startTime),
      endIndex: candles.length,
    };
  });
}

function randomWindows(candles: Candle[], seed: number): WindowSpec[] {
  const rng = mulberry32(seed);
  const firstTime = candles[0].openTime;
  const latestCloseTime = candles[candles.length - 1].closeTime;
  const windows: WindowSpec[] = [];

  for (const spec of randomSpecs) {
    const windowMs = spec.windowDays * DAY_MS;
    const maxStart = latestCloseTime - windowMs;
    if (maxStart <= firstTime) {
      throw new Error(`Not enough candle history for ${spec.label}.`);
    }

    for (let sampleIndex = 1; sampleIndex <= spec.samples; sampleIndex += 1) {
      const startTime = randomInt(rng, firstTime, maxStart);
      const endTime = startTime + windowMs;
      const startIndex = lowerBoundOpenTime(candles, startTime);
      const endIndex = upperBoundOpenTime(candles, endTime);
      if (endIndex <= startIndex) {
        sampleIndex -= 1;
        continue;
      }
      windows.push({
        kind: "random-sample",
        group: spec.label,
        label: `${spec.label} #${sampleIndex}`,
        requestedDays: spec.windowDays,
        sampleIndex,
        sampleCount: spec.samples,
        startIndex,
        endIndex,
      });
    }
  }

  return windows;
}

function resultRow(
  window: WindowSpec,
  candles: Candle[],
  result: BacktestResult,
  elapsedMs: number,
): JsonResultRow {
  const first = candles[window.startIndex];
  const last = candles[window.endIndex - 1];
  const market = marketSummary(candles, window.startIndex, window.endIndex);
  const summary = result.summary;
  const metrics = result.finalState.metrics;
  const actualDays = Math.max((last.closeTime - first.openTime + 1) / DAY_MS, 1 / 24 / 60);

  return {
    type: "result",
    kind: window.kind,
    group: window.group,
    label: window.label,
    requestedDays: window.requestedDays,
    sampleIndex: window.sampleIndex,
    sampleCount: window.sampleCount,
    startTime: first.openTime,
    endTime: last.closeTime,
    start: formatDateTime(first.openTime),
    end: formatDateTime(last.closeTime),
    actualDays: round(actualDays, 4),
    candleCount: window.endIndex - window.startIndex,
    marketReturnPct: round(market.returnPct, 4),
    marketSpanPct: round(market.spanPct, 4),
    marketLow: market.low,
    marketHigh: market.high,
    returnPct: round(summary.returnPct, 4),
    returnPctPerDay: round(summary.returnPct / actualDays, 4),
    netPnl: round(summary.netPnl, 2),
    netPnlPerDay: round(summary.netPnl / actualDays, 2),
    finalEquity: round(summary.finalEquity, 2),
    maxDrawdownPct: round(summary.maxDrawdownPct, 4),
    perfectMarginReturnPct: optionalRound(summary.perfectMarginReturnPct, 4),
    perfectMarginCapturePct: optionalRound(summary.perfectMarginCapturePct, 4),
    perfectMarginCompoundedReturnPct: optionalRound(
      summary.perfectMarginCompoundedReturnPct,
      4,
    ),
    perfectMarginCompoundedCapturePct: optionalRound(
      summary.perfectMarginCompoundedCapturePct,
      4,
    ),
    realizedPnl: round(metrics.realizedPnl, 2),
    unrealizedPnl: round(metrics.unrealizedPnl, 2),
    feesPaid: round(metrics.feesPaid, 2),
    tradeCount: summary.tradeCount,
    winRate: round(summary.winRate, 2),
    closedPositionCount: summary.closedPositionCount,
    profitableClosedPositionCount: summary.profitableClosedPositionCount,
    profitableClosedPositionRate: round(summary.profitableClosedPositionRate, 4),
    liquidatedPositionCount: summary.liquidatedPositionCount,
    maxEntryLeverage: optionalRound(
      summary.maxEntryLeverage ?? metrics.maxEntryLeverage,
      4,
    ),
    maxEffectiveLeverage: optionalRound(
      summary.maxEffectiveLeverage ?? metrics.maxEffectiveLeverage,
      4,
    ),
    stopReason: summary.stopReason,
    elapsedMs,
  };
}

function renderReport(inputPath: string, targetReportPath: string): void {
  const rows = readRows(inputPath);
  const meta = rows.find((row): row is JsonMetaRow => row.type === "meta");
  if (!meta) {
    throw new Error(`No meta row found in ${inputPath}.`);
  }
  const completed = rows.some((row) => row.type === "status" && row.status === "completed");
  const results = rows.filter((row): row is JsonResultRow => row.type === "result");
  const fixed = results.filter((row) => row.kind === "fixed");
  const random = results.filter((row) => row.kind === "random-sample");
  const expectedFixed = meta.fixedSpecs.length;
  const expectedRandom = meta.randomSpecs.reduce((total, spec) => total + spec.samples, 0);

  const lines: string[] = [];
  lines.push("# Current Strategy Static Sigma 0.1/0.1 Benchmark");
  lines.push("");
  lines.push(`Generated: ${formatDateTime(Date.now())}`);
  lines.push(`Status: ${completed ? "completed" : "partial"}`);
  lines.push(`Raw results: \`${path.relative(repoRoot, inputPath)}\``);
  lines.push("");
  lines.push("## Scope");
  lines.push("");
  lines.push(`- Market: ${meta.symbol} ${meta.interval} from \`${meta.candleDir}\`.`);
  lines.push(`- Cache span: ${meta.cacheStart} to ${meta.cacheEnd} (${meta.dailyFiles.toLocaleString()} day files).`);
  lines.push("- Strategy: current `legacy-valley-peak` defaults with only sigma mode overridden.");
  lines.push("- Static sigma override: `sigmaMode=static`, `buySigma=0.1`, `sellSigma=0.1`.");
  lines.push(`- Account model: ${meta.config.startingQuote} ${defaultStrategyConfig.quoteAsset} starting quote, ${meta.config.maxLeverage}x max leverage, ${meta.config.shortMarginModel} shorts, borrow depths ${meta.config.longBorrowDepth}/${meta.config.shortBorrowDepth}.`);
  lines.push(`- Random window seed: ${meta.seed}.`);
  lines.push(`- Completed rows: ${fixed.length}/${expectedFixed} fixed windows, ${random.length}/${expectedRandom} random samples.`);
  if ((meta.dailyFiles as number) < 365 * 5) {
    lines.push(`- Note: the requested 5y fixed window is limited by available local cache to ${meta.dailyFiles.toLocaleString()} day files.`);
  }
  lines.push("");
  lines.push("## Fixed Windows");
  lines.push("");
  lines.push("| window | actual span | candles | market | return | net PnL | max DD | perfect | capture | trades | closed win | liq | stop |");
  lines.push("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|");
  for (const row of fixed) {
    lines.push(
      [
        `| ${row.group}`,
        `${row.actualDays.toLocaleString()}d`,
        row.candleCount.toLocaleString(),
        formatSignedPct(row.marketReturnPct),
        formatSignedPct(row.returnPct),
        formatCurrency(row.netPnl),
        formatPct(row.maxDrawdownPct),
        formatOptionalPct(row.perfectMarginReturnPct),
        formatOptionalPct(row.perfectMarginCapturePct),
        row.tradeCount.toLocaleString(),
        `${row.profitableClosedPositionCount.toLocaleString()}/${row.closedPositionCount.toLocaleString()}`,
        row.liquidatedPositionCount.toLocaleString(),
        row.stopReason ?? "-",
      ].join(" | ") + " |",
    );
  }
  if (fixed.length === 0) {
    lines.push("| _none completed_ |  |  |  |  |  |  |  |  |  |  |  |  |");
  }
  lines.push("");
  lines.push("## Random Window Aggregates");
  lines.push("");
  lines.push("| group | samples | profitable | avg return | median | p10 | worst | best | avg/day | avg max DD | avg capture | avg trades | avg liq |");
  lines.push("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
  for (const group of groupRandomRows(random)) {
    lines.push(
      [
        `| ${group.group}`,
        `${group.completed}/${group.expected}`,
        `${group.profitable}/${group.completed}`,
        formatSignedPct(group.avgReturnPct),
        formatSignedPct(group.medianReturnPct),
        formatSignedPct(group.p10ReturnPct),
        formatSignedPct(group.worstReturnPct),
        formatSignedPct(group.bestReturnPct),
        formatSignedPct(group.avgReturnPctPerDay),
        formatPct(group.avgMaxDrawdownPct),
        formatOptionalPct(group.avgPerfectMarginCapturePct),
        formatNumber(group.avgTradeCount, 1),
        formatNumber(group.avgLiquidatedPositionCount, 2),
      ].join(" | ") + " |",
    );
  }
  if (random.length === 0) {
    lines.push("| _none completed_ |  |  |  |  |  |  |  |  |  |  |  |  |");
  }
  lines.push("");
  lines.push("## Random Sample Details");
  lines.push("");
  for (const spec of meta.randomSpecs) {
    const groupRows = random
      .filter((row) => row.group === spec.label)
      .sort((left, right) => (left.sampleIndex ?? 0) - (right.sampleIndex ?? 0));
    lines.push(`### ${spec.label}`);
    lines.push("");
    lines.push("| sample | dates | market | return | net PnL | max DD | capture | trades | closed win | liq | stop |");
    lines.push("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|");
    for (const row of groupRows) {
      lines.push(
        [
          `| ${row.sampleIndex}`,
          `${row.start.slice(0, 10)} to ${row.end.slice(0, 10)}`,
          formatSignedPct(row.marketReturnPct),
          formatSignedPct(row.returnPct),
          formatCurrency(row.netPnl),
          formatPct(row.maxDrawdownPct),
          formatOptionalPct(row.perfectMarginCapturePct),
          row.tradeCount.toLocaleString(),
          `${row.profitableClosedPositionCount.toLocaleString()}/${row.closedPositionCount.toLocaleString()}`,
          row.liquidatedPositionCount.toLocaleString(),
          row.stopReason ?? "-",
        ].join(" | ") + " |",
      );
    }
    if (groupRows.length === 0) {
      lines.push("| _none completed_ |  |  |  |  |  |  |  |  |  |  |");
    }
    lines.push("");
  }

  fs.mkdirSync(path.dirname(targetReportPath), { recursive: true });
  fs.writeFileSync(targetReportPath, `${lines.join("\n").trimEnd()}\n`);
}

function groupRandomRows(rows: JsonResultRow[]): Array<{
  group: string;
  expected: number;
  completed: number;
  profitable: number;
  avgReturnPct: number;
  medianReturnPct: number;
  p10ReturnPct: number;
  worstReturnPct: number;
  bestReturnPct: number;
  avgReturnPctPerDay: number;
  avgMaxDrawdownPct: number;
  avgPerfectMarginCapturePct?: number;
  avgTradeCount: number;
  avgLiquidatedPositionCount: number;
}> {
  return randomSpecs.map((spec) => {
    const groupRows = rows.filter((row) => row.group === spec.label);
    const returns = groupRows.map((row) => row.returnPct);
    return {
      group: spec.label,
      expected: spec.samples,
      completed: groupRows.length,
      profitable: groupRows.filter((row) => row.returnPct > 0).length,
      avgReturnPct: average(returns),
      medianReturnPct: percentile(returns, 0.5),
      p10ReturnPct: percentile(returns, 0.1),
      worstReturnPct: returns.length ? Math.min(...returns) : Number.NaN,
      bestReturnPct: returns.length ? Math.max(...returns) : Number.NaN,
      avgReturnPctPerDay: average(groupRows.map((row) => row.returnPctPerDay)),
      avgMaxDrawdownPct: average(groupRows.map((row) => row.maxDrawdownPct)),
      avgPerfectMarginCapturePct: averageDefined(
        groupRows.map((row) => row.perfectMarginCapturePct),
      ),
      avgTradeCount: average(groupRows.map((row) => row.tradeCount)),
      avgLiquidatedPositionCount: average(
        groupRows.map((row) => row.liquidatedPositionCount),
      ),
    };
  });
}

function listCandleFiles(dir: string): string[] {
  if (!fs.existsSync(dir)) {
    return [];
  }
  return fs
    .readdirSync(dir, { withFileTypes: true })
    .filter((entry) => entry.isFile() && entry.name.endsWith(".jsonl"))
    .map((entry) => entry.name)
    .sort();
}

function loadCandles(dir: string, files: string[]): Candle[] {
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

function assertCandles(candles: Candle[]): void {
  if (candles.length === 0) {
    throw new Error("No candles loaded.");
  }
  for (let index = 1; index < candles.length; index += 1) {
    if (candles[index].openTime < candles[index - 1].openTime) {
      throw new Error("Candles are not sorted by open time.");
    }
  }
}

function marketSummary(
  candles: Candle[],
  startIndex: number,
  endIndex: number,
): { returnPct: number; spanPct: number; low: number; high: number } {
  const first = candles[startIndex];
  const last = candles[endIndex - 1];
  let low = Number.POSITIVE_INFINITY;
  let high = Number.NEGATIVE_INFINITY;
  for (let index = startIndex; index < endIndex; index += 1) {
    low = Math.min(low, candles[index].low);
    high = Math.max(high, candles[index].high);
  }
  return {
    returnPct: first.open > 0 ? ((last.close - first.open) / first.open) * 100 : 0,
    spanPct: first.open > 0 ? ((high - low) / first.open) * 100 : 0,
    low,
    high,
  };
}

function readRows(inputPath: string): JsonRow[] {
  const content = fs.readFileSync(inputPath, "utf8");
  return content
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line) as JsonRow);
}

function appendJson(filePath: string, row: JsonRow): void {
  fs.appendFileSync(filePath, `${JSON.stringify(row)}\n`);
}

function lowerBoundOpenTime(candles: Candle[], target: number): number {
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

function upperBoundOpenTime(candles: Candle[], target: number): number {
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

function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function randomInt(rng: () => number, minInclusive: number, maxInclusive: number): number {
  return Math.floor(minInclusive + rng() * (maxInclusive - minInclusive + 1));
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
    outputPath: values.get("output"),
    reportPath: values.get("report"),
    inputPath: values.get("input"),
    renderOnly: values.get("render-only") === "true",
    seed: positiveInteger(values.get("seed"), 1337),
  };
}

function positiveInteger(value: string | undefined, fallback: number): number {
  if (value === undefined) {
    return fallback;
  }
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error("Expected a positive integer.");
  }
  return Math.round(parsed);
}

function average(values: number[]): number {
  const finite = values.filter(Number.isFinite);
  return finite.length ? finite.reduce((total, value) => total + value, 0) / finite.length : Number.NaN;
}

function averageDefined(values: Array<number | undefined>): number | undefined {
  const finite = values.filter((value): value is number => value !== undefined && Number.isFinite(value));
  return finite.length ? average(finite) : undefined;
}

function percentile(values: number[], ratio: number): number {
  const sorted = values.filter(Number.isFinite).sort((left, right) => left - right);
  if (sorted.length === 0) {
    return Number.NaN;
  }
  const index = Math.min(sorted.length - 1, Math.max(0, Math.floor((sorted.length - 1) * ratio)));
  return sorted[index];
}

function optionalRound(value: number | undefined, digits: number): number | undefined {
  return value === undefined || !Number.isFinite(value) ? undefined : round(value, digits);
}

function round(value: number, digits: number): number {
  return Number.isFinite(value) ? Number(value.toFixed(digits)) : value;
}

function formatDateTime(timestamp: number): string {
  return new Date(timestamp).toISOString().replace(".000Z", "Z");
}

function formatNumber(value: number, digits: number): string {
  return Number.isFinite(value) ? value.toFixed(digits) : "-";
}

function formatPct(value: number): string {
  return Number.isFinite(value) ? `${value.toFixed(4)}%` : "-";
}

function formatSignedPct(value: number): string {
  if (!Number.isFinite(value)) {
    return "-";
  }
  return `${value >= 0 ? "+" : ""}${value.toFixed(4)}%`;
}

function formatOptionalPct(value: number | undefined): string {
  return value === undefined ? "-" : formatSignedPct(value);
}

function formatCurrency(value: number): string {
  if (!Number.isFinite(value)) {
    return "-";
  }
  return `${value >= 0 ? "+" : "-"}$${Math.abs(value).toFixed(2)}`;
}
