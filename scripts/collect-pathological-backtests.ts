import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  defaultStrategyConfig,
  runBacktestFromCandles,
  type BacktestResult,
  type Candle,
  type PartialStrategyConfig,
  type ShortMarginModel,
  type StrategyAlgorithm,
} from "../packages/bot-algo/src/index.js";

type Mode = "collect" | "rerun";

interface PathologyArgs {
  mode: Mode;
  marketKey?: string;
  symbol: string;
  interval: string;
  output: string;
  caseFile: string;
  seed: number;
  samplesPerBucket: number;
  minCases: number;
  lookbackDays: number;
  durations: DurationBucket[];
  startingQuote: number;
  leverage: number;
  shortMarginModel: ShortMarginModel;
  longBorrowDepth: number;
  shortBorrowDepth: number;
  lockBorrowedLenderCollateral: boolean;
  borrowerProfitShareToLender: number;
  maxOpenOrders: number;
  cooldownSec: number;
  resampleMinutes: number;
}

interface DurationBucket {
  label: string;
  days: number;
}

interface StrategySettings {
  startingQuote: number;
  leverage: number;
  shortMarginModel: ShortMarginModel;
  longBorrowDepth: number;
  shortBorrowDepth: number;
  lockBorrowedLenderCollateral: boolean;
  borrowerProfitShareToLender: number;
  maxOpenOrders: number;
  cooldownSec: number;
}

interface CandleSourceSummary {
  symbol: string;
  interval: string;
  marketKey?: string;
  candleDir: string;
  fileCount: number;
  candleCount: number;
  cacheStartTime: number;
  cacheEndTime: number;
  cacheStartIso: string;
  cacheEndIso: string;
}

interface SampleWindow {
  sampleIndex: number;
  bucket: string;
  durationDays: number;
  startTime: number;
  endTime: number;
  startIndex: number;
  endIndex: number;
}

interface CaseMetrics {
  finalEquity: number;
  netPnl: number;
  returnPct: number;
  maxDrawdownPct: number;
  riskAdjustedReturn?: number;
  sharpeRatio?: number;
  tradeCount: number;
  winRate: number;
  closedPositionCount: number;
  profitableClosedPositionCount: number;
  profitableClosedPositionRate: number;
  liquidatedPositionCount: number;
  perfectMarginNetPnl?: number;
  perfectMarginReturnPct?: number;
  perfectMarginCapturePct?: number;
  perfectMarginCompoundedReturnPct?: number;
  perfectMarginCompoundedCapturePct?: number;
  stopReason?: string;
  stoppedEarly?: boolean;
  candlesProcessed?: number;
  eventsProcessed: number;
  replayDurationMs?: number;
}

interface PathologyCase {
  id: string;
  symbol: string;
  interval: string;
  bucket: string;
  durationDays: number;
  startTime: number;
  endTime: number;
  startIso: string;
  endIso: string;
  candles: number;
  reasons: string[];
  oracleCaptureRank?: number;
  metrics: CaseMetrics;
}

interface PathologySuite {
  schemaVersion: 1;
  generatedAt: string;
  rerunCommand: string;
  source: CandleSourceSummary;
  sampling: {
    seed: number;
    samplesPerBucket: number;
    sampleCount: number;
    minCases: number;
    lookbackDays: number;
    buckets: DurationBucket[];
  };
  strategy: {
    label: string;
    algorithm: StrategyAlgorithm;
    settings: StrategySettings;
  };
  pathologies: PathologyCase[];
}

interface HistoricalCandleSource {
  dir: string;
  files: string[];
}

const DAY_MS = 24 * 60 * 60 * 1000;
const DEFAULT_DURATIONS = "1d,3d,5d,1w,2w,1m,2m,3m";
const DEFAULT_OUTPUT = "docs/pathological-backtest-windows.json";
const FULL_BTC_CYCLE_DAYS = 365 * 5;
const STRATEGY_LABEL = "Legacy Valley/Peak Long/Short";
const STRATEGY_ALGORITHM: StrategyAlgorithm = "legacy-valley-peak";
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

try {
  main();
} catch (error) {
  console.error(error instanceof Error ? (error.stack ?? error.message) : String(error));
  process.exitCode = 1;
}

function main(): void {
  const args = parseArgs(process.argv.slice(2));
  if (args.mode === "rerun") {
    rerunSavedPathologies(args);
    return;
  }

  collectPathologies(args);
}

function collectPathologies(args: PathologyArgs): void {
  const source = historicalCandleSource(args);
  if (source.files.length === 0) {
    throw new Error(
      `No candles found for ${args.symbol.toUpperCase()} ${args.interval}. Searched ${path.relative(
        repoRoot,
        source.dir,
      )}.`,
    );
  }

  const maxDurationDays = Math.max(...args.durations.map((bucket) => bucket.days));
  const selectedFiles = source.files.slice(-Math.ceil(args.lookbackDays + maxDurationDays + 2));
  const candles = loadHistoricalCandles(args, source.dir, selectedFiles);
  assertCandles(candles);
  const windows = createRandomWindows(candles, args);
  const settings = strategySettingsFromArgs(args);

  console.error(
    `Running ${windows.length.toLocaleString()} sampled windows across ${args.durations.length} buckets...`,
  );
  const startedAt = Date.now();
  const samples = windows.map((window, index) => {
    const sampleStartedAt = Date.now();
    const result = runPathologyBacktest(candles, window, args, settings);
    const sample = caseFromResult(window, result, [], args.interval);
    console.error(
      [
        `${index + 1}/${windows.length}`,
        sample.bucket,
        `${sample.startIso} to ${sample.endIso}`,
        `return ${formatPercent(sample.metrics.returnPct, 2)}`,
        `capture ${formatOptionalPercent(sample.metrics.perfectMarginCapturePct, 3)}`,
        `${(Date.now() - sampleStartedAt).toLocaleString()}ms`,
      ].join(", "),
    );
    return sample;
  });

  const selected = selectPathologies(samples, args.minCases);
  const suite: PathologySuite = {
    schemaVersion: 1,
    generatedAt: new Date().toISOString(),
    rerunCommand: `npm run backtest:pathologies -- --mode rerun --case-file ${relativeFromRepo(
      args.output,
    )}`,
    source: summarizeSource(args, source, selectedFiles.length, candles),
    sampling: {
      seed: args.seed,
      samplesPerBucket: args.samplesPerBucket,
      sampleCount: samples.length,
      minCases: args.minCases,
      lookbackDays: args.lookbackDays,
      buckets: args.durations,
    },
    strategy: {
      label: STRATEGY_LABEL,
      algorithm: STRATEGY_ALGORITHM,
      settings,
    },
    pathologies: selected,
  };

  fs.mkdirSync(path.dirname(resolveRepoPath(args.output)), { recursive: true });
  fs.writeFileSync(resolveRepoPath(args.output), `${JSON.stringify(suite, null, 2)}\n`);

  console.log(collectionHeader(suite, Date.now() - startedAt));
  console.log("");
  console.log(pathologyTable(selected));
  console.log("");
  console.log(`Saved ${selected.length.toLocaleString()} cases to ${relativeFromRepo(args.output)}`);
  console.log(`Rerun with: ${suite.rerunCommand}`);
}

function rerunSavedPathologies(args: PathologyArgs): void {
  const suite = readPathologySuite(args.caseFile);
  const sourceArgs: PathologyArgs = {
    ...args,
    marketKey: suite.source.marketKey,
    symbol: suite.source.symbol,
    interval: suite.source.interval,
    startingQuote: suite.strategy.settings.startingQuote,
    leverage: suite.strategy.settings.leverage,
    shortMarginModel: suite.strategy.settings.shortMarginModel,
    longBorrowDepth: suite.strategy.settings.longBorrowDepth,
    shortBorrowDepth: suite.strategy.settings.shortBorrowDepth,
    lockBorrowedLenderCollateral: suite.strategy.settings.lockBorrowedLenderCollateral,
    borrowerProfitShareToLender: suite.strategy.settings.borrowerProfitShareToLender,
    maxOpenOrders: suite.strategy.settings.maxOpenOrders,
    cooldownSec: suite.strategy.settings.cooldownSec,
  };
  const source = historicalCandleSource(sourceArgs);
  if (source.files.length === 0) {
    throw new Error(`No candles found for ${suite.source.symbol} ${suite.source.interval}.`);
  }

  const candles = loadHistoricalCandles(sourceArgs, source.dir, source.files);
  assertCandles(candles);
  const settings = suite.strategy.settings;
  const reruns = suite.pathologies.map((savedCase, index) => {
    const startIndex = lowerBoundCandleOpenTime(candles, savedCase.startTime);
    const endIndex = upperBoundCandleOpenTime(candles, savedCase.endTime);
    if (endIndex <= startIndex) {
      throw new Error(`Saved case ${savedCase.id} no longer resolves to local candles.`);
    }
    console.error(
      `Rerunning ${index + 1}/${suite.pathologies.length}: ${savedCase.id} on ${(
        endIndex - startIndex
      ).toLocaleString()} candles...`,
    );
    const window: SampleWindow = {
      sampleIndex: index,
      bucket: savedCase.bucket,
      durationDays: savedCase.durationDays,
      startTime: savedCase.startTime,
      endTime: savedCase.endTime,
      startIndex,
      endIndex,
    };
    const result = runPathologyBacktest(candles, window, sourceArgs, settings);
    return {
      ...caseFromResult(window, result, savedCase.reasons, sourceArgs.interval),
      id: savedCase.id,
      oracleCaptureRank: savedCase.oracleCaptureRank,
      savedReturnPct: savedCase.metrics.returnPct,
      savedCapturePct: savedCase.metrics.perfectMarginCapturePct,
    };
  });

  console.log(`Reran ${reruns.length.toLocaleString()} saved pathological windows.`);
  console.log("");
  console.log(rerunTable(reruns));
}

function runPathologyBacktest(
  candles: Candle[],
  window: SampleWindow,
  args: Pick<PathologyArgs, "symbol">,
  settings: StrategySettings,
): BacktestResult {
  return runBacktestFromCandles(candles, {
    config: strategyConfig(args.symbol, settings),
    startIndex: window.startIndex,
    endIndex: window.endIndex,
    maxEquityPoints: 200,
    maxReturnedOrders: 0,
    maxReturnedFills: 0,
  });
}

function strategyConfig(symbol: string, settings: StrategySettings): PartialStrategyConfig {
  return {
    symbol: symbol.toUpperCase(),
    algorithm: STRATEGY_ALGORITHM,
    startingQuote: settings.startingQuote,
    maxLeverage: settings.leverage,
    shortMarginModel: settings.shortMarginModel,
    longBorrowDepth: settings.longBorrowDepth,
    shortBorrowDepth: settings.shortBorrowDepth,
    lockBorrowedLenderCollateral: settings.lockBorrowedLenderCollateral,
    borrowerProfitShareToLender: settings.borrowerProfitShareToLender,
    maxOpenOrders: settings.maxOpenOrders,
    cooldownMs: settings.cooldownSec * 1000,
  };
}

function caseFromResult(
  window: SampleWindow,
  result: BacktestResult,
  reasons: string[],
  interval: string,
): PathologyCase {
  const summary = result.summary;
  return {
    id: [
      summary.symbol.toLowerCase(),
      window.bucket,
      isoCompact(summary.startTime),
      isoCompact(summary.endTime),
    ].join("-"),
    symbol: summary.symbol,
    interval,
    bucket: window.bucket,
    durationDays: window.durationDays,
    startTime: summary.startTime,
    endTime: summary.endTime,
    startIso: new Date(summary.startTime).toISOString(),
    endIso: new Date(summary.endTime).toISOString(),
    candles: summary.candlesProcessed ?? window.endIndex - window.startIndex,
    reasons,
    metrics: metricsFromResult(result),
  };
}

function metricsFromResult(result: BacktestResult): CaseMetrics {
  const summary = result.summary;
  return {
    finalEquity: round(summary.finalEquity, 8),
    netPnl: round(summary.netPnl, 8),
    returnPct: round(summary.returnPct, 8),
    maxDrawdownPct: round(summary.maxDrawdownPct, 8),
    riskAdjustedReturn: optionalRound(summary.riskAdjustedReturn, 8),
    sharpeRatio: optionalRound(summary.sharpeRatio, 8),
    tradeCount: summary.tradeCount,
    winRate: round(summary.winRate, 8),
    closedPositionCount: summary.closedPositionCount,
    profitableClosedPositionCount: summary.profitableClosedPositionCount,
    profitableClosedPositionRate: round(summary.profitableClosedPositionRate, 8),
    liquidatedPositionCount: summary.liquidatedPositionCount,
    perfectMarginNetPnl: optionalRound(summary.perfectMarginNetPnl, 8),
    perfectMarginReturnPct: optionalRound(summary.perfectMarginReturnPct, 8),
    perfectMarginCapturePct: optionalRound(summary.perfectMarginCapturePct, 8),
    perfectMarginCompoundedReturnPct: optionalRound(
      summary.perfectMarginCompoundedReturnPct,
      8,
    ),
    perfectMarginCompoundedCapturePct: optionalRound(
      summary.perfectMarginCompoundedCapturePct,
      8,
    ),
    stopReason: summary.stopReason,
    stoppedEarly: summary.stoppedEarly,
    candlesProcessed: summary.candlesProcessed,
    eventsProcessed: summary.eventsProcessed,
    replayDurationMs: summary.replayDurationMs,
  };
}

function selectPathologies(samples: PathologyCase[], minCases: number): PathologyCase[] {
  const selected = new Map<string, PathologyCase>();
  for (const sample of samples) {
    if (sample.metrics.netPnl < 0) {
      selected.set(sample.id, {
        ...sample,
        reasons: [...sample.reasons, "negative-profit"],
      });
    }
  }

  const rankedByCapture = [...samples]
    .filter((sample) => Number.isFinite(sample.metrics.perfectMarginCapturePct))
    .sort(
      (left, right) =>
        (left.metrics.perfectMarginCapturePct as number) -
          (right.metrics.perfectMarginCapturePct as number) ||
        left.metrics.returnPct - right.metrics.returnPct,
    );

  const captureCount = Math.max(minCases, Math.min(10, rankedByCapture.length));
  for (let index = 0; index < captureCount; index += 1) {
    const sample = rankedByCapture[index];
    const existing = selected.get(sample.id);
    selected.set(sample.id, {
      ...(existing ?? sample),
      oracleCaptureRank: index + 1,
      reasons: [...new Set([...(existing?.reasons ?? sample.reasons), "low-oracle-capture"])],
    });
  }

  const pathologies = [...selected.values()].sort(comparePathologies);
  if (pathologies.length < minCases) {
    throw new Error(
      `Only found ${pathologies.length.toLocaleString()} pathological cases; requested ${minCases.toLocaleString()}.`,
    );
  }
  return pathologies;
}

function comparePathologies(left: PathologyCase, right: PathologyCase): number {
  if (left.metrics.netPnl < 0 && right.metrics.netPnl >= 0) {
    return -1;
  }
  if (left.metrics.netPnl >= 0 && right.metrics.netPnl < 0) {
    return 1;
  }
  const leftRank = left.oracleCaptureRank ?? Number.POSITIVE_INFINITY;
  const rightRank = right.oracleCaptureRank ?? Number.POSITIVE_INFINITY;
  return (
    leftRank - rightRank ||
    left.metrics.returnPct - right.metrics.returnPct ||
    left.startTime - right.startTime
  );
}

function createRandomWindows(candles: Candle[], args: PathologyArgs): SampleWindow[] {
  const rng = mulberry32(args.seed);
  const firstTime = candles[0].openTime;
  const latestTime = candles[candles.length - 1].closeTime;
  const lookbackStart = Math.max(firstTime, latestTime - args.lookbackDays * DAY_MS);
  const windows: SampleWindow[] = [];

  for (const bucket of args.durations) {
    const durationMs = bucket.days * DAY_MS;
    const maxStart = latestTime - durationMs;
    if (maxStart <= lookbackStart) {
      throw new Error(
        `Not enough local candle history for ${bucket.label}; cache span is ${formatDays(
          (latestTime - firstTime) / DAY_MS,
        )} days.`,
      );
    }

    const seen = new Set<string>();
    for (let index = 0; index < args.samplesPerBucket; index += 1) {
      let startIndex = -1;
      let endIndex = -1;
      let startTime = 0;
      let endTime = 0;
      for (let attempt = 0; attempt < 100; attempt += 1) {
        startTime = randomInt(rng, lookbackStart, maxStart);
        endTime = startTime + durationMs;
        startIndex = lowerBoundCandleOpenTime(candles, startTime);
        endIndex = upperBoundCandleOpenTime(candles, endTime);
        const key = `${bucket.label}:${startIndex}:${endIndex}`;
        if (endIndex > startIndex && !seen.has(key)) {
          seen.add(key);
          break;
        }
      }

      if (endIndex <= startIndex) {
        throw new Error(`Failed to sample a non-empty ${bucket.label} window.`);
      }

      windows.push({
        sampleIndex: windows.length,
        bucket: bucket.label,
        durationDays: bucket.days,
        startTime,
        endTime,
        startIndex,
        endIndex,
      });
    }
  }

  return windows;
}

function summarizeSource(
  args: PathologyArgs,
  source: HistoricalCandleSource,
  fileCount: number,
  candles: [Candle, ...Candle[]],
): CandleSourceSummary {
  const first = candles[0];
  const last = candles[candles.length - 1];
  return {
    symbol: args.symbol.toUpperCase(),
    interval: args.interval,
    ...(args.marketKey ? { marketKey: args.marketKey } : {}),
    candleDir: relativeFromRepo(source.dir),
    fileCount,
    candleCount: candles.length,
    cacheStartTime: first.openTime,
    cacheEndTime: last.closeTime,
    cacheStartIso: new Date(first.openTime).toISOString(),
    cacheEndIso: new Date(last.closeTime).toISOString(),
  };
}

function loadHistoricalCandles(
  args: Pick<PathologyArgs, "resampleMinutes">,
  dir: string,
  files: string[],
): Candle[] {
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

  return resampleCandles(
    candles.sort((left, right) => left.openTime - right.openTime),
    args.resampleMinutes,
  );
}

function historicalCandleSource(
  args: Pick<PathologyArgs, "marketKey" | "symbol" | "interval">,
): HistoricalCandleSource {
  const candidates = historicalCandleDirCandidates(args)
    .map((dir) => ({
      dir,
      files: listHistoricalCandleFiles(dir),
    }))
    .sort(
      (left, right) =>
        right.files.length - left.files.length || left.dir.localeCompare(right.dir),
    );

  return (
    candidates[0] ?? {
      dir: fallbackHistoricalCandleDir(args),
      files: [],
    }
  );
}

function historicalCandleDirCandidates(
  args: Pick<PathologyArgs, "marketKey" | "symbol" | "interval">,
): string[] {
  const root = path.join(repoRoot, "data", "historical");
  const symbol = safePathPart(args.symbol);
  const interval = safePathPart(args.interval);
  const dirs = new Set<string>();

  if (args.marketKey) {
    dirs.add(path.join(root, safePathPart(args.marketKey), symbol, interval));
  }

  dirs.add(fallbackHistoricalCandleDir(args));

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
  args: Pick<PathologyArgs, "symbol" | "interval">,
): string {
  return path.join(
    repoRoot,
    "data",
    "historical",
    safePathPart(args.symbol),
    safePathPart(args.interval),
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

function readPathologySuite(caseFile: string): PathologySuite {
  const content = fs.readFileSync(resolveRepoPath(caseFile), "utf8");
  const suite = JSON.parse(content) as PathologySuite;
  if (suite.schemaVersion !== 1 || !Array.isArray(suite.pathologies)) {
    throw new Error(`${caseFile} is not a supported pathology suite file.`);
  }
  return suite;
}

function parseArgs(argv: string[]): PathologyArgs {
  const values = parseCliValues(argv);
  const modeValue = values.get("mode");
  const output = values.get("output") ?? DEFAULT_OUTPUT;
  const caseFile = values.get("case-file") ?? values.get("cases") ?? output;
  return {
    mode: modeValue === "rerun" || values.get("rerun") === "true" ? "rerun" : "collect",
    marketKey: values.get("market-key"),
    symbol: values.get("symbol") ?? "BTCUSDT",
    interval: values.get("interval") ?? "1m",
    output,
    caseFile,
    seed: parsePositiveInt(values.get("seed"), 20260624),
    samplesPerBucket: parsePositiveInt(values.get("samples-per-bucket"), 2),
    minCases: parsePositiveInt(values.get("min-cases"), 5),
    lookbackDays: parsePositiveNumber(values.get("lookback-days"), FULL_BTC_CYCLE_DAYS),
    durations: parseDurations(values.get("durations") ?? DEFAULT_DURATIONS),
    startingQuote: parsePositiveNumber(values.get("starting-quote"), 10_000),
    leverage: parsePositiveNumber(values.get("leverage"), defaultStrategyConfig.maxLeverage),
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
    maxOpenOrders: parsePositiveInt(values.get("max-open-orders"), 1024),
    cooldownSec: parsePositiveNumber(values.get("cooldown-sec"), 300),
    resampleMinutes: parsePositiveInt(values.get("resample-minutes"), 1),
  };
}

function parseCliValues(argv: string[]): Map<string, string> {
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
  return values;
}

function parseDurations(value: string): DurationBucket[] {
  const buckets = value
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean)
    .map((label) => ({ label, days: durationSpecToDays(label) }));
  if (buckets.length === 0) {
    throw new Error("At least one duration bucket is required.");
  }
  return buckets;
}

function durationSpecToDays(spec: string): number {
  const match = /^(\d+(?:\.\d+)?)(d|w|m|mo)$/i.exec(spec);
  if (!match) {
    throw new Error(`Unsupported duration "${spec}". Use d, w, m, or mo.`);
  }
  const value = Number(match[1]);
  const unit = match[2].toLowerCase();
  const multiplier = unit === "d" ? 1 : unit === "w" ? 7 : 30;
  return value * multiplier;
}

function parseShortMarginModel(value: string | undefined): ShortMarginModel {
  return value === "spot-borrow" ? "spot-borrow" : "futures-margin";
}

function strategySettingsFromArgs(args: PathologyArgs): StrategySettings {
  return {
    startingQuote: args.startingQuote,
    leverage: args.leverage,
    shortMarginModel: args.shortMarginModel,
    longBorrowDepth: args.longBorrowDepth,
    shortBorrowDepth: args.shortBorrowDepth,
    lockBorrowedLenderCollateral: args.lockBorrowedLenderCollateral,
    borrowerProfitShareToLender: args.borrowerProfitShareToLender,
    maxOpenOrders: args.maxOpenOrders,
    cooldownSec: args.cooldownSec,
  };
}

function collectionHeader(suite: PathologySuite, elapsedMs: number): string {
  return [
    "Pathological backtest collection",
    `${suite.source.symbol} ${suite.source.interval}`,
    `${suite.sampling.sampleCount.toLocaleString()} samples`,
    `${suite.pathologies.length.toLocaleString()} saved cases`,
    `seed ${suite.sampling.seed}`,
    `${suite.source.cacheStartIso} to ${suite.source.cacheEndIso} cache span`,
    `${elapsedMs.toLocaleString()}ms`,
  ].join(", ");
}

function pathologyTable(cases: PathologyCase[]): string {
  return markdownTable(
    [
      "Case",
      "Bucket",
      "Window",
      "Reasons",
      "Return",
      "Net PnL",
      "Max DD",
      "Oracle Capture",
      "Trades",
      "Liq Pos",
    ],
    cases.map((caseItem) => [
      caseItem.id,
      caseItem.bucket,
      `${caseItem.startIso.slice(0, 10)} to ${caseItem.endIso.slice(0, 10)}`,
      caseItem.reasons.join(", "),
      formatPercent(caseItem.metrics.returnPct, 2),
      formatMoney(caseItem.metrics.netPnl),
      formatPercent(caseItem.metrics.maxDrawdownPct, 2),
      formatOptionalPercent(caseItem.metrics.perfectMarginCapturePct, 3),
      caseItem.metrics.tradeCount.toLocaleString(),
      caseItem.metrics.liquidatedPositionCount.toLocaleString(),
    ]),
  );
}

function rerunTable(
  cases: Array<
    PathologyCase & {
      savedReturnPct: number;
      savedCapturePct?: number;
    }
  >,
): string {
  return markdownTable(
    [
      "Case",
      "Bucket",
      "Return",
      "Saved Return",
      "Return Delta",
      "Oracle Capture",
      "Saved Capture",
      "Trades",
    ],
    cases.map((caseItem) => [
      caseItem.id,
      caseItem.bucket,
      formatPercent(caseItem.metrics.returnPct, 2),
      formatPercent(caseItem.savedReturnPct, 2),
      formatPercent(caseItem.metrics.returnPct - caseItem.savedReturnPct, 4),
      formatOptionalPercent(caseItem.metrics.perfectMarginCapturePct, 3),
      formatOptionalPercent(caseItem.savedCapturePct, 3),
      caseItem.metrics.tradeCount.toLocaleString(),
    ]),
  );
}

function markdownTable(header: string[], rows: string[][]): string {
  return [
    `| ${header.join(" | ")} |`,
    `| ${header.map(() => "---").join(" | ")} |`,
    ...rows.map((row) => `| ${row.join(" | ")} |`),
  ].join("\n");
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
    if (current) {
      resampled.push({
        ...current,
        volume,
        closed: true,
      });
    }
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

function assertCandles(candles: Candle[]): asserts candles is [Candle, ...Candle[]] {
  if (candles.length === 0) {
    throw new Error("Backtest requires at least one candle.");
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

function parseFiniteNumber(value: string | undefined, fallback: number): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function round(value: number, digits: number): number {
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
}

function optionalRound(value: number | undefined, digits: number): number | undefined {
  return Number.isFinite(value) ? round(value as number, digits) : undefined;
}

function isoCompact(value: number): string {
  return new Date(value).toISOString().replace(/[-:.]/g, "").replace("T", "t").slice(0, 15);
}

function resolveRepoPath(value: string): string {
  return path.isAbsolute(value) ? value : path.join(repoRoot, value);
}

function relativeFromRepo(value: string): string {
  return path.relative(repoRoot, resolveRepoPath(value)) || ".";
}

function safePathPart(value: string): string {
  return value.replace(/[^a-z0-9_-]+/gi, "-").replace(/^-+|-+$/g, "").toLowerCase();
}

function formatDays(value: number): string {
  return (Number.isFinite(value) ? value : 0).toFixed(1);
}

function formatMoney(value: number): string {
  return `$${formatNumber(value, 2)}`;
}

function formatPercent(value: number, digits: number): string {
  return `${formatNumber(value, digits)}%`;
}

function formatOptionalPercent(value: number | undefined, digits: number): string {
  return Number.isFinite(value) ? `${formatNumber(value as number, digits)}%` : "-";
}

function formatNumber(value: number, digits: number): string {
  return (Number.isFinite(value) ? value : 0).toFixed(digits);
}
