import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  readCandleShardReferenceSync,
  TradingStorageLayout,
  type SequentialCandle,
} from "@trading/storage";
import {
  aggregateLogReturns,
  returnsInWindow,
  summarizeReturnDistribution,
  type ReturnDistribution,
  type TimedReturn,
} from "./lib/log-return-distribution.js";
import {
  buildReturnDistributionChunk,
  StreamingReturnDistribution,
} from "./lib/streaming-return-distribution.js";

const DAY_MS = 86_400_000;
const YEAR_MS = 365 * DAY_MS;
const SCALES = [
  { id: "1s", label: "1 second", intervalMs: 1_000 },
  { id: "1m", label: "1 minute", intervalMs: 60_000 },
  { id: "15m", label: "15 minutes", intervalMs: 15 * 60_000 },
  { id: "1h", label: "1 hour", intervalMs: 60 * 60_000 },
  { id: "4h", label: "4 hours", intervalMs: 4 * 60 * 60_000 },
  { id: "1d", label: "1 day", intervalMs: DAY_MS },
] as const;
const TRAILING_WINDOWS = [
  { id: "7d", label: "Trailing 7 days", durationMs: 7 * DAY_MS },
  { id: "30d", label: "Trailing 30 days", durationMs: 30 * DAY_MS },
  { id: "90d", label: "Trailing 90 days", durationMs: 90 * DAY_MS },
  { id: "365d", label: "Trailing 365 days", durationMs: YEAR_MS },
] as const;

interface Options {
  dataDir: string;
  outputPath: string;
  reportPath?: string;
  requestedEndTime?: number;
  annualWindows: number;
  rollingWindowDays: number;
  rollingStepDays: number;
}

interface WindowResult {
  id: string;
  label: string;
  startTime: string;
  endTime: string;
  durationDays: number;
  scales: ScaleStatistics[];
}

interface WindowDefinition {
  id: string;
  label: string;
  startTime: number;
  endTime: number;
}

interface ScaleStatistics extends ReturnDistribution {
  id: string;
  label: string;
  intervalMs: number;
  annualizedVolatilityPct: number;
  sampleWarning: string | null;
  estimation: "exact" | "exact-moments-approximate-quantiles";
}

interface AnalysisReport {
  version: number;
  generatedAt: string;
  source: {
    market: string;
    symbol: string;
    commonAnalysisEndTime: string;
    oneSecond: SourceSeries;
    oneMinute: SourceSeries;
  };
  methodology: {
    returnDefinition: string;
    alignment: string;
    gapPolicy: string;
    standardization: string;
    annualization: string;
    trailingWindows: string[];
    annualEpochs: string[];
    rollingWindowDays: number;
    rollingStepDays: number;
    oneSecondQuantiles: string;
  };
  fullHistory: WindowResult;
  trailingWindows: WindowResult[];
  annualEpochs: WindowResult[];
  rollingWindows: WindowResult[];
}

interface SourceSeries {
  interval: "1s" | "1m";
  referenceDirectory: string;
  files: number;
  candles: number;
  firstCandleOpenTime: string;
  lastCandleEndTime: string;
}

interface OneSecondSource {
  referenceDirectory: string;
  files: string[];
  firstCandleOpenTime: number;
  lastCandleEndTime: number;
  candles: number;
}

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.stack ?? error.message : String(error));
  process.exitCode = 1;
});

async function main(): Promise<void> {
  const options = parseOptions(process.argv.slice(2));
  const loaded = loadMinuteCandles(options.dataDir);
  const minuteSource: SourceSeries = {
    interval: "1m",
    referenceDirectory: path.relative(repoRoot, loaded.referenceDirectory).replaceAll("\\", "/"),
    files: loaded.files,
    candles: loaded.candles.length,
    firstCandleOpenTime: iso(loaded.candles[0]!.openTime),
    lastCandleEndTime: iso(loaded.candles.at(-1)!.openTime + 60_000),
  };
  const scaleReturns = new Map<string, TimedReturn[]>();
  for (const scale of SCALES.filter((item) => item.id !== "1s")) {
    const returns = aggregateLogReturns(loaded.candles, scale.intervalMs);
    if (returns.length < 2) throw new Error(`Not enough complete ${scale.id} returns.`);
    scaleReturns.set(scale.id, returns);
  }
  loaded.candles.length = 0;
  const oneSecondSource = discoverOneSecondSource(options.dataDir);

  const latestCommonEnd = Math.min(
    oneSecondSource.lastCandleEndTime,
    ...SCALES.filter((scale) => scale.id !== "1s")
      .map((scale) => scaleReturns.get(scale.id)!.at(-1)!.endTime),
  );
  const analysisEnd = options.requestedEndTime ?? latestCommonEnd;
  if (analysisEnd > latestCommonEnd) {
    throw new Error(
      `Requested end ${iso(analysisEnd)} exceeds latest common complete scale ${iso(latestCommonEnd)}.`,
    );
  }
  if (analysisEnd % DAY_MS !== 0) {
    throw new Error("Analysis end must be a UTC day boundary (YYYY-MM-DD).");
  }

  const earliestMinuteReturnEnd = Math.min(
    ...SCALES.filter((scale) => scale.id !== "1s")
      .map((scale) => scaleReturns.get(scale.id)![0]!.endTime),
  );
  const fullStart = Math.max(
    oneSecondSource.firstCandleOpenTime,
    Math.floor((earliestMinuteReturnEnd - 1) / DAY_MS) * DAY_MS,
  );
  const fullDefinition: WindowDefinition = {
    id: "full",
    label: "Full common history",
    startTime: fullStart,
    endTime: analysisEnd,
  };
  const trailingDefinitions: WindowDefinition[] = TRAILING_WINDOWS.map((window) => ({
    id: window.id,
    label: window.label,
    startTime: analysisEnd - window.durationMs,
    endTime: analysisEnd,
  }));
  const annualDefinitions: WindowDefinition[] = Array.from(
    { length: options.annualWindows },
    (_, index) => {
      const epochEnd = subtractUtcYears(analysisEnd, options.annualWindows - index - 1);
      const epochStart = subtractUtcYears(analysisEnd, options.annualWindows - index);
      return {
        id: `${dateOnly(epochStart)}_${dateOnly(epochEnd)}`,
        label: `${dateOnly(epochStart)} to ${dateOnly(epochEnd)}`,
        startTime: epochStart,
        endTime: epochEnd,
      };
    },
  ).filter((window) => window.startTime >= fullStart);
  const rollingDefinitions = buildRollingWindowDefinitions(
    fullStart,
    analysisEnd,
    options.rollingWindowDays,
    options.rollingStepDays,
  );
  const allDefinitions = [
    fullDefinition,
    ...trailingDefinitions,
    ...annualDefinitions,
    ...rollingDefinitions,
  ];
  const oneSecondStatistics = loadOneSecondStatistics(
    oneSecondSource,
    allDefinitions,
    fullStart,
    analysisEnd,
  );
  const fullHistory = analyzeWindow(
    "full",
    "Full common history",
    fullStart,
    analysisEnd,
    scaleReturns,
    oneSecondStatistics,
  );
  const trailingWindows = trailingDefinitions.map((window) => analyzeWindow(
    window.id,
    window.label,
    window.startTime,
    window.endTime,
    scaleReturns,
    oneSecondStatistics,
  ));
  const annualEpochs = annualDefinitions.map((window) => analyzeWindow(
      window.id,
      window.label,
      window.startTime,
      window.endTime,
      scaleReturns,
      oneSecondStatistics,
    ));
  const rollingWindows = rollingDefinitions.map((window) => analyzeWindow(
    window.id,
    window.label,
    window.startTime,
    window.endTime,
    scaleReturns,
    oneSecondStatistics,
  ));

  const report: AnalysisReport = {
    version: 2,
    generatedAt: new Date().toISOString(),
    source: {
      market: "spot-btcusdt",
      symbol: "BTCUSDT",
      commonAnalysisEndTime: iso(analysisEnd),
      oneSecond: {
        interval: "1s",
        referenceDirectory: path.relative(repoRoot, oneSecondSource.referenceDirectory)
          .replaceAll("\\", "/"),
        files: oneSecondSource.files.length,
        candles: oneSecondSource.candles,
        firstCandleOpenTime: iso(oneSecondSource.firstCandleOpenTime),
        lastCandleEndTime: iso(oneSecondSource.lastCandleEndTime),
      },
      oneMinute: minuteSource,
    },
    methodology: {
      returnDefinition: "Natural log of adjacent UTC-aligned bucket close prices.",
      alignment: "Native 1s returns plus non-overlapping 1m, 15m, 1h, 4h, and 1d buckets; every slower scale is derived from one-minute BTCUSDT spot closes.",
      gapPolicy: "A return is omitted unless both buckets are complete and adjacent; gaps are never bridged.",
      standardization: "Within-window sample mean and sample standard deviation.",
      annualization: "Sample standard deviation times sqrt(365 days / native interval).",
      trailingWindows: trailingWindows.map((window) => window.label),
      annualEpochs: annualEpochs.map((window) => window.label),
      rollingWindowDays: options.rollingWindowDays,
      rollingStepDays: options.rollingStepDays,
      oneSecondQuantiles: "Exact moments and counts; deterministic tail-dense weighted-centroid approximation for quantiles and standardized probability masses.",
    },
    fullHistory,
    trailingWindows,
    annualEpochs,
    rollingWindows,
  };

  fs.mkdirSync(path.dirname(options.outputPath), { recursive: true });
  fs.writeFileSync(options.outputPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
  if (options.reportPath) {
    fs.mkdirSync(path.dirname(options.reportPath), { recursive: true });
    fs.writeFileSync(options.reportPath, renderMarkdown(report, options.outputPath), "utf8");
  }
  console.log(`Wrote ${path.relative(repoRoot, options.outputPath)}`);
  if (options.reportPath) console.log(`Wrote ${path.relative(repoRoot, options.reportPath)}`);
  printConsoleSummary(report);
}

function loadMinuteCandles(dataDir: string): {
  referenceDirectory: string;
  files: number;
  candles: SequentialCandle[];
} {
  const referenceDirectory = new TradingStorageLayout(dataDir)
    .candleReferences("spot-btcusdt", "btcusdt", "1m");
  if (!fs.existsSync(referenceDirectory)) {
    throw new Error(`BTCUSDT one-minute reference directory does not exist: ${referenceDirectory}`);
  }
  const files = fs.readdirSync(referenceDirectory, { withFileTypes: true })
    .filter((entry) => entry.isFile() && /^\d{4}-\d{2}-\d{2}\.json$/.test(entry.name))
    .map((entry) => path.join(referenceDirectory, entry.name))
    .sort();
  const candles: SequentialCandle[] = [];
  for (const [index, file] of files.entries()) {
    if (index % 250 === 0) console.error(`Loading minute history ${index}/${files.length}...`);
    candles.push(...readCandleShardReferenceSync(file));
  }
  candles.sort((left, right) => left.openTime - right.openTime);
  const unique: SequentialCandle[] = [];
  for (const candle of candles) {
    const previous = unique.at(-1);
    if (previous?.openTime === candle.openTime) {
      unique[unique.length - 1] = candle;
    } else {
      unique.push(candle);
    }
  }
  if (unique.length === 0) throw new Error("No BTCUSDT one-minute candles were loaded.");
  return { referenceDirectory, files: files.length, candles: unique };
}

function discoverOneSecondSource(dataDir: string): OneSecondSource {
  const referenceDirectory = new TradingStorageLayout(dataDir)
    .candleReferences("spot-btcusdt", "btcusdt", "1s");
  if (!fs.existsSync(referenceDirectory)) {
    throw new Error(`BTCUSDT one-second reference directory does not exist: ${referenceDirectory}`);
  }
  const files = fs.readdirSync(referenceDirectory, { withFileTypes: true })
    .filter((entry) => entry.isFile() && /^\d{4}-\d{2}-\d{2}\.json$/.test(entry.name))
    .map((entry) => path.join(referenceDirectory, entry.name))
    .sort();
  if (files.length === 0) throw new Error("No BTCUSDT one-second shards were found.");
  const firstCandleOpenTime = parseUtcDay(path.basename(files[0]!, ".json"));
  const lastDayStart = parseUtcDay(path.basename(files.at(-1)!, ".json"));
  for (let index = 1; index < files.length; index += 1) {
    const previous = parseUtcDay(path.basename(files[index - 1]!, ".json"));
    const current = parseUtcDay(path.basename(files[index]!, ".json"));
    if (current !== previous + DAY_MS) {
      throw new Error(
        `One-second source has a missing UTC day between ${dateOnly(previous)} and ${dateOnly(current)}.`,
      );
    }
  }
  return {
    referenceDirectory,
    files,
    firstCandleOpenTime,
    lastCandleEndTime: lastDayStart + DAY_MS,
    candles: files.length * 86_400,
  };
}

function loadOneSecondStatistics(
  source: OneSecondSource,
  definitions: readonly WindowDefinition[],
  fullStart: number,
  analysisEnd: number,
): Map<string, ScaleStatistics> {
  const accumulators = new Map(
    definitions.map((definition) => [
      definition.id,
      new StreamingReturnDistribution(800),
    ]),
  );
  let previousClose = Number.NaN;
  let previousOpenTime = Number.NaN;
  for (const [fileIndex, file] of source.files.entries()) {
    const dayStart = parseUtcDay(path.basename(file, ".json"));
    const dayEnd = dayStart + DAY_MS;
    if (dayStart < fullStart || dayEnd > analysisEnd) continue;
    if (fileIndex % 100 === 0) {
      console.error(`Streaming one-second history ${fileIndex}/${source.files.length}...`);
    }
    const candles = readCandleShardReferenceSync(file);
    if (candles.length !== 86_400
      || candles[0]?.openTime !== dayStart
      || candles.at(-1)?.openTime !== dayEnd - 1_000) {
      throw new Error(`${path.basename(file)} is not a complete one-second UTC day.`);
    }
    const values = new Float64Array(candles.length);
    let returnCount = 0;
    for (let index = 0; index < candles.length; index += 1) {
      const candle = candles[index]!;
      if (candle.openTime !== dayStart + index * 1_000
        || candle.closed === false
        || !Number.isFinite(candle.close)
        || candle.close <= 0) {
        throw new Error(`${path.basename(file)} has an invalid one-second candle at ${index}.`);
      }
      if (previousOpenTime === candle.openTime - 1_000
        && Number.isFinite(previousClose)
        && previousClose > 0) {
        values[returnCount] = Math.log(candle.close / previousClose);
        returnCount += 1;
      }
      previousClose = candle.close;
      previousOpenTime = candle.openTime;
    }
    const chunk = buildReturnDistributionChunk(
      values.subarray(0, returnCount),
      dayStart,
      dayEnd,
    );
    for (const definition of definitions) {
      if (dayStart >= definition.startTime && dayEnd <= definition.endTime) {
        accumulators.get(definition.id)!.merge(chunk);
      }
    }
  }
  return new Map(definitions.map((definition) => {
    const summary = accumulators.get(definition.id)!.summarize();
    return [definition.id, scaleStatistics(
      SCALES[0],
      summary,
      "exact-moments-approximate-quantiles",
    )];
  }));
}

function analyzeWindow(
  id: string,
  label: string,
  startTime: number,
  endTime: number,
  returns: ReadonlyMap<string, readonly TimedReturn[]>,
  oneSecondStatistics: ReadonlyMap<string, ScaleStatistics>,
): WindowResult {
  return {
    id,
    label,
    startTime: iso(startTime),
    endTime: iso(endTime),
    durationDays: (endTime - startTime) / DAY_MS,
    scales: SCALES.map((scale) => {
      if (scale.id === "1s") {
        const statistics = oneSecondStatistics.get(id);
        if (!statistics) throw new Error(`Missing one-second statistics for ${id}.`);
        return statistics;
      }
      const values = returnsInWindow(returns.get(scale.id)!, startTime, endTime);
      const summary = summarizeReturnDistribution(values);
      return scaleStatistics(scale, summary, "exact");
    }),
  };
}

function scaleStatistics(
  scale: (typeof SCALES)[number],
  summary: ReturnDistribution,
  estimation: ScaleStatistics["estimation"],
): ScaleStatistics {
  return {
    id: scale.id,
    label: scale.label,
    intervalMs: scale.intervalMs,
    annualizedVolatilityPct: summary.standardDeviationBps / 100
      * Math.sqrt(YEAR_MS / scale.intervalMs),
    sampleWarning: summary.observations < 100
      ? "Shape estimates are unstable below 100 observations."
      : summary.observations < 1_000
        ? "Tail estimates are noisy below 1,000 observations."
        : null,
    estimation,
    ...summary,
  };
}

function buildRollingWindowDefinitions(
  fullStart: number,
  analysisEnd: number,
  windowDays: number,
  stepDays: number,
): WindowDefinition[] {
  const duration = windowDays * DAY_MS;
  const step = stepDays * DAY_MS;
  const firstEnd = Math.ceil((fullStart + duration) / step) * step;
  const windows: WindowDefinition[] = [];
  for (let end = firstEnd; end <= analysisEnd; end += step) {
    windows.push({
      id: `rolling-${windowDays}d-${dateOnly(end)}`,
      label: `${windowDays}d ending ${dateOnly(end)}`,
      startTime: end - duration,
      endTime: end,
    });
  }
  if (windows.at(-1)?.endTime !== analysisEnd) {
    windows.push({
      id: `rolling-${windowDays}d-${dateOnly(analysisEnd)}`,
      label: `${windowDays}d ending ${dateOnly(analysisEnd)}`,
      startTime: analysisEnd - duration,
      endTime: analysisEnd,
    });
  }
  return windows;
}

function renderMarkdown(report: AnalysisReport, outputPath: string): string {
  const fullRows = report.fullHistory.scales.map((scale) => [
    scale.id,
    integer(scale.observations),
    fixed(scale.standardDeviationBps, 3),
    fixed(scale.annualizedVolatilityPct, 2),
    fixed(scale.skewness, 2),
    fixed(scale.excessKurtosis, 1),
    fixed(scale.centralMassVsGaussian, 2),
    fixed(scale.tailMass3SigmaVsGaussian, 1),
    fixed(scale.lag1AbsoluteReturnCorrelation, 3),
  ]);
  const trailingRows = report.trailingWindows.flatMap((window) =>
    window.scales.map((scale) => [
      window.id,
      scale.id,
      integer(scale.observations),
      fixed(scale.annualizedVolatilityPct, 2),
      fixed(scale.skewness, 2),
      fixed(scale.excessKurtosis, 1),
      fixed(scale.tailMass3SigmaVsGaussian, 1),
    ]));
  const epochRows = report.annualEpochs.flatMap((window) =>
    window.scales.map((scale) => [
      window.label.slice(0, 10),
      scale.id,
      fixed(scale.annualizedVolatilityPct, 2),
      fixed(scale.skewness, 2),
      fixed(scale.excessKurtosis, 1),
      fixed(scale.tailMass3SigmaVsGaussian, 1),
      fixed(scale.downsideToUpsideP99, 2),
    ]));
  const insights = deriveInsights(report);
  const relativeOutput = path.relative(repoRoot, outputPath).replaceAll("\\", "/");
  return `# BTCUSDT log-return distributions across scales and history windows

Generated ${report.generatedAt}. Data ends at **${report.source.commonAnalysisEndTime}**.

## Result

${insights.map((item) => `- ${item}`).join("\n")}

## Full-history shape

${markdownTable(
    ["Scale", "n", "σ (bp)", "Ann. vol %", "Skew", "Excess kurt.", "Peak / normal", ">3σ / normal", "ACF |r| lag 1"],
    fullRows,
  )}

\`Peak / normal\` is the observed mass within ±0.25 sample standard deviations,
divided by the Gaussian expectation. \`>3σ / normal\` is the observed two-sided
three-sigma exceedance rate divided by the Gaussian expectation. Values above
one mean a sharper center or heavier tail than a Gaussian with the same variance.

## Dependence on trailing-window length

All rows end at ${report.source.commonAnalysisEndTime}; only the amount of history changes.

${markdownTable(
    ["Window", "Scale", "n", "Ann. vol %", "Skew", "Excess kurt.", ">3σ / normal"],
    trailingRows,
  )}

Daily-scale rows with short lookbacks have too few observations for reliable
shape inference. The JSON includes a \`sampleWarning\` on every affected row.

## Dependence on the particular historical epoch

The one-year epochs are non-overlapping and share the same calendar boundaries.

${markdownTable(
    ["Epoch start", "Scale", "Ann. vol %", "Skew", "Excess kurt.", ">3σ / normal", "Down/up p99"],
    epochRows,
  )}

\`Down/up p99\` compares the magnitude of the centered 1st-percentile loss with
the centered 99th-percentile gain. Values above one indicate the left tail is
larger at that quantile.

## Method

- Source: ${report.source.oneSecond.candles.toLocaleString("en-US")} BTCUSDT spot one-second candles across ${report.source.oneSecond.files.toLocaleString("en-US")} complete daily shards, plus ${report.source.oneMinute.candles.toLocaleString("en-US")} one-minute candles across ${report.source.oneMinute.files.toLocaleString("en-US")} shards.
- Return: native one-second close-to-close log returns and non-overlapping UTC-aligned 1m, 15m, 1h, 4h, and 1d close-to-close log returns.
- The 1m through 1d scales share one minute-close source. The 1s source covers the same full comparison window. Incomplete buckets and returns across gaps are omitted.
- Annualized volatility uses 365-day crypto trading. Shape statistics use each window's own sample mean and sample standard deviation.
- The machine-readable result at \`${relativeOutput}\` also contains quantiles, zero mass, robust scale, return autocorrelation, absolute-return autocorrelation, standardized absolute-tail survival curves, and rolling ${report.methodology.rollingWindowDays}-day estimates stepped every ${report.methodology.rollingStepDays} days. Moments and direct counts are exact at every scale; 1s quantiles and standardized probability masses use a deterministic tail-dense weighted sketch to stay memory-bounded.

## Interpretation limits

- These are unconditional sample distributions, not forecasts and not evidence of independent or identically distributed returns.
- Excess kurtosis and sigma-tail multiples are strongly regime- and sample-dependent because volatility is clustered. A conditional volatility model would remove part, but not all, of the apparent heavy tail.
- Close-to-close returns omit intrabar extremes, spread, fees, slippage, and liquidation paths. One-second closes are last-trade samples, not executable bid/ask returns; tick-size and no-trade seconds create a discrete mass at zero.
- The latest common complete UTC boundary is used; no live or partial candle is included.
`;
}

function deriveInsights(report: AnalysisReport): string[] {
  const full = report.fullHistory.scales;
  const finest = full.find((scale) => scale.id === "1s")!;
  const coarsest = full.find((scale) => scale.id === "1d")!;
  const latestYear = report.trailingWindows.find((window) => window.id === "365d")!;
  const fullOneSecond = finest;
  const latestOneSecond = latestYear.scales.find((scale) => scale.id === "1s")!;
  const epochOneSecond = report.annualEpochs.map((window) => ({
    label: window.label,
    value: window.scales.find((scale) => scale.id === "1s")!.annualizedVolatilityPct,
  }));
  const minVol = epochOneSecond.reduce((left, right) => left.value < right.value ? left : right);
  const maxVol = epochOneSecond.reduce((left, right) => left.value > right.value ? left : right);
  return [
    `The unconditional distribution is sharply peaked and fat-tailed at every scale: full-history central-mass ratios run from ${fixed(Math.min(...full.map((scale) => scale.centralMassVsGaussian ?? Number.NaN)), 2)}× to ${fixed(Math.max(...full.map((scale) => scale.centralMassVsGaussian ?? Number.NaN)), 2)}× Gaussian, while three-sigma events occur ${fixed(Math.min(...full.map((scale) => scale.tailMass3SigmaVsGaussian ?? Number.NaN)), 1)}× to ${fixed(Math.max(...full.map((scale) => scale.tailMass3SigmaVsGaussian ?? Number.NaN)), 1)}× as often.`,
    `Aggregation makes the center more Gaussian but does not eliminate tail risk: excess kurtosis changes from ${fixed(finest.excessKurtosis, 1)} at 1s to ${fixed(coarsest.excessKurtosis, 1)} at 1d.`,
    `The latest 365-day 1s annualized volatility is ${fixed(latestOneSecond.annualizedVolatilityPct, 2)}%, versus ${fixed(fullOneSecond.annualizedVolatilityPct, 2)}% over the full sample; distribution scale is therefore not stationary.`,
    `Window placement matters materially: one-year 1s annualized volatility ranges from ${fixed(minVol.value, 2)}% (${minVol.label}) to ${fixed(maxVol.value, 2)}% (${maxVol.label}).`,
    `Absolute-return lag-1 correlation is ${fixed(finest.lag1AbsoluteReturnCorrelation, 3)} at 1s and ${fixed(coarsest.lag1AbsoluteReturnCorrelation, 3)} at 1d, direct evidence that volatility clustering contributes to the unconditional mixture shape.`,
    `At 1s, ${fixed(finest.zeroFraction * 100, 2)}% of close-to-close returns are exactly zero, so the center contains market microstructure and tick-size effects in addition to continuous price variation.`,
  ];
}

function printConsoleSummary(report: AnalysisReport): void {
  console.log("\nFull-history summary");
  for (const scale of report.fullHistory.scales) {
    console.log(
      `${scale.id.padEnd(3)} n=${integer(scale.observations).padStart(9)} `
      + `annVol=${fixed(scale.annualizedVolatilityPct, 2).padStart(6)}% `
      + `skew=${fixed(scale.skewness, 2).padStart(6)} `
      + `exKurt=${fixed(scale.excessKurtosis, 1).padStart(7)} `
      + `3sigma=${fixed(scale.tailMass3SigmaVsGaussian, 1).padStart(5)}x normal`,
    );
  }
}

function parseOptions(args: string[]): Options {
  if (args.includes("--help")) {
    console.log(`Usage: tsx scripts/analyze-log-return-distributions.ts [options]

Options:
  --data-dir PATH             Trading data root (default: data)
  --output PATH               JSON result (default: data/benchmarks/log-return-distributions.json)
  --report PATH               Optional Markdown report
  --end YYYY-MM-DD            Common exclusive UTC end boundary
  --annual-windows N          Number of one-year epochs (default: 5)
  --rolling-window-days N     Rolling estimator window (default: 90)
  --rolling-step-days N       Rolling estimator step (default: 30)`);
    process.exit(0);
  }
  const values = new Map<string, string>();
  for (let index = 0; index < args.length; index += 1) {
    const key = args[index];
    if (!key?.startsWith("--")) throw new Error(`Unexpected argument: ${key}`);
    const value = args[index + 1];
    if (!value || value.startsWith("--")) throw new Error(`Missing value for ${key}.`);
    values.set(key.slice(2), value);
    index += 1;
  }
  const requestedEnd = values.get("end");
  const requestedEndTime = requestedEnd === undefined
    ? undefined
    : parseUtcDay(requestedEnd);
  return {
    dataDir: path.resolve(repoRoot, values.get("data-dir") ?? "data"),
    outputPath: path.resolve(
      repoRoot,
      values.get("output") ?? "data/benchmarks/log-return-distributions.json",
    ),
    ...(values.has("report")
      ? { reportPath: path.resolve(repoRoot, values.get("report")!) }
      : {}),
    ...(requestedEndTime === undefined ? {} : { requestedEndTime }),
    annualWindows: positiveInteger(values.get("annual-windows") ?? "5", "annual-windows"),
    rollingWindowDays: positiveInteger(
      values.get("rolling-window-days") ?? "90",
      "rolling-window-days",
    ),
    rollingStepDays: positiveInteger(
      values.get("rolling-step-days") ?? "30",
      "rolling-step-days",
    ),
  };
}

function parseUtcDay(value: string): number {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) throw new Error(`Invalid UTC date: ${value}`);
  const result = Date.parse(`${value}T00:00:00.000Z`);
  if (!Number.isFinite(result) || dateOnly(result) !== value) {
    throw new Error(`Invalid UTC date: ${value}`);
  }
  return result;
}

function positiveInteger(value: string, label: string): number {
  const result = Number(value);
  if (!Number.isSafeInteger(result) || result < 1) throw new Error(`${label} must be positive.`);
  return result;
}

function subtractUtcYears(time: number, years: number): number {
  const date = new Date(time);
  return Date.UTC(
    date.getUTCFullYear() - years,
    date.getUTCMonth(),
    date.getUTCDate(),
  );
}

function iso(time: number): string {
  return new Date(time).toISOString();
}

function dateOnly(time: number): string {
  return iso(time).slice(0, 10);
}

function fixed(value: number | null, digits: number): string {
  return value === null || !Number.isFinite(value) ? "n/a" : value.toFixed(digits);
}

function integer(value: number): string {
  return value.toLocaleString("en-US", { maximumFractionDigits: 0 });
}

function markdownTable(headers: string[], rows: string[][]): string {
  const header = `| ${headers.join(" | ")} |`;
  const separator = `| ${headers.map(() => "---").join(" | ")} |`;
  return [header, separator, ...rows.map((row) => `| ${row.join(" | ")} |`)].join("\n");
}
