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
  type TimedReturn,
} from "./lib/log-return-distribution.js";
import {
  fitEllipticalGeneralizedGaussianPower,
  fitRadialCandidates,
  jensenShannonBits,
  PairMoments,
  STANDARDIZED_PAIR_EDGES,
  summarizePairShape,
  type PairMomentsSnapshot,
  type PairSample,
  type PairShapeStatistics,
  type RadialCandidateFit,
} from "./lib/consecutive-return-pairs.js";

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

interface Options {
  dataDir: string;
  outputPath: string;
  reportPath: string;
  requestedEndTime?: number;
  sampleTarget: number;
  fitSampleLimit: number;
}

interface WindowDefinition {
  id: string;
  label: string;
  kind: "full" | "annual-epoch" | "trailing";
  startTime: number;
  endTime: number;
}

interface WindowAccumulator {
  definition: WindowDefinition;
  moments: PairMoments;
  sample: PairSample;
  samplingProbability: number;
}

interface WindowScaleResult {
  id: string;
  label: string;
  kind: WindowDefinition["kind"];
  startTime: string;
  endTime: string;
  durationDays: number;
  observations: number;
  meanXBps: number;
  meanYBps: number;
  standardDeviationXBps: number;
  standardDeviationYBps: number;
  returnCorrelation: number | null;
  absoluteReturnCorrelation: number | null;
  squaredReturnCorrelation: number | null;
  zeroZeroFraction: number;
  anyZeroFraction: number;
  nonzeroSameSignFraction: number | null;
  quadrants: PairMomentsSnapshot["quadrants"];
  continuousObservations: number;
  continuousCorrelation: number | null;
  sampleObservations: number;
  estimation: {
    moments: "exact";
    shape: "deterministic-uniform-sample";
    samplingProbability: number;
  };
  shape: PairShapeStatistics;
  generalizedGaussianApproximation: {
    family: "elliptical-generalized-gaussian";
    power: number;
    scale: number;
    observations: number;
  };
  similarityToFull?: {
    unconditionalJensenShannonBits: number;
    continuousJensenShannonBits: number;
  };
  sampleWarning: string | null;
}

interface ScaleResult {
  id: string;
  label: string;
  intervalMs: number;
  fullHistory: WindowScaleResult;
  annualEpochs: WindowScaleResult[];
  trailingWindows: WindowScaleResult[];
  fullHistoryModelFits: RadialCandidateFit[];
  selectedFullHistoryModel: RadialCandidateFit;
}

interface AnalysisReport {
  version: number;
  generatedAt: string;
  source: {
    market: "spot-btcusdt";
    symbol: "BTCUSDT";
    analysisStartTime: string;
    analysisEndTime: string;
    durationDays: number;
    oneSecond: SourceDescription;
    oneMinute: SourceDescription;
  };
  methodology: Record<string, string | number | string[]>;
  histogram: {
    standardizedEdges: number[];
    layout: string;
  };
  scales: ScaleResult[];
  crossScaleSimilarity: Array<{
    leftScale: string;
    rightScale: string;
    unconditionalJensenShannonBits: number;
    continuousJensenShannonBits: number;
  }>;
  annualWindowStability: Array<{
    scale: string;
    returnCorrelationRange: [number, number] | null;
    absoluteReturnCorrelationRange: [number, number] | null;
    generalizedGaussianPowerRange: [number, number] | null;
    continuousJensenShannonBitsMedian: number;
    continuousJensenShannonBitsMaximum: number;
  }>;
}

interface SourceDescription {
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
}

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.stack ?? error.message : String(error));
  process.exitCode = 1;
});

async function main(): Promise<void> {
  const options = parseOptions(process.argv.slice(2));
  const minute = loadMinuteCandles(options.dataDir);
  const oneSecond = discoverOneSecondSource(options.dataDir);
  const minuteLastEnd = minute.candles.at(-1)!.openTime + 60_000;
  const latestCommonDay = Math.floor(
    Math.min(minuteLastEnd, oneSecond.lastCandleEndTime) / DAY_MS,
  ) * DAY_MS;
  const analysisEnd = options.requestedEndTime ?? latestCommonDay;
  if (analysisEnd > latestCommonDay) {
    throw new Error(`Requested end ${iso(analysisEnd)} exceeds latest common day ${iso(latestCommonDay)}.`);
  }
  if (analysisEnd % DAY_MS !== 0) throw new Error("Analysis end must be a UTC day boundary.");
  const analysisStart = subtractUtcYears(analysisEnd, 5);
  if (analysisStart < oneSecond.firstCandleOpenTime
    || analysisStart < minute.candles[0]!.openTime) {
    throw new Error("The local candle store does not cover the requested five-year window.");
  }
  const definitions = buildWindowDefinitions(analysisStart, analysisEnd);
  const scaleResults: ScaleResult[] = [];
  const minuteCandleCount = minute.candles.length;

  console.error(`Analyzing exact five-year window ${dateOnly(analysisStart)}..${dateOnly(analysisEnd)}.`);
  for (const scale of SCALES) {
    console.error(`Preparing consecutive ${scale.id} return pairs...`);
    const accumulators = createAccumulators(
      definitions,
      scale.intervalMs,
      options.sampleTarget,
    );
    if (scale.id === "1s") {
      processOneSecondPairs(oneSecond, accumulators, analysisStart, analysisEnd);
    } else {
      const returns = aggregateLogReturns(minute.candles, scale.intervalMs);
      processTimedReturnPairs(returns, accumulators, scale.intervalMs);
    }
    const windows = accumulators.map((accumulator) => finalizeWindow(
      accumulator,
      options.fitSampleLimit,
    ));
    const fullHistory = windows.find((window) => window.kind === "full")!;
    for (const window of windows) {
      if (window === fullHistory) continue;
      window.similarityToFull = {
        unconditionalJensenShannonBits: jensenShannonBits(
          window.shape.unconditionalHistogram,
          fullHistory.shape.unconditionalHistogram,
        ),
        continuousJensenShannonBits: jensenShannonBits(
          window.shape.continuousHistogram,
          fullHistory.shape.continuousHistogram,
        ),
      };
    }
    const fullAccumulator = accumulators.find((item) => item.definition.kind === "full")!;
    console.error(`Fitting candidate ${scale.id} joint distributions...`);
    const fits = fitRadialCandidates(
      fullAccumulator.sample,
      fullAccumulator.moments.snapshot(),
      options.fitSampleLimit,
    );
    scaleResults.push({
      id: scale.id,
      label: scale.label,
      intervalMs: scale.intervalMs,
      fullHistory,
      annualEpochs: windows.filter((window) => window.kind === "annual-epoch"),
      trailingWindows: windows.filter((window) => window.kind === "trailing"),
      fullHistoryModelFits: fits,
      selectedFullHistoryModel: fits[0]!,
    });
  }
  minute.candles.length = 0;

  const crossScaleSimilarity = scaleResults.slice(1).map((right, index) => {
    const left = scaleResults[index]!;
    return {
      leftScale: left.id,
      rightScale: right.id,
      unconditionalJensenShannonBits: jensenShannonBits(
        left.fullHistory.shape.unconditionalHistogram,
        right.fullHistory.shape.unconditionalHistogram,
      ),
      continuousJensenShannonBits: jensenShannonBits(
        left.fullHistory.shape.continuousHistogram,
        right.fullHistory.shape.continuousHistogram,
      ),
    };
  });
  const annualWindowStability = scaleResults.map((scale) => {
    const continuousDistances = scale.annualEpochs
      .map((window) => window.similarityToFull!.continuousJensenShannonBits)
      .sort((left, right) => left - right);
    return {
      scale: scale.id,
      returnCorrelationRange: finiteRange(scale.annualEpochs.map((item) => item.returnCorrelation)),
      absoluteReturnCorrelationRange: finiteRange(
        scale.annualEpochs.map((item) => item.absoluteReturnCorrelation),
      ),
      generalizedGaussianPowerRange: finiteRange(
        scale.annualEpochs.map((item) => item.generalizedGaussianApproximation.power),
      ),
      continuousJensenShannonBitsMedian: quantile(continuousDistances, 0.5),
      continuousJensenShannonBitsMaximum: continuousDistances.at(-1)!,
    };
  });
  const report: AnalysisReport = {
    version: 2,
    generatedAt: new Date().toISOString(),
    source: {
      market: "spot-btcusdt",
      symbol: "BTCUSDT",
      analysisStartTime: iso(analysisStart),
      analysisEndTime: iso(analysisEnd),
      durationDays: (analysisEnd - analysisStart) / DAY_MS,
      oneSecond: sourceDescription(oneSecond, "1s"),
      oneMinute: {
        interval: "1m",
        referenceDirectory: relative(minute.referenceDirectory),
        files: minute.files,
        candles: minuteCandleCount,
        firstCandleOpenTime: iso(minute.firstCandleOpenTime),
        lastCandleEndTime: iso(minute.lastCandleEndTime),
      },
    },
    methodology: {
      target: "Ordered adjacent pair (r_t, r_{t+1}) of close-to-close natural-log returns.",
      scales: SCALES.map((scale) => scale.id),
      alignment: "Native one-second closes and non-overlapping UTC-aligned 1m, 15m, 1h, 4h, and 1d closes.",
      gapPolicy: "Both returns must be adjacent at the native scale and both return endpoints must be strictly inside the selected window; gaps are never bridged.",
      windows: "One exact five-calendar-year window, five non-overlapping one-year epochs, and trailing 30d, 90d, and 365d windows sharing the analysis end.",
      standardization: "Each coordinate uses its own within-window mean and sample standard deviation. Continuous fits additionally exclude pairs with either exact-zero return and use that component's covariance whitening.",
      zeroPolicy: "Exact zero-zero and one-zero pair masses are reported as discrete atoms. Continuous densities are fitted only where both returns are nonzero.",
      sampling: `All linear, absolute, squared, sign, quadrant, and zero statistics are exact. Histograms, nonlinear tail statistics, and fits use a deterministic uniform time-hash sample targeting ${options.sampleTarget.toLocaleString("en-US")} pairs per window.`,
      fitting: `Maximum likelihood on at most ${options.fitSampleLimit.toLocaleString("en-US")} evenly spread sampled continuous pairs; AIC compares Gaussian, Student-t, elliptical and product generalized Gaussian, elliptical and product generalized t, and an elliptical lognormal-radius family. All candidates share the empirical continuous mean/covariance standardization.`,
      generalizedGaussianFormula: "f(z)=p/[2*pi*s^2*Gamma(2/p)]*exp(-(rho(z)/s)^p) for the elliptical bivariate generalized Gaussian.",
      generalizedTFormula: "f(z)=p/[2*pi*s^2*B(2/p,q-2/p)]*[1+(rho(z)/s)^p]^(-q) for the elliptical bivariate generalized t.",
      radialLognormalFormula: "f(z)=exp(-(log(rho(z))-m)^2/(2*tau^2))/[2*pi*sqrt(2*pi)*tau*rho(z)^2] for the elliptical lognormal-radius family.",
      modelLimit: "Candidate likelihoods compare symmetric stationary continuous families. They do not model skew, volatility regimes, or the discrete zero axes.",
    },
    histogram: {
      standardizedEdges: [...STANDARDIZED_PAIR_EDGES],
      layout: "Row-major y-by-x probabilities, including the two infinite overflow edges.",
    },
    scales: scaleResults,
    crossScaleSimilarity,
    annualWindowStability,
  };
  fs.mkdirSync(path.dirname(options.outputPath), { recursive: true });
  fs.writeFileSync(
    options.outputPath,
    `${JSON.stringify(report, numberReplacer, 2)}\n`,
    "utf8",
  );
  fs.mkdirSync(path.dirname(options.reportPath), { recursive: true });
  fs.writeFileSync(options.reportPath, renderMarkdown(report, options.outputPath), "utf8");
  console.log(`Wrote ${relative(options.outputPath)}`);
  console.log(`Wrote ${relative(options.reportPath)}`);
  printSummary(report);
}

function loadMinuteCandles(dataDir: string): {
  referenceDirectory: string;
  files: number;
  candles: SequentialCandle[];
  firstCandleOpenTime: number;
  lastCandleEndTime: number;
} {
  const referenceDirectory = new TradingStorageLayout(dataDir)
    .candleReferences("spot-btcusdt", "btcusdt", "1m");
  const files = fs.readdirSync(referenceDirectory, { withFileTypes: true })
    .filter((entry) => entry.isFile() && /^\d{4}-\d{2}-\d{2}\.json$/.test(entry.name))
    .map((entry) => path.join(referenceDirectory, entry.name))
    .sort();
  if (files.length === 0) throw new Error("No BTCUSDT one-minute candle shards were found.");
  const candles: SequentialCandle[] = [];
  for (const [index, file] of files.entries()) {
    if (index % 250 === 0) console.error(`Loading minute history ${index}/${files.length}...`);
    candles.push(...readCandleShardReferenceSync(file));
  }
  candles.sort((left, right) => left.openTime - right.openTime);
  const unique: SequentialCandle[] = [];
  for (const candle of candles) {
    const previous = unique.at(-1);
    if (previous?.openTime === candle.openTime) unique[unique.length - 1] = candle;
    else unique.push(candle);
  }
  return {
    referenceDirectory,
    files: files.length,
    candles: unique,
    firstCandleOpenTime: unique[0]!.openTime,
    lastCandleEndTime: unique.at(-1)!.openTime + 60_000,
  };
}

function discoverOneSecondSource(dataDir: string): OneSecondSource {
  const referenceDirectory = new TradingStorageLayout(dataDir)
    .candleReferences("spot-btcusdt", "btcusdt", "1s");
  const files = fs.readdirSync(referenceDirectory, { withFileTypes: true })
    .filter((entry) => entry.isFile() && /^\d{4}-\d{2}-\d{2}\.json$/.test(entry.name))
    .map((entry) => path.join(referenceDirectory, entry.name))
    .sort();
  if (files.length === 0) throw new Error("No BTCUSDT one-second candle shards were found.");
  for (let index = 1; index < files.length; index += 1) {
    const previous = parseUtcDay(path.basename(files[index - 1]!, ".json"));
    const current = parseUtcDay(path.basename(files[index]!, ".json"));
    if (current !== previous + DAY_MS) {
      throw new Error(`Missing one-second day between ${dateOnly(previous)} and ${dateOnly(current)}.`);
    }
  }
  return {
    referenceDirectory,
    files,
    firstCandleOpenTime: parseUtcDay(path.basename(files[0]!, ".json")),
    lastCandleEndTime: parseUtcDay(path.basename(files.at(-1)!, ".json")) + DAY_MS,
  };
}

function sourceDescription(source: OneSecondSource, interval: "1s"): SourceDescription {
  return {
    interval,
    referenceDirectory: relative(source.referenceDirectory),
    files: source.files.length,
    candles: source.files.length * 86_400,
    firstCandleOpenTime: iso(source.firstCandleOpenTime),
    lastCandleEndTime: iso(source.lastCandleEndTime),
  };
}

function buildWindowDefinitions(startTime: number, endTime: number): WindowDefinition[] {
  const annualEpochs = Array.from({ length: 5 }, (_, index): WindowDefinition => ({
    id: `year-${index + 1}-${dateOnly(subtractUtcYears(endTime, 5 - index))}`,
    label: `${dateOnly(subtractUtcYears(endTime, 5 - index))} to ${dateOnly(subtractUtcYears(endTime, 4 - index))}`,
    kind: "annual-epoch",
    startTime: subtractUtcYears(endTime, 5 - index),
    endTime: subtractUtcYears(endTime, 4 - index),
  }));
  return [
    {
      id: "full-5y",
      label: "Full five-year history",
      kind: "full",
      startTime,
      endTime,
    },
    ...annualEpochs,
    {
      id: "trailing-30d",
      label: "Trailing 30 days",
      kind: "trailing",
      startTime: endTime - 30 * DAY_MS,
      endTime,
    },
    {
      id: "trailing-90d",
      label: "Trailing 90 days",
      kind: "trailing",
      startTime: endTime - 90 * DAY_MS,
      endTime,
    },
    {
      id: "trailing-365d",
      label: "Trailing 365 days",
      kind: "trailing",
      startTime: endTime - YEAR_MS,
      endTime,
    },
  ];
}

function createAccumulators(
  definitions: readonly WindowDefinition[],
  intervalMs: number,
  sampleTarget: number,
): WindowAccumulator[] {
  return definitions.map((definition) => {
    const expectedPairs = Math.max(1, (definition.endTime - definition.startTime) / intervalMs - 1);
    return {
      definition,
      moments: new PairMoments(),
      sample: { x: [], y: [] },
      samplingProbability: Math.min(1, sampleTarget / expectedPairs),
    };
  });
}

function processTimedReturnPairs(
  returns: readonly TimedReturn[],
  accumulators: readonly WindowAccumulator[],
  intervalMs: number,
): void {
  for (let index = 1; index < returns.length; index += 1) {
    const previous = returns[index - 1]!;
    const current = returns[index]!;
    if (current.endTime !== previous.endTime + intervalMs) continue;
    const sampleHash = hashUnit(Math.floor(current.endTime / intervalMs), intervalMs);
    for (const accumulator of accumulators) {
      const definition = accumulator.definition;
      if (previous.endTime <= definition.startTime || current.endTime > definition.endTime) continue;
      accumulator.moments.add(previous.value, current.value);
      if (sampleHash < accumulator.samplingProbability) {
        accumulator.sample.x.push(previous.value);
        accumulator.sample.y.push(current.value);
      }
    }
  }
}

function processOneSecondPairs(
  source: OneSecondSource,
  accumulators: readonly WindowAccumulator[],
  analysisStart: number,
  analysisEnd: number,
): void {
  let previousClose = Number.NaN;
  let previousCandleTime = Number.NaN;
  let previousReturn = Number.NaN;
  let previousReturnEnd = Number.NaN;
  const firstReadDay = analysisStart - DAY_MS;
  for (const [fileIndex, file] of source.files.entries()) {
    const dayStart = parseUtcDay(path.basename(file, ".json"));
    const dayEnd = dayStart + DAY_MS;
    if (dayStart < firstReadDay || dayEnd > analysisEnd) continue;
    if (fileIndex % 100 === 0) {
      console.error(`Streaming one-second history ${fileIndex}/${source.files.length}...`);
    }
    const candles = readCandleShardReferenceSync(file);
    validateOneSecondDay(candles, dayStart, file);
    const active = accumulators.filter((accumulator) => (
      dayStart >= accumulator.definition.startTime
      && dayEnd <= accumulator.definition.endTime
    ));
    const allDay = new PairMoments();
    const insideDay = new PairMoments();
    for (let index = 0; index < candles.length; index += 1) {
      const candle = candles[index]!;
      if (previousCandleTime === candle.openTime - 1_000 && previousClose > 0) {
        const currentReturn = Math.log(candle.close / previousClose);
        const currentReturnEnd = candle.openTime + 1_000;
        if (previousReturnEnd === currentReturnEnd - 1_000 && Number.isFinite(previousReturn)) {
          allDay.add(previousReturn, currentReturn);
          if (previousReturnEnd > dayStart) insideDay.add(previousReturn, currentReturn);
          if (active.length > 0) {
            const sampleHash = hashUnit(Math.floor(currentReturnEnd / 1_000), 1_000);
            for (const accumulator of active) {
              if (previousReturnEnd <= accumulator.definition.startTime) continue;
              if (sampleHash < accumulator.samplingProbability) {
                accumulator.sample.x.push(previousReturn);
                accumulator.sample.y.push(currentReturn);
              }
            }
          }
        }
        previousReturn = currentReturn;
        previousReturnEnd = currentReturnEnd;
      } else {
        previousReturn = Number.NaN;
        previousReturnEnd = Number.NaN;
      }
      previousClose = candle.close;
      previousCandleTime = candle.openTime;
    }
    for (const accumulator of active) {
      accumulator.moments.merge(
        accumulator.definition.startTime === dayStart ? insideDay : allDay,
      );
    }
  }
}

function validateOneSecondDay(
  candles: readonly SequentialCandle[],
  dayStart: number,
  file: string,
): void {
  if (candles.length !== 86_400
    || candles[0]?.openTime !== dayStart
    || candles.at(-1)?.openTime !== dayStart + DAY_MS - 1_000) {
    throw new Error(`${path.basename(file)} is not a complete one-second UTC day.`);
  }
  for (let index = 0; index < candles.length; index += 1) {
    const candle = candles[index]!;
    if (candle.openTime !== dayStart + index * 1_000
      || candle.closed === false
      || !(candle.close > 0)) {
      throw new Error(`${path.basename(file)} has an invalid candle at index ${index}.`);
    }
  }
}

function finalizeWindow(
  accumulator: WindowAccumulator,
  fitSampleLimit: number,
): WindowScaleResult {
  const moments = accumulator.moments.snapshot();
  const shape = summarizePairShape(accumulator.sample, moments);
  const generalizedGaussian = fitEllipticalGeneralizedGaussianPower(
    accumulator.sample,
    moments,
    fitSampleLimit,
  );
  return {
    id: accumulator.definition.id,
    label: accumulator.definition.label,
    kind: accumulator.definition.kind,
    startTime: iso(accumulator.definition.startTime),
    endTime: iso(accumulator.definition.endTime),
    durationDays: (accumulator.definition.endTime - accumulator.definition.startTime) / DAY_MS,
    observations: moments.observations,
    meanXBps: moments.meanX * 10_000,
    meanYBps: moments.meanY * 10_000,
    standardDeviationXBps: moments.standardDeviationX * 10_000,
    standardDeviationYBps: moments.standardDeviationY * 10_000,
    returnCorrelation: moments.correlation,
    absoluteReturnCorrelation: moments.absoluteCorrelation,
    squaredReturnCorrelation: moments.squaredCorrelation,
    zeroZeroFraction: moments.zeroZeroFraction,
    anyZeroFraction: moments.anyZeroFraction,
    nonzeroSameSignFraction: moments.nonzeroSameSignFraction,
    quadrants: moments.quadrants,
    continuousObservations: moments.continuous.observations,
    continuousCorrelation: moments.continuous.correlation,
    sampleObservations: accumulator.sample.x.length,
    estimation: {
      moments: "exact",
      shape: "deterministic-uniform-sample",
      samplingProbability: accumulator.samplingProbability,
    },
    shape,
    generalizedGaussianApproximation: {
      family: "elliptical-generalized-gaussian",
      ...generalizedGaussian,
    },
    sampleWarning: moments.observations < 100
      ? "Fewer than 100 pairs; all shape and fit estimates are unstable."
      : moments.observations < 1_000
        ? "Fewer than 1,000 pairs; tail and model-selection estimates are noisy."
        : null,
  };
}

function renderMarkdown(report: AnalysisReport, outputPath: string): string {
  const fullRows = report.scales.map((scale) => {
    const full = scale.fullHistory;
    const selected = scale.selectedFullHistoryModel;
    return [
      scale.id,
      integer(full.observations),
      fixed(full.returnCorrelation, 4),
      fixed(full.absoluteReturnCorrelation, 3),
      fixed(full.nonzeroSameSignFraction === null ? null : 100 * full.nonzeroSameSignFraction, 2),
      fixed(full.shape.jointAbsoluteTail2SigmaLift, 2),
      fixed(100 * full.zeroZeroFraction, 2),
      fixed(full.generalizedGaussianApproximation.power, 3),
      familyLabel(selected.family),
    ];
  });
  const candidateRows = report.scales.flatMap((scale) => scale.fullHistoryModelFits.map((fit) => [
    scale.id,
    familyLabel(fit.family),
    fixed(fit.parameters.power, 3),
    fixed(fit.parameters.tail, 3),
    fixed(fit.parameters.degreesFreedom, 2),
    fixed(fit.parameters.logRadiusMean, 3),
    fixed(fit.parameters.logRadiusStandardDeviation, 3),
    fixed(fit.nllPerObservation, 5),
    deltaAicText(fit.deltaAic),
  ]));
  const scaleSimilarityRows = report.crossScaleSimilarity.map((item) => [
    `${item.leftScale} → ${item.rightScale}`,
    fixed(item.unconditionalJensenShannonBits, 4),
    fixed(item.continuousJensenShannonBits, 4),
  ]);
  const stabilityRows = report.annualWindowStability.map((item) => [
    item.scale,
    rangeText(item.returnCorrelationRange, 4),
    rangeText(item.absoluteReturnCorrelationRange, 3),
    rangeText(item.generalizedGaussianPowerRange, 3),
    fixed(item.continuousJensenShannonBitsMedian, 4),
    fixed(item.continuousJensenShannonBitsMaximum, 4),
  ]);
  const daily = report.scales.find((scale) => scale.id === "1d")!;
  const dailyFit = daily.selectedFullHistoryModel;
  const dailyRunnerUp = daily.fullHistoryModelFits[1]!;
  const winnerSummary = [...new Set(report.scales.map(
    (scale) => scale.selectedFullHistoryModel.family,
  ))].map((family) => `${familyLabel(family)} at ${report.scales
    .filter((scale) => scale.selectedFullHistoryModel.family === family)
    .map((scale) => scale.id)
    .join(", ")}`).join("; ");
  const adjacentContinuousJs = report.crossScaleSimilarity
    .map((item) => item.continuousJensenShannonBits);
  const absoluteCorrelations = report.scales
    .map((scale) => scale.fullHistory.absoluteReturnCorrelation)
    .filter((value): value is number => value !== null);
  return `# Consecutive BTCUSDT log-return pairs across scales and windows

Generated ${report.generatedAt}. The exact common history is **${report.source.analysisStartTime} to ${report.source.analysisEndTime}** (exclusive end).

## Result

- Consecutive signed returns are nearly uncorrelated, but their magnitudes are not: full-history return correlations span ${fixed(Math.min(...report.scales.map((scale) => scale.fullHistory.returnCorrelation ?? 0)), 4)} to ${fixed(Math.max(...report.scales.map((scale) => scale.fullHistory.returnCorrelation ?? 0)), 4)}, while absolute-return correlations span ${fixed(Math.min(...absoluteCorrelations), 3)} to ${fixed(Math.max(...absoluteCorrelations), 3)}. The joint distribution therefore contains volatility-state dependence that two independent marginals miss.
- After removing exact-zero axes and standardizing each window, adjacent-scale 2D shape distances are ${fixed(Math.min(...adjacentContinuousJs), 4)}–${fixed(Math.max(...adjacentContinuousJs), 4)} bits of Jensen–Shannon divergence. The shapes are related, but not scale-invariant.
- AIC winners by scale are ${winnerSummary}.
- At 1d the closest tested continuous family is **${familyLabel(dailyFit.family)}**${dailyFit.parameters.power === null ? "" : ` with power $p=${fixed(dailyFit.parameters.power, 3)}$`}; ${familyLabel(dailyRunnerUp.family)} follows at $\\Delta$AIC ${fixed(dailyRunnerUp.deltaAic, 2)}.
- The 1s distribution is a mixed distribution, not a single continuous density: ${fixed(100 * report.scales[0]!.fullHistory.zeroZeroFraction, 2)}% of pairs are exactly $(0,0)$ and ${fixed(100 * report.scales[0]!.fullHistory.anyZeroFraction, 2)}% lie on at least one zero axis. Model selection below applies only where both returns are nonzero.

## Full-history pair statistics

${markdownTable(
    ["Scale", "Pairs", "Corr r", "Corr magnitude", "Same sign %", "Joint >2σ lift", "(0,0) %", "GGD p", "AIC winner"],
    fullRows,
  )}

\`Joint >2σ lift\` is $P(|X|>2,|Y|>2)/[P(|X|>2)P(|Y|>2)]$. A value above one means large adjacent moves cluster more than independent marginals predict. \`GGD p\` is a common elliptical generalized-Gaussian shape fit used as a comparable shape index even where another family wins.

## Closest continuous distribution

The elliptical bivariate generalized Gaussian is

$$
f(z)=\\frac{p}{2\\pi s^2|R|^{1/2}\\Gamma(2/p)}
\\exp\\!\\left[-\\left(\\frac{\\sqrt{z^\\top R^{-1}z}}{s}\\right)^p\\right].
$$

The elliptical generalized t replaces the stretched-exponential kernel with

$$
f(z)=\\frac{p}{2\\pi s^2|R|^{1/2}B(2/p,q-2/p)}
\\left[1+\\left(\\frac{\\sqrt{z^\\top R^{-1}z}}{s}\\right)^p\\right]^{-q}.
$$

For radial-lognormal $m$ and $\\tau$, the covariance-whitened radius $\\rho=\\lVert z\\rVert_2$ has $\\log\\rho\\sim\\mathcal N(m,\\tau^2)$, giving

$$
f(z)=\\frac{1}{2\\pi\\sqrt{2\\pi}\\tau\\rho^2}
\\exp\\!\\left[-\\frac{(\\log\\rho-m)^2}{2\\tau^2}\\right].
$$

The product variants apply the analogous 1D density independently along the whitened common/difference axes. They test whether an axis-shaped density is closer than an elliptical common-radius density. AIC uses the same deterministic continuous-pair sample for all candidates within a scale; only differences within a scale are meaningful.

${markdownTable(
    ["Scale", "Candidate", "p", "q", "ν", "mean log ρ", "sd log ρ", "NLL / pair", "ΔAIC"],
    candidateRows,
  )}

## Shape similarity across scales

Every histogram is standardized within its own scale before comparison. The unconditional comparison retains zero atoms; the continuous comparison removes every pair with either return exactly zero.

${markdownTable(
    ["Adjacent scales", "JS bits, all pairs", "JS bits, continuous"],
    scaleSimilarityRows,
  )}

Jensen–Shannon divergence is zero only for identical binned shapes and at most one bit. Daily comparisons are noisier because the five-year daily sample contains only about 1,825 pairs.

## Stability across one-year epochs

${markdownTable(
    ["Scale", "Corr r range", "Corr magnitude range", "GGD p range", "Median JS to 5y", "Max JS to 5y"],
    stabilityRows,
  )}

The machine-readable JSON also contains trailing 30d, 90d, and 365d windows, quadrant probabilities, squared-return correlation, joint 2σ/3σ lifts, the response of next-return magnitude after a 2σ move, whitened radial quantiles, angular non-ellipticity, and the normalized 2D histograms.

## Interpretation

- The best family here is an unconditional descriptive law, not a predictive transition model. Magnitude correlation and joint-tail lift show that $r_{t+1}$ is not independent of $r_t$ even when signed correlation is near zero.
- A symmetric elliptical density can represent common stochastic volatility through a shared radius, but it cannot represent time-varying volatility regimes exactly. The product alternatives test a different contour shape, not a full volatility model.
- The daily result has limited power for distinguishing flexible families. Five years provide thousands of minute-scale pairs but only roughly 1,825 daily pairs and roughly 365 per annual epoch.
- The 4h and especially 1d annual-epoch JS distances have a substantial sparse-histogram noise floor; use their parameter ranges as uncertainty indicators, not as proof of regime changes. The 1s generalized-Gaussian proxy also reaches the fit's lower power bound and should not be interpreted literally.
- Close prices omit intrabar extremes, spreads, fees, slippage, and liquidation paths. One-second last-trade closes also have tick-size and no-trade artifacts.

## Reproduction

Run \`npm run analysis:return-pairs\`, then \`npm run analysis:return-pairs:render\` for the six-scale heatmap. Machine-readable results are in \`${relative(outputPath)}\`.
`;
}

function printSummary(report: AnalysisReport): void {
  console.log("\nFull-history consecutive-return pairs");
  for (const scale of report.scales) {
    const full = scale.fullHistory;
    const selected = scale.selectedFullHistoryModel;
    console.log(
      `${scale.id.padEnd(3)} n=${integer(full.observations).padStart(11)} `
      + `corr=${fixed(full.returnCorrelation, 4).padStart(7)} `
      + `|r|corr=${fixed(full.absoluteReturnCorrelation, 3).padStart(6)} `
      + `GGD-p=${fixed(full.generalizedGaussianApproximation.power, 3).padStart(6)} `
      + `best=${selected.family}`,
    );
  }
}

function parseOptions(args: string[]): Options {
  if (args.includes("--help")) {
    console.log(`Usage: tsx scripts/analyze-consecutive-return-pairs.ts [options]

Options:
  --data-dir PATH          Trading data root (default: data)
  --output PATH            JSON result
  --report PATH            Markdown report
  --end YYYY-MM-DD         Exclusive UTC end boundary
  --sample-target N        Target sampled pairs per window (default: 250000)
  --fit-sample-limit N     Maximum continuous pairs per likelihood fit (default: 30000)`);
    process.exit(0);
  }
  const values = new Map<string, string>();
  for (let index = 0; index < args.length; index += 1) {
    const key = args[index];
    const value = args[index + 1];
    if (!key?.startsWith("--") || !value || value.startsWith("--")) {
      throw new Error(`Invalid argument near ${key ?? "end"}.`);
    }
    values.set(key.slice(2), value);
    index += 1;
  }
  const requestedEnd = values.get("end");
  return {
    dataDir: path.resolve(repoRoot, values.get("data-dir") ?? "data"),
    outputPath: path.resolve(
      repoRoot,
      values.get("output") ?? "data/benchmarks/consecutive-log-return-pairs.json",
    ),
    reportPath: path.resolve(
      repoRoot,
      values.get("report")
        ?? `docs/experiments/consecutive-log-return-pairs-${dateOnly(Date.now())}.md`,
    ),
    ...(requestedEnd === undefined ? {} : { requestedEndTime: parseUtcDay(requestedEnd) }),
    sampleTarget: positiveInteger(values.get("sample-target") ?? "250000", "sample-target"),
    fitSampleLimit: positiveInteger(
      values.get("fit-sample-limit") ?? "30000",
      "fit-sample-limit",
    ),
  };
}

function hashUnit(value: number, salt: number): number {
  let hash = (value ^ salt) >>> 0;
  hash = Math.imul(hash ^ (hash >>> 16), 0x45d9f3b);
  hash = Math.imul(hash ^ (hash >>> 16), 0x45d9f3b);
  hash = (hash ^ (hash >>> 16)) >>> 0;
  return hash / 0x1_0000_0000;
}

function subtractUtcYears(time: number, years: number): number {
  const date = new Date(time);
  return Date.UTC(date.getUTCFullYear() - years, date.getUTCMonth(), date.getUTCDate());
}

function finiteRange(values: readonly (number | null)[]): [number, number] | null {
  const finite = values.filter((value): value is number => value !== null && Number.isFinite(value));
  return finite.length === 0 ? null : [Math.min(...finite), Math.max(...finite)];
}

function quantile(sorted: readonly number[], probability: number): number {
  const position = probability * (sorted.length - 1);
  const lower = Math.floor(position);
  const weight = position - lower;
  return sorted[lower]! * (1 - weight) + sorted[Math.min(lower + 1, sorted.length - 1)]! * weight;
}

function parseUtcDay(value: string): number {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) throw new Error(`Invalid UTC date: ${value}`);
  const result = Date.parse(`${value}T00:00:00.000Z`);
  if (!Number.isFinite(result) || dateOnly(result) !== value) throw new Error(`Invalid UTC date: ${value}`);
  return result;
}

function positiveInteger(value: string, label: string): number {
  const result = Number(value);
  if (!Number.isSafeInteger(result) || result < 1) throw new Error(`${label} must be positive.`);
  return result;
}

function familyLabel(family: string): string {
  return ({
    "bivariate-gaussian": "Gaussian",
    "bivariate-student-t": "Student t",
    "elliptical-generalized-gaussian": "Elliptical generalized Gaussian",
    "elliptical-radial-lognormal": "Elliptical radial lognormal",
    "product-generalized-gaussian": "Product generalized Gaussian",
    "elliptical-generalized-t": "Elliptical generalized t",
    "product-generalized-t": "Product generalized t",
  } as Record<string, string>)[family] ?? family;
}

function rangeText(range: [number, number] | null, digits: number): string {
  return range === null ? "n/a" : `${range[0].toFixed(digits)}–${range[1].toFixed(digits)}`;
}

function markdownTable(headers: string[], rows: string[][]): string {
  return [
    `| ${headers.join(" | ")} |`,
    `| ${headers.map(() => "---").join(" | ")} |`,
    ...rows.map((row) => `| ${row.join(" | ")} |`),
  ].join("\n");
}

function fixed(value: number | null, digits: number): string {
  return value === null || !Number.isFinite(value) ? "n/a" : value.toFixed(digits);
}

function deltaAicText(value: number): string {
  if (value < 10) return value.toFixed(2);
  if (value < 1_000) return value.toFixed(1);
  return value.toFixed(0);
}

function integer(value: number): string {
  return value.toLocaleString("en-US", { maximumFractionDigits: 0 });
}

function iso(time: number): string {
  return new Date(time).toISOString();
}

function dateOnly(time: number): string {
  return iso(time).slice(0, 10);
}

function relative(file: string): string {
  return path.relative(repoRoot, file).replaceAll("\\", "/");
}

function numberReplacer(_key: string, value: unknown): unknown {
  if (typeof value !== "number" || !Number.isFinite(value) || Number.isInteger(value)) return value;
  return Number(value.toPrecision(10));
}
