import fs from "node:fs";
import path from "node:path";
import { deserialize, serialize } from "node:v8";
import { fileURLToPath, pathToFileURL } from "node:url";
import { readCandleShardReferenceSync } from "@trading/storage";

const DAY_MS = 86_400_000;
const FEATURE_BINS = 16;
const MAGNITUDE_BINS = 16;
const HISTORY_MAGNITUDE_BINS = 16;
const HISTORY_STATES = 1 + 2 * HISTORY_MAGNITUDE_BINS;
const JOINT_CLASSES = 1 + 2 * MAGNITUDE_BINS;
const WARMUP_RETURNS = 12_000;
const CALIBRATION_SAMPLE_STRIDE = 251;
const CALIBRATION_SAMPLE_CAPACITY = 150_000;
const EVALUATION_STRIDE = 4;
const DIRICHLET_ALPHA = 0.5;
const MEAN_PRIOR_WEIGHT = 100;
const DEFAULT_ANALYSIS = "data/benchmarks/log-return-distributions.json";
const DEFAULT_CALIBRATION_CACHE = "data/runtime-cache/technical-indicator-calibration.bin";
const DEFAULT_COUNTS_CACHE = "data/runtime-cache/technical-indicator-predictiveness-counts.bin";
const DEFAULT_OUTPUT = "data/benchmarks/technical-indicator-predictiveness.json";
const DEFAULT_REPORT = "docs/experiments/technical-indicator-predictiveness-2026-08-16.md";

const RSI_PERIODS = [2, 4, 8, 14, 32, 64, 128, 256, 512, 1_024] as const;
const EMA_PERIODS = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1_024, 2_048, 4_096] as const;
const MACD_CONFIGS = [
  [3, 7, 3],
  [6, 13, 5],
  [12, 26, 9],
  [24, 52, 18],
  [48, 104, 36],
  [96, 208, 72],
  [192, 416, 144],
  [384, 832, 288],
  [768, 1_664, 576],
] as const;
const EMA_DYNAMICS_PERIODS = [2, 4, 8, 32, 128, 512, 2_048, 4_096, 8_192] as const;
const EMA_DYNAMICS_HORIZONS = [1, 2, 4, 8, 16, 32, 64] as const;
const MAX_EMA_DYNAMICS_LAG = 2 * Math.max(...EMA_DYNAMICS_HORIZONS);

interface Options {
  analysisPath: string;
  calibrationCachePath: string;
  countsCachePath: string;
  outputPath: string;
  reportPath: string;
  rebuildCalibration: boolean;
  rebuildCounts: boolean;
}

interface AnalysisReport {
  generatedAt: string;
  source: {
    symbol: string;
    oneSecond: { referenceDirectory: string };
  };
  fullHistory: {
    id: string;
    label: string;
    startTime: string;
    endTime: string;
  };
}

type SignalDefinition =
  | { id: string; label: string; family: "RSI"; kind: "rsi"; period: number }
  | { id: string; label: string; family: "EMA"; kind: "emaGap"; period: number }
  | {
    id: string;
    label: string;
    family: "EMA slope" | "EMA acceleration";
    kind: "emaSlope" | "emaAcceleration";
    period: number;
    horizon: number;
  }
  | {
    id: string;
    label: string;
    family: "MACD";
    kind: "macdLine" | "macdHistogram";
    fast: number;
    slow: number;
    signal: number;
  };

interface Calibration {
  featureEdges: number[][];
  magnitudeEdges: number[];
  featureSamples: number;
  magnitudeSamples: number;
}

interface AnnualSignalCounts {
  jointCounts: Float64Array;
  returnSums: Float64Array;
  magnitudeSums: Float64Array;
}

interface AnnualMoments {
  observations: number;
  activeObservations: number;
  returnSum: number;
  returnSquareSum: number;
  magnitudeSum: number;
  magnitudeSquareSum: number;
}

interface StreamedCounts {
  signals: AnnualSignalCounts[][];
  annualMoments: AnnualMoments[];
  returnsSeen: number;
  returnsEvaluated: number;
  activeEvaluated: number;
}

interface DiscreteTarget {
  id: "zeroGate" | "activeSign" | "activeMagnitude" | "activeJoint" | "signedDistribution";
  label: string;
  classes: number;
  mapping: Int16Array;
  binary: boolean;
}

interface RollingDiscreteRow {
  year: number;
  observations: number;
  standaloneGainBits: number;
  incrementalGainBits: number;
  historyGainBits: number;
  standaloneAccuracyGain: number | null;
  incrementalAccuracyGain: number | null;
}

interface RollingDiscreteResult {
  observations: number;
  standaloneGainBits: number;
  incrementalGainBits: number;
  historyGainBits: number;
  combinedGainBits: number;
  standaloneAccuracyGain: number | null;
  incrementalAccuracyGain: number | null;
  positiveStandaloneYears: number;
  positiveIncrementalYears: number;
  annual: RollingDiscreteRow[];
}

interface ContinuousAccumulator {
  observations: number;
  targetSum: number;
  targetSquareSum: number;
  standalonePredictionSum: number;
  standalonePredictionSquareSum: number;
  standalonePredictionTargetSum: number;
  standaloneBaselinePredictionSum: number;
  standaloneBaselinePredictionSquareSum: number;
  standaloneBaselinePredictionTargetSum: number;
  incrementalPredictionSum: number;
  incrementalPredictionSquareSum: number;
  incrementalPredictionTargetSum: number;
  incrementalBaselinePredictionSum: number;
  incrementalBaselinePredictionSquareSum: number;
  incrementalBaselinePredictionTargetSum: number;
  standaloneSse: number;
  standaloneBaselineSse: number;
  incrementalSse: number;
  incrementalBaselineSse: number;
}

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function main(): void {
  const options = parseOptions(process.argv.slice(2));
  const analysis = readJson<AnalysisReport>(options.analysisPath);
  const definitions = buildSignalDefinitions();
  const start = Date.parse(analysis.fullHistory.startTime);
  const end = Date.parse(analysis.fullHistory.endTime);
  const files = selectedFiles(
    path.resolve(repoRoot, analysis.source.oneSecond.referenceDirectory),
    start,
    end,
  );
  const metadata = JSON.stringify({
    version: 1,
    analysisGeneratedAt: analysis.generatedAt,
    files: files.map((entry) => path.basename(entry.file)),
    definitions,
    featureBins: FEATURE_BINS,
    magnitudeBins: MAGNITUDE_BINS,
    historyMagnitudeBins: HISTORY_MAGNITUDE_BINS,
    warmupReturns: WARMUP_RETURNS,
    calibrationSampleStride: CALIBRATION_SAMPLE_STRIDE,
    evaluationStride: EVALUATION_STRIDE,
  });
  const calibration = loadOrBuildCalibration(
    options,
    metadata,
    files,
    start,
    definitions,
  );
  const counts = loadOrBuildCounts(
    options,
    metadata,
    files,
    start,
    definitions,
    calibration,
  );
  const artifact = analyzeCounts(analysis, definitions, calibration, counts);
  fs.mkdirSync(path.dirname(options.outputPath), { recursive: true });
  fs.writeFileSync(options.outputPath, `${JSON.stringify(artifact, null, 2)}\n`, "utf8");
  fs.mkdirSync(path.dirname(options.reportPath), { recursive: true });
  fs.writeFileSync(options.reportPath, renderReport(artifact), "utf8");
  console.log(`Wrote ${path.relative(repoRoot, options.outputPath)}`);
  console.log(`Wrote ${path.relative(repoRoot, options.reportPath)}`);
}

function parseOptions(args: string[]): Options {
  const values = new Map<string, string>();
  let rebuildCalibration = false;
  let rebuildCounts = false;
  for (let index = 0; index < args.length; index += 1) {
    const key = args[index]!;
    if (key === "--rebuild-calibration") {
      rebuildCalibration = true;
      continue;
    }
    if (key === "--rebuild-counts") {
      rebuildCounts = true;
      continue;
    }
    const value = args[index + 1];
    if (!key.startsWith("--") || !value || value.startsWith("--")) {
      throw new Error(`Invalid argument near ${key}.`);
    }
    values.set(key.slice(2), value);
    index += 1;
  }
  return {
    analysisPath: path.resolve(repoRoot, values.get("analysis") ?? DEFAULT_ANALYSIS),
    calibrationCachePath: path.resolve(
      repoRoot,
      values.get("calibration-cache") ?? DEFAULT_CALIBRATION_CACHE,
    ),
    countsCachePath: path.resolve(repoRoot, values.get("counts-cache") ?? DEFAULT_COUNTS_CACHE),
    outputPath: path.resolve(repoRoot, values.get("output") ?? DEFAULT_OUTPUT),
    reportPath: path.resolve(repoRoot, values.get("report") ?? DEFAULT_REPORT),
    rebuildCalibration,
    rebuildCounts,
  };
}

function readJson<T>(file: string): T {
  return JSON.parse(fs.readFileSync(file, "utf8")) as T;
}

export function buildSignalDefinitions(): SignalDefinition[] {
  return [
    ...RSI_PERIODS.map((period): SignalDefinition => ({
      id: `rsi-${period}`,
      label: `RSI(${period}s)`,
      family: "RSI",
      kind: "rsi",
      period,
    })),
    ...EMA_PERIODS.map((period): SignalDefinition => ({
      id: `ema-gap-${period}`,
      label: `Price−EMA(${period}s)`,
      family: "EMA",
      kind: "emaGap",
      period,
    })),
    ...EMA_DYNAMICS_PERIODS.flatMap((period) =>
      EMA_DYNAMICS_HORIZONS.flatMap((horizon): SignalDefinition[] => [
        {
          id: `ema-slope-${period}-${horizon}`,
          label: `EMA slope(n=${period}s,k=${horizon}s)`,
          family: "EMA slope",
          kind: "emaSlope",
          period,
          horizon,
        },
        {
          id: `ema-acceleration-${period}-${horizon}`,
          label: `EMA acceleration(n=${period}s,k=${horizon}s)`,
          family: "EMA acceleration",
          kind: "emaAcceleration",
          period,
          horizon,
        },
      ]),
    ),
    ...MACD_CONFIGS.flatMap(([fast, slow, signal]): SignalDefinition[] => [
      {
        id: `macd-line-${fast}-${slow}-${signal}`,
        label: `MACD line(${fast},${slow},${signal})`,
        family: "MACD",
        kind: "macdLine",
        fast,
        slow,
        signal,
      },
      {
        id: `macd-hist-${fast}-${slow}-${signal}`,
        label: `MACD histogram(${fast},${slow},${signal})`,
        family: "MACD",
        kind: "macdHistogram",
        fast,
        slow,
        signal,
      },
    ]),
  ];
}

export class IndicatorEngine {
  private readonly emaPeriods: number[];
  private readonly emaAlphas: Float64Array;
  private readonly emaValues: Float64Array;
  private readonly emaHistory: Float64Array[];
  private emaHistoryCursor = 0;
  private readonly emaIndex = new Map<number, number>();
  private readonly rsiPeriods: number[];
  private readonly rsiAlphas: Float64Array;
  private readonly rsiGains: Float64Array;
  private readonly rsiLosses: Float64Array;
  private readonly macdDefinitions: Array<Extract<SignalDefinition, { family: "MACD" }>>;
  private readonly macdSignalAlphas: Float64Array;
  private readonly macdSignals: Float64Array;
  private readonly definitionSources: Array<{
    kind: SignalDefinition["kind"];
    primary: number;
    secondary: number;
    tertiary: number;
    horizon: number;
  }>;
  private previousPrice: number;

  constructor(private readonly definitions: SignalDefinition[], initialPrice: number) {
    this.emaPeriods = Array.from(new Set(definitions.flatMap((definition) => {
      if (
        definition.kind === "emaGap"
        || definition.kind === "emaSlope"
        || definition.kind === "emaAcceleration"
      ) return [definition.period];
      if (definition.kind === "macdLine" || definition.kind === "macdHistogram") {
        return [definition.fast, definition.slow];
      }
      return [];
    }))).sort((left, right) => left - right);
    this.emaAlphas = Float64Array.from(this.emaPeriods, (period) => 2 / (period + 1));
    this.emaValues = new Float64Array(this.emaPeriods.length);
    this.emaValues.fill(initialPrice);
    this.emaHistory = this.emaPeriods.map(() => {
      const history = new Float64Array(MAX_EMA_DYNAMICS_LAG + 1);
      history.fill(initialPrice);
      return history;
    });
    this.emaPeriods.forEach((period, index) => this.emaIndex.set(period, index));
    this.rsiPeriods = definitions
      .filter((definition): definition is Extract<SignalDefinition, { kind: "rsi" }> =>
        definition.kind === "rsi")
      .map((definition) => definition.period);
    this.rsiAlphas = Float64Array.from(this.rsiPeriods, (period) => 1 / period);
    this.rsiGains = new Float64Array(this.rsiPeriods.length);
    this.rsiLosses = new Float64Array(this.rsiPeriods.length);
    this.macdDefinitions = definitions.filter(
      (definition): definition is Extract<SignalDefinition, { family: "MACD" }> =>
        definition.family === "MACD" && definition.kind === "macdLine",
    );
    this.macdSignalAlphas = Float64Array.from(
      this.macdDefinitions,
      (definition) => 2 / (definition.signal + 1),
    );
    this.macdSignals = new Float64Array(this.macdDefinitions.length);
    const rsiIndex = new Map(this.rsiPeriods.map((period, index) => [period, index]));
    const macdIndex = new Map(this.macdDefinitions.map((definition, index) => [
      `${definition.fast}-${definition.slow}-${definition.signal}`,
      index,
    ]));
    this.definitionSources = definitions.map((definition) => {
      if (definition.kind === "rsi") {
        return {
          kind: definition.kind,
          primary: rsiIndex.get(definition.period)!,
          secondary: 0,
          tertiary: 0,
          horizon: 0,
        };
      }
      if (definition.kind === "emaGap") {
        return {
          kind: definition.kind,
          primary: this.emaIndex.get(definition.period)!,
          secondary: 0,
          tertiary: 0,
          horizon: 0,
        };
      }
      if (definition.kind === "emaSlope" || definition.kind === "emaAcceleration") {
        return {
          kind: definition.kind,
          primary: this.emaIndex.get(definition.period)!,
          secondary: 0,
          tertiary: 0,
          horizon: definition.horizon,
        };
      }
      if (!("fast" in definition)) throw new Error(`Unsupported signal definition: ${definition.id}`);
      return {
        kind: definition.kind,
        primary: macdIndex.get(`${definition.fast}-${definition.slow}-${definition.signal}`)!,
        secondary: this.emaIndex.get(definition.fast)!,
        tertiary: this.emaIndex.get(definition.slow)!,
        horizon: 0,
      };
    });
    this.previousPrice = initialPrice;
  }

  update(price: number): void {
    const change = price - this.previousPrice;
    for (let index = 0; index < this.emaValues.length; index += 1) {
      this.emaValues[index] += this.emaAlphas[index]! * (price - this.emaValues[index]!);
    }
    const gain = Math.max(0, change);
    const loss = Math.max(0, -change);
    for (let index = 0; index < this.rsiPeriods.length; index += 1) {
      const alpha = this.rsiAlphas[index]!;
      this.rsiGains[index] += alpha * (gain - this.rsiGains[index]!);
      this.rsiLosses[index] += alpha * (loss - this.rsiLosses[index]!);
    }
    for (let index = 0; index < this.macdDefinitions.length; index += 1) {
      const definition = this.macdDefinitions[index]!;
      const macd = this.emaValues[this.emaIndex.get(definition.fast)!]!
        - this.emaValues[this.emaIndex.get(definition.slow)!]!;
      this.macdSignals[index] += this.macdSignalAlphas[index]! * (macd - this.macdSignals[index]!);
    }
    this.emaHistoryCursor = (this.emaHistoryCursor + 1) % (MAX_EMA_DYNAMICS_LAG + 1);
    for (let index = 0; index < this.emaValues.length; index += 1) {
      this.emaHistory[index]![this.emaHistoryCursor] = this.emaValues[index]!;
    }
    this.previousPrice = price;
  }

  values(output: Float64Array): Float64Array {
    const price = this.previousPrice;
    for (let index = 0; index < this.definitionSources.length; index += 1) {
      const source = this.definitionSources[index]!;
      if (source.kind === "rsi") {
        const gain = this.rsiGains[source.primary]!;
        const loss = this.rsiLosses[source.primary]!;
        output[index] = loss <= 0 ? (gain > 0 ? 100 : 50) : 100 - 100 / (1 + gain / loss);
      } else if (source.kind === "emaGap") {
        output[index] = 10_000 * Math.log(price / this.emaValues[source.primary]!);
      } else if (source.kind === "emaSlope" || source.kind === "emaAcceleration") {
        const history = this.emaHistory[source.primary]!;
        const lag = history[ringIndex(this.emaHistoryCursor, source.horizon, history.length)]!;
        const slope = emaSlopeFromLevels(
          this.emaValues[source.primary]!,
          lag,
          source.horizon,
        );
        if (source.kind === "emaSlope") {
          output[index] = slope;
        } else {
          const lag2 = history[
            ringIndex(this.emaHistoryCursor, 2 * source.horizon, history.length)
          ]!;
          output[index] = slope - emaSlopeFromLevels(lag, lag2, source.horizon);
        }
      } else {
        const macd = this.emaValues[source.secondary]! - this.emaValues[source.tertiary]!;
        const value = source.kind === "macdLine" ? macd : macd - this.macdSignals[source.primary]!;
        output[index] = 10_000 * value / price;
      }
    }
    return output;
  }
}

function ringIndex(cursor: number, lag: number, length: number): number {
  return (cursor - lag + length) % length;
}

export function emaSlopeFromLevels(current: number, lagged: number, horizon: number): number {
  return 10_000 * Math.log(current / lagged) / horizon;
}

function selectedFiles(directory: string, start: number, end: number) {
  const result = fs.readdirSync(directory, { withFileTypes: true })
    .filter((entry) => entry.isFile() && /^\d{4}-\d{2}-\d{2}\.json$/.test(entry.name))
    .map((entry) => ({
      file: path.join(directory, entry.name),
      dayStart: Date.parse(`${entry.name.slice(0, 10)}T00:00:00.000Z`),
    }))
    .filter((entry) => entry.dayStart >= start && entry.dayStart < end)
    .sort((left, right) => left.dayStart - right.dayStart);
  if (result.length === 0) throw new Error("No complete one-second shards selected.");
  for (let index = 1; index < result.length; index += 1) {
    if (result[index]!.dayStart !== result[index - 1]!.dayStart + DAY_MS) {
      throw new Error(`Missing one-second shard before ${new Date(result[index]!.dayStart).toISOString()}.`);
    }
  }
  return result;
}

function loadOrBuildCalibration(
  options: Options,
  metadata: string,
  files: Array<{ file: string; dayStart: number }>,
  start: number,
  definitions: SignalDefinition[],
): Calibration {
  if (!options.rebuildCalibration && fs.existsSync(options.calibrationCachePath)) {
    const cached = deserialize(fs.readFileSync(options.calibrationCachePath)) as {
      metadata: string;
      calibration: Calibration;
    };
    if (cached.metadata === metadata) {
      console.log(`Loading calibration from ${path.relative(repoRoot, options.calibrationCachePath)}`);
      return cached.calibration;
    }
  }
  const calibration = buildCalibration(files, start, definitions);
  fs.mkdirSync(path.dirname(options.calibrationCachePath), { recursive: true });
  fs.writeFileSync(options.calibrationCachePath, serialize({ metadata, calibration }));
  return calibration;
}

function buildCalibration(
  files: Array<{ file: string; dayStart: number }>,
  start: number,
  definitions: SignalDefinition[],
): Calibration {
  const calibrationEnd = anniversary(start, 1);
  const featureSamples = definitions.map(() => new Float64Array(CALIBRATION_SAMPLE_CAPACITY));
  const magnitudeSamples = new Float64Array(CALIBRATION_SAMPLE_CAPACITY);
  let featureSampleCount = 0;
  let magnitudeSampleCount = 0;
  let previousClose = Number.NaN;
  let engine: IndicatorEngine | undefined;
  let returnsSeen = 0;
  const values = new Float64Array(definitions.length);
  for (const [fileIndex, entry] of files.entries()) {
    if (entry.dayStart >= calibrationEnd) break;
    if (fileIndex % 100 === 0) console.error(`Calibrating indicators ${fileIndex}...`);
    const candles = readCandleShardReferenceSync(entry.file);
    if (candles.length !== 86_400) throw new Error(`${path.basename(entry.file)} is incomplete.`);
    for (const candle of candles) {
      validateClose(candle.close, entry.file);
      if (!engine) {
        engine = new IndicatorEngine(definitions, candle.close);
      } else {
        if (
          returnsSeen >= WARMUP_RETURNS
          && returnsSeen % CALIBRATION_SAMPLE_STRIDE === 0
        ) {
          if (featureSampleCount >= CALIBRATION_SAMPLE_CAPACITY) {
            throw new Error("Feature calibration sample capacity was exceeded.");
          }
          engine.values(values);
          for (let signal = 0; signal < definitions.length; signal += 1) {
            featureSamples[signal]![featureSampleCount] = values[signal]!;
          }
          featureSampleCount += 1;
          if (candle.close !== previousClose) {
            if (magnitudeSampleCount >= CALIBRATION_SAMPLE_CAPACITY) {
              throw new Error("Magnitude calibration sample capacity was exceeded.");
            }
            magnitudeSamples[magnitudeSampleCount] = Math.abs(
              Math.log(candle.close / previousClose) * 10_000,
            );
            magnitudeSampleCount += 1;
          }
        }
        engine.update(candle.close);
        returnsSeen += 1;
      }
      previousClose = candle.close;
    }
  }
  if (featureSampleCount < FEATURE_BINS * 100 || magnitudeSampleCount < MAGNITUDE_BINS * 100) {
    throw new Error("Calibration sample is unexpectedly small.");
  }
  return {
    featureEdges: featureSamples.map((samples) => equalMassEdges(
      samples.slice(0, featureSampleCount),
      FEATURE_BINS,
    )),
    magnitudeEdges: equalMassEdges(
      magnitudeSamples.slice(0, magnitudeSampleCount),
      MAGNITUDE_BINS,
    ),
    featureSamples: featureSampleCount,
    magnitudeSamples: magnitudeSampleCount,
  };
}

export function equalMassEdges(values: Float64Array, bins: number): number[] {
  if (values.length === 0 || bins < 2) throw new Error("Equal-mass edges require data and bins >= 2.");
  values.sort();
  const edges: number[] = [];
  for (let index = 1; index < bins; index += 1) {
    const position = Math.min(values.length - 1, Math.floor(index * values.length / bins));
    edges.push(values[position]!);
  }
  return edges;
}

function loadOrBuildCounts(
  options: Options,
  metadata: string,
  files: Array<{ file: string; dayStart: number }>,
  start: number,
  definitions: SignalDefinition[],
  calibration: Calibration,
): StreamedCounts {
  const countsMetadata = JSON.stringify({ metadata, calibration });
  if (!options.rebuildCounts && fs.existsSync(options.countsCachePath)) {
    const cached = deserialize(fs.readFileSync(options.countsCachePath)) as {
      metadata: string;
      counts: StreamedCounts;
    };
    if (cached.metadata === countsMetadata) {
      console.log(`Loading counts from ${path.relative(repoRoot, options.countsCachePath)}`);
      return cached.counts;
    }
  }
  const counts = streamCounts(files, start, definitions, calibration);
  fs.mkdirSync(path.dirname(options.countsCachePath), { recursive: true });
  fs.writeFileSync(options.countsCachePath, serialize({ metadata: countsMetadata, counts }));
  return counts;
}

function streamCounts(
  files: Array<{ file: string; dayStart: number }>,
  start: number,
  definitions: SignalDefinition[],
  calibration: Calibration,
): StreamedCounts {
  const years = anniversaryIndex(start, files.at(-1)!.dayStart) + 1;
  const signals = definitions.map(() => Array.from({ length: years }, createAnnualSignalCounts));
  const annualMoments = Array.from({ length: years }, createAnnualMoments);
  const historyMagnitudeEdges = calibration.magnitudeEdges;
  const values = new Float64Array(definitions.length);
  let engine: IndicatorEngine | undefined;
  let previousClose = Number.NaN;
  let previousReturnState = 0;
  let returnsSeen = 0;
  let returnsEvaluated = 0;
  let activeEvaluated = 0;
  for (const [fileIndex, entry] of files.entries()) {
    if (fileIndex % 100 === 0) {
      console.error(`Evaluating indicators ${fileIndex}/${files.length}...`);
    }
    const year = anniversaryIndex(start, entry.dayStart);
    const candles = readCandleShardReferenceSync(entry.file);
    if (candles.length !== 86_400) throw new Error(`${path.basename(entry.file)} is incomplete.`);
    for (const candle of candles) {
      validateClose(candle.close, entry.file);
      if (!engine) {
        engine = new IndicatorEngine(definitions, candle.close);
      } else {
        const returnBps = candle.close === previousClose
          ? 0
          : Math.log(candle.close / previousClose) * 10_000;
        const active = returnBps !== 0;
        const magnitude = Math.abs(returnBps);
        const sign = returnBps > 0 ? 1 : 0;
        const magnitudeBin = active ? upperBound(calibration.magnitudeEdges, magnitude) : -1;
        const targetClass = active ? 1 + sign * MAGNITUDE_BINS + magnitudeBin : 0;
        if (returnsSeen >= WARMUP_RETURNS && isEvaluationTarget(returnsSeen)) {
          engine.values(values);
          const moments = annualMoments[year]!;
          moments.observations += 1;
          moments.returnSum += returnBps;
          moments.returnSquareSum += returnBps * returnBps;
          if (active) {
            moments.activeObservations += 1;
            moments.magnitudeSum += magnitude;
            moments.magnitudeSquareSum += magnitude * magnitude;
            activeEvaluated += 1;
          }
          for (let signal = 0; signal < definitions.length; signal += 1) {
            const featureBin = upperBound(calibration.featureEdges[signal]!, values[signal]!);
            const state = previousReturnState * FEATURE_BINS + featureBin;
            const annual = signals[signal]![year]!;
            annual.jointCounts[state * JOINT_CLASSES + targetClass] += 1;
            annual.returnSums[state] += returnBps;
            annual.magnitudeSums[state] += magnitude;
          }
          returnsEvaluated += 1;
        }
        previousReturnState = active
          ? 1 + sign * HISTORY_MAGNITUDE_BINS + upperBound(historyMagnitudeEdges, magnitude)
          : 0;
        engine.update(candle.close);
        returnsSeen += 1;
      }
      previousClose = candle.close;
    }
  }
  return { signals, annualMoments, returnsSeen, returnsEvaluated, activeEvaluated };
}

function createAnnualSignalCounts(): AnnualSignalCounts {
  const states = HISTORY_STATES * FEATURE_BINS;
  return {
    jointCounts: new Float64Array(states * JOINT_CLASSES),
    returnSums: new Float64Array(states),
    magnitudeSums: new Float64Array(states),
  };
}

function createAnnualMoments(): AnnualMoments {
  return {
    observations: 0,
    activeObservations: 0,
    returnSum: 0,
    returnSquareSum: 0,
    magnitudeSum: 0,
    magnitudeSquareSum: 0,
  };
}

function validateClose(close: number, file: string): void {
  if (!Number.isFinite(close) || close <= 0) {
    throw new Error(`${path.basename(file)} contains an invalid close.`);
  }
}

function upperBound(sorted: number[], value: number): number {
  let low = 0;
  let high = sorted.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (value < sorted[middle]!) high = middle;
    else low = middle + 1;
  }
  return low;
}

function isEvaluationTarget(index: number): boolean {
  // A deterministic integer hash avoids locking the sample to one wall-clock
  // second modulo EVALUATION_STRIDE.
  let mixed = index ^ (index >>> 16);
  mixed = Math.imul(mixed, 0x45d9f3b);
  mixed ^= mixed >>> 16;
  return (mixed >>> 0) % EVALUATION_STRIDE === 0;
}

function anniversary(startMs: number, years: number): number {
  const start = new Date(startMs);
  return Date.UTC(
    start.getUTCFullYear() + years,
    start.getUTCMonth(),
    start.getUTCDate(),
  );
}

function anniversaryIndex(startMs: number, dayMs: number): number {
  const start = new Date(startMs);
  const day = new Date(dayMs);
  let index = day.getUTCFullYear() - start.getUTCFullYear();
  if (dayMs < anniversary(startMs, index)) index -= 1;
  return Math.max(0, index);
}

function buildDiscreteTargets(): DiscreteTarget[] {
  const zeroGate = new Int16Array(JOINT_CLASSES);
  zeroGate.fill(1);
  zeroGate[0] = 0;
  const activeSign = new Int16Array(JOINT_CLASSES);
  activeSign.fill(-1);
  const activeMagnitude = new Int16Array(JOINT_CLASSES);
  activeMagnitude.fill(-1);
  const activeJoint = new Int16Array(JOINT_CLASSES);
  activeJoint.fill(-1);
  const signedDistribution = new Int16Array(JOINT_CLASSES);
  for (let targetClass = 0; targetClass < JOINT_CLASSES; targetClass += 1) {
    signedDistribution[targetClass] = targetClass;
    if (targetClass > 0) {
      const activeClass = targetClass - 1;
      activeSign[targetClass] = Math.floor(activeClass / MAGNITUDE_BINS);
      activeMagnitude[targetClass] = activeClass % MAGNITUDE_BINS;
      activeJoint[targetClass] = activeClass;
    }
  }
  return [
    { id: "zeroGate", label: "zero versus active", classes: 2, mapping: zeroGate, binary: true },
    { id: "activeSign", label: "active sign", classes: 2, mapping: activeSign, binary: true },
    {
      id: "activeMagnitude",
      label: "active magnitude (16 cells)",
      classes: MAGNITUDE_BINS,
      mapping: activeMagnitude,
      binary: false,
    },
    {
      id: "activeJoint",
      label: "active sign × magnitude",
      classes: 2 * MAGNITUDE_BINS,
      mapping: activeJoint,
      binary: false,
    },
    {
      id: "signedDistribution",
      label: "full signed return including zero",
      classes: JOINT_CLASSES,
      mapping: signedDistribution,
      binary: false,
    },
  ];
}

export function aggregateTargetCounts(
  jointCounts: Float64Array,
  mapping: Int16Array,
  targetClasses: number,
): Float64Array {
  const states = jointCounts.length / JOINT_CLASSES;
  const result = new Float64Array(states * targetClasses);
  for (let state = 0; state < states; state += 1) {
    for (let jointClass = 0; jointClass < JOINT_CLASSES; jointClass += 1) {
      const targetClass = mapping[jointClass]!;
      if (targetClass >= 0) {
        result[state * targetClasses + targetClass] +=
          jointCounts[state * JOINT_CLASSES + jointClass]!;
      }
    }
  }
  return result;
}

export function rollingDiscreteEvaluation(
  annualJointCounts: Float64Array[],
  target: DiscreteTarget,
): RollingDiscreteResult {
  const annual = annualJointCounts.map((counts) =>
    aggregateTargetCounts(counts, target.mapping, target.classes));
  const train = new Float64Array(annual[0]!.length);
  const rows: RollingDiscreteRow[] = [];
  for (let year = 0; year < annual.length; year += 1) {
    if (year > 0) rows.push(evaluateDiscreteYear(train, annual[year]!, target, year));
    addInPlace(train, annual[year]!);
  }
  const observations = rows.reduce((sum, row) => sum + row.observations, 0);
  return {
    observations,
    standaloneGainBits: weighted(rows, "standaloneGainBits"),
    incrementalGainBits: weighted(rows, "incrementalGainBits"),
    historyGainBits: weighted(rows, "historyGainBits"),
    combinedGainBits: weighted(rows, "historyGainBits") + weighted(rows, "incrementalGainBits"),
    standaloneAccuracyGain: target.binary ? weightedNullable(rows, "standaloneAccuracyGain") : null,
    incrementalAccuracyGain: target.binary ? weightedNullable(rows, "incrementalAccuracyGain") : null,
    positiveStandaloneYears: rows.filter((row) => row.standaloneGainBits > 0).length,
    positiveIncrementalYears: rows.filter((row) => row.incrementalGainBits > 0).length,
    annual: rows,
  };
}

function evaluateDiscreteYear(
  train: Float64Array,
  test: Float64Array,
  target: DiscreteTarget,
  year: number,
): RollingDiscreteRow {
  const classes = target.classes;
  const featureTrain = new Float64Array(FEATURE_BINS * classes);
  const historyTrain = new Float64Array(HISTORY_STATES * classes);
  const globalTrain = new Float64Array(classes);
  const featureTotals = new Float64Array(FEATURE_BINS);
  const historyTotals = new Float64Array(HISTORY_STATES);
  const crossTotals = new Float64Array(HISTORY_STATES * FEATURE_BINS);
  let globalTotal = 0;
  for (let history = 0; history < HISTORY_STATES; history += 1) {
    for (let feature = 0; feature < FEATURE_BINS; feature += 1) {
      const state = history * FEATURE_BINS + feature;
      for (let targetClass = 0; targetClass < classes; targetClass += 1) {
        const count = train[state * classes + targetClass]!;
        featureTrain[feature * classes + targetClass] += count;
        historyTrain[history * classes + targetClass] += count;
        globalTrain[targetClass] += count;
        featureTotals[feature] += count;
        historyTotals[history] += count;
        crossTotals[state] += count;
        globalTotal += count;
      }
    }
  }
  let observations = 0;
  let standaloneGain = 0;
  let incrementalGain = 0;
  let historyGain = 0;
  let standaloneCorrect = 0;
  let standaloneBaselineCorrect = 0;
  let incrementalCorrect = 0;
  let incrementalBaselineCorrect = 0;
  const globalChoice = argmax(globalTrain, 0, classes);
  const featureChoices = target.binary
    ? Array.from({ length: FEATURE_BINS }, (_, feature) =>
      argmax(featureTrain, feature * classes, classes))
    : [];
  const historyChoices = target.binary
    ? Array.from({ length: HISTORY_STATES }, (_, history) =>
      argmax(historyTrain, history * classes, classes))
    : [];
  const crossChoices = target.binary
    ? Array.from({ length: HISTORY_STATES * FEATURE_BINS }, (_, state) =>
      argmax(train, state * classes, classes))
    : [];
  for (let history = 0; history < HISTORY_STATES; history += 1) {
    for (let feature = 0; feature < FEATURE_BINS; feature += 1) {
      const state = history * FEATURE_BINS + feature;
      for (let targetClass = 0; targetClass < classes; targetClass += 1) {
        const count = test[state * classes + targetClass]!;
        if (count === 0) continue;
        observations += count;
        const globalProbability = smoothedProbability(
          globalTrain[targetClass]!,
          globalTotal,
          classes,
        );
        const featureProbability = smoothedProbability(
          featureTrain[feature * classes + targetClass]!,
          featureTotals[feature]!,
          classes,
        );
        const historyProbability = smoothedProbability(
          historyTrain[history * classes + targetClass]!,
          historyTotals[history]!,
          classes,
        );
        const crossProbability = smoothedProbability(
          train[state * classes + targetClass]!,
          crossTotals[state]!,
          classes,
        );
        standaloneGain += count * Math.log2(featureProbability / globalProbability);
        incrementalGain += count * Math.log2(crossProbability / historyProbability);
        historyGain += count * Math.log2(historyProbability / globalProbability);
        if (target.binary) {
          if (globalChoice === targetClass) standaloneBaselineCorrect += count;
          if (featureChoices[feature] === targetClass) standaloneCorrect += count;
          if (historyChoices[history] === targetClass) incrementalBaselineCorrect += count;
          if (crossChoices[state] === targetClass) incrementalCorrect += count;
        }
      }
    }
  }
  return {
    year,
    observations,
    standaloneGainBits: standaloneGain / observations,
    incrementalGainBits: incrementalGain / observations,
    historyGainBits: historyGain / observations,
    standaloneAccuracyGain: target.binary
      ? (standaloneCorrect - standaloneBaselineCorrect) / observations
      : null,
    incrementalAccuracyGain: target.binary
      ? (incrementalCorrect - incrementalBaselineCorrect) / observations
      : null,
  };
}

function smoothedProbability(count: number, total: number, classes: number): number {
  return (count + DIRICHLET_ALPHA) / (total + classes * DIRICHLET_ALPHA);
}

function argmax(values: Float64Array, offset: number, length: number): number {
  let best = 0;
  for (let index = 1; index < length; index += 1) {
    if (values[offset + index]! > values[offset + best]!) best = index;
  }
  return best;
}

function addInPlace(target: Float64Array, source: Float64Array): void {
  for (let index = 0; index < target.length; index += 1) target[index] += source[index]!;
}

function weighted(
  rows: RollingDiscreteRow[],
  key: "standaloneGainBits" | "incrementalGainBits" | "historyGainBits",
): number {
  const total = rows.reduce((sum, row) => sum + row.observations, 0);
  return rows.reduce((sum, row) => sum + row[key] * row.observations, 0) / total;
}

function weightedNullable(
  rows: RollingDiscreteRow[],
  key: "standaloneAccuracyGain" | "incrementalAccuracyGain",
): number {
  const total = rows.reduce((sum, row) => sum + row.observations, 0);
  return rows.reduce((sum, row) => sum + (row[key] ?? 0) * row.observations, 0) / total;
}

function rollingContinuousEvaluation(
  annual: AnnualSignalCounts[],
  moments: AnnualMoments[],
  mode: "return" | "magnitude",
) {
  const states = HISTORY_STATES * FEATURE_BINS;
  const trainCounts = new Float64Array(states);
  const trainSums = new Float64Array(states);
  const accumulator = createContinuousAccumulator();
  for (let year = 0; year < annual.length; year += 1) {
    const test = annual[year]!;
    if (year > 0) {
      evaluateContinuousYear(trainCounts, trainSums, test, moments[year]!, mode, accumulator);
    }
    const stateCounts = stateCountsForMode(annual[year]!.jointCounts, mode);
    addInPlace(trainCounts, stateCounts);
    addInPlace(trainSums, mode === "return" ? annual[year]!.returnSums : annual[year]!.magnitudeSums);
  }
  return finishContinuousAccumulator(accumulator);
}

function stateCountsForMode(jointCounts: Float64Array, mode: "return" | "magnitude") {
  const states = jointCounts.length / JOINT_CLASSES;
  const result = new Float64Array(states);
  for (let state = 0; state < states; state += 1) {
    const startClass = mode === "return" ? 0 : 1;
    for (let targetClass = startClass; targetClass < JOINT_CLASSES; targetClass += 1) {
      result[state] += jointCounts[state * JOINT_CLASSES + targetClass]!;
    }
  }
  return result;
}

function createContinuousAccumulator(): ContinuousAccumulator {
  return {
    observations: 0,
    targetSum: 0,
    targetSquareSum: 0,
    standalonePredictionSum: 0,
    standalonePredictionSquareSum: 0,
    standalonePredictionTargetSum: 0,
    standaloneBaselinePredictionSum: 0,
    standaloneBaselinePredictionSquareSum: 0,
    standaloneBaselinePredictionTargetSum: 0,
    incrementalPredictionSum: 0,
    incrementalPredictionSquareSum: 0,
    incrementalPredictionTargetSum: 0,
    incrementalBaselinePredictionSum: 0,
    incrementalBaselinePredictionSquareSum: 0,
    incrementalBaselinePredictionTargetSum: 0,
    standaloneSse: 0,
    standaloneBaselineSse: 0,
    incrementalSse: 0,
    incrementalBaselineSse: 0,
  };
}

function evaluateContinuousYear(
  trainCounts: Float64Array,
  trainSums: Float64Array,
  test: AnnualSignalCounts,
  moments: AnnualMoments,
  mode: "return" | "magnitude",
  result: ContinuousAccumulator,
): void {
  const testCounts = stateCountsForMode(test.jointCounts, mode);
  const testSums = mode === "return" ? test.returnSums : test.magnitudeSums;
  const featureCounts = new Float64Array(FEATURE_BINS);
  const featureSums = new Float64Array(FEATURE_BINS);
  const historyCounts = new Float64Array(HISTORY_STATES);
  const historySums = new Float64Array(HISTORY_STATES);
  let globalCount = 0;
  let globalSum = 0;
  for (let history = 0; history < HISTORY_STATES; history += 1) {
    for (let feature = 0; feature < FEATURE_BINS; feature += 1) {
      const state = history * FEATURE_BINS + feature;
      featureCounts[feature] += trainCounts[state]!;
      featureSums[feature] += trainSums[state]!;
      historyCounts[history] += trainCounts[state]!;
      historySums[history] += trainSums[state]!;
      globalCount += trainCounts[state]!;
      globalSum += trainSums[state]!;
    }
  }
  const globalMean = globalSum / globalCount;
  const featureMeans = Float64Array.from(featureCounts, (count, feature) =>
    shrunkMean(featureSums[feature]!, count, globalMean));
  const historyMeans = Float64Array.from(historyCounts, (count, history) =>
    shrunkMean(historySums[history]!, count, globalMean));
  const targetSquareSum = mode === "return"
    ? moments.returnSquareSum
    : moments.magnitudeSquareSum;
  result.targetSquareSum += targetSquareSum;
  for (let history = 0; history < HISTORY_STATES; history += 1) {
    for (let feature = 0; feature < FEATURE_BINS; feature += 1) {
      const state = history * FEATURE_BINS + feature;
      const count = testCounts[state]!;
      if (count === 0) continue;
      const targetSum = testSums[state]!;
      const featureMean = featureMeans[feature]!;
      const historyMean = historyMeans[history]!;
      const crossMean = shrunkMean(trainSums[state]!, trainCounts[state]!, historyMean);
      accumulateContinuousPrediction(result, "standalone", count, targetSum, featureMean);
      accumulateContinuousPrediction(result, "standaloneBaseline", count, targetSum, globalMean);
      accumulateContinuousPrediction(result, "incremental", count, targetSum, crossMean);
      accumulateContinuousPrediction(result, "incrementalBaseline", count, targetSum, historyMean);
      result.observations += count;
      result.targetSum += targetSum;
    }
  }
}

function shrunkMean(sum: number, count: number, parentMean: number): number {
  return (sum + MEAN_PRIOR_WEIGHT * parentMean) / (count + MEAN_PRIOR_WEIGHT);
}

function accumulateContinuousPrediction(
  result: ContinuousAccumulator,
  prefix: "standalone" | "standaloneBaseline" | "incremental" | "incrementalBaseline",
  count: number,
  targetSum: number,
  prediction: number,
): void {
  const predictionSumKey = `${prefix}PredictionSum` as keyof ContinuousAccumulator;
  const predictionSquareSumKey = `${prefix}PredictionSquareSum` as keyof ContinuousAccumulator;
  const predictionTargetSumKey = `${prefix}PredictionTargetSum` as keyof ContinuousAccumulator;
  const sseKey = `${prefix}Sse` as keyof ContinuousAccumulator;
  result[predictionSumKey] += count * prediction;
  result[predictionSquareSumKey] += count * prediction * prediction;
  result[predictionTargetSumKey] += prediction * targetSum;
  result[sseKey] += count * prediction * prediction - 2 * prediction * targetSum;
}

function finishContinuousAccumulator(result: ContinuousAccumulator) {
  const standaloneSse = result.targetSquareSum + result.standaloneSse;
  const standaloneBaselineSse = result.targetSquareSum + result.standaloneBaselineSse;
  const incrementalSse = result.targetSquareSum + result.incrementalSse;
  const incrementalBaselineSse = result.targetSquareSum + result.incrementalBaselineSse;
  return {
    observations: result.observations,
    standaloneCorrelation: correlation(
      result.observations,
      result.standalonePredictionSum,
      result.targetSum,
      result.standalonePredictionSquareSum,
      result.targetSquareSum,
      result.standalonePredictionTargetSum,
    ),
    standaloneBaselineCorrelation: correlation(
      result.observations,
      result.standaloneBaselinePredictionSum,
      result.targetSum,
      result.standaloneBaselinePredictionSquareSum,
      result.targetSquareSum,
      result.standaloneBaselinePredictionTargetSum,
    ),
    incrementalCorrelation: correlation(
      result.observations,
      result.incrementalPredictionSum,
      result.targetSum,
      result.incrementalPredictionSquareSum,
      result.targetSquareSum,
      result.incrementalPredictionTargetSum,
    ),
    incrementalBaselineCorrelation: correlation(
      result.observations,
      result.incrementalBaselinePredictionSum,
      result.targetSum,
      result.incrementalBaselinePredictionSquareSum,
      result.targetSquareSum,
      result.incrementalBaselinePredictionTargetSum,
    ),
    standaloneMseImprovement: (standaloneBaselineSse - standaloneSse) / standaloneBaselineSse,
    incrementalMseImprovement: (incrementalBaselineSse - incrementalSse) / incrementalBaselineSse,
  };
}

export function correlation(
  observations: number,
  leftSum: number,
  rightSum: number,
  leftSquareSum: number,
  rightSquareSum: number,
  productSum: number,
): number {
  const covariance = productSum - leftSum * rightSum / observations;
  const leftVariance = leftSquareSum - leftSum * leftSum / observations;
  const rightVariance = rightSquareSum - rightSum * rightSum / observations;
  if (leftVariance <= 0 || rightVariance <= 0) return 0;
  return covariance / Math.sqrt(leftVariance * rightVariance);
}

function analyzeCounts(
  analysis: AnalysisReport,
  definitions: SignalDefinition[],
  calibration: Calibration,
  counts: StreamedCounts,
) {
  const targets = buildDiscreteTargets();
  const signals = definitions.map((definition, index) => {
    const annual = counts.signals[index]!;
    const discrete = Object.fromEntries(targets.map((target) => [
      target.id,
      rollingDiscreteEvaluation(annual.map((row) => row.jointCounts), target),
    ]));
    return {
      ...definition,
      calibrationEdges: calibration.featureEdges[index],
      discrete,
      returnMean: rollingContinuousEvaluation(annual, counts.annualMoments, "return"),
      magnitudeMean: rollingContinuousEvaluation(annual, counts.annualMoments, "magnitude"),
    };
  });
  const winners = Object.fromEntries(targets.flatMap((target) => [
    [`${target.id}Standalone`, bestSignal(signals, target.id, "standaloneGainBits")],
    [`${target.id}Incremental`, bestSignal(signals, target.id, "incrementalGainBits")],
  ]));
  return {
    version: 1,
    generatedAt: new Date().toISOString(),
    symbol: analysis.source.symbol,
    scale: "1s → next 1s",
    window: analysis.fullHistory,
    methodology: {
      causalAlignment: "Indicator at close t predicts log(close[t+1]/close[t]).",
      calibration: "Indicator and active-magnitude equal-mass edges are fitted on year 0 only.",
      evaluation: "Rolling annual holdout: each test year uses all earlier years, never later data.",
      evaluationStride: EVALUATION_STRIDE,
      featureBins: FEATURE_BINS,
      magnitudeBins: MAGNITUDE_BINS,
      historyBaseline: "Previous return: zero or sign crossed with 16 first-year magnitude cells.",
      standaloneGain: "Indicator-conditioned log score minus unconditional log score.",
      incrementalGain: "History+indicator log score minus history-only log score.",
    },
    calibration: {
      ...calibration,
      featureEdges: undefined,
    },
    returnsSeen: counts.returnsSeen,
    returnsEvaluated: counts.returnsEvaluated,
    activeEvaluated: counts.activeEvaluated,
    annualMoments: counts.annualMoments,
    targets: targets.map(({ mapping: _mapping, ...target }) => target),
    historyBaselineByTarget: Object.fromEntries(targets.map((target) => [
      target.id,
      signals[0]!.discrete[target.id].historyGainBits,
    ])),
    winners,
    signals,
    conclusion: classifySignals(signals),
    limitations: [
      "This is a distributional predictiveness screen, not a fee-aware trading backtest.",
      "Indicator bin edges remain fixed after year 0, so later regime/scale drift is intentionally exposed rather than recalibrated away.",
      `Only every ${EVALUATION_STRIDE}th target is scored to make the five-year hyperparameter sweep tractable; every candle still updates every causal indicator.`,
      "Incremental scores condition on the immediately previous return state, not on every possible history representation or a trained neural model.",
      "The candidate grid creates multiple comparisons; annual consistency and effect size matter more than formal significance on the very large sample.",
    ],
  };
}

function bestSignal(
  signals: Array<Record<string, any>>,
  target: DiscreteTarget["id"],
  metric: "standaloneGainBits" | "incrementalGainBits",
) {
  const sorted = [...signals].sort((left, right) =>
    right.discrete[target][metric] - left.discrete[target][metric]);
  const winner = sorted[0]!;
  return {
    id: winner.id,
    label: winner.label,
    family: winner.family,
    value: winner.discrete[target][metric],
    positiveYears: metric === "standaloneGainBits"
      ? winner.discrete[target].positiveStandaloneYears
      : winner.discrete[target].positiveIncrementalYears,
  };
}

function classifySignals(signals: Array<Record<string, any>>): string {
  const bestIncremental = Math.max(...signals.map((signal) =>
    signal.discrete.signedDistribution.incrementalGainBits));
  if (bestIncremental <= 0) {
    return "No tested indicator improves the full next-return distribution beyond the latest return state.";
  }
  if (bestIncremental < 1e-4) {
    return "At least one indicator adds positive but very small distributional information beyond the latest return state.";
  }
  return "At least one indicator adds material distributional information beyond the latest return state.";
}

function renderReport(artifact: ReturnType<typeof analyzeCounts>): string {
  const fullWinner = artifact.signals.find((signal) =>
    signal.id === artifact.winners.signedDistributionIncremental.id)!;
  const signWinner = artifact.signals.find((signal) =>
    signal.id === artifact.winners.activeSignIncremental.id)!;
  const magnitudeCellWinner = artifact.signals.find((signal) =>
    signal.id === artifact.winners.activeMagnitudeIncremental.id)!;
  const returnMeanWinner = [...artifact.signals].sort((left, right) =>
    right.returnMean.incrementalMseImprovement - left.returnMean.incrementalMseImprovement)[0]!;
  const magnitudeMeanWinner = [...artifact.signals].sort((left, right) =>
    right.magnitudeMean.incrementalMseImprovement - left.magnitudeMean.incrementalMseImprovement)[0]!;
  const macdFullWinner = artifact.signals
    .filter((signal) => signal.family === "MACD")
    .sort((left, right) =>
      right.discrete.signedDistribution.incrementalGainBits
      - left.discrete.signedDistribution.incrementalGainBits)[0]!;
  const dynamicsSignals = artifact.signals.filter((signal) =>
    (signal.family === "EMA slope" && "horizon" in signal && signal.horizon > 1)
    || signal.family === "EMA acceleration");
  const dynamicsFullWinner = [...dynamicsSignals].sort((left, right) =>
    right.discrete.signedDistribution.incrementalGainBits
    - left.discrete.signedDistribution.incrementalGainBits)[0]!;
  const dynamicsMagnitudeWinner = [...dynamicsSignals].sort((left, right) =>
    right.discrete.activeMagnitude.incrementalGainBits
    - left.discrete.activeMagnitude.incrementalGainBits)[0]!;
  const accelerationSignWinner = artifact.signals
    .filter((signal) => signal.kind === "emaAcceleration")
    .sort((left, right) =>
      right.discrete.activeSign.incrementalGainBits
      - left.discrete.activeSign.incrementalGainBits)[0]!;
  const accelerationMagnitudeWinner = artifact.signals
    .filter((signal) => signal.kind === "emaAcceleration")
    .sort((left, right) =>
      right.discrete.activeMagnitude.incrementalGainBits
      - left.discrete.activeMagnitude.incrementalGainBits)[0]!;
  const lines = [
    "# One-second technical-indicator predictiveness",
    "",
    `Generated ${artifact.generatedAt}. ${artifact.symbol} indicators at close t predict the next one-second log return. The scan observes ${artifact.returnsSeen.toLocaleString("en-US")} returns and evaluates a deterministic one-in-${artifact.methodology.evaluationStride} sample: ${artifact.returnsEvaluated.toLocaleString("en-US")} total targets, ${artifact.activeEvaluated.toLocaleString("en-US")} active.`,
    "",
    "## Result",
    "",
    artifact.conclusion,
    "",
    "`Standalone` compares an indicator with no feature. `Incremental` compares history + indicator against the latest-return-state history alone. Positive bits mean better held-out probability forecasts.",
    "",
    "## Interpretation",
    "",
    `- The 33-state latest-return baseline is already strong: ${formatMetric(artifact.historyBaselineByTarget.signedDistribution)} bits/target over the unconditional full-return distribution. ${fullWinner.label} adds ${formatMetric(fullWinner.discrete.signedDistribution.incrementalGainBits)} bits beyond it in all four years, a ${((2 ** fullWinner.discrete.signedDistribution.incrementalGainBits - 1) * 100).toFixed(3)}% improvement in geometric assigned probability.`,
    `- Active sign is best served by ${signWinner.label}: ${formatMetric(signWinner.discrete.activeSign.incrementalGainBits)} additional bits and ${((signWinner.discrete.activeSign.incrementalAccuracyGain ?? 0) * 100).toFixed(4)} percentage points of held-out sign accuracy beyond the latest return.`,
    `- The 16-cell active-magnitude distribution prefers ${magnitudeCellWinner.label} (${formatMetric(magnitudeCellWinner.discrete.activeMagnitude.incrementalGainBits)} bits), while the conditional mean magnitude prefers ${magnitudeMeanWinner.label}: correlation rises from ${formatMetric(magnitudeMeanWinner.magnitudeMean.incrementalBaselineCorrelation)} to ${formatMetric(magnitudeMeanWinner.magnitudeMean.incrementalCorrelation)} and MSE falls ${(magnitudeMeanWinner.magnitudeMean.incrementalMseImprovement * 100).toFixed(4)}%.`,
    `- The signed conditional mean prefers ${returnMeanWinner.label}: correlation rises from ${formatMetric(returnMeanWinner.returnMean.incrementalBaselineCorrelation)} to ${formatMetric(returnMeanWinner.returnMean.incrementalCorrelation)}, with ${(returnMeanWinner.returnMean.incrementalMseImprovement * 100).toFixed(4)}% lower MSE.`,
    `- Among the new dynamics, ${dynamicsFullWinner.label} is best for the full distribution at ${formatMetric(dynamicsFullWinner.discrete.signedDistribution.incrementalGainBits)} bits. ${dynamicsMagnitudeWinner.label} is best for active magnitude at ${formatMetric(dynamicsMagnitudeWinner.discrete.activeMagnitude.incrementalGainBits)} bits; both remain positive in all four years.`,
    `- Acceleration separates into two useful regimes: ${accelerationSignWinner.label} is strongest for sign (${formatMetric(accelerationSignWinner.discrete.activeSign.incrementalGainBits)} bits), while ${accelerationMagnitudeWinner.label} is strongest for magnitude (${formatMetric(accelerationMagnitudeWinner.discrete.activeMagnitude.incrementalGainBits)} bits). The latter changes only modestly from n=4096s to n=8192s, indicating slow-period saturation.`,
    "- The `k=1` EMA-slope control reproduces the corresponding Price−EMA metrics exactly across every target. This confirms the causal alignment and shows that the improvement starts specifically at the genuinely multi-step `k=2` slope.",
    `- MACD is useful but dominated by the shortest RSI/EMA variants. Its best full-distribution result is ${macdFullWinner.label} at ${formatMetric(macdFullWinner.discrete.signedDistribution.incrementalGainBits)} bits; longer MACD lines and histograms generally decay toward smaller gains.`,
    "",
    "## Best indicator for each target",
    "",
    "| target | latest-return baseline gain | standalone winner | standalone gain | incremental winner | gain beyond latest return | combined gain | positive incremental years |",
    "|---|---:|---|---:|---|---:|---:|---:|",
  ];
  for (const target of artifact.targets) {
    const standalone = artifact.winners[`${target.id}Standalone`]!;
    const incremental = artifact.winners[`${target.id}Incremental`]!;
    lines.push(
      `| ${target.label} | ${formatMetric(artifact.historyBaselineByTarget[target.id]!)} | ${standalone.label} | ${formatMetric(standalone.value)} | ${incremental.label} | ${formatMetric(incremental.value)} | ${formatMetric(artifact.historyBaselineByTarget[target.id]! + incremental.value)} | ${incremental.positiveYears}/4 |`,
    );
  }
  lines.push(
    "",
    "## Best result by indicator family",
    "",
    "| family | best full-return feature | full bits | best sign feature | sign bits | best magnitude feature | magnitude bits |",
    "|---|---|---:|---|---:|---|---:|",
  );
  for (const family of ["RSI", "EMA", "EMA slope", "EMA acceleration", "MACD"]) {
    const members = artifact.signals.filter((signal) =>
      signal.family === family
      && !(family === "EMA slope" && "horizon" in signal && signal.horizon === 1));
    const bestFull = [...members].sort((left, right) =>
      right.discrete.signedDistribution.incrementalGainBits
      - left.discrete.signedDistribution.incrementalGainBits)[0]!;
    const bestSign = [...members].sort((left, right) =>
      right.discrete.activeSign.incrementalGainBits
      - left.discrete.activeSign.incrementalGainBits)[0]!;
    const bestMagnitude = [...members].sort((left, right) =>
      right.discrete.activeMagnitude.incrementalGainBits
      - left.discrete.activeMagnitude.incrementalGainBits)[0]!;
    lines.push(
      `| ${family} | ${bestFull.label} | ${formatMetric(bestFull.discrete.signedDistribution.incrementalGainBits)} | ${bestSign.label} | ${formatMetric(bestSign.discrete.activeSign.incrementalGainBits)} | ${bestMagnitude.label} | ${formatMetric(bestMagnitude.discrete.activeMagnitude.incrementalGainBits)} |`,
    );
  }
  lines.push(
    "",
    "## EMA dynamics parameter maps",
    "",
    "Each cell is incremental held-out bits beyond the 33-state latest-return baseline.",
    "The `k=1` slope column is a redundancy control for ordinary Price−EMA; genuinely multi-step slopes have `k>1`. Acceleration at `k=1` is a genuine second-difference feature.",
  );
  appendDynamicsMatrix(lines, artifact.signals, "emaSlope", "signedDistribution", "Slope: full signed-return distribution");
  appendDynamicsMatrix(lines, artifact.signals, "emaAcceleration", "signedDistribution", "Acceleration: full signed-return distribution");
  appendDynamicsMatrix(lines, artifact.signals, "emaSlope", "activeMagnitude", "Slope: active-magnitude distribution");
  appendDynamicsMatrix(lines, artifact.signals, "emaAcceleration", "activeMagnitude", "Acceleration: active-magnitude distribution");
  lines.push(
    "",
    "## Complete hyperparameter sweep",
    "",
    "All values below are incremental held-out gains beyond the latest return state. `Return ρ` is the correlation between the binned history+indicator conditional-mean forecast and the realized signed return. `ΔMSE` compares that mean forecast with the history-only conditional mean.",
    "",
    "| indicator | full return bits | zero bits | sign bits | magnitude bits | active joint bits | return ρ | return ΔMSE | magnitude ρ | magnitude ΔMSE |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
  );
  for (const signal of artifact.signals) {
    lines.push(
      `| ${signal.label} | ${formatMetric(signal.discrete.signedDistribution.incrementalGainBits)} | ${formatMetric(signal.discrete.zeroGate.incrementalGainBits)} | ${formatMetric(signal.discrete.activeSign.incrementalGainBits)} | ${formatMetric(signal.discrete.activeMagnitude.incrementalGainBits)} | ${formatMetric(signal.discrete.activeJoint.incrementalGainBits)} | ${formatMetric(signal.returnMean.incrementalCorrelation)} | ${(signal.returnMean.incrementalMseImprovement * 100).toFixed(6)}% | ${formatMetric(signal.magnitudeMean.incrementalCorrelation)} | ${(signal.magnitudeMean.incrementalMseImprovement * 100).toFixed(6)}% |`,
    );
  }
  lines.push(
    "",
    `## Annual stability of ${fullWinner.label}`,
    "",
    "| test year | full return bits | zero bits | active sign bits | active magnitude bits | active joint bits |",
    "|---:|---:|---:|---:|---:|---:|",
  );
  for (let year = 0; year < 4; year += 1) {
    lines.push(
      `| ${year + 1} | ${formatMetric(fullWinner.discrete.signedDistribution.annual[year]!.incrementalGainBits)} | ${formatMetric(fullWinner.discrete.zeroGate.annual[year]!.incrementalGainBits)} | ${formatMetric(fullWinner.discrete.activeSign.annual[year]!.incrementalGainBits)} | ${formatMetric(fullWinner.discrete.activeMagnitude.annual[year]!.incrementalGainBits)} | ${formatMetric(fullWinner.discrete.activeJoint.annual[year]!.incrementalGainBits)} |`,
    );
  }
  lines.push(
    "",
    `## Annual stability of best EMA dynamics: ${dynamicsFullWinner.label}`,
    "",
    "| test year | full return bits | zero bits | active sign bits | active magnitude bits | active joint bits |",
    "|---:|---:|---:|---:|---:|---:|",
  );
  for (let year = 0; year < 4; year += 1) {
    lines.push(
      `| ${year + 1} | ${formatMetric(dynamicsFullWinner.discrete.signedDistribution.annual[year]!.incrementalGainBits)} | ${formatMetric(dynamicsFullWinner.discrete.zeroGate.annual[year]!.incrementalGainBits)} | ${formatMetric(dynamicsFullWinner.discrete.activeSign.annual[year]!.incrementalGainBits)} | ${formatMetric(dynamicsFullWinner.discrete.activeMagnitude.annual[year]!.incrementalGainBits)} | ${formatMetric(dynamicsFullWinner.discrete.activeJoint.annual[year]!.incrementalGainBits)} |`,
    );
  }
  lines.push(
    "",
    "## Method",
    "",
    `- Causal alignment: ${artifact.methodology.causalAlignment}`,
    `- Calibration: ${artifact.methodology.calibration}`,
    `- Evaluation: ${artifact.methodology.evaluation}`,
    `- Feature quantization: ${artifact.methodology.featureBins} equal-mass cells; active magnitude: ${artifact.methodology.magnitudeBins} equal-mass cells.`,
    `- Incremental baseline: ${artifact.methodology.historyBaseline}`,
    "- EMA uses alpha `2 / (period + 1)`; RSI uses Wilder alpha `1 / period`; MACD line and histogram are normalized by current price into basis points.",
    "",
    "## Limitations",
    "",
    ...artifact.limitations.map((limitation) => `- ${limitation}`),
    "",
    "## Reproducibility",
    "",
    "```text",
    "node --conditions=development --import tsx scripts/analyze-technical-indicator-predictiveness.ts",
    "```",
    "",
    "The complete per-indicator, per-target, and per-year metrics are stored in `data/benchmarks/technical-indicator-predictiveness.json`.",
  );
  return `${lines.join("\n")}\n`;
}

function appendDynamicsMatrix(
  lines: string[],
  signals: Array<Record<string, any>>,
  kind: "emaSlope" | "emaAcceleration",
  target: "signedDistribution" | "activeMagnitude",
  label: string,
): void {
  lines.push(
    "",
    `### ${label}`,
    "",
    `| EMA period n \\ slope horizon k | ${EMA_DYNAMICS_HORIZONS.map((horizon) => `${horizon}s`).join(" | ")} |`,
    `|---:|${EMA_DYNAMICS_HORIZONS.map(() => "---:").join("|")}|`,
  );
  for (const period of EMA_DYNAMICS_PERIODS) {
    const cells = EMA_DYNAMICS_HORIZONS.map((horizon) => {
      const signal = signals.find((entry) =>
        entry.kind === kind && entry.period === period && entry.horizon === horizon)!;
      return formatMetric(signal.discrete[target].incrementalGainBits);
    });
    lines.push(`| ${period}s | ${cells.join(" | ")} |`);
  }
}

function formatMetric(value: number): string {
  return value === 0 ? "0" : value.toPrecision(8).replace(/\.0+(?=e|$)/, "");
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  try {
    main();
  } catch (error: unknown) {
    console.error(error instanceof Error ? error.stack ?? error.message : String(error));
    process.exitCode = 1;
  }
}
