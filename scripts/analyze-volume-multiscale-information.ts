import fs from "node:fs";
import path from "node:path";
import { deserialize, serialize } from "node:v8";
import { fileURLToPath, pathToFileURL } from "node:url";
import {
  readCandleShardReferenceSync,
  type SequentialCandle,
} from "@trading/storage";
import {
  buildSignalDefinitions,
  IndicatorEngine,
} from "./analyze-technical-indicator-predictiveness.ts";
import { evaluateCandidateYear } from "./analyze-indicator-information-basis.ts";

const DAY_MS = 86_400_000;
const SECOND_MS = 1_000;
const HISTORY_STATES = 33;
const FEATURE_BINS = 4;
const MAGNITUDE_BINS = 16;
const TARGET_CLASSES = 33;
const WARMUP_RETURNS = 12_000;
const EVALUATION_STRIDE = 4;
const CALIBRATION_SAMPLE_STRIDE = 256;
const CALIBRATION_CAPACITY = 131_072;
const MIN_COMPLETED_BARS = 65;
const DEFAULT_ANALYSIS = "data/benchmarks/log-return-distributions.json";
const DEFAULT_PRICE_CALIBRATION = "data/runtime-cache/technical-indicator-calibration.bin";
const DEFAULT_CALIBRATION = "data/runtime-cache/volume-multiscale-calibration.bin";
const DEFAULT_COUNTS = "data/runtime-cache/volume-multiscale-information-counts.bin";
const DEFAULT_OUTPUT = "data/benchmarks/volume-multiscale-information.json";
const DEFAULT_REPORT = "docs/experiments/volume-multiscale-information-2026-08-16.md";

const PRICE_BASIS_IDS = [
  "rsi-2",
  "ema-acceleration-2-1",
  "ema-slope-8-8",
] as const;
const PRICE_BENCHMARK_IDS = [
  "ema-acceleration-2-2",
  "ema-gap-8",
] as const;
const BAR_SIGNAL_IDS = [
  "rsi-2",
  "ema-acceleration-2-1",
  "ema-slope-8-8",
  "ema-gap-8",
] as const;

export const BAR_RESOLUTIONS = [
  { id: "5s", label: "5 seconds", milliseconds: 5_000 },
  { id: "15s", label: "15 seconds", milliseconds: 15_000 },
  { id: "1m", label: "1 minute", milliseconds: 60_000 },
  { id: "5m", label: "5 minutes", milliseconds: 300_000 },
  { id: "15m", label: "15 minutes", milliseconds: 900_000 },
  { id: "1h", label: "1 hour", milliseconds: 3_600_000 },
  { id: "4h", label: "4 hours", milliseconds: 14_400_000 },
] as const;

interface Options {
  analysisPath: string;
  priceCalibrationPath: string;
  calibrationPath: string;
  countsPath: string;
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

interface PriceCalibration {
  featureEdges: number[][];
  magnitudeEdges: number[];
}

export interface MarketFeatureDefinition {
  id: string;
  label: string;
  family: string;
  resolution: string;
  sourceIndex: number;
}

interface CandidateDefinition {
  id: string;
  label: string;
  family: string;
  resolution: string;
  source: "market" | "price";
  sourceIndex: number;
}

interface AnnualRow {
  year: number;
  observations: number;
  individualGainBits: number;
  cumulativeGainBits: number;
  marginalGainBits: number;
}

interface CandidateResult {
  id: string;
  label: string;
  family: string;
  resolution: string;
  observations: number;
  individualGainBits: number;
  cumulativeGainBits: number;
  marginalGainBits: number;
  positiveMarginalYears: number;
  annual: AnnualRow[];
}

interface AggregateBar {
  openTime: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface BarState {
  aggregate?: AggregateBar;
  completedBars: number;
  previousClose: number;
  logVolumeEma8: number;
  logVolumeEma32: number;
  indicator?: IndicatorEngine;
  indicatorValues: Float64Array;
}

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

export function buildMarketFeatureDefinitions(): MarketFeatureDefinition[] {
  const definitions: Array<Omit<MarketFeatureDefinition, "sourceIndex">> = [
    {
      id: "1s-log-volume",
      label: "1s log(1 + volume)",
      family: "volume level",
      resolution: "1s",
    },
    ...[2, 8, 32, 128, 512, 2048].map((period) => ({
      id: `1s-relative-log-volume-${period}`,
      label: `1s log-volume surprise vs EMA(${period}s)`,
      family: "relative volume",
      resolution: "1s",
    })),
    {
      id: "1s-range",
      label: "1s high-low range",
      family: "candle shape",
      resolution: "1s",
    },
    {
      id: "1s-close-location",
      label: "1s close location in range",
      family: "candle shape",
      resolution: "1s",
    },
    {
      id: "1s-signed-relative-volume",
      label: "1s return sign × relative volume",
      family: "price-volume interaction",
      resolution: "1s",
    },
    {
      id: "1s-volume-adjusted-magnitude",
      label: "1s absolute return / sqrt(volume)",
      family: "price-volume interaction",
      resolution: "1s",
    },
  ];
  for (const resolution of BAR_RESOLUTIONS) {
    const prefix = `last completed ${resolution.id}`;
    definitions.push(
      {
        id: `${resolution.id}-return`,
        label: `${prefix} return`,
        family: "completed-bar return",
        resolution: resolution.id,
      },
      {
        id: `${resolution.id}-absolute-return`,
        label: `${prefix} absolute return`,
        family: "completed-bar volatility",
        resolution: resolution.id,
      },
      {
        id: `${resolution.id}-range`,
        label: `${prefix} high-low range`,
        family: "completed-bar candle shape",
        resolution: resolution.id,
      },
      {
        id: `${resolution.id}-close-location`,
        label: `${prefix} close location in range`,
        family: "completed-bar candle shape",
        resolution: resolution.id,
      },
      {
        id: `${resolution.id}-log-volume`,
        label: `${prefix} log(1 + volume)`,
        family: "completed-bar volume level",
        resolution: resolution.id,
      },
      {
        id: `${resolution.id}-relative-log-volume-8`,
        label: `${prefix} log-volume surprise vs EMA(8 bars)`,
        family: "completed-bar relative volume",
        resolution: resolution.id,
      },
      {
        id: `${resolution.id}-relative-log-volume-32`,
        label: `${prefix} log-volume surprise vs EMA(32 bars)`,
        family: "completed-bar relative volume",
        resolution: resolution.id,
      },
      {
        id: `${resolution.id}-signed-relative-volume`,
        label: `${prefix} return sign × relative volume`,
        family: "completed-bar price-volume interaction",
        resolution: resolution.id,
      },
      ...BAR_SIGNAL_IDS.map((id) => {
        const suffix = id === "rsi-2" ? "RSI(2 bars)"
          : id === "ema-acceleration-2-1" ? "EMA acceleration(n=2,k=1 bars)"
          : id === "ema-slope-8-8" ? "EMA slope(n=8,k=8 bars)"
          : "close−EMA(8 bars)";
        return {
          id: `${resolution.id}-${id}`,
          label: `${prefix} ${suffix}`,
          family: "completed-bar indicator",
          resolution: resolution.id,
        };
      }),
    );
  }
  return definitions.map((definition, sourceIndex) => ({ ...definition, sourceIndex }));
}

export class CausalMarketFeatureEngine {
  readonly definitions = buildMarketFeatureDefinitions();
  private readonly valuesBuffer = new Float64Array(this.definitions.length);
  private readonly validBuffer = new Uint8Array(this.definitions.length);
  private readonly oneSecondVolumePeriods = [2, 8, 32, 128, 512, 2048];
  private readonly oneSecondLogVolumeEmas = new Float64Array(this.oneSecondVolumePeriods.length);
  private oneSecondInitialized = false;
  private previousOneSecondClose = Number.NaN;
  private readonly barSignalDefinitions = selectedPriceDefinitions(BAR_SIGNAL_IDS);
  private readonly barStates: BarState[] = BAR_RESOLUTIONS.map(() => ({
    completedBars: 0,
    previousClose: Number.NaN,
    logVolumeEma8: 0,
    logVolumeEma32: 0,
    indicatorValues: new Float64Array(BAR_SIGNAL_IDS.length),
  }));

  values(): Float64Array {
    return this.valuesBuffer;
  }

  valid(): Uint8Array {
    return this.validBuffer;
  }

  update(candle: SequentialCandle): void {
    this.updateOneSecond(candle);
    BAR_RESOLUTIONS.forEach((resolution, index) => {
      this.updateBar(index, resolution.milliseconds, candle);
    });
  }

  private updateOneSecond(candle: SequentialCandle): void {
    const logVolume = Math.log1p(candle.volume);
    const returnBps = Number.isFinite(this.previousOneSecondClose)
      ? Math.log(candle.close / this.previousOneSecondClose) * 10_000
      : 0;
    this.valuesBuffer[0] = logVolume;
    if (!this.oneSecondInitialized) {
      this.oneSecondLogVolumeEmas.fill(logVolume);
      this.oneSecondInitialized = true;
    }
    for (let index = 0; index < this.oneSecondVolumePeriods.length; index += 1) {
      this.valuesBuffer[1 + index] = logVolume - this.oneSecondLogVolumeEmas[index]!;
      const alpha = 2 / (this.oneSecondVolumePeriods[index]! + 1);
      this.oneSecondLogVolumeEmas[index] +=
        alpha * (logVolume - this.oneSecondLogVolumeEmas[index]!);
    }
    this.valuesBuffer[7] = candle.high > 0 && candle.low > 0
      ? Math.log(candle.high / candle.low) * 10_000
      : 0;
    this.valuesBuffer[8] = closeLocation(candle.high, candle.low, candle.close);
    this.valuesBuffer[9] = Math.sign(returnBps) * this.valuesBuffer[3]!;
    this.valuesBuffer[10] = Math.abs(returnBps) / Math.sqrt(candle.volume + 0.001);
    this.validBuffer.fill(1, 0, 11);
    this.previousOneSecondClose = candle.close;
  }

  private updateBar(index: number, resolutionMs: number, candle: SequentialCandle): void {
    const state = this.barStates[index]!;
    const bucketOpen = Math.floor(candle.openTime / resolutionMs) * resolutionMs;
    if (!state.aggregate || state.aggregate.openTime !== bucketOpen) {
      state.aggregate = {
        openTime: bucketOpen,
        open: candle.open,
        high: candle.high,
        low: candle.low,
        close: candle.close,
        volume: candle.volume,
      };
    } else {
      state.aggregate.high = Math.max(state.aggregate.high, candle.high);
      state.aggregate.low = Math.min(state.aggregate.low, candle.low);
      state.aggregate.close = candle.close;
      state.aggregate.volume += candle.volume;
    }
    if ((candle.openTime + SECOND_MS) % resolutionMs === 0) {
      this.completeBar(index, state);
      state.aggregate = undefined;
    }
  }

  private completeBar(resolutionIndex: number, state: BarState): void {
    const bar = state.aggregate!;
    const offset = 11 + resolutionIndex * 12;
    const returnBps = Number.isFinite(state.previousClose)
      ? Math.log(bar.close / state.previousClose) * 10_000
      : 0;
    const logVolume = Math.log1p(bar.volume);
    const relative8 = state.completedBars === 0 ? 0 : logVolume - state.logVolumeEma8;
    const relative32 = state.completedBars === 0 ? 0 : logVolume - state.logVolumeEma32;
    this.valuesBuffer[offset] = returnBps;
    this.valuesBuffer[offset + 1] = Math.abs(returnBps);
    this.valuesBuffer[offset + 2] = Math.log(bar.high / bar.low) * 10_000;
    this.valuesBuffer[offset + 3] = closeLocation(bar.high, bar.low, bar.close);
    this.valuesBuffer[offset + 4] = logVolume;
    this.valuesBuffer[offset + 5] = relative8;
    this.valuesBuffer[offset + 6] = relative32;
    this.valuesBuffer[offset + 7] = Math.sign(returnBps) * relative32;
    if (!state.indicator) {
      state.indicator = new IndicatorEngine(this.barSignalDefinitions, bar.close);
      state.logVolumeEma8 = logVolume;
      state.logVolumeEma32 = logVolume;
    } else {
      state.indicator.update(bar.close);
      state.logVolumeEma8 += 2 / 9 * (logVolume - state.logVolumeEma8);
      state.logVolumeEma32 += 2 / 33 * (logVolume - state.logVolumeEma32);
    }
    state.indicator.values(state.indicatorValues);
    for (let signal = 0; signal < BAR_SIGNAL_IDS.length; signal += 1) {
      this.valuesBuffer[offset + 8 + signal] = state.indicatorValues[signal]!;
    }
    state.completedBars += 1;
    if (state.completedBars >= MIN_COMPLETED_BARS) {
      this.validBuffer.fill(1, offset, offset + 12);
    }
    state.previousClose = bar.close;
  }
}

function main(): void {
  const options = parseOptions(process.argv.slice(2));
  const analysis = readJson<AnalysisReport>(options.analysisPath);
  const definitions = buildSignalDefinitions();
  const definitionIndex = new Map(definitions.map((definition, index) => [definition.id, index]));
  const priceCalibrationEnvelope = deserialize(fs.readFileSync(options.priceCalibrationPath)) as {
    calibration: PriceCalibration;
  };
  const priceCalibration = priceCalibrationEnvelope.calibration;
  const files = selectedFiles(
    path.resolve(repoRoot, analysis.source.oneSecond.referenceDirectory),
    Date.parse(analysis.fullHistory.startTime),
    Date.parse(analysis.fullHistory.endTime),
  );
  const marketDefinitions = buildMarketFeatureDefinitions();
  const calibrationMetadata = JSON.stringify({
    version: 1,
    analysisGeneratedAt: analysis.generatedAt,
    marketDefinitions,
    calibrationSampleStride: CALIBRATION_SAMPLE_STRIDE,
    calibrationCapacity: CALIBRATION_CAPACITY,
    minimumCompletedBars: MIN_COMPLETED_BARS,
  });
  let marketEdges: number[][];
  if (!options.rebuildCalibration && fs.existsSync(options.calibrationPath)) {
    const cached = deserialize(fs.readFileSync(options.calibrationPath)) as {
      metadata: string;
      marketEdges: number[][];
    };
    if (cached.metadata === calibrationMetadata) {
      console.log(`Loading calibration from ${path.relative(repoRoot, options.calibrationPath)}`);
      marketEdges = cached.marketEdges;
    } else {
      marketEdges = calibrateMarketFeatures(files, analysis, marketDefinitions.length);
      writeBinary(options.calibrationPath, { metadata: calibrationMetadata, marketEdges });
    }
  } else {
    marketEdges = calibrateMarketFeatures(files, analysis, marketDefinitions.length);
    writeBinary(options.calibrationPath, { metadata: calibrationMetadata, marketEdges });
  }
  const candidateDefinitions: CandidateDefinition[] = [
    ...marketDefinitions.map((definition): CandidateDefinition => ({
      ...definition,
      source: "market",
    })),
    ...PRICE_BENCHMARK_IDS.map((id): CandidateDefinition => {
      const sourceIndex = definitionIndex.get(id)!;
      const definition = definitions[sourceIndex]!;
      return {
        id: `price-${id}`,
        label: `${definition.label} (price-basis benchmark)`,
        family: definition.family,
        resolution: "1s",
        source: "price",
        sourceIndex,
      };
    }),
  ];
  const candidateEdges = [
    ...marketEdges,
    ...PRICE_BENCHMARK_IDS.map((id) => quartileEdges(
      priceCalibration.featureEdges[definitionIndex.get(id)!]!,
    )),
  ];
  const countsMetadata = JSON.stringify({
    version: 1,
    analysisGeneratedAt: analysis.generatedAt,
    candidateDefinitions,
    candidateEdges,
    priceBasisIds: PRICE_BASIS_IDS,
    evaluationStride: EVALUATION_STRIDE,
    warmupReturns: WARMUP_RETURNS,
  });
  let annualRows: AnnualRow[][];
  if (!options.rebuildCounts && fs.existsSync(options.countsPath)) {
    const cached = deserialize(fs.readFileSync(options.countsPath)) as {
      metadata: string;
      annualRows: AnnualRow[][];
    };
    if (cached.metadata === countsMetadata) {
      console.log(`Loading counts from ${path.relative(repoRoot, options.countsPath)}`);
      annualRows = cached.annualRows;
    } else {
      annualRows = scanCounts(
        files,
        analysis,
        candidateDefinitions,
        candidateEdges,
        priceCalibration,
        definitionIndex,
      );
      writeBinary(options.countsPath, { metadata: countsMetadata, annualRows });
    }
  } else {
    annualRows = scanCounts(
      files,
      analysis,
      candidateDefinitions,
      candidateEdges,
      priceCalibration,
      definitionIndex,
    );
    writeBinary(options.countsPath, { metadata: countsMetadata, annualRows });
  }
  const results = candidateDefinitions.map((definition, index) => summarize(
    definition,
    annualRows[index]!,
  ));
  const ranked = [...results].sort(compareResults);
  const benchmark = results.find((result) =>
    result.id === "price-ema-acceleration-2-2")!;
  const bestExternal = ranked.find((result) => !result.id.startsWith("price-"))!;
  const coreThreeGainBits = weightedMean(
    benchmark.annual,
    (row) => row.cumulativeGainBits - row.marginalGainBits,
  );
  const artifact = {
    version: 1,
    generatedAt: new Date().toISOString(),
    symbol: analysis.source.symbol,
    scale: "causal history through t → next 1s return",
    window: analysis.fullHistory,
    priceBasis: PRICE_BASIS_IDS,
    target: "33 cells: zero or sign × 16 active-magnitude cells",
    evaluation: {
      rollingWindows: "Four annual held-out epochs; each epoch is scored from all preceding years.",
      sampleStride: EVALUATION_STRIDE,
      calibration: "Candidate quartile edges fitted from deterministic samples in year 1 only.",
      higherTimeframePolicy: "UTC-aligned bars become visible only after their final 1s candle closes.",
      minimumCompletedBars: MIN_COMPLETED_BARS,
    },
    candidatesEvaluated: results.length,
    coreThreeGainBits,
    bestExternal,
    priceFourthBenchmark: benchmark,
    externalBeatsExistingFourth: bestExternal.marginalGainBits > benchmark.marginalGainBits,
    rankings: ranked,
    bestByResolution: ["1s", ...BAR_RESOLUTIONS.map((row) => row.id)].map((resolution) =>
      ranked.find((result) => result.resolution === resolution && !result.id.startsWith("price-"))),
    bestByFamily: Array.from(new Set(results
      .filter((result) => !result.id.startsWith("price-"))
      .map((result) => result.family)))
      .map((family) => ranked.find((result) => result.family === family)),
    limitations: [
      "The spot candle archive contains base volume but not quote volume, trade count, or taker-buy/sell imbalance.",
      "This scan tests additions after the previously selected three-feature price basis; it does not exhaustively enumerate every mixed price/volume subset.",
      "Each candidate is quartile-quantized. Continuous models may retain useful within-cell variation.",
      "Aligned higher-timeframe bars are deliberately stale between closes; partial bars are excluded to prevent future leakage.",
      "Predictive log information is not a trading-PnL estimate and does not include fees, spread, latency, or impact.",
    ],
  };
  fs.mkdirSync(path.dirname(options.outputPath), { recursive: true });
  fs.writeFileSync(options.outputPath, `${JSON.stringify(artifact, null, 2)}\n`, "utf8");
  fs.mkdirSync(path.dirname(options.reportPath), { recursive: true });
  fs.writeFileSync(options.reportPath, renderReport(artifact), "utf8");
  console.log(`Wrote ${path.relative(repoRoot, options.outputPath)}`);
  console.log(`Wrote ${path.relative(repoRoot, options.reportPath)}`);
}

function calibrateMarketFeatures(
  files: Array<{ file: string; dayStart: number }>,
  analysis: AnalysisReport,
  featureCount: number,
): number[][] {
  console.error("Calibrating volume and completed-bar feature quartiles on year 1...");
  const samples = Array.from(
    { length: featureCount },
    () => new Float64Array(CALIBRATION_CAPACITY),
  );
  const sampleCounts = new Uint32Array(featureCount);
  const engine = new CausalMarketFeatureEngine();
  const end = anniversary(Date.parse(analysis.fullHistory.startTime), 1);
  let returnsSeen = 0;
  for (const [fileIndex, entry] of files.entries()) {
    if (entry.dayStart >= end) break;
    if (fileIndex % 50 === 0) console.error(`Calibration ${fileIndex}/366...`);
    const candles = readCompleteDay(entry.file);
    for (const candle of candles) {
      if (returnsSeen >= WARMUP_RETURNS
        && returnsSeen % CALIBRATION_SAMPLE_STRIDE === 0) {
        const values = engine.values();
        const valid = engine.valid();
        for (let feature = 0; feature < featureCount; feature += 1) {
          if (valid[feature] !== 1) continue;
          const count = sampleCounts[feature]!;
          if (count < CALIBRATION_CAPACITY) {
            samples[feature]![count] = values[feature]!;
            sampleCounts[feature] = count + 1;
          }
        }
      }
      engine.update(candle);
      returnsSeen += 1;
    }
  }
  return samples.map((sample, index) => {
    const count = sampleCounts[index]!;
    if (count < 100) throw new Error(`Feature ${index} has only ${count} calibration samples.`);
    const sorted = sample.slice(0, count).sort();
    return [0.25, 0.5, 0.75].map((quantile) =>
      sorted[Math.min(count - 1, Math.floor(quantile * count))]!);
  });
}

function scanCounts(
  files: Array<{ file: string; dayStart: number }>,
  analysis: AnalysisReport,
  candidates: CandidateDefinition[],
  candidateEdges: number[][],
  priceCalibration: PriceCalibration,
  definitionIndex: Map<string, number>,
): AnnualRow[][] {
  const singleLength = HISTORY_STATES * FEATURE_BINS * TARGET_CLASSES;
  const selectedDimensions = PRICE_BASIS_IDS.length;
  const fullFeatureStates = FEATURE_BINS ** (selectedDimensions + 1);
  const conditionalLength = HISTORY_STATES * fullFeatureStates * TARGET_CLASSES;
  console.error(
    `Scanning ${candidates.length} candidates at 1/${EVALUATION_STRIDE} targets; `
      + `${conditionalLength.toLocaleString("en-US")} conditional cells each...`,
  );
  const singleTrain = candidates.map(() => new Float64Array(singleLength));
  const singleCurrent = candidates.map(() => new Float64Array(singleLength));
  const conditionalTrain = candidates.map(() => new Float64Array(conditionalLength));
  const conditionalCurrent = candidates.map(() => new Float64Array(conditionalLength));
  const annualRows = candidates.map((): AnnualRow[] => []);
  const marketEngine = new CausalMarketFeatureEngine();
  const priceDefinitions = selectedPriceDefinitions([
    ...PRICE_BASIS_IDS,
    ...PRICE_BENCHMARK_IDS,
  ]);
  const priceEngineValues = new Float64Array(priceDefinitions.length);
  const priceEngineIndex = new Map(priceDefinitions.map((definition, index) => [definition.id, index]));
  const priceBasisEdges = PRICE_BASIS_IDS.map((id) => quartileEdges(
    priceCalibration.featureEdges[definitionIndex.get(id)!]!,
  ));
  const priceEngine = { current: undefined as IndicatorEngine | undefined };
  const start = Date.parse(analysis.fullHistory.startTime);
  let previousClose = Number.NaN;
  let previousReturnState = 0;
  let returnsSeen = 0;
  let currentYear = 0;
  for (const [fileIndex, entry] of files.entries()) {
    const year = anniversaryIndex(start, entry.dayStart);
    if (year !== currentYear) {
      finalizeCountYear(
        singleTrain,
        singleCurrent,
        conditionalTrain,
        conditionalCurrent,
        annualRows,
        selectedDimensions,
        currentYear,
      );
      currentYear = year;
    }
    if (fileIndex % 50 === 0) console.error(`Information scan ${fileIndex}/${files.length}...`);
    const candles = readCompleteDay(entry.file);
    for (const candle of candles) {
      if (!priceEngine.current) {
        priceEngine.current = new IndicatorEngine(priceDefinitions, candle.close);
        marketEngine.update(candle);
      } else {
        const returnBps = candle.close === previousClose
          ? 0
          : Math.log(candle.close / previousClose) * 10_000;
        const active = returnBps !== 0;
        const sign = returnBps > 0 ? 1 : 0;
        const magnitudeBin = active
          ? upperBound(priceCalibration.magnitudeEdges, Math.abs(returnBps))
          : -1;
        const targetClass = active ? 1 + sign * MAGNITUDE_BINS + magnitudeBin : 0;
        if (returnsSeen >= WARMUP_RETURNS && isEvaluationTarget(returnsSeen)) {
          priceEngine.current.values(priceEngineValues);
          let selectedState = 0;
          for (let basis = 0; basis < PRICE_BASIS_IDS.length; basis += 1) {
            const value = priceEngineValues[priceEngineIndex.get(PRICE_BASIS_IDS[basis]!)!]!;
            selectedState = selectedState * FEATURE_BINS
              + upperBound(priceBasisEdges[basis]!, value);
          }
          const marketValues = marketEngine.values();
          const marketValid = marketEngine.valid();
          for (let candidate = 0; candidate < candidates.length; candidate += 1) {
            const definition = candidates[candidate]!;
            if (definition.source === "market" && marketValid[definition.sourceIndex] !== 1) {
              continue;
            }
            const value = definition.source === "market"
              ? marketValues[definition.sourceIndex]!
              : priceEngineValues[priceEngineIndex.get(
                definition.id.slice("price-".length),
              )!]!;
            const featureBin = upperBound(candidateEdges[candidate]!, value);
            singleCurrent[candidate]![
              (previousReturnState * FEATURE_BINS + featureBin) * TARGET_CLASSES + targetClass
            ] += 1;
            const featureState = selectedState * FEATURE_BINS + featureBin;
            conditionalCurrent[candidate]![
              (previousReturnState * fullFeatureStates + featureState) * TARGET_CLASSES + targetClass
            ] += 1;
          }
        }
        previousReturnState = active
          ? 1 + sign * MAGNITUDE_BINS + magnitudeBin
          : 0;
        priceEngine.current.update(candle.close);
        marketEngine.update(candle);
        returnsSeen += 1;
      }
      previousClose = candle.close;
    }
  }
  finalizeCountYear(
    singleTrain,
    singleCurrent,
    conditionalTrain,
    conditionalCurrent,
    annualRows,
    selectedDimensions,
    currentYear,
  );
  return annualRows;
}

function finalizeCountYear(
  singleTrain: Float64Array[],
  singleCurrent: Float64Array[],
  conditionalTrain: Float64Array[],
  conditionalCurrent: Float64Array[],
  annualRows: AnnualRow[][],
  selectedDimensions: number,
  year: number,
): void {
  console.error(`Finalizing information year ${year}...`);
  for (let candidate = 0; candidate < annualRows.length; candidate += 1) {
    if (year > 0) {
      const single = evaluateCandidateYear(
        singleTrain[candidate]!,
        singleCurrent[candidate]!,
        0,
        year,
      );
      const conditional = evaluateCandidateYear(
        conditionalTrain[candidate]!,
        conditionalCurrent[candidate]!,
        selectedDimensions,
        year,
      );
      annualRows[candidate]!.push({
        year,
        observations: conditional.observations,
        individualGainBits: single.marginalGainBits,
        cumulativeGainBits: conditional.cumulativeGainBits,
        marginalGainBits: conditional.marginalGainBits,
      });
    }
    addInPlace(singleTrain[candidate]!, singleCurrent[candidate]!);
    singleCurrent[candidate]!.fill(0);
    addInPlace(conditionalTrain[candidate]!, conditionalCurrent[candidate]!);
    conditionalCurrent[candidate]!.fill(0);
  }
}

function summarize(definition: CandidateDefinition, annual: AnnualRow[]): CandidateResult {
  const observations = annual.reduce((sum, row) => sum + row.observations, 0);
  return {
    id: definition.id,
    label: definition.label,
    family: definition.family,
    resolution: definition.resolution,
    observations,
    individualGainBits: weightedMean(annual, (row) => row.individualGainBits),
    cumulativeGainBits: weightedMean(annual, (row) => row.cumulativeGainBits),
    marginalGainBits: weightedMean(annual, (row) => row.marginalGainBits),
    positiveMarginalYears: annual.filter((row) => row.marginalGainBits > 0).length,
    annual,
  };
}

function weightedMean(rows: AnnualRow[], value: (row: AnnualRow) => number): number {
  const observations = rows.reduce((sum, row) => sum + row.observations, 0);
  return rows.reduce((sum, row) => sum + value(row) * row.observations, 0) / observations;
}

function renderReport(artifact: any): string {
  const best = artifact.bestExternal;
  const benchmark = artifact.priceFourthBenchmark;
  const lines = [
    "# Volume and completed higher-timeframe information for the next 1s return",
    "",
    `Generated ${artifact.generatedAt} from ${artifact.symbol} spot OHLCV history.`,
    "",
    "## Answer",
    "",
    artifact.externalBeatsExistingFourth
      ? `The earlier price-only basis was not the best tested basis. **${best.label}** adds ${formatMetric(best.marginalGainBits)} bits/target after the price core-three, versus ${formatMetric(benchmark.marginalGainBits)} for the previous fourth feature.`
      : `The earlier price-only fourth feature remains stronger. The best new input, **${best.label}**, adds ${formatMetric(best.marginalGainBits)} bits/target after the price core-three, versus ${formatMetric(benchmark.marginalGainBits)} for the previous fourth feature.`,
    "",
    `The price core-three itself contributes ${formatMetric(artifact.coreThreeGainBits)} bits/target beyond the latest-return state. The best external feature changes geometric assigned probability by ${formatPercent(2 ** best.marginalGainBits - 1)} on top of that state.`,
    "",
    "## Strongest additions after the price core-three",
    "",
    "| rank | candidate | scale | family | individual bits | marginal bits | cumulative bits | positive years |",
    "|---:|---|---|---|---:|---:|---:|---:|",
  ];
  artifact.rankings.slice(0, 30).forEach((row: CandidateResult, index: number) => {
    lines.push(
      `| ${index + 1} | ${row.label} | ${row.resolution} | ${row.family} | ${formatMetric(row.individualGainBits)} | ${formatMetric(row.marginalGainBits)} | ${formatMetric(row.cumulativeGainBits)} | ${row.positiveMarginalYears}/4 |`,
    );
  });
  lines.push(
    "",
    "## Best new feature at each resolution",
    "",
    "| scale | candidate | marginal bits | positive years |",
    "|---|---|---:|---:|",
  );
  for (const row of artifact.bestByResolution as CandidateResult[]) {
    lines.push(`| ${row.resolution} | ${row.label} | ${formatMetric(row.marginalGainBits)} | ${row.positiveMarginalYears}/4 |`);
  }
  lines.push(
    "",
    "## Causal construction",
    "",
    "For a target return from second `t` to `t+1`, every input contains data only through second `t`. A UTC-aligned 5s, 15s, 1m, 5m, 15m, 1h, or 4h bar becomes visible only when its final constituent 1s candle has closed. The current partial bar is never exposed.",
    "",
    "All candidate features are split into four year-1 equal-mass cells. We score the exact next-return distribution with rolling held-out log information:",
    "",
    "```text",
    "mean_test[log2 P_train(R | previous_return, price_core3, candidate)",
    "        - log2 P_train(R | previous_return, price_core3)]",
    "```",
    "",
    `The deterministic evaluation sample uses one of every ${artifact.evaluation.sampleStride} targets. The history is still large enough to score four separate annual test epochs, each trained only on earlier years.`,
    "",
    "## Limits",
    "",
    ...artifact.limitations.map((limitation: string) => `- ${limitation}`),
    "",
    "## Reproducibility",
    "",
    "```text",
    "node --conditions=development --import tsx scripts/analyze-volume-multiscale-information.ts",
    "```",
    "",
    "Complete rankings are stored in `data/benchmarks/volume-multiscale-information.json`.",
  );
  return `${lines.join("\n")}\n`;
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
    priceCalibrationPath: path.resolve(
      repoRoot,
      values.get("price-calibration") ?? DEFAULT_PRICE_CALIBRATION,
    ),
    calibrationPath: path.resolve(repoRoot, values.get("calibration") ?? DEFAULT_CALIBRATION),
    countsPath: path.resolve(repoRoot, values.get("counts") ?? DEFAULT_COUNTS),
    outputPath: path.resolve(repoRoot, values.get("output") ?? DEFAULT_OUTPUT),
    reportPath: path.resolve(repoRoot, values.get("report") ?? DEFAULT_REPORT),
    rebuildCalibration,
    rebuildCounts,
  };
}

function selectedPriceDefinitions(ids: readonly string[]) {
  const wanted = new Set(ids);
  const byId = new Map(buildSignalDefinitions().map((definition) => [definition.id, definition]));
  return ids.map((id) => {
    const definition = byId.get(id);
    if (!definition || !wanted.has(definition.id)) throw new Error(`Missing price definition ${id}.`);
    return definition;
  });
}

function closeLocation(high: number, low: number, close: number): number {
  return high > low ? (2 * close - high - low) / (high - low) : 0;
}

function quartileEdges(edges: number[]): number[] {
  if (edges.length !== 15) throw new Error("Expected 15 price calibration edges.");
  return [edges[3]!, edges[7]!, edges[11]!];
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
  let mixed = index ^ (index >>> 16);
  mixed = Math.imul(mixed, 0x45d9f3b);
  mixed ^= mixed >>> 16;
  return (mixed >>> 0) % EVALUATION_STRIDE === 0;
}

function compareResults(left: CandidateResult, right: CandidateResult): number {
  const stableDifference = right.positiveMarginalYears - left.positiveMarginalYears;
  if (stableDifference !== 0) return stableDifference;
  return right.marginalGainBits - left.marginalGainBits;
}

function addInPlace(target: Float64Array, source: Float64Array): void {
  for (let index = 0; index < target.length; index += 1) target[index] += source[index]!;
}

function readCompleteDay(file: string): SequentialCandle[] {
  const candles = readCandleShardReferenceSync(file);
  if (candles.length !== 86_400) throw new Error(`${path.basename(file)} is incomplete.`);
  return candles;
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
  for (let index = 1; index < result.length; index += 1) {
    if (result[index]!.dayStart !== result[index - 1]!.dayStart + DAY_MS) {
      throw new Error(`Missing shard before ${new Date(result[index]!.dayStart).toISOString()}.`);
    }
  }
  return result;
}

function anniversary(startMs: number, years: number): number {
  const start = new Date(startMs);
  return Date.UTC(start.getUTCFullYear() + years, start.getUTCMonth(), start.getUTCDate());
}

function anniversaryIndex(startMs: number, dayMs: number): number {
  const start = new Date(startMs);
  const day = new Date(dayMs);
  let index = day.getUTCFullYear() - start.getUTCFullYear();
  if (dayMs < anniversary(startMs, index)) index -= 1;
  return Math.max(0, index);
}

function writeBinary(file: string, value: unknown): void {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, serialize(value));
}

function readJson<T>(file: string): T {
  return JSON.parse(fs.readFileSync(file, "utf8")) as T;
}

function formatMetric(value: number): string {
  return value === 0 ? "0" : value.toPrecision(8).replace(/\.0+(?=e|$)/, "");
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(4)}%`;
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  try {
    main();
  } catch (error: unknown) {
    console.error(error instanceof Error ? error.stack ?? error.message : String(error));
    process.exitCode = 1;
  }
}
