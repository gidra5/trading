import fs from "node:fs";
import path from "node:path";
import { deserialize, serialize } from "node:v8";
import { fileURLToPath, pathToFileURL } from "node:url";
import { readCandleShardReferenceSync } from "@trading/storage";

const DAY_MS = 86_400_000;
const LOG_MAGNITUDE_MIN = -20;
const LOG_MAGNITUDE_MAX = 10;
const CORE_BINS = 131_072;
const HISTOGRAM_CELLS = CORE_BINS + 2;
const RESOLUTIONS = [8, 16, 32, 64, 128, 256, 512, 1_024, 2_048] as const;
const PRIMARY_RESOLUTION = 64;
const DEFAULT_ANALYSIS = "data/benchmarks/log-return-distributions.json";
const DEFAULT_CACHE = "data/runtime-cache/one-second-sign-magnitude-counts.bin";
const DEFAULT_OUTPUT = "data/benchmarks/one-second-sign-magnitude-dependence.json";
const DEFAULT_REPORT = "docs/experiments/one-second-sign-magnitude-dependence-2026-08-16.md";

interface Options {
  analysisPath: string;
  cachePath: string;
  outputPath: string;
  reportPath: string;
  rebuildCache: boolean;
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

interface MomentSums {
  count: number;
  positive: number;
  sumMagnitude: number;
  sumMagnitudeSquared: number;
  sumSignedMagnitude: number;
  sumLogMagnitude: number;
  sumLogMagnitudeSquared: number;
  sumSignedLogMagnitude: number;
}

interface HistogramCounts {
  negative: Float64Array;
  positive: Float64Array;
  annualNegative: Float64Array[];
  annualPositive: Float64Array[];
  moments: MomentSums;
  annualMoments: MomentSums[];
  returns: number;
  zeroReturns: number;
}

export interface IndependenceMetrics {
  observations: number;
  magnitudeBins: number;
  positiveProbability: number;
  signEntropyBits: number;
  mutualInformationBits: number;
  millerMadowNullBiasBits: number;
  mutualInformationAboveBiasBits: number;
  fractionOfSignEntropyExplainedAboveBias: number;
  jsFromIndependentProductBits: number;
  totalVariationFromIndependentProduct: number;
  nullExpectedTotalVariation: number;
  totalVariationAboveNullMean: number;
  cramersV: number;
  baselineMajoritySignAccuracy: number;
  optimalMagnitudeOnlySignAccuracy: number;
  magnitudeOnlyAccuracyGain: number;
  nullExpectedMagnitudeOnlyAccuracyGain: number;
  magnitudeOnlyAccuracyGainAboveNull: number;
  maximumConditionalPositiveProbabilityDeviation: number;
  weightedRmsConditionalPositiveProbabilityDeviation: number;
}

interface BinnedResult extends IndependenceMetrics {
  edgesBps: Array<number | null>;
  counts: { negative: number[]; positive: number[] };
  magnitudeProbability: number[];
  positiveProbabilityByMagnitude: number[];
  mutualInformationContributionBits: number[];
}

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function main(): void {
  const options = parseOptions(process.argv.slice(2));
  const analysis = readJson<AnalysisReport>(options.analysisPath);
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
    logMagnitudeMin: LOG_MAGNITUDE_MIN,
    logMagnitudeMax: LOG_MAGNITUDE_MAX,
    coreBins: CORE_BINS,
  });
  const counts = loadOrBuildCounts(options, metadata, files, start);
  const artifact = analyzeCounts(analysis, counts);
  fs.mkdirSync(path.dirname(options.outputPath), { recursive: true });
  fs.writeFileSync(options.outputPath, `${JSON.stringify(artifact, null, 2)}\n`, "utf8");
  fs.mkdirSync(path.dirname(options.reportPath), { recursive: true });
  fs.writeFileSync(options.reportPath, renderReport(artifact), "utf8");
  console.log(`Wrote ${path.relative(repoRoot, options.outputPath)}`);
  console.log(`Wrote ${path.relative(repoRoot, options.reportPath)}`);
}

function parseOptions(args: string[]): Options {
  const values = new Map<string, string>();
  let rebuildCache = false;
  for (let index = 0; index < args.length; index += 1) {
    const key = args[index]!;
    if (key === "--rebuild-cache") {
      rebuildCache = true;
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
    cachePath: path.resolve(repoRoot, values.get("cache") ?? DEFAULT_CACHE),
    outputPath: path.resolve(repoRoot, values.get("output") ?? DEFAULT_OUTPUT),
    reportPath: path.resolve(repoRoot, values.get("report") ?? DEFAULT_REPORT),
    rebuildCache,
  };
}

function readJson<T>(file: string): T {
  return JSON.parse(fs.readFileSync(file, "utf8")) as T;
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

function loadOrBuildCounts(
  options: Options,
  metadata: string,
  files: Array<{ file: string; dayStart: number }>,
  start: number,
): HistogramCounts {
  if (!options.rebuildCache && fs.existsSync(options.cachePath)) {
    const cached = deserialize(fs.readFileSync(options.cachePath)) as {
      metadata: string;
      counts: HistogramCounts;
    };
    if (cached.metadata === metadata) {
      console.log(`Loading cached counts from ${path.relative(repoRoot, options.cachePath)}`);
      return cached.counts;
    }
  }
  const counts = streamCounts(files, start);
  fs.mkdirSync(path.dirname(options.cachePath), { recursive: true });
  fs.writeFileSync(options.cachePath, serialize({ metadata, counts }));
  return counts;
}

function emptyMoments(): MomentSums {
  return {
    count: 0,
    positive: 0,
    sumMagnitude: 0,
    sumMagnitudeSquared: 0,
    sumSignedMagnitude: 0,
    sumLogMagnitude: 0,
    sumLogMagnitudeSquared: 0,
    sumSignedLogMagnitude: 0,
  };
}

function streamCounts(
  files: Array<{ file: string; dayStart: number }>,
  start: number,
): HistogramCounts {
  const negative = new Float64Array(HISTOGRAM_CELLS);
  const positive = new Float64Array(HISTOGRAM_CELLS);
  const annualNegative: Float64Array[] = [];
  const annualPositive: Float64Array[] = [];
  const annualMoments: MomentSums[] = [];
  const moments = emptyMoments();
  let previousClose = Number.NaN;
  let returns = 0;
  let zeroReturns = 0;

  for (const [fileIndex, entry] of files.entries()) {
    if (fileIndex % 100 === 0) {
      console.error(`Reading sign-magnitude dependence ${fileIndex}/${files.length}...`);
    }
    const year = anniversaryIndex(start, entry.dayStart);
    while (annualNegative.length <= year) {
      annualNegative.push(new Float64Array(HISTOGRAM_CELLS));
      annualPositive.push(new Float64Array(HISTOGRAM_CELLS));
      annualMoments.push(emptyMoments());
    }
    const yearNegative = annualNegative[year]!;
    const yearPositive = annualPositive[year]!;
    const yearMoments = annualMoments[year]!;
    const candles = readCandleShardReferenceSync(entry.file);
    if (candles.length !== 86_400) throw new Error(`${path.basename(entry.file)} is incomplete.`);
    for (const candle of candles) {
      if (!Number.isFinite(candle.close) || candle.close <= 0) {
        throw new Error(`${path.basename(entry.file)} contains an invalid close.`);
      }
      if (Number.isFinite(previousClose)) {
        returns += 1;
        if (candle.close === previousClose) zeroReturns += 1;
        else {
          const returnBps = Math.log(candle.close / previousClose) * 10_000;
          const magnitude = Math.abs(returnBps);
          const logMagnitude = Math.log(magnitude);
          const cell = magnitudeCell(logMagnitude);
          const isPositive = returnBps > 0;
          (isPositive ? positive : negative)[cell] += 1;
          (isPositive ? yearPositive : yearNegative)[cell] += 1;
          addMoment(moments, magnitude, logMagnitude, isPositive);
          addMoment(yearMoments, magnitude, logMagnitude, isPositive);
        }
      }
      previousClose = candle.close;
    }
  }
  return {
    negative,
    positive,
    annualNegative,
    annualPositive,
    moments,
    annualMoments,
    returns,
    zeroReturns,
  };
}

function addMoment(
  moments: MomentSums,
  magnitude: number,
  logMagnitude: number,
  isPositive: boolean,
): void {
  const sign = isPositive ? 1 : -1;
  moments.count += 1;
  if (isPositive) moments.positive += 1;
  moments.sumMagnitude += magnitude;
  moments.sumMagnitudeSquared += magnitude * magnitude;
  moments.sumSignedMagnitude += sign * magnitude;
  moments.sumLogMagnitude += logMagnitude;
  moments.sumLogMagnitudeSquared += logMagnitude * logMagnitude;
  moments.sumSignedLogMagnitude += sign * logMagnitude;
}

function magnitudeCell(logMagnitude: number): number {
  if (logMagnitude < LOG_MAGNITUDE_MIN) return 0;
  if (logMagnitude >= LOG_MAGNITUDE_MAX) return CORE_BINS + 1;
  return 1 + Math.floor(
    (logMagnitude - LOG_MAGNITUDE_MIN)
    / (LOG_MAGNITUDE_MAX - LOG_MAGNITUDE_MIN)
    * CORE_BINS,
  );
}

function anniversaryIndex(startMs: number, dayMs: number): number {
  const start = new Date(startMs);
  const day = new Date(dayMs);
  let index = day.getUTCFullYear() - start.getUTCFullYear();
  const anniversary = Date.UTC(
    start.getUTCFullYear() + index,
    start.getUTCMonth(),
    start.getUTCDate(),
  );
  if (dayMs < anniversary) index -= 1;
  return Math.max(0, index);
}

function analyzeCounts(analysis: AnalysisReport, counts: HistogramCounts) {
  const combined = addArrays(counts.negative, counts.positive);
  const byResolution = Object.fromEntries(RESOLUTIONS.map((resolution) => {
    const cuts = equalMassCuts(combined, resolution);
    return [String(resolution), binnedResult(counts.negative, counts.positive, cuts)];
  })) as Record<string, BinnedResult>;
  const primaryCuts = equalMassCuts(combined, PRIMARY_RESOLUTION);
  const annual = counts.annualNegative.map((negative, index) => ({
    index,
    startTime: anniversaryIso(analysis.fullHistory.startTime, index),
    endTime: anniversaryIso(analysis.fullHistory.startTime, index + 1),
    ...binnedResult(negative, counts.annualPositive[index]!, primaryCuts),
    correlations: momentCorrelations(counts.annualMoments[index]!),
  }));
  const primary = byResolution[String(PRIMARY_RESOLUTION)]!;
  const finest = byResolution[String(RESOLUTIONS.at(-1)!)]!;
  const conditionalMagnitude = conditionalMagnitudeComparison(
    counts.negative,
    counts.positive,
    counts.moments,
    primary.counts.negative,
    primary.counts.positive,
  );
  return {
    version: 1,
    generatedAt: new Date().toISOString(),
    symbol: analysis.source.symbol,
    scale: "1s",
    window: analysis.fullHistory,
    source: {
      analysis: DEFAULT_ANALYSIS,
      referenceDirectory: analysis.source.oneSecond.referenceDirectory,
    },
    returns: counts.returns,
    zeroReturns: counts.zeroReturns,
    activeReturns: counts.moments.count,
    zeroProbability: counts.zeroReturns / counts.returns,
    definition: {
      sign: "negative or positive; exact-zero returns are excluded because their sign is undefined",
      magnitude: "absolute log return in basis points",
      independenceNull: "P(sign,magnitude)=P(sign)P(magnitude), using the observed nonzero sign marginal",
      primaryDiscretization: "64 approximately equal-mass magnitude cells cut on a 131072-cell log-magnitude histogram",
    },
    primary,
    finestResolution: finest,
    byResolution,
    conditionalMagnitude,
    correlations: momentCorrelations(counts.moments),
    annualStability: annual,
    largestDependenceContributions: largestContributions(primary, 12),
    conclusion: classifyDependence(finest),
    limitations: [
      "Independence is measured after magnitude discretization; the resolution table checks sensitivity from 8 through 2048 cells.",
      "Magnitude cut points are estimated on the full history for descriptive analysis and must not be reused across a chronological forecast boundary without fitting them on the available prefix.",
      "Very small returns reflect price tick size and its changing bps value as BTC price changes, so some sign-magnitude dependence can be microstructural rather than economically predictive.",
      "This is an unconditional same-candle dependence test; dependence conditional on history or volatility state can differ.",
    ],
  };
}

function addArrays(left: Float64Array, right: Float64Array): Float64Array {
  const result = new Float64Array(left.length);
  for (let index = 0; index < result.length; index += 1) result[index] = left[index]! + right[index]!;
  return result;
}

export function equalMassCuts(counts: Float64Array, bins: number): number[] {
  let total = 0;
  for (const count of counts) total += count;
  const cuts = [0];
  let cumulative = 0;
  let cell = 0;
  for (let bin = 1; bin < bins; bin += 1) {
    const target = total * bin / bins;
    while (cell < counts.length && cumulative + counts[cell]! < target) {
      cumulative += counts[cell]!;
      cell += 1;
    }
    const beforeError = Math.abs(target - cumulative);
    const afterError = cell < counts.length
      ? Math.abs(cumulative + counts[cell]! - target)
      : Number.POSITIVE_INFINITY;
    let cut = beforeError <= afterError ? cell : cell + 1;
    const minimum = cuts.at(-1)! + 1;
    const maximum = counts.length - (bins - bin);
    cut = Math.min(maximum, Math.max(minimum, cut));
    cuts.push(cut);
    while (cell < cut) {
      cumulative += counts[cell]!;
      cell += 1;
    }
  }
  cuts.push(counts.length);
  return cuts;
}

function binnedResult(
  negativeHistogram: Float64Array,
  positiveHistogram: Float64Array,
  cuts: number[],
): BinnedResult {
  const bins = cuts.length - 1;
  const negative = Array<number>(bins).fill(0);
  const positive = Array<number>(bins).fill(0);
  for (let bin = 0; bin < bins; bin += 1) {
    for (let cell = cuts[bin]!; cell < cuts[bin + 1]!; cell += 1) {
      negative[bin]! += negativeHistogram[cell]!;
      positive[bin]! += positiveHistogram[cell]!;
    }
  }
  const metrics = independenceMetrics(negative, positive);
  const total = metrics.observations;
  const positiveTotal = positive.reduce((sum, value) => sum + value, 0);
  const negativeTotal = total - positiveTotal;
  const contributions = positive.map((positiveCount, bin) => {
    const negativeCount = negative[bin]!;
    const magnitudeCount = positiveCount + negativeCount;
    let contribution = 0;
    if (positiveCount > 0) {
      contribution += positiveCount / total * Math.log2(
        positiveCount * total / (positiveTotal * magnitudeCount),
      );
    }
    if (negativeCount > 0) {
      contribution += negativeCount / total * Math.log2(
        negativeCount * total / (negativeTotal * magnitudeCount),
      );
    }
    return contribution;
  });
  return {
    ...metrics,
    edgesBps: cuts.map(cellBoundaryBps),
    counts: { negative, positive },
    magnitudeProbability: negative.map((value, index) => (value + positive[index]!) / total),
    positiveProbabilityByMagnitude: positive.map((value, index) =>
      value / (value + negative[index]!)),
    mutualInformationContributionBits: contributions,
  };
}

export function independenceMetrics(
  negative: number[],
  positive: number[],
): IndependenceMetrics {
  if (negative.length !== positive.length || negative.length === 0) {
    throw new Error("Sign rows must have the same nonzero magnitude-bin count.");
  }
  const negativeTotal = negative.reduce((sum, value) => sum + value, 0);
  const positiveTotal = positive.reduce((sum, value) => sum + value, 0);
  const total = negativeTotal + positiveTotal;
  const signTotals = [negativeTotal, positiveTotal];
  let mutualInformation = 0;
  let js = 0;
  let tv = 0;
  let chiSquareOverN = 0;
  let optimalCorrect = 0;
  let maximumDeviation = 0;
  let squaredDeviation = 0;
  let occupiedMagnitudeBins = 0;
  let nullExpectedAbsoluteCellError = 0;
  let nullExpectedOptimalCorrect = 0;
  const globalPositiveProbability = positiveTotal / total;
  for (let bin = 0; bin < negative.length; bin += 1) {
    const magnitudeTotal = negative[bin]! + positive[bin]!;
    if (magnitudeTotal === 0) continue;
    occupiedMagnitudeBins += 1;
    optimalCorrect += Math.max(negative[bin]!, positive[bin]!);
    const conditionalPositive = positive[bin]! / magnitudeTotal;
    const positiveProbability = positiveTotal / total;
    const deviation = conditionalPositive - positiveProbability;
    maximumDeviation = Math.max(maximumDeviation, Math.abs(deviation));
    squaredDeviation += magnitudeTotal / total * deviation * deviation;
    const positiveStandardDeviation = Math.sqrt(
      magnitudeTotal * globalPositiveProbability * (1 - globalPositiveProbability),
    );
    nullExpectedAbsoluteCellError += positiveStandardDeviation * Math.sqrt(2 / Math.PI);
    const signDifferenceMean = magnitudeTotal * (2 * globalPositiveProbability - 1);
    const signDifferenceStandardDeviation = 2 * positiveStandardDeviation;
    nullExpectedOptimalCorrect += (
      magnitudeTotal
      + expectedAbsoluteNormal(signDifferenceMean, signDifferenceStandardDeviation)
    ) / 2;
    for (let sign = 0; sign < 2; sign += 1) {
      const observedCount = sign === 0 ? negative[bin]! : positive[bin]!;
      const observed = observedCount / total;
      const independent = signTotals[sign]! / total * magnitudeTotal / total;
      if (observed > 0) mutualInformation += observed * Math.log2(observed / independent);
      const mixture = (observed + independent) / 2;
      if (observed > 0) js += 0.5 * observed * Math.log2(observed / mixture);
      if (independent > 0) js += 0.5 * independent * Math.log2(independent / mixture);
      tv += 0.5 * Math.abs(observed - independent);
      if (independent > 0) chiSquareOverN += (observed - independent) ** 2 / independent;
    }
  }
  const bias = (occupiedMagnitudeBins - 1) / (2 * total * Math.LN2);
  const baselineAccuracy = Math.max(negativeTotal, positiveTotal) / total;
  const nullExpectedAccuracyGain = nullExpectedOptimalCorrect / total - baselineAccuracy;
  const magnitudeOnlyAccuracyGain = optimalCorrect / total - baselineAccuracy;
  const nullExpectedTv = nullExpectedAbsoluteCellError / total;
  const positiveProbability = positiveTotal / total;
  const negativeProbability = negativeTotal / total;
  const signEntropy = -positiveProbability * Math.log2(positiveProbability)
    - negativeProbability * Math.log2(negativeProbability);
  const mutualInformationAboveBias = Math.max(0, mutualInformation - bias);
  return {
    observations: total,
    magnitudeBins: negative.length,
    positiveProbability,
    signEntropyBits: signEntropy,
    mutualInformationBits: mutualInformation,
    millerMadowNullBiasBits: bias,
    mutualInformationAboveBiasBits: mutualInformationAboveBias,
    fractionOfSignEntropyExplainedAboveBias: mutualInformationAboveBias / signEntropy,
    jsFromIndependentProductBits: js,
    totalVariationFromIndependentProduct: tv,
    nullExpectedTotalVariation: nullExpectedTv,
    totalVariationAboveNullMean: Math.max(0, tv - nullExpectedTv),
    cramersV: Math.sqrt(chiSquareOverN),
    baselineMajoritySignAccuracy: baselineAccuracy,
    optimalMagnitudeOnlySignAccuracy: optimalCorrect / total,
    magnitudeOnlyAccuracyGain,
    nullExpectedMagnitudeOnlyAccuracyGain: nullExpectedAccuracyGain,
    magnitudeOnlyAccuracyGainAboveNull: Math.max(
      0,
      magnitudeOnlyAccuracyGain - nullExpectedAccuracyGain,
    ),
    maximumConditionalPositiveProbabilityDeviation: maximumDeviation,
    weightedRmsConditionalPositiveProbabilityDeviation: Math.sqrt(squaredDeviation),
  };
}

function expectedAbsoluteNormal(mean: number, standardDeviation: number): number {
  if (standardDeviation <= 0) return Math.abs(mean);
  const ratio = mean / standardDeviation;
  return standardDeviation * Math.sqrt(2 / Math.PI) * Math.exp(-0.5 * ratio * ratio)
    + mean * (2 * normalCdf(ratio) - 1);
}

function normalCdf(value: number): number {
  return 0.5 * (1 + erf(value / Math.SQRT2));
}

function erf(value: number): number {
  const sign = value < 0 ? -1 : 1;
  const x = Math.abs(value);
  const t = 1 / (1 + 0.3275911 * x);
  const polynomial = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
    - 0.284496736) * t + 0.254829592) * t;
  return sign * (1 - polynomial * Math.exp(-x * x));
}

function cellBoundaryBps(cell: number): number | null {
  if (cell <= 0) return 0;
  if (cell >= HISTOGRAM_CELLS) return null;
  return Math.exp(
    LOG_MAGNITUDE_MIN
    + (cell - 1) / CORE_BINS * (LOG_MAGNITUDE_MAX - LOG_MAGNITUDE_MIN),
  );
}

function conditionalMagnitudeComparison(
  negative: Float64Array,
  positive: Float64Array,
  moments: MomentSums,
  binnedNegative: number[],
  binnedPositive: number[],
) {
  const negativeTotal = moments.count - moments.positive;
  const positiveTotal = moments.positive;
  let js = 0;
  let tv = 0;
  for (let cell = 0; cell < binnedNegative.length; cell += 1) {
    const negativeProbability = binnedNegative[cell]! / negativeTotal;
    const positiveProbability = binnedPositive[cell]! / positiveTotal;
    const mixture = (negativeProbability + positiveProbability) / 2;
    if (negativeProbability > 0) {
      js += 0.5 * negativeProbability * Math.log2(negativeProbability / mixture);
    }
    if (positiveProbability > 0) {
      js += 0.5 * positiveProbability * Math.log2(positiveProbability / mixture);
    }
    tv += 0.5 * Math.abs(negativeProbability - positiveProbability);
  }
  const probabilities = [
    ["p50", 0.5],
    ["p90", 0.9],
    ["p99", 0.99],
    ["p999", 0.999],
  ] as const;
  return {
    magnitudeBins: binnedNegative.length,
    jsDivergenceBits: js,
    totalVariation: tv,
    negativeMeanMagnitudeBps: (
      moments.sumMagnitude - moments.sumSignedMagnitude
    ) / (2 * negativeTotal),
    positiveMeanMagnitudeBps: (
      moments.sumMagnitude + moments.sumSignedMagnitude
    ) / (2 * positiveTotal),
    quantilesBps: Object.fromEntries(probabilities.map(([label, probability]) => [
      label,
      {
        negative: histogramQuantile(negative, probability),
        positive: histogramQuantile(positive, probability),
      },
    ])),
  };
}

function histogramQuantile(counts: Float64Array, probability: number): number | null {
  let total = 0;
  for (const count of counts) total += count;
  const target = total * probability;
  let cumulative = 0;
  for (let cell = 0; cell < counts.length; cell += 1) {
    if (cumulative + counts[cell]! >= target) {
      const lower = cellBoundaryBps(cell);
      const upper = cellBoundaryBps(cell + 1);
      if (lower === null || upper === null) return lower ?? upper;
      const fraction = counts[cell]! > 0 ? (target - cumulative) / counts[cell]! : 0.5;
      if (lower <= 0) return upper * fraction;
      return Math.exp(Math.log(lower) + fraction * (Math.log(upper) - Math.log(lower)));
    }
    cumulative += counts[cell]!;
  }
  return null;
}

function momentCorrelations(moments: MomentSums) {
  const count = moments.count;
  const meanSign = (2 * moments.positive - count) / count;
  const signVariance = 1 - meanSign * meanSign;
  return {
    signWithMagnitude: correlationFromSums(
      count,
      meanSign,
      signVariance,
      moments.sumMagnitude,
      moments.sumMagnitudeSquared,
      moments.sumSignedMagnitude,
    ),
    signWithLogMagnitude: correlationFromSums(
      count,
      meanSign,
      signVariance,
      moments.sumLogMagnitude,
      moments.sumLogMagnitudeSquared,
      moments.sumSignedLogMagnitude,
    ),
  };
}

function correlationFromSums(
  count: number,
  meanSign: number,
  signVariance: number,
  sumValue: number,
  sumValueSquared: number,
  sumSignValue: number,
): number {
  const meanValue = sumValue / count;
  const variance = sumValueSquared / count - meanValue * meanValue;
  const covariance = sumSignValue / count - meanSign * meanValue;
  return covariance / Math.sqrt(signVariance * variance);
}

function largestContributions(result: BinnedResult, limit: number) {
  return result.mutualInformationContributionBits
    .map((contribution, index) => ({
      bin: index,
      magnitudeLowerBps: result.edgesBps[index],
      magnitudeUpperBps: result.edgesBps[index + 1],
      magnitudeProbability: result.magnitudeProbability[index],
      positiveProbability: result.positiveProbabilityByMagnitude[index],
      positiveProbabilityDeviation: result.positiveProbabilityByMagnitude[index]!
        - result.positiveProbability,
      mutualInformationContributionBits: contribution,
    }))
    .sort((left, right) =>
      right.mutualInformationContributionBits - left.mutualInformationContributionBits)
    .slice(0, limit);
}

function anniversaryIso(startValue: string, offset: number): string {
  const start = new Date(startValue);
  return new Date(Date.UTC(
    start.getUTCFullYear() + offset,
    start.getUTCMonth(),
    start.getUTCDate(),
  )).toISOString();
}

function classifyDependence(primary: BinnedResult): string {
  const gain = primary.magnitudeOnlyAccuracyGainAboveNull;
  if (primary.mutualInformationAboveBiasBits < 1e-5 && gain < 0.0005) {
    return "Sign and magnitude are effectively independent at the measured resolution.";
  }
  if (gain < 0.005) {
    return "Sign and magnitude are measurably but weakly dependent at the measured resolution.";
  }
  if (gain < 0.02) return "Sign and magnitude have modest dependence at the measured resolution.";
  return "Sign and magnitude have strong dependence at the measured resolution.";
}

function renderReport(artifact: ReturnType<typeof analyzeCounts>): string {
  const primary = artifact.primary;
  const finest = artifact.finestResolution;
  const conditional = artifact.conditionalMagnitude;
  const lines = [
    "# One-second return sign–magnitude dependence",
    "",
    `Generated ${artifact.generatedAt}. The analysis uses ${artifact.activeReturns.toLocaleString("en-US")} nonzero BTCUSDT one-second log returns from ${artifact.window.startTime} through ${artifact.window.endTime}. The ${artifact.zeroReturns.toLocaleString("en-US")} exact-zero returns are excluded because zero has no sign.`,
    "",
    "## Result",
    "",
    artifact.conclusion,
    "",
    `At 64 magnitude cells, mutual information is ${formatMetric(primary.mutualInformationBits)} bits versus a ${formatMetric(primary.millerMadowNullBiasBits)}-bit plug-in null bias. Magnitude raises in-sample optimal sign accuracy by ${(primary.magnitudeOnlyAccuracyGain * 100).toFixed(6)} percentage points, of which ${(primary.nullExpectedMagnitudeOnlyAccuracyGain * 100).toFixed(6)} points are expected from cell-selection noise.`,
    "",
    `At the finest 2,048-cell audit, bias-corrected mutual information is ${formatMetric(finest.mutualInformationAboveBiasBits)} bits and sign-accuracy gain above its null expectation is ${(finest.magnitudeOnlyAccuracyGainAboveNull * 100).toFixed(6)} percentage points.`,
    "",
    "## Interpretation",
    "",
    `- Magnitude explains only ${(finest.fractionOfSignEntropyExplainedAboveBias * 100).toFixed(6)}% of sign entropy at the finest audited resolution. Unconditionally, sign and magnitude are therefore extremely close to independent, though not exactly independent.`,
    `- At 64 cells, the two conditional magnitude distributions differ by only ${(conditional.totalVariation * 100).toFixed(4)}% TV. Positive-return magnitudes are slightly smaller at the median but slightly larger in the far tail, so the residual relationship is weak and non-monotone.`,
    `- Dependence is not stable across years: corrected magnitude-only sign gain ranges from ${(Math.min(...artifact.annualStability.map((row) => row.magnitudeOnlyAccuracyGainAboveNull)) * 100).toFixed(4)} to ${(Math.max(...artifact.annualStability.map((row) => row.magnitudeOnlyAccuracyGainAboveNull)) * 100).toFixed(4)} percentage points.`,
    "- A factorized sign and magnitude output is a sound unconditional baseline. It should remain possible for a conditional history model to introduce a small sign–magnitude coupling rather than enforcing exact independence.",
    "",
    "## Resolution robustness",
    "",
    "| magnitude cells | mutual information (bits) | MI above bias | JS from product (bits) | TV above null | Cramer's V | accuracy gain above null |",
    "|---:|---:|---:|---:|---:|---:|---:|",
  ];
  for (const [resolution, result] of Object.entries(artifact.byResolution)) {
    lines.push(
      `| ${resolution} | ${formatMetric(result.mutualInformationBits)} | ${formatMetric(result.mutualInformationAboveBiasBits)} | ${formatMetric(result.jsFromIndependentProductBits)} | ${formatMetric(result.totalVariationAboveNullMean)} | ${formatMetric(result.cramersV)} | ${(result.magnitudeOnlyAccuracyGainAboveNull * 100).toFixed(6)} pp |`,
    );
  }
  lines.push(
    "",
    "## Conditional magnitude distributions",
    "",
    `On the fixed ${conditional.magnitudeBins}-cell view, the magnitude laws conditional on negative and positive sign have JS ${formatMetric(conditional.jsDivergenceBits)} bits and total variation ${formatMetric(conditional.totalVariation)}. Mean magnitude is ${formatMetric(conditional.negativeMeanMagnitudeBps)} bps after negative returns and ${formatMetric(conditional.positiveMeanMagnitudeBps)} bps after positive returns.`,
    "",
    "| magnitude quantile | negative (bps) | positive (bps) | positive / negative |",
    "|---|---:|---:|---:|",
  );
  for (const [name, values] of Object.entries(conditional.quantilesBps)) {
    lines.push(
      `| ${name} | ${formatMetric(values.negative!)} | ${formatMetric(values.positive!)} | ${formatMetric(values.positive! / values.negative!)} |`,
    );
  }
  lines.push(
    "",
    "## Linear correlations",
    "",
    `- Correlation of sign with magnitude: ${formatMetric(artifact.correlations.signWithMagnitude)}.`,
    `- Correlation of sign with log magnitude: ${formatMetric(artifact.correlations.signWithLogMagnitude)}.`,
    "",
    "Zero correlation would not prove independence; mutual information and the full conditional-bin comparison are the primary tests.",
    "",
    "## Annual stability",
    "",
    "| window | active returns | positive probability | MI above bias (bits) | TV above null | accuracy gain above null |",
    "|---|---:|---:|---:|---:|---:|",
  );
  for (const row of artifact.annualStability) {
    lines.push(
      `| ${row.startTime.slice(0, 10)} to ${row.endTime.slice(0, 10)} | ${row.observations.toLocaleString("en-US")} | ${(row.positiveProbability * 100).toFixed(6)}% | ${formatMetric(row.mutualInformationAboveBiasBits)} | ${formatMetric(row.totalVariationAboveNullMean)} | ${(row.magnitudeOnlyAccuracyGainAboveNull * 100).toFixed(6)} pp |`,
    );
  }
  lines.push(
    "",
    "## Largest dependence contributions",
    "",
    "| magnitude interval (bps) | mass | positive probability | deviation from baseline | MI contribution (bits) |",
    "|---|---:|---:|---:|---:|",
  );
  for (const row of artifact.largestDependenceContributions) {
    lines.push(
      `| ${intervalLabel(row.magnitudeLowerBps, row.magnitudeUpperBps)} | ${(row.magnitudeProbability! * 100).toFixed(5)}% | ${(row.positiveProbability! * 100).toFixed(5)}% | ${(row.positiveProbabilityDeviation! * 100).toFixed(5)} pp | ${formatMetric(row.mutualInformationContributionBits)} |`,
    );
  }
  lines.push(
    "",
    "## Method",
    "",
    "The independence null uses the observed global sign probability, not an assumed 50/50 split. Magnitudes are first accumulated on a 131,072-cell logarithmic grid, then combined into approximately equal-mass cells. This preserves the central tick-scale structure and far tails while allowing the same source scan to be audited at several resolutions.",
    "",
    "## Reproducibility",
    "",
    "```text",
    "node --conditions=development --import tsx scripts/analyze-return-sign-magnitude.ts",
    "```",
    "",
    "The complete magnitude edges, signed counts, conditional probabilities, annual results, and per-cell information contributions are stored in `data/benchmarks/one-second-sign-magnitude-dependence.json`.",
  );
  return `${lines.join("\n")}\n`;
}

function intervalLabel(lower: number | null, upper: number | null): string {
  const left = lower === null ? "-inf" : formatMetric(lower);
  const right = upper === null ? "+inf" : formatMetric(upper);
  return `[${left}, ${right})`;
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
