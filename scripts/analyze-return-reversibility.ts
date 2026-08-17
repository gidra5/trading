import fs from "node:fs";
import path from "node:path";
import { deserialize, serialize } from "node:v8";
import { fileURLToPath, pathToFileURL } from "node:url";
import { readCandleShardReferenceSync } from "@trading/storage";

const DAY_MS = 86_400_000;
const PRIMARY_ACTIVE_STATES = 64;
const PRIMARY_STATE_COUNT = PRIMARY_ACTIVE_STATES + 1;
const COARSE_ACTIVE_STATES = 16;
const COARSE_STATE_COUNT = COARSE_ACTIVE_STATES + 1;
const PRIMARY_LAGS = [1, 2, 5, 10, 30, 60, 300, 900] as const;
const COARSE_BLOCK_LENGTHS = [2, 3, 4, 5] as const;
const DEFAULT_ANALYSIS = "data/benchmarks/log-return-distributions.json";
const DEFAULT_KNOTS = "data/benchmarks/one-second-return-64-knot-fits.json";
const DEFAULT_CACHE = "data/runtime-cache/one-second-return-reversibility-counts.bin";
const DEFAULT_OUTPUT = "data/benchmarks/one-second-return-reversibility.json";
const DEFAULT_REPORT = "docs/experiments/one-second-return-reversibility-2026-08-16.md";

interface Options {
  analysisPath: string;
  knotsPath: string;
  cachePath: string;
  outputPath: string;
  reportPath: string;
  rebuildCache: boolean;
}

interface AnalysisReport {
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

interface DensityFit {
  knotsUnit: number[];
  knotDensityHeights: number[];
}

interface KnotArtifact {
  generatedAt: string;
  transform: {
    alpha: number;
    locationBps: number;
    scaleBps: number;
    [key: string]: unknown;
  };
  fits: { joint: DensityFit };
}

interface Transform {
  alpha: number;
  locationBps: number;
  scaleBps: number;
}

interface StreamCounts {
  primaryByLag: Record<string, Float64Array>;
  coarseByLength: Record<string, Float64Array>;
  annualPrimary: Float64Array[];
  observations: number;
  activeObservations: number;
  firstState: number;
  lastState: number;
}

interface ReversalMetrics {
  observations: number;
  occupiedReversalOrbits: number;
  jsDivergenceBits: number;
  totalVariation: number;
  optimalArrowClassifierAccuracy: number;
  maximumProbabilityFlux: number;
  jeffreysSmoothedEntropyProductionBits: number;
}

interface NullMetric {
  mean: number;
  standardDeviation: number;
  approximateP95: number;
}

interface OrientationNull {
  method: string;
  jsDivergenceBits: NullMetric;
  totalVariation: NullMetric;
}

interface StateDefinition {
  id: number;
  kind: string;
  label: string;
  unitLower?: number;
  unitUpper?: number;
  returnLowerBps: number | null;
  returnUpperBps: number | null;
}

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function main(): void {
  const options = parseOptions(process.argv.slice(2));
  const analysis = readJson<AnalysisReport>(options.analysisPath);
  const knotArtifact = readJson<KnotArtifact>(options.knotsPath);
  const transform: Transform = {
    alpha: knotArtifact.transform.alpha,
    locationBps: knotArtifact.transform.locationBps,
    scaleBps: knotArtifact.transform.scaleBps,
  };
  const primaryKnots = knotArtifact.fits.joint.knotsUnit;
  if (primaryKnots.length !== PRIMARY_ACTIVE_STATES) {
    throw new Error(`Expected ${PRIMARY_ACTIVE_STATES} primary knots, found ${primaryKnots.length}.`);
  }
  const primaryBoundaries = primaryKnots.slice(0, -1)
    .map((value, index) => (value + primaryKnots[index + 1]!) / 2);
  const coarseEdges = fittedQuantileEdges(knotArtifact.fits.joint, COARSE_ACTIVE_STATES);
  const files = selectedFiles(
    path.resolve(repoRoot, analysis.source.oneSecond.referenceDirectory),
    Date.parse(analysis.fullHistory.startTime),
    Date.parse(analysis.fullHistory.endTime),
  );
  const metadata = JSON.stringify({
    version: 1,
    knotGeneratedAt: knotArtifact.generatedAt,
    transform,
    files: files.map((entry) => path.basename(entry.file)),
    primaryKnots,
    coarseEdges,
    lags: PRIMARY_LAGS,
    blockLengths: COARSE_BLOCK_LENGTHS,
  });
  const counts = loadOrBuildCounts(
    options,
    metadata,
    files,
    Date.parse(analysis.fullHistory.startTime),
    transform,
    primaryBoundaries,
    coarseEdges,
  );
  const artifact = analyzeCounts(
    analysis,
    knotArtifact,
    transform,
    primaryBoundaries,
    coarseEdges,
    counts,
  );
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
    knotsPath: path.resolve(repoRoot, values.get("knots") ?? DEFAULT_KNOTS),
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

function fittedQuantileEdges(fit: DensityFit, bins: number): number[] {
  const knots = fit.knotsUnit;
  const heights = fit.knotDensityHeights;
  if (knots.length !== heights.length) throw new Error("Knot and height counts differ.");
  const areas = knots.slice(0, -1).map((value, index) =>
    (knots[index + 1]! - value) * (heights[index]! + heights[index + 1]!) / 2);
  const prefix = [0];
  for (const area of areas) prefix.push(prefix.at(-1)! + area);
  const totalArea = prefix.at(-1)!;
  const edges = Array<number>(bins + 1).fill(0);
  edges[bins] = 1;
  for (let edgeIndex = 1; edgeIndex < bins; edgeIndex += 1) {
    const target = edgeIndex / bins * totalArea;
    let interval = 0;
    while (interval + 1 < prefix.length && prefix[interval + 1]! <= target) interval += 1;
    interval = Math.min(interval, knots.length - 2);
    const left = knots[interval]!;
    const width = knots[interval + 1]! - left;
    const leftHeight = heights[interval]!;
    const slope = (heights[interval + 1]! - leftHeight) / width;
    const targetArea = target - prefix[interval]!;
    let distance: number;
    if (Math.abs(slope) < 1e-14) distance = targetArea / leftHeight;
    else {
      const discriminant = Math.max(0, leftHeight * leftHeight + 2 * slope * targetArea);
      const denominator = leftHeight + Math.sqrt(discriminant);
      distance = denominator === 0 ? width / 2 : 2 * targetArea / denominator;
    }
    edges[edgeIndex] = left + Math.min(width, Math.max(0, distance));
  }
  if (edges.some((value, index) => index > 0 && value <= edges[index - 1]!)) {
    throw new Error("Fitted quantile edges are not strictly increasing.");
  }
  return edges;
}

function loadOrBuildCounts(
  options: Options,
  metadata: string,
  files: Array<{ file: string; dayStart: number }>,
  analysisStart: number,
  transform: Transform,
  primaryBoundaries: number[],
  coarseEdges: number[],
): StreamCounts {
  if (!options.rebuildCache && fs.existsSync(options.cachePath)) {
    const cached = deserialize(fs.readFileSync(options.cachePath)) as {
      metadata: string;
      counts: StreamCounts;
    };
    if (cached.metadata === metadata) {
      console.log(`Loading cached counts from ${path.relative(repoRoot, options.cachePath)}`);
      return cached.counts;
    }
  }
  const counts = streamCounts(
    files,
    analysisStart,
    transform,
    primaryBoundaries,
    coarseEdges,
  );
  fs.mkdirSync(path.dirname(options.cachePath), { recursive: true });
  fs.writeFileSync(options.cachePath, serialize({ metadata, counts }));
  return counts;
}

function streamCounts(
  files: Array<{ file: string; dayStart: number }>,
  analysisStart: number,
  transform: Transform,
  primaryBoundaries: number[],
  coarseEdges: number[],
): StreamCounts {
  const primaryByLag = Object.fromEntries(PRIMARY_LAGS.map((lag) => [
    String(lag),
    new Float64Array(PRIMARY_STATE_COUNT ** 2),
  ])) as Record<string, Float64Array>;
  const coarseByLength = Object.fromEntries(COARSE_BLOCK_LENGTHS.map((length) => [
    String(length),
    new Float64Array(COARSE_STATE_COUNT ** length),
  ])) as Record<string, Float64Array>;
  const annualPrimary: Float64Array[] = [];
  let primaryCarry = new Uint8Array();
  let coarseCarry = new Uint8Array();
  let annualPrevious = -1;
  let annualIndex = -1;
  let previousClose = Number.NaN;
  let observations = 0;
  let activeObservations = 0;
  let firstState = -1;
  let lastState = -1;
  const coarseBoundaries = coarseEdges.slice(1, -1);

  for (const [fileIndex, entry] of files.entries()) {
    if (fileIndex % 100 === 0) {
      console.error(`Reading one-second reversibility ${fileIndex}/${files.length}...`);
    }
    const candles = readCandleShardReferenceSync(entry.file);
    if (candles.length !== 86_400) throw new Error(`${path.basename(entry.file)} is incomplete.`);
    const skip = Number.isFinite(previousClose) ? 0 : 1;
    const primary = new Uint8Array(candles.length - skip);
    const coarse = new Uint8Array(candles.length - skip);
    let cursor = 0;
    for (const candle of candles) {
      if (!Number.isFinite(candle.close) || candle.close <= 0) {
        throw new Error(`${path.basename(entry.file)} contains an invalid close.`);
      }
      if (Number.isFinite(previousClose)) {
        if (candle.close === previousClose) {
          primary[cursor] = 0;
          coarse[cursor] = 0;
        } else {
          const returnBps = Math.log(candle.close / previousClose) * 10_000;
          const unit = transformReturn(returnBps, transform);
          primary[cursor] = 1 + upperBound(primaryBoundaries, unit);
          coarse[cursor] = 1 + upperBound(coarseBoundaries, unit);
          activeObservations += 1;
        }
        cursor += 1;
      }
      previousClose = candle.close;
    }
    if (cursor !== primary.length) throw new Error("Return cursor is inconsistent.");
    if (firstState < 0) firstState = primary[0]!;
    lastState = primary.at(-1)!;
    observations += primary.length;
    primaryCarry = addLagCounts(primaryByLag, primary, primaryCarry, PRIMARY_STATE_COUNT);
    coarseCarry = addBlockCounts(coarseByLength, coarse, coarseCarry, COARSE_STATE_COUNT);

    const nextAnnualIndex = anniversaryIndex(analysisStart, entry.dayStart);
    while (annualPrimary.length <= nextAnnualIndex) {
      annualPrimary.push(new Float64Array(PRIMARY_STATE_COUNT ** 2));
    }
    if (nextAnnualIndex !== annualIndex) {
      annualIndex = nextAnnualIndex;
      annualPrevious = -1;
    }
    const annual = annualPrimary[annualIndex]!;
    if (annualPrevious >= 0) annual[annualPrevious * PRIMARY_STATE_COUNT + primary[0]!] += 1;
    for (let index = 1; index < primary.length; index += 1) {
      annual[primary[index - 1]! * PRIMARY_STATE_COUNT + primary[index]!] += 1;
    }
    annualPrevious = primary.at(-1)!;
  }
  return {
    primaryByLag,
    coarseByLength,
    annualPrimary,
    observations,
    activeObservations,
    firstState,
    lastState,
  };
}

function transformReturn(returnBps: number, transform: Transform): number {
  const latent = transform.alpha * Math.asinh(
    (returnBps - transform.locationBps) / transform.scaleBps,
  );
  if (latent >= 0) return 1 / (1 + Math.exp(-latent));
  const exponential = Math.exp(latent);
  return exponential / (1 + exponential);
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

export function addLagCounts(
  destination: Record<string, Float64Array>,
  values: Uint8Array,
  carry: Uint8Array,
  stateCount: number,
): Uint8Array {
  const combined = concatenate(carry, values);
  for (const lag of PRIMARY_LAGS) {
    const counts = destination[String(lag)];
    if (!counts) continue;
    const start = Math.max(carry.length, lag);
    for (let index = start; index < combined.length; index += 1) {
      counts[combined[index - lag]! * stateCount + combined[index]!] += 1;
    }
  }
  return combined.slice(-Math.min(Math.max(...PRIMARY_LAGS), combined.length));
}

export function addBlockCounts(
  destination: Record<string, Float64Array>,
  values: Uint8Array,
  carry: Uint8Array,
  stateCount: number,
): Uint8Array {
  const combined = concatenate(carry, values);
  for (const length of COARSE_BLOCK_LENGTHS) {
    const counts = destination[String(length)];
    if (!counts) continue;
    const endStart = Math.max(carry.length, length - 1);
    for (let end = endStart; end < combined.length; end += 1) {
      let code = combined[end - length + 1]!;
      for (let offset = end - length + 2; offset <= end; offset += 1) {
        code = code * stateCount + combined[offset]!;
      }
      counts[code] += 1;
    }
  }
  const keep = Math.min(Math.max(...COARSE_BLOCK_LENGTHS) - 1, combined.length);
  return combined.slice(-keep);
}

function concatenate(left: Uint8Array, right: Uint8Array): Uint8Array {
  const result = new Uint8Array(left.length + right.length);
  result.set(left);
  result.set(right, left.length);
  return result;
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

export function reverseCodeIndices(stateCount: number, length: number): Int32Array {
  const size = stateCount ** length;
  const result = new Int32Array(size);
  for (let source = 0; source < size; source += 1) {
    let remainder = source;
    let reversed = 0;
    for (let index = 0; index < length; index += 1) {
      reversed = reversed * stateCount + remainder % stateCount;
      remainder = Math.floor(remainder / stateCount);
    }
    result[source] = reversed;
  }
  return result;
}

export function reversalMetrics(
  counts: Float64Array,
  reverseIndices: Int32Array,
): ReversalMetrics {
  let total = 0;
  for (const count of counts) total += count;
  let occupiedReversalOrbits = 0;
  let jsCount = 0;
  let absoluteDifference = 0;
  let maximumDifference = 0;
  let eprCount = 0;
  for (let index = 0; index < counts.length; index += 1) {
    const reverse = reverseIndices[index]!;
    if (index >= reverse) continue;
    const left = counts[index]!;
    const right = counts[reverse]!;
    const orbitTotal = left + right;
    if (orbitTotal === 0) continue;
    occupiedReversalOrbits += 1;
    if (left > 0) jsCount += left * Math.log2(2 * left / orbitTotal);
    if (right > 0) jsCount += right * Math.log2(2 * right / orbitTotal);
    const difference = Math.abs(left - right);
    absoluteDifference += difference;
    maximumDifference = Math.max(maximumDifference, difference);
    const smoothedLeft = left + 0.5;
    const smoothedRight = right + 0.5;
    eprCount += (smoothedLeft - smoothedRight) * Math.log2(smoothedLeft / smoothedRight);
  }
  const totalVariation = absoluteDifference / total;
  return {
    observations: total,
    occupiedReversalOrbits,
    jsDivergenceBits: jsCount / total,
    totalVariation,
    optimalArrowClassifierAccuracy: (1 + totalVariation) / 2,
    maximumProbabilityFlux: maximumDifference / total,
    jeffreysSmoothedEntropyProductionBits: eprCount / total,
  };
}

const orientationMomentCache = new Map<number, {
  jsMean: number;
  jsVariance: number;
  tvMean: number;
  tvVariance: number;
}>();

function orientationNull(counts: Float64Array, reverseIndices: Int32Array): OrientationNull {
  let total = 0;
  for (const count of counts) total += count;
  let jsMeanCount = 0;
  let jsVarianceCount = 0;
  let tvMeanCount = 0;
  let tvVarianceCount = 0;
  for (let index = 0; index < counts.length; index += 1) {
    const reverse = reverseIndices[index]!;
    if (index >= reverse) continue;
    const orbitTotal = counts[index]! + counts[reverse]!;
    if (orbitTotal === 0) continue;
    const moments = orientationMoments(orbitTotal);
    jsMeanCount += moments.jsMean;
    jsVarianceCount += moments.jsVariance;
    tvMeanCount += moments.tvMean;
    tvVarianceCount += moments.tvVariance;
  }
  const jsMean = jsMeanCount / total;
  const jsStandardDeviation = Math.sqrt(jsVarianceCount) / total;
  const tvMean = tvMeanCount / total;
  const tvStandardDeviation = Math.sqrt(tvVarianceCount) / total;
  return {
    method: "conditional fair-binomial orientation null per block/reverse-block orbit; exact moments through orbit count 256 and normal/chi-square moments above it",
    jsDivergenceBits: {
      mean: jsMean,
      standardDeviation: jsStandardDeviation,
      approximateP95: jsMean + 1.6448536269514722 * jsStandardDeviation,
    },
    totalVariation: {
      mean: tvMean,
      standardDeviation: tvStandardDeviation,
      approximateP95: tvMean + 1.6448536269514722 * tvStandardDeviation,
    },
  };
}

function orientationMoments(total: number) {
  const cached = orientationMomentCache.get(total);
  if (cached) return cached;
  let result: { jsMean: number; jsVariance: number; tvMean: number; tvVariance: number };
  if (total > 256) {
    result = {
      jsMean: 1 / (2 * Math.LN2),
      jsVariance: 1 / (2 * Math.LN2 ** 2),
      tvMean: Math.sqrt(2 * total / Math.PI),
      tvVariance: total * (1 - 2 / Math.PI),
    };
  } else {
    let probability = 2 ** -total;
    let probabilitySum = 0;
    let jsMean = 0;
    let jsSecond = 0;
    let tvMean = 0;
    let tvSecond = 0;
    for (let left = 0; left <= total; left += 1) {
      const right = total - left;
      let js = 0;
      if (left > 0) js += left * Math.log2(2 * left / total);
      if (right > 0) js += right * Math.log2(2 * right / total);
      const tv = Math.abs(left - right);
      probabilitySum += probability;
      jsMean += probability * js;
      jsSecond += probability * js * js;
      tvMean += probability * tv;
      tvSecond += probability * tv * tv;
      if (left < total) probability *= (total - left) / (left + 1);
    }
    jsMean /= probabilitySum;
    jsSecond /= probabilitySum;
    tvMean /= probabilitySum;
    tvSecond /= probabilitySum;
    result = {
      jsMean,
      jsVariance: Math.max(0, jsSecond - jsMean * jsMean),
      tvMean,
      tvVariance: Math.max(0, tvSecond - tvMean * tvMean),
    };
  }
  orientationMomentCache.set(total, result);
  return result;
}

function analyzeCounts(
  analysis: AnalysisReport,
  knotArtifact: KnotArtifact,
  transform: Transform,
  primaryBoundaries: number[],
  coarseEdges: number[],
  counts: StreamCounts,
) {
  const primaryReverse = reverseCodeIndices(PRIMARY_STATE_COUNT, 2);
  const pairwiseByLagSeconds = Object.fromEntries(PRIMARY_LAGS.map((lag) => {
    const values = counts.primaryByLag[String(lag)]!;
    const metrics = reversalMetrics(values, primaryReverse);
    const nullResult = orientationNull(values, primaryReverse);
    return [String(lag), {
      ...metrics,
      orientationNull: nullResult,
      jsAboveNullMeanBits: Math.max(
        0,
        metrics.jsDivergenceBits - nullResult.jsDivergenceBits.mean,
      ),
      totalVariationAboveNullMean: Math.max(
        0,
        metrics.totalVariation - nullResult.totalVariation.mean,
      ),
    }];
  }));
  const higherOrderConsecutiveBlocks = Object.fromEntries(COARSE_BLOCK_LENGTHS.map((length) => {
    const values = counts.coarseByLength[String(length)]!;
    const reverse = reverseCodeIndices(COARSE_STATE_COUNT, length);
    const metrics = reversalMetrics(values, reverse);
    const nullResult = orientationNull(values, reverse);
    return [String(length), {
      ...metrics,
      orientationNull: nullResult,
      jsAboveNullMeanBits: Math.max(
        0,
        metrics.jsDivergenceBits - nullResult.jsDivergenceBits.mean,
      ),
      totalVariationAboveNullMean: Math.max(
        0,
        metrics.totalVariation - nullResult.totalVariation.mean,
      ),
    }];
  }));
  const annualOneStepStability = counts.annualPrimary.map((values, index) => ({
    index,
    startTime: anniversaryIso(analysis.fullHistory.startTime, index),
    endTime: anniversaryIso(analysis.fullHistory.startTime, index + 1),
    ...reversalMetrics(values, primaryReverse),
  }));
  const states = primaryStateDefinitions(primaryBoundaries, transform);
  const oneStepCounts = counts.primaryByLag["1"]!;
  const activeOnlyCounts = activeOnlyPairCounts(oneStepCounts, PRIMARY_STATE_COUNT);
  const exactOneZeroCounts = exactOneZeroPairCounts(oneStepCounts, PRIMARY_STATE_COUNT);
  const activeOnlyReverse = reverseCodeIndices(PRIMARY_ACTIVE_STATES, 2);
  const activeOnlyMetrics = reversalMetrics(activeOnlyCounts, activeOnlyReverse);
  const activeOnlyNull = orientationNull(activeOnlyCounts, activeOnlyReverse);
  const exactOneZeroMetrics = reversalMetrics(exactOneZeroCounts, primaryReverse);
  const exactOneZeroNull = orientationNull(exactOneZeroCounts, primaryReverse);
  const oneStep = pairwiseByLagSeconds["1"] as ReversalMetrics & {
    orientationNull: OrientationNull;
  };
  const conclusion = classifyReversibility(
    oneStep.totalVariation,
    oneStep.jsDivergenceBits,
    oneStep.orientationNull.jsDivergenceBits.mean,
  );
  return {
    version: 1,
    generatedAt: new Date().toISOString(),
    symbol: analysis.source.symbol,
    scale: "1s",
    window: analysis.fullHistory,
    source: {
      analysis: DEFAULT_ANALYSIS,
      knotFit: DEFAULT_KNOTS,
      referenceDirectory: analysis.source.oneSecond.referenceDirectory,
    },
    observations: counts.observations,
    activeObservations: counts.activeObservations,
    zeroObservations: counts.observations - counts.activeObservations,
    zeroProbability: 1 - counts.activeObservations / counts.observations,
    definitions: {
      reversibility: "Every finite ordered return block has the same distribution as its reversed block.",
      pairwiseDetailedBalance: "pi(i) F(i,j) = pi(j) F(j,i).",
      jsDivergenceBits: "JS(P(block), P(reverse(block))) in bits; zero means reversal symmetry at the measured resolution.",
      totalVariation: "Maximum event-probability difference between forward and reversed block laws.",
      optimalArrowClassifierAccuracy: "Bayes accuracy for classifying whether an equally likely block was shown forward or backward; (1+TV)/2.",
      entropyProduction: "Jeffreys-0.5-smoothed KL(P(block)||P(reverse(block))) in bits per block.",
    },
    primaryRepresentation: {
      description: "64 nearest optimized transformed-return knots plus a separate exact-zero state",
      stateCount: PRIMARY_STATE_COUNT,
      occupiedStateCount: occupiedOriginStates(oneStepCounts, PRIMARY_STATE_COUNT),
      transform: knotArtifact.transform,
      states,
    },
    higherOrderRepresentation: {
      description: "16 approximately equal-mass active bins under the fitted joint density plus exact zero",
      stateCount: COARSE_STATE_COUNT,
      activeUnitEdges: coarseEdges,
    },
    pairwiseByLagSeconds,
    higherOrderConsecutiveBlocks,
    annualOneStepStability,
    oneStepDecomposition: {
      bothReturnsActive: {
        ...activeOnlyMetrics,
        shareOfAllTransitions: activeOnlyMetrics.observations / oneStep.observations,
        orientationNull: activeOnlyNull,
        jsAboveNullMeanBits: Math.max(
          0,
          activeOnlyMetrics.jsDivergenceBits - activeOnlyNull.jsDivergenceBits.mean,
        ),
        totalVariationAboveNullMean: Math.max(
          0,
          activeOnlyMetrics.totalVariation - activeOnlyNull.totalVariation.mean,
        ),
      },
      exactlyOneReturnIsZero: {
        ...exactOneZeroMetrics,
        shareOfAllTransitions: exactOneZeroMetrics.observations / oneStep.observations,
        orientationNull: exactOneZeroNull,
        jsAboveNullMeanBits: Math.max(
          0,
          exactOneZeroMetrics.jsDivergenceBits - exactOneZeroNull.jsDivergenceBits.mean,
        ),
        totalVariationAboveNullMean: Math.max(
          0,
          exactOneZeroMetrics.totalVariation - exactOneZeroNull.totalVariation.mean,
        ),
      },
    },
    oneStepConditionalDistribution: transitionPayload(oneStepCounts, PRIMARY_STATE_COUNT),
    largestOneStepProbabilityFluxes: topFluxes(oneStepCounts, PRIMARY_STATE_COUNT, states),
    conclusion,
    limitations: [
      "Reversibility is evaluated after finite-state discretization; asymmetry below a bin's resolution is not visible.",
      "The conditional-binomial null does not preserve every dependency between overlapping blocks; annual stability is a complementary robustness check.",
      "Pairwise detailed balance is necessary but not sufficient for full process reversibility, so consecutive blocks through length five are also tested.",
      "This is a descriptive full-history analysis, not a chronological forecast evaluation.",
    ],
  };
}

function activeOnlyPairCounts(counts: Float64Array, stateCount: number): Float64Array {
  const result = new Float64Array((stateCount - 1) ** 2);
  for (let from = 1; from < stateCount; from += 1) {
    for (let to = 1; to < stateCount; to += 1) {
      result[(from - 1) * (stateCount - 1) + to - 1] = counts[from * stateCount + to]!;
    }
  }
  return result;
}

function exactOneZeroPairCounts(counts: Float64Array, stateCount: number): Float64Array {
  const result = new Float64Array(counts.length);
  for (let state = 1; state < stateCount; state += 1) {
    result[state] = counts[state]!;
    result[state * stateCount] = counts[state * stateCount]!;
  }
  return result;
}

function occupiedOriginStates(counts: Float64Array, stateCount: number): number {
  let occupied = 0;
  for (let from = 0; from < stateCount; from += 1) {
    let total = 0;
    for (let to = 0; to < stateCount; to += 1) total += counts[from * stateCount + to]!;
    if (total > 0) occupied += 1;
  }
  return occupied;
}

function transitionPayload(counts: Float64Array, stateCount: number) {
  const rows = new Float64Array(stateCount);
  const columns = new Float64Array(stateCount);
  let total = 0;
  for (let from = 0; from < stateCount; from += 1) {
    for (let to = 0; to < stateCount; to += 1) {
      const count = counts[from * stateCount + to]!;
      rows[from] += count;
      columns[to] += count;
      total += count;
    }
  }
  const forward = Array.from({ length: stateCount }, (_, from) =>
    Array.from({ length: stateCount }, (_, to) =>
      rows[from]! > 0 ? counts[from * stateCount + to]! / rows[from]! : 0));
  const backward = Array.from({ length: stateCount }, (_, current) =>
    Array.from({ length: stateCount }, (_, previous) =>
      columns[current]! > 0 ? counts[previous * stateCount + current]! / columns[current]! : 0));
  const originProbability = Array.from(rows, (value) => value / total);
  const destinationProbability = Array.from(columns, (value) => value / total);
  let marginalTv = 0;
  let weightedConditionalTv = 0;
  for (let state = 0; state < stateCount; state += 1) {
    marginalTv += Math.abs(originProbability[state]! - destinationProbability[state]!);
    let conditionalTv = 0;
    for (let neighbor = 0; neighbor < stateCount; neighbor += 1) {
      conditionalTv += Math.abs(forward[state]![neighbor]! - backward[state]![neighbor]!);
    }
    weightedConditionalTv += (originProbability[state]! + destinationProbability[state]!) / 2
      * conditionalTv / 2;
  }
  return {
    originStateProbability: originProbability,
    destinationStateProbability: destinationProbability,
    stationarityMarginalTotalVariation: marginalTv / 2,
    stationaryWeightedConditionalTotalVariation: weightedConditionalTv,
    forwardNextGivenCurrent: forward,
    backwardPreviousGivenCurrent: backward,
  };
}

function primaryStateDefinitions(boundaries: number[], transform: Transform): StateDefinition[] {
  const unitEdges = [0, ...boundaries, 1];
  const result: StateDefinition[] = [{
    id: 0,
    kind: "exactZero",
    label: "exact zero",
    returnLowerBps: 0,
    returnUpperBps: 0,
  }];
  for (let index = 0; index < PRIMARY_ACTIVE_STATES; index += 1) {
    const lower = inverseTransform(unitEdges[index]!, transform);
    const upper = inverseTransform(unitEdges[index + 1]!, transform);
    result.push({
      id: index + 1,
      kind: "activeVoronoi",
      label: intervalLabel(lower, upper),
      unitLower: unitEdges[index],
      unitUpper: unitEdges[index + 1],
      returnLowerBps: lower,
      returnUpperBps: upper,
    });
  }
  return result;
}

function inverseTransform(unit: number, transform: Transform): number | null {
  if (unit <= 0 || unit >= 1) return null;
  const logit = Math.log(unit) - Math.log1p(-unit);
  return transform.locationBps + transform.scaleBps * Math.sinh(logit / transform.alpha);
}

function intervalLabel(lower: number | null, upper: number | null): string {
  const left = lower === null ? "-inf" : formatCompact(lower);
  const right = upper === null ? "+inf" : formatCompact(upper);
  return `[${left}, ${right}) bps excluding exact zero`;
}

function formatCompact(value: number): string {
  return Number(value.toPrecision(6)).toString();
}

function topFluxes(
  counts: Float64Array,
  stateCount: number,
  states: StateDefinition[],
  limit = 12,
) {
  let total = 0;
  for (const count of counts) total += count;
  const candidates: Array<{ absolute: number; left: number; right: number }> = [];
  for (let left = 0; left < stateCount; left += 1) {
    for (let right = left + 1; right < stateCount; right += 1) {
      const net = counts[left * stateCount + right]! - counts[right * stateCount + left]!;
      candidates.push({ absolute: Math.abs(net), left, right });
    }
  }
  candidates.sort((left, right) => right.absolute - left.absolute);
  return candidates.slice(0, limit).map((candidate) => {
    let { left, right } = candidate;
    let forwardCount = counts[left * stateCount + right]!;
    let reverseCount = counts[right * stateCount + left]!;
    if (forwardCount < reverseCount) {
      [left, right] = [right, left];
      [forwardCount, reverseCount] = [reverseCount, forwardCount];
    }
    return {
      fromState: left,
      toState: right,
      fromLabel: states[left]!.label,
      toLabel: states[right]!.label,
      forwardCount,
      reverseCount,
      netCount: forwardCount - reverseCount,
      netProbabilityPerTransition: (forwardCount - reverseCount) / total,
    };
  });
}

function anniversaryIso(startValue: string, offset: number): string {
  const start = new Date(startValue);
  return new Date(Date.UTC(
    start.getUTCFullYear() + offset,
    start.getUTCMonth(),
    start.getUTCDate(),
  )).toISOString();
}

function classifyReversibility(tv: number, js: number, nullJs: number): string {
  const ratio = js / Math.max(nullJs, 1e-30);
  if (tv < 0.002 && ratio < 3) {
    return "No practically resolved one-step arrow of time above the finite-sample orientation floor.";
  }
  if (tv < 0.01) {
    return "Statistically resolved but small one-step time asymmetry at the 65-state resolution.";
  }
  if (tv < 0.05) {
    return "Clearly resolved but practically modest one-step time asymmetry at the 65-state resolution.";
  }
  return "Large one-step time asymmetry at the 65-state resolution.";
}

function renderReport(artifact: ReturnType<typeof analyzeCounts>): string {
  const oneStep = artifact.pairwiseByLagSeconds["1"]!;
  const sixtySecond = artifact.pairwiseByLagSeconds["60"]!;
  const lengthFive = artifact.higherOrderConsecutiveBlocks["5"]!;
  const active = artifact.oneStepDecomposition.bothReturnsActive;
  const zeroBoundary = artifact.oneStepDecomposition.exactlyOneReturnIsZero;
  const activeJsContribution = active.shareOfAllTransitions * active.jsDivergenceBits;
  const zeroJsContribution = zeroBoundary.shareOfAllTransitions * zeroBoundary.jsDivergenceBits;
  const activeJsShare = activeJsContribution / (activeJsContribution + zeroJsContribution);
  const lines = [
    "# One-second log-return time reversibility",
    "",
    `Generated ${artifact.generatedAt}. This analysis covers ${artifact.observations.toLocaleString("en-US")} adjacent BTCUSDT one-second log returns from ${artifact.window.startTime} through ${artifact.window.endTime}. Exact zero is a separate state with probability ${(artifact.zeroProbability * 100).toFixed(6)}%.`,
    "",
    "## Result",
    "",
    artifact.conclusion,
    "",
    "Forward and backward marginals being identical is only stationarity. Reversibility additionally requires every ordered block distribution to equal its reversed distribution.",
    "",
    "## Interpretation",
    "",
    `- At one second, reversal JS is ${formatMetric(oneStep.jsDivergenceBits)} bits, ${(oneStep.jsDivergenceBits / oneStep.orientationNull.jsDivergenceBits.mean).toFixed(1)} times the orientation-null mean. Total variation above its null mean is ${(oneStep.totalVariationAboveNullMean * 100).toFixed(4)}%, while the raw optimal arrow classifier reaches only ${(oneStep.optimalArrowClassifierAccuracy * 100).toFixed(4)}%. The arrow is unambiguous statistically but modest as predictive information.`,
    `- The effect decays sharply: at 60 seconds, TV above null is only ${(sixtySecond.totalVariationAboveNullMean * 100).toFixed(4)}% and arrow classification is ${(sixtySecond.optimalArrowClassifierAccuracy * 100).toFixed(4)}%. The measurable arrow is predominantly immediate market microstructure.`,
    `- Histories accumulate additional direction information. A five-return block reaches ${(lengthFive.optimalArrowClassifierAccuracy * 100).toFixed(4)}% raw arrow accuracy; its TV is ${(lengthFive.totalVariation * 100).toFixed(4)}% versus a ${(lengthFive.orientationNull.totalVariation.mean * 100).toFixed(4)}% sampling floor.`,
    `- Both-active transitions contribute about ${(activeJsShare * 100).toFixed(2)}% of one-step reversal JS. Therefore the effect is not merely the zero-gap process, although exact-zero boundaries also carry directional information.`,
    "- The largest currents involve very small returns, zero transitions, and immediate sign reversals. That pattern is consistent with tick-size/price-grid mechanics and bid-ask bounce; it should not be interpreted directly as tradable directional return correlation.",
    "",
    "## Pairwise detailed balance by lag",
    "",
    "The primary representation has 64 optimized transformed-return cells plus the exact-zero state.",
    "",
    "| lag | JS forward vs reverse (bits) | null JS mean | JS above null | total variation | TV above null | arrow classifier | entropy production (bits) |",
    "|---:|---:|---:|---:|---:|---:|---:|---:|",
  ];
  for (const [lag, metrics] of Object.entries(artifact.pairwiseByLagSeconds)) {
    lines.push(
      `| ${lag}s | ${formatMetric(metrics.jsDivergenceBits)} | ${formatMetric(metrics.orientationNull.jsDivergenceBits.mean)} | ${formatMetric(metrics.jsAboveNullMeanBits)} | ${formatMetric(metrics.totalVariation)} | ${formatMetric(metrics.totalVariationAboveNullMean)} | ${(metrics.optimalArrowClassifierAccuracy * 100).toFixed(6)}% | ${formatMetric(metrics.jeffreysSmoothedEntropyProductionBits)} |`,
    );
  }
  lines.push(
    "",
    "`arrow classifier` is the best possible accuracy from the discretized block alone when forward and reversed orientations are equally likely. Chance is 50%.",
    "",
    "## Consecutive history reversal",
    "",
    "Longer histories use 16 approximately equal-mass active-return cells plus zero.",
    "",
    "| block length | horizon | JS (bits) | null JS mean | JS above null | total variation | TV above null | arrow classifier |",
    "|---:|---:|---:|---:|---:|---:|---:|---:|",
  );
  for (const [length, metrics] of Object.entries(artifact.higherOrderConsecutiveBlocks)) {
    lines.push(
      `| ${length} | ${Number(length) - 1}s | ${formatMetric(metrics.jsDivergenceBits)} | ${formatMetric(metrics.orientationNull.jsDivergenceBits.mean)} | ${formatMetric(metrics.jsAboveNullMeanBits)} | ${formatMetric(metrics.totalVariation)} | ${formatMetric(metrics.totalVariationAboveNullMean)} | ${(metrics.optimalArrowClassifierAccuracy * 100).toFixed(6)}% |`,
    );
  }
  lines.push(
    "",
    "## One-step decomposition",
    "",
    "| subset | share of transitions | JS (bits) | JS above null | total variation | TV above null | arrow classifier |",
    "|---|---:|---:|---:|---:|---:|---:|",
  );
  for (const [name, metrics] of Object.entries(artifact.oneStepDecomposition)) {
    lines.push(
      `| ${name} | ${(metrics.shareOfAllTransitions * 100).toFixed(6)}% | ${formatMetric(metrics.jsDivergenceBits)} | ${formatMetric(metrics.jsAboveNullMeanBits)} | ${formatMetric(metrics.totalVariation)} | ${formatMetric(metrics.totalVariationAboveNullMean)} | ${(metrics.optimalArrowClassifierAccuracy * 100).toFixed(6)}% |`,
    );
  }
  lines.push(
    "",
    "## Annual stability of the one-step result",
    "",
    "| window | transitions | JS (bits) | total variation | arrow classifier |",
    "|---|---:|---:|---:|---:|",
  );
  for (const row of artifact.annualOneStepStability) {
    lines.push(
      `| ${row.startTime.slice(0, 10)} to ${row.endTime.slice(0, 10)} | ${row.observations.toLocaleString("en-US")} | ${formatMetric(row.jsDivergenceBits)} | ${formatMetric(row.totalVariation)} | ${(row.optimalArrowClassifierAccuracy * 100).toFixed(6)}% |`,
    );
  }
  lines.push(
    "",
    "## Largest one-step probability currents",
    "",
    "Positive current means the first direction occurs more often than its exact reverse.",
    "",
    "| from | to | forward count | reverse count | net probability |",
    "|---|---|---:|---:|---:|",
  );
  for (const row of artifact.largestOneStepProbabilityFluxes) {
    lines.push(
      `| ${row.fromLabel} | ${row.toLabel} | ${row.forwardCount.toLocaleString("en-US")} | ${row.reverseCount.toLocaleString("en-US")} | ${formatMetric(row.netProbabilityPerTransition)} |`,
    );
  }
  lines.push(
    "",
    "## Conditional kernels",
    "",
    "The machine-readable artifact contains the full 65 by 65 forward kernel `P(next | current)` and Bayes-reversed kernel `P(previous | current)`, along with state boundaries. Their difference is the measured detailed-balance violation, not a violation of Bayes' theorem.",
    "",
    "## Methodological notes",
    "",
    "- JS and total variation remain finite when a block orientation has no observed reverse. Entropy production uses a Jeffreys 0.5 count to avoid infinite plug-in KL.",
    "- The orientation null fixes each block/reverse-block orbit count and assigns its directions with probability one half. Moments are exact for orbit totals through 256 and use their asymptotic normal/chi-square forms above that.",
    "- Longer-block scores naturally have a larger sampling floor; `JS above null` subtracts the measured orientation-null mean.",
    "- Pairwise symmetry does not prove full reversibility. The length-three through length-five tests look for higher-order arrows of time.",
    "",
    "## Reproducibility",
    "",
    "```text",
    "node --conditions=development --import tsx scripts/analyze-return-reversibility.ts",
    "```",
    "",
    "The full conditional matrices, state definitions, null summaries, annual results, and probability currents are stored in `data/benchmarks/one-second-return-reversibility.json`.",
  );
  return `${lines.join("\n")}\n`;
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
