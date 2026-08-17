import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { readCandleShardReferenceSync } from "@trading/storage";
import { circularConvolutionPower } from "./lib/discrete-convolution.js";

const FFT_SIZE = 131_072;
const HORIZONS = [
  { id: "1m", seconds: 60, parentId: "1s", parentFactor: 60 },
  { id: "15m", seconds: 900, parentId: "1m", parentFactor: 15 },
  { id: "1h", seconds: 3_600, parentId: "15m", parentFactor: 4 },
  { id: "4h", seconds: 14_400, parentId: "1h", parentFactor: 4 },
  { id: "1d", seconds: 86_400, parentId: "4h", parentFactor: 6 },
] as const;

interface HistogramWindow {
  id: string;
  histogram: {
    observations: number;
    binWidthBps: number;
    lowerBps: number;
    upperBps: number;
    binCount: number;
    underflowProbability: number;
    overflowProbability: number;
    nonzeroBins: Array<[index: number, probability: number]>;
  };
}

interface HistogramReport {
  version: number;
  symbol: string;
  commonEndTime: string;
  scales: Array<{
    id: string;
    label: string;
    intervalMs: number;
    windows: HistogramWindow[];
  }>;
}

interface AnalysisReport {
  version: number;
  source: {
    symbol: string;
    oneSecond: { referenceDirectory: string };
    oneMinute: { referenceDirectory: string };
  };
}

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

try {
  main();
} catch (error: unknown) {
  console.error(error instanceof Error ? error.stack ?? error.message : String(error));
  process.exitCode = 1;
}

function main(): void {
  const args = parseArgs(process.argv.slice(2));
  const histogramPath = path.resolve(
    repoRoot,
    args.get("histograms") ?? "data/benchmarks/log-return-histograms.json",
  );
  const analysisPath = path.resolve(
    repoRoot,
    args.get("analysis") ?? "data/benchmarks/log-return-distributions.json",
  );
  const outputPath = path.resolve(
    repoRoot,
    args.get("output") ?? "data/benchmarks/return-aggregation-hierarchy.json",
  );
  const histograms = JSON.parse(fs.readFileSync(histogramPath, "utf8")) as HistogramReport;
  const analysis = JSON.parse(fs.readFileSync(analysisPath, "utf8")) as AnalysisReport;
  if (histograms.version !== 1 || histograms.symbol !== "BTCUSDT"
    || analysis.version < 2 || analysis.source.symbol !== histograms.symbol) {
    throw new Error("Inputs are not matching BTCUSDT return-distribution reports.");
  }

  const oneSecond = fullHistogram(histograms, "1s");
  const oneSecondSigmaBps = oneSecond.binWidthBps / 0.1;
  const base = buildFftLattice(oneSecond);
  const hierarchyValidation = validateCandleHierarchy(analysis, 12);
  const scales = HORIZONS.map((horizon) => {
    const actualHistogram = fullHistogram(histograms, horizon.id);
    const parentHistogram = fullHistogram(histograms, horizon.parentId);
    console.error(`Convolving 1s distribution ${horizon.seconds} times for ${horizon.id}...`);
    const oneSecondConvolved = circularConvolutionPower(base, horizon.seconds);
    const oneSecondIndependent = rebinConvolution(
      oneSecondConvolved,
      oneSecond.binWidthBps,
      actualHistogram,
    );
    const parentIndependent = horizon.parentId === "1s"
      ? oneSecondIndependent
      : rebinConvolution(
        circularConvolutionPower(buildFftLattice(parentHistogram), horizon.parentFactor),
        parentHistogram.binWidthBps,
        actualHistogram,
      );
    const actual = denseProbabilities(actualHistogram);
    const sigmaBps = actualHistogram.binWidthBps / 0.1;
    const firstObservedIndex = actualHistogram.nonzeroBins[0]?.[0];
    const lastObservedIndex = actualHistogram.nonzeroBins.at(-1)?.[0];
    if (firstObservedIndex === undefined || lastObservedIndex === undefined) {
      throw new Error(`${horizon.id} has no observed histogram bins.`);
    }
    const points = Array.from({ length: actualHistogram.binCount }, (_, index) => {
      const x = (actualHistogram.lowerBps + (index + 0.5) * actualHistogram.binWidthBps)
        / sigmaBps;
      const observed = actual[index]!;
      const independent1s = oneSecondIndependent.probabilities[index]!;
      const independentParent = parentIndependent.probabilities[index]!;
      const observedCount = observed * actualHistogram.observations;
      const expected1sCount = independent1s * actualHistogram.observations;
      const expectedParentCount = independentParent * actualHistogram.observations;
      return {
        index,
        x,
        observed,
        independent1s,
        independentParent,
        log10Ratio1s: observedCount >= 3 && expected1sCount >= 3
          ? Math.log10(observed / independent1s)
          : null,
        log10RatioParent: observedCount >= 3 && expectedParentCount >= 3
          ? Math.log10(observed / independentParent)
          : null,
      };
    }).filter((point) => point.index >= firstObservedIndex && point.index <= lastObservedIndex);
    const parentSigmaBps = parentHistogram.binWidthBps / 0.1;
    return {
      id: horizon.id,
      label: histograms.scales.find((scale) => scale.id === horizon.id)!.label,
      seconds: horizon.seconds,
      parentId: horizon.parentId,
      parentFactor: horizon.parentFactor,
      observations: actualHistogram.observations,
      sigmaBps: round(sigmaBps),
      oneSecondIndependent: comparisonSummary(
        actual,
        oneSecondIndependent,
        actualHistogram,
        sigmaBps,
        oneSecondSigmaBps * Math.sqrt(horizon.seconds),
      ),
      parentIndependent: comparisonSummary(
        actual,
        parentIndependent,
        actualHistogram,
        sigmaBps,
        parentSigmaBps * Math.sqrt(horizon.parentFactor),
      ),
      points: points.map((point) => [
        round(point.x),
        round(point.observed),
        round(point.independent1s),
        round(point.independentParent),
        point.log10Ratio1s === null ? null : round(point.log10Ratio1s),
        point.log10RatioParent === null ? null : round(point.log10RatioParent),
      ]),
    };
  });

  const report = {
    version: 1,
    generatedAt: new Date().toISOString(),
    symbol: histograms.symbol,
    commonEndTime: histograms.commonEndTime,
    methodology: {
      identity: "An aligned h-second log return is the sum of its h contiguous one-second log returns.",
      independentBaseline: "The empirical full-history 1s probability mass is convolved h times by FFT, which preserves the 1s marginal distribution and removes all serial dependence.",
      parentBaseline: "Each observed parent-scale marginal is also convolved by its exact child/parent factor. This preserves dependence already accumulated inside a parent block and isolates additional dependence between adjacent parent blocks.",
      comparisonGrid: "Both observed and IID-convolved probabilities are rebinned to each observed scale's 0.1-sigma bins.",
      pointColumns: ["returnSigma", "observed", "oneSecondIndependent", "parentIndependent", "log10ObservedOverOneSecondIndependent", "log10ObservedOverParentIndependent"],
      interpretation: "Observed-minus-IID differences contain the aggregate effect of serial dependence, volatility clustering, activity persistence, and temporal concentration of jumps; they do not identify those mechanisms separately.",
      fftSize: FFT_SIZE,
    },
    hierarchyValidation,
    oneSecond: {
      observations: oneSecond.observations,
      sigmaBps: round(oneSecondSigmaBps),
      binWidthBps: round(oneSecond.binWidthBps),
    },
    scales,
  };
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
  console.log(outputPath);
  for (const scale of scales) {
    console.log(
      `${scale.id}: variance ${scale.oneSecondIndependent.varianceRatio}x vs 1s IID, `
      + `3sigma tail ${scale.oneSecondIndependent.threeSigmaTail.ratio}x vs 1s IID; `
      + `parent ${scale.parentId} variance ${scale.parentIndependent.varianceRatio}x, `
      + `3sigma tail ${scale.parentIndependent.threeSigmaTail.ratio}x`,
    );
  }
}

function comparisonSummary(
  actual: Float64Array,
  independent: { probabilities: Float64Array; underflow: number; overflow: number },
  histogram: HistogramWindow["histogram"],
  observedSigmaBps: number,
  independentSigmaBps: number,
) {
  const actualCentral = centralProbability(actual, histogram, observedSigmaBps, 0.25);
  const independentCentral = centralProbability(
    independent.probabilities,
    histogram,
    observedSigmaBps,
    0.25,
  );
  const threeSigma = tailComparison(
    actual,
    independent.probabilities,
    histogram,
    observedSigmaBps,
    3,
  );
  const fiveSigma = tailComparison(
    actual,
    independent.probabilities,
    histogram,
    observedSigmaBps,
    5,
  );
  const varianceRatio = (observedSigmaBps / independentSigmaBps) ** 2;
  const rebinnedMoments = probabilityMoments(independent.probabilities, histogram);
  return {
    sigmaBps: round(independentSigmaBps),
    rebinnedSigmaBps: round(rebinnedMoments.sigmaBps),
    rebinnedSigmaRelativeError: round(
      rebinnedMoments.sigmaBps / independentSigmaBps - 1,
    ),
    varianceRatio: round(varianceRatio),
    integratedReturnCorrelation: round(varianceRatio - 1),
    jsDivergenceBits: round(jensenShannonBits(actual, independent.probabilities)),
    centralMass: {
      observed: round(actualCentral),
      independent: round(independentCentral),
      ratio: round(actualCentral / independentCentral),
    },
    threeSigmaTail: threeSigma,
    fiveSigmaTail: fiveSigma,
    circularMassOutsideTargetGrid: round(independent.underflow + independent.overflow),
  };
}

function probabilityMoments(
  probabilities: Float64Array,
  histogram: HistogramWindow["histogram"],
): { meanBps: number; sigmaBps: number } {
  let mass = 0;
  let first = 0;
  let second = 0;
  for (let index = 0; index < probabilities.length; index += 1) {
    const probability = probabilities[index]!;
    const center = histogram.lowerBps + (index + 0.5) * histogram.binWidthBps;
    mass += probability;
    first += probability * center;
    second += probability * center * center;
  }
  const meanBps = first / mass;
  return {
    meanBps,
    sigmaBps: Math.sqrt(Math.max(0, second / mass - meanBps * meanBps)),
  };
}

function tailComparison(
  actual: Float64Array,
  independent: Float64Array,
  histogram: HistogramWindow["histogram"],
  sigmaBps: number,
  thresholdSigma: number,
) {
  const observed = tailProbability(actual, histogram, sigmaBps, thresholdSigma);
  const baseline = tailProbability(independent, histogram, sigmaBps, thresholdSigma);
  return {
    observed: round(observed),
    independent: round(baseline),
    ratio: baseline > 0 ? round(observed / baseline) : null,
    observedCount: round(observed * histogram.observations),
    independentExpectedCount: round(baseline * histogram.observations),
  };
}

function fullHistogram(report: HistogramReport, scaleId: string): HistogramWindow["histogram"] {
  const scale = report.scales.find((candidate) => candidate.id === scaleId);
  const full = scale?.windows.find((window) => window.id === "full");
  if (!full) throw new Error(`Histogram report is missing ${scaleId}/full.`);
  return full.histogram;
}

function buildFftLattice(histogram: HistogramWindow["histogram"]): Float64Array {
  if (histogram.binCount >= FFT_SIZE / 2) {
    throw new Error(`FFT size ${FFT_SIZE} is too small for ${histogram.binCount} input bins.`);
  }
  const probabilities = new Float64Array(FFT_SIZE);
  for (const [binIndex, probability] of histogram.nonzeroBins) {
    const midpoint = histogram.lowerBps + (binIndex + 0.5) * histogram.binWidthBps;
    const offset = Math.round(midpoint / histogram.binWidthBps);
    const latticeIndex = offset >= 0 ? offset : FFT_SIZE + offset;
    probabilities[latticeIndex] += probability;
  }
  const mass = probabilities.reduce((sum, probability) => sum + probability, 0);
  if (Math.abs(mass - 1) > 1e-9) {
    throw new Error(`1s FFT lattice contains ${mass} probability mass instead of one.`);
  }
  return probabilities;
}

function rebinConvolution(
  convolution: Float64Array,
  latticeWidthBps: number,
  target: HistogramWindow["histogram"],
): { probabilities: Float64Array; underflow: number; overflow: number } {
  const probabilities = new Float64Array(target.binCount);
  let underflow = 0;
  let overflow = 0;
  for (let index = 0; index < convolution.length; index += 1) {
    const probability = convolution[index]!;
    if (probability <= 0) continue;
    const signedOffset = index < convolution.length / 2 ? index : index - convolution.length;
    const valueBps = signedOffset * latticeWidthBps;
    const targetIndex = Math.floor((valueBps - target.lowerBps) / target.binWidthBps);
    if (targetIndex < 0) underflow += probability;
    else if (targetIndex >= target.binCount) overflow += probability;
    else probabilities[targetIndex] += probability;
  }
  return { probabilities, underflow, overflow };
}

function denseProbabilities(histogram: HistogramWindow["histogram"]): Float64Array {
  const probabilities = new Float64Array(histogram.binCount);
  for (const [index, probability] of histogram.nonzeroBins) probabilities[index] = probability;
  return probabilities;
}

function centralProbability(
  probabilities: Float64Array,
  histogram: HistogramWindow["histogram"],
  sigmaBps: number,
  thresholdSigma: number,
): number {
  let sum = 0;
  for (let index = 0; index < probabilities.length; index += 1) {
    const center = histogram.lowerBps + (index + 0.5) * histogram.binWidthBps;
    if (Math.abs(center / sigmaBps) < thresholdSigma) sum += probabilities[index]!;
  }
  return sum;
}

function tailProbability(
  probabilities: Float64Array,
  histogram: HistogramWindow["histogram"],
  sigmaBps: number,
  thresholdSigma: number,
): number {
  let sum = 0;
  for (let index = 0; index < probabilities.length; index += 1) {
    const center = histogram.lowerBps + (index + 0.5) * histogram.binWidthBps;
    if (Math.abs(center / sigmaBps) >= thresholdSigma) sum += probabilities[index]!;
  }
  return sum;
}

function jensenShannonBits(left: Float64Array, right: Float64Array): number {
  let divergence = 0;
  for (let index = 0; index < left.length; index += 1) {
    const p = left[index]!;
    const q = right[index]!;
    const midpoint = (p + q) / 2;
    if (p > 0) divergence += 0.5 * p * Math.log(p / midpoint);
    if (q > 0) divergence += 0.5 * q * Math.log(q / midpoint);
  }
  return divergence / Math.log(2);
}

function validateCandleHierarchy(
  analysis: AnalysisReport,
  requestedDays: number,
): {
  sampledDays: number;
  minuteEndpoints: number;
  mismatchedEndpoints: number;
  maximumCloseDifferenceBps: number;
} {
  const oneSecondDirectory = path.resolve(repoRoot, analysis.source.oneSecond.referenceDirectory);
  const oneMinuteDirectory = path.resolve(repoRoot, analysis.source.oneMinute.referenceDirectory);
  const names = fs.readdirSync(oneSecondDirectory)
    .filter((name) => /^\d{4}-\d{2}-\d{2}\.json$/.test(name)
      && fs.existsSync(path.join(oneMinuteDirectory, name)))
    .sort();
  const selected = new Set(Array.from({ length: Math.min(requestedDays, names.length) }, (_, index) => (
    names[Math.round(index * (names.length - 1) / (Math.min(requestedDays, names.length) - 1))]!
  )));
  let minuteEndpoints = 0;
  let mismatchedEndpoints = 0;
  let maximumCloseDifferenceBps = 0;
  for (const name of selected) {
    const seconds = readCandleShardReferenceSync(path.join(oneSecondDirectory, name));
    const minutes = readCandleShardReferenceSync(path.join(oneMinuteDirectory, name));
    if (seconds.length !== 86_400 || minutes.length !== 1_440) {
      throw new Error(`${name} is incomplete in the hierarchy validation sample.`);
    }
    for (let minute = 0; minute < minutes.length; minute += 1) {
      const secondClose = seconds[(minute + 1) * 60 - 1]!.close;
      const minuteClose = minutes[minute]!.close;
      const differenceBps = Math.abs(Math.log(secondClose / minuteClose) * 10_000);
      maximumCloseDifferenceBps = Math.max(maximumCloseDifferenceBps, differenceBps);
      if (differenceBps > 1e-10) mismatchedEndpoints += 1;
      minuteEndpoints += 1;
    }
  }
  return {
    sampledDays: selected.size,
    minuteEndpoints,
    mismatchedEndpoints,
    maximumCloseDifferenceBps: round(maximumCloseDifferenceBps),
  };
}

function parseArgs(args: string[]): Map<string, string> {
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
  return values;
}

function round(value: number): number {
  return Number(value.toPrecision(10));
}
