import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { readCandleShardReferenceSync } from "@trading/storage";
import {
  circularConvolutionPower,
  compoundCircularConvolution,
} from "./lib/discrete-convolution.js";
import { stationaryRenewalCountDistribution } from "./lib/renewal-process.js";

const FFT_SIZE = 131_072;
const HORIZON_SECONDS = 60;
const DAY_MS = 86_400_000;

interface Histogram {
  observations: number;
  binWidthBps: number;
  lowerBps: number;
  upperBps: number;
  binCount: number;
  nonzeroBins: Array<[number, number]>;
}

interface HistogramWindow {
  id: string;
  zeroProbability: number;
  histogram: Histogram;
}

interface HistogramReport {
  version: number;
  symbol: string;
  commonEndTime: string;
  scales: Array<{ id: string; windows: HistogramWindow[] }>;
}

interface GapReport {
  version: number;
  symbol: string;
  windows: Array<{
    id: string;
    nonzeroReturns: number;
    gaps: number;
    pmf: Array<[seconds: number, count: number, probability: number]>;
  }>;
}

interface AnalysisReport {
  version: number;
  source: {
    symbol: string;
    oneSecond: { referenceDirectory: string };
  };
  fullHistory: { startTime: string; endTime: string };
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
  const histograms = readJson<HistogramReport>(
    path.resolve(repoRoot, args.get("histograms") ?? "data/benchmarks/log-return-histograms.json"),
  );
  const gaps = readJson<GapReport>(
    path.resolve(repoRoot, args.get("gaps") ?? "data/benchmarks/zero-change-gaps.json"),
  );
  const analysis = readJson<AnalysisReport>(
    path.resolve(repoRoot, args.get("analysis") ?? "data/benchmarks/log-return-distributions.json"),
  );
  const outputPath = path.resolve(
    repoRoot,
    args.get("output") ?? "data/benchmarks/delay-aware-minute-return-distribution.json",
  );
  if (histograms.version !== 1 || gaps.version !== 1 || analysis.version < 2
    || histograms.symbol !== gaps.symbol || gaps.symbol !== analysis.source.symbol) {
    throw new Error("Inputs are not matching return histogram, delay, and analysis reports.");
  }

  const oneSecondWindow = fullWindow(histograms, "1s");
  const oneMinuteWindow = fullWindow(histograms, "1m");
  const gapWindow = gaps.windows.find((window) => window.id === "full");
  if (!gapWindow) throw new Error("The delay report has no full-history window.");
  const oneSecondLattice = histogramLattice(oneSecondWindow.histogram);
  const markLattice = conditionalNonzeroMarkLattice(oneSecondWindow, oneSecondLattice);
  const interarrival = interarrivalDistribution(gapWindow);
  const renewalCounts = stationaryRenewalCountDistribution(interarrival, HORIZON_SECONDS);
  const binomialCounts = binomialDistribution(
    HORIZON_SECONDS,
    1 - oneSecondWindow.zeroProbability,
  );
  const actualCountResult = scanActualMinuteActivity(analysis);

  const iidSeconds = rebin(
    circularConvolutionPower(oneSecondLattice, HORIZON_SECONDS),
    oneSecondWindow.histogram.binWidthBps,
    oneMinuteWindow.histogram,
  );
  const renewalDelay = rebin(
    compoundCircularConvolution(markLattice, renewalCounts),
    oneSecondWindow.histogram.binWidthBps,
    oneMinuteWindow.histogram,
  );
  const actualActivity = rebin(
    compoundCircularConvolution(markLattice, actualCountResult.probabilities),
    oneSecondWindow.histogram.binWidthBps,
    oneMinuteWindow.histogram,
  );
  const actual = denseProbabilities(oneMinuteWindow.histogram);
  const sigmaBps = oneMinuteWindow.histogram.binWidthBps / 0.1;
  const models = [
    modelSummary("iidSeconds", "IID one-second returns", iidSeconds, actual, oneMinuteWindow.histogram, sigmaBps),
    modelSummary("renewalDelay", "IID empirical interarrival delays + IID nonzero marks", renewalDelay, actual, oneMinuteWindow.histogram, sigmaBps),
    modelSummary("actualActivity", "Observed per-minute nonzero counts + IID nonzero marks", actualActivity, actual, oneMinuteWindow.histogram, sigmaBps),
  ];
  const report = {
    version: 1,
    generatedAt: new Date().toISOString(),
    symbol: histograms.symbol,
    commonEndTime: histograms.commonEndTime,
    question: "How much of the observed 1s-to-1m aggregation difference can the zero-change delay distribution explain?",
    methodology: {
      mark: "The exact-zero mass is removed from the empirical 1s marginal; the remainder is normalized as the nonzero-return mark distribution.",
      delay: "Positive zero-run lengths are combined with zero-length gaps between adjacent nonzero returns to form the full empirical interarrival distribution D in seconds.",
      renewal: "A stationary renewal process draws IID interarrival delays D. Each arrival draws an IID nonzero-return mark independent of D. The exact 60-second event-count law is mixed over convolution powers of the mark distribution.",
      actualActivityControl: "The observed distribution of nonzero-return counts in aligned 60-second windows replaces the renewal count law, while marks remain IID. This is an upper control for what activity counts alone can explain.",
      limitation: "Neither activity model preserves dependence between delay and return magnitude, sign dependence, volatility regimes, or ordering of nonzero marks.",
    },
    oneSecond: {
      zeroProbability: oneSecondWindow.zeroProbability,
      nonzeroProbability: 1 - oneSecondWindow.zeroProbability,
      interarrivalMeanSeconds: distributionMean(interarrival),
    },
    minuteActivityCounts: {
      actualWindows: actualCountResult.windows,
      iidBernoulli: countSummary(binomialCounts),
      renewalDelay: {
        ...countSummary(renewalCounts),
        jsDivergenceFromActualBits: jensenShannonBits(
          actualCountResult.probabilities,
          renewalCounts,
        ),
      },
      actual: {
        ...countSummary(actualCountResult.probabilities),
        jsDivergenceFromIidBernoulliBits: jensenShannonBits(
          actualCountResult.probabilities,
          binomialCounts,
        ),
      },
    },
    observedMinuteSigmaBps: sigmaBps,
    models,
    delayExplainedFractionOfIidJsDivergence: explainedFraction(
      models[0]!.jsDivergenceBits,
      models[1]!.jsDivergenceBits,
    ),
    activityCountExplainedFractionOfIidJsDivergence: explainedFraction(
      models[0]!.jsDivergenceBits,
      models[2]!.jsDivergenceBits,
    ),
  };
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
  console.log(outputPath);
  console.log(JSON.stringify({
    counts: report.minuteActivityCounts,
    models: models.map((model) => ({
      id: model.id,
      varianceRatio: model.varianceRatioObservedOverModel,
      centralRatio: model.centralMass.ratioObservedOverModel,
      threeSigmaRatio: model.threeSigmaTail.ratioObservedOverModel,
      fiveSigmaRatio: model.fiveSigmaTail.ratioObservedOverModel,
      jsBits: model.jsDivergenceBits,
    })),
    delayExplainedJs: report.delayExplainedFractionOfIidJsDivergence,
    activityExplainedJs: report.activityCountExplainedFractionOfIidJsDivergence,
  }, null, 2));
}

function interarrivalDistribution(
  gapWindow: GapReport["windows"][number],
): number[] {
  const transitionCount = gapWindow.nonzeroReturns - 1;
  const zeroGapCount = transitionCount - gapWindow.gaps;
  if (zeroGapCount < 0) throw new Error("Gap counts exceed nonzero-return transitions.");
  const maximumDelay = Math.max(1, ...gapWindow.pmf.map(([seconds]) => seconds + 1));
  const probabilities = Array.from({ length: maximumDelay + 1 }, () => 0);
  probabilities[1] = zeroGapCount / transitionCount;
  for (const [zeroSeconds, count] of gapWindow.pmf) {
    probabilities[zeroSeconds + 1]! += count / transitionCount;
  }
  assertProbabilityMass(probabilities, "interarrival");
  return probabilities;
}

function scanActualMinuteActivity(
  analysis: AnalysisReport,
): { windows: number; probabilities: number[] } {
  const directory = path.resolve(repoRoot, analysis.source.oneSecond.referenceDirectory);
  const start = Date.parse(analysis.fullHistory.startTime);
  const end = Date.parse(analysis.fullHistory.endTime);
  const files = fs.readdirSync(directory, { withFileTypes: true })
    .filter((entry) => entry.isFile() && /^\d{4}-\d{2}-\d{2}\.json$/.test(entry.name))
    .map((entry) => ({
      file: path.join(directory, entry.name),
      dayStart: Date.parse(`${entry.name.slice(0, 10)}T00:00:00.000Z`),
    }))
    .filter((entry) => entry.dayStart >= start && entry.dayStart + DAY_MS <= end)
    .sort((left, right) => left.dayStart - right.dayStart);
  const counts = Array.from({ length: HORIZON_SECONDS + 1 }, () => 0);
  let previousClose = Number.NaN;
  let previousTime = Number.NaN;
  let currentMinute = Number.NaN;
  let minuteReturns = 0;
  let minuteNonzero = 0;
  let windows = 0;
  const finishMinute = (): void => {
    if (minuteReturns === HORIZON_SECONDS) {
      counts[minuteNonzero]! += 1;
      windows += 1;
    }
  };
  for (const [fileIndex, entry] of files.entries()) {
    if (fileIndex % 100 === 0) console.error(`Scanning minute activity ${fileIndex}/${files.length}...`);
    const candles = readCandleShardReferenceSync(entry.file);
    if (candles.length !== 86_400) throw new Error(`${path.basename(entry.file)} is incomplete.`);
    for (const candle of candles) {
      const minute = Math.floor(candle.openTime / 60_000);
      if (minute !== currentMinute) {
        if (Number.isFinite(currentMinute)) finishMinute();
        currentMinute = minute;
        minuteReturns = 0;
        minuteNonzero = 0;
      }
      if (previousTime === candle.openTime - 1_000) {
        minuteReturns += 1;
        if (candle.close !== previousClose) minuteNonzero += 1;
      }
      previousClose = candle.close;
      previousTime = candle.openTime;
    }
  }
  finishMinute();
  if (windows === 0) throw new Error("No complete one-minute activity windows were found.");
  return { windows, probabilities: counts.map((count) => count / windows) };
}

function conditionalNonzeroMarkLattice(
  window: HistogramWindow,
  unconditional: Float64Array,
): Float64Array {
  const result = unconditional.slice();
  result[0] = result[0]! - window.zeroProbability;
  if (result[0]! < -1e-10) throw new Error("Exact-zero mass exceeds its histogram bin.");
  result[0] = Math.max(0, result[0]!);
  const nonzeroMass = 1 - window.zeroProbability;
  for (let index = 0; index < result.length; index += 1) result[index] /= nonzeroMass;
  assertProbabilityMass(result, "conditional nonzero mark");
  return result;
}

function histogramLattice(histogram: Histogram): Float64Array {
  const result = new Float64Array(FFT_SIZE);
  for (const [binIndex, probability] of histogram.nonzeroBins) {
    const midpoint = histogram.lowerBps + (binIndex + 0.5) * histogram.binWidthBps;
    const offset = Math.round(midpoint / histogram.binWidthBps);
    result[offset >= 0 ? offset : FFT_SIZE + offset] += probability;
  }
  assertProbabilityMass(result, "histogram lattice");
  return result;
}

function rebin(
  lattice: Float64Array,
  latticeWidthBps: number,
  target: Histogram,
): Float64Array {
  const result = new Float64Array(target.binCount);
  for (let index = 0; index < lattice.length; index += 1) {
    const signedOffset = index < lattice.length / 2 ? index : index - lattice.length;
    const valueBps = signedOffset * latticeWidthBps;
    const targetIndex = Math.floor((valueBps - target.lowerBps) / target.binWidthBps);
    if (targetIndex >= 0 && targetIndex < result.length) result[targetIndex] += lattice[index]!;
  }
  assertProbabilityMass(result, "rebinned model");
  return result;
}

function modelSummary(
  id: string,
  label: string,
  model: Float64Array,
  actual: Float64Array,
  histogram: Histogram,
  observedSigmaBps: number,
) {
  const moments = probabilityMoments(model, histogram);
  const observedCentral = intervalProbability(actual, histogram, observedSigmaBps, 0.25, false);
  const modelCentral = intervalProbability(model, histogram, observedSigmaBps, 0.25, false);
  const observedThree = intervalProbability(actual, histogram, observedSigmaBps, 3, true);
  const modelThree = intervalProbability(model, histogram, observedSigmaBps, 3, true);
  const observedFive = intervalProbability(actual, histogram, observedSigmaBps, 5, true);
  const modelFive = intervalProbability(model, histogram, observedSigmaBps, 5, true);
  return {
    id,
    label,
    sigmaBps: moments.sigmaBps,
    varianceRatioObservedOverModel: (observedSigmaBps / moments.sigmaBps) ** 2,
    jsDivergenceBits: jensenShannonBits(actual, model),
    centralMass: {
      observed: observedCentral,
      model: modelCentral,
      ratioObservedOverModel: observedCentral / modelCentral,
    },
    threeSigmaTail: {
      observed: observedThree,
      model: modelThree,
      ratioObservedOverModel: observedThree / modelThree,
    },
    fiveSigmaTail: {
      observed: observedFive,
      model: modelFive,
      ratioObservedOverModel: observedFive / modelFive,
    },
  };
}

function countSummary(probabilities: readonly number[]) {
  const mean = probabilities.reduce((sum, probability, count) => sum + probability * count, 0);
  const variance = probabilities.reduce(
    (sum, probability, count) => sum + probability * (count - mean) ** 2,
    0,
  );
  return { mean, variance, fanoFactor: variance / mean, zeroProbability: probabilities[0] ?? 0 };
}

function probabilityMoments(probabilities: Float64Array, histogram: Histogram) {
  let first = 0;
  let second = 0;
  for (let index = 0; index < probabilities.length; index += 1) {
    const center = histogram.lowerBps + (index + 0.5) * histogram.binWidthBps;
    first += probabilities[index]! * center;
    second += probabilities[index]! * center * center;
  }
  return { meanBps: first, sigmaBps: Math.sqrt(Math.max(0, second - first * first)) };
}

function intervalProbability(
  probabilities: Float64Array,
  histogram: Histogram,
  sigmaBps: number,
  thresholdSigma: number,
  tail: boolean,
): number {
  let result = 0;
  for (let index = 0; index < probabilities.length; index += 1) {
    const center = histogram.lowerBps + (index + 0.5) * histogram.binWidthBps;
    const selected = Math.abs(center / sigmaBps) >= thresholdSigma;
    if (tail ? selected : !selected) result += probabilities[index]!;
  }
  return result;
}

function binomialDistribution(trials: number, successProbability: number): number[] {
  const probabilities = Array.from({ length: trials + 1 }, () => 0);
  probabilities[0] = (1 - successProbability) ** trials;
  for (let successes = 1; successes <= trials; successes += 1) {
    probabilities[successes] = probabilities[successes - 1]!
      * (trials - successes + 1) / successes
      * successProbability / (1 - successProbability);
  }
  assertProbabilityMass(probabilities, "binomial count");
  return probabilities;
}

function denseProbabilities(histogram: Histogram): Float64Array {
  const result = new Float64Array(histogram.binCount);
  for (const [index, probability] of histogram.nonzeroBins) result[index] = probability;
  return result;
}

function jensenShannonBits(left: ArrayLike<number>, right: ArrayLike<number>): number {
  if (left.length !== right.length) throw new Error("JS distributions have different lengths.");
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

function distributionMean(probabilities: readonly number[]): number {
  return probabilities.reduce((sum, probability, value) => sum + probability * value, 0);
}

function explainedFraction(iidDivergence: number, modelDivergence: number): number {
  return (iidDivergence - modelDivergence) / iidDivergence;
}

function assertProbabilityMass(probabilities: ArrayLike<number>, label: string): void {
  let mass = 0;
  for (let index = 0; index < probabilities.length; index += 1) mass += probabilities[index]!;
  if (Math.abs(mass - 1) > 1e-8) throw new Error(`${label} has probability mass ${mass}.`);
}

function fullWindow(report: HistogramReport, scaleId: string): HistogramWindow {
  const window = report.scales.find((scale) => scale.id === scaleId)
    ?.windows.find((candidate) => candidate.id === "full");
  if (!window) throw new Error(`Histogram input is missing ${scaleId}/full.`);
  return window;
}

function readJson<T>(file: string): T {
  return JSON.parse(fs.readFileSync(file, "utf8")) as T;
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
