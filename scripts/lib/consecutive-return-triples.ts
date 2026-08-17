import { STANDARDIZED_PAIR_EDGES } from "./consecutive-return-pairs.js";

export interface TripleSample {
  x: number[];
  y: number[];
  z: number[];
}

export interface TripleBasicMoments {
  observations: number;
  means: [number, number, number];
  standardDeviations: [number, number, number];
  correlations: [number, number, number];
}

export interface TripleMomentsSnapshot extends TripleBasicMoments {
  absoluteCorrelations: [number, number, number];
  squaredCorrelations: [number, number, number];
  allZeroFraction: number;
  anyZeroFraction: number;
  nonzeroAllSameSignFraction: number | null;
  signPatterns: number[];
  continuous: TripleBasicMoments;
}

export interface TripleShapeStatistics {
  sampleObservations: number;
  continuousSampleObservations: number;
  allThreeAbsoluteTail2Sigma: number | null;
  allThreeAbsoluteTail2SigmaLift: number | null;
  allThreeAbsoluteTail3Sigma: number | null;
  allThreeAbsoluteTail3SigmaLift: number | null;
  thirdAbsoluteReturnAfterFirstTwo2SigmaRatio: number | null;
  whitenedRadiusQuantiles: {
    p50: number | null;
    p90: number | null;
    p95: number | null;
    p99: number | null;
  };
  projectionHistograms: {
    firstSecond: number[];
    secondThird: number[];
    firstThird: number[];
  };
  visualizationSample: Array<[number, number, number]>;
}

export type TripleFamily =
  | "trivariate-gaussian"
  | "trivariate-student-t"
  | "trivariate-generalized-gaussian"
  | "trivariate-radial-lognormal"
  | "trivariate-generalized-t";

export interface TripleCandidateFit {
  family: TripleFamily;
  parameters: {
    scale: number;
    power: number | null;
    tail: number | null;
    degreesFreedom: number | null;
    jointPdfTailExponent: number | null;
    marginalPdfTailExponent: number | null;
    marginalSurvivalTailExponent: number | null;
    logRadiusMean: number | null;
    logRadiusStandardDeviation: number | null;
  };
  parameterCountBeyondStandardization: number;
  observations: number;
  nllPerObservation: number;
  aic: number;
  deltaAic: number;
}

export class TripleMoments {
  private readonly raw = new BasicTripleMoments();
  private readonly absolute = new BasicTripleMoments();
  private readonly squared = new BasicTripleMoments();
  private readonly continuous = new BasicTripleMoments();
  private allZero = 0;
  private anyZero = 0;
  private nonzero = 0;
  private allSameSign = 0;
  private readonly patterns = Array.from({ length: 8 }, () => 0);

  add(x: number, y: number, z: number): void {
    if (![x, y, z].every(Number.isFinite)) return;
    this.raw.add(x, y, z);
    this.absolute.add(Math.abs(x), Math.abs(y), Math.abs(z));
    this.squared.add(x * x, y * y, z * z);
    if (x === 0 && y === 0 && z === 0) this.allZero += 1;
    if (x === 0 || y === 0 || z === 0) this.anyZero += 1;
    if (x !== 0 && y !== 0 && z !== 0) {
      this.continuous.add(x, y, z);
      this.nonzero += 1;
      if ((x > 0 && y > 0 && z > 0) || (x < 0 && y < 0 && z < 0)) {
        this.allSameSign += 1;
      }
      const pattern = (x > 0 ? 4 : 0) + (y > 0 ? 2 : 0) + (z > 0 ? 1 : 0);
      this.patterns[pattern]! += 1;
    }
  }

  merge(other: TripleMoments): void {
    this.raw.merge(other.raw);
    this.absolute.merge(other.absolute);
    this.squared.merge(other.squared);
    this.continuous.merge(other.continuous);
    this.allZero += other.allZero;
    this.anyZero += other.anyZero;
    this.nonzero += other.nonzero;
    this.allSameSign += other.allSameSign;
    for (let index = 0; index < 8; index += 1) this.patterns[index]! += other.patterns[index]!;
  }

  snapshot(): TripleMomentsSnapshot {
    const raw = this.raw.snapshot();
    if (raw.observations < 3) throw new Error("At least three return triples are required.");
    const continuous = this.continuous.snapshot();
    const absolute = this.absolute.snapshot();
    const squared = this.squared.snapshot();
    return {
      ...raw,
      absoluteCorrelations: absolute.correlations,
      squaredCorrelations: squared.correlations,
      allZeroFraction: this.allZero / raw.observations,
      anyZeroFraction: this.anyZero / raw.observations,
      nonzeroAllSameSignFraction: this.nonzero === 0 ? null : this.allSameSign / this.nonzero,
      signPatterns: this.patterns.map((count) => count / raw.observations),
      continuous,
    };
  }
}

class BasicTripleMoments {
  private observations = 0;
  private readonly sums = [0, 0, 0];
  private readonly products = [0, 0, 0, 0, 0, 0];

  add(x: number, y: number, z: number): void {
    const values = [x, y, z];
    this.observations += 1;
    for (let index = 0; index < 3; index += 1) this.sums[index]! += values[index]!;
    this.products[0]! += x * x;
    this.products[1]! += y * y;
    this.products[2]! += z * z;
    this.products[3]! += x * y;
    this.products[4]! += y * z;
    this.products[5]! += x * z;
  }

  merge(other: BasicTripleMoments): void {
    this.observations += other.observations;
    for (let index = 0; index < 3; index += 1) this.sums[index]! += other.sums[index]!;
    for (let index = 0; index < 6; index += 1) this.products[index]! += other.products[index]!;
  }

  snapshot(): TripleBasicMoments {
    if (this.observations < 2) {
      return {
        observations: this.observations,
        means: [Number.NaN, Number.NaN, Number.NaN],
        standardDeviations: [Number.NaN, Number.NaN, Number.NaN],
        correlations: [Number.NaN, Number.NaN, Number.NaN],
      };
    }
    const means = this.sums.map((sum) => sum / this.observations) as [number, number, number];
    const variances = [0, 1, 2].map((index) => Math.max(
      0,
      (this.products[index]! - this.observations * means[index]! ** 2)
        / (this.observations - 1),
    ));
    const standardDeviations = variances.map(Math.sqrt) as [number, number, number];
    const covariance = (productIndex: number, left: number, right: number) => (
      (this.products[productIndex]! - this.observations * means[left]! * means[right]!)
        / (this.observations - 1)
    );
    const correlations: [number, number, number] = [
      correlation(covariance(3, 0, 1), standardDeviations[0], standardDeviations[1]),
      correlation(covariance(4, 1, 2), standardDeviations[1], standardDeviations[2]),
      correlation(covariance(5, 0, 2), standardDeviations[0], standardDeviations[2]),
    ];
    return { observations: this.observations, means, standardDeviations, correlations };
  }
}

interface TripleStandardization {
  means: [number, number, number];
  standardDeviations: [number, number, number];
  cholesky: [number, number, number, number, number, number];
  logDeterminantAdjustment: number;
}

export function summarizeTripleShape(
  sample: TripleSample,
  moments: TripleMomentsSnapshot,
  visualizationLimit = 4_000,
): TripleShapeStatistics {
  const all = diagonalStandardization(moments);
  const continuous = tripleStandardization(moments.continuous);
  const histograms = {
    firstSecond: emptyHistogram(),
    secondThird: emptyHistogram(),
    firstThird: emptyHistogram(),
  };
  let tail0Two = 0;
  let tail1Two = 0;
  let tail2Two = 0;
  let jointTwo = 0;
  let tail0Three = 0;
  let tail1Three = 0;
  let tail2Three = 0;
  let jointThree = 0;
  let overallThirdAbsolute = 0;
  let conditionalThirdAbsolute = 0;
  let conditionalCount = 0;
  const radii: number[] = [];
  const eligibleVisualization: number[] = [];
  for (let index = 0; index < sample.x.length; index += 1) {
    const values: [number, number, number] = [
      sample.x[index]!,
      sample.y[index]!,
      sample.z[index]!,
    ];
    const standardized = values.map((value, coordinate) => (
      (value - all.means[coordinate]!) / all.standardDeviations[coordinate]!
    )) as [number, number, number];
    const absolute = standardized.map(Math.abs);
    if (absolute[0]! > 2) tail0Two += 1;
    if (absolute[1]! > 2) tail1Two += 1;
    if (absolute[2]! > 2) tail2Two += 1;
    if (absolute.every((value) => value > 2)) jointTwo += 1;
    if (absolute[0]! > 3) tail0Three += 1;
    if (absolute[1]! > 3) tail1Three += 1;
    if (absolute[2]! > 3) tail2Three += 1;
    if (absolute.every((value) => value > 3)) jointThree += 1;
    overallThirdAbsolute += Math.abs(values[2] - all.means[2]);
    if (absolute[0]! > 2 && absolute[1]! > 2) {
      conditionalThirdAbsolute += Math.abs(values[2] - all.means[2]);
      conditionalCount += 1;
    }
    if (values.some((value) => value === 0)) continue;
    const continuousStandardized = values.map((value, coordinate) => (
      (value - moments.continuous.means[coordinate]!)
        / moments.continuous.standardDeviations[coordinate]!
    )) as [number, number, number];
    addHistogram(histograms.firstSecond, continuousStandardized[0], continuousStandardized[1]);
    addHistogram(histograms.secondThird, continuousStandardized[1], continuousStandardized[2]);
    addHistogram(histograms.firstThird, continuousStandardized[0], continuousStandardized[2]);
    const whitened = whiten(values, continuous);
    radii.push(Math.hypot(...whitened));
    eligibleVisualization.push(index);
  }
  for (const histogram of Object.values(histograms)) normalizeHistogram(histogram);
  radii.sort((left, right) => left - right);
  const visualizationIndices = evenlySpaced(
    eligibleVisualization,
    Math.min(visualizationLimit, eligibleVisualization.length),
  );
  const visualizationSample = visualizationIndices.map((index): [number, number, number] => [
    (sample.x[index]! - moments.continuous.means[0]) / moments.continuous.standardDeviations[0],
    (sample.y[index]! - moments.continuous.means[1]) / moments.continuous.standardDeviations[1],
    (sample.z[index]! - moments.continuous.means[2]) / moments.continuous.standardDeviations[2],
  ]);
  const observations = sample.x.length;
  return {
    sampleObservations: observations,
    continuousSampleObservations: eligibleVisualization.length,
    allThreeAbsoluteTail2Sigma: fraction(jointTwo, observations),
    allThreeAbsoluteTail2SigmaLift: tripleLift(
      jointTwo,
      tail0Two,
      tail1Two,
      tail2Two,
      observations,
    ),
    allThreeAbsoluteTail3Sigma: fraction(jointThree, observations),
    allThreeAbsoluteTail3SigmaLift: tripleLift(
      jointThree,
      tail0Three,
      tail1Three,
      tail2Three,
      observations,
    ),
    thirdAbsoluteReturnAfterFirstTwo2SigmaRatio: conditionalCount === 0
      || overallThirdAbsolute === 0
      ? null
      : (conditionalThirdAbsolute / conditionalCount) / (overallThirdAbsolute / observations),
    whitenedRadiusQuantiles: {
      p50: sampleQuantile(radii, 0.5),
      p90: sampleQuantile(radii, 0.9),
      p95: sampleQuantile(radii, 0.95),
      p99: sampleQuantile(radii, 0.99),
    },
    projectionHistograms: histograms,
    visualizationSample,
  };
}

export function fitTripleCandidates(
  sample: TripleSample,
  moments: TripleMomentsSnapshot,
  fitLimit = 30_000,
): TripleCandidateFit[] {
  const transform = tripleStandardization(moments.continuous);
  const eligible: number[] = [];
  for (let index = 0; index < sample.x.length; index += 1) {
    if (sample.x[index] !== 0 && sample.y[index] !== 0 && sample.z[index] !== 0) {
      eligible.push(index);
    }
  }
  if (eligible.length < 100) throw new Error("At least 100 continuous return triples are required.");
  const selected = evenlySpaced(eligible, Math.min(fitLimit, eligible.length));
  const logR = selected.map((index) => safeLog(Math.hypot(...whiten([
    sample.x[index]!,
    sample.y[index]!,
    sample.z[index]!,
  ], transform))));
  const definitions: Array<Omit<TripleCandidateFit, "deltaAic">> = [];
  const gaussian = generalizedGaussianFit(logR, 3, 2);
  definitions.push(candidate(
    "trivariate-gaussian",
    gaussian.nll + transform.logDeterminantAdjustment,
    selected.length,
    1,
    gaussian.scale,
    null,
    null,
    null,
  ));
  const generalizedGaussian = generalizedGaussianFit(logR, 3);
  definitions.push(candidate(
    "trivariate-generalized-gaussian",
    generalizedGaussian.nll + transform.logDeterminantAdjustment,
    selected.length,
    2,
    generalizedGaussian.scale,
    generalizedGaussian.power,
    null,
    null,
  ));
  const radialLognormal = radialLognormalFit(logR, 3);
  definitions.push(candidate(
    "trivariate-radial-lognormal",
    radialLognormal.nll + transform.logDeterminantAdjustment,
    selected.length,
    2,
    Math.exp(radialLognormal.logRadiusMean),
    null,
    null,
    null,
    radialLognormal.logRadiusMean,
    radialLognormal.logRadiusStandardDeviation,
  ));
  const student = studentFit(logR, 3);
  definitions.push(candidate(
    "trivariate-student-t",
    student.nll + transform.logDeterminantAdjustment,
    selected.length,
    2,
    student.scale,
    2,
    (student.degreesFreedom + 3) / 2,
    student.degreesFreedom,
  ));
  const generalizedT = generalizedTFit(logR, 3);
  definitions.push(candidate(
    "trivariate-generalized-t",
    generalizedT.nll + transform.logDeterminantAdjustment,
    selected.length,
    3,
    generalizedT.scale,
    generalizedT.power,
    generalizedT.tail,
    null,
  ));
  const bestAic = Math.min(...definitions.map((item) => item.aic));
  return definitions.map((item) => ({ ...item, deltaAic: item.aic - bestAic }))
    .sort((left, right) => left.aic - right.aic);
}

export function fitTripleGeneralizedGaussianPower(
  sample: TripleSample,
  moments: TripleMomentsSnapshot,
  fitLimit = 30_000,
): { power: number; scale: number; observations: number } {
  const transform = tripleStandardization(moments.continuous);
  const eligible: number[] = [];
  for (let index = 0; index < sample.x.length; index += 1) {
    if (sample.x[index] !== 0 && sample.y[index] !== 0 && sample.z[index] !== 0) {
      eligible.push(index);
    }
  }
  if (eligible.length < 50) {
    return { power: Number.NaN, scale: Number.NaN, observations: eligible.length };
  }
  const selected = evenlySpaced(eligible, Math.min(fitLimit, eligible.length));
  const logR = selected.map((index) => safeLog(Math.hypot(...whiten([
    sample.x[index]!,
    sample.y[index]!,
    sample.z[index]!,
  ], transform))));
  const fit = generalizedGaussianFit(logR, 3);
  return { power: fit.power, scale: fit.scale, observations: selected.length };
}

export function projectionJensenShannonBits(
  left: TripleShapeStatistics["projectionHistograms"],
  right: TripleShapeStatistics["projectionHistograms"],
): number {
  return (
    jensenShannonBits(left.firstSecond, right.firstSecond)
    + jensenShannonBits(left.secondThird, right.secondThird)
    + jensenShannonBits(left.firstThird, right.firstThird)
  ) / 3;
}

function candidate(
  family: TripleFamily,
  nll: number,
  observations: number,
  parameterCount: number,
  scale: number,
  power: number | null,
  tail: number | null,
  degreesFreedom: number | null,
  logRadiusMean: number | null = null,
  logRadiusStandardDeviation: number | null = null,
): Omit<TripleCandidateFit, "deltaAic"> {
  const jointPdfTailExponent = tail === null || power === null ? null : power * tail;
  const marginalPdfTailExponent = jointPdfTailExponent === null
    ? null
    : jointPdfTailExponent - 2;
  const marginalSurvivalTailExponent = marginalPdfTailExponent === null
    ? null
    : marginalPdfTailExponent - 1;
  return {
    family,
    parameters: {
      scale,
      power,
      tail,
      degreesFreedom,
      jointPdfTailExponent,
      marginalPdfTailExponent,
      marginalSurvivalTailExponent,
      logRadiusMean,
      logRadiusStandardDeviation,
    },
    parameterCountBeyondStandardization: parameterCount,
    observations,
    nllPerObservation: nll,
    aic: 2 * parameterCount + 2 * observations * nll,
  };
}

function radialLognormalFit(
  logR: readonly number[],
  dimensions: number,
): { logRadiusMean: number; logRadiusStandardDeviation: number; nll: number } {
  const logRadiusMean = logR.reduce((sum, value) => sum + value, 0) / logR.length;
  const variance = logR.reduce(
    (sum, value) => sum + (value - logRadiusMean) ** 2,
    0,
  ) / logR.length;
  const logRadiusStandardDeviation = Math.sqrt(Math.max(variance, Number.EPSILON));
  const meanSquaredStandardized = logR.reduce(
    (sum, value) => sum + ((value - logRadiusMean) / logRadiusStandardDeviation) ** 2,
    0,
  ) / logR.length;
  const nll = logSurfaceArea(dimensions)
    + Math.log(logRadiusStandardDeviation)
    + 0.5 * Math.log(2 * Math.PI)
    + dimensions * logRadiusMean
    + 0.5 * meanSquaredStandardized;
  return { logRadiusMean, logRadiusStandardDeviation, nll };
}

function generalizedGaussianFit(
  logR: readonly number[],
  dimensions: number,
  fixedPower?: number,
): { power: number; scale: number; nll: number } {
  const evaluate = (power: number) => {
    if (!(power >= 0.15 && power <= 6)) return Number.POSITIVE_INFINITY;
    const meanPower = meanExponential(logR, power);
    if (!(meanPower > 0 && Number.isFinite(meanPower))) return Number.POSITIVE_INFINITY;
    const logScale = (Math.log(power / dimensions) + Math.log(meanPower)) / power;
    return -Math.log(power) + logSurfaceArea(dimensions) + dimensions * logScale
      + logGamma(dimensions / power) + dimensions / power;
  };
  const power = fixedPower ?? goldenSectionMinimum(evaluate, 0.15, 6);
  const meanPower = meanExponential(logR, power);
  const scale = Math.exp(
    (Math.log(power / dimensions) + Math.log(meanPower)) / power,
  );
  return { power, scale, nll: evaluate(power) };
}

function studentFit(logR: readonly number[], dimensions: number): {
  scale: number;
  degreesFreedom: number;
  nll: number;
} {
  const objective = ([logScale, logDegreesMargin]: number[]) => {
    const degreesFreedom = 2 + Math.exp(logDegreesMargin!);
    if (degreesFreedom > 2_000 || Math.abs(logScale!) > 8) return 1e6;
    const meanLogKernel = meanSoftplus(
      logR,
      2,
      logScale! + 0.5 * Math.log(degreesFreedom),
    );
    const logNormalizer = logGamma((degreesFreedom + dimensions) / 2)
      - logGamma(degreesFreedom / 2)
      - dimensions / 2 * Math.log(degreesFreedom * Math.PI)
      - dimensions * logScale!;
    return -logNormalizer + (degreesFreedom + dimensions) / 2 * meanLogKernel;
  };
  const result = minimizeNelderMead(objective, [Math.log(0.5), Math.log(3)], 0.35);
  return {
    scale: Math.exp(result.parameters[0]!),
    degreesFreedom: 2 + Math.exp(result.parameters[1]!),
    nll: result.value,
  };
}

function generalizedTFit(logR: readonly number[], dimensions: number): {
  scale: number;
  power: number;
  tail: number;
  nll: number;
} {
  const objective = ([logScale, logPower, logTailMargin]: number[]) => {
    const power = Math.exp(logPower!);
    const tail = (dimensions + 2) / power + Math.exp(logTailMargin!);
    if (!(power >= 0.15 && power <= 6 && tail <= 300) || Math.abs(logScale!) > 8) return 1e6;
    const logNormalizer = Math.log(power) - logSurfaceArea(dimensions)
      - dimensions * logScale!
      - logBeta(dimensions / power, tail - dimensions / power);
    return -logNormalizer + tail * meanSoftplus(logR, power, logScale!);
  };
  const starts = [
    [Math.log(0.7), Math.log(1.5), Math.log(1)],
    [Math.log(0.5), Math.log(1), Math.log(2)],
    [Math.log(1), Math.log(2), Math.log(0.5)],
  ];
  const result = starts.map((start) => minimizeNelderMead(objective, start, 0.3))
    .reduce((best, item) => item.value < best.value ? item : best);
  const power = Math.exp(result.parameters[1]!);
  return {
    scale: Math.exp(result.parameters[0]!),
    power,
    tail: (dimensions + 2) / power + Math.exp(result.parameters[2]!),
    nll: result.value,
  };
}

function diagonalStandardization(moments: TripleBasicMoments): {
  means: [number, number, number];
  standardDeviations: [number, number, number];
} {
  if (moments.standardDeviations.some((value) => !(value > 0))) {
    throw new Error("Triple standard deviations must be positive.");
  }
  return { means: moments.means, standardDeviations: moments.standardDeviations };
}

function tripleStandardization(moments: TripleBasicMoments): TripleStandardization {
  diagonalStandardization(moments);
  const r01 = clamp((moments.correlations[0] + moments.correlations[1]) / 2, -0.99, 0.99);
  const r02 = clamp(moments.correlations[2], -0.99, 0.99);
  let l10 = r01;
  let l20 = r02;
  let l11 = Math.sqrt(Math.max(1e-6, 1 - l10 * l10));
  let l21 = (r01 - l20 * l10) / l11;
  let l22Squared = 1 - l20 * l20 - l21 * l21;
  if (l22Squared <= 1e-6) {
    l10 *= 0.99;
    l20 *= 0.99;
    l11 = Math.sqrt(1 - l10 * l10);
    l21 = (r01 * 0.99 - l20 * l10) / l11;
    l22Squared = Math.max(1e-6, 1 - l20 * l20 - l21 * l21);
  }
  const l22 = Math.sqrt(l22Squared);
  return {
    means: moments.means,
    standardDeviations: moments.standardDeviations,
    cholesky: [1, l10, l11, l20, l21, l22],
    logDeterminantAdjustment: moments.standardDeviations
      .reduce((sum, value) => sum + Math.log(value), 0)
      + Math.log(l11) + Math.log(l22),
  };
}

function whiten(
  values: [number, number, number],
  transform: TripleStandardization,
): [number, number, number] {
  const standardized = values.map((value, index) => (
    (value - transform.means[index]!) / transform.standardDeviations[index]!
  ));
  const [, l10, l11, l20, l21, l22] = transform.cholesky;
  const first = standardized[0]!;
  const second = (standardized[1]! - l10 * first) / l11;
  const third = (standardized[2]! - l20 * first - l21 * second) / l22;
  return [first, second, third];
}

function emptyHistogram(): number[] {
  const bins = STANDARDIZED_PAIR_EDGES.length - 1;
  return Array.from({ length: bins * bins }, () => 0);
}

function addHistogram(histogram: number[], x: number, y: number): void {
  const bins = STANDARDIZED_PAIR_EDGES.length - 1;
  histogram[binIndex(y) * bins + binIndex(x)]! += 1;
}

function normalizeHistogram(histogram: number[]): void {
  const total = histogram.reduce((sum, value) => sum + value, 0);
  if (total === 0) return;
  for (let index = 0; index < histogram.length; index += 1) histogram[index]! /= total;
}

function binIndex(value: number): number {
  let lower = 0;
  let upper = STANDARDIZED_PAIR_EDGES.length - 1;
  while (lower + 1 < upper) {
    const middle = Math.floor((lower + upper) / 2);
    if (value < STANDARDIZED_PAIR_EDGES[middle]!) upper = middle;
    else lower = middle;
  }
  return Math.min(lower, STANDARDIZED_PAIR_EDGES.length - 2);
}

function tripleLift(
  joint: number,
  first: number,
  second: number,
  third: number,
  observations: number,
): number | null {
  if (joint === 0 || first === 0 || second === 0 || third === 0 || observations === 0) {
    return null;
  }
  return (joint / observations)
    / ((first / observations) * (second / observations) * (third / observations));
}

function jensenShannonBits(left: readonly number[], right: readonly number[]): number {
  let divergence = 0;
  for (let index = 0; index < left.length; index += 1) {
    const l = left[index]!;
    const r = right[index]!;
    const midpoint = (l + r) / 2;
    if (l > 0) divergence += 0.5 * l * Math.log2(l / midpoint);
    if (r > 0) divergence += 0.5 * r * Math.log2(r / midpoint);
  }
  return divergence;
}

function meanExponential(values: readonly number[], multiplier: number): number {
  let sum = 0;
  for (const value of values) sum += value === Number.NEGATIVE_INFINITY ? 0 : Math.exp(multiplier * value);
  return sum / values.length;
}

function meanSoftplus(
  logValues: readonly number[],
  power: number,
  logScale: number,
): number {
  let sum = 0;
  for (const value of logValues) sum += softplus(power * (value - logScale));
  return sum / logValues.length;
}

function softplus(value: number): number {
  if (value > 35) return value;
  if (value < -35) return Math.exp(value);
  return Math.log1p(Math.exp(value));
}

function goldenSectionMinimum(
  objective: (value: number) => number,
  lower: number,
  upper: number,
): number {
  const ratio = (Math.sqrt(5) - 1) / 2;
  let left = upper - ratio * (upper - lower);
  let right = lower + ratio * (upper - lower);
  let leftValue = objective(left);
  let rightValue = objective(right);
  for (let iteration = 0; iteration < 80; iteration += 1) {
    if (leftValue < rightValue) {
      upper = right;
      right = left;
      rightValue = leftValue;
      left = upper - ratio * (upper - lower);
      leftValue = objective(left);
    } else {
      lower = left;
      left = right;
      leftValue = rightValue;
      right = lower + ratio * (upper - lower);
      rightValue = objective(right);
    }
  }
  return (lower + upper) / 2;
}

function minimizeNelderMead(
  objective: (parameters: number[]) => number,
  start: number[],
  step: number,
): { parameters: number[]; value: number } {
  const dimensions = start.length;
  let simplex = [start.slice()];
  for (let dimension = 0; dimension < dimensions; dimension += 1) {
    const point = start.slice();
    point[dimension] += step;
    simplex.push(point);
  }
  let values = simplex.map(objective);
  for (let iteration = 0; iteration < 2_000; iteration += 1) {
    const order = values.map((_, index) => index)
      .sort((left, right) => values[left]! - values[right]!);
    simplex = order.map((index) => simplex[index]!);
    values = order.map((index) => values[index]!);
    const valueSpread = Math.max(...values.map((value) => Math.abs(value - values[0]!)));
    const pointSpread = Math.max(...simplex.map((point) => Math.max(
      ...point.map((value, index) => Math.abs(value - simplex[0]![index]!)),
    )));
    if (valueSpread < 1e-10 && pointSpread < 1e-6) break;
    const centroid = Array.from({ length: dimensions }, (_, dimension) => (
      simplex.slice(0, dimensions)
        .reduce((sum, point) => sum + point[dimension]!, 0) / dimensions
    ));
    const reflected = centroid.map((value, index) => value + value - simplex.at(-1)![index]!);
    const reflectedValue = objective(reflected);
    if (values[0]! <= reflectedValue && reflectedValue < values[dimensions - 1]!) {
      simplex[dimensions] = reflected;
      values[dimensions] = reflectedValue;
      continue;
    }
    if (reflectedValue < values[0]!) {
      const expanded = centroid.map((value, index) => value + 2 * (reflected[index]! - value));
      const expandedValue = objective(expanded);
      if (expandedValue < reflectedValue) {
        simplex[dimensions] = expanded;
        values[dimensions] = expandedValue;
      } else {
        simplex[dimensions] = reflected;
        values[dimensions] = reflectedValue;
      }
      continue;
    }
    const contracted = centroid.map((value, index) => (
      value + 0.5 * (simplex[dimensions]![index]! - value)
    ));
    const contractedValue = objective(contracted);
    if (contractedValue < values[dimensions]!) {
      simplex[dimensions] = contracted;
      values[dimensions] = contractedValue;
      continue;
    }
    for (let index = 1; index <= dimensions; index += 1) {
      simplex[index] = simplex[index]!.map((value, dimension) => (
        simplex[0]![dimension]! + 0.5 * (value - simplex[0]![dimension]!)
      ));
      values[index] = objective(simplex[index]!);
    }
  }
  const bestIndex = values.indexOf(Math.min(...values));
  return { parameters: simplex[bestIndex]!, value: values[bestIndex]! };
}

function logSurfaceArea(dimensions: number): number {
  return Math.log(2) + dimensions / 2 * Math.log(Math.PI) - logGamma(dimensions / 2);
}

function logBeta(left: number, right: number): number {
  return logGamma(left) + logGamma(right) - logGamma(left + right);
}

function logGamma(value: number): number {
  const coefficients = [
    676.5203681218851,
    -1259.1392167224028,
    771.3234287776531,
    -176.6150291621406,
    12.507343278686905,
    -0.13857109526572012,
    9.984369578019572e-6,
    1.5056327351493116e-7,
  ];
  if (value < 0.5) {
    return Math.log(Math.PI) - Math.log(Math.sin(Math.PI * value)) - logGamma(1 - value);
  }
  const shifted = value - 1;
  let series = 0.9999999999998099;
  for (let index = 0; index < coefficients.length; index += 1) {
    series += coefficients[index]! / (shifted + index + 1);
  }
  const t = shifted + coefficients.length - 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (shifted + 0.5) * Math.log(t)
    - t + Math.log(series);
}

function correlation(covariance: number, left: number, right: number): number {
  const denominator = left * right;
  return denominator > 0 ? clamp(covariance / denominator, -1, 1) : Number.NaN;
}

function sampleQuantile(sorted: readonly number[], probability: number): number | null {
  if (sorted.length === 0) return null;
  const position = probability * (sorted.length - 1);
  const lower = Math.floor(position);
  const weight = position - lower;
  return sorted[lower]! * (1 - weight) + sorted[Math.min(lower + 1, sorted.length - 1)]! * weight;
}

function evenlySpaced(input: readonly number[], count: number): number[] {
  if (count >= input.length) return input.slice();
  return Array.from({ length: count }, (_, index) => (
    input[Math.min(input.length - 1, Math.floor((index + 0.5) * input.length / count))]!
  ));
}

function safeLog(value: number): number {
  return value === 0 ? Number.NEGATIVE_INFINITY : Math.log(value);
}

function fraction(numerator: number, denominator: number): number | null {
  return denominator === 0 ? null : numerator / denominator;
}

function clamp(value: number, lower: number, upper: number): number {
  return Math.max(lower, Math.min(upper, value));
}
