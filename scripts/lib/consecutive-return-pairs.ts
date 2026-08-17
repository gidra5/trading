export interface PairMomentsSnapshot {
  observations: number;
  meanX: number;
  meanY: number;
  standardDeviationX: number;
  standardDeviationY: number;
  correlation: number | null;
  absoluteCorrelation: number | null;
  squaredCorrelation: number | null;
  zeroZeroFraction: number;
  anyZeroFraction: number;
  nonzeroSameSignFraction: number | null;
  quadrants: {
    positivePositive: number;
    positiveNegative: number;
    negativePositive: number;
    negativeNegative: number;
  };
  continuous: {
    observations: number;
    meanX: number;
    meanY: number;
    standardDeviationX: number;
    standardDeviationY: number;
    correlation: number | null;
  };
}

export interface PairSample {
  x: number[];
  y: number[];
}

export interface Standardization {
  meanX: number;
  meanY: number;
  standardDeviationX: number;
  standardDeviationY: number;
  correlation: number;
}

export interface PairShapeStatistics {
  sampleObservations: number;
  continuousSampleObservations: number;
  jointAbsoluteTail2Sigma: number | null;
  jointAbsoluteTail2SigmaLift: number | null;
  jointAbsoluteTail3Sigma: number | null;
  jointAbsoluteTail3SigmaLift: number | null;
  nextAbsoluteReturnAfter2SigmaRatio: number | null;
  whitenedAngularFourfoldAmplitude: number | null;
  whitenedRadiusQuantiles: {
    p50: number | null;
    p90: number | null;
    p95: number | null;
    p99: number | null;
  };
  unconditionalHistogram: number[];
  continuousHistogram: number[];
}

export type RadialFamily =
  | "bivariate-gaussian"
  | "bivariate-student-t"
  | "elliptical-generalized-gaussian"
  | "elliptical-radial-lognormal"
  | "product-generalized-gaussian"
  | "elliptical-generalized-t"
  | "product-generalized-t";

export interface RadialCandidateFit {
  family: RadialFamily;
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

export const STANDARDIZED_PAIR_EDGES = [
  Number.NEGATIVE_INFINITY,
  -4,
  -3,
  -2.5,
  -2,
  -1.5,
  -1.25,
  -1,
  -0.75,
  -0.5,
  -0.25,
  -0.1,
  0,
  0.1,
  0.25,
  0.5,
  0.75,
  1,
  1.25,
  1.5,
  2,
  2.5,
  3,
  4,
  Number.POSITIVE_INFINITY,
] as const;

export class PairMoments {
  private observations = 0;
  private sumX = 0;
  private sumY = 0;
  private sumXX = 0;
  private sumYY = 0;
  private sumXY = 0;
  private sumAbsX = 0;
  private sumAbsY = 0;
  private sumAbsXX = 0;
  private sumAbsYY = 0;
  private sumAbsXY = 0;
  private sumSquaredX = 0;
  private sumSquaredY = 0;
  private sumSquaredXX = 0;
  private sumSquaredYY = 0;
  private sumSquaredXY = 0;
  private zeroZero = 0;
  private anyZero = 0;
  private positivePositive = 0;
  private positiveNegative = 0;
  private negativePositive = 0;
  private negativeNegative = 0;
  private readonly continuousMoments = new BasicPairMoments();

  add(x: number, y: number): void {
    if (!Number.isFinite(x) || !Number.isFinite(y)) return;
    this.observations += 1;
    this.sumX += x;
    this.sumY += y;
    this.sumXX += x * x;
    this.sumYY += y * y;
    this.sumXY += x * y;
    const absoluteX = Math.abs(x);
    const absoluteY = Math.abs(y);
    this.sumAbsX += absoluteX;
    this.sumAbsY += absoluteY;
    this.sumAbsXX += absoluteX * absoluteX;
    this.sumAbsYY += absoluteY * absoluteY;
    this.sumAbsXY += absoluteX * absoluteY;
    const squaredX = x * x;
    const squaredY = y * y;
    this.sumSquaredX += squaredX;
    this.sumSquaredY += squaredY;
    this.sumSquaredXX += squaredX * squaredX;
    this.sumSquaredYY += squaredY * squaredY;
    this.sumSquaredXY += squaredX * squaredY;
    if (x === 0 && y === 0) this.zeroZero += 1;
    if (x === 0 || y === 0) this.anyZero += 1;
    if (x > 0 && y > 0) this.positivePositive += 1;
    else if (x > 0 && y < 0) this.positiveNegative += 1;
    else if (x < 0 && y > 0) this.negativePositive += 1;
    else if (x < 0 && y < 0) this.negativeNegative += 1;
    if (x !== 0 && y !== 0) this.continuousMoments.add(x, y);
  }

  merge(other: PairMoments): void {
    this.observations += other.observations;
    this.sumX += other.sumX;
    this.sumY += other.sumY;
    this.sumXX += other.sumXX;
    this.sumYY += other.sumYY;
    this.sumXY += other.sumXY;
    this.sumAbsX += other.sumAbsX;
    this.sumAbsY += other.sumAbsY;
    this.sumAbsXX += other.sumAbsXX;
    this.sumAbsYY += other.sumAbsYY;
    this.sumAbsXY += other.sumAbsXY;
    this.sumSquaredX += other.sumSquaredX;
    this.sumSquaredY += other.sumSquaredY;
    this.sumSquaredXX += other.sumSquaredXX;
    this.sumSquaredYY += other.sumSquaredYY;
    this.sumSquaredXY += other.sumSquaredXY;
    this.zeroZero += other.zeroZero;
    this.anyZero += other.anyZero;
    this.positivePositive += other.positivePositive;
    this.positiveNegative += other.positiveNegative;
    this.negativePositive += other.negativePositive;
    this.negativeNegative += other.negativeNegative;
    this.continuousMoments.merge(other.continuousMoments);
  }

  snapshot(): PairMomentsSnapshot {
    if (this.observations < 2) throw new Error("At least two return pairs are required.");
    const basic = basicSnapshot(
      this.observations,
      this.sumX,
      this.sumY,
      this.sumXX,
      this.sumYY,
      this.sumXY,
    );
    const absolute = basicSnapshot(
      this.observations,
      this.sumAbsX,
      this.sumAbsY,
      this.sumAbsXX,
      this.sumAbsYY,
      this.sumAbsXY,
    );
    const squared = basicSnapshot(
      this.observations,
      this.sumSquaredX,
      this.sumSquaredY,
      this.sumSquaredXX,
      this.sumSquaredYY,
      this.sumSquaredXY,
    );
    const nonzeroSigned = this.positivePositive + this.positiveNegative
      + this.negativePositive + this.negativeNegative;
    const continuous = this.continuousMoments.snapshot();
    return {
      observations: this.observations,
      meanX: basic.meanX,
      meanY: basic.meanY,
      standardDeviationX: basic.standardDeviationX,
      standardDeviationY: basic.standardDeviationY,
      correlation: basic.correlation,
      absoluteCorrelation: absolute.correlation,
      squaredCorrelation: squared.correlation,
      zeroZeroFraction: this.zeroZero / this.observations,
      anyZeroFraction: this.anyZero / this.observations,
      nonzeroSameSignFraction: nonzeroSigned === 0
        ? null
        : (this.positivePositive + this.negativeNegative) / nonzeroSigned,
      quadrants: {
        positivePositive: this.positivePositive / this.observations,
        positiveNegative: this.positiveNegative / this.observations,
        negativePositive: this.negativePositive / this.observations,
        negativeNegative: this.negativeNegative / this.observations,
      },
      continuous,
    };
  }
}

class BasicPairMoments {
  private observations = 0;
  private sumX = 0;
  private sumY = 0;
  private sumXX = 0;
  private sumYY = 0;
  private sumXY = 0;

  add(x: number, y: number): void {
    this.observations += 1;
    this.sumX += x;
    this.sumY += y;
    this.sumXX += x * x;
    this.sumYY += y * y;
    this.sumXY += x * y;
  }

  merge(other: BasicPairMoments): void {
    this.observations += other.observations;
    this.sumX += other.sumX;
    this.sumY += other.sumY;
    this.sumXX += other.sumXX;
    this.sumYY += other.sumYY;
    this.sumXY += other.sumXY;
  }

  snapshot(): PairMomentsSnapshot["continuous"] {
    if (this.observations < 2) {
      return {
        observations: this.observations,
        meanX: Number.NaN,
        meanY: Number.NaN,
        standardDeviationX: Number.NaN,
        standardDeviationY: Number.NaN,
        correlation: null,
      };
    }
    return {
      observations: this.observations,
      ...basicSnapshot(
        this.observations,
        this.sumX,
        this.sumY,
        this.sumXX,
        this.sumYY,
        this.sumXY,
      ),
    };
  }
}

export function standardization(
  snapshot: PairMomentsSnapshot,
  continuous: boolean,
): Standardization {
  const selected = continuous ? snapshot.continuous : snapshot;
  if (!(selected.standardDeviationX > 0 && selected.standardDeviationY > 0)) {
    throw new Error("Return-pair standard deviations must be positive.");
  }
  return {
    meanX: selected.meanX,
    meanY: selected.meanY,
    standardDeviationX: selected.standardDeviationX,
    standardDeviationY: selected.standardDeviationY,
    correlation: clamp(selected.correlation ?? 0, -0.995, 0.995),
  };
}

export function summarizePairShape(
  sample: PairSample,
  moments: PairMomentsSnapshot,
): PairShapeStatistics {
  const all = standardization(moments, false);
  const continuous = standardization(moments, true);
  let tailX2 = 0;
  let tailY2 = 0;
  let joint2 = 0;
  let tailX3 = 0;
  let tailY3 = 0;
  let joint3 = 0;
  let conditionalNextAbsolute = 0;
  let conditionalCount = 0;
  let overallNextAbsolute = 0;
  let continuousCount = 0;
  let cos4 = 0;
  let sin4 = 0;
  const radii: number[] = [];
  const unconditionalHistogram = emptyHistogram();
  const continuousHistogram = emptyHistogram();
  for (let index = 0; index < sample.x.length; index += 1) {
    const x = sample.x[index]!;
    const y = sample.y[index]!;
    const zx = (x - all.meanX) / all.standardDeviationX;
    const zy = (y - all.meanY) / all.standardDeviationY;
    addHistogram(unconditionalHistogram, zx, zy);
    const absoluteX = Math.abs(zx);
    const absoluteY = Math.abs(zy);
    if (absoluteX > 2) tailX2 += 1;
    if (absoluteY > 2) tailY2 += 1;
    if (absoluteX > 2 && absoluteY > 2) joint2 += 1;
    if (absoluteX > 3) tailX3 += 1;
    if (absoluteY > 3) tailY3 += 1;
    if (absoluteX > 3 && absoluteY > 3) joint3 += 1;
    overallNextAbsolute += Math.abs(y - all.meanY);
    if (absoluteX > 2) {
      conditionalNextAbsolute += Math.abs(y - all.meanY);
      conditionalCount += 1;
    }
    if (x === 0 || y === 0) continue;
    const whitened = whiten(x, y, continuous);
    const radius = Math.hypot(whitened[0], whitened[1]);
    const angle = Math.atan2(whitened[1], whitened[0]);
    radii.push(radius);
    cos4 += Math.cos(4 * angle);
    sin4 += Math.sin(4 * angle);
    continuousCount += 1;
    addHistogram(
      continuousHistogram,
      (x - continuous.meanX) / continuous.standardDeviationX,
      (y - continuous.meanY) / continuous.standardDeviationY,
    );
  }
  normalizeHistogram(unconditionalHistogram);
  normalizeHistogram(continuousHistogram);
  radii.sort((left, right) => left - right);
  const count = sample.x.length;
  return {
    sampleObservations: count,
    continuousSampleObservations: continuousCount,
    jointAbsoluteTail2Sigma: fraction(joint2, count),
    jointAbsoluteTail2SigmaLift: lift(joint2, tailX2, tailY2, count),
    jointAbsoluteTail3Sigma: fraction(joint3, count),
    jointAbsoluteTail3SigmaLift: lift(joint3, tailX3, tailY3, count),
    nextAbsoluteReturnAfter2SigmaRatio: conditionalCount === 0 || overallNextAbsolute === 0
      ? null
      : (conditionalNextAbsolute / conditionalCount) / (overallNextAbsolute / count),
    whitenedAngularFourfoldAmplitude: continuousCount === 0
      ? null
      : Math.hypot(cos4 / continuousCount, sin4 / continuousCount),
    whitenedRadiusQuantiles: {
      p50: sampleQuantile(radii, 0.5),
      p90: sampleQuantile(radii, 0.9),
      p95: sampleQuantile(radii, 0.95),
      p99: sampleQuantile(radii, 0.99),
    },
    unconditionalHistogram,
    continuousHistogram,
  };
}

export function fitRadialCandidates(
  sample: PairSample,
  moments: PairMomentsSnapshot,
  fitLimit = 30_000,
): RadialCandidateFit[] {
  const transform = standardization(moments, true);
  const eligible: number[] = [];
  for (let index = 0; index < sample.x.length; index += 1) {
    if (sample.x[index] !== 0 && sample.y[index] !== 0) eligible.push(index);
  }
  if (eligible.length < 100) throw new Error("At least 100 continuous return pairs are required.");
  const selected = evenlySpaced(eligible, Math.min(fitLimit, eligible.length));
  const logR: number[] = [];
  const logAbsU: number[] = [];
  const logAbsV: number[] = [];
  for (const index of selected) {
    const [u, v] = whiten(sample.x[index]!, sample.y[index]!, transform);
    logR.push(safeLog(Math.hypot(u, v)));
    logAbsU.push(safeLog(Math.abs(u)));
    logAbsV.push(safeLog(Math.abs(v)));
  }
  const determinantAdjustment = 0.5 * Math.log(1 - transform.correlation ** 2)
    + Math.log(transform.standardDeviationX * transform.standardDeviationY);
  const definitions: Array<Omit<RadialCandidateFit, "deltaAic">> = [];
  const gaussian = generalizedGaussianFit(logR, 2);
  definitions.push(candidate(
    "bivariate-gaussian",
    gaussian.nll + determinantAdjustment,
    selected.length,
    1,
    gaussian.scale,
    null,
    null,
    null,
  ));
  const generalizedGaussian = generalizedGaussianFit(logR);
  definitions.push(candidate(
    "elliptical-generalized-gaussian",
    generalizedGaussian.nll + determinantAdjustment,
    selected.length,
    2,
    generalizedGaussian.scale,
    generalizedGaussian.power,
    null,
    null,
  ));
  const radialLognormal = radialLognormalFit(logR, 2);
  definitions.push(candidate(
    "elliptical-radial-lognormal",
    radialLognormal.nll + determinantAdjustment,
    selected.length,
    2,
    Math.exp(radialLognormal.logRadiusMean),
    null,
    null,
    null,
    radialLognormal.logRadiusMean,
    radialLognormal.logRadiusStandardDeviation,
  ));
  const productGaussian = productGeneralizedGaussianFit(logAbsU, logAbsV);
  definitions.push(candidate(
    "product-generalized-gaussian",
    productGaussian.nll + determinantAdjustment,
    selected.length,
    2,
    productGaussian.scale,
    productGaussian.power,
    null,
    null,
  ));
  const student = studentFit(logR);
  definitions.push(candidate(
    "bivariate-student-t",
    student.nll + determinantAdjustment,
    selected.length,
    2,
    student.scale,
    2,
    (student.degreesFreedom + 2) / 2,
    student.degreesFreedom,
  ));
  const generalizedT = ellipticalGeneralizedTFit(logR);
  definitions.push(candidate(
    "elliptical-generalized-t",
    generalizedT.nll + determinantAdjustment,
    selected.length,
    3,
    generalizedT.scale,
    generalizedT.power,
    generalizedT.tail,
    null,
  ));
  const productGeneralizedT = productGeneralizedTFit(logAbsU, logAbsV);
  definitions.push(candidate(
    "product-generalized-t",
    productGeneralizedT.nll + determinantAdjustment,
    selected.length,
    3,
    productGeneralizedT.scale,
    productGeneralizedT.power,
    productGeneralizedT.tail,
    null,
  ));
  const bestAic = Math.min(...definitions.map((item) => item.aic));
  return definitions.map((item) => ({ ...item, deltaAic: item.aic - bestAic }))
    .sort((left, right) => left.aic - right.aic);
}

export function fitEllipticalGeneralizedGaussianPower(
  sample: PairSample,
  moments: PairMomentsSnapshot,
  fitLimit = 30_000,
): { power: number; scale: number; observations: number } {
  const transform = standardization(moments, true);
  const continuousIndices: number[] = [];
  for (let index = 0; index < sample.x.length; index += 1) {
    if (sample.x[index] !== 0 && sample.y[index] !== 0) continuousIndices.push(index);
  }
  if (continuousIndices.length < 50) {
    return { power: Number.NaN, scale: Number.NaN, observations: continuousIndices.length };
  }
  const selected = evenlySpaced(
    continuousIndices,
    Math.min(fitLimit, continuousIndices.length),
  );
  const logR = selected.map((index) => {
    const [u, v] = whiten(sample.x[index]!, sample.y[index]!, transform);
    return safeLog(Math.hypot(u, v));
  });
  const fit = generalizedGaussianFit(logR);
  return { power: fit.power, scale: fit.scale, observations: selected.length };
}

export function jensenShannonBits(left: readonly number[], right: readonly number[]): number {
  if (left.length !== right.length || left.length === 0) {
    throw new Error("Jensen-Shannon inputs must have the same nonzero length.");
  }
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

function candidate(
  family: RadialFamily,
  nll: number,
  observations: number,
  parameterCount: number,
  scale: number,
  power: number | null,
  tail: number | null,
  degreesFreedom: number | null,
  logRadiusMean: number | null = null,
  logRadiusStandardDeviation: number | null = null,
): Omit<RadialCandidateFit, "deltaAic"> {
  const elliptical = family.startsWith("elliptical") || family === "bivariate-student-t";
  const jointPdfTailExponent = tail === null || power === null
    ? null
    : elliptical ? power * tail : power * tail;
  const marginalPdfTailExponent = jointPdfTailExponent === null
    ? null
    : elliptical ? jointPdfTailExponent - 1 : jointPdfTailExponent;
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
  const nll = Math.log(2 * Math.PI)
    + Math.log(logRadiusStandardDeviation)
    + 0.5 * Math.log(2 * Math.PI)
    + dimensions * logRadiusMean
    + 0.5 * meanSquaredStandardized;
  return { logRadiusMean, logRadiusStandardDeviation, nll };
}

function generalizedGaussianFit(
  logR: readonly number[],
  fixedPower?: number,
): { power: number; scale: number; nll: number } {
  const evaluate = (power: number) => {
    if (!(power >= 0.15 && power <= 6)) return Number.POSITIVE_INFINITY;
    const meanPower = meanExponential(logR, power);
    if (!(meanPower > 0 && Number.isFinite(meanPower))) return Number.POSITIVE_INFINITY;
    const logScale = (Math.log(power / 2) + Math.log(meanPower)) / power;
    return -Math.log(power) + Math.log(2 * Math.PI) + 2 * logScale
      + logGamma(2 / power) + 2 / power;
  };
  const power = fixedPower ?? goldenSectionMinimum(evaluate, 0.15, 6);
  const meanPower = meanExponential(logR, power);
  const scale = Math.exp((Math.log(power / 2) + Math.log(meanPower)) / power);
  return { power, scale, nll: evaluate(power) };
}

function productGeneralizedGaussianFit(
  logAbsU: readonly number[],
  logAbsV: readonly number[],
): { power: number; scale: number; nll: number } {
  const evaluate = (power: number) => {
    if (!(power >= 0.15 && power <= 6)) return Number.POSITIVE_INFINITY;
    const meanPower = meanTwoExponentials(logAbsU, logAbsV, power);
    if (!(meanPower > 0 && Number.isFinite(meanPower))) return Number.POSITIVE_INFINITY;
    const logScale = (Math.log(power / 2) + Math.log(meanPower)) / power;
    return -2 * Math.log(power) + Math.log(4) + 2 * logScale
      + 2 * logGamma(1 / power) + 2 / power;
  };
  const power = goldenSectionMinimum(evaluate, 0.15, 6);
  const meanPower = meanTwoExponentials(logAbsU, logAbsV, power);
  const scale = Math.exp((Math.log(power / 2) + Math.log(meanPower)) / power);
  return { power, scale, nll: evaluate(power) };
}

function studentFit(logR: readonly number[]): {
  scale: number;
  degreesFreedom: number;
  nll: number;
} {
  const objective = ([logScale, logDegreesMargin]: number[]) => {
    const degreesFreedom = 2 + Math.exp(logDegreesMargin!);
    if (degreesFreedom > 2_000 || Math.abs(logScale!) > 8) return 1e6;
    const meanLogKernel = meanSoftplus(logR, 2, logScale! + 0.5 * Math.log(degreesFreedom));
    const logNormalizer = logGamma((degreesFreedom + 2) / 2)
      - logGamma(degreesFreedom / 2) - Math.log(degreesFreedom * Math.PI)
      - 2 * logScale!;
    return -logNormalizer + (degreesFreedom + 2) / 2 * meanLogKernel;
  };
  const result = minimizeNelderMead(objective, [Math.log(0.5), Math.log(3)], 0.35);
  return {
    scale: Math.exp(result.parameters[0]!),
    degreesFreedom: 2 + Math.exp(result.parameters[1]!),
    nll: result.value,
  };
}

function ellipticalGeneralizedTFit(logR: readonly number[]): {
  scale: number;
  power: number;
  tail: number;
  nll: number;
} {
  const objective = ([logScale, logPower, logTailMargin]: number[]) => {
    const power = Math.exp(logPower!);
    const tail = 4 / power + Math.exp(logTailMargin!);
    if (!(power >= 0.15 && power <= 6 && tail <= 250) || Math.abs(logScale!) > 8) return 1e6;
    const logNormalizer = Math.log(power) - Math.log(2 * Math.PI) - 2 * logScale!
      - logBeta(2 / power, tail - 2 / power);
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
    tail: 4 / power + Math.exp(result.parameters[2]!),
    nll: result.value,
  };
}

function productGeneralizedTFit(
  logAbsU: readonly number[],
  logAbsV: readonly number[],
): { scale: number; power: number; tail: number; nll: number } {
  const objective = ([logScale, logPower, logTailMargin]: number[]) => {
    const power = Math.exp(logPower!);
    const tail = 3 / power + Math.exp(logTailMargin!);
    if (!(power >= 0.15 && power <= 6 && tail <= 250) || Math.abs(logScale!) > 8) return 1e6;
    const logNormalizer = Math.log(power) - Math.log(2) - logScale!
      - logBeta(1 / power, tail - 1 / power);
    const meanKernel = meanTwoSoftplus(logAbsU, logAbsV, power, logScale!);
    return -2 * logNormalizer + tail * meanKernel;
  };
  const starts = [
    [Math.log(0.5), Math.log(1.5), Math.log(1)],
    [Math.log(0.4), Math.log(1), Math.log(2)],
    [Math.log(0.8), Math.log(2), Math.log(0.5)],
  ];
  const result = starts.map((start) => minimizeNelderMead(objective, start, 0.3))
    .reduce((best, item) => item.value < best.value ? item : best);
  const power = Math.exp(result.parameters[1]!);
  return {
    scale: Math.exp(result.parameters[0]!),
    power,
    tail: 3 / power + Math.exp(result.parameters[2]!),
    nll: result.value,
  };
}

function meanExponential(values: readonly number[], multiplier: number): number {
  let sum = 0;
  for (const value of values) sum += value === Number.NEGATIVE_INFINITY ? 0 : Math.exp(multiplier * value);
  return sum / values.length;
}

function meanTwoExponentials(
  left: readonly number[],
  right: readonly number[],
  multiplier: number,
): number {
  let sum = 0;
  for (let index = 0; index < left.length; index += 1) {
    sum += left[index] === Number.NEGATIVE_INFINITY ? 0 : Math.exp(multiplier * left[index]!);
    sum += right[index] === Number.NEGATIVE_INFINITY ? 0 : Math.exp(multiplier * right[index]!);
  }
  return sum / left.length;
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

function meanTwoSoftplus(
  left: readonly number[],
  right: readonly number[],
  power: number,
  logScale: number,
): number {
  let sum = 0;
  for (let index = 0; index < left.length; index += 1) {
    sum += softplus(power * (left[index]! - logScale));
    sum += softplus(power * (right[index]! - logScale));
  }
  return sum / left.length;
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

function basicSnapshot(
  observations: number,
  sumX: number,
  sumY: number,
  sumXX: number,
  sumYY: number,
  sumXY: number,
): {
  meanX: number;
  meanY: number;
  standardDeviationX: number;
  standardDeviationY: number;
  correlation: number | null;
} {
  const meanX = sumX / observations;
  const meanY = sumY / observations;
  const varianceX = Math.max(0, (sumXX - observations * meanX * meanX) / (observations - 1));
  const varianceY = Math.max(0, (sumYY - observations * meanY * meanY) / (observations - 1));
  const covariance = (sumXY - observations * meanX * meanY) / (observations - 1);
  const denominator = Math.sqrt(varianceX * varianceY);
  return {
    meanX,
    meanY,
    standardDeviationX: Math.sqrt(varianceX),
    standardDeviationY: Math.sqrt(varianceY),
    correlation: denominator > 0 ? clamp(covariance / denominator, -1, 1) : null,
  };
}

function whiten(x: number, y: number, transform: Standardization): [number, number] {
  const zx = (x - transform.meanX) / transform.standardDeviationX;
  const zy = (y - transform.meanY) / transform.standardDeviationY;
  const common = (zx + zy) / Math.sqrt(2 * (1 + transform.correlation));
  const difference = (zx - zy) / Math.sqrt(2 * (1 - transform.correlation));
  return [common, difference];
}

function emptyHistogram(): number[] {
  const bins = STANDARDIZED_PAIR_EDGES.length - 1;
  return Array.from({ length: bins * bins }, () => 0);
}

function addHistogram(histogram: number[], x: number, y: number): void {
  const bins = STANDARDIZED_PAIR_EDGES.length - 1;
  const xIndex = binIndex(x);
  const yIndex = binIndex(y);
  histogram[yIndex * bins + xIndex]! += 1;
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

function lift(joint: number, left: number, right: number, observations: number): number | null {
  if (left === 0 || right === 0 || observations === 0) return null;
  return (joint / observations) / ((left / observations) * (right / observations));
}

function fraction(numerator: number, denominator: number): number | null {
  return denominator === 0 ? null : numerator / denominator;
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

function clamp(value: number, lower: number, upper: number): number {
  return Math.max(lower, Math.min(upper, value));
}
