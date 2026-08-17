export interface TimedClose {
  openTime: number;
  close: number;
  closed?: boolean;
}

export interface TimedReturn {
  endTime: number;
  value: number;
}

export interface ReturnDistribution {
  observations: number;
  meanBps: number;
  medianBps: number;
  meanAbsoluteBps: number;
  standardDeviationBps: number;
  robustSigmaBps: number;
  skewness: number | null;
  excessKurtosis: number | null;
  positiveFraction: number;
  zeroFraction: number;
  lag1ReturnCorrelation: number | null;
  lag1AbsoluteReturnCorrelation: number | null;
  standardDeviationToRobustSigma: number | null;
  iqrToGaussianIqr: number | null;
  centralMass025Sigma: number | null;
  centralMassVsGaussian: number | null;
  tailMass3Sigma: number | null;
  tailMass3SigmaVsGaussian: number | null;
  absoluteP99Sigma: number | null;
  absoluteP999Sigma: number | null;
  downsideToUpsideP99: number | null;
  quantilesBps: {
    p001: number;
    p01: number;
    p05: number;
    p25: number;
    p50: number;
    p75: number;
    p95: number;
    p99: number;
    p999: number;
  };
  absoluteTail: Array<{
    sigma: number;
    probability: number;
  }>;
}

const MINUTE_MS = 60_000;
const GAUSSIAN_CENTRAL_MASS_025 = 0.197_412_651_365_847_4;
const GAUSSIAN_TAIL_MASS_3 = 0.002_699_796_063_260_207;
const GAUSSIAN_IQR_SIGMA = 1.348_979_500_392_163_4;
const TAIL_THRESHOLDS = [
  0.25,
  0.5,
  0.75,
  1,
  1.25,
  1.5,
  2,
  2.5,
  3,
  3.5,
  4,
  5,
  6,
  8,
  10,
] as const;

/**
 * Aggregate complete one-minute closes into UTC-aligned buckets, then calculate
 * close-to-close log returns only across adjacent complete buckets. A data gap
 * therefore never becomes a synthetic long-horizon return.
 */
export function aggregateLogReturns(
  candles: readonly TimedClose[],
  intervalMs: number,
): TimedReturn[] {
  if (!Number.isSafeInteger(intervalMs) || intervalMs < MINUTE_MS
    || intervalMs % MINUTE_MS !== 0) {
    throw new Error("Return interval must be a whole number of minutes.");
  }

  const expectedCandles = intervalMs / MINUTE_MS;
  const output: TimedReturn[] = [];
  let bucketStart = Number.NaN;
  let bucketCount = 0;
  let bucketFirstOpen = Number.NaN;
  let bucketLastOpen = Number.NaN;
  let bucketLastClose = Number.NaN;
  let previousCompleteBucketStart = Number.NaN;
  let previousCompleteClose = Number.NaN;

  const finishBucket = () => {
    if (!Number.isFinite(bucketStart)) return;
    const complete = bucketCount === expectedCandles
      && bucketFirstOpen === bucketStart
      && bucketLastOpen === bucketStart + intervalMs - MINUTE_MS
      && Number.isFinite(bucketLastClose)
      && bucketLastClose > 0;
    if (complete) {
      if (previousCompleteBucketStart === bucketStart - intervalMs
        && Number.isFinite(previousCompleteClose)
        && previousCompleteClose > 0) {
        output.push({
          endTime: bucketStart + intervalMs,
          value: Math.log(bucketLastClose / previousCompleteClose),
        });
      }
      previousCompleteBucketStart = bucketStart;
      previousCompleteClose = bucketLastClose;
    } else {
      previousCompleteBucketStart = Number.NaN;
      previousCompleteClose = Number.NaN;
    }
  };

  let previousOpenTime = Number.NEGATIVE_INFINITY;
  for (const candle of candles) {
    if (!Number.isSafeInteger(candle.openTime)
      || candle.openTime <= previousOpenTime
      || candle.openTime % MINUTE_MS !== 0) {
      throw new Error("Minute candles must be strictly ordered and UTC minute-aligned.");
    }
    previousOpenTime = candle.openTime;
    const nextBucketStart = Math.floor(candle.openTime / intervalMs) * intervalMs;
    if (nextBucketStart !== bucketStart) {
      finishBucket();
      bucketStart = nextBucketStart;
      bucketCount = 0;
      bucketFirstOpen = candle.openTime;
    }
    bucketCount += 1;
    bucketLastOpen = candle.openTime;
    bucketLastClose = candle.closed === false ? Number.NaN : candle.close;
  }
  finishBucket();
  return output;
}

export function returnsInWindow(
  returns: readonly TimedReturn[],
  startTime: number,
  endTime: number,
): number[] {
  if (!(endTime > startTime)) throw new Error("Return window must have positive duration.");
  const start = upperBound(returns, startTime);
  const end = upperBound(returns, endTime);
  return returns.slice(start, end).map((item) => item.value);
}

export function summarizeReturnDistribution(
  input: readonly number[],
): ReturnDistribution {
  const values = input.filter(Number.isFinite);
  if (values.length < 2) {
    throw new Error("At least two finite returns are required.");
  }
  const observations = values.length;
  const mean = values.reduce((total, value) => total + value, 0) / observations;
  let squared = 0;
  let cubed = 0;
  let fourth = 0;
  let positive = 0;
  let zero = 0;
  const centeredAbsolute = new Array<number>(observations);
  for (let index = 0; index < observations; index += 1) {
    const value = values[index]!;
    const delta = value - mean;
    const delta2 = delta * delta;
    squared += delta2;
    cubed += delta2 * delta;
    fourth += delta2 * delta2;
    centeredAbsolute[index] = Math.abs(delta);
    if (value > 0) positive += 1;
    if (value === 0) zero += 1;
  }
  const variance = squared / (observations - 1);
  const sigma = Math.sqrt(variance);
  const sorted = [...values].sort((left, right) => left - right);
  const median = quantileSorted(sorted, 0.5);
  const absoluteFromMedian = values
    .map((value) => Math.abs(value - median))
    .sort((left, right) => left - right);
  const robustSigma = quantileSorted(absoluteFromMedian, 0.5) * 1.4826;
  centeredAbsolute.sort((left, right) => left - right);
  const p25 = quantileSorted(sorted, 0.25);
  const p75 = quantileSorted(sorted, 0.75);

  const skewness = observations > 2 && sigma > 0
    ? observations / ((observations - 1) * (observations - 2))
      * cubed / (sigma ** 3)
    : null;
  const excessKurtosis = observations > 3 && sigma > 0
    ? observations * (observations + 1)
      / ((observations - 1) * (observations - 2) * (observations - 3))
      * fourth / (sigma ** 4)
      - 3 * (observations - 1) ** 2
      / ((observations - 2) * (observations - 3))
    : null;
  const centralMass = sigma > 0
    ? centeredAbsolute.filter((value) => value <= 0.25 * sigma).length / observations
    : null;
  const tailMass3 = sigma > 0
    ? centeredAbsolute.filter((value) => value > 3 * sigma).length / observations
    : null;
  const absoluteP99Sigma = sigma > 0
    ? quantileSorted(centeredAbsolute, 0.99) / sigma
    : null;
  const absoluteP999Sigma = sigma > 0
    ? quantileSorted(centeredAbsolute, 0.999) / sigma
    : null;
  const downsideP01 = Math.abs(quantileSorted(sorted, 0.01) - mean);
  const upsideP99 = Math.abs(quantileSorted(sorted, 0.99) - mean);

  return {
    observations,
    meanBps: mean * 10_000,
    medianBps: median * 10_000,
    meanAbsoluteBps: values.reduce((total, value) => total + Math.abs(value), 0)
      / observations * 10_000,
    standardDeviationBps: sigma * 10_000,
    robustSigmaBps: robustSigma * 10_000,
    skewness,
    excessKurtosis,
    positiveFraction: positive / observations,
    zeroFraction: zero / observations,
    lag1ReturnCorrelation: pearsonLag1(values, false),
    lag1AbsoluteReturnCorrelation: pearsonLag1(values, true),
    standardDeviationToRobustSigma: robustSigma > 0 ? sigma / robustSigma : null,
    iqrToGaussianIqr: sigma > 0 ? (p75 - p25) / (GAUSSIAN_IQR_SIGMA * sigma) : null,
    centralMass025Sigma: centralMass,
    centralMassVsGaussian: centralMass === null
      ? null
      : centralMass / GAUSSIAN_CENTRAL_MASS_025,
    tailMass3Sigma: tailMass3,
    tailMass3SigmaVsGaussian: tailMass3 === null
      ? null
      : tailMass3 / GAUSSIAN_TAIL_MASS_3,
    absoluteP99Sigma,
    absoluteP999Sigma,
    downsideToUpsideP99: upsideP99 > 0 ? downsideP01 / upsideP99 : null,
    quantilesBps: {
      p001: quantileSorted(sorted, 0.001) * 10_000,
      p01: quantileSorted(sorted, 0.01) * 10_000,
      p05: quantileSorted(sorted, 0.05) * 10_000,
      p25: p25 * 10_000,
      p50: median * 10_000,
      p75: p75 * 10_000,
      p95: quantileSorted(sorted, 0.95) * 10_000,
      p99: quantileSorted(sorted, 0.99) * 10_000,
      p999: quantileSorted(sorted, 0.999) * 10_000,
    },
    absoluteTail: TAIL_THRESHOLDS.map((threshold) => ({
      sigma: threshold,
      probability: sigma > 0
        ? (observations - upperBoundNumbers(centeredAbsolute, threshold * sigma)) / observations
        : 0,
    })),
  };
}

function quantileSorted(sorted: readonly number[], probability: number): number {
  if (sorted.length === 0) return Number.NaN;
  const position = (sorted.length - 1) * probability;
  const lower = Math.floor(position);
  const fraction = position - lower;
  const left = sorted[lower]!;
  const right = sorted[Math.min(lower + 1, sorted.length - 1)]!;
  return left + fraction * (right - left);
}

function pearsonLag1(values: readonly number[], absolute: boolean): number | null {
  if (values.length < 3) return null;
  let leftMean = 0;
  let rightMean = 0;
  for (let index = 1; index < values.length; index += 1) {
    leftMean += absolute ? Math.abs(values[index - 1]!) : values[index - 1]!;
    rightMean += absolute ? Math.abs(values[index]!) : values[index]!;
  }
  const count = values.length - 1;
  leftMean /= count;
  rightMean /= count;
  let covariance = 0;
  let leftVariance = 0;
  let rightVariance = 0;
  for (let index = 1; index < values.length; index += 1) {
    const left = (absolute ? Math.abs(values[index - 1]!) : values[index - 1]!) - leftMean;
    const right = (absolute ? Math.abs(values[index]!) : values[index]!) - rightMean;
    covariance += left * right;
    leftVariance += left * left;
    rightVariance += right * right;
  }
  const denominator = Math.sqrt(leftVariance * rightVariance);
  return denominator > 0 ? covariance / denominator : null;
}

function upperBound(values: readonly TimedReturn[], target: number): number {
  let low = 0;
  let high = values.length;
  while (low < high) {
    const middle = low + Math.floor((high - low) / 2);
    if (values[middle]!.endTime <= target) low = middle + 1;
    else high = middle;
  }
  return low;
}

function upperBoundNumbers(values: readonly number[], target: number): number {
  let low = 0;
  let high = values.length;
  while (low < high) {
    const middle = low + Math.floor((high - low) / 2);
    if (values[middle]! <= target) low = middle + 1;
    else high = middle;
  }
  return low;
}
