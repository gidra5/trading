import type { ReturnDistribution } from "./log-return-distribution.js";

interface Centroid {
  mean: number;
  weight: number;
}

interface Moments {
  count: number;
  mean: number;
  m2: number;
  m3: number;
  m4: number;
}

interface PairSums {
  count: number;
  sumLeft: number;
  sumRight: number;
  sumLeftSquared: number;
  sumRightSquared: number;
  sumProduct: number;
}

export interface ReturnDistributionChunk {
  startTime: number;
  endTime: number;
  moments: Moments;
  meanAbsoluteSum: number;
  positiveCount: number;
  zeroCount: number;
  returnPairs: PairSums;
  absoluteReturnPairs: PairSums;
  firstReturn: number;
  lastReturn: number;
  centroids: Centroid[];
}

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
 * Summarize a contiguous return chunk. Moments and counts remain exact; the
 * sorted returns are compressed into tail-dense weighted centroids for bounded-
 * memory quantiles and standardized probability estimates.
 */
export function buildReturnDistributionChunk(
  values: Float64Array,
  startTime: number,
  endTime: number,
  compression = 300,
): ReturnDistributionChunk {
  if (values.length < 1 || !(endTime > startTime)) {
    throw new Error("A return chunk needs values and a positive time range.");
  }
  const moments = emptyMoments();
  const returnPairs = emptyPairs();
  const absoluteReturnPairs = emptyPairs();
  let meanAbsoluteSum = 0;
  let positiveCount = 0;
  let zeroCount = 0;
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index]!;
    if (!Number.isFinite(value)) throw new Error("A return chunk contains a non-finite value.");
    addMoment(moments, value);
    meanAbsoluteSum += Math.abs(value);
    if (value > 0) positiveCount += 1;
    if (value === 0) zeroCount += 1;
    if (index > 0) {
      const previous = values[index - 1]!;
      addPair(returnPairs, previous, value);
      addPair(absoluteReturnPairs, Math.abs(previous), Math.abs(value));
    }
  }
  const firstReturn = values[0]!;
  const lastReturn = values.at(-1)!;
  values.sort();
  return {
    startTime,
    endTime,
    moments,
    meanAbsoluteSum,
    positiveCount,
    zeroCount,
    returnPairs,
    absoluteReturnPairs,
    firstReturn,
    lastReturn,
    centroids: centroidsFromSorted(values, compression),
  };
}

export class StreamingReturnDistribution {
  private readonly moments = emptyMoments();
  private readonly returnPairs = emptyPairs();
  private readonly absoluteReturnPairs = emptyPairs();
  private readonly sketch: WeightedQuantileSketch;
  private meanAbsoluteSum = 0;
  private positiveCount = 0;
  private zeroCount = 0;
  private lastChunkEnd = Number.NaN;
  private lastReturn = Number.NaN;

  constructor(compression = 800) {
    this.sketch = new WeightedQuantileSketch(compression);
  }

  merge(chunk: ReturnDistributionChunk): void {
    if (chunk.moments.count < 1) return;
    if (this.moments.count > 0 && this.lastChunkEnd === chunk.startTime) {
      addPair(this.returnPairs, this.lastReturn, chunk.firstReturn);
      addPair(
        this.absoluteReturnPairs,
        Math.abs(this.lastReturn),
        Math.abs(chunk.firstReturn),
      );
    }
    mergeMoments(this.moments, chunk.moments);
    mergePairs(this.returnPairs, chunk.returnPairs);
    mergePairs(this.absoluteReturnPairs, chunk.absoluteReturnPairs);
    this.meanAbsoluteSum += chunk.meanAbsoluteSum;
    this.positiveCount += chunk.positiveCount;
    this.zeroCount += chunk.zeroCount;
    this.sketch.addCentroids(chunk.centroids);
    this.lastChunkEnd = chunk.endTime;
    this.lastReturn = chunk.lastReturn;
  }

  summarize(): ReturnDistribution {
    this.sketch.compress();
    const observations = this.moments.count;
    if (observations < 2) throw new Error("At least two streamed returns are required.");
    const mean = this.moments.mean;
    const sigma = Math.sqrt(this.moments.m2 / (observations - 1));
    const median = this.sketch.quantile(0.5);
    const absoluteFromMedian = this.sketch.transformed(
      (value) => Math.abs(value - median),
    );
    const absoluteFromMean = this.sketch.transformed(
      (value) => Math.abs(value - mean),
    );
    const robustSigma = absoluteFromMedian.quantile(0.5) * 1.4826;
    const p25 = this.sketch.quantile(0.25);
    const p75 = this.sketch.quantile(0.75);
    const centralMass = sigma > 0
      ? probabilityBetween(this.sketch, mean - 0.25 * sigma, mean + 0.25 * sigma)
      : null;
    const tailMass3 = sigma > 0
      ? 1 - probabilityBetween(this.sketch, mean - 3 * sigma, mean + 3 * sigma)
      : null;
    const downsideP01 = Math.abs(this.sketch.quantile(0.01) - mean);
    const upsideP99 = Math.abs(this.sketch.quantile(0.99) - mean);
    const skewness = observations > 2 && sigma > 0
      ? observations / ((observations - 1) * (observations - 2))
        * this.moments.m3 / (sigma ** 3)
      : null;
    const excessKurtosis = observations > 3 && sigma > 0
      ? observations * (observations + 1)
        / ((observations - 1) * (observations - 2) * (observations - 3))
        * this.moments.m4 / (sigma ** 4)
        - 3 * (observations - 1) ** 2
        / ((observations - 2) * (observations - 3))
      : null;

    return {
      observations,
      meanBps: mean * 10_000,
      medianBps: median * 10_000,
      meanAbsoluteBps: this.meanAbsoluteSum / observations * 10_000,
      standardDeviationBps: sigma * 10_000,
      robustSigmaBps: robustSigma * 10_000,
      skewness,
      excessKurtosis,
      positiveFraction: this.positiveCount / observations,
      zeroFraction: this.zeroCount / observations,
      lag1ReturnCorrelation: pairCorrelation(this.returnPairs),
      lag1AbsoluteReturnCorrelation: pairCorrelation(this.absoluteReturnPairs),
      standardDeviationToRobustSigma: robustSigma > 0 ? sigma / robustSigma : null,
      iqrToGaussianIqr: sigma > 0
        ? (p75 - p25) / (GAUSSIAN_IQR_SIGMA * sigma)
        : null,
      centralMass025Sigma: centralMass,
      centralMassVsGaussian: centralMass === null
        ? null
        : centralMass / GAUSSIAN_CENTRAL_MASS_025,
      tailMass3Sigma: tailMass3,
      tailMass3SigmaVsGaussian: tailMass3 === null
        ? null
        : tailMass3 / GAUSSIAN_TAIL_MASS_3,
      absoluteP99Sigma: sigma > 0 ? absoluteFromMean.quantile(0.99) / sigma : null,
      absoluteP999Sigma: sigma > 0 ? absoluteFromMean.quantile(0.999) / sigma : null,
      downsideToUpsideP99: upsideP99 > 0 ? downsideP01 / upsideP99 : null,
      quantilesBps: {
        p001: this.sketch.quantile(0.001) * 10_000,
        p01: this.sketch.quantile(0.01) * 10_000,
        p05: this.sketch.quantile(0.05) * 10_000,
        p25: p25 * 10_000,
        p50: median * 10_000,
        p75: p75 * 10_000,
        p95: this.sketch.quantile(0.95) * 10_000,
        p99: this.sketch.quantile(0.99) * 10_000,
        p999: this.sketch.quantile(0.999) * 10_000,
      },
      absoluteTail: TAIL_THRESHOLDS.map((threshold) => ({
        sigma: threshold,
        probability: sigma > 0
          ? 1 - probabilityBetween(
            this.sketch,
            mean - threshold * sigma,
            mean + threshold * sigma,
          )
          : 0,
      })),
    };
  }
}

class WeightedQuantileSketch {
  private centroids: Centroid[] = [];
  private pending: Centroid[] = [];

  constructor(private readonly compression: number) {
    if (!Number.isFinite(compression) || compression < 20) {
      throw new Error("Quantile-sketch compression must be at least 20.");
    }
  }

  addCentroids(values: readonly Centroid[]): void {
    this.pending.push(...values);
    if (this.pending.length >= this.compression * 8) this.compress();
  }

  compress(): void {
    if (this.pending.length === 0) return;
    const sorted = [...this.centroids, ...this.pending]
      .sort((left, right) => left.mean - right.mean);
    this.pending = [];
    this.centroids = compressCentroids(sorted, this.compression);
  }

  quantile(probability: number): number {
    this.compress();
    if (this.centroids.length === 0) return Number.NaN;
    if (this.centroids.length === 1) return this.centroids[0]!.mean;
    const total = totalWeight(this.centroids);
    const target = Math.max(0, Math.min(1, probability)) * total;
    let cumulative = 0;
    for (let index = 0; index < this.centroids.length - 1; index += 1) {
      const current = this.centroids[index]!;
      const next = this.centroids[index + 1]!;
      const currentMidpoint = cumulative + current.weight / 2;
      const nextMidpoint = cumulative + current.weight + next.weight / 2;
      if (target <= currentMidpoint) return current.mean;
      if (target <= nextMidpoint) {
        const fraction = (target - currentMidpoint) / (nextMidpoint - currentMidpoint);
        return current.mean + fraction * (next.mean - current.mean);
      }
      cumulative += current.weight;
    }
    return this.centroids.at(-1)!.mean;
  }

  cdf(value: number): number {
    this.compress();
    if (this.centroids.length === 0) return Number.NaN;
    const total = totalWeight(this.centroids);
    if (value < this.centroids[0]!.mean) return 0;
    if (value >= this.centroids.at(-1)!.mean) return 1;
    let cumulative = 0;
    for (let index = 0; index < this.centroids.length - 1; index += 1) {
      const current = this.centroids[index]!;
      const next = this.centroids[index + 1]!;
      if (value <= next.mean) {
        const leftCdf = (cumulative + current.weight / 2) / total;
        const rightCdf = (cumulative + current.weight + next.weight / 2) / total;
        const fraction = (value - current.mean) / (next.mean - current.mean);
        return leftCdf + Math.max(0, Math.min(1, fraction)) * (rightCdf - leftCdf);
      }
      cumulative += current.weight;
    }
    return 1;
  }

  transformed(transform: (value: number) => number): WeightedQuantileSketch {
    this.compress();
    const output = new WeightedQuantileSketch(this.compression);
    output.addCentroids(this.centroids.map((centroid) => ({
      mean: transform(centroid.mean),
      weight: centroid.weight,
    })));
    output.compress();
    return output;
  }
}

function centroidsFromSorted(
  sorted: Float64Array,
  compression: number,
): Centroid[] {
  return compressCentroids(
    Array.from(sorted, (mean) => ({ mean, weight: 1 })),
    compression,
  );
}

function compressCentroids(
  sortedInput: readonly Centroid[],
  compression: number,
): Centroid[] {
  if (sortedInput.length === 0) return [];
  const sorted = isSorted(sortedInput)
    ? sortedInput
    : [...sortedInput].sort((left, right) => left.mean - right.mean);
  const total = totalWeight(sorted);
  const output: Centroid[] = [];
  let cumulative = 0;
  let current = { ...sorted[0]! };
  for (let index = 1; index < sorted.length; index += 1) {
    const next = sorted[index]!;
    const proposedWeight = current.weight + next.weight;
    const quantile = (cumulative + proposedWeight / 2) / total;
    const maximumWeight = Math.max(
      1,
      4 * total * quantile * (1 - quantile) / compression,
    );
    if (proposedWeight <= maximumWeight) {
      current.mean = (
        current.mean * current.weight + next.mean * next.weight
      ) / proposedWeight;
      current.weight = proposedWeight;
    } else {
      output.push(current);
      cumulative += current.weight;
      current = { ...next };
    }
  }
  output.push(current);
  return output;
}

function isSorted(values: readonly Centroid[]): boolean {
  for (let index = 1; index < values.length; index += 1) {
    if (values[index]!.mean < values[index - 1]!.mean) return false;
  }
  return true;
}

function totalWeight(centroids: readonly Centroid[]): number {
  return centroids.reduce((total, centroid) => total + centroid.weight, 0);
}

function probabilityBetween(
  sketch: WeightedQuantileSketch,
  lower: number,
  upper: number,
): number {
  return Math.max(0, Math.min(1, sketch.cdf(upper) - sketch.cdf(lower)));
}

function emptyMoments(): Moments {
  return { count: 0, mean: 0, m2: 0, m3: 0, m4: 0 };
}

function addMoment(state: Moments, value: number): void {
  const previousCount = state.count;
  state.count += 1;
  const delta = value - state.mean;
  const deltaOverCount = delta / state.count;
  const deltaOverCount2 = deltaOverCount * deltaOverCount;
  const term = delta * deltaOverCount * previousCount;
  state.m4 += term * deltaOverCount2
    * (state.count * state.count - 3 * state.count + 3)
    + 6 * deltaOverCount2 * state.m2
    - 4 * deltaOverCount * state.m3;
  state.m3 += term * deltaOverCount * (state.count - 2)
    - 3 * deltaOverCount * state.m2;
  state.m2 += term;
  state.mean += deltaOverCount;
}

function mergeMoments(target: Moments, source: Moments): void {
  if (source.count === 0) return;
  if (target.count === 0) {
    Object.assign(target, source);
    return;
  }
  const leftCount = target.count;
  const rightCount = source.count;
  const count = leftCount + rightCount;
  const delta = source.mean - target.mean;
  const delta2 = delta * delta;
  const delta3 = delta2 * delta;
  const delta4 = delta3 * delta;
  const leftM2 = target.m2;
  const leftM3 = target.m3;
  const rightM2 = source.m2;
  const rightM3 = source.m3;
  target.m4 += source.m4
    + delta4 * leftCount * rightCount
      * (leftCount * leftCount - leftCount * rightCount + rightCount * rightCount)
      / (count * count * count)
    + 6 * delta2 * (leftCount * leftCount * rightM2 + rightCount * rightCount * leftM2)
      / (count * count)
    + 4 * delta * (leftCount * rightM3 - rightCount * leftM3) / count;
  target.m3 += source.m3
    + delta3 * leftCount * rightCount * (leftCount - rightCount) / (count * count)
    + 3 * delta * (leftCount * rightM2 - rightCount * leftM2) / count;
  target.m2 += rightM2 + delta2 * leftCount * rightCount / count;
  target.mean += delta * rightCount / count;
  target.count = count;
}

function emptyPairs(): PairSums {
  return {
    count: 0,
    sumLeft: 0,
    sumRight: 0,
    sumLeftSquared: 0,
    sumRightSquared: 0,
    sumProduct: 0,
  };
}

function addPair(state: PairSums, left: number, right: number): void {
  state.count += 1;
  state.sumLeft += left;
  state.sumRight += right;
  state.sumLeftSquared += left * left;
  state.sumRightSquared += right * right;
  state.sumProduct += left * right;
}

function mergePairs(target: PairSums, source: PairSums): void {
  target.count += source.count;
  target.sumLeft += source.sumLeft;
  target.sumRight += source.sumRight;
  target.sumLeftSquared += source.sumLeftSquared;
  target.sumRightSquared += source.sumRightSquared;
  target.sumProduct += source.sumProduct;
}

function pairCorrelation(state: PairSums): number | null {
  if (state.count < 2) return null;
  const covariance = state.sumProduct - state.sumLeft * state.sumRight / state.count;
  const leftVariance = state.sumLeftSquared - state.sumLeft ** 2 / state.count;
  const rightVariance = state.sumRightSquared - state.sumRight ** 2 / state.count;
  const denominator = Math.sqrt(leftVariance * rightVariance);
  return denominator > 0 ? covariance / denominator : null;
}
