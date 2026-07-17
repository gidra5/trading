export interface ExposureValueOracleOptions {
  scoreStartIndex: number;
  friction: number;
  gridSize: number;
  temperature: number;
  minExposure?: number;
  maxExposure?: number;
  opportunityEpsilon?: number;
  quoteLendRate?: number;
  quoteBorrowRate?: number;
  assetBorrowRate?: number;
}

export interface ExposureValueOracle {
  scoreStartIndex: number;
  grid: Float64Array;
  means: Float32Array;
  secondMoments: Float32Array;
  entropies: Float32Array;
  weights: Float32Array;
  opportunities: Float32Array;
}

export interface ExposureValueDistillationAccumulator {
  weightedCrossEntropy: number;
  weightedOracleEntropy: number;
  weightSum: number;
  opportunitySum: number;
  sampleCount: number;
}

export interface ExposureValueDistillationMetrics extends ExposureValueDistillationAccumulator {
  crossEntropy: number;
  oracleEntropy: number;
  klDivergence: number;
  score: number;
  meanOpportunity: number;
}

interface HoldingOptions {
  friction: number;
  minExposure: number;
  maxExposure: number;
  quoteLendRate: number;
  quoteBorrowRate: number;
  assetBorrowRate: number;
}

const INVALID_VALUE_TEMPERATURES = 50;

/**
 * Builds p_t(a) ∝ exp(Q_t(a) / temperature) over a fixed exposure grid.
 *
 * Q_t(a) forces exposure a for the next price move, then follows the optimal
 * friction-aware policy. Portfolio value is homogeneous, so the Bellman state
 * needs only the current marked exposure rather than absolute equity.
 */
export function prepareExposureValueOracle(
  prices: ArrayLike<number>,
  options: ExposureValueOracleOptions,
  shared = false,
): ExposureValueOracle {
  validateOracleOptions(prices, options);
  const minExposure = options.minExposure ?? -1;
  const maxExposure = options.maxExposure ?? 1;
  const grid = float64(options.gridSize, shared);
  for (let index = 0; index < grid.length; index += 1) {
    grid[index] = minExposure + index / (grid.length - 1) * (maxExposure - minExposure);
  }
  const means = float32(prices.length, shared);
  const secondMoments = float32(prices.length, shared);
  const entropies = float32(prices.length, shared);
  const weights = float32(prices.length, shared);
  const opportunities = float32(prices.length, shared);
  const opportunityEpsilon = Math.max(0, options.opportunityEpsilon ?? 1e-6);
  const holdingOptions: HoldingOptions = {
    friction: Math.max(0, options.friction),
    minExposure,
    maxExposure,
    quoteLendRate: options.quoteLendRate ?? 0,
    quoteBorrowRate: options.quoteBorrowRate ?? 0,
    assetBorrowRate: options.assetBorrowRate ?? 0,
  };
  let continuation = new Float64Array(grid.length);
  const forcedValues = new Float64Array(grid.length);
  let nextContinuation = new Float64Array(grid.length);

  for (let time = prices.length - 1; time >= options.scoreStartIndex; time -= 1) {
    if (time === prices.length - 1) {
      forcedValues.fill(0);
    } else {
      const price = prices[time]!;
      const nextPrice = prices[time + 1]!;
      for (let exposureIndex = 0; exposureIndex < grid.length; exposureIndex += 1) {
        forcedValues[exposureIndex] = forcedExposureValue(
          grid[exposureIndex]!,
          price,
          nextPrice,
          continuation,
          grid,
          holdingOptions,
        );
      }
    }

    const statistics = preferenceStatistics(forcedValues, grid, options.temperature);
    means[time] = statistics.mean;
    secondMoments[time] = statistics.secondMoment;
    entropies[time] = statistics.entropy;
    opportunities[time] = statistics.opportunity;
    weights[time] = statistics.opportunity + opportunityEpsilon;

    if (time === prices.length - 1) continue;
    for (let currentIndex = 0; currentIndex < grid.length; currentIndex += 1) {
      let best = Number.NEGATIVE_INFINITY;
      for (let targetIndex = 0; targetIndex < grid.length; targetIndex += 1) {
        const rebalance = rebalanceEquityFactor(
          grid[currentIndex]!,
          grid[targetIndex]!,
          holdingOptions.friction,
        );
        const forced = forcedValues[targetIndex]!;
        if (rebalance <= 0 || !Number.isFinite(forced)) continue;
        best = Math.max(best, Math.log(rebalance) + forced);
      }
      nextContinuation[currentIndex] = best;
    }
    [continuation, nextContinuation] = [nextContinuation, continuation];
  }

  return {
    scoreStartIndex: options.scoreStartIndex,
    grid,
    means,
    secondMoments,
    entropies,
    weights,
    opportunities,
  };
}

export function shareExposureValueOracle(oracle: ExposureValueOracle): ExposureValueOracle {
  return {
    scoreStartIndex: oracle.scoreStartIndex,
    grid: sharedCopy(oracle.grid),
    means: sharedCopy(oracle.means),
    secondMoments: sharedCopy(oracle.secondMoments),
    entropies: sharedCopy(oracle.entropies),
    weights: sharedCopy(oracle.weights),
    opportunities: sharedCopy(oracle.opportunities),
  };
}

export function exposureValueOracleBytes(oracle: ExposureValueOracle): number {
  return oracle.grid.byteLength
    + oracle.means.byteLength
    + oracle.secondMoments.byteLength
    + oracle.entropies.byteLength
    + oracle.weights.byteLength
    + oracle.opportunities.byteLength;
}

export function createExposureValueDistillationAccumulator(): ExposureValueDistillationAccumulator {
  return {
    weightedCrossEntropy: 0,
    weightedOracleEntropy: 0,
    weightSum: 0,
    opportunitySum: 0,
    sampleCount: 0,
  };
}

export function observeExposureValueDistillation(
  accumulator: ExposureValueDistillationAccumulator,
  oracle: ExposureValueOracle,
  candleIndex: number,
  strategyExposure: number,
  strategySigma: number,
): void {
  if (candleIndex < oracle.scoreStartIndex || candleIndex >= oracle.means.length) return;
  if (!Number.isFinite(strategyExposure) || !Number.isFinite(strategySigma) || strategySigma <= 0) {
    throw new Error("Exposure value distillation requires finite exposure and positive strategy sigma.");
  }
  const center = clamp(strategyExposure, oracle.grid[0]!, oracle.grid[oracle.grid.length - 1]!);
  const inverseTwoVariance = 1 / (2 * strategySigma * strategySigma);
  let maximumLogWeight = Number.NEGATIVE_INFINITY;
  for (const exposure of oracle.grid) {
    maximumLogWeight = Math.max(maximumLogWeight, -((exposure - center) ** 2) * inverseTwoVariance);
  }
  let normalizer = 0;
  for (const exposure of oracle.grid) {
    normalizer += Math.exp(-((exposure - center) ** 2) * inverseTwoVariance - maximumLogWeight);
  }
  const logNormalizer = maximumLogWeight + Math.log(normalizer);
  const mean = oracle.means[candleIndex]!;
  const secondMoment = oracle.secondMoments[candleIndex]!;
  const expectedSquaredDistance = Math.max(0, secondMoment - 2 * center * mean + center * center);
  const crossEntropy = logNormalizer + expectedSquaredDistance * inverseTwoVariance;
  const weight = oracle.weights[candleIndex]!;
  accumulator.weightedCrossEntropy += weight * crossEntropy;
  accumulator.weightedOracleEntropy += weight * oracle.entropies[candleIndex]!;
  accumulator.weightSum += weight;
  accumulator.opportunitySum += oracle.opportunities[candleIndex]!;
  accumulator.sampleCount += 1;
}

export function finalizeExposureValueDistillation(
  accumulator: ExposureValueDistillationAccumulator,
): ExposureValueDistillationMetrics {
  const crossEntropy = accumulator.weightSum > 0
    ? accumulator.weightedCrossEntropy / accumulator.weightSum
    : 0;
  const oracleEntropy = accumulator.weightSum > 0
    ? accumulator.weightedOracleEntropy / accumulator.weightSum
    : 0;
  const klDivergence = Math.max(0, crossEntropy - oracleEntropy);
  return {
    ...accumulator,
    crossEntropy,
    oracleEntropy,
    klDivergence,
    score: Math.exp(-klDivergence),
    meanOpportunity: accumulator.sampleCount > 0
      ? accumulator.opportunitySum / accumulator.sampleCount
      : 0,
  };
}

export function rebalanceEquityFactor(fromExposure: number, toExposure: number, friction: number): number {
  const fee = Math.max(0, friction);
  const difference = toExposure - fromExposure;
  if (Math.abs(difference) <= Number.EPSILON) return 1;
  if (difference > 0) {
    const denominator = 1 - fee + fee * toExposure;
    if (denominator <= 0) return Number.NEGATIVE_INFINITY;
    return 1 - fee * difference / denominator;
  }
  const denominator = 1 - fee * toExposure;
  if (denominator <= 0) return Number.NEGATIVE_INFINITY;
  return 1 - fee * -difference / denominator;
}

function forcedExposureValue(
  exposure: number,
  price: number,
  nextPrice: number,
  continuation: Float64Array,
  grid: Float64Array,
  options: HoldingOptions,
): number {
  const quote = 1 - exposure;
  const asset = exposure / price;
  const maintainedQuote = quote >= 0
    ? quote * (1 + options.quoteLendRate)
    : quote * (1 + options.quoteBorrowRate);
  const maintainedAsset = asset >= 0 ? asset : asset * (1 + options.assetBorrowRate);
  const assetValue = maintainedAsset * nextPrice;
  const markedEquity = maintainedQuote + assetValue;
  const liquidatedAssetValue = assetValue >= 0
    ? assetValue * (1 - options.friction)
    : assetValue / Math.max(Number.EPSILON, 1 - options.friction);
  const liquidationEquity = maintainedQuote + liquidatedAssetValue;
  if (liquidationEquity <= 0 || !Number.isFinite(liquidationEquity)) {
    return Number.NEGATIVE_INFINITY;
  }
  const liquidationExposure = liquidatedAssetValue / liquidationEquity;
  if (liquidationExposure < options.minExposure || liquidationExposure > options.maxExposure) {
    return Math.log(liquidationEquity) + interpolate(continuation, grid, 0);
  }
  if (markedEquity <= 0 || !Number.isFinite(markedEquity)) return Number.NEGATIVE_INFINITY;
  const nextExposure = assetValue / markedEquity;
  return Math.log(markedEquity) + interpolate(continuation, grid, nextExposure);
}

function preferenceStatistics(
  values: Float64Array,
  grid: Float64Array,
  temperature: number,
): {
  mean: number;
  secondMoment: number;
  entropy: number;
  opportunity: number;
} {
  let maximum = Number.NEGATIVE_INFINITY;
  let minimum = Number.POSITIVE_INFINITY;
  let invalid = false;
  for (const value of values) {
    if (!Number.isFinite(value)) {
      invalid = true;
      continue;
    }
    maximum = Math.max(maximum, value);
    minimum = Math.min(minimum, value);
  }
  if (!Number.isFinite(maximum)) {
    return { mean: 0, secondMoment: 0, entropy: 0, opportunity: 0 };
  }
  let total = 0;
  let weighted = 0;
  let weightedSquared = 0;
  let weightedLogWeight = 0;
  for (let index = 0; index < values.length; index += 1) {
    const scaled = Number.isFinite(values[index])
      ? (values[index]! - maximum) / temperature
      : -INVALID_VALUE_TEMPERATURES;
    const weight = Math.exp(Math.max(-INVALID_VALUE_TEMPERATURES, scaled));
    const exposure = grid[index]!;
    total += weight;
    weighted += weight * exposure;
    weightedSquared += weight * exposure * exposure;
    weightedLogWeight += weight * Math.log(weight);
  }
  const entropy = Math.log(total) - weightedLogWeight / total;
  return {
    mean: weighted / total,
    secondMoment: weightedSquared / total,
    entropy,
    opportunity: Math.max(maximum - minimum, invalid ? temperature * INVALID_VALUE_TEMPERATURES : 0),
  };
}

function interpolate(values: Float64Array, grid: Float64Array, exposure: number): number {
  const position = clamp(
    (exposure - grid[0]!) / (grid[grid.length - 1]! - grid[0]!) * (grid.length - 1),
    0,
    grid.length - 1,
  );
  const lower = Math.floor(position);
  const fraction = position - lower;
  const left = values[lower]!;
  const right = values[Math.min(grid.length - 1, lower + 1)]!;
  if (!Number.isFinite(left)) return right;
  if (!Number.isFinite(right)) return left;
  return left * (1 - fraction) + right * fraction;
}

function validateOracleOptions(prices: ArrayLike<number>, options: ExposureValueOracleOptions): void {
  if (prices.length < 2) {
    throw new Error("Exposure value oracle requires at least two positive finite prices.");
  }
  for (let index = 0; index < prices.length; index += 1) {
    const price = prices[index]!;
    if (!Number.isFinite(price) || price <= 0) {
      throw new Error("Exposure value oracle requires at least two positive finite prices.");
    }
  }
  if (!Number.isInteger(options.scoreStartIndex)
    || options.scoreStartIndex < 0
    || options.scoreStartIndex >= prices.length) {
    throw new Error("Exposure value oracle score start is outside the price series.");
  }
  if (!Number.isInteger(options.gridSize) || options.gridSize < 3) {
    throw new Error("Exposure value oracle grid size must be an integer of at least three.");
  }
  const minimum = options.minExposure ?? -1;
  const maximum = options.maxExposure ?? 1;
  if (!Number.isFinite(minimum) || !Number.isFinite(maximum) || minimum >= 0 || maximum <= 0) {
    throw new Error("Exposure value oracle bounds must contain zero.");
  }
  const finiteNonNegative = [
    options.friction,
    options.opportunityEpsilon ?? 0,
    options.quoteLendRate ?? 0,
    options.quoteBorrowRate ?? 0,
    options.assetBorrowRate ?? 0,
  ];
  if (finiteNonNegative.some((value) => !Number.isFinite(value) || value < 0)) {
    throw new Error("Exposure value oracle costs and rates must be finite and non-negative.");
  }
  if (options.friction >= 1) {
    throw new Error("Exposure value oracle friction must be less than one.");
  }
  if (!Number.isFinite(options.temperature) || options.temperature <= 0) {
    throw new Error("Exposure value oracle temperature must be positive.");
  }
}

function float32(length: number, shared: boolean): Float32Array {
  return shared
    ? new Float32Array(new SharedArrayBuffer(length * Float32Array.BYTES_PER_ELEMENT))
    : new Float32Array(length);
}

function float64(length: number, shared: boolean): Float64Array {
  return shared
    ? new Float64Array(new SharedArrayBuffer(length * Float64Array.BYTES_PER_ELEMENT))
    : new Float64Array(length);
}

function sharedCopy<T extends Float32Array | Float64Array>(values: T): T {
  const buffer = new SharedArrayBuffer(values.byteLength);
  const copy = (values instanceof Float32Array
    ? new Float32Array(buffer)
    : new Float64Array(buffer)) as T;
  copy.set(values);
  return copy;
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}
