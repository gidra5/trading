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
  includeProbabilities?: boolean;
}

export interface ExposureValueOracle {
  scoreStartIndex: number;
  grid: Float64Array;
  means: Float32Array;
  secondMoments: Float32Array;
  modalExposures: Float32Array;
  optimalExposures: Float32Array;
  entropies: Float32Array;
  weights: Float32Array;
  opportunities: Float32Array;
  probabilities?: Float32Array;
  execution: ExposureExecutionOptions;
}

export interface ExposureExecutionOptions {
  friction: number;
  minExposure: number;
  maxExposure: number;
  quoteLendRate: number;
  quoteBorrowRate: number;
  assetBorrowRate: number;
}

export interface ExposureReturnAccumulator {
  equity: number;
  exposure: number;
  peakEquity: number;
  maxDrawdown: number;
  turnover: number;
  rebalanceCount: number;
  liquidationCount: number;
  sampleCount: number;
}

export interface ExposureReturnMetrics extends ExposureReturnAccumulator {
  totalReturn: number;
  logReturn: number;
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
  const modalExposures = float32(prices.length, shared);
  const optimalExposures = float32(prices.length, shared);
  const entropies = float32(prices.length, shared);
  const weights = float32(prices.length, shared);
  const opportunities = float32(prices.length, shared);
  const probabilities = options.includeProbabilities
    ? float32(prices.length * grid.length, shared)
    : undefined;
  const opportunityEpsilon = Math.max(0, options.opportunityEpsilon ?? 1e-6);
  const execution: ExposureExecutionOptions = {
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
  const policy = new Uint16Array(prices.length * grid.length);

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
          execution,
        );
      }
    }

    const probabilityRow = probabilities?.subarray(time * grid.length, (time + 1) * grid.length);
    const statistics = preferenceStatistics(forcedValues, grid, options.temperature, probabilityRow);
    means[time] = statistics.mean;
    secondMoments[time] = statistics.secondMoment;
    modalExposures[time] = statistics.optimalExposure;
    entropies[time] = statistics.entropy;
    opportunities[time] = statistics.opportunity;
    weights[time] = statistics.opportunity + opportunityEpsilon;

    if (time === prices.length - 1) {
      for (let currentIndex = 0; currentIndex < grid.length; currentIndex += 1) {
        policy[time * grid.length + currentIndex] = currentIndex;
      }
      continue;
    }
    for (let currentIndex = 0; currentIndex < grid.length; currentIndex += 1) {
      let best = Number.NEGATIVE_INFINITY;
      let bestTargetIndex = currentIndex;
      for (let targetIndex = 0; targetIndex < grid.length; targetIndex += 1) {
        const rebalance = rebalanceEquityFactor(
          grid[currentIndex]!,
          grid[targetIndex]!,
          execution.friction,
        );
        const forced = forcedValues[targetIndex]!;
        if (rebalance <= 0 || !Number.isFinite(forced)) continue;
        const value = Math.log(rebalance) + forced;
        if (value > best) {
          best = value;
          bestTargetIndex = targetIndex;
        }
      }
      nextContinuation[currentIndex] = best;
      policy[time * grid.length + currentIndex] = bestTargetIndex;
    }
    [continuation, nextContinuation] = [nextContinuation, continuation];
  }

  let currentExposure = 0;
  for (let time = options.scoreStartIndex; time < prices.length; time += 1) {
    const currentIndex = Math.round(clamp(
      (currentExposure - grid[0]!) / (grid[grid.length - 1]! - grid[0]!) * (grid.length - 1),
      0,
      grid.length - 1,
    ));
    const target = grid[policy[time * grid.length + currentIndex]!]!;
    optimalExposures[time] = target;
    if (time + 1 >= prices.length) continue;
    const rebalance = rebalanceEquityFactor(currentExposure, target, execution.friction);
    currentExposure = rebalance > 0
      ? holdingOutcome(target, prices[time]!, prices[time + 1]!, execution).exposure
      : 0;
  }

  return {
    scoreStartIndex: options.scoreStartIndex,
    grid,
    means,
    secondMoments,
    modalExposures,
    optimalExposures,
    entropies,
    weights,
    opportunities,
    probabilities,
    execution,
  };
}

export function shareExposureValueOracle(oracle: ExposureValueOracle): ExposureValueOracle {
  return {
    scoreStartIndex: oracle.scoreStartIndex,
    grid: sharedCopy(oracle.grid),
    means: sharedCopy(oracle.means),
    secondMoments: sharedCopy(oracle.secondMoments),
    modalExposures: sharedCopy(oracle.modalExposures),
    optimalExposures: sharedCopy(oracle.optimalExposures),
    entropies: sharedCopy(oracle.entropies),
    weights: sharedCopy(oracle.weights),
    opportunities: sharedCopy(oracle.opportunities),
    ...(oracle.probabilities ? { probabilities: sharedCopy(oracle.probabilities) } : {}),
    execution: { ...oracle.execution },
  };
}

export function exposureValueOracleBytes(oracle: ExposureValueOracle): number {
  return oracle.grid.byteLength
    + oracle.means.byteLength
    + oracle.secondMoments.byteLength
    + oracle.modalExposures.byteLength
    + oracle.optimalExposures.byteLength
    + oracle.entropies.byteLength
    + oracle.weights.byteLength
    + oracle.opportunities.byteLength
    + (oracle.probabilities?.byteLength ?? 0);
}

export function exposureValueOracleProbabilities(
  oracle: ExposureValueOracle,
  candleIndex: number,
): Float32Array {
  if (!oracle.probabilities) {
    throw new Error("Exposure value oracle probabilities were not retained.");
  }
  if (candleIndex < 0 || candleIndex >= oracle.means.length) {
    throw new Error("Exposure value oracle probability index is outside the price series.");
  }
  return oracle.probabilities.subarray(
    candleIndex * oracle.grid.length,
    (candleIndex + 1) * oracle.grid.length,
  );
}

export function strategyExposureProbabilities(
  grid: ArrayLike<number>,
  strategyExposure: number,
  strategySigma: number,
): Float64Array {
  if (grid.length < 2 || !Number.isFinite(strategyExposure)
    || !Number.isFinite(strategySigma) || strategySigma <= 0) {
    throw new Error("Strategy exposure distribution requires a grid, finite exposure, and positive sigma.");
  }
  const center = clamp(strategyExposure, grid[0]!, grid[grid.length - 1]!);
  const inverseTwoVariance = 1 / (2 * strategySigma * strategySigma);
  const probabilities = new Float64Array(grid.length);
  let maximum = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < grid.length; index += 1) {
    const logWeight = -((grid[index]! - center) ** 2) * inverseTwoVariance;
    probabilities[index] = logWeight;
    maximum = Math.max(maximum, logWeight);
  }
  let total = 0;
  for (let index = 0; index < probabilities.length; index += 1) {
    const weight = Math.exp(probabilities[index]! - maximum);
    probabilities[index] = weight;
    total += weight;
  }
  for (let index = 0; index < probabilities.length; index += 1) {
    probabilities[index] /= total;
  }
  return probabilities;
}

export function createExposureReturnAccumulator(): ExposureReturnAccumulator {
  return {
    equity: 1,
    exposure: 0,
    peakEquity: 1,
    maxDrawdown: 0,
    turnover: 0,
    rebalanceCount: 0,
    liquidationCount: 0,
    sampleCount: 0,
  };
}

export function observeExposureReturn(
  accumulator: ExposureReturnAccumulator,
  targetExposure: number,
  price: number,
  nextPrice: number,
  options: ExposureExecutionOptions,
): void {
  if (![targetExposure, price, nextPrice].every(Number.isFinite) || price <= 0 || nextPrice <= 0) {
    throw new Error("Exposure return requires finite exposure and positive prices.");
  }
  if (!(accumulator.equity > 0)) return;
  const target = clamp(targetExposure, options.minExposure, options.maxExposure);
  const change = Math.abs(target - accumulator.exposure);
  if (change > Number.EPSILON) {
    const rebalance = rebalanceEquityFactor(accumulator.exposure, target, options.friction);
    if (!(rebalance > 0)) {
      accumulator.equity = 0;
      accumulator.maxDrawdown = 1;
      return;
    }
    accumulator.equity *= rebalance;
    accumulator.turnover += change;
    accumulator.rebalanceCount += 1;
    updateDrawdown(accumulator);
  }
  const holding = holdingOutcome(target, price, nextPrice, options);
  accumulator.equity *= holding.equityFactor;
  accumulator.exposure = holding.exposure;
  accumulator.liquidationCount += holding.liquidated ? 1 : 0;
  accumulator.sampleCount += 1;
  updateDrawdown(accumulator);
}

export function finalizeExposureReturn(
  accumulator: ExposureReturnAccumulator,
): ExposureReturnMetrics {
  return {
    ...accumulator,
    totalReturn: accumulator.equity - 1,
    logReturn: accumulator.equity > 0 ? Math.log(accumulator.equity) : Number.NEGATIVE_INFINITY,
  };
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
  options: ExposureExecutionOptions,
): number {
  const outcome = holdingOutcome(exposure, price, nextPrice, options);
  if (!(outcome.equityFactor > 0)) return Number.NEGATIVE_INFINITY;
  return Math.log(outcome.equityFactor) + interpolate(continuation, grid, outcome.exposure);
}

function preferenceStatistics(
  values: Float64Array,
  grid: Float64Array,
  temperature: number,
  probabilities?: Float32Array,
): {
  mean: number;
  secondMoment: number;
  optimalExposure: number;
  entropy: number;
  opportunity: number;
} {
  let maximum = Number.NEGATIVE_INFINITY;
  let minimum = Number.POSITIVE_INFINITY;
  let invalid = false;
  let maximumIndex = 0;
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index]!;
    if (!Number.isFinite(value)) {
      invalid = true;
      continue;
    }
    if (value > maximum) {
      maximum = value;
      maximumIndex = index;
    }
    minimum = Math.min(minimum, value);
  }
  if (!Number.isFinite(maximum)) {
    return { mean: 0, secondMoment: 0, optimalExposure: 0, entropy: 0, opportunity: 0 };
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
    if (probabilities) probabilities[index] = weight;
    const exposure = grid[index]!;
    total += weight;
    weighted += weight * exposure;
    weightedSquared += weight * exposure * exposure;
    weightedLogWeight += weight * Math.log(weight);
  }
  const entropy = Math.log(total) - weightedLogWeight / total;
  if (probabilities) {
    for (let index = 0; index < probabilities.length; index += 1) probabilities[index] /= total;
  }
  return {
    mean: weighted / total,
    secondMoment: weightedSquared / total,
    optimalExposure: grid[maximumIndex]!,
    entropy,
    opportunity: Math.max(maximum - minimum, invalid ? temperature * INVALID_VALUE_TEMPERATURES : 0),
  };
}

function holdingOutcome(
  exposure: number,
  price: number,
  nextPrice: number,
  options: ExposureExecutionOptions,
): { equityFactor: number; exposure: number; liquidated: boolean } {
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
  if (!(liquidationEquity > 0) || !Number.isFinite(liquidationEquity)) {
    return { equityFactor: 0, exposure: 0, liquidated: true };
  }
  const liquidationExposure = liquidatedAssetValue / liquidationEquity;
  if (liquidationExposure < options.minExposure || liquidationExposure > options.maxExposure) {
    return { equityFactor: liquidationEquity, exposure: 0, liquidated: true };
  }
  if (!(markedEquity > 0) || !Number.isFinite(markedEquity)) {
    return { equityFactor: 0, exposure: 0, liquidated: true };
  }
  return { equityFactor: markedEquity, exposure: assetValue / markedEquity, liquidated: false };
}

function updateDrawdown(accumulator: ExposureReturnAccumulator): void {
  accumulator.peakEquity = Math.max(accumulator.peakEquity, accumulator.equity);
  accumulator.maxDrawdown = Math.max(
    accumulator.maxDrawdown,
    accumulator.peakEquity > 0 ? 1 - accumulator.equity / accumulator.peakEquity : 1,
  );
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
  if (!Number.isInteger(options.gridSize) || options.gridSize < 3 || options.gridSize > 65_535) {
    throw new Error("Exposure value oracle grid size must be an integer from three to 65,535.");
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
