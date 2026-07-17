export interface ExposureValueOracleOptions {
  scoreStartIndex: number;
  horizonSteps?: number;
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
  horizonSteps: number;
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
const HOUR_MS = 3_600_000;

export interface StrategyExposureTemperatureOptions {
  intervalMs: number;
  horizonSteps: number;
  temperature: number;
  scaleByVolatility?: boolean;
}

/**
 * Builds p_t(a) ∝ exp(Q_t(a) / temperature) over a fixed exposure grid.
 *
 * Q_t(a) keeps target exposure a for H price moves, then follows the optimal
 * friction-aware policy. Keeping the target fixed includes the intermediate
 * rebalances needed to undo price drift. Portfolio value is homogeneous, so
 * the Bellman state needs only the current marked exposure rather than equity.
 */
export function prepareExposureValueOracle(
  prices: ArrayLike<number>,
  options: ExposureValueOracleOptions,
  shared = false,
): ExposureValueOracle {
  const oracle = createExposureValueOracleStorage(prices, options, shared);
  const {
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
  } = oracle;
  const horizonSteps = oracle.horizonSteps;
  const opportunityEpsilon = Math.max(0, options.opportunityEpsilon ?? 1e-6);
  const forcedValues = new Float64Array(grid.length);
  const continuationRing = Array.from(
    { length: Math.min(prices.length, horizonSteps + 1) },
    () => new Float64Array(grid.length),
  );
  const rollingValues = new Float64Array(grid.length);
  const rollingInvalidCounts = new Int32Array(grid.length);
  const policy = new Uint16Array(prices.length * grid.length);

  for (let time = prices.length - 1; time >= options.scoreStartIndex; time -= 1) {
    if (time === prices.length - 1) {
      forcedValues.fill(0);
    } else {
      const endpointTime = Math.min(prices.length - 1, time + horizonSteps);
      const endpointContinuation = continuationRing[
        endpointTime % continuationRing.length
      ]!;
      const endpointMove = endpointTime - 1;
      for (let exposureIndex = 0; exposureIndex < grid.length; exposureIndex += 1) {
        const exposure = grid[exposureIndex]!;
        let rolling = rollingValues[exposureIndex]!;
        let invalidCount = rollingInvalidCounts[exposureIndex]!;
        if (time + 1 < prices.length - 1) {
          const removedFirst = holdingLogValue(exposure, time + 1, prices, execution);
          if (Number.isFinite(removedFirst)) rolling -= removedFirst;
          else invalidCount -= 1;
          const addedContinuation = continuedHoldingLogValue(
            exposure,
            time + 1,
            prices,
            execution,
          );
          if (Number.isFinite(addedContinuation)) rolling += addedContinuation;
          else invalidCount += 1;
        }
        const removedTime = time + horizonSteps;
        if (removedTime < prices.length - 1) {
          const removedContinuation = continuedHoldingLogValue(
            exposure,
            removedTime,
            prices,
            execution,
          );
          if (Number.isFinite(removedContinuation)) rolling -= removedContinuation;
          else invalidCount -= 1;
        }
        const addedFirst = holdingLogValue(exposure, time, prices, execution);
        if (Number.isFinite(addedFirst)) rolling += addedFirst;
        else invalidCount += 1;
        rollingValues[exposureIndex] = rolling;
        rollingInvalidCounts[exposureIndex] = invalidCount;

        const endpointOutcome = holdingOutcome(
          exposure,
          prices[endpointMove]!,
          prices[endpointTime]!,
          execution,
        );
        forcedValues[exposureIndex] = invalidCount === 0
          ? rolling + interpolate(endpointContinuation, grid, endpointOutcome.exposure)
          : Number.NEGATIVE_INFINITY;
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

    const timeContinuation = continuationRing[time % continuationRing.length]!;
    if (time === prices.length - 1) {
      timeContinuation.fill(0);
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
      timeContinuation[currentIndex] = best;
      policy[time * grid.length + currentIndex] = bestTargetIndex;
    }
  }

  let currentExposure = 0;
  let target = 0;
  let remainingHoldSteps = 0;
  for (let time = options.scoreStartIndex; time < prices.length; time += 1) {
    if (remainingHoldSteps === 0) {
      const currentIndex = Math.round(clamp(
        (currentExposure - grid[0]!) / (grid[grid.length - 1]! - grid[0]!) * (grid.length - 1),
        0,
        grid.length - 1,
      ));
      target = grid[policy[time * grid.length + currentIndex]!]!;
      remainingHoldSteps = Math.min(horizonSteps, prices.length - 1 - time);
    }
    optimalExposures[time] = target;
    if (time + 1 >= prices.length) continue;
    const rebalance = rebalanceEquityFactor(currentExposure, target, execution.friction);
    currentExposure = rebalance > 0
      ? holdingOutcome(target, prices[time]!, prices[time + 1]!, execution).exposure
      : 0;
    remainingHoldSteps -= 1;
  }

  return oracle;
}

/** Allocate and validate an oracle result without running the Bellman recurrence. */
export function createExposureValueOracleStorage(
  prices: ArrayLike<number>,
  options: ExposureValueOracleOptions,
  shared = false,
): ExposureValueOracle {
  validateExposureValueOracleOptions(prices, options);
  const minExposure = options.minExposure ?? -1;
  const maxExposure = options.maxExposure ?? 1;
  // Retain the selected H for temperature scaling and reporting. The Bellman
  // recurrence truncates each hold at the available segment tail itself.
  const horizonSteps = options.horizonSteps ?? 1;
  const grid = float64(options.gridSize, shared);
  for (let index = 0; index < grid.length; index += 1) {
    grid[index] = minExposure + index / (grid.length - 1) * (maxExposure - minExposure);
  }
  return {
    scoreStartIndex: options.scoreStartIndex,
    horizonSteps,
    grid,
    means: float32(prices.length, shared),
    secondMoments: float32(prices.length, shared),
    modalExposures: float32(prices.length, shared),
    optimalExposures: float32(prices.length, shared),
    entropies: float32(prices.length, shared),
    weights: float32(prices.length, shared),
    opportunities: float32(prices.length, shared),
    ...(options.includeProbabilities
      ? { probabilities: float32(prices.length * grid.length, shared) }
      : {}),
    execution: {
      friction: Math.max(0, options.friction),
      minExposure,
      maxExposure,
      quoteLendRate: options.quoteLendRate ?? 0,
      quoteBorrowRate: options.quoteBorrowRate ?? 0,
      assetBorrowRate: options.assetBorrowRate ?? 0,
    },
  };
}

export function shareExposureValueOracle(oracle: ExposureValueOracle): ExposureValueOracle {
  return {
    scoreStartIndex: oracle.scoreStartIndex,
    horizonSteps: oracle.horizonSteps,
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
  rateBpsPerHour: number,
  horizonMs: number,
  temperature: number,
): Float64Array {
  if (grid.length < 2) {
    throw new Error("Strategy exposure distribution requires at least two grid points.");
  }
  const slope = strategyExposureLogSlope(rateBpsPerHour, horizonMs, temperature);
  const probabilities = new Float64Array(grid.length);
  let maximum = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < grid.length; index += 1) {
    const logWeight = slope * grid[index]!;
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

/**
 * Converts the signed KAMA rate into the natural parameter of a truncated
 * exponential exposure distribution: s(a) ∝ exp(slope * a).
 */
export function strategyExposureLogSlope(
  rateBpsPerHour: number,
  horizonMs: number,
  temperature: number,
): number {
  if (!Number.isFinite(rateBpsPerHour) || !Number.isFinite(horizonMs) || horizonMs <= 0
    || !Number.isFinite(temperature) || temperature <= 0) {
    throw new Error("Strategy exposure distribution requires finite rate and positive timing temperature.");
  }
  const expectedLogReturn = rateBpsPerHour / 10_000 * horizonMs / HOUR_MS;
  return expectedLogReturn / temperature;
}

/**
 * Builds causal per-candle temperatures for an H-step value horizon. Temperature
 * scales by sqrt(H / dt), which is sqrt(horizonSteps). Optional volatility is
 * the unannualized population standard deviation of simple close returns over
 * the same trailing H-step interval.
 */
export function strategyExposureTemperatures(
  prices: ArrayLike<number>,
  options: StrategyExposureTemperatureOptions,
  shared = false,
): Float32Array {
  if (!Number.isFinite(options.intervalMs) || options.intervalMs <= 0
    || !Number.isInteger(options.horizonSteps) || options.horizonSteps < 1
    || !Number.isFinite(options.temperature) || options.temperature <= 0
  ) {
    throw new Error("Strategy exposure temperature configuration is invalid.");
  }
  const base = options.temperature * Math.sqrt(options.horizonSteps);
  const result = float32(prices.length, shared);
  if (!options.scaleByVolatility) {
    result.fill(base);
    return result;
  }

  const capacity = options.horizonSteps;
  const returns = new Float64Array(capacity);
  let position = 0;
  let count = 0;
  let sum = 0;
  let sumSquares = 0;
  for (let index = 0; index < prices.length; index += 1) {
    const price = prices[index]!;
    if (!Number.isFinite(price) || price <= 0) {
      throw new Error("Strategy exposure volatility requires positive finite prices.");
    }
    if (index > 0) {
      const previous = prices[index - 1]!;
      if (!Number.isFinite(previous) || previous <= 0) {
        throw new Error("Strategy exposure volatility requires positive finite prices.");
      }
      const value = price / previous - 1;
      if (count === capacity) {
        const removed = returns[position]!;
        sum -= removed;
        sumSquares -= removed * removed;
      } else {
        count += 1;
      }
      returns[position] = value;
      position = (position + 1) % capacity;
      sum += value;
      sumSquares += value * value;
    }
    if (count === 0) {
      result[index] = base;
      continue;
    }
    const mean = sum / count;
    const volatility = Math.sqrt(Math.max(0, sumSquares / count - mean * mean));
    result[index] = base * Math.max(Number.EPSILON, volatility);
  }
  return result;
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
  rateBpsPerHour: number,
  intervalMs: number,
  temperature: number,
): void {
  if (candleIndex < oracle.scoreStartIndex || candleIndex >= oracle.means.length) return;
  const slope = strategyExposureLogSlope(
    rateBpsPerHour,
    intervalMs * oracle.horizonSteps,
    temperature,
  );
  let maximumLogWeight = Number.NEGATIVE_INFINITY;
  for (const exposure of oracle.grid) {
    maximumLogWeight = Math.max(maximumLogWeight, slope * exposure);
  }
  let normalizer = 0;
  for (const exposure of oracle.grid) {
    normalizer += Math.exp(slope * exposure - maximumLogWeight);
  }
  const logNormalizer = maximumLogWeight + Math.log(normalizer);
  const mean = oracle.means[candleIndex]!;
  const crossEntropy = Math.max(0, logNormalizer - slope * mean);
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

function holdingLogValue(
  exposure: number,
  time: number,
  prices: ArrayLike<number>,
  options: ExposureExecutionOptions,
): number {
  const outcome = holdingOutcome(exposure, prices[time]!, prices[time + 1]!, options);
  if (!(outcome.equityFactor > 0)) return Number.NEGATIVE_INFINITY;
  return Math.log(outcome.equityFactor);
}

function continuedHoldingLogValue(
  exposure: number,
  time: number,
  prices: ArrayLike<number>,
  options: ExposureExecutionOptions,
): number {
  const previous = holdingOutcome(exposure, prices[time - 1]!, prices[time]!, options);
  const rebalance = rebalanceEquityFactor(previous.exposure, exposure, options.friction);
  const outcome = holdingOutcome(exposure, prices[time]!, prices[time + 1]!, options);
  if (!(rebalance > 0) || !(outcome.equityFactor > 0)) return Number.NEGATIVE_INFINITY;
  return Math.log(rebalance) + Math.log(outcome.equityFactor);
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

export function validateExposureValueOracleOptions(
  prices: ArrayLike<number>,
  options: ExposureValueOracleOptions,
): void {
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
  if (!Number.isInteger(options.horizonSteps ?? 1) || (options.horizonSteps ?? 1) < 1) {
    throw new Error("Exposure value oracle horizon must be a positive integer number of steps.");
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
