export interface ExposureValueOracleOptions {
  scoreStartIndex: number;
  holdingPeriodSteps?: number;
  valueHorizonSteps?: number;
  friction: number;
  gridSize: number;
  temperature: number;
  minExposure?: number;
  maxExposure?: number;
  maxEffectiveExposure?: number;
  opportunityEpsilon?: number;
  quoteLendRate?: number;
  quoteBorrowRate?: number;
  assetBorrowRate?: number;
  includeProbabilities?: boolean;
}

export interface ExposureValueOracle {
  scoreStartIndex: number;
  holdingPeriodSteps: number;
  valueHorizonSteps: number;
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
  maxEffectiveExposure: number;
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
  weightedStrategyEntropy: number;
  weightedEntropyGap: number;
  weightSum: number;
  opportunitySum: number;
  sampleCount: number;
}

export interface ExposureValueDistillationMetrics extends ExposureValueDistillationAccumulator {
  crossEntropy: number;
  oracleEntropy: number;
  strategyEntropy: number;
  entropyGap: number;
  stateMutualInformation: number;
  oracleMutualInformation: number;
  oracleMutualInformationMode: ExposureValueOracleMutualInformationMode;
  mixedLoss: number;
  mixedScore: number;
  klDivergence: number;
  score: number;
  meanOpportunity: number;
}

export type ExposureValueOracleMutualInformationMode = "approximate" | "precise";

export interface ExposureValueDistillationLossConfig {
  entropyGapLambda: number;
  stateMutualInformationLambda: number;
  oracleMutualInformationLambda: number;
  oracleMutualInformationMode: ExposureValueOracleMutualInformationMode;
  mutualInformationBins: number;
}

export const DEFAULT_EXPOSURE_VALUE_DISTILLATION_LOSS: Readonly<ExposureValueDistillationLossConfig> = {
  entropyGapLambda: 0,
  stateMutualInformationLambda: 0,
  oracleMutualInformationLambda: 0,
  oracleMutualInformationMode: "approximate",
  mutualInformationBins: 15,
};

interface ExposureValueDistillationState {
  config: ExposureValueDistillationLossConfig;
  gridSize: number;
  weightedStrategyMean: number;
  weightedStrategySecondMoment: number;
  weightedStrategyVariance: number;
  weightedOracleMean: number;
  weightedOracleSecondMoment: number;
  weightedOracleStrategyMeanProduct: number;
  preciseJoint: Float64Array | null;
  preciseOracleBins: Float64Array | null;
  preciseStrategyBins: Float64Array | null;
}

interface QuadraticExponentialStatistics {
  logNormalizer: number;
  mean: number;
  secondMoment: number;
  entropy: number;
  probabilities?: Float64Array;
}

const distillationStates = new WeakMap<
  ExposureValueDistillationAccumulator,
  ExposureValueDistillationState
>();

const INVALID_VALUE_TEMPERATURES = 50;
const HOUR_MS = 3_600_000;

export interface StrategyExposureTemperatureOptions {
  intervalMs: number;
  holdingPeriodSteps: number;
  temperature: number;
  scaleByVolatility?: boolean;
  volatilities?: ArrayLike<number>;
}

/**
 * Builds p_t(a) ∝ exp(Q_t(a) / temperature) over a fixed exposure grid.
 *
 * Q_t(a) keeps target exposure a for H price moves, then follows the optimal
 * friction-aware policy until T. Keeping the target fixed includes the intermediate
 * rebalances needed to undo price drift. Portfolio value is homogeneous, so
 * the Bellman state needs only the current marked exposure rather than equity.
 */
export function prepareExposureValueOracle(
  prices: ArrayLike<number>,
  options: ExposureValueOracleOptions,
  shared = false,
): ExposureValueOracle {
  const oracle = createExposureValueOracleStorage(prices, options, shared);
  if (oracle.valueHorizonSteps >= prices.length - 1) {
    return prepareSegmentEndingExposureValueOracle(prices, options, oracle);
  }
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
  const holdingPeriodSteps = oracle.holdingPeriodSteps;
  const valueHorizonSteps = oracle.valueHorizonSteps;
  const opportunityEpsilon = Math.max(0, options.opportunityEpsilon ?? 1e-6);
  const forcedValues = new Float64Array(grid.length);
  const cellCount = prices.length * grid.length;
  let previousContinuations: Float64Array | null = null;
  let currentContinuations = new Float64Array(cellCount);
  const policy = new Uint16Array((prices.length - options.scoreStartIndex) * grid.length);
  const prefixValues = new Float64Array(grid.length);
  const suffixValues = new Float64Array(grid.length);
  const prefixTargets = new Uint16Array(grid.length);
  const suffixTargets = new Uint16Array(grid.length);
  const separableRebalanceCosts = 1 - execution.friction * grid[grid.length - 1]! > 0
    && 1 - execution.friction + execution.friction * grid[0]! > 0;

  const blockDurations: number[] = [];
  for (let remaining = valueHorizonSteps; remaining > 0;) {
    const duration = Math.min(holdingPeriodSteps, remaining);
    blockDurations.push(duration);
    remaining -= duration;
  }
  const transitionCache = new Map<number, HoldingTransitions>();
  for (let level = blockDurations.length - 1; level >= 0; level -= 1) {
    const duration = blockDurations[level]!;
    let transitions = transitionCache.get(duration);
    if (!transitions) {
      transitions = prepareHoldingTransitions(prices, grid, duration, execution);
      transitionCache.set(duration, transitions);
    }
    const finalLevel = level === 0;
    for (let time = options.scoreStartIndex; time < prices.length; time += 1) {
      const row = time * grid.length;
      const endpointTime = Math.min(prices.length - 1, time + duration);
      const endpointRow = endpointTime * grid.length;
      for (let exposureIndex = 0; exposureIndex < grid.length; exposureIndex += 1) {
        const cell = row + exposureIndex;
        const holdingValue = transitions.values[cell]!;
        forcedValues[exposureIndex] = Number.isFinite(holdingValue)
          ? holdingValue + (previousContinuations
              ? interpolateRow(
                  previousContinuations,
                  endpointRow,
                  grid,
                  transitions.endpointExposures[cell]!,
                )
              : 0)
          : Number.NEGATIVE_INFINITY;
      }

      if (finalLevel) {
        const probabilityRow = probabilities?.subarray(row, row + grid.length);
        const statistics = preferenceStatistics(
          forcedValues,
          grid,
          options.temperature,
          probabilityRow,
        );
        means[time] = statistics.mean;
        secondMoments[time] = statistics.secondMoment;
        modalExposures[time] = statistics.optimalExposure;
        entropies[time] = statistics.entropy;
        opportunities[time] = statistics.opportunity;
        weights[time] = statistics.opportunity + opportunityEpsilon;
      }

      fillOptimalContinuation(
        forcedValues,
        currentContinuations,
        row,
        grid,
        execution.friction,
        separableRebalanceCosts,
        prefixValues,
        suffixValues,
        prefixTargets,
        suffixTargets,
        finalLevel ? policy : undefined,
        finalLevel ? (time - options.scoreStartIndex) * grid.length : 0,
      );
    }
    previousContinuations = currentContinuations;
    currentContinuations = new Float64Array(cellCount);
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
      target = grid[policy[(time - options.scoreStartIndex) * grid.length + currentIndex]!]!;
      remainingHoldSteps = Math.min(holdingPeriodSteps, prices.length - 1 - time);
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

function prepareSegmentEndingExposureValueOracle(
  prices: ArrayLike<number>,
  options: ExposureValueOracleOptions,
  oracle: ExposureValueOracle,
): ExposureValueOracle {
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
  const holdingPeriodSteps = oracle.holdingPeriodSteps;
  const opportunityEpsilon = Math.max(0, options.opportunityEpsilon ?? 1e-6);
  const forcedValues = new Float64Array(grid.length);
  const ringLength = Math.min(prices.length, holdingPeriodSteps + 1);
  const continuationRing = Array.from(
    { length: ringLength },
    () => new Float64Array(grid.length),
  );
  const outcomeExposureRing = Array.from(
    { length: ringLength },
    () => new Float64Array(grid.length),
  );
  const continuedValueRing = Array.from(
    { length: ringLength },
    () => new Float64Array(grid.length),
  );
  const nextOutcomeEquities = new Float64Array(grid.length);
  const rollingValues = new Float64Array(grid.length);
  const rollingInvalidCounts = new Int32Array(grid.length);
  const policy = new Uint16Array((prices.length - options.scoreStartIndex) * grid.length);
  const prefixValues = new Float64Array(grid.length);
  const suffixValues = new Float64Array(grid.length);
  const prefixTargets = new Uint16Array(grid.length);
  const suffixTargets = new Uint16Array(grid.length);
  const separableRebalanceCosts = 1 - execution.friction * grid[grid.length - 1]! > 0
    && 1 - execution.friction + execution.friction * grid[0]! > 0;

  for (let time = prices.length - 1; time >= options.scoreStartIndex; time -= 1) {
    if (time === prices.length - 1) {
      forcedValues.fill(0);
    } else {
      const endpointTime = Math.min(prices.length - 1, time + holdingPeriodSteps);
      const endpointContinuation = continuationRing[endpointTime % ringLength]!;
      const endpointExposures = outcomeExposureRing[(endpointTime - 1) % ringLength]!;
      const currentOutcomeExposures = outcomeExposureRing[time % ringLength]!;
      const addedContinuations = continuedValueRing[(time + 1) % ringLength]!;
      for (let exposureIndex = 0; exposureIndex < grid.length; exposureIndex += 1) {
        const exposure = grid[exposureIndex]!;
        const currentOutcome = holdingOutcome(
          exposure,
          prices[time]!,
          prices[time + 1]!,
          execution,
        );
        currentOutcomeExposures[exposureIndex] = currentOutcome.exposure;
        let rolling = rollingValues[exposureIndex]!;
        let invalidCount = rollingInvalidCounts[exposureIndex]!;
        if (time + 1 < prices.length - 1) {
          const nextEquity = nextOutcomeEquities[exposureIndex]!;
          if (nextEquity > 0) rolling -= Math.log(nextEquity);
          else invalidCount -= 1;
          const rebalance = rebalanceEquityFactor(
            currentOutcome.exposure,
            exposure,
            execution.friction,
          );
          const added = rebalance > 0 && nextEquity > 0
            ? Math.log(rebalance) + Math.log(nextEquity)
            : Number.NEGATIVE_INFINITY;
          addedContinuations[exposureIndex] = added;
          if (Number.isFinite(added)) rolling += added;
          else invalidCount += 1;
        }
        const removedTime = time + holdingPeriodSteps;
        if (removedTime < prices.length - 1) {
          const removed = continuedValueRing[removedTime % ringLength]![exposureIndex]!;
          if (Number.isFinite(removed)) rolling -= removed;
          else invalidCount -= 1;
        }
        const first = currentOutcome.equityFactor > 0
          ? Math.log(currentOutcome.equityFactor)
          : Number.NEGATIVE_INFINITY;
        if (Number.isFinite(first)) rolling += first;
        else invalidCount += 1;
        rollingValues[exposureIndex] = rolling;
        rollingInvalidCounts[exposureIndex] = invalidCount;
        nextOutcomeEquities[exposureIndex] = currentOutcome.equityFactor;
        forcedValues[exposureIndex] = invalidCount === 0
          ? rolling + interpolateRow(endpointContinuation, 0, grid, endpointExposures[exposureIndex]!)
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

    fillOptimalContinuation(
      forcedValues,
      continuationRing[time % ringLength]!,
      0,
      grid,
      execution.friction,
      separableRebalanceCosts,
      prefixValues,
      suffixValues,
      prefixTargets,
      suffixTargets,
      policy,
      (time - options.scoreStartIndex) * grid.length,
    );
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
      target = grid[policy[(time - options.scoreStartIndex) * grid.length + currentIndex]!]!;
      remainingHoldSteps = Math.min(holdingPeriodSteps, prices.length - 1 - time);
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

interface HoldingTransitions {
  values: Float64Array;
  endpointExposures: Float64Array;
}

function prepareHoldingTransitions(
  prices: ArrayLike<number>,
  grid: Float64Array,
  duration: number,
  execution: ExposureExecutionOptions,
): HoldingTransitions {
  const values = new Float64Array(prices.length * grid.length);
  const endpointExposures = new Float64Array(values.length);
  const firstLogs = new Float64Array(prices.length);
  const outcomeExposures = new Float64Array(prices.length);
  const continuedPrefix = new Float64Array(prices.length + 1);
  const invalidPrefix = new Int32Array(prices.length + 1);
  for (let exposureIndex = 0; exposureIndex < grid.length; exposureIndex += 1) {
    const target = grid[exposureIndex]!;
    for (let move = 0; move + 1 < prices.length; move += 1) {
      const outcome = holdingOutcome(target, prices[move]!, prices[move + 1]!, execution);
      outcomeExposures[move] = outcome.exposure;
      firstLogs[move] = outcome.equityFactor > 0
        ? Math.log(outcome.equityFactor)
        : Number.NEGATIVE_INFINITY;
      const rebalance = move === 0
        ? 1
        : rebalanceEquityFactor(outcomeExposures[move - 1]!, target, execution.friction);
      const continued = rebalance > 0 && outcome.equityFactor > 0
        ? Math.log(rebalance) + Math.log(outcome.equityFactor)
        : Number.NEGATIVE_INFINITY;
      continuedPrefix[move + 1] = continuedPrefix[move]! + (Number.isFinite(continued) ? continued : 0);
      invalidPrefix[move + 1] = invalidPrefix[move]! + (Number.isFinite(continued) ? 0 : 1);
    }
    for (let time = 0; time < prices.length; time += 1) {
      const cell = time * grid.length + exposureIndex;
      const endpointTime = Math.min(prices.length - 1, time + duration);
      if (endpointTime === time) {
        values[cell] = 0;
        endpointExposures[cell] = target;
        continue;
      }
      const invalidCount = invalidPrefix[endpointTime]! - invalidPrefix[time + 1]!
        + (Number.isFinite(firstLogs[time]!) ? 0 : 1);
      values[cell] = invalidCount === 0
        ? firstLogs[time]! + continuedPrefix[endpointTime]! - continuedPrefix[time + 1]!
        : Number.NEGATIVE_INFINITY;
      endpointExposures[cell] = outcomeExposures[endpointTime - 1]!;
    }
  }
  return { values, endpointExposures };
}

function fillOptimalContinuation(
  forcedValues: Float64Array,
  continuations: Float64Array,
  row: number,
  grid: Float64Array,
  friction: number,
  separableRebalanceCosts: boolean,
  prefixValues: Float64Array,
  suffixValues: Float64Array,
  prefixTargets: Uint16Array,
  suffixTargets: Uint16Array,
  policy?: Uint16Array,
  policyRow = 0,
): void {
  if (separableRebalanceCosts) {
    prepareSeparableRebalanceScans(
      forcedValues,
      grid,
      friction,
      prefixValues,
      suffixValues,
      prefixTargets,
      suffixTargets,
    );
    for (let currentIndex = 0; currentIndex < grid.length; currentIndex += 1) {
      const current = grid[currentIndex]!;
      const sell = Math.log(1 - friction * current) + prefixValues[currentIndex]!;
      const buy = Math.log(1 - friction + friction * current) + suffixValues[currentIndex]!;
      const sellTarget = prefixTargets[currentIndex]!;
      const buyTarget = suffixTargets[currentIndex]!;
      const chooseBuy = buy > sell || (buy === sell && buyTarget < sellTarget);
      const best = chooseBuy ? buy : sell;
      continuations[row + currentIndex] = best;
      if (policy) policy[policyRow + currentIndex] = Number.isFinite(best)
        ? (chooseBuy ? buyTarget : sellTarget)
        : currentIndex;
    }
    return;
  }
  for (let currentIndex = 0; currentIndex < grid.length; currentIndex += 1) {
    let best = Number.NEGATIVE_INFINITY;
    let bestTargetIndex = currentIndex;
    for (let targetIndex = 0; targetIndex < grid.length; targetIndex += 1) {
      const rebalance = rebalanceEquityFactor(grid[currentIndex]!, grid[targetIndex]!, friction);
      const forced = forcedValues[targetIndex]!;
      if (rebalance <= 0 || !Number.isFinite(forced)) continue;
      const value = Math.log(rebalance) + forced;
      if (value > best) {
        best = value;
        bestTargetIndex = targetIndex;
      }
    }
    continuations[row + currentIndex] = best;
    if (policy) policy[policyRow + currentIndex] = bestTargetIndex;
  }
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
  const holdingPeriodSteps = options.holdingPeriodSteps ?? 1;
  const valueHorizonSteps = options.valueHorizonSteps ?? holdingPeriodSteps;
  const grid = float64(options.gridSize, shared);
  for (let index = 0; index < grid.length; index += 1) {
    grid[index] = minExposure + index / (grid.length - 1) * (maxExposure - minExposure);
  }
  return {
    scoreStartIndex: options.scoreStartIndex,
    holdingPeriodSteps,
    valueHorizonSteps,
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
      maxEffectiveExposure: options.maxEffectiveExposure ?? 250,
      quoteLendRate: options.quoteLendRate ?? 0,
      quoteBorrowRate: options.quoteBorrowRate ?? 0,
      assetBorrowRate: options.assetBorrowRate ?? 0,
    },
  };
}

export function shareExposureValueOracle(oracle: ExposureValueOracle): ExposureValueOracle {
  return {
    scoreStartIndex: oracle.scoreStartIndex,
    holdingPeriodSteps: oracle.holdingPeriodSteps,
    valueHorizonSteps: oracle.valueHorizonSteps,
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

/** Retains oracle targets for a scored prefix while allowing their values to use later prices. */
export function truncateExposureValueOracle(
  oracle: ExposureValueOracle,
  length: number,
): ExposureValueOracle {
  if (!Number.isInteger(length) || length <= oracle.scoreStartIndex || length > oracle.means.length) {
    throw new Error("Exposure value oracle truncation is outside its prepared price series.");
  }
  return {
    ...oracle,
    means: oracle.means.subarray(0, length),
    secondMoments: oracle.secondMoments.subarray(0, length),
    modalExposures: oracle.modalExposures.subarray(0, length),
    optimalExposures: oracle.optimalExposures.subarray(0, length),
    entropies: oracle.entropies.subarray(0, length),
    weights: oracle.weights.subarray(0, length),
    opportunities: oracle.opportunities.subarray(0, length),
    ...(oracle.probabilities
      ? { probabilities: oracle.probabilities.subarray(0, length * oracle.grid.length) }
      : {}),
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
  holdingPeriodMs: number,
  temperature: number,
  quadraticCoefficient = 0,
): Float64Array {
  if (grid.length < 2) {
    throw new Error("Strategy exposure distribution requires at least two grid points.");
  }
  const slope = strategyExposureLogSlope(rateBpsPerHour, holdingPeriodMs, temperature);
  if (!Number.isFinite(quadraticCoefficient)) {
    throw new Error("Strategy exposure distribution requires a finite quadratic coefficient.");
  }
  const probabilities = new Float64Array(grid.length);
  let maximum = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < grid.length; index += 1) {
    const exposure = grid[index]!;
    const logWeight = slope * exposure + quadraticCoefficient * exposure * exposure;
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
  holdingPeriodMs: number,
  temperature: number,
): number {
  if (!Number.isFinite(rateBpsPerHour)
    || !Number.isFinite(holdingPeriodMs) || holdingPeriodMs <= 0
    || !Number.isFinite(temperature) || temperature <= 0) {
    throw new Error("Strategy exposure distribution requires finite rate and positive timing temperature.");
  }
  const expectedLogReturn = rateBpsPerHour / 10_000 * holdingPeriodMs / HOUR_MS;
  return expectedLogReturn / temperature;
}

/** Exact log partition for a truncated exponential on a uniform discrete grid. */
export function truncatedExponentialLogNormalizer(
  gridSize: number,
  minimumExposure: number,
  maximumExposure: number,
  slope: number,
): number {
  if (!Number.isInteger(gridSize) || gridSize < 2
    || !Number.isFinite(minimumExposure) || !Number.isFinite(maximumExposure)
    || minimumExposure >= maximumExposure || !Number.isFinite(slope)) {
    throw new Error("Truncated exponential normalizer requires a finite uniform grid and slope.");
  }
  if (slope === 0) return Math.log(gridSize);
  const spacing = (maximumExposure - minimumExposure) / (gridSize - 1);
  const step = Math.abs(slope) * spacing;
  const maximumLogWeight = Math.max(slope * minimumExposure, slope * maximumExposure);
  if (!Number.isFinite(step)) return maximumLogWeight;
  return maximumLogWeight
    + Math.log(-Math.expm1(-step * gridSize))
    - Math.log(-Math.expm1(-step));
}

/** Machine-precision log partition for exp(linear*a + quadratic*a^2) on a uniform grid. */
export function quadraticExponentialLogNormalizer(
  gridSize: number,
  minimumExposure: number,
  maximumExposure: number,
  linearCoefficient: number,
  quadraticCoefficient: number,
): number {
  if (quadraticCoefficient === 0) {
    return truncatedExponentialLogNormalizer(
      gridSize,
      minimumExposure,
      maximumExposure,
      linearCoefficient,
    );
  }
  if (!Number.isInteger(gridSize) || gridSize < 2
    || !Number.isFinite(minimumExposure) || !Number.isFinite(maximumExposure)
    || minimumExposure >= maximumExposure || !Number.isFinite(linearCoefficient)
    || !Number.isFinite(quadraticCoefficient)) {
    throw new Error("Quadratic exponential normalizer requires a finite uniform grid and coefficients.");
  }
  const spacing = (maximumExposure - minimumExposure) / (gridSize - 1);
  const logWeight = (exposure: number) =>
    linearCoefficient * exposure + quadraticCoefficient * exposure * exposure;
  const minimumLogWeight = logWeight(minimumExposure);
  const maximumLogWeight = logWeight(maximumExposure);
  let maximum = Math.max(minimumLogWeight, maximumLogWeight);
  let peakIndex = minimumLogWeight >= maximumLogWeight ? 0 : gridSize - 1;
  if (quadraticCoefficient < 0) {
    const vertexPosition = clamp(
      (-linearCoefficient / (2 * quadraticCoefficient) - minimumExposure) / spacing,
      0,
      gridSize - 1,
    );
    const lower = Math.floor(vertexPosition);
    const upper = Math.min(gridSize - 1, lower + 1);
    const lowerLogWeight = logWeight(minimumExposure + lower * spacing);
    const upperLogWeight = logWeight(minimumExposure + upper * spacing);
    if (lowerLogWeight > maximum) {
      maximum = lowerLogWeight;
      peakIndex = lower;
    }
    if (upperLogWeight > maximum) {
      maximum = upperLogWeight;
      peakIndex = upper;
    }

    // Concavity gives a mode-centered fixed-width window. The dynamic cutoff bounds the
    // complete omitted tail below one Float64 epsilon. A fixed forward recurrence also mirrors
    // the SIMD-friendly CUDA implementation.
    const tailLogCutoff = Math.log(gridSize / Number.EPSILON) + 2;
    const indexCurvature = -quadraticCoefficient * spacing * spacing;
    const radius = Math.min(
      gridSize,
      Math.ceil(0.5 + Math.sqrt(0.25 + tailLogCutoff / indexCurvature)),
    );
    const activeCount = Math.min(gridSize, 2 * radius + 1);
    const startIndex = clamp(peakIndex - radius, 0, gridSize - activeCount);
    const startExposure = minimumExposure + startIndex * spacing;
    const secondDifference = 2 * quadraticCoefficient * spacing * spacing;
    let total = 0;
    let relativeLogWeight = logWeight(startExposure) - maximum;
    let delta = linearCoefficient * spacing
      + quadraticCoefficient * (2 * startExposure * spacing + spacing * spacing);
    for (let offset = 0; offset < activeCount; offset += 1) {
      total += Math.exp(relativeLogWeight);
      relativeLogWeight += delta;
      delta += secondDifference;
    }
    return maximum + Math.log(total);
  }
  let total = 0;
  for (let index = 0; index < gridSize; index += 1) {
    total += Math.exp(logWeight(minimumExposure + index * spacing) - maximum);
  }
  return maximum + Math.log(total);
}

function quadraticExponentialStatistics(
  gridSize: number,
  minimumExposure: number,
  maximumExposure: number,
  linearCoefficient: number,
  quadraticCoefficient: number,
  includeProbabilities: boolean,
): QuadraticExponentialStatistics {
  const logNormalizer = quadraticExponentialLogNormalizer(
    gridSize,
    minimumExposure,
    maximumExposure,
    linearCoefficient,
    quadraticCoefficient,
  );
  const spacing = (maximumExposure - minimumExposure) / (gridSize - 1);
  const probabilities = includeProbabilities ? new Float64Array(gridSize) : undefined;
  let mean = 0;
  let secondMoment = 0;
  let probabilitySum = 0;
  for (let index = 0; index < gridSize; index += 1) {
    const exposure = minimumExposure + index * spacing;
    const probability = Math.exp(
      linearCoefficient * exposure
      + quadraticCoefficient * exposure * exposure
      - logNormalizer,
    );
    if (probabilities) probabilities[index] = probability;
    probabilitySum += probability;
    mean += probability * exposure;
    secondMoment += probability * exposure * exposure;
  }
  // The analytic normalizer and the explicit moment sum can differ by a few ulps.
  // Renormalizing the moments keeps entropy and MI invariant to that drift.
  mean /= probabilitySum;
  secondMoment /= probabilitySum;
  if (probabilities) {
    for (let index = 0; index < probabilities.length; index += 1) {
      probabilities[index] /= probabilitySum;
    }
  }
  return {
    logNormalizer,
    mean,
    secondMoment,
    entropy: Math.max(
      0,
      logNormalizer - linearCoefficient * mean - quadraticCoefficient * secondMoment,
    ),
    ...(probabilities ? { probabilities } : {}),
  };
}

/**
 * Builds causal, unannualized volatility over the trailing H price moves as the
 * population standard deviation of log close returns.
 */
export function strategyExposureVolatilities(
  prices: ArrayLike<number>,
  holdingPeriodSteps: number,
  shared = false,
): Float32Array {
  if (!Number.isInteger(holdingPeriodSteps) || holdingPeriodSteps < 1) {
    throw new Error("Strategy exposure volatility window must be a positive integer.");
  }
  const result = float32(prices.length, shared);
  const capacity = holdingPeriodSteps;
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
      const value = Math.log(price / previous);
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
      result[index] = 0;
      continue;
    }
    const mean = sum / count;
    const volatility = Math.sqrt(Math.max(0, sumSquares / count - mean * mean));
    result[index] = volatility;
  }
  return result;
}

/**
 * Builds causal per-candle temperatures for an H-step holding period. Temperature
 * scales by sqrt(H / dt), which is sqrt(holdingPeriodSteps). Optional volatility uses
 * the trailing-H population standard deviation of log close returns.
 */
export function strategyExposureTemperatures(
  prices: ArrayLike<number>,
  options: StrategyExposureTemperatureOptions,
  shared = false,
): Float32Array {
  if (!Number.isFinite(options.intervalMs) || options.intervalMs <= 0
    || !Number.isInteger(options.holdingPeriodSteps) || options.holdingPeriodSteps < 1
    || !Number.isFinite(options.temperature) || options.temperature <= 0
  ) {
    throw new Error("Strategy exposure temperature configuration is invalid.");
  }
  const base = options.temperature * Math.sqrt(options.holdingPeriodSteps);
  const result = float32(prices.length, shared);
  if (!options.scaleByVolatility) {
    result.fill(base);
    return result;
  }
  const volatilities = options.volatilities
    ?? strategyExposureVolatilities(prices, options.holdingPeriodSteps);
  if (volatilities.length < prices.length) {
    throw new Error("Strategy exposure volatilities do not cover the price series.");
  }
  for (let index = 0; index < prices.length; index += 1) {
    const volatility = volatilities[index]!;
    if (!Number.isFinite(volatility) || volatility < 0) {
      throw new Error("Strategy exposure volatility must be finite and non-negative.");
    }
    result[index] = base * Math.max(Number.EPSILON, volatility);
  }
  return result;
}

/** Effective concave coefficient in exp(b1*a + b2*a^2), with b2 = -b2' * v^2. */
export function strategyExposureQuadraticCoefficient(
  quadraticScale: number,
  volatility: number,
): number {
  if (!Number.isFinite(quadraticScale) || quadraticScale < 0
    || !Number.isFinite(volatility) || volatility < 0) {
    throw new Error("Strategy quadratic scale and volatility must be finite and non-negative.");
  }
  if (quadraticScale === 0 || volatility === 0) return 0;
  const coefficient = -quadraticScale * volatility * volatility;
  if (!Number.isFinite(coefficient)) {
    throw new Error("Strategy quadratic coefficient exceeds the numeric range.");
  }
  return coefficient;
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

export function normalizeExposureValueDistillationLossConfig(
  config: Partial<ExposureValueDistillationLossConfig> = {},
  gridSize = Number.MAX_SAFE_INTEGER,
): ExposureValueDistillationLossConfig {
  const normalized = {
    ...DEFAULT_EXPOSURE_VALUE_DISTILLATION_LOSS,
    ...config,
  };
  if (![normalized.entropyGapLambda, normalized.stateMutualInformationLambda,
    normalized.oracleMutualInformationLambda].every((value) => Number.isFinite(value) && value >= 0)) {
    throw new Error("Value-distillation loss weights must be finite and non-negative.");
  }
  if (normalized.oracleMutualInformationMode !== "approximate"
    && normalized.oracleMutualInformationMode !== "precise") {
    throw new Error("Oracle mutual information mode must be approximate or precise.");
  }
  if (!Number.isInteger(normalized.mutualInformationBins)
    || normalized.mutualInformationBins < 2
    || normalized.mutualInformationBins > 32) {
    throw new Error("Mutual-information bins must be an integer from 2 through 32.");
  }
  normalized.mutualInformationBins = Math.min(normalized.mutualInformationBins, gridSize);
  return normalized;
}

export function createExposureValueDistillationAccumulator(
  lossConfig: Partial<ExposureValueDistillationLossConfig> = {},
  gridSize = Number.MAX_SAFE_INTEGER,
): ExposureValueDistillationAccumulator {
  const accumulator: ExposureValueDistillationAccumulator = {
    weightedCrossEntropy: 0,
    weightedOracleEntropy: 0,
    weightedStrategyEntropy: 0,
    weightedEntropyGap: 0,
    weightSum: 0,
    opportunitySum: 0,
    sampleCount: 0,
  };
  distillationStates.set(accumulator, {
    config: normalizeExposureValueDistillationLossConfig(lossConfig, gridSize),
    gridSize,
    weightedStrategyMean: 0,
    weightedStrategySecondMoment: 0,
    weightedStrategyVariance: 0,
    weightedOracleMean: 0,
    weightedOracleSecondMoment: 0,
    weightedOracleStrategyMeanProduct: 0,
    preciseJoint: null,
    preciseOracleBins: null,
    preciseStrategyBins: null,
  });
  return accumulator;
}

export function observeExposureValueDistillation(
  accumulator: ExposureValueDistillationAccumulator,
  oracle: ExposureValueOracle,
  candleIndex: number,
  rateBpsPerHour: number,
  intervalMs: number,
  temperature: number,
  quadraticCoefficient = 0,
): void {
  if (candleIndex < oracle.scoreStartIndex || candleIndex >= oracle.means.length) return;
  const state = distillationStates.get(accumulator);
  if (!state) throw new Error("Value-distillation accumulator was not initialized.");
  if (state.gridSize === Number.MAX_SAFE_INTEGER) state.gridSize = oracle.grid.length;
  if (state.gridSize !== oracle.grid.length) {
    throw new Error("Value-distillation accumulator grid does not match its oracle.");
  }
  const slope = strategyExposureLogSlope(
    rateBpsPerHour,
    intervalMs * oracle.holdingPeriodSteps,
    temperature,
  );
  const lossEnabled = state.config.entropyGapLambda > 0
    || state.config.stateMutualInformationLambda > 0
    || state.config.oracleMutualInformationLambda > 0;
  const preciseEnabled = state.config.oracleMutualInformationLambda > 0
    && state.config.oracleMutualInformationMode === "precise";
  const statistics = lossEnabled
    ? quadraticExponentialStatistics(
        oracle.grid.length,
        oracle.grid[0]!,
        oracle.grid[oracle.grid.length - 1]!,
        slope,
        quadraticCoefficient,
        preciseEnabled,
      )
    : null;
  const logNormalizer = statistics?.logNormalizer ?? quadraticExponentialLogNormalizer(
    oracle.grid.length,
    oracle.grid[0]!,
    oracle.grid[oracle.grid.length - 1]!,
    slope,
    quadraticCoefficient,
  );
  const mean = oracle.means[candleIndex]!;
  const secondMoment = oracle.secondMoments[candleIndex]!;
  const crossEntropy = Math.max(
    0,
    logNormalizer - slope * mean - quadraticCoefficient * secondMoment,
  );
  const weight = oracle.weights[candleIndex]!;
  accumulator.weightedCrossEntropy += weight * crossEntropy;
  accumulator.weightedOracleEntropy += weight * oracle.entropies[candleIndex]!;
  if (statistics) {
    const logGridSize = Math.log(oracle.grid.length);
    const oracleEntropy = oracle.entropies[candleIndex]!;
    const normalizedExcessEntropy = Math.max(0, statistics.entropy - oracleEntropy) / logGridSize;
    accumulator.weightedStrategyEntropy += weight * statistics.entropy;
    accumulator.weightedEntropyGap += weight * normalizedExcessEntropy * normalizedExcessEntropy;
    state.weightedStrategyMean += weight * statistics.mean;
    state.weightedStrategySecondMoment += weight * statistics.secondMoment;
    state.weightedStrategyVariance += weight * Math.max(
      0,
      statistics.secondMoment - statistics.mean * statistics.mean,
    );
    state.weightedOracleMean += weight * mean;
    state.weightedOracleSecondMoment += weight * secondMoment;
    state.weightedOracleStrategyMeanProduct += weight * mean * statistics.mean;
    if (preciseEnabled) {
      const oracleProbabilities = oracle.probabilities;
      if (!oracleProbabilities || !statistics.probabilities) {
        throw new Error("Precise oracle mutual information requires retained oracle probabilities.");
      }
      const bins = state.config.mutualInformationBins;
      state.preciseJoint ??= new Float64Array(bins * bins);
      state.preciseOracleBins ??= new Float64Array(bins);
      state.preciseStrategyBins ??= new Float64Array(bins);
      const oracleBins = state.preciseOracleBins;
      const strategyBins = state.preciseStrategyBins;
      oracleBins.fill(0);
      strategyBins.fill(0);
      const probabilityOffset = candleIndex * oracle.grid.length;
      for (let index = 0; index < oracle.grid.length; index += 1) {
        const bin = Math.min(bins - 1, Math.floor(index * bins / oracle.grid.length));
        oracleBins[bin] += oracleProbabilities[probabilityOffset + index]!;
        strategyBins[bin] += statistics.probabilities[index]!;
      }
      for (let oracleBin = 0; oracleBin < bins; oracleBin += 1) {
        for (let strategyBin = 0; strategyBin < bins; strategyBin += 1) {
          state.preciseJoint[oracleBin * bins + strategyBin] += weight
            * oracleBins[oracleBin]!
            * strategyBins[strategyBin]!;
        }
      }
    }
  }
  accumulator.weightSum += weight;
  accumulator.opportunitySum += oracle.opportunities[candleIndex]!;
  accumulator.sampleCount += 1;
}

export function finalizeExposureValueDistillation(
  accumulator: ExposureValueDistillationAccumulator,
): ExposureValueDistillationMetrics {
  const state = distillationStates.get(accumulator);
  if (!state) throw new Error("Value-distillation accumulator was not initialized.");
  const crossEntropy = accumulator.weightSum > 0
    ? accumulator.weightedCrossEntropy / accumulator.weightSum
    : 0;
  const oracleEntropy = accumulator.weightSum > 0
    ? accumulator.weightedOracleEntropy / accumulator.weightSum
    : 0;
  const klDivergence = Math.max(0, crossEntropy - oracleEntropy);
  const strategyEntropy = accumulator.weightSum > 0
    ? accumulator.weightedStrategyEntropy / accumulator.weightSum
    : 0;
  const entropyGap = accumulator.weightSum > 0
    ? accumulator.weightedEntropyGap / accumulator.weightSum
    : 0;
  const stateMutualInformation = gaussianStateMutualInformation(state, accumulator.weightSum);
  const oracleMutualInformation = state.config.oracleMutualInformationMode === "precise"
    ? categoricalMutualInformation(state.preciseJoint, state.config.mutualInformationBins)
    : gaussianOracleMutualInformation(state, accumulator.weightSum);
  const mixedLoss = crossEntropy
    + state.config.entropyGapLambda * entropyGap
    - state.config.stateMutualInformationLambda * stateMutualInformation
    - state.config.oracleMutualInformationLambda * oracleMutualInformation;
  const mixedKlDivergence = Math.max(0, mixedLoss - oracleEntropy);
  return {
    ...accumulator,
    crossEntropy,
    oracleEntropy,
    strategyEntropy,
    entropyGap,
    stateMutualInformation,
    oracleMutualInformation,
    oracleMutualInformationMode: state.config.oracleMutualInformationMode,
    mixedLoss,
    mixedScore: Math.exp(-mixedKlDivergence),
    klDivergence,
    score: Math.exp(-klDivergence),
    meanOpportunity: accumulator.sampleCount > 0
      ? accumulator.opportunitySum / accumulator.sampleCount
      : 0,
  };
}

function gaussianStateMutualInformation(
  state: ExposureValueDistillationState,
  weightSum: number,
): number {
  if (weightSum <= 0 || state.config.stateMutualInformationLambda === 0) return 0;
  const mean = state.weightedStrategyMean / weightSum;
  const totalVariance = Math.max(0, state.weightedStrategySecondMoment / weightSum - mean * mean);
  const conditionalVariance = Math.max(0, state.weightedStrategyVariance / weightSum);
  return normalizedGaussianVarianceInformation(totalVariance, conditionalVariance, state.gridSize);
}

function gaussianOracleMutualInformation(
  state: ExposureValueDistillationState,
  weightSum: number,
): number {
  if (weightSum <= 0 || state.config.oracleMutualInformationLambda === 0) return 0;
  const oracleMean = state.weightedOracleMean / weightSum;
  const strategyMean = state.weightedStrategyMean / weightSum;
  const oracleVariance = Math.max(
    0,
    state.weightedOracleSecondMoment / weightSum - oracleMean * oracleMean,
  );
  const strategyVariance = Math.max(
    0,
    state.weightedStrategySecondMoment / weightSum - strategyMean * strategyMean,
  );
  if (oracleVariance <= Number.EPSILON || strategyVariance <= Number.EPSILON) return 0;
  const covariance = state.weightedOracleStrategyMeanProduct / weightSum
    - oracleMean * strategyMean;
  const correlationSquared = clamp(
    covariance * covariance / (oracleVariance * strategyVariance),
    0,
    1 - 1e-12,
  );
  return clamp(-0.5 * Math.log1p(-correlationSquared) / Math.log(state.gridSize), 0, 1);
}

function normalizedGaussianVarianceInformation(
  totalVariance: number,
  conditionalVariance: number,
  gridSize: number,
): number {
  if (totalVariance <= Number.EPSILON) return 0;
  if (conditionalVariance <= Number.EPSILON) return 1;
  return clamp(0.5 * Math.log(totalVariance / conditionalVariance) / Math.log(gridSize), 0, 1);
}

function categoricalMutualInformation(joint: Float64Array | null, bins: number): number {
  if (!joint) return 0;
  let total = 0;
  for (const value of joint) total += value;
  if (total <= 0) return 0;
  const oracleMarginal = new Float64Array(bins);
  const strategyMarginal = new Float64Array(bins);
  for (let oracleBin = 0; oracleBin < bins; oracleBin += 1) {
    for (let strategyBin = 0; strategyBin < bins; strategyBin += 1) {
      const probability = joint[oracleBin * bins + strategyBin]! / total;
      oracleMarginal[oracleBin] += probability;
      strategyMarginal[strategyBin] += probability;
    }
  }
  let mutualInformation = 0;
  let oracleEntropy = 0;
  let strategyEntropy = 0;
  for (let bin = 0; bin < bins; bin += 1) {
    const oracleProbability = oracleMarginal[bin]!;
    const strategyProbability = strategyMarginal[bin]!;
    if (oracleProbability > 0) oracleEntropy -= oracleProbability * Math.log(oracleProbability);
    if (strategyProbability > 0) strategyEntropy -= strategyProbability * Math.log(strategyProbability);
  }
  if (oracleEntropy <= Number.EPSILON || strategyEntropy <= Number.EPSILON) return 0;
  for (let oracleBin = 0; oracleBin < bins; oracleBin += 1) {
    for (let strategyBin = 0; strategyBin < bins; strategyBin += 1) {
      const probability = joint[oracleBin * bins + strategyBin]! / total;
      if (probability > 0) {
        mutualInformation += probability * Math.log(
          probability / (oracleMarginal[oracleBin]! * strategyMarginal[strategyBin]!),
        );
      }
    }
  }
  return clamp(mutualInformation / Math.sqrt(oracleEntropy * strategyEntropy), 0, 1);
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

function prepareSeparableRebalanceScans(
  forcedValues: Float64Array,
  grid: Float64Array,
  friction: number,
  prefixValues: Float64Array,
  suffixValues: Float64Array,
  prefixTargets: Uint16Array,
  suffixTargets: Uint16Array,
): void {
  let best = Number.NEGATIVE_INFINITY;
  let bestTarget = 0;
  for (let target = 0; target < grid.length; target += 1) {
    const forced = forcedValues[target]!;
    const adjusted = Number.isFinite(forced)
      ? forced - Math.log(1 - friction * grid[target]!)
      : Number.NEGATIVE_INFINITY;
    if (adjusted > best) {
      best = adjusted;
      bestTarget = target;
    }
    prefixValues[target] = best;
    prefixTargets[target] = bestTarget;
  }

  best = Number.NEGATIVE_INFINITY;
  bestTarget = grid.length - 1;
  for (let target = grid.length - 1; target >= 0; target -= 1) {
    const forced = forcedValues[target]!;
    const adjusted = Number.isFinite(forced)
      ? forced - Math.log(1 - friction + friction * grid[target]!)
      : Number.NEGATIVE_INFINITY;
    // Scanning downward means an equal value at this index is the lower target,
    // matching the original increasing target loop's deterministic tie break.
    if (adjusted >= best) {
      best = adjusted;
      bestTarget = target;
    }
    suffixValues[target] = best;
    suffixTargets[target] = bestTarget;
  }
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
  if (Math.abs(liquidationExposure) > options.maxEffectiveExposure) {
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

function interpolateRow(
  values: Float64Array,
  row: number,
  grid: Float64Array,
  exposure: number,
): number {
  const position = clamp(
    (exposure - grid[0]!) / (grid[grid.length - 1]! - grid[0]!) * (grid.length - 1),
    0,
    grid.length - 1,
  );
  const lower = Math.floor(position);
  const fraction = position - lower;
  const left = values[row + lower]!;
  const right = values[row + Math.min(grid.length - 1, lower + 1)]!;
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
  const holdingPeriodSteps = options.holdingPeriodSteps ?? 1;
  const valueHorizonSteps = options.valueHorizonSteps ?? holdingPeriodSteps;
  if (!Number.isInteger(holdingPeriodSteps) || holdingPeriodSteps < 1) {
    throw new Error("Exposure value oracle holding period must be a positive integer number of steps.");
  }
  if (!Number.isInteger(valueHorizonSteps) || valueHorizonSteps < holdingPeriodSteps) {
    throw new Error("Exposure value oracle horizon must be an integer at least as long as its holding period.");
  }
  const minimum = options.minExposure ?? -1;
  const maximum = options.maxExposure ?? 1;
  const maximumEffective = options.maxEffectiveExposure ?? 250;
  if (!Number.isFinite(minimum) || !Number.isFinite(maximum) || minimum >= 0 || maximum <= 0) {
    throw new Error("Exposure value oracle bounds must contain zero.");
  }
  if (!Number.isFinite(maximumEffective)
    || maximumEffective < Math.max(Math.abs(minimum), Math.abs(maximum))) {
    throw new Error("Exposure value oracle effective exposure must cover the tradable grid.");
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
