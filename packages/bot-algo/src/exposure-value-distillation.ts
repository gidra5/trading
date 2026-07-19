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
  initialExposure?: number;
  terminalIndex?: number;
  includeProbabilities?: boolean;
}

export interface ExposureValueOraclePath {
  startIndex: number;
  terminalIndex: number;
  initialExposure: number;
  terminalExposure: 0;
  logReturn: number;
  totalReturn: number;
  exposures: Float32Array;
  equities: Float64Array;
  maxDrawdown: number;
  turnover: number;
  rebalanceCount: number;
  liquidationCount: number;
}

export interface ExposureValueOracle {
  scoreStartIndex: number;
  holdingPeriodSteps: number;
  valueHorizonSteps: number;
  temperature: number;
  /** Executable target/action exposures. */
  grid: Float64Array;
  /** Current strategy states, including drift up to the liquidation boundary. */
  currentGrid: Float64Array;
  means: Float32Array;
  secondMoments: Float32Array;
  modalExposures: Float32Array;
  entropies: Float32Array;
  policyMeans: Float32Array;
  policySecondMoments: Float32Array;
  policyMeanLogRebalances: Float32Array;
  policyEntropies: Float32Array;
  averageRegrets: Float32Array;
  weights: Float32Array;
  opportunities: Float32Array;
  probabilities?: Float32Array;
  execution: ExposureExecutionOptions;
  path: ExposureValueOraclePath;
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
  averageRegretSum: number;
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
  meanAverageRegret: number;
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
  conditionalBinScratch: ExposureConditionalBinScratch | null;
  transitionScratch: TransitionDistributionScratch | null;
}

export interface ExposureConditionalBinScratch {
  baseLogs: Float64Array;
  sellTargetWeights: Float64Array;
  buyTargetWeights: Float64Array;
  sellBins: Float64Array;
  buyBins: Float64Array;
}

interface QuadraticExponentialStatistics {
  logNormalizer: number;
  mean: number;
  secondMoment: number;
  entropy: number;
  probabilities?: Float64Array;
}

export interface ExposureTransitionPolicyStatistics {
  logNormalizer: number;
  mean: number;
  secondMoment: number;
  meanLogRebalance: number;
  entropy: number;
}

type TransitionDistributionStatistics = ExposureTransitionPolicyStatistics;

interface TransitionOracleStatistics extends ExposureTransitionPolicyStatistics {
  averageRegret: number;
}

interface TransitionDistributionScratch {
  baseLogits: Float64Array;
  sellWeights: Float64Array;
  sellMeans: Float64Array;
  sellSecondMoments: Float64Array;
  sellBaseLogits: Float64Array;
  sellLogDenominators: Float64Array;
  buyWeights: Float64Array;
  buyMeans: Float64Array;
  buySecondMoments: Float64Array;
  buyBaseLogits: Float64Array;
  buyLogDenominators: Float64Array;
  prefixRawSellLogs: Float64Array;
  suffixRawBuyLogs: Float64Array;
  prefixRawSellMaxima: Float64Array;
  suffixRawBuyMaxima: Float64Array;
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
 * Builds the post-action F_t(a) preference plus sufficient statistics for
 * p_t(a | x) ∝ exp((F_t(a) + log R(x→a)) / temperature) on a fixed grid.
 *
 * Q_t(a) rebalances to exposure a once, leaves the resulting quote and asset
 * quantities untouched for H price moves, then follows the optimal friction-aware
 * policy until T. Portfolio value is homogeneous, so
 * the Bellman state needs only the current marked exposure rather than equity.
 */
export function prepareExposureValueOracle(
  prices: ArrayLike<number>,
  options: ExposureValueOracleOptions,
  shared = false,
): ExposureValueOracle {
  const oracle = createExposureValueOracleStorage(prices, options, shared);
  if (oracle.valueHorizonSteps >= prices.length - 1 - options.scoreStartIndex) {
    prepareSegmentEndingExposureValueOracle(prices, options, oracle);
    prepareExposureValueOraclePath(prices, options, oracle);
    return oracle;
  }
  const {
    grid,
    means,
    secondMoments,
    modalExposures,
    entropies,
    policyMeans,
    policySecondMoments,
    policyMeanLogRebalances,
    policyEntropies,
    averageRegrets,
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
  const transitionScratch = createTransitionDistributionScratch(grid.length);
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
        const policy = transitionOracleStatistics(
          forcedValues,
          grid,
          oracle.currentGrid,
          execution.friction,
          options.temperature,
          transitionScratch,
        );
        policyMeans[time] = policy.mean;
        policySecondMoments[time] = policy.secondMoment;
        policyMeanLogRebalances[time] = policy.meanLogRebalance;
        policyEntropies[time] = policy.entropy;
        averageRegrets[time] = policy.averageRegret;
        weights[time] = policy.averageRegret + opportunityEpsilon;
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

  prepareExposureValueOraclePath(prices, options, oracle);
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
    entropies,
    policyMeans,
    policySecondMoments,
    policyMeanLogRebalances,
    policyEntropies,
    averageRegrets,
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
  const prefixValues = new Float64Array(grid.length);
  const suffixValues = new Float64Array(grid.length);
  const prefixTargets = new Uint16Array(grid.length);
  const suffixTargets = new Uint16Array(grid.length);
  const transitionScratch = createTransitionDistributionScratch(grid.length);
  const separableRebalanceCosts = 1 - execution.friction * grid[grid.length - 1]! > 0
    && 1 - execution.friction + execution.friction * grid[0]! > 0;
  for (let time = prices.length - 1; time >= options.scoreStartIndex; time -= 1) {
    if (time === prices.length - 1) {
      forcedValues.fill(0);
    } else {
      const endpointTime = Math.min(prices.length - 1, time + holdingPeriodSteps);
      const endpointContinuation = continuationRing[endpointTime % ringLength]!;
      for (let exposureIndex = 0; exposureIndex < grid.length; exposureIndex += 1) {
        const holding = holdingBlockOutcome(
          prices,
          time,
          endpointTime,
          grid[exposureIndex]!,
          execution,
        );
        forcedValues[exposureIndex] = Number.isFinite(holding.logReturn)
          ? holding.logReturn + interpolateRow(
              endpointContinuation,
              0,
              grid,
              holding.exposure,
            )
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
    const policy = transitionOracleStatistics(
      forcedValues,
      grid,
      oracle.currentGrid,
      execution.friction,
      options.temperature,
      transitionScratch,
    );
    policyMeans[time] = policy.mean;
    policySecondMoments[time] = policy.secondMoment;
    policyMeanLogRebalances[time] = policy.meanLogRebalance;
    policyEntropies[time] = policy.entropy;
    averageRegrets[time] = policy.averageRegret;
    weights[time] = policy.averageRegret + opportunityEpsilon;

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
    );
  }

  return oracle;
}

/**
 * Solves one coherent hindsight control problem from score start through the
 * selected terminal candle. The terminal value is the fee-aware rebalance to
 * zero exposure; rolling Q targets and their shorter horizons remain separate.
 *
 * The Bellman rows are checkpointed and recomputed block-by-block while the
 * policy is rolled forward. This keeps memory at O(grid * sqrt(decisions))
 * rather than retaining a policy row for every candle.
 */
export function prepareExposureValueOraclePath(
  prices: ArrayLike<number>,
  options: ExposureValueOracleOptions,
  oracle: ExposureValueOracle,
): void {
  const start = options.scoreStartIndex;
  const terminal = options.terminalIndex ?? prices.length - 1;
  const holdingSteps = oracle.holdingPeriodSteps;
  const decisionTimes = [start];
  while (decisionTimes.at(-1)! < terminal) {
    decisionTimes.push(Math.min(terminal, decisionTimes.at(-1)! + holdingSteps));
  }
  const decisionCount = decisionTimes.length;
  const grid = oracle.grid;
  const gridSize = grid.length;
  const blockSize = Math.max(1, Math.ceil(Math.sqrt(decisionCount - 1)));
  const checkpoints = new Map<number, Float64Array>();
  let nextValues: Float64Array<ArrayBufferLike> = new Float64Array(gridSize);
  for (let index = 0; index < gridSize; index += 1) {
    const closeout = rebalanceEquityFactor(grid[index]!, 0, oracle.execution.friction);
    nextValues[index] = closeout > 0 ? Math.log(closeout) : Number.NEGATIVE_INFINITY;
  }
  checkpoints.set(decisionCount - 1, nextValues.slice());
  const scratch = createFullWindowScratch(gridSize);
  for (let decision = decisionCount - 2; decision >= 0; decision -= 1) {
    const currentValues = fullWindowBellmanRow(
      prices,
      decisionTimes[decision]!,
      decisionTimes[decision + 1]!,
      nextValues,
      grid,
      oracle.execution,
      scratch,
    );
    nextValues = currentValues;
    if (decision % blockSize === 0) checkpoints.set(decision, currentValues.slice());
  }

  const accumulator = createExposureReturnAccumulator(options.initialExposure ?? 0);
  const exposures = oracle.path.exposures;
  const equities = oracle.path.equities;
  equities[start] = accumulator.equity;
  let currentExposure = accumulator.exposure;
  for (let blockStart = 0; blockStart < decisionCount - 1; blockStart += blockSize) {
    const blockEnd = Math.min(decisionCount - 1, blockStart + blockSize);
    const rows = new Float64Array((blockEnd - blockStart + 1) * gridSize);
    rows.set(checkpoints.get(blockEnd)!, (blockEnd - blockStart) * gridSize);
    for (let decision = blockEnd - 1; decision >= blockStart; decision -= 1) {
      const nextRow = rows.subarray(
        (decision - blockStart + 1) * gridSize,
        (decision - blockStart + 2) * gridSize,
      );
      const currentRow = fullWindowBellmanRow(
        prices,
        decisionTimes[decision]!,
        decisionTimes[decision + 1]!,
        nextRow,
        grid,
        oracle.execution,
        scratch,
      );
      rows.set(currentRow, (decision - blockStart) * gridSize);
    }
    for (let decision = blockStart; decision < blockEnd; decision += 1) {
      const from = decisionTimes[decision]!;
      const to = decisionTimes[decision + 1]!;
      const nextRow = rows.subarray(
        (decision - blockStart + 1) * gridSize,
        (decision - blockStart + 2) * gridSize,
      );
      prepareFullWindowForcedValues(
        prices,
        from,
        to,
        nextRow,
        grid,
        oracle.execution,
        scratch,
      );
      const noTrade = holdingBlockOutcome(
        prices,
        from,
        to,
        currentExposure,
        oracle.execution,
      );
      let best = Number.isFinite(noTrade.logReturn)
        ? noTrade.logReturn + interpolateRow(nextRow, 0, grid, noTrade.exposure)
        : Number.NEGATIVE_INFINITY;
      let target = currentExposure;
      for (let targetIndex = 0; targetIndex < gridSize; targetIndex += 1) {
        const rebalance = rebalanceEquityFactor(
          currentExposure,
          grid[targetIndex]!,
          oracle.execution.friction,
        );
        const forced = scratch.forcedValues[targetIndex]!;
        const value = rebalance > 0 && Number.isFinite(forced)
          ? Math.log(rebalance) + forced
          : Number.NEGATIVE_INFINITY;
        if (value > best) {
          best = value;
          target = grid[targetIndex]!;
        }
      }
      const actionChangesExposure = Math.abs(target - currentExposure) > Number.EPSILON;
      for (let time = from; time < to; time += 1) {
        const activeExposure = time === from ? target : accumulator.exposure;
        exposures[time] = activeExposure;
        if (time === from && actionChangesExposure) {
          observeExposureReturn(
            accumulator,
            activeExposure,
            prices[time]!,
            prices[time + 1]!,
            oracle.execution,
          );
        } else {
          observeHeldExposureReturn(
            accumulator,
            prices[time]!,
            prices[time + 1]!,
            oracle.execution,
          );
        }
        equities[time + 1] = accumulator.equity;
      }
      currentExposure = accumulator.exposure;
    }
  }
  closeExposureReturn(accumulator, oracle.execution);
  exposures[terminal] = 0;
  equities[terminal] = accumulator.equity;
  const metrics = finalizeExposureReturn(accumulator);
  const cashBaseline = createExposureReturnAccumulator(options.initialExposure ?? 0);
  for (let time = start; time < terminal; time += 1) {
    observeExposureReturn(
      cashBaseline,
      0,
      prices[time]!,
      prices[time + 1]!,
      oracle.execution,
    );
  }
  closeExposureReturn(cashBaseline, oracle.execution);
  if (metrics.equity + 1e-10 < cashBaseline.equity) {
    throw new Error(
      `Exposure value oracle Q0 ${metrics.equity} is below its cash baseline ${cashBaseline.equity}.`,
    );
  }
  oracle.path = {
    startIndex: start,
    terminalIndex: terminal,
    initialExposure: options.initialExposure ?? 0,
    terminalExposure: 0,
    logReturn: metrics.logReturn,
    totalReturn: metrics.totalReturn,
    exposures,
    equities,
    maxDrawdown: metrics.maxDrawdown,
    turnover: metrics.turnover,
    rebalanceCount: metrics.rebalanceCount,
    liquidationCount: metrics.liquidationCount,
  };
}

interface FullWindowScratch {
  forcedValues: Float64Array;
  endpointExposures: Float64Array;
  prefixValues: Float64Array;
  suffixValues: Float64Array;
  prefixTargets: Uint16Array;
  suffixTargets: Uint16Array;
}

function createFullWindowScratch(gridSize: number): FullWindowScratch {
  return {
    forcedValues: new Float64Array(gridSize),
    endpointExposures: new Float64Array(gridSize),
    prefixValues: new Float64Array(gridSize),
    suffixValues: new Float64Array(gridSize),
    prefixTargets: new Uint16Array(gridSize),
    suffixTargets: new Uint16Array(gridSize),
  };
}

function fullWindowBellmanRow(
  prices: ArrayLike<number>,
  start: number,
  end: number,
  nextValues: Float64Array,
  grid: Float64Array,
  execution: ExposureExecutionOptions,
  scratch: FullWindowScratch,
): Float64Array {
  prepareFullWindowForcedValues(prices, start, end, nextValues, grid, execution, scratch);
  const result = new Float64Array(grid.length);
  fillOptimalContinuation(
    scratch.forcedValues,
    result,
    0,
    grid,
    execution.friction,
    1 - execution.friction * grid[grid.length - 1]! > 0
      && 1 - execution.friction + execution.friction * grid[0]! > 0,
    scratch.prefixValues,
    scratch.suffixValues,
    scratch.prefixTargets,
    scratch.suffixTargets,
  );
  return result;
}

function prepareFullWindowForcedValues(
  prices: ArrayLike<number>,
  start: number,
  end: number,
  nextValues: Float64Array,
  grid: Float64Array,
  execution: ExposureExecutionOptions,
  scratch: FullWindowScratch,
): void {
  for (let targetIndex = 0; targetIndex < grid.length; targetIndex += 1) {
    const target = grid[targetIndex]!;
    const holding = holdingBlockOutcome(prices, start, end, target, execution);
    scratch.endpointExposures[targetIndex] = holding.exposure;
    scratch.forcedValues[targetIndex] = Number.isFinite(holding.logReturn)
      ? holding.logReturn + interpolateRow(nextValues, 0, grid, holding.exposure)
      : Number.NEGATIVE_INFINITY;
  }
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
  for (let exposureIndex = 0; exposureIndex < grid.length; exposureIndex += 1) {
    const target = grid[exposureIndex]!;
    for (let time = 0; time < prices.length; time += 1) {
      const cell = time * grid.length + exposureIndex;
      const endpointTime = Math.min(prices.length - 1, time + duration);
      if (endpointTime === time) {
        values[cell] = 0;
        endpointExposures[cell] = target;
        continue;
      }
      const holding = holdingBlockOutcome(prices, time, endpointTime, target, execution);
      values[cell] = holding.logReturn;
      endpointExposures[cell] = holding.exposure;
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
  const maxEffectiveExposure = options.maxEffectiveExposure ?? 250;
  const currentHalfIntervals = Math.max(1, Math.floor(options.gridSize / 2));
  const currentGrid = float64(currentHalfIntervals * 2 + 1, shared);
  for (let index = 0; index < currentGrid.length; index += 1) {
    currentGrid[index] = -maxEffectiveExposure
      + index / (currentGrid.length - 1) * (2 * maxEffectiveExposure);
  }
  const initialExposure = options.initialExposure ?? 0;
  const emptyPath = {
    startIndex: options.scoreStartIndex,
    terminalIndex: options.terminalIndex ?? prices.length - 1,
    initialExposure,
    terminalExposure: 0 as const,
    logReturn: 0,
    totalReturn: 0,
    exposures: float32(prices.length, shared),
    equities: float64(prices.length, shared),
    maxDrawdown: 0,
    turnover: 0,
    rebalanceCount: 0,
    liquidationCount: 0,
  };
  return {
    scoreStartIndex: options.scoreStartIndex,
    holdingPeriodSteps,
    valueHorizonSteps,
    temperature: options.temperature,
    grid,
    currentGrid,
    means: float32(prices.length, shared),
    secondMoments: float32(prices.length, shared),
    modalExposures: float32(prices.length, shared),
    entropies: float32(prices.length, shared),
    policyMeans: float32(prices.length, shared),
    policySecondMoments: float32(prices.length, shared),
    policyMeanLogRebalances: float32(prices.length, shared),
    policyEntropies: float32(prices.length, shared),
    averageRegrets: float32(prices.length, shared),
    weights: float32(prices.length, shared),
    opportunities: float32(prices.length, shared),
    ...(options.includeProbabilities
      ? { probabilities: float32(prices.length * grid.length, shared) }
      : {}),
    execution: {
      friction: Math.max(0, options.friction),
      minExposure,
      maxExposure,
      maxEffectiveExposure,
      quoteLendRate: options.quoteLendRate ?? 0,
      quoteBorrowRate: options.quoteBorrowRate ?? 0,
      assetBorrowRate: options.assetBorrowRate ?? 0,
    },
    path: emptyPath,
  };
}

export function shareExposureValueOracle(oracle: ExposureValueOracle): ExposureValueOracle {
  return {
    scoreStartIndex: oracle.scoreStartIndex,
    holdingPeriodSteps: oracle.holdingPeriodSteps,
    valueHorizonSteps: oracle.valueHorizonSteps,
    temperature: oracle.temperature,
    grid: sharedCopy(oracle.grid),
    currentGrid: sharedCopy(oracle.currentGrid),
    means: sharedCopy(oracle.means),
    secondMoments: sharedCopy(oracle.secondMoments),
    modalExposures: sharedCopy(oracle.modalExposures),
    entropies: sharedCopy(oracle.entropies),
    policyMeans: sharedCopy(oracle.policyMeans),
    policySecondMoments: sharedCopy(oracle.policySecondMoments),
    policyMeanLogRebalances: sharedCopy(oracle.policyMeanLogRebalances),
    policyEntropies: sharedCopy(oracle.policyEntropies),
    averageRegrets: sharedCopy(oracle.averageRegrets),
    weights: sharedCopy(oracle.weights),
    opportunities: sharedCopy(oracle.opportunities),
    ...(oracle.probabilities ? { probabilities: sharedCopy(oracle.probabilities) } : {}),
    execution: { ...oracle.execution },
    path: {
      ...oracle.path,
      exposures: sharedCopy(oracle.path.exposures),
      equities: sharedCopy(oracle.path.equities),
    },
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
    entropies: oracle.entropies.subarray(0, length),
    policyMeans: oracle.policyMeans.subarray(0, length),
    policySecondMoments: oracle.policySecondMoments.subarray(0, length),
    policyMeanLogRebalances: oracle.policyMeanLogRebalances.subarray(0, length),
    policyEntropies: oracle.policyEntropies.subarray(0, length),
    averageRegrets: oracle.averageRegrets.subarray(0, length),
    weights: oracle.weights.subarray(0, length),
    opportunities: oracle.opportunities.subarray(0, length),
    ...(oracle.probabilities
      ? { probabilities: oracle.probabilities.subarray(0, length * oracle.grid.length) }
      : {}),
    path: {
      ...oracle.path,
      exposures: oracle.path.exposures.subarray(0, length),
      equities: oracle.path.equities.subarray(0, length),
    },
  };
}

export function exposureValueOracleBytes(oracle: ExposureValueOracle): number {
  return oracle.grid.byteLength
    + oracle.currentGrid.byteLength
    + oracle.means.byteLength
    + oracle.secondMoments.byteLength
    + oracle.modalExposures.byteLength
    + oracle.entropies.byteLength
    + oracle.policyMeans.byteLength
    + oracle.policySecondMoments.byteLength
    + oracle.policyMeanLogRebalances.byteLength
    + oracle.policyEntropies.byteLength
    + oracle.averageRegrets.byteLength
    + oracle.weights.byteLength
    + oracle.opportunities.byteLength
    + oracle.path.exposures.byteLength
    + oracle.path.equities.byteLength
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
  normalCenter = 0,
  normalMixture = 0,
  normalSigma = 25,
): Float64Array {
  if (grid.length < 2) {
    throw new Error("Strategy exposure distribution requires at least two grid points.");
  }
  const slope = strategyExposureLogSlope(rateBpsPerHour, holdingPeriodMs, temperature);
  if (!Number.isFinite(quadraticCoefficient)) {
    throw new Error("Strategy exposure distribution requires a finite quadratic coefficient.");
  }
  validateStrategyNormalMixture(normalCenter, normalMixture, normalSigma);
  const quadraticLogNormalizer = quadraticExponentialLogNormalizer(
    grid.length,
    grid[0]!,
    grid[grid.length - 1]!,
    slope,
    quadraticCoefficient,
  );
  const normalQuadraticCoefficient = -0.5 / (normalSigma * normalSigma);
  const normalLinearCoefficient = normalCenter / (normalSigma * normalSigma);
  const normalLogNormalizer = normalMixture > 0
    ? quadraticExponentialLogNormalizer(
        grid.length,
        grid[0]!,
        grid[grid.length - 1]!,
        normalLinearCoefficient,
        normalQuadraticCoefficient,
      )
    : 0;
  const probabilities = new Float64Array(grid.length);
  let maximum = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < grid.length; index += 1) {
    const exposure = grid[index]!;
    const logWeight = strategyMixtureBaseLogit(
      exposure,
      slope,
      quadraticCoefficient,
      quadraticLogNormalizer,
      normalCenter,
      normalMixture,
      normalSigma,
      normalLinearCoefficient,
      normalQuadraticCoefficient,
      normalLogNormalizer,
    );
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

function validateStrategyNormalMixture(
  normalCenter: number,
  normalMixture: number,
  normalSigma: number,
): void {
  if (!Number.isFinite(normalCenter)
    || !Number.isFinite(normalMixture) || normalMixture < 0 || normalMixture > 1
    || !Number.isFinite(normalSigma) || normalSigma <= 0) {
    throw new Error(
      "Strategy normal mixture requires a finite center, a weight from zero through one, and positive sigma.",
    );
  }
}

function strategyMixtureBaseLogit(
  exposure: number,
  linearCoefficient: number,
  quadraticCoefficient: number,
  quadraticLogNormalizer: number,
  normalCenter: number,
  normalMixture: number,
  normalSigma: number,
  normalLinearCoefficient = normalCenter / (normalSigma * normalSigma),
  normalQuadraticCoefficient = -0.5 / (normalSigma * normalSigma),
  normalLogNormalizer = 0,
): number {
  const quadraticLogProbability = linearCoefficient * exposure
    + quadraticCoefficient * exposure * exposure
    - quadraticLogNormalizer;
  if (normalMixture <= 0) return quadraticLogProbability;
  const normalLogProbability = normalLinearCoefficient * exposure
    + normalQuadraticCoefficient * exposure * exposure
    - normalLogNormalizer;
  if (normalMixture >= 1) return normalLogProbability;
  return logAddExp(
    Math.log1p(-normalMixture) + quadraticLogProbability,
    Math.log(normalMixture) + normalLogProbability,
  );
}

function logAddExp(left: number, right: number): number {
  if (!Number.isFinite(left)) return right;
  if (!Number.isFinite(right)) return left;
  const maximum = Math.max(left, right);
  return maximum + Math.log(Math.exp(left - maximum) + Math.exp(right - maximum));
}

function fillStrategyMixtureBaseLogits(
  output: Float64Array,
  grid: Float64Array,
  linearCoefficient: number,
  quadraticCoefficient: number,
  normalCenter: number,
  normalMixture: number,
  normalSigma: number,
): number {
  validateStrategyNormalMixture(normalCenter, normalMixture, normalSigma);
  const quadraticLogNormalizer = quadraticExponentialLogNormalizer(
    grid.length,
    grid[0]!,
    grid[grid.length - 1]!,
    linearCoefficient,
    quadraticCoefficient,
  );
  const normalQuadraticCoefficient = -0.5 / (normalSigma * normalSigma);
  const normalLinearCoefficient = normalCenter / (normalSigma * normalSigma);
  const normalLogNormalizer = normalMixture > 0
    ? quadraticExponentialLogNormalizer(
        grid.length,
        grid[0]!,
        grid[grid.length - 1]!,
        normalLinearCoefficient,
        normalQuadraticCoefficient,
      )
    : 0;
  for (let index = 0; index < grid.length; index += 1) {
    output[index] = strategyMixtureBaseLogit(
      grid[index]!,
      linearCoefficient,
      quadraticCoefficient,
      quadraticLogNormalizer,
      normalCenter,
      normalMixture,
      normalSigma,
      normalLinearCoefficient,
      normalQuadraticCoefficient,
      normalLogNormalizer,
    );
  }
  return quadraticLogNormalizer;
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

export function createExposureReturnAccumulator(initialExposure = 0): ExposureReturnAccumulator {
  if (!Number.isFinite(initialExposure)) {
    throw new Error("Exposure return initial exposure must be finite.");
  }
  return {
    equity: 1,
    exposure: initialExposure,
    peakEquity: 1,
    maxDrawdown: 0,
    turnover: 0,
    rebalanceCount: 0,
    liquidationCount: 0,
    sampleCount: 0,
  };
}

export function closeExposureReturn(
  accumulator: ExposureReturnAccumulator,
  options: ExposureExecutionOptions,
): void {
  if (!(accumulator.equity > 0) || Math.abs(accumulator.exposure) <= Number.EPSILON) {
    accumulator.exposure = 0;
    return;
  }
  const change = Math.abs(accumulator.exposure);
  const rebalance = rebalanceEquityFactor(accumulator.exposure, 0, options.friction);
  if (!(rebalance > 0)) {
    accumulator.equity = 0;
    accumulator.exposure = 0;
    accumulator.maxDrawdown = 1;
    return;
  }
  accumulator.equity *= rebalance;
  accumulator.exposure = 0;
  accumulator.turnover += change;
  accumulator.rebalanceCount += 1;
  updateDrawdown(accumulator);
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
  observeHeldExposureReturn(accumulator, price, nextPrice, options, target);
}

function observeHeldExposureReturn(
  accumulator: ExposureReturnAccumulator,
  price: number,
  nextPrice: number,
  options: ExposureExecutionOptions,
  exposure = accumulator.exposure,
): void {
  const holding = holdingOutcome(exposure, price, nextPrice, options);
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

export function exposureValueOraclePathMetrics(oracle: ExposureValueOracle): ExposureReturnMetrics {
  const equity = Math.exp(oracle.path.logReturn);
  let peakEquity = 1;
  for (const value of oracle.path.equities) peakEquity = Math.max(peakEquity, value);
  return {
    equity,
    exposure: 0,
    peakEquity,
    maxDrawdown: oracle.path.maxDrawdown,
    turnover: oracle.path.turnover,
    rebalanceCount: oracle.path.rebalanceCount,
    liquidationCount: oracle.path.liquidationCount,
    sampleCount: Math.max(0, oracle.path.terminalIndex - oracle.path.startIndex),
    totalReturn: oracle.path.totalReturn,
    logReturn: oracle.path.logReturn,
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
    averageRegretSum: 0,
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
    conditionalBinScratch: null,
    transitionScratch: null,
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
  frictionFraction = 1,
  normalCenter = 0,
  normalMixture = 0,
  normalSigma = 25,
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
  if (!Number.isFinite(frictionFraction) || frictionFraction < 0) {
    throw new Error("Strategy transition friction fraction must be finite and non-negative.");
  }
  const lossEnabled = state.config.entropyGapLambda > 0
    || state.config.stateMutualInformationLambda > 0
    || state.config.oracleMutualInformationLambda > 0;
  const preciseEnabled = state.config.oracleMutualInformationLambda > 0
    && state.config.oracleMutualInformationMode === "precise";
  state.transitionScratch ??= createTransitionDistributionScratch(oracle.grid.length);
  const scratch = state.transitionScratch;
  const quadraticLogNormalizer = fillStrategyMixtureBaseLogits(
    scratch.baseLogits,
    oracle.grid,
    slope,
    quadraticCoefficient,
    normalCenter,
    normalMixture,
    normalSigma,
  );
  const statistics = transitionDistributionStatistics(
    scratch.baseLogits,
    oracle.grid,
    oracle.currentGrid,
    oracle.execution.friction,
    frictionFraction / temperature,
    scratch,
  );
  const mean = oracle.policyMeans[candleIndex]!;
  const secondMoment = oracle.policySecondMoments[candleIndex]!;
  let targetBaseLogit = slope * mean
    + quadraticCoefficient * secondMoment
    - quadraticLogNormalizer;
  if (normalMixture > 0) {
    if (!oracle.probabilities) {
      throw new Error(
        "Strategy normal-mixture loss requires retained oracle probabilities.",
      );
    }
    targetBaseLogit = transitionCrossExpectation(
      oracle.probabilities.subarray(
        candleIndex * oracle.grid.length,
        (candleIndex + 1) * oracle.grid.length,
      ),
      scratch.baseLogits,
      oracle.grid,
      oracle.currentGrid,
      oracle.execution.friction,
      1 / oracle.temperature,
      scratch,
    );
  }
  const crossEntropy = Math.max(
    0,
    statistics.logNormalizer
      - targetBaseLogit
      - frictionFraction / temperature * oracle.policyMeanLogRebalances[candleIndex]!,
  );
  const weight = oracle.weights[candleIndex]!;
  accumulator.weightedCrossEntropy += weight * crossEntropy;
  accumulator.weightedOracleEntropy += weight * oracle.policyEntropies[candleIndex]!;
  if (lossEnabled) {
    const logGridSize = Math.log(oracle.grid.length);
    const oracleEntropy = oracle.policyEntropies[candleIndex]!;
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
      const diagonalProbabilities = strategyExposureProbabilities(
        oracle.grid,
        rateBpsPerHour,
        intervalMs * oracle.holdingPeriodSteps,
        temperature,
        quadraticCoefficient,
        normalCenter,
        normalMixture,
        normalSigma,
      );
      const oracleProbabilities = oracle.probabilities;
      if (!oracleProbabilities) {
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
      state.conditionalBinScratch ??= createExposureConditionalBinScratch(
        oracle.grid.length,
        bins,
      );
      binnedConditionalExposureProbabilities(
        oracleProbabilities.subarray(probabilityOffset, probabilityOffset + oracle.grid.length),
        oracle.grid,
        oracle.currentGrid,
        oracle.execution.friction,
        1 / oracle.temperature,
        bins,
        oracleBins,
        state.conditionalBinScratch,
      );
      binnedConditionalExposureProbabilities(
        diagonalProbabilities,
        oracle.grid,
        oracle.currentGrid,
        oracle.execution.friction,
        frictionFraction / temperature,
        bins,
        strategyBins,
        state.conditionalBinScratch,
      );
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
  accumulator.averageRegretSum += oracle.averageRegrets[candleIndex]!;
  accumulator.sampleCount += 1;
}

/** Bins the conditional target marginal after uniformly averaging current grid states. */
export function binnedConditionalExposureProbabilities(
  baseProbabilities: ArrayLike<number>,
  grid: Float64Array,
  currentGrid: Float64Array,
  friction: number,
  transitionLogScale: number,
  bins: number,
  result: Float64Array<ArrayBufferLike> = new Float64Array(bins),
  scratch = createExposureConditionalBinScratch(grid.length, bins),
): Float64Array<ArrayBufferLike> {
  result.fill(0);
  if (scratch.baseLogs.length !== grid.length || scratch.sellBins.length !== bins) {
    throw new Error("Conditional exposure bin scratch does not match the grid and bin count.");
  }
  if (!hasSeparableCurrentStateFactors(currentGrid, friction)) {
    for (const current of currentGrid) {
      let total = 0;
      const weights = new Float64Array(grid.length);
      for (let targetIndex = 0; targetIndex < grid.length; targetIndex += 1) {
        const factor = rebalanceEquityFactor(current, grid[targetIndex]!, friction);
        const weight = factor > 0 && baseProbabilities[targetIndex]! > 0
          ? baseProbabilities[targetIndex]! * Math.exp(transitionLogScale * Math.log(factor))
          : 0;
        weights[targetIndex] = weight;
        total += weight;
      }
      if (!(total > 0)) continue;
      for (let targetIndex = 0; targetIndex < grid.length; targetIndex += 1) {
        const bin = Math.min(bins - 1, Math.floor(targetIndex * bins / grid.length));
        result[bin] += weights[targetIndex]! / total / currentGrid.length;
      }
    }
    return result;
  }
  const { baseLogs, sellTargetWeights, buyTargetWeights, sellBins, buyBins } = scratch;
  sellBins.fill(0);
  buyBins.fill(0);
  let maximumAdjusted = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < grid.length; index += 1) {
    const probability = baseProbabilities[index]!;
    const base = probability > 0 ? Math.log(probability) : Math.log(1e-300);
    const exposure = grid[index]!;
    const sellDenominator = 1 - friction * exposure;
    const buyDenominator = 1 - friction + friction * exposure;
    baseLogs[index] = base;
    maximumAdjusted = Math.max(
      maximumAdjusted,
      base - transitionLogScale * Math.log(sellDenominator),
      base - transitionLogScale * Math.log(buyDenominator),
    );
  }
  let sellTotal = 0;
  let buyTotal = 0;
  for (let index = 0; index < grid.length; index += 1) {
    const exposure = grid[index]!;
    const sellWeight = Math.exp(
      baseLogs[index]! - transitionLogScale * Math.log(1 - friction * exposure)
        - maximumAdjusted,
    );
    const buyWeight = Math.exp(
      baseLogs[index]! - transitionLogScale * Math.log(1 - friction + friction * exposure)
        - maximumAdjusted,
    );
    const bin = Math.min(bins - 1, Math.floor(index * bins / grid.length));
    sellTargetWeights[index] = sellWeight;
    buyTargetWeights[index] = buyWeight;
    buyBins[bin] += buyWeight;
    buyTotal += buyWeight;
  }
  let targetCursor = 0;
  for (const current of currentGrid) {
    while (targetCursor < grid.length && grid[targetCursor]! <= current) {
      const targetBin = Math.min(bins - 1, Math.floor(targetCursor * bins / grid.length));
      sellBins[targetBin] += sellTargetWeights[targetCursor]!;
      sellTotal += sellTargetWeights[targetCursor]!;
      buyBins[targetBin] = Math.max(0, buyBins[targetBin]! - buyTargetWeights[targetCursor]!);
      buyTotal = Math.max(0, buyTotal - buyTargetWeights[targetCursor]!);
      targetCursor += 1;
    }
    const sellLogScale = transitionLogScale * Math.log(1 - friction * current);
    const buyLogScale = transitionLogScale * Math.log(1 - friction + friction * current);
    const rowMaximum = Math.max(
      sellTotal > 0 ? sellLogScale + Math.log(sellTotal) : Number.NEGATIVE_INFINITY,
      buyTotal > 0 ? buyLogScale + Math.log(buyTotal) : Number.NEGATIVE_INFINITY,
    );
    const sellScale = Math.exp(sellLogScale - rowMaximum);
    const buyScale = Math.exp(buyLogScale - rowMaximum);
    const rowTotal = sellScale * sellTotal + buyScale * buyTotal;
    if (!(rowTotal > 0)) continue;
    for (let bin = 0; bin < bins; bin += 1) {
      result[bin] += (sellScale * sellBins[bin]! + buyScale * buyBins[bin]!)
        / rowTotal / currentGrid.length;
    }
  }
  return result;
}

export function createExposureConditionalBinScratch(
  gridSize: number,
  bins: number,
): ExposureConditionalBinScratch {
  return {
    baseLogs: new Float64Array(gridSize),
    sellTargetWeights: new Float64Array(gridSize),
    buyTargetWeights: new Float64Array(gridSize),
    sellBins: new Float64Array(bins),
    buyBins: new Float64Array(bins),
  };
}

/**
 * Returns the exact conditional target policy represented by the transition-aware
 * distillation loss for one current exposure. The input probabilities are the
 * post-action target distribution before transition friction is applied.
 */
export function conditionalExposureProbabilities(
  baseProbabilities: ArrayLike<number>,
  grid: ArrayLike<number>,
  currentExposure: number,
  friction: number,
  transitionLogScale: number,
  result: Float64Array<ArrayBufferLike> = new Float64Array(grid.length),
): Float64Array<ArrayBufferLike> {
  if (baseProbabilities.length !== grid.length || grid.length < 2) {
    throw new Error("Conditional exposure probabilities require matching probability and grid arrays.");
  }
  if (!Number.isFinite(currentExposure) || !Number.isFinite(friction) || friction < 0
    || !Number.isFinite(transitionLogScale) || transitionLogScale < 0) {
    throw new Error("Conditional exposure probabilities require finite state and non-negative costs.");
  }
  if (result.length !== grid.length) {
    throw new Error("Conditional exposure probability output does not match the grid.");
  }
  let maximum = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < grid.length; index += 1) {
    const probability = baseProbabilities[index]!;
    const factor = rebalanceEquityFactor(currentExposure, grid[index]!, friction);
    const logit = probability > 0 && factor > 0
      ? Math.log(probability) + transitionLogScale * Math.log(factor)
      : Number.NEGATIVE_INFINITY;
    result[index] = logit;
    maximum = Math.max(maximum, logit);
  }
  let total = 0;
  for (let index = 0; index < result.length; index += 1) {
    const probability = Number.isFinite(result[index]!)
      ? Math.exp(result[index]! - maximum)
      : 0;
    result[index] = probability;
    total += probability;
  }
  if (!(total > 0)) {
    throw new Error("Conditional exposure policy has no valid target action.");
  }
  for (let index = 0; index < result.length; index += 1) result[index] /= total;
  return result;
}

export interface ConditionalQuadraticPolicyFitOptions {
  friction: number;
  transitionLogScale: number;
  objective?: "forward-kl" | "probability-mse";
  initialLinearCoefficient?: number;
  initialQuadraticCoefficient?: number;
  constrainQuadraticNonPositive?: boolean;
  maxIterations?: number;
  tolerance?: number;
}

export interface ConditionalQuadraticPolicyFit {
  objective: "forward-kl" | "probability-mse";
  linearCoefficient: number;
  quadraticCoefficient: number;
  crossEntropy: number;
  klDivergence: number;
  meanSquaredError: number;
  iterations: number;
  converged: boolean;
}

/** Evaluates the predictor's exact conditional quadratic-exponential policy. */
export function conditionalQuadraticExposureProbabilities(
  grid: ArrayLike<number>,
  currentExposure: number,
  linearCoefficient: number,
  quadraticCoefficient: number,
  friction: number,
  transitionLogScale: number,
  result: Float64Array<ArrayBufferLike> = new Float64Array(grid.length),
): Float64Array<ArrayBufferLike> {
  if (grid.length < 2 || result.length !== grid.length
    || !Number.isFinite(currentExposure) || !Number.isFinite(linearCoefficient)
    || !Number.isFinite(quadraticCoefficient) || !Number.isFinite(friction) || friction < 0
    || !Number.isFinite(transitionLogScale) || transitionLogScale < 0) {
    throw new Error("Conditional quadratic policy requires a valid grid and finite coefficients.");
  }
  let maximum = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < grid.length; index += 1) {
    const exposure = grid[index]!;
    const factor = rebalanceEquityFactor(currentExposure, exposure, friction);
    const logit = factor > 0
      ? linearCoefficient * exposure + quadraticCoefficient * exposure * exposure
        + transitionLogScale * Math.log(factor)
      : Number.NEGATIVE_INFINITY;
    result[index] = logit;
    maximum = Math.max(maximum, logit);
  }
  let total = 0;
  for (let index = 0; index < result.length; index += 1) {
    const weight = Number.isFinite(result[index]!) ? Math.exp(result[index]! - maximum) : 0;
    result[index] = weight;
    total += weight;
  }
  if (!(total > 0)) throw new Error("Conditional quadratic policy has no valid target action.");
  for (let index = 0; index < result.length; index += 1) result[index] /= total;
  return result;
}

/**
 * Projects one or more target policy rows onto the predictor's shared
 * b1*a + b2*a^2 family. Forward KL is the default training-aligned objective;
 * probability MSE provides a literal pointwise least-squares visualization fit.
 * Current exposures identify each row and the transition term remains fixed.
 */
export function fitConditionalQuadraticPolicy(
  grid: ArrayLike<number>,
  targetProbabilities: ArrayLike<number>,
  currentExposures: ArrayLike<number>,
  options: ConditionalQuadraticPolicyFitOptions,
): ConditionalQuadraticPolicyFit {
  if (options.objective === "probability-mse") {
    return fitConditionalQuadraticPolicyMse(
      grid,
      targetProbabilities,
      currentExposures,
      options,
    );
  }
  const size = grid.length;
  const rows = currentExposures.length;
  if (size < 3 || rows < 1 || targetProbabilities.length !== rows * size) {
    throw new Error("Conditional quadratic fit requires complete target rows on one exposure grid.");
  }
  const minimum = grid[0]!;
  const maximum = grid[size - 1]!;
  const center = (minimum + maximum) / 2;
  const exposureScale = (maximum - minimum) / 2;
  if (!(exposureScale > 0) || !Number.isFinite(options.friction) || options.friction < 0
    || !Number.isFinite(options.transitionLogScale) || options.transitionLogScale < 0) {
    throw new Error("Conditional quadratic fit requires ordered exposures and non-negative costs.");
  }
  const normalizedExposures = Float64Array.from(
    { length: size },
    (_, index) => (grid[index]! - center) / exposureScale,
  );
  const targetMeans = new Float64Array(rows * 3);
  let targetEntropy = 0;
  for (let row = 0; row < rows; row += 1) {
    const current = currentExposures[row]!;
    if (!Number.isFinite(current)) throw new Error("Conditional quadratic fit state is not finite.");
    let probabilityTotal = 0;
    for (let index = 0; index < size; index += 1) {
      const probability = targetProbabilities[row * size + index]!;
      if (!Number.isFinite(probability) || probability < 0) {
        throw new Error("Conditional quadratic fit probabilities must be finite and non-negative.");
      }
      probabilityTotal += probability;
    }
    if (!(probabilityTotal > 0)) throw new Error("Conditional quadratic fit target row is empty.");
    for (let index = 0; index < size; index += 1) {
      const probability = targetProbabilities[row * size + index]! / probabilityTotal;
      if (!(probability > 0)) continue;
      const z = normalizedExposures[index]!;
      const factor = rebalanceEquityFactor(current, grid[index]!, options.friction);
      if (!(factor > 0)) continue;
      targetMeans[row * 3] += probability * z;
      targetMeans[row * 3 + 1] += probability * z * z;
      targetMeans[row * 3 + 2] += probability * Math.log(factor);
      targetEntropy -= probability * Math.log(probability) / rows;
    }
  }

  const constrained = options.constrainQuadraticNonPositive ?? true;
  const initialB2 = options.initialQuadraticCoefficient ?? 0;
  let beta2 = initialB2 * exposureScale * exposureScale;
  let beta1 = exposureScale * (options.initialLinearCoefficient ?? 0)
    + 2 * center * exposureScale * initialB2;
  if (constrained) beta2 = Math.min(0, beta2);
  const evaluate = (candidateBeta1: number, candidateBeta2: number) => {
    let objective = 0;
    let gradient1 = 0;
    let gradient2 = 0;
    let hessian11 = 0;
    let hessian12 = 0;
    let hessian22 = 0;
    const logits = new Float64Array(size);
    for (let row = 0; row < rows; row += 1) {
      const current = currentExposures[row]!;
      let rowMaximum = Number.NEGATIVE_INFINITY;
      for (let index = 0; index < size; index += 1) {
        const z = normalizedExposures[index]!;
        const factor = rebalanceEquityFactor(current, grid[index]!, options.friction);
        const logit = factor > 0
          ? candidateBeta1 * z + candidateBeta2 * z * z
            + options.transitionLogScale * Math.log(factor)
          : Number.NEGATIVE_INFINITY;
        logits[index] = logit;
        rowMaximum = Math.max(rowMaximum, logit);
      }
      let total = 0;
      let moment1 = 0;
      let moment2 = 0;
      let moment3 = 0;
      let moment4 = 0;
      for (let index = 0; index < size; index += 1) {
        if (!Number.isFinite(logits[index])) continue;
        const weight = Math.exp(logits[index]! - rowMaximum);
        const z = normalizedExposures[index]!;
        const z2 = z * z;
        total += weight;
        moment1 += weight * z;
        moment2 += weight * z2;
        moment3 += weight * z2 * z;
        moment4 += weight * z2 * z2;
      }
      const modelMean1 = moment1 / total;
      const modelMean2 = moment2 / total;
      const targetMean1 = targetMeans[row * 3]!;
      const targetMean2 = targetMeans[row * 3 + 1]!;
      objective += (
        rowMaximum + Math.log(total)
          - candidateBeta1 * targetMean1
          - candidateBeta2 * targetMean2
          - options.transitionLogScale * targetMeans[row * 3 + 2]!
      ) / rows;
      gradient1 += (modelMean1 - targetMean1) / rows;
      gradient2 += (modelMean2 - targetMean2) / rows;
      hessian11 += Math.max(0, moment2 / total - modelMean1 * modelMean1) / rows;
      hessian12 += (moment3 / total - modelMean1 * modelMean2) / rows;
      hessian22 += Math.max(0, moment4 / total - modelMean2 * modelMean2) / rows;
    }
    return { objective, gradient1, gradient2, hessian11, hessian12, hessian22 };
  };

  const maximumIterations = Math.max(1, Math.floor(options.maxIterations ?? 40));
  const tolerance = Math.max(Number.EPSILON, options.tolerance ?? 1e-10);
  let state = evaluate(beta1, beta2);
  let converged = false;
  let iterations = 0;
  for (; iterations < maximumIterations; iterations += 1) {
    const quadraticAtBoundary = constrained && beta2 >= -tolerance && state.gradient2 < 0;
    const projectedGradient2 = quadraticAtBoundary ? 0 : state.gradient2;
    if (Math.hypot(state.gradient1, projectedGradient2) <= tolerance) {
      converged = true;
      break;
    }
    const ridge = 1e-10;
    const h11 = state.hessian11 + ridge;
    const h22 = state.hessian22 + ridge;
    let step1: number;
    let step2: number;
    if (quadraticAtBoundary) {
      step1 = -state.gradient1 / h11;
      step2 = 0;
    } else {
      const determinant = h11 * h22 - state.hessian12 * state.hessian12;
      if (determinant > 1e-14) {
        step1 = -(h22 * state.gradient1 - state.hessian12 * state.gradient2) / determinant;
        step2 = -(-state.hessian12 * state.gradient1 + h11 * state.gradient2) / determinant;
      } else {
        step1 = -state.gradient1 / h11;
        step2 = -state.gradient2 / h22;
      }
    }
    let accepted = false;
    for (let lineScale = 1; lineScale >= 1 / 4096; lineScale /= 2) {
      const nextBeta1 = beta1 + lineScale * step1;
      const nextBeta2 = constrained
        ? Math.min(0, beta2 + lineScale * step2)
        : beta2 + lineScale * step2;
      const next = evaluate(nextBeta1, nextBeta2);
      if (next.objective <= state.objective + 1e-13) {
        beta1 = nextBeta1;
        beta2 = nextBeta2;
        state = next;
        accepted = true;
        break;
      }
    }
    if (!accepted) break;
  }
  if (!converged) {
    const quadraticAtBoundary = constrained && beta2 >= -tolerance && state.gradient2 < 0;
    converged = Math.hypot(state.gradient1, quadraticAtBoundary ? 0 : state.gradient2)
      <= Math.sqrt(tolerance);
  }
  const quadraticCoefficient = beta2 / (exposureScale * exposureScale);
  const linearCoefficient = beta1 / exposureScale - 2 * center * quadraticCoefficient;
  const diagnostics = conditionalQuadraticFitDiagnostics(
    grid,
    targetProbabilities,
    currentExposures,
    linearCoefficient,
    quadraticCoefficient,
    options,
  );
  return {
    objective: "forward-kl",
    linearCoefficient,
    quadraticCoefficient,
    crossEntropy: state.objective,
    klDivergence: Math.max(0, state.objective - targetEntropy),
    meanSquaredError: diagnostics.meanSquaredError,
    iterations,
    converged,
  };
}

function fitConditionalQuadraticPolicyMse(
  grid: ArrayLike<number>,
  targetProbabilities: ArrayLike<number>,
  currentExposures: ArrayLike<number>,
  options: ConditionalQuadraticPolicyFitOptions,
): ConditionalQuadraticPolicyFit {
  const klSeed = fitConditionalQuadraticPolicy(
    grid,
    targetProbabilities,
    currentExposures,
    { ...options, objective: "forward-kl" },
  );
  const size = grid.length;
  const rows = currentExposures.length;
  const minimum = grid[0]!;
  const maximum = grid[size - 1]!;
  const center = (minimum + maximum) / 2;
  const exposureScale = (maximum - minimum) / 2;
  const normalizedExposures = Float64Array.from(
    { length: size },
    (_, index) => (grid[index]! - center) / exposureScale,
  );
  const targets = new Float64Array(rows * size);
  const fixedLogits = new Float64Array(rows * size);
  for (let row = 0; row < rows; row += 1) {
    let total = 0;
    for (let index = 0; index < size; index += 1) {
      total += targetProbabilities[row * size + index]!;
    }
    for (let index = 0; index < size; index += 1) {
      const offset = row * size + index;
      targets[offset] = targetProbabilities[offset]! / total;
      const factor = rebalanceEquityFactor(
        currentExposures[row]!,
        grid[index]!,
        options.friction,
      );
      fixedLogits[offset] = factor > 0
        ? options.transitionLogScale * Math.log(factor)
        : Number.NEGATIVE_INFINITY;
    }
  }
  const divisor = rows * size;
  const logits = new Float64Array(size);
  const probabilities = new Float64Array(size);
  const evaluate = (candidateBeta1: number, candidateBeta2: number) => {
    let objective = 0;
    let gradient1 = 0;
    let gradient2 = 0;
    let hessian11 = 0;
    let hessian12 = 0;
    let hessian22 = 0;
    for (let row = 0; row < rows; row += 1) {
      let rowMaximum = Number.NEGATIVE_INFINITY;
      for (let index = 0; index < size; index += 1) {
        const fixed = fixedLogits[row * size + index]!;
        const z = normalizedExposures[index]!;
        const logit = Number.isFinite(fixed)
          ? candidateBeta1 * z + candidateBeta2 * z * z + fixed
          : Number.NEGATIVE_INFINITY;
        logits[index] = logit;
        rowMaximum = Math.max(rowMaximum, logit);
      }
      let total = 0;
      let modelMean1 = 0;
      let modelMean2 = 0;
      for (let index = 0; index < size; index += 1) {
        const probability = Number.isFinite(logits[index])
          ? Math.exp(logits[index]! - rowMaximum)
          : 0;
        probabilities[index] = probability;
        total += probability;
      }
      for (let index = 0; index < size; index += 1) {
        const probability = probabilities[index]! / total;
        const z = normalizedExposures[index]!;
        probabilities[index] = probability;
        modelMean1 += probability * z;
        modelMean2 += probability * z * z;
      }
      for (let index = 0; index < size; index += 1) {
        const z = normalizedExposures[index]!;
        const probability = probabilities[index]!;
        const residual = probability - targets[row * size + index]!;
        const jacobian1 = probability * (z - modelMean1);
        const jacobian2 = probability * (z * z - modelMean2);
        objective += residual * residual / divisor;
        gradient1 += 2 * residual * jacobian1 / divisor;
        gradient2 += 2 * residual * jacobian2 / divisor;
        hessian11 += 2 * jacobian1 * jacobian1 / divisor;
        hessian12 += 2 * jacobian1 * jacobian2 / divisor;
        hessian22 += 2 * jacobian2 * jacobian2 / divisor;
      }
    }
    return { objective, gradient1, gradient2, hessian11, hessian12, hessian22 };
  };

  const toNormalized = (linearCoefficient: number, quadraticCoefficient: number) => ({
    beta1: exposureScale * linearCoefficient
      + 2 * center * exposureScale * quadraticCoefficient,
    beta2: quadraticCoefficient * exposureScale * exposureScale,
  });
  const constrained = options.constrainQuadraticNonPositive ?? true;
  const klNormalized = toNormalized(
    klSeed.linearCoefficient,
    klSeed.quadraticCoefficient,
  );
  const initialNormalized = toNormalized(
    options.initialLinearCoefficient ?? 0,
    options.initialQuadraticCoefficient ?? 0,
  );
  const rawSeeds = [
    klNormalized,
    initialNormalized,
    { beta1: 0, beta2: 0 },
    { beta1: klNormalized.beta1, beta2: 0 },
  ];
  const seeds = rawSeeds.filter((seed, index) => rawSeeds.findIndex((candidate) =>
    Math.abs(candidate.beta1 - seed.beta1) <= 1e-12
      && Math.abs(candidate.beta2 - seed.beta2) <= 1e-12) === index);
  const maximumIterations = Math.max(1, Math.floor(options.maxIterations ?? 32));
  const tolerance = Math.max(Number.EPSILON, options.tolerance ?? 1e-10);
  let best: {
    beta1: number;
    beta2: number;
    state: ReturnType<typeof evaluate>;
    iterations: number;
    converged: boolean;
  } | undefined;
  for (const seed of seeds) {
    let beta1 = seed.beta1;
    let beta2 = constrained ? Math.min(0, seed.beta2) : seed.beta2;
    let state = evaluate(beta1, beta2);
    let converged = false;
    let iterations = 0;
    for (; iterations < maximumIterations; iterations += 1) {
      const quadraticAtBoundary = constrained
        && beta2 >= -tolerance && state.gradient2 < 0;
      const projectedGradient2 = quadraticAtBoundary ? 0 : state.gradient2;
      if (Math.hypot(state.gradient1, projectedGradient2) <= tolerance) {
        converged = true;
        break;
      }
      const ridge = Math.max(1e-12, (state.hessian11 + state.hessian22) * 1e-8);
      const h11 = state.hessian11 + ridge;
      const h22 = state.hessian22 + ridge;
      let step1: number;
      let step2: number;
      if (quadraticAtBoundary) {
        step1 = -state.gradient1 / h11;
        step2 = 0;
      } else {
        const determinant = h11 * h22 - state.hessian12 * state.hessian12;
        if (determinant > 1e-20) {
          step1 = -(h22 * state.gradient1 - state.hessian12 * state.gradient2)
            / determinant;
          step2 = -(-state.hessian12 * state.gradient1 + h11 * state.gradient2)
            / determinant;
        } else {
          step1 = -state.gradient1 / h11;
          step2 = -state.gradient2 / h22;
        }
      }
      if (!(state.gradient1 * step1 + projectedGradient2 * step2 < 0)) {
        step1 = -state.gradient1;
        step2 = -projectedGradient2;
      }
      let accepted = false;
      for (let lineScale = 1; lineScale >= 1 / 1_024; lineScale /= 2) {
        const nextBeta1 = beta1 + lineScale * step1;
        const nextBeta2 = constrained
          ? Math.min(0, beta2 + lineScale * step2)
          : beta2 + lineScale * step2;
        const next = evaluate(nextBeta1, nextBeta2);
        if (next.objective < state.objective - 1e-16) {
          beta1 = nextBeta1;
          beta2 = nextBeta2;
          state = next;
          accepted = true;
          break;
        }
      }
      if (!accepted) break;
    }
    if (!converged) {
      const quadraticAtBoundary = constrained
        && beta2 >= -tolerance && state.gradient2 < 0;
      converged = Math.hypot(
        state.gradient1,
        quadraticAtBoundary ? 0 : state.gradient2,
      ) <= Math.sqrt(tolerance);
    }
    if (!best || state.objective < best.state.objective) {
      best = { beta1, beta2, state, iterations, converged };
    }
  }
  if (!best) throw new Error("Conditional quadratic MSE fit did not produce a candidate.");
  const quadraticCoefficient = best.beta2 / (exposureScale * exposureScale);
  const linearCoefficient = best.beta1 / exposureScale
    - 2 * center * quadraticCoefficient;
  const diagnostics = conditionalQuadraticFitDiagnostics(
    grid,
    targetProbabilities,
    currentExposures,
    linearCoefficient,
    quadraticCoefficient,
    options,
  );
  return {
    objective: "probability-mse",
    linearCoefficient,
    quadraticCoefficient,
    crossEntropy: diagnostics.crossEntropy,
    klDivergence: diagnostics.klDivergence,
    meanSquaredError: diagnostics.meanSquaredError,
    iterations: best.iterations,
    converged: best.converged,
  };
}

function conditionalQuadraticFitDiagnostics(
  grid: ArrayLike<number>,
  targetProbabilities: ArrayLike<number>,
  currentExposures: ArrayLike<number>,
  linearCoefficient: number,
  quadraticCoefficient: number,
  options: Pick<ConditionalQuadraticPolicyFitOptions, "friction" | "transitionLogScale">,
): { crossEntropy: number; klDivergence: number; meanSquaredError: number } {
  const size = grid.length;
  const rows = currentExposures.length;
  const model = new Float64Array(size);
  let crossEntropy = 0;
  let targetEntropy = 0;
  let meanSquaredError = 0;
  for (let row = 0; row < rows; row += 1) {
    let targetTotal = 0;
    for (let index = 0; index < size; index += 1) {
      targetTotal += targetProbabilities[row * size + index]!;
    }
    conditionalQuadraticExposureProbabilities(
      grid,
      currentExposures[row]!,
      linearCoefficient,
      quadraticCoefficient,
      options.friction,
      options.transitionLogScale,
      model,
    );
    for (let index = 0; index < size; index += 1) {
      const target = targetProbabilities[row * size + index]! / targetTotal;
      const predicted = model[index]!;
      if (target > 0) {
        crossEntropy -= target * Math.log(Math.max(1e-300, predicted)) / rows;
        targetEntropy -= target * Math.log(target) / rows;
      }
      const residual = predicted - target;
      meanSquaredError += residual * residual / (rows * size);
    }
  }
  return {
    crossEntropy,
    klDivergence: Math.max(0, crossEntropy - targetEntropy),
    meanSquaredError,
  };
}

/**
 * Statistics of the strategy policy p(target | current exposure), averaged with
 * the uniform operational prior over every current-exposure grid state.
 */
export function strategyExposureTransitionStatistics(
  grid: Float64Array,
  currentGrid: Float64Array,
  rateBpsPerHour: number,
  holdingPeriodMs: number,
  temperature: number,
  quadraticCoefficient: number,
  friction: number,
  frictionFraction = 1,
  normalCenter = 0,
  normalMixture = 0,
  normalSigma = 25,
): ExposureTransitionPolicyStatistics {
  const slope = strategyExposureLogSlope(rateBpsPerHour, holdingPeriodMs, temperature);
  const scratch = createTransitionDistributionScratch(grid.length);
  fillStrategyMixtureBaseLogits(
    scratch.baseLogits,
    grid,
    slope,
    quadraticCoefficient,
    normalCenter,
    normalMixture,
    normalSigma,
  );
  return transitionDistributionStatistics(
    scratch.baseLogits,
    grid,
    currentGrid,
    friction,
    Math.max(0, frictionFraction) / temperature,
    scratch,
  );
}

export function strategyExposureTransitionCrossEntropy(
  oracle: ExposureValueOracle,
  candleIndex: number,
  rateBpsPerHour: number,
  holdingPeriodMs: number,
  temperature: number,
  quadraticCoefficient: number,
  frictionFraction = 1,
  normalCenter = 0,
  normalMixture = 0,
  normalSigma = 25,
): number {
  const slope = strategyExposureLogSlope(rateBpsPerHour, holdingPeriodMs, temperature);
  const scratch = createTransitionDistributionScratch(oracle.grid.length);
  const quadraticLogNormalizer = fillStrategyMixtureBaseLogits(
    scratch.baseLogits,
    oracle.grid,
    slope,
    quadraticCoefficient,
    normalCenter,
    normalMixture,
    normalSigma,
  );
  const transitionLogScale = frictionFraction / temperature;
  const statistics = transitionDistributionStatistics(
    scratch.baseLogits,
    oracle.grid,
    oracle.currentGrid,
    oracle.execution.friction,
    transitionLogScale,
    scratch,
  );
  let targetBaseLogit = slope * oracle.policyMeans[candleIndex]!
    + quadraticCoefficient * oracle.policySecondMoments[candleIndex]!
    - quadraticLogNormalizer;
  if (normalMixture > 0) {
    if (!oracle.probabilities) {
      throw new Error("Strategy normal-mixture loss requires retained oracle probabilities.");
    }
    targetBaseLogit = transitionCrossExpectation(
      oracle.probabilities.subarray(
        candleIndex * oracle.grid.length,
        (candleIndex + 1) * oracle.grid.length,
      ),
      scratch.baseLogits,
      oracle.grid,
      oracle.currentGrid,
      oracle.execution.friction,
      1 / oracle.temperature,
      scratch,
    );
  }
  return Math.max(
    0,
    statistics.logNormalizer
      - targetBaseLogit
      - transitionLogScale * oracle.policyMeanLogRebalances[candleIndex]!,
  );
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
    meanAverageRegret: accumulator.sampleCount > 0
      ? accumulator.averageRegretSum / accumulator.sampleCount
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

function upperBoundExposure(grid: Float64Array, exposure: number): number {
  let low = 0;
  let high = grid.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (grid[middle]! <= exposure) low = middle + 1;
    else high = middle;
  }
  return low;
}

function hasSeparableCurrentStateFactors(grid: Float64Array, friction: number): boolean {
  for (const current of grid) {
    if (!(1 - friction * current > 0)
      || !(1 - friction + friction * current > 0)) return false;
  }
  return true;
}

function createTransitionDistributionScratch(gridSize: number): TransitionDistributionScratch {
  const array = () => new Float64Array(gridSize);
  return {
    baseLogits: array(),
    sellWeights: array(),
    sellMeans: array(),
    sellSecondMoments: array(),
    sellBaseLogits: array(),
    sellLogDenominators: array(),
    buyWeights: array(),
    buyMeans: array(),
    buySecondMoments: array(),
    buyBaseLogits: array(),
    buyLogDenominators: array(),
    prefixRawSellLogs: array(),
    suffixRawBuyLogs: array(),
    prefixRawSellMaxima: array(),
    suffixRawBuyMaxima: array(),
  };
}

/**
 * Uniformly averages E[targetBaseLogit | current exposure] under an oracle
 * conditional policy. Buy and sell rebalance factors are separable, so the
 * complete conditional map is reduced with two stable linear scans.
 */
function transitionCrossExpectation(
  oracleBaseProbabilities: ArrayLike<number>,
  targetBaseLogits: Float64Array,
  grid: Float64Array,
  currentGrid: Float64Array,
  friction: number,
  transitionLogScale: number,
  scratch: TransitionDistributionScratch,
): number {
  const size = grid.length;
  if (oracleBaseProbabilities.length !== size || targetBaseLogits.length !== size) {
    throw new Error("Transition cross-expectation arrays do not match the exposure grid.");
  }
  for (const current of currentGrid) {
    if (!(1 - friction * current > 0)
      || !(1 - friction + friction * current > 0)) {
      return bruteTransitionCrossExpectation(
        oracleBaseProbabilities,
        targetBaseLogits,
        grid,
        currentGrid,
        friction,
        transitionLogScale,
      );
    }
  }
  let sellLogSum = Number.NEGATIVE_INFINITY;
  let sellMeanTarget = 0;
  for (let index = 0; index < size; index += 1) {
    const exposure = grid[index]!;
    const denominator = 1 - friction * exposure;
    const probability = oracleBaseProbabilities[index]!;
    if (!(denominator > 0) || !Number.isFinite(probability) || probability < 0) {
      return bruteTransitionCrossExpectation(
        oracleBaseProbabilities,
        targetBaseLogits,
        grid,
        currentGrid,
        friction,
        transitionLogScale,
      );
    }
    const adjusted = probability > 0
      ? Math.log(probability) - transitionLogScale * Math.log(denominator)
      : Number.NEGATIVE_INFINITY;
    const nextLogSum = logAddExp(sellLogSum, adjusted);
    const addedShare = Number.isFinite(adjusted) ? Math.exp(adjusted - nextLogSum) : 0;
    const retainedShare = Number.isFinite(sellLogSum) ? Math.exp(sellLogSum - nextLogSum) : 0;
    sellMeanTarget = retainedShare * sellMeanTarget
      + addedShare * targetBaseLogits[index]!;
    sellLogSum = nextLogSum;
    scratch.sellWeights[index] = sellLogSum;
    scratch.sellBaseLogits[index] = sellMeanTarget;
  }

  let buyLogSum = Number.NEGATIVE_INFINITY;
  let buyMeanTarget = 0;
  for (let index = size - 1; index >= 0; index -= 1) {
    const exposure = grid[index]!;
    const denominator = 1 - friction + friction * exposure;
    const probability = oracleBaseProbabilities[index]!;
    if (!(denominator > 0)) {
      return bruteTransitionCrossExpectation(
        oracleBaseProbabilities,
        targetBaseLogits,
        grid,
        currentGrid,
        friction,
        transitionLogScale,
      );
    }
    const adjusted = probability > 0
      ? Math.log(probability) - transitionLogScale * Math.log(denominator)
      : Number.NEGATIVE_INFINITY;
    const nextLogSum = logAddExp(buyLogSum, adjusted);
    const addedShare = Number.isFinite(adjusted) ? Math.exp(adjusted - nextLogSum) : 0;
    const retainedShare = Number.isFinite(buyLogSum) ? Math.exp(buyLogSum - nextLogSum) : 0;
    buyMeanTarget = retainedShare * buyMeanTarget
      + addedShare * targetBaseLogits[index]!;
    buyLogSum = nextLogSum;
    scratch.buyWeights[index] = buyLogSum;
    scratch.buyBaseLogits[index] = buyMeanTarget;
  }

  let expectation = 0;
  for (const stateExposure of currentGrid) {
    const buyIndex = upperBoundExposure(grid, stateExposure);
    const sellIndex = buyIndex - 1;
    const sellTotal = sellIndex >= 0
      ? transitionLogScale * Math.log(1 - friction * stateExposure)
        + scratch.sellWeights[sellIndex]!
      : Number.NEGATIVE_INFINITY;
    const buyTotal = buyIndex < size
      ? transitionLogScale * Math.log(1 - friction + friction * stateExposure)
        + scratch.buyWeights[buyIndex]!
      : Number.NEGATIVE_INFINITY;
    const total = logAddExp(sellTotal, buyTotal);
    const sellShare = Number.isFinite(sellTotal) ? Math.exp(sellTotal - total) : 0;
    const buyShare = Number.isFinite(buyTotal) ? Math.exp(buyTotal - total) : 0;
    expectation += (sellIndex >= 0 ? sellShare * scratch.sellBaseLogits[sellIndex]! : 0)
      + (buyIndex < size ? buyShare * scratch.buyBaseLogits[buyIndex]! : 0);
  }
  return expectation / currentGrid.length;
}

function bruteTransitionCrossExpectation(
  oracleBaseProbabilities: ArrayLike<number>,
  targetBaseLogits: Float64Array,
  grid: Float64Array,
  currentGrid: Float64Array,
  friction: number,
  transitionLogScale: number,
): number {
  let result = 0;
  for (const currentExposure of currentGrid) {
    let total = 0;
    let weighted = 0;
    for (let targetIndex = 0; targetIndex < grid.length; targetIndex += 1) {
      const factor = rebalanceEquityFactor(currentExposure, grid[targetIndex]!, friction);
      const probability = oracleBaseProbabilities[targetIndex]!;
      if (!(factor > 0) || !(probability > 0)) continue;
      const weight = probability * Math.exp(transitionLogScale * Math.log(factor));
      total += weight;
      weighted += weight * targetBaseLogits[targetIndex]!;
    }
    if (total > 0) result += weighted / total;
  }
  return result / currentGrid.length;
}

/**
 * Uniformly averages p(target | current exposure) over every input exposure on the grid.
 * Exact buy/sell fee factors are separable, so all rows require two linear scans rather
 * than materializing the grid-squared policy kernel.
 */
function transitionDistributionStatistics(
  baseLogits: Float64Array,
  grid: Float64Array,
  currentGrid: Float64Array,
  friction: number,
  transitionLogScale: number,
  scratch: TransitionDistributionScratch,
): TransitionDistributionStatistics {
  const size = grid.length;
  let maximum = Number.NEGATIVE_INFINITY;
  let separable = Number.isFinite(transitionLogScale) && transitionLogScale >= 0;
  for (const current of currentGrid) {
    if (!(1 - friction * current > 0)
      || !(1 - friction + friction * current > 0)) {
      separable = false;
      break;
    }
  }
  for (let index = 0; index < size; index += 1) {
    const exposure = grid[index]!;
    const sellDenominator = 1 - friction * exposure;
    const buyDenominator = 1 - friction + friction * exposure;
    const base = baseLogits[index]!;
    if (!(sellDenominator > 0) || !(buyDenominator > 0) || !Number.isFinite(base)) {
      separable = false;
      break;
    }
    maximum = Math.max(
      maximum,
      base - transitionLogScale * Math.log(sellDenominator),
      base - transitionLogScale * Math.log(buyDenominator),
    );
  }
  if (!separable) {
    return bruteTransitionDistributionStatistics(
      baseLogits,
      grid,
      currentGrid,
      friction,
      transitionLogScale,
    );
  }

  let sellWeight = 0;
  let sellMean = 0;
  let sellSecondMoment = 0;
  let sellBaseLogit = 0;
  let sellLogDenominator = 0;
  let rawSellLogs = 0;
  let rawSellMaximum = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < size; index += 1) {
    const exposure = grid[index]!;
    const logDenominator = Math.log(1 - friction * exposure);
    const base = baseLogits[index]!;
    const weight = Math.exp(base - transitionLogScale * logDenominator - maximum);
    sellWeight += weight;
    sellMean += weight * exposure;
    sellSecondMoment += weight * exposure * exposure;
    sellBaseLogit += weight * base;
    sellLogDenominator += weight * logDenominator;
    rawSellLogs += logDenominator;
    rawSellMaximum = Math.max(rawSellMaximum, base - logDenominator);
    scratch.sellWeights[index] = sellWeight;
    scratch.sellMeans[index] = sellMean;
    scratch.sellSecondMoments[index] = sellSecondMoment;
    scratch.sellBaseLogits[index] = sellBaseLogit;
    scratch.sellLogDenominators[index] = sellLogDenominator;
    scratch.prefixRawSellLogs[index] = rawSellLogs;
    scratch.prefixRawSellMaxima[index] = rawSellMaximum;
  }

  let buyWeight = 0;
  let buyMean = 0;
  let buySecondMoment = 0;
  let buyBaseLogit = 0;
  let buyLogDenominator = 0;
  let rawBuyLogs = 0;
  let rawBuyMaximum = Number.NEGATIVE_INFINITY;
  for (let index = size - 1; index >= 0; index -= 1) {
    const exposure = grid[index]!;
    const logDenominator = Math.log(1 - friction + friction * exposure);
    const base = baseLogits[index]!;
    const weight = Math.exp(base - transitionLogScale * logDenominator - maximum);
    buyWeight += weight;
    buyMean += weight * exposure;
    buySecondMoment += weight * exposure * exposure;
    buyBaseLogit += weight * base;
    buyLogDenominator += weight * logDenominator;
    rawBuyLogs += logDenominator;
    rawBuyMaximum = Math.max(rawBuyMaximum, base - logDenominator);
    scratch.buyWeights[index] = buyWeight;
    scratch.buyMeans[index] = buyMean;
    scratch.buySecondMoments[index] = buySecondMoment;
    scratch.buyBaseLogits[index] = buyBaseLogit;
    scratch.buyLogDenominators[index] = buyLogDenominator;
    scratch.suffixRawBuyLogs[index] = rawBuyLogs;
    scratch.suffixRawBuyMaxima[index] = rawBuyMaximum;
  }

  let logNormalizer = 0;
  let mean = 0;
  let secondMoment = 0;
  let meanLogRebalance = 0;
  let entropy = 0;
  for (const stateExposure of currentGrid) {
    const buyIndex = upperBoundExposure(grid, stateExposure);
    const sellIndex = buyIndex - 1;
    const sellCurrentLog = Math.log(1 - friction * stateExposure);
    const buyCurrentLog = Math.log(1 - friction + friction * stateExposure);
    const sellScale = Math.exp(transitionLogScale * sellCurrentLog);
    const buyScale = Math.exp(transitionLogScale * buyCurrentLog);
    const sellTotal = sellIndex >= 0 ? sellScale * scratch.sellWeights[sellIndex]! : 0;
    const buyTotal = buyIndex < size ? buyScale * scratch.buyWeights[buyIndex]! : 0;
    const total = sellTotal + buyTotal;
    if (!(total > 0) || !Number.isFinite(total)) {
      return bruteTransitionDistributionStatistics(
        baseLogits,
        grid,
        currentGrid,
        friction,
        transitionLogScale,
      );
    }
    const weightedMean = (sellIndex >= 0 ? sellScale * scratch.sellMeans[sellIndex]! : 0)
      + (buyIndex < size ? buyScale * scratch.buyMeans[buyIndex]! : 0);
    const weightedSecondMoment = (sellIndex >= 0
      ? sellScale * scratch.sellSecondMoments[sellIndex]!
      : 0)
      + (buyIndex < size ? buyScale * scratch.buySecondMoments[buyIndex]! : 0);
    const weightedBaseLogit = (sellIndex >= 0
      ? sellScale * scratch.sellBaseLogits[sellIndex]!
      : 0)
      + (buyIndex < size ? buyScale * scratch.buyBaseLogits[buyIndex]! : 0);
    const weightedLogRebalance = (sellIndex >= 0 ? sellScale * (
      sellCurrentLog * scratch.sellWeights[sellIndex]!
      - scratch.sellLogDenominators[sellIndex]!
    ) : 0) + (buyIndex < size ? buyScale * (
      buyCurrentLog * scratch.buyWeights[buyIndex]!
      - scratch.buyLogDenominators[buyIndex]!
    ) : 0);
    const rowLogNormalizer = maximum + Math.log(total);
    const rowMeanLogRebalance = weightedLogRebalance / total;
    logNormalizer += rowLogNormalizer;
    mean += weightedMean / total;
    secondMoment += weightedSecondMoment / total;
    meanLogRebalance += rowMeanLogRebalance;
    entropy += Math.max(
      0,
      rowLogNormalizer
        - weightedBaseLogit / total
        - transitionLogScale * rowMeanLogRebalance,
    );
  }
  return {
    logNormalizer: logNormalizer / currentGrid.length,
    mean: mean / currentGrid.length,
    secondMoment: secondMoment / currentGrid.length,
    meanLogRebalance: meanLogRebalance / currentGrid.length,
    entropy: entropy / currentGrid.length,
  };
}

function bruteTransitionDistributionStatistics(
  baseLogits: Float64Array,
  grid: Float64Array,
  currentGrid: Float64Array,
  friction: number,
  transitionLogScale: number,
): TransitionDistributionStatistics {
  let averageLogNormalizer = 0;
  let averageMean = 0;
  let averageSecondMoment = 0;
  let averageMeanLogRebalance = 0;
  let averageEntropy = 0;
  for (const stateExposure of currentGrid) {
    let maximum = Number.NEGATIVE_INFINITY;
    const logits = new Float64Array(grid.length);
    const logRebalances = new Float64Array(grid.length);
    for (let targetIndex = 0; targetIndex < grid.length; targetIndex += 1) {
      const factor = rebalanceEquityFactor(stateExposure, grid[targetIndex]!, friction);
      const logRebalance = factor > 0 ? Math.log(factor) : Number.NEGATIVE_INFINITY;
      const logit = baseLogits[targetIndex]! + transitionLogScale * logRebalance;
      logits[targetIndex] = logit;
      logRebalances[targetIndex] = logRebalance;
      maximum = Math.max(maximum, logit);
    }
    let total = 0;
    let mean = 0;
    let secondMoment = 0;
    let meanLogRebalance = 0;
    let meanLogit = 0;
    for (let targetIndex = 0; targetIndex < grid.length; targetIndex += 1) {
      if (!Number.isFinite(logits[targetIndex])) continue;
      const weight = Math.exp(logits[targetIndex]! - maximum);
      const exposure = grid[targetIndex]!;
      total += weight;
      mean += weight * exposure;
      secondMoment += weight * exposure * exposure;
      meanLogRebalance += weight * logRebalances[targetIndex]!;
      meanLogit += weight * logits[targetIndex]!;
    }
    const rowLogNormalizer = maximum + Math.log(total);
    averageLogNormalizer += rowLogNormalizer;
    averageMean += mean / total;
    averageSecondMoment += secondMoment / total;
    averageMeanLogRebalance += meanLogRebalance / total;
    averageEntropy += Math.max(0, rowLogNormalizer - meanLogit / total);
  }
  const scale = 1 / currentGrid.length;
  return {
    logNormalizer: averageLogNormalizer * scale,
    mean: averageMean * scale,
    secondMoment: averageSecondMoment * scale,
    meanLogRebalance: averageMeanLogRebalance * scale,
    entropy: averageEntropy * scale,
  };
}

function transitionOracleStatistics(
  values: Float64Array,
  grid: Float64Array,
  currentGrid: Float64Array,
  friction: number,
  temperature: number,
  scratch: TransitionDistributionScratch,
): TransitionOracleStatistics {
  for (let index = 0; index < values.length; index += 1) {
    scratch.baseLogits[index] = values[index]! / temperature;
  }
  const statistics = transitionDistributionStatistics(
    scratch.baseLogits,
    grid,
    currentGrid,
    friction,
    1 / temperature,
    scratch,
  );
  let averageRegret = 0;
  const meanValue = values.reduce((sum, value) => sum + value, 0) / values.length;
  if (values.every(Number.isFinite)
    && 1 - friction * grid[grid.length - 1]! > 0
    && 1 - friction + friction * grid[0]! > 0
    && hasSeparableCurrentStateFactors(currentGrid, friction)) {
    let sellMaximum = Number.NEGATIVE_INFINITY;
    for (let index = 0; index < grid.length; index += 1) {
      sellMaximum = Math.max(
        sellMaximum,
        values[index]! - Math.log(1 - friction * grid[index]!),
      );
      scratch.prefixRawSellMaxima[index] = sellMaximum;
    }
    let buyMaximum = Number.NEGATIVE_INFINITY;
    for (let index = grid.length - 1; index >= 0; index -= 1) {
      buyMaximum = Math.max(
        buyMaximum,
        values[index]! - Math.log(1 - friction + friction * grid[index]!),
      );
      scratch.suffixRawBuyMaxima[index] = buyMaximum;
    }
    for (const stateExposure of currentGrid) {
      const buyIndex = upperBoundExposure(grid, stateExposure);
      const sellIndex = buyIndex - 1;
      const sellCurrentLog = Math.log(1 - friction * stateExposure);
      const buyCurrentLog = Math.log(1 - friction + friction * stateExposure);
      const best = Math.max(
        sellIndex >= 0
          ? sellCurrentLog + scratch.prefixRawSellMaxima[sellIndex]!
          : Number.NEGATIVE_INFINITY,
        buyIndex < grid.length
          ? buyCurrentLog + scratch.suffixRawBuyMaxima[buyIndex]!
          : Number.NEGATIVE_INFINITY,
      );
      const sellLogSum = sellIndex >= 0
        ? (sellIndex + 1) * sellCurrentLog - scratch.prefixRawSellLogs[sellIndex]!
        : 0;
      const buyLogSum = buyIndex < grid.length
        ? (grid.length - buyIndex) * buyCurrentLog - scratch.suffixRawBuyLogs[buyIndex]!
        : 0;
      averageRegret += Math.max(
        0,
        best - meanValue - (sellLogSum + buyLogSum) / grid.length,
      );
    }
    averageRegret /= currentGrid.length;
  } else {
    for (const stateExposure of currentGrid) {
      let best = Number.NEGATIVE_INFINITY;
      let total = 0;
      let count = 0;
      for (let targetIndex = 0; targetIndex < grid.length; targetIndex += 1) {
        const factor = rebalanceEquityFactor(stateExposure, grid[targetIndex]!, friction);
        if (!(factor > 0) || !Number.isFinite(values[targetIndex])) continue;
        const value = values[targetIndex]! + Math.log(factor);
        best = Math.max(best, value);
        total += value;
        count += 1;
      }
      if (count > 0) averageRegret += Math.max(0, best - total / count);
    }
    averageRegret /= currentGrid.length;
  }
  return { ...statistics, averageRegret };
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

function holdingBlockOutcome(
  prices: ArrayLike<number>,
  start: number,
  end: number,
  initialExposure: number,
  options: ExposureExecutionOptions,
): { logReturn: number; exposure: number } {
  let logReturn = 0;
  let exposure = initialExposure;
  for (let time = start; time < end; time += 1) {
    const outcome = holdingOutcome(exposure, prices[time]!, prices[time + 1]!, options);
    if (!(outcome.equityFactor > 0)) {
      return { logReturn: Number.NEGATIVE_INFINITY, exposure: 0 };
    }
    logReturn += Math.log(outcome.equityFactor);
    exposure = outcome.exposure;
  }
  return { logReturn, exposure };
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
  const zeroGridPosition = -minimum / (maximum - minimum) * (options.gridSize - 1);
  if (Math.abs(zeroGridPosition - Math.round(zeroGridPosition)) > 1e-9) {
    throw new Error("Exposure value oracle grid must contain an exact zero-exposure point.");
  }
  if (!Number.isFinite(maximumEffective)
    || maximumEffective < Math.max(Math.abs(minimum), Math.abs(maximum))) {
    throw new Error("Exposure value oracle effective exposure must cover the tradable grid.");
  }
  const initialExposure = options.initialExposure ?? 0;
  if (!Number.isFinite(initialExposure) || Math.abs(initialExposure) > maximumEffective) {
    throw new Error("Exposure value oracle initial exposure exceeds the effective-exposure limit.");
  }
  const terminalIndex = options.terminalIndex ?? prices.length - 1;
  if (!Number.isInteger(terminalIndex)
    || terminalIndex <= options.scoreStartIndex
    || terminalIndex >= prices.length) {
    throw new Error("Exposure value oracle terminal candle is outside the scored price series.");
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
