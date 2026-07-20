import {
  rebalanceEquityFactor,
  type ExposureExecutionOptions,
} from "./exposure-value-distillation.js";

export interface HandcraftedIndicatorPredictorParameters {
  driftEstimateHalfLifeMs: number;
  driftForecastHalfLifeMs: number;
  driftScale: number;
  varianceEstimateHalfLifeMs: number;
  longRunVarianceHalfLifeMs: number;
  varianceForecastHalfLifeMs: number;
}

export interface HandcraftedIndicatorState {
  drift: number;
  variance: number;
  longRunVariance: number;
}

export interface HandcraftedIndicatorPredictionOptions {
  intervalMs: number;
  holdingPeriodSteps: number;
  valueHorizonSteps: number;
  temperature: number;
  execution: ExposureExecutionOptions;
  /** Use the oracle's exact per-step maintenance equity factor in log space. */
  exactMaintenanceUtility?: boolean;
}

export interface HandcraftedIndicatorPrediction {
  values: Float64Array;
  regrets: Float64Array;
  probabilities: Float64Array;
  optimalExposure: number;
  meanExposure: number;
}

export interface HandcraftedIndicatorForecastBackground {
  /** One-dimensional F(a): initial forced holding utility plus optimal continuation value. */
  values: Float64Array;
  holdingPeriodSteps: number;
  valueHorizonSteps: number;
}

export const HANDCRAFTED_INDICATOR_PARAMETER_BOUNDS: Readonly<{
  [K in keyof HandcraftedIndicatorPredictorParameters]: readonly [number, number];
}> = {
  driftEstimateHalfLifeMs: [60_000, 6 * 3_600_000],
  driftForecastHalfLifeMs: [60_000, 6 * 3_600_000],
  driftScale: [0.002, 1.5],
  varianceEstimateHalfLifeMs: [5 * 60_000, 24 * 3_600_000],
  longRunVarianceHalfLifeMs: [3 * 3_600_000, 72 * 3_600_000],
  varianceForecastHalfLifeMs: [10 * 60_000, 48 * 3_600_000],
};

export const DEFAULT_HANDCRAFTED_INDICATOR_PARAMETERS:
Readonly<HandcraftedIndicatorPredictorParameters> = {
  driftEstimateHalfLifeMs: 2_657_296.394970352,
  driftForecastHalfLifeMs: 1_316_056.907073569,
  driftScale: 0.631158796528097,
  varianceEstimateHalfLifeMs: 29_220_215.856925953,
  longRunVarianceHalfLifeMs: 28_897_920.874858618,
  varianceForecastHalfLifeMs: 7_716_496.503579344,
};

/**
 * Builds causal close-to-close forecast state. State i uses prices through i and
 * can therefore be consumed by a decision made at that close.
 */
export function prepareHandcraftedIndicatorStates(
  prices: ArrayLike<number>,
  intervalMs: number,
  parameters: HandcraftedIndicatorPredictorParameters,
): HandcraftedIndicatorState[] {
  validateParameters(intervalMs, parameters);
  if (prices.length === 0) return [];
  const driftAlpha = halfLifeAlpha(parameters.driftEstimateHalfLifeMs, intervalMs);
  const varianceAlpha = halfLifeAlpha(parameters.varianceEstimateHalfLifeMs, intervalMs);
  const longRunAlpha = halfLifeAlpha(parameters.longRunVarianceHalfLifeMs, intervalMs);
  const result = new Array<HandcraftedIndicatorState>(prices.length);
  let drift = 0;
  let variance = 0;
  let longRunVariance = 0;
  result[0] = { drift, variance, longRunVariance };
  for (let index = 1; index < prices.length; index += 1) {
    const previousPrice = prices[index - 1]!;
    const price = prices[index]!;
    if (!(Number.isFinite(previousPrice) && previousPrice > 0
      && Number.isFinite(price) && price > 0)) {
      throw new Error("Handcrafted indicator predictor requires finite positive prices.");
    }
    const returnValue = price / previousPrice - 1;
    if (index === 1) {
      drift = returnValue;
      variance = returnValue * returnValue;
      longRunVariance = variance;
      result[index] = { drift, variance, longRunVariance };
      continue;
    }
    const innovation = returnValue - drift;
    drift += driftAlpha * innovation;
    variance += varianceAlpha * (innovation * innovation - variance);
    longRunVariance += longRunAlpha * (returnValue * returnValue - longRunVariance);
    result[index] = {
      drift,
      variance: Math.max(0, variance),
      longRunVariance: Math.max(0, longRunVariance),
    };
  }
  return result;
}

/** Computes one causal state without allocating the complete state history. */
export function prepareHandcraftedIndicatorStateAt(
  prices: ArrayLike<number>,
  index: number,
  intervalMs: number,
  parameters: HandcraftedIndicatorPredictorParameters,
): HandcraftedIndicatorState {
  validateParameters(intervalMs, parameters);
  if (!Number.isInteger(index) || index < 0 || index >= prices.length) {
    throw new Error("Handcrafted indicator state index is outside the price series.");
  }
  if (index === 0) return { drift: 0, variance: 0, longRunVariance: 0 };
  const driftAlpha = halfLifeAlpha(parameters.driftEstimateHalfLifeMs, intervalMs);
  const varianceAlpha = halfLifeAlpha(parameters.varianceEstimateHalfLifeMs, intervalMs);
  const longRunAlpha = halfLifeAlpha(parameters.longRunVarianceHalfLifeMs, intervalMs);
  let drift = 0;
  let variance = 0;
  let longRunVariance = 0;
  for (let cursor = 1; cursor <= index; cursor += 1) {
    const previousPrice = prices[cursor - 1]!;
    const price = prices[cursor]!;
    if (!(Number.isFinite(previousPrice) && previousPrice > 0
      && Number.isFinite(price) && price > 0)) {
      throw new Error("Handcrafted indicator predictor requires finite positive prices.");
    }
    const returnValue = price / previousPrice - 1;
    if (cursor === 1) {
      drift = returnValue;
      variance = returnValue * returnValue;
      longRunVariance = variance;
      continue;
    }
    const innovation = returnValue - drift;
    drift += driftAlpha * innovation;
    variance += varianceAlpha * (innovation * innovation - variance);
    longRunVariance += longRunAlpha * (returnValue * returnValue - longRunVariance);
  }
  return {
    drift,
    variance: Math.max(0, variance),
    longRunVariance: Math.max(0, longRunVariance),
  };
}

/**
 * Solves the same fixed-H, finite-T exposure problem as the value oracle, but
 * under causal EWMA return/variance forecasts rather than realized prices.
 */
export function predictHandcraftedIndicatorRegret(
  exposureGrid: ArrayLike<number>,
  state: HandcraftedIndicatorState,
  parameters: HandcraftedIndicatorPredictorParameters,
  options: HandcraftedIndicatorPredictionOptions,
): HandcraftedIndicatorPrediction {
  validatePredictionInputs(exposureGrid, state, parameters, options);
  const grid = Float64Array.from(exposureGrid);
  const background = forecastHandcraftedIndicatorBackground(
    grid,
    grid,
    state,
    parameters,
    options,
  );
  const baseValues = background.values;
  let maximum = Number.NEGATIVE_INFINITY;
  let optimalIndex = 0;
  for (let index = 0; index < baseValues.length; index += 1) {
    if (baseValues[index]! > maximum) {
      maximum = baseValues[index]!;
      optimalIndex = index;
    }
  }
  const regrets = Float64Array.from(baseValues, (value) => maximum - value);
  const probabilities = softmaxValues(baseValues, options.temperature);
  let meanExposure = 0;
  for (let index = 0; index < grid.length; index += 1) {
    meanExposure += grid[index]! * probabilities[index]!;
  }
  return {
    values: baseValues,
    regrets,
    probabilities,
    optimalExposure: grid[optimalIndex]!,
    meanExposure,
  };
}

/**
 * Computes only the one-dimensional forecast background F(a). Continuation
 * optimization scans target actions but never materializes an x-by-a value or
 * regret surface.
 */
export function forecastHandcraftedIndicatorBackground(
  backgroundGrid: ArrayLike<number>,
  actionGrid: ArrayLike<number>,
  state: HandcraftedIndicatorState,
  parameters: HandcraftedIndicatorPredictorParameters,
  options: HandcraftedIndicatorPredictionOptions,
): HandcraftedIndicatorForecastBackground {
  validatePredictionInputs(backgroundGrid, state, parameters, options);
  validateOrderedGrid(actionGrid, "action");
  const backgrounds = Float64Array.from(backgroundGrid);
  const actions = Float64Array.from(actionGrid);
  const holdingSteps = Math.max(1, Math.floor(options.holdingPeriodSteps));
  const horizonSteps = Math.max(holdingSteps, Math.floor(options.valueHorizonSteps));
  const driftPersistence = halfLifePersistence(parameters.driftForecastHalfLifeMs, options.intervalMs);
  const variancePersistence = halfLifePersistence(
    parameters.varianceForecastHalfLifeMs,
    options.intervalMs,
  );
  const forecastDrifts = new Float64Array(horizonSteps);
  const forecastSecondMoments = new Float64Array(horizonSteps);
  let driftDecay = 1;
  let varianceDecay = 1;
  for (let step = 0; step < horizonSteps; step += 1) {
    const drift = parameters.driftScale * driftDecay * state.drift;
    const variance = Math.max(0, state.longRunVariance
      + varianceDecay * (state.variance - state.longRunVariance));
    forecastDrifts[step] = drift;
    forecastSecondMoments[step] = variance + drift * drift;
    driftDecay *= driftPersistence;
    varianceDecay *= variancePersistence;
  }

  const utilities = new Float64Array(horizonSteps * actions.length);
  for (let step = 0; step < horizonSteps; step += 1) {
    const row = step * actions.length;
    for (let index = 0; index < actions.length; index += 1) {
      utilities[row + index] = forecastUtility(
        actions[index]!,
        forecastDrifts[step]!,
        forecastSecondMoments[step]!,
        options.execution,
        options.exactMaintenanceUtility ?? false,
      );
    }
  }

  const blockDurations: number[] = [];
  for (let remaining = horizonSteps; remaining > 0;) {
    const duration = Math.min(holdingSteps, remaining);
    blockDurations.push(duration);
    remaining -= duration;
  }
  let nextValues: Float64Array<ArrayBufferLike> = new Float64Array(actions.length);
  const forcedValues = new Float64Array(actions.length);
  const prefixValues = new Float64Array(actions.length);
  const suffixValues = new Float64Array(actions.length);
  let firstContinuationForcedValues: Float64Array | null = null;
  let blockEnd = horizonSteps;
  for (let block = blockDurations.length - 1; block >= 1; block -= 1) {
    const duration = blockDurations[block]!;
    const blockStart = blockEnd - duration;
    for (let action = 0; action < actions.length; action += 1) {
      let value = block === blockDurations.length - 1 ? 0 : nextValues[action]!;
      for (let step = blockStart; step < blockEnd; step += 1) {
        value += utilities[step * actions.length + action]!;
      }
      forcedValues[action] = value;
    }
    if (block === 1) firstContinuationForcedValues = Float64Array.from(forcedValues);
    nextValues = maximizeRebalanceValues(
      actions,
      forcedValues,
      options.execution.friction,
      prefixValues,
      suffixValues,
    );
    blockEnd = blockStart;
  }

  const initialHoldingSteps = blockDurations[0]!;
  const values = new Float64Array(backgrounds.length);
  for (let index = 0; index < backgrounds.length; index += 1) {
    const exposure = backgrounds[index]!;
    let value = 0;
    for (let step = 0; step < initialHoldingSteps; step += 1) {
      value += forecastUtility(
        exposure,
        forecastDrifts[step]!,
        forecastSecondMoments[step]!,
        options.execution,
        options.exactMaintenanceUtility ?? false,
      );
    }
    if (firstContinuationForcedValues) {
      value += maximizeRebalanceValueAt(
        exposure,
        actions,
        firstContinuationForcedValues,
        options.execution.friction,
      );
    }
    values[index] = value;
  }
  return {
    values,
    holdingPeriodSteps: holdingSteps,
    valueHorizonSteps: horizonSteps,
  };
}

function maximizeRebalanceValueAt(
  current: number,
  targets: Float64Array,
  forcedValues: Float64Array,
  friction: number,
): number {
  let maximum = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < targets.length; index += 1) {
    const factor = rebalanceEquityFactor(current, targets[index]!, friction);
    if (factor > 0) maximum = Math.max(maximum, Math.log(factor) + forcedValues[index]!);
  }
  if (!Number.isFinite(maximum)) {
    throw new Error("Handcrafted indicator continuation has no feasible target action.");
  }
  return maximum;
}

function maximizeRebalanceValues(
  grid: Float64Array,
  forcedValues: Float64Array,
  friction: number,
  prefixValues: Float64Array,
  suffixValues: Float64Array,
): Float64Array {
  let best = Number.NEGATIVE_INFINITY;
  for (let target = 0; target < grid.length; target += 1) {
    const denominator = 1 - friction * grid[target]!;
    const adjusted = denominator > 0
      ? forcedValues[target]! - Math.log(denominator)
      : Number.NEGATIVE_INFINITY;
    if (adjusted > best) best = adjusted;
    prefixValues[target] = best;
  }
  best = Number.NEGATIVE_INFINITY;
  for (let target = grid.length - 1; target >= 0; target -= 1) {
    const denominator = 1 - friction + friction * grid[target]!;
    const adjusted = denominator > 0
      ? forcedValues[target]! - Math.log(denominator)
      : Number.NEGATIVE_INFINITY;
    if (adjusted >= best) best = adjusted;
    suffixValues[target] = best;
  }
  const result = new Float64Array(grid.length);
  for (let current = 0; current < grid.length; current += 1) {
    const sellFactor = 1 - friction * grid[current]!;
    const buyFactor = 1 - friction + friction * grid[current]!;
    const sell = sellFactor > 0
      ? Math.log(sellFactor) + prefixValues[current]!
      : Number.NEGATIVE_INFINITY;
    const buy = buyFactor > 0
      ? Math.log(buyFactor) + suffixValues[current]!
      : Number.NEGATIVE_INFINITY;
    result[current] = Math.max(sell, buy);
    if (!Number.isFinite(result[current]!)) {
      let fallback = Number.NEGATIVE_INFINITY;
      for (let target = 0; target < grid.length; target += 1) {
        const factor = rebalanceEquityFactor(grid[current]!, grid[target]!, friction);
        if (factor > 0) fallback = Math.max(fallback, Math.log(factor) + forcedValues[target]!);
      }
      result[current] = fallback;
    }
  }
  return result;
}

function maintenanceUtility(exposure: number, execution: ExposureExecutionOptions): number {
  const quote = 1 - exposure;
  const quoteUtility = quote >= 0
    ? quote * execution.quoteLendRate
    : quote * execution.quoteBorrowRate;
  const assetUtility = exposure < 0 ? exposure * execution.assetBorrowRate : 0;
  return quoteUtility + assetUtility;
}

function forecastUtility(
  exposure: number,
  drift: number,
  secondMoment: number,
  execution: ExposureExecutionOptions,
  exactMaintenance: boolean,
): number {
  const maintenance = maintenanceUtility(exposure, execution);
  const maintenanceLogUtility = maintenance > -1
    ? Math.log1p(maintenance)
    : Number.NEGATIVE_INFINITY;
  return exposure * drift
    - 0.5 * exposure * exposure * secondMoment
    + (exactMaintenance ? maintenanceLogUtility : maintenance);
}

function softmaxValues(values: Float64Array, temperature: number): Float64Array {
  const maximum = values.reduce((best, value) => Math.max(best, value), Number.NEGATIVE_INFINITY);
  const result = new Float64Array(values.length);
  let total = 0;
  for (let index = 0; index < values.length; index += 1) {
    const probability = Math.max(
      Number.MIN_VALUE,
      Math.exp((values[index]! - maximum) / temperature),
    );
    result[index] = probability;
    total += probability;
  }
  if (!(total > 0 && Number.isFinite(total))) {
    throw new Error("Handcrafted indicator prediction has no finite action probability.");
  }
  for (let index = 0; index < result.length; index += 1) result[index] /= total;
  return result;
}

function halfLifeAlpha(halfLifeMs: number, intervalMs: number): number {
  return 1 - halfLifePersistence(halfLifeMs, intervalMs);
}

function halfLifePersistence(halfLifeMs: number, intervalMs: number): number {
  return Math.exp(-Math.LN2 * intervalMs / halfLifeMs);
}

function validateParameters(
  intervalMs: number,
  parameters: HandcraftedIndicatorPredictorParameters,
): void {
  if (!(Number.isFinite(intervalMs) && intervalMs > 0)) {
    throw new Error("Handcrafted indicator interval must be positive.");
  }
  if (![parameters.driftEstimateHalfLifeMs, parameters.driftForecastHalfLifeMs,
    parameters.varianceEstimateHalfLifeMs, parameters.longRunVarianceHalfLifeMs,
    parameters.varianceForecastHalfLifeMs].every((value) => Number.isFinite(value) && value > 0)
    || !(Number.isFinite(parameters.driftScale) && parameters.driftScale >= 0)) {
    throw new Error("Handcrafted indicator forecast parameters must be finite and non-negative.");
  }
}

function validatePredictionInputs(
  exposureGrid: ArrayLike<number>,
  state: HandcraftedIndicatorState,
  parameters: HandcraftedIndicatorPredictorParameters,
  options: HandcraftedIndicatorPredictionOptions,
): void {
  validateParameters(options.intervalMs, parameters);
  if (exposureGrid.length < 2
    || ![state.drift, state.variance, state.longRunVariance].every(Number.isFinite)
    || state.variance < 0 || state.longRunVariance < 0
    || !Number.isInteger(options.holdingPeriodSteps) || options.holdingPeriodSteps < 1
    || !Number.isInteger(options.valueHorizonSteps)
    || options.valueHorizonSteps < options.holdingPeriodSteps
    || !(Number.isFinite(options.temperature) && options.temperature > 0)) {
    throw new Error("Invalid handcrafted indicator prediction input.");
  }
  validateOrderedGrid(exposureGrid, "exposure");
}

function validateOrderedGrid(grid: ArrayLike<number>, label: string): void {
  if (grid.length < 2) throw new Error(`Handcrafted indicator ${label} grid is too short.`);
  for (let index = 0; index < grid.length; index += 1) {
    if (!Number.isFinite(grid[index]!)
      || (index > 0 && !(grid[index]! > grid[index - 1]!))) {
      throw new Error(`Handcrafted indicator ${label} grid must be finite and strictly ordered.`);
    }
  }
}
