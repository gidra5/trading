import {
  conditionalFourSegmentExposureProbabilities,
  type ConditionalFourSegmentParameters,
} from "./conditional-exposure-distribution.js";
import type { ExposureExecutionOptions } from "./exposure-value-distillation.js";
import {
  DEFAULT_HANDCRAFTED_INDICATOR_PARAMETERS,
  HANDCRAFTED_INDICATOR_PARAMETER_BOUNDS,
  forecastHandcraftedIndicatorBackground,
  prepareHandcraftedIndicatorForecast,
  type HandcraftedIndicatorPredictionOptions,
  type HandcraftedIndicatorPredictorParameters,
  type HandcraftedIndicatorState,
} from "./handcrafted-indicator-predictor.js";

const LOGISTIC_TEN_TO_NINETY = 4.394;
const SUPPORT_WIDTH_FRACTION = 0.2;

export interface DirectIndicatorPredictorParameters
  extends HandcraftedIndicatorPredictorParameters {
  /** Fallback smoothing width for unresolved one-dimensional kinks, in action-grid cells. */
  transitionWidthGridCells: number;
}

export interface DirectIndicatorParameterVector6 {
  c1: number;
  c2: number;
  b: number;
  lambda: number;
  betaC1: number;
  betaC2: number;
}

export interface DirectIndicatorConditionalMetadata {
  parameterVector6: DirectIndicatorParameterVector6;
  backgroundSecantSlopes: readonly [number, number, number];
  effectiveSellCostSlope: number;
  effectiveBuyCostSlope: number;
  c1MarginalBuyCost: number;
  c2MarginalSellCost: number;
  transitionWidths10To90: readonly [number, number, number];
  transitionWidthSources: readonly ["background-10-90" | "grid-convention", "grid-convention", "background-10-90" | "grid-convention"];
  supportInterior: readonly [number, number];
}

export interface DirectIndicatorConditionalPrediction {
  probabilities: Float64Array;
  optimalExposure: number;
  meanExposure: number;
  conditionalParameters: ConditionalFourSegmentParameters;
  metadata: DirectIndicatorConditionalMetadata;
  /** Retained only as the permitted one-dimensional F(a) diagnostic. */
  backgroundValues: Float64Array;
}

export interface PreparedDirectIndicatorConditionalPredictor {
  predict(
    currentExposure: number,
    state: HandcraftedIndicatorState,
  ): DirectIndicatorConditionalPrediction;
}

export const DEFAULT_DIRECT_INDICATOR_PARAMETERS: Readonly<DirectIndicatorPredictorParameters> = {
  ...DEFAULT_HANDCRAFTED_INDICATOR_PARAMETERS,
  transitionWidthGridCells: 2,
};

export const DIRECT_INDICATOR_PARAMETER_BOUNDS: Readonly<{
  [K in keyof DirectIndicatorPredictorParameters]: readonly [number, number];
}> = {
  ...HANDCRAFTED_INDICATOR_PARAMETER_BOUNDS,
  transitionWidthGridCells: [0.5, 8],
};

/**
 * Decodes causal indicator forecasts directly into the analytic conditional
 * distribution. This routine only asks the forecast engine for one-dimensional
 * F(a), and deliberately has no regret-projection dependency.
 */
export function decodeDirectIndicatorConditionalParameters(
  actionGridInput: ArrayLike<number>,
  backgroundGridInput: ArrayLike<number>,
  state: HandcraftedIndicatorState,
  parameters: DirectIndicatorPredictorParameters,
  options: HandcraftedIndicatorPredictionOptions,
): {
  parameters: ConditionalFourSegmentParameters;
  metadata: DirectIndicatorConditionalMetadata;
  backgroundValues: Float64Array;
} {
  validateDirectInputs(actionGridInput, backgroundGridInput, parameters, options);
  const actionGrid = Float64Array.from(actionGridInput);
  const backgroundGrid = Float64Array.from(backgroundGridInput);
  const backgroundValues = forecastHandcraftedIndicatorBackground(
    backgroundGrid,
    actionGrid,
    state,
    parameters,
    { ...options, exactMaintenanceUtility: true },
  ).values;
  return decodeDirectIndicatorConditionalParametersFromBackground(
    actionGrid,
    backgroundGrid,
    backgroundValues,
    parameters,
    options,
  );
}

function decodeDirectIndicatorConditionalParametersFromBackground(
  actionGrid: Float64Array,
  backgroundGrid: Float64Array,
  backgroundValues: Float64Array,
  parameters: DirectIndicatorPredictorParameters,
  options: HandcraftedIndicatorPredictionOptions,
): {
  parameters: ConditionalFourSegmentParameters;
  metadata: DirectIndicatorConditionalMetadata;
  backgroundValues: Float64Array;
} {
  const latentLower = backgroundGrid[0]!;
  const latentUpper = backgroundGrid.at(-1)!;
  const visibleLower = actionGrid[0]!;
  const visibleUpper = actionGrid.at(-1)!;
  const latentSpan = latentUpper - latentLower;
  const leftSupportWidth = SUPPORT_WIDTH_FRACTION * latentSpan;
  const rightSupportWidth = SUPPORT_WIDTH_FRACTION * latentSpan;
  const supportLower = latentLower + leftSupportWidth;
  const supportUpper = latentUpper - rightSupportWidth;
  if (!backgroundValues.every(Number.isFinite)) {
    throw new Error("Direct indicator forecast background is infeasible under maintenance constraints.");
  }
  const derivatives = finiteDifferenceDerivatives(backgroundGrid, backgroundValues);
  const spacing = medianSpacing(actionGrid);
  const minimumSeparation = Math.max(spacing, latentSpan * 1e-6);
  const c1Residual = Float64Array.from(backgroundGrid, (exposure, index) =>
    derivatives[index]! - marginalBuyCost(exposure, options.execution.friction));
  const c2Residual = Float64Array.from(backgroundGrid, (exposure, index) =>
    derivatives[index]! + marginalSellCost(exposure, options.execution.friction));
  let c1 = decreasingRoot(backgroundGrid, c1Residual);
  let c2 = decreasingRoot(backgroundGrid, c2Residual);
  c1 = clamp(c1, supportLower + minimumSeparation, supportUpper - 2 * minimumSeparation);
  c2 = clamp(c2, c1 + minimumSeparation, supportUpper - minimumSeparation);
  if (!(latentLower < c1 && c1 < c2 && c2 < latentUpper)) {
    const center = clamp((c1 + c2) / 2, supportLower + minimumSeparation, supportUpper - minimumSeparation);
    c1 = center - minimumSeparation / 2;
    c2 = center + minimumSeparation / 2;
  }

  const fLower = interpolate(backgroundGrid, backgroundValues, supportLower);
  const fC1 = interpolate(backgroundGrid, backgroundValues, c1);
  const fC2 = interpolate(backgroundGrid, backgroundValues, c2);
  const fUpper = interpolate(backgroundGrid, backgroundValues, supportUpper);
  const s0 = (fC1 - fLower) / (c1 - supportLower);
  const s1 = (fC2 - fC1) / (c2 - c1);
  const s2 = (fUpper - fC2) / (supportUpper - c2);
  const qSell = effectiveSellCostSlope(
    supportLower,
    c1,
    options.execution.friction,
  );
  const qBuy = effectiveBuyCostSlope(c2, supportUpper, options.execution.friction);
  const inverseTemperature = 1 / options.temperature;

  const fallbackWidth = Math.max(Number.EPSILON, parameters.transitionWidthGridCells * spacing);
  const c1MeasuredWidth = transitionWidth10To90(
    backgroundGrid,
    derivatives,
    c1,
    s0,
    s1,
  );
  const c2MeasuredWidth = transitionWidth10To90(
    backgroundGrid,
    derivatives,
    c2,
    s1,
    s2,
  );
  const c1Width = c1MeasuredWidth ?? fallbackWidth;
  const c2Width = c2MeasuredWidth ?? fallbackWidth;
  const kappaC1 = LOGISTIC_TEN_TO_NINETY / c1Width;
  const kappaX = LOGISTIC_TEN_TO_NINETY / fallbackWidth;
  const kappaC2 = LOGISTIC_TEN_TO_NINETY / c2Width;

  // Fit the screenshot's linear design after fixing c/kappa and subtracting
  // the moving fee transition. This estimates smooth curvature independently
  // from the localized beta transitions instead of making the betas absorb it.
  const coefficients = fitQuadraticBackgroundCoefficients(
    backgroundGrid,
    backgroundValues,
    supportLower,
    supportUpper,
    c1,
    c2,
    kappaC1,
    kappaC2,
    qSell,
    inverseTemperature,
  );
  const conditionalParameters: ConditionalFourSegmentParameters = {
    latentLower,
    latentUpper,
    visibleLower,
    visibleUpper,
    cutoffLower: latentLower,
    cutoffUpper: latentUpper,
    basisCenter: (supportLower + supportUpper) / 2,
    c1,
    c2,
    baseSlope: coefficients.baseSlope,
    quadraticPrecision: coefficients.quadraticPrecision,
    betaC1: coefficients.betaC1,
    betaX: -(qBuy + qSell) * inverseTemperature,
    betaC2: coefficients.betaC2,
    kappaC1,
    kappaX,
    kappaC2,
  };
  const parameterVector6: DirectIndicatorParameterVector6 = {
    c1,
    c2,
    b: conditionalParameters.baseSlope,
    lambda: conditionalParameters.quadraticPrecision,
    betaC1: conditionalParameters.betaC1,
    betaC2: conditionalParameters.betaC2,
  };
  return {
    parameters: conditionalParameters,
    metadata: {
      parameterVector6,
      backgroundSecantSlopes: [s0, s1, s2],
      effectiveSellCostSlope: qSell,
      effectiveBuyCostSlope: qBuy,
      c1MarginalBuyCost: marginalBuyCost(c1, options.execution.friction),
      c2MarginalSellCost: marginalSellCost(c2, options.execution.friction),
      transitionWidths10To90: [c1Width, fallbackWidth, c2Width],
      transitionWidthSources: [
        c1MeasuredWidth === null ? "grid-convention" : "background-10-90",
        "grid-convention",
        c2MeasuredWidth === null ? "grid-convention" : "background-10-90",
      ],
      supportInterior: [supportLower, supportUpper],
    },
    backgroundValues,
  };
}

export function predictDirectIndicatorConditionalDistribution(
  actionGridInput: ArrayLike<number>,
  backgroundGridInput: ArrayLike<number>,
  currentExposure: number,
  state: HandcraftedIndicatorState,
  parameters: DirectIndicatorPredictorParameters,
  options: HandcraftedIndicatorPredictionOptions,
): DirectIndicatorConditionalPrediction {
  const decoded = decodeDirectIndicatorConditionalParameters(
    actionGridInput,
    backgroundGridInput,
    state,
    parameters,
    options,
  );
  const actionGrid = Float64Array.from(actionGridInput);
  return directPredictionFromDecoded(actionGrid, currentExposure, decoded);
}

/** Compiles the expensive horizon forecast terms for sequential direct-model predictions. */
export function prepareDirectIndicatorConditionalPredictor(
  actionGridInput: ArrayLike<number>,
  backgroundGridInput: ArrayLike<number>,
  parameters: DirectIndicatorPredictorParameters,
  options: HandcraftedIndicatorPredictionOptions,
): PreparedDirectIndicatorConditionalPredictor {
  validateDirectInputs(actionGridInput, backgroundGridInput, parameters, options);
  const actionGrid = Float64Array.from(actionGridInput);
  const backgroundGrid = Float64Array.from(backgroundGridInput);
  const forecast = prepareHandcraftedIndicatorForecast(
    backgroundGrid,
    actionGrid,
    parameters,
    { ...options, exactMaintenanceUtility: true },
  );
  return { predict: (currentExposure, state) => {
    const decoded = decodeDirectIndicatorConditionalParametersFromBackground(
      actionGrid,
      backgroundGrid,
      forecast.forecast(state).values,
      parameters,
      options,
    );
    return directPredictionFromDecoded(actionGrid, currentExposure, decoded);
  } };
}

function directPredictionFromDecoded(
  actionGrid: Float64Array,
  currentExposure: number,
  decoded: ReturnType<typeof decodeDirectIndicatorConditionalParameters>,
): DirectIndicatorConditionalPrediction {
  const probabilities = conditionalFourSegmentExposureProbabilities(
    actionGrid,
    currentExposure,
    decoded.parameters,
  );
  let optimalIndex = 0;
  let meanExposure = 0;
  for (let index = 0; index < actionGrid.length; index += 1) {
    if (probabilities[index]! > probabilities[optimalIndex]!) optimalIndex = index;
    meanExposure += actionGrid[index]! * probabilities[index]!;
  }
  return {
    probabilities,
    optimalExposure: actionGrid[optimalIndex]!,
    meanExposure,
    conditionalParameters: decoded.parameters,
    metadata: decoded.metadata,
    backgroundValues: decoded.backgroundValues,
  };
}

function marginalBuyCost(exposure: number, friction: number): number {
  if (!(friction > 0)) return 0;
  const denominator = 1 - friction + friction * exposure;
  return denominator > 0 ? friction / denominator : Number.POSITIVE_INFINITY;
}

function marginalSellCost(exposure: number, friction: number): number {
  if (!(friction > 0)) return 0;
  const denominator = 1 - friction * exposure;
  return denominator > 0 ? friction / denominator : Number.POSITIVE_INFINITY;
}

function effectiveSellCostSlope(lower: number, upper: number, friction: number): number {
  if (!(friction > 0)) return 0;
  return (-Math.log(1 - friction * upper) + Math.log(1 - friction * lower))
    / (upper - lower);
}

function effectiveBuyCostSlope(lower: number, upper: number, friction: number): number {
  if (!(friction > 0)) return 0;
  return (Math.log(1 - friction + friction * upper)
      - Math.log(1 - friction + friction * lower))
    / (upper - lower);
}

function fitQuadraticBackgroundCoefficients(
  grid: Float64Array,
  values: Float64Array,
  lower: number,
  upper: number,
  c1: number,
  c2: number,
  kappaC1: number,
  kappaC2: number,
  sellCostSlope: number,
  inverseTemperature: number,
): {
  baseSlope: number;
  quadraticPrecision: number;
  betaC1: number;
  betaC2: number;
} {
  const size = 5;
  const center = (lower + upper) / 2;
  const normal = new Float64Array(size * size);
  const right = new Float64Array(size);
  for (let index = 0; index < grid.length; index += 1) {
    const action = grid[index]!;
    if (action < lower || action > upper) continue;
    const edgeDistance = Math.min(action - lower, upper - action);
    const weight = Math.max(0.05, Math.min(1, edgeDistance / Math.max(1e-12, (upper - lower) / 8)));
    const features = [
      1,
      action - lower,
      -0.5 * (action - center) ** 2,
      scaledSoftplus(action - c1, kappaC1),
      scaledSoftplus(action - c2, kappaC2),
    ];
    const target = (values[index]! + sellCostSlope * (action - lower)) * inverseTemperature;
    for (let row = 0; row < size; row += 1) {
      right[row] += weight * features[row]! * target;
      for (let column = 0; column < size; column += 1) {
        normal[row * size + column] += weight * features[row]! * features[column]!;
      }
    }
  }
  for (let index = 0; index < size; index += 1) normal[index * size + index] += 1e-10;
  const fitted = solveLinearSystem(normal, right, size);
  return {
    baseSlope: fitted[1]!,
    quadraticPrecision: fitted[2]!,
    betaC1: fitted[3]!,
    betaC2: fitted[4]!,
  };
}

function solveLinearSystem(
  matrixInput: Float64Array,
  vectorInput: Float64Array,
  size: number,
): Float64Array {
  const matrix = matrixInput.slice();
  const vector = vectorInput.slice();
  for (let pivot = 0; pivot < size; pivot += 1) {
    let best = pivot;
    for (let row = pivot + 1; row < size; row += 1) {
      if (Math.abs(matrix[row * size + pivot]!) > Math.abs(matrix[best * size + pivot]!)) best = row;
    }
    if (best !== pivot) {
      for (let column = pivot; column < size; column += 1) {
        [matrix[pivot * size + column], matrix[best * size + column]]
          = [matrix[best * size + column]!, matrix[pivot * size + column]!];
      }
      [vector[pivot], vector[best]] = [vector[best]!, vector[pivot]!];
    }
    const diagonal = matrix[pivot * size + pivot]!;
    if (Math.abs(diagonal) < 1e-18) continue;
    for (let row = pivot + 1; row < size; row += 1) {
      const factor = matrix[row * size + pivot]! / diagonal;
      for (let column = pivot; column < size; column += 1) {
        matrix[row * size + column] -= factor * matrix[pivot * size + column]!;
      }
      vector[row] -= factor * vector[pivot]!;
    }
  }
  const result = new Float64Array(size);
  for (let row = size - 1; row >= 0; row -= 1) {
    let value = vector[row]!;
    for (let column = row + 1; column < size; column += 1) {
      value -= matrix[row * size + column]! * result[column]!;
    }
    result[row] = value / matrix[row * size + row]!;
  }
  return result;
}

function scaledSoftplus(offset: number, kappa: number): number {
  const scaled = kappa * offset;
  const value = scaled > 35
    ? scaled
    : scaled < -35
      ? Math.exp(scaled)
      : Math.log1p(Math.exp(scaled));
  return value / kappa;
}

function finiteDifferenceDerivatives(grid: Float64Array, values: Float64Array): Float64Array {
  const result = new Float64Array(grid.length);
  for (let index = 0; index < grid.length; index += 1) {
    const left = Math.max(0, index - 1);
    const right = Math.min(grid.length - 1, index + 1);
    result[index] = (values[right]! - values[left]!) / (grid[right]! - grid[left]!);
  }
  return result;
}

function decreasingRoot(grid: Float64Array, residuals: Float64Array): number {
  let closestIndex = 0;
  for (let index = 0; index < residuals.length - 1; index += 1) {
    if (Math.abs(residuals[index]!) < Math.abs(residuals[closestIndex]!)) closestIndex = index;
    const left = residuals[index]!;
    const right = residuals[index + 1]!;
    if (left >= 0 && right <= 0) {
      const weight = left === right ? 0.5 : left / (left - right);
      return grid[index]! + weight * (grid[index + 1]! - grid[index]!);
    }
  }
  if (Math.abs(residuals.at(-1)!) < Math.abs(residuals[closestIndex]!)) {
    closestIndex = residuals.length - 1;
  }
  return grid[closestIndex]!;
}

function transitionWidth10To90(
  grid: Float64Array,
  derivatives: Float64Array,
  center: number,
  leftSlope: number,
  rightSlope: number,
): number | null {
  if (Math.abs(rightSlope - leftSlope) < 1e-12) return null;
  const ten = crossingNearest(grid, derivatives, leftSlope + 0.1 * (rightSlope - leftSlope), center);
  const ninety = crossingNearest(grid, derivatives, leftSlope + 0.9 * (rightSlope - leftSlope), center);
  if (ten === null || ninety === null) return null;
  const width = Math.abs(ninety - ten);
  return width > medianSpacing(grid) * 0.25 ? width : null;
}

function crossingNearest(
  grid: Float64Array,
  values: Float64Array,
  level: number,
  center: number,
): number | null {
  let result: number | null = null;
  let distance = Number.POSITIVE_INFINITY;
  for (let index = 0; index < grid.length - 1; index += 1) {
    const left = values[index]! - level;
    const right = values[index + 1]! - level;
    if (left * right > 0) continue;
    const weight = left === right ? 0.5 : left / (left - right);
    const crossing = grid[index]! + weight * (grid[index + 1]! - grid[index]!);
    const nextDistance = Math.abs(crossing - center);
    if (nextDistance < distance) {
      result = crossing;
      distance = nextDistance;
    }
  }
  return result;
}

function interpolate(grid: Float64Array, values: Float64Array, exposure: number): number {
  if (exposure <= grid[0]!) return values[0]!;
  if (exposure >= grid.at(-1)!) return values.at(-1)!;
  let low = 0;
  let high = grid.length - 1;
  while (high - low > 1) {
    const middle = (low + high) >>> 1;
    if (grid[middle]! <= exposure) low = middle;
    else high = middle;
  }
  const weight = (exposure - grid[low]!) / (grid[high]! - grid[low]!);
  return values[low]! + weight * (values[high]! - values[low]!);
}

function medianSpacing(grid: ArrayLike<number>): number {
  const spacings = Array.from({ length: grid.length - 1 }, (_, index) =>
    grid[index + 1]! - grid[index]!).sort((left, right) => left - right);
  return spacings[Math.floor(spacings.length / 2)]!;
}

function validateDirectInputs(
  actions: ArrayLike<number>,
  backgrounds: ArrayLike<number>,
  parameters: DirectIndicatorPredictorParameters,
  options: HandcraftedIndicatorPredictionOptions,
): void {
  if (actions.length < 5 || backgrounds.length < 5
    || !(backgrounds[0]! < actions[0]!)
    || !(backgrounds[backgrounds.length - 1]! > actions[actions.length - 1]!)) {
    throw new Error("Direct indicator predictor requires visible actions inside latent support.");
  }
  for (const grid of [actions, backgrounds]) {
    for (let index = 0; index < grid.length; index += 1) {
      if (!Number.isFinite(grid[index]!)
        || (index > 0 && !(grid[index]! > grid[index - 1]!))) {
        throw new Error("Direct indicator predictor grids must be finite and strictly ordered.");
      }
    }
  }
  if (!(Number.isFinite(parameters.transitionWidthGridCells)
      && parameters.transitionWidthGridCells > 0)
    || !(Number.isFinite(options.temperature) && options.temperature > 0)) {
    throw new Error("Direct indicator predictor smoothing and temperature must be positive.");
  }
  validateExecutionSupport(backgrounds, options.execution);
}

function validateExecutionSupport(
  backgrounds: ArrayLike<number>,
  execution: ExposureExecutionOptions,
): void {
  const lower = backgrounds[0]!;
  const upper = backgrounds[backgrounds.length - 1]!;
  if (!(1 - execution.friction * upper > 0)
    || !(1 - execution.friction + execution.friction * lower > 0)) {
    throw new Error("Direct indicator latent support is infeasible under the configured friction.");
  }
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}
