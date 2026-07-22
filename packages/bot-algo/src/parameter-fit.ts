import {
  conditionalFourSegmentLogKernel,
  fitConditionalFourSegmentPolicy,
  type ConditionalFourSegmentFitTermination,
  type ConditionalFourSegmentModelOptions,
  type ConditionalFourSegmentParameters,
} from "./conditional-exposure-distribution.js";

export interface ConditionalFourSegmentScoreFitOptions
  extends ConditionalFourSegmentModelOptions {
  /** Per-cell non-negative weights in row-major state/action order. */
  weights?: ArrayLike<number>;
  /** L2 penalty reported for the four learned linear coefficients. */
  ridge?: number;
}

export interface ConditionalFourSegmentScoreFit {
  parameters: ConditionalFourSegmentParameters;
  /** Analytically eliminated alpha(x) for every supplied current-exposure row. */
  sliceOffsets: Float64Array;
  regularizedLoss: number;
  weightedMeanSquaredError: number;
  rootMeanSquaredError: number;
  maximumAbsoluteError: number;
  weightedRSquared: number;
  iterations: number;
  restarts: number;
  termination: ConditionalFourSegmentFitTermination;
  converged: boolean;
  /** Cutoffs are supplied independently rather than estimated from score MSE. */
  fittedSupport: false;
}

/**
 * Fit score rows after eliminating their arbitrary alpha(x) offsets. Score
 * rows are converted to their equivalent conditional probabilities; the
 * shared cutoff-aware probability fitter then performs the weighted
 * quadratic/hinge projection and optional cross-entropy refinement.
 */
export function fitConditionalFourSegmentScores(
  actionGrid: ArrayLike<number>,
  scores: ArrayLike<number>,
  currentExposures: ArrayLike<number>,
  options: ConditionalFourSegmentScoreFitOptions,
): ConditionalFourSegmentScoreFit {
  validateScoreInputs(actionGrid, scores, currentExposures, options.weights);
  const weights = options.weights
    ? Float64Array.from(options.weights)
    : Float64Array.from({ length: scores.length }, () => 1);
  const probabilities = new Float64Array(scores.length);
  for (let row = 0; row < currentExposures.length; row += 1) {
    const offset = row * actionGrid.length;
    let maximum = Number.NEGATIVE_INFINITY;
    for (let action = 0; action < actionGrid.length; action += 1) {
      if (weights[offset + action]! > 0) maximum = Math.max(maximum, scores[offset + action]!);
    }
    let total = 0;
    for (let action = 0; action < actionGrid.length; action += 1) {
      const value = weights[offset + action]! > 0
        ? Math.exp(Math.max(-745, scores[offset + action]! - maximum)) * weights[offset + action]!
        : 0;
      probabilities[offset + action] = value;
      total += value;
    }
    if (!(total > 0)) throw new Error("Conditional four-segment score rows need positive weight.");
    for (let action = 0; action < actionGrid.length; action += 1) {
      probabilities[offset + action] /= total;
    }
  }
  const policyFit = fitConditionalFourSegmentPolicy(
    actionGrid,
    probabilities,
    currentExposures,
    {
      ...options,
      refineProjectedFit: options.maxIterations === 0 ? false : options.refineProjectedFit,
    },
  );
  const diagnostics = scoreDiagnostics(
    actionGrid,
    scores,
    currentExposures,
    weights,
    policyFit.parameters,
  );
  const ridge = Math.max(0, options.ridge ?? 0);
  const ridgeValue = policyFit.rawParameters.subarray(2, 6)
    .reduce((sum, value) => sum + value * value, 0) * ridge;
  return {
    parameters: policyFit.parameters,
    sliceOffsets: diagnostics.offsets,
    regularizedLoss: diagnostics.weightedMeanSquaredError + ridgeValue,
    weightedMeanSquaredError: diagnostics.weightedMeanSquaredError,
    rootMeanSquaredError: Math.sqrt(diagnostics.weightedMeanSquaredError),
    maximumAbsoluteError: diagnostics.maximumAbsoluteError,
    weightedRSquared: diagnostics.weightedRSquared,
    iterations: policyFit.iterations,
    restarts: policyFit.restarts,
    termination: policyFit.termination,
    converged: policyFit.converged,
    fittedSupport: false,
  };
}

/** Fit an oracle regret table after converting it to Y=-R/temperature. */
export function fitConditionalFourSegmentRegret(
  actionGrid: ArrayLike<number>,
  regrets: ArrayLike<number>,
  currentExposures: ArrayLike<number>,
  temperature: number,
  options: ConditionalFourSegmentScoreFitOptions,
): ConditionalFourSegmentScoreFit {
  if (!(Number.isFinite(temperature) && temperature > 0)) {
    throw new Error("Conditional four-segment regret temperature must be positive.");
  }
  return fitConditionalFourSegmentScores(
    actionGrid,
    Float64Array.from(regrets, (regret) => -regret / temperature),
    currentExposures,
    { ...options, temperature: options.temperature ?? temperature },
  );
}

export function conditionalFourSegmentScore(
  action: number,
  currentExposure: number,
  parameters: ConditionalFourSegmentParameters,
): number {
  return conditionalFourSegmentLogKernel(action, currentExposure, parameters);
}

export function conditionalFourSegmentScoreMatrix(
  actionGrid: ArrayLike<number>,
  currentExposures: ArrayLike<number>,
  parameters: ConditionalFourSegmentParameters,
  sliceOffsets?: ArrayLike<number>,
): Float64Array {
  if (sliceOffsets && sliceOffsets.length !== currentExposures.length) {
    throw new Error("Conditional four-segment slice offsets do not match the state grid.");
  }
  return Float64Array.from(
    { length: actionGrid.length * currentExposures.length },
    (_, index) => {
      const row = Math.floor(index / actionGrid.length);
      return conditionalFourSegmentScore(
        actionGrid[index % actionGrid.length]!,
        currentExposures[row]!,
        parameters,
      ) + (sliceOffsets?.[row] ?? 0);
    },
  );
}

function scoreDiagnostics(
  actions: ArrayLike<number>,
  scores: ArrayLike<number>,
  states: ArrayLike<number>,
  weights: Float64Array,
  parameters: ConditionalFourSegmentParameters,
) {
  const offsets = new Float64Array(states.length);
  let totalWeight = 0;
  let squaredError = 0;
  let maximumAbsoluteError = 0;
  let weightedMean = 0;
  for (let row = 0; row < states.length; row += 1) {
    let rowWeight = 0;
    let rowOffset = 0;
    for (let action = 0; action < actions.length; action += 1) {
      const index = row * actions.length + action;
      const weight = weights[index]!;
      rowWeight += weight;
      rowOffset += weight * (scores[index]!
        - conditionalFourSegmentScore(actions[action]!, states[row]!, parameters));
    }
    offsets[row] = rowOffset / rowWeight;
  }
  for (let index = 0; index < scores.length; index += 1) {
    totalWeight += weights[index]!;
    weightedMean += weights[index]! * scores[index]!;
  }
  weightedMean /= totalWeight;
  let totalVariation = 0;
  for (let row = 0; row < states.length; row += 1) {
    for (let action = 0; action < actions.length; action += 1) {
      const index = row * actions.length + action;
      const predicted = conditionalFourSegmentScore(actions[action]!, states[row]!, parameters)
        + offsets[row]!;
      const residual = predicted - scores[index]!;
      squaredError += weights[index]! * residual * residual;
      maximumAbsoluteError = Math.max(maximumAbsoluteError, Math.abs(residual));
      totalVariation += weights[index]! * (scores[index]! - weightedMean) ** 2;
    }
  }
  const weightedMeanSquaredError = squaredError / totalWeight;
  return {
    offsets,
    weightedMeanSquaredError,
    maximumAbsoluteError,
    weightedRSquared: totalVariation > 0 ? 1 - squaredError / totalVariation : squaredError === 0 ? 1 : 0,
  };
}

function validateScoreInputs(
  actions: ArrayLike<number>,
  scores: ArrayLike<number>,
  states: ArrayLike<number>,
  inputWeights?: ArrayLike<number>,
): void {
  if (actions.length < 5 || states.length < 3 || scores.length !== actions.length * states.length) {
    throw new Error("Conditional four-segment score fit requires complete score rows.");
  }
  if (inputWeights && inputWeights.length !== scores.length) {
    throw new Error("Conditional four-segment weights must match the score table.");
  }
  for (let index = 0; index < actions.length; index += 1) {
    if (!Number.isFinite(actions[index]!) || (index > 0 && !(actions[index]! > actions[index - 1]!))) {
      throw new Error("Conditional four-segment actions must be finite and ordered.");
    }
  }
  for (const value of [...Array.from(scores), ...Array.from(states)]) {
    if (!Number.isFinite(value)) throw new Error("Conditional four-segment score inputs must be finite.");
  }
  if (inputWeights) {
    for (const weight of Array.from(inputWeights)) {
      if (!(Number.isFinite(weight) && weight >= 0)) {
        throw new Error("Conditional four-segment weights must be finite and non-negative.");
      }
    }
  }
}
