import type {
  ConditionalFourSegmentFitTermination,
  ConditionalFourSegmentModelOptions,
  ConditionalFourSegmentParameters,
} from "./conditional-exposure-distribution.js";

const LINEAR_PARAMETER_COUNT = 8;
const INTERIOR_RAW_PARAMETER_COUNT = 5;
const COMPLETE_RAW_PARAMETER_COUNT = 9;

export interface ConditionalFourSegmentScoreFitOptions
  extends ConditionalFourSegmentModelOptions {
  /** Per-cell least-squares weights in row-major state/action order. */
  weights?: ArrayLike<number>;
  /** Ridge penalty for the eight slope coefficients. Slice offsets are never penalized. */
  ridge?: number;
  /** Fit endpoint widths and sharpnesses. Keep false without latent boundary-layer samples. */
  fitSupport?: boolean;
  finiteDifferenceStep?: number;
}

export interface ConditionalFourSegmentScoreFit {
  parameters: ConditionalFourSegmentParameters;
  /** The analytically eliminated alpha(x) value for every supplied current-exposure row. */
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
  fittedSupport: boolean;
}

interface FitContext {
  actions: Float64Array;
  scores: Float64Array;
  states: Float64Array;
  weights: Float64Array;
  latentLower: number;
  latentUpper: number;
  visibleLower: number;
  visibleUpper: number;
  minimumKappa: number;
  minimumSupportSharpness: number;
  ridge: number;
  fitSupport: boolean;
  fixedSupport: readonly [number, number, number, number];
}

interface ProjectedFit {
  loss: number;
  weightedMeanSquaredError: number;
  coefficients: Float64Array;
  offsets: Float64Array;
  maximumAbsoluteError: number;
  weightedRSquared: number;
}

interface OptimizationResult {
  raw: Float64Array;
  projected: ProjectedFit;
  iterations: number;
  termination: ConditionalFourSegmentFitTermination;
  converged: boolean;
}

/**
 * Fits score values Y(x,a) directly using variable projection. The input is
 * row-major: one complete action-grid row for every current exposure.
 */
export function fitConditionalFourSegmentScores(
  actionGrid: ArrayLike<number>,
  scores: ArrayLike<number>,
  currentExposures: ArrayLike<number>,
  options: ConditionalFourSegmentScoreFitOptions,
): ConditionalFourSegmentScoreFit {
  const context = createFitContext(actionGrid, scores, currentExposures, options);
  const starts = initialStructuralValues(context, options);
  const restartCount = Math.max(1, Math.min(
    starts.length,
    Math.floor(options.restartCount ?? starts.length),
  ));
  const maximumIterations = Math.max(0, Math.floor(options.maxIterations ?? 80));
  const tolerance = Math.max(Number.EPSILON, options.tolerance ?? 1e-7);
  const finiteDifferenceStep = Math.max(1e-7, options.finiteDifferenceStep ?? 2e-4);
  let best: OptimizationResult | undefined;
  for (let restart = 0; restart < restartCount; restart += 1) {
    const candidate = optimizeStructuralParameters(
      starts[restart]!,
      context,
      maximumIterations,
      tolerance,
      finiteDifferenceStep,
    );
    if (!best || candidate.projected.loss < best.projected.loss) best = candidate;
  }
  if (!best) throw new Error("Conditional four-segment score optimizer did not run.");
  const structural = structuralParametersFromRaw(best.raw, context);
  const coefficients = best.projected.coefficients;
  const parameters: ConditionalFourSegmentParameters = {
    latentLower: context.latentLower,
    latentUpper: context.latentUpper,
    visibleLower: context.visibleLower,
    visibleUpper: context.visibleUpper,
    c1: structural.c1,
    c2: structural.c2,
    leftSupportWidth: structural.leftSupportWidth,
    rightSupportWidth: structural.rightSupportWidth,
    leftSupportSharpness: structural.leftSupportSharpness,
    rightSupportSharpness: structural.rightSupportSharpness,
    baseSlope: [coefficients[0]!, coefficients[1]!],
    betaC1: [coefficients[2]!, coefficients[3]!],
    betaX: [coefficients[4]!, coefficients[5]!],
    betaC2: [coefficients[6]!, coefficients[7]!],
    kappaC1: structural.kappaC1,
    kappaX: structural.kappaX,
    kappaC2: structural.kappaC2,
  };
  return {
    parameters,
    sliceOffsets: best.projected.offsets,
    regularizedLoss: best.projected.loss,
    weightedMeanSquaredError: best.projected.weightedMeanSquaredError,
    rootMeanSquaredError: Math.sqrt(best.projected.weightedMeanSquaredError),
    maximumAbsoluteError: best.projected.maximumAbsoluteError,
    weightedRSquared: best.projected.weightedRSquared,
    iterations: best.iterations,
    restarts: restartCount,
    termination: best.termination,
    converged: best.converged,
    fittedSupport: context.fitSupport,
  };
}

/** Fits an oracle regret table after converting it to Y=-R/temperature. */
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
  const scores = Float64Array.from(regrets, (regret) => {
    if (!Number.isFinite(regret)) {
      throw new Error("Conditional four-segment regrets must be finite.");
    }
    return -regret / temperature;
  });
  return fitConditionalFourSegmentScores(actionGrid, scores, currentExposures, options);
}

/** Evaluates the unnormalized fitted score without a nuisance slice offset. */
export function conditionalFourSegmentScore(
  action: number,
  currentExposure: number,
  parameters: ConditionalFourSegmentParameters,
): number {
  if (!(action > parameters.latentLower && action < parameters.latentUpper)) {
    return Number.NEGATIVE_INFINITY;
  }
  const xi = scaledCurrentExposure(
    currentExposure,
    parameters.latentLower,
    parameters.latentUpper,
  );
  return (parameters.baseSlope[0] + parameters.baseSlope[1] * xi)
      * (action - parameters.latentLower)
    + (parameters.betaC1[0] + parameters.betaC1[1] * xi)
      * scaledSoftplus(action - parameters.c1, parameters.kappaC1)
    + (parameters.betaX[0] + parameters.betaX[1] * xi)
      * scaledSoftplus(action - currentExposure, parameters.kappaX)
    + (parameters.betaC2[0] + parameters.betaC2[1] * xi)
      * scaledSoftplus(action - parameters.c2, parameters.kappaC2)
    + compactEnvelopeLogValue(action, parameters);
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

function createFitContext(
  actionGrid: ArrayLike<number>,
  scores: ArrayLike<number>,
  currentExposures: ArrayLike<number>,
  options: ConditionalFourSegmentScoreFitOptions,
): FitContext {
  if (actionGrid.length < 5 || currentExposures.length < 3
    || scores.length !== actionGrid.length * currentExposures.length) {
    throw new Error("Conditional four-segment score fit requires complete score rows.");
  }
  const actions = Float64Array.from(actionGrid);
  const states = Float64Array.from(currentExposures);
  const scoreValues = Float64Array.from(scores);
  for (let index = 0; index < actions.length; index += 1) {
    if (!Number.isFinite(actions[index]!)
      || (index > 0 && !(actions[index]! > actions[index - 1]!))) {
      throw new Error("Conditional four-segment action grid must be finite and strictly ordered.");
    }
  }
  if (!(Number.isFinite(options.latentLower) && Number.isFinite(options.latentUpper)
    && options.latentLower < actions[0]! && actions.at(-1)! < options.latentUpper)) {
    throw new Error("Conditional four-segment score rows must lie strictly inside latent support.");
  }
  for (const state of states) {
    if (!Number.isFinite(state)) {
      throw new Error("Conditional four-segment current exposures must be finite.");
    }
  }
  for (const score of scoreValues) {
    if (!Number.isFinite(score)) {
      throw new Error("Conditional four-segment scores must be finite.");
    }
  }
  const visibleLower = options.visibleLower ?? actions[0]!;
  const visibleUpper = options.visibleUpper ?? actions.at(-1)!;
  if (!(options.latentLower < visibleLower && visibleLower < visibleUpper
    && visibleUpper < options.latentUpper)) {
    throw new Error("Conditional four-segment visible support must be inside latent support.");
  }
  const weights = options.weights
    ? Float64Array.from(options.weights)
    : Float64Array.from({ length: scoreValues.length }, () => 1);
  if (weights.length !== scoreValues.length) {
    throw new Error("Conditional four-segment weights must match the score table.");
  }
  for (let row = 0; row < states.length; row += 1) {
    let rowWeight = 0;
    for (let action = 0; action < actions.length; action += 1) {
      const weight = weights[row * actions.length + action]!;
      if (!(Number.isFinite(weight) && weight >= 0)) {
        throw new Error("Conditional four-segment weights must be finite and non-negative.");
      }
      rowWeight += weight;
    }
    if (!(rowWeight > 0)) {
      throw new Error("Every conditional four-segment slice must have positive weight.");
    }
  }
  const latentSpan = options.latentUpper - options.latentLower;
  const leftSupportWidth = options.initialLeftSupportWidth ?? latentSpan / 5;
  const rightSupportWidth = options.initialRightSupportWidth ?? latentSpan / 5;
  const leftSupportSharpness = options.leftSupportSharpness ?? 1;
  const rightSupportSharpness = options.rightSupportSharpness ?? 1;
  if (!(leftSupportWidth > 0 && rightSupportWidth > 0
    && leftSupportWidth + rightSupportWidth < latentSpan
    && leftSupportSharpness > 0 && rightSupportSharpness > 0)) {
    throw new Error("Conditional four-segment support taper parameters are invalid.");
  }
  const minimumKappa = options.minimumKappa ?? 4.394 / latentSpan;
  const minimumSupportSharpness = options.minimumSupportSharpness ?? 1e-5;
  const ridge = options.ridge ?? 1e-10;
  if (!(minimumKappa > 0 && minimumSupportSharpness > 0
    && Number.isFinite(ridge) && ridge >= 0)) {
    throw new Error("Conditional four-segment fitting regularization is invalid.");
  }
  return {
    actions,
    scores: scoreValues,
    states,
    weights,
    latentLower: options.latentLower,
    latentUpper: options.latentUpper,
    visibleLower,
    visibleUpper,
    minimumKappa,
    minimumSupportSharpness,
    ridge,
    fitSupport: options.fitSupport ?? false,
    fixedSupport: [
      leftSupportWidth,
      rightSupportWidth,
      leftSupportSharpness,
      rightSupportSharpness,
    ],
  };
}

function initialStructuralValues(
  context: FitContext,
  options: ConditionalFourSegmentScoreFitOptions,
): Float64Array[] {
  const detected = detectFixedBreakpoints(context);
  const span = context.latentUpper - context.latentLower;
  const visibleSpan = context.visibleUpper - context.visibleLower;
  const requestedC1 = options.initialC1;
  const requestedC2 = options.initialC2;
  const breakpointStarts: Array<readonly [number, number]> = [];
  if (requestedC1 !== undefined || requestedC2 !== undefined) {
    breakpointStarts.push([
      requestedC1 ?? detected?.[0] ?? context.visibleLower + visibleSpan / 3,
      requestedC2 ?? detected?.[1] ?? context.visibleUpper - visibleSpan / 3,
    ]);
  }
  if (detected) breakpointStarts.push(detected);
  breakpointStarts.push(
    [context.visibleLower + visibleSpan / 3, context.visibleUpper - visibleSpan / 3],
    [context.latentLower + span / 4, context.latentUpper - span / 4],
  );
  const typicalStep = medianActionStep(context.actions);
  const baseKappa = Math.max(context.minimumKappa * 1.01, 4.394 / (4 * typicalStep));
  const kappas = [
    options.initialKappaC1 ?? baseKappa,
    options.initialKappaX ?? baseKappa,
    options.initialKappaC2 ?? baseKappa,
  ] as const;
  const unique = new Map<string, Float64Array>();
  for (let index = 0; index < breakpointStarts.length; index += 1) {
    const [unorderedC1, unorderedC2] = breakpointStarts[index]!;
    const c1 = clamp(
      Math.min(unorderedC1, unorderedC2),
      context.latentLower + span * 1e-6,
      context.latentUpper - span * 2e-6,
    );
    const c2 = clamp(
      Math.max(unorderedC1, unorderedC2),
      c1 + span * 1e-6,
      context.latentUpper - span * 1e-6,
    );
    const sharpnessScale = index < 2 ? 1 : 0.5;
    const raw = rawStructuralParameters(
      c1,
      c2,
      kappas[0] * sharpnessScale,
      kappas[1] * sharpnessScale,
      kappas[2] * sharpnessScale,
      context,
    );
    unique.set(Array.from(raw, (value) => value.toFixed(6)).join(","), raw);
  }
  return [...unique.values()];
}

function detectFixedBreakpoints(context: FitContext): readonly [number, number] | undefined {
  const actions = context.actions;
  const typicalStep = medianActionStep(actions);
  const actionSpan = actions.at(-1)! - actions[0]!;
  const movingExclusion = Math.max(typicalStep * 2.5, actionSpan / 14);
  const energy = new Float64Array(actions.length);
  for (let actionIndex = 1; actionIndex < actions.length - 1; actionIndex += 1) {
    const location = actions[actionIndex]!;
    let total = 0;
    let totalWeight = 0;
    for (let row = 0; row < context.states.length; row += 1) {
      if (Math.abs(context.states[row]! - location) <= movingExclusion) continue;
      const leftOffset = row * actions.length + actionIndex - 1;
      const centerOffset = leftOffset + 1;
      const rightOffset = centerOffset + 1;
      const weight = Math.min(
        context.weights[leftOffset]!,
        context.weights[centerOffset]!,
        context.weights[rightOffset]!,
      );
      if (!(weight > 0)) continue;
      const leftSlope = (context.scores[centerOffset]! - context.scores[leftOffset]!)
        / (actions[actionIndex]! - actions[actionIndex - 1]!);
      const rightSlope = (context.scores[rightOffset]! - context.scores[centerOffset]!)
        / (actions[actionIndex + 1]! - actions[actionIndex]!);
      total += weight * Math.abs(rightSlope - leftSlope);
      totalWeight += weight;
    }
    energy[actionIndex] = totalWeight > 0 ? total / totalWeight : 0;
  }
  const smoothed = Float64Array.from(energy, (_, index) => {
    let total = 0;
    let totalWeight = 0;
    for (let offset = -2; offset <= 2; offset += 1) {
      const neighbor = index + offset;
      if (neighbor <= 0 || neighbor >= energy.length - 1) continue;
      const weight = 3 - Math.abs(offset);
      total += weight * energy[neighbor]!;
      totalWeight += weight;
    }
    return totalWeight > 0 ? total / totalWeight : 0;
  });
  const candidates = Array.from({ length: actions.length - 2 }, (_, index) => index + 1)
    .sort((left, right) => smoothed[right]! - smoothed[left]!);
  const selected: number[] = [];
  const minimumSeparation = Math.max(typicalStep * 4, actionSpan / 8);
  for (const candidate of candidates) {
    if (!(smoothed[candidate]! > 0)) break;
    const location = actions[candidate]!;
    if (selected.every((other) => Math.abs(location - other) >= minimumSeparation)) {
      selected.push(location);
      if (selected.length === 2) break;
    }
  }
  return selected.length === 2
    ? selected.sort((left, right) => left - right) as [number, number]
    : undefined;
}

function optimizeStructuralParameters(
  initial: Float64Array,
  context: FitContext,
  maximumIterations: number,
  tolerance: number,
  finiteDifferenceStep: number,
): OptimizationResult {
  let raw: Float64Array<ArrayBufferLike> = initial.slice();
  let projected = projectLinearParameters(raw, context);
  if (maximumIterations === 0) {
    return {
      raw,
      projected,
      iterations: 0,
      termination: "iteration-limit",
      converged: false,
    };
  }
  let gradient = structuralGradient(raw, context, finiteDifferenceStep);
  let inverseHessian = identityMatrix(raw.length);
  let stableIterations = 0;
  for (let iteration = 0; iteration < maximumIterations; iteration += 1) {
    if (maximumAbsolute(gradient) <= tolerance) {
      return { raw, projected, iterations: iteration, termination: "gradient", converged: true };
    }
    let direction = matrixVectorProduct(inverseHessian, gradient);
    for (let index = 0; index < direction.length; index += 1) direction[index] *= -1;
    let directionalDerivative = dot(gradient, direction);
    if (!(directionalDerivative < 0) || !Number.isFinite(directionalDerivative)) {
      inverseHessian = identityMatrix(raw.length);
      direction = Float64Array.from(gradient, (value) => -value);
      directionalDerivative = -dot(gradient, gradient);
    }
    const maximumDirection = maximumAbsolute(direction);
    if (maximumDirection > 3) {
      const scale = 3 / maximumDirection;
      for (let index = 0; index < direction.length; index += 1) direction[index] *= scale;
      directionalDerivative *= scale;
    }
    let step = 1;
    let nextRaw: Float64Array | undefined;
    let nextProjected: ProjectedFit | undefined;
    for (let search = 0; search < 24; search += 1) {
      const candidate = Float64Array.from(raw, (value, index) =>
        clamp(value + step * direction[index]!, -16, 16));
      const candidateProjected = projectLinearParameters(candidate, context);
      if (Number.isFinite(candidateProjected.loss)
        && candidateProjected.loss <= projected.loss + 1e-4 * step * directionalDerivative) {
        nextRaw = candidate;
        nextProjected = candidateProjected;
        break;
      }
      step *= 0.5;
    }
    if (!nextRaw || !nextProjected) {
      return {
        raw,
        projected,
        iterations: iteration,
        termination: "line-search",
        converged: false,
      };
    }
    const nextGradient = structuralGradient(nextRaw, context, finiteDifferenceStep);
    const parameterDelta = subtract(nextRaw, raw);
    const gradientDelta = subtract(nextGradient, gradient);
    inverseHessian = updateInverseHessian(
      inverseHessian,
      parameterDelta,
      gradientDelta,
    );
    const relativeImprovement = (projected.loss - nextProjected.loss)
      / Math.max(1e-12, Math.abs(projected.loss));
    stableIterations = relativeImprovement <= tolerance ? stableIterations + 1 : 0;
    raw = nextRaw;
    projected = nextProjected;
    gradient = nextGradient;
    if (stableIterations >= 4) {
      return {
        raw,
        projected,
        iterations: iteration + 1,
        termination: "relative-loss",
        converged: true,
      };
    }
  }
  return {
    raw,
    projected,
    iterations: maximumIterations,
    termination: "iteration-limit",
    converged: false,
  };
}

function structuralGradient(
  raw: Float64Array,
  context: FitContext,
  finiteDifferenceStep: number,
): Float64Array {
  const gradient = new Float64Array(raw.length);
  for (let index = 0; index < raw.length; index += 1) {
    const step = finiteDifferenceStep * Math.max(1, Math.abs(raw[index]!));
    const lower = raw.slice();
    const upper = raw.slice();
    lower[index] = clamp(lower[index]! - step, -16, 16);
    upper[index] = clamp(upper[index]! + step, -16, 16);
    const span = upper[index]! - lower[index]!;
    gradient[index] = span > 0
      ? (projectLinearParameters(upper, context).loss
        - projectLinearParameters(lower, context).loss) / span
      : 0;
  }
  return gradient;
}

function projectLinearParameters(raw: Float64Array, context: FitContext): ProjectedFit {
  const structural = structuralParametersFromRaw(raw, context);
  const sampleCount = context.actions.length * context.states.length;
  const features = new Float64Array(sampleCount * LINEAR_PARAMETER_COUNT);
  const responses = new Float64Array(sampleCount);
  const featureMeans = new Float64Array(context.states.length * LINEAR_PARAMETER_COUNT);
  const responseMeans = new Float64Array(context.states.length);
  const rowWeights = new Float64Array(context.states.length);
  let totalWeight = 0;
  for (let row = 0; row < context.states.length; row += 1) {
    const current = context.states[row]!;
    const xi = scaledCurrentExposure(current, context.latentLower, context.latentUpper);
    for (let actionIndex = 0; actionIndex < context.actions.length; actionIndex += 1) {
      const sample = row * context.actions.length + actionIndex;
      const action = context.actions[actionIndex]!;
      const weight = context.weights[sample]!;
      const featureOffset = sample * LINEAR_PARAMETER_COUNT;
      const c1 = scaledSoftplus(action - structural.c1, structural.kappaC1);
      const moving = scaledSoftplus(action - current, structural.kappaX);
      const c2 = scaledSoftplus(action - structural.c2, structural.kappaC2);
      const values = [
        action - context.latentLower,
        xi * (action - context.latentLower),
        c1,
        xi * c1,
        moving,
        xi * moving,
        c2,
        xi * c2,
      ];
      for (let parameter = 0; parameter < LINEAR_PARAMETER_COUNT; parameter += 1) {
        features[featureOffset + parameter] = values[parameter]!;
        featureMeans[row * LINEAR_PARAMETER_COUNT + parameter] += weight * values[parameter]!;
      }
      const gate = compactEnvelopeLogValueStructural(action, structural, context);
      responses[sample] = context.scores[sample]! - gate;
      responseMeans[row] += weight * responses[sample]!;
      rowWeights[row] += weight;
      totalWeight += weight;
    }
    for (let parameter = 0; parameter < LINEAR_PARAMETER_COUNT; parameter += 1) {
      featureMeans[row * LINEAR_PARAMETER_COUNT + parameter] /= rowWeights[row]!;
    }
    responseMeans[row] /= rowWeights[row]!;
  }
  const normal = new Float64Array(LINEAR_PARAMETER_COUNT * LINEAR_PARAMETER_COUNT);
  const right = new Float64Array(LINEAR_PARAMETER_COUNT);
  for (let row = 0; row < context.states.length; row += 1) {
    for (let actionIndex = 0; actionIndex < context.actions.length; actionIndex += 1) {
      const sample = row * context.actions.length + actionIndex;
      const weight = context.weights[sample]!;
      if (!(weight > 0)) continue;
      const featureOffset = sample * LINEAR_PARAMETER_COUNT;
      const centeredResponse = responses[sample]! - responseMeans[row]!;
      for (let left = 0; left < LINEAR_PARAMETER_COUNT; left += 1) {
        const centeredLeft = features[featureOffset + left]!
          - featureMeans[row * LINEAR_PARAMETER_COUNT + left]!;
        right[left] += weight * centeredLeft * centeredResponse;
        for (let column = 0; column <= left; column += 1) {
          normal[left * LINEAR_PARAMETER_COUNT + column] += weight * centeredLeft
            * (features[featureOffset + column]!
              - featureMeans[row * LINEAR_PARAMETER_COUNT + column]!);
        }
      }
    }
  }
  for (let row = 0; row < LINEAR_PARAMETER_COUNT; row += 1) {
    for (let column = 0; column < row; column += 1) {
      normal[column * LINEAR_PARAMETER_COUNT + row]
        = normal[row * LINEAR_PARAMETER_COUNT + column]!;
    }
    normal[row * LINEAR_PARAMETER_COUNT + row] += context.ridge;
  }
  const coefficients = solveSymmetricSystem(normal, right);
  const offsets = new Float64Array(context.states.length);
  for (let row = 0; row < context.states.length; row += 1) {
    let modeledMean = 0;
    for (let parameter = 0; parameter < LINEAR_PARAMETER_COUNT; parameter += 1) {
      modeledMean += coefficients[parameter]!
        * featureMeans[row * LINEAR_PARAMETER_COUNT + parameter]!;
    }
    offsets[row] = responseMeans[row]! - modeledMean;
  }
  let squaredError = 0;
  let maximumAbsoluteError = 0;
  let totalVariation = 0;
  for (let row = 0; row < context.states.length; row += 1) {
    let rowScoreMean = 0;
    for (let actionIndex = 0; actionIndex < context.actions.length; actionIndex += 1) {
      const sample = row * context.actions.length + actionIndex;
      rowScoreMean += context.weights[sample]! * context.scores[sample]!;
    }
    rowScoreMean /= rowWeights[row]!;
    for (let actionIndex = 0; actionIndex < context.actions.length; actionIndex += 1) {
      const sample = row * context.actions.length + actionIndex;
      let fitted = offsets[row]!;
      for (let parameter = 0; parameter < LINEAR_PARAMETER_COUNT; parameter += 1) {
        fitted += coefficients[parameter]!
          * features[sample * LINEAR_PARAMETER_COUNT + parameter]!;
      }
      const residual = responses[sample]! - fitted;
      const weight = context.weights[sample]!;
      squaredError += weight * residual * residual;
      maximumAbsoluteError = Math.max(maximumAbsoluteError, Math.abs(residual));
      totalVariation += weight * (context.scores[sample]! - rowScoreMean) ** 2;
    }
  }
  const ridgePenalty = context.ridge * dot(coefficients, coefficients);
  return {
    loss: (squaredError + ridgePenalty) / totalWeight,
    weightedMeanSquaredError: squaredError / totalWeight,
    coefficients,
    offsets,
    maximumAbsoluteError,
    weightedRSquared: totalVariation > 0 ? 1 - squaredError / totalVariation : 1,
  };
}

function solveSymmetricSystem(matrix: Float64Array, right: Float64Array): Float64Array {
  const size = right.length;
  const maximumDiagonal = Array.from({ length: size }, (_, index) =>
    Math.abs(matrix[index * size + index]!))
    .reduce((maximum, value) => Math.max(maximum, value), 0);
  const jitter = Math.max(1e-14, maximumDiagonal * 1e-12);
  const coefficients = matrix.slice();
  const values = right.slice();
  for (let index = 0; index < size; index += 1) {
    coefficients[index * size + index] += jitter;
  }
  for (let column = 0; column < size; column += 1) {
    let pivot = column;
    for (let row = column + 1; row < size; row += 1) {
      if (Math.abs(coefficients[row * size + column]!)
        > Math.abs(coefficients[pivot * size + column]!)) pivot = row;
    }
    if (pivot !== column) {
      for (let index = column; index < size; index += 1) {
        const left = column * size + index;
        const rightOffset = pivot * size + index;
        [coefficients[left], coefficients[rightOffset]] = [
          coefficients[rightOffset]!,
          coefficients[left]!,
        ];
      }
      [values[column], values[pivot]] = [values[pivot]!, values[column]!];
    }
    const diagonal = coefficients[column * size + column]!;
    if (Math.abs(diagonal) <= Number.EPSILON) continue;
    for (let row = column + 1; row < size; row += 1) {
      const factor = coefficients[row * size + column]! / diagonal;
      coefficients[row * size + column] = 0;
      for (let index = column + 1; index < size; index += 1) {
        coefficients[row * size + index] -= factor
          * coefficients[column * size + index]!;
      }
      values[row] -= factor * values[column]!;
    }
  }
  const result = new Float64Array(size);
  for (let row = size - 1; row >= 0; row -= 1) {
    let value = values[row]!;
    for (let column = row + 1; column < size; column += 1) {
      value -= coefficients[row * size + column]! * result[column]!;
    }
    const diagonal = coefficients[row * size + row]!;
    result[row] = Math.abs(diagonal) > Number.EPSILON ? value / diagonal : 0;
  }
  return result;
}

function structuralParametersFromRaw(raw: Float64Array, context: FitContext) {
  const span = context.latentUpper - context.latentLower;
  const firstFraction = sigmoid(raw[0]!);
  const c1 = context.latentLower + span * firstFraction;
  const secondFraction = sigmoid(raw[1]!);
  const c2 = c1 + (context.latentUpper - c1) * secondFraction;
  const supportRaw = context.fitSupport ? raw : undefined;
  const leftFraction = supportRaw ? sigmoid(supportRaw[5]!) : 0;
  const leftSupportWidth = supportRaw
    ? span * leftFraction
    : context.fixedSupport[0];
  const rightSupportWidth = supportRaw
    ? (span - leftSupportWidth) * sigmoid(supportRaw[6]!)
    : context.fixedSupport[1];
  return {
    c1,
    c2,
    kappaC1: context.minimumKappa + softplus(raw[2]!),
    kappaX: context.minimumKappa + softplus(raw[3]!),
    kappaC2: context.minimumKappa + softplus(raw[4]!),
    leftSupportWidth,
    rightSupportWidth,
    leftSupportSharpness: supportRaw
      ? context.minimumSupportSharpness + softplus(supportRaw[7]!)
      : context.fixedSupport[2],
    rightSupportSharpness: supportRaw
      ? context.minimumSupportSharpness + softplus(supportRaw[8]!)
      : context.fixedSupport[3],
  };
}

function rawStructuralParameters(
  c1: number,
  c2: number,
  kappaC1: number,
  kappaX: number,
  kappaC2: number,
  context: FitContext,
): Float64Array {
  const span = context.latentUpper - context.latentLower;
  const result = new Float64Array(
    context.fitSupport ? COMPLETE_RAW_PARAMETER_COUNT : INTERIOR_RAW_PARAMETER_COUNT,
  );
  const firstFraction = clamp((c1 - context.latentLower) / span, 1e-7, 1 - 1e-7);
  const secondFraction = clamp(
    (c2 - c1) / (context.latentUpper - c1),
    1e-7,
    1 - 1e-7,
  );
  result[0] = logit(firstFraction);
  result[1] = logit(secondFraction);
  result[2] = inverseSoftplus(Math.max(1e-12, kappaC1 - context.minimumKappa));
  result[3] = inverseSoftplus(Math.max(1e-12, kappaX - context.minimumKappa));
  result[4] = inverseSoftplus(Math.max(1e-12, kappaC2 - context.minimumKappa));
  if (context.fitSupport) {
    const leftFraction = clamp(context.fixedSupport[0] / span, 1e-7, 1 - 1e-7);
    const remaining = span - context.fixedSupport[0];
    const rightFraction = clamp(context.fixedSupport[1] / remaining, 1e-7, 1 - 1e-7);
    result[5] = logit(leftFraction);
    result[6] = logit(rightFraction);
    result[7] = inverseSoftplus(Math.max(
      1e-12,
      context.fixedSupport[2] - context.minimumSupportSharpness,
    ));
    result[8] = inverseSoftplus(Math.max(
      1e-12,
      context.fixedSupport[3] - context.minimumSupportSharpness,
    ));
  }
  return result;
}

function compactEnvelopeLogValueStructural(
  action: number,
  structural: ReturnType<typeof structuralParametersFromRaw>,
  context: FitContext,
): number {
  return logCompactStep(
    (action - context.latentLower) / structural.leftSupportWidth,
    structural.leftSupportSharpness,
  ) + logCompactStep(
    (context.latentUpper - action) / structural.rightSupportWidth,
    structural.rightSupportSharpness,
  );
}

function compactEnvelopeLogValue(
  action: number,
  parameters: ConditionalFourSegmentParameters,
): number {
  return logCompactStep(
    (action - parameters.latentLower) / parameters.leftSupportWidth,
    parameters.leftSupportSharpness,
  ) + logCompactStep(
    (parameters.latentUpper - action) / parameters.rightSupportWidth,
    parameters.rightSupportSharpness,
  );
}

function logCompactStep(t: number, sharpness: number): number {
  if (t <= 0) return Number.NEGATIVE_INFINITY;
  if (t >= 1) return 0;
  const shape = 1 / (1 - t) - 1 / t;
  return -softplus(-sharpness * shape);
}

function updateInverseHessian(
  inverseHessian: Float64Array,
  parameterDelta: Float64Array,
  gradientDelta: Float64Array,
): Float64Array {
  const size = parameterDelta.length;
  const curvature = dot(gradientDelta, parameterDelta);
  const curvatureScale = Math.sqrt(dot(parameterDelta, parameterDelta)
    * dot(gradientDelta, gradientDelta));
  if (!(curvature > 1e-10 * Math.max(1, curvatureScale))) return identityMatrix(size);
  const hessianGradient = matrixVectorProduct(inverseHessian, gradientDelta);
  const gradientHessianGradient = dot(gradientDelta, hessianGradient);
  const result = inverseHessian.slice();
  const firstScale = (curvature + gradientHessianGradient) / (curvature * curvature);
  for (let row = 0; row < size; row += 1) {
    for (let column = 0; column < size; column += 1) {
      const offset = row * size + column;
      result[offset] += firstScale * parameterDelta[row]! * parameterDelta[column]!
        - (hessianGradient[row]! * parameterDelta[column]!
          + parameterDelta[row]! * hessianGradient[column]!) / curvature;
    }
  }
  return result;
}

function identityMatrix(size: number): Float64Array {
  const result = new Float64Array(size * size);
  for (let index = 0; index < size; index += 1) result[index * size + index] = 1;
  return result;
}

function matrixVectorProduct(matrix: Float64Array, vector: Float64Array): Float64Array {
  const result = new Float64Array(vector.length);
  for (let row = 0; row < vector.length; row += 1) {
    for (let column = 0; column < vector.length; column += 1) {
      result[row] += matrix[row * vector.length + column]! * vector[column]!;
    }
  }
  return result;
}

function medianActionStep(actions: Float64Array): number {
  const differences = Array.from(
    { length: actions.length - 1 },
    (_, index) => actions[index + 1]! - actions[index]!,
  ).sort((left, right) => left - right);
  return differences[Math.floor(differences.length / 2)]!;
}

function scaledCurrentExposure(current: number, lower: number, upper: number): number {
  return clamp((2 * current - lower - upper) / (upper - lower), -1, 1);
}

function scaledSoftplus(value: number, kappa: number): number {
  return softplus(kappa * value) / kappa;
}

function softplus(value: number): number {
  if (value > 35) return value;
  if (value < -35) return Math.exp(value);
  return Math.log1p(Math.exp(value));
}

function inverseSoftplus(value: number): number {
  if (value > 35) return value;
  return Math.log(Math.expm1(Math.max(1e-12, value)));
}

function sigmoid(value: number): number {
  if (value >= 0) {
    const inverse = Math.exp(-value);
    return 1 / (1 + inverse);
  }
  const exponential = Math.exp(value);
  return exponential / (1 + exponential);
}

function logit(value: number): number {
  return Math.log(value / (1 - value));
}

function subtract(left: Float64Array, right: Float64Array): Float64Array {
  return Float64Array.from(left, (value, index) => value - right[index]!);
}

function dot(left: Float64Array, right: Float64Array): number {
  return left.reduce((sum, value, index) => sum + value * right[index]!, 0);
}

function maximumAbsolute(values: Float64Array): number {
  return values.reduce((maximum, value) => Math.max(maximum, Math.abs(value)), 0);
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}
