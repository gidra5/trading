import { fitConditionalFourSegmentScores } from "./parameter-fit.js";

export interface ConditionalFourSegmentModelOptions {
  latentLower: number;
  latentUpper: number;
  visibleLower?: number;
  visibleUpper?: number;
  initialC1?: number;
  initialC2?: number;
  initialLeftSupportWidth?: number;
  initialRightSupportWidth?: number;
  initialKappaC1?: number;
  initialKappaX?: number;
  initialKappaC2?: number;
  leftSupportSharpness?: number;
  rightSupportSharpness?: number;
  minimumKappa?: number;
  minimumSupportSharpness?: number;
  maxIterations?: number;
  restartCount?: number;
  sampleStates?: number;
  sampleActions?: number;
  tolerance?: number;
}

export interface ConditionalFourSegmentParameters {
  latentLower: number;
  latentUpper: number;
  visibleLower: number;
  visibleUpper: number;
  c1: number;
  c2: number;
  leftSupportWidth: number;
  rightSupportWidth: number;
  leftSupportSharpness: number;
  rightSupportSharpness: number;
  baseSlope: readonly [number, number];
  betaC1: readonly [number, number];
  betaX: readonly [number, number];
  betaC2: readonly [number, number];
  kappaC1: number;
  kappaX: number;
  kappaC2: number;
}

export interface ConditionalFourSegmentSliceParameters {
  xi: number;
  baseSlope: number;
  betaC1: number;
  betaX: number;
  betaC2: number;
  kappaC1: number;
  kappaX: number;
  kappaC2: number;
  segmentSlopes: readonly [number, number, number, number];
}

export type ConditionalFourSegmentFitTermination =
  | "gradient"
  | "relative-loss"
  | "line-search"
  | "iteration-limit";

export interface ConditionalFourSegmentPolicyFit {
  parameters: ConditionalFourSegmentParameters;
  crossEntropy: number;
  klDivergence: number;
  meanSquaredError: number;
  iterations: number;
  restarts: number;
  termination: ConditionalFourSegmentFitTermination;
  converged: boolean;
}

interface FixedParameters {
  latentLower: number;
  latentUpper: number;
  visibleLower: number;
  visibleUpper: number;
  minimumKappa: number;
  minimumSupportSharpness: number;
  slopeScale: number;
}

interface ObjectiveValue {
  loss: number;
  gradient: Float64Array;
}

interface OptimizationResult extends ObjectiveValue {
  raw: Float64Array;
  iterations: number;
  termination: ConditionalFourSegmentFitTermination;
  converged: boolean;
}

const PARAMETER_COUNT = 17;
const C1_RAW = 0;
const C2_RAW = 1;
const BASE_0 = 2;
const BASE_1 = 3;
const BETA_C1_0 = 4;
const BETA_C1_1 = 5;
const BETA_X_0 = 6;
const BETA_X_1 = 7;
const BETA_C2_0 = 8;
const BETA_C2_1 = 9;
const KAPPA_C1_RAW = 10;
const KAPPA_X_RAW = 11;
const KAPPA_C2_RAW = 12;
const LEFT_WIDTH_RAW = 13;
const RIGHT_WIDTH_RAW = 14;
const LEFT_SHARPNESS_RAW = 15;
const RIGHT_SHARPNESS_RAW = 16;
const LINEAR_PARAMETER_INDICES = [
  BASE_0,
  BASE_1,
  BETA_C1_0,
  BETA_C1_1,
  BETA_X_0,
  BETA_X_1,
  BETA_C2_0,
  BETA_C2_1,
] as const;

/** Fits the documented order-independent conditional four-segment surface. */
export function fitConditionalFourSegmentPolicy(
  actionGrid: ArrayLike<number>,
  targetProbabilities: ArrayLike<number>,
  currentExposures: ArrayLike<number>,
  options: ConditionalFourSegmentModelOptions,
): ConditionalFourSegmentPolicyFit {
  validateInputs(actionGrid, targetProbabilities, currentExposures, options);
  const visibleLower = options.visibleLower ?? actionGrid[0]!;
  const visibleUpper = options.visibleUpper ?? actionGrid[actionGrid.length - 1]!;
  const visibleSpan = visibleUpper - visibleLower;
  const latentSpan = options.latentUpper - options.latentLower;
  const fixed: FixedParameters = {
    latentLower: options.latentLower,
    latentUpper: options.latentUpper,
    visibleLower,
    visibleUpper,
    minimumKappa: options.minimumKappa ?? 4.394 / latentSpan,
    minimumSupportSharpness: options.minimumSupportSharpness ?? 1e-5,
    slopeScale: 1 / visibleSpan,
  };
  validateFixed(fixed);

  const stateIndices = sampledIndices(
    currentExposures.length,
    Math.max(3, Math.floor(options.sampleStates ?? 31)),
  );
  const actionIndices = sampledIndices(
    actionGrid.length,
    Math.max(5, Math.floor(options.sampleActions ?? 41)),
  );
  const sampledActions = Float64Array.from(actionIndices, (index) => actionGrid[index]!);
  const sampledStates = Float64Array.from(stateIndices, (index) => currentExposures[index]!);
  const sampledTargets = normalizedSampledTargets(
    actionGrid.length,
    targetProbabilities,
    stateIndices,
    actionIndices,
  );
  const objective = (raw: Float64Array) => conditionalCrossEntropyWithGradient(
    sampledActions,
    sampledTargets,
    sampledStates,
    raw,
    fixed,
  );
  const empiricalValues = empiricalInitialValues(
    sampledActions,
    sampledTargets,
    sampledStates,
    fixed,
    options,
  );
  const projectedInitial = variableProjectionInitialRaw(
    sampledActions,
    sampledTargets,
    sampledStates,
    fixed,
    options,
  );
  const initialValues = [projectedInitial, ...empiricalValues];
  const restartCount = Math.max(1, Math.min(
    initialValues.length,
    Math.floor(options.restartCount ?? initialValues.length),
  ));
  const maximumIterations = Math.max(1, Math.floor(options.maxIterations ?? 100));
  const tolerance = Math.max(Number.EPSILON, options.tolerance ?? 1e-6);
  let best: OptimizationResult | undefined;
  for (let restart = 0; restart < restartCount; restart += 1) {
    let candidate: OptimizationResult;
    if (maximumIterations === 1) {
      candidate = optimizeBfgs(initialValues[restart]!, objective, 1, tolerance);
    } else {
      const warmupIterations = Math.min(
        30,
        maximumIterations - 1,
        Math.max(1, Math.floor(maximumIterations / 4)),
      );
      const warmup = optimizeBfgs(
        initialValues[restart]!,
        maskedObjective(objective, LINEAR_PARAMETER_INDICES),
        warmupIterations,
        tolerance,
      );
      candidate = optimizeBfgs(
        warmup.raw,
        objective,
        maximumIterations - warmup.iterations,
        tolerance,
      );
      candidate.iterations += warmup.iterations;
    }
    if (!best || candidate.loss < best.loss) best = candidate;
  }
  if (!best) throw new Error("Conditional four-segment optimizer did not run.");
  const parameters = parametersFromRaw(best.raw, fixed);
  return {
    parameters,
    ...fitDiagnostics(actionGrid, targetProbabilities, currentExposures, parameters),
    iterations: best.iterations,
    restarts: restartCount,
    termination: best.termination,
    converged: best.converged,
  };
}

/**
 * Fits log probabilities as score values first. Conditional normalization is
 * absorbed by the projected slice offsets, so this provides the documented
 * direct-value estimate before the probability objective jointly refines it.
 */
function variableProjectionInitialRaw(
  actions: Float64Array,
  targets: Float64Array,
  states: Float64Array,
  fixed: FixedParameters,
  options: ConditionalFourSegmentModelOptions,
): Float64Array {
  const scores = new Float64Array(targets.length);
  for (let row = 0; row < states.length; row += 1) {
    scores.set(
      stableLogProbabilities(targets.subarray(row * actions.length, (row + 1) * actions.length)),
      row * actions.length,
    );
  }
  const supportWidths = initialSupportWidths(fixed, options);
  const fitsLeftBoundary = actions[0]! < fixed.latentLower + supportWidths[0];
  const fitsRightBoundary = actions.at(-1)! > fixed.latentUpper - supportWidths[1];
  const projected = fitConditionalFourSegmentScores(actions, scores, states, {
    ...options,
    latentLower: fixed.latentLower,
    latentUpper: fixed.latentUpper,
    visibleLower: fixed.visibleLower,
    visibleUpper: fixed.visibleUpper,
    initialLeftSupportWidth: supportWidths[0],
    initialRightSupportWidth: supportWidths[1],
    minimumKappa: fixed.minimumKappa,
    minimumSupportSharpness: fixed.minimumSupportSharpness,
    fitSupport: fitsLeftBoundary || fitsRightBoundary,
    ridge: 1e-2,
    maxIterations: Math.min(40, Math.max(12, Math.floor((options.maxIterations ?? 100) / 2))),
    restartCount: Math.min(3, Math.max(1, Math.floor(options.restartCount ?? 3))),
    tolerance: Math.min(options.tolerance ?? 1e-6, 1e-7),
  });
  return rawFromParameters(projected.parameters, fixed);
}

function maskedObjective(
  objective: (raw: Float64Array) => ObjectiveValue,
  activeParameters: readonly number[],
): (raw: Float64Array) => ObjectiveValue {
  const active = new Set(activeParameters);
  return (raw) => {
    const value = objective(raw);
    for (let parameter = 0; parameter < value.gradient.length; parameter += 1) {
      if (!active.has(parameter)) value.gradient[parameter] = 0;
    }
    return value;
  };
}

/** Evaluates one hard-truncated visible row of the fitted model. */
export function conditionalFourSegmentExposureProbabilities(
  actionGrid: ArrayLike<number>,
  currentExposure: number,
  parameters: ConditionalFourSegmentParameters,
  result: Float64Array<ArrayBufferLike> = new Float64Array(actionGrid.length),
): Float64Array<ArrayBufferLike> {
  if (result.length !== actionGrid.length) {
    throw new Error("Conditional four-segment result does not match its action grid.");
  }
  const slice = conditionalFourSegmentParametersAt(currentExposure, parameters);
  let maximum = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < actionGrid.length; index += 1) {
    const action = actionGrid[index]!;
    const value = action >= parameters.visibleLower && action <= parameters.visibleUpper
      ? logKernel(action, currentExposure, slice, parameters)
      : Number.NEGATIVE_INFINITY;
    result[index] = value;
    maximum = Math.max(maximum, value);
  }
  let total = 0;
  for (let index = 0; index < result.length; index += 1) {
    const weight = Number.isFinite(result[index]!) ? Math.exp(result[index]! - maximum) : 0;
    result[index] = weight;
    total += weight;
  }
  if (!(total > 0)) throw new Error("Conditional four-segment row has no visible support.");
  for (let index = 0; index < result.length; index += 1) result[index] /= total;
  return result;
}

export function conditionalFourSegmentPolicyMatrix(
  actionGrid: ArrayLike<number>,
  currentExposures: ArrayLike<number>,
  parameters: ConditionalFourSegmentParameters,
): Float64Array {
  const result = new Float64Array(actionGrid.length * currentExposures.length);
  for (let row = 0; row < currentExposures.length; row += 1) {
    conditionalFourSegmentExposureProbabilities(
      actionGrid,
      currentExposures[row]!,
      parameters,
      result.subarray(row * actionGrid.length, (row + 1) * actionGrid.length),
    );
  }
  return result;
}

export function conditionalFourSegmentParametersAt(
  currentExposure: number,
  parameters: ConditionalFourSegmentParameters,
): ConditionalFourSegmentSliceParameters {
  const xi = scaledCurrentExposure(currentExposure, parameters.latentLower, parameters.latentUpper);
  const baseSlope = linear(parameters.baseSlope, xi);
  const betaC1 = linear(parameters.betaC1, xi);
  const betaX = linear(parameters.betaX, xi);
  const betaC2 = linear(parameters.betaC2, xi);
  const orderedChanges = [
    { location: parameters.c1, change: betaC1 },
    { location: currentExposure, change: betaX },
    { location: parameters.c2, change: betaC2 },
  ].sort((left, right) => left.location - right.location);
  const slopes = [baseSlope];
  for (const transition of orderedChanges) slopes.push(slopes.at(-1)! + transition.change);
  return {
    xi,
    baseSlope,
    betaC1,
    betaX,
    betaC2,
    kappaC1: parameters.kappaC1,
    kappaX: parameters.kappaX,
    kappaC2: parameters.kappaC2,
    segmentSlopes: slopes as [number, number, number, number],
  };
}

/** Effective action derivative of the normalized row's log density. */
export function conditionalFourSegmentLogSlope(
  action: number,
  currentExposure: number,
  parameters: ConditionalFourSegmentParameters,
): number {
  if (!(action > parameters.latentLower && action < parameters.latentUpper)) {
    return Number.NaN;
  }
  const slice = conditionalFourSegmentParametersAt(currentExposure, parameters);
  return slice.baseSlope
    + slice.betaC1 * sigmoid(slice.kappaC1 * (action - parameters.c1))
    + slice.betaX * sigmoid(slice.kappaX * (action - currentExposure))
    + slice.betaC2 * sigmoid(slice.kappaC2 * (action - parameters.c2))
    + compactEnvelopeLogSlope(action, parameters);
}

function logKernel(
  action: number,
  current: number,
  slice: ConditionalFourSegmentSliceParameters,
  parameters: ConditionalFourSegmentParameters,
): number {
  if (!(action > parameters.latentLower && action < parameters.latentUpper)) {
    return Number.NEGATIVE_INFINITY;
  }
  return slice.baseSlope * (action - parameters.latentLower)
    + slice.betaC1 * scaledSoftplus(action - parameters.c1, slice.kappaC1)
    + slice.betaX * scaledSoftplus(action - current, slice.kappaX)
    + slice.betaC2 * scaledSoftplus(action - parameters.c2, slice.kappaC2)
    + compactEnvelopeLogValue(action, parameters);
}

function conditionalCrossEntropyWithGradient(
  actions: Float64Array,
  targets: Float64Array,
  states: Float64Array,
  raw: Float64Array,
  fixed: FixedParameters,
): ObjectiveValue {
  const mapped = mappedRawParameters(raw, fixed);
  const logits = new Float64Array(actions.length);
  const derivatives = new Float64Array(actions.length * PARAMETER_COUNT);
  const gradient = new Float64Array(PARAMETER_COUNT);
  let loss = 0;
  for (let row = 0; row < states.length; row += 1) {
    const current = states[row]!;
    const xi = scaledCurrentExposure(current, fixed.latentLower, fixed.latentUpper);
    const baseSlope = mapped.base0 + mapped.base1 * xi;
    const betaC1 = mapped.betaC10 + mapped.betaC11 * xi;
    const betaX = mapped.betaX0 + mapped.betaX1 * xi;
    const betaC2 = mapped.betaC20 + mapped.betaC21 * xi;
    let maximum = Number.NEGATIVE_INFINITY;
    for (let actionIndex = 0; actionIndex < actions.length; actionIndex += 1) {
      const action = actions[actionIndex]!;
      const c1Offset = action - mapped.c1;
      const xOffset = action - current;
      const c2Offset = action - mapped.c2;
      const spC1 = scaledSoftplus(c1Offset, mapped.kappaC1);
      const spX = scaledSoftplus(xOffset, mapped.kappaX);
      const spC2 = scaledSoftplus(c2Offset, mapped.kappaC2);
      const gate = compactEnvelopeLogValueMapped(action, mapped, fixed);
      const logit = baseSlope * (action - fixed.latentLower)
        + betaC1 * spC1 + betaX * spX + betaC2 * spC2 + gate.value;
      logits[actionIndex] = logit;
      maximum = Math.max(maximum, logit);
      const derivativeOffset = actionIndex * PARAMETER_COUNT;
      const sigmoidC1 = sigmoid(mapped.kappaC1 * c1Offset);
      const sigmoidC2 = sigmoid(mapped.kappaC2 * c2Offset);
      derivatives[derivativeOffset + C1_RAW] = -betaC1 * sigmoidC1 * mapped.c1Derivative
        - betaC2 * sigmoidC2 * mapped.c2C1Derivative;
      derivatives[derivativeOffset + C2_RAW] = -betaC2 * sigmoidC2 * mapped.c2Derivative;
      derivatives[derivativeOffset + BASE_0] = fixed.slopeScale * (action - fixed.latentLower);
      derivatives[derivativeOffset + BASE_1] = fixed.slopeScale * xi
        * (action - fixed.latentLower);
      derivatives[derivativeOffset + BETA_C1_0] = fixed.slopeScale * spC1;
      derivatives[derivativeOffset + BETA_C1_1] = fixed.slopeScale * xi * spC1;
      derivatives[derivativeOffset + BETA_X_0] = fixed.slopeScale * spX;
      derivatives[derivativeOffset + BETA_X_1] = fixed.slopeScale * xi * spX;
      derivatives[derivativeOffset + BETA_C2_0] = fixed.slopeScale * spC2;
      derivatives[derivativeOffset + BETA_C2_1] = fixed.slopeScale * xi * spC2;
      derivatives[derivativeOffset + KAPPA_C1_RAW] = betaC1
        * scaledSoftplusKappaDerivative(c1Offset, mapped.kappaC1)
        * mapped.kappaC1Derivative;
      derivatives[derivativeOffset + KAPPA_X_RAW] = betaX
        * scaledSoftplusKappaDerivative(xOffset, mapped.kappaX)
        * mapped.kappaXDerivative;
      derivatives[derivativeOffset + KAPPA_C2_RAW] = betaC2
        * scaledSoftplusKappaDerivative(c2Offset, mapped.kappaC2)
        * mapped.kappaC2Derivative;
      derivatives[derivativeOffset + LEFT_WIDTH_RAW] = gate.leftWidthDerivative
        * mapped.leftWidthDerivative
        + gate.rightWidthDerivative * mapped.rightLeftWidthDerivative;
      derivatives[derivativeOffset + RIGHT_WIDTH_RAW] = gate.rightWidthDerivative
        * mapped.rightWidthDerivative;
      derivatives[derivativeOffset + LEFT_SHARPNESS_RAW] = gate.leftDerivative;
      derivatives[derivativeOffset + RIGHT_SHARPNESS_RAW] = gate.rightDerivative;
    }
    let normalizer = 0;
    for (let actionIndex = 0; actionIndex < actions.length; actionIndex += 1) {
      normalizer += Math.exp(logits[actionIndex]! - maximum);
    }
    const logNormalizer = maximum + Math.log(normalizer);
    for (let actionIndex = 0; actionIndex < actions.length; actionIndex += 1) {
      const target = targets[row * actions.length + actionIndex]!;
      const model = Math.exp(logits[actionIndex]! - logNormalizer);
      if (target > 0) loss -= target * (logits[actionIndex]! - logNormalizer) / states.length;
      const residual = (model - target) / states.length;
      const derivativeOffset = actionIndex * PARAMETER_COUNT;
      for (let parameter = 0; parameter < PARAMETER_COUNT; parameter += 1) {
        gradient[parameter] += residual * derivatives[derivativeOffset + parameter]!;
      }
    }
  }
  const slopeMagnitudePenalty = 1e-9;
  const unpenalizedSlopeMagnitude = 100;
  for (const parameter of LINEAR_PARAMETER_INDICES) {
    const magnitude = Math.abs(raw[parameter]!);
    const excess = Math.max(0, magnitude - unpenalizedSlopeMagnitude);
    loss += slopeMagnitudePenalty * excess ** 2;
    gradient[parameter] += 2 * slopeMagnitudePenalty * excess
      * Math.sign(raw[parameter]!);
  }
  const variationPenalty = 1e-8;
  for (const parameter of [BASE_1, BETA_C1_1, BETA_X_1, BETA_C2_1]) {
    loss += variationPenalty * raw[parameter]! ** 2;
    gradient[parameter] += 2 * variationPenalty * raw[parameter]!;
  }
  const visibleSpan = fixed.visibleUpper - fixed.visibleLower;
  const leftPlateauLimit = fixed.visibleLower - fixed.latentLower;
  const rightPlateauLimit = fixed.latentUpper - fixed.visibleUpper;
  const leftOverlap = Math.max(0, mapped.leftSupportWidth - leftPlateauLimit);
  const rightOverlap = Math.max(0, mapped.rightSupportWidth - rightPlateauLimit);
  const overlapPenalty = 5e-3 / (visibleSpan * visibleSpan);
  loss += overlapPenalty * (leftOverlap * leftOverlap + rightOverlap * rightOverlap);
  const leftWidthPenaltyDerivative = 2 * overlapPenalty * leftOverlap;
  const rightWidthPenaltyDerivative = 2 * overlapPenalty * rightOverlap;
  gradient[LEFT_WIDTH_RAW] += leftWidthPenaltyDerivative * mapped.leftWidthDerivative
    + rightWidthPenaltyDerivative * mapped.rightLeftWidthDerivative;
  gradient[RIGHT_WIDTH_RAW] += rightWidthPenaltyDerivative * mapped.rightWidthDerivative;
  return { loss, gradient };
}

function optimizeBfgs(
  initial: Float64Array,
  objective: (raw: Float64Array) => ObjectiveValue,
  maximumIterations: number,
  tolerance: number,
): OptimizationResult {
  let raw: Float64Array<ArrayBufferLike> = initial.slice();
  let value = objective(raw);
  let inverseHessian = identityMatrix(PARAMETER_COUNT);
  let stableIterations = 0;
  for (let iteration = 0; iteration < maximumIterations; iteration += 1) {
    if (maximumAbsolute(value.gradient) <= tolerance) {
      return { ...value, raw, iterations: iteration, termination: "gradient", converged: true };
    }
    let direction = matrixVectorProduct(inverseHessian, value.gradient);
    for (let index = 0; index < direction.length; index += 1) direction[index] *= -1;
    let directionalDerivative = dot(value.gradient, direction);
    if (!(directionalDerivative < 0) || !Number.isFinite(directionalDerivative)) {
      inverseHessian = identityMatrix(PARAMETER_COUNT);
      direction = Float64Array.from(value.gradient, (gradient) => -gradient);
      directionalDerivative = -dot(value.gradient, value.gradient);
    }
    const maximumDirection = maximumAbsolute(direction);
    if (maximumDirection > 4) {
      const scale = 4 / maximumDirection;
      for (let index = 0; index < direction.length; index += 1) direction[index] *= scale;
      directionalDerivative *= scale;
    }
    let step = 1;
    let nextRaw: Float64Array | undefined;
    let nextValue: ObjectiveValue | undefined;
    for (let search = 0; search < 24; search += 1) {
      const candidate = Float64Array.from(raw, (parameter, index) =>
        parameter + step * direction[index]!);
      boundUnconstrainedParameters(candidate);
      const candidateValue = objective(candidate);
      if (Number.isFinite(candidateValue.loss)
        && candidateValue.loss <= value.loss + 1e-4 * step * directionalDerivative) {
        nextRaw = candidate;
        nextValue = candidateValue;
        break;
      }
      step *= 0.5;
    }
    if (!nextRaw || !nextValue) {
      return {
        ...value,
        raw,
        iterations: iteration,
        termination: "line-search",
        converged: false,
      };
    }
    const relativeImprovement = (value.loss - nextValue.loss) / Math.max(1, Math.abs(value.loss));
    stableIterations = relativeImprovement <= tolerance ? stableIterations + 1 : 0;
    const parameterDelta = subtract(nextRaw, raw);
    const gradientDelta = subtract(nextValue.gradient, value.gradient);
    inverseHessian = updateInverseHessian(inverseHessian, parameterDelta, gradientDelta);
    raw = nextRaw;
    value = nextValue;
    if (stableIterations >= 4
      && maximumAbsolute(value.gradient) <= Math.max(tolerance * 10, 1e-5)) {
      return {
        ...value,
        raw,
        iterations: iteration + 1,
        termination: "relative-loss",
        converged: true,
      };
    }
  }
  return {
    ...value,
    raw,
    iterations: maximumIterations,
    termination: "iteration-limit",
    converged: false,
  };
}

function empiricalInitialValues(
  actions: Float64Array,
  targets: Float64Array,
  states: Float64Array,
  fixed: FixedParameters,
  options: ConditionalFourSegmentModelOptions,
): Float64Array[] {
  const visibleSpan = fixed.visibleUpper - fixed.visibleLower;
  const outsideC1 = clamp(
    options.initialC1 ?? (fixed.latentLower + fixed.visibleLower) / 2,
    fixed.latentLower + 1e-6,
    fixed.latentUpper - 2e-6,
  );
  const outsideC2 = clamp(
    options.initialC2 ?? (fixed.visibleUpper + fixed.latentUpper) / 2,
    outsideC1 + 1e-6,
    fixed.latentUpper - 1e-6,
  );
  const detected = empiricalFixedBreakpointLocations(actions, targets, states);
  const locations: Array<readonly [number, number, boolean]> = [
    ...(detected ? [[detected[0], detected[1], true] as const] : []),
    [outsideC1, outsideC2, false],
    [fixed.visibleLower + visibleSpan / 3, fixed.visibleUpper - visibleSpan / 3, true],
    [
      Math.max(fixed.latentLower + 1e-6, fixed.visibleLower - visibleSpan / 8),
      Math.min(fixed.latentUpper - 1e-6, fixed.visibleUpper + visibleSpan / 8),
      true,
    ],
  ];
  const baselineWidths = initialSupportWidths(fixed, options);
  const overlappingWidths = overlappingSupportWidths(fixed, baselineWidths);
  return locations.map(([c1, c2, seedTransitions], index) => empiricalInitialRaw(
    actions,
    targets,
    states,
    c1,
    c2,
    seedTransitions,
    fixed,
    options,
    index === 0 ? baselineWidths : overlappingWidths,
  ));
}

function empiricalFixedBreakpointLocations(
  actions: Float64Array,
  targets: Float64Array,
  states: Float64Array,
): readonly [number, number] | undefined {
  const actionSpan = actions.at(-1)! - actions[0]!;
  const typicalStep = actionSpan / (actions.length - 1);
  const movingExclusion = Math.max(typicalStep * 3, actionSpan / 12);
  const scores = new Float64Array(actions.length);
  for (let actionIndex = 1; actionIndex < actions.length - 1; actionIndex += 1) {
    const location = actions[actionIndex]!;
    let signedChange = 0;
    let absoluteChange = 0;
    let contributingRows = 0;
    for (let row = 0; row < states.length; row += 1) {
      if (Math.abs(states[row]! - location) <= movingExclusion) continue;
      const probabilities = targets.subarray(row * actions.length, (row + 1) * actions.length);
      const maximum = probabilities.reduce((best, value) => Math.max(best, value), 0);
      if (probabilities[actionIndex]! < maximum * 1e-6) continue;
      const logProbabilities = stableLogProbabilities(probabilities);
      const leftSlope = (logProbabilities[actionIndex]! - logProbabilities[actionIndex - 1]!)
        / (actions[actionIndex]! - actions[actionIndex - 1]!);
      const rightSlope = (logProbabilities[actionIndex + 1]! - logProbabilities[actionIndex]!)
        / (actions[actionIndex + 1]! - actions[actionIndex]!);
      const change = rightSlope - leftSlope;
      signedChange += change;
      absoluteChange += Math.abs(change);
      contributingRows += 1;
    }
    if (contributingRows > 0) {
      const meanChange = signedChange / contributingRows;
      const meanAbsoluteChange = absoluteChange / contributingRows;
      scores[actionIndex] = Math.abs(meanChange) + meanAbsoluteChange * 0.15;
    }
  }
  const smoothed = new Float64Array(scores.length);
  for (let index = 1; index < scores.length - 1; index += 1) {
    let weightedScore = 0;
    let totalWeight = 0;
    for (let offset = -2; offset <= 2; offset += 1) {
      const neighbor = index + offset;
      if (neighbor <= 0 || neighbor >= scores.length - 1) continue;
      const weight = 3 - Math.abs(offset);
      weightedScore += scores[neighbor]! * weight;
      totalWeight += weight;
    }
    smoothed[index] = totalWeight > 0 ? weightedScore / totalWeight : 0;
  }
  const minimumSeparation = Math.max(typicalStep * 3, actionSpan / 10);
  const candidates = Array.from(
    { length: actions.length - 2 },
    (_, offset) => offset + 1,
  ).filter((index) => smoothed[index]! > 1e-9)
    .sort((left, right) => smoothed[right]! - smoothed[left]!);
  const selected: number[] = [];
  for (const index of candidates) {
    const location = actions[index]!;
    if (selected.every((selectedLocation) =>
      Math.abs(selectedLocation - location) >= minimumSeparation)) {
      selected.push(location);
      if (selected.length === 2) break;
    }
  }
  return selected.length === 2
    ? selected.sort((left, right) => left - right) as [number, number]
    : undefined;
}

function empiricalInitialRaw(
  actions: Float64Array,
  targets: Float64Array,
  states: Float64Array,
  c1: number,
  c2: number,
  seedTransitions: boolean,
  fixed: FixedParameters,
  options: ConditionalFourSegmentModelOptions,
  supportWidths: readonly [number, number],
): Float64Array {
  const xi = Float64Array.from(states, (state) =>
    scaledCurrentExposure(state, fixed.latentLower, fixed.latentUpper));
  const rowSlopes = new Float64Array(states.length);
  const c1Changes = new Float64Array(states.length);
  const xChanges = new Float64Array(states.length);
  const c2Changes = new Float64Array(states.length);
  for (let row = 0; row < states.length; row += 1) {
    const probabilities = targets.subarray(row * actions.length, (row + 1) * actions.length);
    const logProbabilities = stableLogProbabilities(probabilities);
    rowSlopes[row] = empiricalLogSlope(actions, logProbabilities, probabilities);
    if (seedTransitions) {
      c1Changes[row] = empiricalSlopeChange(actions, logProbabilities, c1);
      xChanges[row] = empiricalSlopeChange(actions, logProbabilities, states[row]!);
      c2Changes[row] = empiricalSlopeChange(actions, logProbabilities, c2);
    }
  }
  const base = linearRegression(xi, rowSlopes);
  const betaC1 = seedTransitions ? linearRegression(xi, c1Changes) : [0, 0] as const;
  const betaX = seedTransitions ? linearRegression(xi, xChanges) : [0, 0] as const;
  const betaC2 = seedTransitions ? linearRegression(xi, c2Changes) : [0, 0] as const;
  const initialKappa = 4.394 / Math.max(1e-6, (fixed.visibleUpper - fixed.visibleLower) / 4);
  const raw = new Float64Array(PARAMETER_COUNT);
  [raw[C1_RAW], raw[C2_RAW]] = rawBreakpoints(c1, c2, fixed);
  setSlopeCoefficients(raw, BASE_0, base, fixed.slopeScale);
  setSlopeCoefficients(raw, BETA_C1_0, betaC1, fixed.slopeScale);
  setSlopeCoefficients(raw, BETA_X_0, betaX, fixed.slopeScale);
  setSlopeCoefficients(raw, BETA_C2_0, betaC2, fixed.slopeScale);
  raw[KAPPA_C1_RAW] = inverseSoftplus(
    Math.max(fixed.minimumKappa * 2, options.initialKappaC1 ?? initialKappa)
      - fixed.minimumKappa,
  );
  raw[KAPPA_X_RAW] = inverseSoftplus(
    Math.max(fixed.minimumKappa * 2, options.initialKappaX ?? initialKappa)
      - fixed.minimumKappa,
  );
  raw[KAPPA_C2_RAW] = inverseSoftplus(
    Math.max(fixed.minimumKappa * 2, options.initialKappaC2 ?? initialKappa)
      - fixed.minimumKappa,
  );
  [raw[LEFT_WIDTH_RAW], raw[RIGHT_WIDTH_RAW]] = rawSupportWidths(
    supportWidths[0],
    supportWidths[1],
    fixed,
  );
  raw[LEFT_SHARPNESS_RAW] = inverseSoftplus(
    Math.max(fixed.minimumSupportSharpness * 2, options.leftSupportSharpness ?? 1)
      - fixed.minimumSupportSharpness,
  );
  raw[RIGHT_SHARPNESS_RAW] = inverseSoftplus(
    Math.max(fixed.minimumSupportSharpness * 2, options.rightSupportSharpness ?? 1)
      - fixed.minimumSupportSharpness,
  );
  boundUnconstrainedParameters(raw);
  return raw;
}

function mappedRawParameters(raw: Float64Array, fixed: FixedParameters) {
  const latentSpan = fixed.latentUpper - fixed.latentLower;
  const firstFraction = sigmoid(raw[C1_RAW]!);
  const c1 = fixed.latentLower + (fixed.latentUpper - fixed.latentLower) * firstFraction;
  const secondFraction = sigmoid(raw[C2_RAW]!);
  const c2 = c1 + (fixed.latentUpper - c1) * secondFraction;
  const c1Derivative = (fixed.latentUpper - fixed.latentLower)
    * firstFraction * (1 - firstFraction);
  const c2C1Derivative = c1Derivative * (1 - secondFraction);
  const c2Derivative = (fixed.latentUpper - c1) * secondFraction * (1 - secondFraction);
  const kappaC1 = fixed.minimumKappa + softplus(raw[KAPPA_C1_RAW]!);
  const kappaX = fixed.minimumKappa + softplus(raw[KAPPA_X_RAW]!);
  const kappaC2 = fixed.minimumKappa + softplus(raw[KAPPA_C2_RAW]!);
  const leftWidthFraction = sigmoid(raw[LEFT_WIDTH_RAW]!);
  const leftSupportWidth = latentSpan * leftWidthFraction;
  const rightWidthFraction = sigmoid(raw[RIGHT_WIDTH_RAW]!);
  const remainingWidth = latentSpan - leftSupportWidth;
  const rightSupportWidth = remainingWidth * rightWidthFraction;
  const leftWidthDerivative = latentSpan * leftWidthFraction * (1 - leftWidthFraction);
  const rightLeftWidthDerivative = -leftWidthDerivative * rightWidthFraction;
  const rightWidthDerivative = remainingWidth * rightWidthFraction * (1 - rightWidthFraction);
  const leftSupportSharpness = fixed.minimumSupportSharpness
    + softplus(raw[LEFT_SHARPNESS_RAW]!);
  const rightSupportSharpness = fixed.minimumSupportSharpness
    + softplus(raw[RIGHT_SHARPNESS_RAW]!);
  return {
    c1,
    c2,
    c1Derivative,
    c2C1Derivative,
    c2Derivative,
    base0: raw[BASE_0]! * fixed.slopeScale,
    base1: raw[BASE_1]! * fixed.slopeScale,
    betaC10: raw[BETA_C1_0]! * fixed.slopeScale,
    betaC11: raw[BETA_C1_1]! * fixed.slopeScale,
    betaX0: raw[BETA_X_0]! * fixed.slopeScale,
    betaX1: raw[BETA_X_1]! * fixed.slopeScale,
    betaC20: raw[BETA_C2_0]! * fixed.slopeScale,
    betaC21: raw[BETA_C2_1]! * fixed.slopeScale,
    kappaC1,
    kappaX,
    kappaC2,
    kappaC1Derivative: sigmoid(raw[KAPPA_C1_RAW]!),
    kappaXDerivative: sigmoid(raw[KAPPA_X_RAW]!),
    kappaC2Derivative: sigmoid(raw[KAPPA_C2_RAW]!),
    leftSupportWidth,
    rightSupportWidth,
    leftWidthDerivative,
    rightLeftWidthDerivative,
    rightWidthDerivative,
    leftSupportSharpness,
    rightSupportSharpness,
    leftSupportSharpnessDerivative: sigmoid(raw[LEFT_SHARPNESS_RAW]!),
    rightSupportSharpnessDerivative: sigmoid(raw[RIGHT_SHARPNESS_RAW]!),
  };
}

function parametersFromRaw(
  raw: Float64Array,
  fixed: FixedParameters,
): ConditionalFourSegmentParameters {
  const mapped = mappedRawParameters(raw, fixed);
  return {
    latentLower: fixed.latentLower,
    latentUpper: fixed.latentUpper,
    visibleLower: fixed.visibleLower,
    visibleUpper: fixed.visibleUpper,
    c1: mapped.c1,
    c2: mapped.c2,
    leftSupportWidth: mapped.leftSupportWidth,
    rightSupportWidth: mapped.rightSupportWidth,
    leftSupportSharpness: mapped.leftSupportSharpness,
    rightSupportSharpness: mapped.rightSupportSharpness,
    baseSlope: [mapped.base0, mapped.base1],
    betaC1: [mapped.betaC10, mapped.betaC11],
    betaX: [mapped.betaX0, mapped.betaX1],
    betaC2: [mapped.betaC20, mapped.betaC21],
    kappaC1: mapped.kappaC1,
    kappaX: mapped.kappaX,
    kappaC2: mapped.kappaC2,
  };
}

function rawFromParameters(
  parameters: ConditionalFourSegmentParameters,
  fixed: FixedParameters,
): Float64Array {
  const raw = new Float64Array(PARAMETER_COUNT);
  [raw[C1_RAW], raw[C2_RAW]] = rawBreakpoints(parameters.c1, parameters.c2, fixed);
  setSlopeCoefficients(raw, BASE_0, parameters.baseSlope, fixed.slopeScale);
  setSlopeCoefficients(raw, BETA_C1_0, parameters.betaC1, fixed.slopeScale);
  setSlopeCoefficients(raw, BETA_X_0, parameters.betaX, fixed.slopeScale);
  setSlopeCoefficients(raw, BETA_C2_0, parameters.betaC2, fixed.slopeScale);
  raw[KAPPA_C1_RAW] = inverseSoftplus(
    Math.max(1e-12, parameters.kappaC1 - fixed.minimumKappa),
  );
  raw[KAPPA_X_RAW] = inverseSoftplus(
    Math.max(1e-12, parameters.kappaX - fixed.minimumKappa),
  );
  raw[KAPPA_C2_RAW] = inverseSoftplus(
    Math.max(1e-12, parameters.kappaC2 - fixed.minimumKappa),
  );
  [raw[LEFT_WIDTH_RAW], raw[RIGHT_WIDTH_RAW]] = rawSupportWidths(
    parameters.leftSupportWidth,
    parameters.rightSupportWidth,
    fixed,
  );
  raw[LEFT_SHARPNESS_RAW] = inverseSoftplus(Math.max(
    1e-12,
    parameters.leftSupportSharpness - fixed.minimumSupportSharpness,
  ));
  raw[RIGHT_SHARPNESS_RAW] = inverseSoftplus(Math.max(
    1e-12,
    parameters.rightSupportSharpness - fixed.minimumSupportSharpness,
  ));
  boundUnconstrainedParameters(raw);
  return raw;
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

function compactEnvelopeLogSlope(
  action: number,
  parameters: ConditionalFourSegmentParameters,
): number {
  const left = logCompactStepWithDerivative(
    (action - parameters.latentLower) / parameters.leftSupportWidth,
    parameters.leftSupportSharpness,
  );
  const right = logCompactStepWithDerivative(
    (parameters.latentUpper - action) / parameters.rightSupportWidth,
    parameters.rightSupportSharpness,
  );
  return left.coordinateDerivative / parameters.leftSupportWidth
    - right.coordinateDerivative / parameters.rightSupportWidth;
}

function compactEnvelopeLogValueMapped(
  action: number,
  mapped: ReturnType<typeof mappedRawParameters>,
  fixed: FixedParameters,
): {
  value: number;
  leftWidthDerivative: number;
  rightWidthDerivative: number;
  leftDerivative: number;
  rightDerivative: number;
} {
  const left = logCompactStepWithDerivative(
    (action - fixed.latentLower) / mapped.leftSupportWidth,
    mapped.leftSupportSharpness,
  );
  const right = logCompactStepWithDerivative(
    (fixed.latentUpper - action) / mapped.rightSupportWidth,
    mapped.rightSupportSharpness,
  );
  return {
    value: left.value + right.value,
    leftWidthDerivative: left.coordinateDerivative
      * -(action - fixed.latentLower) / (mapped.leftSupportWidth ** 2),
    rightWidthDerivative: right.coordinateDerivative
      * -(fixed.latentUpper - action) / (mapped.rightSupportWidth ** 2),
    leftDerivative: left.sharpnessDerivative * mapped.leftSupportSharpnessDerivative,
    rightDerivative: right.sharpnessDerivative * mapped.rightSupportSharpnessDerivative,
  };
}

function logCompactStepWithDerivative(
  t: number,
  sharpness: number,
): { value: number; coordinateDerivative: number; sharpnessDerivative: number } {
  if (t <= 0) {
    return { value: Number.NEGATIVE_INFINITY, coordinateDerivative: 0, sharpnessDerivative: 0 };
  }
  if (t >= 1) return { value: 0, coordinateDerivative: 0, sharpnessDerivative: 0 };
  const shape = 1 / (1 - t) - 1 / t;
  const argument = sharpness * shape;
  const argumentDerivative = sharpness * (1 / ((1 - t) ** 2) + 1 / (t * t));
  return {
    value: -softplus(-argument),
    coordinateDerivative: sigmoid(-argument) * argumentDerivative,
    sharpnessDerivative: shape * sigmoid(-argument),
  };
}

function logCompactStep(t: number, sharpness: number): number {
  return logCompactStepWithDerivative(t, sharpness).value;
}

function fitDiagnostics(
  actions: ArrayLike<number>,
  targets: ArrayLike<number>,
  states: ArrayLike<number>,
  parameters: ConditionalFourSegmentParameters,
): Pick<ConditionalFourSegmentPolicyFit, "crossEntropy" | "klDivergence" | "meanSquaredError"> {
  const model = new Float64Array(actions.length);
  let crossEntropy = 0;
  let targetEntropy = 0;
  let meanSquaredError = 0;
  for (let row = 0; row < states.length; row += 1) {
    conditionalFourSegmentExposureProbabilities(actions, states[row]!, parameters, model);
    let targetTotal = 0;
    for (let action = 0; action < actions.length; action += 1) {
      targetTotal += targets[row * actions.length + action]!;
    }
    for (let action = 0; action < actions.length; action += 1) {
      const target = targets[row * actions.length + action]! / targetTotal;
      if (target > 0) {
        crossEntropy -= target * Math.log(Math.max(1e-300, model[action]!)) / states.length;
        targetEntropy -= target * Math.log(target) / states.length;
      }
      meanSquaredError += (model[action]! - target) ** 2 / (states.length * actions.length);
    }
  }
  return {
    crossEntropy,
    klDivergence: Math.max(0, crossEntropy - targetEntropy),
    meanSquaredError,
  };
}

function normalizedSampledTargets(
  actionCount: number,
  targets: ArrayLike<number>,
  stateIndices: number[],
  actionIndices: number[],
): Float64Array {
  const sampled = new Float64Array(stateIndices.length * actionIndices.length);
  for (let row = 0; row < stateIndices.length; row += 1) {
    let total = 0;
    for (let action = 0; action < actionIndices.length; action += 1) {
      const probability = targets[stateIndices[row]! * actionCount + actionIndices[action]!]!;
      sampled[row * actionIndices.length + action] = probability;
      total += probability;
    }
    if (!(total > 0)) throw new Error("Conditional four-segment fit sampled an empty row.");
    for (let action = 0; action < actionIndices.length; action += 1) {
      sampled[row * actionIndices.length + action] /= total;
    }
  }
  return sampled;
}

function stableLogProbabilities(probabilities: Float64Array): Float64Array {
  const maximum = probabilities.reduce((best, value) => Math.max(best, value), 0);
  const floor = Math.max(1e-300, maximum * 1e-6);
  return Float64Array.from(probabilities, (probability) => Math.log(Math.max(floor, probability)));
}

function empiricalLogSlope(
  actions: Float64Array,
  logProbabilities: Float64Array,
  probabilities: Float64Array,
): number {
  const maximum = probabilities.reduce((best, value) => Math.max(best, value), 0);
  const threshold = maximum * 1e-5;
  let count = 0;
  let meanAction = 0;
  let meanLogProbability = 0;
  for (let index = 0; index < actions.length; index += 1) {
    if (probabilities[index]! < threshold) continue;
    count += 1;
    meanAction += actions[index]!;
    meanLogProbability += logProbabilities[index]!;
  }
  if (count < 2) return 0;
  meanAction /= count;
  meanLogProbability /= count;
  let numerator = 0;
  let denominator = 0;
  for (let index = 0; index < actions.length; index += 1) {
    if (probabilities[index]! < threshold) continue;
    const centeredAction = actions[index]! - meanAction;
    numerator += centeredAction * (logProbabilities[index]! - meanLogProbability);
    denominator += centeredAction * centeredAction;
  }
  return denominator > 0 ? numerator / denominator : 0;
}

function empiricalSlopeChange(
  actions: Float64Array,
  logProbabilities: Float64Array,
  location: number,
): number {
  if (location <= actions[1]! || location >= actions.at(-2)!) return 0;
  const slopes = new Float64Array(actions.length - 1);
  const midpoints = new Float64Array(actions.length - 1);
  for (let index = 0; index < slopes.length; index += 1) {
    slopes[index] = (logProbabilities[index + 1]! - logProbabilities[index]!)
      / (actions[index + 1]! - actions[index]!);
    midpoints[index] = (actions[index + 1]! + actions[index]!) / 2;
  }
  const nearest = (side: "left" | "right") => Array.from(slopes, (slope, index) => ({
    slope,
    distance: Math.abs(midpoints[index]! - location),
    side: midpoints[index]! < location ? "left" : "right",
  })).filter((candidate) => candidate.side === side)
    .sort((left, right) => left.distance - right.distance)
    .slice(0, 3);
  const left = nearest("left");
  const right = nearest("right");
  if (left.length === 0 || right.length === 0) return 0;
  return average(right.map((candidate) => candidate.slope))
    - average(left.map((candidate) => candidate.slope));
}

function linearRegression(
  x: Float64Array,
  y: Float64Array,
): readonly [number, number] {
  const meanX = x.reduce((sum, value) => sum + value, 0) / x.length;
  const meanY = y.reduce((sum, value) => sum + value, 0) / y.length;
  let numerator = 0;
  let denominator = 0;
  for (let index = 0; index < x.length; index += 1) {
    numerator += (x[index]! - meanX) * (y[index]! - meanY);
    denominator += (x[index]! - meanX) ** 2;
  }
  const slope = denominator > 0 ? numerator / denominator : 0;
  return [meanY - slope * meanX, slope];
}

function rawBreakpoints(
  c1: number,
  c2: number,
  fixed: FixedParameters,
): readonly [number, number] {
  const firstFraction = clamp(
    (c1 - fixed.latentLower) / (fixed.latentUpper - fixed.latentLower),
    1e-6,
    1 - 1e-6,
  );
  const secondFraction = clamp(
    (c2 - c1) / (fixed.latentUpper - c1),
    1e-6,
    1 - 1e-6,
  );
  return [logit(firstFraction), logit(secondFraction)];
}

function initialSupportWidths(
  fixed: FixedParameters,
  options: ConditionalFourSegmentModelOptions,
): readonly [number, number] {
  const latentSpan = fixed.latentUpper - fixed.latentLower;
  const left = options.initialLeftSupportWidth
    ?? Math.max(1, Math.min(latentSpan / 4, (fixed.visibleLower - fixed.latentLower) / 2));
  const right = options.initialRightSupportWidth
    ?? Math.max(1, Math.min(latentSpan / 4, (fixed.latentUpper - fixed.visibleUpper) / 2));
  if (!(left > 0 && right > 0 && left + right < latentSpan)) {
    throw new Error("Initial compact-envelope widths must be positive and leave an interior plateau.");
  }
  return [left, right];
}

function rawSupportWidths(
  left: number,
  right: number,
  fixed: FixedParameters,
): readonly [number, number] {
  const latentSpan = fixed.latentUpper - fixed.latentLower;
  const leftFraction = clamp(left / latentSpan, 1e-6, 1 - 1e-6);
  const rightFraction = clamp(right / (latentSpan - left), 1e-6, 1 - 1e-6);
  return [logit(leftFraction), logit(rightFraction)];
}

function overlappingSupportWidths(
  fixed: FixedParameters,
  baseline: readonly [number, number],
): readonly [number, number] {
  const latentSpan = fixed.latentUpper - fixed.latentLower;
  const visibleSpan = fixed.visibleUpper - fixed.visibleLower;
  let left = Math.max(
    baseline[0],
    fixed.visibleLower - fixed.latentLower + visibleSpan / 4,
  );
  let right = Math.max(
    baseline[1],
    fixed.latentUpper - fixed.visibleUpper + visibleSpan / 4,
  );
  const maximumCombinedWidth = latentSpan * 0.9;
  if (left + right > maximumCombinedWidth) {
    const scale = maximumCombinedWidth / (left + right);
    left *= scale;
    right *= scale;
  }
  return [left, right];
}

function setSlopeCoefficients(
  raw: Float64Array,
  offset: number,
  coefficients: readonly [number, number],
  scale: number,
): void {
  raw[offset] = coefficients[0] / scale;
  raw[offset + 1] = coefficients[1] / scale;
}

function updateInverseHessian(
  inverseHessian: Float64Array,
  parameterDelta: Float64Array,
  gradientDelta: Float64Array,
): Float64Array {
  const curvature = dot(gradientDelta, parameterDelta);
  const curvatureScale = Math.sqrt(dot(parameterDelta, parameterDelta)
    * dot(gradientDelta, gradientDelta));
  if (!(curvature > 1e-10 * Math.max(1, curvatureScale))) {
    return identityMatrix(PARAMETER_COUNT);
  }
  const hessianGradient = matrixVectorProduct(inverseHessian, gradientDelta);
  const gradientHessianGradient = dot(gradientDelta, hessianGradient);
  const result = inverseHessian.slice();
  const firstScale = (curvature + gradientHessianGradient) / (curvature * curvature);
  for (let row = 0; row < PARAMETER_COUNT; row += 1) {
    for (let column = 0; column < PARAMETER_COUNT; column += 1) {
      const offset = row * PARAMETER_COUNT + column;
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

function subtract(left: Float64Array, right: Float64Array): Float64Array {
  return Float64Array.from(left, (value, index) => value - right[index]!);
}

function dot(left: Float64Array, right: Float64Array): number {
  return left.reduce((sum, value, index) => sum + value * right[index]!, 0);
}

function maximumAbsolute(values: Float64Array): number {
  return values.reduce((maximum, value) => Math.max(maximum, Math.abs(value)), 0);
}

function boundUnconstrainedParameters(raw: Float64Array): void {
  raw[C1_RAW] = clamp(raw[C1_RAW]!, -14, 14);
  raw[C2_RAW] = clamp(raw[C2_RAW]!, -14, 14);
  for (let index = KAPPA_C1_RAW; index <= RIGHT_SHARPNESS_RAW; index += 1) {
    raw[index] = clamp(raw[index]!, -16, 12);
  }
}

function validateInputs(
  actions: ArrayLike<number>,
  targets: ArrayLike<number>,
  states: ArrayLike<number>,
  options: ConditionalFourSegmentModelOptions,
): void {
  if (actions.length < 5 || states.length < 3 || targets.length !== actions.length * states.length
    || !(options.latentLower < actions[0]!)
    || !(options.latentUpper > actions[actions.length - 1]!)) {
    throw new Error("Conditional four-segment fit requires complete rows inside latent support.");
  }
  for (let index = 1; index < actions.length; index += 1) {
    if (!(actions[index]! > actions[index - 1]!)) {
      throw new Error("Conditional four-segment action grid must be strictly ordered.");
    }
  }
  for (let index = 0; index < states.length; index += 1) {
    if (!Number.isFinite(states[index]!)) {
      throw new Error("Conditional four-segment current exposures must be finite.");
    }
  }
  for (let index = 0; index < targets.length; index += 1) {
    if (!Number.isFinite(targets[index]!) || targets[index]! < 0) {
      throw new Error("Conditional four-segment targets must be finite and non-negative.");
    }
  }
  for (let row = 0; row < states.length; row += 1) {
    let total = 0;
    for (let action = 0; action < actions.length; action += 1) {
      total += targets[row * actions.length + action]!;
    }
    if (!(total > 0)) {
      throw new Error("Conditional four-segment targets must have positive row mass.");
    }
  }
}

function validateFixed(parameters: FixedParameters): void {
  if (!(parameters.latentLower < parameters.visibleLower
      && parameters.visibleLower < parameters.visibleUpper
      && parameters.visibleUpper < parameters.latentUpper)
    || !(parameters.minimumKappa > 0)
    || !(parameters.minimumSupportSharpness > 0)) {
    throw new Error("Conditional four-segment support is invalid.");
  }
}

function scaledSoftplus(value: number, kappa: number): number {
  return softplus(kappa * value) / kappa;
}

function scaledSoftplusKappaDerivative(value: number, kappa: number): number {
  const scaled = kappa * value;
  return (scaled * sigmoid(scaled) - softplus(scaled)) / (kappa * kappa);
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

function linear(coefficients: readonly [number, number], value: number): number {
  return coefficients[0] + coefficients[1] * value;
}

function scaledCurrentExposure(current: number, lower: number, upper: number): number {
  return clamp((2 * current - lower - upper) / (upper - lower), -1, 1);
}

function sampledIndices(length: number, maximumCount: number): number[] {
  if (length <= maximumCount) return Array.from({ length }, (_, index) => index);
  const result = new Set<number>();
  for (let index = 0; index < maximumCount; index += 1) {
    result.add(Math.round(index / (maximumCount - 1) * (length - 1)));
  }
  return [...result].sort((left, right) => left - right);
}

function average(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}
