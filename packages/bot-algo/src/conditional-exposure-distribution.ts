export interface ConditionalFourSegmentModelOptions {
  latentLower: number;
  latentUpper: number;
  visibleLower?: number;
  visibleUpper?: number;
  /** Action interval used only for CE, KL, and probability MSE. */
  metricVisibleLower?: number;
  metricVisibleUpper?: number;
  /** Optional separate current-exposure interval for diagnostics. */
  metricCurrentLower?: number;
  metricCurrentUpper?: number;
  /** One-way execution friction as a decimal rate. It determines beta_x. */
  friction?: number;
  /** Oracle policy temperature. It determines beta_x. */
  temperature?: number;
  initialC1?: number;
  initialC2?: number;
  /** Exact feasible holding interval; defaults to the complete effective range. */
  cutoffLower?: number;
  cutoffUpper?: number;
  maxIterations?: number;
  restartCount?: number;
  sampleStates?: number;
  sampleActions?: number;
  tolerance?: number;
  /** Jointly refine the weighted score projection against cross-entropy. */
  refineProjectedFit?: boolean;
}

export interface ConditionalFourSegmentParameters {
  latentLower: number;
  latentUpper: number;
  visibleLower: number;
  visibleUpper: number;
  /** Hard survival bounds learned from the mandatory holding path. */
  cutoffLower: number;
  cutoffUpper: number;
  /** Center used by the fitted quadratic basis (the effective-range center for new models). */
  basisCenter: number;
  c1: number;
  c2: number;
  /** Linear score coefficient b. */
  baseSlope: number;
  /** Signed score curvature lambda in -lambda (a-a_center)^2 / 2. */
  quadraticPrecision: number;
  betaC1: number;
  /** Fee-derived moving transition coefficient; not learned. */
  betaX: number;
  betaC2: number;
  /** Fixed breakpoint sharpnesses calibrated for the fitted effective span. */
  kappaC1: number;
  kappaX: number;
  kappaC2: number;
}

export interface ConditionalFourSegmentSliceParameters {
  baseSlope: number;
  quadraticPrecision: number;
  betaC1: number;
  betaX: number;
  betaC2: number;
  kappaC1: number;
  kappaX: number;
  kappaC2: number;
  /** Piecewise slope offsets before the smooth -lambda(a-a_center) contribution. */
  segmentSlopeOffsets: readonly [number, number, number, number];
}

export interface ConditionalFourSegmentRawParameterOptions {
  latentLower: number;
  latentUpper: number;
  visibleLower: number;
  visibleUpper: number;
  friction?: number;
  temperature?: number;
}

/** [c1, c2, b, lambda, beta_c1, beta_c2, cutoff_lower, cutoff_upper]. */
export const CONDITIONAL_FOUR_SEGMENT_PARAMETER_COUNT = 8;
export const LEGACY_CONDITIONAL_FOUR_SEGMENT_PARAMETER_COUNT = 6;

export type ConditionalFourSegmentFitTermination =
  | "gradient"
  | "relative-loss"
  | "line-search"
  | "iteration-limit";

export interface ConditionalFourSegmentPolicyFit {
  parameters: ConditionalFourSegmentParameters;
  /** Canonical [c1, c2, b, lambda, beta_c1, beta_c2, cutoff_lower, cutoff_upper] representation. */
  rawParameters: Float64Array;
  crossEntropy: number;
  klDivergence: number;
  meanSquaredError: number;
  iterations: number;
  restarts: number;
  termination: ConditionalFourSegmentFitTermination;
  converged: boolean;
  refined: boolean;
}

interface FixedParameters {
  latentLower: number;
  latentUpper: number;
  visibleLower: number;
  visibleUpper: number;
  visibleCenter: number;
  visibleSpan: number;
  basisCenter: number;
  basisSpan: number;
  halfBasisSpan: number;
  slopeScale: number;
  precisionScale: number;
  betaX: number;
  kappaC1: number;
  kappaX: number;
  kappaC2: number;
}

interface MappedParameters {
  c1: number;
  c2: number;
  c1Derivative: number;
  c2C1Derivative: number;
  c2Derivative: number;
  baseSlope: number;
  quadraticPrecision: number;
  betaC1: number;
  betaC2: number;
  cutoffLower: number;
  cutoffUpper: number;
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

const PARAMETER_COUNT = CONDITIONAL_FOUR_SEGMENT_PARAMETER_COUNT;
const C1_RAW = 0;
const C2_RAW = 1;
const BASE_RAW = 2;
const PRECISION_RAW = 3;
const BETA_C1_RAW = 4;
const BETA_C2_RAW = 5;
const CUTOFF_LOWER_RAW = 6;
const CUTOFF_UPPER_RAW = 7;
const LINEAR_PARAMETER_INDICES = [BASE_RAW, PRECISION_RAW, BETA_C1_RAW, BETA_C2_RAW] as const;
const FIXED_KAPPA_C_VISIBLE_PRODUCT = 82;
const FIXED_KAPPA_X_VISIBLE_PRODUCT = 678;

/**
 * Decode the learned coordinates. New eight-coordinate models fit and scale on
 * the complete effective range, then apply the usable range only as an
 * execution-time truncation. Six-coordinate artifacts retain their original
 * visible-range scaling and receive effective-range cutoff defaults.
 */
export function conditionalFourSegmentParametersFromRaw(
  rawInput: ArrayLike<number>,
  options: ConditionalFourSegmentRawParameterOptions,
): ConditionalFourSegmentParameters {
  const legacy = rawInput.length === LEGACY_CONDITIONAL_FOUR_SEGMENT_PARAMETER_COUNT;
  if (!legacy && rawInput.length !== PARAMETER_COUNT) {
    throw new Error(
      `Conditional four-segment raw output must contain ${LEGACY_CONDITIONAL_FOUR_SEGMENT_PARAMETER_COUNT} or ${PARAMETER_COUNT} values.`,
    );
  }
  const fixed = createFixedParameters(options, legacy);
  const raw = new Float64Array(PARAMETER_COUNT);
  raw.set(rawInput);
  if (legacy) {
    raw[CUTOFF_LOWER_RAW] = -14;
    raw[CUTOFF_UPPER_RAW] = 14;
  }
  if (!raw.every(Number.isFinite)) {
    throw new Error("Conditional four-segment raw output must be finite.");
  }
  boundRaw(raw);
  return parametersFromRaw(raw, fixed);
}

/** Encode an exact feasible interval into the two bounded MLP coordinates. */
export function conditionalCutoffRawParameters(
  cutoffLower: number,
  cutoffUpper: number,
  support: Pick<ConditionalFourSegmentRawParameterOptions, "latentLower" | "latentUpper">,
): readonly [number, number] {
  if (!(Number.isFinite(cutoffLower) && Number.isFinite(cutoffUpper))
    || !(support.latentLower < 0 && support.latentUpper > 0)
    || cutoffLower < support.latentLower || cutoffLower > 0
    || cutoffUpper < 0 || cutoffUpper > support.latentUpper) {
    throw new Error("Conditional hard cutoffs must form a finite interval around zero inside effective support.");
  }
  const lowerFraction = (cutoffLower - support.latentLower) / -support.latentLower;
  const upperFraction = cutoffUpper / support.latentUpper;
  return [
    lowerFraction <= 0 ? -14 : lowerFraction >= 1 ? 14 : clamp(logit(lowerFraction), -14, 14),
    upperFraction <= 0 ? -14 : upperFraction >= 1 ? 14 : clamp(logit(upperFraction), -14, 14),
  ];
}

/** Fit the shared score surface while keeping exact hard cutoffs fixed. */
export function fitConditionalFourSegmentPolicy(
  actionGrid: ArrayLike<number>,
  targetProbabilities: ArrayLike<number>,
  currentExposures: ArrayLike<number>,
  options: ConditionalFourSegmentModelOptions,
): ConditionalFourSegmentPolicyFit {
  validateInputs(actionGrid, targetProbabilities, currentExposures, options);
  const visibleLower = options.visibleLower ?? actionGrid[0]!;
  const visibleUpper = options.visibleUpper ?? actionGrid[actionGrid.length - 1]!;
  const metricVisibleLower = options.metricVisibleLower ?? visibleLower;
  const metricVisibleUpper = options.metricVisibleUpper ?? visibleUpper;
  const metricCurrentLower = options.metricCurrentLower ?? metricVisibleLower;
  const metricCurrentUpper = options.metricCurrentUpper ?? metricVisibleUpper;
  const metricActionCount = Array.from(actionGrid).filter((value) =>
    value >= metricVisibleLower && value <= metricVisibleUpper).length;
  const metricStateCount = Array.from(currentExposures).filter((value) =>
    value >= metricCurrentLower && value <= metricCurrentUpper).length;
  if (!Number.isFinite(metricVisibleLower) || !Number.isFinite(metricVisibleUpper)
    || metricVisibleLower < visibleLower || metricVisibleUpper > visibleUpper
    || !(metricVisibleLower < metricVisibleUpper)
    || !Number.isFinite(metricCurrentLower) || !Number.isFinite(metricCurrentUpper)
    || metricCurrentLower < options.latentLower || metricCurrentUpper > options.latentUpper
    || !(metricCurrentLower < metricCurrentUpper)
    || metricActionCount < 5 || metricStateCount < 3) {
    throw new Error("Conditional four-segment metric range must be a complete subset of visible support.");
  }
  const fixed = createFixedParameters({
    ...options,
    visibleLower,
    visibleUpper,
  });
  const stateIndices = sampledIndices(
    currentExposures.length,
    Math.max(3, Math.floor(options.sampleStates ?? 31)),
  );
  const actionIndices = sampledIndices(
    actionGrid.length,
    Math.max(5, Math.floor(options.sampleActions ?? 51)),
  );
  const actions = Float64Array.from(actionIndices, (index) => actionGrid[index]!);
  const states = Float64Array.from(stateIndices, (index) => currentExposures[index]!);
  const targets = normalizedSampledTargets(
    actionGrid.length,
    targetProbabilities,
    stateIndices,
    actionIndices,
  );
  const starts = projectedInitialValues(actions, targets, states, fixed, options);
  const restartCount = Math.max(1, Math.min(
    starts.length,
    Math.floor(options.restartCount ?? starts.length),
  ));
  const objective = (raw: Float64Array) => conditionalCrossEntropyWithGradient(
    actions,
    targets,
    states,
    raw,
    fixed,
  );
  starts.sort((left, right) => objective(left).loss - objective(right).loss);
  const refine = options.refineProjectedFit !== false;
  const maximumIterations = refine ? Math.max(1, Math.floor(options.maxIterations ?? 100)) : 0;
  const tolerance = Math.max(Number.EPSILON, options.tolerance ?? 1e-6);
  let best: OptimizationResult | undefined;
  for (let restart = 0; restart < restartCount; restart += 1) {
    const initial = starts[restart]!;
    let candidate: OptimizationResult;
    if (maximumIterations === 0) {
      candidate = {
        ...objective(initial),
        raw: initial,
        iterations: 0,
        termination: "iteration-limit",
        converged: false,
      };
    } else {
      const linearIterations = Math.min(20, Math.max(0, maximumIterations - 1));
      const warmup = optimizeBfgs(
        initial,
        maskedObjective(objective, LINEAR_PARAMETER_INDICES),
        linearIterations,
        tolerance,
      );
      candidate = optimizeBfgs(
        warmup.raw,
        objective,
        maximumIterations - linearIterations,
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
    rawParameters: Float64Array.from(best.raw),
    ...fitDiagnostics(
      actionGrid,
      targetProbabilities,
      currentExposures,
      parameters,
      metricVisibleLower,
      metricVisibleUpper,
      metricCurrentLower,
      metricCurrentUpper,
    ),
    iterations: best.iterations,
    restarts: restartCount,
    termination: best.termination,
    converged: best.converged,
    refined: refine,
  };
}

/** Evaluate one hard-truncated visible row of the fitted model. */
export function conditionalFourSegmentExposureProbabilities(
  actionGrid: ArrayLike<number>,
  currentExposure: number,
  parameters: ConditionalFourSegmentParameters,
  result: Float64Array<ArrayBufferLike> = new Float64Array(actionGrid.length),
): Float64Array<ArrayBufferLike> {
  if (result.length !== actionGrid.length) {
    throw new Error("Conditional four-segment result does not match its action grid.");
  }
  let maximum = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < actionGrid.length; index += 1) {
    const action = actionGrid[index]!;
    const value = action >= parameters.visibleLower && action <= parameters.visibleUpper
      ? conditionalFourSegmentLogKernel(action, currentExposure, parameters)
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
  const orderedChanges = [
    { location: parameters.c1, change: parameters.betaC1 },
    { location: currentExposure, change: parameters.betaX },
    { location: parameters.c2, change: parameters.betaC2 },
  ].sort((left, right) => left.location - right.location);
  const offsets = [parameters.baseSlope];
  for (const transition of orderedChanges) offsets.push(offsets.at(-1)! + transition.change);
  return {
    baseSlope: parameters.baseSlope,
    quadraticPrecision: parameters.quadraticPrecision,
    betaC1: parameters.betaC1,
    betaX: parameters.betaX,
    betaC2: parameters.betaC2,
    kappaC1: parameters.kappaC1,
    kappaX: parameters.kappaX,
    kappaC2: parameters.kappaC2,
    segmentSlopeOffsets: offsets as [number, number, number, number],
  };
}

/** Effective action derivative of a normalized row's log density. */
export function conditionalFourSegmentLogSlope(
  action: number,
  currentExposure: number,
  parameters: ConditionalFourSegmentParameters,
): number {
  if (action < parameters.latentLower || action > parameters.latentUpper
    || action < parameters.cutoffLower || action > parameters.cutoffUpper) return Number.NaN;
  const center = parameters.basisCenter;
  return parameters.baseSlope
    - parameters.quadraticPrecision * (action - center)
    + parameters.betaC1 * sigmoid(parameters.kappaC1 * (action - parameters.c1))
    + parameters.betaX * sigmoid(parameters.kappaX * (action - currentExposure))
    + parameters.betaC2 * sigmoid(parameters.kappaC2 * (action - parameters.c2));
}

/** Unnormalized conditional score; row-wise additive constants are immaterial. */
export function conditionalFourSegmentLogKernel(
  action: number,
  currentExposure: number,
  parameters: ConditionalFourSegmentParameters,
): number {
  if (action < parameters.latentLower || action > parameters.latentUpper
    || action < parameters.cutoffLower || action > parameters.cutoffUpper) {
    return Number.NEGATIVE_INFINITY;
  }
  const center = parameters.basisCenter;
  return parameters.baseSlope * (action - parameters.latentLower)
    - 0.5 * parameters.quadraticPrecision * (action - center) ** 2
    + parameters.betaC1 * scaledSoftplus(action - parameters.c1, parameters.kappaC1)
    + parameters.betaX * scaledSoftplus(action - currentExposure, parameters.kappaX)
    + parameters.betaC2 * scaledSoftplus(action - parameters.c2, parameters.kappaC2);
}

function projectedInitialValues(
  actions: Float64Array,
  targets: Float64Array,
  states: Float64Array,
  fixed: FixedParameters,
  options: ConditionalFourSegmentModelOptions,
): Float64Array[] {
  const span = fixed.visibleSpan;
  const [cutoffLowerRaw, cutoffUpperRaw] = conditionalCutoffRawParameters(
    options.cutoffLower ?? fixed.latentLower,
    options.cutoffUpper ?? fixed.latentUpper,
    fixed,
  );
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
  const locations: Array<readonly [number, number]> = [
    ...(detected ? [detected] : []),
    [outsideC1, outsideC2],
    [fixed.visibleLower + span / 3, fixed.visibleUpper - span / 3],
    [
      Math.max(fixed.latentLower + 1e-6, fixed.visibleLower - span / 8),
      Math.min(fixed.latentUpper - 1e-6, fixed.visibleUpper + span / 8),
    ],
    [fixed.visibleLower + span / 5, fixed.visibleUpper - span / 5],
  ];
  const unique = new Map<string, Float64Array>();
  for (const [inputC1, inputC2] of locations) {
    const c1 = clamp(inputC1, fixed.latentLower + 1e-6, fixed.latentUpper - 2e-6);
    const c2 = clamp(inputC2, c1 + 1e-6, fixed.latentUpper - 1e-6);
    const raw = projectedLinearRaw(actions, targets, states, c1, c2, fixed);
    raw[CUTOFF_LOWER_RAW] = cutoffLowerRaw;
    raw[CUTOFF_UPPER_RAW] = cutoffUpperRaw;
    unique.set(`${raw[C1_RAW]!.toFixed(7)}:${raw[C2_RAW]!.toFixed(7)}`, raw);
  }
  return [...unique.values()];
}

/**
 * Weighted variable projection from the screenshot derivation. The nuisance
 * alpha(x) is eliminated by centering each row, leaving the linear design
 * [a, -a^2/2, H_c1(a), H_c2(a)] after subtracting the fixed fee transition.
 */
function projectedLinearRaw(
  actions: Float64Array,
  targets: Float64Array,
  states: Float64Array,
  c1: number,
  c2: number,
  fixed: FixedParameters,
): Float64Array {
  const featureCount = 4;
  const normal = new Float64Array(featureCount * featureCount);
  const right = new Float64Array(featureCount);
  const rowFeatures = new Float64Array(actions.length * featureCount);
  const rowResidual = new Float64Array(actions.length);
  const means = new Float64Array(featureCount);
  for (let row = 0; row < states.length; row += 1) {
    means.fill(0);
    let residualMean = 0;
    let totalWeight = 0;
    let maximumProbability = 0;
    for (let actionIndex = 0; actionIndex < actions.length; actionIndex += 1) {
      maximumProbability = Math.max(
        maximumProbability,
        targets[row * actions.length + actionIndex]!,
      );
    }
    const probabilityFloor = Math.max(1e-300, maximumProbability * 1e-6);
    for (let actionIndex = 0; actionIndex < actions.length; actionIndex += 1) {
      const action = actions[actionIndex]!;
      const target = targets[row * actions.length + actionIndex]!;
      const weight = Math.max(target, maximumProbability * 1e-3);
      const z = (action - fixed.basisCenter) / fixed.halfBasisSpan;
      const featureOffset = actionIndex * featureCount;
      rowFeatures[featureOffset] = (action - fixed.latentLower) / fixed.basisSpan;
      rowFeatures[featureOffset + 1] = -0.5 * z * z;
      rowFeatures[featureOffset + 2]
        = scaledSoftplus(action - c1, fixed.kappaC1) / fixed.basisSpan;
      rowFeatures[featureOffset + 3]
        = scaledSoftplus(action - c2, fixed.kappaC2) / fixed.basisSpan;
      rowResidual[actionIndex] = Math.log(Math.max(probabilityFloor, target))
        - fixed.betaX * scaledSoftplus(action - states[row]!, fixed.kappaX);
      totalWeight += weight;
      residualMean += weight * rowResidual[actionIndex]!;
      for (let feature = 0; feature < featureCount; feature += 1) {
        means[feature] += weight * rowFeatures[featureOffset + feature]!;
      }
    }
    residualMean /= totalWeight;
    for (let feature = 0; feature < featureCount; feature += 1) means[feature] /= totalWeight;
    for (let actionIndex = 0; actionIndex < actions.length; actionIndex += 1) {
      const target = targets[row * actions.length + actionIndex]!;
      const weight = Math.max(target, maximumProbability * 1e-3);
      const featureOffset = actionIndex * featureCount;
      const centeredResidual = rowResidual[actionIndex]! - residualMean;
      for (let left = 0; left < featureCount; left += 1) {
        const centeredLeft = rowFeatures[featureOffset + left]! - means[left]!;
        right[left] += weight * centeredLeft * centeredResidual;
        for (let column = 0; column <= left; column += 1) {
          normal[left * featureCount + column] += weight * centeredLeft
            * (rowFeatures[featureOffset + column]! - means[column]!);
        }
      }
    }
  }
  for (let row = 0; row < featureCount; row += 1) {
    for (let column = 0; column < row; column += 1) {
      normal[column * featureCount + row] = normal[row * featureCount + column]!;
    }
    normal[row * featureCount + row] += 1e-7;
  }
  const coefficients = solveLinearSystem(normal, right, featureCount)
    ?? new Float64Array(featureCount);
  const raw = new Float64Array(PARAMETER_COUNT);
  [raw[C1_RAW], raw[C2_RAW]] = rawBreakpoints(c1, c2, fixed);
  raw[BASE_RAW] = coefficients[0]!;
  raw[PRECISION_RAW] = coefficients[1]!;
  raw[BETA_C1_RAW] = coefficients[2]!;
  raw[BETA_C2_RAW] = coefficients[3]!;
  boundRaw(raw);
  return raw;
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
    let maximum = Number.NEGATIVE_INFINITY;
    for (let actionIndex = 0; actionIndex < actions.length; actionIndex += 1) {
      const action = actions[actionIndex]!;
      const c1Offset = action - mapped.c1;
      const c2Offset = action - mapped.c2;
      const spC1 = scaledSoftplus(c1Offset, fixed.kappaC1);
      const spC2 = scaledSoftplus(c2Offset, fixed.kappaC2);
      const z = (action - fixed.basisCenter) / fixed.halfBasisSpan;
      const feasible = action >= mapped.cutoffLower && action <= mapped.cutoffUpper;
      const logit = feasible ? mapped.baseSlope * (action - fixed.latentLower)
        - 0.5 * mapped.quadraticPrecision * (action - fixed.basisCenter) ** 2
        + mapped.betaC1 * spC1
        + fixed.betaX * scaledSoftplus(action - current, fixed.kappaX)
        + mapped.betaC2 * spC2 : Number.NEGATIVE_INFINITY;
      logits[actionIndex] = logit;
      maximum = Math.max(maximum, logit);
      const derivativeOffset = actionIndex * PARAMETER_COUNT;
      const sigmoidC1 = sigmoid(fixed.kappaC1 * c1Offset);
      const sigmoidC2 = sigmoid(fixed.kappaC2 * c2Offset);
      derivatives[derivativeOffset + C1_RAW]
        = -mapped.betaC1 * sigmoidC1 * mapped.c1Derivative
          - mapped.betaC2 * sigmoidC2 * mapped.c2C1Derivative;
      derivatives[derivativeOffset + C2_RAW]
        = -mapped.betaC2 * sigmoidC2 * mapped.c2Derivative;
      derivatives[derivativeOffset + BASE_RAW]
        = fixed.slopeScale * (action - fixed.latentLower);
      derivatives[derivativeOffset + PRECISION_RAW] = -0.5 * z * z;
      derivatives[derivativeOffset + BETA_C1_RAW] = fixed.slopeScale * spC1;
      derivatives[derivativeOffset + BETA_C2_RAW] = fixed.slopeScale * spC2;
      if (!feasible) derivatives.fill(0, derivativeOffset, derivativeOffset + PARAMETER_COUNT);
    }
    let normalizer = 0;
    for (const logit of logits) normalizer += Math.exp(logit - maximum);
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
  const magnitudePenalty = 1e-9;
  for (const parameter of LINEAR_PARAMETER_INDICES) {
    const excess = Math.max(0, Math.abs(raw[parameter]!) - 100);
    loss += magnitudePenalty * excess ** 2;
    gradient[parameter] += 2 * magnitudePenalty * excess * Math.sign(raw[parameter]!);
  }
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
      boundRaw(candidate);
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
      return { ...value, raw, iterations: iteration, termination: "line-search", converged: false };
    }
    const relativeImprovement = (value.loss - nextValue.loss) / Math.max(1, Math.abs(value.loss));
    stableIterations = relativeImprovement <= tolerance ? stableIterations + 1 : 0;
    inverseHessian = updateInverseHessian(
      inverseHessian,
      subtract(nextRaw, raw),
      subtract(nextValue.gradient, value.gradient),
    );
    raw = nextRaw;
    value = nextValue;
    if (stableIterations >= 4 && maximumAbsolute(value.gradient) <= Math.max(1e-5, tolerance * 10)) {
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

function mappedRawParameters(raw: Float64Array, fixed: FixedParameters): MappedParameters {
  const firstFraction = sigmoid(raw[C1_RAW]!);
  const c1 = fixed.latentLower + (fixed.latentUpper - fixed.latentLower) * firstFraction;
  const secondFraction = sigmoid(raw[C2_RAW]!);
  const c2 = c1 + (fixed.latentUpper - c1) * secondFraction;
  const c1Derivative = (fixed.latentUpper - fixed.latentLower)
    * firstFraction * (1 - firstFraction);
  const cutoffLower = decodeLowerCutoff(raw[CUTOFF_LOWER_RAW]!, fixed);
  const cutoffUpper = decodeUpperCutoff(raw[CUTOFF_UPPER_RAW]!, fixed);
  return {
    c1,
    c2,
    c1Derivative,
    c2C1Derivative: c1Derivative * (1 - secondFraction),
    c2Derivative: (fixed.latentUpper - c1) * secondFraction * (1 - secondFraction),
    baseSlope: raw[BASE_RAW]! * fixed.slopeScale,
    quadraticPrecision: raw[PRECISION_RAW]! * fixed.precisionScale,
    betaC1: raw[BETA_C1_RAW]! * fixed.slopeScale,
    betaC2: raw[BETA_C2_RAW]! * fixed.slopeScale,
    cutoffLower,
    cutoffUpper,
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
    cutoffLower: mapped.cutoffLower,
    cutoffUpper: mapped.cutoffUpper,
    basisCenter: fixed.basisCenter,
    c1: mapped.c1,
    c2: mapped.c2,
    baseSlope: mapped.baseSlope,
    quadraticPrecision: mapped.quadraticPrecision,
    betaC1: mapped.betaC1,
    betaX: fixed.betaX,
    betaC2: mapped.betaC2,
    kappaC1: fixed.kappaC1,
    kappaX: fixed.kappaX,
    kappaC2: fixed.kappaC2,
  };
}

function createFixedParameters(
  options: ConditionalFourSegmentRawParameterOptions,
  legacyVisibleBasis = false,
): FixedParameters {
  const values = [options.latentLower, options.latentUpper, options.visibleLower, options.visibleUpper];
  if (!values.every(Number.isFinite)
    || !(options.latentLower <= options.visibleLower)
    || !(options.visibleLower < options.visibleUpper)
    || !(options.visibleUpper <= options.latentUpper)) {
    throw new Error("Conditional four-segment support must be ordered and finite.");
  }
  const friction = options.friction ?? 0;
  const temperature = options.temperature ?? 0.01;
  if (!(Number.isFinite(friction) && friction >= 0 && friction < 1)
    || !(Number.isFinite(temperature) && temperature > 0)) {
    throw new Error("Conditional four-segment friction and temperature are invalid.");
  }
  const latentSpan = options.latentUpper - options.latentLower;
  const visibleSpan = options.visibleUpper - options.visibleLower;
  const basisSpan = legacyVisibleBasis ? visibleSpan : latentSpan;
  const basisCenter = legacyVisibleBasis
    ? (options.visibleLower + options.visibleUpper) / 2
    : (options.latentLower + options.latentUpper) / 2;
  const halfBasisSpan = basisSpan / 2;
  const buySlopeAtZero = friction > 0 ? friction / (1 - friction) : 0;
  const sellSlopeAtZero = friction;
  return {
    latentLower: options.latentLower,
    latentUpper: options.latentUpper,
    visibleLower: options.visibleLower,
    visibleUpper: options.visibleUpper,
    visibleCenter: (options.visibleLower + options.visibleUpper) / 2,
    visibleSpan,
    basisCenter,
    basisSpan,
    halfBasisSpan,
    slopeScale: 1 / basisSpan,
    precisionScale: 1 / (halfBasisSpan * halfBasisSpan),
    betaX: -(buySlopeAtZero + sellSlopeAtZero) / temperature,
    kappaC1: FIXED_KAPPA_C_VISIBLE_PRODUCT / basisSpan,
    kappaX: FIXED_KAPPA_X_VISIBLE_PRODUCT / basisSpan,
    kappaC2: FIXED_KAPPA_C_VISIBLE_PRODUCT / basisSpan,
  };
}

function empiricalFixedBreakpointLocations(
  actions: Float64Array,
  targets: Float64Array,
  states: Float64Array,
): readonly [number, number] | undefined {
  const actionSpan = actions.at(-1)! - actions[0]!;
  const step = actionSpan / (actions.length - 1);
  const movingExclusion = Math.max(step * 3, actionSpan / 12);
  const curvature = new Float64Array(actions.length);
  for (let actionIndex = 1; actionIndex < actions.length - 1; actionIndex += 1) {
    const location = actions[actionIndex]!;
    let signed = 0;
    let absolute = 0;
    let count = 0;
    for (let row = 0; row < states.length; row += 1) {
      if (Math.abs(states[row]! - location) <= movingExclusion) continue;
      const offset = row * actions.length;
      const maximum = targets.subarray(offset, offset + actions.length)
        .reduce((best, value) => Math.max(best, value), 0);
      if (targets[offset + actionIndex]! < maximum * 1e-6) continue;
      const left = (Math.log(Math.max(maximum * 1e-6, targets[offset + actionIndex]!))
          - Math.log(Math.max(maximum * 1e-6, targets[offset + actionIndex - 1]!)))
        / (actions[actionIndex]! - actions[actionIndex - 1]!);
      const right = (Math.log(Math.max(maximum * 1e-6, targets[offset + actionIndex + 1]!))
          - Math.log(Math.max(maximum * 1e-6, targets[offset + actionIndex]!)))
        / (actions[actionIndex + 1]! - actions[actionIndex]!);
      const change = right - left;
      signed += change;
      absolute += Math.abs(change);
      count += 1;
    }
    if (count > 0) curvature[actionIndex] = Math.abs(signed / count) + 0.15 * absolute / count;
  }
  const candidates = Array.from({ length: actions.length - 2 }, (_, index) => index + 1)
    .sort((left, right) => curvature[right]! - curvature[left]!);
  const selected: number[] = [];
  for (const index of candidates) {
    const location = actions[index]!;
    if (curvature[index]! <= 1e-9) break;
    if (selected.every((value) => Math.abs(value - location) >= Math.max(step * 3, actionSpan / 10))) {
      selected.push(location);
      if (selected.length === 2) break;
    }
  }
  return selected.length === 2
    ? selected.sort((left, right) => left - right) as [number, number]
    : undefined;
}

function fitDiagnostics(
  actions: ArrayLike<number>,
  targets: ArrayLike<number>,
  states: ArrayLike<number>,
  parameters: ConditionalFourSegmentParameters,
  metricVisibleLower: number,
  metricVisibleUpper: number,
  metricCurrentLower: number,
  metricCurrentUpper: number,
): Pick<ConditionalFourSegmentPolicyFit, "crossEntropy" | "klDivergence" | "meanSquaredError"> {
  const model = new Float64Array(actions.length);
  const metricActions = Array.from({ length: actions.length }, (_, index) => index)
    .filter((index) => actions[index]! >= metricVisibleLower
      && actions[index]! <= metricVisibleUpper);
  const metricStates = Array.from({ length: states.length }, (_, index) => index)
    .filter((index) => states[index]! >= metricCurrentLower
      && states[index]! <= metricCurrentUpper);
  let crossEntropy = 0;
  let targetEntropy = 0;
  let meanSquaredError = 0;
  for (const row of metricStates) {
    conditionalFourSegmentExposureProbabilities(actions, states[row]!, parameters, model);
    let targetTotal = 0;
    let modelTotal = 0;
    for (const action of metricActions) {
      targetTotal += targets[row * actions.length + action]!;
      modelTotal += model[action]!;
    }
    if (!(targetTotal > 0) || !(modelTotal > 0)) {
      throw new Error("Conditional four-segment metric rows need visible probability mass.");
    }
    for (const action of metricActions) {
      const target = targets[row * actions.length + action]! / targetTotal;
      const predicted = model[action]! / modelTotal;
      if (target > 0) {
        crossEntropy -= target * Math.log(Math.max(1e-300, predicted)) / metricStates.length;
        targetEntropy -= target * Math.log(target) / metricStates.length;
      }
      meanSquaredError += (predicted - target) ** 2
        / (metricStates.length * metricActions.length);
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

function sampledIndices(length: number, requested: number): number[] {
  if (requested >= length) return Array.from({ length }, (_, index) => index);
  const selected = new Set<number>();
  for (let index = 0; index < requested; index += 1) {
    selected.add(Math.round(index * (length - 1) / (requested - 1)));
  }
  return [...selected].sort((left, right) => left - right);
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

function solveLinearSystem(
  matrixInput: Float64Array,
  vectorInput: Float64Array,
  size: number,
): Float64Array | undefined {
  const matrix = matrixInput.slice();
  const vector = vectorInput.slice();
  for (let pivot = 0; pivot < size; pivot += 1) {
    let best = pivot;
    for (let row = pivot + 1; row < size; row += 1) {
      if (Math.abs(matrix[row * size + pivot]!) > Math.abs(matrix[best * size + pivot]!)) best = row;
    }
    if (Math.abs(matrix[best * size + pivot]!) < 1e-14) return undefined;
    if (best !== pivot) {
      for (let column = pivot; column < size; column += 1) {
        [matrix[pivot * size + column], matrix[best * size + column]]
          = [matrix[best * size + column]!, matrix[pivot * size + column]!];
      }
      [vector[pivot], vector[best]] = [vector[best]!, vector[pivot]!];
    }
    const diagonal = matrix[pivot * size + pivot]!;
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

function updateInverseHessian(
  inverseHessian: Float64Array,
  parameterDelta: Float64Array,
  gradientDelta: Float64Array,
): Float64Array {
  const curvature = dot(gradientDelta, parameterDelta);
  const curvatureScale = Math.sqrt(dot(parameterDelta, parameterDelta) * dot(gradientDelta, gradientDelta));
  if (!(curvature > 1e-10 * Math.max(1, curvatureScale))) return identityMatrix(PARAMETER_COUNT);
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

function validateInputs(
  actions: ArrayLike<number>,
  targets: ArrayLike<number>,
  states: ArrayLike<number>,
  options: ConditionalFourSegmentModelOptions,
): void {
  if (actions.length < 5 || states.length < 3 || targets.length !== actions.length * states.length
    || !(options.latentLower <= actions[0]!)
    || !(options.latentUpper >= actions[actions.length - 1]!)) {
    throw new Error("Conditional four-segment fit requires complete rows inside latent support.");
  }
  for (let index = 1; index < actions.length; index += 1) {
    if (!(actions[index]! > actions[index - 1]!)) {
      throw new Error("Conditional four-segment action grid must be strictly ordered.");
    }
  }
  for (const state of Array.from(states)) {
    if (!Number.isFinite(state)) throw new Error("Conditional four-segment states must be finite.");
  }
  for (const target of Array.from(targets)) {
    if (!Number.isFinite(target) || target < 0) {
      throw new Error("Conditional four-segment targets must be finite and non-negative.");
    }
  }
  for (let row = 0; row < states.length; row += 1) {
    let total = 0;
    for (let action = 0; action < actions.length; action += 1) {
      total += targets[row * actions.length + action]!;
    }
    if (!(total > 0)) throw new Error("Conditional four-segment target rows must have mass.");
  }
}

function boundRaw(raw: Float64Array): void {
  raw[C1_RAW] = clamp(raw[C1_RAW]!, -14, 14);
  raw[C2_RAW] = clamp(raw[C2_RAW]!, -14, 14);
  raw[CUTOFF_LOWER_RAW] = clamp(raw[CUTOFF_LOWER_RAW]!, -14, 14);
  raw[CUTOFF_UPPER_RAW] = clamp(raw[CUTOFF_UPPER_RAW]!, -14, 14);
  for (let index = BASE_RAW; index <= BETA_C2_RAW; index += 1) {
    raw[index] = clamp(raw[index]!, -1e4, 1e4);
  }
}

function decodeLowerCutoff(raw: number, fixed: FixedParameters): number {
  if (raw <= -13.999999) return fixed.latentLower;
  if (raw >= 13.999999) return 0;
  return fixed.latentLower + -fixed.latentLower * sigmoid(raw);
}

function decodeUpperCutoff(raw: number, fixed: FixedParameters): number {
  if (raw <= -13.999999) return 0;
  if (raw >= 13.999999) return fixed.latentUpper;
  return fixed.latentUpper * sigmoid(raw);
}

function scaledSoftplus(value: number, kappa: number): number {
  return softplus(kappa * value) / kappa;
}

function softplus(value: number): number {
  if (value > 35) return value;
  if (value < -35) return Math.exp(value);
  return Math.log1p(Math.exp(value));
}

function sigmoid(value: number): number {
  if (value >= 0) {
    const exponential = Math.exp(-value);
    return 1 / (1 + exponential);
  }
  const exponential = Math.exp(value);
  return exponential / (1 + exponential);
}

function logit(value: number): number {
  return Math.log(value / (1 - value));
}

function clamp(value: number, lower: number, upper: number): number {
  return Math.max(lower, Math.min(upper, value));
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
