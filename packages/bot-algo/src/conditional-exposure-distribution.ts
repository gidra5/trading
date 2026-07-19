export interface ConditionalFourSegmentModelOptions {
  latentLower: number;
  latentUpper: number;
  visibleLower?: number;
  visibleUpper?: number;
  c1?: number;
  c2?: number;
  leftSupportWidth?: number;
  rightSupportWidth?: number;
  supportSharpness?: number;
  minimumStrength?: number;
  minimumWidth?: number;
  minimumKappa?: number;
  maxIterations?: number;
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
  supportSharpness: number;
  betaC1: readonly [number, number];
  betaC2: readonly [number, number];
  tilt: readonly [number, number];
  strengthRaw: readonly [number, number];
  widthRaw: readonly [number, number];
  kappaC1Raw: number;
  kappaC2Raw: number;
  minimumStrength: number;
  minimumWidth: number;
  minimumKappa: number;
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
  tilt: number;
  strength: number;
  width: number;
  segmentSlopes: readonly [number, number, number, number];
}

export interface ConditionalFourSegmentPolicyFit {
  parameters: ConditionalFourSegmentParameters;
  crossEntropy: number;
  klDivergence: number;
  meanSquaredError: number;
  iterations: number;
  converged: boolean;
}

type FixedParameters = Omit<ConditionalFourSegmentParameters,
  "betaC1" | "betaC2" | "tilt" | "strengthRaw" | "widthRaw"
    | "kappaC1Raw" | "kappaC2Raw">;

const PARAMETER_COUNT = 12;

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
    c1: options.c1 ?? visibleLower + visibleSpan / 3,
    c2: options.c2 ?? visibleUpper - visibleSpan / 3,
    leftSupportWidth: options.leftSupportWidth
      ?? Math.max(1, Math.min(latentSpan / 4, (visibleLower - options.latentLower) / 2)),
    rightSupportWidth: options.rightSupportWidth
      ?? Math.max(1, Math.min(latentSpan / 4, (options.latentUpper - visibleUpper) / 2)),
    supportSharpness: options.supportSharpness ?? 1,
    minimumStrength: options.minimumStrength ?? 1e-5,
    minimumWidth: options.minimumWidth ?? Math.max(1e-3, visibleSpan / 200),
    minimumKappa: options.minimumKappa ?? 1e-4,
  };
  validateFixed(fixed);
  const initialStrength = Math.max(fixed.minimumStrength * 2, 2 / visibleSpan);
  const initialWidth = Math.max(fixed.minimumWidth * 2, visibleSpan / 8);
  const initialKappa = 4.394 / Math.max(1e-6, visibleSpan / 4);
  const raw = new Float64Array([
    0, 0, 0, 0, 0, 0,
    inverseSoftplus(initialStrength - fixed.minimumStrength), 0,
    inverseSoftplus(initialWidth - fixed.minimumWidth), 0,
    inverseSoftplus(initialKappa - fixed.minimumKappa),
    inverseSoftplus(initialKappa - fixed.minimumKappa),
  ]);

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
  const sampledTargets = new Float64Array(sampledStates.length * sampledActions.length);
  for (let row = 0; row < stateIndices.length; row += 1) {
    let total = 0;
    for (let action = 0; action < actionIndices.length; action += 1) {
      const probability = targetProbabilities[
        stateIndices[row]! * actionGrid.length + actionIndices[action]!
      ]!;
      sampledTargets[row * sampledActions.length + action] = probability;
      total += probability;
    }
    if (!(total > 0)) throw new Error("Conditional four-segment fit sampled an empty row.");
    for (let action = 0; action < sampledActions.length; action += 1) {
      sampledTargets[row * sampledActions.length + action] /= total;
    }
  }

  const evaluate = (candidate: Float64Array) => sampledCrossEntropy(
    sampledActions,
    sampledTargets,
    sampledStates,
    parametersFromRaw(candidate, fixed),
  );
  const firstMoment = new Float64Array(PARAMETER_COUNT);
  const secondMoment = new Float64Array(PARAMETER_COUNT);
  const gradient = new Float64Array(PARAMETER_COUNT);
  const maximumIterations = Math.max(1, Math.floor(options.maxIterations ?? 90));
  const tolerance = Math.max(Number.EPSILON, options.tolerance ?? 1e-7);
  let bestLoss = evaluate(raw);
  let best = raw.slice();
  let stagnant = 0;
  let iterations = 0;
  let converged = false;
  for (; iterations < maximumIterations; iterations += 1) {
    for (let parameter = 0; parameter < PARAMETER_COUNT; parameter += 1) {
      const original = raw[parameter]!;
      const step = 1e-3 * Math.max(1, Math.abs(original));
      raw[parameter] = original + step;
      const right = evaluate(raw);
      raw[parameter] = original - step;
      const left = evaluate(raw);
      raw[parameter] = original;
      gradient[parameter] = (right - left) / (2 * step);
    }
    if (Math.hypot(...gradient) <= tolerance) {
      converged = true;
      break;
    }
    const time = iterations + 1;
    for (let parameter = 0; parameter < PARAMETER_COUNT; parameter += 1) {
      const value = gradient[parameter]!;
      firstMoment[parameter] = 0.9 * firstMoment[parameter]! + 0.1 * value;
      secondMoment[parameter] = 0.999 * secondMoment[parameter]! + 0.001 * value * value;
      const correctedFirst = firstMoment[parameter]! / (1 - 0.9 ** time);
      const correctedSecond = secondMoment[parameter]! / (1 - 0.999 ** time);
      raw[parameter] -= learningRate(parameter)
        * correctedFirst / (Math.sqrt(correctedSecond) + 1e-8);
    }
    clampRaw(raw);
    const loss = evaluate(raw);
    if (loss + tolerance < bestLoss) {
      bestLoss = loss;
      best = raw.slice();
      stagnant = 0;
    } else if (++stagnant >= 14) {
      converged = true;
      break;
    }
  }
  const parameters = parametersFromRaw(best, fixed);
  return {
    parameters,
    ...fitDiagnostics(actionGrid, targetProbabilities, currentExposures, parameters),
    iterations,
    converged,
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
  const xi = clamp(
    (2 * currentExposure - parameters.latentLower - parameters.latentUpper)
      / (parameters.latentUpper - parameters.latentLower),
    -1,
    1,
  );
  const betaC1 = linear(parameters.betaC1, xi);
  const betaC2 = linear(parameters.betaC2, xi);
  const tilt = linear(parameters.tilt, xi);
  const strength = parameters.minimumStrength + softplus(linear(parameters.strengthRaw, xi));
  const width = parameters.minimumWidth + softplus(linear(parameters.widthRaw, xi));
  const kappaC1 = parameters.minimumKappa + softplus(parameters.kappaC1Raw);
  const kappaC2 = parameters.minimumKappa + softplus(parameters.kappaC2Raw);
  const betaX = -2 * strength;
  const kappaX = Math.max(parameters.minimumKappa, 2 / (strength * width * width));
  const u1 = sigmoid(kappaC1 * (currentExposure - parameters.c1));
  const u2 = sigmoid(kappaC2 * (currentExposure - parameters.c2));
  const baseSlope = tilt + strength - betaC1 * u1 - betaC2 * u2
    - compactEnvelopeLogSlope(currentExposure, parameters);
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
    kappaC1,
    kappaX,
    kappaC2,
    tilt,
    strength,
    width,
    segmentSlopes: slopes as [number, number, number, number],
  };
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

function sampledCrossEntropy(
  actions: Float64Array,
  targets: Float64Array,
  states: Float64Array,
  parameters: ConditionalFourSegmentParameters,
): number {
  const model = new Float64Array(actions.length);
  let result = 0;
  for (let row = 0; row < states.length; row += 1) {
    conditionalFourSegmentExposureProbabilities(actions, states[row]!, parameters, model);
    for (let action = 0; action < actions.length; action += 1) {
      const target = targets[row * actions.length + action]!;
      if (target > 0) result -= target * Math.log(Math.max(1e-300, model[action]!));
    }
  }
  const variationPenalty = 1e-4 * (
    parameters.betaC1[1] ** 2 + parameters.betaC2[1] ** 2 + parameters.tilt[1] ** 2
      + parameters.strengthRaw[1] ** 2 + parameters.widthRaw[1] ** 2 / 100
  );
  return result / states.length + variationPenalty;
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

function parametersFromRaw(raw: Float64Array, fixed: FixedParameters): ConditionalFourSegmentParameters {
  return {
    ...fixed,
    betaC1: [raw[0]!, raw[1]!],
    betaC2: [raw[2]!, raw[3]!],
    tilt: [raw[4]!, raw[5]!],
    strengthRaw: [raw[6]!, raw[7]!],
    widthRaw: [raw[8]!, raw[9]!],
    kappaC1Raw: raw[10]!,
    kappaC2Raw: raw[11]!,
  };
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
  if (!(parameters.latentLower < parameters.c1 && parameters.c1 < parameters.c2
      && parameters.c2 < parameters.latentUpper)
    || !(parameters.visibleLower < parameters.visibleUpper)
    || parameters.visibleLower < parameters.latentLower
    || parameters.visibleUpper > parameters.latentUpper
    || !(parameters.supportSharpness > 0)
    || !(parameters.minimumStrength > 0)
    || !(parameters.minimumWidth > 0)
    || !(parameters.minimumKappa > 0)
    || !(parameters.leftSupportWidth > 0 && parameters.rightSupportWidth > 0)
    || parameters.leftSupportWidth + parameters.rightSupportWidth
      >= parameters.latentUpper - parameters.latentLower) {
    throw new Error("Conditional four-segment support or breakpoints are invalid.");
  }
}

function compactEnvelopeLogValue(
  action: number,
  parameters: ConditionalFourSegmentParameters,
): number {
  return logCompactStep(
    (action - parameters.latentLower) / parameters.leftSupportWidth,
    parameters.supportSharpness,
  ) + logCompactStep(
    (parameters.latentUpper - action) / parameters.rightSupportWidth,
    parameters.supportSharpness,
  );
}

function compactEnvelopeLogSlope(
  action: number,
  parameters: ConditionalFourSegmentParameters,
): number {
  const leftT = (action - parameters.latentLower) / parameters.leftSupportWidth;
  const rightT = (parameters.latentUpper - action) / parameters.rightSupportWidth;
  const derivative = (t: number, width: number, direction: number) => {
    if (!(t > 0 && t < 1)) return 0;
    const argument = parameters.supportSharpness * (1 / (1 - t) - 1 / t);
    const argumentDerivative = parameters.supportSharpness
      * (1 / ((1 - t) ** 2) + 1 / (t * t));
    return sigmoid(-argument) * argumentDerivative / width * direction;
  };
  // Exact latent centering is undefined at the compact endpoints. Their
  // observed visible rows use the plateau slope and remain well-defined.
  return derivative(leftT, parameters.leftSupportWidth, 1)
    + derivative(rightT, parameters.rightSupportWidth, -1);
}

function logCompactStep(t: number, sharpness: number): number {
  if (t <= 0) return Number.NEGATIVE_INFINITY;
  if (t >= 1) return 0;
  return -softplus(-sharpness * (1 / (1 - t) - 1 / t));
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

function linear(coefficients: readonly [number, number], value: number): number {
  return coefficients[0] + coefficients[1] * value;
}

function sampledIndices(length: number, maximumCount: number): number[] {
  if (length <= maximumCount) return Array.from({ length }, (_, index) => index);
  const result = new Set<number>();
  for (let index = 0; index < maximumCount; index += 1) {
    result.add(Math.round(index / (maximumCount - 1) * (length - 1)));
  }
  return [...result].sort((left, right) => left - right);
}

function learningRate(parameter: number): number {
  return parameter === 8 || parameter === 9 ? 0.35 : 0.035;
}

function clampRaw(raw: Float64Array): void {
  for (let index = 0; index < 8; index += 1) raw[index] = clamp(raw[index]!, -4, 4);
  raw[8] = clamp(raw[8]!, -8, 250);
  raw[9] = clamp(raw[9]!, -100, 100);
  raw[10] = clamp(raw[10]!, -10, 5);
  raw[11] = clamp(raw[11]!, -10, 5);
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}
