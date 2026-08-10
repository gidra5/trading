import crypto from "node:crypto";

export const KRONOS_RETURN_CALIBRATION_CONTRACT =
  "kronos-return-ridge-calibration-v1" as const;

export const KRONOS_RETURN_FEATURE_NAMES = [
  "horizonMean",
  "horizonMedian",
  "horizonStd",
  "horizonUpLogit",
  "horizonP10",
  "horizonP90",
  "horizonInterquantileRange",
  "executionMeanFraction",
  "executionDirectionalMass",
  "utilityMeanFraction",
  "utilityDirectionalMass",
  "meanPathAverage",
  "meanPathTerminal",
  "meanPathSlope",
  "medianPathAverage",
  "medianPathTerminal",
] as const;

export interface KronosReturnFeatureInput {
  row: {
    horizonLogReturnMean: number;
    horizonLogReturnMedian: number;
    horizonLogReturnStd: number;
    horizonUpProbability: number;
    horizonLogReturnP10: number;
    horizonLogReturnP90: number;
    meanCloseLogPath: readonly number[];
    medianCloseLogPath: readonly number[];
  };
  distribution: {
    grid: ArrayLike<number>;
    probabilities: ArrayLike<number>;
    mean: number;
  };
  utilityDistribution: {
    grid: ArrayLike<number>;
    probabilities: ArrayLike<number>;
    mean: number;
  };
}

export interface KronosReturnCalibrationExample {
  episodeId: string;
  decisionTime: number;
  forecast: KronosReturnFeatureInput;
  actualLogReturn: number;
}

export interface FrozenKronosReturnCalibration {
  version: 1;
  contract: typeof KRONOS_RETURN_CALIBRATION_CONTRACT;
  id: string;
  ridgeLambda: number;
  featureNames: string[];
  featureMeans: number[];
  featureScales: number[];
  coefficients: number[];
  trainingExamples: number;
  trainingEpisodeIds: string[];
  residualStd: number;
}

export interface CrossValidatedKronosReturnCalibration {
  id: string;
  ridgeLambda: number;
  predictions: Map<number, number>;
  frozen: FrozenKronosReturnCalibration;
  oofMse: number;
  oofZeroMseSkill: number;
  oofCorrelation: number;
  oofDirectionAccuracy: number;
}

export function kronosReturnFeatures(input: KronosReturnFeatureInput): number[] {
  const row = input.row;
  const probability = Math.min(1 - 1e-6, Math.max(1e-6, row.horizonUpProbability));
  const meanPath = row.meanCloseLogPath;
  const medianPath = row.medianCloseLogPath;
  if (meanPath.length === 0 || medianPath.length === 0) {
    throw new Error("Kronos return calibration requires non-empty close paths.");
  }
  return [
    row.horizonLogReturnMean,
    row.horizonLogReturnMedian,
    row.horizonLogReturnStd,
    Math.log(probability / (1 - probability)),
    row.horizonLogReturnP10,
    row.horizonLogReturnP90,
    row.horizonLogReturnP90 - row.horizonLogReturnP10,
    normalizedDistributionMean(input.distribution),
    directionalMass(input.distribution),
    normalizedDistributionMean(input.utilityDistribution),
    directionalMass(input.utilityDistribution),
    average(meanPath),
    meanPath.at(-1)!,
    meanPath.at(-1)! - meanPath[0]!,
    average(medianPath),
    medianPath.at(-1)!,
  ];
}

export function fitKronosReturnCalibration(
  examples: readonly KronosReturnCalibrationExample[],
  ridgeLambda: number,
): FrozenKronosReturnCalibration {
  if (examples.length <= KRONOS_RETURN_FEATURE_NAMES.length + 1) {
    throw new Error("Kronos return calibration has too few examples.");
  }
  if (!Number.isFinite(ridgeLambda) || ridgeLambda < 0) {
    throw new Error("ridgeLambda must be finite and non-negative.");
  }
  const features = examples.map((example) => kronosReturnFeatures(example.forecast));
  const targets = examples.map((example) => example.actualLogReturn);
  const dimension = KRONOS_RETURN_FEATURE_NAMES.length;
  const featureMeans = Array.from({ length: dimension }, (_, feature) =>
    average(features.map((row) => row[feature]!)));
  const featureScales = Array.from({ length: dimension }, (_, feature) => {
    const variance = average(features.map((row) =>
      (row[feature]! - featureMeans[feature]!) ** 2));
    return Math.max(1e-12, Math.sqrt(variance));
  });
  const design = features.map((row) => [
    1,
    ...row.map((value, feature) =>
      (value - featureMeans[feature]!) / featureScales[feature]!),
  ]);
  const coefficients = ridgeSolve(design, targets, ridgeLambda);
  const residuals = design.map((row, index) =>
    targets[index]! - dot(row, coefficients));
  const content: Omit<FrozenKronosReturnCalibration, "id"> = {
    version: 1,
    contract: KRONOS_RETURN_CALIBRATION_CONTRACT,
    ridgeLambda,
    featureNames: [...KRONOS_RETURN_FEATURE_NAMES],
    featureMeans,
    featureScales,
    coefficients,
    trainingExamples: examples.length,
    trainingEpisodeIds: [...new Set(examples.map((example) => example.episodeId))].sort(),
    residualStd: Math.sqrt(average(residuals.map((value) => value ** 2))),
  };
  return { ...content, id: calibrationId(content) };
}

export function predictKronosReturn(
  calibration: FrozenKronosReturnCalibration,
  forecast: KronosReturnFeatureInput,
): number {
  validateKronosReturnCalibration(calibration);
  const features = kronosReturnFeatures(forecast);
  const standardized = [
    1,
    ...features.map((value, index) =>
      (value - calibration.featureMeans[index]!) / calibration.featureScales[index]!),
  ];
  return dot(standardized, calibration.coefficients);
}

export function crossValidateKronosReturnCalibrations(
  examples: readonly KronosReturnCalibrationExample[],
  ridgeLambdas: readonly number[],
): CrossValidatedKronosReturnCalibration[] {
  const episodeIds = [...new Set(examples.map((example) => example.episodeId))].sort();
  if (episodeIds.length < 2) {
    throw new Error("Cross-validation requires at least two independent episodes.");
  }
  if (new Set(examples.map((example) => example.decisionTime)).size !== examples.length) {
    throw new Error("Return calibration examples contain duplicate decision times.");
  }
  return ridgeLambdas.map((ridgeLambda) => {
    const predictions = new Map<number, number>();
    for (const heldOutEpisode of episodeIds) {
      const train = examples.filter((example) => example.episodeId !== heldOutEpisode);
      const test = examples.filter((example) => example.episodeId === heldOutEpisode);
      const fold = fitKronosReturnCalibration(train, ridgeLambda);
      for (const example of test) {
        predictions.set(example.decisionTime, predictKronosReturn(fold, example.forecast));
      }
    }
    if (predictions.size !== examples.length) {
      throw new Error("Return calibration did not produce exactly one OOF prediction per row.");
    }
    const predicted = examples.map((example) => predictions.get(example.decisionTime)!);
    const actual = examples.map((example) => example.actualLogReturn);
    const mse = average(actual.map((value, index) => (value - predicted[index]!) ** 2));
    const zeroMse = average(actual.map((value) => value ** 2));
    const frozen = fitKronosReturnCalibration(examples, ridgeLambda);
    return {
      id: frozen.id,
      ridgeLambda,
      predictions,
      frozen,
      oofMse: mse,
      oofZeroMseSkill: zeroMse > 0 ? 1 - mse / zeroMse : 0,
      oofCorrelation: correlation(predicted, actual),
      oofDirectionAccuracy: average(actual.map((value, index) =>
        Math.sign(value) === Math.sign(predicted[index]!) ? 1 : 0)),
    };
  });
}

export function validateKronosReturnCalibration(
  calibration: FrozenKronosReturnCalibration,
): void {
  const dimension = KRONOS_RETURN_FEATURE_NAMES.length;
  if (calibration.version !== 1
    || calibration.contract !== KRONOS_RETURN_CALIBRATION_CONTRACT
    || typeof calibration.id !== "string"
    || !Number.isFinite(calibration.ridgeLambda)
    || calibration.ridgeLambda < 0
    || JSON.stringify(calibration.featureNames) !== JSON.stringify(KRONOS_RETURN_FEATURE_NAMES)
    || calibration.featureMeans.length !== dimension
    || calibration.featureScales.length !== dimension
    || calibration.coefficients.length !== dimension + 1
    || !calibration.featureMeans.every(Number.isFinite)
    || !calibration.featureScales.every((value) => Number.isFinite(value) && value > 0)
    || !calibration.coefficients.every(Number.isFinite)
    || !Number.isSafeInteger(calibration.trainingExamples)
    || calibration.trainingExamples <= dimension + 1
    || !Array.isArray(calibration.trainingEpisodeIds)
    || calibration.trainingEpisodeIds.length < 1
    || !Number.isFinite(calibration.residualStd)
    || calibration.residualStd < 0) {
    throw new Error("Invalid frozen Kronos return calibration.");
  }
  const { id, ...content } = calibration;
  if (id !== calibrationId(content)) {
    throw new Error("Frozen Kronos return calibration fingerprint does not match its content.");
  }
}

function calibrationId(content: Omit<FrozenKronosReturnCalibration, "id">): string {
  const digest = crypto.createHash("sha256").update(JSON.stringify(content)).digest("hex");
  return `ridge-${content.ridgeLambda.toExponential(0).replace("+", "")}-${digest.slice(0, 12)}`;
}

function normalizedDistributionMean(distribution: KronosReturnFeatureInput["distribution"]): number {
  const bound = Math.max(
    ...Array.from(distribution.grid, (value) => Math.abs(Number(value))),
  );
  return bound > 0 ? distribution.mean / bound : 0;
}

function directionalMass(distribution: KronosReturnFeatureInput["distribution"]): number {
  let total = 0;
  let signed = 0;
  for (let index = 0; index < distribution.grid.length; index += 1) {
    const probability = Number(distribution.probabilities[index]);
    total += probability;
    signed += probability * Math.sign(Number(distribution.grid[index]));
  }
  return total > 0 ? signed / total : 0;
}

function ridgeSolve(design: readonly number[][], targets: readonly number[], lambda: number): number[] {
  const dimension = design[0]!.length;
  const gram = Array.from({ length: dimension }, () => Array(dimension).fill(0) as number[]);
  const rhs = Array(dimension).fill(0) as number[];
  for (let row = 0; row < design.length; row += 1) {
    for (let left = 0; left < dimension; left += 1) {
      rhs[left] += design[row]![left]! * targets[row]!;
      for (let right = 0; right < dimension; right += 1) {
        gram[left]![right] += design[row]![left]! * design[row]![right]!;
      }
    }
  }
  const inverseExamples = 1 / design.length;
  for (let left = 0; left < dimension; left += 1) {
    rhs[left] *= inverseExamples;
    for (let right = 0; right < dimension; right += 1) {
      gram[left]![right] *= inverseExamples;
    }
  }
  for (let index = 1; index < dimension; index += 1) gram[index]![index] += lambda;
  return solveLinearSystem(gram, rhs);
}

function solveLinearSystem(matrix: number[][], values: number[]): number[] {
  const size = values.length;
  const augmented = matrix.map((row, index) => [...row, values[index]!]);
  for (let column = 0; column < size; column += 1) {
    let pivot = column;
    for (let row = column + 1; row < size; row += 1) {
      if (Math.abs(augmented[row]![column]!) > Math.abs(augmented[pivot]![column]!)) pivot = row;
    }
    if (Math.abs(augmented[pivot]![column]!) < 1e-14) {
      throw new Error("Kronos return calibration matrix is singular.");
    }
    [augmented[column], augmented[pivot]] = [augmented[pivot]!, augmented[column]!];
    const divisor = augmented[column]![column]!;
    for (let entry = column; entry <= size; entry += 1) augmented[column]![entry]! /= divisor;
    for (let row = 0; row < size; row += 1) {
      if (row === column) continue;
      const factor = augmented[row]![column]!;
      for (let entry = column; entry <= size; entry += 1) {
        augmented[row]![entry]! -= factor * augmented[column]![entry]!;
      }
    }
  }
  return augmented.map((row) => row[size]!);
}

function correlation(left: readonly number[], right: readonly number[]): number {
  const leftMean = average(left);
  const rightMean = average(right);
  let covariance = 0;
  let leftVariance = 0;
  let rightVariance = 0;
  for (let index = 0; index < left.length; index += 1) {
    const leftDelta = left[index]! - leftMean;
    const rightDelta = right[index]! - rightMean;
    covariance += leftDelta * rightDelta;
    leftVariance += leftDelta ** 2;
    rightVariance += rightDelta ** 2;
  }
  return leftVariance > 0 && rightVariance > 0
    ? covariance / Math.sqrt(leftVariance * rightVariance)
    : 0;
}

function dot(left: readonly number[], right: readonly number[]): number {
  return left.reduce((sum, value, index) => sum + value * right[index]!, 0);
}

function average(values: readonly number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}
