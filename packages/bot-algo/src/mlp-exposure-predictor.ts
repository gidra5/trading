import type { Candle } from "./legacy/types.js";

export const MLP_FEATURE_SCHEMA_VERSION = 5;
export const MLP_OUTPUT_ACTION_COUNT = 255;
export const MLP_CANDLE_FEATURE_COUNT = 4;
export const MLP_VOLUME_EMA_WARMUP_MULTIPLE = 4;

export const MLP_CANDLE_WINDOWS = [
  { id: "1s", intervalMs: 1_000, candleCount: 64 },
  { id: "1m", intervalMs: 60_000, candleCount: 64 },
  { id: "1h", intervalMs: 3_600_000, candleCount: 32 },
  { id: "1d", intervalMs: 86_400_000, candleCount: 32 },
  { id: "1M", intervalMs: "calendar-month", candleCount: 16 },
  { id: "3M", intervalMs: "calendar-quarter", candleCount: 16 },
] as const;

export const MLP_CANDLE_INPUT_COUNT = MLP_CANDLE_WINDOWS.reduce(
  (sum, window) => sum + window.candleCount,
  0,
);
export const MLP_CANDLE_FILL_FRACTION_COUNT = MLP_CANDLE_WINDOWS.length - 1;
export const MLP_HISTORIC_INPUT_COUNT = MLP_CANDLE_INPUT_COUNT * MLP_CANDLE_FEATURE_COUNT
  + MLP_CANDLE_FILL_FRACTION_COUNT;
export const MLP_INPUT_FEATURE_COUNT = MLP_HISTORIC_INPUT_COUNT;

export interface MlpModelManifest {
  id: string;
  label: string;
  createdAt: string;
  featureSchemaVersion: number;
  inputFeatureCount: number;
  outputRepresentation: "base-action-logits";
  outputActionCount: number;
  actionGrid: number[];
  hiddenLayerCount: number;
  hiddenWidth: number;
  predictionDelayMs: number;
  modelFile: string;
  checkpointFile?: string;
  verificationFixture?: {
    batchSize: number;
    inputFile: string;
    outputFile: string;
  };
  verification?: {
    executionProvider: "cuda" | "cpu";
    maxAbsolutePyTorchError: number;
    maxAbsoluteProviderError: number;
    verifiedAt: string;
  };
  training?: {
    datasetPlanId?: string;
    targetRepresentation?: "rawOracleProbabilities" | "minuteOracleProbabilities";
    trainExamples: number;
    validationExamples: number;
    testExamples: number;
    bestEpoch: number;
    bestValidationLoss: number;
    testLoss: number;
    bestValidationMetrics?: MlpTrainingMetrics;
    testMetrics?: MlpTrainingMetrics;
    teacherFitMetrics?: Record<string, number>;
    lossWeights?: Record<string, number>;
    selectionMetric?: "loss" | "klDivergence";
    policyMetricDefinitions?: {
      klDivergence: string;
      klDivergenceVariance?: string;
      baseKlDivergence: string;
      probabilityMse?: string;
      probabilityMseVariance?: string;
    };
    predictionDelayMs?: number;
    curriculum?: {
      version: number;
      stage: number;
      delayMs: number;
      parentKey: string;
      weightProfile: string;
      lineage: Array<{
        stage: number;
        delayMs: number;
        weights: Record<string, number>;
      }>;
    };
    finalizedEarly?: boolean;
    seed: number;
    device: string;
  };
}

export interface MlpTrainingMetrics {
  loss: number;
  klDivergence: number;
  klDivergenceVariance?: number;
  klDivergenceStdDev?: number;
  baseKlDivergence: number;
  probabilityMse: number;
  probabilityMseVariance?: number;
  probabilityMseStdDev?: number;
  excessEntropy: number;
  temporalMutualInformation: number;
  targetTemporalMutualInformation: number;
  temporalMutualInformationReward: number;
  oracleMutualInformation: number;
  targetEntropy: number;
  predictedEntropy: number;
  distanceImbalanceWeight?: number;
  timeWeightEffectiveSampleRatio?: number;
}

export interface MlpDistributionPrediction {
  probabilities: Float64Array;
  optimalExposure: number;
  meanExposure: number;
  actionLogits: Float64Array;
}

/**
 * Encode the last N causal candles as close/open log return, upward and
 * downward log deviations from the geometric candle midpoint, and log volume
 * relative to a slow causal EMA. Older candles may be supplied to warm the EMA;
 * only the final expectedCount candles are emitted.
 */
export function encodeMlpCandleWindow(
  candles: readonly Pick<Candle, "open" | "high" | "low" | "close" | "volume">[],
  expectedCount: number,
  output = new Float32Array(expectedCount * MLP_CANDLE_FEATURE_COUNT),
  outputOffset = 0,
): Float32Array {
  if (!Number.isInteger(expectedCount) || expectedCount <= 0
    || outputOffset < 0
    || outputOffset + expectedCount * MLP_CANDLE_FEATURE_COUNT > output.length) {
    throw new Error("MLP candle encoding requires a positive window and sufficient output storage.");
  }
  const observed = candles.slice(-expectedCount * MLP_VOLUME_EMA_WARMUP_MULTIPLE);
  const emitStart = Math.max(0, observed.length - expectedCount);
  const padding = expectedCount - (observed.length - emitStart);
  const volumeAlpha = 2 / (expectedCount * MLP_VOLUME_EMA_WARMUP_MULTIPLE + 1);
  let volumeEma = 0;
  for (let index = 0; index < observed.length; index += 1) {
    const candle = observed[index]!;
    validateMlpCandle(candle);
    volumeEma = index === 0
      ? candle.volume
      : volumeAlpha * candle.volume + (1 - volumeAlpha) * volumeEma;
    if (index < emitStart) continue;
    const destination = outputOffset
      + (padding + index - emitStart) * MLP_CANDLE_FEATURE_COUNT;
    const logOpen = Math.log(candle.open);
    const logClose = Math.log(candle.close);
    const logMiddle = (logOpen + logClose) / 2;
    output[destination] = logClose - logOpen;
    output[destination + 1] = Math.max(0, Math.log(candle.high) - logMiddle);
    output[destination + 2] = Math.max(0, logMiddle - Math.log(candle.low));
    output[destination + 3] = volumeEma > 0 && candle.volume > 0
      ? Math.log(candle.volume / volumeEma)
      : 0;
  }
  return output;
}

export function predictMlpDistribution(
  actionGridInput: ArrayLike<number>,
  actionLogitInput: ArrayLike<number>,
  modelActionGridInput: ArrayLike<number>,
): MlpDistributionPrediction {
  if (actionGridInput.length < 5
    || actionLogitInput.length !== MLP_OUTPUT_ACTION_COUNT
    || modelActionGridInput.length !== MLP_OUTPUT_ACTION_COUNT) {
    throw new Error("MLP exposure prediction requires compatible runtime and model action grids.");
  }
  const actionGrid = Float64Array.from(actionGridInput);
  const modelActionGrid = Float64Array.from(modelActionGridInput);
  const actionLogits = Float64Array.from(actionLogitInput);
  validateStrictlyIncreasingGrid(actionGrid, "runtime");
  validateStrictlyIncreasingGrid(modelActionGrid, "model");
  if (![...actionLogits].every(Number.isFinite)
    || actionGrid[0]! < modelActionGrid[0]!
    || actionGrid[actionGrid.length - 1]! > modelActionGrid[modelActionGrid.length - 1]!) {
    throw new Error("MLP action logits or runtime action-grid range are invalid.");
  }
  const interpolatedLogits = interpolateActionLogits(
    actionGrid,
    modelActionGrid,
    actionLogits,
  );
  const maximum = Math.max(...interpolatedLogits);
  const probabilities = new Float64Array(actionGrid.length);
  let total = 0;
  for (let index = 0; index < probabilities.length; index += 1) {
    const probability = Math.exp(interpolatedLogits[index]! - maximum);
    probabilities[index] = probability;
    total += probability;
  }
  let optimalIndex = 0;
  let meanExposure = 0;
  for (let index = 0; index < actionGrid.length; index += 1) {
    probabilities[index] /= total;
    if (probabilities[index]! > probabilities[optimalIndex]!) optimalIndex = index;
    meanExposure += probabilities[index]! * actionGrid[index]!;
  }
  return {
    probabilities,
    optimalExposure: actionGrid[optimalIndex]!,
    meanExposure,
    actionLogits,
  };
}

export function validateMlpModelManifest(manifest: MlpModelManifest): void {
  if (!manifest.id || !manifest.label || !manifest.modelFile
    || !Number.isFinite(Date.parse(manifest.createdAt))
    || manifest.featureSchemaVersion !== MLP_FEATURE_SCHEMA_VERSION
    || manifest.inputFeatureCount !== MLP_INPUT_FEATURE_COUNT
    || manifest.outputRepresentation !== "base-action-logits"
    || manifest.outputActionCount !== MLP_OUTPUT_ACTION_COUNT
    || !Array.isArray(manifest.actionGrid)
    || manifest.actionGrid.length !== MLP_OUTPUT_ACTION_COUNT
    || manifest.hiddenLayerCount !== 16
    || manifest.hiddenWidth !== 1_024
    || !Number.isInteger(manifest.predictionDelayMs)
    || manifest.predictionDelayMs < 0) {
    throw new Error(
      "MLP model manifest is incompatible with the current 901 -> 16x1024 -> 255 contract.",
    );
  }
  validateStrictlyIncreasingGrid(manifest.actionGrid, "manifest");
  const fixture = manifest.verificationFixture;
  if (fixture && (!Number.isInteger(fixture.batchSize) || fixture.batchSize <= 0
    || !fixture.inputFile || !fixture.outputFile)) {
    throw new Error("MLP model manifest has an invalid verification fixture.");
  }
  const training = manifest.training;
  if (training && (![training.trainExamples, training.validationExamples, training.testExamples]
    .every((value) => Number.isInteger(value) && value >= 0)
    || ![
      training.bestEpoch,
      training.bestValidationLoss,
      training.testLoss,
      training.seed,
    ].every(Number.isFinite)
    || !training.device
    || !validTrainingMetrics(training.bestValidationMetrics)
    || !validTrainingMetrics(training.testMetrics)
    || (training.targetRepresentation !== undefined
      && training.targetRepresentation !== "rawOracleProbabilities"
      && training.targetRepresentation !== "minuteOracleProbabilities")
    || (training.selectionMetric !== undefined
      && training.selectionMetric !== "loss"
      && training.selectionMetric !== "klDivergence")
    || (training.lossWeights !== undefined
      && !Object.values(training.lossWeights).every((value) =>
        Number.isFinite(value) && value >= 0)))) {
    throw new Error("MLP model manifest has invalid training metadata.");
  }
  const verification = manifest.verification;
  if (verification && ((verification.executionProvider !== "cuda"
    && verification.executionProvider !== "cpu")
    || !Number.isFinite(verification.maxAbsolutePyTorchError)
    || verification.maxAbsolutePyTorchError < 0
    || !Number.isFinite(verification.maxAbsoluteProviderError)
    || verification.maxAbsoluteProviderError < 0
    || !Number.isFinite(Date.parse(verification.verifiedAt)))) {
    throw new Error("MLP model manifest has invalid verification metadata.");
  }
}

function validTrainingMetrics(metrics: MlpTrainingMetrics | undefined): boolean {
  return metrics === undefined || Object.values(metrics).every(Number.isFinite);
}

function interpolateActionLogits(
  targetGrid: Float64Array,
  sourceGrid: Float64Array,
  sourceLogits: Float64Array,
): Float64Array {
  const result = new Float64Array(targetGrid.length);
  let source = 0;
  while (source < sourceGrid.length - 1 && sourceGrid[source]! < targetGrid[0]!) {
    source += 1;
  }
  let sourceEnd = sourceGrid.length - 1;
  while (sourceEnd > source && sourceGrid[sourceEnd]! > targetGrid[targetGrid.length - 1]!) {
    sourceEnd -= 1;
  }
  for (let target = 0; target < targetGrid.length; target += 1) {
    const action = targetGrid[target]!;
    while (source + 1 <= sourceEnd && sourceGrid[source + 1]! < action) source += 1;
    const next = Math.min(source + 1, sourceEnd);
    const lower = sourceGrid[source]!;
    const upper = sourceGrid[next]!;
    const fraction = upper > lower
      ? Math.max(0, Math.min(1, (action - lower) / (upper - lower)))
      : 0;
    result[target] = sourceLogits[source]!
      + fraction * (sourceLogits[next]! - sourceLogits[source]!);
  }
  return result;
}

function validateStrictlyIncreasingGrid(
  grid: ArrayLike<number>,
  name: string,
): void {
  for (let index = 0; index < grid.length; index += 1) {
    if (!Number.isFinite(grid[index])
      || (index > 0 && grid[index]! <= grid[index - 1]!)) {
      throw new Error(`MLP ${name} action grid must be finite and strictly increasing.`);
    }
  }
}

function validateMlpCandle(
  candle: Pick<Candle, "open" | "high" | "low" | "close" | "volume">,
): void {
  if (![candle.open, candle.high, candle.low, candle.close, candle.volume].every(Number.isFinite)
    || candle.open <= 0 || candle.high <= 0 || candle.low <= 0 || candle.close <= 0
    || candle.volume < 0 || candle.high < Math.max(candle.open, candle.close)
    || candle.low > Math.min(candle.open, candle.close)) {
    throw new Error("MLP candle features require finite, positive, internally consistent OHLCV values.");
  }
}
