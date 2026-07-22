import {
  conditionalFourSegmentExposureProbabilities,
  conditionalFourSegmentParametersFromRaw,
  type ConditionalFourSegmentParameters,
} from "./conditional-exposure-distribution.js";
import type { Candle } from "./legacy/types.js";

export const MLP_FEATURE_SCHEMA_VERSION = 4;
export const MLP_OUTPUT_PARAMETER_COUNT = 8;
export const MLP_SUPPORTED_OUTPUT_PARAMETER_COUNTS = [6, MLP_OUTPUT_PARAMETER_COUNT] as const;
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

export const MLP_STATE_FEATURES = [
  "feeRate",
  "minimumUsableExposure",
  "maximumUsableExposure",
  "minimumEffectiveExposure",
  "maximumEffectiveExposure",
  "quoteLendRate",
  "quoteBorrowRate",
  "assetBorrowRate",
] as const;

export const MLP_CANDLE_INPUT_COUNT = MLP_CANDLE_WINDOWS.reduce(
  (sum, window) => sum + window.candleCount,
  0,
);
export const MLP_CANDLE_FILL_FRACTION_COUNT = MLP_CANDLE_WINDOWS.length - 1;
export const MLP_HISTORIC_INPUT_COUNT = MLP_CANDLE_INPUT_COUNT * MLP_CANDLE_FEATURE_COUNT
  + MLP_CANDLE_FILL_FRACTION_COUNT;
export const MLP_INPUT_FEATURE_COUNT = MLP_HISTORIC_INPUT_COUNT
  + MLP_STATE_FEATURES.length;

export interface MlpExposureStateInputs {
  feeRate: number;
  minimumUsableExposure: number;
  maximumUsableExposure: number;
  minimumEffectiveExposure: number;
  maximumEffectiveExposure: number;
  quoteLendRate: number;
  quoteBorrowRate: number;
  assetBorrowRate: number;
}

export interface MlpModelManifest {
  id: string;
  label: string;
  createdAt: string;
  featureSchemaVersion: number;
  inputFeatureCount: number;
  outputParameterCount: number;
  hiddenLayerCount: number;
  hiddenWidth: number;
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
    finalizedEarly?: boolean;
    seed: number;
    device: string;
  };
}

export interface MlpTrainingMetrics {
  loss: number;
  crossEntropy: number;
  probabilityMse: number;
  parameterMse: number;
  excessEntropy: number;
  stateMutualInformation: number;
  oracleMutualInformation: number;
  targetEntropy: number;
  predictedEntropy: number;
  rawParameterMae: number;
  distanceImbalanceWeight?: number;
  timeWeightEffectiveSampleRatio?: number;
}

export interface MlpConditionalPrediction {
  probabilities: Float64Array;
  optimalExposure: number;
  meanExposure: number;
  conditionalParameters: ConditionalFourSegmentParameters;
  rawParameters: Float64Array;
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

export function encodeMlpStateInputs(
  state: MlpExposureStateInputs,
  output = new Float32Array(MLP_STATE_FEATURES.length),
  outputOffset = 0,
): Float32Array {
  if (outputOffset < 0 || outputOffset + MLP_STATE_FEATURES.length > output.length) {
    throw new Error("MLP state encoding output storage is too small.");
  }
  for (let index = 0; index < MLP_STATE_FEATURES.length; index += 1) {
    const value = state[MLP_STATE_FEATURES[index]!];
    if (!Number.isFinite(value)) throw new Error("MLP state inputs must be finite.");
    output[outputOffset + index] = value;
  }
  return output;
}

export function predictMlpConditionalDistribution(
  actionGridInput: ArrayLike<number>,
  currentExposure: number,
  rawParameterInput: ArrayLike<number>,
  support: {
    latentLower: number;
    latentUpper: number;
    visibleLower?: number;
    visibleUpper?: number;
    friction: number;
    temperature: number;
  },
): MlpConditionalPrediction {
  if (actionGridInput.length < 5 || !Number.isFinite(currentExposure)) {
    throw new Error("MLP exposure prediction requires an action grid and finite current exposure.");
  }
  const actionGrid = Float64Array.from(actionGridInput);
  const visibleLower = support.visibleLower ?? actionGrid[0]!;
  const visibleUpper = support.visibleUpper ?? actionGrid[actionGrid.length - 1]!;
  const rawParameters = Float64Array.from(rawParameterInput);
  const conditionalParameters = conditionalFourSegmentParametersFromRaw(rawParameters, {
    latentLower: support.latentLower,
    latentUpper: support.latentUpper,
    visibleLower,
    visibleUpper,
    friction: support.friction,
    temperature: support.temperature,
  });
  const probabilities = conditionalFourSegmentExposureProbabilities(
    actionGrid,
    currentExposure,
    conditionalParameters,
  );
  let optimalIndex = 0;
  let meanExposure = 0;
  for (let index = 0; index < actionGrid.length; index += 1) {
    if (probabilities[index]! > probabilities[optimalIndex]!) optimalIndex = index;
    meanExposure += probabilities[index]! * actionGrid[index]!;
  }
  return {
    probabilities,
    optimalExposure: actionGrid[optimalIndex]!,
    meanExposure,
    conditionalParameters,
    rawParameters,
  };
}

export function validateMlpModelManifest(manifest: MlpModelManifest): void {
  if (!manifest.id || !manifest.label || !manifest.modelFile
    || !Number.isFinite(Date.parse(manifest.createdAt))
    || manifest.featureSchemaVersion !== MLP_FEATURE_SCHEMA_VERSION
    || manifest.inputFeatureCount !== MLP_INPUT_FEATURE_COUNT
    || !MLP_SUPPORTED_OUTPUT_PARAMETER_COUNTS.includes(
      manifest.outputParameterCount as typeof MLP_SUPPORTED_OUTPUT_PARAMETER_COUNTS[number],
    )
    || manifest.hiddenLayerCount !== 16
    || manifest.hiddenWidth !== 1_024) {
    throw new Error("MLP model manifest is incompatible with the current 16x1024 feature contract.");
  }
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
    || !validTrainingMetrics(training.testMetrics))) {
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
