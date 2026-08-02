import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { isDeepStrictEqual } from "node:util";
import * as ort from "onnxruntime-node";
import type { ExposureValueOracleActionDistribution } from "@trading/bot-algo";

const DEFAULT_BATCH_SIZE = 512;
export const JOINT_PRICE_ORACLE_CONTEXT_LENGTH = 3_600;
export const JOINT_PRICE_ORACLE_DECISION_INTERVAL_MS = 60_000;
export const JOINT_PRICE_ORACLE_DECISION_PHASE_MS = 999;
export const JOINT_PRICE_ORACLE_DATA_CONTRACT =
  "causal-1s-close-context-ending-at-minute-t-future-closes-t-plus-1-through-1h-verified-oracle-policy-at-t-hold-60-delay-60-v2";
export const JOINT_PRICE_ORACLE_FRICTION = 0.00175;
export const JOINT_PRICE_ORACLE_TEACHER_TEMPERATURE = 0.01;
export const JOINT_PRICE_ORACLE_ACTION_CALIBRATION_OBJECTIVE =
  "actions.signedTransitionF1";
export const JOINT_PRICE_ORACLE_EXACT_STATE_CALIBRATION_OBJECTIVE =
  "exactStateActions.signedTransitionF1";
export const JOINT_PRICE_ORACLE_ACTION_CALIBRATION_TIE_BREAKERS = [
  "actions.signedTransitionPrecision:maximize",
  "actions.signedTransitionRecall:maximize",
  "actions.exactTransitionF1:maximize",
  "actions.pathDirectionalAgreement:maximize",
  "actions.pathMeanAbsoluteError:minimize",
  "actions.turnoverRatioDistanceFromOne:minimize",
  "logitTemperatureLogDistanceFromIdentity:minimize",
  "logitTemperature:minimize",
] as const;
export const JOINT_PRICE_ORACLE_EXACT_STATE_CALIBRATION_TIE_BREAKERS = [
  "exactStateActions.signedTransitionPrecision:maximize",
  "exactStateActions.signedTransitionRecall:maximize",
  "exactStateActions.exactTransitionF1:maximize",
  "exactStateActions.executableTargetDirectionalAgreement:maximize",
  "exactStateActions.executableTargetMeanAbsoluteError:minimize",
  "exactStateActions.turnoverRelativeError:minimize",
  "logitTemperatureLogDistanceFromIdentity:minimize",
  "logitTemperature:minimize",
] as const;

export interface OneSecondClose {
  closeTime: number;
  close: number;
}

export interface JointPriceOracleManifest {
  version: 2 | 3;
  kind: "joint-price-oracle";
  id: string;
  label: string;
  createdAt: string;
  modelFile: string;
  modelSha256: string;
  architectureContract: string;
  dataContract: string;
  input: {
    name: "closes";
    dtype: "float32";
    intervalMs: 1_000;
    contextLength: number;
    variableCount: 1;
  };
  output: {
    name: "action_logits";
    dtype: "float32";
    actionCount: number;
    actionGrid: number[];
  };
  oracle: {
    intervalMs: 1_000;
    decisionIntervalMs: 60_000;
    decisionPhaseMs: 999;
    options: {
      holdingPeriodSteps: 60;
      decisionDelaySteps: 60;
      valueHorizonSteps: 3_600;
      friction: number;
      temperature: number;
    };
  };
  training: {
    bestEpoch: number;
    globalStep: number;
    validation: Record<string, number | boolean | null>;
    test?: Record<string, number | boolean | null>;
    datasetFingerprint: string;
  };
  calibration: {
    method: "identity" | "validation-kl-temperature-scaling"
      | "validation-action-temperature-scaling";
    logitTemperature: number;
    validationKlBefore?: number;
    validationKlAfter?: number;
    calibratedAt?: string;
    objective?: typeof JOINT_PRICE_ORACLE_ACTION_CALIBRATION_OBJECTIVE
      | typeof JOINT_PRICE_ORACLE_EXACT_STATE_CALIBRATION_OBJECTIVE;
    tieBreakers?: string[];
    validationReportFile?: string;
    validationReportSha256?: string;
    validationReportVersion?: number;
    validationExamples?: number;
    candidateCount?: number;
    checkpoint?: {
      kind: "best";
      epoch: number;
      globalStep: number;
    };
    datasetFingerprint?: string;
    executionPolicy?: JointPriceOracleExecutionPolicy | null;
    rolloutScoreVersion?: 1 | 2;
    rolloutResetPolicy?: "reset-only-at-true-timeline-gaps";
    selectedMetrics?: JointPriceOracleCalibrationMetrics;
    identityMetrics?: JointPriceOracleCalibrationMetrics;
  };
}

export interface JointPriceOracleCalibrationMetrics {
  raw: Record<string, number | null>;
  actions: Record<string, number | null>;
}

export interface JointPriceOracleExecutionPolicy {
  version: 2;
  maximumLeverage: number;
  minimumConfidence: number;
  confidenceExposurePower: number;
  confidenceLeverageFloor: number;
}

interface JointPriceOracleArtifact {
  directory: string;
  manifest: JointPriceOracleManifest;
}

interface LoadedJointPriceOracleArtifact extends JointPriceOracleArtifact {
  session: ort.InferenceSession;
}

export class JointPriceOracleRuntime {
  private loaded?: Promise<LoadedJointPriceOracleArtifact>;

  constructor(
    private readonly dataDir: string,
    private readonly artifactId?: string,
    private readonly logitTemperatureOverride?: number,
  ) {}

  models(): JointPriceOracleManifest[] {
    return discoverJointPriceOracleArtifacts(this.dataDir).map(({ manifest }) => manifest);
  }

  async predictDistributions(
    oneSecondCandles: readonly OneSecondClose[],
    times: readonly number[],
  ): Promise<ExposureValueOracleActionDistribution[]> {
    if (times.length === 0) return [];
    const artifact = await this.model();
    const contextLength = artifact.manifest.input.contextLength;
    const actionCount = artifact.manifest.output.actionCount;
    const logitTemperature = this.logitTemperatureOverride
      ?? artifact.manifest.calibration.logitTemperature;
    if (!(logitTemperature > 0) || !Number.isFinite(logitTemperature)) {
      throw new Error("Joint price-oracle logit temperature must be finite and positive.");
    }
    const indexes = candleIndexes(oneSecondCandles, times, contextLength);
    const result: ExposureValueOracleActionDistribution[] = new Array(times.length);
    const requestedBatch = Number(
      process.env.TRADING_JOINT_PRICE_ORACLE_BATCH_SIZE ?? DEFAULT_BATCH_SIZE,
    );
    const batchSize = Number.isFinite(requestedBatch) && requestedBatch > 0
      ? Math.max(1, Math.floor(requestedBatch))
      : DEFAULT_BATCH_SIZE;
    for (let start = 0; start < times.length; start += batchSize) {
      const end = Math.min(times.length, start + batchSize);
      const closes = new Float32Array((end - start) * contextLength);
      for (let row = start; row < end; row += 1) {
        const candleEnd = indexes[row]!;
        const candleStart = candleEnd - contextLength + 1;
        const outputOffset = (row - start) * contextLength;
        for (let index = 0; index < contextLength; index += 1) {
          closes[outputOffset + index] = oneSecondCandles[candleStart + index]!.close;
        }
      }
      const tensor = new ort.Tensor(
        "float32",
        closes,
        [end - start, contextLength, 1],
      );
      const inference = await artifact.session.run(
        { closes: tensor },
        ["action_logits"],
      );
      const output = inference.action_logits;
      if (!output || output.type !== "float32"
        || output.dims.length !== 2
        || output.dims[0] !== end - start
        || output.dims[1] !== actionCount
        || !(output.data instanceof Float32Array)) {
        throw new Error(
          `Joint price-oracle output must be [batch, ${actionCount}] float32 logits.`,
        );
      }
      for (let row = 0; row < end - start; row += 1) {
        result[start + row] = logitsDistribution(
          output.data.subarray(row * actionCount, (row + 1) * actionCount),
          artifact.manifest.output.actionGrid,
          logitTemperature,
        );
      }
    }
    return result;
  }

  async predictLatest(
    oneSecondCandles: readonly OneSecondClose[],
  ): Promise<ExposureValueOracleActionDistribution | null> {
    const latest = oneSecondCandles.at(-1);
    if (!latest || !isJointPriceOracleDecisionTime(latest.closeTime)) return null;
    return (await this.predictDistributions(
      oneSecondCandles,
      [latest.closeTime],
    ))[0]!;
  }

  private model(): Promise<LoadedJointPriceOracleArtifact> {
    return this.loaded ??= this.loadModel();
  }

  private async loadModel(): Promise<LoadedJointPriceOracleArtifact> {
    const artifacts = discoverJointPriceOracleArtifacts(this.dataDir);
    const artifact = this.artifactId
      ? artifacts.find(({ manifest }) => manifest.id === this.artifactId)
      : artifacts[0];
    if (!artifact) {
      throw new Error(
        this.artifactId
          ? `Unknown joint price-oracle artifact '${this.artifactId}'.`
          : "No trained joint price-oracle artifact is available.",
      );
    }
    const modelFile = path.resolve(artifact.directory, artifact.manifest.modelFile);
    const expectedRoot = `${path.resolve(artifact.directory)}${path.sep}`;
    if (!modelFile.startsWith(expectedRoot) || !fs.statSync(modelFile).isFile()) {
      throw new Error("Joint price-oracle modelFile leaves its artifact directory.");
    }
    const actualHash = crypto.createHash("sha256")
      .update(fs.readFileSync(modelFile))
      .digest("hex");
    if (actualHash !== artifact.manifest.modelSha256) {
      throw new Error("Joint price-oracle ONNX hash does not match its manifest.");
    }
    const preference = process.env.TRADING_JOINT_PRICE_ORACLE_EXECUTION_PROVIDER
      ?.trim().toLowerCase() ?? "auto";
    if (preference !== "auto" && preference !== "cuda" && preference !== "cpu") {
      throw new Error(
        "TRADING_JOINT_PRICE_ORACLE_EXECUTION_PROVIDER must be auto, cuda, or cpu.",
      );
    }
    let session: ort.InferenceSession | undefined;
    if (preference !== "cpu") {
      try {
        session = await ort.InferenceSession.create(modelFile, {
          executionProviders: [{ name: "cuda", deviceId: 0 }],
          graphOptimizationLevel: "all",
          executionMode: "sequential",
          enableMemPattern: true,
        });
      } catch (error) {
        if (preference === "cuda") throw error;
      }
    }
    session ??= await ort.InferenceSession.create(modelFile, {
      executionProviders: ["cpu"],
      graphOptimizationLevel: "all",
      executionMode: "parallel",
    });
    if (session.inputNames.length !== 1 || session.inputNames[0] !== "closes"
      || session.outputNames.length !== 1 || session.outputNames[0] !== "action_logits") {
      throw new Error("Joint price-oracle ONNX graph must expose closes -> action_logits.");
    }
    return { ...artifact, session };
  }
}

export function discoverJointPriceOracleArtifacts(
  dataDir: string,
): JointPriceOracleArtifact[] {
  const root = path.join(dataDir, "models", "joint-price-oracle");
  let entries: fs.Dirent[];
  try {
    entries = fs.readdirSync(root, { withFileTypes: true });
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return [];
    throw error;
  }
  const artifacts: JointPriceOracleArtifact[] = [];
  for (const entry of entries) {
    if (!entry.isDirectory() || entry.isSymbolicLink()) continue;
    const directory = path.join(root, entry.name);
    const manifestFile = path.join(directory, "manifest.json");
    if (!fs.existsSync(manifestFile)) continue;
    const manifest = JSON.parse(
      fs.readFileSync(manifestFile, "utf8"),
    ) as JointPriceOracleManifest;
    validateJointPriceOracleManifest(manifest);
    validateJointPriceOracleCalibrationArtifact(directory, manifest);
    artifacts.push({ directory, manifest });
  }
  return artifacts.sort((left, right) =>
    Date.parse(right.manifest.createdAt) - Date.parse(left.manifest.createdAt));
}

export function validateJointPriceOracleManifest(
  manifest: JointPriceOracleManifest,
): void {
  if ((manifest.version !== 2 && manifest.version !== 3)
    || manifest.kind !== "joint-price-oracle"
    || !manifest.id || !manifest.modelFile || !/^[a-f0-9]{64}$/.test(manifest.modelSha256)
    || !manifest.architectureContract?.trim()
    || manifest.dataContract !== JOINT_PRICE_ORACLE_DATA_CONTRACT
    || manifest.input?.name !== "closes" || manifest.input.dtype !== "float32"
    || manifest.input.intervalMs !== 1_000
    || manifest.input.contextLength !== JOINT_PRICE_ORACLE_CONTEXT_LENGTH
    || manifest.input.variableCount !== 1
    || manifest.output?.name !== "action_logits" || manifest.output.dtype !== "float32"
    || manifest.output.actionCount !== 101
    || !validJointPriceOracleGrid(manifest.output.actionGrid)
    || manifest.oracle?.intervalMs !== 1_000
    || manifest.oracle.decisionIntervalMs !== JOINT_PRICE_ORACLE_DECISION_INTERVAL_MS
    || manifest.oracle.decisionPhaseMs !== JOINT_PRICE_ORACLE_DECISION_PHASE_MS
    || manifest.oracle.options?.holdingPeriodSteps !== 60
    || manifest.oracle.options.decisionDelaySteps !== 60
    || manifest.oracle.options.valueHorizonSteps !== 3_600
    || manifest.oracle.options.friction !== JOINT_PRICE_ORACLE_FRICTION
    || manifest.oracle.options.temperature !== JOINT_PRICE_ORACLE_TEACHER_TEMPERATURE
    || !(manifest.calibration?.logitTemperature > 0)
    || !Number.isFinite(manifest.calibration.logitTemperature)
    || !validCalibrationManifest(manifest)) {
    throw new Error("Joint price-oracle manifest does not satisfy the 1h/1m/1m contract.");
  }
}

function validJointPriceOracleGrid(value: readonly number[] | undefined): boolean {
  if (value?.length !== 101) return false;
  const step = 500 / 254;
  return value.every((item, index) =>
    Number.isFinite(item) && Math.abs(item - (index - 50) * step) <= 1e-10);
}

function validCalibrationManifest(manifest: JointPriceOracleManifest): boolean {
  const calibration = manifest.calibration;
  if (calibration.method === "identity") {
    return calibration.logitTemperature === 1;
  }
  if (calibration.method === "validation-kl-temperature-scaling") {
    return manifest.version === 2
      && Number.isFinite(calibration.validationKlBefore)
      && Number.isFinite(calibration.validationKlAfter)
      && typeof calibration.calibratedAt === "string";
  }
  const expectedTieBreakers = actionCalibrationTieBreakers(
    calibration.objective,
  );
  return manifest.version === 3
    && calibration.method === "validation-action-temperature-scaling"
    && expectedTieBreakers !== undefined
    && isDeepStrictEqual(
      calibration.tieBreakers,
      expectedTieBreakers,
    )
    && typeof calibration.calibratedAt === "string"
    && Boolean(calibration.calibratedAt)
    && typeof calibration.validationReportFile === "string"
    && Boolean(calibration.validationReportFile)
    && typeof calibration.validationReportSha256 === "string"
    && /^[a-f0-9]{64}$/.test(calibration.validationReportSha256)
    && calibration.validationReportVersion === 3
    && Number.isInteger(calibration.validationExamples)
    && calibration.validationExamples! > 1
    && Number.isInteger(calibration.candidateCount)
    && calibration.candidateCount! > 0
    && calibration.checkpoint?.kind === "best"
    && Number.isInteger(calibration.checkpoint.epoch)
    && calibration.checkpoint.epoch > 0
    && Number.isInteger(calibration.checkpoint.globalStep)
    && calibration.checkpoint.globalStep >= 0
    && calibration.datasetFingerprint === manifest.training.datasetFingerprint
    && validExecutionPolicy(calibration.executionPolicy)
    && (calibration.rolloutScoreVersion === 1
      || calibration.rolloutScoreVersion === 2
        && calibration.executionPolicy !== null)
    && calibration.rolloutResetPolicy === "reset-only-at-true-timeline-gaps"
    && isCalibrationMetrics(calibration.selectedMetrics)
    && isCalibrationMetrics(calibration.identityMetrics);
}

function actionCalibrationTieBreakers(
  objective: JointPriceOracleManifest["calibration"]["objective"],
): string[] | undefined {
  if (objective === JOINT_PRICE_ORACLE_ACTION_CALIBRATION_OBJECTIVE) {
    return [...JOINT_PRICE_ORACLE_ACTION_CALIBRATION_TIE_BREAKERS];
  }
  if (objective === JOINT_PRICE_ORACLE_EXACT_STATE_CALIBRATION_OBJECTIVE) {
    return [...JOINT_PRICE_ORACLE_EXACT_STATE_CALIBRATION_TIE_BREAKERS];
  }
  return undefined;
}

function validExecutionPolicy(
  value: JointPriceOracleExecutionPolicy | null | undefined,
): boolean {
  if (value === null) return true;
  return value !== undefined
    && value.version === 2
    && Number.isFinite(value.maximumLeverage)
    && value.maximumLeverage > 0
    && Number.isFinite(value.minimumConfidence)
    && value.minimumConfidence >= 0
    && value.minimumConfidence <= 1
    && Number.isFinite(value.confidenceExposurePower)
    && value.confidenceExposurePower >= 0
    && Number.isFinite(value.confidenceLeverageFloor)
    && value.confidenceLeverageFloor >= 0
    && value.confidenceLeverageFloor <= 1;
}

function isCalibrationMetrics(
  value: JointPriceOracleCalibrationMetrics | undefined,
): value is JointPriceOracleCalibrationMetrics {
  return Boolean(value)
    && isMetricRecord(value!.raw)
    && isMetricRecord(value!.actions);
}

function isMetricRecord(value: unknown): value is Record<string, number | null> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    && Object.values(value).every((item) =>
      item === null || typeof item === "number" && Number.isFinite(item));
}

function validateJointPriceOracleCalibrationArtifact(
  directory: string,
  manifest: JointPriceOracleManifest,
): void {
  const calibration = manifest.calibration;
  if (calibration.method !== "validation-action-temperature-scaling") return;
  const reportFile = path.resolve(directory, calibration.validationReportFile!);
  const expectedRoot = `${path.resolve(directory)}${path.sep}`;
  const reportStat = fs.existsSync(reportFile) ? fs.lstatSync(reportFile) : undefined;
  if (!reportFile.startsWith(expectedRoot) || !reportStat?.isFile()
    || reportStat.isSymbolicLink()) {
    throw new Error("Joint price-oracle calibration report leaves its artifact directory.");
  }
  const bytes = fs.readFileSync(reportFile);
  const actualHash = crypto.createHash("sha256").update(bytes).digest("hex");
  if (actualHash !== calibration.validationReportSha256) {
    throw new Error("Joint price-oracle calibration report hash does not match its manifest.");
  }
  const report = JSON.parse(bytes.toString("utf8")) as Record<string, any>;
  const selection = report.actionTemperatureCalibration as Record<string, any> | undefined;
  const expectedTieBreakers = actionCalibrationTieBreakers(calibration.objective);
  if (report.version !== calibration.validationReportVersion
    || report.plan?.id !== manifest.id
    || report.split?.name !== "validation"
    || report.split.examples !== calibration.validationExamples
    || report.checkpoint?.kind !== "best"
    || report.checkpoint.epoch !== calibration.checkpoint?.epoch
    || report.checkpoint.globalStep !== calibration.checkpoint?.globalStep
    || report.datasetFingerprint !== calibration.datasetFingerprint
    || !isDeepStrictEqual(
      report.actionEvaluation?.executionPolicy,
      calibration.executionPolicy,
    )
    || report.actionEvaluation?.rolloutScoreVersion !== calibration.rolloutScoreVersion
    || report.actionEvaluation?.rolloutResetPolicy !== calibration.rolloutResetPolicy
    || selection?.method !== "validation-action-temperature-scaling-v1"
    || selection.objective?.metric !== calibration.objective
    || selection.objective?.direction !== "maximize"
    || !isDeepStrictEqual(
      selection.tieBreakers,
      expectedTieBreakers,
    )
    || selection.candidateCount !== calibration.candidateCount
    || selection.selectedLogitTemperature !== calibration.logitTemperature
    || !isDeepStrictEqual(selection.selectedMetrics, calibration.selectedMetrics)
    || !isDeepStrictEqual(selection.identityMetrics, calibration.identityMetrics)) {
    throw new Error("Joint price-oracle calibration report provenance is incompatible.");
  }
}

export function isJointPriceOracleDecisionTime(timestamp: number): boolean {
  return positiveModulo(timestamp, JOINT_PRICE_ORACLE_DECISION_INTERVAL_MS)
    === JOINT_PRICE_ORACLE_DECISION_PHASE_MS;
}

function positiveModulo(value: number, divisor: number): number {
  return ((value % divisor) + divisor) % divisor;
}

export function logitsDistribution(
  logits: ArrayLike<number>,
  actionGrid: readonly number[],
  logitTemperature = 1,
): ExposureValueOracleActionDistribution {
  if (logits.length !== actionGrid.length || logits.length < 2) {
    throw new Error("Joint price-oracle logits and action grid are incompatible.");
  }
  if (!(logitTemperature > 0) || !Number.isFinite(logitTemperature)) {
    throw new Error("Logit temperature must be finite and positive.");
  }
  let maximum = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < logits.length; index += 1) {
    maximum = Math.max(maximum, logits[index]!);
  }
  const probabilities = new Float32Array(logits.length);
  let total = 0;
  for (let index = 0; index < logits.length; index += 1) {
    const value = Math.exp((logits[index]! - maximum) / logitTemperature);
    probabilities[index] = value;
    total += value;
  }
  const grid = Float64Array.from(actionGrid);
  let modalIndex = 0;
  let mean = 0;
  let secondMoment = 0;
  let entropy = 0;
  for (let index = 0; index < probabilities.length; index += 1) {
    const probability = probabilities[index]! / total;
    probabilities[index] = probability;
    if (probability > probabilities[modalIndex]!) modalIndex = index;
    mean += probability * grid[index]!;
    secondMoment += probability * grid[index]! ** 2;
    if (probability > 0) entropy -= probability * Math.log(probability);
  }
  return {
    grid,
    probabilities,
    mean,
    secondMoment,
    modalExposure: grid[modalIndex]!,
    entropy,
    opportunity: 0,
    feasibleActionCount: probabilities.length,
  };
}

function candleIndexes(
  candles: readonly OneSecondClose[],
  times: readonly number[],
  contextLength: number,
): number[] {
  const indexesByTime = new Map<number, number>();
  const gapPrefix = new Uint32Array(candles.length);
  for (let index = 0; index < candles.length; index += 1) {
    indexesByTime.set(candles[index]!.closeTime, index);
    if (index > 0) {
      gapPrefix[index] = gapPrefix[index - 1]!
        + (candles[index]!.closeTime === candles[index - 1]!.closeTime + 1_000 ? 0 : 1);
    }
  }
  return times.map((time) => {
    const index = indexesByTime.get(time);
    if (index === undefined || index + 1 < contextLength) {
      throw new Error(`Joint price-oracle lacks ${contextLength} closes through ${time}.`);
    }
    const first = index - contextLength + 1;
    if (candles[first]!.closeTime !== time - (contextLength - 1) * 1_000) {
      throw new Error(`Joint price-oracle context through ${time} is not continuous.`);
    }
    if (gapPrefix[index]! !== gapPrefix[first]!) {
      throw new Error(`Joint price-oracle context through ${time} contains a gap.`);
    }
    return index;
  });
}
