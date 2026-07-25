import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import * as ort from "onnxruntime-node";
import {
  MLP_INPUT_FEATURE_COUNT,
  validateMlpModelManifest,
  type Candle,
  type MlpModelManifest,
  type MlpTrainingMetrics,
} from "@trading/bot-algo";
import { MlpFeatureStore } from "./mlp-feature-store.js";

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../..");
const DEFAULT_BATCH_SIZE = 1_024;

interface DiscoveredMlpModel {
  manifest: MlpModelManifest;
  directory: string;
}

interface LoadedMlpModel extends DiscoveredMlpModel {
  session: ort.InferenceSession;
  executionProvider: "cuda" | "cpu";
}

export interface MlpModelSummary {
  id: string;
  label: string;
  createdAt: string;
  predictionDelayMs: number;
  executionProvider: "cuda" | "cpu" | "unavailable";
  training?: {
    trainExamples: number;
    validationExamples: number;
    testExamples: number;
    bestValidationLoss: number;
    testLoss: number;
    bestValidationMetrics?: MlpTrainingMetrics;
    testMetrics?: MlpTrainingMetrics;
    teacherFitMetrics?: Record<string, number>;
    lossWeights?: Record<string, number>;
    bestEpoch?: number;
    selectionMetric?: "loss" | "klDivergence";
    curriculum?: NonNullable<MlpModelManifest["training"]>["curriculum"];
    finalizedEarly?: boolean;
  };
}

export interface MlpModelPrediction {
  actionLogits: Float32Array[];
  modelActionGrid: number[];
  predictionDelayMs: number;
}

export function mlpInferenceTimes(
  oracleTimes: readonly number[],
  predictionDelayMs: number,
  alignToModelTarget: boolean,
): readonly number[] {
  return alignToModelTarget && predictionDelayMs > 0
    ? oracleTimes.map((time) => time + predictionDelayMs)
    : oracleTimes;
}

export function discoverMlpModels(dataDir: string): DiscoveredMlpModel[] {
  const result = new Map<string, DiscoveredMlpModel>();
  const roots: Array<{ directory: string; depth: number }> = [
    { directory: path.join(REPO_ROOT, "models", "mlp"), depth: 1 },
    { directory: path.join(dataDir, "models", "mlp"), depth: 1 },
    { directory: path.join(dataDir, "ml-joint-studies"), depth: 5 },
    { directory: path.join(dataDir, "ml-dynamic-studies"), depth: 5 },
  ];
  for (const root of roots) {
    for (const directory of artifactDirectories(root.directory, root.depth)) {
      try {
        const manifest = JSON.parse(
          fs.readFileSync(path.join(directory, "manifest.json"), "utf8"),
        ) as MlpModelManifest;
        validateMlpModelManifest(manifest);
        const modelFile = path.resolve(directory, manifest.modelFile);
        if (!modelFile.startsWith(`${path.resolve(directory)}${path.sep}`)
          || !fs.statSync(modelFile).isFile()) {
          throw new Error("MLP manifest modelFile must resolve to a file inside its artifact directory.");
        }
        const existing = result.get(manifest.id);
        if (!existing || Date.parse(manifest.createdAt) > Date.parse(existing.manifest.createdAt)) {
          result.set(manifest.id, { manifest, directory });
        }
      } catch (error) {
        const reason = error instanceof Error ? error.message : String(error);
        console.error(`Ignoring invalid MLP model artifact ${directory}: ${reason}`);
      }
    }
  }
  return [...result.values()].sort((left, right) =>
    Date.parse(right.manifest.createdAt) - Date.parse(left.manifest.createdAt));
}

function artifactDirectories(root: string, maximumDepth: number): string[] {
  const result: string[] = [];
  const visit = (directory: string, depth: number): void => {
    let entries: fs.Dirent[];
    try {
      entries = fs.readdirSync(directory, { withFileTypes: true });
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") return;
      throw error;
    }
    if (entries.some((entry) => entry.isFile() && entry.name === "manifest.json")) {
      result.push(directory);
      return;
    }
    if (depth >= maximumDepth) return;
    for (const entry of entries) {
      if (entry.isDirectory() && !entry.isSymbolicLink()) {
        visit(path.join(directory, entry.name), depth + 1);
      }
    }
  };
  visit(root, 0);
  return result;
}

export function mlpModelSummaries(dataDir: string): MlpModelSummary[] {
  return discoverMlpModels(dataDir).map(({ manifest }) => ({
    id: manifest.id,
    label: manifest.label,
    createdAt: manifest.createdAt,
    predictionDelayMs: manifest.predictionDelayMs,
    executionProvider: manifest.verification?.executionProvider ?? "unavailable",
    ...(manifest.training ? { training: {
      trainExamples: manifest.training.trainExamples,
      validationExamples: manifest.training.validationExamples,
      testExamples: manifest.training.testExamples,
      bestValidationLoss: manifest.training.bestValidationLoss,
      testLoss: manifest.training.testLoss,
      bestEpoch: manifest.training.bestEpoch,
      ...(manifest.training.bestValidationMetrics
        ? { bestValidationMetrics: manifest.training.bestValidationMetrics }
        : {}),
      ...(manifest.training.testMetrics ? { testMetrics: manifest.training.testMetrics } : {}),
      ...(manifest.training.teacherFitMetrics
        ? { teacherFitMetrics: manifest.training.teacherFitMetrics }
        : {}),
      ...(manifest.training.lossWeights ? { lossWeights: manifest.training.lossWeights } : {}),
      ...(manifest.training.selectionMetric
        ? { selectionMetric: manifest.training.selectionMetric }
        : {}),
      ...(manifest.training.curriculum
        ? { curriculum: manifest.training.curriculum }
        : {}),
      ...(manifest.training.finalizedEarly ? { finalizedEarly: true } : {}),
    } } : {}),
  }));
}

export class MlpModelRuntime {
  private readonly featureStore: MlpFeatureStore;
  private readonly loaded = new Map<string, Promise<LoadedMlpModel>>();
  private predictionCache: {
    modelId: string;
    oneSecondCandles: readonly Candle[];
    times: readonly number[];
    outputs: Float32Array[];
  } | undefined;

  constructor(private readonly dataDir: string) {
    this.featureStore = new MlpFeatureStore(dataDir);
  }

  models(): MlpModelSummary[] {
    return mlpModelSummaries(this.dataDir);
  }

  async predict(
    modelId: string,
    oneSecondCandles: readonly Candle[],
    times: readonly number[],
    cancelFlag?: Int32Array,
    alignToModelTarget = false,
  ): Promise<MlpModelPrediction> {
    if (!modelId) throw new Error("MLP prediction requires a selected model artifact.");
    const model = await this.model(modelId);
    const featureTimes = mlpInferenceTimes(
      times,
      model.manifest.predictionDelayMs,
      alignToModelTarget,
    );
    const featureCount = MLP_INPUT_FEATURE_COUNT;
    const cached = this.predictionCache;
    if (cached
      && cached.modelId === modelId
      && cached.oneSecondCandles === oneSecondCandles
      && cached.times.length >= featureTimes.length
      && cached.times[0] === featureTimes[0]
      && cached.times[featureTimes.length - 1] === featureTimes[featureTimes.length - 1]) {
      return {
        actionLogits: cached.outputs.slice(0, featureTimes.length),
        modelActionGrid: model.manifest.actionGrid,
        predictionDelayMs: model.manifest.predictionDelayMs,
      };
    }
    const features = await this.featureStore.prepare(oneSecondCandles, featureTimes);
    const outputs: Float32Array[] = new Array(featureTimes.length);
    const requestedBatch = Number(process.env.TRADING_MLP_BATCH_SIZE ?? DEFAULT_BATCH_SIZE);
    const batchSize = Number.isFinite(requestedBatch) && requestedBatch > 0
      ? Math.max(1, Math.floor(requestedBatch))
      : DEFAULT_BATCH_SIZE;
    for (let start = 0; start < featureTimes.length; start += batchSize) {
      throwIfCancelled(cancelFlag);
      const end = Math.min(featureTimes.length, start + batchSize);
      const input = new Float32Array((end - start) * featureCount);
      for (let index = start; index < end; index += 1) {
        features.encode(featureTimes[index]!, input, (index - start) * featureCount);
      }
      const tensor = new ort.Tensor("float32", input, [end - start, featureCount]);
      const result = await model.session.run({ features: tensor }, ["action_logits"]);
      const output = result.action_logits;
      if (!output || output.type !== "float32"
        || output.dims.length !== 2
        || output.dims[0] !== end - start
        || output.dims[1] !== model.manifest.outputActionCount
        || !(output.data instanceof Float32Array)) {
        throw new Error(
          `MLP ONNX output does not match [batch, ${model.manifest.outputActionCount}] float32 contract.`,
        );
      }
      for (let row = 0; row < end - start; row += 1) {
        outputs[start + row] = output.data.slice(
          row * model.manifest.outputActionCount,
          (row + 1) * model.manifest.outputActionCount,
        );
      }
    }
    this.predictionCache = { modelId, oneSecondCandles, times: featureTimes, outputs };
    return {
      actionLogits: outputs,
      modelActionGrid: model.manifest.actionGrid,
      predictionDelayMs: model.manifest.predictionDelayMs,
    };
  }

  async predictionDelayMs(modelId: string): Promise<number> {
    if (!modelId) throw new Error("MLP prediction requires a selected model artifact.");
    return (await this.model(modelId)).manifest.predictionDelayMs;
  }

  private model(modelId: string): Promise<LoadedMlpModel> {
    const existing = this.loaded.get(modelId);
    if (existing) return existing;
    const pending = this.loadModel(modelId);
    this.loaded.set(modelId, pending);
    pending.catch(() => {
      if (this.loaded.get(modelId) === pending) this.loaded.delete(modelId);
    });
    return pending;
  }

  private async loadModel(modelId: string): Promise<LoadedMlpModel> {
    const discovered = discoverMlpModels(this.dataDir).find((item) => item.manifest.id === modelId);
    if (!discovered) {
      throw new Error(
        `Unknown MLP model '${modelId}'. Train one with npm run mlp:train before selecting it.`,
      );
    }
    if (!discovered.manifest.verification) {
      throw new Error(
        `MLP model '${modelId}' has not been verified; run npm run mlp:verify -- ${discovered.directory}.`,
      );
    }
    const modelFile = path.join(discovered.directory, discovered.manifest.modelFile);
    const preference = process.env.TRADING_MLP_EXECUTION_PROVIDER?.trim().toLowerCase() ?? "auto";
    if (preference !== "auto" && preference !== "cuda" && preference !== "cpu") {
      throw new Error("TRADING_MLP_EXECUTION_PROVIDER must be auto, cuda, or cpu.");
    }
    if (preference !== "cpu") {
      try {
        const session = await ort.InferenceSession.create(modelFile, {
          executionProviders: [{ name: "cuda", deviceId: 0 }],
          graphOptimizationLevel: "all",
          executionMode: "sequential",
          enableMemPattern: true,
        });
        validateSession(session);
        console.info(`MLP model '${modelId}' loaded with CUDA inference.`);
        return { ...discovered, session, executionProvider: "cuda" };
      } catch (error) {
        if (preference === "cuda") {
          throw new Error(`MLP CUDA inference initialization failed: ${errorMessage(error)}`);
        }
        console.error(`MLP CUDA inference unavailable; using CPU: ${errorMessage(error)}`);
      }
    }
    const session = await ort.InferenceSession.create(modelFile, {
      executionProviders: ["cpu"],
      graphOptimizationLevel: "all",
      executionMode: "parallel",
    });
    validateSession(session);
    console.info(`MLP model '${modelId}' loaded with CPU inference.`);
    return { ...discovered, session, executionProvider: "cpu" };
  }
}

function validateSession(session: ort.InferenceSession): void {
  if (session.inputNames.length !== 1 || session.inputNames[0] !== "features"
    || session.outputNames.length !== 1 || session.outputNames[0] !== "action_logits") {
    throw new Error("MLP ONNX graph must expose features -> action_logits only.");
  }
}

function throwIfCancelled(flag: Int32Array | undefined): void {
  if (flag && Atomics.load(flag, 0) !== 0) throw new Error("MLP prediction cancelled.");
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
