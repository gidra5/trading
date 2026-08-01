import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import * as ort from "onnxruntime-node";
import type { ExposureValueOracleActionDistribution } from "@trading/bot-algo";

const DEFAULT_BATCH_SIZE = 512;

export interface OneSecondClose {
  closeTime: number;
  close: number;
}

export interface JointPriceOracleManifest {
  version: 1;
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
    validation: Record<string, number>;
    test?: Record<string, number>;
    datasetFingerprint: string;
  };
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
        );
      }
    }
    return result;
  }

  async predictLatest(
    oneSecondCandles: readonly OneSecondClose[],
  ): Promise<ExposureValueOracleActionDistribution | null> {
    const latest = oneSecondCandles.at(-1);
    if (!latest || (latest.closeTime + 1) % 60_000 !== 0) return null;
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
    artifacts.push({ directory, manifest });
  }
  return artifacts.sort((left, right) =>
    Date.parse(right.manifest.createdAt) - Date.parse(left.manifest.createdAt));
}

export function validateJointPriceOracleManifest(
  manifest: JointPriceOracleManifest,
): void {
  if (manifest.version !== 1 || manifest.kind !== "joint-price-oracle"
    || !manifest.id || !manifest.modelFile || !/^[a-f0-9]{64}$/.test(manifest.modelSha256)
    || manifest.input?.name !== "closes" || manifest.input.dtype !== "float32"
    || manifest.input.intervalMs !== 1_000 || manifest.input.contextLength !== 3_600
    || manifest.input.variableCount !== 1
    || manifest.output?.name !== "action_logits" || manifest.output.dtype !== "float32"
    || manifest.output.actionCount !== 101
    || manifest.output.actionGrid?.length !== manifest.output.actionCount
    || manifest.oracle?.intervalMs !== 1_000
    || manifest.oracle.decisionIntervalMs !== 60_000
    || manifest.oracle.options?.holdingPeriodSteps !== 60
    || manifest.oracle.options.decisionDelaySteps !== 60
    || manifest.oracle.options.valueHorizonSteps !== 3_600) {
    throw new Error("Joint price-oracle manifest does not satisfy the 1h/1m/1m contract.");
  }
}

export function logitsDistribution(
  logits: ArrayLike<number>,
  actionGrid: readonly number[],
): ExposureValueOracleActionDistribution {
  if (logits.length !== actionGrid.length || logits.length < 2) {
    throw new Error("Joint price-oracle logits and action grid are incompatible.");
  }
  let maximum = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < logits.length; index += 1) {
    maximum = Math.max(maximum, logits[index]!);
  }
  const probabilities = new Float32Array(logits.length);
  let total = 0;
  for (let index = 0; index < logits.length; index += 1) {
    const value = Math.exp(logits[index]! - maximum);
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
