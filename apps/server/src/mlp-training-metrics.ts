import fs from "node:fs/promises";
import path from "node:path";

const MAX_LOG_CHUNK_BYTES = 64 * 1024 * 1024;
const METRIC_EVENTS = new Set([
  "dataset-complete",
  "dataset-component",
  "dataset-example-weighting",
  "dataset-feature-refresh-complete",
  "dataset-feature-refresh-progress",
  "dataset-oracle",
  "dataset-progress",
  "dataset-shard",
  "dataset-source-recovered",
  "dataset-source-recovery-complete",
  "dataset-source-recovery-start",
  "dataset-source-rejected",
  "dataset-stage-timing",
  "epoch",
  "baseline",
  "train-step",
  "training-complete",
  "training-start",
  "time-weighting-ready",
]);

interface TrainingPlan {
  id: string;
  label: string;
  runDir: string;
  datasetDir: string;
  samplingIntervalMs?: number;
  predictionDelayMs?: number;
  training?: {
    epochs?: number;
    patience?: number;
    lossWeights?: Record<string, number>;
    timeWeighting?: MlpTimeWeightingPlan;
  };
}

export interface MlpTimeWeightingPlan {
  mode: "distanceImbalance";
  distanceEpsilon: number;
  minimumWeight: number;
  stateAggregation: "globalDistanceRatio";
  minimumAdviceMagnitude: number;
  memoryHalfLifeSteps: number;
  growthPerPriorAdvice: number;
  maximumMultiplier: number;
  resetAfterGapSteps: number;
  resolutionDivergenceMultiplier: number;
}

interface TrainingStatus {
  pid?: number;
  stage?: string;
  startedAt?: string;
  updatedAt?: string;
  completedAt?: string;
  failedAt?: string;
  pausedAt?: string;
  message?: string;
  error?: string;
  latest?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface MlpTrainingMetricEvent {
  event: string;
  [key: string]: unknown;
}

export interface MlpTrainingRunSummary {
  key: string;
  id: string;
  label: string;
  running: boolean;
  stage?: string;
  updatedAt?: string;
  epochs?: number;
  patience?: number;
}

export interface MlpTrainingMetricsResponse {
  runs: MlpTrainingRunSummary[];
  selectedRunKey: string;
  plan: {
    id: string;
    label: string;
    epochs?: number;
    patience?: number;
    samplingIntervalMs?: number;
    predictionDelayMs?: number;
    lossWeights?: Record<string, number>;
    timeWeighting?: MlpTimeWeightingPlan;
  };
  status?: TrainingStatus;
  running: boolean;
  finalizeRequested: boolean;
  progress: {
    refinedShards?: number;
    totalShards?: number;
    remainingTeacherFits?: number;
    sourceRejectedDays?: number;
    featureComponents?: number;
    oracleComponents?: number;
  };
  cursor: number;
  reset: boolean;
  events: MlpTrainingMetricEvent[];
}

export class MlpTrainingRunNotFoundError extends Error {
  constructor(readonly runKey: string) {
    super(`Unknown MLP training run: ${runKey}`);
    this.name = "MlpTrainingRunNotFoundError";
  }
}

interface LoadedTrainingPlan {
  key: string;
  plan: TrainingPlan;
  runDir: string;
  datasetDir: string;
  statusFile: string;
  logFile: string;
  finalizeFile: string;
  status?: TrainingStatus;
  updatedAt?: string;
  running: boolean;
}

/** Incrementally exposes the append-only MLP run log to the local dashboard. */
export class MlpTrainingMetricsReader {
  private readonly repoRoot: string;

  constructor(
    private readonly planFile: string,
    repoRoot = path.resolve(path.dirname(planFile), ".."),
  ) {
    this.repoRoot = path.resolve(repoRoot);
  }

  async read(
    cursor: number,
    requestedRunKey?: string,
  ): Promise<MlpTrainingMetricsResponse> {
    const availableRuns = await this.discoverRuns();
    const files = requestedRunKey
      ? availableRuns.find((candidate) => candidate.key === requestedRunKey)
      : availableRuns[0];
    if (!files) {
      throw new MlpTrainingRunNotFoundError(requestedRunKey ?? "");
    }
    const [log, progress, queue, sourceQueue, finalizeRequested] = await Promise.all([
      readMetricLog(files.logFile, cursor),
      readOptionalJson<{
        shards?: Array<{ refinementPass?: number }>;
        featureComponents?: unknown[];
        oracleComponents?: unknown[];
      }>(
        path.join(files.datasetDir, "progress.json"),
      ),
      readOptionalJson<{ cases?: unknown[] }>(
        path.join(files.datasetDir, "teacher-refinement-queue.json"),
      ),
      readOptionalJson<{ cases?: unknown[] }>(
        path.join(files.datasetDir, "source-rejection-queue.json"),
      ),
      exists(files.finalizeFile),
    ]);
    const shards = Array.isArray(progress?.shards) ? progress.shards : undefined;
    return {
      runs: availableRuns.map((candidate) => ({
        key: candidate.key,
        id: candidate.plan.id,
        label: candidate.plan.label,
        running: candidate.running,
        ...(candidate.status?.stage ? { stage: candidate.status.stage } : {}),
        ...(candidate.updatedAt ? { updatedAt: candidate.updatedAt } : {}),
        ...(candidate.plan.training?.epochs === undefined
          ? {}
          : { epochs: candidate.plan.training.epochs }),
        ...(candidate.plan.training?.patience === undefined
          ? {}
          : { patience: candidate.plan.training.patience }),
      })),
      selectedRunKey: files.key,
      plan: {
        id: files.plan.id,
        label: files.plan.label,
        ...(files.plan.training?.epochs === undefined
          ? {}
          : { epochs: files.plan.training.epochs }),
        ...(files.plan.training?.patience === undefined
          ? {}
          : { patience: files.plan.training.patience }),
        ...(files.plan.samplingIntervalMs === undefined
          ? {}
          : { samplingIntervalMs: files.plan.samplingIntervalMs }),
        ...((typeof files.status?.predictionDelayMs === "number"
          ? files.status.predictionDelayMs
          : files.plan.predictionDelayMs) === undefined
          ? {}
          : {
              predictionDelayMs: typeof files.status?.predictionDelayMs === "number"
                ? files.status.predictionDelayMs
                : files.plan.predictionDelayMs,
            }),
        ...(files.plan.training?.lossWeights
          ? { lossWeights: files.plan.training.lossWeights }
          : {}),
        ...(files.plan.training?.timeWeighting
          ? { timeWeighting: files.plan.training.timeWeighting }
          : {}),
      },
      ...(files.status ? { status: files.status } : {}),
      running: files.running,
      finalizeRequested,
      progress: {
        ...(shards ? {
          totalShards: shards.length,
          refinedShards: shards.filter((shard) => (shard.refinementPass ?? 0) > 0).length,
        } : {}),
        ...(Array.isArray(queue?.cases) ? { remainingTeacherFits: queue.cases.length } : {}),
        ...(Array.isArray(sourceQueue?.cases)
          ? { sourceRejectedDays: sourceQueue.cases.length }
          : {}),
        ...(Array.isArray(progress?.featureComponents)
          ? { featureComponents: progress.featureComponents.length }
          : {}),
        ...(Array.isArray(progress?.oracleComponents)
          ? { oracleComponents: progress.oracleComponents.length }
          : {}),
      },
      cursor: log.cursor,
      reset: log.reset,
      events: log.events,
    };
  }

  private async discoverRuns(): Promise<LoadedTrainingPlan[]> {
    const planFiles = new Set<string>([path.resolve(this.planFile)]);
    const planDirectory = path.join(this.repoRoot, "ml", "training-plans");
    try {
      for (const entry of await fs.readdir(planDirectory, { withFileTypes: true })) {
        if (entry.isFile() && entry.name.endsWith(".json")) {
          planFiles.add(path.join(planDirectory, entry.name));
        }
      }
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    }
    const loaded = await Promise.all([...planFiles].map(async (planFile) => {
      try {
        return await this.loadPlan(planFile);
      } catch (error) {
        if (path.resolve(planFile) === path.resolve(this.planFile)) throw error;
        return undefined;
      }
    }));
    return loaded
      .filter((candidate): candidate is LoadedTrainingPlan => candidate !== undefined)
      .sort(compareRuns);
  }

  private async loadPlan(planFile: string): Promise<LoadedTrainingPlan> {
    const plan = JSON.parse(await fs.readFile(planFile, "utf8")) as TrainingPlan;
    if (!plan.id || !plan.label || !plan.runDir || !plan.datasetDir) {
      throw new Error(`Invalid MLP training plan: ${planFile}`);
    }
    const runDir = path.resolve(this.repoRoot, plan.runDir);
    const datasetDir = path.resolve(this.repoRoot, plan.datasetDir);
    const statusFile = path.join(runDir, "status.json");
    const [status, planStat] = await Promise.all([
      readOptionalJson<TrainingStatus>(statusFile),
      fs.stat(planFile),
    ]);
    const updatedAt = validTimestamp(status?.updatedAt)
      ?? validTimestamp(status?.completedAt)
      ?? validTimestamp(status?.failedAt)
      ?? validTimestamp(status?.pausedAt)
      ?? planStat.mtime.toISOString();
    return {
      key: path.relative(this.repoRoot, planFile).split(path.sep).join("/"),
      plan,
      runDir,
      datasetDir,
      statusFile,
      logFile: path.join(runDir, "training.log"),
      finalizeFile: path.join(runDir, "FINALIZE"),
      ...(status ? { status } : {}),
      ...(updatedAt ? { updatedAt } : {}),
      running: processIsAlive(status?.pid) && !isTerminalStage(status?.stage),
    };
  }
}

function compareRuns(left: LoadedTrainingPlan, right: LoadedTrainingPlan): number {
  if (left.running !== right.running) return left.running ? -1 : 1;
  const updatedDifference = Date.parse(right.updatedAt ?? "") - Date.parse(left.updatedAt ?? "");
  if (Number.isFinite(updatedDifference) && updatedDifference !== 0) return updatedDifference;
  return left.plan.label.localeCompare(right.plan.label);
}

function validTimestamp(value: string | undefined): string | undefined {
  if (!value) return undefined;
  return Number.isFinite(Date.parse(value)) ? value : undefined;
}

function isTerminalStage(stage: string | undefined): boolean {
  return stage === "complete"
    || stage === "failed"
    || stage === "paused"
    || stage === "cancelled";
}

async function readMetricLog(file: string, requestedCursor: number): Promise<{
  cursor: number;
  reset: boolean;
  events: MlpTrainingMetricEvent[];
}> {
  let descriptor;
  try {
    descriptor = await fs.open(file, "r");
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      return { cursor: 0, reset: requestedCursor > 0, events: [] };
    }
    throw error;
  }
  try {
    const size = (await descriptor.stat()).size;
    const validCursor = Number.isSafeInteger(requestedCursor) && requestedCursor >= 0
      ? requestedCursor
      : 0;
    let start = validCursor <= size ? validCursor : 0;
    let reset = start !== validCursor;
    if (size - start > MAX_LOG_CHUNK_BYTES) {
      start = size - MAX_LOG_CHUNK_BYTES;
      reset = true;
    }
    if (start === size) return { cursor: size, reset, events: [] };
    const buffer = Buffer.allocUnsafe(size - start);
    const { bytesRead } = await descriptor.read(buffer, 0, buffer.length, start);
    const content = buffer.subarray(0, bytesRead);
    const finalNewline = content.lastIndexOf(0x0a);
    if (finalNewline < 0) return { cursor: start, reset, events: [] };
    const complete = content.subarray(0, finalNewline + 1).toString("utf8");
    const lines = complete.split("\n");
    if (start > 0 && reset) lines.shift();
    const events: MlpTrainingMetricEvent[] = [];
    for (const line of lines) {
      if (!line.startsWith("{")) continue;
      try {
        const value = JSON.parse(line) as MlpTrainingMetricEvent;
        if (typeof value.event === "string" && METRIC_EVENTS.has(value.event)) {
          events.push(value);
        }
      } catch {
        // Runner diagnostics and interrupted final lines remain in the log.
      }
    }
    return { cursor: start + finalNewline + 1, reset, events };
  } finally {
    await descriptor.close();
  }
}

async function readOptionalJson<T>(file: string): Promise<T | undefined> {
  try {
    return JSON.parse(await fs.readFile(file, "utf8")) as T;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return undefined;
    throw error;
  }
}

async function exists(file: string): Promise<boolean> {
  try {
    await fs.access(file);
    return true;
  } catch {
    return false;
  }
}

function processIsAlive(pid: number | undefined): boolean {
  if (typeof pid !== "number" || !Number.isInteger(pid) || pid <= 0) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}
