import fs from "node:fs/promises";
import path from "node:path";

const MAX_LOG_CHUNK_BYTES = 64 * 1024 * 1024;
const METRIC_EVENTS = new Set([
  "dataset-complete",
  "dataset-feature-refresh-complete",
  "dataset-feature-refresh-progress",
  "dataset-oracle",
  "dataset-progress",
  "dataset-shard",
  "dataset-source-recovered",
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
  training?: {
    epochs?: number;
    lossWeights?: Record<string, number>;
  };
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

export interface MlpTrainingMetricsResponse {
  plan: {
    id: string;
    label: string;
    epochs?: number;
    lossWeights?: Record<string, number>;
  };
  status?: TrainingStatus;
  running: boolean;
  finalizeRequested: boolean;
  progress: {
    refinedShards?: number;
    totalShards?: number;
    remainingTeacherFits?: number;
    sourceRejectedDays?: number;
  };
  cursor: number;
  reset: boolean;
  events: MlpTrainingMetricEvent[];
}

/** Incrementally exposes the append-only MLP run log to the local dashboard. */
export class MlpTrainingMetricsReader {
  constructor(private readonly planFile: string) {}

  async read(cursor: number): Promise<MlpTrainingMetricsResponse> {
    const files = await this.loadPlan();
    const [status, log, progress, queue, sourceQueue, finalizeRequested] = await Promise.all([
      readOptionalJson<TrainingStatus>(files.statusFile),
      readMetricLog(files.logFile, cursor),
      readOptionalJson<{ shards?: Array<{ refinementPass?: number }> }>(
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
      plan: {
        id: files.plan.id,
        label: files.plan.label,
        ...(files.plan.training?.epochs === undefined
          ? {}
          : { epochs: files.plan.training.epochs }),
        ...(files.plan.training?.lossWeights
          ? { lossWeights: files.plan.training.lossWeights }
          : {}),
      },
      ...(status ? { status } : {}),
      running: processIsAlive(status?.pid),
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
      },
      cursor: log.cursor,
      reset: log.reset,
      events: log.events,
    };
  }

  private async loadPlan() {
    const plan = JSON.parse(await fs.readFile(this.planFile, "utf8")) as TrainingPlan;
    if (!plan.id || !plan.label || !plan.runDir || !plan.datasetDir) {
      throw new Error(`Invalid MLP training plan: ${this.planFile}`);
    }
    const repoRoot = path.resolve(path.dirname(this.planFile), "..");
    const runDir = path.resolve(repoRoot, plan.runDir);
    const datasetDir = path.resolve(repoRoot, plan.datasetDir);
    return {
      plan,
      runDir,
      datasetDir,
      statusFile: path.join(runDir, "status.json"),
      logFile: path.join(runDir, "training.log"),
      finalizeFile: path.join(runDir, "FINALIZE"),
    };
  }
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
