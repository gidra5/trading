import fs from "node:fs/promises";
import path from "node:path";

const EVENT_ALIASES: ReadonlyMap<string, string> = new Map([
  ["minute-return-dataset-selected", "dataset-complete"],
  ["minute-return-epoch", "epoch"],
  ["minute-return-complete", "training-complete"],
  ["distribution-q-training-start", "training-start"],
  ["distribution-q-optimization-epoch", "epoch"],
  ["distribution-q-complete", "training-complete"],
  ["theory-q-prime-training-start", "training-start"],
  ["theory-q-prime-optimization-epoch", "epoch"],
  ["theory-q-prime-complete", "training-complete"],
] as const);

type JsonObject = Readonly<Record<string, unknown>>;

interface RunRecord {
  readonly key: string;
  readonly id: string;
  readonly label: string;
  readonly runDir: string;
  readonly running: boolean;
  readonly stage?: string;
  readonly updatedAt?: string;
  readonly epochs?: number;
  readonly plan: JsonObject;
  readonly status?: JsonObject;
}

async function readJson(file: string): Promise<JsonObject | undefined> {
  try {
    const parsed: unknown = JSON.parse(await fs.readFile(file, "utf8"));
    return parsed && typeof parsed === "object" && !Array.isArray(parsed)
      ? (parsed as JsonObject)
      : undefined;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return undefined;
    throw error;
  }
}

function object(value: unknown): JsonObject | undefined {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as JsonObject)
    : undefined;
}

function text(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

function finite(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function positiveInteger(value: unknown): number | undefined {
  const number = finite(value);
  return number !== undefined && Number.isInteger(number) && number > 0
    ? number
    : undefined;
}

function isRunningStage(stage: string | undefined): boolean {
  return stage === "initializing"
    || stage === "computing-training-statistics"
    || stage === "compiling"
    || stage === "training"
    || stage === "evaluating";
}

async function runRecord(runRoot: string): Promise<RunRecord | undefined> {
  const snapshot = await readJson(path.join(runRoot, "state", "plan.json"));
  const plan = object(snapshot?.plan) ?? snapshot;
  if (!plan) return undefined;
  const id = text(plan.id);
  const runDir = text(plan.runDir);
  if (!id || !runDir) return undefined;
  const [status, display] = await Promise.all([
    readJson(path.join(runRoot, "state", "status.json")),
    readJson(path.join(runRoot, "state", "display.json")),
  ]);
  const latest = object(status?.latest);
  const training = object(plan.training);
  const stage = text(status?.stage);
  return {
    key: id,
    id,
    label: text(display?.label) ?? text(plan.label) ?? id,
    runDir,
    running: isRunningStage(stage),
    ...(stage ? { stage } : {}),
    ...(text(status?.updatedAt) ? { updatedAt: text(status?.updatedAt) } : {}),
    ...(positiveInteger(latest?.epochs)
      ?? positiveInteger(status?.epochs)
      ?? positiveInteger(display?.epochs)
      ?? positiveInteger(training?.epochs)
      ? {
          epochs: positiveInteger(latest?.epochs)
            ?? positiveInteger(status?.epochs)
            ?? positiveInteger(display?.epochs)
            ?? positiveInteger(training?.epochs),
        }
      : {}),
    plan,
    ...(status ? { status } : {}),
  };
}

async function availableRuns(repoRoot: string): Promise<readonly RunRecord[]> {
  const root = path.join(repoRoot, "data", "training", "runs");
  let entries: readonly import("node:fs").Dirent[];
  try {
    entries = await fs.readdir(root, { withFileTypes: true });
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return [];
    throw error;
  }
  const runs = (await Promise.all(
    entries
      .filter((entry) => entry.isDirectory())
      .map((entry) => runRecord(path.join(root, entry.name))),
  )).filter((run): run is RunRecord => run !== undefined);
  return runs.sort((left, right) =>
    Number(right.running) - Number(left.running)
    || (right.updatedAt ?? "").localeCompare(left.updatedAt ?? "")
    || left.label.localeCompare(right.label),
  );
}

function publicRun(run: RunRecord) {
  return {
    key: run.key,
    id: run.id,
    label: run.label,
    running: run.running,
    ...(run.stage ? { stage: run.stage } : {}),
    ...(run.updatedAt ? { updatedAt: run.updatedAt } : {}),
    ...(run.epochs ? { epochs: run.epochs } : {}),
  };
}

function publicPlan(run: RunRecord) {
  const training = object(run.plan.training);
  const architecture = object(run.plan.architecture);
  return {
    id: run.id,
    label: run.label,
    ...(run.epochs ? { epochs: run.epochs } : {}),
    ...(finite(run.plan.samplingIntervalMs) !== undefined
      ? { samplingIntervalMs: finite(run.plan.samplingIntervalMs) }
      : {}),
    ...(finite(run.plan.predictionDelayMs) !== undefined
      ? { predictionDelayMs: finite(run.plan.predictionDelayMs) }
      : {}),
    ...(finite(architecture?.dropout) !== undefined
      ? { dropout: finite(architecture?.dropout) }
      : {}),
    ...(finite(architecture?.dropoutRate) !== undefined
      ? { dropoutRate: finite(architecture?.dropoutRate) }
      : {}),
    ...(object(training?.lossWeights)
      ? { lossWeights: object(training?.lossWeights) }
      : {}),
  };
}

async function logEvents(
  repoRoot: string,
  run: RunRecord,
  cursor: number,
): Promise<{ readonly cursor: number; readonly reset: boolean; readonly events: readonly JsonObject[] }> {
  const file = path.resolve(repoRoot, run.runDir, "logs", "training.jsonl");
  const runsRoot = path.resolve(repoRoot, "data", "training", "runs") + path.sep;
  if (!file.startsWith(runsRoot)) throw new Error("Training log escaped the run store");
  let content: Buffer;
  try {
    content = await fs.readFile(file);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      return { cursor: 0, reset: cursor !== 0, events: [] };
    }
    throw error;
  }
  const reset = cursor > content.length;
  const start = reset ? 0 : cursor;
  // A training writer may still be appending the last record. Only advance
  // the byte cursor past complete lines so that record is retried next poll.
  const end = Math.max(start, content.lastIndexOf(10) + 1);
  const events: JsonObject[] = [];
  for (const line of content.subarray(start, end).toString("utf8").split(/\r?\n/)) {
    if (!line) continue;
    const event = object(JSON.parse(line));
    if (!event) continue;
    const name = text(event.event);
    events.push({
      ...event,
      ...(name && EVENT_ALIASES.has(name)
        ? { event: EVENT_ALIASES.get(name) }
        : {}),
    });
  }
  return { cursor: end, reset, events };
}

function comparisonMetrics(value: unknown): JsonObject | undefined {
  const item = object(value);
  if (!item) return undefined;
  return object(item.nextReturn) ?? item;
}

async function comparisonRun(repoRoot: string, run: RunRecord) {
  const root = path.resolve(repoRoot, run.runDir);
  const [result, evaluation] = await Promise.all([
    readJson(path.join(root, "state", "result.json")),
    readJson(path.join(root, "state", "stopped-evaluation.json")),
  ]);
  const log = await logEvents(repoRoot, run, 0);
  const epochs = new Map<number, JsonObject>();
  for (const event of log.events) {
    if (event.event === "training-start") {
      const start = finite(event.startEpoch) ?? 0;
      for (const epoch of epochs.keys()) if (epoch >= start) epochs.delete(epoch);
    } else if (event.event === "epoch" && finite(event.epoch) !== undefined) {
      epochs.set(finite(event.epoch)!, {
        ...event,
        distribution: object(event.distribution) ?? {
          train: object(event.trainDistribution),
          validation: object(event.validationDistribution),
          test: object(event.testDistribution),
        },
      });
    }
  }
  const history = [...epochs.values()].sort((a, b) => Number(a.epoch) - Number(b.epoch));
  const fit = history
    .map((event) => ({
      epoch: finite(event.epoch) ?? 0,
      x: (finite(event.epoch) ?? 0) + 1,
      live: run.running,
      ...(comparisonMetrics(event.train) ? { train: comparisonMetrics(event.train) } : {}),
      ...(comparisonMetrics(event.validation)
        ? { validation: comparisonMetrics(event.validation) }
        : {}),
      trainNormalizedMse: finite(comparisonMetrics(event.train)?.normalizedMse),
      validationNormalizedMse: finite(comparisonMetrics(event.validation)?.normalizedMse),
      trainNegativeLogLikelihood: finite(object(object(event.distribution)?.train)?.negativeLogLikelihood),
      validationNegativeLogLikelihood: finite(object(object(event.distribution)?.validation)?.negativeLogLikelihood),
    }));
  const latest = history.at(-1) ?? object(run.status?.latest);
  const selections: Record<string, JsonObject> = {};
  // Recover online validation selections from the same score and tie rule
  // as the trainer; their displayed metrics remain next-return metrics.
  for (const event of history) {
    const scope = text(event.checkpointSelectionScope);
    const scoreMetrics = scope === "complete-next-feature-state"
      ? object(object(event.featureState)?.validation)
      : comparisonMetrics(event.validation);
    const scores: readonly [string, number | undefined, boolean][] = [
      ["validation-mse", finite(scoreMetrics?.normalizedMse), false],
      ["validation-correlation", finite(scoreMetrics?.correlation), true],
      ["validation-nll", finite(object(object(event.distribution)?.validation)?.negativeLogLikelihood), false],
    ];
    for (const [policy, score, maximize] of scores) {
      if (score === undefined) continue;
      const previous = finite(selections[policy]?.selectionScore);
      if (previous === undefined || (maximize ? score > previous : score < previous)) {
        selections[policy] = { ...event, selectionScore: score };
      }
    }
  }
  if (latest && finite(latest.epoch) !== undefined) selections["stopped-last"] = latest;
  for (const [policy, value] of Object.entries(object(result?.checkpointSelections) ?? {})) {
    const selection = object(value);
    if (selection) selections[policy] = selection;
  }
  for (const [policy, value] of Object.entries(object(evaluation?.policies) ?? {})) {
    const selection = object(value);
    if (selection) selections[policy === "last" ? "stopped-last" : policy.replace(/^best-/, "")] = selection;
  }
  const summary = result ?? latest;
  const trainingStart = log.events.filter((event) => event.event === "training-start").at(-1);
  return {
    ...publicRun(run),
    checkpointSelections: selections,
    evaluationHorizonSteps: text(summary?.evaluationScope) === "next-completed-1s-signed-log-return" ? 1 : undefined,
    ...(finite(latest?.epoch) !== undefined ? { epoch: finite(latest?.epoch) } : {}),
    examples: finite(result?.examples) ?? finite(object(run.plan.subset)?.examples),
    parameterCount: finite(summary?.parameterCount) ?? finite(trainingStart?.parameters),
    trainableParameterCount: finite(result?.trainableParameterCount) ?? finite(trainingStart?.trainableParameters),
    ...(finite(result?.bestEpoch) !== undefined ? { bestEpoch: finite(result?.bestEpoch) } : {}),
    ...(comparisonMetrics(summary?.train) ? { train: comparisonMetrics(summary?.train) } : {}),
    ...(comparisonMetrics(summary?.validation)
      ? { validation: comparisonMetrics(summary?.validation) }
      : {}),
    ...(comparisonMetrics(summary?.test) ? { test: comparisonMetrics(summary?.test) } : {}),
    ...(object(summary?.distribution) ? { distribution: object(summary?.distribution) } : {}),
    fit,
  };
}

export class TrainingRunNotFoundError extends Error {}

export class TrainingMetricsReader {
  constructor(private readonly repoRoot: string) {}

  async read(cursor: number, requested?: string) {
    const runs = await availableRuns(this.repoRoot);
    const selected = requested
      ? runs.find((run) => run.key === requested)
      : runs[0];
    if (!selected) throw new TrainingRunNotFoundError(
      requested ? `Unknown training run: ${requested}` : "No training runs are available",
    );
    const log = await logEvents(this.repoRoot, selected, cursor);
    return {
      runs: runs.map(publicRun),
      matrices: [],
      selectedRunKey: selected.key,
      plan: publicPlan(selected),
      status: selected.status,
      running: selected.running,
      finalizeRequested: false,
      progress: {},
      ...log,
    };
  }

  async compare(keys: readonly string[]) {
    const runs = await availableRuns(this.repoRoot);
    const selected = keys.map((key) => {
      const run = runs.find((candidate) => candidate.key === key);
      if (!run) throw new TrainingRunNotFoundError(`Unknown training run: ${key}`);
      return run;
    });
    return { runs: await Promise.all(selected.map((run) => comparisonRun(this.repoRoot, run))) };
  }
}
