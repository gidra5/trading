import {
  For,
  Show,
  createMemo,
  createSignal,
  onCleanup,
  onMount,
} from "solid-js";
import { Activity, ArrowLeft, BarChart3, Pause, Play, Plus, Search, X } from "lucide-solid";

const apiBase = import.meta.env.DEV ? "/backend" : "";
const POLL_MS = 2_000;
const COMPARISON_COLORS = ["#38bdf8", "#f5b84b", "#a78bfa", "#34d399"] as const;

interface MetricValues {
  loss?: number;
  normalizedMse?: number;
  mse?: number;
  rmse?: number;
  mae?: number;
  zeroBaselineMse?: number;
  mseSkillVsZero?: number;
  directionAccuracy?: number;
  correlation?: number;
  predictionStd?: number;
  targetStd?: number;
  crossEntropy?: number;
  klDivergence?: number;
  klDivergenceVariance?: number;
  klDivergenceStdDev?: number;
  baseKlDivergence?: number;
  reverseKlDivergence?: number;
  probabilityMse?: number;
  probabilityMseVariance?: number;
  probabilityMseStdDev?: number;
  excessEntropy?: number;
  oracleMutualInformation?: number;
  targetEntropy?: number;
  predictedEntropy?: number;
  entropyGap?: number;
  entropySharpness?: number;
  reverseKlGate?: number;
  entropySharpnessGate?: number;
  softLayerNorm?: number;
  softLayerNormMeanPenalty?: number;
  softLayerNormVariancePenalty?: number;
  distributionLayer?: number;
  distributionLayerSumPenalty?: number;
  distributionLayerNegativePenalty?: number;
  softWeightBound?: number;
  centeringConstraint?: number;
  centeringIdempotence?: number;
  centeringSymmetry?: number;
  distanceImbalanceWeight?: number;
  timeWeightEffectiveSampleRatio?: number;
  curriculumTargetKl?: number;
  curriculumTargetProbabilityMse?: number;
  curriculumTargetEntropy?: number;
  trainingCrossEntropy?: number;
  rawCrossEntropy?: number;
  rawBaseActionKl?: number;
  rawProbabilityMse?: number;
  rawTargetEntropy?: number;
  regularizationLoss?: number;
}

interface TrainingEvent {
  event: string;
  [key: string]: unknown;
}

interface TrainingStatus {
  stage?: string;
  updatedAt?: string;
  startedAt?: string;
  message?: string;
  error?: string;
  latest?: TrainingEvent;
  [key: string]: unknown;
}

interface TimeWeightingPlan {
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

interface SoftWeightBoundPlan {
  desiredMagnitude: number;
  sharpness: number;
  absoluteEpsilon: number;
}

interface ReverseKlPlan {
  predictionMixtureWeight: number;
}

interface OutputRegularizerPlan {
  applicationProbabilities: {
    reverseKl: number;
    entropySharpness: number;
  };
  samplingUnit: "optimizer-update";
  independentGates: boolean;
  inverseProbabilityScaling: boolean;
}

interface BranchNormalizationPlan {
  learnableCentering?: boolean;
}

interface MetricsResponse {
  runs: Array<{
    key: string;
    id: string;
    label: string;
    running: boolean;
    stage?: string;
    updatedAt?: string;
    epochs?: number;
    patience?: number;
    archived?: boolean;
    bestValidationKl?: number;
    bestValidationKlEpoch?: number;
  }>;
  matrices: Array<{
    id: string;
    label: string;
    totalRuns: number;
    completedRuns: number;
    failedRuns: number;
    queuedRuns: number;
    progress: number;
    controllable: boolean;
    pauseRequested: boolean;
    paused: boolean;
    active?: {
      id: string;
      label: string;
      stage: string;
      epoch?: number;
      epochs?: number;
      bestTrainScore?: number;
    };
  }>;
  selectedRunKey: string;
  plan: {
    id: string;
    label: string;
    epochs?: number;
    patience?: number;
    samplingIntervalMs?: number;
    predictionDelayMs?: number;
    archived?: boolean;
    archivedAt?: string;
    bestValidationKl?: number;
    bestValidationKlEpoch?: number;
    dropout?: number;
    dropoutRate?: number;
    lossWeights?: Record<string, number>;
    reverseKl?: ReverseKlPlan;
    outputRegularizer?: OutputRegularizerPlan;
    softWeightBound?: SoftWeightBoundPlan;
    branchNormalization?: BranchNormalizationPlan;
    timeWeighting?: TimeWeightingPlan;
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
  events: TrainingEvent[];
}

interface ComparisonMetricValues {
  normalizedMse?: number;
  mse?: number;
  rmse?: number;
  mae?: number;
  zeroBaselineMse?: number;
  mseSkillVsZero?: number;
  directionAccuracy?: number;
  correlation?: number;
}

interface ComparisonResponse {
  runs: Array<{
    key: string;
    id: string;
    label: string;
    running: boolean;
    stage?: string;
    epochs?: number;
    examples?: number;
    parameterCount?: number;
    trainableParameterCount?: number;
    bestEpoch?: number;
    train?: ComparisonMetricValues;
    validation?: ComparisonMetricValues;
    test?: ComparisonMetricValues;
    validationCheckpointEpoch?: number;
    fit: Array<{
      epoch: number;
      trainNormalizedMse?: number;
      validationNormalizedMse?: number;
      bestTrainScore?: number;
    }>;
  }>;
}

interface DatasetPoint {
  key: string;
  x: number;
  date: string;
  split: string;
  days?: number;
  preparationExamplesPerSecond?: number;
  examplesPerSecond?: number;
  gpuMemoryMiB?: number;
  kl?: number;
  probabilityMse?: number;
  temporalBefore?: number;
  temporalAfter?: number;
  temporalSelectedPct?: number;
  acceptedPct?: number;
  oracleKernelMs?: number;
  oracleWallMs?: number;
  oraclePipelineWaitMs?: number;
  directDiagnosticsKernelMs?: number;
  directDiagnosticsCutoffKernelMs?: number;
  directDiagnosticsWorkerWallMs?: number;
  teacherWallMs?: number;
  featureEncodingMs?: number;
  persistMs?: number;
}

interface TrainStepPoint {
  globalStep: number;
  epoch: number;
  learningRate?: number;
  gradientNorm?: number;
  examplesPerSecond?: number;
  networkRowsPerSecond?: number;
  batchCompactionRatio?: number;
  gpuAllocatedMiB?: number;
  gpuReservedMiB?: number;
  gpuDeviceUsedMiB?: number;
  latest: MetricValues;
}

interface EpochPoint {
  epoch: number;
  globalStep: number;
  train: MetricValues;
  validation: MetricValues;
  bestValidation?: number;
  staleEpochs?: number;
  trainingTargetTemperature?: number;
  curriculumTargetKl?: number;
  learningRate?: number;
  seconds?: number;
}

interface PlotPoint {
  x: number;
  y: number;
}

interface PlotSeries {
  label: string;
  color: string;
  values: PlotPoint[];
}

interface ChartViewport {
  xMin: number;
  xMax: number;
  followLatest: boolean;
}

export function MlpTrainingPage() {
  const [snapshot, setSnapshot] = createSignal<MetricsResponse>();
  const [runs, setRuns] = createSignal<MetricsResponse["runs"]>([]);
  const [selectedRunKey, setSelectedRunKey] = createSignal<string>();
  const [comparisonRunKeys, setComparisonRunKeys] = createSignal<string[]>([]);
  const [comparison, setComparison] = createSignal<ComparisonResponse>();
  const [comparisonError, setComparisonError] = createSignal<string>();
  const [matrixControlPending, setMatrixControlPending] = createSignal<string>();
  const [matrixControlError, setMatrixControlError] = createSignal<string>();
  const [datasetPoints, setDatasetPoints] = createSignal<DatasetPoint[]>([]);
  const [currentDataset, setCurrentDataset] = createSignal<DatasetPoint>();
  const [trainSteps, setTrainSteps] = createSignal<TrainStepPoint[]>([]);
  const [epochs, setEpochs] = createSignal<EpochPoint[]>([]);
  const [error, setError] = createSignal<string>();
  const [now, setNow] = createSignal(Date.now());
  const datasetByKey = new Map<string, DatasetPoint>();
  const oracleByDate = new Map<string, Partial<DatasetPoint>>();
  const stepsById = new Map<number, TrainStepPoint>();
  const epochsById = new Map<number, EpochPoint>();
  let cursor = 0;
  let pollTimer: number | undefined;
  let clockTimer: number | undefined;
  let comparisonTimer: number | undefined;
  let disposed = false;
  let requestGeneration = 0;
  let comparisonRequestGeneration = 0;

  const clearSeries = () => {
    datasetByKey.clear();
    oracleByDate.clear();
    stepsById.clear();
    epochsById.clear();
    setDatasetPoints([]);
    setCurrentDataset(undefined);
    setTrainSteps([]);
    setEpochs([]);
  };

  const publishSeries = () => {
    setDatasetPoints([...datasetByKey.values()].sort((left, right) => left.x - right.x));
    setTrainSteps([...stepsById.values()].sort((left, right) => left.globalStep - right.globalStep));
    setEpochs([...epochsById.values()].sort((left, right) => left.epoch - right.epoch));
  };

  const applyEvents = (events: TrainingEvent[]) => {
    for (const event of events) {
      if (event.event === "training-start") {
        const startEpoch = numberValue(event.startEpoch) ?? 0;
        for (const [globalStep, point] of stepsById) {
          if (point.epoch >= startEpoch) stepsById.delete(globalStep);
        }
        for (const epoch of epochsById.keys()) {
          if (epoch >= startEpoch) epochsById.delete(epoch);
        }
      } else if (event.event === "dataset-oracle") {
        const date = textValue(event.date);
        const x = numberValue(event.day);
        if (!date || x === undefined) continue;
        const oracle: Partial<DatasetPoint> = {
          x,
          date,
          days: numberValue(event.days),
          oracleKernelMs: numberValue(event.kernelMs),
          oracleWallMs: numberValue(event.wallMs),
          oraclePipelineWaitMs: numberValue(event.pipelineWaitMs),
        };
        oracleByDate.set(date, oracle);
        const key = `${date}:oracle`;
        const point: DatasetPoint = {
          ...datasetByKey.get(key),
          ...oracle,
          key,
          x,
          date,
          split: "oracle",
        };
        datasetByKey.set(key, point);
        if (!currentDataset() || x >= currentDataset()!.x) setCurrentDataset(point);
      } else if (event.event === "dataset-progress") {
        const date = textValue(event.date);
        const split = textValue(event.split) ?? textValue(event.component) ?? "unknown";
        const x = numberValue(event.day);
        if (!date || x === undefined) continue;
        const key = `${date}:${split}`;
        const examplesPerSecond = numberValue(event.examplesPerSecond);
        const point: DatasetPoint = {
          ...datasetByKey.get(key),
          ...oracleByDate.get(date),
          key,
          x,
          date,
          split,
          days: numberValue(event.days),
          preparationExamplesPerSecond: examplesPerSecond,
          examplesPerSecond,
          gpuMemoryMiB: numberValue(event.gpuMemoryMiB),
          kl: numberValue(event.meanKlDivergence),
          probabilityMse: numberValue(event.meanSquaredError),
          temporalBefore: numberValue(event.temporalMeanNormalizedStepBefore),
          temporalAfter: numberValue(event.temporalMeanNormalizedStepAfter),
          temporalSelectedPct: percentValue(event.temporalWarmSelectedFraction),
        };
        datasetByKey.set(key, point);
        if (!currentDataset() || x >= currentDataset()!.x) setCurrentDataset(point);
      } else if (event.event === "dataset-stage-timing") {
        const date = textValue(event.date);
        const split = textValue(event.split) ?? textValue(event.component) ?? "unknown";
        if (!date) continue;
        const key = `${date}:${split}`;
        const existing = datasetByKey.get(key);
        if (!existing) continue;
        const examples = numberValue(event.examples);
        const accepted = numberValue(event.acceptedExamples);
        const directDiagnosticsWorkerWallMs = numberValue(
          event.directDiagnosticsWorkerWallMs,
        );
        const point: DatasetPoint = {
          ...existing,
          preparationExamplesPerSecond: directDiagnosticsWorkerWallMs
            && examples
            ? examples * 1_000 / directDiagnosticsWorkerWallMs
            : existing.preparationExamplesPerSecond,
          acceptedPct: examples && accepted !== undefined ? 100 * accepted / examples : undefined,
          directDiagnosticsKernelMs: numberValue(event.directDiagnosticsKernelMs),
          directDiagnosticsCutoffKernelMs: numberValue(
            event.directDiagnosticsCutoffKernelMs,
          ),
          directDiagnosticsWorkerWallMs,
          teacherWallMs: numberValue(event.teacherWallMs),
          featureEncodingMs: numberValue(event.featureEncodingMs),
          persistMs: numberValue(event.persistMs),
        };
        datasetByKey.set(key, point);
        if (!currentDataset() || point.x >= currentDataset()!.x) setCurrentDataset(point);
      } else if (event.event === "train-step") {
        const globalStep = numberValue(event.globalStep);
        if (globalStep === undefined) continue;
        stepsById.set(globalStep, {
          globalStep,
          epoch: numberValue(event.epoch) ?? 0,
          learningRate: numberValue(event.learningRate),
          gradientNorm: numberValue(event.gradientNorm),
          examplesPerSecond: numberValue(event.examplesPerSecond),
          networkRowsPerSecond: numberValue(event.networkRowsPerSecond),
          batchCompactionRatio: numberValue(event.batchCompactionRatio),
          gpuAllocatedMiB: numberValue(event.gpuAllocatedMiB),
          gpuReservedMiB: numberValue(event.gpuReservedMiB),
          gpuDeviceUsedMiB: numberValue(event.gpuDeviceUsedMiB),
          latest: metricValues(event.latest),
        });
      } else if (event.event === "epoch") {
        const epoch = numberValue(event.epoch);
        if (epoch === undefined) continue;
        epochsById.set(epoch, {
          epoch,
          globalStep: numberValue(event.globalStep) ?? epoch,
          train: metricValues(event.train),
          validation: metricValues(event.validation),
          bestValidation: numberValue(event.bestValidation)
            ?? numberValue(event.bestValidationScore)
            ?? numberValue(event.bestRawValidationKl),
          staleEpochs: numberValue(event.staleEpochs),
          trainingTargetTemperature: numberValue(
            event.trainingTargetTemperature,
          ),
          curriculumTargetKl: numberValue(event.curriculumValidationKl)
            ?? metricValues(event.validation).curriculumTargetKl,
          learningRate: numberValue(event.learningRate),
          seconds: numberValue(event.seconds),
        });
      }
    }
    publishSeries();
  };

  const loadComparison = async (keys = comparisonRunKeys()) => {
    if (keys.length === 0) return;
    const generation = ++comparisonRequestGeneration;
    const search = new URLSearchParams({ runs: keys.join(",") });
    try {
      const response = await fetch(`${apiBase}/api/mlp-training/comparison?${search}`, {
        cache: "no-store",
      });
      const payload = await response.json() as ComparisonResponse & { error?: string };
      if (!response.ok) {
        throw new Error(payload.error ?? `Training comparison request failed: ${response.status}`);
      }
      if (disposed || generation !== comparisonRequestGeneration) return;
      setComparison(payload);
      setComparisonError(undefined);
    } catch (reason) {
      if (disposed || generation !== comparisonRequestGeneration) return;
      setComparisonError(
        reason instanceof Error ? reason.message : "Training comparison request failed.",
      );
    }
  };

  const updateComparisonRun = (index: number, runKey: string) => {
    if (!runKey) return;
    const next = [...comparisonRunKeys()];
    if (next.some((key, candidateIndex) => candidateIndex !== index && key === runKey)) return;
    next[index] = runKey;
    setComparisonRunKeys(next);
    setComparison(undefined);
    void loadComparison(next);
  };

  const addComparisonRun = () => {
    if (comparisonRunKeys().length >= 4) return;
    const nextKey = runs().find((run) => !comparisonRunKeys().includes(run.key))?.key;
    if (!nextKey) return;
    const next = [...comparisonRunKeys(), nextKey];
    setComparisonRunKeys(next);
    setComparison(undefined);
    void loadComparison(next);
  };

  const removeComparisonRun = (index: number) => {
    if (index === 0 || comparisonRunKeys().length <= 1) return;
    const next = comparisonRunKeys().filter((_, candidateIndex) => candidateIndex !== index);
    setComparisonRunKeys(next);
    setComparison(undefined);
    void loadComparison(next);
  };

  const setMatrixPaused = async (matrixId: string, paused: boolean) => {
    setMatrixControlPending(matrixId);
    setMatrixControlError(undefined);
    try {
      const response = await fetch(
        `${apiBase}/api/mlp-training/matrices/${encodeURIComponent(matrixId)}/control`,
        {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ action: paused ? "pause" : "resume" }),
        },
      );
      const payload = await response.json() as { error?: string };
      if (!response.ok) {
        throw new Error(payload.error ?? `Matrix control request failed: ${response.status}`);
      }
      setSnapshot((current) => current ? {
        ...current,
        matrices: current.matrices.map((matrix) => matrix.id === matrixId ? {
          ...matrix,
          pauseRequested: paused,
          paused: paused ? matrix.paused : false,
        } : matrix),
      } : current);
    } catch (reason) {
      setMatrixControlError(
        reason instanceof Error ? reason.message : "Matrix control request failed.",
      );
    } finally {
      setMatrixControlPending(undefined);
    }
  };

  const load = async () => {
    const generation = ++requestGeneration;
    const requestedRunKey = selectedRunKey();
    const search = new URLSearchParams({ cursor: String(cursor) });
    if (requestedRunKey) search.set("run", requestedRunKey);
    try {
      const response = await fetch(`${apiBase}/api/mlp-training/metrics?${search}`, {
        cache: "no-store",
      });
      const payload = await response.json() as MetricsResponse & { error?: string };
      if (!response.ok) throw new Error(payload.error ?? `Training metrics request failed: ${response.status}`);
      if (disposed || generation !== requestGeneration) return;
      const previous = snapshot();
      const runChanged = previous !== undefined
        && (previous.selectedRunKey !== payload.selectedRunKey
          || previous.plan.id !== payload.plan.id);
      if (payload.reset || runChanged) clearSeries();
      cursor = payload.cursor;
      applyEvents(payload.events);
      setRuns(payload.runs);
      if (comparisonRunKeys().length === 0) {
        const initialKeys = initialComparisonRunKeys(payload.runs, payload.selectedRunKey);
        setComparisonRunKeys(initialKeys);
        void loadComparison(initialKeys);
      }
      setSelectedRunKey(payload.selectedRunKey);
      setSnapshot(payload);
      setError(undefined);
    } catch (reason) {
      if (disposed || generation !== requestGeneration) return;
      setError(reason instanceof Error ? reason.message : "Training metrics request failed.");
    } finally {
      if (!disposed && generation === requestGeneration) {
        pollTimer = window.setTimeout(() => void load(), POLL_MS);
      }
    }
  };

  const selectRun = (runKey: string) => {
    if (!runKey || runKey === selectedRunKey()) return;
    requestGeneration += 1;
    if (pollTimer !== undefined) window.clearTimeout(pollTimer);
    cursor = 0;
    clearSeries();
    setSnapshot(undefined);
    setSelectedRunKey(runKey);
    setError(undefined);
    void load();
  };

  onMount(() => {
    void load();
    clockTimer = window.setInterval(() => setNow(Date.now()), 1_000);
    comparisonTimer = window.setInterval(() => void loadComparison(), 5_000);
  });
  onCleanup(() => {
    disposed = true;
    if (pollTimer !== undefined) window.clearTimeout(pollTimer);
    if (clockTimer !== undefined) window.clearInterval(clockTimer);
    if (comparisonTimer !== undefined) window.clearInterval(comparisonTimer);
  });

  const latestDataset = createMemo(() => currentDataset());
  const latestCompletedDataset = createMemo(() => {
    const points = datasetPoints();
    for (let index = points.length - 1; index >= 0; index -= 1) {
      if (points[index]!.persistMs !== undefined) return points[index];
    }
    return latestDataset();
  });
  const latestStep = createMemo(() => trainSteps().at(-1));
  const latestEpoch = createMemo(() => epochs().at(-1));
  const hasCurriculumEpochs = createMemo(() => epochs().some(
    (point) => point.trainingTargetTemperature !== undefined
      && point.curriculumTargetKl !== undefined,
  ));
  const hasRegressionEpochs = createMemo(() => epochs().some(
    (point) => point.validation.normalizedMse !== undefined,
  ));
  const stage = createMemo(() => snapshot()?.status?.stage ?? "idle");
  const stageProgress = createMemo(() => {
    if (stage().startsWith("dataset")) {
      const current = latestDataset();
      return current?.days ? current.x / current.days : 0;
    }
    if (stage() === "training") {
      const currentEpoch = latestStep()?.epoch ?? latestEpoch()?.epoch;
      const total = snapshot()?.plan.epochs;
      return total && currentEpoch !== undefined ? (currentEpoch + 1) / total : 0;
    }
    return stage() === "complete" || stage() === "archived" ? 1 : 0;
  });
  const updatedAgo = createMemo(() => {
    const timestamp = Date.parse(snapshot()?.status?.updatedAt ?? "");
    if (!Number.isFinite(timestamp)) return "never";
    return `${Math.max(0, Math.floor((now() - timestamp) / 1_000))}s ago`;
  });
  const comparisonTrainFit = createMemo<PlotSeries[]>(() => (comparison()?.runs ?? [])
    .map((run, index) => ({
      label: run.label,
      color: COMPARISON_COLORS[index % COMPARISON_COLORS.length]!,
      values: run.fit.flatMap((point) => point.trainNormalizedMse === undefined
        ? []
        : [{ x: point.epoch + 1, y: point.trainNormalizedMse }]),
    }))
    .filter((series) => series.values.length > 0));
  const comparisonValidationFit = createMemo<PlotSeries[]>(() => (comparison()?.runs ?? [])
    .map((run, index) => ({
      label: run.label,
      color: COMPARISON_COLORS[index % COMPARISON_COLORS.length]!,
      values: run.fit.flatMap((point) => point.validationNormalizedMse === undefined
        ? []
        : [{ x: point.epoch + 1, y: point.validationNormalizedMse }]),
    }))
    .filter((series) => series.values.length > 0));
  const comparisonRows = createMemo(() => {
    const comparedRuns = comparison()?.runs ?? [];
    return [
      { label: "Training examples", values: comparedRuns.map((run) => formatCount(run.examples)) },
      {
        label: "Parameters",
        values: comparedRuns.map((run) => formatCount(
          run.trainableParameterCount ?? run.parameterCount,
        )),
      },
      {
        label: "Best epoch",
        values: comparedRuns.map((run) => run.bestEpoch === undefined
          ? "—"
          : String(run.bestEpoch + 1)),
      },
      { label: "Train normalized MSE", values: comparedRuns.map((run) => formatMetric(run.train?.normalizedMse)) },
      { label: "Train MSE skill", values: comparedRuns.map((run) => formatPercent(run.train?.mseSkillVsZero)) },
      { label: "Validation normalized MSE", values: comparedRuns.map((run) => formatMetric(run.validation?.normalizedMse)) },
      { label: "Validation MSE skill", values: comparedRuns.map((run) => formatPercent(run.validation?.mseSkillVsZero)) },
      { label: "Validation correlation", values: comparedRuns.map((run) => formatMetric(run.validation?.correlation)) },
      { label: "Validation direction", values: comparedRuns.map((run) => formatPercent(run.validation?.directionAccuracy)) },
      { label: "Validation MAE", values: comparedRuns.map((run) => formatMetric(run.validation?.mae)) },
      { label: "Test normalized MSE", values: comparedRuns.map((run) => formatMetric(run.test?.normalizedMse)) },
    ];
  });

  return (
    <main class="min-h-screen overflow-x-hidden bg-ink-950 text-ink-100">
      <div class="mx-auto flex w-full max-w-[96rem] flex-col gap-4 px-4 py-4 lg:px-6">
        <header class="flex min-w-0 flex-col gap-3 border-b border-line pb-4 lg:flex-row lg:items-center lg:justify-between">
          <div class="min-w-0 flex-1">
            <div class="muted-label">Machine learning</div>
            <div class="mt-1 flex flex-wrap items-center gap-3">
              <h1 class="text-2xl font-semibold">Live MLP training</h1>
              <StatusPill running={Boolean(snapshot()?.running)} label={snapshot()?.running ? "Live" : stage()} />
            </div>
            <p class="mt-1 max-w-4xl break-words text-sm text-ink-300">
              {snapshot()?.plan.label ?? "Loading the active training plan…"}
            </p>
          </div>
          <div class="flex w-full min-w-0 flex-col gap-2 lg:w-[32rem] lg:max-w-[45vw] lg:flex-none lg:items-end">
            <label class="flex w-full min-w-0 flex-col gap-1">
              <span class="muted-label">Training run</span>
              <select
                class="block w-full min-w-0 max-w-full truncate rounded border border-line bg-ink-900 px-3 py-2 text-sm text-ink-100 outline-none focus:border-accent"
                value={selectedRunKey() ?? ""}
                disabled={runs().length === 0}
                onChange={(event) => selectRun(event.currentTarget.value)}
              >
                <Show when={runs().length === 0}>
                  <option value="">Loading runs…</option>
                </Show>
                <For each={runs()}>
                  {(run) => (
                    <option
                      value={run.key}
                      selected={run.key === selectedRunKey()}
                    >
                      {runSummaryLabel(run)}
                    </option>
                  )}
                </For>
              </select>
            </label>
            <div class="flex flex-wrap gap-2">
              <a class="btn" href="#/portfolio-index"><BarChart3 size={16} /> Basis Index</a>
              <a class="btn" href="#/kama-inspector"><Search size={16} /> KAMA Inspector</a>
              <a class="btn" href="#/"><ArrowLeft size={16} /> Dashboard</a>
            </div>
          </div>
        </header>

        <Show when={error()}>
          {(message) => (
            <div class="rounded-2 border border-loss/50 bg-loss/10 px-4 py-3 text-sm text-loss">
              {message()} Retrying automatically.
            </div>
          )}
        </Show>

        <For each={snapshot()?.matrices ?? []}>
          {(matrix) => (
            <section class="panel flex flex-col gap-3" data-testid={`training-matrix-${matrix.id}`}>
              <div class="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <div class="muted-label">Experiment matrix</div>
                  <div class="mt-1 text-lg font-semibold">{matrix.label}</div>
                </div>
                <div class="flex items-start gap-3">
                  <div class="text-right text-sm tabular-nums text-ink-300">
                    <div>{matrix.completedRuns} / {matrix.totalRuns} runs complete</div>
                    <div>{Math.round(matrix.progress * 100)}% total progress</div>
                  </div>
                  <Show when={matrix.controllable && matrix.completedRuns < matrix.totalRuns}>
                    <button
                      class="btn"
                      type="button"
                      disabled={matrixControlPending() === matrix.id}
                      title={matrix.pauseRequested
                        ? "Resume from the last durable checkpoint"
                        : "Pause after the current epoch or validation pass"}
                      onClick={() => void setMatrixPaused(matrix.id, !matrix.pauseRequested)}
                    >
                      <Show when={matrix.pauseRequested} fallback={<><Pause size={16} /> Pause</>}>
                        <Play size={16} /> Resume
                      </Show>
                    </button>
                  </Show>
                </div>
              </div>
              <div class="h-2 overflow-hidden rounded-full bg-ink-800">
                <div
                  class="h-full rounded-full bg-accent transition-[width] duration-500"
                  style={{ width: `${Math.max(0, Math.min(100, matrix.progress * 100))}%` }}
                />
              </div>
              <div class="flex flex-wrap gap-x-5 gap-y-1 text-xs text-ink-300">
                <span>{matrix.queuedRuns} queued</span>
                <Show when={matrix.paused}>
                  <span class="text-warn">Paused at a durable epoch boundary</span>
                </Show>
                <Show when={matrix.pauseRequested && !matrix.paused}>
                  <span class="text-warn">Pause requested; finishing the current safe boundary</span>
                </Show>
                <Show when={matrix.failedRuns > 0}>
                  <span class="text-loss">{matrix.failedRuns} failed</span>
                </Show>
                <Show when={matrix.active}>
                  {(active) => (
                    <>
                      <span class="text-accent">Active: {active().label}</span>
                      <Show when={active().epoch !== undefined && active().epochs !== undefined}>
                        <span>
                          Epoch {(active().epoch ?? 0) + 1} / {active().epochs}
                        </span>
                      </Show>
                      <Show when={active().bestTrainScore !== undefined}>
                        <span>Best normalized MSE: {formatMetric(active().bestTrainScore)}</span>
                      </Show>
                    </>
                  )}
                </Show>
              </div>
              <Show when={matrixControlError()}>
                {(message) => (
                  <div class="rounded border border-loss/50 bg-loss/10 px-3 py-2 text-sm text-loss">
                    {message()}
                  </div>
                )}
              </Show>
            </section>
          )}
        </For>

        <section class="panel flex flex-col gap-4" data-testid="training-run-comparison">
          <div class="flex flex-wrap items-start justify-between gap-3">
            <div>
              <div class="muted-label">Experiment analysis</div>
              <div class="mt-1 text-lg font-semibold">Run comparison</div>
              <p class="mt-1 text-sm text-ink-300">
                Compare final train and held-out results, then inspect how each model fit over time.
              </p>
            </div>
            <button
              class="btn"
              type="button"
              disabled={comparisonRunKeys().length >= 4
                || comparisonRunKeys().length >= runs().length}
              onClick={addComparisonRun}
            >
              <Plus size={16} /> Add run
            </button>
          </div>

          <div class="grid min-w-0 gap-3 md:grid-cols-2 xl:grid-cols-4">
            <For each={comparisonRunKeys()}>
              {(runKey, index) => (
                <label class="flex min-w-0 flex-col gap-1">
                  <span class="muted-label">
                    {index() === 0 ? "Baseline" : `Candidate ${index()}`}
                  </span>
                  <div class="flex min-w-0 gap-2">
                    <select
                      class="block min-w-0 flex-1 truncate rounded border border-line bg-ink-900 px-3 py-2 text-sm text-ink-100 outline-none focus:border-accent"
                      value={runKey}
                      onChange={(event) => updateComparisonRun(index(), event.currentTarget.value)}
                    >
                      <For each={runs()}>
                        {(run) => (
                          <option
                            value={run.key}
                            selected={run.key === runKey}
                            disabled={comparisonRunKeys().some(
                              (selected, selectedIndex) => selectedIndex !== index()
                                && selected === run.key,
                            )}
                          >
                            {runSummaryLabel(run)}
                          </option>
                        )}
                      </For>
                    </select>
                    <Show when={index() > 0}>
                      <button
                        class="btn px-2"
                        type="button"
                        title="Remove comparison run"
                        onClick={() => removeComparisonRun(index())}
                      >
                        <X size={16} />
                      </button>
                    </Show>
                  </div>
                </label>
              )}
            </For>
          </div>

          <Show when={comparisonError()}>
            {(message) => (
              <div class="rounded border border-loss/50 bg-loss/10 px-3 py-2 text-sm text-loss">
                {message()}
              </div>
            )}
          </Show>

          <Show
            when={comparison()?.runs.length}
            fallback={<div class="rounded border border-line bg-ink-900/40 px-4 py-6 text-sm text-ink-400">Loading comparison…</div>}
          >
            <div class="min-w-0 overflow-x-auto rounded border border-line">
              <table class="w-full min-w-[48rem] border-collapse text-sm">
                <thead class="bg-ink-900/80 text-left">
                  <tr>
                    <th class="px-3 py-3 text-xs font-medium uppercase tracking-wide text-ink-400">Metric</th>
                    <For each={comparison()?.runs ?? []}>
                      {(run, index) => (
                        <th class="min-w-[12rem] px-3 py-3 font-medium">
                          <div class="flex items-center gap-2">
                            <span
                              class="h-2.5 w-2.5 flex-none rounded-full"
                              style={{ "background-color": COMPARISON_COLORS[index() % COMPARISON_COLORS.length] }}
                            />
                            <span class="line-clamp-2">{run.label}</span>
                          </div>
                          <div class="mt-1 text-xs font-normal text-ink-400">
                            {run.running ? "training" : stageLabel(run.stage ?? "idle")}
                          </div>
                        </th>
                      )}
                    </For>
                  </tr>
                </thead>
                <tbody>
                  <For each={comparisonRows()}>
                    {(row) => (
                      <tr class="border-t border-line even:bg-ink-900/25">
                        <th class="whitespace-nowrap px-3 py-2 text-left font-normal text-ink-300">{row.label}</th>
                        <For each={row.values}>
                          {(value) => <td class="px-3 py-2 tabular-nums text-ink-100">{value}</td>}
                        </For>
                      </tr>
                    )}
                  </For>
                </tbody>
              </table>
            </div>

            <div class="grid min-w-0 gap-3 xl:grid-cols-2">
              <Show when={comparisonTrainFit().length > 0}>
                <MetricChart
                  title="Training fit"
                  subtitle="Normalized MSE by epoch; lower is better"
                  scale="log"
                  xLabel="epoch"
                  series={comparisonTrainFit()}
                />
              </Show>
              <Show when={comparisonValidationFit().length > 0}>
                <MetricChart
                  title="Validation fit"
                  subtitle="Held-out normalized MSE by epoch; lower is better"
                  scale="log"
                  xLabel="epoch"
                  series={comparisonValidationFit()}
                />
              </Show>
            </div>
          </Show>
        </section>

        <section class="panel flex flex-col gap-3">
          <div class="flex flex-wrap items-center justify-between gap-3">
            <div>
              <div class="muted-label">Current stage</div>
              <div class="mt-1 text-lg font-semibold">{stageLabel(stage())}</div>
            </div>
            <div class="min-w-0 text-right text-sm text-ink-300">
              <div class="break-all">{snapshot()?.plan.id ?? "—"}</div>
              <div>Updated {updatedAgo()}</div>
            </div>
          </div>
          <div class="h-2 overflow-hidden rounded-full bg-ink-800">
            <div
              class="h-full rounded-full bg-accent transition-[width] duration-500"
              style={{ width: `${Math.max(0, Math.min(100, stageProgress() * 100))}%` }}
            />
          </div>
          <div class="flex flex-wrap justify-between gap-2 text-xs text-ink-300">
            <span>{progressLabel(
              stage(),
              latestDataset(),
              latestStep(),
              latestEpoch(),
              snapshot()?.plan.epochs,
            )}</span>
            <Show when={snapshot()?.plan.patience !== undefined}>
              <span>Early-stop patience: {snapshot()?.plan.patience} epochs</span>
            </Show>
            <Show when={snapshot()?.finalizeRequested}><span class="text-warn">Finalize requested</span></Show>
          </div>
          <Show when={snapshot()?.plan.archived && snapshot()?.plan.bestValidationKl !== undefined}>
            <div class="rounded border border-accent/30 bg-accent/5 px-3 py-2 text-sm text-ink-200">
              Best logged validation forward KL: <span class="font-semibold text-accent">
                {snapshot()!.plan.bestValidationKl!.toFixed(5)}
              </span> at epoch {(snapshot()!.plan.bestValidationKlEpoch ?? 0) + 1}
              <span class="text-ink-400"> (stored epoch {snapshot()!.plan.bestValidationKlEpoch ?? 0})</span>
            </div>
          </Show>
        </section>

        <section class="flex flex-col gap-3">
          <SectionHeading
            title="Network optimization"
            subtitle={stage() === "archived"
              ? `${trainSteps().length} historical updates · ${epochs().length} completed epochs`
              : stage() === "training"
              ? epochs().length > 0
                ? `${epochs().length} completed epochs · updating live`
                : `${trainSteps().length} logged updates · updating live`
              : trainSteps().length > 0
                ? `${trainSteps().length} updates from the most recent training attempt; live updates resume after refinement`
                : "Plots will populate when refinement hands off to weight training"}
          />
          <Show when={trainSteps().length > 0 || epochs().length > 0} fallback={<WaitingForTraining stage={stage()} />}>
            <div class="grid min-w-0 gap-3 xl:grid-cols-2">
              <Show when={hasCurriculumEpochs()}>
                <MetricChart title="Gate: curriculum target KL" subtitle="Temperature takes one downward step only when validation KL is at or below 0.05" scale="log" xLabel="epoch" series={[
                  epochPlot("Train", "#38bdf8", epochs(), "train", "curriculumTargetKl"),
                  epochPlot("Validation", "#f5b84b", epochs(), "validation", "curriculumTargetKl"),
                  constantEpochPlot("Gate threshold", "#fb7185", epochs(), 0.05),
                ]} />
              </Show>
              <Show when={epochs().some((point) => point.trainingTargetTemperature !== undefined)}>
                <MetricChart title="Curriculum target temperature" subtitle="Held when validation KL exceeds 0.05; every allowed change targets one equal validation-entropy decrease toward 0.01" scale="log" xLabel="epoch" series={[
                  directEpochPlot("Target temperature", "#fb7185", epochs(), "trainingTargetTemperature"),
                ]} />
              </Show>
              <Show when={hasCurriculumEpochs()}>
                <MetricChart title="Curriculum target cross-entropy" subtitle="The optimization objective at the current target temperature" scale="log" xLabel="epoch" series={[
                  epochPlot("Train", "#38bdf8", epochs(), "train", "trainingCrossEntropy"),
                  epochPlot("Validation", "#f5b84b", epochs(), "validation", "trainingCrossEntropy"),
                ]} />
                <MetricChart title="Combined training objective" subtitle="Curriculum cross-entropy plus restored regularizers" scale="log" xLabel="epoch" series={[
                  epochPlot("Train", "#38bdf8", epochs(), "train", "loss"),
                  epochPlot("Validation", "#f5b84b", epochs(), "validation", "loss"),
                ]} />
                <MetricChart title="Soft-layer normalization" subtitle="Weight 1 · raw value and gate projections" scale="log" xLabel="epoch" series={[
                  epochPlot("Train", "#38bdf8", epochs(), "train", "softLayerNorm"),
                  epochPlot("Validation", "#f5b84b", epochs(), "validation", "softLayerNorm"),
                ]} />
                <MetricChart title="Soft weight bound" subtitle="Displayed before its 0.01 loss weight" scale="log" xLabel="epoch" series={[
                  epochPlot("Train", "#38bdf8", epochs(), "train", "softWeightBound"),
                  epochPlot("Validation", "#f5b84b", epochs(), "validation", "softWeightBound"),
                ]} />
                <MetricChart title="Soft-target cross-entropy" subtitle="Validation distribution at the current curriculum temperature" xLabel="epoch" series={[
                  epochPlot("Cross entropy", "#f5b84b", epochs(), "validation", "trainingCrossEntropy"),
                  epochPlot("Oracle entropy", "#22c55e", epochs(), "validation", "curriculumTargetEntropy"),
                  epochPlot("Predicted entropy", "#38bdf8", epochs(), "validation", "predictedEntropy"),
                ]} />
                <MetricChart title="Raw production-temperature KL" subtitle="At target temperature 0.01, this becomes identical to curriculum target KL" scale="log" xLabel="epoch" series={[
                  epochPlot("Train", "#38bdf8", epochs(), "train", "rawBaseActionKl"),
                  epochPlot("Validation", "#f5b84b", epochs(), "validation", "rawBaseActionKl"),
                  directEpochPlot("Best validation", "#22c55e", epochs(), "bestValidation"),
                ]} />
                <MetricChart title="Raw production-temperature cross-entropy" scale="log" xLabel="epoch" series={[
                  epochPlot("Train", "#38bdf8", epochs(), "train", "rawCrossEntropy"),
                  epochPlot("Validation", "#f5b84b", epochs(), "validation", "rawCrossEntropy"),
                ]} />
                <MetricChart title="Production-temperature entropy" subtitle="Validation distribution against the production oracle at temperature 0.01" xLabel="epoch" series={[
                  epochPlot("Cross entropy", "#f5b84b", epochs(), "validation", "rawCrossEntropy"),
                  epochPlot("Oracle entropy", "#22c55e", epochs(), "validation", "rawTargetEntropy"),
                  epochPlot("Predicted entropy", "#38bdf8", epochs(), "validation", "predictedEntropy"),
                ]} />
                <MetricChart title="Raw production-temperature probability MSE" scale="log" xLabel="epoch" series={[
                  epochPlot("Train", "#38bdf8", epochs(), "train", "rawProbabilityMse"),
                  epochPlot("Validation", "#f05252", epochs(), "validation", "rawProbabilityMse"),
                ]} />
                <MetricChart title="Curriculum target probability MSE" scale="log" xLabel="epoch" series={[
                  epochPlot("Train", "#38bdf8", epochs(), "train", "curriculumTargetProbabilityMse"),
                  epochPlot("Validation", "#f05252", epochs(), "validation", "curriculumTargetProbabilityMse"),
                ]} />
                <MetricChart title="Curriculum target entropy" subtitle="Every allowed temperature change targets the same validation-entropy decrease; KL-gate holds create plateaus" xLabel="epoch" series={[
                  epochPlot("Train", "#38bdf8", epochs(), "train", "curriculumTargetEntropy"),
                  epochPlot("Validation", "#f5b84b", epochs(), "validation", "curriculumTargetEntropy"),
                ]} />
                <MetricChart title="Learning rate" subtitle="Rate calibrated from 1e-4 toward 1e-6 over 300 stale epochs, applied in 24-epoch blocks; an improvement resets the plateau counter" scale="log" xLabel="epoch" series={[
                  directEpochPlot("Learning rate", "#38bdf8", epochs(), "learningRate"),
                ]} />
                <MetricChart title="Epoch duration" unit="seconds" xLabel="epoch" series={[
                  directEpochPlot("Duration", "#a78bfa", epochs(), "seconds"),
                ]} />
              </Show>
              <Show when={!hasCurriculumEpochs()}>
                <>
              <Show when={hasRegressionEpochs()}>
                <MetricChart title="Normalized MSE" subtitle="Lower is better; validation determines checkpoint selection" scale="log" xLabel="epoch" series={[
                  epochPlot("Train", "#38bdf8", epochs(), "train", "normalizedMse"),
                  epochPlot("Validation", "#f5b84b", epochs(), "validation", "normalizedMse"),
                  directEpochPlot("Best validation", "#22c55e", epochs(), "bestValidation"),
                ]} />
                <MetricChart title="MSE skill versus zero" subtitle="Positive means lower MSE than always predicting zero" unit="ratio" xLabel="epoch" series={[
                  epochPlot("Train", "#38bdf8", epochs(), "train", "mseSkillVsZero"),
                  epochPlot("Validation", "#22c55e", epochs(), "validation", "mseSkillVsZero"),
                ]} />
                <MetricChart title="Direction accuracy" unit="ratio" yDomain={[0, 1]} xLabel="epoch" series={[
                  epochPlot("Train", "#38bdf8", epochs(), "train", "directionAccuracy"),
                  epochPlot("Validation", "#f5b84b", epochs(), "validation", "directionAccuracy"),
                ]} />
                <MetricChart title="Return correlation" xLabel="epoch" series={[
                  epochPlot("Train", "#38bdf8", epochs(), "train", "correlation"),
                  epochPlot("Validation", "#a78bfa", epochs(), "validation", "correlation"),
                ]} />
                <MetricChart title="Prediction and target dispersion" scale="log" xLabel="epoch" series={[
                  epochPlot("Prediction std", "#38bdf8", epochs(), "validation", "predictionStd"),
                  epochPlot("Target std", "#f5b84b", epochs(), "validation", "targetStd"),
                ]} />
              </Show>
              <MetricChart title="Latest batch loss" scale="log" xLabel="global step" series={[
                metricPlot("Total", "#38bdf8", trainSteps(), "loss"),
              ]} />
              <MetricChart title="Epoch loss" scale="log" xLabel="epoch" series={[
                epochPlot("Train", "#38bdf8", epochs(), "train", "loss"),
                epochPlot("Validation", "#f5b84b", epochs(), "validation", "loss"),
                directEpochPlot("Best validation", "#22c55e", epochs(), "bestValidation"),
              ]} />
              <MetricChart title="Distribution losses" scale="log" xLabel="global step" series={[
                metricPlot("Cross entropy", "#f5b84b", trainSteps(), "crossEntropy"),
                metricPlot("Conditional KL", "#38bdf8", trainSteps(), "klDivergence"),
                metricPlot("Base-action KL", "#22c55e", trainSteps(), "baseKlDivergence"),
                metricPlot("Skew reverse KL", "#a78bfa", trainSteps(), "reverseKlDivergence"),
                metricPlot("Entropy sharpness", "#fb7185", trainSteps(), "entropySharpness"),
                metricPlot("Probability MSE", "#f05252", trainSteps(), "probabilityMse"),
              ]} />
              <MetricChart title="Validation conditional KL" subtitle="Actual fee-conditioned distribution on the visible range" scale="log" xLabel="epoch" series={[
                epochPlot("Mean conditional KL", "#38bdf8", epochs(), "validation", "klDivergence"),
                epochPlot("Conditional KL standard deviation", "#f5b84b", epochs(), "validation", "klDivergenceStdDev"),
                epochPlot("Conditional KL variance", "#a78bfa", epochs(), "validation", "klDivergenceVariance"),
              ]} />
              <MetricChart title="Validation base-action KL" subtitle="Stored raw oracle factor versus the direct 255-logit head" scale="log" xLabel="epoch" series={[
                epochPlot("Mean base-action KL", "#22c55e", epochs(), "validation", "baseKlDivergence"),
              ]} />
              <MetricChart title="Validation skew reverse KL" subtitle={reverseKlLabel(
                snapshot()?.plan.lossWeights?.reverseKl,
                snapshot()?.plan.reverseKl?.predictionMixtureWeight,
                snapshot()?.plan.outputRegularizer,
              )} scale="log" xLabel="epoch" series={[
                epochPlot("KL(P || (1-epsilon)Q + epsilon P)", "#a78bfa", epochs(), "validation", "reverseKlDivergence"),
              ]} />
              <MetricChart title="Validation probability MSE" subtitle="Per-example visible-range surface error" scale="log" xLabel="epoch" series={[
                epochPlot("Mean pMSE", "#f05252", epochs(), "validation", "probabilityMse"),
                epochPlot("pMSE standard deviation", "#f5b84b", epochs(), "validation", "probabilityMseStdDev"),
                epochPlot("pMSE variance", "#a78bfa", epochs(), "validation", "probabilityMseVariance"),
              ]} />
              <MetricChart title="Excess entropy" subtitle={lossWeightLabel(snapshot()?.plan.lossWeights?.excessEntropy)} xLabel="global step" series={[
                metricPlot("Excess entropy", "#a78bfa", trainSteps(), "excessEntropy"),
              ]} />
              <MetricChart title="Oracle mutual information" subtitle={lossWeightLabel(snapshot()?.plan.lossWeights?.oracleMutualInformation)} xLabel="global step" series={[
                metricPlot("Oracle MI", "#22c55e", trainSteps(), "oracleMutualInformation"),
              ]} />
              <MetricChart title="Soft-target cross-entropy" subtitle={lossWeightLabel(snapshot()?.plan.lossWeights?.crossEntropy)} xLabel="global step" series={[
                metricPlot("Cross entropy", "#f5b84b", trainSteps(), "crossEntropy"),
                metricPlot("Oracle entropy", "#22c55e", trainSteps(), "targetEntropy"),
                metricPlot("Predicted entropy", "#38bdf8", trainSteps(), "predictedEntropy"),
              ]} />
              <MetricChart title="Skew reverse KL" subtitle={reverseKlLabel(
                snapshot()?.plan.lossWeights?.reverseKl,
                snapshot()?.plan.reverseKl?.predictionMixtureWeight,
                snapshot()?.plan.outputRegularizer,
              )} scale="log" xLabel="global step" series={[
                metricPlot("KL(P || (1-epsilon)Q + epsilon P)", "#a78bfa", trainSteps(), "reverseKlDivergence"),
              ]} />
              <MetricChart title="One-sided entropy sharpness" subtitle={gatedLossWeightLabel(
                snapshot()?.plan.lossWeights?.entropySharpness,
                snapshot()?.plan.outputRegularizer,
                "entropySharpness",
              )} scale="log" xLabel="global step" series={[
                metricPlot("ReLU(H(P) - H(Q)) squared", "#fb7185", trainSteps(), "entropySharpness"),
              ]} />
              <MetricChart title="Soft LayerNorm" subtitle={lossWeightLabel(snapshot()?.plan.lossWeights?.softLayerNorm)} scale="log" xLabel="global step" series={[
                metricPlot("Total penalty", "#a78bfa", trainSteps(), "softLayerNorm"),
                metricPlot("Mean penalty", "#38bdf8", trainSteps(), "softLayerNormMeanPenalty"),
                metricPlot("Variance penalty", "#f5b84b", trainSteps(), "softLayerNormVariancePenalty"),
              ]} />
              <Show when={snapshot()?.plan.branchNormalization?.learnableCentering !== false}>
                <MetricChart title="Learnable centering projector" subtitle={centeringConstraintLabel(
                  snapshot()?.plan.lossWeights?.centeringIdempotence,
                  snapshot()?.plan.lossWeights?.centeringSymmetry,
                )} scale="log" xLabel="global step" series={[
                  metricPlot("Weighted constraint", "#22c55e", trainSteps(), "centeringConstraint"),
                  metricPlot("Idempotence C^2 - C", "#38bdf8", trainSteps(), "centeringIdempotence"),
                  metricPlot("Symmetry C^T - C", "#f5b84b", trainSteps(), "centeringSymmetry"),
                ]} />
              </Show>
              <MetricChart title="Distribution-like hidden layers" subtitle={distributionLayerLabel(
                snapshot()?.plan.lossWeights?.distributionLayerSum,
                snapshot()?.plan.lossWeights?.distributionLayerNegative,
              )} scale="log" xLabel="global step" series={[
                metricPlot("Weighted loss", "#22c55e", trainSteps(), "distributionLayer"),
                metricPlot("Width-normalized unit-sum", "#38bdf8", trainSteps(), "distributionLayerSumPenalty"),
                metricPlot("Mean negative penalty", "#f5b84b", trainSteps(), "distributionLayerNegativePenalty"),
              ]} />
              <MetricChart title="Soft weight bound" subtitle={softWeightBoundLabel(
                snapshot()?.plan.lossWeights?.softWeightBound,
                snapshot()?.plan.softWeightBound?.desiredMagnitude,
              )} scale="log" xLabel="global step" series={[
                metricPlot("Mean squared smooth excess", "#f05252", trainSteps(), "softWeightBound"),
              ]} />
              <MetricChart title="Distance-imbalance time weighting" unit="ratio" yDomain={[0, 1]} xLabel="global step" series={[
                metricPlot("Mean weight", "#38bdf8", trainSteps(), "distanceImbalanceWeight"),
                metricPlot("Effective sample ratio", "#f5b84b", trainSteps(), "timeWeightEffectiveSampleRatio"),
              ]} />
              <MetricChart title="Learning rate" scale="log" xLabel="global step" series={[
                directStepPlot("Learning rate", "#38bdf8", trainSteps(), "learningRate"),
              ]} />
              <MetricChart title="Gradient norm" scale="log" xLabel="global step" series={[
                directStepPlot("Gradient norm", "#f5b84b", trainSteps(), "gradientNorm"),
              ]} />
              <MetricChart title="Training throughput" unit="examples/s" xLabel="global step" series={[
                directStepPlot("Weighted source examples", "#22c55e", trainSteps(), "examplesPerSecond"),
                directStepPlot("Network rows", "#38bdf8", trainSteps(), "networkRowsPerSecond"),
              ]} />
              <MetricChart title="Training GPU memory" unit="MiB" xLabel="global step" series={[
                directStepPlot("Allocated tensors", "#a78bfa", trainSteps(), "gpuAllocatedMiB"),
                directStepPlot("PyTorch reserved", "#38bdf8", trainSteps(), "gpuReservedMiB"),
                directStepPlot("Whole device", "#f5b84b", trainSteps(), "gpuDeviceUsedMiB"),
              ]} />
                </>
              </Show>
            </div>
          </Show>
        </section>

        <Show when={snapshot()?.plan.timeWeighting}>
          {(weighting) => (
            <section class="panel flex flex-col gap-3">
              <SectionHeading
                title="Persistent same-side example weighting"
                subtitle="Active training-plan values. Dₜ and Wₜ are each one scalar for the complete timestamp example; training only normalizes Wₜ by the current batch mean."
              />
              <div class="grid grid-cols-2 gap-3 md:grid-cols-4 xl:grid-cols-9">
                <MetricCard
                  label="Signal aggregation"
                  value={weighting().stateAggregation === "globalDistanceRatio"
                    ? "Global distance ratio"
                    : weighting().stateAggregation}
                />
                <MetricCard
                  label="Advice threshold"
                  value={`|Dₜ| ≥ ${formatMetric(weighting().minimumAdviceMagnitude)}`}
                />
                <MetricCard
                  label="Evidence half-life"
                  value={stepDuration(
                    weighting().memoryHalfLifeSteps,
                    snapshot()?.plan.samplingIntervalMs,
                  )}
                />
                <MetricCard
                  label="Growth / prior advice"
                  value={`+${formatMetric(weighting().growthPerPriorAdvice)}×`}
                />
                <MetricCard
                  label="Multiplier cap"
                  value={`${formatMetric(weighting().maximumMultiplier)}×`}
                />
                <MetricCard
                  label="Gap reset"
                  value={`>${stepDuration(
                    weighting().resetAfterGapSteps,
                    snapshot()?.plan.samplingIntervalMs,
                  )}`}
                />
                <MetricCard
                  label="Distance epsilon"
                  value={formatMetric(weighting().distanceEpsilon)}
                />
                <MetricCard
                  label="Minimum weight"
                  value={formatMetric(weighting().minimumWeight)}
                />
                <MetricCard
                  label="Resolution JSD boost"
                  value={`1 + ${formatMetric(weighting().resolutionDivergenceMultiplier)}×JSD`}
                />
              </div>
            </section>
          )}
        </Show>

        <section class="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
          <MetricCard label="Prediction delay" value={formatDurationMs(snapshot()?.plan.predictionDelayMs)} />
          <MetricCard label="Input days" value={integer(snapshot()?.progress.featureComponents)} />
          <MetricCard label="Oracle days" value={integer(snapshot()?.progress.oracleComponents)} />
          <MetricCard label="Preparation rate" value={formatUnit(latestCompletedDataset()?.preparationExamplesPerSecond, " ex/s")} />
          <MetricCard label="Preparation GPU" value={formatUnit(latestCompletedDataset()?.gpuMemoryMiB, " MiB", 1)} />
          <MetricCard label="Fit KL" value={formatMetric(latestCompletedDataset()?.kl)} />
          <MetricCard label="Fit pMSE" value={formatMetric(latestCompletedDataset()?.probabilityMse)} />
          <MetricCard label="Oracle kernel" value={formatUnit(latestDataset()?.oracleKernelMs, " ms", 1)} />
          <MetricCard label="Feature encoding" value={formatUnit(latestCompletedDataset()?.featureEncodingMs, " ms", 1)} />
          <MetricCard label="Persistence" value={formatUnit(latestCompletedDataset()?.persistMs, " ms", 1)} />
          <MetricCard label="Pipeline wait" value={formatUnit(latestDataset()?.oraclePipelineWaitMs, " ms", 3)} />
          <MetricCard label="Refined shards" value={ratio(snapshot()?.progress.refinedShards, snapshot()?.progress.totalShards)} />
          <MetricCard label="Queued fits" value={integer(snapshot()?.progress.remainingTeacherFits)} />
          <MetricCard label="Last global step" value={integer(latestStep()?.globalStep)} />
          <MetricCard label="Last batch loss" value={formatMetric(latestStep()?.latest.loss)} />
          <MetricCard label="Last best validation" value={formatMetric(latestEpoch()?.bestValidation)} />
          <MetricCard label="Validation MSE skill" value={formatPercent(latestEpoch()?.validation.mseSkillVsZero)} />
          <MetricCard label="Validation direction" value={formatPercent(latestEpoch()?.validation.directionAccuracy)} />
          <MetricCard label="Validation correlation" value={formatMetric(latestEpoch()?.validation.correlation)} />
          <MetricCard label="Curriculum KL" value={formatMetric(latestEpoch()?.curriculumTargetKl)} />
          <MetricCard label="Target temperature" value={formatMetric(latestEpoch()?.trainingTargetTemperature)} />
          <MetricCard label="Learning rate" value={formatMetric(latestStep()?.learningRate)} />
          <MetricCard label="Activation dropout" value={formatUnit(
            snapshot()?.plan.dropout === undefined ? undefined : snapshot()!.plan.dropout! * 100,
            "%",
            1,
          )} />
          <MetricCard label="Dropout application" value={formatUnit(
            snapshot()?.plan.dropoutRate === undefined ? undefined : snapshot()!.plan.dropoutRate! * 100,
            "%",
            1,
          )} />
          <MetricCard label="Train rate" value={formatUnit(latestStep()?.examplesPerSecond, " ex/s", 1)} />
          <MetricCard label="Network rate" value={formatUnit(latestStep()?.networkRowsPerSecond, " rows/s", 1)} />
          <MetricCard label="Batch compaction" value={formatUnit(latestStep()?.batchCompactionRatio, "×", 1)} />
          <MetricCard label="GPU tensors" value={formatUnit(latestStep()?.gpuAllocatedMiB, " MiB", 1)} />
          <MetricCard label="GPU reserved" value={formatUnit(latestStep()?.gpuReservedMiB, " MiB", 1)} />
          <MetricCard label="GPU device" value={formatUnit(latestStep()?.gpuDeviceUsedMiB, " MiB", 1)} />
        </section>

        <Show when={datasetPoints().length > 0}>
          <section class="flex flex-col gap-3">
            <SectionHeading title="Dataset preparation" subtitle={`${datasetPoints().length} completed day/component batches`} />
            <div class="grid min-w-0 gap-3 xl:grid-cols-2">
              <MetricChart title="Preparation throughput" unit="examples/s" series={[
                plot("Examples / second", "#38bdf8", datasetPoints(), "preparationExamplesPerSecond"),
              ]} />
              <MetricChart title="Accepted examples" unit="%" yDomain={[0, 100]} series={[
                plot("Accepted", "#22c55e", datasetPoints(), "acceptedPct"),
              ]} />
              <MetricChart title="KL divergence" series={[
                plot("Mean KL", "#f5b84b", datasetPoints(), "kl"),
              ]} />
              <MetricChart title="Probability MSE" scale="log" series={[
                plot("pMSE", "#f05252", datasetPoints(), "probabilityMse"),
              ]} />
              <MetricChart title="Temporal parameter continuity" subtitle="Lower is smoother" series={[
                plot("Before", "#aeb6c8", datasetPoints(), "temporalBefore"),
                plot("After", "#38bdf8", datasetPoints(), "temporalAfter"),
              ]} />
              <MetricChart title="Daily pipeline time" unit="ms" scale="log" series={[
                plot("Teacher / CPU materialization", "#38bdf8", datasetPoints(), "teacherWallMs"),
                plot("Oracle kernel", "#a78bfa", datasetPoints(), "oracleKernelMs"),
                plot("Direct diagnostics", "#22c55e", datasetPoints(), "directDiagnosticsKernelMs"),
                plot("Feature encoding", "#f05252", datasetPoints(), "featureEncodingMs"),
                plot("Persist", "#f5b84b", datasetPoints(), "persistMs"),
              ]} />
              <MetricChart title="GPU memory" unit="MiB" series={[
                plot("Allocated", "#a78bfa", datasetPoints(), "gpuMemoryMiB"),
              ]} />
              <MetricChart title="Temporal warm selections" unit="%" yDomain={[0, 100]} series={[
                plot("Selected", "#22c55e", datasetPoints(), "temporalSelectedPct"),
              ]} />
            </div>
          </section>
        </Show>

        <Show when={epochs().length > 0}>
          <section class="panel overflow-x-auto">
            <div class="mb-3"><SectionHeading title="Recent epochs" subtitle="Validation checkpoints and early-stopping state" /></div>
            <table class="w-full">
              <thead><tr>
                <th class="table-head">Epoch</th><th class="table-head">Step</th>
                <th class="table-head">{hasCurriculumEpochs() ? "Train target CE" : "Train loss"}</th>
                <th class="table-head">{hasCurriculumEpochs() ? "Validation target CE" : "Validation loss"}</th>
                <th class="table-head">{hasCurriculumEpochs() ? "Curriculum KL" : "Conditional KL"}</th>
                <th class="table-head">{hasCurriculumEpochs() ? "Raw 0.01 KL" : "Base KL"}</th>
                <th class="table-head">Reverse KL</th><th class="table-head">Entropy gap</th>
                <th class="table-head">pMSE mean</th><th class="table-head">pMSE variance</th>
                <Show when={hasCurriculumEpochs()}><th class="table-head">Temperature</th></Show>
                <th class="table-head">{hasCurriculumEpochs() ? "Best raw KL" : "Best"}</th><th class="table-head">Stale</th>
              </tr></thead>
              <tbody><For each={epochs().slice(-12).reverse()}>{(item) => <tr>
                <td class="td-cell">{item.epoch + 1}</td><td class="td-cell">{item.globalStep}</td>
                <td class="td-cell">{formatMetric(item.train.loss)}</td>
                <td class="td-cell">{formatMetric(item.validation.loss)}</td>
                <td class="td-cell">{formatMetric(item.validation.klDivergence)}</td>
                <td class="td-cell">{formatMetric(item.validation.baseKlDivergence)}</td>
                <td class="td-cell">{formatMetric(item.validation.reverseKlDivergence)}</td>
                <td class="td-cell">{formatMetric(item.validation.entropyGap)}</td>
                <td class="td-cell">{formatMetric(item.validation.probabilityMse)}</td>
                <td class="td-cell">{formatMetric(item.validation.probabilityMseVariance)}</td>
                <Show when={hasCurriculumEpochs()}><td class="td-cell">{formatMetric(item.trainingTargetTemperature)}</td></Show>
                <td class="td-cell text-gain">{formatMetric(item.bestValidation)}</td>
                <td class="td-cell">{integer(item.staleEpochs)}</td>
              </tr>}</For></tbody>
            </table>
          </section>
        </Show>

        <Show when={snapshot()?.status?.error || snapshot()?.status?.message}>
          <section class="panel text-sm text-ink-300">
            <Show when={snapshot()?.status?.error}>{(value) => <div class="text-loss">{String(value())}</div>}</Show>
            <Show when={snapshot()?.status?.message}>{(value) => <div>{String(value())}</div>}</Show>
          </section>
        </Show>
      </div>
    </main>
  );
}

function MetricChart(props: {
  title: string;
  subtitle?: string;
  unit?: string;
  scale?: "linear" | "log";
  xLabel?: string;
  yDomain?: [number, number];
  series: PlotSeries[];
}) {
  const width = 900;
  const height = 260;
  const margin = { left: 72, right: 18, top: 16, bottom: 38 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;
  const scale = () => props.scale ?? "linear";
  const [viewport, setViewport] = createSignal<ChartViewport>();
  const [plotPixelWidth, setPlotPixelWidth] = createSignal(innerWidth);
  const [dragging, setDragging] = createSignal(false);
  let svg!: SVGSVGElement;
  let resizeObserver: ResizeObserver | undefined;
  let drag: { pointerId: number; clientX: number } | undefined;
  const usable = createMemo(() => props.series.map((series) => ({
    ...series,
    values: series.values.filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y)
      && (scale() !== "log" || point.y > 0)),
  })).filter((series) => series.values.length > 0));
  const fullXDomain = createMemo(() => {
    let xMin = Number.POSITIVE_INFINITY;
    let xMax = Number.NEGATIVE_INFINITY;
    for (const series of usable()) {
      for (const point of series.values) {
        xMin = Math.min(xMin, point.x);
        xMax = Math.max(xMax, point.x);
      }
    }
    return Number.isFinite(xMin) && Number.isFinite(xMax)
      ? { xMin, xMax }
      : undefined;
  });
  const visibleXDomain = createMemo(() => {
    const full = fullXDomain();
    if (!full) return undefined;
    const requested = viewport();
    if (!requested || full.xMax <= full.xMin) return full;
    const fullSpan = full.xMax - full.xMin;
    const minimumSpan = Math.max(fullSpan / 1_000_000, Number.EPSILON);
    const span = Math.min(
      fullSpan,
      Math.max(minimumSpan, requested.xMax - requested.xMin),
    );
    if (requested.followLatest) {
      return {
        xMin: Math.max(full.xMin, full.xMax - span),
        xMax: full.xMax,
      };
    }
    const xMin = Math.max(
      full.xMin,
      Math.min(requested.xMin, full.xMax - span),
    );
    return { xMin, xMax: xMin + span };
  });
  const rendered = createMemo(() => {
    const domain = visibleXDomain();
    if (!domain) return [];
    return usable().map((series) => ({
      ...series,
      values: convolveLinearChartOnGeometricGrid(
        series.values,
        domain.xMin,
        domain.xMax,
        plotPixelWidth(),
        scale(),
      ),
    })).filter((series) => series.values.length > 0);
  });
  const bounds = createMemo(() => {
    const domain = visibleXDomain();
    const points = rendered().flatMap((series) => series.values);
    if (!domain || points.length === 0) return undefined;
    const transform = (value: number) => scale() === "log" ? Math.log10(value) : value;
    let observedMin = Number.POSITIVE_INFINITY;
    let observedMax = Number.NEGATIVE_INFINITY;
    for (const point of points) {
      const transformed = transform(point.y);
      observedMin = Math.min(observedMin, transformed);
      observedMax = Math.max(observedMax, transformed);
    }
    let yMin = props.yDomain ? transform(props.yDomain[0] || Number.MIN_VALUE) : observedMin;
    let yMax = props.yDomain ? transform(props.yDomain[1]) : observedMax;
    if (yMin === yMax) {
      const pad = Math.max(Math.abs(yMin) * 0.05, 1e-6);
      yMin -= pad;
      yMax += pad;
    } else if (!props.yDomain) {
      const pad = (yMax - yMin) * 0.08;
      yMin -= pad;
      yMax += pad;
      if (scale() === "linear" && observedMin >= 0
        && observedMin <= (observedMax - observedMin) * 0.1) yMin = 0;
    }
    return {
      xMin: domain.xMin,
      xMax: domain.xMax,
      yMin,
      yMax,
      transform,
    };
  });
  const xPosition = (x: number) => {
    const value = bounds();
    if (!value || value.xMax === value.xMin) return margin.left + innerWidth / 2;
    return margin.left + (x - value.xMin) / (value.xMax - value.xMin) * innerWidth;
  };
  const yPosition = (y: number) => {
    const value = bounds();
    if (!value) return margin.top + innerHeight / 2;
    return margin.top + (value.yMax - value.transform(y)) / (value.yMax - value.yMin) * innerHeight;
  };
  const ticks = createMemo(() => {
    const value = bounds();
    if (!value) return [];
    return Array.from({ length: 5 }, (_, index) => {
      const ratio = index / 4;
      const transformed = value.yMax - ratio * (value.yMax - value.yMin);
      return {
        y: margin.top + ratio * innerHeight,
        value: scale() === "log" ? 10 ** transformed : transformed,
      };
    });
  });
  const isZoomed = createMemo(() => {
    const full = fullXDomain();
    const visible = visibleXDomain();
    if (!full || !visible) return false;
    const tolerance = Math.max(
      Number.EPSILON,
      (full.xMax - full.xMin) * 1e-9,
    );
    return Math.abs(visible.xMin - full.xMin) > tolerance
      || Math.abs(visible.xMax - full.xMax) > tolerance;
  });
  const isFollowingLatest = createMemo(() =>
    isZoomed() && viewport()?.followLatest === true);
  const zoomViewport = (factor: number, anchor = 0.5) => {
    const full = fullXDomain();
    const current = visibleXDomain();
    if (!full || !current || full.xMax <= full.xMin) return;
    const fullSpan = full.xMax - full.xMin;
    const minimumSpan = Math.max(fullSpan / 1_000_000, Number.EPSILON);
    const currentSpan = current.xMax - current.xMin;
    const nextSpan = Math.max(
      minimumSpan,
      Math.min(fullSpan, currentSpan * factor),
    );
    if (nextSpan >= fullSpan) {
      setViewport(undefined);
      return;
    }
    const preserveRightPin = viewport()?.followLatest === true;
    const normalizedAnchor = Math.max(0, Math.min(1, anchor));
    const anchorX = current.xMin + normalizedAnchor * currentSpan;
    const requestedMin = anchorX - normalizedAnchor * nextSpan;
    const xMin = preserveRightPin
      ? full.xMax - nextSpan
      : Math.max(
        full.xMin,
        Math.min(requestedMin, full.xMax - nextSpan),
      );
    const xMax = xMin + nextSpan;
    const rightTolerance = Math.max(Number.EPSILON, fullSpan * 1e-9);
    setViewport({
      xMin,
      xMax,
      followLatest: preserveRightPin
        || Math.abs(xMax - full.xMax) <= rightTolerance,
    });
  };
  const panViewport = (fraction: number) => {
    const full = fullXDomain();
    const current = visibleXDomain();
    if (!full || !current) return;
    const span = current.xMax - current.xMin;
    const fullSpan = full.xMax - full.xMin;
    if (span >= fullSpan) return;
    const requestedMin = current.xMin + fraction * span;
    const xMin = Math.max(
      full.xMin,
      Math.min(requestedMin, full.xMax - span),
    );
    const xMax = xMin + span;
    const rightTolerance = Math.max(Number.EPSILON, fullSpan * 1e-9);
    setViewport({
      xMin,
      xMax,
      followLatest: Math.abs(xMax - full.xMax) <= rightTolerance,
    });
  };
  const resetViewport = () => setViewport(undefined);
  const plotClientGeometry = () => {
    const bounds = svg.getBoundingClientRect();
    const scaleX = bounds.width / width;
    return {
      left: bounds.left + margin.left * scaleX,
      width: Math.max(1, innerWidth * scaleX),
    };
  };
  const handleWheel = (event: WheelEvent) => {
    event.preventDefault();
    const geometry = plotClientGeometry();
    if (event.shiftKey || Math.abs(event.deltaX) > Math.abs(event.deltaY)) {
      panViewport((event.deltaX || event.deltaY) / geometry.width);
      return;
    }
    const anchor = Math.max(
      0,
      Math.min(1, (event.clientX - geometry.left) / geometry.width),
    );
    zoomViewport(Math.exp(Math.sign(event.deltaY) * 0.18), anchor);
  };
  const handlePointerDown = (event: PointerEvent) => {
    if (event.button !== 0) return;
    event.preventDefault();
    drag = { pointerId: event.pointerId, clientX: event.clientX };
    setDragging(true);
    svg.setPointerCapture(event.pointerId);
  };
  const handlePointerMove = (event: PointerEvent) => {
    if (!drag || event.pointerId !== drag.pointerId) return;
    event.preventDefault();
    const distance = drag.clientX - event.clientX;
    drag.clientX = event.clientX;
    panViewport(distance / plotClientGeometry().width);
  };
  const handlePointerUp = (event: PointerEvent) => {
    if (!drag || event.pointerId !== drag.pointerId) return;
    if (svg.hasPointerCapture(event.pointerId)) {
      svg.releasePointerCapture(event.pointerId);
    }
    drag = undefined;
    setDragging(false);
  };
  const handleKeyDown = (event: KeyboardEvent) => {
    if (event.key === "+" || event.key === "=") {
      event.preventDefault();
      zoomViewport(0.5);
    } else if (event.key === "-") {
      event.preventDefault();
      zoomViewport(2);
    } else if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
      event.preventDefault();
      panViewport(event.key === "ArrowLeft" ? -0.2 : 0.2);
    } else if (event.key === "Home" || event.key === "0") {
      event.preventDefault();
      resetViewport();
    }
  };
  const attachSvg = (element: SVGSVGElement) => {
    svg = element;
    resizeObserver?.disconnect();
    const measure = () => {
      const renderedWidth = svg.getBoundingClientRect().width;
      setPlotPixelWidth(Math.max(
        1,
        Math.round(renderedWidth * innerWidth / width),
      ));
    };
    resizeObserver = new ResizeObserver(measure);
    resizeObserver.observe(svg);
    measure();
  };
  onCleanup(() => resizeObserver?.disconnect());

  return (
    <article class="panel min-w-0 overflow-hidden">
      <div class="flex flex-wrap items-start justify-between gap-2">
        <div><h3 class="font-semibold">{props.title}</h3><Show when={props.subtitle}><p class="text-xs text-ink-300">{props.subtitle}</p></Show></div>
        <div class="flex flex-col items-end gap-1.5">
          <div class="flex flex-wrap justify-end gap-x-3 gap-y-1 text-xs">
            <For each={rendered()}>{(series) => <span class="inline-flex items-center gap-1 text-ink-300">
              <span class="h-2 w-2 rounded-full" style={{ background: series.color }} />
              {series.label} <span class="tabular-nums text-ink-100">{formatMetric(series.values.at(-1)?.y)}{props.unit ? ` ${props.unit}` : ""}</span>
            </span>}</For>
          </div>
          <div class="flex items-center gap-1 text-[11px] text-ink-400">
            <span class={`mr-1 ${isFollowingLatest() ? "text-gain" : ""}`}>
              {isFollowingLatest()
                ? "Following newest · pan left to detach"
                : "Wheel to zoom · drag to pan"}
            </span>
            <button class="btn min-w-7 px-1.5 py-0.5" type="button" title="Zoom in" aria-label={`Zoom ${props.title} in`} onClick={() => zoomViewport(0.5)}>+</button>
            <button class="btn min-w-7 px-1.5 py-0.5" type="button" title="Zoom out" aria-label={`Zoom ${props.title} out`} onClick={() => zoomViewport(2)}>−</button>
            <button class="btn px-2 py-0.5" type="button" title="Reset view" aria-label={`Reset ${props.title} view`} disabled={!isZoomed()} onClick={resetViewport}>Reset</button>
          </div>
        </div>
      </div>
      <Show when={bounds()} fallback={<div class="flex h-64 items-center justify-center text-sm text-ink-300">Waiting for data</div>}>
        <svg
          ref={attachSvg}
          class={`mt-2 block h-auto w-full select-none outline-none focus-visible:ring-2 focus-visible:ring-accent/45 ${dragging() ? "cursor-grabbing" : "cursor-grab"}`}
          viewBox={`0 0 ${width} ${height}`}
          role="img"
          aria-label={`${props.title}. Interactive chart`}
          tabIndex={0}
          style={{ "touch-action": "none" }}
          onDblClick={resetViewport}
          onKeyDown={handleKeyDown}
          onPointerCancel={handlePointerUp}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onWheel={handleWheel}
        >
          <title>Wheel to zoom. Drag to pan. Double-click to reset.</title>
          <For each={ticks()}>{(tick) => <g>
            <line class="training-chart-grid" x1={margin.left} x2={width - margin.right} y1={tick.y} y2={tick.y} />
            <text class="training-chart-axis" x={margin.left - 9} y={tick.y + 4} text-anchor="end">{shortNumber(tick.value)}</text>
          </g>}</For>
          <line class="training-chart-axis-line" x1={margin.left} x2={margin.left} y1={margin.top} y2={height - margin.bottom} />
          <line class="training-chart-axis-line" x1={margin.left} x2={width - margin.right} y1={height - margin.bottom} y2={height - margin.bottom} />
          <For each={rendered()}>{(series) => {
            const points = () => series.values
              .map((point) => `${xPosition(point.x)},${yPosition(point.y)}`).join(" ");
            return <polyline points={points()} fill="none" stroke={series.color} stroke-width="2" stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke" />;
          }}</For>
          <Show when={bounds()}>{(value) => <>
            <text class="training-chart-axis" x={margin.left} y={height - 13} text-anchor="start">{shortNumber(value().xMin)}</text>
            <text class="training-chart-axis" x={width - margin.right} y={height - 13} text-anchor="end">{shortNumber(value().xMax)}</text>
            <text class="training-chart-axis" x={width / 2} y={height - 13} text-anchor="middle">{props.xLabel ?? "dataset day"}</text>
          </>}</Show>
        </svg>
      </Show>
    </article>
  );
}

function MetricCard(props: { label: string; value: string }) {
  return <div class="panel-tight min-w-0"><div class="muted-label truncate">{props.label}</div><div class="metric-value mt-1 truncate">{props.value}</div></div>;
}

function SectionHeading(props: { title: string; subtitle: string }) {
  return <div><h2 class="text-lg font-semibold">{props.title}</h2><p class="text-sm text-ink-300">{props.subtitle}</p></div>;
}

function StatusPill(props: { running: boolean; label: string }) {
  return <span class={`inline-flex items-center gap-2 rounded-full border px-2.5 py-1 text-xs font-semibold ${props.running ? "border-gain/50 bg-gain/10 text-gain" : "border-line bg-ink-800 text-ink-300"}`}>
    <Activity size={13} class={props.running ? "animate-pulse" : ""} /> {props.label}
  </span>;
}

function WaitingForTraining(props: { stage: string }) {
  return <div class="panel flex min-h-40 items-center justify-center text-center text-sm text-ink-300">
    <div><Activity class="mx-auto mb-2 text-accent" size={22} /><p>Current stage: {stageLabel(props.stage)}</p><p>Weight-training plots will appear here automatically.</p></div>
  </div>;
}

function plot(label: string, color: string, points: DatasetPoint[], key: keyof DatasetPoint): PlotSeries {
  return { label, color, values: points.flatMap((point) => typeof point[key] === "number" ? [{ x: point.x, y: point[key] as number }] : []) };
}

function metricPlot(label: string, color: string, points: TrainStepPoint[], key: keyof MetricValues): PlotSeries {
  return { label, color, values: points.flatMap((point) => typeof point.latest[key] === "number" ? [{ x: point.globalStep, y: point.latest[key]! }] : []) };
}

function directStepPlot(label: string, color: string, points: TrainStepPoint[], key: keyof TrainStepPoint): PlotSeries {
  return { label, color, values: points.flatMap((point) => typeof point[key] === "number" ? [{ x: point.globalStep, y: point[key] as number }] : []) };
}

function epochPlot(label: string, color: string, points: EpochPoint[], group: "train" | "validation", key: keyof MetricValues): PlotSeries {
  return { label, color, values: points.flatMap((point) => typeof point[group][key] === "number" ? [{ x: point.epoch + 1, y: point[group][key]! }] : []) };
}

function directEpochPlot(label: string, color: string, points: EpochPoint[], key: keyof EpochPoint): PlotSeries {
  return { label, color, values: points.flatMap((point) => typeof point[key] === "number" ? [{ x: point.epoch + 1, y: point[key] as number }] : []) };
}

function constantEpochPlot(label: string, color: string, points: EpochPoint[], value: number): PlotSeries {
  return { label, color, values: points.map((point) => ({ x: point.epoch + 1, y: value })) };
}

function metricValues(value: unknown): MetricValues {
  if (!value || typeof value !== "object") return {};
  const record = value as Record<string, unknown>;
  return {
    loss: numberValue(record.loss)
      ?? numberValue(record.normalizedMse)
      ?? numberValue(record.trainingCrossEntropy),
    normalizedMse: numberValue(record.normalizedMse),
    mse: numberValue(record.mse),
    rmse: numberValue(record.rmse),
    mae: numberValue(record.mae),
    zeroBaselineMse: numberValue(record.zeroBaselineMse),
    mseSkillVsZero: numberValue(record.mseSkillVsZero),
    directionAccuracy: numberValue(record.directionAccuracy),
    correlation: numberValue(record.correlation),
    predictionStd: numberValue(record.predictionStd),
    targetStd: numberValue(record.targetStd),
    crossEntropy: numberValue(record.crossEntropy)
      ?? numberValue(record.trainingCrossEntropy)
      ?? numberValue(record.rawCrossEntropy),
    klDivergence: numberValue(record.klDivergence)
      ?? numberValue(record.curriculumTargetKl),
    klDivergenceVariance: numberValue(record.klDivergenceVariance),
    klDivergenceStdDev: numberValue(record.klDivergenceStdDev),
    baseKlDivergence: numberValue(record.baseKlDivergence)
      ?? numberValue(record.rawBaseActionKl),
    reverseKlDivergence: numberValue(record.reverseKlDivergence),
    probabilityMse: numberValue(record.probabilityMse)
      ?? numberValue(record.rawProbabilityMse),
    probabilityMseVariance: numberValue(record.probabilityMseVariance),
    probabilityMseStdDev: numberValue(record.probabilityMseStdDev),
    excessEntropy: numberValue(record.excessEntropy),
    oracleMutualInformation: numberValue(record.oracleMutualInformation),
    targetEntropy: numberValue(record.targetEntropy)
      ?? numberValue(record.curriculumTargetEntropy)
      ?? numberValue(record.rawTargetEntropy),
    predictedEntropy: numberValue(record.predictedEntropy),
    entropyGap: numberValue(record.entropyGap),
    entropySharpness: numberValue(record.entropySharpness),
    reverseKlGate: numberValue(record.reverseKlGate),
    entropySharpnessGate: numberValue(record.entropySharpnessGate),
    softLayerNorm: numberValue(record.softLayerNorm),
    softLayerNormMeanPenalty: numberValue(record.softLayerNormMeanPenalty),
    softLayerNormVariancePenalty: numberValue(record.softLayerNormVariancePenalty),
    distributionLayer: numberValue(record.distributionLayer),
    distributionLayerSumPenalty: numberValue(record.distributionLayerSumPenalty),
    distributionLayerNegativePenalty: numberValue(record.distributionLayerNegativePenalty),
    softWeightBound: numberValue(record.softWeightBound),
    centeringConstraint: numberValue(record.centeringConstraint),
    centeringIdempotence: numberValue(record.centeringIdempotence),
    centeringSymmetry: numberValue(record.centeringSymmetry),
    distanceImbalanceWeight: numberValue(record.distanceImbalanceWeight),
    timeWeightEffectiveSampleRatio: numberValue(record.timeWeightEffectiveSampleRatio),
    curriculumTargetKl: numberValue(record.curriculumTargetKl),
    curriculumTargetProbabilityMse: numberValue(
      record.curriculumTargetProbabilityMse,
    ),
    curriculumTargetEntropy: numberValue(record.curriculumTargetEntropy),
    trainingCrossEntropy: numberValue(record.trainingCrossEntropy),
    rawCrossEntropy: numberValue(record.rawCrossEntropy),
    rawBaseActionKl: numberValue(record.rawBaseActionKl),
    rawProbabilityMse: numberValue(record.rawProbabilityMse),
    rawTargetEntropy: numberValue(record.rawTargetEntropy),
    regularizationLoss: numberValue(record.regularizationLoss),
  };
}

function numberValue(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function textValue(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

function percentValue(value: unknown): number | undefined {
  const number = numberValue(value);
  return number === undefined ? undefined : number * 100;
}

function convolveLinearChartOnGeometricGrid(
  points: PlotPoint[],
  xMin: number,
  xMax: number,
  pixelWidth: number,
  scale: "linear" | "log",
): PlotPoint[] {
  if (points.length === 0) return [];
  const constantY = points[0]!.y;
  if (points.every((point) => point.y === constantY)) {
    const left = Math.max(xMin, points[0]!.x);
    const right = Math.min(xMax, points.at(-1)!.x);
    if (right < left) return [];
    return right === left
      ? [{ x: left, y: constantY }]
      : [{ x: left, y: constantY }, { x: right, y: constantY }];
  }
  if (points.length === 1) {
    const point = points[0]!;
    return point.x >= xMin && point.x <= xMax ? [point] : [];
  }
  if (xMax <= xMin) return [];
  const targetCellPixelWidth = 2;
  const chartUnitsPerPixel = (
    (xMax - xMin) / Math.max(1, pixelWidth)
  );
  const targetCellWidth = chartUnitsPerPixel * targetCellPixelWidth;
  const cellScaleRatio = 1.2;
  const cellWidth = cellScaleRatio ** Math.ceil(
    Math.log(targetCellWidth) / Math.log(cellScaleRatio),
  );
  const chartGridOrigin = 0;
  const firstGridCell = Math.ceil(
    (xMin - chartGridOrigin) / cellWidth - 0.5,
  );
  const lastGridCell = Math.floor(
    (xMax - chartGridOrigin) / cellWidth - 0.5,
  );
  const firstCellStart = chartGridOrigin + firstGridCell * cellWidth;
  const lastCellEnd = chartGridOrigin + (lastGridCell + 1) * cellWidth;
  const lowerIndex = lowerBoundPlotPoint(points, firstCellStart);
  const upperIndex = upperBoundPlotPoint(points, lastCellEnd);
  const start = Math.max(0, lowerIndex - 1);
  const end = Math.min(points.length, upperIndex + 1);
  const transformed: Array<{
    x: number;
    y: number;
    count: number;
  }> = [];
  for (let index = start; index < end; index += 1) {
    const point = points[index]!;
    const displayY = scale === "log" ? Math.log10(point.y) : point.y;
    const previous = transformed.at(-1);
    if (previous?.x === point.x) {
      previous.y = (
        previous.y * previous.count + displayY
      ) / (previous.count + 1);
      previous.count += 1;
    } else {
      transformed.push({ x: point.x, y: displayY, count: 1 });
    }
  }
  if (transformed.length === 0) return [];
  if (transformed.length === 1) {
    const point = transformed[0]!;
    return point.x >= xMin && point.x <= xMax
      ? [{ x: point.x, y: restoreChartValue(point.y, scale) }]
      : [];
  }

  const prefixArea = new Float64Array(transformed.length);
  for (let index = 1; index < transformed.length; index += 1) {
    const previous = transformed[index - 1]!;
    const point = transformed[index]!;
    prefixArea[index] = prefixArea[index - 1]!
      + (previous.y + point.y) / 2 * (point.x - previous.x);
  }
  const supportMin = transformed[0]!.x;
  const supportMax = transformed.at(-1)!.x;
  const result: PlotPoint[] = [];
  for (
    let gridCell = firstGridCell;
    gridCell <= lastGridCell;
    gridCell += 1
  ) {
    // Every geometric scale shares one chart-space origin. Cell membership
    // stays fixed within a level, while 1.2x steps keep level transitions
    // much finer than the previous binary ladder.
    const cellStart = chartGridOrigin + gridCell * cellWidth;
    const cellEnd = cellStart + cellWidth;
    const x = (cellStart + cellEnd) / 2;
    if (x < supportMin || x > supportMax) continue;
    const left = Math.max(supportMin, cellStart);
    const right = Math.min(supportMax, cellEnd);
    if (right <= left) continue;
    const displayY = (
      integrateLinearChartTo(transformed, prefixArea, right)
      - integrateLinearChartTo(transformed, prefixArea, left)
    ) / (right - left);
    result.push({ x, y: restoreChartValue(displayY, scale) });
  }
  return result;
}

function integrateLinearChartTo(
  points: Array<{ x: number; y: number }>,
  prefixArea: Float64Array,
  x: number,
): number {
  if (x <= points[0]!.x) return 0;
  const lastIndex = points.length - 1;
  if (x >= points[lastIndex]!.x) return prefixArea[lastIndex]!;
  let low = 0;
  let high = lastIndex;
  while (low + 1 < high) {
    const middle = (low + high) >>> 1;
    if (points[middle]!.x <= x) low = middle;
    else high = middle;
  }
  const left = points[low]!;
  const right = points[low + 1]!;
  const fraction = (x - left.x) / (right.x - left.x);
  const y = left.y + fraction * (right.y - left.y);
  return prefixArea[low]! + (left.y + y) / 2 * (x - left.x);
}

function lowerBoundPlotPoint(points: PlotPoint[], x: number): number {
  let low = 0;
  let high = points.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (points[middle]!.x < x) low = middle + 1;
    else high = middle;
  }
  return low;
}

function upperBoundPlotPoint(points: PlotPoint[], x: number): number {
  let low = 0;
  let high = points.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (points[middle]!.x <= x) low = middle + 1;
    else high = middle;
  }
  return low;
}

function restoreChartValue(
  displayValue: number,
  scale: "linear" | "log",
): number {
  return scale === "log" ? 10 ** displayValue : displayValue;
}

function formatMetric(value: number | undefined): string {
  if (value === undefined || !Number.isFinite(value)) return "—";
  if (value === 0) return "0";
  const absolute = Math.abs(value);
  return absolute >= 1_000 || absolute < 0.0001 ? value.toExponential(3) : value.toPrecision(5);
}

function lossWeightLabel(value: number | undefined): string | undefined {
  return value === undefined ? undefined : `Loss weight ${value}`;
}

function reverseKlLabel(
  lossWeight: number | undefined,
  predictionMixtureWeight: number | undefined,
  gate: OutputRegularizerPlan | undefined,
): string | undefined {
  const parts = [
    ...gatedLossWeightParts(lossWeight, gate, "reverseKl"),
    predictionMixtureWeight === undefined
      ? undefined
      : `Prediction-mixture epsilon ${predictionMixtureWeight}`,
  ].filter((part): part is string => part !== undefined);
  return parts.length > 0 ? parts.join(" · ") : undefined;
}

function gatedLossWeightLabel(
  lossWeight: number | undefined,
  gate: OutputRegularizerPlan | undefined,
  loss: keyof OutputRegularizerPlan["applicationProbabilities"],
): string | undefined {
  const parts = gatedLossWeightParts(lossWeight, gate, loss);
  return parts.length > 0 ? parts.join(" · ") : undefined;
}

function gatedLossWeightParts(
  lossWeight: number | undefined,
  gate: OutputRegularizerPlan | undefined,
  loss: keyof OutputRegularizerPlan["applicationProbabilities"],
): string[] {
  if (lossWeight === undefined) return [];
  if (!gate) return [`Loss weight ${lossWeight}`];
  const probability = gate.applicationProbabilities[loss];
  const activeWeight = gate.inverseProbabilityScaling
    ? lossWeight / probability
    : lossWeight;
  const expectedWeight = gate.inverseProbabilityScaling
    ? lossWeight
    : lossWeight * probability;
  return [
    `Expected weight ${expectedWeight}`,
    `Active weight ${activeWeight}`,
    `${probability * 100}% of optimizer updates`,
  ];
}

function softWeightBoundLabel(
  lossWeight: number | undefined,
  desiredMagnitude: number | undefined,
): string | undefined {
  const parts = [
    lossWeight === undefined ? undefined : `Loss weight ${lossWeight}`,
    desiredMagnitude === undefined
      ? undefined
      : `Desired |weight| <= ${desiredMagnitude}`,
  ].filter((part): part is string => part !== undefined);
  return parts.length > 0 ? parts.join(" · ") : undefined;
}

function distributionLayerLabel(
  sumWeight: number | undefined,
  negativeWeight: number | undefined,
): string | undefined {
  if (sumWeight === undefined && negativeWeight === undefined) return undefined;
  return [
    sumWeight === undefined ? undefined : `λsum ${sumWeight}`,
    negativeWeight === undefined ? undefined : `λnegative ${negativeWeight}`,
    "post-GLU · pre-dropout",
  ].filter((part): part is string => part !== undefined).join(" · ");
}

function centeringConstraintLabel(
  idempotenceWeight: number | undefined,
  symmetryWeight: number | undefined,
): string | undefined {
  if (idempotenceWeight === undefined && symmetryWeight === undefined) {
    return undefined;
  }
  return [
    idempotenceWeight === undefined
      ? undefined
      : `Idempotence weight ${idempotenceWeight}`,
    symmetryWeight === undefined
      ? undefined
      : `Symmetry weight ${symmetryWeight}`,
    "mean squared matrix residuals",
  ].filter((part): part is string => part !== undefined).join(" · ");
}

function shortNumber(value: number): string {
  const absolute = Math.abs(value);
  if (absolute >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}m`;
  if (absolute >= 1_000) return `${(value / 1_000).toFixed(1)}k`;
  if (absolute > 0 && absolute < 0.001) return value.toExponential(1);
  return Number(value.toPrecision(4)).toString();
}

function formatUnit(value: number | undefined, unit: string, digits = 2): string {
  return value === undefined ? "—" : `${value.toFixed(digits)}${unit}`;
}

function formatPercent(value: number | undefined): string {
  return value === undefined ? "—" : `${(value * 100).toFixed(3)}%`;
}

function formatCount(value: number | undefined): string {
  return value === undefined ? "—" : Math.round(value).toLocaleString();
}

function initialComparisonRunKeys(
  runs: MetricsResponse["runs"],
  selectedRunKey: string,
): string[] {
  const selected = runs.find((run) => run.key === selectedRunKey) ?? runs[0];
  if (!selected) return [];
  const dropoutMarker = "-dropout-";
  const baselineId = selected.id.includes(dropoutMarker)
    ? selected.id.slice(0, selected.id.indexOf(dropoutMarker))
    : selected.id;
  const baseline = runs.find((run) => run.id === baselineId) ?? selected;
  const paired = runs.find((run) => run.id.startsWith(`${baselineId}${dropoutMarker}`));
  const candidate = selected.key !== baseline.key
    ? selected
    : paired ?? runs.find((run) => run.key !== baseline.key);
  return candidate ? [baseline.key, candidate.key] : [baseline.key];
}

function integer(value: number | undefined): string {
  return value === undefined ? "—" : Math.round(value).toLocaleString();
}

function ratio(value: number | undefined, total: number | undefined): string {
  return value === undefined || total === undefined ? "—" : `${value.toLocaleString()} / ${total.toLocaleString()}`;
}

function formatDurationMs(value: number | undefined): string {
  if (value === undefined) return "—";
  if (value % 60_000 === 0) return `${value / 60_000}m`;
  if (value % 1_000 === 0) return `${value / 1_000}s`;
  return `${value}ms`;
}

function stepDuration(steps: number, samplingIntervalMs: number | undefined): string {
  const stepLabel = `${formatMetric(steps)} step${steps === 1 ? "" : "s"}`;
  if (samplingIntervalMs === undefined) return stepLabel;
  const seconds = steps * samplingIntervalMs / 1_000;
  return `${stepLabel} · ${formatMetric(seconds)}s`;
}

function stageLabel(stage: string): string {
  return ({
    "cuda-build": "Building CUDA kernels", dataset: "Preparing dataset",
    "dataset-refinement": "Refining teacher fits", "dataset-features": "Refreshing features",
    training: "Training network weights", verification: "Verifying model artifact",
    complete: "Complete", archived: "Archived", paused: "Paused", failed: "Failed", starting: "Starting",
  } as Record<string, string>)[stage] ?? stage;
}

function runSummaryLabel(run: MetricsResponse["runs"][number]): string {
  const prefix = run.running ? "LIVE · " : run.archived ? "ARCHIVE · " : "";
  const best = run.bestValidationKl === undefined
    ? ""
    : ` · best KL ${run.bestValidationKl.toFixed(5)} @ epoch ${(run.bestValidationKlEpoch ?? 0) + 1}`;
  return `${prefix}${run.label}${best} · ${stageLabel(run.stage ?? "idle")}`;
}

function progressLabel(
  stage: string,
  dataset: DatasetPoint | undefined,
  step: TrainStepPoint | undefined,
  epoch: EpochPoint | undefined,
  totalEpochs: number | undefined,
): string {
  if (stage.startsWith("dataset") && dataset) return `Day ${dataset.x} / ${dataset.days ?? "—"} · ${dataset.date} ${dataset.split}`;
  if (stage === "training" && step) return `Epoch ${step.epoch + 1} / ${totalEpochs ?? "—"} · global step ${step.globalStep.toLocaleString()}`;
  if (stage === "training" && epoch) return `Epoch ${epoch.epoch + 1} / ${totalEpochs ?? "—"} · global step ${epoch.globalStep.toLocaleString()}`;
  return stageLabel(stage);
}
