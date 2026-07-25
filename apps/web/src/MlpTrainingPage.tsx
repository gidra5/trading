import { For, Show, createMemo, createSignal, onCleanup, onMount } from "solid-js";
import { Activity, ArrowLeft, BarChart3, Search } from "lucide-solid";

const apiBase = "/backend";
const POLL_MS = 2_000;

interface MetricValues {
  loss?: number;
  klDivergence?: number;
  klDivergenceVariance?: number;
  klDivergenceStdDev?: number;
  baseKlDivergence?: number;
  probabilityMse?: number;
  probabilityMseVariance?: number;
  probabilityMseStdDev?: number;
  excessEntropy?: number;
  temporalMutualInformation?: number;
  targetTemporalMutualInformation?: number;
  temporalMutualInformationReward?: number;
  oracleMutualInformation?: number;
  distanceImbalanceWeight?: number;
  timeWeightEffectiveSampleRatio?: number;
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
  }>;
  selectedRunKey: string;
  plan: {
    id: string;
    label: string;
    epochs?: number;
    patience?: number;
    samplingIntervalMs?: number;
    predictionDelayMs?: number;
    lossWeights?: Record<string, number>;
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

interface DatasetPoint {
  key: string;
  x: number;
  date: string;
  split: string;
  days?: number;
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
  gpuMemoryMiB?: number;
  latest: MetricValues;
}

interface EpochPoint {
  epoch: number;
  globalStep: number;
  train: MetricValues;
  validation: MetricValues;
  bestValidation?: number;
  staleEpochs?: number;
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

export function MlpTrainingPage() {
  const [snapshot, setSnapshot] = createSignal<MetricsResponse>();
  const [runs, setRuns] = createSignal<MetricsResponse["runs"]>([]);
  const [selectedRunKey, setSelectedRunKey] = createSignal<string>();
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
  let disposed = false;
  let requestGeneration = 0;

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
      if (event.event === "dataset-oracle") {
        const date = textValue(event.date);
        if (!date) continue;
        oracleByDate.set(date, {
          x: numberValue(event.day),
          date,
          days: numberValue(event.days),
          oracleKernelMs: numberValue(event.kernelMs),
          oracleWallMs: numberValue(event.wallMs),
        });
        for (const [key, point] of datasetByKey) {
          if (point.date === date) datasetByKey.set(key, { ...point, ...oracleByDate.get(date) });
        }
      } else if (event.event === "dataset-progress") {
        const date = textValue(event.date);
        const split = textValue(event.split) ?? textValue(event.component) ?? "unknown";
        const x = numberValue(event.day);
        if (!date || x === undefined) continue;
        const key = `${date}:${split}`;
        const point: DatasetPoint = {
          ...datasetByKey.get(key),
          ...oracleByDate.get(date),
          key,
          x,
          date,
          split,
          days: numberValue(event.days),
          examplesPerSecond: numberValue(event.examplesPerSecond),
          gpuMemoryMiB: numberValue(event.gpuMemoryMiB),
          kl: numberValue(event.meanKlDivergence),
          probabilityMse: numberValue(event.meanSquaredError),
          temporalBefore: numberValue(event.temporalMeanNormalizedStepBefore),
          temporalAfter: numberValue(event.temporalMeanNormalizedStepAfter),
          temporalSelectedPct: percentValue(event.temporalWarmSelectedFraction),
        };
        datasetByKey.set(key, point);
        setCurrentDataset(point);
      } else if (event.event === "dataset-stage-timing") {
        const date = textValue(event.date);
        const split = textValue(event.split) ?? textValue(event.component) ?? "unknown";
        if (!date) continue;
        const key = `${date}:${split}`;
        const existing = datasetByKey.get(key);
        if (!existing) continue;
        const examples = numberValue(event.examples);
        const accepted = numberValue(event.acceptedExamples);
        const point: DatasetPoint = {
          ...existing,
          acceptedPct: examples && accepted !== undefined ? 100 * accepted / examples : undefined,
          teacherWallMs: numberValue(event.teacherWallMs),
          featureEncodingMs: numberValue(event.featureEncodingMs),
          persistMs: numberValue(event.persistMs),
        };
        datasetByKey.set(key, point);
        if (currentDataset()?.key === key) setCurrentDataset(point);
      } else if (event.event === "train-step") {
        const globalStep = numberValue(event.globalStep);
        if (globalStep === undefined) continue;
        stepsById.set(globalStep, {
          globalStep,
          epoch: numberValue(event.epoch) ?? 0,
          learningRate: numberValue(event.learningRate),
          gradientNorm: numberValue(event.gradientNorm),
          examplesPerSecond: numberValue(event.examplesPerSecond),
          gpuMemoryMiB: numberValue(event.gpuMemoryMiB),
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
          bestValidation: numberValue(event.bestValidation),
          staleEpochs: numberValue(event.staleEpochs),
        });
      }
    }
    publishSeries();
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
  });
  onCleanup(() => {
    disposed = true;
    if (pollTimer !== undefined) window.clearTimeout(pollTimer);
    if (clockTimer !== undefined) window.clearInterval(clockTimer);
  });

  const latestDataset = createMemo(() => currentDataset());
  const latestStep = createMemo(() => trainSteps().at(-1));
  const latestEpoch = createMemo(() => epochs().at(-1));
  const stage = createMemo(() => snapshot()?.status?.stage ?? "idle");
  const stageProgress = createMemo(() => {
    if (stage().startsWith("dataset")) {
      const current = latestDataset();
      return current?.days ? current.x / current.days : 0;
    }
    if (stage() === "training") {
      const current = latestStep();
      const total = snapshot()?.plan.epochs;
      return total ? (current?.epoch ?? 0) / total : 0;
    }
    return stage() === "complete" ? 1 : 0;
  });
  const updatedAgo = createMemo(() => {
    const timestamp = Date.parse(snapshot()?.status?.updatedAt ?? "");
    if (!Number.isFinite(timestamp)) return "never";
    return `${Math.max(0, Math.floor((now() - timestamp) / 1_000))}s ago`;
  });

  return (
    <main class="min-h-screen bg-ink-950 text-ink-100">
      <div class="mx-auto flex w-full max-w-[96rem] flex-col gap-4 px-4 py-4 lg:px-6">
        <header class="flex flex-col gap-3 border-b border-line pb-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <div class="muted-label">Machine learning</div>
            <div class="mt-1 flex flex-wrap items-center gap-3">
              <h1 class="text-2xl font-semibold">Live MLP training</h1>
              <StatusPill running={Boolean(snapshot()?.running)} label={snapshot()?.running ? "Live" : stage()} />
            </div>
            <p class="mt-1 max-w-4xl break-words text-sm text-ink-300">
              {snapshot()?.plan.label ?? "Loading the active training plan…"}
            </p>
          </div>
          <div class="flex min-w-0 flex-col gap-2 lg:items-end">
            <label class="flex min-w-0 flex-col gap-1">
              <span class="muted-label">Training run</span>
              <select
                class="min-w-64 max-w-full rounded border border-line bg-ink-900 px-3 py-2 text-sm text-ink-100 outline-none focus:border-accent"
                value={selectedRunKey() ?? ""}
                disabled={runs().length === 0}
                onChange={(event) => selectRun(event.currentTarget.value)}
              >
                <Show when={runs().length === 0}>
                  <option value="">Loading runs…</option>
                </Show>
                <For each={runs()}>
                  {(run) => (
                    <option value={run.key}>
                      {run.running ? "LIVE · " : ""}{run.label} · {stageLabel(run.stage ?? "idle")}
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
            <span>{progressLabel(stage(), latestDataset(), latestStep(), snapshot()?.plan.epochs)}</span>
            <Show when={snapshot()?.plan.patience !== undefined}>
              <span>Early-stop patience: {snapshot()?.plan.patience} epochs</span>
            </Show>
            <Show when={snapshot()?.finalizeRequested}><span class="text-warn">Finalize requested</span></Show>
          </div>
        </section>

        <section class="flex flex-col gap-3">
          <SectionHeading
            title="Network optimization"
            subtitle={stage() === "training"
              ? `${trainSteps().length} logged updates · updating live`
              : trainSteps().length > 0
                ? `${trainSteps().length} updates from the most recent training attempt; live updates resume after refinement`
                : "Plots will populate when refinement hands off to weight training"}
          />
          <Show when={trainSteps().length > 0 || epochs().length > 0} fallback={<WaitingForTraining stage={stage()} />}>
            <div class="grid min-w-0 gap-3 xl:grid-cols-2">
              <MetricChart title="Latest batch loss" scale="log" xLabel="global step" series={[
                metricPlot("Total", "#38bdf8", trainSteps(), "loss"),
              ]} />
              <MetricChart title="Epoch loss" scale="log" xLabel="epoch" series={[
                epochPlot("Train", "#38bdf8", epochs(), "train", "loss"),
                epochPlot("Validation", "#f5b84b", epochs(), "validation", "loss"),
                directEpochPlot("Best validation", "#22c55e", epochs(), "bestValidation"),
              ]} />
              <MetricChart title="Distribution losses" scale="log" xLabel="global step" series={[
                metricPlot("Conditional KL", "#38bdf8", trainSteps(), "klDivergence"),
                metricPlot("Base-action KL", "#22c55e", trainSteps(), "baseKlDivergence"),
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
              <MetricChart title="Validation probability MSE" subtitle="Per-example visible-range surface error" scale="log" xLabel="epoch" series={[
                epochPlot("Mean pMSE", "#f05252", epochs(), "validation", "probabilityMse"),
                epochPlot("pMSE standard deviation", "#f5b84b", epochs(), "validation", "probabilityMseStdDev"),
                epochPlot("pMSE variance", "#a78bfa", epochs(), "validation", "probabilityMseVariance"),
              ]} />
              <MetricChart title="Excess entropy" subtitle={lossWeightLabel(snapshot()?.plan.lossWeights?.excessEntropy)} xLabel="global step" series={[
                metricPlot("Excess entropy", "#a78bfa", trainSteps(), "excessEntropy"),
              ]} />
              <MetricChart title="Temporal mutual information" subtitle={lossWeightLabel(snapshot()?.plan.lossWeights?.temporalMutualInformation)} xLabel="global step" series={[
                metricPlot("Predicted MI", "#38bdf8", trainSteps(), "temporalMutualInformation"),
                metricPlot("Capped reward", "#a78bfa", trainSteps(), "temporalMutualInformationReward"),
                metricPlot("Teacher MI", "#f5b84b", trainSteps(), "targetTemporalMutualInformation"),
              ]} />
              <MetricChart title="Oracle mutual information" subtitle={lossWeightLabel(snapshot()?.plan.lossWeights?.oracleMutualInformation)} xLabel="global step" series={[
                metricPlot("Oracle MI", "#22c55e", trainSteps(), "oracleMutualInformation"),
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
                directStepPlot("Throughput", "#22c55e", trainSteps(), "examplesPerSecond"),
              ]} />
              <MetricChart title="Training GPU memory" unit="MiB" xLabel="global step" series={[
                directStepPlot("Allocated", "#a78bfa", trainSteps(), "gpuMemoryMiB"),
              ]} />
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
          <MetricCard label="Teacher rate" value={formatUnit(latestDataset()?.examplesPerSecond, " fits/s")} />
          <MetricCard label="Teacher GPU" value={formatUnit(latestDataset()?.gpuMemoryMiB, " MiB", 1)} />
          <MetricCard label="Fit KL" value={formatMetric(latestDataset()?.kl)} />
          <MetricCard label="Fit pMSE" value={formatMetric(latestDataset()?.probabilityMse)} />
          <MetricCard label="Refined shards" value={ratio(snapshot()?.progress.refinedShards, snapshot()?.progress.totalShards)} />
          <MetricCard label="Queued fits" value={integer(snapshot()?.progress.remainingTeacherFits)} />
          <MetricCard label="Last global step" value={integer(latestStep()?.globalStep)} />
          <MetricCard label="Last batch loss" value={formatMetric(latestStep()?.latest.loss)} />
          <MetricCard label="Last best validation" value={formatMetric(latestEpoch()?.bestValidation)} />
          <MetricCard label="Learning rate" value={formatMetric(latestStep()?.learningRate)} />
          <MetricCard label="Train rate" value={formatUnit(latestStep()?.examplesPerSecond, " ex/s", 1)} />
          <MetricCard label="Train GPU" value={formatUnit(latestStep()?.gpuMemoryMiB, " MiB", 1)} />
        </section>

        <Show when={datasetPoints().length > 0}>
          <section class="flex flex-col gap-3">
            <SectionHeading title="Teacher fitting and refinement" subtitle={`${datasetPoints().length} completed day/split batches`} />
            <div class="grid min-w-0 gap-3 xl:grid-cols-2">
              <MetricChart title="Teacher throughput" unit="fits/s" series={[
                plot("Fits / second", "#38bdf8", datasetPoints(), "examplesPerSecond"),
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
                plot("Teacher", "#38bdf8", datasetPoints(), "teacherWallMs"),
                plot("Oracle kernel", "#a78bfa", datasetPoints(), "oracleKernelMs"),
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
                <th class="table-head">Train loss</th><th class="table-head">Validation loss</th>
                <th class="table-head">Conditional KL</th><th class="table-head">Base KL</th>
                <th class="table-head">pMSE mean</th><th class="table-head">pMSE variance</th>
                <th class="table-head">Best</th><th class="table-head">Stale</th>
              </tr></thead>
              <tbody><For each={epochs().slice(-12).reverse()}>{(item) => <tr>
                <td class="td-cell">{item.epoch + 1}</td><td class="td-cell">{item.globalStep}</td>
                <td class="td-cell">{formatMetric(item.train.loss)}</td>
                <td class="td-cell">{formatMetric(item.validation.loss)}</td>
                <td class="td-cell">{formatMetric(item.validation.klDivergence)}</td>
                <td class="td-cell">{formatMetric(item.validation.baseKlDivergence)}</td>
                <td class="td-cell">{formatMetric(item.validation.probabilityMse)}</td>
                <td class="td-cell">{formatMetric(item.validation.probabilityMseVariance)}</td>
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
  const usable = createMemo(() => props.series.map((series) => ({
    ...series,
    values: series.values.filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y)
      && (scale() !== "log" || point.y > 0)),
  })).filter((series) => series.values.length > 0));
  const bounds = createMemo(() => {
    const points = usable().flatMap((series) => series.values);
    if (points.length === 0) return undefined;
    const xValues = points.map((point) => point.x);
    const transform = (value: number) => scale() === "log" ? Math.log10(value) : value;
    const yValues = points.map((point) => transform(point.y));
    const observedMin = Math.min(...yValues);
    const observedMax = Math.max(...yValues);
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
      xMin: Math.min(...xValues),
      xMax: Math.max(...xValues),
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

  return (
    <article class="panel min-w-0 overflow-hidden">
      <div class="flex flex-wrap items-start justify-between gap-2">
        <div><h3 class="font-semibold">{props.title}</h3><Show when={props.subtitle}><p class="text-xs text-ink-300">{props.subtitle}</p></Show></div>
        <div class="flex flex-wrap justify-end gap-x-3 gap-y-1 text-xs">
          <For each={usable()}>{(series) => <span class="inline-flex items-center gap-1 text-ink-300">
            <span class="h-2 w-2 rounded-full" style={{ background: series.color }} />
            {series.label} <span class="tabular-nums text-ink-100">{formatMetric(series.values.at(-1)?.y)}{props.unit ? ` ${props.unit}` : ""}</span>
          </span>}</For>
        </div>
      </div>
      <Show when={bounds()} fallback={<div class="flex h-64 items-center justify-center text-sm text-ink-300">Waiting for data</div>}>
        <svg class="mt-2 block h-auto w-full" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={props.title}>
          <For each={ticks()}>{(tick) => <g>
            <line class="training-chart-grid" x1={margin.left} x2={width - margin.right} y1={tick.y} y2={tick.y} />
            <text class="training-chart-axis" x={margin.left - 9} y={tick.y + 4} text-anchor="end">{shortNumber(tick.value)}</text>
          </g>}</For>
          <line class="training-chart-axis-line" x1={margin.left} x2={margin.left} y1={margin.top} y2={height - margin.bottom} />
          <line class="training-chart-axis-line" x1={margin.left} x2={width - margin.right} y1={height - margin.bottom} y2={height - margin.bottom} />
          <For each={usable()}>{(series) => {
            const points = () => downsampleMinMax(series.values, 900)
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

function metricValues(value: unknown): MetricValues {
  if (!value || typeof value !== "object") return {};
  const record = value as Record<string, unknown>;
  return {
    loss: numberValue(record.loss), klDivergence: numberValue(record.klDivergence),
    klDivergenceVariance: numberValue(record.klDivergenceVariance),
    klDivergenceStdDev: numberValue(record.klDivergenceStdDev),
    baseKlDivergence: numberValue(record.baseKlDivergence),
    probabilityMse: numberValue(record.probabilityMse),
    probabilityMseVariance: numberValue(record.probabilityMseVariance),
    probabilityMseStdDev: numberValue(record.probabilityMseStdDev),
    excessEntropy: numberValue(record.excessEntropy), temporalMutualInformation: numberValue(record.temporalMutualInformation),
    targetTemporalMutualInformation: numberValue(record.targetTemporalMutualInformation),
    temporalMutualInformationReward: numberValue(record.temporalMutualInformationReward),
    oracleMutualInformation: numberValue(record.oracleMutualInformation),
    distanceImbalanceWeight: numberValue(record.distanceImbalanceWeight),
    timeWeightEffectiveSampleRatio: numberValue(record.timeWeightEffectiveSampleRatio),
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

function downsampleMinMax(points: PlotPoint[], buckets: number): PlotPoint[] {
  if (points.length <= buckets * 2) return points;
  const result: PlotPoint[] = [];
  const size = points.length / buckets;
  for (let bucket = 0; bucket < buckets; bucket += 1) {
    const slice = points.slice(Math.floor(bucket * size), Math.floor((bucket + 1) * size));
    if (slice.length === 0) continue;
    let low = slice[0]!;
    let high = slice[0]!;
    for (const point of slice) {
      if (point.y < low.y) low = point;
      if (point.y > high.y) high = point;
    }
    if (low.x <= high.x) result.push(low, ...(low === high ? [] : [high]));
    else result.push(high, low);
  }
  return result;
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
    complete: "Complete", paused: "Paused", failed: "Failed", starting: "Starting",
  } as Record<string, string>)[stage] ?? stage;
}

function progressLabel(stage: string, dataset: DatasetPoint | undefined, step: TrainStepPoint | undefined, totalEpochs: number | undefined): string {
  if (stage.startsWith("dataset") && dataset) return `Day ${dataset.x} / ${dataset.days ?? "—"} · ${dataset.date} ${dataset.split}`;
  if (stage === "training" && step) return `Epoch ${step.epoch + 1} / ${totalEpochs ?? "—"} · global step ${step.globalStep.toLocaleString()}`;
  return stageLabel(stage);
}
