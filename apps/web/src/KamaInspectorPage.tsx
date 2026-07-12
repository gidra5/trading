import { For, Show, batch, createEffect, createMemo, createSignal, createUniqueId, onCleanup, onMount } from "solid-js";
import { Activity, ArrowLeft, Info, RefreshCw } from "lucide-solid";
import type {
  BacktestChartAnnotation,
  BacktestChartSmaSeries,
  BacktestOraclePath,
  BacktestOraclePoint,
  BacktestTrace,
  Candle,
  VwKamaCandleRangeResponse,
  VwKamaInspectorCatalog,
  VwKamaInspectorRequest,
  VwKamaInspectorResponse,
  VwKamaParameters,
  VwKamaPreset,
  VwKamaTransition,
} from "@trading/bot-algo";
import {
  CandleChart,
  type CandleChartOverlayVisibility,
  type CandleChartViewport,
} from "./components/CandleChart";
import { formatDateTime, formatDuration, formatQuote } from "./format";

const apiBase = "/backend";
const debounceMs = 150;
const detailDebounceMs = 120;
const detailMaxCandles = 5_000;
const detailTriggerCandles = detailMaxCandles * 4;

type OracleState = BacktestOraclePoint["state"];
interface TimeRange { start: number; end: number }

const defaults: VwKamaParameters = {
  efficiencyMs: 14 * 60_000,
  fastMs: 28 * 60_000,
  slowMs: 153 * 60_000,
  power: 0.49045,
  volumeMs: 130 * 60_000,
  volumeCap: 2.65003,
  volumePower: 0,
  deadbandBpsHour: 67.56654,
  deadbandMode: "flat",
  thresholdMode: "static",
  thresholdLookbackMs: 60 * 60_000,
  thresholdNoiseMultiplier: 0,
};

const emptyOracle: BacktestOraclePath = {
  mode: "fixed-notional",
  eventMode: "close",
  leverage: 1,
  friction: 0,
  points: [],
};

const metricHelp = {
  score: "Overall oracle-path similarity: 60% one-to-one timing-weighted F1, 30% exposure agreement, and 10% signal cleanliness. It does not measure profitability.",
  f1: "Harmonic mean of one-to-one timing-weighted precision and recall. It is high only when signals are both accurate and complete.",
  precision: "The timing-weighted share of candidate transitions paired with one oracle transition reaching the same target state. Extra and time-offset signals reduce it.",
  recall: "The timing-weighted share of oracle transitions covered by one candidate transition reaching the same target state. Missed and time-offset transitions reduce it.",
  agreement: "Average candle-by-candle exposure similarity: same state 100%, flat versus directional 50%, and opposite directions 0%.",
  timingError: "Median absolute time offset across one-to-one matched transitions. Extra candidate signals and missed oracle transitions are excluded.",
  matched: "One-to-one matches divided by all oracle transitions. The remainder are missed oracle transitions.",
  extra: "Candidate transitions that could not be paired with an oracle transition. These are unnecessary signals under this comparison.",
  noise: "Extra candidate transitions divided by one-to-one matched transitions. For example, 2× means two unnecessary transitions per matched transition. Lower is better; signal cleanliness uses its bounded inverse in the score.",
  cleanliness: "Matched transitions divided by all candidate transitions: matched / (matched + extra). Higher is better, and this contributes 10% of the overall score.",
  signals: "Candidate long, short, or flat state changes per scored day.",
  candles: "Candles included in scoring after warmup candles are excluded.",
} as const;

export function KamaInspectorPage() {
  const [catalog, setCatalog] = createSignal<VwKamaInspectorCatalog>();
  const [windowId, setWindowId] = createSignal("");
  const [intervalMs, setIntervalMs] = createSignal(60_000);
  const [parameters, setParameters] = createSignal({ ...defaults });
  const [selectedPresetId, setSelectedPresetId] = createSignal("custom");
  const [rankedPair, setRankedPair] = createSignal<"current" | "best" | "worst">("current");
  const [oracleFriction, setOracleFriction] = createSignal(0.00175);
  const [matchWindowMs, setMatchWindowMs] = createSignal(2 * 3_600_000);
  const [timingHalfLifeMs, setTimingHalfLifeMs] = createSignal(10 * 60_000);
  const [warmupMultiple, setWarmupMultiple] = createSignal(3);
  const [result, setResult] = createSignal<VwKamaInspectorResponse>();
  const [comparisonResult, setComparisonResult] = createSignal<VwKamaInspectorResponse>();
  const [resultRequest, setResultRequest] = createSignal<VwKamaInspectorRequest>();
  const [catalogError, setCatalogError] = createSignal<string>();
  const [analysisError, setAnalysisError] = createSignal<string>();
  const [loading, setLoading] = createSignal(false);
  const [showKama, setShowKama] = createSignal(true);
  const [showSignals, setShowSignals] = createSignal(true);
  const [showOracle, setShowOracle] = createSignal(true);
  const [timeViewport, setTimeViewport] = createSignal<TimeRange>();
  const [detail, setDetail] = createSignal<VwKamaCandleRangeResponse>();
  const [detailLoading, setDetailLoading] = createSignal(false);
  const [detailError, setDetailError] = createSignal<string>();
  const [hoveredOracle, setHoveredOracle] = createSignal<BacktestOraclePoint>();
  let analysisTimer: number | undefined;
  let analysisController: AbortController | undefined;
  let detailTimer: number | undefined;
  let detailController: AbortController | undefined;
  let requestSequence = 0;
  let detailSequence = 0;
  let viewportWindowId = "";

  const selectedWindow = createMemo(() =>
    catalog()?.windows.find((window) => window.id === windowId()));
  const presets = createMemo(() => (catalog()?.presets ?? []).filter((preset) =>
    preset.scope === "global"
      || preset.windowId === windowId() && (preset.intervalMs == null || preset.intervalMs === intervalMs())));
  const selectedPreset = createMemo(() =>
    catalog()?.presets.find((preset) => preset.id === selectedPresetId()));
  const globalPreset = createMemo(() =>
    catalog()?.presets.find((preset) => preset.scope === "global"));
  const rankedPresets = createMemo(() => (catalog()?.presets ?? [])
    .filter((preset): preset is VwKamaPreset & { score: number } =>
      preset.scope === "window" && Number.isFinite(preset.score))
    .sort((left, right) => right.score - left.score));
  const bestPreset = createMemo(() => rankedPresets()[0]);
  const worstPreset = createMemo(() => rankedPresets().at(-1));
  const candles = createMemo(() => result()?.candles ?? []);
  const oracle = createMemo(() => result()?.oracle ?? emptyOracle);
  const candidatePoints = createMemo(() => result()?.candidatePath.points ?? []);
  const kamaSeries = createMemo(() => showKama() && result()?.kamaSeries
    ? [mergeDetailKamaSeries(result()!.kamaSeries, detail(), result())]
    : []);
  const annotations = createMemo(() => showSignals()
    ? candidateAnnotations(result()?.candidateTransitions ?? [])
    : []);
  const matchedCandidate = createMemo(() => {
    const oraclePoint = hoveredOracle();
    if (!oraclePoint) return undefined;
    const oracleTransition = result()?.oracleTransitions.find((point) =>
      point.time === oraclePoint.time && point.state === oraclePoint.state);
    if (oracleTransition?.matchedTime === null || oracleTransition?.matchedTime === undefined) return undefined;
    return annotations().find((annotation) =>
      annotation.time === oracleTransition.matchedTime && annotation.signalState === oracleTransition.state);
  });
  const trace = createMemo<BacktestTrace>(() => ({
    positions: [],
    grids: [],
    orders: [],
    signals: [],
    extrema: [],
    oracle: showOracle() ? oracle() : emptyOracle,
    frames: [],
  }));
  const overlays = createMemo<CandleChartOverlayVisibility>(() => ({
    averages: showKama(),
    signals: showSignals(),
    orders: false,
    fills: false,
    positions: false,
    extrema: false,
    oracle: showOracle(),
  }));
  const timeRange = createMemo(() => {
    const analysis = result();
    const values = candles();
    return {
      start: analysis?.window.startTime ?? values[0]?.openTime ?? selectedWindow()?.startTime ?? 0,
      end: analysis?.window.endTime ?? (values.at(-1)?.closeTime ?? (selectedWindow()?.endTime ?? 1) - 1) + 1,
    };
  });
  const visibleTimeRange = createMemo(() => normalizeTimeRange(
    timeViewport() ?? timeRange(),
    timeRange(),
    Math.min(timeRange().end - timeRange().start, Math.max(2, result()?.intervalMs ?? intervalMs()) * 2),
  ));
  const chartCandles = createMemo(() => mergeDetailCandles(candles(), detail(), result()));
  const chartViewport = createMemo(() => candleViewport(chartCandles(), visibleTimeRange()));
  const chartResolution = createMemo(() => visibleResolution(
    candles(),
    detail(),
    visibleTimeRange(),
    result()?.intervalMs ?? intervalMs(),
    result()?.renderIntervalMs,
  ));
  const comparisons = createMemo(() => compareTransitions(
    result()?.candidateTransitions ?? [],
    result()?.oracleTransitions ?? [],
  ));
  const visibleComparisons = createMemo(() => comparisons().filter((row) => {
    const inside = (time: number | undefined) => time !== undefined
      && time >= visibleTimeRange().start
      && time < visibleTimeRange().end;
    return inside(row.candidate?.time) || inside(row.oracle?.time);
  }));
  const visibleCandidateCount = createMemo(() => (result()?.candidateTransitions ?? []).filter(
    (point) => point.time >= visibleTimeRange().start && point.time < visibleTimeRange().end,
  ).length);

  onMount(() => void loadCatalog());

  createEffect(() => {
    const id = windowId();
    const preset = selectedPreset();
    const interval = intervalMs();
    if (!id || preset?.scope !== "window"
      || preset.windowId === id && (preset.intervalMs == null || preset.intervalMs === interval)) return;
    const next = catalog()?.presets.find((item) => item.scope === "window"
      && item.windowId === id
      && (item.intervalMs == null || item.intervalMs === interval))
      ?? globalPreset();
    setSelectedPresetId(next?.id ?? "custom");
    if (next) setParameters({ ...next.parameters });
  });

  createEffect(() => {
    const id = windowId();
    const interval = intervalMs();
    const config = parameters();
    const friction = oracleFriction();
    const matchWindow = matchWindowMs();
    const timingHalfLife = timingHalfLifeMs();
    const warmup = warmupMultiple();
    const preset = selectedPreset();
    if (!id || !catalog()) return;
    if (analysisTimer !== undefined) window.clearTimeout(analysisTimer);
    analysisController?.abort();
    const sequence = ++requestSequence;
    analysisTimer = window.setTimeout(() => {
      analysisTimer = undefined;
      const request = {
        windowId: id,
        intervalMs: interval,
        parameters: config,
        oracleFriction: friction,
        matchWindowMs: matchWindow,
        timingHalfLifeMs: timingHalfLife,
        warmupMultiple: warmup,
      };
      const baseline = preset?.scope === "window" ? globalPreset() : undefined;
      void analyze(request, sequence, baseline ? { ...request, parameters: { ...baseline.parameters } } : undefined);
    }, debounceMs);
  });

  createEffect(() => {
    const analysis = result();
    if (!analysis) return;
    const bounds = timeRange();
    const reset = analysis.window.id !== viewportWindowId;
    viewportWindowId = analysis.window.id;
    setTimeViewport((current) => {
      const next = reset || !current
        ? bounds
        : normalizeTimeRange(current, bounds, Math.min(bounds.end - bounds.start, analysis.intervalMs * 2));
      return current && current.start === next.start && current.end === next.end ? current : next;
    });
    setDetail(undefined);
  });

  createEffect(() => {
    const analysis = result();
    const baseRequest = resultRequest();
    const viewport = visibleTimeRange();
    if (!analysis || !baseRequest
      || baseRequest.windowId !== analysis.window.id
      || baseRequest.intervalMs !== analysis.intervalMs
      || viewport.end <= viewport.start) return;
    const sourceCount = Math.ceil((viewport.end - viewport.start) / analysis.intervalMs);
    if (sourceCount > detailTriggerCandles) {
      cancelDetailRequest();
      setDetail(undefined);
      return;
    }
    if (detailTimer !== undefined) window.clearTimeout(detailTimer);
    detailController?.abort();
    const sequence = ++detailSequence;
    const span = viewport.end - viewport.start;
    const startTime = Math.max(timeRange().start, viewport.start - span);
    const endTime = Math.min(timeRange().end, viewport.end + span);
    detailTimer = window.setTimeout(() => {
      detailTimer = undefined;
      void loadDetail({
        ...baseRequest,
        parameters: { ...baseRequest.parameters },
        startTime,
        endTime,
        maxCandles: detailMaxCandles,
      }, sequence);
    }, detailDebounceMs);
  });

  onCleanup(() => {
    if (analysisTimer !== undefined) window.clearTimeout(analysisTimer);
    analysisController?.abort();
    cancelDetailRequest();
  });

  async function loadCatalog(): Promise<void> {
    setCatalogError(undefined);
    try {
      const response = await fetch(`${apiBase}/api/kama-inspector/windows`);
      const payload = await response.json() as unknown;
      if (!response.ok) throw new Error(errorMessage(payload, "Could not load inspector windows"));
      const next = payload as VwKamaInspectorCatalog;
      if (next.windows.length === 0 || next.scales.length === 0) {
        throw new Error("No cached KAMA inspection windows or scales are available.");
      }
      setCatalog(next);
      setWindowId(next.defaults.windowId ?? next.windows[0]!.id);
      setIntervalMs(next.defaults.intervalMs ?? next.scales[0]!.intervalMs);
      setParameters({ ...next.defaults.parameters });
      setSelectedPresetId(next.presets.find((preset) => preset.scope === "global")?.id ?? "custom");
      setRankedPair("current");
      setOracleFriction(next.defaults.oracleFriction);
      setMatchWindowMs(next.defaults.matchWindowMs);
      setTimingHalfLifeMs(next.defaults.timingHalfLifeMs);
      setWarmupMultiple(next.defaults.warmupMultiple);
    } catch (error) {
      setCatalogError(error instanceof Error ? error.message : "Could not load inspector windows");
    }
  }

  async function analyze(
    request: VwKamaInspectorRequest,
    sequence: number,
    comparisonRequest?: VwKamaInspectorRequest,
  ): Promise<void> {
    const controller = new AbortController();
    analysisController = controller;
    setLoading(true);
    setAnalysisError(undefined);
    try {
      const [payload, comparison] = await Promise.all([
        fetchAnalysis(request, controller.signal),
        comparisonRequest ? fetchAnalysis(comparisonRequest, controller.signal) : Promise.resolve(undefined),
      ]);
      if (sequence === requestSequence) {
        batch(() => {
          setResultRequest({ ...request, parameters: { ...request.parameters } });
          setResult(payload);
          setComparisonResult(comparison);
        });
      }
    } catch (error) {
      if (!controller.signal.aborted && sequence === requestSequence) {
        setAnalysisError(error instanceof Error ? error.message : "KAMA analysis failed");
      }
    } finally {
      if (sequence === requestSequence) {
        setLoading(false);
        if (analysisController === controller) analysisController = undefined;
      }
    }
  }

  async function loadDetail(request: VwKamaInspectorRequest & {
    startTime: number;
    endTime: number;
    maxCandles: number;
  }, sequence: number): Promise<void> {
    const controller = new AbortController();
    detailController = controller;
    setDetailLoading(true);
    setDetailError(undefined);
    try {
      const response = await fetch(`${apiBase}/api/kama-inspector/candles`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          ...request,
          startTime: Math.floor(request.startTime),
          endTime: Math.ceil(request.endTime),
        }),
        signal: controller.signal,
      });
      const payload = await response.json() as unknown;
      if (!response.ok) throw new Error(errorMessage(payload, "Could not load detailed candles"));
      const next = payload as VwKamaCandleRangeResponse;
      if (sequence === detailSequence
        && next.windowId === request.windowId
        && next.intervalMs === request.intervalMs) {
        setDetail(next);
      }
    } catch (error) {
      if (!controller.signal.aborted && sequence === detailSequence) {
        setDetailError(error instanceof Error ? error.message : "Could not load detailed candles");
      }
    } finally {
      if (sequence === detailSequence) {
        setDetailLoading(false);
        if (detailController === controller) detailController = undefined;
      }
    }
  }

  function cancelDetailRequest(): void {
    if (detailTimer !== undefined) window.clearTimeout(detailTimer);
    detailTimer = undefined;
    detailController?.abort();
    detailController = undefined;
    detailSequence += 1;
    setDetailLoading(false);
    setDetailError(undefined);
  }

  const update = <K extends keyof VwKamaParameters>(key: K, value: VwKamaParameters[K]) => {
    setRankedPair("current");
    setSelectedPresetId("custom");
    setParameters((current) => ({ ...current, [key]: value }));
  };

  const applyPreset = (id: string) => {
    setRankedPair("current");
    setSelectedPresetId(id);
    const preset = catalog()?.presets.find((item) => item.id === id);
    if (preset) setParameters({ ...preset.parameters });
  };

  const applyRankedPair = (value: string) => {
    const ranking = value === "best" ? "best" : value === "worst" ? "worst" : "current";
    setRankedPair(ranking);
    if (ranking === "current") return;
    const preset = ranking === "best" ? bestPreset() : worstPreset();
    if (!preset?.windowId) return;
    batch(() => {
      setWindowId(preset.windowId!);
      if (preset.intervalMs) setIntervalMs(preset.intervalMs);
      setSelectedPresetId(preset.id);
      setParameters({ ...preset.parameters });
    });
  };

  const setViewport = (viewport: TimeRange) =>
    setTimeViewport(normalizeTimeRange(
      viewport,
      timeRange(),
      Math.min(timeRange().end - timeRange().start, (result()?.intervalMs ?? intervalMs()) * 2),
    ));

  const updateViewportFromChart = (viewport: CandleChartViewport) => {
    const values = chartCandles();
    const normalized = inspectorViewport(viewport, values.length);
    const first = values[normalized.start];
    const last = values[Math.max(normalized.start, normalized.end - 1)];
    if (first && last) setViewport({ start: first.openTime, end: last.closeTime + 1 });
  };

  const zoomViewport = (scale: number, anchor = 0.5) => {
    const bounds = timeRange();
    const current = visibleTimeRange();
    const visible = current.end - current.start;
    const minimum = Math.min(bounds.end - bounds.start, (result()?.intervalMs ?? intervalMs()) * 2);
    const nextVisible = clamp(Math.round(visible * scale), minimum, bounds.end - bounds.start);
    const normalizedAnchor = clamp(anchor, 0, 1);
    const anchorTime = current.start + visible * normalizedAnchor;
    const start = Math.round(anchorTime - nextVisible * normalizedAnchor);
    setViewport({ start, end: start + nextVisible });
  };

  const panViewport = (fraction: number) => {
    const current = visibleTimeRange();
    const shift = Math.round((current.end - current.start) * fraction);
    if (shift !== 0) setViewport({ start: current.start + shift, end: current.end + shift });
  };

  const resetViewport = () => setViewport(timeRange());

  return (
    <main class="min-h-screen bg-ink-950 text-ink-100">
      <div class="mx-auto flex w-full max-w-7xl flex-col gap-4 px-4 py-4 lg:px-6">
        <header class="flex flex-col gap-3 border-b border-line pb-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <div class="muted-label">Research workspace</div>
            <h1 class="mt-1 text-2xl font-semibold">Volume-weighted KAMA inspector</h1>
            <p class="mt-1 max-w-3xl text-sm text-ink-300">
              Compare causal KAMA state changes with the close-only perfect-margin oracle. Editing a field reruns the selected cached window.
            </p>
          </div>
          <a class="btn" href="#/">
            <ArrowLeft size={16} /> Dashboard
          </a>
        </header>

        <Show when={catalogError()}>
          {(message) => <ErrorNotice message={message()} onRetry={() => void loadCatalog()} />}
        </Show>

        <section class="panel">
          <div class="mb-3 flex flex-col gap-2 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <div class="muted-label">Experiment</div>
              <h2 class="text-lg font-semibold">Window and signal parameters</h2>
            </div>
            <div class="flex items-center gap-2 text-xs text-ink-300">
              <Activity class={loading() ? "animate-pulse text-accent" : "text-gain"} size={15} />
              {loading() ? "Analyzing…" : result() ? "Up to date" : "Waiting for data"}
            </div>
          </div>

          <div class="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-4">
            <InspectorSelect
              label="History window"
              value={windowId()}
              options={(catalog()?.windows ?? []).map((window) => ({
                value: window.id,
                label: `${window.label} · ${formatDateRange(window.startTime, window.endTime)}`,
              }))}
              onInput={(value) => {
                setRankedPair("current");
                setWindowId(value);
              }}
            />
            <InspectorSelect
              label="Candle scale"
              value={String(intervalMs())}
              options={(catalog()?.scales ?? []).map((scale) => ({
                value: String(scale.intervalMs),
                label: scale.label,
              }))}
              onInput={(value) => {
                setRankedPair("current");
                setIntervalMs(Number(value));
              }}
            />
            <InspectorSelect
              label="Ranked window/config pair"
              value={rankedPair()}
              options={[
                { value: "current", label: "Current selection" },
                ...(bestPreset() ? [{ value: "best", label: `Best · ${rankedPresetLabel(bestPreset()!, catalog())}` }] : []),
                ...(worstPreset() ? [{ value: "worst", label: `Worst · ${rankedPresetLabel(worstPreset()!, catalog())}` }] : []),
              ]}
              onInput={applyRankedPair}
            />
            <InspectorSelect
              label="Parameter preset"
              value={selectedPresetId()}
              options={[
                { value: "custom", label: "Custom" },
                ...presets().map(presetOption),
              ]}
              onInput={applyPreset}
            />
            <InspectorSelect
              label="Deadband behavior"
              value={parameters().deadbandMode}
              options={[{ value: "hold", label: "Hold prior state" }, { value: "flat", label: "Go flat" }]}
              onInput={(value) => update("deadbandMode", value as VwKamaParameters["deadbandMode"])}
            />
            <InspectorSelect
              label="Threshold mode"
              value={parameters().thresholdMode ?? "static"}
              options={[
                { value: "static", label: "Static" },
                { value: "adaptive", label: "Volatility-adaptive" },
              ]}
              onInput={(value) => update("thresholdMode", value as VwKamaParameters["thresholdMode"])}
            />
            <InspectorNumber
              label="Base threshold (bps/hour)"
              value={parameters().deadbandBpsHour}
              min={0}
              step={1}
              onInput={(value) => update("deadbandBpsHour", value)}
            />
            <Show when={parameters().thresholdMode === "adaptive"}>
              <DurationInput
                label="Noise lookback"
                value={parameters().thresholdLookbackMs ?? 60 * 60_000}
                onInput={(value) => update("thresholdLookbackMs", value)}
              />
              <InspectorNumber
                label="Noise multiplier"
                value={parameters().thresholdNoiseMultiplier ?? 0}
                min={0}
                step={0.1}
                onInput={(value) => update("thresholdNoiseMultiplier", value)}
              />
            </Show>
            <DurationInput label="Efficiency" value={parameters().efficiencyMs} onInput={(value) => update("efficiencyMs", value)} />
            <DurationInput label="Fast" value={parameters().fastMs} onInput={(value) => update("fastMs", value)} />
            <DurationInput label="Slow" value={parameters().slowMs} onInput={(value) => update("slowMs", value)} />
            <DurationInput label="Volume EMA" value={parameters().volumeMs} onInput={(value) => update("volumeMs", value)} />
            <InspectorNumber label="KAMA power" value={parameters().power} min={0.1} step={0.05} onInput={(value) => update("power", value)} />
            <InspectorNumber label="Volume cap" value={parameters().volumeCap} min={1} step={0.1} onInput={(value) => update("volumeCap", value)} />
            <InspectorNumber label="Volume power" value={parameters().volumePower} min={0} step={0.1} onInput={(value) => update("volumePower", value)} />
            <div class="flex items-end">
              <button class="btn w-full" type="button" onClick={() => {
                const preset = globalPreset();
                setRankedPair("current");
                setSelectedPresetId(preset?.id ?? "custom");
                setParameters({ ...(preset?.parameters ?? catalog()?.defaults.parameters ?? defaults) });
              }}>
                <RefreshCw size={15} /> Reset parameters
              </button>
            </div>
          </div>
          <div class="mt-2 text-xs text-ink-400">
            At {formatDuration(intervalMs())}, the periods round to {sampleCount(parameters().efficiencyMs, intervalMs())} / {sampleCount(parameters().fastMs, intervalMs())} / {sampleCount(parameters().slowMs, intervalMs())} / {sampleCount(parameters().volumeMs, intervalMs())} samples.
            <Show when={parameters().thresholdMode === "adaptive"}>
              {` Threshold = base + ${formatQuote(parameters().thresholdNoiseMultiplier ?? 0, 3)} × EWMA of absolute KAMA-rate changes.`}
            </Show>
          </div>
          <details class="mt-3 rounded-2 border border-line bg-ink-800 p-3">
            <summary class="cursor-pointer text-sm font-semibold text-ink-200">Scoring controls</summary>
            <div class="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
              <InspectorNumber label="Oracle + signal friction (bps)" value={oracleFriction() * 10_000} min={0} step={1} onInput={(value) => setOracleFriction(value / 10_000)} />
              <DurationInput label="Match window" value={matchWindowMs()} onInput={setMatchWindowMs} />
              <DurationInput label="Timing half-life" value={timingHalfLifeMs()} onInput={setTimingHalfLifeMs} />
              <InspectorNumber label="Warmup multiple" value={warmupMultiple()} min={1} step={0.25} onInput={setWarmupMultiple} />
            </div>
            <div class="mt-2 text-xs text-ink-400">
              This friction shapes the oracle path and requires each candidate state change to move farther from the last emitted signal price.
            </div>
          </details>
        </section>

        <Show when={analysisError()}>
          {(message) => <ErrorNotice message={message()} />}
        </Show>

        <Show when={result()} fallback={<EmptyResult loading={loading()} />}>
          {(analysis) => (
            <>
              <section class="grid grid-cols-2 gap-3 md:grid-cols-4 xl:grid-cols-8">
                <div class="col-span-full flex items-center gap-1.5 text-xs text-ink-400">
                  <Info aria-hidden="true" size={13} />
                  These compare VW-KAMA with the perfect-margin oracle; they do not measure trading profitability. Hover or focus an info icon for details.
                </div>
                <ScoreCard label="Score" value={ratioPercent(analysis().metrics.score)} description={metricHelp.score} />
                <ScoreCard label="F1" value={ratioPercent(analysis().metrics.f1)} description={metricHelp.f1} />
                <ScoreCard label="Precision" value={ratioPercent(analysis().metrics.precision)} description={metricHelp.precision} />
                <ScoreCard label="Recall" value={ratioPercent(analysis().metrics.recall)} description={metricHelp.recall} />
                <ScoreCard label="Agreement" value={ratioPercent(analysis().metrics.exposureAgreement)} description={metricHelp.agreement} />
                <ScoreCard
                  label="Matched / total"
                  value={`${formatQuote(analysis().metrics.matchedCount, 0)} / ${formatQuote(analysis().metrics.oracleCount, 0)}`}
                  description={metricHelp.matched}
                />
                <ScoreCard label="Extra" value={formatQuote(analysis().metrics.extraSignalCount, 0)} description={metricHelp.extra} />
                <ScoreCard label="Noise / matched" value={formatNoiseRatio(analysis().metrics.noiseSignalRatio)} description={metricHelp.noise} />
                <ScoreCard label="Cleanliness" value={ratioPercent(analysis().metrics.signalCleanliness)} description={metricHelp.cleanliness} />
                <ScoreCard label="Timing error P50" value={formatDuration(analysis().metrics.lagP50Ms ?? undefined)} description={metricHelp.timingError} />
                <ScoreCard label="Signals/day" value={formatQuote(analysis().metrics.signalsPerDay, 1)} description={metricHelp.signals} />
                <ScoreCard label="Candles" value={formatQuote(analysis().candleCount, 0)} description={metricHelp.candles} />
              </section>

              <Show when={comparisonResult()}>
                {(baseline) => (
                  <section class="panel-tight flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                    <div>
                      <div class="muted-label">Per-window comparison</div>
                      <div class="text-sm text-ink-200">
                        {selectedPreset()?.label ?? "Selected config"} against {globalPreset()?.label ?? "global config"}
                      </div>
                    </div>
                    <div class="flex gap-5 text-sm tabular-nums">
                      <div><span class="text-ink-400">Window </span>{ratioPercent(analysis().metrics.score)}</div>
                      <div><span class="text-ink-400">Global </span>{ratioPercent(baseline().metrics.score)}</div>
                      <div class={analysis().metrics.score >= baseline().metrics.score ? "text-gain" : "text-loss"}>
                        {signedPercent(analysis().metrics.score - baseline().metrics.score)}
                      </div>
                    </div>
                  </section>
                )}
              </Show>

              <section class="panel">
                <div class="mb-3 flex flex-col gap-2 lg:flex-row lg:items-center lg:justify-between">
                  <div>
                    <div class="muted-label">Signal overlay</div>
                    <h2 class="text-lg font-semibold">Candidate against oracle</h2>
                    <div class="mt-1 text-xs text-ink-300">
                      {formatDateRange(timeRange().start, timeRange().end)} · {formatQuote(chartCandles().length, 0)} chart points · {formatDuration(analysis().intervalMs)} source · {chartResolution().exact ? "exact price candles" : `~${formatDuration(chartResolution().renderIntervalMs)} price buckets`}
                      <Show when={showKama()}> · {chartResolution().exact ? "exact KAMA" : "sampled KAMA"}</Show>
                      <Show when={detailLoading()}> · loading detail…</Show>
                      <Show when={analysis().elapsedMs}> · computed in {formatComputeTime(analysis().elapsedMs)}</Show>
                    </div>
                    <Show when={detailError()}>{(message) => <div class="mt-1 text-xs text-loss">{message()}</div>}</Show>
                  </div>
                  <div class="flex flex-wrap gap-2">
                    <OverlayButton label="KAMA" active={showKama()} onClick={() => setShowKama(!showKama())} />
                    <OverlayButton label="Candidate signals" active={showSignals()} onClick={() => setShowSignals(!showSignals())} />
                    <OverlayButton label="Oracle" active={showOracle()} onClick={() => setShowOracle(!showOracle())} />
                  </div>
                </div>
                <div class="mb-3 text-xs text-ink-400">
                  Purple L / F / S marks the candidate target state. Red and green L / S marks are retrospective oracle actions.
                </div>
                <div class="h-110 lg:h-[620px]">
                  <CandleChart
                    candles={chartCandles()}
                    orders={[]}
                    lastPrice={candles().at(-1)?.close ?? 0}
                    smaSeries={kamaSeries()}
                    annotations={annotations()}
                    trace={trace()}
                    overlays={overlays()}
                    maxCandles={0}
                    minInteractiveCandles={2}
                    priceDisplay="line"
                    interactive
                    timeNavigation
                    viewport={chartViewport()}
                    highlightedAnnotation={matchedCandidate()}
                    onOracleHoverChange={setHoveredOracle}
                    onViewportChange={updateViewportFromChart}
                    emptyLabel="No analyzed candles"
                  />
                </div>
              </section>

              <section class="panel">
                <div class="mb-3 flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
                  <div>
                    <div class="muted-label">State comparison</div>
                    <h2 class="text-lg font-semibold">Oracle vs candidate transitions</h2>
                    <div class="mt-1 text-xs text-ink-300">
                      {formatQuote(visibleCandidateCount(), 0)} / {formatQuote(analysis().metrics.signalCount, 0)} candidate transitions visible · {formatQuote(analysis().scoredSegmentCount, 0)} / {formatQuote(analysis().sourceSegmentCount, 0)} continuous segments scored after warmup
                    </div>
                    <div class="mt-1 text-xs tabular-nums text-ink-400">
                      {formatUtcRange(visibleTimeRange().start, visibleTimeRange().end)} · {formatDuration(visibleTimeRange().end - visibleTimeRange().start)} visible
                    </div>
                  </div>
                  <div class="flex flex-wrap items-center gap-2">
                    <button class="btn min-w-9 px-2" type="button" title="Pan left" aria-label="Pan state comparison left" onClick={() => panViewport(-0.25)}>←</button>
                    <button class="btn min-w-9 px-2" type="button" title="Zoom in" aria-label="Zoom state comparison in" onClick={() => zoomViewport(0.5)}>+</button>
                    <button class="btn min-w-9 px-2" type="button" title="Zoom out" aria-label="Zoom state comparison out" onClick={() => zoomViewport(2)}>−</button>
                    <button class="btn min-w-9 px-2" type="button" title="Pan right" aria-label="Pan state comparison right" onClick={() => panViewport(0.25)}>→</button>
                    <button class="btn" type="button" onClick={resetViewport}>Reset view</button>
                  </div>
                </div>
                <StateStrip
                  label="Oracle"
                  points={oracle().points}
                  start={visibleTimeRange().start}
                  end={visibleTimeRange().end}
                  onZoom={zoomViewport}
                  onPan={panViewport}
                />
                <StateStrip
                  label="Candidate"
                  points={candidatePoints()}
                  start={visibleTimeRange().start}
                  end={visibleTimeRange().end}
                  onZoom={zoomViewport}
                  onPan={panViewport}
                />
                <div class="ml-[96px] text-xs text-ink-400">Wheel to zoom · drag to pan · synchronized with the price chart</div>
                <div class="mt-4 overflow-auto">
                  <table class="w-full min-w-190 border-collapse">
                    <thead>
                      <tr>
                        <th class="table-head">Candidate time</th>
                        <th class="table-head">Candidate action</th>
                        <th class="table-head">Matched oracle</th>
                        <th class="table-head">Oracle action</th>
                        <th class="table-head" title="Candidate time minus oracle time; positive means the candidate was later.">Time offset</th>
                        <th class="table-head">Price</th>
                      </tr>
                    </thead>
                    <tbody>
                      <For each={visibleComparisons().slice(0, 120)} fallback={
                        <tr><td class="td-cell text-ink-300" colSpan={6}>No candidate or oracle transitions in this window.</td></tr>
                      }>
                        {(row) => (
                          <tr>
                            <td class="td-cell">{formatDateTime(row.candidate?.time)}</td>
                            <td class={`td-cell ${stateText(row.candidate?.state)}`}>{row.candidate ? stateAction(row.candidate) : "missed"}</td>
                            <td class="td-cell">{formatDateTime(row.oracle?.time)}</td>
                            <td class={`td-cell ${stateText(row.oracle?.state)}`}>{row.oracle ? stateAction(row.oracle) : "extra candidate"}</td>
                            <td class="td-cell">{row.candidate && row.oracle ? signedDuration(row.candidate.time - row.oracle.time) : "-"}</td>
                            <td class="td-cell">${formatQuote(row.candidate?.price ?? row.oracle?.price, 4)}</td>
                          </tr>
                        )}
                      </For>
                    </tbody>
                  </table>
                </div>
              </section>
            </>
          )}
        </Show>
      </div>
    </main>
  );
}

function InspectorSelect(props: {
  label: string;
  value: string;
  options: Array<{ value: string; label: string }>;
  onInput: (value: string) => void;
}) {
  return (
    <label class="block">
      <span class="mb-1 block text-xs font-medium text-ink-300">{props.label}</span>
      <select class="min-h-10 w-full rounded-2 border border-line bg-ink-800 px-3 text-sm text-ink-100" value={props.value} onInput={(event) => props.onInput(event.currentTarget.value)}>
        <For each={props.options}>
          {(option) => (
            <option value={option.value} selected={option.value === props.value}>
              {option.label}
            </option>
          )}
        </For>
      </select>
    </label>
  );
}

function InspectorNumber(props: {
  label: string;
  value: number;
  min?: number;
  step?: number;
  onInput: (value: number) => void;
}) {
  return (
    <label class="block">
      <span class="mb-1 block text-xs font-medium text-ink-300">{props.label}</span>
      <input
        class="min-h-10 w-full rounded-2 border border-line bg-ink-800 px-3 text-sm tabular-nums text-ink-100"
        type="number"
        value={props.value}
        min={props.min}
        step={props.step ?? "any"}
        onInput={(event) => {
          const value = Number(event.currentTarget.value);
          if (Number.isFinite(value)) props.onInput(value);
        }}
      />
    </label>
  );
}

function DurationInput(props: { label: string; value: number; onInput: (value: number) => void }) {
  return <InspectorNumber label={`${props.label} (seconds)`} value={round(props.value / 1_000, 3)} min={0.001} step={1} onInput={(value) => props.onInput(value * 1_000)} />;
}

function OverlayButton(props: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button class="btn" classList={{ "border-accent text-accent": props.active }} aria-pressed={props.active} type="button" onClick={props.onClick}>
      {props.label}
    </button>
  );
}

function ScoreCard(props: { label: string; value: string; description: string }) {
  const descriptionId = createUniqueId();
  return (
    <div class="panel-tight group relative min-w-0">
      <div class="flex min-w-0 items-center gap-1">
        <div class="muted-label truncate">{props.label}</div>
        <button
          type="button"
          class="shrink-0 rounded-full text-ink-300 transition hover:text-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/45"
          aria-label={`About ${props.label}`}
          aria-describedby={descriptionId}
        >
          <Info aria-hidden="true" size={12} />
        </button>
      </div>
      <div class="mt-1 truncate text-lg font-semibold tabular-nums">{props.value}</div>
      <div
        id={descriptionId}
        role="tooltip"
        class="pointer-events-none invisible absolute left-0 right-0 top-[calc(100%+0.35rem)] z-30 rounded-2 border border-line bg-ink-800 p-2 text-xs normal-case leading-relaxed tracking-normal text-ink-200 opacity-0 shadow-xl transition-opacity group-hover:visible group-hover:opacity-100 group-focus-within:visible group-focus-within:opacity-100"
      >
        {props.description}
      </div>
    </div>
  );
}

function StateStrip(props: {
  label: string;
  points: BacktestOraclePoint[];
  start: number;
  end: number;
  onZoom: (scale: number, anchor: number) => void;
  onPan: (fraction: number) => void;
}) {
  let canvas!: HTMLCanvasElement;
  let observer: ResizeObserver | undefined;
  let drag: { pointerId: number; x: number } | undefined;
  const [dragging, setDragging] = createSignal(false);
  const draw = () => {
    if (!canvas) return;
    const width = Math.max(1, canvas.clientWidth);
    const height = Math.max(1, canvas.clientHeight);
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.floor(width * ratio);
    canvas.height = Math.floor(height * ratio);
    const context = canvas.getContext("2d");
    if (!context) return;
    context.scale(ratio, ratio);
    context.clearRect(0, 0, width, height);
    for (const segment of stateSegments(props.points, props.start, props.end)) {
      const left = (segment.start - props.start) / Math.max(1, props.end - props.start) * width;
      const right = (segment.end - props.start) / Math.max(1, props.end - props.start) * width;
      context.fillStyle = stateColor(segment.state);
      context.fillRect(left, 0, Math.max(1, right - left), height);
      if (right - left < 28) continue;
      context.fillStyle = segment.state === "flat" ? "#e5e7eb" : "#071018";
      context.font = "600 10px Inter, sans-serif";
      context.textAlign = "center";
      context.textBaseline = "middle";
      context.fillText(segment.state.slice(0, 1).toUpperCase(), (left + right) / 2, height / 2);
    }
  };
  onMount(() => {
    observer = new ResizeObserver(draw);
    observer.observe(canvas);
    draw();
  });
  onCleanup(() => observer?.disconnect());
  createEffect(() => {
    props.points;
    props.start;
    props.end;
    draw();
  });
  const handleWheel = (event: WheelEvent) => {
    event.preventDefault();
    const bounds = canvas.getBoundingClientRect();
    if (event.shiftKey || Math.abs(event.deltaX) > Math.abs(event.deltaY)) {
      props.onPan((event.deltaX || event.deltaY) / Math.max(1, bounds.width));
      return;
    }
    const anchor = clamp((event.clientX - bounds.left) / Math.max(1, bounds.width), 0, 1);
    props.onZoom(Math.exp(Math.sign(event.deltaY) * 0.18), anchor);
  };
  const handlePointerDown = (event: PointerEvent) => {
    if (event.button !== 0) return;
    event.preventDefault();
    drag = { pointerId: event.pointerId, x: event.clientX };
    setDragging(true);
    canvas.setPointerCapture(event.pointerId);
  };
  const handlePointerMove = (event: PointerEvent) => {
    if (!drag || event.pointerId !== drag.pointerId) return;
    event.preventDefault();
    const distance = drag.x - event.clientX;
    drag.x = event.clientX;
    props.onPan(distance / Math.max(1, canvas.clientWidth));
  };
  const handlePointerUp = (event: PointerEvent) => {
    if (!drag || event.pointerId !== drag.pointerId) return;
    if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
    drag = undefined;
    setDragging(false);
  };
  return (
    <div class="mb-3 grid grid-cols-[84px_minmax(0,1fr)] items-center gap-3">
      <div class="text-xs font-semibold text-ink-300">{props.label}</div>
      <canvas
        class="h-8 w-full rounded-2 border border-line bg-ink-800"
        classList={{ "cursor-grab": !dragging(), "cursor-grabbing": dragging() }}
        ref={canvas}
        title="Wheel to zoom. Drag to pan."
        style={{ "touch-action": "none" }}
        onPointerCancel={handlePointerUp}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onWheel={handleWheel}
      />
    </div>
  );
}

function ErrorNotice(props: { message: string; onRetry?: () => void }) {
  return (
    <div class="flex items-center justify-between gap-3 rounded-2 border border-loss/50 bg-loss/10 px-3 py-2 text-sm text-loss">
      <span>{props.message}</span>
      <Show when={props.onRetry}><button class="btn" type="button" onClick={props.onRetry}>Retry</button></Show>
    </div>
  );
}

function EmptyResult(props: { loading: boolean }) {
  return (
    <div class="panel flex min-h-52 items-center justify-center text-center text-sm text-ink-300">
      {props.loading ? "Evaluating the selected KAMA configuration…" : "Choose an available window to inspect its signal path."}
    </div>
  );
}

function candidateAnnotations(points: VwKamaTransition[]): BacktestChartAnnotation[] {
  return points.map((point) => ({
    time: point.time,
    price: point.price,
    kind: transitionSide(point) === "buy" ? "buy-signal" : "sell-signal",
    label: `KAMA ${point.fromState} → ${point.state}`,
    reason: point.matchedTime === null
      ? `Volume-weighted KAMA candidate: ${stateAction(point)} · unmatched`
      : `Volume-weighted KAMA candidate: ${stateAction(point)} · matched oracle ${signedDuration(point.lagMs ?? 0)}`,
    signalState: point.state,
  }));
}

function compareTransitions(candidate: VwKamaTransition[], oracle: VwKamaTransition[]) {
  const byTarget = new Map(oracle.map((point) => [`${point.state}:${point.time}`, point]));
  const paired = new Set<VwKamaTransition>();
  const rows: Array<{ candidate?: VwKamaTransition; oracle?: VwKamaTransition }> = candidate.map((point) => {
    const match = point.matchedTime === null
      ? undefined
      : byTarget.get(`${point.state}:${point.matchedTime}`);
    const reciprocal = match?.matchedTime === point.time ? match : undefined;
    if (reciprocal) paired.add(reciprocal);
    return { candidate: point, oracle: reciprocal };
  });
  rows.push(...oracle.filter((point) => !paired.has(point)).map((point) => ({ oracle: point })));
  return rows.sort((left, right) =>
    (left.candidate?.time ?? left.oracle?.time ?? 0) - (right.candidate?.time ?? right.oracle?.time ?? 0));
}

function transitionSide(point: Pick<BacktestOraclePoint, "fromState" | "state">): "buy" | "sell" {
  return exposure(point.state) > exposure(point.fromState) ? "buy" : "sell";
}

function presetOption(preset: VwKamaPreset): { value: string; label: string } {
  const score = preset.score === undefined ? "" : ` · ${ratioPercent(preset.score)}`;
  return { value: preset.id, label: `${preset.label}${score}` };
}

function rankedPresetLabel(
  preset: VwKamaPreset & { score: number },
  catalog: VwKamaInspectorCatalog | undefined,
): string {
  const window = catalog?.windows.find((item) => item.id === preset.windowId);
  return `${window?.label ?? preset.windowId ?? "Unknown window"} · ${formatDuration(preset.intervalMs ?? 0)} · ${ratioPercent(preset.score)}`;
}

async function fetchAnalysis(
  request: VwKamaInspectorRequest,
  signal: AbortSignal,
): Promise<VwKamaInspectorResponse> {
  const response = await fetch(`${apiBase}/api/kama-inspector/analyze`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(request),
    signal,
  });
  const payload = await response.json() as unknown;
  if (!response.ok) throw new Error(errorMessage(payload, "KAMA analysis failed"));
  return payload as VwKamaInspectorResponse;
}

function stateSegments(points: BacktestOraclePoint[], start: number, end: number) {
  if (end <= start || points.length === 0) return [];
  const ordered = points.slice().sort((left, right) => left.time - right.time);
  let state: OracleState = ordered.filter((point) => point.time <= start).at(-1)?.state ?? "flat";
  let cursor = start;
  const segments: Array<{ state: OracleState; start: number; end: number; width: number }> = [];
  for (const point of ordered) {
    if (point.time <= start) continue;
    if (point.time >= end) break;
    if (point.state === state) continue;
    segments.push({ state, start: cursor, end: point.time, width: (point.time - cursor) / (end - start) * 100 });
    cursor = point.time;
    state = point.state;
  }
  segments.push({ state, start: cursor, end, width: (end - cursor) / (end - start) * 100 });
  return segments;
}

function ratioPercent(value: number): string {
  return `${formatQuote(value * 100, 2)}%`;
}

function formatNoiseRatio(value: number | null): string {
  return value === null ? "∞" : `${formatQuote(value, 2)}×`;
}

function signedPercent(value: number): string {
  return `${value >= 0 ? "+" : ""}${formatQuote(value * 100, 2)} pp`;
}

function exposure(state: OracleState): number {
  return state === "long" ? 1 : state === "short" ? -1 : 0;
}

function stateAction(point: Pick<BacktestOraclePoint, "fromState" | "state">): string {
  return point.state === "flat" ? "go flat" : point.fromState === "flat"
    ? `go ${point.state}` : `switch ${point.state}`;
}

function stateColor(state: OracleState): string {
  return state === "long" ? "#22c55e" : state === "short" ? "#f05252" : "#4b5563";
}

function stateText(state: OracleState | undefined): string {
  return state === "long" ? "text-gain" : state === "short" ? "text-loss" : "text-ink-300";
}

function signedDuration(value: number): string {
  if (value === 0) return "0s";
  return `${value > 0 ? "+" : value < 0 ? "−" : ""}${formatDuration(Math.abs(value))}`;
}

function formatComputeTime(value: number): string {
  return value < 1_000 ? `${value}ms` : formatDuration(value);
}

function sampleCount(durationMs: number, intervalMs: number): number {
  return Math.max(1, Math.round(durationMs / Math.max(1, intervalMs)));
}

function formatDateRange(start: number, end: number): string {
  if (!start || !end) return "-";
  const endTime = end % 86_400_000 === 0 ? end - 1 : end;
  return `${new Date(start).toISOString().slice(0, 10)} – ${new Date(endTime).toISOString().slice(0, 10)} UTC`;
}

function formatUtcRange(start: number, end: number): string {
  if (!start || !end) return "-";
  const timestamp = (value: number) => new Date(value).toISOString().replace("T", " ").slice(0, 19);
  return `${timestamp(start)} – ${timestamp(Math.max(start, end - 1))} UTC`;
}

function normalizeTimeRange(viewport: TimeRange, bounds: TimeRange, minimum: number): TimeRange {
  if (bounds.end <= bounds.start) return bounds;
  const duration = clamp(
    Math.round(viewport.end - viewport.start),
    Math.min(minimum, bounds.end - bounds.start),
    bounds.end - bounds.start,
  );
  const start = clamp(Math.round(viewport.start), bounds.start, bounds.end - duration);
  return { start, end: start + duration };
}

function candleViewport(candles: Candle[], range: TimeRange): CandleChartViewport {
  if (candles.length === 0) return { start: 0, end: 0 };
  let low = 0;
  let high = candles.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (candles[middle]!.closeTime < range.start) low = middle + 1;
    else high = middle;
  }
  const start = low;
  low = start;
  high = candles.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (candles[middle]!.openTime < range.end) low = middle + 1;
    else high = middle;
  }
  return inspectorViewport({ start, end: Math.max(start + 1, low) }, candles.length);
}

function mergeDetailCandles(
  overview: Candle[],
  detail: VwKamaCandleRangeResponse | undefined,
  analysis: VwKamaInspectorResponse | undefined,
): Candle[] {
  if (!detail || !analysis
    || detail.windowId !== analysis.window.id
    || detail.intervalMs !== analysis.intervalMs
    || detail.candles.length === 0) return overview;
  const replacement = detailReplacementRange(overview, detail.candles);
  if (!replacement) return overview;
  return [
    ...overview.filter((candle) =>
      candle.openTime < replacement.start || candle.closeTime + 1 > replacement.end),
    ...detail.candles.filter((candle) =>
      candle.openTime >= replacement.start && candle.closeTime + 1 <= replacement.end),
  ].sort((left, right) => left.openTime - right.openTime);
}

function mergeDetailKamaSeries(
  overview: BacktestChartSmaSeries,
  detail: VwKamaCandleRangeResponse | undefined,
  analysis: VwKamaInspectorResponse | undefined,
): BacktestChartSmaSeries {
  if (!detail?.kamaSeries || !analysis
    || detail.windowId !== analysis.window.id
    || detail.intervalMs !== analysis.intervalMs
    || detail.kamaSeries.points.length === 0) return overview;
  const replacement = detailReplacementRange(analysis.candles, detail.candles);
  if (!replacement) return overview;
  return {
    ...overview,
    points: [
      ...overview.points.filter((point) => point.time < replacement.start || point.time >= replacement.end),
      ...detail.kamaSeries.points.filter((point) => point.time >= replacement.start && point.time < replacement.end),
    ].sort((left, right) => left.time - right.time),
  };
}

function detailReplacementRange(overview: Candle[], detail: Candle[]): TimeRange | undefined {
  if (overview.length === 0 || detail.length === 0) return undefined;
  const overviewBoundaries = new Set<number>();
  for (const candle of overview) {
    overviewBoundaries.add(candle.openTime);
    overviewBoundaries.add(candle.closeTime + 1);
  }
  const common = new Set<number>();
  for (const candle of detail) {
    if (overviewBoundaries.has(candle.openTime)) common.add(candle.openTime);
    if (overviewBoundaries.has(candle.closeTime + 1)) common.add(candle.closeTime + 1);
  }
  const ordered = [...common].sort((left, right) => left - right);
  const start = ordered[0];
  const end = ordered.at(-1);
  return start !== undefined && end !== undefined && end > start ? { start, end } : undefined;
}

function visibleResolution(
  overview: Candle[],
  detail: VwKamaCandleRangeResponse | undefined,
  range: TimeRange,
  sourceIntervalMs: number,
  overviewIntervalMs?: number,
): { renderIntervalMs: number; exact: boolean } {
  const replacement = detail ? detailReplacementRange(overview, detail.candles) : undefined;
  if (detail && replacement
    && detail.intervalMs === sourceIntervalMs
    && replacement.start <= range.start
    && replacement.end >= range.end) {
    return {
      renderIntervalMs: detail.renderIntervalMs,
      exact: detail.renderIntervalMs === sourceIntervalMs,
    };
  }
  const candle = overview.find((item) => item.closeTime >= range.start && item.openTime < range.end)
    ?? overview[0];
  const renderIntervalMs = overviewIntervalMs
    ?? (candle ? candle.closeTime - candle.openTime + 1 : sourceIntervalMs);
  return { renderIntervalMs, exact: renderIntervalMs === sourceIntervalMs };
}

function inspectorViewport(
  viewport: CandleChartViewport | undefined,
  total: number,
): CandleChartViewport {
  if (total <= 0) return { start: 0, end: 0 };
  const minimum = Math.min(2, total);
  const visible = clampInt(Math.round((viewport?.end ?? total) - (viewport?.start ?? 0)), minimum, total);
  const start = clampInt(Math.round(viewport?.start ?? 0), 0, Math.max(0, total - visible));
  return { start, end: start + visible };
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

function clampInt(value: number, minimum: number, maximum: number): number {
  return Math.round(clamp(value, minimum, maximum));
}

function errorMessage(payload: unknown, fallback: string): string {
  const value = record(payload);
  return typeof value.error === "string" ? value.error : fallback;
}

function record(input: unknown): Record<string, unknown> {
  return input && typeof input === "object" && !Array.isArray(input) ? input as Record<string, unknown> : {};
}

function round(value: number, digits: number): number {
  return Number(value.toFixed(digits));
}
