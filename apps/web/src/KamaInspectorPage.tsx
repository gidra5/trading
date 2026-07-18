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
  VwKamaIndicatorPoint,
  VwKamaInspectorRequest,
  VwKamaInspectorResponse,
  VwKamaParameters,
  VwKamaPreset,
  VwKamaTransition,
  VwKamaValueDistillationConfig,
  VwKamaValueDistributionPoint,
} from "@trading/bot-algo";
import {
  CandleChart,
  type CandleChartOverlayVisibility,
  type CandleChartStateBand,
  type CandleChartViewport,
} from "./components/CandleChart";
import {
  IndicatorChart,
  type IndicatorChartEvent,
  type IndicatorChartSeries,
} from "./components/IndicatorChart";
import { formatDateTime, formatDuration, formatQuote } from "./format";

const apiBase = "/backend";
const debounceMs = 150;
const detailDebounceMs = 120;
const detailMaxCandles = 5_000;
const detailTriggerCandles = detailMaxCandles * 4;
const defaultValueDistillation: VwKamaValueDistillationConfig = {
  gridSize: 150,
  minExposure: -100,
  maxExposure: 100,
  maxEffectiveExposure: 250,
  holdingPeriodMode: "fixed",
  holdingPeriodMs: 1_000,
  valueHorizonMs: 1_000,
  horizonEndMode: "truncate",
  oracleTemperature: 0.01,
  strategyVolatilityScaling: false,
  opportunityEpsilon: 0.000001,
  quoteLendRate: 0,
  quoteBorrowRate: 0,
  assetBorrowRate: 0,
};

type OracleState = BacktestOraclePoint["state"];
interface TimeRange { start: number; end: number }
interface ScoreWeights { f1: number; agreement: number; cleanliness: number }
type DiagnosticIndicatorId =
  | "confirmationEma"
  | "meanReversionKama"
  | "signalFriction"
  | "kamaRate"
  | "efficiencyRatio"
  | "effectiveEfficiencyRatio"
  | "volume"
  | "relativeVolume"
  | "alpha"
  | "rsi"
  | "dmi"
  | "adx"
  | "meanDistance";
type DiagnosticVisibility = Record<DiagnosticIndicatorId, boolean>;

const defaultScoreWeights: ScoreWeights = { f1: 0.2, agreement: 0.6, cleanliness: 0.2 };
const defaultDiagnosticVisibility: DiagnosticVisibility = {
  confirmationEma: false,
  meanReversionKama: false,
  signalFriction: false,
  kamaRate: false,
  efficiencyRatio: false,
  effectiveEfficiencyRatio: false,
  volume: false,
  relativeVolume: false,
  alpha: false,
  rsi: false,
  dmi: false,
  adx: false,
  meanDistance: false,
};
const diagnosticOptions: Array<{ id: DiagnosticIndicatorId; label: string; placement: "price" | "pane" }> = [
  { id: "confirmationEma", label: "Confirmation EMA", placement: "price" },
  { id: "meanReversionKama", label: "Mean-reversion KAMA", placement: "price" },
  { id: "signalFriction", label: "Signal friction bands", placement: "price" },
  { id: "kamaRate", label: "KAMA derivative", placement: "pane" },
  { id: "efficiencyRatio", label: "Efficiency ratio", placement: "pane" },
  { id: "effectiveEfficiencyRatio", label: "Effective ER", placement: "pane" },
  { id: "volume", label: "Volume + EMA", placement: "pane" },
  { id: "relativeVolume", label: "Relative volume", placement: "pane" },
  { id: "alpha", label: "KAMA alpha", placement: "pane" },
  { id: "rsi", label: "RSI", placement: "pane" },
  { id: "dmi", label: "DMI direction", placement: "pane" },
  { id: "adx", label: "ADX", placement: "pane" },
  { id: "meanDistance", label: "Mean distance", placement: "pane" },
];

const defaults: VwKamaParameters = {
  efficiencyMs: 14 * 60_000,
  efficiencyVolumeEmaMs: 60 * 60_000,
  efficiencyVolumePower: 0,
  fastMs: 28 * 60_000,
  slowMs: 153 * 60_000,
  power: 0.49045,
  volumeMs: 130 * 60_000,
  volumeCap: 2.65003,
  volumePower: 0,
  rateMode: "relative",
  rateEmaMs: 60_000,
  deadbandBpsHour: 67.56654,
  deadbandMode: "flat",
  hysteresisReleaseRatio: 0.25,
  thresholdLookbackMs: 60 * 60_000,
  thresholdNoiseResponse: "proportional",
  thresholdNoiseMultiplier: 0,
  thresholdInverseMaxBpsHour: 0,
  thresholdInverseNoiseScaleBpsHour: 30,
  signalFrictionFraction: 1,
  strategyTemperature: 0.001,
  strategyQuadraticScale: 0,
  strategyQuadraticVolatilityMs: 60 * 60_000,
  buyMaxFraction: 1,
  sellMaxFraction: 1,
  buySizingSigmaBpsHour: 1e12,
  sellSizingSigmaBpsHour: 1e12,
  agreementMode: "sizing",
  confirmationMix: 0,
  confirmationMinQuality: 0,
  confirmationAccelerationLookbackMs: 60 * 60_000,
  confirmationDistanceLookbackMs: 60 * 60_000,
  confirmationAccelerationWeight: 1,
  confirmationDistanceWeight: 1,
  confirmationEmaMs: 60 * 60_000,
  confirmationEmaThresholdBpsHour: 0,
  confirmationEmaWeight: 0,
  confirmationEmaGateStrength: 0,
  confirmationRsiMs: 14 * 60_000,
  confirmationRsiThreshold: 0,
  confirmationRsiWeight: 0,
  confirmationDmiMs: 14 * 60_000,
  confirmationDmiWeight: 0,
  confirmationAdxThreshold: 20,
  confirmationBias: 0,
  meanReversionEfficiencyMs: 14 * 60_000,
  meanReversionFastMs: 28 * 60_000,
  meanReversionSlowMs: 60 * 60_000,
  meanReversionVolatilityMs: 60 * 60_000,
  meanReversionSuppressionThreshold: 1,
  meanReversionReversalThreshold: 0,
};

const emptyOracle: BacktestOraclePath = {
  mode: "fixed-notional",
  eventMode: "close",
  leverage: 1,
  friction: 0,
  points: [],
};

const metricHelp = {
  score: "Weighted oracle-path similarity using the adjustable F1, agreement, and cleanliness weights. It does not measure profitability.",
  f1: "Harmonic mean of one-to-one timing-weighted precision and recall. It is high only when signals are both accurate and complete.",
  precision: "The timing-weighted share of candidate transitions paired with one oracle transition reaching the same target state. Extra and time-offset signals reduce it.",
  recall: "The timing-weighted share of oracle transitions covered by one candidate transition reaching the same target state. Missed and time-offset transitions reduce it.",
  agreement: "Average candle-by-candle oracle overlap. Sizing mode marks partial positions at current prices. Confidence mode gives matching direction its confidence, oracle-flat the remaining uncertainty, and opposite direction zero.",
  timingError: "Median absolute time offset across one-to-one matched transitions. Extra candidate signals and missed oracle transitions are excluded.",
  matched: "One-to-one matches divided by all oracle transitions. The remainder are missed oracle transitions.",
  extra: "Candidate transitions that could not be paired with an oracle transition. These are unnecessary signals under this comparison.",
  noise: "Extra candidate transitions divided by one-to-one matched transitions. For example, 2× means two unnecessary transitions per matched transition. Lower is better; signal cleanliness uses its bounded inverse in the score.",
  cleanliness: "Matched transitions divided by all candidate transitions: matched / (matched + extra). Higher is better, and this contributes 20% of the overall score.",
  signals: "Candidate long, short, or flat state changes per scored day.",
  candles: "Candles included in scoring after warmup candles are excluded.",
  distillation: "Opportunity-weighted cross-entropy between the hindsight value oracle and the strategy exposure distribution. Lower is better.",
  valueHoldingPeriod: "Resolved H for which each candidate exposure is forced. Adaptive mode uses half the mean time between consecutive executable oracle state changes.",
  valueHorizon: "Rolling T−t interval between E_t and E_T. Truncate mode caps it at the window end; future-candle mode loads post-window prices so every scored target reaches t + horizon.",
  strategyReturn: "Close-to-close marked return from equity 1 and zero initial exposure, using the strategy's actual exposure and the configured friction.",
  oracleReturn: "Close-to-close marked return from equity 1 and zero initial exposure, using the executable hindsight Bellman policy and the same friction.",
  drawdown: "Largest peak-to-trough equity loss in the scored path. Continuous segments each restart at equity 1 and zero exposure.",
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
  const [valueConfig, setValueConfig] = createSignal({ ...defaultValueDistillation });
  const [scoreWeights, setScoreWeights] = createSignal({ ...defaultScoreWeights });
  const [result, setResult] = createSignal<VwKamaInspectorResponse>();
  const [comparisonResult, setComparisonResult] = createSignal<VwKamaInspectorResponse>();
  const [resultRequest, setResultRequest] = createSignal<VwKamaInspectorRequest>();
  const [catalogError, setCatalogError] = createSignal<string>();
  const [analysisError, setAnalysisError] = createSignal<string>();
  const [loading, setLoading] = createSignal(false);
  const [showKama, setShowKama] = createSignal(true);
  const [showSignals, setShowSignals] = createSignal(true);
  const [showOracle, setShowOracle] = createSignal(true);
  const [diagnosticVisibility, setDiagnosticVisibility] = createSignal({ ...defaultDiagnosticVisibility });
  const [timeViewport, setTimeViewport] = createSignal<TimeRange>();
  const [detail, setDetail] = createSignal<VwKamaCandleRangeResponse>();
  const [detailLoading, setDetailLoading] = createSignal(false);
  const [detailError, setDetailError] = createSignal<string>();
  const [hoveredOracle, setHoveredOracle] = createSignal<BacktestOraclePoint>();
  const [cursorTime, setCursorTime] = createSignal<number>();
  const [selectedTime, setSelectedTime] = createSignal<number>();
  let analysisTimer: number | undefined;
  let analysisController: AbortController | undefined;
  let detailTimer: number | undefined;
  let detailController: AbortController | undefined;
  let requestSequence = 0;
  let detailSequence = 0;
  let viewportWindowId = "";

  const selectedWindow = createMemo(() =>
    catalog()?.windows.find((window) => window.id === windowId()));
  const availableScales = createMemo(() => (catalog()?.scales ?? []).filter((scale) =>
    scale.intervalMs >= (selectedWindow()?.sourceIntervalMs ?? 1_000)));
  const selectedPreset = createMemo(() =>
    catalog()?.presets.find((preset) => preset.id === selectedPresetId()));
  const globalPreset = createMemo(() =>
    catalog()?.presets.find((preset) => preset.scope === "global"));
  const rankedPresets = createMemo(() => (catalog()?.presets ?? [])
    .flatMap((preset) => {
      const score = preset.score ?? preset.historicalScore;
      return preset.scope === "window" && Number.isFinite(score)
        ? [{ ...preset, score: score!, historical: preset.score === undefined }]
        : [];
    })
    .sort((left, right) => right.score - left.score));
  const bestPreset = createMemo(() => rankedPresets()[0]);
  const worstPreset = createMemo(() => rankedPresets().at(-1));
  const scaleWindowResults = createMemo(() => rankedPresets()
    .filter((preset) => preset.intervalMs === intervalMs()));
  const scaleMedianScore = createMemo(() => medianValue(
    scaleWindowResults().map((preset) => preset.score),
  ));
  const latestOptimization = createMemo(() => (catalog()?.presets ?? [])
    .filter((preset) => preset.scope === "window" && preset.generatedAt)
    .sort((left, right) => Date.parse(right.generatedAt!) - Date.parse(left.generatedAt!))[0]);
  const normalizedScoreWeights = createMemo(() => normalizeScoreWeights(scoreWeights()));
  const weightedScore = (metrics: VwKamaInspectorResponse["metrics"]): number => {
    const weights = normalizedScoreWeights();
    return metrics.f1 * weights.f1
      + metrics.exposureAgreement * weights.agreement
      + metrics.signalCleanliness * weights.cleanliness;
  };
  const candles = createMemo(() => result()?.candles ?? []);
  const oracle = createMemo(() => result()?.oracle ?? emptyOracle);
  const candidatePoints = createMemo(() => result()?.candidatePath.points ?? []);
  const candidateStatePoints = createMemo(() => {
    const analysis = result();
    const points = new Map((analysis?.statePoints ?? []).map((point) => [point.time, {
      time: point.time,
      state: point.candidate,
      exposure: point.candidateExposure,
    }]));
    for (const transition of analysis?.candidateTransitions ?? []) {
      points.set(transition.time, {
        time: transition.time,
        state: transition.state,
        exposure: transition.exposure,
      });
    }
    return [...points.values()].sort((left, right) => left.time - right.time);
  });
  const chartStateBands = createMemo<CandleChartStateBand[]>(() => [
    { label: "Oracle", points: oracle().points },
    { label: "Candidate", points: candidateStatePoints() },
  ]);
  const indicatorPoints = createMemo(() => mergeDetailIndicatorPoints(
    result()?.indicatorPoints ?? [],
    detail(),
    result(),
  ));
  const priceIndicatorSeries = createMemo<BacktestChartSmaSeries[]>(() => {
    const analysis = result();
    if (!analysis) return [];
    const visibility = diagnosticVisibility();
    return [
      ...(showKama() ? [mergeDetailKamaSeries(analysis.kamaSeries, detail(), analysis)] : []),
      ...(visibility.confirmationEma
        ? [priceIndicatorLine(indicatorPoints(), "confirmationEma", "Confirmation EMA", "#f5b84b", parameters().confirmationEmaMs ?? 0)]
        : []),
      ...(visibility.meanReversionKama
        ? [priceIndicatorLine(indicatorPoints(), "meanReversionKama", "Mean-reversion KAMA", "#22c55e", parameters().meanReversionSlowMs ?? 0)]
        : []),
      ...(visibility.signalFriction
        ? signalFrictionSeries(indicatorPoints(), parameters().signalFrictionFraction ?? 1, oracleFriction())
        : []),
    ];
  });
  const diagnosticSeries = createMemo<IndicatorChartSeries[]>(() => {
    const points = indicatorPoints();
    const visible = diagnosticVisibility();
    const values = parameters();
    return [
      ...(visible.kamaRate ? [diagnosticLine(points, "kamaRate", `KAMA derivative · ${values.rateMode === "log" ? "log-relative" : "relative"} · ${formatDuration(values.rateEmaMs ?? intervalMs())} EMA`, "#f472b6", {
        symmetric: true,
        suffix: " bps/h",
        references: [{ value: 0 }],
        overlays: [
          ...(sampleCount(values.rateEmaMs ?? intervalMs(), intervalMs()) > 1 ? [{
            label: values.rateMode === "log" ? "Raw log-relative source" : "Raw relative source",
            color: "#38bdf8",
            points: indicatorValues(points, "kamaRateRaw"),
          }] : []),
          {
            label: "Positive live threshold",
            color: "#f5b84b",
            dashed: true,
            points: indicatorValues(points, "threshold"),
          },
          {
            label: "Negative live threshold",
            color: "#f5b84b",
            dashed: true,
            points: indicatorValues(points, "threshold", -1),
          },
        ],
      })] : []),
      ...(visible.efficiencyRatio ? [diagnosticLine(points, "efficiencyRatio", "Efficiency ratio", "#38bdf8", { minimum: 0, maximum: 1, decimals: 3 })] : []),
      ...(visible.effectiveEfficiencyRatio ? [diagnosticLine(points, "effectiveEfficiencyRatio", "Effective ER", "#a78bfa", { minimum: 0, maximum: 1, decimals: 3 })] : []),
      ...(visible.volume ? [diagnosticLine(points, "volume", `Volume · ${formatDuration(values.volumeMs)} EMA`, "#a78bfa", {
        minimum: 0,
        decimals: 2,
        overlays: [{
          label: "Volume EMA",
          color: "#38bdf8",
          points: indicatorValues(points, "volumeAverage"),
        }],
      })] : []),
      ...(visible.relativeVolume ? [diagnosticLine(points, "relativeVolume", "Relative volume", "#eab308", {
        minimum: 0,
        maximum: Math.max(1, values.volumeCap),
        references: [{ value: 1 }],
        suffix: "×",
      })] : []),
      ...(visible.alpha ? [diagnosticLine(points, "alpha", "KAMA alpha", "#fb7185", { minimum: 0, maximum: 1, decimals: 4 })] : []),
      ...(visible.rsi ? [diagnosticLine(points, "rsi", "RSI", "#22c55e", {
        minimum: 0,
        maximum: 100,
        references: [
          { value: 50 },
          { value: 50 + (values.confirmationRsiThreshold ?? 0), color: "#f5b84b" },
          { value: 50 - (values.confirmationRsiThreshold ?? 0), color: "#f5b84b" },
        ],
      })] : []),
      ...(visible.dmi ? [diagnosticLine(points, "dmi", "DMI direction", "#38bdf8", { minimum: -100, maximum: 100, references: [{ value: 0 }] })] : []),
      ...(visible.adx ? [diagnosticLine(points, "adx", "ADX", "#f5b84b", {
        minimum: 0,
        maximum: 100,
        references: [{ value: values.confirmationAdxThreshold ?? 20, color: "#f5b84b" }],
      })] : []),
      ...(visible.meanDistance ? [diagnosticLine(points, "meanDistance", "Mean distance", "#c084fc", {
        symmetric: true,
        suffix: "σ",
        references: [
          ...((values.meanReversionReversalThreshold ?? 0) > 0 ? [
            { value: values.meanReversionSuppressionThreshold ?? 1, color: "#f5b84b" },
            { value: -(values.meanReversionSuppressionThreshold ?? 1), color: "#f5b84b" },
            { value: values.meanReversionReversalThreshold ?? 0, color: "#fb7185" },
            { value: -(values.meanReversionReversalThreshold ?? 0), color: "#fb7185" },
          ] : []),
          { value: 0 },
        ],
      })] : []),
    ];
  });
  const rejectionEvents = createMemo<IndicatorChartEvent[]>(() =>
    diagnosticVisibility().kamaRate ? signalRejectionEvents(indicatorPoints()) : []);
  const activeDiagnosticCount = createMemo(() =>
    diagnosticOptions.filter((option) => diagnosticVisibility()[option.id]).length);
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
    averages: priceIndicatorSeries().length > 0,
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
  const valueDistributions = createMemo(() => mergeDetailValueDistributions(
    result()?.valueDistributions ?? [],
    detail(),
    result(),
  ));
  const selectedDistribution = createMemo(() => nearestValueDistribution(
    valueDistributions(),
    selectedTime() ?? cursorTime(),
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
    const distillation = valueConfig();
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
        valueDistillation: { ...distillation },
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
    const distributions = analysis.valueDistributions;
    if (distributions.length > 0) {
      setSelectedTime(distributions.reduce((best, point) =>
        point.opportunity > best.opportunity ? point : best).time);
    }
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
      setValueConfig({ ...defaultValueDistillation, ...next.defaults.valueDistillation });
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

  const updateScoreWeight = (key: keyof ScoreWeights, percent: number) =>
    setScoreWeights((current) => ({ ...current, [key]: Math.max(0, percent) / 100 }));

  const toggleDiagnostic = (id: DiagnosticIndicatorId) =>
    setDiagnosticVisibility((current) => ({ ...current, [id]: !current[id] }));

  const applyPreset = (id: string) => {
    const preset = catalog()?.presets.find((item) => item.id === id);
    if (!preset) {
      setRankedPair("current");
      setSelectedPresetId("custom");
      return;
    }
    inspectPreset(preset);
  };

  const inspectPreset = (preset: VwKamaPreset) => batch(() => {
    if (preset.windowId) setWindowId(preset.windowId);
    if (preset.intervalMs) setIntervalMs(preset.intervalMs);
    setRankedPair("current");
    setSelectedPresetId(preset.id);
    setParameters({ ...preset.parameters });
    if (preset.optimization?.valueDistillation) {
      setValueConfig({ ...defaultValueDistillation, ...preset.optimization.valueDistillation });
    }
  });

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
      if (preset.optimization?.valueDistillation) {
        setValueConfig({ ...defaultValueDistillation, ...preset.optimization.valueDistillation });
      }
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

        <Show when={latestOptimization()}>
          {(preset) => {
            const run = () => preset().optimization;
            const incumbent = () => preset().incumbentScore;
            const score = () => preset().score ?? preset().historicalScore;
            return (
              <section class="panel-tight flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
                <div class="min-w-0">
                  <div class="muted-label">Latest completed optimization</div>
                  <div class="mt-1 flex flex-wrap items-baseline gap-x-3 gap-y-1">
                    <h2 class="text-lg font-semibold">{preset().label}</h2>
                    <span class="text-xl font-semibold tabular-nums text-accent">
                      {score() === undefined ? "—" : ratioPercent(score()!)}
                    </span>
                    <Show when={incumbent() !== undefined && score() !== undefined}>
                      <span class="text-sm tabular-nums text-gain">
                        {signedPercent(score()! - incumbent()!)} vs incumbent
                      </span>
                    </Show>
                    <Show when={preset().score === undefined && preset().historicalScore !== undefined}>
                      <span class="rounded-full border border-warn/40 px-2 py-0.5 text-xs text-warn">
                        Historical score v{preset().scoreVersion ?? "?"}
                      </span>
                    </Show>
                  </div>
                  <div class="mt-1 text-xs text-ink-300">
                    {preset().source ?? "Per-window optimizer result"}
                    <Show when={run()}>
                      {` · ${run()!.algorithm.toUpperCase()} ${formatQuote(run()!.population, 0)} × ${run()!.generations} × ${run()!.restarts} + ${run()!.refinementRounds} refinements · ${formatComputeTime(run()!.elapsedMs)}`}
                    </Show>
                  </div>
                  <div class="mt-1 text-xs text-ink-400">
                    Single-window hindsight upper bound—not a deployable or out-of-sample result. ER volume power {formatQuote(preset().parameters.efficiencyVolumePower ?? 0, 3)}; power 0 is standard ER.
                  </div>
                </div>
                <button class="btn shrink-0" type="button" onClick={() => inspectPreset(preset())}>
                  Inspect result
                </button>
              </section>
            );
          }}
        </Show>

        <Show when={scaleWindowResults().length > 0}>
          <section class="panel">
            <div class="mb-3 flex flex-col gap-2 lg:flex-row lg:items-end lg:justify-between">
              <div>
                <div class="muted-label">Per-window results</div>
                <h2 class="text-lg font-semibold">Hindsight optima at {formatDuration(intervalMs())}</h2>
                <div class="mt-1 text-xs text-ink-400">
                  {scaleWindowResults().length} representative windows · each configuration was optimized on its own window and is not out-of-sample evidence.
                  <Show when={scaleWindowResults().some((preset) => preset.historical)}>
                    {" Rankings use historical score v2 until these presets are re-optimized under the agreement-dominant v3 objective."}
                  </Show>
                </div>
              </div>
              <div class="grid grid-cols-3 gap-2 text-sm tabular-nums">
                <div class="rounded-2 border border-line bg-ink-800 px-3 py-2"><span class="text-ink-400">Best </span>{ratioPercent(scaleWindowResults()[0]!.score)}</div>
                <div class="rounded-2 border border-line bg-ink-800 px-3 py-2"><span class="text-ink-400">Median </span>{ratioPercent(scaleMedianScore())}</div>
                <div class="rounded-2 border border-line bg-ink-800 px-3 py-2"><span class="text-ink-400">Worst </span>{ratioPercent(scaleWindowResults().at(-1)!.score)}</div>
              </div>
            </div>
            <div class="max-h-96 overflow-auto rounded-2 border border-line">
              <table class="w-full min-w-180 border-collapse">
                <thead class="sticky top-0 bg-ink-900">
                  <tr>
                    <th class="table-head w-14">Rank</th>
                    <th class="table-head">Window</th>
                    <th class="table-head w-32">Score</th>
                    <th class="table-head w-44">Search</th>
                    <th class="table-head w-24"></th>
                  </tr>
                </thead>
                <tbody>
                  <For each={scaleWindowResults()}>
                    {(preset, index) => (
                      <tr classList={{ "bg-accent/5": preset.id === latestOptimization()?.id }}>
                        <td class="td-cell text-ink-400">{index() + 1}</td>
                        <td class="td-cell">
                          <div>{presetWindowName(preset, catalog())}</div>
                          <Show when={preset.id === latestOptimization()?.id}>
                            <div class="mt-0.5 text-xs text-accent">Latest deep rerun</div>
                          </Show>
                        </td>
                        <td class="td-cell">
                          <div class="font-medium tabular-nums">{ratioPercent(preset.score)}</div>
                          <div class="mt-1 h-1.5 overflow-hidden rounded-full bg-ink-700">
                            <div class="h-full rounded-full bg-accent" style={{ width: `${clamp(preset.score, 0, 1) * 100}%` }} />
                          </div>
                        </td>
                        <td class="td-cell text-xs text-ink-400">
                          <Show when={preset.optimization} fallback="Shallow per-window DE">
                            {(run) => `${run().population} × ${run().generations} × ${run().restarts} + ${run().refinementRounds} refinements`}
                          </Show>
                        </td>
                        <td class="td-cell text-right">
                          <button class="btn px-2 py-1 text-xs" type="button" onClick={() => inspectPreset(preset)}>Inspect</button>
                        </td>
                      </tr>
                    )}
                  </For>
                </tbody>
              </table>
            </div>
          </section>
        </Show>

        <div class="kama-inspector-workspace">
          <section class="panel kama-inspector-controls">
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

          <div class="grid grid-cols-1 gap-3 md:grid-cols-2">
            <InspectorSelect
              label="History window"
              value={windowId()}
              options={(catalog()?.windows ?? []).map((window) => ({
                value: window.id,
                label: `${window.label} · ${formatDateRange(window.startTime, window.endTime)}`,
              }))}
              onInput={(value) => {
                const nextWindow = catalog()?.windows.find((window) => window.id === value);
                batch(() => {
                  setRankedPair("current");
                  setWindowId(value);
                  if (nextWindow && intervalMs() < nextWindow.sourceIntervalMs) {
                    setIntervalMs(nextWindow.sourceIntervalMs);
                  }
                });
              }}
            />
            <InspectorSelect
              label="Candle scale"
              value={String(intervalMs())}
              options={availableScales().map((scale) => ({
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
            <InspectorSearchSelect
              label={`Parameter preset · ${(catalog()?.presets.length ?? 0) + 1} configs`}
              value={selectedPresetId()}
              options={[
                { value: "custom", label: "Custom parameters", keywords: "manual editable" },
                ...(catalog()?.presets ?? []).map(presetOption),
              ]}
              onInput={applyPreset}
            />
            <Show when={selectedPreset()?.source}>
              {(source) => (
                <div class="flex items-end text-xs text-ink-400 md:col-span-2">
                  {source()}
                  <Show when={selectedPreset()?.score !== undefined}>
                    {` · stored ${presetScoreLabel(selectedPreset()!, selectedPreset()!.score!)}`}
                  </Show>
                </div>
              )}
            </Show>
            <div class="rounded-2 border border-line bg-ink-800/50 p-3 md:col-span-2">
              <div class="mb-3 flex flex-wrap items-center justify-between gap-2">
                <div>
                  <div class="text-sm font-medium">Scoring weights</div>
                  <div class="mt-0.5 text-xs text-ink-400">
                    Applied immediately to the displayed live score. Values are normalized to 100%; optimizer default is 20 / 60 / 20.
                  </div>
                </div>
                <button class="btn px-2 py-1 text-xs" type="button" onClick={() => setScoreWeights({ ...defaultScoreWeights })}>
                  Reset 20 / 60 / 20
                </button>
              </div>
              <div class="grid grid-cols-1 gap-3 sm:grid-cols-3">
                <InspectorNumber label="F1 weight (%)" value={round(scoreWeights().f1 * 100, 3)} min={0} step={1} onInput={(value) => updateScoreWeight("f1", value)} />
                <InspectorNumber label="Agreement weight (%)" value={round(scoreWeights().agreement * 100, 3)} min={0} step={1} onInput={(value) => updateScoreWeight("agreement", value)} />
                <InspectorNumber label="Cleanliness weight (%)" value={round(scoreWeights().cleanliness * 100, 3)} min={0} step={1} onInput={(value) => updateScoreWeight("cleanliness", value)} />
              </div>
              <div class="mt-2 text-xs tabular-nums text-ink-400">
                Effective mix: {ratioPercent(normalizedScoreWeights().f1)} F1 · {ratioPercent(normalizedScoreWeights().agreement)} agreement · {ratioPercent(normalizedScoreWeights().cleanliness)} cleanliness
              </div>
            </div>
            <InspectorSelect
              label="Signal rate"
              value={parameters().rateMode ?? "relative"}
              options={[
                { value: "relative", label: "Relative KAMA derivative" },
                { value: "log", label: "Log-relative KAMA rate" },
              ]}
              onInput={(value) => update("rateMode", value as VwKamaParameters["rateMode"])}
            />
            <DurationInput
              label="Rate EMA"
              value={parameters().rateEmaMs ?? intervalMs()}
              onInput={(value) => update("rateEmaMs", value)}
            />
            <InspectorSelect
              label="Deadband behavior"
              value={parameters().deadbandMode}
              options={[
                { value: "hold", label: "Hold prior state" },
                { value: "flat", label: "Go flat" },
                { value: "hysteresis", label: "Hysteresis" },
              ]}
              onInput={(value) => update("deadbandMode", value as VwKamaParameters["deadbandMode"])}
            />
            <Show when={parameters().deadbandMode === "hysteresis"}>
              <InspectorNumber label="Hysteresis release ratio" value={parameters().hysteresisReleaseRatio ?? 0.25} min={0} max={1} step={0.05} onInput={(value) => update("hysteresisReleaseRatio", value)} />
            </Show>
            <InspectorSelect
              label="Agreement strategy"
              value={parameters().agreementMode ?? "sizing"}
              options={[
                { value: "sizing", label: "Price-marked sizing" },
                { value: "confidence", label: "Signal confidence" },
              ]}
              onInput={(value) => update("agreementMode", value as VwKamaParameters["agreementMode"])}
            />
            <DurationInput
              label="Mean-reversion ER"
              value={parameters().meanReversionEfficiencyMs ?? parameters().efficiencyMs}
              onInput={(value) => update("meanReversionEfficiencyMs", value)}
            />
            <DurationInput
              label="Mean-reversion fast"
              value={parameters().meanReversionFastMs ?? parameters().fastMs}
              onInput={(value) => update("meanReversionFastMs", value)}
            />
            <DurationInput
              label="Mean-reversion slow"
              value={parameters().meanReversionSlowMs ?? 60 * 60_000}
              onInput={(value) => update("meanReversionSlowMs", value)}
            />
            <DurationInput
              label="Mean-reversion volatility"
              value={parameters().meanReversionVolatilityMs ?? 60 * 60_000}
              onInput={(value) => update("meanReversionVolatilityMs", value)}
            />
            <InspectorNumber
              label="Suppress at (σ)"
              value={parameters().meanReversionSuppressionThreshold ?? 1}
              min={0.01}
              step={0.1}
              onInput={(value) => update("meanReversionSuppressionThreshold", value)}
            />
            <InspectorNumber
              label="Reverse at (σ; 0 disables)"
              value={parameters().meanReversionReversalThreshold ?? 0}
              min={0}
              step={0.1}
              onInput={(value) => update("meanReversionReversalThreshold", value)}
            />
            <div class="flex items-end text-xs text-ink-400 md:col-span-2">
              The mean baseline is a second KAMA. Its ER can weight price changes by the configured ER-volume EMA/power, and it shares the main KAMA power and post-ER volume settings. On a rate-band exit, distance below suppression follows KAMA, the middle zone suppresses, and distance at or beyond reversal trades against KAMA. Reversal 0 disables it; equal thresholds make a direct switch.
            </div>
            <InspectorNumber
              label="Base threshold (bps/hour)"
              value={parameters().deadbandBpsHour}
              min={0}
              step={1}
              onInput={(value) => update("deadbandBpsHour", value)}
            />
            <DurationInput
              label="Noise lookback"
              value={parameters().thresholdLookbackMs ?? 60 * 60_000}
              onInput={(value) => update("thresholdLookbackMs", value)}
            />
            <InspectorSelect
              label="Noise response"
              value={parameters().thresholdNoiseResponse ?? "proportional"}
              options={[
                { value: "proportional", label: "Proportional to noise" },
                { value: "inverse", label: "Inversely proportional" },
              ]}
              onInput={(value) => update("thresholdNoiseResponse", value as VwKamaParameters["thresholdNoiseResponse"])}
            />
            <Show
              when={parameters().thresholdNoiseResponse === "inverse"}
              fallback={(
                <InspectorNumber
                  label="Noise multiplier"
                  value={parameters().thresholdNoiseMultiplier ?? 0}
                  min={0}
                  step={0.1}
                  onInput={(value) => update("thresholdNoiseMultiplier", value)}
                />
              )}
            >
              <InspectorNumber
                label="Maximum inverse bonus (bps/hour)"
                value={parameters().thresholdInverseMaxBpsHour ?? 0}
                min={0}
                step={1}
                onInput={(value) => update("thresholdInverseMaxBpsHour", value)}
              />
              <InspectorNumber
                label="Half-decay noise scale (bps/hour)"
                value={parameters().thresholdInverseNoiseScaleBpsHour ?? 30}
                min={0.000001}
                step={1}
                onInput={(value) => update("thresholdInverseNoiseScaleBpsHour", value)}
              />
            </Show>
            <InspectorNumber label="Candidate friction fraction" value={parameters().signalFrictionFraction ?? 1} min={0} max={1} step={0.05} onInput={(value) => update("signalFrictionFraction", value)} />
            <InspectorNumber label="Base strategy temperature" value={parameters().strategyTemperature ?? 0.001} min={0.000001} step={0.0001} onInput={(value) => update("strategyTemperature", value)} />
            <InspectorNumber label="Strategy quadratic scale b₂′" value={parameters().strategyQuadraticScale ?? 0} min={0} step={1} onInput={(value) => update("strategyQuadraticScale", value)} />
            <DurationInput label="Quadratic volatility window" value={parameters().strategyQuadraticVolatilityMs ?? 60 * 60_000} onInput={(value) => update("strategyQuadraticVolatilityMs", value)} />
            <DurationInput label="Efficiency" value={parameters().efficiencyMs} onInput={(value) => update("efficiencyMs", value)} />
            <DurationInput label="ER volume EMA" value={parameters().efficiencyVolumeEmaMs ?? parameters().volumeMs} onInput={(value) => update("efficiencyVolumeEmaMs", value)} />
            <InspectorNumber label="ER volume power" value={parameters().efficiencyVolumePower ?? 0} min={0} step={0.1} onInput={(value) => update("efficiencyVolumePower", value)} />
            <DurationInput label="Fast" value={parameters().fastMs} onInput={(value) => update("fastMs", value)} />
            <DurationInput label="Slow" value={parameters().slowMs} onInput={(value) => update("slowMs", value)} />
            <DurationInput label="Volume EMA" value={parameters().volumeMs} onInput={(value) => update("volumeMs", value)} />
            <InspectorNumber label="KAMA power" value={parameters().power} min={0.1} step={0.05} onInput={(value) => update("power", value)} />
            <InspectorNumber label="Volume cap" value={parameters().volumeCap} min={1} step={0.1} onInput={(value) => update("volumeCap", value)} />
            <InspectorNumber label="Volume power" value={parameters().volumePower} min={0} step={0.1} onInput={(value) => update("volumePower", value)} />
            <InspectorNumber label={parameters().agreementMode === "confidence" ? "Buy max confidence" : "Buy max equity fraction"} value={parameters().buyMaxFraction ?? 1} min={0} max={1} step={0.05} onInput={(value) => update("buyMaxFraction", value)} />
            <InspectorNumber label={parameters().agreementMode === "confidence" ? "Sell max confidence" : "Sell max equity fraction"} value={parameters().sellMaxFraction ?? 1} min={0} max={1} step={0.05} onInput={(value) => update("sellMaxFraction", value)} />
            <InspectorNumber label="Buy sizing sigma (bps/hour)" value={parameters().buySizingSigmaBpsHour ?? 1e12} min={0.000001} step={0.1} onInput={(value) => update("buySizingSigmaBpsHour", value)} />
            <InspectorNumber label="Sell sizing sigma (bps/hour)" value={parameters().sellSizingSigmaBpsHour ?? 1e12} min={0.000001} step={0.1} onInput={(value) => update("sellSizingSigmaBpsHour", value)} />
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
            At {formatDuration(intervalMs())}, the periods round to {sampleCount(parameters().efficiencyMs, intervalMs())} / {sampleCount(parameters().fastMs, intervalMs())} / {sampleCount(parameters().slowMs, intervalMs())} / {sampleCount(parameters().volumeMs, intervalMs())} samples; the ER volume EMA is {sampleCount(parameters().efficiencyVolumeEmaMs ?? parameters().volumeMs, intervalMs())} samples. ER move weights are (volume / volume EMA)^{parameters().efficiencyVolumePower ?? 0}; power 0 is standard ER.
            <Show
              when={parameters().thresholdNoiseResponse === "inverse"}
              fallback={` Threshold = base + noise × ${formatQuote(parameters().thresholdNoiseMultiplier ?? 0, 3)}; multiplier 0 is static.`}
            >
              {` Threshold = base + ${formatQuote(parameters().thresholdInverseMaxBpsHour ?? 0, 3)} / (1 + noise / ${formatQuote(parameters().thresholdInverseNoiseScaleBpsHour ?? 30, 3)}); maximum bonus 0 is static.`}
            </Show>
            {` A one-candle rate EMA is the raw selected rate.`}
            {` Signal strength = side maximum × exp(−0.5 × (KAMA rate / side sigma)²). ${parameters().agreementMode === "confidence" ? "Confidence stays fixed until the next signal." : "Partial allocations are marked from their signal price."}`}
          </div>
          <details class="mt-3 rounded-2 border border-line bg-ink-800 p-3">
            <summary class="cursor-pointer text-sm font-semibold text-ink-200">Peak/valley confirmation</summary>
            <div class="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2">
              <InspectorNumber label="Confirmation mix" value={parameters().confirmationMix ?? 0} min={0} max={1} step={0.05} onInput={(value) => update("confirmationMix", value)} />
              <InspectorNumber label="Minimum quality" value={parameters().confirmationMinQuality ?? 0} min={0} max={1} step={0.05} onInput={(value) => update("confirmationMinQuality", value)} />
              <DurationInput label="Acceleration lookback" value={parameters().confirmationAccelerationLookbackMs ?? 60 * 60_000} onInput={(value) => update("confirmationAccelerationLookbackMs", value)} />
              <DurationInput label="Distance lookback" value={parameters().confirmationDistanceLookbackMs ?? 60 * 60_000} onInput={(value) => update("confirmationDistanceLookbackMs", value)} />
              <InspectorNumber label="Acceleration weight" value={parameters().confirmationAccelerationWeight ?? 1} min={0} step={0.1} onInput={(value) => update("confirmationAccelerationWeight", value)} />
              <InspectorNumber label="Overextension weight" value={parameters().confirmationDistanceWeight ?? 1} min={0} step={0.1} onInput={(value) => update("confirmationDistanceWeight", value)} />
              <DurationInput label="Slow EMA" value={parameters().confirmationEmaMs ?? 60 * 60_000} onInput={(value) => update("confirmationEmaMs", value)} />
              <InspectorNumber label="EMA tolerance (bps/hour)" value={parameters().confirmationEmaThresholdBpsHour ?? 0} min={0} step={1} onInput={(value) => update("confirmationEmaThresholdBpsHour", value)} />
              <InspectorNumber label="EMA trend weight" value={parameters().confirmationEmaWeight ?? 0} min={0} step={0.1} onInput={(value) => update("confirmationEmaWeight", value)} />
              <InspectorNumber label="EMA hard-gate strength" value={parameters().confirmationEmaGateStrength ?? 0} min={0} max={1} step={0.05} onInput={(value) => update("confirmationEmaGateStrength", value)} />
              <DurationInput label="RSI period" value={parameters().confirmationRsiMs ?? 14 * 60_000} onInput={(value) => update("confirmationRsiMs", value)} />
              <InspectorNumber label="RSI neutral tolerance" value={parameters().confirmationRsiThreshold ?? 0} min={0} max={50} step={1} onInput={(value) => update("confirmationRsiThreshold", value)} />
              <InspectorNumber label="RSI trend weight" value={parameters().confirmationRsiWeight ?? 0} min={0} step={0.1} onInput={(value) => update("confirmationRsiWeight", value)} />
              <DurationInput label="DMI / ADX period" value={parameters().confirmationDmiMs ?? 14 * 60_000} onInput={(value) => update("confirmationDmiMs", value)} />
              <InspectorNumber label="DMI direction weight" value={parameters().confirmationDmiWeight ?? 0} min={0} step={0.1} onInput={(value) => update("confirmationDmiWeight", value)} />
              <InspectorNumber label="ADX trend threshold" value={parameters().confirmationAdxThreshold ?? 20} min={0} max={100} step={1} onInput={(value) => update("confirmationAdxThreshold", value)} />
              <InspectorNumber label="Quality bias" value={parameters().confirmationBias ?? 0} step={0.1} onInput={(value) => update("confirmationBias", value)} />
            </div>
            <div class="mt-2 text-xs text-ink-400">
              Quality blends KAMA acceleration, price overextension, slow-EMA direction, RSI, and ADX-strength-weighted DMI direction. It scales the Gaussian strength; transitions below minimum quality are suppressed. A full EMA gate closes a countertrend position to flat and waits before entering the opposite side. Mix 0 and gate 0 disable confirmation exactly.
            </div>
          </details>
          <details class="mt-3 rounded-2 border border-line bg-ink-800 p-3">
            <summary class="cursor-pointer text-sm font-semibold text-ink-200">Scoring controls</summary>
            <div class="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2">
              <InspectorNumber label="Oracle friction (bps)" value={oracleFriction() * 10_000} min={0} step={1} onInput={(value) => setOracleFriction(value / 10_000)} />
              <DurationInput label="Match window" value={matchWindowMs()} onInput={setMatchWindowMs} />
              <DurationInput label="Timing half-life" value={timingHalfLifeMs()} onInput={setTimingHalfLifeMs} />
              <InspectorNumber label="Warmup multiple" value={warmupMultiple()} min={1} step={0.25} onInput={setWarmupMultiple} />
              <InspectorNumber label="Value grid points" value={valueConfig().gridSize} min={3} max={1024} step={1} onInput={(value) => setValueConfig((current) => ({ ...current, gridSize: Math.round(value) }))} />
              <InspectorSelect label="Holding-period source" value={valueConfig().holdingPeriodMode} options={[{ value: "fixed", label: "Fixed duration" }, { value: "oracle-half-average-trade", label: "Half average time between oracle trades" }]} onInput={(value) => setValueConfig((current) => ({ ...current, holdingPeriodMode: value as VwKamaValueDistillationConfig["holdingPeriodMode"] }))} />
              <DurationInput label={valueConfig().holdingPeriodMode === "fixed" ? "Holding period H" : "Fallback holding period H"} value={valueConfig().holdingPeriodMs} onInput={(value) => setValueConfig((current) => ({ ...current, holdingPeriodMs: value, valueHorizonMs: Math.max(current.valueHorizonMs, value) }))} />
              <DurationInput label="Value horizon T−t" value={valueConfig().valueHorizonMs} onInput={(value) => setValueConfig((current) => ({ ...current, valueHorizonMs: Math.max(value, current.holdingPeriodMs) }))} />
              <InspectorSelect label="Horizon at window end" value={valueConfig().horizonEndMode} options={[{ value: "truncate", label: "Truncate at window end" }, { value: "extend", label: "Use future candles" }]} onInput={(value) => setValueConfig((current) => ({ ...current, horizonEndMode: value as VwKamaValueDistillationConfig["horizonEndMode"] }))} />
              <InspectorNumber label="Oracle temperature" value={valueConfig().oracleTemperature} min={0.000001} step={0.001} onInput={(value) => setValueConfig((current) => ({ ...current, oracleTemperature: value }))} />
              <InspectorSelect label="Volatility temperature scaling" value={valueConfig().strategyVolatilityScaling ? "enabled" : "disabled"} options={[{ value: "disabled", label: "Disabled" }, { value: "enabled", label: "Trailing-H log-return stddev" }]} onInput={(value) => setValueConfig((current) => ({ ...current, strategyVolatilityScaling: value === "enabled" }))} />
              <InspectorNumber label="Minimum exposure" value={valueConfig().minExposure} max={-0.000001} step={0.1} onInput={(value) => setValueConfig((current) => ({ ...current, minExposure: value }))} />
              <InspectorNumber label="Maximum exposure" value={valueConfig().maxExposure} min={0.000001} step={0.1} onInput={(value) => setValueConfig((current) => ({ ...current, maxExposure: value }))} />
              <InspectorNumber label="Maximum effective exposure" value={valueConfig().maxEffectiveExposure} min={Math.max(Math.abs(valueConfig().minExposure), Math.abs(valueConfig().maxExposure))} step={1} onInput={(value) => setValueConfig((current) => ({ ...current, maxEffectiveExposure: value }))} />
            </div>
            <div class="mt-2 text-xs text-ink-400">
              Oracle friction shapes both retrospective paths. H defaults to one candle and can be fixed or resolved per continuous segment as half the average interval between consecutive executable oracle trades; the fallback is used when fewer than two oracle transitions exist. Q holds each input exposure for H, including drift-correcting rebalances, before perfect-oracle continuation. Trades target only the configured exposure grid, while price and fee drift may reach the separate effective-exposure limit before liquidation. The strategy curve is exp(b₁a + b₂a²), where b₁ is signed H-return divided by temperature and b₂ = −b₂′v² from the configured nonnegative scale and causal log-return standard deviation over the separate quadratic volatility window. Temperature scales by √(H/dt); optional temperature-volatility scaling continues to use trailing-H volatility. Return paths start at equity 1 and exposure 0.
            </div>
          </details>
        </section>

        <Show when={analysisError()}>
          {(message) => <ErrorNotice message={message()} />}
        </Show>

        <Show when={result()} fallback={<div class="kama-inspector-chart"><EmptyResult loading={loading()} /></div>}>
          {(analysis) => (
            <>
              <section class="grid grid-cols-2 gap-3 md:grid-cols-4 xl:grid-cols-8">
                <div class="col-span-full flex items-center gap-1.5 text-xs text-ink-400">
                  <Info aria-hidden="true" size={13} />
                  These compare VW-KAMA with the perfect-margin oracle; they do not measure trading profitability. Hover or focus an info icon for details.
                </div>
                <ScoreCard label="Score" value={ratioPercent(weightedScore(analysis().metrics))} description={metricHelp.score} />
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
                <Show when={analysis().metrics.valueDistillation}>
                  {(value) => (
                    <>
                      <ScoreCard label="Distillation CE" value={formatQuote(value().crossEntropy, 5)} description={metricHelp.distillation} />
                      <ScoreCard label="exp(−KL)" value={ratioPercent(value().score)} description={metricHelp.distillation} />
                      <ScoreCard label="Resolved holding H" value={formatDuration(value().holdingPeriodMs)} description={metricHelp.valueHoldingPeriod} />
                      <ScoreCard label="Value horizon T−t" value={formatDuration(value().valueHorizonMs)} description={metricHelp.valueHorizon} />
                      <ScoreCard label="Strategy return" value={ratioPercent(value().returns.strategy.totalReturn)} description={metricHelp.strategyReturn} />
                      <ScoreCard label="Oracle return" value={ratioPercent(value().returns.oracle.totalReturn)} description={metricHelp.oracleReturn} />
                      <ScoreCard label="Strategy drawdown" value={ratioPercent(value().returns.strategy.maxDrawdown)} description={metricHelp.drawdown} />
                      <ScoreCard label="Oracle drawdown" value={ratioPercent(value().returns.oracle.maxDrawdown)} description={metricHelp.drawdown} />
                    </>
                  )}
                </Show>
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
                      <div><span class="text-ink-400">Window </span>{ratioPercent(weightedScore(analysis().metrics))}</div>
                      <div><span class="text-ink-400">Global </span>{ratioPercent(weightedScore(baseline().metrics))}</div>
                      <div class={weightedScore(analysis().metrics) >= weightedScore(baseline().metrics) ? "text-gain" : "text-loss"}>
                        {signedPercent(weightedScore(analysis().metrics) - weightedScore(baseline().metrics))}
                      </div>
                    </div>
                  </section>
                )}
              </Show>

              <section class="panel kama-inspector-chart">
                <div class="mb-3 flex flex-col gap-2 lg:flex-row lg:items-center lg:justify-between">
                  <div>
                    <div class="muted-label">Signal overlay</div>
                    <h2 class="text-lg font-semibold">Candidate against oracle</h2>
                    <div class="mt-1 text-xs text-ink-300">
                      {formatDateRange(timeRange().start, timeRange().end)} · {formatQuote(chartCandles().length, 0)} chart points · {formatDuration(analysis().intervalMs)} source · {chartResolution().exact ? "exact price candles" : `~${formatDuration(chartResolution().renderIntervalMs)} price buckets`}
                      <Show when={showKama()}> · {chartResolution().exact ? "exact KAMA" : "sampled KAMA"}</Show>
                      <Show when={diagnosticSeries().length > 0}> · {chartResolution().exact ? "exact indicators" : "resolution-matched indicators"}</Show>
                      <Show when={detailLoading()}> · loading detail…</Show>
                      <Show when={analysis().elapsedMs}> · computed in {formatComputeTime(analysis().elapsedMs)}</Show>
                    </div>
                    <Show when={detailError()}>{(message) => <div class="mt-1 text-xs text-loss">{message()}</div>}</Show>
                  </div>
                  <div class="flex flex-wrap gap-2">
                    <OverlayButton label="KAMA" active={showKama()} onClick={() => setShowKama(!showKama())} />
                    <OverlayButton label="Candidate signals" active={showSignals()} onClick={() => setShowSignals(!showSignals())} />
                    <OverlayButton label="Oracle" active={showOracle()} onClick={() => setShowOracle(!showOracle())} />
                    <details class="relative">
                      <summary class="btn list-none cursor-pointer">
                        Indicators{activeDiagnosticCount() > 0 ? ` (${activeDiagnosticCount()})` : ""}
                      </summary>
                      <div class="absolute right-0 z-30 mt-2 w-80 rounded-2 border border-line bg-ink-900 p-3 shadow-xl">
                        <div class="mb-2 text-xs text-ink-400">
                          EMAs and signal-friction bands overlay price. Oscillators use independently scaled, time-aligned lanes below it.
                        </div>
                        <div class="grid grid-cols-2 gap-2">
                          <For each={diagnosticOptions}>
                            {(option) => (
                              <button
                                class="btn justify-start px-2 py-1.5 text-xs"
                                classList={{ "border-accent text-accent": diagnosticVisibility()[option.id] }}
                                type="button"
                                aria-pressed={diagnosticVisibility()[option.id]}
                                title={option.placement === "price" ? "Overlay on price chart" : "Show in aligned indicator pane"}
                                onClick={() => toggleDiagnostic(option.id)}
                              >
                                {option.label}
                              </button>
                            )}
                          </For>
                        </div>
                        <button
                          class="btn mt-2 w-full px-2 py-1 text-xs"
                          type="button"
                          onClick={() => setDiagnosticVisibility({ ...defaultDiagnosticVisibility })}
                        >
                          Clear indicator overlays
                        </button>
                      </div>
                    </details>
                  </div>
                </div>
                <div class="mb-3 text-xs text-ink-400">
                  Purple L / F / S marks the candidate target state. Red and green L / S marks are retrospective oracle actions.
                </div>
                <div class={diagnosticSeries().length > 0
                  ? "h-80 lg:h-[calc(100vh-27rem)] lg:min-h-[300px]"
                  : "h-110 lg:h-[calc(100vh-12rem)] lg:min-h-[420px]"}>
                  <CandleChart
                    candles={chartCandles()}
                    orders={[]}
                    lastPrice={candles().at(-1)?.close ?? 0}
                    smaSeries={priceIndicatorSeries()}
                    annotations={annotations()}
                    trace={trace()}
                    overlays={overlays()}
                    maxCandles={0}
                    minInteractiveCandles={2}
                    priceDisplay="line"
                    interactive
                    timeNavigation
                    viewport={chartViewport()}
                    cursorTime={cursorTime()}
                    selectedTime={selectedTime()}
                    highlightedAnnotation={matchedCandidate()}
                    stateBands={chartStateBands()}
                    onOracleHoverChange={setHoveredOracle}
                    onViewportChange={updateViewportFromChart}
                    onCursorTimeChange={setCursorTime}
                    onSelectionChange={(selection) => {
                      if (selection) setSelectedTime(selection.candle.closeTime);
                    }}
                    emptyLabel="No analyzed candles"
                  />
                </div>
                <Show when={selectedDistribution()}>
                  {(point) => (
                    <div class="mt-3 rounded-2 border border-line bg-ink-800/50 p-3">
                      <div class="mb-2 flex flex-wrap items-end justify-between gap-2">
                        <div>
                          <div class="muted-label">Selected chart point</div>
                          <h3 class="text-sm font-semibold">Oracle and predicted exposure distributions</h3>
                        </div>
                        <div class="text-xs tabular-nums text-ink-300">
                          {formatDateTime(point().time)} · oracle mode {signedExposure(point().oracleModalExposure)} / target {signedExposure(point().oracleOptimalExposure)} · strategy mode {signedExposure(distributionMode(point().values, "strategyProbability"))} / target {signedExposure(point().candidateExposure)} · rate {formatQuote(point().strategyRateBpsHour, 2)} bps/h · b₂ v {formatQuote(point().strategyQuadraticVolatility, 8)} · b₂′ {formatQuote(point().strategyQuadraticScale, 4)} · b₂ {formatQuote(point().strategyQuadraticCoefficient, 8)} · effective T {formatQuote(point().strategyTemperature, 6)} · CE {formatQuote(point().crossEntropy, 5)} · opportunity {formatQuote(point().opportunity, 6)}
                        </div>
                      </div>
                      <ExposureDistributionChart point={point()} />
                      <div class="mt-2 text-xs text-ink-400">
                        Cyan is the hindsight soft value oracle; purple is the signed-rate truncated exponential used by the loss. Click or hover the price chart to inspect another rendered candle.
                      </div>
                    </div>
                  )}
                </Show>
                <Show when={diagnosticSeries().length > 0}>
                  <div class="mt-3">
                    <div class="mb-1 flex items-center justify-between gap-3 text-xs text-ink-400">
                      <span>Signal diagnostics · causal trace · gold lines are the live ±threshold · orange marks are rejected state changes</span>
                      <span>Wheel to zoom · drag to pan</span>
                    </div>
                    <div class="h-60">
                      <IndicatorChart
                        series={diagnosticSeries()}
                        events={rejectionEvents()}
                        start={visibleTimeRange().start}
                        end={visibleTimeRange().end}
                        cursorTime={cursorTime()}
                        onCursorTimeChange={setCursorTime}
                        onZoom={zoomViewport}
                        onPan={panViewport}
                      />
                    </div>
                  </div>
                </Show>
              </section>

              <section class="panel mx-auto w-full max-w-7xl">
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
                <div class="mt-4 overflow-auto">
                  <table class="w-full min-w-190 border-collapse">
                    <thead>
                      <tr>
                        <th class="table-head">Candidate time</th>
                        <th class="table-head">Candidate action</th>
                        <th class="table-head">{parameters().agreementMode === "confidence" ? "Confidence" : "Equity target"}</th>
                        <th class="table-head" title="Weighted acceleration/overextension confirmation applied to this signal.">Quality</th>
                        <th class="table-head">Matched oracle</th>
                        <th class="table-head">Oracle action</th>
                        <th class="table-head" title="Candidate time minus oracle time; positive means the candidate was later.">Time offset</th>
                        <th class="table-head">Price</th>
                      </tr>
                    </thead>
                    <tbody>
                      <For each={visibleComparisons().slice(0, 120)} fallback={
                        <tr><td class="td-cell text-ink-300" colSpan={8}>No candidate or oracle transitions in this window.</td></tr>
                      }>
                        {(row) => (
                          <tr>
                            <td class="td-cell">{formatDateTime(row.candidate?.time)}</td>
                            <td class={`td-cell ${stateText(row.candidate?.state)}`}>{row.candidate ? stateAction(row.candidate) : "missed"}</td>
                            <td class="td-cell">{row.candidate ? ratioPercent(row.candidate.sizeFraction) : "-"}</td>
                            <td class="td-cell" title={row.candidate ? `Acceleration ${formatQuote(row.candidate.acceleration, 3)} · overextension ${formatQuote(row.candidate.overextension, 3)} · EMA ${formatQuote(row.candidate.emaRate, 2)} bps/h · RSI ${formatQuote(row.candidate.rsi, 1)} · DMI ${formatQuote(row.candidate.dmi, 1)} · ADX ${formatQuote(row.candidate.adx, 1)}` : undefined}>{row.candidate ? ratioPercent(row.candidate.quality) : "-"}</td>
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
      </div>
    </main>
  );
}

function ExposureDistributionChart(props: { point: VwKamaValueDistributionPoint }) {
  const maximum = () => Math.max(
    Number.EPSILON,
    ...props.point.values.flatMap((value) => [value.oracleProbability, value.strategyProbability]),
  );
  const strategyMode = () => distributionMode(props.point.values, "strategyProbability");
  const markerPosition = (exposure: number) => distributionMarkerPosition(
    props.point.values.map((value) => value.exposure),
    exposure,
  );
  return (
    <div>
      <div class="mb-2 flex flex-wrap gap-4 text-xs tabular-nums text-ink-300">
        <span><span class="mr-1 inline-block h-2.5 w-2.5 rounded-sm bg-cyan-400" />Oracle mode <strong class="text-cyan-200">{signedExposure(props.point.oracleModalExposure)}</strong> · target <strong class="text-cyan-100">{signedExposure(props.point.oracleOptimalExposure)}</strong> · mean {signedExposure(props.point.oracleMeanExposure)}</span>
        <span><span class="mr-1 inline-block h-2.5 w-2.5 rounded-sm bg-violet-400" />Strategy mode <strong class="text-violet-200">{signedExposure(strategyMode())}</strong> · target <strong class="text-violet-100">{signedExposure(props.point.candidateExposure)}</strong> · mean {signedExposure(props.point.strategyMeanExposure)}</span>
        <span>Oracle entropy {formatQuote(props.point.oracleEntropy, 4)}</span>
      </div>
      <div class="relative h-52 border-b border-line" role="img" aria-label={`Oracle mode ${signedExposure(props.point.oracleModalExposure)} and target ${signedExposure(props.point.oracleOptimalExposure)}; strategy mode ${signedExposure(strategyMode())} and target ${signedExposure(props.point.candidateExposure)}`}>
        <div
          class="grid h-full items-end gap-0.5 px-1 pt-2"
          style={{ "grid-template-columns": `repeat(${props.point.values.length}, minmax(0, 1fr))` }}
        >
          <For each={props.point.values}>
            {(value) => {
              const oracleMode = () => value.exposure === props.point.oracleModalExposure;
              const predictedMode = () => value.exposure === strategyMode();
              return (
                <div
                  class="group relative flex h-full items-end justify-center gap-px"
                  classList={{
                    "bg-cyan-400/8": oracleMode() && !predictedMode(),
                    "bg-violet-400/8": predictedMode() && !oracleMode(),
                    "bg-gradient-to-r from-cyan-400/10 to-violet-400/10": oracleMode() && predictedMode(),
                  }}
                  title={`Exposure ${signedExposure(value.exposure)} · oracle ${ratioPercent(value.oracleProbability)}${oracleMode() ? " (mode)" : ""} · strategy ${ratioPercent(value.strategyProbability)}${predictedMode() ? " (mode)" : ""}`}
                >
                  <div
                    class="w-[44%] min-w-px rounded-t-sm bg-cyan-400/80"
                    classList={{ "outline outline-2 outline-offset-1 outline-cyan-200 shadow-[0_0_10px_rgba(34,211,238,0.8)]": oracleMode() }}
                    style={{ height: `${value.oracleProbability / maximum() * 100}%` }}
                  />
                  <div
                    class="w-[44%] min-w-px rounded-t-sm bg-violet-400/80"
                    classList={{ "outline outline-2 outline-offset-1 outline-violet-200 shadow-[0_0_10px_rgba(167,139,250,0.8)]": predictedMode() }}
                    style={{ height: `${value.strategyProbability / maximum() * 100}%` }}
                  />
                </div>
              );
            }}
          </For>
        </div>
        <div class="pointer-events-none absolute inset-x-1 bottom-0 top-2" aria-hidden="true">
          <div
            class="absolute inset-y-0 -translate-x-px border-l-2 border-dashed border-cyan-100/90"
            style={{ left: markerPosition(props.point.oracleOptimalExposure) }}
          />
          <div
            class="absolute inset-y-0 translate-x-px border-l-2 border-dashed border-violet-100/90"
            style={{ left: markerPosition(props.point.candidateExposure) }}
          />
        </div>
      </div>
      <div
        class="grid gap-0.5 px-1 pt-1 text-center text-[10px] tabular-nums text-ink-400"
        style={{ "grid-template-columns": `repeat(${props.point.values.length}, minmax(0, 1fr))` }}
      >
        <For each={props.point.values}>
          {(value, index) => (
            <span classList={{ invisible: index() % Math.max(1, Math.ceil(props.point.values.length / 9)) !== 0 && index() !== props.point.values.length - 1 }}>
              {formatQuote(value.exposure, 1)}
            </span>
          )}
        </For>
      </div>
      <div class="mt-0.5 flex flex-wrap items-center justify-center gap-x-3 gap-y-1 text-[10px] uppercase tracking-wider text-ink-500">
        <span>Target exposure</span>
        <span class="text-cyan-200">outlined bar = oracle mode · dashed line = oracle target</span>
        <span class="text-violet-200">outlined bar = strategy mode · dashed line = strategy target</span>
      </div>
    </div>
  );
}

function distributionMode(
  values: VwKamaValueDistributionPoint["values"],
  probability: "oracleProbability" | "strategyProbability",
): number {
  if (values.length === 0) return 0;
  return values.reduce((best, value) =>
    value[probability] > best[probability]
      || (value[probability] === best[probability]
        && Math.abs(value.exposure) < Math.abs(best.exposure))
      ? value
      : best, values[0]!).exposure;
}

function distributionMarkerPosition(exposures: number[], target: number): string {
  if (exposures.length <= 1) return "50%";
  const first = exposures[0]!;
  const last = exposures.at(-1)!;
  const position = last === first ? 0 : (target - first) / (last - first);
  const centered = (Math.max(0, Math.min(1, position)) * (exposures.length - 1) + 0.5)
    / exposures.length;
  return `${centered * 100}%`;
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

interface SearchableOption {
  value: string;
  label: string;
  keywords?: string;
}

function InspectorSearchSelect(props: {
  label: string;
  value: string;
  options: SearchableOption[];
  onInput: (value: string) => void;
}) {
  const inputId = createUniqueId();
  const listId = createUniqueId();
  const [query, setQuery] = createSignal("");
  const [expanded, setExpanded] = createSignal(false);
  const [activeIndex, setActiveIndex] = createSignal(0);
  let input: HTMLInputElement | undefined;
  const selected = createMemo(() => props.options.find((option) => option.value === props.value));
  const matches = createMemo(() => {
    const terms = query().trim().toLocaleLowerCase().split(/\s+/).filter(Boolean);
    if (terms.length === 0) return props.options;
    return props.options.filter((option) => {
      const haystack = `${option.label} ${option.value} ${option.keywords ?? ""}`.toLocaleLowerCase();
      return terms.every((term) => haystack.includes(term));
    });
  });
  const visibleMatches = createMemo(() => matches().slice(0, 80));
  const choose = (option: SearchableOption) => {
    props.onInput(option.value);
    setExpanded(false);
    setQuery("");
  };
  const open = () => {
    setQuery("");
    setActiveIndex(0);
    setExpanded(true);
  };

  return (
    <div class="relative block">
      <label for={inputId} class="mb-1 block text-xs font-medium text-ink-300">{props.label}</label>
      <input
        id={inputId}
        ref={input}
        class="min-h-10 w-full rounded-2 border border-line bg-ink-800 px-3 text-sm text-ink-100"
        type="search"
        role="combobox"
        aria-autocomplete="list"
        aria-controls={listId}
        aria-expanded={expanded()}
        autocomplete="off"
        placeholder="Search by name, window, scale, ID, or objective"
        value={expanded() ? query() : selected()?.label ?? "Custom parameters"}
        onFocus={open}
        onInput={(event) => {
          setQuery(event.currentTarget.value);
          setActiveIndex(0);
          setExpanded(true);
        }}
        onBlur={() => setExpanded(false)}
        onKeyDown={(event) => {
          const options = visibleMatches();
          if (event.key === "ArrowDown") {
            event.preventDefault();
            setActiveIndex((index) => Math.max(0, Math.min(index + 1, options.length - 1)));
          } else if (event.key === "ArrowUp") {
            event.preventDefault();
            setActiveIndex((index) => Math.max(index - 1, 0));
          } else if (event.key === "Enter" && options[activeIndex()]) {
            event.preventDefault();
            choose(options[activeIndex()]!);
          } else if (event.key === "Escape") {
            setExpanded(false);
            setQuery("");
            input?.blur();
          }
        }}
      />
      <Show when={expanded()}>
        <div id={listId} role="listbox" class="absolute left-0 right-0 z-40 mt-1 max-h-80 overflow-auto rounded-2 border border-line bg-ink-900 p-1 shadow-xl">
          <For each={visibleMatches()} fallback={<div class="px-3 py-2 text-sm text-ink-400">No matching configs</div>}>
            {(option, index) => (
              <button
                type="button"
                role="option"
                aria-selected={option.value === props.value}
                class="block w-full rounded-2 px-3 py-2 text-left text-sm text-ink-200 hover:bg-ink-700"
                classList={{ "bg-ink-700": index() === activeIndex(), "text-accent": option.value === props.value }}
                onMouseEnter={() => setActiveIndex(index())}
                onMouseDown={(event) => event.preventDefault()}
                onClick={() => choose(option)}
              >
                <div class="truncate">{option.label}</div>
                <div class="mt-0.5 truncate text-xs text-ink-400">{option.value}</div>
              </button>
            )}
          </For>
          <Show when={matches().length > visibleMatches().length}>
            <div class="px-3 py-2 text-xs text-ink-400">Refine the search to see the remaining {matches().length - visibleMatches().length} configs.</div>
          </Show>
        </div>
      </Show>
    </div>
  );
}

function InspectorNumber(props: {
  label: string;
  value: number;
  min?: number;
  max?: number;
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
        max={props.max}
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

function ErrorNotice(props: { message: string; onRetry?: () => void }) {
  return (
    <div class="flex items-center justify-between gap-3 rounded-2 border border-loss/50 bg-loss/10 px-3 py-2 text-sm text-loss">
      <span>{props.message}</span>
      <Show when={props.onRetry}><button class="btn" type="button" onClick={props.onRetry}>Retry</button></Show>
    </div>
  );
}

function normalizeScoreWeights(weights: ScoreWeights): ScoreWeights {
  const total = weights.f1 + weights.agreement + weights.cleanliness;
  if (!(total > 0)) return { ...defaultScoreWeights };
  return {
    f1: weights.f1 / total,
    agreement: weights.agreement / total,
    cleanliness: weights.cleanliness / total,
  };
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
    label: `KAMA ${point.fromState} → ${point.state} · ${ratioPercent(point.sizeFraction)} · Q ${ratioPercent(point.quality)}`,
    reason: point.matchedTime === null
      ? `Quality ${ratioPercent(point.quality)} · acceleration ${formatQuote(point.acceleration, 3)} · overextension ${formatQuote(point.overextension, 3)} · EMA ${formatQuote(point.emaRate, 2)} bps/h · RSI ${formatQuote(point.rsi, 1)} · DMI/ADX ${formatQuote(point.dmi, 1)}/${formatQuote(point.adx, 1)} · unmatched`
      : `Quality ${ratioPercent(point.quality)} · acceleration ${formatQuote(point.acceleration, 3)} · overextension ${formatQuote(point.overextension, 3)} · EMA ${formatQuote(point.emaRate, 2)} bps/h · RSI ${formatQuote(point.rsi, 1)} · DMI/ADX ${formatQuote(point.dmi, 1)}/${formatQuote(point.adx, 1)} · matched oracle ${signedDuration(point.lagMs ?? 0)}`,
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

function presetOption(preset: VwKamaPreset): SearchableOption {
  const value = preset.score ?? preset.historicalScore;
  const version = preset.score === undefined && value !== undefined
    ? ` v${preset.scoreVersion ?? "?"}`
    : "";
  const score = value === undefined ? "" : ` · ${presetScoreLabel(preset, value)}${version}`;
  return {
    value: preset.id,
    label: `${preset.label}${score}`,
    keywords: [
      preset.scope,
      preset.source,
      preset.optimization?.objective,
      preset.optimization?.algorithm,
      preset.generatedAt,
    ].filter(Boolean).join(" "),
  };
}

function presetScoreLabel(preset: VwKamaPreset, value: number): string {
  return preset.optimization?.objective === "value-distillation"
    ? `CE ${formatQuote(-value, 5)}`
    : `score ${ratioPercent(value)}`;
}

function rankedPresetLabel(
  preset: VwKamaPreset & { score: number; historical: boolean },
  catalog: VwKamaInspectorCatalog | undefined,
): string {
  const window = catalog?.windows.find((item) => item.id === preset.windowId);
  const version = preset.historical ? ` v${preset.scoreVersion ?? "?"}` : "";
  return `${window?.label ?? preset.windowId ?? "Unknown window"} · ${formatDuration(preset.intervalMs ?? 0)} · ${ratioPercent(preset.score)}${version}`;
}

function presetWindowName(
  preset: VwKamaPreset,
  catalog: VwKamaInspectorCatalog | undefined,
): string {
  return catalog?.windows.find((item) => item.id === preset.windowId)?.label
    ?? preset.windowId
    ?? "Unknown window";
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

function ratioPercent(value: number): string {
  return `${formatQuote(value * 100, 2)}%`;
}

function signedExposure(value: number): string {
  return `${value >= 0 ? "+" : ""}${formatQuote(value, 3)}`;
}

function medianValue(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = values.slice().sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[middle - 1]! + sorted[middle]!) / 2
    : sorted[middle]!;
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

type NumericIndicatorKey = {
  [K in keyof VwKamaIndicatorPoint]: VwKamaIndicatorPoint[K] extends number ? K : never
}[keyof VwKamaIndicatorPoint];

function indicatorValues(
  points: VwKamaIndicatorPoint[],
  key: NumericIndicatorKey,
  multiplier = 1,
): Array<{ time: number; value: number }> {
  return points
    .map((point) => ({ time: point.time, value: point[key] * multiplier }))
    .filter((point) => Number.isFinite(point.value));
}

function priceIndicatorLine(
  points: VwKamaIndicatorPoint[],
  key: NumericIndicatorKey,
  label: string,
  color: string,
  windowMs: number,
): BacktestChartSmaSeries {
  return {
    index: -1,
    windowSec: windowMs / 1_000,
    label,
    color,
    points: indicatorValues(points, key),
  };
}

function signalFrictionSeries(
  points: VwKamaIndicatorPoint[],
  fraction: number,
  oracleFriction: number,
): BacktestChartSmaSeries[] {
  const bps = oracleFriction * fraction * 10_000;
  return [
    {
      index: -1,
      windowSec: 0,
      label: `Signal +${formatQuote(bps, 2)} bps`,
      color: "#fb923c",
      points: steppedIndicatorValues(points, "signalFrictionUpper"),
    },
    {
      index: -1,
      windowSec: 0,
      label: `Signal −${formatQuote(bps, 2)} bps`,
      color: "#fb923c",
      points: steppedIndicatorValues(points, "signalFrictionLower"),
    },
  ];
}

function steppedIndicatorValues(
  points: VwKamaIndicatorPoint[],
  key: "signalFrictionLower" | "signalFrictionUpper",
): Array<{ time: number; value: number }> {
  const result: Array<{ time: number; value: number }> = [];
  let previous: number | undefined;
  for (const point of points) {
    const value = point[key];
    if (value === null || !Number.isFinite(value)) continue;
    if (previous !== undefined && value !== previous) {
      result.push({ time: point.time, value: previous });
    }
    result.push({ time: point.time, value });
    previous = value;
  }
  return result;
}

function diagnosticLine(
  points: VwKamaIndicatorPoint[],
  key: NumericIndicatorKey,
  label: string,
  color: string,
  options: Omit<IndicatorChartSeries, "id" | "label" | "color" | "points"> = {},
): IndicatorChartSeries {
  return {
    id: key,
    label,
    color,
    points: indicatorValues(points, key),
    ...options,
  };
}

function signalRejectionEvents(points: VwKamaIndicatorPoint[]): IndicatorChartEvent[] {
  const result: IndicatorChartEvent[] = [];
  let previous = "";
  for (const point of points) {
    const reasons = point.rejectionReasons ?? [];
    if (reasons.length === 0 || !point.signalIntent) {
      previous = "";
      continue;
    }
    const key = `${point.signalIntent}:${reasons.join(",")}`;
    if (key === previous) continue;
    previous = key;
    result.push({
      time: point.time,
      seriesId: "kamaRate",
      color: "#fb923c",
      label: `${point.signalIntent.toUpperCase()} rejected · ${reasons.map(rejectionReasonLabel).join(" + ")}`,
    });
  }
  return result;
}

function rejectionReasonLabel(reason: VwKamaIndicatorPoint["rejectionReasons"][number]): string {
  switch (reason) {
    case "mean-reversion": return "mean-reversion threshold";
    case "ema-hard-gate": return "EMA hard gate";
    case "zero-quality": return "zero confirmation quality";
    case "minimum-quality": return "below minimum quality";
    case "signal-friction": return "signal friction";
  }
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

function mergeDetailIndicatorPoints(
  overview: VwKamaIndicatorPoint[],
  detail: VwKamaCandleRangeResponse | undefined,
  analysis: VwKamaInspectorResponse | undefined,
): VwKamaIndicatorPoint[] {
  if (!detail?.indicatorPoints?.length || !analysis
    || detail.windowId !== analysis.window.id
    || detail.intervalMs !== analysis.intervalMs) return overview;
  const replacement = detailReplacementRange(analysis.candles, detail.candles);
  if (!replacement) return overview;
  return [
    ...overview.filter((point) => point.time < replacement.start || point.time >= replacement.end),
    ...detail.indicatorPoints.filter((point) =>
      point.time >= replacement.start && point.time < replacement.end),
  ].sort((left, right) => left.time - right.time);
}

function mergeDetailValueDistributions(
  overview: VwKamaValueDistributionPoint[],
  detail: VwKamaCandleRangeResponse | undefined,
  analysis: VwKamaInspectorResponse | undefined,
): VwKamaValueDistributionPoint[] {
  if (!detail?.valueDistributions?.length || !analysis
    || detail.windowId !== analysis.window.id
    || detail.intervalMs !== analysis.intervalMs) return overview;
  const replacement = detailReplacementRange(analysis.candles, detail.candles);
  if (!replacement) return overview;
  return [
    ...overview.filter((point) => point.time < replacement.start || point.time >= replacement.end),
    ...detail.valueDistributions.filter((point) =>
      point.time >= replacement.start && point.time < replacement.end),
  ].sort((left, right) => left.time - right.time);
}

function nearestValueDistribution(
  points: VwKamaValueDistributionPoint[],
  time: number | undefined,
): VwKamaValueDistributionPoint | undefined {
  if (points.length === 0) return undefined;
  if (time === undefined) return points.at(-1);
  let low = 0;
  let high = points.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (points[middle]!.time < time) low = middle + 1;
    else high = middle;
  }
  const right = points[Math.min(points.length - 1, low)]!;
  const left = points[Math.max(0, low - 1)]!;
  return Math.abs(left.time - time) <= Math.abs(right.time - time) ? left : right;
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
