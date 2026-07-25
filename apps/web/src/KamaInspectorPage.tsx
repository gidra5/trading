import { For, Show, batch, createEffect, createMemo, createSignal, createUniqueId, onCleanup, onMount } from "solid-js";
import { Activity, ArrowLeft, BarChart3, Info, RefreshCw } from "lucide-solid";
import {
  DEFAULT_EXPOSURE_VALUE_GRID_SIZE,
  conditionalExposureProbabilities,
  conditionalQuadraticExposureProbabilities,
  fitConditionalQuadraticPolicy,
} from "@trading/bot-algo/exposure-value-distillation";
import {
  conditionalFourSegmentExposureProbabilities,
  conditionalFourSegmentLogSlope,
  conditionalFourSegmentParametersAt,
  conditionalFourSegmentPolicyMatrix,
  fitConditionalFourSegmentPolicy,
} from "@trading/bot-algo/conditional-exposure-distribution";
import { DEFAULT_HANDCRAFTED_INDICATOR_PARAMETERS } from "@trading/bot-algo/handcrafted-indicator-predictor";
import { DEFAULT_DIRECT_INDICATOR_PARAMETERS } from "@trading/bot-algo/direct-indicator-conditional-predictor";
import type {
  BacktestChartAnnotation,
  BacktestChartSmaSeries,
  BacktestOraclePath,
  BacktestOraclePoint,
  BacktestTrace,
  Candle,
  ConditionalFourSegmentPolicyFit,
  ConditionalFourSegmentSliceParameters,
  ConditionalQuadraticPolicyFit,
  DirectIndicatorPredictorParameters,
  DirectIndicatorParameterVector6,
  HandcraftedIndicatorPredictorParameters,
  VwKamaCandleRangeResponse,
  VwKamaInspectorCatalog,
  VwKamaIndicatorPoint,
  VwKamaHistoricalAveragesResponse,
  VwKamaInspectorRequest,
  VwKamaInspectorResponse,
  VwKamaMlpOracleAlignment,
  VwKamaParameters,
  VwKamaPreset,
  VwKamaPredictorFitResponse,
  VwKamaPredictorModel,
  VwKamaPredictorPreset,
  VwKamaTimestampPredictionResponse,
  VwKamaTransition,
  VwKamaValueDistillationConfig,
  VwKamaValueDistributionPoint,
  VwKamaValuePathPoint,
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

const apiBase = import.meta.env.DEV ? "/backend" : "";
const debounceMs = 150;
const detailDebounceMs = 120;
const detailMaxCandles = 5_000;
const detailTriggerCandles = detailMaxCandles * 4;
const hoverPredictionThrottleMs = 300;
const hoverPredictionCacheSize = 64;
const defaultValueDistillation: VwKamaValueDistillationConfig = {
  gridSize: DEFAULT_EXPOSURE_VALUE_GRID_SIZE,
  minExposure: -100,
  maxExposure: 100,
  maxEffectiveExposure: 250,
  initialExposure: 0,
  holdingPeriodMode: "fixed",
  holdingPeriodMs: 60_000,
  valueHorizonMode: "fixed",
  valueHorizonMs: 60 * 60_000,
  horizonEndMode: "truncate",
  oracleTemperature: 0.01,
  strategyVolatilityScaling: false,
  opportunityEpsilon: 0.000001,
  quoteLendRate: 0,
  quoteBorrowRate: 0,
  assetBorrowRate: 0,
  entropyGapLambda: 0,
  stateMutualInformationLambda: 0,
  oracleMutualInformationLambda: 0,
  oracleMutualInformationMode: "approximate",
  mutualInformationBins: 15,
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
  strategyNormalMixture: 0,
  strategyNormalSigma: 25,
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
  distillation: "Average-regret-weighted cross-entropy over the complete conditional current-exposure → target-exposure policy. Current exposure is weighted uniformly. Lower is better.",
  averageRegret: "Time weight before epsilon: for each current exposure, best conditional oracle value minus the mean value of all target actions, averaged uniformly over current exposures.",
  mixedLoss: "Optimizer objective: cross-entropy plus the weighted excess-entropy penalty, minus weighted state and oracle mutual information. Lower is better.",
  entropyGap: "Opportunity-weighted squared excess of forecast entropy over oracle entropy, normalized by the grid's maximum entropy.",
  stateMutualInformation: "Normalized Gaussian variance-decomposition estimate of how much the strategy exposure distribution changes across market states. Higher is more state-responsive.",
  oracleMutualInformation: "Normalized dependence between soft oracle and strategy exposure distributions. Approximate mode uses distribution moments; precise mode computes categorical MI over the configured exposure bins.",
  valueHoldingPeriod: "Resolved H between oracle decisions. An action rebalances once, then quote and asset quantities remain untouched while exposure drifts until the next decision. Adaptive mode uses half the mean time between consecutive executable oracle state changes.",
  valueHorizon: "Rolling T−t interval between E_t and E_T. Truncate mode caps it at the window end; future-candle mode loads post-window prices so every scored target reaches t + horizon.",
  strategyReturn: "Close-to-close marked return from equity 1 and zero initial exposure, using the sampled forecast-regret mean exposure and configured friction.",
  oracleReturn: "exp(Q₀(initial exposure))−1 for one full-window Bellman policy after the mandatory terminal rebalance to zero, including friction and maintenance.",
  drawdown: "Largest peak-to-trough equity loss in the scored path. Continuous segments each restart at equity 1 and zero exposure.",
} as const;

export function KamaInspectorPage() {
  const [catalog, setCatalog] = createSignal<VwKamaInspectorCatalog>();
  const [windowId, setWindowId] = createSignal("");
  const [latestDays, setLatestDays] = createSignal(7);
  const [intervalMs, setIntervalMs] = createSignal(60_000);
  const [parameters, setParameters] = createSignal({ ...defaults });
  const [selectedPresetId, setSelectedPresetId] = createSignal("custom");
  const [predictorModel, setPredictorModel] = createSignal<VwKamaPredictorModel>("handcrafted");
  const [handcraftedParameters, setHandcraftedParameters] = createSignal<HandcraftedIndicatorPredictorParameters>({
    ...DEFAULT_HANDCRAFTED_INDICATOR_PARAMETERS,
  });
  const [directIndicatorParameters, setDirectIndicatorParameters] = createSignal<DirectIndicatorPredictorParameters>({
    ...DEFAULT_DIRECT_INDICATOR_PARAMETERS,
  });
  const [mlpModelId, setMlpModelId] = createSignal("");
  const [mlpOracleAlignment, setMlpOracleAlignment] =
    createSignal<VwKamaMlpOracleAlignment>("model-target");
  const [selectedPredictorPresetId, setSelectedPredictorPresetId] = createSignal("custom");
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
  const [valuePathVisibility, setValuePathVisibility] = createSignal({
    oracleExposure: true,
    oracleEquity: false,
    candidateExposure: true,
    candidateEquity: true,
  });
  const toggleValuePathSeries = (
    series: "oracleExposure" | "oracleEquity" | "candidateExposure" | "candidateEquity",
  ) => setValuePathVisibility((visible) => ({ ...visible, [series]: !visible[series] }));
  const [diagnosticVisibility, setDiagnosticVisibility] = createSignal({ ...defaultDiagnosticVisibility });
  const [timeViewport, setTimeViewport] = createSignal<TimeRange>();
  const [detail, setDetail] = createSignal<VwKamaCandleRangeResponse>();
  const [detailLoading, setDetailLoading] = createSignal(false);
  const [detailError, setDetailError] = createSignal<string>();
  const [predictionError, setPredictionError] = createSignal<string>();
  const [hoveredOracle, setHoveredOracle] = createSignal<BacktestOraclePoint>();
  const [cursorTime, setCursorTime] = createSignal<number>();
  const [selectedTime, setSelectedTime] = createSignal<number>();
  const [timestampPredictions, setTimestampPredictions] = createSignal(
    new Map<number, VwKamaTimestampPredictionResponse>(),
  );
  let analysisTimer: number | undefined;
  let analysisController: AbortController | undefined;
  let detailTimer: number | undefined;
  let detailController: AbortController | undefined;
  let requestSequence = 0;
  let detailSequence = 0;
  let predictionTimer: number | undefined;
  let predictionController: AbortController | undefined;
  let predictionLastStartedAt = 0;
  let predictionSequence = 0;
  let viewportWindowId = "";

  const selectedWindow = createMemo(() =>
    catalog()?.windows.find((window) => window.id === windowId()));
  const availableScales = createMemo(() => (catalog()?.scales ?? []).filter((scale) =>
    scale.intervalMs >= (selectedWindow()?.sourceIntervalMs ?? 1_000)));
  const selectedPreset = createMemo(() =>
    catalog()?.presets.find((preset) => preset.id === selectedPresetId()));
  const globalPreset = createMemo(() =>
    catalog()?.presets.find((preset) => preset.scope === "global"));
  const selectedPredictorPreset = createMemo(() =>
    catalog()?.predictorPresets.find((preset) => preset.id === selectedPredictorPresetId()));
  const selectedMlpModel = createMemo(() =>
    catalog()?.mlpModels.find((model) => model.id === mlpModelId()));
  const modelPredictorPresets = createMemo(() => (catalog()?.predictorPresets ?? [])
    .filter((preset) => preset.model === predictorModel()));
  const globalPredictorPreset = createMemo(() =>
    modelPredictorPresets().find((preset) => preset.scope === "global"));
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
  const displayedPredictorModel = createMemo(() =>
    resultRequest()?.predictor?.model ?? predictorModel());
  const chartCandles = createMemo(() => mergeDetailCandles(candles(), detail(), result()));
  const valueOraclePath = createMemo(() => {
    const analysis = result();
    return mergeDetailValueOraclePath(
      analysis?.valueOraclePath ?? [],
      detail(),
      analysis,
    );
  });
  const valueCandidatePath = createMemo(() => {
    const analysis = result();
    return mergeDetailValueCandidatePath(
      analysis?.valueCandidatePath ?? [],
      detail(),
      analysis,
    );
  });
  const oracle = createMemo(() => exposurePathOracle(
    valueOraclePath(),
    chartCandles(),
    oracleFriction(),
  ));
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
  const valuePathSeries = createMemo<IndicatorChartSeries[]>(() => {
    const visible = valuePathVisibility();
    const oraclePath = visible.oracleExposure || visible.oracleEquity ? valueOraclePath() : [];
    const candidatePath = visible.candidateExposure || visible.candidateEquity ? valueCandidatePath() : [];
    return [
      ...(visible.oracleExposure && oraclePath.length > 0 ? [{
        id: "value-oracle-exposure",
        label: "Full-window optimal exposure",
        color: "#22d3ee",
        points: oraclePath.map((point) => ({ time: point.time, value: point.exposure })),
        symmetric: true,
        references: [{ value: 0 }],
        decimals: 3,
      }] : []),
      ...(visible.oracleEquity && oraclePath.length > 0 ? [{
        id: "value-oracle-equity",
        label: "Full-window oracle equity",
        color: "#22c55e",
        points: oraclePath.map((point) => ({ time: point.time, value: point.equity })),
        minimum: 0,
        decimals: 4,
      }] : []),
      ...(visible.candidateExposure && candidatePath.length > 0 ? [{
        id: "value-candidate-exposure",
        label: "Candidate exposure",
        color: "#a78bfa",
        points: candidatePath.map((point) => ({ time: point.time, value: point.exposure })),
        symmetric: true,
        references: [{ value: 0 }],
        decimals: 3,
      }] : []),
      ...(visible.candidateEquity && candidatePath.length > 0 ? [{
        id: "value-candidate-equity",
        label: "Candidate equity",
        color: "#f5b84b",
        points: candidatePath.map((point) => ({ time: point.time, value: point.equity })),
        minimum: 0,
        decimals: 4,
      }] : []),
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
  const chartViewport = createMemo(() => candleViewport(chartCandles(), visibleTimeRange()));
  const chartResolution = createMemo(() => visibleResolution(
    candles(),
    detail(),
    visibleTimeRange(),
    result()?.intervalMs ?? intervalMs(),
    result()?.renderIntervalMs,
  ));
  const valueDistributions = createMemo(() => {
    const merged = mergeDetailValueDistributions(
      result()?.valueDistributions ?? [],
      detail(),
      result(),
    );
    const request = resultRequest();
    if (request?.predictor?.model !== "mlp") return merged;
    const exact = [...timestampPredictions().values()]
      .filter((prediction) => prediction.modelId === request.predictor!.mlpModelId)
      .map((prediction) => prediction.point);
    const exactTimes = new Set(exact.map((point) => point.time));
    return [...merged.filter((point) => !exactTimes.has(point.time)), ...exact]
      .sort((left, right) => left.time - right.time);
  });
  const inspectedTime = () => selectedTime() ?? cursorTime();
  const alignedPredictionUnavailable = createMemo(() => {
    const time = inspectedTime();
    const analysis = result();
    const model = selectedMlpModel();
    return time !== undefined
      && analysis !== undefined
      && displayedPredictorModel() === "mlp"
      && mlpOracleAlignment() === "model-target"
      && model !== undefined
      && time + model.predictionDelayMs >= analysis.window.endTime;
  });
  const selectedDistribution = createMemo(() => {
    if (alignedPredictionUnavailable()) return undefined;
    return nearestValueDistribution(valueDistributions(), inspectedTime());
  });
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
    const preset = selectedPredictorPreset();
    const model = predictorModel();
    if ((model !== "handcrafted" && model !== "direct-indicator") || !id
      || preset?.scope !== "window"
      || preset.windowId === id && (preset.intervalMs == null || preset.intervalMs === interval)) return;
    const next = modelPredictorPresets().find((item) => item.scope === "window"
      && item.windowId === id
      && (item.intervalMs == null || item.intervalMs === interval))
      ?? globalPredictorPreset();
    setSelectedPredictorPresetId(next?.id ?? "custom");
    if (next?.model === "handcrafted") {
      setHandcraftedParameters({ ...next.parameters } as HandcraftedIndicatorPredictorParameters);
    } else if (next?.model === "direct-indicator") {
      setDirectIndicatorParameters({ ...next.parameters } as DirectIndicatorPredictorParameters);
    }
  });

  createEffect(() => {
    const id = windowId();
    const recentDays = latestDays();
    const interval = intervalMs();
    const config = parameters();
    const model = predictorModel();
    const forecastParameters = handcraftedParameters();
    const directParameters = directIndicatorParameters();
    const selectedMlpModel = mlpModelId();
    const oracleAlignment = mlpOracleAlignment();
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
        latestDays: recentDays,
        intervalMs: interval,
        parameters: config,
        predictor: {
          model,
          handcraftedParameters: { ...forecastParameters },
          directIndicatorParameters: { ...directParameters },
          mlpModelId: selectedMlpModel,
          mlpOracleAlignment: oracleAlignment,
        },
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
    const viewportKey = `${analysis.window.id}:${analysis.window.startTime}:${analysis.window.endTime}`;
    const reset = viewportKey !== viewportWindowId;
    viewportWindowId = viewportKey;
    setTimeViewport((current) => {
      const next = reset || !current
        ? bounds
        : normalizeTimeRange(current, bounds, Math.min(bounds.end - bounds.start, analysis.intervalMs * 2));
      return current && current.start === next.start && current.end === next.end ? current : next;
    });
    setDetail(undefined);
    setTimestampPredictions(new Map());
    const distributions = analysis.valueDistributions;
    setSelectedTime(undefined);
    if (distributions.length > 0) {
      setCursorTime(distributions.reduce((best, point) =>
        point.opportunity > best.opportunity ? point : best).time);
    } else {
      setCursorTime(undefined);
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

  createEffect(() => {
    const request = resultRequest();
    const analysis = result();
    const time = inspectedTime();
    if (predictionTimer !== undefined) window.clearTimeout(predictionTimer);
    predictionTimer = undefined;
    const sequence = ++predictionSequence;
    if (!request || !analysis || request.predictor?.model !== "mlp" || time === undefined) {
      predictionController?.abort();
      predictionController = undefined;
      setPredictionError(undefined);
      return;
    }
    const model = catalog()?.mlpModels.find((item) => item.id === request.predictor!.mlpModelId);
    if (request.predictor.mlpOracleAlignment === "model-target"
      && model
      && time + model.predictionDelayMs >= analysis.window.endTime) {
      predictionController?.abort();
      predictionController = undefined;
      setPredictionError(undefined);
      return;
    }
    const existing = [
      ...analysis.valueDistributions,
      ...(detail()?.valueDistributions ?? []),
    ].some((point) => point.time === time);
    if (existing || timestampPredictions().has(time)) {
      predictionController?.abort();
      predictionController = undefined;
      setPredictionError(undefined);
      return;
    }
    const delay = Math.max(0, hoverPredictionThrottleMs - (performance.now() - predictionLastStartedAt));
    predictionTimer = window.setTimeout(() => {
      predictionTimer = undefined;
      predictionLastStartedAt = performance.now();
      predictionController?.abort();
      const controller = new AbortController();
      predictionController = controller;
      setPredictionError(undefined);
      void fetch(`${apiBase}/api/kama-inspector/predict`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ ...request, time }),
        signal: controller.signal,
      }).then(async (response) => {
        const payload = await response.json() as unknown;
        if (!response.ok) throw new Error(errorMessage(payload, "MLP timestamp prediction failed"));
        if (!controller.signal.aborted && sequence === predictionSequence) {
          const prediction = payload as VwKamaTimestampPredictionResponse;
          setTimestampPredictions((current) => {
            const next = new Map(current);
            next.delete(prediction.time);
            next.set(prediction.time, prediction);
            while (next.size > hoverPredictionCacheSize) {
              const oldest = next.keys().next().value as number | undefined;
              if (oldest === undefined) break;
              next.delete(oldest);
            }
            return next;
          });
        }
      }).catch((error) => {
        if (!controller.signal.aborted && sequence === predictionSequence) {
          setPredictionError(error instanceof Error ? error.message : "MLP timestamp prediction failed");
        }
      });
    }, delay);
  });

  onCleanup(() => {
    if (analysisTimer !== undefined) window.clearTimeout(analysisTimer);
    analysisController?.abort();
    cancelDetailRequest();
    if (predictionTimer !== undefined) window.clearTimeout(predictionTimer);
    predictionController?.abort();
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
      setLatestDays(next.defaults.latestDays ?? 7);
      setIntervalMs(next.defaults.intervalMs ?? next.scales[0]!.intervalMs);
      setParameters({ ...next.defaults.parameters });
      setSelectedPresetId(next.presets.find((preset) => preset.scope === "global")?.id ?? "custom");
      const predictor = next.defaults.predictor;
      setPredictorModel(predictor?.model ?? "handcrafted");
      setHandcraftedParameters({
        ...DEFAULT_HANDCRAFTED_INDICATOR_PARAMETERS,
        ...predictor?.handcraftedParameters,
      });
      setDirectIndicatorParameters({
        ...DEFAULT_DIRECT_INDICATOR_PARAMETERS,
        ...predictor?.directIndicatorParameters,
      });
      setMlpModelId(predictor?.mlpModelId || next.mlpModels[0]?.id || "");
      setMlpOracleAlignment(predictor?.mlpOracleAlignment ?? "model-target");
      setSelectedPredictorPresetId(
        next.predictorPresets.find((preset) => preset.scope === "global"
          && preset.model === (predictor?.model ?? "handcrafted"))?.id ?? "custom",
      );
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

  const updateHandcraftedParameter = <K extends keyof HandcraftedIndicatorPredictorParameters>(
    key: K,
    value: HandcraftedIndicatorPredictorParameters[K],
  ) => {
    setSelectedPredictorPresetId("custom");
    setHandcraftedParameters((current) => ({ ...current, [key]: value }));
  };

  const updateDirectIndicatorParameter = <K extends keyof DirectIndicatorPredictorParameters>(
    key: K,
    value: DirectIndicatorPredictorParameters[K],
  ) => {
    setSelectedPredictorPresetId("custom");
    setDirectIndicatorParameters((current) => ({ ...current, [key]: value }));
  };

  const choosePredictorModel = (model: VwKamaPredictorModel) => batch(() => {
    setPredictorModel(model);
    if (model === "legacy" || model === "mlp") {
      setSelectedPredictorPresetId("custom");
      if (model === "mlp" && !mlpModelId()) setMlpModelId(catalog()?.mlpModels[0]?.id ?? "");
      return;
    }
    const preset = (catalog()?.predictorPresets ?? []).find((item) =>
      item.model === model && item.scope === "global");
    setSelectedPredictorPresetId(preset?.id ?? "custom");
    if (model === "handcrafted") {
      setHandcraftedParameters({
        ...(preset?.parameters ?? DEFAULT_HANDCRAFTED_INDICATOR_PARAMETERS),
      } as HandcraftedIndicatorPredictorParameters);
    } else if (model === "direct-indicator") {
      setDirectIndicatorParameters({
        ...(preset?.parameters ?? DEFAULT_DIRECT_INDICATOR_PARAMETERS),
      } as DirectIndicatorPredictorParameters);
    }
  });

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

  const applyPredictorPreset = (id: string) => {
    const preset = catalog()?.predictorPresets.find((item) => item.id === id);
    if (!preset) {
      setSelectedPredictorPresetId("custom");
      return;
    }
    batch(() => {
      if (preset.windowId) setWindowId(preset.windowId);
      if (preset.intervalMs) setIntervalMs(preset.intervalMs);
      setPredictorModel(preset.model);
      setSelectedPredictorPresetId(preset.id);
      if (preset.model === "handcrafted") {
        setHandcraftedParameters({ ...preset.parameters } as HandcraftedIndicatorPredictorParameters);
      } else {
        setDirectIndicatorParameters({ ...preset.parameters } as DirectIndicatorPredictorParameters);
      }
    });
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
          <div class="flex flex-wrap gap-2">
            <a class="btn" href="#/portfolio-index">
              <BarChart3 size={16} /> Basis Index
            </a>
            <a class="btn" href="#/mlp-training">
              <Activity size={16} /> MLP Training
            </a>
            <a class="btn" href="#/">
              <ArrowLeft size={16} /> Dashboard
            </a>
          </div>
        </header>

        <Show when={catalogError()}>
          {(message) => <ErrorNotice message={message()} onRetry={() => void loadCatalog()} />}
        </Show>

        <Show when={predictorModel() === "legacy" ? latestOptimization() : undefined}>
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

        <Show when={predictorModel() === "legacy" && scaleWindowResults().length > 0}>
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
              <h2 class="text-lg font-semibold">{predictorModel() === "legacy" ? "Window and signal parameters" : "Window and predictor parameters"}</h2>
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
                label: window.id === "latest"
                  ? latestHistoryLabel(latestDays(), window.endTime)
                  : `${window.label} · ${formatDateRange(window.startTime, window.endTime)}`,
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
            <Show when={windowId() === "latest"}>
              <InspectorNumber
                label="Latest history (days)"
                value={latestDays()}
                min={1}
                step={1}
                onInput={(value) => setLatestDays(Math.round(value))}
              />
            </Show>
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
            <Show when={predictorModel() === "legacy"}>
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
                label={`VW-KAMA signal preset · ${(catalog()?.presets.length ?? 0) + 1} configs`}
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
            </Show>
            <div class="rounded-2 border border-violet-400/20 bg-violet-400/5 p-3 md:col-span-2">
              <div class="mb-3">
                <div class="text-sm font-medium text-violet-100">Exposure prediction model</div>
                <div class="mt-0.5 text-xs text-ink-400">
                  Choose the distribution compared with the exposure-value oracle. Only controls used by the selected model are shown.
                </div>
              </div>
              <div class="grid grid-cols-1 gap-3 md:grid-cols-2">
                <InspectorSelect
                  label="Prediction model"
                  value={predictorModel()}
                  options={[
                    { value: "handcrafted", label: "Handcrafted drift/variance forecast" },
                    { value: "direct-indicator", label: "Direct indicator → 6-parameter quadratic distribution" },
                    { value: "mlp", label: "MLP · direct 255-action distribution" },
                    { value: "legacy", label: "Legacy KAMA-rate distribution" },
                  ]}
                  onInput={(value) => choosePredictorModel(value as VwKamaPredictorModel)}
                />
                <Show when={predictorModel() === "handcrafted"}>
                  <InspectorSearchSelect
                    label={`Forecast preset · ${modelPredictorPresets().length + 1} configs`}
                    value={selectedPredictorPresetId()}
                    options={[
                      { value: "custom", label: "Custom forecast parameters", keywords: "manual editable" },
                      ...modelPredictorPresets().map((preset) => predictorPresetOption(preset, catalog())),
                    ]}
                    onInput={applyPredictorPreset}
                  />
                  <DurationInput label="Drift estimate half-life" value={handcraftedParameters().driftEstimateHalfLifeMs} onInput={(value) => updateHandcraftedParameter("driftEstimateHalfLifeMs", value)} />
                  <DurationInput label="Drift forecast half-life" value={handcraftedParameters().driftForecastHalfLifeMs} onInput={(value) => updateHandcraftedParameter("driftForecastHalfLifeMs", value)} />
                  <InspectorNumber label="Drift scale" value={handcraftedParameters().driftScale} min={0} step={0.01} onInput={(value) => updateHandcraftedParameter("driftScale", value)} />
                  <DurationInput label="Variance estimate half-life" value={handcraftedParameters().varianceEstimateHalfLifeMs} onInput={(value) => updateHandcraftedParameter("varianceEstimateHalfLifeMs", value)} />
                  <DurationInput label="Long-run variance half-life" value={handcraftedParameters().longRunVarianceHalfLifeMs} onInput={(value) => updateHandcraftedParameter("longRunVarianceHalfLifeMs", value)} />
                  <DurationInput label="Variance forecast half-life" value={handcraftedParameters().varianceForecastHalfLifeMs} onInput={(value) => updateHandcraftedParameter("varianceForecastHalfLifeMs", value)} />
                  <div class="flex items-end">
                    <button class="btn w-full" type="button" onClick={() => {
                      const preset = globalPredictorPreset();
                      setSelectedPredictorPresetId(preset?.id ?? "custom");
                      setHandcraftedParameters({
                        ...(preset?.parameters ?? DEFAULT_HANDCRAFTED_INDICATOR_PARAMETERS),
                      } as HandcraftedIndicatorPredictorParameters);
                    }}>
                      <RefreshCw size={15} /> Reset forecast
                    </button>
                  </div>
                  <Show when={selectedPredictorPreset()}>
                    {(preset) => (
                      <div class="text-xs text-ink-400 md:col-span-2">
                        {preset().source} · fitted mean decision regret {formatQuote(preset().loss, 6)}
                        <Show when={preset().diagnosticCrossEntropy !== undefined}>
                          {` · diagnostic CE ${formatQuote(preset().diagnosticCrossEntropy!, 5)}`}
                        </Show>
                      </div>
                    )}
                  </Show>
                </Show>
                <Show when={predictorModel() === "direct-indicator"}>
                  <InspectorSearchSelect
                    label={`Direct-model preset · ${modelPredictorPresets().length + 1} configs`}
                    value={selectedPredictorPresetId()}
                    options={[
                      { value: "custom", label: "Custom direct parameters", keywords: "manual editable" },
                      ...modelPredictorPresets().map((preset) => predictorPresetOption(preset, catalog())),
                    ]}
                    onInput={applyPredictorPreset}
                  />
                  <DurationInput label="Drift estimate half-life" value={directIndicatorParameters().driftEstimateHalfLifeMs} onInput={(value) => updateDirectIndicatorParameter("driftEstimateHalfLifeMs", value)} />
                  <DurationInput label="Drift forecast half-life" value={directIndicatorParameters().driftForecastHalfLifeMs} onInput={(value) => updateDirectIndicatorParameter("driftForecastHalfLifeMs", value)} />
                  <InspectorNumber label="Drift scale" value={directIndicatorParameters().driftScale} min={0} step={0.01} onInput={(value) => updateDirectIndicatorParameter("driftScale", value)} />
                  <DurationInput label="Variance estimate half-life" value={directIndicatorParameters().varianceEstimateHalfLifeMs} onInput={(value) => updateDirectIndicatorParameter("varianceEstimateHalfLifeMs", value)} />
                  <DurationInput label="Long-run variance half-life" value={directIndicatorParameters().longRunVarianceHalfLifeMs} onInput={(value) => updateDirectIndicatorParameter("longRunVarianceHalfLifeMs", value)} />
                  <DurationInput label="Variance forecast half-life" value={directIndicatorParameters().varianceForecastHalfLifeMs} onInput={(value) => updateDirectIndicatorParameter("varianceForecastHalfLifeMs", value)} />
                  <InspectorNumber label="Unresolved transition width (grid cells)" value={directIndicatorParameters().transitionWidthGridCells} min={0.1} step={0.1} onInput={(value) => updateDirectIndicatorParameter("transitionWidthGridCells", value)} />
                  <div class="flex items-end">
                    <button class="btn w-full" type="button" onClick={() => {
                      const preset = globalPredictorPreset();
                      setSelectedPredictorPresetId(preset?.id ?? "custom");
                      setDirectIndicatorParameters({
                        ...(preset?.parameters ?? DEFAULT_DIRECT_INDICATOR_PARAMETERS),
                      } as DirectIndicatorPredictorParameters);
                    }}>
                      <RefreshCw size={15} /> Reset direct model
                    </button>
                  </div>
                  <Show when={selectedPredictorPreset()}>
                    {(preset) => (
                      <div class="text-xs text-ink-400 md:col-span-2">
                        {preset().source} · fitted mean conditional decision regret {formatQuote(preset().loss, 6)}
                        <Show when={preset().diagnosticCrossEntropy !== undefined}>
                          {` · conditional CE ${formatQuote(preset().diagnosticCrossEntropy!, 5)}`}
                        </Show>
                      </div>
                    )}
                  </Show>
                  <div class="text-xs text-ink-400 md:col-span-2">
                    The six causal forecast constants are calibrated. H, T, fees, maintenance, constraints, and both exposure grids come directly from the oracle. The extra width is the documented grid convention for unresolved analytic kinks; support widths and sharpness remain fixed latent-design settings.
                  </div>
                </Show>
                <Show when={predictorModel() === "mlp"}>
                  <InspectorSelect
                    label="Trained MLP artifact"
                    value={mlpModelId()}
                    options={(catalog()?.mlpModels.length ?? 0) > 0
                      ? catalog()!.mlpModels.map((model) => ({
                          value: model.id,
                          label: `${model.label} · ${model.executionProvider.toUpperCase()}`,
                        }))
                      : [{ value: "", label: "No trained model · run npm run mlp:train" }]}
                    onInput={setMlpModelId}
                  />
                  <InspectorSelect
                    label="Oracle comparison timing"
                    value={mlpOracleAlignment()}
                    options={[
                      {
                        value: "model-target",
                        label: "Aligned target time · prediction t + delay vs oracle t",
                      },
                      {
                        value: "prediction-time",
                        label: "Same timestamp · prediction t vs oracle t",
                      },
                    ]}
                    onInput={(value) => setMlpOracleAlignment(value as VwKamaMlpOracleAlignment)}
                  />
                  <Show when={selectedMlpModel()} fallback={(
                    <div class="text-xs text-amber-300 md:col-span-2">
                      Build an oracle dataset and train an artifact before running the MLP predictor.
                    </div>
                  )}>
                    {(model) => (
                      <div class="text-xs text-ink-400 md:col-span-2">
                        {model().id} · verified {model().executionProvider.toUpperCase()}
                        {' '}· policy delay {formatDuration(model().predictionDelayMs)}
                        {' '}· {mlpOracleAlignment() === "model-target"
                          ? `aligned predictions are plotted ${formatDuration(model().predictionDelayMs)} earlier at their oracle target time`
                          : "same-timestamp diagnostic; model delay is not applied to the comparison"}
                        <Show when={model().training}>
                          {(training) => <>
                            {' '}· train / validation / test {training().trainExamples.toLocaleString()} / {training().validationExamples.toLocaleString()} / {training().testExamples.toLocaleString()}
                            {' '}· validation loss {formatQuote(training().bestValidationLoss, 5)} · test loss {formatQuote(training().testLoss, 5)}
                            <Show when={training().bestEpoch !== undefined}>
                              {' '}· best epoch {(training().bestEpoch ?? 0) + 1}
                            </Show>
                            <Show when={training().finalizedEarly}>{' '}· finalized early</Show>
                            <Show when={training().testMetrics}>
                              {(metrics) => <>
                                {' '}· test KL {formatQuote(metrics().klDivergence, 5)}
                                <Show when={metrics().klDivergenceStdDev !== undefined}>
                                  {' '}± {formatQuote(metrics().klDivergenceStdDev ?? 0, 5)}
                                </Show>
                                {' '}· base KL {formatQuote(metrics().baseKlDivergence, 5)}
                                {' '}· probability MSE {formatQuote(metrics().probabilityMse, 7)}
                                {' '}· excess entropy {formatQuote(metrics().excessEntropy, 5)}
                                {' '}· Temporal MI {formatQuote(metrics().temporalMutualInformation, 5)}
                                {' '}· Oracle MI {formatQuote(metrics().oracleMutualInformation, 5)}
                                <Show when={metrics().distanceImbalanceWeight}>
                                  {(value) => <>{' '}· mean persistent advice weight {formatQuote(value(), 5)}</>}
                                </Show>
                                <Show when={metrics().timeWeightEffectiveSampleRatio}>
                                  {(value) => <>{' '}· time-weight ESS {ratioPercent(value())}</>}
                                </Show>
                              </>}
                            </Show>
                            <Show when={training().bestValidationMetrics}>
                              {(metrics) => <>
                                {' '}· validation KL {formatQuote(metrics().klDivergence, 5)}
                                <Show when={metrics().klDivergenceStdDev !== undefined}>
                                  {' '}± {formatQuote(metrics().klDivergenceStdDev ?? 0, 5)}
                                </Show>
                                {' '}· validation base KL {formatQuote(metrics().baseKlDivergence, 5)}
                              </>}
                            </Show>
                            <Show when={training().curriculum}>
                              {(curriculum) => <>
                                {' '}· curriculum stage {curriculum().stage}
                                {' '}· parent {curriculum().parentKey}
                                {' '}· weight profile {curriculum().weightProfile}
                              </>}
                            </Show>
                            <Show when={training().teacherFitMetrics}>
                              {(metrics) => <>
                                {' '}· revised-fitter oracle CE {formatQuote(metrics().crossEntropy, 5)}
                                {' '}· revised-fitter oracle MSE {formatQuote(metrics().meanSquaredError, 7)}
                                {' '}· fitter convergence {ratioPercent(metrics().converged)}
                              </>}
                            </Show>
                            <Show when={training().lossWeights}>
                              {(weights) => <>
                                {' '}· weights {Object.entries(weights()).map(([name, value]) => `${name}=${formatQuote(value, 3)}`).join(', ')}
                              </>}
                            </Show>
                          </>}
                        </Show>
                      </div>
                    )}
                  </Show>
                </Show>
                <Show when={predictorModel() === "legacy"}>
                  <InspectorNumber label="Base temperature" value={parameters().strategyTemperature ?? 0.001} min={0.000001} step={0.0001} onInput={(value) => update("strategyTemperature", value)} />
                  <InspectorNumber label="Quadratic scale" value={parameters().strategyQuadraticScale ?? 0} step={0.1} onInput={(value) => update("strategyQuadraticScale", value)} />
                  <DurationInput label="Quadratic volatility window" value={parameters().strategyQuadraticVolatilityMs ?? 60 * 60_000} onInput={(value) => update("strategyQuadraticVolatilityMs", value)} />
                  <InspectorNumber label="Target-normal mixture" value={parameters().strategyNormalMixture ?? 0} min={0} max={1} step={0.05} onInput={(value) => update("strategyNormalMixture", value)} />
                  <InspectorNumber label="Target-normal sigma" value={parameters().strategyNormalSigma ?? 25} min={0.000001} step={1} onInput={(value) => update("strategyNormalSigma", value)} />
                  <InspectorSelect
                    label="Temperature scaling"
                    value={valueConfig().strategyVolatilityScaling ? "volatility" : "fixed"}
                    options={[
                      { value: "fixed", label: "Fixed temperature" },
                      { value: "volatility", label: "Scale by realized volatility" },
                    ]}
                    onInput={(value) => setValueConfig((current) => ({
                      ...current,
                      strategyVolatilityScaling: value === "volatility",
                    }))}
                  />
                  <div class="text-xs text-ink-400 md:col-span-2">
                    Legacy mode maps the current KAMA rate into a linear/quadratic exposure distribution, optionally mixed with a normal target centered on the VW-KAMA candidate exposure.
                  </div>
                </Show>
              </div>
            </div>
          </div>
            <Show when={predictorModel() === "legacy"}>
              <>
            <div class="mt-3 grid grid-cols-1 gap-3 md:grid-cols-2">
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
            <InspectorNumber label="KAMA signal friction fraction" value={parameters().signalFrictionFraction ?? 1} min={0} max={1} step={0.05} onInput={(value) => update("signalFrictionFraction", value)} />
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
              </>
            </Show>
          <details class="mt-3 rounded-2 border border-line bg-ink-800 p-3">
            <summary class="cursor-pointer text-sm font-semibold text-ink-200">Oracle and scoring controls</summary>
            <div class="mt-3 rounded-2 border border-line bg-ink-900/30 p-3">
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
            <div class="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2">
              <InspectorNumber label="Oracle friction (bps)" value={oracleFriction() * 10_000} min={0} step={1} onInput={(value) => setOracleFriction(value / 10_000)} />
              <Show when={predictorModel() === "legacy"}>
                <DurationInput label="Match window" value={matchWindowMs()} onInput={setMatchWindowMs} />
                <DurationInput label="Timing half-life" value={timingHalfLifeMs()} onInput={setTimingHalfLifeMs} />
                <InspectorNumber label="Warmup multiple" value={warmupMultiple()} min={1} step={0.25} onInput={setWarmupMultiple} />
              </Show>
              <InspectorNumber
                label="Value grid points"
                value={valueConfig().gridSize}
                min={3}
                max={1024}
                step={1}
                onInput={(value) => setValueConfig((current) => {
                  const gridSize = Math.round(value);
                  return {
                    ...current,
                    gridSize,
                    mutualInformationBins: Math.min(current.mutualInformationBins, gridSize, 32),
                  };
                })}
              />
              <InspectorSelect label="Holding-period source" value={valueConfig().holdingPeriodMode} options={[{ value: "fixed", label: "Fixed duration" }, { value: "oracle-half-average-trade", label: "Half average time between oracle trades" }]} onInput={(value) => setValueConfig((current) => ({ ...current, holdingPeriodMode: value as VwKamaValueDistillationConfig["holdingPeriodMode"] }))} />
              <DurationInput label={valueConfig().holdingPeriodMode === "fixed" ? "Holding period H" : "Fallback holding period H"} value={valueConfig().holdingPeriodMs} onInput={(value) => setValueConfig((current) => ({ ...current, holdingPeriodMs: value, valueHorizonMs: Math.max(current.valueHorizonMs, value) }))} />
              <InspectorSelect label="Value horizon" value={valueConfig().valueHorizonMode ?? "full-window"} options={[{ value: "full-window", label: "Full selected window" }, { value: "fixed", label: "Fixed duration" }]} onInput={(value) => setValueConfig((current) => ({ ...current, valueHorizonMode: value as "full-window" | "fixed" }))} />
              <Show when={valueConfig().valueHorizonMode === "fixed"}>
                <DurationInput label="Fixed value horizon T−t" value={valueConfig().valueHorizonMs} onInput={(value) => setValueConfig((current) => ({ ...current, valueHorizonMs: Math.max(value, current.holdingPeriodMs) }))} />
              </Show>
              <Show when={valueConfig().valueHorizonMode === "fixed"}>
                <InspectorSelect label="Horizon at window end" value={valueConfig().horizonEndMode} options={[{ value: "truncate", label: "Truncate at window end" }, { value: "extend", label: "Use future candles" }]} onInput={(value) => setValueConfig((current) => ({ ...current, horizonEndMode: value as VwKamaValueDistillationConfig["horizonEndMode"] }))} />
              </Show>
              <InspectorNumber label="Oracle temperature" value={valueConfig().oracleTemperature} min={0.000001} step={0.001} onInput={(value) => setValueConfig((current) => ({ ...current, oracleTemperature: value }))} />
              <InspectorNumber label="Excess-entropy λ" value={valueConfig().entropyGapLambda} min={0} step={0.01} onInput={(value) => setValueConfig((current) => ({ ...current, entropyGapLambda: value }))} />
              <InspectorNumber label="State MI λ" value={valueConfig().stateMutualInformationLambda} min={0} step={0.01} onInput={(value) => setValueConfig((current) => ({ ...current, stateMutualInformationLambda: value }))} />
              <InspectorNumber label="Oracle MI λ" value={valueConfig().oracleMutualInformationLambda} min={0} step={0.01} onInput={(value) => setValueConfig((current) => ({ ...current, oracleMutualInformationLambda: value }))} />
              <InspectorSelect label="Oracle MI estimator" value={valueConfig().oracleMutualInformationMode} options={[{ value: "approximate", label: "Approximate · Gaussian moments" }, { value: "precise", label: "Precise · soft categorical bins" }]} onInput={(value) => setValueConfig((current) => ({ ...current, oracleMutualInformationMode: value as VwKamaValueDistillationConfig["oracleMutualInformationMode"] }))} />
              <Show when={valueConfig().oracleMutualInformationMode === "precise"}>
                <InspectorNumber label="Mutual-information bins" value={valueConfig().mutualInformationBins} min={2} max={Math.min(32, valueConfig().gridSize)} step={1} onInput={(value) => setValueConfig((current) => ({ ...current, mutualInformationBins: Math.round(value) }))} />
              </Show>
              <InspectorNumber label="Minimum exposure" value={valueConfig().minExposure} max={-0.000001} step={0.1} onInput={(value) => setValueConfig((current) => ({ ...current, minExposure: value }))} />
              <InspectorNumber label="Maximum exposure" value={valueConfig().maxExposure} min={0.000001} step={0.1} onInput={(value) => setValueConfig((current) => ({ ...current, maxExposure: value }))} />
              <InspectorNumber label="Maximum effective exposure" value={valueConfig().maxEffectiveExposure} min={Math.max(Math.abs(valueConfig().minExposure), Math.abs(valueConfig().maxExposure))} step={1} onInput={(value) => setValueConfig((current) => ({ ...current, maxEffectiveExposure: value }))} />
              <InspectorNumber label="Initial exposure" value={valueConfig().initialExposure} min={-valueConfig().maxEffectiveExposure} max={valueConfig().maxEffectiveExposure} step={0.1} onInput={(value) => setValueConfig((current) => ({ ...current, initialExposure: value }))} />
              <InspectorNumber label="Quote lend maintenance / hour" value={valueConfig().quoteLendRate} min={0} step={0.000001} onInput={(value) => setValueConfig((current) => ({ ...current, quoteLendRate: value }))} />
              <InspectorNumber label="Quote borrow maintenance / hour" value={valueConfig().quoteBorrowRate} min={0} step={0.000001} onInput={(value) => setValueConfig((current) => ({ ...current, quoteBorrowRate: value }))} />
              <InspectorNumber label="Asset borrow maintenance / hour" value={valueConfig().assetBorrowRate} min={0} step={0.000001} onInput={(value) => setValueConfig((current) => ({ ...current, assetBorrowRate: value }))} />
            </div>
            <div class="mt-2 text-xs text-ink-400">
              The displayed oracle return is exp(Q₀(initial exposure))−1 for one coherent policy ending at the selected terminal candle, where exposure is forcibly closed to zero. Rolling targets default to T−t=1h and H=60s.
              <Show when={predictorModel() === "handcrafted"}>
                <> The handcrafted predictor estimates causal EWMA drift and variance, forecasts them over T, and runs a friction-aware Bellman recursion directly on the exposure grid. Its plotted prediction is raw forecast regret; local presets and per-candle fitting are explicitly hindsight diagnostics.</>
              </Show>
              <Show when={predictorModel() === "direct-indicator"}>
                <> The direct indicator model computes only F(a), solves its two no-trade boundaries with the oracle’s exact fee and maintenance semantics, decodes the 17 analytic parameters, and evaluates the conditional four-segment distribution without constructing or fitting a regret surface.</>
              </Show>
              <Show when={predictorModel() === "mlp"}>
                <> The MLP reads causal standardized OHLCV histories at 1s, 1m, 1h, 1d, 1M, and 3M plus execution state, then predicts six quadratic score coordinates and two exact survival cutoffs. Full-window metrics and paths use batched backend inference; unsampled hover timestamps use the throttled timestamp API.</>
              </Show>
              <Show when={predictorModel() === "legacy"}>
                <> The legacy predictor maps the causal KAMA rate into its configured linear/quadratic and target-normal exposure distribution.</>
              </Show>
              {" Accuracy metrics use at most 2,000 evenly spaced causal samples per continuous segment so long 1-second windows remain interactive. Loss terms with λ=0 are skipped."}
            </div>
          </details>
        </section>

        <Show when={analysisError()}>
          {(message) => <ErrorNotice message={message()} />}
        </Show>
        <Show when={predictionError()}>
          {(message) => <ErrorNotice message={message()} />}
        </Show>

        <Show when={result()} fallback={<div class="kama-inspector-chart"><EmptyResult loading={loading()} /></div>}>
          {(analysis) => (
            <>
              <section class="min-w-0 max-w-full rounded-2 border border-line bg-ink-900/60 shadow-sm">
                <div class="flex flex-wrap items-center justify-between gap-3 px-3 py-2">
                  <div class="min-w-0">
                    <div class="muted-label">Performance</div>
                    <div class="mt-0.5 flex items-center gap-1.5 text-[11px] text-ink-400">
                      <Info aria-hidden="true" size={12} />
                      VW-KAMA against the perfect-margin oracle · diagnostic accuracy, not trading profitability
                    </div>
                  </div>
                  <PerformanceMetric
                    label="Score"
                    value={ratioPercent(weightedScore(analysis().metrics))}
                    description={metricHelp.score}
                    prominent
                  />
                </div>
                <div class="grid border-t border-line xl:grid-cols-2">
                  <div class="min-w-0 p-2.5">
                    <div class="mb-1 px-1 text-[10px] font-semibold uppercase tracking-wider text-ink-500">Signal quality</div>
                    <div class="grid grid-cols-[repeat(auto-fit,minmax(7.25rem,1fr))] gap-x-1 gap-y-0.5">
                      <PerformanceMetric label="F1" value={ratioPercent(analysis().metrics.f1)} description={metricHelp.f1} />
                      <PerformanceMetric label="Precision" value={ratioPercent(analysis().metrics.precision)} description={metricHelp.precision} />
                      <PerformanceMetric label="Recall" value={ratioPercent(analysis().metrics.recall)} description={metricHelp.recall} />
                      <PerformanceMetric label="Agreement" value={ratioPercent(analysis().metrics.exposureAgreement)} description={metricHelp.agreement} />
                      <PerformanceMetric
                        label="Matched / total"
                        value={`${formatQuote(analysis().metrics.matchedCount, 0)} / ${formatQuote(analysis().metrics.oracleCount, 0)}`}
                        description={metricHelp.matched}
                      />
                      <PerformanceMetric label="Extra" value={formatQuote(analysis().metrics.extraSignalCount, 0)} description={metricHelp.extra} />
                      <PerformanceMetric label="Noise / matched" value={formatNoiseRatio(analysis().metrics.noiseSignalRatio)} description={metricHelp.noise} />
                      <PerformanceMetric label="Cleanliness" value={ratioPercent(analysis().metrics.signalCleanliness)} description={metricHelp.cleanliness} />
                      <PerformanceMetric label="Timing error P50" value={formatDuration(analysis().metrics.lagP50Ms ?? undefined)} description={metricHelp.timingError} />
                      <PerformanceMetric label="Signals/day" value={formatQuote(analysis().metrics.signalsPerDay, 1)} description={metricHelp.signals} />
                      <PerformanceMetric label="Candles" value={formatQuote(analysis().candleCount, 0)} description={metricHelp.candles} />
                    </div>
                  </div>
                  <Show when={analysis().metrics.valueDistillation}>
                    {(value) => (
                      <div class="min-w-0 border-t border-line p-2.5 xl:border-l xl:border-t-0">
                        <div class="mb-1 px-1 text-[10px] font-semibold uppercase tracking-wider text-ink-500">Forecast fit and return</div>
                        <div class="grid grid-cols-[repeat(auto-fit,minmax(7.25rem,1fr))] gap-x-1 gap-y-0.5">
                          <PerformanceMetric label="Forecast CE" value={formatQuote(value().crossEntropy, 5)} description={metricHelp.distillation} />
                          <PerformanceMetric label="Mean avg regret" value={formatQuote(value().meanAverageRegret, 6)} description={metricHelp.averageRegret} />
                          <PerformanceMetric label="Mixed loss" value={formatQuote(value().mixedLoss, 5)} description={metricHelp.mixedLoss} />
                          <PerformanceMetric label="Entropy gap" value={formatQuote(value().entropyGap, 5)} description={metricHelp.entropyGap} />
                          <PerformanceMetric label="State MI" value={ratioPercent(value().stateMutualInformation)} description={metricHelp.stateMutualInformation} />
                          <PerformanceMetric label={`Oracle MI · ${value().oracleMutualInformationMode}`} value={ratioPercent(value().oracleMutualInformation)} description={metricHelp.oracleMutualInformation} />
                          <PerformanceMetric label="exp(−KL)" value={ratioPercent(value().score)} description={metricHelp.distillation} />
                          <PerformanceMetric label="Mixed exp(−KL)" value={ratioPercent(value().mixedScore)} description={metricHelp.mixedLoss} />
                          <PerformanceMetric label="Resolved holding H" value={formatDuration(value().holdingPeriodMs)} description={metricHelp.valueHoldingPeriod} />
                          <PerformanceMetric
                            label="Value horizon T−t"
                            value={valueConfig().valueHorizonMode === "fixed"
                              ? formatDuration(value().valueHorizonMs)
                              : "Full selected window"}
                            description={metricHelp.valueHorizon}
                          />
                          <PerformanceMetric label="Forecast return" value={ratioPercent(value().returns.strategy.totalReturn)} description={metricHelp.strategyReturn} />
                          <PerformanceMetric label="Oracle exp(Q₀)−1" value={ratioPercent(value().returns.oracle.totalReturn)} description={metricHelp.oracleReturn} />
                          <PerformanceMetric label="Forecast drawdown" value={ratioPercent(value().returns.strategy.maxDrawdown)} description={metricHelp.drawdown} />
                        </div>
                      </div>
                    )}
                  </Show>
                </div>
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
                    <OverlayButton label={displayedPredictorModel() === "legacy" ? "Candidate signals" : "Forecast trades"} active={showSignals()} onClick={() => setShowSignals(!showSignals())} />
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
                  {displayedPredictorModel() === "legacy"
                    ? "Purple L / F / S marks the peak–valley candidate target state."
                    : "Purple L / F / S marks a trade to the selected forecast’s modal target cell; no trade is emitted while the mode remains in the current exposure cell."} Red and green L / S marks are retrospective oracle actions. Hover to inspect a timestamp, click to pin it across the charts, and double-click to unpin.
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
                    cursorTime={inspectedTime()}
                    selectedTime={selectedTime()}
                    highlightedAnnotation={matchedCandidate()}
                    stateBands={chartStateBands()}
                    onOracleHoverChange={setHoveredOracle}
                    onViewportChange={updateViewportFromChart}
                    onCursorTimeChange={setCursorTime}
                    onPinnedSelectionChange={(selection) => setSelectedTime(selection?.time)}
                    emptyLabel="No analyzed candles"
                  />
                </div>
                <Show when={alignedPredictionUnavailable() && selectedMlpModel()}>
                  {(model) => (
                    <div class="mt-3 rounded-2 border border-amber-400/20 bg-amber-400/5 p-3 text-xs text-amber-200">
                      This oracle timestamp has no aligned forecast yet: the selected model needs
                      {' '}{formatDuration(model().predictionDelayMs)} of later input. Switch comparison
                      timing to “Same timestamp” to inspect the unshifted diagnostic instead.
                    </div>
                  )}
                </Show>
                <Show when={selectedDistribution()}>
                  {(point) => (
                    <div class="mt-3 rounded-2 border border-line bg-ink-800/50 p-3">
                      <div class="mb-2 flex flex-wrap items-end justify-between gap-2">
                        <div>
                          <div class="muted-label">{selectedTime() !== undefined
                            ? "Pinned chart point"
                            : cursorTime() !== undefined
                              ? "Hovered chart point"
                              : "Latest chart point"}</div>
                        </div>
                        <div class="text-xs tabular-nums text-ink-300">
                          {formatDateTime(point().time)} · oracle mode {signedExposure(point().oracleModalExposure)} / path {signedExposure(point().oraclePathExposure)} · forecast target {signedExposure(point().predictor?.optimalExposure ?? distributionMode(point().values, "strategyProbability"))}
                          <Show when={point().predictor} fallback={<> · legacy rate {formatQuote(point().strategyRateBpsHour, 2)} bps/h · b₂ {formatQuote(point().strategyQuadraticCoefficient, 8)} · normal {ratioPercent(point().strategyNormalMixture)} / σ {formatQuote(point().strategyNormalSigma, 3)} · τ {formatQuote(point().strategyTemperature, 6)}</>}>
                            {(predictor) => predictor().model === "mlp"
                              ? <> · MLP {predictor().modelId} policy</>
                              : <> · {predictor().model} policy · μ {formatCoefficient(predictor().drift!)} · σ² {formatCoefficient(predictor().variance!)}</>}
                          </Show>
                          {` · conditional CE ${formatQuote(point().crossEntropy, 5)} · avg regret ${formatQuote(point().averageRegret, 6)}`}
                        </div>
                      </div>
                      <TransitionPolicyDiagnostics
                        point={point()}
                        request={resultRequest()}
                      />
                      <div class="mt-5 border-t border-line pt-4">
                        <h3 class="mb-2 text-sm font-semibold">Oracle and predicted exposure distributions</h3>
                        <ExposureDistributionChart point={point()} />
                        <div class="mt-2 text-xs text-ink-400">
                          These post-action target preferences set current exposure equal to target, so transition cost is zero. This chart retains the selected causal model; the optional per-candle hindsight fit applies to the transition diagnostics above. Hover the price chart to preview another rendered candle, click to pin it across charts, or double-click to resume following hover.
                        </div>
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
                        cursorTime={inspectedTime()}
                        onCursorTimeChange={setCursorTime}
                        onZoom={zoomViewport}
                        onPan={panViewport}
                      />
                    </div>
                  </div>
                </Show>
                <Show when={valueOraclePath().length > 0 || valueCandidatePath().length > 0}>
                  <div class="mt-3">
                    <div class="mb-2 flex flex-wrap items-end justify-between gap-2">
                      <div class="text-xs text-ink-400">
                        Full-window paths · oracle equity includes friction, maintenance, and terminal liquidation; candidate equity follows the evaluated forecast exposure with the same friction and maintenance semantics.
                      </div>
                      <div class="flex flex-wrap gap-1.5">
                        <ChartSeriesToggle label="Oracle exposure" color="#22d3ee" active={valuePathVisibility().oracleExposure} onClick={() => toggleValuePathSeries("oracleExposure")} />
                        <ChartSeriesToggle label="Oracle equity" color="#22c55e" active={valuePathVisibility().oracleEquity} onClick={() => toggleValuePathSeries("oracleEquity")} />
                        <ChartSeriesToggle label="Candidate exposure" color="#a78bfa" active={valuePathVisibility().candidateExposure} onClick={() => toggleValuePathSeries("candidateExposure")} />
                        <ChartSeriesToggle label="Candidate equity" color="#f5b84b" active={valuePathVisibility().candidateEquity} onClick={() => toggleValuePathSeries("candidateEquity")} />
                      </div>
                    </div>
                    <Show when={valuePathSeries().length > 0}>
                      <div style={{ height: `${Math.max(15, valuePathSeries().length * 12)}rem` }}>
                        <IndicatorChart
                          series={valuePathSeries()}
                          start={visibleTimeRange().start}
                          end={visibleTimeRange().end}
                          cursorTime={inspectedTime()}
                          onCursorTimeChange={setCursorTime}
                          onZoom={zoomViewport}
                          onPan={panViewport}
                        />
                      </div>
                    </Show>
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

type ExposureDistributionSeries = "oracle" | "oracleFit" | "prediction";

function ExposureDistributionChart(props: {
  point: VwKamaValueDistributionPoint;
}) {
  const [visibleSeries, setVisibleSeries] = createSignal<Record<ExposureDistributionSeries, boolean>>({
    oracle: true,
    oracleFit: true,
    prediction: true,
  });
  const toggleSeries = (series: ExposureDistributionSeries) => setVisibleSeries((visible) => ({
    ...visible,
    [series]: !visible[series],
  }));
  const exposures = createMemo(() => Float64Array.from(
    props.point.values,
    (value) => value.exposure,
  ));
  const oracleProbabilities = createMemo(() => Float64Array.from(
    props.point.values,
    (value) => value.oracleProbability,
  ));
  const oracleFit = createMemo(() => fitConditionalQuadraticPolicy(
    exposures(),
    oracleProbabilities(),
    Float64Array.of(0),
    {
      friction: 0,
      transitionLogScale: 0,
      objective: "probability-mse",
      initialLinearCoefficient: props.point.strategyLinearCoefficient,
      initialQuadraticCoefficient: props.point.strategyQuadraticCoefficient,
    },
  ));
  const fittedOracleProbabilities = createMemo(() => conditionalQuadraticExposureProbabilities(
    exposures(),
    0,
    oracleFit().linearCoefficient,
    oracleFit().quadraticCoefficient,
    0,
    0,
  ));
  const predictedProbabilities = createMemo(() => Float64Array.from(
    props.point.values,
    (value) => value.strategyProbability,
  ));
  const maximum = () => Math.max(
    Number.EPSILON,
    ...(visibleSeries().oracle ? oracleProbabilities() : []),
    ...(visibleSeries().oracleFit ? fittedOracleProbabilities() : []),
    ...(visibleSeries().prediction ? predictedProbabilities() : []),
  );
  const strategyMode = () => distributionMode(props.point.values, "strategyProbability");
  const predictionTarget = () => props.point.predictor?.optimalExposure
    ?? props.point.candidateExposure;
  const markerPosition = (exposure: number) => distributionMarkerPosition(
    props.point.values.map((value) => value.exposure),
    exposure,
  );
  return (
    <div>
      <div class="mb-2 flex flex-wrap gap-4 text-xs tabular-nums text-ink-300">
        <span><span class="mr-1 inline-block h-2.5 w-2.5 rounded-sm bg-cyan-400" />Oracle mode <strong class="text-cyan-200">{signedExposure(props.point.oracleModalExposure)}</strong> · path <strong class="text-cyan-100">{signedExposure(props.point.oraclePathExposure)}</strong> · mean {signedExposure(props.point.oracleMeanExposure)}</span>
        <span><span class="mr-1 inline-block h-2.5 w-2.5 rounded-sm bg-violet-400" />Forecast mode <strong class="text-violet-200">{signedExposure(strategyMode())}</strong> · optimum <strong class="text-violet-100">{signedExposure(predictionTarget())}</strong> · mean {signedExposure(props.point.strategyMeanExposure)}</span>
        <span>Oracle entropy {formatQuote(props.point.oracleEntropy, 4)}</span>
        <span>Post-action CE {formatQuote(props.point.postActionCrossEntropy, 5)}</span>
        <span class="text-cyan-200">Oracle probability-MSE fit {formatQuadraticFit(oracleFit())}</span>
        <Show when={props.point.predictor} fallback={<span class="text-violet-200">Legacy prediction exact b₁ {formatCoefficient(props.point.strategyLinearCoefficient)} · b₂ {formatCoefficient(props.point.strategyQuadraticCoefficient)}</span>}>
          {(predictor) => <span class="text-violet-200">{predictor().model === "mlp"
            ? `MLP ${predictor().modelId ?? "model"} → 6 quadratic parameters`
            : `${predictor().model === "direct-indicator" ? "Direct indicator → 6 quadratic parameters" : "Handcrafted forecast"} · μ ${formatCoefficient(predictor().drift!)} · σ² ${formatCoefficient(predictor().variance!)} · long σ² ${formatCoefficient(predictor().longRunVariance!)}`}</span>}
        </Show>
      </div>
      <div class="relative h-52 pl-11" role="img" aria-label={`Oracle mode ${signedExposure(props.point.oracleModalExposure)} and path ${signedExposure(props.point.oraclePathExposure)}; strategy mode ${signedExposure(strategyMode())} and target ${signedExposure(predictionTarget())}`}>
        <ProbabilityScaleLabels maximum={maximum()} />
        <div class="relative h-full overflow-hidden border-y border-line bg-ink-900/25">
          <svg
            class="absolute inset-0 h-full w-full"
            viewBox="0 0 1000 200"
            preserveAspectRatio="none"
            aria-hidden="true"
          >
            <DistributionGrid />
            <Show when={visibleSeries().oracle}>
              <DistributionSamples
                probabilities={oracleProbabilities()}
                exposures={exposures()}
                maximum={maximum()}
                color="#22d3ee"
              />
            </Show>
            <Show when={visibleSeries().oracleFit}>
              <DistributionCurve probabilities={fittedOracleProbabilities()} maximum={maximum()} color="#a5f3fc" lineWidth={2.25} opacity={0.74} />
            </Show>
            <Show when={visibleSeries().prediction}>
              <DistributionCurve probabilities={predictedProbabilities()} maximum={maximum()} color="#c4b5fd" lineWidth={2.25} opacity={0.74} />
            </Show>
          </svg>
          <div class="pointer-events-none absolute inset-0" aria-hidden="true">
            <Show when={visibleSeries().oracle}>
              <div
                class="absolute inset-y-0 -translate-x-px border-l-2 border-dashed border-cyan-100/90"
                style={{ left: markerPosition(props.point.oraclePathExposure) }}
              />
            </Show>
            <Show when={visibleSeries().prediction}>
              <div
                class="absolute inset-y-0 translate-x-px border-l-2 border-dashed border-violet-100/90"
                style={{ left: markerPosition(predictionTarget()) }}
              />
            </Show>
          </div>
        </div>
      </div>
      <div class="ml-11">
        <DistributionExposureAxis values={props.point.values} />
      </div>
      <div class="mt-1 flex flex-wrap items-center justify-center gap-1.5">
        <ChartSeriesToggle label="Oracle samples + path" color="#22d3ee" points active={visibleSeries().oracle} onClick={() => toggleSeries("oracle")} />
        <ChartSeriesToggle label="Oracle probability-MSE fit" color="#a5f3fc" active={visibleSeries().oracleFit} onClick={() => toggleSeries("oracleFit")} />
        <ChartSeriesToggle label="Forecast-regret prediction" color="#c4b5fd" active={visibleSeries().prediction} onClick={() => toggleSeries("prediction")} />
      </div>
    </div>
  );
}

type DistributionProbability = "oracleProbability" | "strategyProbability";

interface TransitionHeatmaps {
  oracle: Float64Array;
  strategy: Float64Array;
  residual: Float64Array;
}

interface HistoricalTransitionHeatmaps {
  regret: Float64Array;
  maximumReturnGap: Float64Array;
}

const EMPTY_PROBABILITIES = new Float64Array(0);
const EMPTY_MODE_INDICES = new Uint16Array(0);

type TransitionHeatmapPanel =
  | "oracle"
  | "prediction"
  | "difference"
  | "fit"
  | "fitDifference"
  | "averageRegret"
  | "averageReturnGap"
  | "averageReturnDifference";
type TransitionHeatmapExposureMode = "usable" | "possible" | "custom";
type TransitionDistributionSource = "oracle" | "prediction" | "fit";
interface TransitionHeatmapSelection {
  source: TransitionDistributionSource;
  stateIndex: number;
  targetIndex: number;
  currentExposure: number;
  targetExposure: number;
}
interface TransitionHoverSlices extends TransitionHeatmapSelection {
  row: Float64Array;
  column: Float64Array;
  columnExposures: Float64Array;
  pinned: boolean;
}
type TransitionHeatmapMarking =
  | "oracleMode"
  | "oracleMean"
  | "predictionMode"
  | "predictionMean"
  | "oraclePathRow"
  | "candidateRow"
  | "oppositeExposure";
type TransitionHeatmapMarkings = Record<TransitionHeatmapMarking, boolean>;

function TransitionPolicyDiagnostics(props: {
  point: VwKamaValueDistributionPoint;
  request?: VwKamaInspectorRequest;
}) {
  const [visibleHeatmaps, setVisibleHeatmaps] = createSignal<Record<TransitionHeatmapPanel, boolean>>({
    oracle: true,
    prediction: true,
    difference: true,
    fit: false,
    fitDifference: false,
    averageRegret: false,
    averageReturnGap: false,
    averageReturnDifference: false,
  });
  const toggleHeatmap = (panel: TransitionHeatmapPanel) => setVisibleHeatmaps((visible) => ({
    ...visible,
    [panel]: !visible[panel],
  }));
  const visibleHeatmapCount = () => Object.values(visibleHeatmaps())
    .filter(Boolean)
    .length;
  const historicalHeatmapsVisible = createMemo(() => visibleHeatmaps().averageRegret
    || visibleHeatmaps().averageReturnGap
    || visibleHeatmaps().averageReturnDifference);
  const [refineConditionalFit, setRefineConditionalFit] = createSignal(false);
  const [historicalLookbackMs, setHistoricalLookbackMs] = createSignal(24 * 60 * 60_000);
  const [historicalHoldingPeriodMs, setHistoricalHoldingPeriodMs] = createSignal(
    props.point.oracleHoldingPeriodMs,
  );
  const [historicalValueHorizonMs, setHistoricalValueHorizonMs] = createSignal(
    props.point.oracleValueHorizonMs,
  );
  const [historicalAverages, setHistoricalAverages] = createSignal<VwKamaHistoricalAveragesResponse>();
  const [historicalAveragesLoading, setHistoricalAveragesLoading] = createSignal(false);
  const [historicalAveragesError, setHistoricalAveragesError] = createSignal<string>();
  let historicalDurationSupport = "";
  createEffect(() => {
    const maximumHolding = props.point.oracleHoldingPeriodMs;
    const maximumHorizon = props.point.oracleValueHorizonMs;
    const key = `${maximumHolding}:${maximumHorizon}`;
    if (key === historicalDurationSupport) return;
    historicalDurationSupport = key;
    setHistoricalHoldingPeriodMs((value) => Math.min(value, maximumHolding));
    setHistoricalValueHorizonMs((value) => Math.max(
      Math.min(value, maximumHorizon),
      Math.min(historicalHoldingPeriodMs(), maximumHolding),
    ));
  });
  createEffect(() => {
    if (!historicalHeatmapsVisible()) {
      setHistoricalAverages();
      setHistoricalAveragesLoading(false);
      setHistoricalAveragesError();
      return;
    }
    const request = props.request;
    const time = props.point.time;
    const lookback = historicalLookbackMs();
    const holdingPeriod = historicalHoldingPeriodMs();
    const valueHorizon = historicalValueHorizonMs();
    if (!request) {
      setHistoricalAverages();
      setHistoricalAveragesLoading(false);
      return;
    }
    setHistoricalAverages();
    setHistoricalAveragesError();
    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      setHistoricalAveragesLoading(true);
      setHistoricalAveragesError();
      void fetch(`${apiBase}/api/kama-inspector/historical-averages`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          ...request,
          time,
          lookbackMs: lookback,
          holdingPeriodMs: holdingPeriod,
          valueHorizonMs: valueHorizon,
        }),
        signal: controller.signal,
      }).then(async (response) => {
        const payload = await response.json() as unknown;
        if (!response.ok) throw new Error(errorMessage(payload, "Historical averages failed"));
        if (!controller.signal.aborted) setHistoricalAverages(payload as VwKamaHistoricalAveragesResponse);
      }).catch((error) => {
        if (!controller.signal.aborted) {
          setHistoricalAverages();
          setHistoricalAveragesError(error instanceof Error ? error.message : "Historical averages failed");
        }
      }).finally(() => {
        if (!controller.signal.aborted) setHistoricalAveragesLoading(false);
      });
    }, 200);
    onCleanup(() => {
      window.clearTimeout(timer);
      controller.abort();
    });
  });
  const [heatmapExposureMode, setHeatmapExposureMode] = createSignal<TransitionHeatmapExposureMode>("usable");
  const [customTargetLower, setCustomTargetLower] = createSignal(props.point.values[0]!.exposure);
  const [customTargetUpper, setCustomTargetUpper] = createSignal(props.point.values.at(-1)!.exposure);
  const [customCurrentLower, setCustomCurrentLower] = createSignal(props.point.currentExposureMinimum);
  const [customCurrentUpper, setCustomCurrentUpper] = createSignal(props.point.currentExposureMaximum);
  let heatmapSupportKey = "";
  createEffect(() => {
    const targetLower = props.point.values[0]!.exposure;
    const targetUpper = props.point.values.at(-1)!.exposure;
    const key = `${targetLower}:${targetUpper}:${props.point.currentExposureMinimum}:${props.point.currentExposureMaximum}`;
    if (key === heatmapSupportKey) return;
    heatmapSupportKey = key;
    setCustomTargetLower(targetLower);
    setCustomTargetUpper(targetUpper);
    setCustomCurrentLower(props.point.currentExposureMinimum);
    setCustomCurrentUpper(props.point.currentExposureMaximum);
  });
  const [fitForecastAtCandle, setFitForecastAtCandle] = createSignal(false);
  const [perCandleFit, setPerCandleFit] = createSignal<VwKamaPredictorFitResponse>();
  const [perCandleFitLoading, setPerCandleFitLoading] = createSignal(false);
  const [perCandleFitError, setPerCandleFitError] = createSignal<string>();
  createEffect(() => {
    props.point.time;
    setFitForecastAtCandle(false);
    setPerCandleFit();
    setPerCandleFitError();
  });
  createEffect(() => {
    const enabled = fitForecastAtCandle();
    const request = props.request;
    const time = props.point.time;
    if (!enabled || !request || request.predictor?.model !== "handcrafted") {
      setPerCandleFitLoading(false);
      return;
    }
    const controller = new AbortController();
    setPerCandleFitLoading(true);
    setPerCandleFitError();
    void fetch(`${apiBase}/api/kama-inspector/predictor-fit`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ ...request, time }),
      signal: controller.signal,
    }).then(async (response) => {
      const payload = await response.json() as unknown;
      if (!response.ok) throw new Error(errorMessage(payload, "Per-candle forecast fit failed"));
      if (!controller.signal.aborted) setPerCandleFit(payload as VwKamaPredictorFitResponse);
    }).catch((error) => {
      if (!controller.signal.aborted) {
        setPerCandleFitError(error instanceof Error ? error.message : "Per-candle forecast fit failed");
      }
    }).finally(() => {
      if (!controller.signal.aborted) setPerCandleFitLoading(false);
    });
    onCleanup(() => controller.abort());
  });
  const activePoint = createMemo(() => fitForecastAtCandle() && perCandleFit()?.time === props.point.time
    ? perCandleFit()!.point
    : props.point);
  const [visibleHeatmapMarkings, setVisibleHeatmapMarkings] = createSignal<TransitionHeatmapMarkings>({
    oracleMode: false,
    oracleMean: false,
    predictionMode: false,
    predictionMean: false,
    oraclePathRow: true,
    candidateRow: true,
    oppositeExposure: false,
  });
  const [visibleConditionalSeries, setVisibleConditionalSeries] = createSignal<Record<
    ConditionalPolicySeries,
    boolean
  >>({
    oraclePath: false,
    oracleCandidate: false,
    oracleOppositeExposure: false,
    oraclePathFit: false,
    oracleCandidateFit: false,
    wholePolicyCandidateFit: false,
    prediction: false,
    predictionOppositeExposure: false,
    inspectedRow: true,
    inspectedColumn: true,
  });
  const toggleConditionalSeries = (series: ConditionalPolicySeries) => setVisibleConditionalSeries((visible) => ({
    ...visible,
    [series]: !visible[series],
  }));
  const toggleHeatmapMarking = (marking: TransitionHeatmapMarking) => setVisibleHeatmapMarkings((visible) => ({
    ...visible,
    [marking]: !visible[marking],
  }));
  const [hoveredHeatmapCell, setHoveredHeatmapCell] = createSignal<TransitionHeatmapSelection>();
  const [selectedHeatmapCells, setSelectedHeatmapCells] = createSignal<Partial<
    Record<TransitionDistributionSource, TransitionHeatmapSelection>
  >>({});
  const selectHeatmapCell = (selection: TransitionHeatmapSelection) => {
    setSelectedHeatmapCells((selected) => ({
      ...selected,
      [selection.source]: selection,
    }));
  };
  const deselectHeatmap = (source: TransitionDistributionSource) => {
    setSelectedHeatmapCells((selected) => {
      const next = { ...selected };
      delete next[source];
      return next;
    });
  };
  createEffect(() => {
    props.point.time;
    setHoveredHeatmapCell();
    setSelectedHeatmapCells({});
  });
  createEffect(() => {
    if (visibleHeatmaps().fit) return;
    setHoveredHeatmapCell((hovered) => hovered?.source === "fit" ? undefined : hovered);
    setSelectedHeatmapCells((selected) => {
      if (!selected.fit) return selected;
      const next = { ...selected };
      delete next.fit;
      return next;
    });
  });
  const fullTargetExposures = createMemo(() => activePoint().values.map((value) => value.exposure));
  const fullCurrentExposures = createMemo(() => currentExposureGrid(activePoint()));
  const targetIndices = createMemo(() => heatmapExposureMode() === "custom"
    ? boundedGridIndices(
        fullTargetExposures(),
        customTargetLower(),
        customTargetUpper(),
        5,
      )
    : Array.from({ length: fullTargetExposures().length }, (_, index) => index));
  const activeValues = createMemo(() => targetIndices().map((index) => activePoint().values[index]!));
  const targetExposures = createMemo(() => activeValues().map((value) => value.exposure));
  const exposureGrid = createMemo(() => Float64Array.from(targetExposures()));
  const currentExposures = createMemo(() => {
    const mode = heatmapExposureMode();
    const full = fullCurrentExposures();
    if (mode === "possible") return full;
    if (mode === "usable") return Float64Array.from(fullTargetExposures());
    const lower = customCurrentLower();
    const upper = customCurrentUpper();
    return Float64Array.from(boundedGridIndices(full, lower, upper, 3), (index) => full[index]!);
  });
  const targetExposureStep = createMemo(() => exposureGridStep(fullTargetExposures()));
  const currentExposureStep = createMemo(() => targetExposureStep());
  createEffect(() => {
    heatmapExposureMode();
    targetExposures()[0];
    targetExposures().at(-1);
    currentExposures()[0];
    currentExposures().at(-1);
    setHoveredHeatmapCell();
    setSelectedHeatmapCells({});
  });
  const matrices = createMemo(() => transitionHeatmaps(
    activePoint(),
    targetIndices(),
    currentExposures(),
    activePoint().oracleTemperature,
    activePoint().friction,
  ));
  const historicalMatrices = createMemo(() => historicalTransitionHeatmaps(
    historicalAverages(),
    targetIndices(),
    currentExposures(),
  ));
  const averageReturnDifference = createMemo(() => (historicalAverages()?.sampleCount ?? 0) > 0
    ? Float64Array.from(
        historicalMatrices().maximumReturnGap,
        (probability, index) => probability - matrices().oracle[index]!,
      )
    : new Float64Array(matrices().oracle.length));
  const averageReturnDifferenceScale = createMemo(() => robustAbsoluteScale(
    averageReturnDifference(),
  ));
  const fittedHeatmapNeeded = () => visibleHeatmaps().fit || visibleHeatmaps().fitDifference;
  const conditionalModelNeeded = () => fittedHeatmapNeeded()
    || visibleConditionalSeries().wholePolicyCandidateFit;
  const conditionalModelFit = createMemo(() => conditionalModelNeeded()
    ? fitConditionalFourSegmentPolicy(
        exposureGrid(),
        matrices().oracle,
        currentExposures(),
        {
          latentLower: props.point.currentExposureMinimum,
          latentUpper: props.point.currentExposureMaximum,
          visibleLower: targetExposures()[0]!,
          visibleUpper: targetExposures().at(-1)!,
          metricCurrentLower: heatmapExposureMode() === "custom"
            ? currentExposures()[0]!
            : targetExposures()[0]!,
          metricCurrentUpper: heatmapExposureMode() === "custom"
            ? currentExposures().at(-1)!
            : targetExposures().at(-1)!,
          friction: activePoint().friction,
          temperature: activePoint().oracleTemperature,
          refineProjectedFit: refineConditionalFit(),
        },
      )
    : undefined);
  const fittedPolicy = createMemo(() => {
    const fit = conditionalModelFit();
    return fittedHeatmapNeeded() && fit
      ? conditionalFourSegmentPolicyMatrix(
          exposureGrid(),
          currentExposures(),
          fit.parameters,
        )
      : EMPTY_PROBABILITIES;
  });
  const fittedPolicyResidual = createMemo(() => visibleHeatmaps().fitDifference
    ? Float64Array.from(
        fittedPolicy(),
        (probability, index) => probability - matrices().oracle[index]!,
      )
    : EMPTY_PROBABILITIES);
  const fittedInspection = createMemo(() => {
    const selected = selectedHeatmapCells().fit;
    const hovered = hoveredHeatmapCell();
    const selectedTargets = targetExposures();
    const selectedCurrents = currentExposures();
    return selected ?? (hovered?.source === "fit" ? hovered : undefined) ?? {
      currentExposure: clamp(
        props.point.candidateExposure,
        selectedCurrents[0]!,
        selectedCurrents.at(-1)!,
      ),
      targetExposure: clamp(
        props.point.candidateExposure,
        selectedTargets[0]!,
        selectedTargets.at(-1)!,
      ),
    };
  });
  const fittedInspectionParameters = createMemo(() => {
    const fit = conditionalModelFit();
    return fit
      ? conditionalFourSegmentParametersAt(fittedInspection().currentExposure, fit.parameters)
      : undefined;
  });
  const fittedInspectionSlopes = createMemo(() => {
    const fit = conditionalModelFit();
    if (!fit) return undefined;
    const inspection = fittedInspection();
    const parameters = fit.parameters;
    return {
      left: conditionalFourSegmentLogSlope(
        targetExposures()[0]!,
        inspection.currentExposure,
        parameters,
      ),
      selected: conditionalFourSegmentLogSlope(
        inspection.targetExposure,
        inspection.currentExposure,
        parameters,
      ),
      right: conditionalFourSegmentLogSlope(
        targetExposures().at(-1)!,
        inspection.currentExposure,
        parameters,
      ),
    };
  });
  const heatmapSlices = (selection: TransitionHeatmapSelection, pinned: boolean) => {
    const matrix = selection.source === "oracle"
      ? matrices().oracle
      : selection.source === "prediction"
        ? matrices().strategy
        : fittedPolicy();
    const targetSize = targetExposures().length;
    const stateSize = currentExposures().length;
    const row = matrix.slice(
      selection.stateIndex * targetSize,
      (selection.stateIndex + 1) * targetSize,
    );
    const column = new Float64Array(stateSize);
    for (let stateIndex = 0; stateIndex < stateSize; stateIndex += 1) {
      column[stateIndex] = matrix[stateIndex * targetSize + selection.targetIndex]!;
    }
    return { ...selection, row, column, columnExposures: currentExposures(), pinned };
  };
  const inspectedHeatmapSlices = createMemo<TransitionHoverSlices[]>(() => {
    const selected = selectedHeatmapCells();
    const slices = (["oracle", "prediction", "fit"] as const)
      .flatMap((source) => selected[source] ? [heatmapSlices(selected[source]!, true)] : []);
    const hovered = hoveredHeatmapCell();
    const duplicatesPinnedCell = hovered && selected[hovered.source]
      && selected[hovered.source]!.stateIndex === hovered.stateIndex
      && selected[hovered.source]!.targetIndex === hovered.targetIndex;
    if (hovered && !duplicatesPinnedCell) slices.push(heatmapSlices(hovered, false));
    return slices;
  });
  const probabilityScale = createMemo(() => robustPositiveScale(
    matrices().oracle,
    matrices().strategy,
    fittedHeatmapNeeded() ? fittedPolicy() : EMPTY_PROBABILITIES,
    visibleHeatmaps().averageRegret ? historicalMatrices().regret : EMPTY_PROBABILITIES,
    visibleHeatmaps().averageReturnGap
      ? historicalMatrices().maximumReturnGap
      : EMPTY_PROBABILITIES,
  ));
  const oracleModes = createMemo(() => visibleHeatmapMarkings().oracleMode
    ? transitionModeIndices(matrices().oracle, targetExposures(), currentExposures().length)
    : EMPTY_MODE_INDICES);
  const strategyModes = createMemo(() => visibleHeatmapMarkings().predictionMode
    ? transitionModeIndices(matrices().strategy, targetExposures(), currentExposures().length)
    : EMPTY_MODE_INDICES);
  const oracleMeans = createMemo(() => visibleHeatmapMarkings().oracleMean
    ? transitionRowMeans(matrices().oracle, targetExposures(), currentExposures().length)
    : EMPTY_PROBABILITIES);
  const strategyMeans = createMemo(() => visibleHeatmapMarkings().predictionMean
    ? transitionRowMeans(matrices().strategy, targetExposures(), currentExposures().length)
    : EMPTY_PROBABILITIES);
  const oracleOppositeExposureSlice = createMemo(() => visibleConditionalSeries().oracleOppositeExposure
    ? transitionAntiDiagonalSlice(matrices().oracle, targetExposures(), currentExposures())
    : EMPTY_PROBABILITIES);
  const strategyOppositeExposureSlice = createMemo(() => visibleConditionalSeries().predictionOppositeExposure
    ? transitionAntiDiagonalSlice(matrices().strategy, targetExposures(), currentExposures())
    : EMPTY_PROBABILITIES);
  const oraclePathSlice = createMemo(() => visibleConditionalSeries().oraclePath
    || visibleConditionalSeries().oraclePathFit
    ? conditionalExposureSlice(
        activeValues(),
        "oracleProbability",
        props.point.oraclePathExposure,
        props.point.friction,
        props.point.oracleTemperature,
        1,
      )
    : EMPTY_PROBABILITIES);
  const oracleCandidateSlice = createMemo(() => visibleConditionalSeries().oracleCandidate
    || visibleConditionalSeries().oracleCandidateFit
    ? conditionalExposureSlice(
        activeValues(),
        "oracleProbability",
        props.point.candidateExposure,
        props.point.friction,
        props.point.oracleTemperature,
        1,
      )
    : EMPTY_PROBABILITIES);
  const predictionTransitionScale = () => activePoint().frictionFraction
    / activePoint().strategyTemperature;
  const fitOptions = () => ({
    friction: activePoint().friction,
    transitionLogScale: predictionTransitionScale(),
    objective: "probability-mse" as const,
    initialLinearCoefficient: activePoint().strategyLinearCoefficient,
    initialQuadraticCoefficient: activePoint().strategyQuadraticCoefficient,
  });
  const oraclePathFit = createMemo(() => visibleConditionalSeries().oraclePathFit
    ? fitConditionalQuadraticPolicy(
        exposureGrid(),
        oraclePathSlice(),
        Float64Array.of(props.point.oraclePathExposure),
        fitOptions(),
      )
    : undefined);
  const oracleCandidateFit = createMemo(() => visibleConditionalSeries().oracleCandidateFit
    ? fitConditionalQuadraticPolicy(
        exposureGrid(),
        oracleCandidateSlice(),
        Float64Array.of(props.point.candidateExposure),
        fitOptions(),
      )
    : undefined);
  const oraclePathFitCurve = createMemo(() => {
    const fit = oraclePathFit();
    return fit
      ? fittedConditionalCurve(
          exposureGrid(),
          props.point.oraclePathExposure,
          fit,
          props.point.friction,
          predictionTransitionScale(),
        )
      : EMPTY_PROBABILITIES;
  });
  const oracleCandidateFitCurve = createMemo(() => {
    const fit = oracleCandidateFit();
    return fit
      ? fittedConditionalCurve(
          exposureGrid(),
          props.point.candidateExposure,
          fit,
          props.point.friction,
          predictionTransitionScale(),
        )
      : EMPTY_PROBABILITIES;
  });
  const wholePolicyCandidateCurve = createMemo(() => {
    const fit = conditionalModelFit();
    return visibleConditionalSeries().wholePolicyCandidateFit && fit
      ? conditionalFourSegmentExposureProbabilities(
          exposureGrid(),
          props.point.candidateExposure,
          fit.parameters,
        )
      : EMPTY_PROBABILITIES;
  });
  const exactPredictionCurve = createMemo(() => visibleConditionalSeries().prediction
    ? predictedConditionalExposureSlice(
        activePoint(),
        props.point.candidateExposure,
        targetIndices(),
      )
    : EMPTY_PROBABILITIES);
  return (
    <div class="mt-5 border-t border-line pt-4">
      <div class="mb-2 flex flex-wrap items-end justify-between gap-2">
        <div>
          <h4 class="text-sm font-semibold">Transition-aware policy</h4>
          <div class="text-xs text-ink-400">
            X-axis columns are target exposure a ({signedExposure(targetExposures()[0]!)} to {signedExposure(targetExposures().at(-1)!)}); y-axis rows are current exposure x ({signedExposure(currentExposures()[0]!)} to {signedExposure(currentExposures().at(-1)!)}). Probability rows are normalized inside the selected x-axis range; regret and return retain their raw values. Scoring and regret maxima still use the complete exposure grids.
          </div>
        </div>
        <div class="text-xs tabular-nums text-ink-300">
          uniform input-state prior · oracle policy mean {signedExposure(activePoint().oraclePolicyMeanExposure)} · prediction mean {signedExposure(activePoint().strategyPolicyMeanExposure)} · entropy {formatQuote(activePoint().oraclePolicyEntropy, 4)} / {formatQuote(activePoint().strategyPolicyEntropy, 4)}
        </div>
      </div>
      <div class="mb-3 rounded-xl border border-line bg-ink-900/40 p-3">
        <div class="grid gap-3 md:grid-cols-[minmax(0,18rem)_1fr] md:items-end">
          <InspectorSelect
            label="Heatmap exposure coverage"
            value={heatmapExposureMode()}
            options={[
              { value: "usable", label: "Usable exposures · configured min/max" },
              { value: "possible", label: "Possible exposures · full state range" },
              { value: "custom", label: "Custom · separate x/y ranges" },
            ]}
            onInput={(value) => setHeatmapExposureMode(value as TransitionHeatmapExposureMode)}
          />
          <div class="text-[11px] leading-relaxed tabular-nums text-ink-400">
            <Show
              when={heatmapExposureMode() === "custom"}
              fallback={heatmapExposureMode() === "usable"
                ? "Both axes use the executable minimum and maximum exposure range."
                : "The target axis remains executable; the current-exposure axis covers the full possible/effective range at the resolution configured for the usable window."}
            >
              X bounds select available target-grid cells. Y uses the same exposure resolution configured for the usable window; very narrow ranges expand around their midpoint so at least five target columns and three current rows remain visible for the fitted-model diagnostic.
            </Show>
            <div class="mt-1 text-ink-300">
              Showing x target [{signedExposure(targetExposures()[0]!)}, {signedExposure(targetExposures().at(-1)!)}] · y current [{signedExposure(currentExposures()[0]!)}, {signedExposure(currentExposures().at(-1)!)}]
            </div>
          </div>
        </div>
        <Show when={heatmapExposureMode() === "custom"}>
          <div class="mt-3 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
            <InspectorNumber
              label="X target minimum"
              value={customTargetLower()}
              min={fullTargetExposures()[0]!}
              max={customTargetUpper()}
              step={targetExposureStep()}
              onInput={(value) => setCustomTargetLower(Math.min(value, customTargetUpper()))}
            />
            <InspectorNumber
              label="X target maximum"
              value={customTargetUpper()}
              min={customTargetLower()}
              max={fullTargetExposures().at(-1)!}
              step={targetExposureStep()}
              onInput={(value) => setCustomTargetUpper(Math.max(value, customTargetLower()))}
            />
            <InspectorNumber
              label="Y current minimum"
              value={customCurrentLower()}
              min={fullCurrentExposures()[0]!}
              max={customCurrentUpper()}
              step={currentExposureStep()}
              onInput={(value) => setCustomCurrentLower(Math.min(value, customCurrentUpper()))}
            />
            <InspectorNumber
              label="Y current maximum"
              value={customCurrentUpper()}
              min={customCurrentLower()}
              max={fullCurrentExposures().at(-1)!}
              step={currentExposureStep()}
              onInput={(value) => setCustomCurrentUpper(Math.max(value, customCurrentLower()))}
            />
          </div>
        </Show>
        <Show when={historicalHeatmapsVisible()}>
          <div class="mt-3 grid gap-3 border-t border-line pt-3 sm:grid-cols-2 xl:grid-cols-3">
            <DurationInput
              label="Historical averages lookback"
              value={historicalLookbackMs()}
              onInput={setHistoricalLookbackMs}
            />
            <DurationInput
              label="Historical holding period H"
              value={historicalHoldingPeriodMs()}
              min={props.request?.intervalMs}
              max={Math.min(props.point.oracleHoldingPeriodMs, historicalValueHorizonMs())}
              onInput={(value) => setHistoricalHoldingPeriodMs(clamp(
                value,
                props.request?.intervalMs ?? 1,
                Math.min(props.point.oracleHoldingPeriodMs, historicalValueHorizonMs()),
              ))}
            />
            <DurationInput
              label="Historical value horizon T"
              value={historicalValueHorizonMs()}
              min={historicalHoldingPeriodMs()}
              max={props.point.oracleValueHorizonMs}
              onInput={(value) => setHistoricalValueHorizonMs(clamp(
                value,
                historicalHoldingPeriodMs(),
                props.point.oracleValueHorizonMs,
              ))}
            />
            <div class="rounded-lg border border-line bg-ink-900/55 px-3 py-2">
              <div class="muted-label">Past average oracle regret</div>
              <div class="mt-1 text-sm font-semibold tabular-nums text-amber-200">
                {historicalAveragesLoading()
                  ? "Computing…"
                  : (historicalAverages()?.sampleCount ?? 0) > 0
                    ? formatQuote(historicalAverages()!.averageRegret, 6)
                    : "Unavailable"}
              </div>
              <div class="mt-0.5 text-[10px] tabular-nums text-ink-500">
                {formatQuote(historicalAverages()?.sampleCount ?? 0, 0)} completed outcomes · {formatDuration(historicalLookbackMs())}
              </div>
            </div>
            <div class="rounded-lg border border-line bg-ink-900/55 px-3 py-2">
              <div class="muted-label">Past average maximum return</div>
              <div class="mt-1 text-sm font-semibold tabular-nums text-emerald-200">
                {historicalAveragesLoading()
                  ? "Computing…"
                  : (historicalAverages()?.sampleCount ?? 0) > 0
                    ? ratioPercent(historicalAverages()!.averageMaximumReturn)
                    : "Unavailable"}
              </div>
              <div class="mt-0.5 text-[10px] tabular-nums text-ink-500">
                {formatQuote(historicalAverages()?.sampleCount ?? 0, 0)} completed outcomes · {formatDuration(historicalLookbackMs())}
              </div>
            </div>
          </div>
          <Show when={historicalAveragesError()}>{(message) => (
            <div class="mt-2 text-[11px] text-loss">{message()}</div>
          )}</Show>
        </Show>
      </div>
      <div class="mb-2 flex flex-wrap items-center justify-center gap-1.5">
        <ChartSeriesToggle label="Oracle heatmap" color="#22d3ee" heatmap active={visibleHeatmaps().oracle} onClick={() => toggleHeatmap("oracle")} />
        <ChartSeriesToggle label="Prediction heatmap" color="#a78bfa" heatmap active={visibleHeatmaps().prediction} onClick={() => toggleHeatmap("prediction")} />
        <ChartSeriesToggle label="Difference heatmap" color="#fb7185" secondaryColor="#22d3ee" heatmap active={visibleHeatmaps().difference} onClick={() => toggleHeatmap("difference")} />
        <ChartSeriesToggle label="Fitted model heatmap" color="#fbbf24" heatmap active={visibleHeatmaps().fit} onClick={() => toggleHeatmap("fit")} />
        <ChartSeriesToggle label="Fit difference heatmap" color="#fb7185" secondaryColor="#fbbf24" heatmap active={visibleHeatmaps().fitDifference} onClick={() => toggleHeatmap("fitDifference")} />
        <ChartSeriesToggle label="Past average-regret distribution" color="#fbbf24" heatmap active={visibleHeatmaps().averageRegret} onClick={() => toggleHeatmap("averageRegret")} />
        <ChartSeriesToggle label="Past max-return-gap distribution" color="#34d399" heatmap active={visibleHeatmaps().averageReturnGap} onClick={() => toggleHeatmap("averageReturnGap")} />
        <ChartSeriesToggle label="Max-return-gap difference" color="#fb7185" secondaryColor="#34d399" heatmap active={visibleHeatmaps().averageReturnDifference} onClick={() => toggleHeatmap("averageReturnDifference")} />
        <ChartSeriesToggle label="Joint fit refinement" color="#fbbf24" active={refineConditionalFit()} onClick={() => setRefineConditionalFit((active) => !active)} />
        <Show when={props.point.predictor && props.request?.predictor?.model === "handcrafted"}>
          <ChartSeriesToggle label="Hindsight-fit this candle" color="#c4b5fd" active={fitForecastAtCandle()} onClick={() => setFitForecastAtCandle((active) => !active)} />
        </Show>
      </div>
      <Show when={historicalHeatmapsVisible()}>
        <div class="mb-2 text-center text-[10px] leading-relaxed text-ink-500">
          Fully causal at the selected candle: an L lookback uses origin candles in [t − T − L, t − T], so every included T-horizon outcome has completed by t. Historical H {formatDuration(historicalAverages()?.holdingPeriodMs ?? historicalHoldingPeriodMs())} and T {formatDuration(historicalAverages()?.valueHorizonMs ?? historicalValueHorizonMs())} are independent of, and capped by, the main oracle. Average regret uses the full-row maximum conditional log return minus the target log return. Max-return gap uses the cash-baselined post-action oracle maximum, before current-state rebalance cost, minus average conditional simple return per target. Both become p(a | x) ∝ exp(−cost / τ).
        </div>
      </Show>
      <Show when={fitForecastAtCandle()}>
        <div class="mb-2 text-center text-[11px] tabular-nums text-violet-200">
          <Show when={!perCandleFitLoading()} fallback={<>Searching six forecast parameters against this candle’s realized oracle…</>}>
            <Show when={perCandleFit()} fallback={<span class="text-loss">{perCandleFitError() ?? "No fitted result."}</span>}>
              {(fit) => <>Hindsight diagnostic only · conditional CE {formatQuote(fit().baselineCrossEntropy, 5)} → {formatQuote(fit().fittedCrossEntropy, 5)} · {fit().iterations} trials · {formatComputeTime(fit().elapsedMs)}</>}
            </Show>
          </Show>
        </div>
      </Show>
      <Show
        when={visibleHeatmapCount() > 0}
        fallback={<div class="rounded-xl border border-dashed border-line py-8 text-center text-xs text-ink-500">All heatmaps hidden.</div>}
      >
        <div
          class="grid gap-3"
          classList={{
            "lg:grid-cols-3": visibleHeatmapCount() >= 3,
            "lg:grid-cols-2": visibleHeatmapCount() === 2,
            "lg:grid-cols-1": visibleHeatmapCount() <= 1,
          }}
        >
          <Show when={visibleHeatmaps().oracle}>
            <TransitionHeatmapCanvas
              title="Oracle p(a | x)"
              matrix={matrices().oracle}
              targetExposures={targetExposures()}
              currentExposures={currentExposures()}
              scale={probabilityScale()}
              palette="cyan"
              oracleModes={oracleModes()}
              strategyModes={strategyModes()}
              oracleMeans={oracleMeans()}
              strategyMeans={strategyMeans()}
              oracleCurrentExposure={props.point.oraclePathExposure}
              candidateCurrentExposure={props.point.candidateExposure}
              markings={visibleHeatmapMarkings()}
              sliceSource="oracle"
              selectedCell={selectedHeatmapCells().oracle}
              onSliceHover={setHoveredHeatmapCell}
              onSliceSelect={selectHeatmapCell}
              onSliceDeselect={() => deselectHeatmap("oracle")}
            />
          </Show>
          <Show when={visibleHeatmaps().prediction}>
            <TransitionHeatmapCanvas
              title={fitForecastAtCandle() && perCandleFit()
                ? "Per-candle hindsight fit s(a | x)"
                : "Prediction s(a | x)"}
              matrix={matrices().strategy}
              targetExposures={targetExposures()}
              currentExposures={currentExposures()}
              scale={probabilityScale()}
              palette="violet"
              oracleModes={oracleModes()}
              strategyModes={strategyModes()}
              oracleMeans={oracleMeans()}
              strategyMeans={strategyMeans()}
              oracleCurrentExposure={props.point.oraclePathExposure}
              candidateCurrentExposure={props.point.candidateExposure}
              markings={visibleHeatmapMarkings()}
              sliceSource="prediction"
              selectedCell={selectedHeatmapCells().prediction}
              onSliceHover={setHoveredHeatmapCell}
              onSliceSelect={selectHeatmapCell}
              onSliceDeselect={() => deselectHeatmap("prediction")}
            />
          </Show>
          <Show when={visibleHeatmaps().difference}>
            <TransitionHeatmapCanvas
              title="Prediction − oracle probability"
              matrix={matrices().residual}
              targetExposures={targetExposures()}
              currentExposures={currentExposures()}
              scale={robustAbsoluteScale(matrices().residual)}
              palette="difference"
              oracleModes={oracleModes()}
              strategyModes={strategyModes()}
              oracleMeans={oracleMeans()}
              strategyMeans={strategyMeans()}
              oracleCurrentExposure={props.point.oraclePathExposure}
              candidateCurrentExposure={props.point.candidateExposure}
              markings={visibleHeatmapMarkings()}
            />
          </Show>
          <Show when={visibleHeatmaps().fit}>
            <TransitionHeatmapCanvas
              title="Fitted four-segment q(a | x)"
              matrix={fittedPolicy()}
              targetExposures={targetExposures()}
              currentExposures={currentExposures()}
              scale={probabilityScale()}
              palette="amber"
              oracleModes={oracleModes()}
              strategyModes={strategyModes()}
              oracleMeans={oracleMeans()}
              strategyMeans={strategyMeans()}
              oracleCurrentExposure={props.point.oraclePathExposure}
              candidateCurrentExposure={props.point.candidateExposure}
              markings={visibleHeatmapMarkings()}
              sliceSource="fit"
              selectedCell={selectedHeatmapCells().fit}
              onSliceHover={setHoveredHeatmapCell}
              onSliceSelect={selectHeatmapCell}
              onSliceDeselect={() => deselectHeatmap("fit")}
            />
          </Show>
          <Show when={visibleHeatmaps().fitDifference}>
            <TransitionHeatmapCanvas
              title="Fitted model − oracle probability"
              matrix={fittedPolicyResidual()}
              targetExposures={targetExposures()}
              currentExposures={currentExposures()}
              scale={robustAbsoluteScale(fittedPolicyResidual())}
              palette="difference"
              oracleModes={oracleModes()}
              strategyModes={strategyModes()}
              oracleMeans={oracleMeans()}
              strategyMeans={strategyMeans()}
              oracleCurrentExposure={props.point.oraclePathExposure}
              candidateCurrentExposure={props.point.candidateExposure}
              markings={visibleHeatmapMarkings()}
            />
          </Show>
          <Show when={visibleHeatmaps().averageRegret}>
            <TransitionHeatmapCanvas
              title={`Causal average-regret p(a | x) · ${historicalAverages()?.sampleCount ?? 0} outcomes`}
              matrix={historicalMatrices().regret}
              targetExposures={targetExposures()}
              currentExposures={currentExposures()}
              scale={probabilityScale()}
              palette="amber"
              oracleModes={oracleModes()}
              strategyModes={strategyModes()}
              oracleMeans={oracleMeans()}
              strategyMeans={strategyMeans()}
              oracleCurrentExposure={props.point.oraclePathExposure}
              candidateCurrentExposure={props.point.candidateExposure}
              markings={visibleHeatmapMarkings()}
            />
          </Show>
          <Show when={visibleHeatmaps().averageReturnGap}>
            <TransitionHeatmapCanvas
              title={`Causal max-return-gap p(a | x) · ${historicalAverages()?.sampleCount ?? 0} outcomes`}
              matrix={historicalMatrices().maximumReturnGap}
              targetExposures={targetExposures()}
              currentExposures={currentExposures()}
              scale={probabilityScale()}
              palette="emerald"
              oracleModes={oracleModes()}
              strategyModes={strategyModes()}
              oracleMeans={oracleMeans()}
              strategyMeans={strategyMeans()}
              oracleCurrentExposure={props.point.oraclePathExposure}
              candidateCurrentExposure={props.point.candidateExposure}
              markings={visibleHeatmapMarkings()}
            />
          </Show>
          <Show when={visibleHeatmaps().averageReturnDifference}>
            <TransitionHeatmapCanvas
              title="Past max-return-gap − current oracle probability"
              matrix={averageReturnDifference()}
              targetExposures={targetExposures()}
              currentExposures={currentExposures()}
              scale={averageReturnDifferenceScale()}
              palette="difference"
              oracleModes={oracleModes()}
              strategyModes={strategyModes()}
              oracleMeans={oracleMeans()}
              strategyMeans={strategyMeans()}
              oracleCurrentExposure={props.point.oraclePathExposure}
              candidateCurrentExposure={props.point.candidateExposure}
              markings={visibleHeatmapMarkings()}
            />
          </Show>
        </div>
      </Show>
      <div class="mt-2 flex flex-wrap items-center justify-center gap-x-4 gap-y-1 text-[10px] uppercase tracking-wider text-ink-500">
        <span><span class="mr-1 inline-block h-2.5 w-2.5 rounded-sm bg-cyan-400" />oracle probability</span>
        <span><span class="mr-1 inline-block h-2.5 w-2.5 rounded-sm bg-violet-400" />predicted probability</span>
        <span><span class="mr-1 inline-block h-2.5 w-2.5 rounded-sm bg-amber-400" />fitted four-segment probability</span>
        <span><span class="mr-1 inline-block h-2.5 w-2.5 rounded-sm bg-amber-400" />past average-regret distribution</span>
        <span><span class="mr-1 inline-block h-2.5 w-2.5 rounded-sm bg-emerald-400" />past max-return-gap distribution</span>
        <span><span class="mr-1 inline-block h-2.5 w-2.5 rounded-sm bg-rose-400" />difference below oracle</span>
        <span>historical maps use only T-horizon outcomes completed by this candle</span>
      </div>
      <div class="mt-2 flex flex-wrap items-center justify-center gap-1.5">
        <span class="mr-1 text-[10px] font-medium uppercase tracking-wider text-ink-500">Heatmap markings</span>
        <ChartSeriesToggle label="Oracle row mode" color="#67e8f9" points active={visibleHeatmapMarkings().oracleMode} onClick={() => toggleHeatmapMarking("oracleMode")} />
        <ChartSeriesToggle label="Oracle row mean" color="#67e8f9" points active={visibleHeatmapMarkings().oracleMean} onClick={() => toggleHeatmapMarking("oracleMean")} />
        <ChartSeriesToggle label="Prediction row mode" color="#c4b5fd" points active={visibleHeatmapMarkings().predictionMode} onClick={() => toggleHeatmapMarking("predictionMode")} />
        <ChartSeriesToggle label="Prediction row mean" color="#c4b5fd" points active={visibleHeatmapMarkings().predictionMean} onClick={() => toggleHeatmapMarking("predictionMean")} />
        <ChartSeriesToggle label={`Oracle path row ${signedExposure(props.point.oraclePathExposure)}`} color="#67e8f9" active={visibleHeatmapMarkings().oraclePathRow} onClick={() => toggleHeatmapMarking("oraclePathRow")} />
        <ChartSeriesToggle label={`Candidate row ${signedExposure(props.point.candidateExposure)}`} color="#6ee7b7" dashed active={visibleHeatmapMarkings().candidateRow} onClick={() => toggleHeatmapMarking("candidateRow")} />
        <ChartSeriesToggle label="x + a = 0 trace" color="#fbbf24" dashed active={visibleHeatmapMarkings().oppositeExposure} onClick={() => toggleHeatmapMarking("oppositeExposure")} />
      </div>
      <div class="mt-4">
        <div class="mb-2 text-xs font-medium text-ink-300">Conditional slices across exposure</div>
        <div class="mb-2 text-[11px] text-ink-400">
          The two local quadratic curves minimize pointwise probability MSE. The global fitted heatmap initializes the documented order-independent three-transition surface by direct score fitting. Joint fit refinement optionally continues against truncated conditional cross-entropy over the whole oracle map; disable it to inspect the projected parameters directly. The per-candle forecast fit is a deterministic 97-trial search inside the calibration bounds: a good result demonstrates an attainable hindsight fit, while a miss does not prove that no parameter set exists. Hover an oracle, prediction, or fitted-model cell to add its row and column here. Click once to pin one point independently on each heatmap; double-click a heatmap to release only its point. The x + a = 0 and inspected-column cross-sections keep raw conditional cell probabilities and are not renormalized.
        </div>
        <div class="mb-2 grid gap-1 text-[11px] tabular-nums text-ink-300 md:grid-cols-2 xl:grid-cols-4">
          <Show when={oraclePathFit()}>{(fit) => (
            <span class="text-cyan-200">Oracle @ path · {formatQuadraticFit(fit())}</span>
          )}</Show>
          <Show when={oracleCandidateFit()}>{(fit) => (
            <span class="text-emerald-200">Oracle @ candidate · {formatQuadraticFit(fit())}</span>
          )}</Show>
          <Show when={conditionalModelFit()}>{(fit) => (
            <span class="text-amber-200">Four-segment {refineConditionalFit() ? "projected + joint fit" : "projected fit"} · {formatConditionalFit(fit())}</span>
          )}</Show>
          <span class="text-violet-200">{activePoint().predictor
            ? activePoint().predictor?.model === "direct-indicator"
              ? "Direct causal 6-parameter quadratic conditional policy"
              : activePoint().predictor?.model === "mlp"
                ? `MLP ${activePoint().predictor?.modelId ?? "model"} conditional policy`
              : fitForecastAtCandle() && perCandleFit() ? "Handcrafted per-candle hindsight fit" : "Handcrafted causal forecast-regret policy"
            : "Legacy exact-mixture prediction"}</span>
        </div>
        <Show when={conditionalModelFit()}>{(fit) => {
          const inspectionParameters = () => fittedInspectionParameters()!;
          const inspectionSlopes = () => fittedInspectionSlopes()!;
          return (
            <div class="mb-3 rounded-lg border border-amber-400/20 bg-amber-400/5 px-3 py-2 text-[10px] leading-relaxed tabular-nums text-ink-300">
              <div class="font-medium uppercase tracking-wider text-amber-200">Fitted conditional-distribution parameters</div>
              <div>{formatConditionalGlobalParameters(fit())}</div>
              <div class="text-amber-100">At inspected fitted row x {signedExposure(fittedInspection().currentExposure)} · {formatConditionalSliceParameters(inspectionParameters())}</div>
              <div class="text-amber-100">Effective ∂a log q · V− {formatCoefficient(inspectionSlopes().left)} · a {signedExposure(fittedInspection().targetExposure)} {formatCoefficient(inspectionSlopes().selected)} · V+ {formatCoefficient(inspectionSlopes().right)}</div>
            </div>
          );
        }}</Show>
        <Show when={activePoint().predictor?.directMetadata}>
          {(metadata) => (
            <div class="mb-3 rounded-lg border border-violet-400/20 bg-violet-400/5 px-3 py-2 text-[10px] leading-relaxed tabular-nums text-ink-300">
              <div class="font-medium uppercase tracking-wider text-violet-200">Direct decoder · computed 6 quadratic parameters</div>
              <div>{formatDirectParameterVector(metadata().parameterVector6)}</div>
              <div class="text-violet-100">F secants [{metadata().backgroundSecantSlopes.map(formatCoefficient).join(", ")}] · q sell/buy [{formatCoefficient(metadata().effectiveSellCostSlope)}, {formatCoefficient(metadata().effectiveBuyCostSlope)}] · widths [{metadata().transitionWidths10To90.map(formatCoefficient).join(", ")}] ({metadata().transitionWidthSources.join(" / ")})</div>
            </div>
          )}
        </Show>
        <ConditionalPolicyCurveChart
          values={activeValues()}
          visibleSeries={visibleConditionalSeries()}
          oraclePath={oraclePathSlice()}
          oracleCandidate={oracleCandidateSlice()}
          oraclePathFit={oraclePathFitCurve()}
          oracleCandidateFit={oracleCandidateFitCurve()}
          wholePolicyCandidateFit={wholePolicyCandidateCurve()}
          prediction={exactPredictionCurve()}
          oracleOppositeExposure={oracleOppositeExposureSlice()}
          predictionOppositeExposure={strategyOppositeExposureSlice()}
          inspectedSlices={inspectedHeatmapSlices()}
          onToggleSeries={toggleConditionalSeries}
          onReleasePinnedSlice={deselectHeatmap}
          onReleasePinnedSlices={() => setSelectedHeatmapCells({})}
        />
      </div>
    </div>
  );
}

function TransitionHeatmapCanvas(props: {
  title: string;
  matrix: Float64Array;
  targetExposures: number[];
  currentExposures: Float64Array;
  scale: number;
  palette: "cyan" | "violet" | "amber" | "emerald" | "difference";
  oracleModes: Uint16Array;
  strategyModes: Uint16Array;
  oracleMeans: Float64Array;
  strategyMeans: Float64Array;
  oracleCurrentExposure: number;
  candidateCurrentExposure: number;
  markings: TransitionHeatmapMarkings;
  sliceSource?: TransitionDistributionSource;
  selectedCell?: TransitionHeatmapSelection;
  onSliceHover?: (selection?: TransitionHeatmapSelection) => void;
  onSliceSelect?: (selection: TransitionHeatmapSelection) => void;
  onSliceDeselect?: () => void;
}) {
  let canvas!: HTMLCanvasElement;
  const [hovered, setHovered] = createSignal<{
    current: number;
    target: number;
    value: number;
    oracleMode?: number;
    strategyMode?: number;
    oracleMean?: number;
    strategyMean?: number;
  }>();
  createEffect(() => {
    const selectedCell = props.selectedCell;
    const selectedHere = selectedCell?.source === props.sliceSource ? selectedCell : undefined;
    drawTransitionHeatmap(
      canvas,
      props.matrix,
      props.scale,
      props.palette,
      props.targetExposures,
      props.currentExposures,
      props.oracleModes,
      props.strategyModes,
      props.oracleMeans,
      props.strategyMeans,
      props.oracleCurrentExposure,
      props.candidateCurrentExposure,
      props.markings,
      selectedHere?.stateIndex,
      selectedHere?.targetIndex,
    );
  });
  const cellAt = (event: MouseEvent) => {
    const bounds = canvas.getBoundingClientRect();
    const targetSize = props.targetExposures.length;
    const stateSize = props.currentExposures.length;
    const targetIndex = Math.max(0, Math.min(
      targetSize - 1,
      Math.floor((event.clientX - bounds.left) / bounds.width * targetSize),
    ));
    const displayRow = Math.max(0, Math.min(
      stateSize - 1,
      Math.floor((event.clientY - bounds.top) / bounds.height * stateSize),
    ));
    const stateIndex = stateSize - 1 - displayRow;
    return {
      stateIndex,
      targetIndex,
      current: props.currentExposures[stateIndex]!,
      target: props.targetExposures[targetIndex]!,
      value: props.matrix[stateIndex * targetSize + targetIndex]!,
      oracleMode: props.markings.oracleMode
        ? props.targetExposures[props.oracleModes[stateIndex]!]
        : undefined,
      strategyMode: props.markings.predictionMode
        ? props.targetExposures[props.strategyModes[stateIndex]!]
        : undefined,
      oracleMean: props.markings.oracleMean ? props.oracleMeans[stateIndex] : undefined,
      strategyMean: props.markings.predictionMean ? props.strategyMeans[stateIndex] : undefined,
    };
  };
  const sliceSelection = (cell: ReturnType<typeof cellAt>): TransitionHeatmapSelection | undefined =>
    props.sliceSource
      ? {
          source: props.sliceSource,
          stateIndex: cell.stateIndex,
          targetIndex: cell.targetIndex,
          currentExposure: cell.current,
          targetExposure: cell.target,
        }
      : undefined;
  const inspect = (event: MouseEvent) => {
    const cell = cellAt(event);
    setHovered({
      current: cell.current,
      target: cell.target,
      value: cell.value,
      oracleMode: cell.oracleMode,
      strategyMode: cell.strategyMode,
      oracleMean: cell.oracleMean,
      strategyMean: cell.strategyMean,
    });
    props.onSliceHover?.(sliceSelection(cell));
  };
  return (
    <div class="rounded-xl border border-line bg-ink-900/55 p-2">
      <div class="mb-1 flex items-center justify-between gap-2 text-xs">
        <span class="font-medium text-ink-200">{props.title}</span>
        <span class="tabular-nums text-ink-500">
          {props.palette === "difference" ? "±" : "color max "}{ratioPercent(props.scale)}
        </span>
      </div>
      <canvas
        ref={canvas}
        width={props.targetExposures.length * 4}
        height={props.currentExposures.length * 4}
        class="w-full cursor-crosshair border border-line [image-rendering:pixelated]"
        style={{ "aspect-ratio": `${props.targetExposures.length} / ${props.currentExposures.length}` }}
        aria-label={`${props.title}; horizontal target exposure, vertical current exposure`}
        onMouseMove={inspect}
        onMouseLeave={() => {
          setHovered();
          props.onSliceHover?.();
        }}
        onClick={(event) => {
          const selection = sliceSelection(cellAt(event));
          if (selection) props.onSliceSelect?.(selection);
        }}
        onDblClick={(event) => {
          event.preventDefault();
          props.onSliceDeselect?.();
        }}
      />
      <div class="mt-1 flex justify-between text-[10px] text-ink-500">
        <span>current + / target −</span><span>target exposure →</span><span>current + / target +</span>
      </div>
      <div class="min-h-4 text-center text-[10px] tabular-nums text-ink-300">
        <Show when={hovered()}>{(cell) => (
          <>
            x {signedExposure(cell().current)} → a {signedExposure(cell().target)} · {props.palette === "difference" ? "Δp " : "p "}{ratioPercent(cell().value)}
            <Show when={cell().oracleMode !== undefined}> · mode <span class="text-cyan-300">oracle {signedExposure(cell().oracleMode!)}</span></Show>
            <Show when={cell().strategyMode !== undefined}> · mode <span class="text-violet-300">prediction {signedExposure(cell().strategyMode!)}</span></Show>
            <Show when={cell().oracleMean !== undefined}> · mean <span class="text-cyan-300">oracle {signedExposure(cell().oracleMean!)}</span></Show>
            <Show when={cell().strategyMean !== undefined}> · mean <span class="text-violet-300">prediction {signedExposure(cell().strategyMean!)}</span></Show>
          </>
        )}</Show>
      </div>
    </div>
  );
}

function fittedConditionalCurve(
  grid: Float64Array,
  currentExposure: number,
  fit: ConditionalQuadraticPolicyFit,
  friction: number,
  transitionLogScale: number,
): Float64Array<ArrayBufferLike> {
  return conditionalQuadraticExposureProbabilities(
    grid,
    currentExposure,
    fit.linearCoefficient,
    fit.quadraticCoefficient,
    friction,
    transitionLogScale,
  );
}

type ConditionalPolicySeries =
  | "oraclePath"
  | "oracleCandidate"
  | "oracleOppositeExposure"
  | "oraclePathFit"
  | "oracleCandidateFit"
  | "wholePolicyCandidateFit"
  | "prediction"
  | "predictionOppositeExposure"
  | "inspectedRow"
  | "inspectedColumn";

function ConditionalPolicyCurveChart(props: {
  values: VwKamaValueDistributionPoint["values"];
  visibleSeries: Record<ConditionalPolicySeries, boolean>;
  oraclePath: Float64Array;
  oracleCandidate: Float64Array;
  oraclePathFit: Float64Array<ArrayBufferLike>;
  oracleCandidateFit: Float64Array<ArrayBufferLike>;
  wholePolicyCandidateFit: Float64Array<ArrayBufferLike>;
  prediction: Float64Array<ArrayBufferLike>;
  oracleOppositeExposure: Float64Array;
  predictionOppositeExposure: Float64Array;
  inspectedSlices: TransitionHoverSlices[];
  onToggleSeries: (series: ConditionalPolicySeries) => void;
  onReleasePinnedSlice: (source: TransitionDistributionSource) => void;
  onReleasePinnedSlices: () => void;
}) {
  const visibleSeries = () => props.visibleSeries;
  const toggleSeries = (series: ConditionalPolicySeries) => props.onToggleSeries(series);
  const maximum = () => Math.max(
    Number.EPSILON,
    ...(visibleSeries().oraclePath ? props.oraclePath : []),
    ...(visibleSeries().oracleCandidate ? props.oracleCandidate : []),
    ...(visibleSeries().oraclePathFit ? props.oraclePathFit : []),
    ...(visibleSeries().oracleCandidateFit ? props.oracleCandidateFit : []),
    ...(visibleSeries().wholePolicyCandidateFit ? props.wholePolicyCandidateFit : []),
    ...(visibleSeries().prediction ? props.prediction : []),
    ...(visibleSeries().oracleOppositeExposure ? props.oracleOppositeExposure : []),
    ...(visibleSeries().predictionOppositeExposure ? props.predictionOppositeExposure : []),
    ...(visibleSeries().inspectedRow
      ? props.inspectedSlices.flatMap((slices) => Array.from(slices.row))
      : []),
    ...(visibleSeries().inspectedColumn
      ? props.inspectedSlices.flatMap((slices) => Array.from(slices.column))
      : []),
  );
  const hasPinnedSlices = () => props.inspectedSlices.some((slices) => slices.pinned);
  return (
    <div>
      <Show when={props.inspectedSlices.length > 0}>
        <div class="mb-2 flex flex-wrap items-center justify-between gap-2 rounded-lg border border-line bg-ink-900/45 px-2.5 py-1.5 text-[11px] tabular-nums text-ink-300">
          <div class="flex flex-wrap items-center gap-x-3 gap-y-1">
            <For each={props.inspectedSlices}>{(slices) => (
              <span>
                <strong style={{ color: transitionInspectionColor(slices.source) }}>
                  {slices.pinned ? "Pinned" : "Hovered"} {transitionInspectionLabel(slices.source)}
                </strong>
                {" · "}x {signedExposure(slices.currentExposure)} · a {signedExposure(slices.targetExposure)}
                <Show when={slices.pinned}>
                  <button
                    type="button"
                    class="ml-1 text-ink-500 transition hover:text-ink-100"
                    aria-label={`Release ${transitionInspectionLabel(slices.source)} selection`}
                    onClick={() => props.onReleasePinnedSlice(slices.source)}
                  >×</button>
                </Show>
              </span>
            )}</For>
            <span class="text-ink-400">solid rows are normalized p(target | x); dashed columns are raw p(a | current), clipped to the action-axis range</span>
          </div>
          <Show when={hasPinnedSlices()}>
            <button
              type="button"
              class="rounded-full border border-line px-2 py-1 text-[10px] font-medium uppercase tracking-wider text-ink-300 transition hover:border-ink-300 hover:text-ink-100"
              onClick={props.onReleasePinnedSlices}
            >
              Clear pinned
            </button>
          </Show>
        </div>
      </Show>
      <div class="relative h-52 pl-11">
        <ProbabilityScaleLabels maximum={maximum()} />
        <div class="relative h-full overflow-hidden border-y border-line bg-ink-900/25">
          <svg
            class="absolute inset-0 h-full w-full"
            viewBox="0 0 1000 200"
            preserveAspectRatio="none"
            role="img"
            aria-label="Conditional oracle samples, local quadratic fits, fitted four-segment policy, exact predicted policy, x plus a equals zero cross-sections, and inspected heatmap row and column slices"
          >
            <DistributionGrid />
            <Show when={visibleSeries().oraclePath}>
              <DistributionSamples
                probabilities={props.oraclePath}
                exposures={props.values.map((value) => value.exposure)}
                maximum={maximum()}
                color="#22d3ee"
              />
            </Show>
            <Show when={visibleSeries().oracleCandidate}>
              <DistributionSamples
                probabilities={props.oracleCandidate}
                exposures={props.values.map((value) => value.exposure)}
                maximum={maximum()}
                color="#34d399"
              />
            </Show>
            <Show when={visibleSeries().oracleOppositeExposure}>
              <DistributionSamples
                probabilities={props.oracleOppositeExposure}
                exposures={props.values.map((value) => value.exposure)}
                maximum={maximum()}
                color="#fb923c"
              />
            </Show>
            <Show when={visibleSeries().oraclePathFit}>
              <DistributionCurve probabilities={props.oraclePathFit} maximum={maximum()} color="#a5f3fc" lineWidth={2.25} opacity={0.72} />
            </Show>
            <Show when={visibleSeries().oracleCandidateFit}>
              <DistributionCurve probabilities={props.oracleCandidateFit} maximum={maximum()} color="#6ee7b7" lineWidth={2.25} opacity={0.72} />
            </Show>
            <Show when={visibleSeries().wholePolicyCandidateFit}>
              <DistributionCurve probabilities={props.wholePolicyCandidateFit} maximum={maximum()} color="#fcd34d" lineWidth={2.25} opacity={0.76} dashed />
            </Show>
            <Show when={visibleSeries().prediction}>
              <DistributionCurve probabilities={props.prediction} maximum={maximum()} color="#c4b5fd" lineWidth={2.5} opacity={0.74} />
            </Show>
            <Show when={visibleSeries().predictionOppositeExposure}>
              <DistributionCurve probabilities={props.predictionOppositeExposure} maximum={maximum()} color="#f472b6" lineWidth={2.25} opacity={0.78} dashed />
            </Show>
            <For each={visibleSeries().inspectedRow ? props.inspectedSlices : []}>{(slices) => (
              <DistributionCurve
                probabilities={slices.row}
                maximum={maximum()}
                color={transitionInspectionColor(slices.source)}
                lineWidth={slices.pinned ? 3 : 2.25}
                opacity={slices.pinned ? 0.96 : 0.7}
              />
            )}</For>
            <For each={visibleSeries().inspectedColumn ? props.inspectedSlices : []}>{(slices) => (
              <DistributionCurve
                probabilities={slices.column}
                exposures={slices.columnExposures}
                minimumExposure={props.values[0]!.exposure}
                maximumExposure={props.values.at(-1)!.exposure}
                maximum={maximum()}
                color={transitionInspectionColor(slices.source)}
                lineWidth={slices.pinned ? 2.75 : 2}
                opacity={slices.pinned ? 0.9 : 0.62}
                dashed
              />
            )}</For>
          </svg>
        </div>
      </div>
      <div class="ml-11">
        <DistributionExposureAxis values={props.values} />
      </div>
      <div class="mt-1 flex flex-wrap items-center justify-center gap-1.5">
        <ChartSeriesToggle label="Oracle @ path" color="#22d3ee" points active={visibleSeries().oraclePath} onClick={() => toggleSeries("oraclePath")} />
        <ChartSeriesToggle label="Path fit" color="#a5f3fc" active={visibleSeries().oraclePathFit} onClick={() => toggleSeries("oraclePathFit")} />
        <ChartSeriesToggle label="Oracle @ candidate" color="#34d399" points active={visibleSeries().oracleCandidate} onClick={() => toggleSeries("oracleCandidate")} />
        <ChartSeriesToggle label="Candidate fit" color="#6ee7b7" active={visibleSeries().oracleCandidateFit} onClick={() => toggleSeries("oracleCandidateFit")} />
        <ChartSeriesToggle label="Four-segment fit @ candidate" color="#fcd34d" dashed active={visibleSeries().wholePolicyCandidateFit} onClick={() => toggleSeries("wholePolicyCandidateFit")} />
        <ChartSeriesToggle label="Forecast-regret prediction" color="#c4b5fd" active={visibleSeries().prediction} onClick={() => toggleSeries("prediction")} />
        <ChartSeriesToggle label="Oracle x + a = 0" color="#fb923c" points active={visibleSeries().oracleOppositeExposure} onClick={() => toggleSeries("oracleOppositeExposure")} />
        <ChartSeriesToggle label="Prediction x + a = 0" color="#f472b6" dashed active={visibleSeries().predictionOppositeExposure} onClick={() => toggleSeries("predictionOppositeExposure")} />
        <Show when={props.inspectedSlices.length > 0}>
          <>
            <ChartSeriesToggle label="Inspected rows" color="#f8fafc" active={visibleSeries().inspectedRow} onClick={() => toggleSeries("inspectedRow")} />
            <ChartSeriesToggle label="Inspected columns" color="#f8fafc" dashed active={visibleSeries().inspectedColumn} onClick={() => toggleSeries("inspectedColumn")} />
          </>
        </Show>
      </div>
    </div>
  );
}

function transitionInspectionColor(source: TransitionDistributionSource): string {
  return source === "oracle" ? "#67e8f9" : source === "prediction" ? "#c4b5fd" : "#fbbf24";
}

function transitionInspectionLabel(source: TransitionDistributionSource): string {
  return source === "oracle" ? "oracle" : source === "prediction" ? "prediction" : "fitted model";
}

function ChartSeriesToggle(props: {
  label: string;
  color: string;
  secondaryColor?: string;
  active: boolean;
  dashed?: boolean;
  heatmap?: boolean;
  points?: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      class="inline-flex min-h-7 items-center gap-1.5 rounded-full border px-2.5 py-1 text-[10px] font-medium uppercase tracking-wider transition hover:border-ink-300 hover:text-ink-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/45"
      classList={{
        "border-ink-400 bg-ink-800 text-ink-200": props.active,
        "border-line bg-ink-900/40 text-ink-500 opacity-50": !props.active,
      }}
      aria-pressed={props.active}
      onClick={props.onClick}
    >
      <Show
        when={props.heatmap}
        fallback={(
          <span class="relative inline-block h-2 w-5 shrink-0" aria-hidden="true">
            <span
              class="absolute inset-x-0 top-1/2 border-t-2"
              classList={{ "border-dashed": props.dashed }}
              style={{ "border-color": props.color }}
            />
            <Show when={props.points}>
              <span
                class="absolute left-1/2 top-1/2 h-1.5 w-1.5 -translate-x-1/2 -translate-y-1/2 rounded-full"
                style={{ "background-color": props.color }}
              />
            </Show>
          </span>
        )}
      >
        <span
          class="inline-block h-3 w-3 shrink-0 rounded-sm border border-white/20"
          style={{
            background: props.secondaryColor
              ? `linear-gradient(135deg, ${props.color}, ${props.secondaryColor})`
              : props.color,
          }}
          aria-hidden="true"
        />
      </Show>
      <span>{props.label}</span>
    </button>
  );
}

function DistributionCurve(props: {
  probabilities: ArrayLike<number>;
  exposures?: ArrayLike<number>;
  minimumExposure?: number;
  maximumExposure?: number;
  maximum: number;
  color: string;
  lineWidth?: number;
  opacity?: number;
  dashed?: boolean;
}) {
  return (
    <path
      d={distributionCurvePath(
        props.probabilities,
        props.maximum,
        1_000,
        200,
        props.exposures,
        props.minimumExposure,
        props.maximumExposure,
      )}
      vector-effect="non-scaling-stroke"
      style={{
        fill: "none",
        stroke: props.color,
        "stroke-width": `${props.lineWidth ?? 2}px`,
        "stroke-dasharray": props.dashed ? "7 5" : "none",
        opacity: `${props.opacity ?? 1}`,
      }}
    />
  );
}

function DistributionSamples(props: {
  probabilities: ArrayLike<number>;
  exposures: ArrayLike<number>;
  maximum: number;
  color: string;
}) {
  const points = () => Array.from(
    { length: props.probabilities.length },
    (_, index) => ({
      x: (index + 0.5) / props.probabilities.length * 1_000,
      y: distributionCurveY(props.probabilities[index]!, props.maximum),
      exposure: props.exposures[index]!,
      probability: props.probabilities[index]!,
    }),
  );
  return (
    <g>
      <DistributionCurve
        probabilities={props.probabilities}
        maximum={props.maximum}
        color={props.color}
        lineWidth={1}
        opacity={0.38}
      />
      <For each={points()}>{(point) => (
        <circle
          cx={point.x}
          cy={point.y}
          r="1.6"
          vector-effect="non-scaling-stroke"
          style={{ fill: props.color, opacity: "0.58" }}
        >
          <title>{`Exposure ${signedExposure(point.exposure)} · probability ${ratioPercent(point.probability)}`}</title>
        </circle>
      )}</For>
    </g>
  );
}

function DistributionGrid() {
  return (
    <g aria-hidden="true">
      <For each={[5, 100, 195]}>{(y) => (
        <line
          x1="0"
          x2="1000"
          y1={y}
          y2={y}
          vector-effect="non-scaling-stroke"
          style={{ stroke: "rgba(100, 116, 139, 0.28)", "stroke-width": "1px" }}
        />
      )}</For>
    </g>
  );
}

function ProbabilityScaleLabels(props: { maximum: number }) {
  return (
    <div class="pointer-events-none absolute inset-y-0 left-0 w-10 text-right text-[10px] tabular-nums text-ink-500" aria-hidden="true">
      <span class="absolute right-1 top-0 -translate-y-1/2">{ratioPercent(props.maximum)}</span>
      <span class="absolute right-1 top-1/2 -translate-y-1/2">{ratioPercent(props.maximum / 2)}</span>
      <span class="absolute bottom-0 right-1 translate-y-1/2">0%</span>
    </div>
  );
}

function distributionCurvePath(
  probabilities: ArrayLike<number>,
  maximum: number,
  width = 1_000,
  height = 200,
  exposures?: ArrayLike<number>,
  minimumExposure?: number,
  maximumExposure?: number,
): string {
  if (probabilities.length === 0) return "";
  const verticalPadding = 5;
  return Array.from({ length: probabilities.length }, (_, index) => {
    const exposureSpan = (maximumExposure ?? 0) - (minimumExposure ?? 0);
    const x = exposures && exposureSpan > 0
      ? (exposures[index]! - minimumExposure!) / exposureSpan * width
      : (index + 0.5) / probabilities.length * width;
    const y = distributionCurveY(probabilities[index]!, maximum, height, verticalPadding);
    return `${index === 0 ? "M" : "L"}${x.toFixed(3)},${y.toFixed(3)}`;
  }).join(" ");
}

function distributionCurveY(
  probability: number,
  maximum: number,
  height = 200,
  verticalPadding = 5,
): number {
  return height - verticalPadding
    - Math.max(0, probability) / Math.max(Number.EPSILON, maximum)
      * (height - verticalPadding * 2);
}

function formatCoefficient(value: number): string {
  const magnitude = Math.abs(value);
  if (magnitude === 0) return "0";
  return magnitude >= 1e-4 && magnitude < 1e4
    ? formatQuote(value, 7)
    : value.toExponential(4);
}

function formatQuadraticFit(fit: ConditionalQuadraticPolicyFit): string {
  return `${fit.converged ? "converged" : "not converged"} · b₁ ${formatCoefficient(fit.linearCoefficient)}`
    + ` · b₂ ${formatCoefficient(fit.quadraticCoefficient)}`
    + ` · MSE ${formatCoefficient(fit.meanSquaredError)}`
    + ` · KL(p∥q) ${formatQuote(fit.klDivergence, 6)}`;
}

function formatConditionalFit(fit: ConditionalFourSegmentPolicyFit): string {
  return `${fit.converged ? "converged" : "stopped"} by ${fit.termination}`
    + ` · ${fit.restarts} starts · ${fit.iterations} best-start iterations`
    + ` · visible CE ${formatQuote(fit.crossEntropy, 6)}`
    + ` · visible KL(p∥q) ${formatQuote(fit.klDivergence, 6)}`
    + ` · visible MSE ${formatCoefficient(fit.meanSquaredError)}`;
}

function formatConditionalGlobalParameters(fit: ConditionalFourSegmentPolicyFit): string {
  const parameters = fit.parameters;
  return `latent [${signedExposure(parameters.latentLower)}, ${signedExposure(parameters.latentUpper)}]`
    + ` · visible [${signedExposure(parameters.visibleLower)}, ${signedExposure(parameters.visibleUpper)}]`
    + ` · survival [${signedExposure(parameters.cutoffLower)}, ${signedExposure(parameters.cutoffUpper)}]`
    + ` · c₁ ${signedExposure(parameters.c1)} · c₂ ${signedExposure(parameters.c2)}`
    + ` · b ${formatCoefficient(parameters.baseSlope)}`
    + ` · λ ${formatCoefficient(parameters.quadraticPrecision)}`
    + ` · βc₁ ${formatCoefficient(parameters.betaC1)}`
    + ` · βx(fee) ${formatCoefficient(parameters.betaX)}`
    + ` · βc₂ ${formatCoefficient(parameters.betaC2)}`
    + ` · κ [${formatCoefficient(parameters.kappaC1)}, ${formatCoefficient(parameters.kappaX)}, ${formatCoefficient(parameters.kappaC2)}]`;
}

function formatDirectParameterVector(parameters: DirectIndicatorParameterVector6): string {
  return `c₁ ${signedExposure(parameters.c1)} · c₂ ${signedExposure(parameters.c2)}`
    + ` · b ${formatCoefficient(parameters.b)}`
    + ` · λ ${formatCoefficient(parameters.lambda)}`
    + ` · βc₁ ${formatCoefficient(parameters.betaC1)}`
    + ` · βc₂ ${formatCoefficient(parameters.betaC2)}`;
}

function formatConditionalSliceParameters(parameters: ConditionalFourSegmentSliceParameters): string {
  return `b ${formatCoefficient(parameters.baseSlope)}`
    + ` · λ ${formatCoefficient(parameters.quadraticPrecision)}`
    + ` · β [${formatCoefficient(parameters.betaC1)}, ${formatCoefficient(parameters.betaX)}, ${formatCoefficient(parameters.betaC2)}]`
    + ` · κ [${formatCoefficient(parameters.kappaC1)}, ${formatCoefficient(parameters.kappaX)}, ${formatCoefficient(parameters.kappaC2)}]`
    + ` · ordered slope offsets [${parameters.segmentSlopeOffsets.map(formatCoefficient).join(", ")}]`;
}

function DistributionExposureAxis(props: { values: VwKamaValueDistributionPoint["values"] }) {
  return (
    <div class="grid gap-0.5 px-1 pt-1 text-center text-[10px] tabular-nums text-ink-400"
      style={{ "grid-template-columns": `repeat(${props.values.length}, minmax(0, 1fr))` }}>
      <For each={props.values}>{(value, index) => (
        <span classList={{ invisible: index() % Math.max(1, Math.ceil(props.values.length / 9)) !== 0 && index() !== props.values.length - 1 }}>
          {formatQuote(value.exposure, 1)}
        </span>
      )}</For>
    </div>
  );
}

function transitionHeatmaps(
  point: VwKamaValueDistributionPoint,
  targetIndices: readonly number[],
  currentGrid: Float64Array,
  oracleTemperature: number,
  friction: number,
): TransitionHeatmaps {
  const targetSize = targetIndices.length;
  const oracle = new Float64Array(currentGrid.length * targetSize);
  const strategy = new Float64Array(currentGrid.length * targetSize);
  const residual = new Float64Array(currentGrid.length * targetSize);
  for (let stateIndex = 0; stateIndex < currentGrid.length; stateIndex += 1) {
    const current = currentGrid[stateIndex]!;
    const oracleRow = conditionalExposureSlice(
      point.values,
      "oracleProbability",
      current,
      friction,
      oracleTemperature,
      1,
    );
    const strategyRow = predictedConditionalExposureSlice(point, current);
    const visibleOracleRow = selectedNormalizedProbabilities(oracleRow, targetIndices);
    const visibleStrategyRow = selectedNormalizedProbabilities(strategyRow, targetIndices);
    for (let targetIndex = 0; targetIndex < targetSize; targetIndex += 1) {
      const offset = stateIndex * targetSize + targetIndex;
      oracle[offset] = visibleOracleRow[targetIndex]!;
      strategy[offset] = visibleStrategyRow[targetIndex]!;
      residual[offset] = visibleStrategyRow[targetIndex]! - visibleOracleRow[targetIndex]!;
    }
  }
  return { oracle, strategy, residual };
}

function historicalTransitionHeatmaps(
  response: VwKamaHistoricalAveragesResponse | undefined,
  targetIndices: readonly number[],
  currentGrid: Float64Array,
): HistoricalTransitionHeatmaps {
  const targetSize = targetIndices.length;
  const cellCount = currentGrid.length * targetSize;
  const result = {
    regret: new Float64Array(cellCount),
    maximumReturnGap: new Float64Array(cellCount),
  };
  if (!response || targetSize === 0 || currentGrid.length === 0) return result;
  const fullTargetSize = response.targetExposures.length;
  for (let stateIndex = 0; stateIndex < currentGrid.length; stateIndex += 1) {
    const sourceStateIndex = nearestExposureIndex(response.currentExposures, currentGrid[stateIndex]!);
    const sourceOffset = sourceStateIndex * fullTargetSize;
    const selectedRegret = selectedNormalizedProbabilities(
      response.regretProbabilities.slice(sourceOffset, sourceOffset + fullTargetSize),
      targetIndices,
    );
    const selectedReturnGap = selectedNormalizedProbabilities(
      response.maximumReturnGapProbabilities.slice(sourceOffset, sourceOffset + fullTargetSize),
      targetIndices,
    );
    const targetOffset = stateIndex * targetSize;
    result.regret.set(selectedRegret, targetOffset);
    result.maximumReturnGap.set(selectedReturnGap, targetOffset);
  }
  return result;
}

function boundedGridIndices(
  grid: ArrayLike<number>,
  lower: number,
  upper: number,
  minimumCount: number,
): number[] {
  if (grid.length === 0) return [];
  const minimum = Math.min(lower, upper);
  const maximum = Math.max(lower, upper);
  const selected = Array.from({ length: grid.length }, (_, index) => index)
    .filter((index) => grid[index]! >= minimum && grid[index]! <= maximum);
  const requiredCount = Math.min(Math.max(1, minimumCount), grid.length);
  if (selected.length >= requiredCount) return selected;

  const midpoint = (minimum + maximum) / 2;
  let nearestIndex = 0;
  for (let index = 1; index < grid.length; index += 1) {
    if (Math.abs(grid[index]! - midpoint) < Math.abs(grid[nearestIndex]! - midpoint)) {
      nearestIndex = index;
    }
  }
  const start = clamp(
    nearestIndex - Math.floor(requiredCount / 2),
    0,
    grid.length - requiredCount,
  );
  return Array.from({ length: requiredCount }, (_, index) => start + index);
}

function exposureGridStep(grid: ArrayLike<number>): number {
  if (grid.length < 2) return 0.1;
  return Math.max(Number.EPSILON, Math.abs(grid[grid.length - 1]! - grid[0]!) / (grid.length - 1));
}

function selectedNormalizedProbabilities(
  probabilities: ArrayLike<number>,
  indices: readonly number[],
): Float64Array {
  const selected = Float64Array.from(indices, (index) => probabilities[index] ?? 0);
  const total = selected.reduce((sum, probability) => sum + probability, 0);
  if (!(total > Number.EPSILON)) {
    selected.fill(1 / Math.max(1, selected.length));
    return selected;
  }
  for (let index = 0; index < selected.length; index += 1) selected[index] /= total;
  return selected;
}

function currentExposureGrid(point: VwKamaValueDistributionPoint): Float64Array {
  const targetExposures = point.values.map((value) => value.exposure);
  const spacing = exposureGridStep(targetExposures);
  const lower = point.currentExposureMinimum;
  const upper = point.currentExposureMaximum;
  const crossesZero = lower < 0 && upper > 0;
  const negativeIntervals = crossesZero ? Math.max(1, Math.round(-lower / spacing)) : 0;
  const positiveIntervals = crossesZero ? Math.max(1, Math.round(upper / spacing)) : 0;
  const intervalCount = crossesZero
    ? negativeIntervals + positiveIntervals
    : Math.max(2, Math.round((upper - lower) / spacing));
  const result = new Float64Array(intervalCount + 1);
  for (let index = 0; index < result.length; index += 1) {
    if (crossesZero && index <= negativeIntervals) {
      result[index] = lower + index / negativeIntervals * -lower;
    } else if (crossesZero) {
      result[index] = (index - negativeIntervals) / positiveIntervals * upper;
    } else {
      result[index] = lower + index / intervalCount * (upper - lower);
    }
  }
  return result;
}

function conditionalExposureSlice(
  values: VwKamaValueDistributionPoint["values"],
  probability: DistributionProbability,
  currentExposure: number,
  friction: number,
  temperature: number,
  frictionScale: number,
): Float64Array {
  return conditionalExposureProbabilities(
    values.map((value) => value[probability]),
    values.map((value) => value.exposure),
    currentExposure,
    friction,
    frictionScale / temperature,
  );
}

function predictedConditionalExposureSlice(
  point: VwKamaValueDistributionPoint,
  currentExposure: number,
  targetIndices?: readonly number[],
): Float64Array {
  const directParameters = point.predictor?.conditionalParameters;
  const probabilities = directParameters
    ? conditionalFourSegmentExposureProbabilities(
      point.values.map((value) => value.exposure),
      currentExposure,
      directParameters,
    )
    : conditionalExposureSlice(
      point.values,
      "strategyProbability",
      currentExposure,
      point.friction,
      point.strategyTemperature,
      point.frictionFraction,
    );
  return targetIndices ? selectedNormalizedProbabilities(probabilities, targetIndices) : probabilities;
}

function robustAbsoluteScale(matrix: Float64Array): number {
  const magnitudes = Array.from(matrix, (value) => Number.isFinite(value) ? Math.abs(value) : 0)
    .filter((value) => value > 0)
    .sort((left, right) => left - right);
  if (magnitudes.length === 0) return 1;
  return Math.max(Number.EPSILON, magnitudes[Math.floor((magnitudes.length - 1) * 0.95)]!);
}

function robustPositiveScale(...matrices: Float64Array[]): number {
  const values = matrices.flatMap((matrix) => Array.from(
    matrix,
    (value) => Number.isFinite(value) && value > 0 ? value : 0,
  )).filter((value) => value > 0).sort((left, right) => left - right);
  if (values.length === 0) return 1;
  return Math.max(Number.EPSILON, values[Math.floor((values.length - 1) * 0.99)]!);
}

function transitionModeIndices(
  matrix: Float64Array,
  exposures: number[],
  stateSize: number,
): Uint16Array {
  const targetSize = exposures.length;
  const result = new Uint16Array(stateSize);
  for (let stateIndex = 0; stateIndex < stateSize; stateIndex += 1) {
    let bestIndex = 0;
    let bestValue = Number.NEGATIVE_INFINITY;
    for (let targetIndex = 0; targetIndex < targetSize; targetIndex += 1) {
      const value = matrix[stateIndex * targetSize + targetIndex]!;
      if (value > bestValue || (value === bestValue
        && Math.abs(exposures[targetIndex]!) < Math.abs(exposures[bestIndex]!))) {
        bestValue = value;
        bestIndex = targetIndex;
      }
    }
    result[stateIndex] = bestIndex;
  }
  return result;
}

function transitionRowMeans(
  matrix: Float64Array,
  exposures: number[],
  stateSize: number,
): Float64Array {
  const targetSize = exposures.length;
  const result = new Float64Array(stateSize);
  for (let stateIndex = 0; stateIndex < stateSize; stateIndex += 1) {
    let total = 0;
    let weightedExposure = 0;
    for (let targetIndex = 0; targetIndex < targetSize; targetIndex += 1) {
      const probability = matrix[stateIndex * targetSize + targetIndex]!;
      total += probability;
      weightedExposure += probability * exposures[targetIndex]!;
    }
    result[stateIndex] = total > 0 ? weightedExposure / total : 0;
  }
  return result;
}

function transitionAntiDiagonalSlice(
  matrix: Float64Array,
  targetExposures: number[],
  currentExposures: Float64Array,
): Float64Array {
  const targetSize = targetExposures.length;
  const result = new Float64Array(targetSize);
  for (let targetIndex = 0; targetIndex < targetSize; targetIndex += 1) {
    const stateIndex = nearestExposureIndex(currentExposures, -targetExposures[targetIndex]!);
    result[targetIndex] = matrix[stateIndex * targetSize + targetIndex]!;
  }
  return result;
}

function nearestExposureIndex(exposures: ArrayLike<number>, target: number): number {
  let bestIndex = 0;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (let index = 0; index < exposures.length; index += 1) {
    const distance = Math.abs(exposures[index]! - target);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = index;
    }
  }
  return bestIndex;
}

function exposureCanvasPosition(
  exposures: number[],
  target: number,
  cellSize: number,
): number {
  if (exposures.length <= 1) return cellSize / 2;
  if (target <= exposures[0]!) return cellSize / 2;
  if (target >= exposures.at(-1)!) return (exposures.length - 0.5) * cellSize;
  let low = 0;
  let high = exposures.length - 1;
  while (high - low > 1) {
    const middle = Math.floor((low + high) / 2);
    if (exposures[middle]! <= target) low = middle;
    else high = middle;
  }
  const span = exposures[high]! - exposures[low]!;
  const fraction = span > 0 ? (target - exposures[low]!) / span : 0;
  return (low + fraction + 0.5) * cellSize;
}

function drawTransitionHeatmap(
  canvas: HTMLCanvasElement,
  matrix: Float64Array,
  scale: number,
  palette: "cyan" | "violet" | "amber" | "emerald" | "difference",
  targetExposures: number[],
  currentExposures: Float64Array,
  oracleModes: Uint16Array,
  strategyModes: Uint16Array,
  oracleMeans: Float64Array,
  strategyMeans: Float64Array,
  oracleCurrentExposure: number,
  candidateCurrentExposure: number,
  markings: TransitionHeatmapMarkings,
  selectedStateIndex?: number,
  selectedTargetIndex?: number,
): void {
  const context = canvas.getContext("2d");
  if (!context) return;
  const cellSize = 4;
  const targetSize = targetExposures.length;
  const stateSize = currentExposures.length;
  const renderWidth = targetSize * cellSize;
  const renderHeight = stateSize * cellSize;
  if (canvas.width !== renderWidth) canvas.width = renderWidth;
  if (canvas.height !== renderHeight) canvas.height = renderHeight;
  const image = context.createImageData(renderWidth, renderHeight);
  for (let stateIndex = 0; stateIndex < stateSize; stateIndex += 1) {
    const displayRow = stateSize - 1 - stateIndex;
    for (let targetIndex = 0; targetIndex < targetSize; targetIndex += 1) {
      const value = matrix[stateIndex * targetSize + targetIndex]!;
      const normalized = Number.isFinite(value) && scale > 0
        ? palette === "difference"
          ? Math.max(-1, Math.min(1, value / scale))
          : Math.max(0, Math.min(1, value / scale))
        : 0;
      const intensity = Math.sqrt(Math.abs(normalized));
      const base = [24, 31, 45];
      const endpoint = palette === "violet"
        ? [167, 139, 250]
        : palette === "amber"
          ? [251, 191, 36]
        : palette === "emerald"
          ? [52, 211, 153]
        : palette === "difference" && normalized < 0
          ? [251, 113, 133]
          : [34, 211, 238];
      const red = Math.round(base[0]! + (endpoint[0]! - base[0]!) * intensity);
      const green = Math.round(base[1]! + (endpoint[1]! - base[1]!) * intensity);
      const blue = Math.round(base[2]! + (endpoint[2]! - base[2]!) * intensity);
      for (let y = 0; y < cellSize; y += 1) {
        for (let x = 0; x < cellSize; x += 1) {
          const pixel = (
            (displayRow * cellSize + y) * renderWidth
              + targetIndex * cellSize + x
          ) * 4;
          image.data[pixel] = red;
          image.data[pixel + 1] = green;
          image.data[pixel + 2] = blue;
          image.data[pixel + 3] = 255;
        }
      }
    }
  }
  context.putImageData(image, 0, 0);

  const rowTop = (stateIndex: number) => (stateSize - 1 - stateIndex) * cellSize;
  const oracleRow = nearestExposureIndex(currentExposures, oracleCurrentExposure);
  const candidateRow = nearestExposureIndex(currentExposures, candidateCurrentExposure);
  context.lineWidth = 1;
  context.setLineDash([]);
  if (markings.oraclePathRow) {
    context.strokeStyle = "rgba(103, 232, 249, 0.95)";
    context.strokeRect(0.5, rowTop(oracleRow) + 0.5, renderWidth - 1, cellSize - 1);
  }
  if (markings.candidateRow) {
    context.setLineDash([cellSize * 2, cellSize]);
    context.strokeStyle = "rgba(110, 231, 183, 0.95)";
    context.strokeRect(0.5, rowTop(candidateRow) + 0.5, renderWidth - 1, cellSize - 1);
  }
  context.setLineDash([]);

  const traceOppositeExposure = () => {
    context.beginPath();
    for (let targetIndex = 0; targetIndex < targetSize; targetIndex += 1) {
      const stateIndex = nearestExposureIndex(currentExposures, -targetExposures[targetIndex]!);
      const x = targetIndex * cellSize + cellSize / 2;
      const y = rowTop(stateIndex) + cellSize / 2;
      if (targetIndex === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    }
  };
  if (markings.oppositeExposure) {
    context.setLineDash([cellSize * 1.5, cellSize]);
    traceOppositeExposure();
    context.strokeStyle = "rgba(2, 6, 23, 0.88)";
    context.lineWidth = 2.8;
    context.stroke();
    traceOppositeExposure();
    context.strokeStyle = "rgba(251, 191, 36, 0.88)";
    context.lineWidth = 1.1;
    context.stroke();
  }
  context.setLineDash([]);

  const drawMode = (x: number, y: number, color: string) => {
    context.fillStyle = "rgba(2, 6, 23, 0.95)";
    context.fillRect(x - 1.5, y - 1.5, 3, 3);
    context.fillStyle = color;
    context.fillRect(x - 0.5, y - 0.5, 1, 1);
  };
  if (markings.oracleMode || markings.predictionMode) {
    for (let stateIndex = 0; stateIndex < stateSize; stateIndex += 1) {
      const y = rowTop(stateIndex) + cellSize / 2;
      const oracleX = markings.oracleMode
        ? oracleModes[stateIndex]! * cellSize + cellSize / 2
        : 0;
      const strategyX = markings.predictionMode
        ? strategyModes[stateIndex]! * cellSize + cellSize / 2
        : 0;
      if (markings.oracleMode && markings.predictionMode && oracleX === strategyX) {
        context.fillStyle = "rgba(2, 6, 23, 0.95)";
        context.fillRect(oracleX - 2, y - 2, 4, 4);
        context.fillStyle = "rgb(103, 232, 249)";
        context.fillRect(oracleX - 1, y - 1, 1, 2);
        context.fillStyle = "rgb(196, 181, 253)";
        context.fillRect(oracleX, y - 1, 1, 2);
        continue;
      }
      if (markings.oracleMode) drawMode(oracleX, y, "rgb(103, 232, 249)");
      if (markings.predictionMode) drawMode(strategyX, y, "rgb(196, 181, 253)");
    }
  }

  const drawDiamond = (x: number, y: number, color: string) => {
    context.beginPath();
    context.moveTo(x, y - 1.8);
    context.lineTo(x + 1.8, y);
    context.lineTo(x, y + 1.8);
    context.lineTo(x - 1.8, y);
    context.closePath();
    context.strokeStyle = "rgba(2, 6, 23, 0.9)";
    context.lineWidth = 2.25;
    context.stroke();
    context.strokeStyle = color;
    context.lineWidth = 0.9;
    context.stroke();
  };
  const drawCircle = (x: number, y: number, color: string) => {
    context.beginPath();
    context.arc(x, y, 1.55, 0, Math.PI * 2);
    context.strokeStyle = "rgba(2, 6, 23, 0.9)";
    context.lineWidth = 2.25;
    context.stroke();
    context.strokeStyle = color;
    context.lineWidth = 0.9;
    context.stroke();
  };
  if (markings.oracleMean || markings.predictionMean) {
    for (let stateIndex = 0; stateIndex < stateSize; stateIndex += 1) {
      const y = rowTop(stateIndex) + cellSize / 2;
      if (markings.oracleMean) {
        drawDiamond(
          exposureCanvasPosition(targetExposures, oracleMeans[stateIndex]!, cellSize),
          y,
          "rgba(103, 232, 249, 0.88)",
        );
      }
      if (markings.predictionMean) {
        drawCircle(
          exposureCanvasPosition(targetExposures, strategyMeans[stateIndex]!, cellSize),
          y,
          "rgba(196, 181, 253, 0.88)",
        );
      }
    }
  }
  if (selectedStateIndex !== undefined && selectedTargetIndex !== undefined) {
    const x = selectedTargetIndex * cellSize;
    const y = rowTop(selectedStateIndex);
    context.setLineDash([]);
    context.strokeStyle = "rgba(2, 6, 23, 0.98)";
    context.lineWidth = 3;
    context.strokeRect(x + 0.5, y + 0.5, cellSize - 1, cellSize - 1);
    context.strokeStyle = "rgba(248, 250, 252, 0.98)";
    context.lineWidth = 1;
    context.strokeRect(x + 0.5, y + 0.5, cellSize - 1, cellSize - 1);
  }
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

function DurationInput(props: {
  label: string;
  value: number;
  min?: number;
  max?: number;
  onInput: (value: number) => void;
}) {
  return <InspectorNumber
    label={`${props.label} (seconds)`}
    value={round(props.value / 1_000, 3)}
    min={(props.min ?? 1) / 1_000}
    max={props.max === undefined ? undefined : props.max / 1_000}
    step={1}
    onInput={(value) => props.onInput(value * 1_000)}
  />;
}

function OverlayButton(props: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button class="btn" classList={{ "border-accent text-accent": props.active }} aria-pressed={props.active} type="button" onClick={props.onClick}>
      {props.label}
    </button>
  );
}

function PerformanceMetric(props: {
  label: string;
  value: string;
  description: string;
  prominent?: boolean;
}) {
  const descriptionId = createUniqueId();
  return (
    <div
      class="group relative min-w-0 rounded-lg px-2 py-1.5 transition-colors hover:bg-ink-800/70 focus-within:bg-ink-800/70"
      classList={{ "min-w-28 bg-accent/10 px-3": props.prominent }}
    >
      <div class="flex min-w-0 items-center gap-1">
        <div class="truncate text-[10px] font-medium uppercase tracking-wide text-ink-400">{props.label}</div>
        <button
          type="button"
          class="shrink-0 rounded-full text-ink-300 transition hover:text-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/45"
          aria-label={`About ${props.label}`}
          aria-describedby={descriptionId}
        >
          <Info aria-hidden="true" size={12} />
        </button>
      </div>
      <div
        class="truncate font-semibold tabular-nums text-ink-100"
        classList={{ "mt-0.5 text-xl": props.prominent, "text-sm": !props.prominent }}
      >{props.value}</div>
      <div
        id={descriptionId}
        role="tooltip"
        class="pointer-events-none invisible absolute left-0 right-0 top-[calc(100%+0.25rem)] z-40 min-w-0 rounded-2 border border-line bg-ink-800 p-2 text-xs normal-case leading-relaxed tracking-normal text-ink-200 opacity-0 shadow-xl transition-opacity group-hover:visible group-hover:opacity-100 group-focus-within:visible group-focus-within:opacity-100"
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
  return points.map((point) => {
    const forecast = point.targetProbability !== null;
    return {
      time: point.time,
      price: point.price,
      kind: transitionSide(point) === "buy" ? "buy-signal" : "sell-signal",
      label: forecast
        ? `Forecast ${signedExposure(point.fromExposure)} → ${signedExposure(point.exposure)} · p ${ratioPercent(point.targetProbability!)}`
        : `KAMA ${point.fromState} → ${point.state} · ${ratioPercent(point.sizeFraction)} · Q ${ratioPercent(point.quality)}`,
      reason: forecast
        ? `The modal target cell at current exposure ${signedExposure(point.fromExposure)} is ${signedExposure(point.exposure)} with ${ratioPercent(point.targetProbability!)} probability.`
        : point.matchedTime === null
          ? `Quality ${ratioPercent(point.quality)} · acceleration ${formatQuote(point.acceleration, 3)} · overextension ${formatQuote(point.overextension, 3)} · EMA ${formatQuote(point.emaRate, 2)} bps/h · RSI ${formatQuote(point.rsi, 1)} · DMI/ADX ${formatQuote(point.dmi, 1)}/${formatQuote(point.adx, 1)} · unmatched`
          : `Quality ${ratioPercent(point.quality)} · acceleration ${formatQuote(point.acceleration, 3)} · overextension ${formatQuote(point.overextension, 3)} · EMA ${formatQuote(point.emaRate, 2)} bps/h · RSI ${formatQuote(point.rsi, 1)} · DMI/ADX ${formatQuote(point.dmi, 1)}/${formatQuote(point.adx, 1)} · matched oracle ${signedDuration(point.lagMs ?? 0)}`,
      signalState: point.state,
    };
  });
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

function predictorPresetOption(
  preset: VwKamaPredictorPreset,
  catalog: VwKamaInspectorCatalog | undefined,
): SearchableOption {
  const window = preset.windowId
    ? catalog?.windows.find((item) => item.id === preset.windowId)?.label ?? preset.windowId
    : "all static windows";
  const scale = preset.intervalMs ? ` · ${formatDuration(preset.intervalMs)}` : "";
  return {
    value: preset.id,
    label: `${preset.scope === "global" ? "Global fit" : `Local fit · ${window}`}${scale} · regret ${formatQuote(preset.loss, 6)}`,
    keywords: `${preset.scope} ${preset.windowId ?? "global"} ${preset.source}`,
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

function latestHistoryLabel(days: number, endTime: number): string {
  const startTime = endTime - days * 86_400_000;
  const range = Number.isFinite(startTime) && !Number.isNaN(new Date(startTime).getTime())
    ? ` · ${formatDateRange(startTime, endTime)}`
    : "";
  return `Latest history · last ${days} day${days === 1 ? "" : "s"}${range}`;
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

function exposurePathOracle(
  path: VwKamaValuePathPoint[],
  candles: Candle[],
  friction: number,
): BacktestOraclePath {
  if (path.length === 0) return { ...emptyOracle, friction };
  let previous: OracleState = "flat";
  const points = path.map((point): BacktestOraclePoint => {
    const state: OracleState = point.exposure > Number.EPSILON
      ? "long"
      : point.exposure < -Number.EPSILON ? "short" : "flat";
    const candle = nearestCandle(candles, point.time);
    const result = {
      time: point.time,
      price: candle?.close ?? 0,
      fromState: previous,
      state,
      action: previous === state
        ? "hold" as const
        : previous === "flat"
          ? "open" as const
          : state === "flat" ? "close" as const : "switch" as const,
    };
    previous = state;
    return result;
  });
  return {
    mode: "fixed-notional",
    eventMode: "close",
    leverage: Math.max(...path.map((point) => Math.abs(point.exposure)), 0),
    friction,
    points,
  };
}

function nearestCandle(candles: Candle[], time: number): Candle | undefined {
  if (candles.length === 0) return undefined;
  let low = 0;
  let high = candles.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (candles[middle]!.closeTime < time) low = middle + 1;
    else high = middle;
  }
  const right = candles[Math.min(candles.length - 1, low)]!;
  const left = candles[Math.max(0, low - 1)]!;
  return Math.abs(left.closeTime - time) <= Math.abs(right.closeTime - time) ? left : right;
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

function mergeDetailValueOraclePath(
  overview: VwKamaValuePathPoint[],
  detail: VwKamaCandleRangeResponse | undefined,
  analysis: VwKamaInspectorResponse | undefined,
): VwKamaValuePathPoint[] {
  if (!detail?.valueOraclePath?.length || !analysis
    || detail.windowId !== analysis.window.id
    || detail.intervalMs !== analysis.intervalMs) return overview;
  const replacement = detailReplacementRange(analysis.candles, detail.candles);
  if (!replacement) return overview;
  return [
    ...overview.filter((point) => point.time < replacement.start || point.time >= replacement.end),
    ...detail.valueOraclePath.filter((point) =>
      point.time >= replacement.start && point.time < replacement.end),
  ].sort((left, right) => left.time - right.time);
}

function mergeDetailValueCandidatePath(
  overview: VwKamaValuePathPoint[],
  detail: VwKamaCandleRangeResponse | undefined,
  analysis: VwKamaInspectorResponse | undefined,
): VwKamaValuePathPoint[] {
  if (!detail?.valueCandidatePath?.length || !analysis
    || detail.windowId !== analysis.window.id
    || detail.intervalMs !== analysis.intervalMs) return overview;
  const replacement = detailReplacementRange(analysis.candles, detail.candles);
  if (!replacement) return overview;
  return [
    ...overview.filter((point) => point.time < replacement.start || point.time >= replacement.end),
    ...detail.valueCandidatePath.filter((point) =>
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
