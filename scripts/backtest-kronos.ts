import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
import {
  type BacktestSummary,
  type Candle,
  type ExposureValueOracleActionDistribution,
  type TradeFill,
  type VwKamaInspectorWindow,
} from "@trading/bot-algo";
import { readCandleShardReferenceSync } from "@trading/storage";
import { appConfig } from "../apps/server/src/config.js";
import {
  HINDSIGHT_ORACLE_MAINTENANCE_BPS_HOUR,
  HINDSIGHT_ORACLE_TEMPERATURE,
  type OracleBacktestDecision,
  hindsightOracleTargetDecision,
  oracleMaximumEffectiveLeverage,
  runBotBacktestFromCandles,
} from "../apps/server/src/bot-backtest.js";
import { KamaInspector } from "../apps/server/src/kama-inspector.js";
import {
  crossValidateKronosReturnCalibrations,
  predictKronosReturn,
  validateKronosReturnCalibration,
  type CrossValidatedKronosReturnCalibration,
  type FrozenKronosReturnCalibration,
  type KronosReturnCalibrationExample,
} from "./kronos-return-calibrator.js";

const REPO_ROOT = path.resolve(import.meta.dirname, "..");
const HISTORY_ROOT = path.join(
  REPO_ROOT,
  "data/market/immutable/refs/candles/spot-btcusdt/btcusdt/1m",
);
const STEP_MS = 60_000;
const HORIZON_CANDLES = 15;
const HORIZON_MS = HORIZON_CANDLES * STEP_MS;
const PREDICTOR_TRAINING_END = Date.parse("2024-01-01T00:00:00.000Z");
const VALIDATION_CUTOFF = Date.parse("2024-07-01T00:00:00.000Z");
const KRONOS_FORECAST_CONTRACT = "kronos-causal-15x1m-forecast-v2";
const FOUNDATION_FORECAST_CONTRACT = "foundation-forecast-causal-15x1m-v1";
const FORECAST_CONTRACTS = [
  KRONOS_FORECAST_CONTRACT,
  FOUNDATION_FORECAST_CONTRACT,
] as const;
const POLICY_CONTRACT = "kronos-bot-policy-v3";
const POLICY_SELECTION_CONTRACT =
  "pretraining-ranking-with-untouched-post-training-confirmation-v2";
const RETURN_CALIBRATION_LAMBDAS = [0.0001, 0.001, 0.01, 0.1, 1, 10] as const;

interface KronosExecutionOracleConfig {
  holding_period_steps: number;
  decision_delay_steps: number;
  value_horizon_steps: number;
  friction: number;
  grid_size: number;
  temperature: number;
  min_exposure: number;
  max_exposure: number;
  max_effective_exposure: number;
  quote_borrow_rate: number;
  asset_borrow_rate: number;
}

const ROW_KEYS = new Set([
  "decisionTime",
  "targetStartTime",
  "windowIds",
  "anchorPrice",
  "horizonLogReturnMean",
  "horizonLogReturnMedian",
  "horizonLogReturnStd",
  "horizonUpProbability",
  "horizonLogReturnP10",
  "horizonLogReturnP90",
  "meanCloseLogPath",
  "medianCloseLogPath",
  "oracleProbabilities",
  "executionOracleProbabilities",
  "executionUtilityProbabilities",
]);

export interface KronosForecastRow {
  decisionTime: number;
  targetStartTime: number;
  windowIds: string[];
  anchorPrice: number;
  horizonLogReturnMean: number;
  horizonLogReturnMedian: number;
  horizonLogReturnStd: number;
  horizonUpProbability: number;
  horizonLogReturnP10: number;
  horizonLogReturnP90: number;
  meanCloseLogPath: number[];
  medianCloseLogPath: number[];
  oracleProbabilities: number[];
  executionOracleProbabilities?: number[];
  executionUtilityProbabilities?: number[];
}

export interface KronosForecastArtifact {
  version: 1 | 2;
  contract: (typeof FORECAST_CONTRACTS)[number];
  generatedAt: string;
  runSignature: string;
  modelId: string;
  market: string;
  intervalMs: number;
  lookbackCandles: number;
  horizonCandles: number;
  temperature: number;
  topP: number;
  sampleCount: number;
  originStrideCandles?: number;
  oracleGrid: number[];
  executionOracleGrid?: number[];
  executionOracle?: KronosExecutionOracleConfig;
  rows: KronosForecastRow[];
}

export interface KronosBotPolicy {
  signalSource:
    | "execution-oracle"
    | "execution-utility"
    | "execution-utility-mean"
    | "execution-utility-sign"
    | "execution-mean"
    | "execution-sign"
    | "horizon-mean"
    | "horizon-median"
    | "calibrated-return";
  returnCalibrationId?: string;
  filteredAction: "flat" | "hold";
  minimumConsecutiveDirections: number;
  maximumLeverage: number;
  staticConfidenceScale: number;
  confidenceExposurePower: number;
  confidenceLeverageFloor: number;
  expansionConfirmationMass: number;
  expansionDeltaCapFraction: number;
  minimumAbsoluteMeanReturnBps: number;
  minimumDirectionalConfidence: number;
  requireOracleReturnAgreement: boolean;
}

interface KronosPolicyArtifact {
  version: 3;
  contract: typeof POLICY_CONTRACT;
  createdAt: string;
  forecastRunSignature: string;
  forecastArtifactSha256: string;
  forecastModelId: string;
  validationCutoff: number;
  calibrationWindowIds: string[];
  selectionEpisodes: EvaluationEpisode[];
  postTrainingConfirmationEpisodes: EvaluationEpisode[];
  candidateCount: number;
  selectionContract: typeof POLICY_SELECTION_CONTRACT;
  selectionObjective: string;
  selectedPolicy: KronosBotPolicy;
  selectedPolicyId: string;
  returnCalibration: FrozenKronosReturnCalibration | null;
  returnCalibrationCandidates: Array<{
    id: string;
    ridgeLambda: number;
    oofMse: number;
    oofZeroMseSkill: number;
    oofCorrelation: number;
    oofDirectionAccuracy: number;
  }>;
  selection: AggregateReport;
  selectionEpisodeReports: WindowReport[];
  postTrainingConfirmation: AggregateReport;
  calibration: AggregateReport;
  calibrationWindows: WindowReport[];
  topCandidates: Array<{
    policy: KronosBotPolicy;
    policyId: string;
    aggregate: AggregateReport;
    postTrainingConfirmation: AggregateReport;
    selectionEligible: boolean;
    selectionScore: number;
  }>;
}

export interface PreparedForecastRow {
  row: KronosForecastRow;
  distribution: ExposureValueOracleActionDistribution;
  utilityDistribution: ExposureValueOracleActionDistribution;
  calibratedReturns: Map<string, number>;
}

interface PreparedWindow {
  window: VwKamaInspectorWindow;
  warmup: Candle[];
  replay: Candle[];
  forecasts: Map<number, PreparedForecastRow>;
}

export interface PolicyFilterState {
  direction: -1 | 0 | 1;
  consecutiveDirections: number;
}

interface CompactFill {
  side: TradeFill["side"];
  price: number;
  quantity: number;
  quoteQuantity: number;
  feeQuote: number;
  realizedPnl: number;
  filledAt: number;
  reason: string;
  positionEffect?: TradeFill["positionEffect"];
  liquidation?: boolean;
}

export interface WindowReport {
  windowId: string;
  label: string;
  startTime: number;
  endTime: number;
  forecastRows: number;
  forecastRowsConsumed: number;
  oracleDecisions: number;
  emittedSignals: number;
  filteredForecasts: number;
  heldForecasts: number;
  flattenedForecasts: number;
  summary: Pick<
    BacktestSummary,
    | "finalEquity"
    | "netPnl"
    | "returnPct"
    | "maxInitialBalanceDrawdownPct"
    | "maxDrawdownPct"
    | "maxEffectiveLeverage"
    | "perfectMarginNetPnl"
    | "perfectMarginReturnPct"
    | "perfectMarginCapturePct"
    | "tradeCount"
    | "feesPaid"
    | "maintenancePaid"
    | "winRate"
    | "closedPositionCount"
    | "profitableClosedPositionCount"
    | "liquidatedPositionCount"
  >;
  fills?: CompactFill[];
}

export interface EvaluationEpisode extends VwKamaInspectorWindow {
  sourceWindowIds: string[];
}

export interface AggregateReport {
  windows: number;
  startingEquity: number;
  finalEquity: number;
  netPnl: number;
  returnPct: number;
  geometricMeanReturnPct: number;
  medianWindowReturnPct: number;
  chronologicalFoldCount: number;
  worstChronologicalFoldReturnPct: number;
  perfectMarginNetPnl: number;
  meanPerfectMarginReturnPct: number;
  perfectMarginCapturePct: number | null;
  profitableWindows: number;
  activeWindows: number;
  tradeCount: number;
  feesPaid: number;
  maintenancePaid: number;
  maximumDrawdownPct: number;
  meanDrawdownPct: number;
  liquidatedPositionCount: number;
  forecastRows: number;
  forecastRowsConsumed: number;
  oracleDecisions: number;
  emittedSignals: number;
  filteredForecasts: number;
  heldForecasts: number;
  flattenedForecasts: number;
  selectionEligible: boolean;
  selectionScore: number;
}

export function confirmedPolicySelection(
  aggregate: AggregateReport,
  postTrainingConfirmation: AggregateReport,
): { eligible: boolean; score: number } {
  const eligible = aggregate.selectionEligible
    && postTrainingConfirmation.selectionEligible;
  return {
    eligible,
    score: eligible ? aggregate.selectionScore : -1e12,
  };
}

interface PhaseReport {
  version: 3;
  contract: "kronos-bot-backtest-v3";
  createdAt: string;
  phase: "smoke" | "validation";
  forecast: {
    file: string;
    sha256: string;
    runSignature: string;
    modelId: string;
    sampleCount: number;
    temperature: number;
  };
  policyArtifact: {
    file: string;
    sha256: string;
  } | null;
  policy: KronosBotPolicy;
  policyId: string;
  returnCalibration: FrozenKronosReturnCalibration | null;
  split: {
    validationCutoff: number;
    windowIds: string[];
    leakagePolicy: string;
  };
  execution: {
    feeBps: number;
    marketSlippageBps: number;
    borrowBpsHour: number;
    closeAtWindowEnd: boolean;
    perWindowIndependentStartingQuote: number;
  };
  gates: {
    allForecastsConsumed: boolean;
    actualFills: boolean;
    noLiquidations: boolean;
    positiveNetPnl: boolean;
  };
  controls: Array<{
    id: "constant-long-1x" | "constant-short-1x" | "constant-long-5x" | "constant-short-5x";
    aggregate: AggregateReport;
    episodes: WindowReport[];
  }>;
  controlComparison: {
    bestControlId: PhaseReport["controls"][number]["id"] | null;
    bestControlReturnPct: number | null;
    selectedMinusBestControlPct: number | null;
  };
  aggregate: AggregateReport;
  episodes: WindowReport[];
  inspectorAggregate: AggregateReport;
  windows: WindowReport[];
}

export function kronosInspectorWindows(): VwKamaInspectorWindow[] {
  const inspector = new KamaInspector(path.join(REPO_ROOT, "data"));
  const windows = inspector.catalog().windows.filter((window) =>
    window.id !== "latest" && !window.id.startsWith("fit-"));
  if (windows.length !== 28) {
    throw new Error(`Expected 28 non-fit static inspector windows; found ${windows.length}.`);
  }
  return windows;
}

export function mergeOverlappingWindows(
  windows: readonly VwKamaInspectorWindow[],
  prefix = "episode",
): EvaluationEpisode[] {
  const sorted = [...windows].sort((left, right) =>
    left.startTime - right.startTime || left.endTime - right.endTime
    || left.id.localeCompare(right.id));
  const merged: Array<{
    startTime: number;
    endTime: number;
    sourceIntervalMs: number;
    sourceWindowIds: string[];
  }> = [];
  for (const window of sorted) {
    const previous = merged.at(-1);
    if (previous && window.startTime <= previous.endTime) {
      previous.endTime = Math.max(previous.endTime, window.endTime);
      previous.sourceIntervalMs = Math.min(previous.sourceIntervalMs, window.sourceIntervalMs);
      previous.sourceWindowIds.push(window.id);
    } else {
      merged.push({
        startTime: window.startTime,
        endTime: window.endTime,
        sourceIntervalMs: window.sourceIntervalMs,
        sourceWindowIds: [window.id],
      });
    }
  }
  return merged.map((episode, index) => ({
    id: `${prefix}-episode-${String(index + 1).padStart(2, "0")}`,
    label: `${prefix} episode ${index + 1}`,
    group: `${prefix} episodes`,
    ...episode,
  }));
}

export function policyCalibrationEpisodeSplit(
  episodes: readonly EvaluationEpisode[],
): {
  selectionEpisodes: EvaluationEpisode[];
  postTrainingConfirmationEpisodes: EvaluationEpisode[];
} {
  const crossing = episodes.filter((episode) =>
    episode.startTime < PREDICTOR_TRAINING_END
    && episode.endTime > PREDICTOR_TRAINING_END);
  if (crossing.length > 0) {
    throw new Error(
      "A policy-calibration episode crosses the predictor training cutoff: "
      + crossing.map((episode) => episode.id).join(", "),
    );
  }
  const selectionEpisodes = episodes.filter((episode) =>
    episode.endTime <= PREDICTOR_TRAINING_END);
  const postTrainingConfirmationEpisodes = episodes.filter((episode) =>
    episode.startTime >= PREDICTOR_TRAINING_END);
  if (selectionEpisodes.length !== 11 || postTrainingConfirmationEpisodes.length !== 2
    || selectionEpisodes.length + postTrainingConfirmationEpisodes.length !== episodes.length) {
    throw new Error(
      "Expected 11 pretraining policy-selection episodes and two untouched "
      + `post-training confirmation episodes; found ${selectionEpisodes.length}/`
      + `${postTrainingConfirmationEpisodes.length}.`,
    );
  }
  return { selectionEpisodes, postTrainingConfirmationEpisodes };
}

export function parseKronosForecastArtifact(source: unknown): KronosForecastArtifact {
  const root = objectValue(source, "forecast artifact");
  const isKronos = root.version === 2 && root.contract === KRONOS_FORECAST_CONTRACT;
  const isFoundation = root.version === 1 && root.contract === FOUNDATION_FORECAST_CONTRACT;
  if (!isKronos && !isFoundation) {
    throw new Error(
      `Forecast artifact must use ${KRONOS_FORECAST_CONTRACT} version 2 or `
      + `${FOUNDATION_FORECAST_CONTRACT} version 1.`,
    );
  }
  const intervalMs = finiteNumber(root.intervalMs, "intervalMs");
  const horizonCandles = finiteNumber(root.horizonCandles, "horizonCandles");
  if (intervalMs !== STEP_MS || horizonCandles !== HORIZON_CANDLES) {
    throw new Error("Kronos bot backtests require causal 15x1m forecasts.");
  }
  const oracleGrid = numberArray(root.oracleGrid, "oracleGrid");
  validateOracleGrid(oracleGrid, "oracleGrid");
  const executionOracleGrid = root.executionOracleGrid === undefined
    ? undefined
    : numberArray(root.executionOracleGrid, "executionOracleGrid");
  let parsedExecutionOracle: KronosExecutionOracleConfig | undefined;
  if (executionOracleGrid !== undefined) {
    validateOracleGrid(executionOracleGrid, "executionOracleGrid");
    const executionOracle = objectValue(root.executionOracle, "executionOracle");
    const holdingPeriod = integerNumber(
      executionOracle.holding_period_steps,
      "executionOracle.holding_period_steps",
    );
    const decisionDelay = integerNumber(
      executionOracle.decision_delay_steps,
      "executionOracle.decision_delay_steps",
    );
    const valueHorizon = integerNumber(
      executionOracle.value_horizon_steps,
      "executionOracle.value_horizon_steps",
    );
    if (holdingPeriod !== HORIZON_CANDLES
      || decisionDelay !== HORIZON_CANDLES
      || valueHorizon !== HORIZON_CANDLES) {
      throw new Error(
        "The execution oracle must hold each action for the complete 15-minute bot decision interval.",
      );
    }
    const friction = finiteNumber(executionOracle.friction, "executionOracle.friction");
    const gridSize = integerNumber(executionOracle.grid_size, "executionOracle.grid_size");
    const temperature = finiteNumber(
      executionOracle.temperature,
      "executionOracle.temperature",
    );
    const minimumExposure = finiteNumber(
      executionOracle.min_exposure,
      "executionOracle.min_exposure",
    );
    const maximumExposure = finiteNumber(
      executionOracle.max_exposure,
      "executionOracle.max_exposure",
    );
    const maximumEffectiveExposure = finiteNumber(
      executionOracle.max_effective_exposure,
      "executionOracle.max_effective_exposure",
    );
    const quoteBorrowRate = finiteNumber(
      executionOracle.quote_borrow_rate,
      "executionOracle.quote_borrow_rate",
    );
    const assetBorrowRate = finiteNumber(
      executionOracle.asset_borrow_rate,
      "executionOracle.asset_borrow_rate",
    );
    const expectedFriction = (
      appConfig.strategy.feeBps + appConfig.strategy.positionRisk.marketSlippageBps
    ) / 10_000;
    const expectedBorrowRate = Math.expm1(
      Math.log1p(HINDSIGHT_ORACLE_MAINTENANCE_BPS_HOUR / 10_000) / 60,
    );
    const gridMaximum = Math.max(
      Math.abs(executionOracleGrid[0]!),
      Math.abs(executionOracleGrid.at(-1)!),
    );
    const expectedMaximumEffectiveExposure = oracleMaximumEffectiveLeverage(gridMaximum);
    const approximately = (left: number, right: number) =>
      Math.abs(left - right) <= 1e-9 * Math.max(1, Math.abs(left), Math.abs(right));
    if (gridSize !== executionOracleGrid.length
      || !approximately(minimumExposure, executionOracleGrid[0]!)
      || !approximately(maximumExposure, executionOracleGrid.at(-1)!)
      || !approximately(friction, expectedFriction)
      || !approximately(temperature, HINDSIGHT_ORACLE_TEMPERATURE)
      || !approximately(maximumEffectiveExposure, expectedMaximumEffectiveExposure)
      || !approximately(quoteBorrowRate, expectedBorrowRate)
      || !approximately(assetBorrowRate, expectedBorrowRate)) {
      throw new Error(
        "The execution oracle costs, temperature, and risk bounds must match the bot simulator.",
      );
    }
    parsedExecutionOracle = {
      holding_period_steps: holdingPeriod,
      decision_delay_steps: decisionDelay,
      value_horizon_steps: valueHorizon,
      friction,
      grid_size: gridSize,
      temperature,
      min_exposure: minimumExposure,
      max_exposure: maximumExposure,
      max_effective_exposure: maximumEffectiveExposure,
      quote_borrow_rate: quoteBorrowRate,
      asset_borrow_rate: assetBorrowRate,
    };
  }
  if (!Array.isArray(root.rows) || root.rows.length === 0) {
    throw new Error("Forecast artifact contains no rows.");
  }
  const seenDecisionTimes = new Set<number>();
  const rows = root.rows.map((sourceRow, index): KronosForecastRow => {
    const row = objectValue(sourceRow, `rows[${index}]`);
    for (const key of Object.keys(row)) {
      if (!ROW_KEYS.has(key)) {
        throw new Error(`rows[${index}] contains forbidden or unknown field ${key}.`);
      }
    }
    const decisionTime = integerNumber(row.decisionTime, `rows[${index}].decisionTime`);
    const targetStartTime = integerNumber(
      row.targetStartTime,
      `rows[${index}].targetStartTime`,
    );
    if (targetStartTime !== decisionTime + 1 || targetStartTime % STEP_MS !== 0) {
      throw new Error(`rows[${index}] violates the causal decision-time contract.`);
    }
    if (seenDecisionTimes.has(decisionTime)) {
      throw new Error(`Duplicate forecast decision time ${decisionTime}.`);
    }
    seenDecisionTimes.add(decisionTime);
    const probabilities = numberArray(
      row.oracleProbabilities,
      `rows[${index}].oracleProbabilities`,
    );
    if (probabilities.length !== oracleGrid.length
      || probabilities.some((value) => value < 0)
      || probabilities.reduce((sum, value) => sum + value, 0) <= 0) {
      throw new Error(`rows[${index}] has an invalid oracle distribution.`);
    }
    const executionProbabilities = row.executionOracleProbabilities === undefined
      ? undefined
      : numberArray(
          row.executionOracleProbabilities,
          `rows[${index}].executionOracleProbabilities`,
        );
    if (executionProbabilities !== undefined && (
      executionOracleGrid === undefined
      || executionProbabilities.length !== executionOracleGrid.length
      || executionProbabilities.some((value) => value < 0)
      || executionProbabilities.reduce((sum, value) => sum + value, 0) <= 0
    )) {
      throw new Error(`rows[${index}] has an invalid execution-oracle distribution.`);
    }
    const executionUtilityProbabilities = row.executionUtilityProbabilities === undefined
      ? undefined
      : numberArray(
          row.executionUtilityProbabilities,
          `rows[${index}].executionUtilityProbabilities`,
        );
    if (executionUtilityProbabilities !== undefined && (
      executionOracleGrid === undefined
      || executionUtilityProbabilities.length !== executionOracleGrid.length
      || executionUtilityProbabilities.some((value) => value < 0)
      || executionUtilityProbabilities.reduce((sum, value) => sum + value, 0) <= 0
    )) {
      throw new Error(`rows[${index}] has an invalid execution-utility distribution.`);
    }
    if (executionOracleGrid !== undefined && (
      executionProbabilities === undefined || executionUtilityProbabilities === undefined
    )) {
      throw new Error(
        `rows[${index}] must contain both execution-oracle aggregations.`,
      );
    }
    const meanPath = numberArray(row.meanCloseLogPath, `rows[${index}].meanCloseLogPath`);
    const medianPath = numberArray(
      row.medianCloseLogPath,
      `rows[${index}].medianCloseLogPath`,
    );
    if (meanPath.length !== HORIZON_CANDLES || medianPath.length !== HORIZON_CANDLES) {
      throw new Error(`rows[${index}] must contain ${HORIZON_CANDLES}-candle paths.`);
    }
    const windowIds = stringArray(row.windowIds, `rows[${index}].windowIds`);
    if (windowIds.length === 0 || new Set(windowIds).size !== windowIds.length) {
      throw new Error(`rows[${index}] has invalid window memberships.`);
    }
    const horizonUpProbability = finiteNumber(
      row.horizonUpProbability,
      `rows[${index}].horizonUpProbability`,
    );
    if (horizonUpProbability < 0 || horizonUpProbability > 1) {
      throw new Error(`rows[${index}].horizonUpProbability must be in [0, 1].`);
    }
    return {
      decisionTime,
      targetStartTime,
      windowIds,
      anchorPrice: positiveNumber(row.anchorPrice, `rows[${index}].anchorPrice`),
      horizonLogReturnMean: finiteNumber(
        row.horizonLogReturnMean,
        `rows[${index}].horizonLogReturnMean`,
      ),
      horizonLogReturnMedian: finiteNumber(
        row.horizonLogReturnMedian,
        `rows[${index}].horizonLogReturnMedian`,
      ),
      horizonLogReturnStd: nonNegativeNumber(
        row.horizonLogReturnStd,
        `rows[${index}].horizonLogReturnStd`,
      ),
      horizonUpProbability,
      horizonLogReturnP10: finiteNumber(
        row.horizonLogReturnP10,
        `rows[${index}].horizonLogReturnP10`,
      ),
      horizonLogReturnP90: finiteNumber(
        row.horizonLogReturnP90,
        `rows[${index}].horizonLogReturnP90`,
      ),
      meanCloseLogPath: meanPath,
      medianCloseLogPath: medianPath,
      oracleProbabilities: probabilities,
      ...(executionProbabilities === undefined
        ? {}
        : { executionOracleProbabilities: executionProbabilities }),
      ...(executionUtilityProbabilities === undefined
        ? {}
        : { executionUtilityProbabilities }),
    };
  }).sort((left, right) => left.decisionTime - right.decisionTime);
  const originStrideCandles = root.originStrideCandles === undefined
    ? undefined
    : integerNumber(root.originStrideCandles, "originStrideCandles");
  if (originStrideCandles !== undefined && originStrideCandles !== HORIZON_CANDLES) {
    throw new Error(`originStrideCandles must equal ${HORIZON_CANDLES}.`);
  }
  return {
    version: isKronos ? 2 : 1,
    contract: isKronos ? KRONOS_FORECAST_CONTRACT : FOUNDATION_FORECAST_CONTRACT,
    generatedAt: stringValue(root.generatedAt, "generatedAt"),
    runSignature: stringValue(root.runSignature, "runSignature"),
    modelId: stringValue(root.modelId, "modelId"),
    market: stringValue(root.market, "market"),
    intervalMs,
    lookbackCandles: integerNumber(root.lookbackCandles, "lookbackCandles"),
    horizonCandles,
    temperature: root.temperature === undefined
      ? 1
      : positiveNumber(root.temperature, "temperature"),
    topP: root.topP === undefined ? 1 : positiveNumber(root.topP, "topP"),
    sampleCount: root.sampleCount === undefined
      ? 9
      : integerNumber(root.sampleCount, "sampleCount"),
    ...(originStrideCandles === undefined ? {} : { originStrideCandles }),
    oracleGrid,
    ...(executionOracleGrid === undefined ? {} : { executionOracleGrid }),
    ...(parsedExecutionOracle === undefined
      ? {}
      : { executionOracle: parsedExecutionOracle }),
    rows,
  };
}

export function loadKronosForecastArtifact(file: string): KronosForecastArtifact {
  return parseKronosForecastArtifact(JSON.parse(fs.readFileSync(file, "utf8")));
}

export function probabilityDistribution(
  source: ArrayLike<number>,
  sourceGrid: readonly number[],
): ExposureValueOracleActionDistribution {
  if (source.length !== sourceGrid.length || source.length < 3) {
    throw new Error("Oracle probabilities and grid must have matching dimensions.");
  }
  const probabilities = Float32Array.from(source);
  const grid = Float64Array.from(sourceGrid);
  let total = 0;
  for (const probability of probabilities) {
    if (!Number.isFinite(probability) || probability < 0) {
      throw new Error("Oracle probabilities must be finite and non-negative.");
    }
    total += probability;
  }
  if (!(total > 0)) throw new Error("Oracle probabilities must have positive mass.");
  let modalIndex = 0;
  let mean = 0;
  let secondMoment = 0;
  let entropy = 0;
  let feasibleActionCount = 0;
  for (let index = 0; index < probabilities.length; index += 1) {
    const probability = probabilities[index]! / total;
    probabilities[index] = probability;
    if (probability > probabilities[modalIndex]!) modalIndex = index;
    mean += probability * grid[index]!;
    secondMoment += probability * grid[index]! ** 2;
    if (probability > 0) {
      entropy -= probability * Math.log(probability);
      feasibleActionCount += 1;
    }
  }
  return {
    grid,
    probabilities,
    mean,
    secondMoment,
    modalExposure: grid[modalIndex]!,
    entropy,
    opportunity: 0,
    feasibleActionCount,
  };
}

export function validateDenseForecastCoverage(
  artifact: KronosForecastArtifact,
  windows: readonly VwKamaInspectorWindow[],
): void {
  const rowsAt = new Map(artifact.rows.map((row) => [row.targetStartTime, row]));
  const expectedByWindow = new Map<string, Set<number>>();
  for (const window of windows) {
    const expected = new Set<number>();
    for (
      let targetStart = window.startTime;
      targetStart <= window.endTime - HORIZON_MS;
      targetStart += HORIZON_MS
    ) {
      expected.add(targetStart);
      const row = rowsAt.get(targetStart);
      if (!row || !row.windowIds.includes(window.id)) {
        throw new Error(
          `${window.id} is missing dense forecast origin ${new Date(targetStart).toISOString()}.`,
        );
      }
    }
    expectedByWindow.set(window.id, expected);
  }
  for (const row of artifact.rows) {
    for (const windowId of row.windowIds) {
      const expected = expectedByWindow.get(windowId);
      if (expected && !expected.has(row.targetStartTime)) {
        throw new Error(`${windowId} contains an off-cadence forecast row.`);
      }
    }
  }
}

export function policyId(policy: KronosBotPolicy): string {
  return crypto.createHash("sha256").update(JSON.stringify(policy)).digest("hex").slice(0, 16);
}

export function validatePolicy(policy: KronosBotPolicy): void {
  if (![
    "execution-oracle",
    "execution-utility",
    "execution-utility-mean",
    "execution-utility-sign",
    "execution-mean",
    "execution-sign",
    "horizon-mean",
    "horizon-median",
    "calibrated-return",
  ].includes(policy.signalSource)) {
    throw new Error(
      "signalSource must be execution-oracle, execution-utility, execution-mean, "
      + "execution-utility-mean, execution-utility-sign, execution-sign, "
      + "horizon-mean, horizon-median, or calibrated-return.",
    );
  }
  if (policy.signalSource === "calibrated-return") {
    if (typeof policy.returnCalibrationId !== "string"
      || policy.returnCalibrationId.trim() === "") {
      throw new Error("calibrated-return policies require returnCalibrationId.");
    }
  } else if (policy.returnCalibrationId !== undefined) {
    throw new Error("returnCalibrationId is only valid for calibrated-return policies.");
  }
  if (!["flat", "hold"].includes(policy.filteredAction)) {
    throw new Error("filteredAction must be flat or hold.");
  }
  if (!Number.isSafeInteger(policy.minimumConsecutiveDirections)
    || policy.minimumConsecutiveDirections < 1) {
    throw new Error("minimumConsecutiveDirections must be a positive integer.");
  }
  positiveNumber(policy.maximumLeverage, "maximumLeverage");
  unitNumber(policy.staticConfidenceScale, "staticConfidenceScale");
  nonNegativeNumber(policy.confidenceExposurePower, "confidenceExposurePower");
  unitNumber(policy.confidenceLeverageFloor, "confidenceLeverageFloor");
  nonNegativeNumber(policy.expansionConfirmationMass, "expansionConfirmationMass");
  nonNegativeNumber(policy.expansionDeltaCapFraction, "expansionDeltaCapFraction");
  nonNegativeNumber(policy.minimumAbsoluteMeanReturnBps, "minimumAbsoluteMeanReturnBps");
  unitNumber(policy.minimumDirectionalConfidence, "minimumDirectionalConfidence");
  if (typeof policy.requireOracleReturnAgreement !== "boolean") {
    throw new Error("requireOracleReturnAgreement must be boolean.");
  }
}

export function calibrationPolicies(
  maximumLeverage: number,
  returnCalibrationIds: readonly string[] = [],
): KronosBotPolicy[] {
  const filters = [
    ["execution-oracle", "flat", 0, 0, false, 1],
    ["execution-oracle", "flat", 10, 0, false, 1],
    ["execution-oracle", "flat", 20, 0.2, false, 1],
    ["execution-oracle", "flat", 20, 0.2, true, 1],
    ["execution-oracle", "hold", 10, 0.1, false, 1],
    ["execution-oracle", "hold", 20, 0.2, false, 1],
    ["execution-oracle", "hold", 40, 0.2, false, 1],
    ["execution-oracle", "hold", 20, 0.2, true, 1],
    ["execution-oracle", "hold", 20, 0.2, false, 2],
    ["execution-oracle", "hold", 40, 0.2, false, 2],
    ["execution-oracle", "hold", 20, 0.4, true, 2],
    ["execution-oracle", "hold", 20, 0.2, false, 3],
    ["execution-oracle", "hold", 20, 0.2, false, 6],
    ["execution-oracle", "hold", 20, 0.2, false, 12],
    ["execution-utility", "flat", 0, 0, false, 1],
    ["execution-utility", "flat", 20, 0.2, false, 1],
    ["execution-utility", "hold", 10, 0.1, false, 1],
    ["execution-utility", "hold", 20, 0.2, false, 2],
    ["execution-utility", "hold", 40, 0.2, false, 2],
    ["execution-utility", "hold", 20, 0.2, false, 6],
    ["execution-utility-mean", "hold", 20, 0.2, false, 1],
    ["execution-utility-mean", "hold", 20, 0.2, false, 2],
    ["execution-utility-mean", "hold", 40, 0.2, false, 2],
    ["execution-utility-mean", "hold", 20, 0.2, false, 6],
    ["execution-utility-sign", "flat", 20, 0.2, false, 1],
    ["execution-utility-sign", "hold", 10, 0.1, false, 1],
    ["execution-utility-sign", "hold", 20, 0.2, false, 1],
    ["execution-utility-sign", "hold", 40, 0.2, false, 2],
    ["execution-utility-sign", "hold", 20, 0.2, false, 2],
    ["execution-utility-sign", "hold", 20, 0.2, false, 6],
    ["execution-mean", "hold", 20, 0.2, false, 1],
    ["execution-mean", "hold", 20, 0.2, false, 2],
    ["execution-mean", "hold", 40, 0.2, false, 2],
    ["execution-mean", "hold", 20, 0.2, false, 6],
    ["execution-sign", "flat", 20, 0.2, false, 1],
    ["execution-sign", "hold", 10, 0.1, false, 1],
    ["execution-sign", "hold", 20, 0.2, false, 1],
    ["execution-sign", "hold", 20, 0.2, true, 1],
    ["execution-sign", "hold", 20, 0.2, false, 2],
    ["execution-sign", "hold", 40, 0.2, false, 2],
    ["execution-sign", "hold", 20, 0.2, false, 3],
    ["execution-sign", "hold", 20, 0.2, false, 6],
    ["execution-sign", "hold", 20, 0.2, false, 12],
    ["horizon-mean", "flat", 20, 0.1, false, 1],
    ["horizon-mean", "flat", 40, 0.2, false, 1],
    ["horizon-mean", "hold", 10, 0.1, false, 1],
    ["horizon-mean", "hold", 20, 0.1, false, 1],
    ["horizon-mean", "hold", 40, 0.2, false, 1],
    ["horizon-mean", "hold", 80, 0.2, false, 1],
    ["horizon-mean", "hold", 20, 0.2, false, 2],
    ["horizon-mean", "hold", 40, 0.2, false, 2],
    ["horizon-mean", "hold", 20, 0.2, false, 3],
    ["horizon-mean", "hold", 20, 0.2, false, 6],
    ["horizon-mean", "hold", 20, 0.2, false, 12],
    ["horizon-median", "flat", 20, 0.1, false, 1],
    ["horizon-median", "flat", 40, 0.2, false, 1],
    ["horizon-median", "hold", 10, 0.1, false, 1],
    ["horizon-median", "hold", 20, 0.1, false, 1],
    ["horizon-median", "hold", 40, 0.2, false, 1],
    ["horizon-median", "hold", 80, 0.2, false, 1],
    ["horizon-median", "hold", 20, 0.2, false, 2],
    ["horizon-median", "hold", 40, 0.2, false, 2],
    ["horizon-median", "hold", 20, 0.4, false, 2],
    ["horizon-median", "hold", 20, 0.2, false, 3],
    ["horizon-median", "hold", 20, 0.2, false, 6],
    ["horizon-median", "hold", 20, 0.2, false, 12],
  ] as const;
  const policies: KronosBotPolicy[] = [];
  for (const [
    signalSource,
    filteredAction,
    minimumAbsoluteMeanReturnBps,
    minimumDirectionalConfidence,
    requireOracleReturnAgreement,
    minimumConsecutiveDirections,
  ] of filters) {
    const controlGrid = signalSource === "execution-oracle"
      || signalSource === "execution-utility"
      ? [
          [0.25, 0, 0.25, 0], [0.25, 0, 0.75, 0.5],
          [0.25, 1, 0.25, 0], [0.25, 1, 0.75, 0.5],
          [0.5, 0, 0.25, 0], [0.5, 0, 0.75, 0.5],
          [0.5, 1, 0.25, 0], [0.5, 1, 0.75, 0.5],
          [0.75, 0, 0.25, 0], [0.75, 0, 0.75, 0.5],
          [0.75, 1, 0.25, 0], [0.75, 1, 0.75, 0.5],
          [1, 0, 0.25, 0], [1, 0, 0.75, 0.5],
          [1, 1, 0.25, 0], [1, 1, 0.75, 0.5],
        ] as const
      : [
          // Point distributions have one feasible action and therefore
          // confidence=1. Their leverage floor cannot affect the exposure cap,
          // so varying it would duplicate every point-policy backtest.
          [0.25, 0, 0.25, 0],
          [0.5, 0, 0.25, 0],
          [0.75, 0, 0.25, 0],
          [1, 0, 0.25, 0],
        ] as const;
    for (const [
      staticConfidenceScale,
      confidenceExposurePower,
      confidenceLeverageFloor,
      expansionConfirmationMass,
    ] of controlGrid) {
      const expansionDeltaCapFractions = staticConfidenceScale === 1
        ? [0.75, 1] as const
        : [1] as const;
      for (const expansionDeltaCapFraction of expansionDeltaCapFractions) {
        policies.push({
          signalSource,
          filteredAction,
          minimumConsecutiveDirections,
          maximumLeverage,
          staticConfidenceScale,
          confidenceExposurePower,
          confidenceLeverageFloor,
          expansionConfirmationMass,
          expansionDeltaCapFraction,
          minimumAbsoluteMeanReturnBps,
          minimumDirectionalConfidence,
          requireOracleReturnAgreement,
        });
      }
    }
  }
  const calibratedFilters = [
    ["flat", 0, 1],
    ["flat", 10, 1],
    ["flat", 20, 1],
    ["hold", 0, 1],
    ["hold", 5, 1],
    ["hold", 10, 1],
    ["hold", 20, 1],
    ["hold", 40, 1],
    ["hold", 10, 2],
    ["hold", 20, 3],
    ["hold", 20, 6],
    ["hold", 20, 12],
  ] as const;
  for (const returnCalibrationId of returnCalibrationIds) {
    for (const [filteredAction, minimumAbsoluteMeanReturnBps, minimumConsecutiveDirections]
      of calibratedFilters) {
      for (const staticConfidenceScale of [0.25, 0.5, 0.75, 1] as const) {
        const expansionDeltaCapFractions = staticConfidenceScale === 1
          ? [0.75, 1] as const
          : [1] as const;
        for (const expansionDeltaCapFraction of expansionDeltaCapFractions) {
          policies.push({
            signalSource: "calibrated-return",
            returnCalibrationId,
            filteredAction,
            minimumConsecutiveDirections,
            maximumLeverage,
            staticConfidenceScale,
            confidenceExposurePower: 0,
            confidenceLeverageFloor: 0.25,
            expansionConfirmationMass: 0,
            expansionDeltaCapFraction,
            minimumAbsoluteMeanReturnBps,
            minimumDirectionalConfidence: 0,
            requireOracleReturnAgreement: false,
          });
        }
      }
    }
  }
  return policies;
}

function aggressiveSmokePolicy(maximumLeverage: number): KronosBotPolicy {
  return {
    signalSource: "execution-oracle",
    filteredAction: "flat",
    minimumConsecutiveDirections: 1,
    maximumLeverage,
    staticConfidenceScale: 1,
    confidenceExposurePower: 0,
    confidenceLeverageFloor: 1,
    expansionConfirmationMass: 0,
    expansionDeltaCapFraction: 1,
    minimumAbsoluteMeanReturnBps: 0,
    minimumDirectionalConfidence: 0,
    requireOracleReturnAgreement: false,
  };
}

export function distributionForPolicy(
  forecast: PreparedForecastRow,
  policy: KronosBotPolicy,
  state: PolicyFilterState,
  neutral: ExposureValueOracleActionDistribution,
  short: ExposureValueOracleActionDistribution,
  long: ExposureValueOracleActionDistribution,
): { distribution: ExposureValueOracleActionDistribution | null; filtered: boolean } {
  const row = forecast.row;
  const executionDistribution = policy.signalSource.startsWith("execution-utility")
    ? forecast.utilityDistribution
    : forecast.distribution;
  const selectedReturn = policy.signalSource === "calibrated-return"
    ? calibratedReturn(forecast, policy.returnCalibrationId!)
    : policy.signalSource === "horizon-median"
      ? row.horizonLogReturnMedian
      : row.horizonLogReturnMean;
  const meanReturnBps = Math.abs(selectedReturn) * 10_000;
  const directionalConfidence = 2 * Math.abs(row.horizonUpProbability - 0.5);
  const meanSign = Math.sign(selectedReturn) as -1 | 0 | 1;
  const oracleSign = Math.sign(executionDistribution.mean) as -1 | 0 | 1;
  const policyDirection = policy.signalSource.startsWith("execution-")
    ? oracleSign
    : meanSign;
  if (policyDirection === 0) {
    state.direction = 0;
    state.consecutiveDirections = 0;
  } else if (policyDirection === state.direction) {
    state.consecutiveDirections += 1;
  } else {
    state.direction = policyDirection;
    state.consecutiveDirections = 1;
  }
  const filtered = state.consecutiveDirections < policy.minimumConsecutiveDirections
    || meanReturnBps < policy.minimumAbsoluteMeanReturnBps
    || directionalConfidence < policy.minimumDirectionalConfidence
    || (policy.requireOracleReturnAgreement && meanSign !== oracleSign);
  return filtered
    ? {
        distribution: policy.filteredAction === "flat" ? neutral : null,
        filtered: true,
      }
    : {
        distribution: policy.signalSource === "execution-oracle"
          || policy.signalSource === "execution-utility"
          ? executionDistribution
          : policy.signalSource === "execution-mean"
            || policy.signalSource === "execution-utility-mean"
            ? pointDistribution(
                forecast.distribution.grid,
                executionDistribution.mean,
              )
            : policyDirection < 0
            ? short
            : policyDirection > 0
              ? long
              : neutral,
        filtered: false,
      };
}

function calibratedReturn(forecast: PreparedForecastRow, id: string): number {
  const value = forecast.calibratedReturns.get(id);
  if (value === undefined || !Number.isFinite(value)) {
    throw new Error(`Missing calibrated return ${id} for forecast ${forecast.row.decisionTime}.`);
  }
  return value;
}

function pointDistribution(
  grid: readonly number[] | Float64Array,
  target: number,
): ExposureValueOracleActionDistribution {
  let nearest = 0;
  for (let index = 1; index < grid.length; index += 1) {
    if (Math.abs(grid[index]! - target) < Math.abs(grid[nearest]! - target)) {
      nearest = index;
    }
  }
  const probabilities = new Float32Array(grid.length);
  probabilities[nearest] = 1;
  return probabilityDistribution(probabilities, Array.from(grid));
}

function neutralDistribution(grid: readonly number[]): ExposureValueOracleActionDistribution {
  const probabilities = new Float32Array(grid.length);
  probabilities[(grid.length - 1) / 2] = 1;
  return probabilityDistribution(probabilities, grid);
}

function directionalDistribution(
  grid: readonly number[],
  direction: -1 | 1,
): ExposureValueOracleActionDistribution {
  const probabilities = new Float32Array(grid.length);
  probabilities[direction < 0 ? 0 : grid.length - 1] = 1;
  return probabilityDistribution(probabilities, grid);
}

function prepareWindows(
  windows: readonly VwKamaInspectorWindow[],
  artifact: KronosForecastArtifact,
): PreparedWindow[] {
  const preparedRows = artifact.rows.map((row): PreparedForecastRow => ({
    row,
    distribution: probabilityDistribution(
      row.executionOracleProbabilities ?? row.oracleProbabilities,
      artifact.executionOracleGrid ?? artifact.oracleGrid,
    ),
    utilityDistribution: probabilityDistribution(
      row.executionUtilityProbabilities
        ?? row.executionOracleProbabilities
        ?? row.oracleProbabilities,
      artifact.executionOracleGrid ?? artifact.oracleGrid,
    ),
    calibratedReturns: new Map<string, number>(),
  }));
  return windows.map((window) => {
    const replayStart = window.startTime - STEP_MS;
    const warmupStart = replayStart - artifact.lookbackCandles * STEP_MS;
    const candles = loadCandleRange(warmupStart, window.endTime);
    const warmup = candles.filter((candle) => candle.openTime < replayStart);
    const replay = candles.filter((candle) => candle.openTime >= replayStart);
    assertContiguousCandles(replay, replayStart, window.endTime, window.id);
    if (warmup.length < artifact.lookbackCandles) {
      throw new Error(`${window.id} has insufficient one-minute warmup history.`);
    }
    const forecasts = new Map<number, PreparedForecastRow>();
    for (const forecast of preparedRows) {
      if (forecast.row.targetStartTime >= window.startTime
        && forecast.row.targetStartTime + HORIZON_MS <= window.endTime) {
        forecasts.set(forecast.row.decisionTime, forecast);
      }
    }
    if (forecasts.size === 0) throw new Error(`${window.id} has no forecast rows.`);
    return {
      window,
      warmup: warmup.slice(-artifact.lookbackCandles),
      replay,
      forecasts,
    };
  });
}

function returnCalibrationExamples(
  prepared: readonly PreparedWindow[],
): KronosReturnCalibrationExample[] {
  const examples: KronosReturnCalibrationExample[] = [];
  for (const episode of prepared) {
    const closeAt = new Map(episode.replay.map((candle) => [candle.closeTime, candle.close]));
    for (const forecast of episode.forecasts.values()) {
      const targetCloseTime = forecast.row.targetStartTime + HORIZON_MS - 1;
      const targetClose = closeAt.get(targetCloseTime);
      if (targetClose === undefined) {
        throw new Error(
          `${episode.window.id} is missing return-calibration target ${targetCloseTime}.`,
        );
      }
      examples.push({
        episodeId: episode.window.id,
        decisionTime: forecast.row.decisionTime,
        forecast,
        actualLogReturn: Math.log(targetClose / forecast.row.anchorPrice),
      });
    }
  }
  return examples;
}

function attachOofReturnCalibrations(
  prepared: readonly PreparedWindow[],
  calibrations: readonly CrossValidatedKronosReturnCalibration[],
): void {
  for (const episode of prepared) {
    for (const forecast of episode.forecasts.values()) {
      for (const calibration of calibrations) {
        const value = calibration.predictions.get(forecast.row.decisionTime);
        if (value === undefined) {
          throw new Error(
            `OOF return calibration ${calibration.id} misses ${forecast.row.decisionTime}.`,
          );
        }
        forecast.calibratedReturns.set(calibration.id, value);
      }
    }
  }
}

function attachFrozenReturnCalibration(
  prepared: readonly PreparedWindow[],
  calibration: FrozenKronosReturnCalibration,
): void {
  validateKronosReturnCalibration(calibration);
  for (const episode of prepared) {
    for (const forecast of episode.forecasts.values()) {
      forecast.calibratedReturns.set(
        calibration.id,
        predictKronosReturn(calibration, forecast),
      );
    }
  }
}

function directionalControlPreparedWindows(
  prepared: readonly PreparedWindow[],
  direction: -1 | 1,
): PreparedWindow[] {
  return prepared.map((episode) => ({
    ...episode,
    forecasts: new Map([...episode.forecasts].map(([timestamp, forecast]) => {
      const distribution = directionalDistribution(
        Array.from(forecast.distribution.grid),
        direction,
      );
      return [timestamp, {
        ...forecast,
        distribution,
        utilityDistribution: distribution,
        calibratedReturns: new Map(forecast.calibratedReturns),
      }];
    })),
  }));
}

function directionalControlPolicy(maximumLeverage: 1 | 5): KronosBotPolicy {
  return {
    signalSource: "execution-sign",
    filteredAction: "hold",
    minimumConsecutiveDirections: 1,
    maximumLeverage,
    staticConfidenceScale: 1,
    confidenceExposurePower: 0,
    confidenceLeverageFloor: 1,
    expansionConfirmationMass: 0,
    expansionDeltaCapFraction: 1,
    minimumAbsoluteMeanReturnBps: 0,
    minimumDirectionalConfidence: 0,
    requireOracleReturnAgreement: false,
  };
}

async function validationControls(
  prepared: readonly PreparedWindow[],
  artifact: KronosForecastArtifact,
): Promise<PhaseReport["controls"]> {
  const controls: PhaseReport["controls"] = [];
  for (const [direction, label] of [[1, "long"], [-1, "short"]] as const) {
    for (const leverage of [1, 5] as const) {
      const result = await runPolicy(
        directionalControlPreparedWindows(prepared, direction),
        artifact,
        directionalControlPolicy(leverage),
        false,
      );
      controls.push({
        id: `constant-${label}-${leverage}x`,
        aggregate: result.aggregate,
        episodes: result.windows,
      });
    }
  }
  return controls;
}

async function runPreparedWindow(
  prepared: PreparedWindow,
  artifact: KronosForecastArtifact,
  policy: KronosBotPolicy,
  captureFills: boolean,
): Promise<WindowReport> {
  validatePolicy(policy);
  const executionGrid = artifact.executionOracleGrid ?? artifact.oracleGrid;
  const neutral = neutralDistribution(executionGrid);
  const short = directionalDistribution(executionGrid, -1);
  const long = directionalDistribution(executionGrid, 1);
  const consumed = new Set<number>();
  const filtered = new Set<number>();
  const filterState: PolicyFilterState = { direction: 0, consecutiveDirections: 0 };
  const oracleDecisions: OracleBacktestDecision[] = [];
  const provider = (timestamp: number): ExposureValueOracleActionDistribution | null => {
    if (timestamp === prepared.window.endTime - 1) return neutral;
    const forecast = prepared.forecasts.get(timestamp);
    if (!forecast) return null;
    consumed.add(timestamp);
    const selected = distributionForPolicy(
      forecast,
      policy,
      filterState,
      neutral,
      short,
      long,
    );
    if (selected.filtered) filtered.add(timestamp);
    return selected.distribution;
  };
  const result = await runBotBacktestFromCandles(prepared.replay, {
    config: appConfig.strategy,
    strategy: "learned-oracle-1m",
    warmup: prepared.warmup,
    learnedOracleDistributionAt: provider,
    learnedOracleMaximumLeverage: policy.maximumLeverage,
    hindsightOracleConfidenceExposurePower: policy.confidenceExposurePower,
    hindsightOracleConfidenceLeverageFloor: policy.confidenceLeverageFloor,
    oracleStaticConfidenceScale: policy.staticConfidenceScale,
    oracleExpansionConfirmationMass: policy.expansionConfirmationMass,
    oracleExpansionDeltaCapFraction: policy.expansionDeltaCapFraction,
    summaryOnly: !captureFills,
    onOracleDecision: (decision) => oracleDecisions.push(decision),
  });
  validateForecastConsumption(
    consumed.size,
    prepared.forecasts.size,
    result.summary.liquidatedPositionCount,
    prepared.window.id,
  );
  if (captureFills && result.fills.length !== result.summary.tradeCount) {
    throw new Error(`${prepared.window.id} fill records do not match the trade count.`);
  }
  return {
    windowId: prepared.window.id,
    label: prepared.window.label,
    startTime: prepared.window.startTime,
    endTime: prepared.window.endTime,
    forecastRows: prepared.forecasts.size,
    forecastRowsConsumed: consumed.size,
    oracleDecisions: oracleDecisions.length,
    emittedSignals: oracleDecisions.filter((decision) => decision.signalEmitted).length,
    filteredForecasts: filtered.size,
    heldForecasts: policy.filteredAction === "hold" ? filtered.size : 0,
    flattenedForecasts: policy.filteredAction === "flat" ? filtered.size : 0,
    summary: compactSummary(result.summary),
    ...(captureFills ? { fills: result.fills.map(compactFill) } : {}),
  };
}

export function validateForecastConsumption(
  consumed: number,
  expected: number,
  liquidations: number,
  label: string,
): void {
  if (consumed !== expected && liquidations === 0) {
    throw new Error(`${label} consumed ${consumed}/${expected} forecasts.`);
  }
}

async function runPolicy(
  prepared: readonly PreparedWindow[],
  artifact: KronosForecastArtifact,
  policy: KronosBotPolicy,
  captureFills: boolean,
): Promise<{ aggregate: AggregateReport; windows: WindowReport[] }> {
  const reports: WindowReport[] = [];
  for (const window of prepared) {
    reports.push(await runPreparedWindow(window, artifact, policy, captureFills));
  }
  return { aggregate: aggregateReports(reports), windows: reports };
}

export function aggregateReports(reports: readonly WindowReport[]): AggregateReport {
  const startingQuote = appConfig.strategy.startingQuote;
  const logReturns = reports.map((report) =>
    Math.log(Math.max(Number.MIN_VALUE, report.summary.finalEquity / startingQuote)));
  const returns = reports.map((report) => report.summary.returnPct);
  const drawdowns = reports.map((report) => report.summary.maxDrawdownPct);
  const meanLogReturn = average(logReturns);
  const medianLogReturn = median(logReturns);
  const downsideRms = Math.sqrt(average(logReturns.map((value) => Math.min(0, value) ** 2)));
  const meanDrawdownFraction = average(drawdowns) / 100;
  const activeWindows = reports.filter((report) => report.summary.tradeCount > 0).length;
  const profitableWindows = reports.filter((report) => report.summary.netPnl > 0).length;
  const tradeCount = sum(reports.map((report) => report.summary.tradeCount));
  const liquidatedPositionCount = sum(
    reports.map((report) => report.summary.liquidatedPositionCount),
  );
  const netPnl = sum(reports.map((report) => report.summary.netPnl));
  const perfectMarginNetPnl = sum(
    reports.map((report) => report.summary.perfectMarginNetPnl ?? 0),
  );
  const minimumActivity = Math.max(1, Math.ceil(reports.length / 2));
  const minimumProfitableWindows = Math.max(1, Math.ceil(activeWindows / 2));
  const chronological = [...reports].sort((left, right) =>
    left.startTime - right.startTime || left.windowId.localeCompare(right.windowId));
  const chronologicalFoldCount = Math.min(4, chronological.length);
  const chronologicalFoldSize = Math.ceil(chronological.length / chronologicalFoldCount);
  const chronologicalFoldMeanLogReturns: number[] = [];
  for (let offset = 0; offset < chronological.length; offset += chronologicalFoldSize) {
    chronologicalFoldMeanLogReturns.push(average(
      chronological.slice(offset, offset + chronologicalFoldSize).map((report) =>
        Math.log(Math.max(Number.MIN_VALUE, report.summary.finalEquity / startingQuote))),
    ));
  }
  const worstChronologicalFoldMeanLogReturn = Math.min(...chronologicalFoldMeanLogReturns);
  const eligible = meanLogReturn > 0
    && activeWindows >= minimumActivity
    && profitableWindows >= minimumProfitableWindows
    && tradeCount >= reports.length
    && liquidatedPositionCount === 0;
  return {
    windows: reports.length,
    startingEquity: startingQuote * reports.length,
    finalEquity: sum(reports.map((report) => report.summary.finalEquity)),
    netPnl,
    returnPct: average(returns),
    geometricMeanReturnPct: (Math.exp(meanLogReturn) - 1) * 100,
    medianWindowReturnPct: median(returns),
    chronologicalFoldCount,
    worstChronologicalFoldReturnPct:
      (Math.exp(worstChronologicalFoldMeanLogReturn) - 1) * 100,
    perfectMarginNetPnl,
    meanPerfectMarginReturnPct: average(
      reports.map((report) => report.summary.perfectMarginReturnPct ?? 0),
    ),
    perfectMarginCapturePct: perfectMarginNetPnl !== 0
      ? netPnl / perfectMarginNetPnl * 100
      : null,
    profitableWindows,
    activeWindows,
    tradeCount,
    feesPaid: sum(reports.map((report) => report.summary.feesPaid ?? 0)),
    maintenancePaid: sum(reports.map((report) => report.summary.maintenancePaid ?? 0)),
    maximumDrawdownPct: Math.max(...drawdowns),
    meanDrawdownPct: average(drawdowns),
    liquidatedPositionCount,
    forecastRows: sum(reports.map((report) => report.forecastRows)),
    forecastRowsConsumed: sum(reports.map((report) => report.forecastRowsConsumed)),
    oracleDecisions: sum(reports.map((report) => report.oracleDecisions)),
    emittedSignals: sum(reports.map((report) => report.emittedSignals)),
    filteredForecasts: sum(reports.map((report) => report.filteredForecasts)),
    heldForecasts: sum(reports.map((report) => report.heldForecasts)),
    flattenedForecasts: sum(reports.map((report) => report.flattenedForecasts)),
    selectionEligible: eligible,
    selectionScore: eligible
      ? meanLogReturn - 0.5 * downsideRms - 0.1 * meanDrawdownFraction
        + 0.25 * Math.min(0, medianLogReturn)
        + 0.25 * Math.min(0, worstChronologicalFoldMeanLogReturn)
        - liquidatedPositionCount
      : -1e12,
  };
}

function compactSummary(summary: BacktestSummary): WindowReport["summary"] {
  return {
    finalEquity: summary.finalEquity,
    netPnl: summary.netPnl,
    returnPct: summary.returnPct,
    maxInitialBalanceDrawdownPct: summary.maxInitialBalanceDrawdownPct,
    maxDrawdownPct: summary.maxDrawdownPct,
    maxEffectiveLeverage: summary.maxEffectiveLeverage,
    perfectMarginNetPnl: summary.perfectMarginNetPnl,
    perfectMarginReturnPct: summary.perfectMarginReturnPct,
    perfectMarginCapturePct: summary.perfectMarginCapturePct,
    tradeCount: summary.tradeCount,
    feesPaid: summary.feesPaid,
    maintenancePaid: summary.maintenancePaid,
    winRate: summary.winRate,
    closedPositionCount: summary.closedPositionCount,
    profitableClosedPositionCount: summary.profitableClosedPositionCount,
    liquidatedPositionCount: summary.liquidatedPositionCount,
  };
}

function compactFill(fill: TradeFill): CompactFill {
  return {
    side: fill.side,
    price: fill.price,
    quantity: fill.quantity,
    quoteQuantity: fill.quoteQuantity,
    feeQuote: fill.feeQuote,
    realizedPnl: fill.realizedPnl,
    filledAt: fill.filledAt,
    reason: fill.reason,
    ...(fill.positionEffect === undefined ? {} : { positionEffect: fill.positionEffect }),
    ...(fill.liquidation === undefined ? {} : { liquidation: fill.liquidation }),
  };
}

function loadCandleRange(startTime: number, endTime: number): Candle[] {
  const candles: Candle[] = [];
  for (const date of utcDates(startTime, endTime)) {
    const file = path.join(HISTORY_ROOT, `${date}.json`);
    if (!fs.existsSync(file)) throw new Error(`Missing canonical one-minute history ${file}.`);
    candles.push(...readCandleShardReferenceSync(file).filter((candle) =>
      candle.openTime >= startTime && candle.openTime < endTime));
  }
  candles.sort((left, right) => left.openTime - right.openTime);
  return candles;
}

function utcDates(startTime: number, endTime: number): string[] {
  const output: string[] = [];
  let day = Math.floor(startTime / 86_400_000) * 86_400_000;
  const final = Math.floor((endTime - 1) / 86_400_000) * 86_400_000;
  while (day <= final) {
    output.push(new Date(day).toISOString().slice(0, 10));
    day += 86_400_000;
  }
  return output;
}

function assertContiguousCandles(
  candles: readonly Candle[],
  startTime: number,
  endTime: number,
  label: string,
): void {
  const expected = (endTime - startTime) / STEP_MS;
  if (candles.length !== expected) {
    throw new Error(`${label} has ${candles.length}/${expected} replay candles.`);
  }
  for (let index = 0; index < candles.length; index += 1) {
    const candle = candles[index]!;
    const openTime = startTime + index * STEP_MS;
    // Binance occasionally closes a bar early during an exchange interruption
    // (for example 2023-03-24 12:39 UTC), while the one-minute open-time grid
    // remains complete. Preserve that real close timestamp, but still reject
    // missing/duplicate opens and any close outside its own minute.
    if (candle.openTime !== openTime
      || candle.closeTime < openTime
      || candle.closeTime >= openTime + STEP_MS) {
      throw new Error(`${label} has non-contiguous one-minute candles at index ${index}.`);
    }
  }
}

async function main(): Promise<void> {
  const phase = choiceArgument(
    "phase",
    ["smoke", "diagnose", "confirm", "calibrate", "validate"] as const,
    "smoke",
  );
  const forecastFile = path.resolve(stringArgument(
    "forecasts",
    "data/benchmarks/kronos-base-ensemble-calibration-forecasts.json",
  ));
  const artifact = loadKronosForecastArtifact(forecastFile);
  const forecastArtifactSha256 = fileSha256(forecastFile);
  const executionMaximumLeverage = executionGridMaximum(artifact);
  const windows = kronosInspectorWindows();
  const calibrationWindows = windows.filter((window) => window.startTime < VALIDATION_CUTOFF);
  const validationWindows = windows.filter((window) => window.startTime >= VALIDATION_CUTOFF);
  if (calibrationWindows.length !== 20 || validationWindows.length !== 8) {
    throw new Error(
      `Expected a 20/8 chronological policy split; found ${calibrationWindows.length}/${validationWindows.length}.`,
    );
  }
  const calibrationEpisodes = mergeOverlappingWindows(calibrationWindows, "calibration");
  const validationEpisodes = mergeOverlappingWindows(validationWindows, "validation");
  const {
    selectionEpisodes,
    postTrainingConfirmationEpisodes,
  } = policyCalibrationEpisodeSplit(calibrationEpisodes);

  if (phase === "smoke") {
    const policy = aggressiveSmokePolicy(executionMaximumLeverage);
    const smokeWindow = calibrationWindows.find((window) => artifact.rows.some((row) => {
      if (!row.windowIds.includes(window.id)) return false;
      const distribution = probabilityDistribution(
        row.executionOracleProbabilities ?? row.oracleProbabilities,
        artifact.executionOracleGrid ?? artifact.oracleGrid,
      );
      return hindsightOracleTargetDecision(distribution, 0, 0.00175).targetExposure !== 0;
    }));
    if (!smokeWindow) {
      throw new Error("Smoke artifact contains no executable non-flat calibration decision.");
    }
    const smokeWindows = [smokeWindow];
    const result = await runPolicy(prepareWindows(smokeWindows, artifact), artifact, policy, true);
    if (result.aggregate.forecastRowsConsumed !== result.aggregate.forecastRows) {
      throw new Error("Smoke backtest did not consume every causal forecast row.");
    }
    if (result.aggregate.tradeCount === 0
      || result.windows.every((window) => (window.fills?.length ?? 0) === 0)) {
      throw new Error(
        "Smoke backtest produced no actual bot fills: "
        + JSON.stringify({ aggregate: result.aggregate, windows: result.windows }),
      );
    }
    const report = phaseReport(
      "smoke",
      forecastFile,
      artifact,
      policy,
      smokeWindows,
      result,
    );
    const output = path.resolve(stringArgument(
      "output",
      "data/benchmarks/kronos-bot-integration-smoke.json",
    ));
    atomicJson(output, report);
    console.log(JSON.stringify({ output, aggregate: report.aggregate }, null, 2));
    return;
  }

  if (phase === "diagnose") {
    requireExecutionOracle(artifact);
    const windowId = requiredStringArgument("window-id");
    const sourceWindow = calibrationWindows.find((candidate) => candidate.id === windowId);
    if (!sourceWindow) {
      throw new Error(
        `Diagnostic window must be one of the pre-validation calibration windows: ${windowId}.`,
      );
    }
    const partial = process.argv.includes("--partial");
    const matchingRows = artifact.rows.filter((row) => row.windowIds.includes(windowId));
    if (matchingRows.length === 0) throw new Error(`${windowId} has no diagnostic rows.`);
    const window = partial
      ? {
          ...sourceWindow,
          label: `${sourceWindow.label} (partial progress)`,
          endTime: Math.max(...matchingRows.map((row) => row.targetStartTime)) + HORIZON_MS,
        }
      : sourceWindow;
    validateDenseForecastCoverage(artifact, [window]);
    const prepared = prepareWindows([window], artifact);
    const policies = calibrationPolicies(executionMaximumLeverage);
    const ranked: Array<{
      policy: KronosBotPolicy;
      policyId: string;
      aggregate: AggregateReport;
    }> = [];
    for (let index = 0; index < policies.length; index += 1) {
      const policy = policies[index]!;
      const result = await runPolicy(prepared, artifact, policy, false);
      ranked.push({ policy, policyId: policyId(policy), aggregate: result.aggregate });
      if ((index + 1) % 50 === 0 || index + 1 === policies.length) {
        console.log(`DIAGNOSE ${index + 1}/${policies.length}`);
      }
    }
    ranked.sort((left, right) =>
      right.aggregate.selectionScore - left.aggregate.selectionScore
      || right.aggregate.geometricMeanReturnPct - left.aggregate.geometricMeanReturnPct
      || left.aggregate.maximumDrawdownPct - right.aggregate.maximumDrawdownPct);
    const sourceBestCandidates = [...new Set(policies.map((policy) => policy.signalSource))]
      .map((signalSource) => ranked.find((candidate) =>
        candidate.policy.signalSource === signalSource)!)
      .filter(Boolean);
    const selected = ranked[0]!;
    const selectedResult = await runPolicy(prepared, artifact, selected.policy, true);
    const controls = await validationControls(prepared, artifact);
    const output = path.resolve(stringArgument(
      "output",
      `data/benchmarks/kronos-bot-diagnostic-${window.id}-${artifact.runSignature.slice(0, 12)}.json`,
    ));
    const report = {
      version: 2,
      contract: "kronos-bot-policy-diagnostic-v2",
      createdAt: new Date().toISOString(),
      leakagePolicy:
        `Diagnostic-only replay of one pre-validation ${partial ? "partial " : ""}window; `
        + "never accepted as the frozen final policy artifact.",
      forecastRunSignature: artifact.runSignature,
      forecastArtifactSha256,
      forecastModelId: artifact.modelId,
      window,
      candidateCount: policies.length,
      selectedPolicy: selected.policy,
      selectedPolicyId: selected.policyId,
      result: selectedResult,
      controls,
      sourceBestCandidates,
      topCandidates: ranked.slice(0, 20),
    };
    atomicJson(output, report);
    console.log(JSON.stringify({
      output,
      selectedPolicy: selected.policy,
      aggregate: selectedResult.aggregate,
      controls: controls.map((control) => ({
        id: control.id,
        aggregate: control.aggregate,
      })),
    }, null, 2));
    return;
  }

  if (phase === "confirm") {
    requireExecutionOracle(artifact);
    const windowId = requiredStringArgument("window-id");
    const sourceWindow = calibrationWindows.find((candidate) => candidate.id === windowId);
    if (!sourceWindow) {
      throw new Error(
        `Temporal confirmation window must be a pre-validation calibration window: ${windowId}.`,
      );
    }
    const matchingRows = artifact.rows.filter((row) => row.windowIds.includes(windowId));
    if (matchingRows.length === 0) throw new Error(`${windowId} has no confirmation rows.`);
    const window = {
      ...sourceWindow,
      label: `${sourceWindow.label} (temporal confirmation)`,
      startTime: Math.min(...matchingRows.map((row) => row.targetStartTime)),
      endTime: Math.max(...matchingRows.map((row) => row.targetStartTime)) + HORIZON_MS,
    };
    validateDenseForecastCoverage(artifact, [window]);
    const policyFile = path.resolve(requiredStringArgument("policy"));
    const policySource = objectValue(
      JSON.parse(fs.readFileSync(policyFile, "utf8")),
      "diagnostic policy artifact",
    );
    if (policySource.contract !== "kronos-bot-policy-diagnostic-v2"
      || stringValue(policySource.forecastRunSignature, "forecastRunSignature")
        !== artifact.runSignature) {
      throw new Error("Temporal confirmation policy must come from this forecast run.");
    }
    const policyWindow = objectValue(policySource.window, "diagnostic policy window");
    if (stringValue(policyWindow.id, "diagnostic policy window id") !== windowId
      || finiteNumber(policyWindow.endTime, "diagnostic policy window endTime")
        > window.startTime) {
      throw new Error(
        "Temporal confirmation rows must start after the diagnostic policy-selection slice.",
      );
    }
    const policy = parsePolicy(policySource.selectedPolicy);
    const result = await runPolicy(prepareWindows([window], artifact), artifact, policy, true);
    const controls = await validationControls(prepareWindows([window], artifact), artifact);
    const output = path.resolve(stringArgument(
      "output",
      `data/benchmarks/kronos-bot-confirmation-${windowId}-${artifact.runSignature.slice(0, 12)}.json`,
    ));
    const report = {
      version: 1,
      contract: "kronos-bot-policy-temporal-confirmation-v1",
      createdAt: new Date().toISOString(),
      leakagePolicy:
        "Policy frozen on an earlier non-overlapping slice of the same calibration episode; "
        + "confirmation result cannot select or modify the final policy.",
      forecastRunSignature: artifact.runSignature,
      forecastArtifactSha256,
      forecastModelId: artifact.modelId,
      policySource: path.relative(REPO_ROOT, policyFile).replaceAll("\\", "/"),
      policySourceSha256: fileSha256(policyFile),
      selectedPolicy: policy,
      selectedPolicyId: policyId(policy),
      sourceWindow: policySource.window,
      confirmationWindow: window,
      result,
      controls,
    };
    atomicJson(output, report);
    console.log(JSON.stringify({
      output,
      selectedPolicy: policy,
      aggregate: result.aggregate,
      controls: controls.map((control) => ({
        id: control.id,
        aggregate: control.aggregate,
      })),
    }, null, 2));
    return;
  }

  if (phase === "calibrate") {
    requireExecutionOracle(artifact);
    validateDenseForecastCoverage(artifact, calibrationWindows);
    const selectionPrepared = prepareWindows(selectionEpisodes, artifact);
    const confirmationPrepared = prepareWindows(
      postTrainingConfirmationEpisodes,
      artifact,
    );
    const returnCalibrations = crossValidateKronosReturnCalibrations(
      returnCalibrationExamples(selectionPrepared),
      RETURN_CALIBRATION_LAMBDAS,
    );
    console.log("RETURN CALIBRATION OOF", JSON.stringify(
      returnCalibrations.map((calibration) => ({
        id: calibration.id,
        ridgeLambda: calibration.ridgeLambda,
        mse: calibration.oofMse,
        zeroMseSkill: calibration.oofZeroMseSkill,
        correlation: calibration.oofCorrelation,
        directionAccuracy: calibration.oofDirectionAccuracy,
      })),
    ));
    attachOofReturnCalibrations(selectionPrepared, returnCalibrations);
    for (const calibration of returnCalibrations) {
      attachFrozenReturnCalibration(confirmationPrepared, calibration.frozen);
    }
    const prepared = [...selectionPrepared, ...confirmationPrepared];
    const policies = calibrationPolicies(
      executionMaximumLeverage,
      returnCalibrations.map((calibration) => calibration.id),
    );
    const output = path.resolve(stringArgument(
      "output",
      `data/benchmarks/kronos-bot-policy-${artifact.runSignature.slice(0, 12)}.json`,
    ));
    const progressFile = `${output}.progress.json`;
    const policyGridSignature = crypto.createHash("sha256")
      .update(JSON.stringify({
        selectionContract: POLICY_SELECTION_CONTRACT,
        policies,
        selectionEpisodes,
        postTrainingConfirmationEpisodes,
        forecastArtifactSha256,
        returnCalibrations: returnCalibrations.map((calibration) => calibration.frozen),
      }))
      .digest("hex");
    const policyStart = optionalIntegerArgument("policy-start");
    const policyEnd = optionalIntegerArgument("policy-end");
    const shardMode = policyStart !== undefined || policyEnd !== undefined;
    if (shardMode && (policyStart === undefined || policyEnd === undefined
      || policyStart < 0 || policyStart >= policyEnd || policyEnd > policies.length)) {
      throw new Error(
        `Policy shard must satisfy 0 <= --policy-start < --policy-end <= ${policies.length}.`,
      );
    }
    const resumed = shardMode
      ? { completedCandidates: policyStart!, results: [] as KronosPolicyArtifact["topCandidates"] }
      : readCalibrationProgress(
          progressFile,
          artifact.runSignature,
          policyGridSignature,
          policies,
        );
    const ranked: KronosPolicyArtifact["topCandidates"] = resumed.results;
    const loopEnd = shardMode ? policyEnd! : policies.length;
    for (let index = resumed.completedCandidates; index < loopEnd; index += 1) {
      const policy = policies[index]!;
      const result = await runPolicy(prepared, artifact, policy, false);
      const selectionAggregate = aggregateReports(
        result.windows.filter((report) => report.endTime <= PREDICTOR_TRAINING_END),
      );
      const postTrainingConfirmation = aggregateReports(
        result.windows.filter((report) => report.startTime >= PREDICTOR_TRAINING_END),
      );
      const selection = confirmedPolicySelection(
        selectionAggregate,
        postTrainingConfirmation,
      );
      ranked.push({
        policy,
        policyId: policyId(policy),
        aggregate: selectionAggregate,
        postTrainingConfirmation,
        selectionEligible: selection.eligible,
        selectionScore: selection.score,
      });
      if (!shardMode && (index + 1) % 5 === 0 && index + 1 < policies.length) {
        atomicJson(progressFile, {
          version: 1,
          contract: "kronos-bot-policy-calibration-progress-v1",
          forecastRunSignature: artifact.runSignature,
          policyGridSignature,
          candidateCount: policies.length,
          completedCandidates: index + 1,
          results: ranked,
        });
      }
      if ((index + 1) % 10 === 0 || index + 1 === loopEnd) {
        console.log(
          `CALIBRATE ${index + 1}/${loopEnd} `
          + `best=${Math.max(...ranked.map((item) => item.selectionScore))}`,
        );
      }
    }
    if (shardMode) {
      atomicJson(output, {
        version: 1,
        contract: "foundation-bot-policy-calibration-shard-v1",
        createdAt: new Date().toISOString(),
        forecastRunSignature: artifact.runSignature,
        policyGridSignature,
        candidateCount: policies.length,
        start: policyStart,
        end: policyEnd,
        results: ranked,
      });
      console.log(`WROTE POLICY SHARD ${policyStart}-${policyEnd} ${output}`);
      return;
    }
    ranked.sort((left, right) =>
      right.selectionScore - left.selectionScore
      || right.aggregate.geometricMeanReturnPct - left.aggregate.geometricMeanReturnPct
      || left.aggregate.maximumDrawdownPct - right.aggregate.maximumDrawdownPct);
    const selected = ranked[0]!;
    if (!selected.selectionEligible) {
      throw new Error(
        "No calibration policy met both the broad and post-training real-trading gates.",
      );
    }
    const selectedResult = await runPolicy(prepared, artifact, selected.policy, false);
    const selectedSelectionReports = selectedResult.windows.filter((report) =>
      report.endTime <= PREDICTOR_TRAINING_END);
    const selectedConfirmationReports = selectedResult.windows.filter((report) =>
      report.startTime >= PREDICTOR_TRAINING_END);
    const selectedSelection = aggregateReports(selectedSelectionReports);
    const selectedConfirmation = aggregateReports(selectedConfirmationReports);
    if (!selectedSelection.selectionEligible || !selectedConfirmation.selectionEligible) {
      throw new Error("Selected policy failed its deterministic selection/confirmation replay.");
    }
    const preparedInspectorWindows = prepareWindows(calibrationWindows, artifact);
    const selectedReturnCalibration = selected.policy.signalSource === "calibrated-return"
      ? returnCalibrations.find((calibration) =>
          calibration.id === selected.policy.returnCalibrationId)?.frozen
      : null;
    if (selected.policy.signalSource === "calibrated-return"
      && selectedReturnCalibration === undefined) {
      throw new Error("Selected return calibration is missing from the frozen candidates.");
    }
    if (selectedReturnCalibration) {
      attachFrozenReturnCalibration(preparedInspectorWindows, selectedReturnCalibration);
    }
    const inspectorResult = await runPolicy(
      preparedInspectorWindows,
      artifact,
      selected.policy,
      false,
    );
    const policyArtifact: KronosPolicyArtifact = {
      version: 3,
      contract: POLICY_CONTRACT,
      createdAt: new Date().toISOString(),
      forecastRunSignature: artifact.runSignature,
      forecastArtifactSha256,
      forecastModelId: artifact.modelId,
      validationCutoff: VALIDATION_CUTOFF,
      calibrationWindowIds: calibrationWindows.map((window) => window.id),
      selectionEpisodes,
      postTrainingConfirmationEpisodes,
      candidateCount: policies.length,
      selectionContract: POLICY_SELECTION_CONTRACT,
      selectionObjective:
        "11 merged episodes ending before predictor training stopped rank candidates by mean log return - 0.5 downside RMS - 0.1 mean drawdown + 0.25 negative-median penalty + 0.25 negative-worst-chronological-fold penalty; requires positive geometric mean return, fills in at least half of episodes, profit in at least half of active episodes, at least one fill per episode on average, and zero liquidations. The two post-training episodes never affect ranking and independently apply the same complete eligibility gate. Calibrated-return candidates use episode-OOF predictions for ranking and a calibration frozen on those 11 episodes for confirmation and final validation",
      selectedPolicy: selected.policy,
      selectedPolicyId: selected.policyId,
      returnCalibration: selectedReturnCalibration ?? null,
      returnCalibrationCandidates: returnCalibrations.map((calibration) => ({
        id: calibration.id,
        ridgeLambda: calibration.ridgeLambda,
        oofMse: calibration.oofMse,
        oofZeroMseSkill: calibration.oofZeroMseSkill,
        oofCorrelation: calibration.oofCorrelation,
        oofDirectionAccuracy: calibration.oofDirectionAccuracy,
      })),
      selection: selectedSelection,
      selectionEpisodeReports: selectedSelectionReports,
      postTrainingConfirmation: selectedConfirmation,
      calibration: inspectorResult.aggregate,
      calibrationWindows: inspectorResult.windows,
      topCandidates: ranked.slice(0, 20),
    };
    atomicJson(output, policyArtifact);
    if (fs.existsSync(progressFile)) fs.unlinkSync(progressFile);
    console.log(JSON.stringify({
      output,
      selectedPolicy: selected.policy,
      selectionAggregate: selected.aggregate,
      inspectorAggregate: inspectorResult.aggregate,
    }, null, 2));
    return;
  }

  requireExecutionOracle(artifact);
  validateDenseForecastCoverage(artifact, validationWindows);
  const policyFile = path.resolve(requiredStringArgument("policy"));
  const policyArtifact = parsePolicyArtifact(JSON.parse(fs.readFileSync(policyFile, "utf8")));
  if (policyArtifact.forecastRunSignature !== artifact.runSignature) {
    throw new Error("Frozen policy was calibrated against a different forecast artifact.");
  }
  if (policyArtifact.forecastArtifactSha256 !== forecastArtifactSha256) {
    throw new Error("Frozen policy forecast hash does not match the supplied artifact bytes.");
  }
  if (policyArtifact.validationCutoff !== VALIDATION_CUTOFF
    || JSON.stringify(policyArtifact.calibrationWindowIds)
      !== JSON.stringify(calibrationWindows.map((window) => window.id))) {
    throw new Error("Frozen policy does not match the chronological calibration split.");
  }
  if (JSON.stringify(policyArtifact.selectionEpisodes)
      !== JSON.stringify(selectionEpisodes)
    || JSON.stringify(policyArtifact.postTrainingConfirmationEpisodes)
      !== JSON.stringify(postTrainingConfirmationEpisodes)
    || !policyArtifact.selection.selectionEligible
    || !policyArtifact.postTrainingConfirmation.selectionEligible) {
    throw new Error(
      "Frozen policy did not pass the broad and post-training calibration gates.",
    );
  }
  const canonicalValidationOutput = path.resolve(
    `data/benchmarks/kronos-bot-validation-${artifact.runSignature.slice(0, 12)}.json`,
  );
  const output = path.resolve(stringArgument(
    "output",
    path.relative(REPO_ROOT, canonicalValidationOutput),
  ));
  if (output !== canonicalValidationOutput) {
    throw new Error(
      `Validation output is fixed by the forecast run signature: ${canonicalValidationOutput}.`,
    );
  }
  if (fs.existsSync(output)) {
    throw new Error(
      `Validation output already exists: ${output}. Refusing to silently repeat the frozen test.`,
    );
  }
  const preparedValidationEpisodes = prepareWindows(validationEpisodes, artifact);
  if (policyArtifact.returnCalibration !== null) {
    attachFrozenReturnCalibration(
      preparedValidationEpisodes,
      policyArtifact.returnCalibration,
    );
  }
  const result = await runPolicy(
    preparedValidationEpisodes,
    artifact,
    policyArtifact.selectedPolicy,
    true,
  );
  const controls = await validationControls(preparedValidationEpisodes, artifact);
  const preparedValidationWindows = prepareWindows(validationWindows, artifact);
  if (policyArtifact.returnCalibration !== null) {
    attachFrozenReturnCalibration(
      preparedValidationWindows,
      policyArtifact.returnCalibration,
    );
  }
  const inspectorResult = await runPolicy(
    preparedValidationWindows,
    artifact,
    policyArtifact.selectedPolicy,
    false,
  );
  const report = phaseReport(
    "validation",
    forecastFile,
    artifact,
    policyArtifact.selectedPolicy,
    validationWindows,
    result,
    inspectorResult,
    policyArtifact.returnCalibration,
    controls,
    policyFile,
  );
  atomicJson(output, report);
  console.log(JSON.stringify({ output, aggregate: report.aggregate }, null, 2));
}

function phaseReport(
  phase: PhaseReport["phase"],
  forecastFile: string,
  artifact: KronosForecastArtifact,
  policy: KronosBotPolicy,
  windows: readonly VwKamaInspectorWindow[],
  result: { aggregate: AggregateReport; windows: WindowReport[] },
  inspectorResult: { aggregate: AggregateReport; windows: WindowReport[] } = result,
  returnCalibration: FrozenKronosReturnCalibration | null = null,
  controls: PhaseReport["controls"] = [],
  policyArtifactFile: string | null = null,
): PhaseReport {
  const bestControl = [...controls].sort((left, right) =>
    right.aggregate.geometricMeanReturnPct - left.aggregate.geometricMeanReturnPct)[0];
  return {
    version: 3,
    contract: "kronos-bot-backtest-v3",
    createdAt: new Date().toISOString(),
    phase,
    forecast: {
      file: path.relative(REPO_ROOT, forecastFile).replaceAll("\\", "/"),
      sha256: fileSha256(forecastFile),
      runSignature: artifact.runSignature,
      modelId: artifact.modelId,
      sampleCount: artifact.sampleCount,
      temperature: artifact.temperature,
    },
    policyArtifact: policyArtifactFile === null
      ? null
      : {
          file: path.relative(REPO_ROOT, policyArtifactFile).replaceAll("\\", "/"),
          sha256: fileSha256(policyArtifactFile),
        },
    policy,
    policyId: policyId(policy),
    returnCalibration,
    split: {
      validationCutoff: VALIDATION_CUTOFF,
      windowIds: windows.map((window) => window.id),
      leakagePolicy: phase === "validation"
        ? "11 pretraining episodes ranked policies with episode-OOF learned signals; two post-training/pre-validation episodes only gated the already-ranked policy and frozen coefficients; no later-window bot PnL was used (forecast-model metrics were inspected earlier)"
        : "integration-only smoke on one pre-cutoff window; no policy selection",
    },
    execution: {
      feeBps: appConfig.strategy.feeBps,
      marketSlippageBps: appConfig.strategy.positionRisk.marketSlippageBps,
      borrowBpsHour: 10,
      closeAtWindowEnd: true,
      perWindowIndependentStartingQuote: appConfig.strategy.startingQuote,
    },
    gates: {
      allForecastsConsumed:
        result.aggregate.forecastRowsConsumed === result.aggregate.forecastRows,
      actualFills: result.aggregate.tradeCount > 0,
      noLiquidations: result.aggregate.liquidatedPositionCount === 0,
      positiveNetPnl: result.aggregate.netPnl > 0,
    },
    controls,
    controlComparison: bestControl
      ? {
          bestControlId: bestControl.id,
          bestControlReturnPct: bestControl.aggregate.geometricMeanReturnPct,
          selectedMinusBestControlPct:
            result.aggregate.geometricMeanReturnPct
            - bestControl.aggregate.geometricMeanReturnPct,
        }
      : {
          bestControlId: null,
          bestControlReturnPct: null,
          selectedMinusBestControlPct: null,
        },
    aggregate: result.aggregate,
    episodes: result.windows,
    inspectorAggregate: inspectorResult.aggregate,
    windows: inspectorResult.windows,
  };
}

function parsePolicyArtifact(source: unknown): KronosPolicyArtifact {
  const root = objectValue(source, "policy artifact");
  if (root.version !== 3 || root.contract !== POLICY_CONTRACT) {
    throw new Error(`Policy artifact must use ${POLICY_CONTRACT} version 3.`);
  }
  if (root.selectionContract !== POLICY_SELECTION_CONTRACT) {
    throw new Error(
      `Policy artifact must use selection contract ${POLICY_SELECTION_CONTRACT}.`,
    );
  }
  const policy = parsePolicy(root.selectedPolicy);
  const returnCalibration = parseReturnCalibration(root.returnCalibration);
  if ((policy.signalSource === "calibrated-return")
    !== (returnCalibration !== null)
    || (returnCalibration !== null
      && returnCalibration.id !== policy.returnCalibrationId)) {
    throw new Error("Frozen return calibration does not match the selected policy.");
  }
  const selectedPolicyId = stringValue(root.selectedPolicyId, "selectedPolicyId");
  if (selectedPolicyId !== policyId(policy)) {
    throw new Error("selectedPolicyId does not match the frozen policy parameters.");
  }
  return {
    ...(root as unknown as KronosPolicyArtifact),
    forecastRunSignature: stringValue(
      root.forecastRunSignature,
      "forecastRunSignature",
    ),
    forecastArtifactSha256: sha256Value(
      root.forecastArtifactSha256,
      "forecastArtifactSha256",
    ),
    forecastModelId: stringValue(root.forecastModelId, "forecastModelId"),
    validationCutoff: integerNumber(root.validationCutoff, "validationCutoff"),
    calibrationWindowIds: stringArray(
      root.calibrationWindowIds,
      "calibrationWindowIds",
    ),
    candidateCount: integerNumber(root.candidateCount, "candidateCount"),
    selectedPolicy: policy,
    selectedPolicyId,
    returnCalibration,
  };
}

function readCalibrationProgress(
  file: string,
  forecastRunSignature: string,
  policyGridSignature: string,
  policies: readonly KronosBotPolicy[],
): {
  completedCandidates: number;
  results: KronosPolicyArtifact["topCandidates"];
} {
  if (!fs.existsSync(file)) return { completedCandidates: 0, results: [] };
  const root = objectValue(JSON.parse(fs.readFileSync(file, "utf8")), "calibration progress");
  if (root.version !== 1
    || root.contract !== "kronos-bot-policy-calibration-progress-v1"
    || root.forecastRunSignature !== forecastRunSignature
    || root.policyGridSignature !== policyGridSignature
    || root.candidateCount !== policies.length) {
    throw new Error(`Calibration progress does not match this run: ${file}.`);
  }
  const completedCandidates = integerNumber(
    root.completedCandidates,
    "completedCandidates",
  );
  if (completedCandidates < 0 || completedCandidates > policies.length
    || !Array.isArray(root.results)
    || root.results.length !== completedCandidates) {
    throw new Error(`Calibration progress is incomplete or corrupt: ${file}.`);
  }
  const results = root.results as KronosPolicyArtifact["topCandidates"];
  for (let index = 0; index < results.length; index += 1) {
    if (results[index]!.policyId !== policyId(policies[index]!)) {
      throw new Error(`Calibration progress policy order changed at candidate ${index}.`);
    }
  }
  console.log(`RESUME CALIBRATION ${completedCandidates}/${policies.length} from ${file}`);
  return { completedCandidates, results };
}

function parsePolicy(source: unknown): KronosBotPolicy {
  const root = objectValue(source, "selectedPolicy");
  const signalSource = stringValue(root.signalSource, "signalSource");
  const policy: KronosBotPolicy = {
    signalSource: signalSource as KronosBotPolicy["signalSource"],
    ...(root.returnCalibrationId === undefined
      ? {}
      : { returnCalibrationId: stringValue(
          root.returnCalibrationId,
          "returnCalibrationId",
        ) }),
    filteredAction: stringValue(
      root.filteredAction,
      "filteredAction",
    ) as KronosBotPolicy["filteredAction"],
    minimumConsecutiveDirections: integerNumber(
      root.minimumConsecutiveDirections,
      "minimumConsecutiveDirections",
    ),
    maximumLeverage: finiteNumber(root.maximumLeverage, "maximumLeverage"),
    staticConfidenceScale: finiteNumber(
      root.staticConfidenceScale,
      "staticConfidenceScale",
    ),
    confidenceExposurePower: finiteNumber(
      root.confidenceExposurePower,
      "confidenceExposurePower",
    ),
    confidenceLeverageFloor: finiteNumber(
      root.confidenceLeverageFloor,
      "confidenceLeverageFloor",
    ),
    expansionConfirmationMass: finiteNumber(
      root.expansionConfirmationMass,
      "expansionConfirmationMass",
    ),
    expansionDeltaCapFraction: finiteNumber(
      root.expansionDeltaCapFraction,
      "expansionDeltaCapFraction",
    ),
    minimumAbsoluteMeanReturnBps: finiteNumber(
      root.minimumAbsoluteMeanReturnBps,
      "minimumAbsoluteMeanReturnBps",
    ),
    minimumDirectionalConfidence: finiteNumber(
      root.minimumDirectionalConfidence,
      "minimumDirectionalConfidence",
    ),
    requireOracleReturnAgreement: root.requireOracleReturnAgreement as boolean,
  };
  validatePolicy(policy);
  return policy;
}

function parseReturnCalibration(source: unknown): FrozenKronosReturnCalibration | null {
  if (source === null) return null;
  const root = objectValue(source, "returnCalibration");
  const calibration: FrozenKronosReturnCalibration = {
    version: integerNumber(root.version, "returnCalibration.version") as 1,
    contract: stringValue(
      root.contract,
      "returnCalibration.contract",
    ) as FrozenKronosReturnCalibration["contract"],
    id: stringValue(root.id, "returnCalibration.id"),
    ridgeLambda: finiteNumber(root.ridgeLambda, "returnCalibration.ridgeLambda"),
    featureNames: stringArray(root.featureNames, "returnCalibration.featureNames"),
    featureMeans: numberArray(root.featureMeans, "returnCalibration.featureMeans"),
    featureScales: numberArray(root.featureScales, "returnCalibration.featureScales"),
    coefficients: numberArray(root.coefficients, "returnCalibration.coefficients"),
    trainingExamples: integerNumber(
      root.trainingExamples,
      "returnCalibration.trainingExamples",
    ),
    trainingEpisodeIds: stringArray(
      root.trainingEpisodeIds,
      "returnCalibration.trainingEpisodeIds",
    ),
    residualStd: finiteNumber(root.residualStd, "returnCalibration.residualStd"),
  };
  validateKronosReturnCalibration(calibration);
  return calibration;
}

type AtomicReplace = (temporary: string, destination: string) => void;

export function atomicJson(
  file: string,
  value: unknown,
  replace: AtomicReplace = fs.renameSync,
  wait: (milliseconds: number) => void = waitForMilliseconds,
): void {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const temporary = `${file}.${process.pid}.tmp`;
  fs.writeFileSync(temporary, `${JSON.stringify(value, jsonNumber, 2)}\n`);
  try {
    for (let attempt = 0; ; attempt += 1) {
      try {
        replace(temporary, file);
        return;
      } catch (error) {
        if (!isTransientAtomicReplaceError(error) || attempt >= 11) throw error;
        wait(Math.min(1_000, 50 * 2 ** attempt));
      }
    }
  } finally {
    if (fs.existsSync(temporary)) fs.unlinkSync(temporary);
  }
}

function isTransientAtomicReplaceError(error: unknown): boolean {
  if (!(error instanceof Error) || !("code" in error)) return false;
  return ["EACCES", "EBUSY", "EPERM"].includes(String(error.code));
}

function waitForMilliseconds(milliseconds: number): void {
  Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, milliseconds);
}

function fileSha256(file: string): string {
  return crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex");
}

function sha256Value(value: unknown, label: string): string {
  const result = stringValue(value, label);
  if (!/^[a-f0-9]{64}$/.test(result)) throw new Error(`${label} must be a SHA-256 digest.`);
  return result;
}

function jsonNumber(_key: string, value: unknown): unknown {
  return typeof value === "number" && !Number.isFinite(value) ? null : value;
}

function objectValue(value: unknown, label: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${label} must be an object.`);
  }
  return value as Record<string, unknown>;
}

function validateOracleGrid(grid: readonly number[], label: string): void {
  if (grid.length < 3 || grid.length % 2 !== 1) {
    throw new Error(`${label} must contain an odd number of at least three actions.`);
  }
  for (let index = 1; index < grid.length; index += 1) {
    if (!(grid[index]! > grid[index - 1]!)) {
      throw new Error(`${label} must be strictly increasing.`);
    }
  }
  if (Math.abs(grid[(grid.length - 1) / 2]!) > 1e-12) {
    throw new Error(`${label} must have zero at its center.`);
  }
}

function requireExecutionOracle(artifact: KronosForecastArtifact): void {
  if (!artifact.executionOracleGrid
    || artifact.rows.some((row) => row.executionOracleProbabilities === undefined
      || row.executionUtilityProbabilities === undefined)) {
    throw new Error(
      "Dense bot calibration/validation requires the executable -5x..+5x "
      + "oracle-vote and expected-utility exports.",
    );
  }
}

function executionGridMaximum(artifact: KronosForecastArtifact): number {
  const grid = artifact.executionOracleGrid ?? artifact.oracleGrid;
  return Math.max(Math.abs(grid[0]!), Math.abs(grid.at(-1)!));
}

function stringValue(value: unknown, label: string): string {
  if (typeof value !== "string" || value.trim() === "") {
    throw new Error(`${label} must be a non-empty string.`);
  }
  return value;
}

function finiteNumber(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${label} must be finite.`);
  }
  return value;
}

function integerNumber(value: unknown, label: string): number {
  const result = finiteNumber(value, label);
  if (!Number.isSafeInteger(result)) throw new Error(`${label} must be a safe integer.`);
  return result;
}

function positiveNumber(value: unknown, label: string): number {
  const result = finiteNumber(value, label);
  if (!(result > 0)) throw new Error(`${label} must be positive.`);
  return result;
}

function nonNegativeNumber(value: unknown, label: string): number {
  const result = finiteNumber(value, label);
  if (result < 0) throw new Error(`${label} must be non-negative.`);
  return result;
}

function unitNumber(value: unknown, label: string): number {
  const result = finiteNumber(value, label);
  if (result < 0 || result > 1) throw new Error(`${label} must be in [0, 1].`);
  return result;
}

function numberArray(value: unknown, label: string): number[] {
  if (!Array.isArray(value)) throw new Error(`${label} must be an array.`);
  return value.map((item, index) => finiteNumber(item, `${label}[${index}]`));
}

function stringArray(value: unknown, label: string): string[] {
  if (!Array.isArray(value)) throw new Error(`${label} must be an array.`);
  return value.map((item, index) => stringValue(item, `${label}[${index}]`));
}

function sum(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0);
}

function average(values: readonly number[]): number {
  return values.length > 0 ? sum(values) / values.length : 0;
}

function median(values: readonly number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1
    ? sorted[middle]!
    : (sorted[middle - 1]! + sorted[middle]!) / 2;
}

function argument(name: string): string | undefined {
  const prefix = `--${name}=`;
  return process.argv.find((item) => item.startsWith(prefix))?.slice(prefix.length);
}

function stringArgument(name: string, fallback: string): string {
  return argument(name)?.trim() || fallback;
}

function requiredStringArgument(name: string): string {
  const value = argument(name)?.trim();
  if (!value) throw new Error(`--${name}=... is required.`);
  return value;
}

function optionalIntegerArgument(name: string): number | undefined {
  const value = argument(name);
  if (value === undefined) return undefined;
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed)) throw new Error(`--${name} must be a safe integer.`);
  return parsed;
}

function choiceArgument<const T extends readonly string[]>(
  name: string,
  choices: T,
  fallback: T[number],
): T[number] {
  const value = argument(name) ?? fallback;
  if (!choices.includes(value)) {
    throw new Error(`--${name} must be one of ${choices.join(", ")}.`);
  }
  return value as T[number];
}

if (process.argv[1]
  && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  void main().catch((error: unknown) => {
    console.error(error);
    process.exitCode = 1;
  });
}
