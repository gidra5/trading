import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath, pathToFileURL } from "node:url";
import { fork } from "node:child_process";
import { availableParallelism, cpus } from "node:os";
import {
  isMainThread,
  parentPort,
  Worker,
  workerData,
} from "node:worker_threads";
import {
  exposureValueOracleBytes,
  prepareExposureValueOracle,
  shareExposureValueOracle,
  strategyExposureTemperatures,
  type ExposureValueOracle,
} from "../packages/bot-algo/src/exposure-value-distillation.js";
import {
  columnarVwKamaCandles,
  evaluateVwKamaOracle,
  prepareVwKamaOracle,
  resolveVwKamaValueHorizonSteps,
  VW_KAMA_SCORE_VERSION,
  vwKamaParametersFromPeakValleySignal,
  vwKamaScore,
  type VwKamaCandleColumns,
  type VwKamaCandleSeries,
  type VwKamaPreparedOracle,
  type VwKamaPreset,
  type VwKamaValueHorizonMode,
} from "../packages/bot-algo/src/kama-signal-evaluator.js";
import { createPeakValleyStrategyConfig } from "../packages/bot-algo/src/peak-valley-strategy.js";
import { volumeWeightedKamaWarmupSamples } from "../packages/bot-algo/src/indicators.js";
import { perfectMarginOracle } from "../packages/bot-algo/src/perfect-margin-oracle.js";
import type { PerfectMarginOracleResult } from "../packages/bot-algo/src/perfect-margin-oracle.js";
import type { TradingCandle } from "../packages/bot-algo/src/trading-api.js";
import {
  evaluateVwKamaCudaBatch,
  prepareExposureValueOracleCuda,
  VW_KAMA_CUDA_VALUE_ORACLE_AUTO_MIN_GRID_SIZE,
  vwKamaCudaStatus,
} from "../packages/bot-algo/src/vw-kama-cuda.js";

type Family = "volume" | "canonical";
type Stage = "fit" | "validation" | "test";
type DeadbandMode = "flat" | "hold" | "hysteresis";
type AgreementMode = "sizing" | "confidence";
const CONFIRMATION_MASKS = [
  "base", "acceleration", "ema", "rsi", "dmi",
  "acceleration+ema", "acceleration+rsi", "acceleration+dmi",
  "ema+rsi", "ema+dmi", "rsi+dmi",
  "acceleration+ema+rsi", "acceleration+ema+dmi", "acceleration+rsi+dmi", "ema+rsi+dmi",
  "all",
] as const;
type ConfirmationMask = typeof CONFIRMATION_MASKS[number];
type SearchAlgorithm = "random" | "de";
type SearchMode = "standard" | "per-window";
type Accelerator = "auto" | "cpu" | "cuda";
type SearchObjective = "signal" | "value-distillation";
type ScoreVersion = typeof VW_KAMA_SCORE_VERSION;
type ValueOracleBackend = "cpu" | "cuda";

interface Window { label: string; start: number; end: number }
interface Range { min: number; max: number }
interface Args {
  sourceDir: string;
  sourceIntervalMs: number;
  scales: number[];
  fitWindows?: Window[];
  validationWindows?: Window[];
  testWindows?: Window[];
  trials: number;
  algorithm: SearchAlgorithm;
  mode: SearchMode;
  generations: number;
  restarts: number;
  fullEvaluationInterval: number;
  refinementRounds: number;
  pbestFraction: number;
  confirmationMasks: ConfirmationMask[];
  differentialWeight: number;
  crossoverRate: number;
  immigrantRate: number;
  screenWindows: number;
  screenScales: number[];
  workers: number;
  accelerator: Accelerator;
  objective: SearchObjective;
  scoreVersion: ScoreVersion;
  exposureGridSize: number;
  exposureMinimum: number;
  exposureMaximum: number;
  valueHorizonMode: VwKamaValueHorizonMode;
  valueHorizonMs: number;
  oracleTemperature: number;
  strategyTemperature: number;
  strategyVolatilityScaling: boolean;
  opportunityEpsilon: number;
  quoteLendRate: number;
  quoteBorrowRate: number;
  assetBorrowRate: number;
  seedCandidatePaths: string[];
  presetWindowIds?: string[];
  presetOutputPath: string;
  top: number;
  seed: number;
  oracleFriction: number;
  matchWindowMs: number;
  timingHalfLifeMs: number;
  warmupMultiple: number;
  caseWarmupMs: number;
  efficiency: Range;
  efficiencyVolumeEma: Range;
  efficiencyVolumePower: Range;
  fast: Range;
  slow: Range;
  volume: Range;
  power: Range;
  volumeCap: Range;
  volumePower: Range;
  deadbandBpsHour: Range;
  hysteresisReleaseRatio: Range;
  thresholdLookback: Range;
  thresholdNoiseMultiplier: Range;
  buyMaxFraction: Range;
  sellMaxFraction: Range;
  buySizingSigma: Range;
  sellSizingSigma: Range;
  agreementModes: AgreementMode[];
  deadbandModes: DeadbandMode[];
  confirmationMix: Range;
  confirmationMinQuality: Range;
  confirmationAccelerationLookback: Range;
  confirmationDistanceLookback: Range;
  confirmationAccelerationWeight: Range;
  confirmationDistanceWeight: Range;
  confirmationBias: Range;
  confirmationEma: Range;
  confirmationEmaThreshold: Range;
  confirmationEmaWeight: Range;
  confirmationEmaGateStrength: Range;
  confirmationRsi: Range;
  confirmationRsiThreshold: Range;
  confirmationRsiWeight: Range;
  confirmationDmi: Range;
  confirmationDmiWeight: Range;
  confirmationAdxThreshold: Range;
  meanReversionSuppressionThreshold: Range;
  meanReversionEfficiency: Range;
  meanReversionFast: Range;
  meanReversionSlow: Range;
  meanReversionVolatility: Range;
  meanReversionReversalThreshold: Range;
  outputPath: string;
  reportPath: string;
}

interface Candidate {
  id: string;
  family: Family;
  efficiencyMs: number;
  efficiencyVolumeEmaMs: number;
  efficiencyVolumePower: number;
  fastMs: number;
  slowMs: number;
  power: number;
  volumeMs: number;
  volumeCap: number;
  volumePower: number;
  rateMode?: "relative" | "log";
  rateEmaMs?: number;
  deadbandBpsHour: number;
  deadbandMode: DeadbandMode;
  hysteresisReleaseRatio: number;
  thresholdLookbackMs: number;
  thresholdNoiseResponse?: "proportional" | "inverse";
  thresholdNoiseMultiplier: number;
  thresholdInverseMaxBpsHour?: number;
  thresholdInverseNoiseScaleBpsHour?: number;
  signalFrictionFraction?: number;
  buyMaxFraction: number;
  sellMaxFraction: number;
  buySizingSigmaBpsHour: number;
  sellSizingSigmaBpsHour: number;
  agreementMode: AgreementMode;
  confirmationMix: number;
  confirmationMinQuality: number;
  confirmationAccelerationLookbackMs: number;
  confirmationDistanceLookbackMs: number;
  confirmationAccelerationWeight: number;
  confirmationDistanceWeight: number;
  confirmationBias: number;
  confirmationEmaMs: number;
  confirmationEmaThresholdBpsHour: number;
  confirmationEmaWeight: number;
  confirmationEmaGateStrength: number;
  confirmationRsiMs: number;
  confirmationRsiThreshold: number;
  confirmationRsiWeight: number;
  confirmationDmiMs: number;
  confirmationDmiWeight: number;
  confirmationAdxThreshold: number;
  meanReversionSuppressionThreshold: number;
  meanReversionEfficiencyMs: number;
  meanReversionFastMs: number;
  meanReversionSlowMs: number;
  meanReversionVolatilityMs: number;
  meanReversionReversalThreshold: number;
}

type Genome = Omit<Candidate, "id" | "family" | "volumePower"> & { volumePower: number };

interface CaseData {
  id: string;
  stage: Stage;
  scaleMs: number;
  window: Window;
  candles: VwKamaCandleSeries;
  scoreStart: number;
  oracle?: PerfectMarginOracleResult;
  preparedOracle?: VwKamaPreparedOracle;
  valueOracle?: ExposureValueOracle;
  strategyTemperatures?: Float32Array;
  valueHorizonMs?: number;
  days: number;
}

interface PreparedStageWindow {
  key: string;
  sourceCount: number;
  segmentCount: number;
  cases: CaseData[];
  bytes: number;
  valueOracleBackend: ValueOracleBackend;
  valueOracleKernelMs: number;
}

interface CaseScore {
  caseId: string;
  scale: string;
  window: string;
  score: number;
  signalScore: number;
  valueDistillationScore: number | null;
  valueDistillationLoss: number | null;
  valueDistillationKl: number | null;
  oracleEntropy: number | null;
  meanOpportunity: number | null;
  valueHorizonMs: number | null;
  strategyReturn: number | null;
  oracleReturn: number | null;
  strategyMaxDrawdown: number | null;
  oracleMaxDrawdown: number | null;
  strategyTurnover: number | null;
  oracleTurnover: number | null;
  precision: number;
  recall: number;
  f1: number;
  rawPrecision: number;
  rawRecall: number;
  exposureAgreement: number;
  noiseSignalRatio: number | null;
  signalCleanliness: number;
  signalsPerDay: number;
  lagP50Ms: number | null;
  lagP90Ms: number | null;
  lagP95Ms: number | null;
  lagMedianSignedMs: number | null;
}

type FitnessCache = Map<string, CaseScore[]>;

interface CaseStats {
  caseId: string;
  scale: string;
  window: string;
  timingCredit: number;
  matchedCount: number;
  signalCount: number;
  oracleCount: number;
  stateCredit: number;
  stateCount: number;
  distillationWeightedCrossEntropy?: number;
  distillationWeightedOracleEntropy?: number;
  distillationWeight?: number;
  distillationOpportunity?: number;
  distillationSamples?: number;
  valueHorizonMs?: number;
  strategyFinalEquity?: number;
  oracleFinalEquity?: number;
  strategyMaxDrawdown?: number;
  oracleMaxDrawdown?: number;
  strategyTurnover?: number;
  oracleTurnover?: number;
  days: number;
  absoluteLags: number[];
  signedLags: number[];
  lagP50Ms?: number | null;
  lagP90Ms?: number | null;
  lagP95Ms?: number | null;
  lagMedianSignedMs?: number | null;
}

interface AggregateScore {
  objective: number;
  median: number;
  p10: number;
  signalScore: number;
  valueDistillationScore: number | null;
  valueDistillationLoss: number | null;
  valueDistillationKl: number | null;
  meanOpportunity: number | null;
  strategyReturn: number | null;
  oracleReturn: number | null;
  strategyMaxDrawdown: number | null;
  oracleMaxDrawdown: number | null;
  precision: number;
  recall: number;
  f1: number;
  exposureAgreement: number;
  noiseSignalRatio: number | null;
  signalCleanliness: number;
  signalsPerDay: number;
  lagP50Ms: number | null;
  lagP90Ms: number | null;
}

interface CandidateResult {
  candidate: Candidate;
  stage: Stage;
  aggregate: AggregateScore;
  cases: CaseScore[];
}

interface SearchMember {
  genome: Genome;
  family: Family;
  mask: ConfirmationMask;
}

interface SearchDescriptor {
  family: Family;
  agreementMode: AgreementMode;
  mask: ConfirmationMask;
}

interface AdaptiveIsland {
  meanF: number;
  meanCr: number;
  archive: Genome[];
}

interface SearchTelemetry {
  restart: number;
  generation: number;
  fullFit: boolean;
  windows: string[];
  best: number;
  median: number;
  unique: number;
  diversity: number;
  meanF: number;
  meanCr: number;
}

const DAY = 86_400_000;
const HOUR = 3_600_000;
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const stamp = new Date().toISOString().replace(/[:.]/g, "").replace("T", "-").slice(0, 17);
// RTX 3060 measurements put the serial-CPU/CUDA crossover between two and four
// candidates across 5k-60k-candle cases. Keep only genuinely tiny batches on CPU.
const CUDA_AUTO_MIN_CANDIDATES = 4;
const searchTelemetry: SearchTelemetry[] = [];
let candidatePool: CandidatePool | null = null;
let cudaDeviceReported = false;
let cudaAutoFailureReported = false;
const windowJob = process.env.VW_KAMA_WINDOW_JOB;
const candidateWorker = !isMainThread && workerData?.kind === "vw-kama-candidate";

if (candidateWorker) {
  parentPort!.on("message", (request: CandidateWorkerRequest) => {
    try {
      parentPort!.postMessage({
        id: request.id,
        results: evaluatePreparedStage(
          request.job.stage,
          request.job.windows,
          request.job.candidates,
          request.job.prepared,
          request.job.config,
        ),
      } satisfies CandidateWorkerResponse);
    } catch (error) {
      parentPort!.postMessage({
        id: request.id,
        error: error instanceof Error ? error.stack ?? error.message : String(error),
      } satisfies CandidateWorkerResponse);
    }
  });
} else if (!windowJob) {
  queueMicrotask(() => {
    void run(parseArgs(process.argv.slice(2)))
      .finally(closeCandidatePool)
      .catch((error) => {
        console.error(error instanceof Error ? error.stack ?? error.message : error);
        process.exitCode = 1;
      });
  });
} else {
  queueMicrotask(() => {
    void optimizeWindow(JSON.parse(windowJob) as WindowJob)
      .then((result) => process.send?.(result))
      .catch((error) => process.send?.({
        error: error instanceof Error ? error.stack ?? error.message : String(error),
      }))
      .finally(closeCandidatePool);
  });
}

interface CandidateJob {
  stage: Stage;
  windows: Window[];
  candidates: Candidate[];
  prepared: PreparedStageWindow;
  config: Args;
}

interface CandidateWorkerRequest { id: number; job: CandidateJob }
interface CandidateWorkerResponse { id: number; results?: CandidateResult[]; error?: string }

async function run(config: Args): Promise<void> {
  const startedAt = Date.now();
  if (config.buyMaxFraction.max > 1 || config.sellMaxFraction.max > 1) {
    throw new Error("Sizing fractions cannot exceed one.");
  }
  validateScales(config.scales, config.sourceIntervalMs);
  validateScales(config.screenScales, config.sourceIntervalMs);
  if (config.algorithm === "de" && config.trials < 4) {
    throw new Error("Differential evolution requires at least four genomes.");
  }
  const bounds = sourceBounds(config.sourceDir);
  const defaults = defaultWindows(bounds.start, bounds.end, config.sourceIntervalMs);
  const windows = {
    fit: config.fitWindows ?? defaults.fit,
    validation: config.validationWindows ?? defaults.validation,
    test: config.testWindows ?? defaults.test,
  };
  if (config.mode === "per-window") {
    await runPerWindow(config, windows.fit, bounds, startedAt);
    return;
  }
  assertChronological(windows.fit, windows.validation, windows.test);
  const maxWarmupMs = maximumWarmupMs(config);
  const seeds = loadSeedCandidates(config.seedCandidatePaths);
  if (seeds.length > 0) {
    console.error(`Warm start: ${seeds.length} unique genomes from ${config.seedCandidatePaths.length} input file(s).`);
  }
  const candidates = await searchCandidates(config, windows.fit, bounds, maxWarmupMs, seeds, true);

  fs.mkdirSync(path.dirname(config.outputPath), { recursive: true });
  fs.mkdirSync(path.dirname(config.reportPath), { recursive: true });
  fs.writeFileSync(config.outputPath, "");
  appendJson(config.outputPath, {
    type: "meta",
    scoreVersion: config.scoreVersion,
    generatedAt: new Date(startedAt).toISOString(),
    sourceDir: path.relative(repoRoot, config.sourceDir),
    sourceInterval: formatDuration(config.sourceIntervalMs),
    scales: config.scales.map(formatDuration),
    windows,
    trials: config.trials,
    algorithm: config.algorithm,
    generations: config.generations,
    optimization: {
      restarts: config.restarts,
      fullEvaluationInterval: config.fullEvaluationInterval,
      refinementRounds: config.refinementRounds,
      pbestFraction: config.pbestFraction,
      confirmationMasks: config.confirmationMasks,
      strategy: "Latin-hypercube warm start -> score/novelty island selection -> adaptive current-to-pbest DE -> elite refinement",
      telemetry: searchTelemetry,
    },
    workers: config.workers,
    accelerator: config.accelerator,
    objective: config.objective,
    valueDistillation: {
      exposureGridSize: config.exposureGridSize,
      exposureRange: [config.exposureMinimum, config.exposureMaximum],
      horizonMode: config.valueHorizonMode,
      horizonMs: config.valueHorizonMs,
      oracleTemperature: config.oracleTemperature,
      strategyTemperature: config.strategyTemperature,
      strategyVolatilityScaling: config.strategyVolatilityScaling,
      opportunityWeight: `max(Q)-min(Q)+${config.opportunityEpsilon}`,
      quoteLendRate: config.quoteLendRate,
      quoteBorrowRate: config.quoteBorrowRate,
      assetBorrowRate: config.assetBorrowRate,
    },
    warmStart: {
      files: config.seedCandidatePaths.map((file) => path.relative(repoRoot, file)),
      genomes: seeds.length,
      quasiRandomGenomes: config.trials,
      selectedPopulation: config.trials,
    },
    screening: {
      windows: config.screenWindows,
      scales: config.screenScales.map(formatDuration),
    },
    seed: config.seed,
    oracle: {
      friction: config.oracleFriction,
      matchWindow: formatDuration(config.matchWindowMs),
      timingHalfLife: formatDuration(config.timingHalfLifeMs),
      warmupMultiple: config.warmupMultiple,
      caseWarmup: formatDuration(maxWarmupMs),
    },
    signalMemory: {
      friction: config.oracleFriction,
      rule: "absolute close-to-last-signal return must be greater than friction",
    },
    ranges: {
      efficiency: formatRange(config.efficiency, formatDuration),
      efficiencyVolumeEma: formatRange(config.efficiencyVolumeEma, formatDuration),
      efficiencyVolumePower: formatRange(config.efficiencyVolumePower),
      fast: formatRange(config.fast, formatDuration),
      slow: formatRange(config.slow, formatDuration),
      volume: formatRange(config.volume, formatDuration),
      power: formatRange(config.power),
      volumeCap: formatRange(config.volumeCap),
      volumePower: formatRange(config.volumePower),
      deadbandBpsHour: formatRange(config.deadbandBpsHour),
      hysteresisReleaseRatio: formatRange(config.hysteresisReleaseRatio),
      thresholdLookback: formatRange(config.thresholdLookback, formatDuration),
      thresholdNoiseMultiplier: formatRange(config.thresholdNoiseMultiplier),
      buyMaxFraction: formatRange(config.buyMaxFraction),
      sellMaxFraction: formatRange(config.sellMaxFraction),
      buySizingSigma: formatRange(config.buySizingSigma),
      sellSizingSigma: formatRange(config.sellSizingSigma),
      agreementModes: config.agreementModes,
      deadbandModes: config.deadbandModes,
      confirmationMix: formatRange(config.confirmationMix),
      confirmationMinQuality: formatRange(config.confirmationMinQuality),
      confirmationAccelerationLookback: formatRange(config.confirmationAccelerationLookback, formatDuration),
      confirmationDistanceLookback: formatRange(config.confirmationDistanceLookback, formatDuration),
      confirmationAccelerationWeight: formatRange(config.confirmationAccelerationWeight),
      confirmationDistanceWeight: formatRange(config.confirmationDistanceWeight),
      confirmationBias: formatRange(config.confirmationBias),
      confirmationEma: formatRange(config.confirmationEma, formatDuration),
      confirmationEmaThreshold: formatRange(config.confirmationEmaThreshold),
      confirmationEmaWeight: formatRange(config.confirmationEmaWeight),
      confirmationEmaGateStrength: formatRange(config.confirmationEmaGateStrength),
      confirmationRsi: formatRange(config.confirmationRsi, formatDuration),
      confirmationRsiThreshold: formatRange(config.confirmationRsiThreshold),
      confirmationRsiWeight: formatRange(config.confirmationRsiWeight),
      confirmationDmi: formatRange(config.confirmationDmi, formatDuration),
      confirmationDmiWeight: formatRange(config.confirmationDmiWeight),
      confirmationAdxThreshold: formatRange(config.confirmationAdxThreshold),
      meanReversionSuppressionThreshold: formatRange(config.meanReversionSuppressionThreshold),
      meanReversionEfficiency: formatRange(config.meanReversionEfficiency, formatDuration),
      meanReversionFast: formatRange(config.meanReversionFast, formatDuration),
      meanReversionSlow: formatRange(config.meanReversionSlow, formatDuration),
      meanReversionVolatility: formatRange(config.meanReversionVolatility, formatDuration),
      meanReversionReversalThreshold: formatRange(config.meanReversionReversalThreshold),
    },
    selection: "fit rank -> validation finalist per family -> finalists plus fixed production signal baseline on holdout",
    matching: "chronological one-to-one target-state alignment maximizing timing credit",
  });

  const fit = await evaluateStageParallel("fit", windows.fit, candidates, bounds, maxWarmupMs, config);
  for (const result of fit) appendResult(config.outputPath, result, false);
  const fitTop = dedupeCandidateResults([
    ...(["volume", "canonical"] as const).flatMap((family) =>
      rank(fit.filter((result) => result.candidate.family === family)).slice(0, config.top)),
    ...fit.filter((result) => result.candidate.id === "production-current"),
  ]);
  const validation = await evaluateStageParallel(
    "validation",
    windows.validation,
    fitTop.map((result) => result.candidate),
    bounds,
    maxWarmupMs,
    config,
  );
  for (const result of validation) appendResult(config.outputPath, result, true);
  const finalists = (["volume", "canonical"] as const).map((family) => {
    const selected = rank(validation.filter((result) => result.candidate.family === family))[0];
    if (!selected) throw new Error(`No ${family} validation finalist.`);
    return selected;
  });
  const testCandidates = dedupeCandidates([
    ...finalists.map((result) => result.candidate),
    ...validation
      .filter((result) => result.candidate.id === "production-current")
      .map((result) => result.candidate),
  ]);
  const test = await evaluateStageParallel(
    "test",
    windows.test,
    testCandidates,
    bounds,
    maxWarmupMs,
    config,
  );
  for (const result of test) appendResult(config.outputPath, result, true);
  appendJson(config.outputPath, {
    type: "selection",
    finalists: finalists.map((result) => ({
      family: result.candidate.family,
      id: result.candidate.id,
      validationObjective: round(result.aggregate.objective, 6),
    })),
  });
  writeGlobalPresets(config, finalists, startedAt);
  appendJson(config.outputPath, {
    type: "status",
    status: "completed",
    completedAt: new Date().toISOString(),
    elapsedMs: Date.now() - startedAt,
  });
  fs.writeFileSync(config.reportPath, report(config, windows, fit, validation, test, maxWarmupMs));
  console.error(`Results: ${path.relative(repoRoot, config.outputPath)}`);
  console.error(`Report: ${path.relative(repoRoot, config.reportPath)}`);
}

function writeGlobalPresets(
  config: Args,
  finalists: CandidateResult[],
  startedAt: number,
): void {
  const completedAt = Date.now();
  const generated = finalists.map((result): VwKamaPreset => ({
    id: `global-${config.objective}-${result.candidate.family}`,
    label: `Global · ${config.objective} · ${result.candidate.family}`,
    scope: "global",
    windowId: null,
    parameters: candidateParameters(result.candidate),
    score: round(result.aggregate.objective, 6),
    scoreVersion: config.scoreVersion,
    source: `${config.algorithm.toUpperCase()} chronological validation finalist; holdout excluded from selection`,
    generatedAt: new Date(completedAt).toISOString(),
    optimization: {
      algorithm: config.algorithm,
      objective: config.objective,
      population: config.trials,
      generations: config.generations,
      restarts: config.restarts,
      refinementRounds: config.refinementRounds,
      elapsedMs: completedAt - startedAt,
      hindsight: false,
      valueDistillation: {
        gridSize: config.exposureGridSize,
        minExposure: config.exposureMinimum,
        maxExposure: config.exposureMaximum,
        horizonMode: config.valueHorizonMode,
        horizonMs: config.valueHorizonMs,
        oracleTemperature: config.oracleTemperature,
        strategyTemperature: config.strategyTemperature,
        strategyVolatilityScaling: config.strategyVolatilityScaling,
        opportunityEpsilon: config.opportunityEpsilon,
        quoteLendRate: config.quoteLendRate,
        quoteBorrowRate: config.quoteBorrowRate,
        assetBorrowRate: config.assetBorrowRate,
      },
    },
  }));
  const replaced = new Set(generated.map((preset) => preset.id));
  const presets = [
    ...loadPresets(config.presetOutputPath).filter((preset) => !replaced.has(preset.id)),
    ...generated,
  ];
  fs.mkdirSync(path.dirname(config.presetOutputPath), { recursive: true });
  fs.writeFileSync(config.presetOutputPath, `${JSON.stringify(presets, null, 2)}\n`);
  console.error(`Global presets: ${path.relative(repoRoot, config.presetOutputPath)}`);
}

async function searchCandidates(
  config: Args,
  windows: Window[],
  bounds: { start: number; end: number },
  warmupMs: number,
  seeds: Candidate[] = [],
  _initialOnly = false,
): Promise<Candidate[]> {
  seeds = dedupeSeedCandidates(seeds.flatMap((candidate) =>
    config.agreementModes.map((agreementMode) => ({ ...candidate, agreementMode }))));
  if (config.algorithm === "random") return [...globalCandidates(config.agreementModes, config.oracleFriction), ...seeds, ...generateCandidates(config)];
  const screenConfig = {
    ...config,
    scales: config.screenScales.filter((scale) => config.scales.includes(scale)),
  };
  if (screenConfig.scales.length === 0) screenConfig.scales = [config.scales[0]!];
  const cache: FitnessCache = new Map();
  const populations: SearchMember[] = [];
  for (let restart = 0; restart < config.restarts; restart += 1) {
    populations.push(...await evolvePopulation(
      config,
      screenConfig,
      windows,
      bounds,
      warmupMs,
      seeds,
      cache,
      restart,
    ));
  }
  const refined = await refineMembers(
    dedupeMembers(populations),
    windows,
    bounds,
    warmupMs,
    screenConfig,
    cache,
  );
  const combined = dedupeMembers([...populations, ...refined]);
  const finalScores = await memberFitness(
    combined,
    windows,
    bounds,
    warmupMs,
    screenConfig,
    cache,
    "final full-fit selection",
  );
  const selected = selectPopulation(
    combined,
    finalScores,
    config.trials,
    searchDescriptors(config),
    config,
  );
  const candidates = selected.map(memberCandidate);
  return [
    ...globalCandidates(config.agreementModes, config.oracleFriction),
    ...candidates,
  ];
}

async function evolvePopulation(
  config: Args,
  screenConfig: Args,
  windows: Window[],
  bounds: { start: number; end: number },
  warmupMs: number,
  seeds: Candidate[],
  cache: FitnessCache,
  restart: number,
): Promise<SearchMember[]> {
  const random = mulberry32(config.seed + restart * 1_000_003);
  const descriptors = searchDescriptors(config);
  const fresh = latinMembers(config.trials, descriptors, config, random);
  const warm = restart === 0 ? seedMembers(seeds, config) : [];
  const initial = dedupeMembers([...warm, ...fresh]);
  const initialWindows = rotatingWindows(windows, config.screenWindows, 0);
  const initialScores = await memberFitness(
    initial,
    initialWindows,
    bounds,
    warmupMs,
    screenConfig,
    cache,
    `restart ${restart + 1} initial`,
  );
  let population = selectPopulation(initial, initialScores, config.trials, descriptors, config);
  let hallOfFame: SearchMember[] = [];
  const states = new Map(descriptors.map((descriptor) => [descriptorKey(descriptor), {
    meanF: config.differentialWeight,
    meanCr: config.crossoverRate,
    archive: [],
  } satisfies AdaptiveIsland]));

  for (let generation = 0; generation < config.generations; generation += 1) {
    const useFull = (generation + 1) % config.fullEvaluationInterval === 0;
    const fold = useFull
      ? windows
      : rotatingWindows(windows, config.screenWindows, generation + 1);
    const parentScores = await memberFitness(
      population,
      fold,
      bounds,
      warmupMs,
      screenConfig,
      cache,
      `restart ${restart + 1} generation ${generation + 1} parents`,
    );
    const trials: SearchMember[] = [];
    const metadata: Array<{ key: string; f: number; cr: number; target: SearchMember }> = [];
    const grouped = groupMembers(population);
    for (const [key, members] of grouped) {
      const state = states.get(key)!;
      const scores = members.map((member) => parentScores.get(memberKey(member)) ?? 0);
      const order = members.map((_, index) => index).sort((a, b) => scores[b]! - scores[a]!);
      for (let index = 0; index < members.length; index += 1) {
        if (random() < config.immigrantRate) {
          trials.push(randomMemberLike(members[index]!, config, random));
          metadata.push({ key, f: state.meanF, cr: state.meanCr, target: members[index]! });
          continue;
        }
        const trial = adaptiveTrial(members, scores, order, index, state, population, config, random);
        trials.push(trial.member);
        metadata.push({ key, f: trial.f, cr: trial.cr, target: members[index]! });
      }
    }
    const trialScores = await memberFitness(
      trials,
      fold,
      bounds,
      warmupMs,
      screenConfig,
      cache,
      `restart ${restart + 1} generation ${generation + 1} trials`,
    );
    const parentByKey = new Map(population.map((member) => [memberKey(member), member]));
    const success = new Map<string, Array<{ f: number; cr: number; gain: number }>>();
    population = trials.map((trial, index) => {
      const key = metadata[index]!.key;
      const island = grouped.get(key)!;
      const target = metadata[index]!.target;
      const targetScore = parentScores.get(memberKey(target)) ?? 0;
      const trialScore = trialScores.get(memberKey(trial)) ?? 0;
      if (trialScore <= targetScore) return target;
      const state = states.get(key)!;
      state.archive.push(target.genome);
      if (state.archive.length > island.length) state.archive.splice(0, state.archive.length - island.length);
      const list = success.get(key) ?? [];
      list.push({ ...metadata[index]!, gain: trialScore - targetScore });
      success.set(key, list);
      return trial;
    });
    for (const [key, values] of success) updateAdaptiveState(states.get(key)!, values);
    population = selectPopulation(
      dedupeMembers([...population, ...parentByKey.values()]),
      new Map([...parentScores, ...trialScores]),
      config.trials,
      descriptors,
      config,
    );
    const scores = await memberFitness(population, fold, bounds, warmupMs, screenConfig, cache, "telemetry");
    const values = population.map((member) => scores.get(memberKey(member)) ?? 0);
    const diversity = populationDiversity(population, config);
    const adaptive = [...states.values()];
    const meanF = mean(adaptive.map((item) => item.meanF));
    const meanCr = mean(adaptive.map((item) => item.meanCr));
    if (useFull) {
      const hallCandidates = dedupeMembers([...hallOfFame, ...population]);
      const hallScores = await memberFitness(
        hallCandidates,
        windows,
        bounds,
        warmupMs,
        screenConfig,
        cache,
        "hall of fame",
      );
      hallOfFame = hallCandidates
        .sort((left, right) => (hallScores.get(memberKey(right)) ?? 0) - (hallScores.get(memberKey(left)) ?? 0))
        .slice(0, Math.max(32, config.top * 4));
    }
    searchTelemetry.push({
      restart: restart + 1,
      generation: generation + 1,
      fullFit: useFull,
      windows: fold.map((window) => window.label),
      best: Math.max(...values),
      median: median(values),
      unique: population.length,
      diversity,
      meanF,
      meanCr,
    });
    console.error(
      `DE restart ${restart + 1}/${config.restarts} generation ${generation + 1}/${config.generations}: `
      + `best ${formatSearchFitness(Math.max(...values), config.objective)}, `
      + `median ${formatSearchFitness(median(values), config.objective)}, `
      + `unique ${population.length}, diversity ${diversity.toFixed(3)}, `
      + `F ${meanF.toFixed(3)}, CR ${meanCr.toFixed(3)}, `
      + `${useFull ? "full fit" : `fold ${generation % Math.max(1, windows.length) + 1}`}.`,
    );
  }
  return dedupeMembers([...population, ...hallOfFame]);
}

function adaptiveTrial(
  island: SearchMember[],
  scores: number[],
  order: number[],
  targetIndex: number,
  state: AdaptiveIsland,
  population: SearchMember[],
  config: Args,
  random: () => number,
): { member: SearchMember; f: number; cr: number } {
  const target = island[targetIndex]!;
  const pbestCount = Math.max(2, Math.ceil(island.length * config.pbestFraction));
  const pbest = island[order[Math.floor(random() * Math.min(pbestCount, order.length))]!]!;
  const available = island.map((_, index) => index).filter((index) => index !== targetIndex);
  shuffle(available, random);
  const r1 = island[available[0] ?? targetIndex]!;
  const useMigration = random() < 0.1 && population.length > island.length;
  const r2Genome = useMigration
    ? population[Math.floor(random() * population.length)]!.genome
    : state.archive.length > 0 && random() < 0.5
      ? state.archive[Math.floor(random() * state.archive.length)]!
      : island[available[1] ?? available[0] ?? targetIndex]!.genome;
  const f = adaptiveF(state.meanF, random);
  const cr = clamp01(randomNormal(state.meanCr, 0.1, random));
  const targetVector = genomeVector(target.genome, config);
  const pbestVector = genomeVector(pbest.genome, config);
  const r1Vector = genomeVector(r1.genome, config);
  const r2Vector = genomeVector(r2Genome, config);
  const forced = Math.floor(random() * targetVector.length);
  const vector = targetVector.map((value, index) => index === forced || random() < cr
    ? clamp01(value + f * (pbestVector[index]! - value) + f * (r1Vector[index]! - r2Vector[index]!))
    : value);
  let genome = vectorGenome(vector, config);
  if (random() < cr) genome.deadbandMode = random() < 0.5 ? pbest.genome.deadbandMode : r1.genome.deadbandMode;
  genome = applyDescriptor(genome, memberDescriptor(target), config);
  return { member: { ...target, genome }, f, cr };
}

function searchDescriptors(config: Args): SearchDescriptor[] {
  return (["volume", "canonical"] as const).flatMap((family) =>
    config.agreementModes.flatMap((agreementMode) =>
      config.confirmationMasks.map((mask) => ({ family, agreementMode, mask }))));
}

function descriptorKey(descriptor: SearchDescriptor): string {
  return `${descriptor.family}:${descriptor.agreementMode}:${descriptor.mask}`;
}

function memberDescriptor(member: SearchMember): SearchDescriptor {
  return { family: member.family, agreementMode: member.genome.agreementMode, mask: member.mask };
}

function groupMembers(members: SearchMember[]): Map<string, SearchMember[]> {
  const grouped = new Map<string, SearchMember[]>();
  for (const member of members) {
    const key = descriptorKey(memberDescriptor(member));
    const list = grouped.get(key) ?? [];
    list.push(member);
    grouped.set(key, list);
  }
  return grouped;
}

function latinMembers(
  count: number,
  descriptors: SearchDescriptor[],
  config: Args,
  random: () => number,
): SearchMember[] {
  const base = Math.floor(count / descriptors.length);
  let remainder = count % descriptors.length;
  return descriptors.flatMap((descriptor) => {
    const size = base + (remainder-- > 0 ? 1 : 0);
    return latinVectors(size, 42, random).map((vector) => ({
      family: descriptor.family,
      mask: descriptor.mask,
      genome: applyDescriptor(vectorGenome(vector, config), descriptor, config),
    }));
  });
}

function latinVectors(count: number, dimensions: number, random: () => number): number[][] {
  if (count === 0) return [];
  const vectors = Array.from({ length: count }, () => Array<number>(dimensions));
  for (let dimension = 0; dimension < dimensions; dimension += 1) {
    const strata = Array.from({ length: count }, (_, index) => (index + random()) / count);
    shuffle(strata, random);
    for (let index = 0; index < count; index += 1) vectors[index]![dimension] = strata[index]!;
  }
  return vectors;
}

function seedMembers(seeds: Candidate[], config: Args): SearchMember[] {
  return dedupeMembers(seeds.flatMap((candidate) => {
    const mask = config.confirmationMasks.includes(candidateMask(candidate))
      ? candidateMask(candidate)
      : config.confirmationMasks.includes("all") ? "all" : config.confirmationMasks[0]!;
    const descriptor = { family: candidate.family, agreementMode: candidate.agreementMode, mask };
    const genome = {
      ...candidateGenome(candidate),
      agreementMode: descriptor.agreementMode,
      volumePower: descriptor.family === "canonical" ? 0 : candidate.volumePower,
      efficiencyVolumePower: descriptor.family === "canonical" ? 0 : candidate.efficiencyVolumePower,
    };
    const exact = {
      family: descriptor.family,
      mask,
      genome: applyDescriptor(genome, descriptor, config),
    } satisfies SearchMember;
    if (!volumeAware(candidate) || candidate.family === "canonical") return [exact];
    const canonical = {
      ...exact,
      family: "canonical" as const,
      genome: applyDescriptor(
        { ...exact.genome, volumePower: 0, efficiencyVolumePower: 0 },
        { ...descriptor, family: "canonical" },
        config,
      ),
    };
    return [exact, canonical];
  }));
}

function candidateMask(candidate: Candidate): ConfirmationMask {
  if (candidate.confirmationMix <= 0 && candidate.confirmationEmaGateStrength <= 0) return "base";
  const active = [
    candidate.confirmationAccelerationWeight > 0 || candidate.confirmationDistanceWeight > 0 ? "acceleration" : null,
    candidate.confirmationEmaWeight > 0 || candidate.confirmationEmaGateStrength > 0 ? "ema" : null,
    candidate.confirmationRsiWeight > 0 ? "rsi" : null,
    candidate.confirmationDmiWeight > 0 ? "dmi" : null,
  ].filter(Boolean);
  if (active.length === 0) return "base";
  if (active.length === 4) return "all";
  const mask = active.join("+");
  return CONFIRMATION_MASKS.includes(mask as ConfirmationMask) ? mask as ConfirmationMask : "all";
}

function maskHas(mask: ConfirmationMask, feature: "acceleration" | "ema" | "rsi" | "dmi"): boolean {
  return mask === "all" || mask.split("+").includes(feature);
}

function applyDescriptor(genome: Genome, descriptor: SearchDescriptor, config: Args): Genome {
  const active = (value: number, range: Range): number => range.max <= 0
    ? 0
    : Math.max(value, range.min > 0 ? range.min : Math.min(range.max, 0.05));
  const result = {
    ...vectorGenome(genomeVector(genome, config), config),
    agreementMode: config.agreementModes.includes(descriptor.agreementMode)
      ? descriptor.agreementMode
      : config.agreementModes[0]!,
  };
  if (descriptor.family === "canonical") {
    result.volumePower = 0;
    result.efficiencyVolumePower = 0;
  } else if (result.volumePower <= 0 && result.efficiencyVolumePower <= 0) {
    if (config.efficiencyVolumePower.max > 0) {
      result.efficiencyVolumePower = Math.max(
        config.efficiencyVolumePower.min,
        Math.min(config.efficiencyVolumePower.max, 0.05),
      );
    } else if (config.volumePower.max > 0) {
      result.volumePower = Math.max(config.volumePower.min, Math.min(config.volumePower.max, 0.05));
    }
  }
  if (result.efficiencyVolumePower <= 0) {
    result.efficiencyVolumeEmaMs = config.efficiencyVolumeEma.min;
  }
  if (result.volumePower <= 0) {
    result.volumeMs = config.volume.min;
    result.volumeCap = config.volumeCap.min;
  }
  if (result.meanReversionSuppressionThreshold <= 0
    || result.meanReversionReversalThreshold <= 0) {
    result.meanReversionSuppressionThreshold = 0;
    result.meanReversionEfficiencyMs = config.meanReversionEfficiency.min;
    result.meanReversionFastMs = config.meanReversionFast.min;
    result.meanReversionSlowMs = config.meanReversionSlow.min;
    result.meanReversionVolatilityMs = config.meanReversionVolatility.min;
    result.meanReversionReversalThreshold = 0;
  } else {
    result.meanReversionReversalThreshold = Math.max(
      result.meanReversionSuppressionThreshold,
      result.meanReversionReversalThreshold,
    );
  }
  if (descriptor.mask === "base") {
    result.confirmationMix = 0;
    result.confirmationMinQuality = 0;
    result.confirmationAccelerationWeight = 0;
    result.confirmationDistanceWeight = 0;
    result.confirmationEmaWeight = 0;
    result.confirmationEmaGateStrength = 0;
    result.confirmationRsiWeight = 0;
    result.confirmationDmiWeight = 0;
    return result;
  }
  result.confirmationMix = active(result.confirmationMix, config.confirmationMix);
  result.confirmationAccelerationWeight = maskHas(descriptor.mask, "acceleration")
    ? active(result.confirmationAccelerationWeight, config.confirmationAccelerationWeight) : 0;
  result.confirmationDistanceWeight = maskHas(descriptor.mask, "acceleration")
    ? active(result.confirmationDistanceWeight, config.confirmationDistanceWeight) : 0;
  result.confirmationEmaWeight = maskHas(descriptor.mask, "ema")
    ? active(result.confirmationEmaWeight, config.confirmationEmaWeight) : 0;
  result.confirmationEmaGateStrength = maskHas(descriptor.mask, "ema")
    ? result.confirmationEmaGateStrength : 0;
  result.confirmationRsiWeight = maskHas(descriptor.mask, "rsi")
    ? active(result.confirmationRsiWeight, config.confirmationRsiWeight) : 0;
  result.confirmationDmiWeight = maskHas(descriptor.mask, "dmi")
    ? active(result.confirmationDmiWeight, config.confirmationDmiWeight) : 0;
  return result;
}

function randomMemberLike(member: SearchMember, config: Args, random: () => number): SearchMember {
  const descriptor = memberDescriptor(member);
  return { ...member, genome: applyDescriptor(randomGenome(config, random), descriptor, config) };
}

function rotatingWindows(windows: Window[], count: number, generation: number): Window[] {
  const size = Math.max(1, Math.min(count, windows.length));
  return Array.from({ length: size }, (_, offset) => windows[(generation + offset) % windows.length]!);
}

async function memberFitness(
  members: SearchMember[],
  windows: Window[],
  bounds: { start: number; end: number },
  warmupMs: number,
  config: Args,
  cache: FitnessCache,
  label: string,
): Promise<Map<string, number>> {
  const unique = dedupeMembers(members);
  for (const window of windows) {
    const missing = unique.filter((member) => !cache.has(fitnessCacheKey(member, window, config)));
    if (missing.length === 0) continue;
    console.error(
      `DE ${label}: evaluating ${missing.length}/${unique.length} uncached members on `
      + `${window.label} × ${config.scales.length} scale(s).`,
    );
    const evaluated = await evaluateStageParallel(
      "fit",
      [window],
      missing.map(memberCandidate),
      bounds,
      warmupMs,
      config,
    );
    for (let index = 0; index < missing.length; index += 1) {
      cache.set(fitnessCacheKey(missing[index]!, window, config), evaluated[index]?.cases ?? []);
    }
  }
  return new Map(unique.map((member) => {
    const cases = windows.flatMap((window) => cache.get(fitnessCacheKey(member, window, config)) ?? []);
    return [memberKey(member), aggregate(cases).objective];
  }));
}

function fitnessCacheKey(member: SearchMember, window: Window, config: Args): string {
  return `${config.scales.join(",")}:${window.label}:${window.start}-${window.end}:${memberKey(member)}`;
}

function memberCandidate(member: SearchMember, index = 0): Candidate {
  return {
    ...member.genome,
    id: `${member.family === "volume" ? "v" : "k"}${String(index + 1).padStart(5, "0")}`,
    family: member.family,
    volumePower: member.family === "canonical" ? 0 : member.genome.volumePower,
    efficiencyVolumePower: member.family === "canonical" ? 0 : member.genome.efficiencyVolumePower,
  };
}

function memberKey(member: SearchMember): string {
  return `${member.family}:${member.mask}:${JSON.stringify(candidateParameters(memberCandidate(member)))}`;
}

function dedupeMembers(members: SearchMember[]): SearchMember[] {
  return [...new Map(members.map((member) => [memberKey(member), member])).values()];
}

function selectPopulation(
  members: SearchMember[],
  scores: Map<string, number>,
  count: number,
  descriptors: SearchDescriptor[],
  config: Args,
): SearchMember[] {
  const grouped = groupMembers(members);
  const islandFloor = Math.max(1, Math.floor(count / descriptors.length / 4));
  const selected = descriptors.flatMap((descriptor) => selectDiverse(
    grouped.get(descriptorKey(descriptor)) ?? [],
    scores,
    islandFloor,
    config,
  ));
  if (selected.length >= count) return selected.slice(0, count);
  const selectedKeys = new Set(selected.map(memberKey));
  const remaining = members.filter((member) => !selectedKeys.has(memberKey(member)));
  return [...selected, ...selectDiverseAdditions(
    remaining,
    selected,
    scores,
    count - selected.length,
    config,
  )];
}

function selectDiverse(
  members: SearchMember[],
  scores: Map<string, number>,
  count: number,
  config: Args,
): SearchMember[] {
  const ranked = members.slice().sort((left, right) =>
    (scores.get(memberKey(right)) ?? 0) - (scores.get(memberKey(left)) ?? 0));
  if (ranked.length <= count) return ranked;
  const selected = ranked.slice(0, Math.max(1, Math.ceil(count * 0.4)));
  return [...selected, ...selectDiverseAdditions(
    ranked.slice(selected.length),
    selected,
    scores,
    count - selected.length,
    config,
  )];
}

function selectDiverseAdditions(
  members: SearchMember[],
  selected: SearchMember[],
  scores: Map<string, number>,
  count: number,
  config: Args,
): SearchMember[] {
  const references = selected.map((member) => genomeVector(member.genome, config));
  const additions: SearchMember[] = [];
  if (members.length <= count) return members;
  const values = members.map((member) => scores.get(memberKey(member)) ?? 0);
  const low = Math.min(...values);
  const span = Math.max(Number.EPSILON, Math.max(...values) - low);
  const remaining = members.map((member, index) => {
    const vector = genomeVector(member.genome, config);
    return {
      member,
      vector,
      quality: (values[index]! - low) / span,
      novelty: references.length === 0
        ? 1
        : Math.min(...references.map((reference) => vectorDistance(vector, reference))),
    };
  });
  while (additions.length < count && remaining.length > 0) {
    let bestIndex = 0;
    let bestValue = -Infinity;
    for (let index = 0; index < remaining.length; index += 1) {
      const entry = remaining[index]!;
      const value = entry.quality * 0.7 + entry.novelty * 0.3;
      if (value > bestValue) {
        bestValue = value;
        bestIndex = index;
      }
    }
    const next = remaining.splice(bestIndex, 1)[0]!;
    additions.push(next.member);
    for (const entry of remaining) {
      entry.novelty = Math.min(entry.novelty, vectorDistance(entry.vector, next.vector));
    }
  }
  return additions;
}

function memberDistance(left: SearchMember, right: SearchMember, config: Args): number {
  return vectorDistance(genomeVector(left.genome, config), genomeVector(right.genome, config));
}

function vectorDistance(left: number[], right: number[]): number {
  let total = 0;
  for (let index = 0; index < left.length; index += 1) total += (left[index]! - right[index]!) ** 2;
  return Math.sqrt(total / left.length);
}

function updateAdaptiveState(
  state: AdaptiveIsland,
  success: Array<{ f: number; cr: number; gain: number }>,
): void {
  const total = sum(success.map((item) => item.gain));
  if (total <= 0) return;
  const weighted = (selector: (item: typeof success[number]) => number): number =>
    sum(success.map((item) => selector(item) * item.gain)) / total;
  const fMean = weighted((item) => item.f ** 2) / Math.max(Number.EPSILON, weighted((item) => item.f));
  state.meanF = state.meanF * 0.9 + fMean * 0.1;
  state.meanCr = state.meanCr * 0.9 + weighted((item) => item.cr) * 0.1;
}

function adaptiveF(meanF: number, random: () => number): number {
  for (let attempt = 0; attempt < 20; attempt += 1) {
    const value = meanF + 0.1 * Math.tan(Math.PI * (random() - 0.5));
    if (value > 0) return Math.min(1, value);
  }
  return Math.min(1, Math.max(0.1, meanF));
}

function randomNormal(meanValue: number, sigma: number, random: () => number): number {
  const radius = Math.sqrt(-2 * Math.log(Math.max(Number.MIN_VALUE, random())));
  return meanValue + sigma * radius * Math.cos(2 * Math.PI * random());
}

function populationDiversity(members: SearchMember[], config: Args): number {
  if (members.length < 2) return 0;
  const stride = Math.max(1, Math.floor(members.length / 64));
  const sample = members.filter((_, index) => index % stride === 0).slice(0, 64);
  const distances: number[] = [];
  for (let left = 0; left < sample.length; left += 1) {
    for (let right = left + 1; right < sample.length; right += 1) {
      distances.push(memberDistance(sample[left]!, sample[right]!, config));
    }
  }
  return mean(distances);
}

async function refineMembers(
  members: SearchMember[],
  windows: Window[],
  bounds: { start: number; end: number },
  warmupMs: number,
  config: Args,
  cache: FitnessCache,
): Promise<SearchMember[]> {
  if (config.refinementRounds === 0 || members.length === 0) return [];
  const random = mulberry32(config.seed + 9_999_991);
  const scores = await memberFitness(members, windows, bounds, warmupMs, config, cache, "elite selection");
  let elites = members.slice().sort((left, right) =>
    (scores.get(memberKey(right)) ?? 0) - (scores.get(memberKey(left)) ?? 0))
    .slice(0, Math.max(24, config.top * 4));
  for (let round = 0; round < config.refinementRounds; round += 1) {
    const sigma = 0.08 / 2 ** round;
    const neighbors = elites.flatMap((elite) => [0, 1].map(() => perturbMember(elite, sigma, config, random)));
    const roundScores = await memberFitness(
      [...elites, ...neighbors],
      windows,
      bounds,
      warmupMs,
      config,
      cache,
      `refinement ${round + 1}`,
    );
    elites = elites.map((elite, index) => {
      const options = [elite, neighbors[index * 2]!, neighbors[index * 2 + 1]!];
      return options.sort((left, right) =>
        ((roundScores.get(memberKey(right)) ?? 0)
        - (roundScores.get(memberKey(left)) ?? 0)))[0]!;
    });
  }
  return elites;
}

function perturbMember(
  member: SearchMember,
  sigma: number,
  config: Args,
  random: () => number,
): SearchMember {
  const vector = genomeVector(member.genome, config).map((value) =>
    clamp01(value + randomNormal(0, sigma, random)));
  return { ...member, genome: applyDescriptor(vectorGenome(vector, config), memberDescriptor(member), config) };
}

function genomeVector(genome: Genome, config: Args): number[] {
  return [
    logUnit(genome.efficiencyMs, config.efficiency),
    logUnit(genome.fastMs, config.fast),
    logUnit(genome.slowMs, config.slow),
    unit(genome.power, config.power),
    logUnit(genome.volumeMs, config.volume),
    unit(genome.volumeCap, config.volumeCap),
    sparseUnit(genome.volumePower, config.volumePower),
    logUnit(genome.deadbandBpsHour, config.deadbandBpsHour),
    genome.deadbandMode === "hold" ? 1 : genome.deadbandMode === "hysteresis" ? 0.5 : 0,
    logUnit(genome.thresholdLookbackMs, config.thresholdLookback),
    unit(genome.thresholdNoiseMultiplier, config.thresholdNoiseMultiplier),
    unit(genome.buyMaxFraction, config.buyMaxFraction),
    unit(genome.sellMaxFraction, config.sellMaxFraction),
    logUnit(genome.buySizingSigmaBpsHour, config.buySizingSigma),
    logUnit(genome.sellSizingSigmaBpsHour, config.sellSizingSigma),
    genome.agreementMode === "confidence" ? 1 : 0,
    unit(genome.confirmationMix, config.confirmationMix),
    unit(genome.confirmationMinQuality, config.confirmationMinQuality),
    logUnit(genome.confirmationAccelerationLookbackMs, config.confirmationAccelerationLookback),
    logUnit(genome.confirmationDistanceLookbackMs, config.confirmationDistanceLookback),
    unit(genome.confirmationAccelerationWeight, config.confirmationAccelerationWeight),
    unit(genome.confirmationDistanceWeight, config.confirmationDistanceWeight),
    unit(genome.confirmationBias, config.confirmationBias),
    unit(genome.hysteresisReleaseRatio, config.hysteresisReleaseRatio),
    logUnit(genome.confirmationEmaMs, config.confirmationEma),
    logUnit(genome.confirmationEmaThresholdBpsHour, config.confirmationEmaThreshold),
    unit(genome.confirmationEmaWeight, config.confirmationEmaWeight),
    unit(genome.confirmationEmaGateStrength, config.confirmationEmaGateStrength),
    logUnit(genome.confirmationRsiMs, config.confirmationRsi),
    unit(genome.confirmationRsiThreshold, config.confirmationRsiThreshold),
    unit(genome.confirmationRsiWeight, config.confirmationRsiWeight),
    logUnit(genome.confirmationDmiMs, config.confirmationDmi),
    unit(genome.confirmationDmiWeight, config.confirmationDmiWeight),
    unit(genome.confirmationAdxThreshold, config.confirmationAdxThreshold),
    logUnit(genome.efficiencyVolumeEmaMs, config.efficiencyVolumeEma),
    sparseUnit(genome.efficiencyVolumePower, config.efficiencyVolumePower),
    sparseUnit(genome.meanReversionSuppressionThreshold, config.meanReversionSuppressionThreshold),
    logUnit(genome.meanReversionSlowMs, config.meanReversionSlow),
    logUnit(genome.meanReversionVolatilityMs, config.meanReversionVolatility),
    sparseUnit(genome.meanReversionReversalThreshold, config.meanReversionReversalThreshold),
    logUnit(genome.meanReversionEfficiencyMs, config.meanReversionEfficiency),
    logUnit(genome.meanReversionFastMs, config.meanReversionFast),
  ];
}

function vectorGenome(vector: number[], config: Args): Genome {
  const fastMs = logRange(vector[1]!, config.fast);
  const meanReversionFastMs = logRange(vector[41]!, config.meanReversionFast);
  return {
    efficiencyMs: logRange(vector[0]!, config.efficiency),
    fastMs,
    slowMs: Math.max(fastMs, logRange(vector[2]!, config.slow)),
    power: fromUnit(vector[3]!, config.power),
    volumeMs: logRange(vector[4]!, config.volume),
    volumeCap: fromUnit(vector[5]!, config.volumeCap),
    volumePower: sparseRange(vector[6]!, config.volumePower),
    deadbandBpsHour: logRange(vector[7]!, config.deadbandBpsHour),
    deadbandMode: config.deadbandModes.length === 1
      ? config.deadbandModes[0]!
      : config.deadbandModes[Math.min(
        config.deadbandModes.length - 1,
        Math.floor(clamp01(vector[8]!) * config.deadbandModes.length),
      )]!,
    thresholdLookbackMs: logRange(vector[9]!, config.thresholdLookback),
    thresholdNoiseMultiplier: fromUnit(vector[10]!, config.thresholdNoiseMultiplier),
    buyMaxFraction: fromUnit(vector[11]!, config.buyMaxFraction),
    sellMaxFraction: fromUnit(vector[12]!, config.sellMaxFraction),
    buySizingSigmaBpsHour: logRange(vector[13]!, config.buySizingSigma),
    sellSizingSigmaBpsHour: logRange(vector[14]!, config.sellSizingSigma),
    agreementMode: config.agreementModes.length === 1
      ? config.agreementModes[0]!
      : vector[15]! >= 0.5 ? "confidence" : "sizing",
    confirmationMix: fromUnit(vector[16]!, config.confirmationMix),
    confirmationMinQuality: fromUnit(vector[17]!, config.confirmationMinQuality),
    confirmationAccelerationLookbackMs: logRange(vector[18]!, config.confirmationAccelerationLookback),
    confirmationDistanceLookbackMs: logRange(vector[19]!, config.confirmationDistanceLookback),
    confirmationAccelerationWeight: fromUnit(vector[20]!, config.confirmationAccelerationWeight),
    confirmationDistanceWeight: fromUnit(vector[21]!, config.confirmationDistanceWeight),
    confirmationBias: fromUnit(vector[22]!, config.confirmationBias),
    hysteresisReleaseRatio: fromUnit(vector[23]!, config.hysteresisReleaseRatio),
    confirmationEmaMs: logRange(vector[24]!, config.confirmationEma),
    confirmationEmaThresholdBpsHour: logRange(vector[25]!, config.confirmationEmaThreshold),
    confirmationEmaWeight: fromUnit(vector[26]!, config.confirmationEmaWeight),
    confirmationEmaGateStrength: fromUnit(vector[27]!, config.confirmationEmaGateStrength),
    confirmationRsiMs: logRange(vector[28]!, config.confirmationRsi),
    confirmationRsiThreshold: fromUnit(vector[29]!, config.confirmationRsiThreshold),
    confirmationRsiWeight: fromUnit(vector[30]!, config.confirmationRsiWeight),
    confirmationDmiMs: logRange(vector[31]!, config.confirmationDmi),
    confirmationDmiWeight: fromUnit(vector[32]!, config.confirmationDmiWeight),
    confirmationAdxThreshold: fromUnit(vector[33]!, config.confirmationAdxThreshold),
    efficiencyVolumeEmaMs: logRange(vector[34]!, config.efficiencyVolumeEma),
    efficiencyVolumePower: sparseRange(vector[35]!, config.efficiencyVolumePower),
    meanReversionSuppressionThreshold: sparseRange(
      vector[36]!,
      config.meanReversionSuppressionThreshold,
    ),
    meanReversionSlowMs: Math.max(
      meanReversionFastMs,
      logRange(vector[37]!, config.meanReversionSlow),
    ),
    meanReversionVolatilityMs: logRange(vector[38]!, config.meanReversionVolatility),
    meanReversionReversalThreshold: sparseRange(
      vector[39]!,
      config.meanReversionReversalThreshold,
    ),
    meanReversionEfficiencyMs: logRange(vector[40]!, config.meanReversionEfficiency),
    meanReversionFastMs,
  };
}

interface WindowJob {
  config: Args;
  window: Window;
  windowId: string;
  index: number;
  bounds: { start: number; end: number };
  warmupMs: number;
  seeds: Candidate[];
  initialOnlySeeds?: boolean;
}

async function runPerWindow(
  config: Args,
  windows: Window[],
  bounds: { start: number; end: number },
  startedAt: number,
): Promise<void> {
  if (config.presetWindowIds && config.presetWindowIds.length !== windows.length) {
    throw new Error("--preset-window-ids must contain one id per fit window.");
  }
  const warmupMs = maximumWarmupMs(config);
  const existing = loadPresets(config.presetOutputPath);
  const importedSeeds = loadSeedCandidates(config.seedCandidatePaths);
  const singleWindow = windows.length === 1;
  const cuda = config.accelerator === "cpu" ? null : await vwKamaCudaStatus();
  if (config.accelerator === "cuda" && !cuda?.available) {
    throw new Error(`CUDA acceleration was requested but is unavailable: ${cuda?.reason}`);
  }
  const serialCudaWindows = !singleWindow && cuda?.available === true;
  const jobs = windows.map((window, index): WindowJob => ({
    config: {
      ...config,
      workers: singleWindow || serialCudaWindows ? config.workers : 1,
      accelerator: singleWindow || serialCudaWindows ? config.accelerator : "cpu",
      seed: config.seed + index * 10_007,
    },
    window,
    windowId: config.presetWindowIds?.[index] ?? window.label,
    index,
    bounds,
    warmupMs,
    seeds: dedupeSeedCandidates([
      ...importedSeeds,
      ...existing
        .filter((preset) => preset.scope === "window" && preset.windowId === (config.presetWindowIds?.[index] ?? window.label))
        .map(presetCandidate),
    ]),
    initialOnlySeeds: importedSeeds.length > 0,
  }));
  let grouped: Awaited<ReturnType<typeof optimizeWindow>>[];
  if (serialCudaWindows) {
    console.error(`Per-window CUDA: serializing ${jobs.length} optimizations through ${cuda.device}.`);
    grouped = [];
    for (const job of jobs) grouped.push(await optimizeWindow(job));
  } else {
    grouped = config.workers === 1 || singleWindow
      ? await Promise.all(jobs.map(optimizeWindow))
      : await parallelMap(jobs, Math.min(config.workers, jobs.length), runWindowWorker);
  }
  const generated = grouped.flat();
  const incumbentScores = new Map<string, number>();
  for (const preset of generated) {
    const incumbent = existing.find((item) =>
      item.scope === "window"
      && item.windowId === preset.windowId
      && item.intervalMs === preset.intervalMs);
    const job = jobs.find((item) => item.windowId === preset.windowId);
    if (!incumbent || !job) continue;
    const [evaluated] = await evaluateStageParallel(
      "fit",
      [job.window],
      [presetCandidate(incumbent)],
      bounds,
      warmupMs,
      { ...config, scales: [preset.intervalMs], accelerator: "cpu" },
    );
    const score = evaluated?.cases.find((item) => item.scale === formatDuration(preset.intervalMs));
    if (score) incumbentScores.set(`${preset.windowId}:${preset.intervalMs}`, score.score);
  }
  const completedAt = Date.now();
  const documented = generated.map((preset): VwKamaPreset => ({
    ...preset,
    generatedAt: new Date(completedAt).toISOString(),
    incumbentScore: incumbentScores.get(`${preset.windowId}:${preset.intervalMs}`),
    optimization: {
      algorithm: config.algorithm,
      objective: config.objective,
      population: config.trials,
      generations: config.generations,
      restarts: config.restarts,
      refinementRounds: config.refinementRounds,
      elapsedMs: completedAt - startedAt,
      hindsight: true,
      valueDistillation: {
        gridSize: config.exposureGridSize,
        minExposure: config.exposureMinimum,
        maxExposure: config.exposureMaximum,
        horizonMode: config.valueHorizonMode,
        horizonMs: config.valueHorizonMs,
        oracleTemperature: config.oracleTemperature,
        strategyTemperature: config.strategyTemperature,
        strategyVolatilityScaling: config.strategyVolatilityScaling,
        opportunityEpsilon: config.opportunityEpsilon,
        quoteLendRate: config.quoteLendRate,
        quoteBorrowRate: config.quoteBorrowRate,
        assetBorrowRate: config.assetBorrowRate,
      },
    },
  }));
  const replaced = new Set(documented.map((preset) => `${preset.windowId}:${preset.intervalMs ?? "all"}`));
  const presets = [
    ...existing.filter((preset) => !replaced.has(`${preset.windowId}:${preset.intervalMs ?? "all"}`)),
    ...documented,
  ].sort((left, right) => `${left.windowId}:${left.intervalMs ?? 0}`.localeCompare(
    `${right.windowId}:${right.intervalMs ?? 0}`,
  ));
  fs.mkdirSync(path.dirname(config.presetOutputPath), { recursive: true });
  fs.writeFileSync(config.presetOutputPath, `${JSON.stringify(presets, null, 2)}\n`);
  fs.mkdirSync(path.dirname(config.reportPath), { recursive: true });
  fs.writeFileSync(config.reportPath, [
    "# VW-KAMA per-window optimization",
    "",
    `Generated: ${new Date().toISOString()}`,
    `Score version: ${config.scoreVersion}`,
    `Algorithm: ${config.algorithm}; ${config.trials} population; ${config.generations} generations × ${config.restarts} restart(s); ${config.refinementRounds} refinement round(s); ${config.scales.map(formatDuration).join(", ")}.`,
    `Workers: ${config.workers}; ${windows.length === 1
      ? cuda?.available
        ? "single-window CUDA candidate batches"
        : "candidate-parallel single-window CPU search"
      : serialCudaWindows
        ? "serial windows with CUDA candidate batches"
        : "parallel CPU window searches"}.`,
    `Windows: ${windows.map((window, index) => `${config.presetWindowIds?.[index] ?? window.label} (${formatWindow(window)})`).join(", ")}.`,
    config.seedCandidatePaths.length > 0
      ? `Warm start: ${config.seedCandidatePaths.map((file) => `\`${path.relative(repoRoot, file)}\``).join(", ")}.`
      : "Warm start: existing matching presets only.",
    "",
    "These are hindsight upper-bound configurations for comparison, not deployable validation results.",
    "",
    "Every selected preset is scored against the global candidates at the same window and candle scale; the global score is therefore a hard lower bound.",
    "",
    "Existing presets are re-scored with the current evaluator before replacement; `incumbent` is therefore directly comparable even when evaluator semantics changed.",
    "",
    `| window | scale | ${config.objective === "value-distillation" ? "cross-entropy" : "score"} | incumbent | improvement | preset |`,
    "|---|---:|---:|---:|---:|---|",
    ...documented.map((preset) => {
      const incumbent = incumbentScores.get(`${preset.windowId}:${preset.intervalMs}`);
      return `| ${preset.windowId} | ${formatDuration(preset.intervalMs)} | ${formatSearchFitness(preset.score, config.objective)} | `
        + `${incumbent === undefined ? "—" : formatSearchFitness(incumbent, config.objective)} | `
        + `${incumbent === undefined
          ? "—"
          : config.objective === "value-distillation"
            ? round(preset.score - incumbent, 5)
            : signedPct(preset.score - incumbent)} | ${preset.id} |`;
    }),
    "",
    "## Selected parameters",
    "",
    "| preset | ER | ER volume EMA/power | fast | slow | KAMA power | post-ER volume EMA/cap/power | threshold | state/agreement | confirmation mix/gate | mean reversion |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ...documented.map((preset) => {
      const parameters = preset.parameters;
      return `| ${preset.id} | ${formatDuration(parameters.efficiencyMs)} | `
        + `${formatDuration(parameters.efficiencyVolumeEmaMs ?? parameters.volumeMs)} / ${round(parameters.efficiencyVolumePower ?? 0, 3)} | `
        + `${formatDuration(parameters.fastMs)} | ${formatDuration(parameters.slowMs)} | ${round(parameters.power, 3)} | `
        + `${formatDuration(parameters.volumeMs)} / ${round(parameters.volumeCap, 3)} / ${round(parameters.volumePower, 3)} | `
        + `${round(parameters.deadbandBpsHour, 3)} bps/h + noise×${round(parameters.thresholdNoiseMultiplier ?? 0, 3)} | `
        + `${parameters.deadbandMode}/${parameters.agreementMode ?? "sizing"} | `
        + `${round(parameters.confirmationMix ?? 0, 3)} / ${round(parameters.confirmationEmaGateStrength ?? 0, 3)} | `
        + `${formatDuration(parameters.meanReversionEfficiencyMs ?? parameters.efficiencyMs)} ER / ${formatDuration(parameters.meanReversionFastMs ?? parameters.fastMs)}–${formatDuration(parameters.meanReversionSlowMs ?? HOUR)} KAMA / ${formatDuration(parameters.meanReversionVolatilityMs ?? HOUR)} vol @ ${round(parameters.meanReversionSuppressionThreshold ?? 1, 3)}σ suppress / ${round(parameters.meanReversionReversalThreshold ?? 0, 3)}σ reverse |`;
    }),
    "",
  ].join("\n"));
  console.error(`Presets: ${path.relative(repoRoot, config.presetOutputPath)}`);
  console.error(`Completed in ${formatDuration(Date.now() - startedAt)}.`);
}

async function optimizeWindow(job: WindowJob) {
  const { config, window, bounds, warmupMs } = job;
  console.error(`Optimizing window ${job.index + 1}: ${formatWindow(window)}.`);
  const candidates = await searchCandidates(
    config,
    [window],
    bounds,
    warmupMs,
    job.seeds,
    job.initialOnlySeeds ?? false,
  );
  const evaluated = await evaluateStageParallel(
    "fit",
    [window],
    candidates,
    bounds,
    warmupMs,
    config,
  );
  return config.scales.map((intervalMs) => {
    const scale = formatDuration(intervalMs);
    const scored = evaluated.flatMap((result) => {
      const score = result.cases.find((item) => item.scale === scale);
      return score ? [{ result, score }] : [];
    }).sort((left, right) => right.score.score - left.score.score
      || left.result.candidate.id.localeCompare(right.result.candidate.id));
    const best = scored[0];
    const baseline = scored.find((item) => item.result.candidate.id.startsWith("global-"));
    if (!best || !baseline || best.score.score + Number.EPSILON < baseline.score.score) {
      throw new Error(`Global lower bound failed for ${window.label} at ${scale}.`);
    }
    return {
      id: `window-${job.windowId}-${scale}`,
      label: `Window best found · ${window.label} · ${scale}`,
      scope: "window" as const,
      windowId: job.windowId,
      intervalMs,
      parameters: candidateParameters(best.result.candidate),
      score: round(best.score.score, 6),
      scoreVersion: config.scoreVersion,
      source: `${config.algorithm.toUpperCase()} ${config.objective} hindsight best; global baseline retained`,
    };
  });
}

function runWindowWorker(job: WindowJob): Promise<Awaited<ReturnType<typeof optimizeWindow>>> {
  return new Promise((resolve, reject) => {
    const worker = fork(fileURLToPath(import.meta.url), [], {
      execArgv: ["--import", "tsx"],
      env: { ...process.env, VW_KAMA_WINDOW_JOB: JSON.stringify(job) },
      stdio: ["ignore", "ignore", "inherit", "ipc"],
    });
    worker.once("message", (message: Awaited<ReturnType<typeof optimizeWindow>> | { error: string }) => {
      worker.disconnect();
      if (!Array.isArray(message)) reject(new Error(message.error));
      else resolve(message);
    });
    worker.once("error", reject);
    worker.once("exit", (code) => {
      if (code !== 0) reject(new Error(`VW-KAMA optimizer worker exited with code ${code}.`));
    });
  });
}

async function parallelMap<T, R>(
  values: T[],
  workers: number,
  operation: (value: T) => Promise<R>,
): Promise<R[]> {
  const results = new Array<R>(values.length);
  let next = 0;
  await Promise.all(Array.from({ length: workers }, async () => {
    while (next < values.length) {
      const index = next++;
      results[index] = await operation(values[index]!);
    }
  }));
  return results;
}

function loadPresets(file: string): VwKamaPreset[] {
  if (!fs.existsSync(file)) return [];
  const parsed = JSON.parse(fs.readFileSync(file, "utf8")) as unknown;
  return Array.isArray(parsed) ? parsed as VwKamaPreset[] : [];
}

function loadSeedCandidates(files: string[]): Candidate[] {
  const loaded = files.flatMap((file) => {
    if (!fs.existsSync(file)) throw new Error(`Seed candidate file does not exist: ${file}`);
    const text = fs.readFileSync(file, "utf8");
    if (file.endsWith(".jsonl")) {
      return text.split("\n").flatMap((line, index) => {
        if (!line.trim()) return [];
        const entry = JSON.parse(line) as { type?: string; stage?: string; candidate?: Record<string, unknown> };
        return entry.type === "candidate" && entry.stage === "fit" && entry.candidate
          ? [storedCandidate(entry.candidate, index)]
          : [];
      });
    }
    const parsed = JSON.parse(text) as unknown;
    if (!Array.isArray(parsed)) throw new Error(`Seed preset file must contain an array: ${file}`);
    return (parsed as VwKamaPreset[]).map(presetCandidate);
  });
  return dedupeSeedCandidates(loaded);
}

function dedupeSeedCandidates(candidates: Candidate[]): Candidate[] {
  const unique = [...new Map(candidates.map((candidate) => [
    JSON.stringify(candidateParameters(candidate)),
    candidate,
  ])).values()];
  const pairedVolume = new Set(unique
    .filter(volumeAware)
    .map(seedPairKey));
  return unique
    .filter((candidate) => volumeAware(candidate) || !pairedVolume.has(seedPairKey(candidate)))
    .map((candidate, index) => ({ ...candidate, id: `warm-${String(index + 1).padStart(4, "0")}` }));
}

function dedupeCandidates(candidates: Candidate[]): Candidate[] {
  return [...new Map(candidates.map((candidate) => [candidate.id, candidate])).values()];
}

function dedupeCandidateResults(results: CandidateResult[]): CandidateResult[] {
  return [...new Map(results.map((result) => [result.candidate.id, result])).values()];
}

function seedPairKey(candidate: Candidate): string {
  return JSON.stringify({
    ...candidateParameters(candidate),
    volumePower: 0,
    efficiencyVolumePower: 0,
  });
}

function volumeAware(candidate: Pick<Candidate, "volumePower" | "efficiencyVolumePower">): boolean {
  return candidate.volumePower > 0 || candidate.efficiencyVolumePower > 0;
}

function storedCandidate(value: Record<string, unknown>, index: number): Candidate {
  const duration = (msKey: string, formattedKey: string, fallback?: number): number => {
    const direct = value[msKey];
    if (typeof direct === "number" && Number.isFinite(direct) && direct > 0) return direct;
    const formatted = value[formattedKey];
    if (typeof formatted === "string") return parseDuration(formatted);
    if (fallback !== undefined) return fallback;
    throw new Error(`Stored candidate ${index} is missing ${formattedKey}.`);
  };
  const number = (key: string, fallback: number): number => {
    const result = value[key];
    return typeof result === "number" && Number.isFinite(result) ? result : fallback;
  };
  const text = <T extends string>(key: string, allowed: readonly T[], fallback: T): T => {
    const result = value[key];
    return typeof result === "string" && allowed.includes(result as T) ? result as T : fallback;
  };
  const volumePower = number("volumePower", 0);
  const efficiencyVolumePower = number("efficiencyVolumePower", 0);
  const volumeMs = duration("volumeMs", "volume");
  return {
    id: typeof value.id === "string" ? value.id : `stored-${index}`,
    family: value.family === "canonical" || (volumePower === 0 && efficiencyVolumePower === 0)
      ? "canonical"
      : "volume",
    efficiencyMs: duration("efficiencyMs", "efficiency"),
    efficiencyVolumeEmaMs: duration(
      "efficiencyVolumeEmaMs",
      "efficiencyVolumeEma",
      volumeMs,
    ),
    efficiencyVolumePower,
    fastMs: duration("fastMs", "fast"),
    slowMs: duration("slowMs", "slow"),
    power: number("power", 1),
    volumeMs,
    volumeCap: number("volumeCap", 1),
    volumePower,
    deadbandBpsHour: number("deadbandBpsHour", 0),
    deadbandMode: text("deadbandMode", ["flat", "hold", "hysteresis"] as const, "hold"),
    hysteresisReleaseRatio: number("hysteresisReleaseRatio", 0.25),
    thresholdLookbackMs: duration("thresholdLookbackMs", "thresholdLookback", HOUR),
    thresholdNoiseMultiplier: number("thresholdNoiseMultiplier", 0),
    buyMaxFraction: number("buyMaxFraction", 1),
    sellMaxFraction: number("sellMaxFraction", 1),
    buySizingSigmaBpsHour: number("buySizingSigmaBpsHour", 1e12),
    sellSizingSigmaBpsHour: number("sellSizingSigmaBpsHour", 1e12),
    agreementMode: text("agreementMode", ["sizing", "confidence"] as const, "sizing"),
    confirmationMix: number("confirmationMix", 0),
    confirmationMinQuality: number("confirmationMinQuality", 0),
    confirmationAccelerationLookbackMs: duration(
      "confirmationAccelerationLookbackMs",
      "confirmationAccelerationLookback",
      HOUR,
    ),
    confirmationDistanceLookbackMs: duration(
      "confirmationDistanceLookbackMs",
      "confirmationDistanceLookback",
      HOUR,
    ),
    confirmationAccelerationWeight: number("confirmationAccelerationWeight", 1),
    confirmationDistanceWeight: number("confirmationDistanceWeight", 1),
    confirmationBias: number("confirmationBias", 0),
    confirmationEmaMs: duration("confirmationEmaMs", "confirmationEma", HOUR),
    confirmationEmaThresholdBpsHour: number("confirmationEmaThresholdBpsHour", 0),
    confirmationEmaWeight: number("confirmationEmaWeight", 0),
    confirmationEmaGateStrength: number("confirmationEmaGateStrength", 0),
    confirmationRsiMs: duration("confirmationRsiMs", "confirmationRsi", 14 * 60_000),
    confirmationRsiThreshold: number("confirmationRsiThreshold", 0),
    confirmationRsiWeight: number("confirmationRsiWeight", 0),
    confirmationDmiMs: duration("confirmationDmiMs", "confirmationDmi", 14 * 60_000),
    confirmationDmiWeight: number("confirmationDmiWeight", 0),
    confirmationAdxThreshold: number("confirmationAdxThreshold", 20),
    meanReversionSuppressionThreshold: number("meanReversionSuppressionThreshold", 1),
    meanReversionEfficiencyMs: duration(
      "meanReversionEfficiencyMs",
      "meanReversionEfficiency",
      HOUR,
    ),
    meanReversionFastMs: duration("meanReversionFastMs", "meanReversionFast", 15 * 60_000),
    meanReversionSlowMs: duration("meanReversionSlowMs", "meanReversionSlow", HOUR),
    meanReversionVolatilityMs: duration(
      "meanReversionVolatilityMs",
      "meanReversionVolatility",
      HOUR,
    ),
    meanReversionReversalThreshold: number("meanReversionReversalThreshold", 0),
  };
}

async function evaluateStageParallel(
  stage: Stage,
  windows: Window[],
  candidates: Candidate[],
  bounds: { start: number; end: number },
  warmupMs: number,
  config: Args,
): Promise<CandidateResult[]> {
  const gpu = await evaluateStageCuda(stage, windows, candidates, bounds, warmupMs, config);
  if (gpu) return gpu;
  const workers = Math.min(config.workers, candidates.length);
  if (workers <= 1 || candidates.length < workers * 4) {
    return evaluateStage(stage, windows, candidates, bounds, warmupMs, config);
  }
  const prepared = await prepareStageWindow(stage, windows, bounds, warmupMs, config);
  const batchCount = Math.min(candidates.length, workers * 4);
  const shards = Array.from({ length: batchCount }, (): Candidate[] => []);
  const loads = Array.from({ length: batchCount }, () => 0);
  const weighted = candidates.map((candidate) => ({
    candidate,
    work: estimatedCandidateWork(candidate, config),
  })).sort((left, right) =>
    right.work - left.work
    || candidateIdentity(left.candidate).localeCompare(candidateIdentity(right.candidate)));
  for (const item of weighted) {
    let shard = 0;
    for (let index = 1; index < batchCount; index += 1) {
      if (loads[index]! < loads[shard]!
        || (loads[index] === loads[shard] && shards[index]!.length < shards[shard]!.length)) {
        shard = index;
      }
    }
    shards[shard]!.push(item.candidate);
    loads[shard]! += item.work;
  }
  const jobs = shards.map((shard): CandidateJob => ({
    stage,
    windows,
    candidates: shard,
    prepared,
    config: { ...config, workers: 1 },
  })).filter((job) => job.candidates.length > 0);
  const pool = candidatePoolFor(workers);
  const nonzeroLoads = loads.filter((load) => load > 0);
  const imbalance = Math.max(...nonzeroLoads) / Math.min(...nonzeroLoads);
  console.error(
    `Parallel ${stage}: ${candidates.length} candidates in ${jobs.length} dynamic batches across `
    + `${workers} shared-memory workers; estimated batch imbalance ${imbalance.toFixed(3)}×.`,
  );
  const order = new Map(candidates.map((candidate, index) => [candidateIdentity(candidate), index]));
  if (order.size !== candidates.length) throw new Error("Parallel candidate ids must be unique.");
  return (await pool.runAll(jobs))
    .flat()
    .sort((left, right) => order.get(candidateIdentity(left.candidate))! - order.get(candidateIdentity(right.candidate))!);
}

async function evaluateStageCuda(
  stage: Stage,
  windows: Window[],
  candidates: Candidate[],
  bounds: { start: number; end: number },
  warmupMs: number,
  config: Args,
): Promise<CandidateResult[] | null> {
  if (config.accelerator === "cpu" || candidates.length === 0) return null;
  if (config.accelerator === "auto" && candidates.length < CUDA_AUTO_MIN_CANDIDATES) return null;
  const status = await vwKamaCudaStatus();
  if (!status.available) {
    if (config.accelerator === "cuda") {
      throw new Error(`CUDA acceleration was requested but is unavailable: ${status.reason}`);
    }
    if (!cudaAutoFailureReported) {
      console.error(`CUDA unavailable; using CPU workers: ${status.reason}`);
      cudaAutoFailureReported = true;
    }
    return null;
  }
  if (!cudaDeviceReported) {
    console.error(
      `CUDA acceleration: ${status.device}; ${config.accelerator === "cuda"
        ? "forced for every non-empty batch"
        : `auto uses batches of ${CUDA_AUTO_MIN_CANDIDATES}+ candidates`}.`,
    );
    cudaDeviceReported = true;
  }
  const prepared = await prepareStageWindow(stage, windows, bounds, warmupMs, config);
  const grouped = candidates.map(() => new Map<string, CaseStats[]>());
  const parameters = candidates.map(candidateParameters);
  let kernelMs = 0;
  const wallStarted = performance.now();
  for (const testCase of prepared.cases) {
    if (Array.isArray(testCase.candles) || !testCase.preparedOracle) {
      if (config.accelerator === "cuda") {
        throw new Error("CUDA evaluation requires prepared columnar candles and oracle states.");
      }
      return null;
    }
    const results = await evaluateVwKamaCudaBatch(
      testCase.candles as VwKamaCandleColumns,
      testCase.preparedOracle,
      parameters,
      {
        intervalMs: testCase.scaleMs,
        scoreStartIndex: testCase.scoreStart,
        oracleFriction: config.oracleFriction,
        matchWindowMs: config.matchWindowMs,
        timingHalfLifeMs: config.timingHalfLifeMs,
        warmupMultiple: config.warmupMultiple,
        valueDistillation: testCase.valueOracle
          ? {
              oracle: testCase.valueOracle,
              strategyTemperature: config.strategyTemperature,
              strategyVolatilityScaling: config.strategyVolatilityScaling,
              strategyTemperatures: testCase.strategyTemperatures,
            }
          : undefined,
      },
    );
    kernelMs += results[0]?.elapsedMs ?? 0;
    for (let index = 0; index < candidates.length; index += 1) {
      const result = results[index]!;
      const parts = grouped[index]!.get(testCase.id) ?? [];
      parts.push({
        caseId: testCase.id,
        scale: formatDuration(testCase.scaleMs),
        window: testCase.window.label,
        timingCredit: result.timingCredit,
        matchedCount: result.matchedCount,
        signalCount: result.signalCount,
        oracleCount: result.oracleCount,
        stateCredit: result.stateCredit,
        stateCount: result.stateCount,
        distillationWeightedCrossEntropy: result.distillationWeightedCrossEntropy,
        distillationWeightedOracleEntropy: result.distillationWeightedOracleEntropy,
        distillationWeight: result.distillationWeight,
        distillationOpportunity: result.distillationOpportunity,
        distillationSamples: testCase.valueOracle ? result.stateCount : 0,
        valueHorizonMs: testCase.valueHorizonMs,
        strategyFinalEquity: testCase.valueOracle ? result.strategyFinalEquity : undefined,
        oracleFinalEquity: testCase.valueOracle ? result.oracleFinalEquity : undefined,
        strategyMaxDrawdown: testCase.valueOracle ? result.strategyMaxDrawdown : undefined,
        oracleMaxDrawdown: testCase.valueOracle ? result.oracleMaxDrawdown : undefined,
        strategyTurnover: testCase.valueOracle ? result.strategyTurnover : undefined,
        oracleTurnover: testCase.valueOracle ? result.oracleTurnover : undefined,
        days: testCase.days,
        absoluteLags: [],
        signedLags: [],
        lagP50Ms: result.lagP50Ms,
        lagP90Ms: result.lagP90Ms,
        lagP95Ms: result.lagP95Ms,
        lagMedianSignedMs: result.lagMedianSignedMs,
      });
      grouped[index]!.set(testCase.id, parts);
    }
  }
  const results = candidates.map((candidate, index): CandidateResult => {
    const cases = orderCases(
      [...grouped[index]!.values()].map((parts) => scoreCase(parts, config)),
      windows,
      config.scales,
    );
    return { candidate, stage, aggregate: aggregate(cases), cases };
  });
  console.error(
    `${capitalize(stage)} CUDA: ${candidates.length} candidates × ${prepared.cases.length} segment(s) in `
    + `${(performance.now() - wallStarted).toFixed(1)} ms (${kernelMs.toFixed(1)} ms kernels).`,
  );
  return results;
}

function estimatedCandidateWork(candidate: Candidate, config: Args): number {
  const multiple = config.warmupMultiple;
  const thresholdEnabled = candidate.thresholdNoiseMultiplier > 0;
  const confirmationEnabled = candidate.confirmationMix > 0;
  const accelerationEnabled = confirmationEnabled && candidate.confirmationAccelerationWeight > 0;
  const distanceEnabled = confirmationEnabled && candidate.confirmationDistanceWeight > 0;
  const emaEnabled = confirmationEnabled && candidate.confirmationEmaWeight > 0
    || candidate.confirmationEmaGateStrength > 0;
  const rsiEnabled = confirmationEnabled && candidate.confirmationRsiWeight > 0;
  const dmiEnabled = confirmationEnabled && candidate.confirmationDmiWeight > 0;
  return Math.max(
    candidate.efficiencyMs + Math.min(...config.scales),
    candidate.efficiencyVolumePower > 0 ? candidate.efficiencyVolumeEmaMs : 0,
    candidate.slowMs,
    candidate.volumeMs,
    thresholdEnabled ? candidate.thresholdLookbackMs : 0,
    accelerationEnabled ? candidate.confirmationAccelerationLookbackMs : 0,
    distanceEnabled ? candidate.confirmationDistanceLookbackMs : 0,
    emaEnabled ? candidate.confirmationEmaMs : 0,
    rsiEnabled ? candidate.confirmationRsiMs : 0,
    dmiEnabled ? candidate.confirmationDmiMs * 2 : 0,
    candidate.meanReversionReversalThreshold > 0
      ? Math.max(
        candidate.meanReversionEfficiencyMs,
        candidate.meanReversionFastMs,
        candidate.meanReversionSlowMs,
        candidate.meanReversionVolatilityMs,
      )
      : 0,
  ) * multiple;
}

function candidateIdentity(candidate: Candidate): string {
  return `${candidate.family}:${candidate.id}:${candidate.agreementMode}`;
}

class CandidatePool {
  readonly workers: Worker[];
  private nextId = 1;

  constructor(count: number) {
    const tsxRoot = path.dirname(createRequire(import.meta.url).resolve("tsx/package.json"));
    const tsxApi = pathToFileURL(path.join(tsxRoot, "dist/esm/api/index.mjs")).href;
    const entry = new URL(`data:text/javascript,${encodeURIComponent(
      `import { tsImport } from ${JSON.stringify(tsxApi)};\n`
      + `import { workerData } from "node:worker_threads";\n`
      + `await tsImport(workerData.module, import.meta.url);`,
    )}`);
    this.workers = Array.from({ length: count }, () => new Worker(entry, {
      workerData: { kind: "vw-kama-candidate", module: import.meta.url },
    }));
  }

  async runAll(jobs: CandidateJob[]): Promise<CandidateResult[][]> {
    let next = 0;
    const results = Array.from({ length: jobs.length }, (): CandidateResult[] => []);
    await Promise.all(this.workers.map(async (_, workerIndex) => {
      while (next < jobs.length) {
        const index = next++;
        results[index] = await this.run(workerIndex, jobs[index]!);
      }
    }));
    return results;
  }

  run(index: number, job: CandidateJob): Promise<CandidateResult[]> {
    const worker = this.workers[index]!;
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      const cleanup = (): void => {
        worker.off("message", onMessage);
        worker.off("error", onError);
        worker.off("exit", onExit);
      };
      const onMessage = (message: CandidateWorkerResponse): void => {
        if (message.id !== id) return;
        cleanup();
        if (message.error) reject(new Error(message.error));
        else resolve(message.results ?? []);
      };
      const onError = (error: Error): void => { cleanup(); reject(error); };
      const onExit = (code: number | null): void => {
        cleanup();
        reject(new Error(`VW-KAMA candidate worker exited with code ${code}.`));
      };
      worker.on("message", onMessage);
      worker.once("error", onError);
      worker.once("exit", onExit);
      worker.postMessage({ id, job } satisfies CandidateWorkerRequest);
    });
  }

  close(): void {
    for (const worker of this.workers) void worker.terminate();
  }
}

function candidatePoolFor(workers: number): CandidatePool {
  if (candidatePool && candidatePool.workers.length !== workers) {
    candidatePool.close();
    candidatePool = null;
  }
  candidatePool ??= new CandidatePool(workers);
  return candidatePool;
}

function closeCandidatePool(): void {
  candidatePool?.close();
  candidatePool = null;
}

async function evaluateStage(
  stage: Stage,
  windows: Window[],
  candidates: Candidate[],
  bounds: { start: number; end: number },
  warmupMs: number,
  config: Args,
): Promise<CandidateResult[]> {
  const prepared = await prepareStageWindow(stage, windows, bounds, warmupMs, config);
  return evaluatePreparedStage(stage, windows, candidates, prepared, config);
}

const preparedStageCache = new Map<string, PreparedStageWindow>();
const MAX_PREPARED_BYTES = 1_500_000_000;
let preparedStageBytes = 0;

async function prepareStageWindow(
  stage: Stage,
  windows: Window[],
  bounds: { start: number; end: number },
  warmupMs: number,
  config: Args,
  forcedValueOracleBackend?: ValueOracleBackend,
): Promise<PreparedStageWindow> {
  const valueOracleBackend = forcedValueOracleBackend
    ?? await resolveValueOracleBackend(config);
  const key = JSON.stringify({
    stage,
    sourceDir: config.sourceDir,
    sourceIntervalMs: config.sourceIntervalMs,
    scales: config.scales,
    windows,
    warmupMs,
    oracleFriction: config.oracleFriction,
    objective: config.objective,
    exposureGridSize: config.exposureGridSize,
    exposureMinimum: config.exposureMinimum,
    exposureMaximum: config.exposureMaximum,
    valueHorizonMode: config.valueHorizonMode,
    valueHorizonMs: config.valueHorizonMs,
    oracleTemperature: config.oracleTemperature,
    strategyTemperature: config.strategyTemperature,
    strategyVolatilityScaling: config.strategyVolatilityScaling,
    opportunityEpsilon: config.opportunityEpsilon,
    quoteLendRate: config.quoteLendRate,
    quoteBorrowRate: config.quoteBorrowRate,
    assetBorrowRate: config.assetBorrowRate,
    valueOracleBackend,
  });
  const cached = preparedStageCache.get(key);
  if (cached) {
    preparedStageCache.delete(key);
    preparedStageCache.set(key, cached);
    return cached;
  }
  const prepared: PreparedStageWindow = {
    key,
    sourceCount: 0,
    segmentCount: 0,
    cases: [],
    bytes: 0,
    valueOracleBackend,
    valueOracleKernelMs: 0,
  };
  try {
    for (const window of windows) {
      const sourceStart = Math.max(bounds.start, window.start - warmupMs);
      for (const source of loadSourceSegments(config.sourceDir, {
        start: sourceStart,
        end: window.end,
      }, config.sourceIntervalMs)) {
        prepared.sourceCount += source.length;
        const built = await buildCases(
          stage,
          [window],
          config.scales,
          source,
          warmupMs,
          config,
          valueOracleBackend,
          false,
        );
        if (built.cases.length === 0) continue;
        prepared.segmentCount += 1;
        prepared.valueOracleKernelMs += built.valueOracleKernelMs;
        for (const testCase of built.cases) {
          const candles = columnarVwKamaCandles(testCase.candles as TradingCandle[], true);
          const preparedOracle = prepareVwKamaOracle(
            candles,
            testCase.scoreStart,
            testCase.oracle!,
            true,
          );
          const valueOracle = testCase.valueOracle
            ? shareExposureValueOracle(testCase.valueOracle)
            : undefined;
          const strategyTemperatures = testCase.strategyTemperatures;
          prepared.bytes += candleColumnBytes(candles)
            + preparedOracle.stateCodes.byteLength
            + (valueOracle ? exposureValueOracleBytes(valueOracle) : 0)
            + (strategyTemperatures?.byteLength ?? 0);
          prepared.cases.push({
            ...testCase,
            candles,
            oracle: undefined,
            preparedOracle,
            valueOracle,
            strategyTemperatures,
          });
        }
      }
    }
  } catch (error) {
    if (valueOracleBackend === "cuda" && config.accelerator === "auto") {
      console.error(
        `CUDA value-oracle preparation failed; retrying on CPU: ${error instanceof Error ? error.message : String(error)}`,
      );
      return prepareStageWindow(stage, windows, bounds, warmupMs, config, "cpu");
    }
    throw error;
  }
  if (prepared.cases.length === 0) {
    throw new Error(`Insufficient continuous candles for ${windows.map((item) => item.label).join(", ")}.`);
  }
  while (preparedStageCache.size > 0 && preparedStageBytes + prepared.bytes > MAX_PREPARED_BYTES) {
    const oldest = preparedStageCache.entries().next().value as [string, PreparedStageWindow] | undefined;
    if (!oldest) break;
    preparedStageCache.delete(oldest[0]);
    preparedStageBytes -= oldest[1].bytes;
  }
  preparedStageCache.set(key, prepared);
  preparedStageBytes += prepared.bytes;
  const resolvedHorizons = prepared.cases.flatMap((item) => item.valueHorizonMs === undefined
    ? []
    : [item.valueHorizonMs]);
  const horizonLabel = resolvedHorizons.length === 0
    ? ""
    : `; resolved H ${formatDuration(Math.min(...resolvedHorizons))}${Math.min(...resolvedHorizons) === Math.max(...resolvedHorizons)
      ? ""
      : `..${formatDuration(Math.max(...resolvedHorizons))}`}`;
  console.error(
    `${capitalize(stage)} prepared ${windows.map((item) => item.label).join(", ")}: `
    + `${prepared.sourceCount.toLocaleString()} source candles, ${prepared.segmentCount} segment(s), `
    + `${formatBytes(prepared.bytes)} shared columns; value oracle ${prepared.valueOracleBackend.toUpperCase()}`
    + `${prepared.valueOracleBackend === "cuda"
      ? ` (${prepared.valueOracleKernelMs.toFixed(1)} ms kernels)`
      : ""}${horizonLabel}; cached for subsequent generations.`,
  );
  return prepared;
}

async function resolveValueOracleBackend(config: Args): Promise<ValueOracleBackend> {
  if (config.objective === "value-distillation"
    && config.accelerator === "cuda"
    && config.exposureGridSize > 1_024) {
    throw new Error("CUDA value-oracle preparation supports at most 1,024 exposure grid points.");
  }
  if (config.objective !== "value-distillation"
    || config.accelerator === "cpu"
    || config.exposureGridSize > 1_024
    || (config.accelerator === "auto"
      && config.exposureGridSize < VW_KAMA_CUDA_VALUE_ORACLE_AUTO_MIN_GRID_SIZE)) return "cpu";
  const status = await vwKamaCudaStatus();
  return status.available ? "cuda" : "cpu";
}

function evaluatePreparedStage(
  stage: Stage,
  windows: Window[],
  candidates: Candidate[],
  prepared: PreparedStageWindow,
  config: Args,
): CandidateResult[] {
  const progressEvery = Math.max(10, Math.ceil(candidates.length / 10));
  const results = candidates.map((candidate, index): CandidateResult => {
    const grouped = new Map<string, CaseStats[]>();
    for (const testCase of prepared.cases) {
      const parts = grouped.get(testCase.id) ?? [];
      parts.push(evaluateCase(candidate, testCase, config));
      grouped.set(testCase.id, parts);
    }
    const cases = orderCases(
      [...grouped.values()].map((parts) => scoreCase(parts, config)),
      windows,
      config.scales,
    );
    if ((index + 1) % progressEvery === 0 || index + 1 === candidates.length) {
      console.error(`${capitalize(stage)} shared: ${index + 1}/${candidates.length} candidates.`);
    }
    return { candidate, stage, aggregate: aggregate(cases), cases };
  });
  console.error(
    `${capitalize(stage)} shared: ${prepared.sourceCount.toLocaleString()} source candles `
    + `across ${prepared.segmentCount} scored segments.`,
  );
  return results;
}

function candleColumnBytes(candles: VwKamaCandleSeries): number {
  if (!("openTime" in candles)) return 0;
  return candles.openTime.byteLength
    + candles.closeTime.byteLength
    + candles.open.byteLength
    + candles.high.byteLength
    + candles.low.byteLength
    + candles.close.byteLength
    + candles.volume.byteLength;
}

function formatBytes(bytes: number): string {
  return `${(bytes / 1_000_000).toFixed(1)} MB`;
}

function orderCases(cases: CaseScore[], windows: Window[], scales: number[]): CaseScore[] {
  const order = new Map(scales.flatMap((scale) => windows.map((window) =>
    `${formatDuration(scale)}:${window.label}`)).map((id, index) => [id, index]));
  return cases.slice().sort((left, right) =>
    (order.get(left.caseId) ?? Number.MAX_SAFE_INTEGER)
    - (order.get(right.caseId) ?? Number.MAX_SAFE_INTEGER));
}

async function buildCases(
  stage: Stage,
  windows: Window[],
  scales: number[],
  source: TradingCandle[],
  warmupMs: number,
  config: Args,
  valueOracleBackend: ValueOracleBackend,
  required = true,
): Promise<{ cases: CaseData[]; valueOracleKernelMs: number }> {
  const result: CaseData[] = [];
  let valueOracleKernelMs = 0;
  for (const scaleMs of scales) {
    for (const window of windows) {
      const relevant = source.filter((candle) =>
        candle.openTime >= window.start - warmupMs && candle.openTime < window.end);
      const segments = continuousSegments(
        aggregateCandles(relevant, config.sourceIntervalMs, scaleMs),
        scaleMs,
      );
      const cases: CaseData[] = [];
      for (const candles of segments) {
        const scoreAt = Math.max(window.start, (candles[0]?.openTime ?? window.end) + warmupMs);
        const scoreStart = candles.findIndex((candle) => candle.openTime >= scoreAt);
        if (scoreStart < 0 || candles.length - scoreStart < 3) continue;
        const scored = candles.slice(scoreStart).filter((candle) => candle.openTime < window.end);
        if (scored.length < 3) continue;
        const caseCandles = candles.slice(0, scoreStart).concat(scored);
        const oracle = perfectMarginOracle(caseCandles, {
          startingQuote: 1,
          leverage: 1,
          friction: config.oracleFriction,
          eventMode: "close",
          maxPathCandles: 1,
        });
        let valueOracle: ExposureValueOracle | undefined;
        let strategyTemperatures: Float32Array | undefined;
        let valueHorizonMs: number | undefined;
        if (config.objective === "value-distillation") {
          const prices = caseCandles.map((candle) => candle.close);
          const horizonSteps = resolveVwKamaValueHorizonSteps(
            caseCandles,
            scoreStart,
            oracle.stateCodes,
            scaleMs,
            { horizonMode: config.valueHorizonMode, horizonMs: config.valueHorizonMs },
          );
          valueHorizonMs = horizonSteps * scaleMs;
          const options = {
            scoreStartIndex: scoreStart,
            horizonSteps,
            friction: config.oracleFriction,
            gridSize: config.exposureGridSize,
            minExposure: config.exposureMinimum,
            maxExposure: config.exposureMaximum,
            temperature: config.oracleTemperature,
            opportunityEpsilon: config.opportunityEpsilon,
            quoteLendRate: config.quoteLendRate,
            quoteBorrowRate: config.quoteBorrowRate,
            assetBorrowRate: config.assetBorrowRate,
          };
          if (valueOracleBackend === "cuda") {
            const prepared = await prepareExposureValueOracleCuda(
              prices,
              options,
            );
            valueOracle = prepared.oracle;
            valueOracleKernelMs += prepared.kernelMs;
          } else {
            valueOracle = prepareExposureValueOracle(
              prices,
              options,
            );
          }
          strategyTemperatures = strategyExposureTemperatures(prices, {
            intervalMs: scaleMs,
            horizonSteps,
            temperature: config.strategyTemperature,
            scaleByVolatility: config.strategyVolatilityScaling,
          }, true);
        }
        cases.push({
          id: `${formatDuration(scaleMs)}:${window.label}`,
          stage,
          scaleMs,
          window,
          candles: caseCandles,
          scoreStart,
          oracle,
          valueOracle,
          strategyTemperatures,
          valueHorizonMs,
          days: Math.max(scaleMs / DAY, scored.length * scaleMs / DAY),
        });
      }
      if (required && cases.length === 0) {
        throw new Error(`Insufficient continuous candles for ${window.label} at ${formatDuration(scaleMs)}.`);
      }
      result.push(...cases);
    }
  }
  return { cases: result, valueOracleKernelMs };
}

function evaluateCase(candidate: Candidate, testCase: CaseData, config: Args): CaseStats {
  const result = evaluateVwKamaOracle(testCase.candles, {
    intervalMs: testCase.scaleMs,
    scoreStartTime: testCase.window.start,
    scoreStartIndex: testCase.scoreStart,
    parameters: {
      efficiencyMs: candidate.efficiencyMs,
      efficiencyVolumeEmaMs: candidate.efficiencyVolumeEmaMs,
      efficiencyVolumePower: candidate.efficiencyVolumePower,
      fastMs: candidate.fastMs,
      slowMs: candidate.slowMs,
      power: candidate.power,
      volumeMs: candidate.volumeMs,
      volumeCap: candidate.volumeCap,
      volumePower: candidate.volumePower,
      deadbandBpsHour: candidate.deadbandBpsHour,
      deadbandMode: candidate.deadbandMode,
      hysteresisReleaseRatio: candidate.hysteresisReleaseRatio,
      thresholdLookbackMs: candidate.thresholdLookbackMs,
      thresholdNoiseMultiplier: candidate.thresholdNoiseMultiplier,
      buyMaxFraction: candidate.buyMaxFraction,
      sellMaxFraction: candidate.sellMaxFraction,
      buySizingSigmaBpsHour: candidate.buySizingSigmaBpsHour,
      sellSizingSigmaBpsHour: candidate.sellSizingSigmaBpsHour,
      agreementMode: candidate.agreementMode,
      confirmationMix: candidate.confirmationMix,
      confirmationMinQuality: candidate.confirmationMinQuality,
      confirmationAccelerationLookbackMs: candidate.confirmationAccelerationLookbackMs,
      confirmationDistanceLookbackMs: candidate.confirmationDistanceLookbackMs,
      confirmationAccelerationWeight: candidate.confirmationAccelerationWeight,
      confirmationDistanceWeight: candidate.confirmationDistanceWeight,
      confirmationBias: candidate.confirmationBias,
      confirmationEmaMs: candidate.confirmationEmaMs,
      confirmationEmaThresholdBpsHour: candidate.confirmationEmaThresholdBpsHour,
      confirmationEmaWeight: candidate.confirmationEmaWeight,
      confirmationEmaGateStrength: candidate.confirmationEmaGateStrength,
      confirmationRsiMs: candidate.confirmationRsiMs,
      confirmationRsiThreshold: candidate.confirmationRsiThreshold,
      confirmationRsiWeight: candidate.confirmationRsiWeight,
      confirmationDmiMs: candidate.confirmationDmiMs,
      confirmationDmiWeight: candidate.confirmationDmiWeight,
      confirmationAdxThreshold: candidate.confirmationAdxThreshold,
      meanReversionSuppressionThreshold: candidate.meanReversionSuppressionThreshold,
      meanReversionEfficiencyMs: candidate.meanReversionEfficiencyMs,
      meanReversionFastMs: candidate.meanReversionFastMs,
      meanReversionSlowMs: candidate.meanReversionSlowMs,
      meanReversionVolatilityMs: candidate.meanReversionVolatilityMs,
      meanReversionReversalThreshold: candidate.meanReversionReversalThreshold,
    },
    oracleFriction: config.oracleFriction,
    matchWindowMs: config.matchWindowMs,
    timingHalfLifeMs: config.timingHalfLifeMs,
    warmupMultiple: config.warmupMultiple,
    oracleResult: testCase.oracle,
    preparedOracle: testCase.preparedOracle,
    includeTrace: false,
    valueDistillation: testCase.valueOracle
      ? {
          oracle: testCase.valueOracle,
          strategyTemperature: config.strategyTemperature,
          strategyVolatilityScaling: config.strategyVolatilityScaling,
          strategyTemperatures: testCase.strategyTemperatures,
        }
      : undefined,
  });
  const metrics = result.metrics;
  const distillation = metrics.valueDistillation;
  return {
    caseId: testCase.id,
    scale: formatDuration(testCase.scaleMs),
    window: testCase.window.label,
    timingCredit: result.candidateTransitions.reduce((sum, item) => sum + item.timingCredit, 0),
    matchedCount: metrics.matchedCount,
    signalCount: metrics.signalCount,
    oracleCount: metrics.oracleCount,
    absoluteLags: result.candidateTransitions.flatMap((item) =>
      item.lagMs === null ? [] : [Math.abs(item.lagMs)]),
    signedLags: result.candidateTransitions.flatMap((item) =>
      item.lagMs === null ? [] : [item.lagMs]),
    stateCredit: metrics.exposureAgreement * result.candleCount,
    stateCount: result.candleCount,
    distillationWeightedCrossEntropy: distillation?.weightedCrossEntropy,
    distillationWeightedOracleEntropy: distillation?.weightedOracleEntropy,
    distillationWeight: distillation?.weightSum,
    distillationOpportunity: distillation?.opportunitySum,
    distillationSamples: distillation?.sampleCount,
    valueHorizonMs: testCase.valueHorizonMs,
    strategyFinalEquity: distillation?.returns.strategy.equity,
    oracleFinalEquity: distillation?.returns.oracle.equity,
    strategyMaxDrawdown: distillation?.returns.strategy.maxDrawdown,
    oracleMaxDrawdown: distillation?.returns.oracle.maxDrawdown,
    strategyTurnover: distillation?.returns.strategy.turnover,
    oracleTurnover: distillation?.returns.oracle.turnover,
    days: testCase.days,
  };
}

function scoreCase(parts: CaseStats[], config: Args): CaseScore {
  const first = parts[0]!;
  const signalCount = sum(parts.map((part) => part.signalCount));
  const oracleCount = sum(parts.map((part) => part.oracleCount));
  const timingCredit = sum(parts.map((part) => part.timingCredit));
  const matchedCount = sum(parts.map((part) => part.matchedCount));
  const precision = eventRatio(timingCredit, signalCount, oracleCount);
  const recall = eventRatio(timingCredit, oracleCount, signalCount);
  const f1 = harmonic(precision, recall);
  const stateCount = sum(parts.map((part) => part.stateCount));
  const exposureAgreement = ratio(sum(parts.map((part) => part.stateCredit)), stateCount);
  const extraSignalCount = signalCount - matchedCount;
  const signalCleanliness = signalCount > 0 ? matchedCount / signalCount : 1;
  const days = sum(parts.map((part) => part.days));
  const absoluteLags = parts.flatMap((part) => part.absoluteLags);
  const signedLags = parts.flatMap((part) => part.signedLags);
  const lagP50Ms = absoluteLags.length > 0
    ? nullablePercentile(absoluteLags, 0.5)
    : nullableMedian(parts.map((part) => part.lagP50Ms ?? null));
  const lagP90Ms = absoluteLags.length > 0
    ? nullablePercentile(absoluteLags, 0.9)
    : nullableMedian(parts.map((part) => part.lagP90Ms ?? null));
  const lagP95Ms = absoluteLags.length > 0
    ? nullablePercentile(absoluteLags, 0.95)
    : nullableMedian(parts.map((part) => part.lagP95Ms ?? null));
  const lagMedianSignedMs = signedLags.length > 0
    ? nullablePercentile(signedLags, 0.5)
    : nullableMedian(parts.map((part) => part.lagMedianSignedMs ?? null));
  const signalScore = searchScore(f1, exposureAgreement, signalCleanliness, config.scoreVersion);
  const distillationWeight = sum(parts.map((part) => part.distillationWeight ?? 0));
  const valueDistillationLoss = distillationWeight > 0
    ? sum(parts.map((part) => part.distillationWeightedCrossEntropy ?? 0)) / distillationWeight
    : null;
  const oracleEntropy = distillationWeight > 0
    ? sum(parts.map((part) => part.distillationWeightedOracleEntropy ?? 0)) / distillationWeight
    : null;
  const valueDistillationKl = valueDistillationLoss !== null && oracleEntropy !== null
    ? Math.max(0, valueDistillationLoss - oracleEntropy)
    : null;
  const valueDistillationScore = valueDistillationKl === null
    ? null
    : Math.exp(-valueDistillationKl);
  const distillationSamples = sum(parts.map((part) => part.distillationSamples ?? 0));
  const meanOpportunity = distillationSamples > 0
    ? sum(parts.map((part) => part.distillationOpportunity ?? 0)) / distillationSamples
    : null;
  const horizonSamples = sum(parts.map((part) => part.valueHorizonMs === undefined
    ? 0
    : part.distillationSamples ?? 0));
  const valueHorizonMs = horizonSamples > 0
    ? sum(parts.map((part) => (part.valueHorizonMs ?? 0) * (part.distillationSamples ?? 0)))
      / horizonSamples
    : null;
  const strategyEquities = parts.flatMap((part) => part.strategyFinalEquity === undefined
    ? [] : [part.strategyFinalEquity]);
  const oracleEquities = parts.flatMap((part) => part.oracleFinalEquity === undefined
    ? [] : [part.oracleFinalEquity]);
  const strategyReturn = strategyEquities.length > 0 ? compoundEquity(strategyEquities) - 1 : null;
  const oracleReturn = oracleEquities.length > 0 ? compoundEquity(oracleEquities) - 1 : null;
  const strategyDrawdowns = parts.flatMap((part) => part.strategyMaxDrawdown === undefined
    ? [] : [part.strategyMaxDrawdown]);
  const oracleDrawdowns = parts.flatMap((part) => part.oracleMaxDrawdown === undefined
    ? [] : [part.oracleMaxDrawdown]);
  return {
    caseId: first.caseId,
    scale: first.scale,
    window: first.window,
    score: config.objective === "value-distillation"
      ? valueDistillationLoss === null ? Number.NEGATIVE_INFINITY : -valueDistillationLoss
      : signalScore,
    signalScore,
    valueDistillationScore,
    valueDistillationLoss,
    valueDistillationKl,
    oracleEntropy,
    meanOpportunity,
    valueHorizonMs,
    strategyReturn,
    oracleReturn,
    strategyMaxDrawdown: strategyDrawdowns.length > 0 ? Math.max(...strategyDrawdowns) : null,
    oracleMaxDrawdown: oracleDrawdowns.length > 0 ? Math.max(...oracleDrawdowns) : null,
    strategyTurnover: nullableSum(parts.map((part) => part.strategyTurnover ?? null)),
    oracleTurnover: nullableSum(parts.map((part) => part.oracleTurnover ?? null)),
    precision,
    recall,
    f1,
    rawPrecision: eventRatio(matchedCount, signalCount, oracleCount),
    rawRecall: eventRatio(matchedCount, oracleCount, signalCount),
    exposureAgreement,
    noiseSignalRatio: matchedCount > 0 ? extraSignalCount / matchedCount : extraSignalCount > 0 ? null : 0,
    signalCleanliness,
    signalsPerDay: ratio(signalCount, days),
    lagP50Ms,
    lagP90Ms,
    lagP95Ms,
    lagMedianSignedMs,
  };
}

function searchScore(
  f1: number,
  exposureAgreement: number,
  signalCleanliness: number,
  _scoreVersion: ScoreVersion,
): number {
  return vwKamaScore(f1, exposureAgreement, signalCleanliness);
}

function scoreWeightsDescription(config: Args): string {
  return config.objective === "value-distillation"
    ? "negative weighted cross-entropy -CE(p_oracle, s_candidate); p is derived from friction-aware future value and weighted by max(Q)-min(Q)"
    : "timing-credited transition F1 20%, sizing/confidence agreement 60%, and signal cleanliness 20%";
}

function aggregate(scores: CaseScore[]): AggregateScore {
  const values = scores.map((score) => score.score);
  const medianScore = percentile(values, 0.5);
  const p10 = percentile(values, 0.1);
  return {
    objective: (medianScore + p10) / 2,
    median: medianScore,
    p10,
    signalScore: median(scores.map((score) => score.signalScore)),
    valueDistillationScore: nullableMedian(scores.map((score) => score.valueDistillationScore)),
    valueDistillationLoss: nullableMedian(scores.map((score) => score.valueDistillationLoss)),
    valueDistillationKl: nullableMedian(scores.map((score) => score.valueDistillationKl)),
    meanOpportunity: nullableMedian(scores.map((score) => score.meanOpportunity)),
    strategyReturn: nullableMedian(scores.map((score) => score.strategyReturn)),
    oracleReturn: nullableMedian(scores.map((score) => score.oracleReturn)),
    strategyMaxDrawdown: nullableMedian(scores.map((score) => score.strategyMaxDrawdown)),
    oracleMaxDrawdown: nullableMedian(scores.map((score) => score.oracleMaxDrawdown)),
    precision: median(scores.map((score) => score.precision)),
    recall: median(scores.map((score) => score.recall)),
    f1: median(scores.map((score) => score.f1)),
    exposureAgreement: median(scores.map((score) => score.exposureAgreement)),
    noiseSignalRatio: nullableMedian(scores.map((score) => score.noiseSignalRatio)),
    signalCleanliness: median(scores.map((score) => score.signalCleanliness)),
    signalsPerDay: median(scores.map((score) => score.signalsPerDay)),
    lagP50Ms: nullableMedian(scores.map((score) => score.lagP50Ms)),
    lagP90Ms: nullableMedian(scores.map((score) => score.lagP90Ms)),
  };
}

function generateCandidates(config: Args): Candidate[] {
  const random = mulberry32(config.seed);
  return candidatesFromGenomes(Array.from({ length: config.trials }, () => randomGenome(config, random)));
}

function globalCandidates(agreementModes: AgreementMode[], oracleFriction: number): Candidate[] {
  const productionSignal = vwKamaParametersFromPeakValleySignal(
    createPeakValleyStrategyConfig({ kamaSignalFriction: oracleFriction }),
    oracleFriction,
  );
  const confirmation = {
    confirmationMix: 0,
    confirmationMinQuality: 0,
    confirmationAccelerationLookbackMs: HOUR,
    confirmationDistanceLookbackMs: HOUR,
    confirmationAccelerationWeight: 1,
    confirmationDistanceWeight: 1,
    confirmationBias: 0,
    hysteresisReleaseRatio: 0.25,
    confirmationEmaMs: HOUR,
    confirmationEmaThresholdBpsHour: 0,
    confirmationEmaWeight: 0,
    confirmationEmaGateStrength: 0,
    confirmationRsiMs: 14 * 60_000,
    confirmationRsiThreshold: 0,
    confirmationRsiWeight: 0,
    confirmationDmiMs: 14 * 60_000,
    confirmationDmiWeight: 0,
    confirmationAdxThreshold: 20,
    meanReversionSuppressionThreshold: 1,
    meanReversionEfficiencyMs: HOUR,
    meanReversionFastMs: 15 * 60_000,
    meanReversionSlowMs: HOUR,
    meanReversionVolatilityMs: HOUR,
    meanReversionReversalThreshold: 0,
  };
  const fullSizing = {
    efficiencyVolumeEmaMs: HOUR,
    efficiencyVolumePower: 0,
    buyMaxFraction: 1,
    sellMaxFraction: 1,
    buySizingSigmaBpsHour: 1e12,
    sellSizingSigmaBpsHour: 1e12,
    agreementMode: "sizing" as const,
    ...confirmation,
  };
  const threshold = {
    thresholdLookbackMs: HOUR,
    thresholdNoiseMultiplier: 0,
  };
  const refined = {
    efficiencyMs: 1_856_036,
    fastMs: 1_873_481,
    slowMs: 3_701_806,
    power: 0.61357,
    volumeMs: 440_028,
    volumeCap: 1.97271,
    deadbandBpsHour: 0.44351,
    deadbandMode: "hold" as const,
    ...fullSizing,
    ...threshold,
  };
  const production: Candidate = {
    ...confirmation,
    id: "production-current",
    family: "canonical",
    efficiencyMs: productionSignal.efficiencyMs,
    efficiencyVolumeEmaMs: productionSignal.efficiencyVolumeEmaMs!,
    efficiencyVolumePower: productionSignal.efficiencyVolumePower ?? 0,
    fastMs: productionSignal.fastMs,
    slowMs: productionSignal.slowMs,
    power: productionSignal.power,
    volumeMs: productionSignal.volumeMs,
    volumeCap: productionSignal.volumeCap,
    volumePower: productionSignal.volumePower,
    rateMode: productionSignal.rateMode,
    rateEmaMs: productionSignal.rateEmaMs,
    deadbandBpsHour: productionSignal.deadbandBpsHour,
    deadbandMode: productionSignal.deadbandMode,
    hysteresisReleaseRatio: productionSignal.hysteresisReleaseRatio ?? 0,
    thresholdLookbackMs: productionSignal.thresholdLookbackMs ?? HOUR,
    thresholdNoiseResponse: productionSignal.thresholdNoiseResponse,
    thresholdNoiseMultiplier: productionSignal.thresholdNoiseMultiplier ?? 0,
    thresholdInverseMaxBpsHour: productionSignal.thresholdInverseMaxBpsHour,
    thresholdInverseNoiseScaleBpsHour: productionSignal.thresholdInverseNoiseScaleBpsHour,
    signalFrictionFraction: productionSignal.signalFrictionFraction,
    buyMaxFraction: productionSignal.buyMaxFraction ?? 1,
    sellMaxFraction: productionSignal.sellMaxFraction ?? 1,
    buySizingSigmaBpsHour: 1e12,
    sellSizingSigmaBpsHour: 1e12,
    agreementMode: "sizing",
  };
  const candidates: Candidate[] = [
    {
      id: "global-clean-v0182",
      family: "volume",
      efficiencyMs: 265_846,
      fastMs: 1_113_418,
      slowMs: 64_800_000,
      power: 0.62547,
      volumeMs: 283_525,
      volumeCap: 5.08117,
      volumePower: 0.48543,
      deadbandBpsHour: 30,
      deadbandMode: "hold",
      thresholdLookbackMs: 2_457_073,
      thresholdNoiseMultiplier: 0,
      ...fullSizing,
    },
    {
      id: "global-clean-k0050",
      family: "canonical",
      efficiencyMs: 1_420_250,
      fastMs: 976_600,
      slowMs: 976_600,
      power: 0.79612,
      volumeMs: 82_113,
      volumeCap: 4.27354,
      volumePower: 0,
      deadbandBpsHour: 21.21468,
      deadbandMode: "hold",
      thresholdLookbackMs: 900_000,
      thresholdNoiseMultiplier: 0,
      ...fullSizing,
    },
    { id: "global-refined-v0016", family: "volume", volumePower: 1.72738, ...refined },
    { id: "global-refined-k0016", family: "canonical", volumePower: 0, ...refined },
  ];
  return [
    production,
    ...agreementModes.flatMap((agreementMode) => candidates.map((candidate) => ({
      ...candidate,
      id: agreementMode === "sizing" ? candidate.id : `${candidate.id}-confidence`,
      agreementMode,
    }))),
  ];
}

function randomGenome(config: Args, random: () => number): Genome {
  const fastMs = logRandom(random, config.fast);
  const meanReversionFastMs = logRandom(random, config.meanReversionFast);
  const meanReversionSuppressionThreshold = sparseRandom(
    random,
    config.meanReversionSuppressionThreshold,
  );
  const meanReversionReversalThreshold = meanReversionSuppressionThreshold > 0
    ? Math.max(
      meanReversionSuppressionThreshold,
      sparseRandom(random, config.meanReversionReversalThreshold),
    )
    : 0;
  return {
    efficiencyMs: logRandom(random, config.efficiency),
    efficiencyVolumeEmaMs: logRandom(random, config.efficiencyVolumeEma),
    efficiencyVolumePower: sparseRandom(random, config.efficiencyVolumePower),
    fastMs,
    slowMs: Math.max(fastMs, logRandom(random, config.slow)),
    power: linearRandom(random, config.power),
    volumeMs: logRandom(random, config.volume),
    volumeCap: linearRandom(random, config.volumeCap),
    volumePower: sparseRandom(random, config.volumePower),
    deadbandBpsHour: logRandom(random, config.deadbandBpsHour),
    deadbandMode: config.deadbandModes[Math.floor(random() * config.deadbandModes.length)]!,
    hysteresisReleaseRatio: linearRandom(random, config.hysteresisReleaseRatio),
    thresholdLookbackMs: logRandom(random, config.thresholdLookback),
    thresholdNoiseMultiplier: linearRandom(random, config.thresholdNoiseMultiplier),
    buyMaxFraction: linearRandom(random, config.buyMaxFraction),
    sellMaxFraction: linearRandom(random, config.sellMaxFraction),
    buySizingSigmaBpsHour: logRandom(random, config.buySizingSigma),
    sellSizingSigmaBpsHour: logRandom(random, config.sellSizingSigma),
    agreementMode: config.agreementModes[Math.floor(random() * config.agreementModes.length)]!,
    confirmationMix: sparseRandom(random, config.confirmationMix, 0.1),
    confirmationMinQuality: sparseRandom(random, config.confirmationMinQuality, 0.2),
    confirmationAccelerationLookbackMs: logRandom(random, config.confirmationAccelerationLookback),
    confirmationDistanceLookbackMs: logRandom(random, config.confirmationDistanceLookback),
    confirmationAccelerationWeight: sparseRandom(random, config.confirmationAccelerationWeight),
    confirmationDistanceWeight: sparseRandom(random, config.confirmationDistanceWeight),
    confirmationBias: linearRandom(random, config.confirmationBias),
    confirmationEmaMs: logRandom(random, config.confirmationEma),
    confirmationEmaThresholdBpsHour: logRandom(random, config.confirmationEmaThreshold),
    confirmationEmaWeight: sparseRandom(random, config.confirmationEmaWeight),
    confirmationEmaGateStrength: edgeRandom(random, config.confirmationEmaGateStrength),
    confirmationRsiMs: logRandom(random, config.confirmationRsi),
    confirmationRsiThreshold: linearRandom(random, config.confirmationRsiThreshold),
    confirmationRsiWeight: sparseRandom(random, config.confirmationRsiWeight),
    confirmationDmiMs: logRandom(random, config.confirmationDmi),
    confirmationDmiWeight: sparseRandom(random, config.confirmationDmiWeight),
    confirmationAdxThreshold: linearRandom(random, config.confirmationAdxThreshold),
    meanReversionSuppressionThreshold,
    meanReversionEfficiencyMs: logRandom(random, config.meanReversionEfficiency),
    meanReversionFastMs,
    meanReversionSlowMs: Math.max(
      meanReversionFastMs,
      logRandom(random, config.meanReversionSlow),
    ),
    meanReversionVolatilityMs: logRandom(random, config.meanReversionVolatility),
    meanReversionReversalThreshold,
  };
}

function candidatesFromGenomes(genomes: Genome[]): Candidate[] {
  return genomes.flatMap((genome, index) => {
    const suffix = String(index + 1).padStart(4, "0");
    return [
      { ...genome, id: `v${suffix}`, family: "volume" as const },
      {
        ...genome,
        id: `k${suffix}`,
        family: "canonical" as const,
        volumePower: 0,
        efficiencyVolumePower: 0,
      },
    ];
  });
}

function candidateParameters(candidate: Candidate) {
  return {
    efficiencyMs: candidate.efficiencyMs,
    efficiencyVolumeEmaMs: candidate.efficiencyVolumeEmaMs,
    efficiencyVolumePower: candidate.efficiencyVolumePower,
    fastMs: candidate.fastMs,
    slowMs: candidate.slowMs,
    power: candidate.power,
    volumeMs: candidate.volumeMs,
    volumeCap: candidate.volumeCap,
    volumePower: candidate.volumePower,
    rateMode: candidate.rateMode,
    rateEmaMs: candidate.rateEmaMs,
    deadbandBpsHour: candidate.deadbandBpsHour,
    deadbandMode: candidate.deadbandMode,
    hysteresisReleaseRatio: candidate.hysteresisReleaseRatio,
    thresholdLookbackMs: candidate.thresholdLookbackMs,
    thresholdNoiseResponse: candidate.thresholdNoiseResponse,
    thresholdNoiseMultiplier: candidate.thresholdNoiseMultiplier,
    thresholdInverseMaxBpsHour: candidate.thresholdInverseMaxBpsHour,
    thresholdInverseNoiseScaleBpsHour: candidate.thresholdInverseNoiseScaleBpsHour,
    signalFrictionFraction: candidate.signalFrictionFraction,
    buyMaxFraction: candidate.buyMaxFraction,
    sellMaxFraction: candidate.sellMaxFraction,
    buySizingSigmaBpsHour: candidate.buySizingSigmaBpsHour,
    sellSizingSigmaBpsHour: candidate.sellSizingSigmaBpsHour,
    agreementMode: candidate.agreementMode,
    confirmationMix: candidate.confirmationMix,
    confirmationMinQuality: candidate.confirmationMinQuality,
    confirmationAccelerationLookbackMs: candidate.confirmationAccelerationLookbackMs,
    confirmationDistanceLookbackMs: candidate.confirmationDistanceLookbackMs,
    confirmationAccelerationWeight: candidate.confirmationAccelerationWeight,
    confirmationDistanceWeight: candidate.confirmationDistanceWeight,
    confirmationBias: candidate.confirmationBias,
    confirmationEmaMs: candidate.confirmationEmaMs,
    confirmationEmaThresholdBpsHour: candidate.confirmationEmaThresholdBpsHour,
    confirmationEmaWeight: candidate.confirmationEmaWeight,
    confirmationEmaGateStrength: candidate.confirmationEmaGateStrength,
    confirmationRsiMs: candidate.confirmationRsiMs,
    confirmationRsiThreshold: candidate.confirmationRsiThreshold,
    confirmationRsiWeight: candidate.confirmationRsiWeight,
    confirmationDmiMs: candidate.confirmationDmiMs,
    confirmationDmiWeight: candidate.confirmationDmiWeight,
    confirmationAdxThreshold: candidate.confirmationAdxThreshold,
    meanReversionSuppressionThreshold: candidate.meanReversionSuppressionThreshold,
    meanReversionEfficiencyMs: candidate.meanReversionEfficiencyMs,
    meanReversionFastMs: candidate.meanReversionFastMs,
    meanReversionSlowMs: candidate.meanReversionSlowMs,
    meanReversionVolatilityMs: candidate.meanReversionVolatilityMs,
    meanReversionReversalThreshold: candidate.meanReversionReversalThreshold,
  };
}

function candidateGenome(candidate: Candidate): Genome {
  const { id: _id, family: _family, ...genome } = candidate;
  return genome;
}

function presetCandidate(preset: VwKamaPreset, index = 0): Candidate {
  return {
    id: `seed-${preset.intervalMs ?? index}-${index}`,
    family: (preset.parameters.volumePower > 0 || (preset.parameters.efficiencyVolumePower ?? 0) > 0)
      ? "volume"
      : "canonical",
    ...preset.parameters,
    efficiencyVolumeEmaMs: preset.parameters.efficiencyVolumeEmaMs ?? preset.parameters.volumeMs,
    efficiencyVolumePower: preset.parameters.efficiencyVolumePower ?? 0,
    thresholdLookbackMs: preset.parameters.thresholdLookbackMs ?? HOUR,
    thresholdNoiseMultiplier: preset.parameters.thresholdNoiseMultiplier ?? 0,
    hysteresisReleaseRatio: preset.parameters.hysteresisReleaseRatio ?? 0.25,
    buyMaxFraction: preset.parameters.buyMaxFraction ?? 1,
    sellMaxFraction: preset.parameters.sellMaxFraction ?? 1,
    buySizingSigmaBpsHour: preset.parameters.buySizingSigmaBpsHour ?? 1e12,
    sellSizingSigmaBpsHour: preset.parameters.sellSizingSigmaBpsHour ?? 1e12,
    agreementMode: preset.parameters.agreementMode ?? "sizing",
    confirmationMix: preset.parameters.confirmationMix ?? 0,
    confirmationMinQuality: preset.parameters.confirmationMinQuality ?? 0,
    confirmationAccelerationLookbackMs: preset.parameters.confirmationAccelerationLookbackMs ?? HOUR,
    confirmationDistanceLookbackMs: preset.parameters.confirmationDistanceLookbackMs ?? HOUR,
    confirmationAccelerationWeight: preset.parameters.confirmationAccelerationWeight ?? 1,
    confirmationDistanceWeight: preset.parameters.confirmationDistanceWeight ?? 1,
    confirmationBias: preset.parameters.confirmationBias ?? 0,
    confirmationEmaMs: preset.parameters.confirmationEmaMs ?? HOUR,
    confirmationEmaThresholdBpsHour: preset.parameters.confirmationEmaThresholdBpsHour ?? 0,
    confirmationEmaWeight: preset.parameters.confirmationEmaWeight ?? 0,
    confirmationEmaGateStrength: preset.parameters.confirmationEmaGateStrength ?? 0,
    confirmationRsiMs: preset.parameters.confirmationRsiMs ?? 14 * 60_000,
    confirmationRsiThreshold: preset.parameters.confirmationRsiThreshold ?? 0,
    confirmationRsiWeight: preset.parameters.confirmationRsiWeight ?? 0,
    confirmationDmiMs: preset.parameters.confirmationDmiMs ?? 14 * 60_000,
    confirmationDmiWeight: preset.parameters.confirmationDmiWeight ?? 0,
    confirmationAdxThreshold: preset.parameters.confirmationAdxThreshold ?? 20,
    meanReversionSuppressionThreshold: preset.parameters.meanReversionSuppressionThreshold ?? 1,
    meanReversionEfficiencyMs: preset.parameters.meanReversionEfficiencyMs
      ?? preset.parameters.efficiencyMs,
    meanReversionFastMs: preset.parameters.meanReversionFastMs ?? preset.parameters.fastMs,
    meanReversionSlowMs: preset.parameters.meanReversionSlowMs ?? HOUR,
    meanReversionVolatilityMs: preset.parameters.meanReversionVolatilityMs ?? HOUR,
    meanReversionReversalThreshold: preset.parameters.meanReversionReversalThreshold ?? 0,
  };
}

function rank(results: CandidateResult[]): CandidateResult[] {
  return results.slice().sort((left, right) =>
    right.aggregate.objective - left.aggregate.objective
    || right.aggregate.p10 - left.aggregate.p10
    || left.candidate.id.localeCompare(right.candidate.id));
}

function aggregateCandles(candles: TradingCandle[], sourceMs: number, targetMs: number): TradingCandle[] {
  if (targetMs === sourceMs) return candles.slice();
  const result: TradingCandle[] = [];
  const expected = targetMs / sourceMs;
  let current: TradingCandle | undefined;
  let bucket = -1;
  let count = 0;
  let contiguous = true;
  let lastOpen = -1;
  const flush = () => {
    if (current && contiguous && count === expected) result.push(current);
  };
  for (const candle of candles) {
    const nextBucket = Math.floor(candle.openTime / targetMs) * targetMs;
    if (nextBucket !== bucket) {
      flush();
      bucket = nextBucket;
      count = 1;
      contiguous = candle.openTime === bucket;
      lastOpen = candle.openTime;
      current = { ...candle, openTime: bucket, closeTime: bucket + targetMs - 1 };
      continue;
    }
    if (!current) continue;
    contiguous &&= candle.openTime === lastOpen + sourceMs;
    lastOpen = candle.openTime;
    current.high = Math.max(current.high, candle.high);
    current.low = Math.min(current.low, candle.low);
    current.close = candle.close;
    current.volume += candle.volume;
    count += 1;
  }
  flush();
  return result;
}

function continuousSegments(candles: TradingCandle[], intervalMs: number): TradingCandle[][] {
  const segments: TradingCandle[][] = [];
  let current: TradingCandle[] = [];
  for (const candle of candles) {
    const previous = current.at(-1);
    if (previous && candle.openTime !== previous.openTime + intervalMs) {
      if (current.length > 0) segments.push(current);
      current = [];
    }
    current.push(candle);
  }
  if (current.length > 0) segments.push(current);
  return segments;
}

function* loadSourceSegments(
  sourceDir: string,
  range: { start: number; end: number },
  intervalMs: number,
): Generator<TradingCandle[]> {
  let candles: TradingCandle[] = [];
  let found = false;
  for (let time = utcDay(range.start); time < range.end; time += DAY) {
    const date = new Date(time).toISOString().slice(0, 10);
    const file = path.join(sourceDir, `${date}.jsonl`);
    if (!fs.existsSync(file)) continue;
    for (const line of fs.readFileSync(file, "utf8").split("\n")) {
      if (!line.trim()) continue;
      const parsed = JSON.parse(line) as TradingCandle & { interval?: string; closed?: boolean };
      if (!validCandle(parsed) || parsed.closed === false) continue;
      if (parsed.openTime < range.start || parsed.openTime >= range.end) continue;
      if (parsed.interval && parseDuration(parsed.interval) !== intervalMs) {
        throw new Error(`Candle interval ${parsed.interval} does not match --source-interval ${formatDuration(intervalMs)}.`);
      }
      const candle: TradingCandle = {
        openTime: parsed.openTime,
        closeTime: parsed.closeTime,
        open: parsed.open,
        high: parsed.high,
        low: parsed.low,
        close: parsed.close,
        volume: parsed.volume,
      };
      const previous = candles.at(-1);
      if (previous && candle.openTime < previous.openTime) {
        throw new Error(`Source candles are not ordered at ${new Date(candle.openTime).toISOString()}.`);
      }
      if (previous?.openTime === candle.openTime) {
        candles[candles.length - 1] = candle;
      } else {
        if (previous && candle.openTime !== previous.openTime + intervalMs) {
          yield candles;
          candles = [];
        }
        candles.push(candle);
        found = true;
      }
    }
  }
  if (candles.length > 0) yield candles;
  if (!found) throw new Error("No source candles found for the selected window.");
}

function sourceBounds(sourceDir: string): { start: number; end: number } {
  const dates = fs.readdirSync(sourceDir)
    .map((file) => /^(\d{4}-\d{2}-\d{2})\.jsonl$/.exec(file)?.[1])
    .filter((date): date is string => Boolean(date))
    .sort();
  if (dates.length === 0) throw new Error(`No daily JSONL shards in ${sourceDir}.`);
  const start = Date.parse(`${dates[0]}T00:00:00Z`);
  const shardEnd = Date.parse(`${dates.at(-1)}T00:00:00Z`) + DAY;
  const lastCompletedDay = utcDay(Date.now());
  return { start, end: Math.max(start + 1, Math.min(shardEnd, lastCompletedDay)) };
}

function defaultWindows(start: number, end: number, sourceMs: number): Record<Stage, Window[]> {
  const unit = Math.min(60 * DAY, 250_000 * sourceMs, Math.floor((end - start) / 8));
  if (unit < 1_000) throw new Error("Source history is too short for automatic chronological windows.");
  const base = end - unit * 8;
  const make = (label: string, index: number): Window => ({ label, start: base + index * unit, end: base + (index + 1) * unit });
  return {
    fit: [0, 1, 2, 3].map((index) => make(`fit-${index + 1}`, index)),
    validation: [4, 5].map((index) => make(`validation-${index - 3}`, index)),
    test: [6, 7].map((index) => make(`test-${index - 5}`, index)),
  };
}

function assertChronological(fit: Window[], validation: Window[], test: Window[]): void {
  for (const [label, windows] of [["fit", fit], ["validation", validation], ["test", test]] as const) {
    if (windows.length === 0) throw new Error(`${label} windows cannot be empty.`);
    if (windows.some((window) => window.end <= window.start)) throw new Error(`Invalid ${label} window.`);
  }
  if (Math.max(...fit.map((window) => window.end)) > Math.min(...validation.map((window) => window.start))
    || Math.max(...validation.map((window) => window.end)) > Math.min(...test.map((window) => window.start))) {
    throw new Error("Fit, validation, and test windows must be chronological and non-overlapping.");
  }
}

function parseArgs(argv: string[]): Args {
  const allowed = new Set([
    "source-dir", "source-interval", "scales", "fit-windows", "validation-windows",
    "test-windows", "trials", "top", "seed", "oracle-friction", "match-window",
    "timing-half-life", "warmup-multiple", "case-warmup", "efficiency-range", "fast-range",
    "efficiency-volume-ema-range", "efficiency-volume-power-range",
    "slow-range", "volume-range", "power-range", "volume-cap-range",
    "volume-power-range", "deadband-bps-hour-range", "threshold-lookback-range",
    "threshold-noise-multiplier-range", "algorithm", "mode", "generations", "restarts",
    "full-evaluation-interval", "refinement-rounds", "pbest-fraction", "confirmation-masks",
    "buy-max-fraction-range", "sell-max-fraction-range",
    "buy-sizing-sigma-range", "sell-sizing-sigma-range",
    "agreement-modes", "deadband-modes",
    "confirmation-mix-range", "confirmation-min-quality-range",
    "confirmation-acceleration-lookback-range", "confirmation-distance-lookback-range",
    "confirmation-acceleration-weight-range", "confirmation-distance-weight-range",
    "confirmation-bias-range",
    "hysteresis-release-ratio-range",
    "confirmation-ema-range", "confirmation-ema-threshold-range",
    "confirmation-ema-weight-range", "confirmation-ema-gate-strength-range",
    "confirmation-rsi-range", "confirmation-rsi-threshold-range", "confirmation-rsi-weight-range",
    "confirmation-dmi-range", "confirmation-dmi-weight-range", "confirmation-adx-threshold-range",
    "mean-reversion-suppression-threshold-range", "mean-reversion-efficiency-range",
    "mean-reversion-fast-range", "mean-reversion-slow-range",
    "mean-reversion-volatility-range", "mean-reversion-reversal-threshold-range",
    "differential-weight", "crossover-rate", "immigrant-rate", "screen-windows",
    "screen-scales", "workers", "accelerator", "objective", "score-version",
    "exposure-grid-size", "exposure-min", "exposure-max", "value-horizon-mode", "value-horizon", "oracle-temperature",
    "strategy-temperature", "strategy-volatility-scaling",
    "opportunity-epsilon", "quote-lend-rate", "quote-borrow-rate",
    "asset-borrow-rate", "seed-candidates", "preset-window-ids", "preset-output", "output", "report",
  ]);
  const values = new Map<string, string>();
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index]!;
    if (token === "--help") { help(); process.exit(0); }
    if (!token.startsWith("--")) throw new Error(`Unexpected argument: ${token}`);
    const [rawKey, inline] = token.slice(2).split("=", 2);
    const value = inline ?? argv[++index];
    if (!rawKey || !value || value.startsWith("--")) throw new Error(`Missing value for --${rawKey}.`);
    if (!allowed.has(rawKey)) throw new Error(`Unknown option: --${rawKey}.`);
    values.set(rawKey, value);
  }
  const get = (key: string, fallback: string) => values.get(key) ?? fallback;
  const sourceDir = path.resolve(repoRoot, get("source-dir", "data/historical/spot-btcusdt/btcusdt/1m"));
  const outputPath = path.resolve(repoRoot, get("output", `data/benchmarks/volume-weighted-kama-${stamp}.jsonl`));
  const reportPath = path.resolve(repoRoot, get("report", `docs/volume-weighted-kama-${stamp}.md`));
  const algorithm = get("algorithm", "de");
  const mode = get("mode", "standard");
  const accelerator = get("accelerator", "auto");
  const objective = get("objective", "signal");
  const valueHorizonMode = get("value-horizon-mode", "fixed");
  const scoreVersion = integer(get("score-version", String(VW_KAMA_SCORE_VERSION)), "score-version");
  if (algorithm !== "random" && algorithm !== "de") throw new Error("--algorithm must be random or de.");
  if (mode !== "standard" && mode !== "per-window") throw new Error("--mode must be standard or per-window.");
  if (accelerator !== "auto" && accelerator !== "cpu" && accelerator !== "cuda") {
    throw new Error("--accelerator must be auto, cpu, or cuda.");
  }
  if (objective !== "signal" && objective !== "value-distillation") {
    throw new Error("--objective must be signal or value-distillation.");
  }
  if (valueHorizonMode !== "fixed" && valueHorizonMode !== "oracle-average-trade") {
    throw new Error("--value-horizon-mode must be fixed or oracle-average-trade.");
  }
  const exposureMinimum = bounded(get("exposure-min", "-1"), "exposure-min", -100, -Number.EPSILON);
  const exposureMaximum = bounded(get("exposure-max", "1"), "exposure-max", Number.EPSILON, 100);
  if (scoreVersion !== VW_KAMA_SCORE_VERSION) {
    throw new Error(`--score-version must be ${VW_KAMA_SCORE_VERSION}.`);
  }
  const agreementModes = unique(get("agreement-modes", "sizing,confidence").split(","));
  if (agreementModes.length === 0 || agreementModes.some((value) => value !== "sizing" && value !== "confidence")) {
    throw new Error("--agreement-modes must contain sizing and/or confidence.");
  }
  const deadbandModes = unique(get("deadband-modes", "flat,hysteresis,hold").split(","));
  if (deadbandModes.length === 0
    || deadbandModes.some((value) => !["flat", "hold", "hysteresis"].includes(value))) {
    throw new Error("--deadband-modes must contain flat, hold, and/or hysteresis.");
  }
  return {
    sourceDir,
    sourceIntervalMs: parseDuration(get("source-interval", "1m")),
    scales: unique(get("scales", "1m,5m,15m,1h").split(",").map(parseDuration)).sort((a, b) => a - b),
    fitWindows: values.has("fit-windows") ? parseWindows(values.get("fit-windows")!, "fit") : undefined,
    validationWindows: values.has("validation-windows") ? parseWindows(values.get("validation-windows")!, "validation") : undefined,
    testWindows: values.has("test-windows") ? parseWindows(values.get("test-windows")!, "test") : undefined,
    trials: integer(get("trials", "384"), "trials"),
    algorithm,
    mode,
    generations: integer(get("generations", "12"), "generations", 0),
    restarts: integer(get("restarts", "2"), "restarts"),
    fullEvaluationInterval: integer(get("full-evaluation-interval", "4"), "full-evaluation-interval"),
    refinementRounds: integer(get("refinement-rounds", "3"), "refinement-rounds", 0),
    pbestFraction: bounded(get("pbest-fraction", "0.15"), "pbest-fraction", 0.02, 1),
    confirmationMasks: parseConfirmationMasks(get("confirmation-masks", CONFIRMATION_MASKS.join(","))),
    differentialWeight: bounded(get("differential-weight", "0.75"), "differential-weight", 0, 2),
    crossoverRate: bounded(get("crossover-rate", "0.8"), "crossover-rate", 0, 1),
    immigrantRate: bounded(get("immigrant-rate", "0.08"), "immigrant-rate", 0, 1),
    screenWindows: integer(get("screen-windows", "1"), "screen-windows"),
    screenScales: unique(get("screen-scales", "1m,15m").split(",").map(parseDuration)).sort((a, b) => a - b),
    workers: integer(get("workers", String(Math.max(
      1,
      Math.min(12, (typeof availableParallelism === "function" ? availableParallelism() : cpus().length) - 4),
    ))), "workers"),
    accelerator,
    objective: objective as SearchObjective,
    scoreVersion,
    exposureGridSize: integer(get("exposure-grid-size", "21"), "exposure-grid-size", 3),
    exposureMinimum,
    exposureMaximum,
    valueHorizonMode,
    valueHorizonMs: parseDuration(get("value-horizon", "1h")),
    oracleTemperature: positive(get("oracle-temperature", "0.01"), "oracle-temperature"),
    strategyTemperature: positive(get("strategy-temperature", "0.001"), "strategy-temperature"),
    strategyVolatilityScaling: booleanValue(
      get("strategy-volatility-scaling", "false"),
      "strategy-volatility-scaling",
    ),
    opportunityEpsilon: nonNegative(get("opportunity-epsilon", "0.000001"), "opportunity-epsilon"),
    quoteLendRate: nonNegative(get("quote-lend-rate", "0"), "quote-lend-rate"),
    quoteBorrowRate: nonNegative(get("quote-borrow-rate", "0"), "quote-borrow-rate"),
    assetBorrowRate: nonNegative(get("asset-borrow-rate", "0"), "asset-borrow-rate"),
    seedCandidatePaths: values.has("seed-candidates")
      ? values.get("seed-candidates")!.split(",").filter(Boolean).map((file) => path.resolve(repoRoot, file))
      : [],
    presetWindowIds: values.has("preset-window-ids")
      ? values.get("preset-window-ids")!.split(",").filter(Boolean)
      : undefined,
    presetOutputPath: path.resolve(repoRoot, get(
      "preset-output",
      mode === "per-window"
        ? "data/benchmarks/vw-kama-window-presets.json"
        : "data/benchmarks/vw-kama-global-presets.json",
    )),
    top: integer(get("top", "6"), "top"),
    seed: integer(get("seed", "17"), "seed", 0),
    oracleFriction: nonNegative(get("oracle-friction", "0.00175"), "oracle-friction"),
    matchWindowMs: parseDuration(get("match-window", "2h")),
    timingHalfLifeMs: parseDuration(get("timing-half-life", "10m")),
    warmupMultiple: positive(get("warmup-multiple", "3"), "warmup-multiple"),
    caseWarmupMs: values.has("case-warmup") ? parseDuration(values.get("case-warmup")!) : 0,
    efficiency: durationRange(get("efficiency-range", "1m..2h"), "efficiency-range"),
    efficiencyVolumeEma: durationRange(
      get("efficiency-volume-ema-range", "1m..12h"),
      "efficiency-volume-ema-range",
    ),
    efficiencyVolumePower: numberRange(
      get("efficiency-volume-power-range", "0..4"),
      "efficiency-volume-power-range",
      0,
    ),
    fast: durationRange(get("fast-range", "1s..30m"), "fast-range"),
    slow: durationRange(get("slow-range", "5m..24h"), "slow-range"),
    volume: durationRange(get("volume-range", "1m..12h"), "volume-range"),
    power: numberRange(get("power-range", "0.3..5"), "power-range", 0.1),
    volumeCap: numberRange(get("volume-cap-range", "1..10"), "volume-cap-range", 1),
    volumePower: numberRange(get("volume-power-range", "0..3"), "volume-power-range", 0),
    deadbandBpsHour: numberRange(get("deadband-bps-hour-range", "0.05..2000"), "deadband-bps-hour-range", 0),
    hysteresisReleaseRatio: fractionRange(get("hysteresis-release-ratio-range", "0..1"), "hysteresis-release-ratio-range"),
    thresholdLookback: durationRange(get("threshold-lookback-range", "5m..24h"), "threshold-lookback-range"),
    thresholdNoiseMultiplier: numberRange(
      get("threshold-noise-multiplier-range", "0..8"),
      "threshold-noise-multiplier-range",
      0,
    ),
    buyMaxFraction: fractionRange(get("buy-max-fraction-range", "0.05..1"), "buy-max-fraction-range"),
    sellMaxFraction: fractionRange(get("sell-max-fraction-range", "0.05..1"), "sell-max-fraction-range"),
    buySizingSigma: numberRange(get("buy-sizing-sigma-range", "0.01..300"), "buy-sizing-sigma-range", 0.000001),
    sellSizingSigma: numberRange(get("sell-sizing-sigma-range", "0.01..300"), "sell-sizing-sigma-range", 0.000001),
    agreementModes: agreementModes as AgreementMode[],
    deadbandModes: deadbandModes as DeadbandMode[],
    confirmationMix: fractionRange(get("confirmation-mix-range", "0..1"), "confirmation-mix-range"),
    confirmationMinQuality: fractionRange(get("confirmation-min-quality-range", "0..0.95"), "confirmation-min-quality-range"),
    confirmationAccelerationLookback: durationRange(get("confirmation-acceleration-lookback-range", "1m..6h"), "confirmation-acceleration-lookback-range"),
    confirmationDistanceLookback: durationRange(get("confirmation-distance-lookback-range", "1m..6h"), "confirmation-distance-lookback-range"),
    confirmationAccelerationWeight: numberRange(get("confirmation-acceleration-weight-range", "0..5"), "confirmation-acceleration-weight-range", 0),
    confirmationDistanceWeight: numberRange(get("confirmation-distance-weight-range", "0..5"), "confirmation-distance-weight-range", 0),
    confirmationBias: numberRange(get("confirmation-bias-range", "-5..5"), "confirmation-bias-range", -50),
    confirmationEma: durationRange(get("confirmation-ema-range", "5m..24h"), "confirmation-ema-range"),
    confirmationEmaThreshold: numberRange(get("confirmation-ema-threshold-range", "0.1..300"), "confirmation-ema-threshold-range", 0.000001),
    confirmationEmaWeight: numberRange(get("confirmation-ema-weight-range", "0..5"), "confirmation-ema-weight-range", 0),
    confirmationEmaGateStrength: fractionRange(get("confirmation-ema-gate-strength-range", "0..1"), "confirmation-ema-gate-strength-range"),
    confirmationRsi: durationRange(get("confirmation-rsi-range", "2m..6h"), "confirmation-rsi-range"),
    confirmationRsiThreshold: numberRange(get("confirmation-rsi-threshold-range", "0..20"), "confirmation-rsi-threshold-range", 0),
    confirmationRsiWeight: numberRange(get("confirmation-rsi-weight-range", "0..5"), "confirmation-rsi-weight-range", 0),
    confirmationDmi: durationRange(get("confirmation-dmi-range", "2m..6h"), "confirmation-dmi-range"),
    confirmationDmiWeight: numberRange(get("confirmation-dmi-weight-range", "0..5"), "confirmation-dmi-weight-range", 0),
    confirmationAdxThreshold: numberRange(get("confirmation-adx-threshold-range", "5..50"), "confirmation-adx-threshold-range", 0),
    meanReversionSuppressionThreshold: numberRange(
      get("mean-reversion-suppression-threshold-range", "0..3"),
      "mean-reversion-suppression-threshold-range",
      0,
    ),
    meanReversionEfficiency: durationRange(
      get("mean-reversion-efficiency-range", "1m..24h"),
      "mean-reversion-efficiency-range",
    ),
    meanReversionFast: durationRange(
      get("mean-reversion-fast-range", "1m..6h"),
      "mean-reversion-fast-range",
    ),
    meanReversionSlow: durationRange(
      get("mean-reversion-slow-range", "5m..24h"),
      "mean-reversion-slow-range",
    ),
    meanReversionVolatility: durationRange(
      get("mean-reversion-volatility-range", "1m..24h"),
      "mean-reversion-volatility-range",
    ),
    meanReversionReversalThreshold: numberRange(
      get("mean-reversion-reversal-threshold-range", "0..6"),
      "mean-reversion-reversal-threshold-range",
      0,
    ),
    outputPath,
    reportPath,
  };
}

function parseWindows(value: string, prefix: string): Window[] {
  return value.split(",").filter(Boolean).map((part, index) => {
    const pieces = part.split("..");
    if (pieces.length !== 2) throw new Error(`--${prefix}-windows entries must be start..end.`);
    const start = endpoint(pieces[0]!, false);
    const end = endpoint(pieces[1]!, true);
    return { label: `${prefix}-${index + 1}`, start, end };
  });
}

function parseConfirmationMasks(value: string): ConfirmationMask[] {
  const masks = unique(value.split(",").filter(Boolean));
  if (masks.length === 0 || masks.some((mask) => !CONFIRMATION_MASKS.includes(mask as ConfirmationMask))) {
    throw new Error(`--confirmation-masks must contain values from ${CONFIRMATION_MASKS.join(", ")}.`);
  }
  return masks as ConfirmationMask[];
}

function endpoint(value: string, end: boolean): number {
  const dateOnly = /^\d{4}-\d{2}-\d{2}$/.test(value);
  const parsed = Date.parse(dateOnly ? `${value}T00:00:00Z` : value);
  if (!Number.isFinite(parsed)) throw new Error(`Invalid date/time: ${value}`);
  return parsed + (end && dateOnly ? DAY : 0);
}

function durationRange(value: string, label: string): Range {
  const [min, max, ...extra] = value.split("..").map(parseDuration);
  if (extra.length || min === undefined || max === undefined || max < min) throw new Error(`Invalid --${label}.`);
  return { min, max };
}

function numberRange(value: string, label: string, minimum: number): Range {
  const [min, max, ...extra] = value.split("..").map(Number);
  if (extra.length || min === undefined || max === undefined || !Number.isFinite(min) || !Number.isFinite(max)
    || min < minimum || max < min) throw new Error(`Invalid --${label}.`);
  return { min, max };
}

function fractionRange(value: string, label: string): Range {
  const range = numberRange(value, label, 0);
  if (range.max > 1) throw new Error(`--${label} cannot exceed one.`);
  return range;
}

function parseDuration(value: string): number {
  const match = /^(\d+(?:\.\d+)?)(ms|s|m|h|d|w)?$/.exec(value.trim());
  if (!match) throw new Error(`Invalid duration: ${value}`);
  const factors: Record<string, number> = { ms: 1, s: 1_000, m: 60_000, h: HOUR, d: DAY, w: 7 * DAY };
  const result = Number(match[1]) * factors[match[2] ?? "ms"]!;
  if (!Number.isFinite(result) || result <= 0) throw new Error(`Duration must be positive: ${value}`);
  return Math.round(result);
}

function validateScales(scales: number[], sourceMs: number): void {
  for (const scale of scales) {
    if (scale < sourceMs) throw new Error(`Target ${formatDuration(scale)} is finer than source ${formatDuration(sourceMs)}.`);
    if (scale % sourceMs !== 0) throw new Error(`Target ${formatDuration(scale)} is not divisible by source ${formatDuration(sourceMs)}.`);
  }
}

function candidateWarmupSamples(candidate: Candidate, scaleMs: number, multiple: number): number {
  const kama = volumeWeightedKamaWarmupSamples({
    efficiencyPeriod: samples(candidate.efficiencyMs, scaleMs),
    efficiencyVolumePeriod: samples(candidate.efficiencyVolumeEmaMs, scaleMs),
    efficiencyVolumePower: candidate.efficiencyVolumePower,
    slowPeriod: samples(candidate.slowMs, scaleMs),
    volumePeriod: samples(candidate.volumeMs, scaleMs),
  }, multiple);
  const threshold = candidate.thresholdNoiseMultiplier > 0
    ? samples(candidate.thresholdLookbackMs, scaleMs) * multiple
    : 0;
  const confirmation = candidate.confirmationMix > 0
    ? Math.max(
      samples(candidate.confirmationAccelerationLookbackMs, scaleMs),
      samples(candidate.confirmationDistanceLookbackMs, scaleMs),
    ) * multiple
    : 0;
  const ema = (candidate.confirmationMix > 0 && candidate.confirmationEmaWeight > 0)
    || candidate.confirmationEmaGateStrength > 0
    ? samples(candidate.confirmationEmaMs, scaleMs) * multiple
    : 0;
  const rsi = candidate.confirmationMix > 0 && candidate.confirmationRsiWeight > 0
    ? samples(candidate.confirmationRsiMs, scaleMs) * multiple
    : 0;
  const dmi = candidate.confirmationMix > 0 && candidate.confirmationDmiWeight > 0
    ? samples(candidate.confirmationDmiMs, scaleMs) * multiple * 2
    : 0;
  const meanReversion = candidate.meanReversionReversalThreshold > 0
    ? Math.max(
      samples(candidate.meanReversionEfficiencyMs, scaleMs),
      samples(candidate.meanReversionFastMs, scaleMs),
      samples(candidate.meanReversionSlowMs, scaleMs),
      samples(candidate.meanReversionVolatilityMs, scaleMs),
    ) * multiple
    : 0;
  return Math.max(kama, threshold, confirmation, ema, rsi, dmi, meanReversion);
}

function maximumWarmupMs(config: Args): number {
  return Math.max(
    config.caseWarmupMs,
    config.efficiency.max * config.warmupMultiple,
    config.efficiencyVolumeEma.max * config.warmupMultiple,
    config.slow.max * config.warmupMultiple,
    config.volume.max * config.warmupMultiple,
    config.thresholdLookback.max * config.warmupMultiple,
    config.confirmationAccelerationLookback.max * config.warmupMultiple,
    config.confirmationDistanceLookback.max * config.warmupMultiple,
    config.confirmationEma.max * config.warmupMultiple,
    config.confirmationRsi.max * config.warmupMultiple,
    config.confirmationDmi.max * config.warmupMultiple * 2,
    config.meanReversionEfficiency.max * config.warmupMultiple,
    config.meanReversionFast.max * config.warmupMultiple,
    config.meanReversionSlow.max * config.warmupMultiple,
    config.meanReversionVolatility.max * config.warmupMultiple,
  );
}

function samples(durationMs: number, scaleMs: number): number {
  return Math.max(1, Math.round(durationMs / scaleMs));
}

function appendResult(file: string, result: CandidateResult, includeCases: boolean): void {
  appendJson(file, {
    type: "candidate",
    stage: result.stage,
    candidate: compactCandidate(result.candidate),
    aggregate: roundObject(result.aggregate),
    ...(includeCases ? { cases: result.cases.map(roundObject) } : {}),
  });
}

function report(
  config: Args,
  windows: Record<Stage, Window[]>,
  fit: CandidateResult[],
  validation: CandidateResult[],
  test: CandidateResult[],
  caseWarmupMs: number,
): string {
  const lines = [
    "# Volume-weighted KAMA oracle approximation search",
    "",
    `Generated: ${new Date().toISOString()}`,
    `Score version: ${config.scoreVersion}`,
    `Raw results: \`${path.relative(repoRoot, config.outputPath)}\``,
    "",
    "The fit ranking never reads validation or holdout scores. Holdout evaluates only the best validated volume-aware/canonical candidates and the fixed current-production signal baseline.",
    "",
    "## Data and objective",
    "",
    `- Source: \`${path.relative(repoRoot, config.sourceDir)}\` (${formatDuration(config.sourceIntervalMs)}); target scales: ${config.scales.map(formatDuration).join(", ")}.`,
    `- Windows: ${Object.entries(windows).map(([stage, list]) => `${stage} ${list.map(formatWindow).join(", ")}`).join("; ")}.`,
    `- Each continuous segment reserves ${formatDuration(caseWarmupMs)} before scoring.`,
    `- Candidate evaluation uses ${config.workers} persistent shared-memory worker thread${config.workers === 1 ? "" : "s"}, cross-generation score caching, and stage-wide prepared columnar candle/oracle caches.`,
    config.seedCandidatePaths.length > 0
      ? `- Generation zero evaluates all ${loadSeedCandidates(config.seedCandidatePaths).length} warm genomes plus ${config.trials} Latin-hypercube genomes, then selects ${config.trials} by 70% score / 30% parameter novelty. Warm sources: ${config.seedCandidatePaths.map((file) => `\`${path.relative(repoRoot, file)}\``).join(", ")}.`
      : `- Generation zero evaluates ${config.trials} deterministic Latin-hypercube genomes and selects by 70% score / 30% parameter novelty.`,
    `- Search uses ${config.restarts} independent restart(s), adaptive current-to-pbest differential evolution, family/agreement/confirmation-mask islands with cross-island migration, rotating fit folds with a full-fit pass every ${config.fullEvaluationInterval} generations, and ${config.refinementRounds} shrinking elite-refinement round(s).`,
    "- Signal: completed-candle volume-weighted KAMA derivative rate with flat, hold, or hysteresis state handling. The rate calculation, threshold clamp, adaptive-noise threshold, and consumed-edge friction memory are shared directly with the live PeakValleyStrategy. ER optionally weights every price move by `(volume / causal volume EMA)^ER volume power`; zero recovers standard ER. A second volume-aware KAMA supplies the optional mean-reversion baseline: it has independent ER/fast/slow periods and shares the strategy's ER-volume, KAMA-power, and post-ER volume behavior. Its causal volatility-normalized distance follows KAMA below the suppression threshold, goes flat between thresholds, and reverses KAMA at the reversal threshold. A causal logistic confirmation can combine KAMA acceleration, price overextension, independent slow-EMA trend, RSI, and ADX-strength-weighted DMI direction, then scale or filter the signal. Those experimental layers are disabled for `production-current`. Sizing mode price-marks the fraction, while confidence mode holds it as uncertainty until the next signal.",
    `- Signal memory: after the first signal, a candidate state change emits only when the current close is strictly more than ${round(config.oracleFriction * 10_000, 3)} bps from the last emitted signal price; rejected changes retain the prior state. This is the same friction used by the oracle.`,
    "- Matching is one chronological one-to-one alignment by resulting state. It maximizes total timing credit, so extra candidate transitions reduce precision and uncovered oracle transitions reduce recall.",
    `- Search objective: ${config.objective}; ${scoreWeightsDescription(config)}. Cleanliness is matched / (matched + extra); the displayed noise/signal ratio is extra / matched.`,
    config.objective === "value-distillation"
      ? `- Oracle exposures: ${config.exposureGridSize} points over [${config.exposureMinimum}, ${config.exposureMaximum}], ${config.valueHorizonMode === "fixed" ? `${formatDuration(config.valueHorizonMs)} fixed` : `average time between consecutive oracle trades (${formatDuration(config.valueHorizonMs)} fallback)`} forced-exposure horizon, value temperature ${config.oracleTemperature}; strategy truncated-exponential temperature ${config.strategyTemperature} scales with sqrt(H/dt)${config.strategyVolatilityScaling ? " and the trailing-H standard deviation of simple returns" : ""}; opportunity weight is max(Q)-min(Q)+${config.opportunityEpsilon}.`
      : "- The signal score remains available as a diagnostic beside the selected objective.",
    config.objective === "value-distillation"
      ? "- Candidate fitness is the negative of median/P90 cross-entropy, equally weighted; every scale/window case has equal weight."
      : "- Candidate objective equally weights the median and P10 case score; every scale/window case has equal weight.",
    "",
    "## Holdout finalists and production signal baseline",
    "",
    resultTable(test, config),
    "",
    "## Validation finalists",
    "",
    resultTable(rank(validation), config),
    "",
    "## Best fit candidates",
    "",
    resultTable(rank(fit).slice(0, 12), config),
    "",
    "## Finalist parameters",
    "",
    "| family | id | agreement | efficiency | ER volume EMA/power | fast | slow | power | volume | cap | volume power | base threshold | state mode | noise lookback | noise multiplier | buy max | sell max | buy sigma | sell sigma |",
    "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ...test.map(({ candidate }) => `| ${candidate.family} | ${candidate.id} | ${candidate.agreementMode} | ${formatDuration(candidate.efficiencyMs)} | ${formatDuration(candidate.efficiencyVolumeEmaMs)} / ${round(candidate.efficiencyVolumePower, 3)} | ${formatDuration(candidate.fastMs)} | ${formatDuration(candidate.slowMs)} | ${round(candidate.power, 3)} | ${formatDuration(candidate.volumeMs)} | ${round(candidate.volumeCap, 3)} | ${round(candidate.volumePower, 3)} | ${round(candidate.deadbandBpsHour, 3)} bps/hour | ${candidate.deadbandMode} | ${formatDuration(candidate.thresholdLookbackMs)} | ${round(candidate.thresholdNoiseMultiplier, 3)} | ${pct(candidate.buyMaxFraction)} | ${pct(candidate.sellMaxFraction)} | ${round(candidate.buySizingSigmaBpsHour, 3)} | ${round(candidate.sellSizingSigmaBpsHour, 3)} |`),
    "",
    "## Finalist confirmation parameters",
    "",
    "| candidate | mix | minimum quality | acceleration lookback | distance lookback | acceleration weight | overextension weight | bias |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
    ...test.map(({ candidate }) => `| ${candidate.id} | ${pct(candidate.confirmationMix)} | ${pct(candidate.confirmationMinQuality)} | ${formatDuration(candidate.confirmationAccelerationLookbackMs)} | ${formatDuration(candidate.confirmationDistanceLookbackMs)} | ${round(candidate.confirmationAccelerationWeight, 3)} | ${round(candidate.confirmationDistanceWeight, 3)} | ${round(candidate.confirmationBias, 3)} |`),
    "",
    "| candidate | hysteresis release | slow EMA | EMA tolerance | EMA weight/gate | RSI period | RSI tolerance/weight | DMI period | DMI weight | ADX threshold |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ...test.map(({ candidate }) => `| ${candidate.id} | ${pct(candidate.hysteresisReleaseRatio)} | ${formatDuration(candidate.confirmationEmaMs)} | ${round(candidate.confirmationEmaThresholdBpsHour, 3)} bps/h | ${round(candidate.confirmationEmaWeight, 3)} / ${pct(candidate.confirmationEmaGateStrength)} | ${formatDuration(candidate.confirmationRsiMs)} | ${round(candidate.confirmationRsiThreshold, 2)} / ${round(candidate.confirmationRsiWeight, 3)} | ${formatDuration(candidate.confirmationDmiMs)} | ${round(candidate.confirmationDmiWeight, 3)} | ${round(candidate.confirmationAdxThreshold, 2)} |`),
    "",
    "## Finalist mean-reversion parameters",
    "",
    "| candidate | ER | fast | slow | volatility | suppress at | reverse at |",
    "|---|---:|---:|---:|---:|---:|---:|",
    ...test.map(({ candidate }) => `| ${candidate.id} | ${formatDuration(candidate.meanReversionEfficiencyMs)} | ${formatDuration(candidate.meanReversionFastMs)} | ${formatDuration(candidate.meanReversionSlowMs)} | ${formatDuration(candidate.meanReversionVolatilityMs)} | ${round(candidate.meanReversionSuppressionThreshold, 3)}σ | ${round(candidate.meanReversionReversalThreshold, 3)}σ |`),
    "",
    "## Holdout cases",
    "",
    ...(config.objective === "value-distillation"
      ? [
        "| candidate | scale/window | H | cross-entropy | KL | exp(-KL) | strategy return | oracle return | strategy DD | oracle DD | opportunity | signal score | agreement | signals/day |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ...test.flatMap((result) => result.cases.map((item) => `| ${result.candidate.id} | ${item.caseId} | ${formatNullableDuration(item.valueHorizonMs)} | ${formatNullableNumber(item.valueDistillationLoss)} | ${formatNullableNumber(item.valueDistillationKl)} | ${formatNullablePercent(item.valueDistillationScore)} | ${formatNullablePercent(item.strategyReturn)} | ${formatNullablePercent(item.oracleReturn)} | ${formatNullablePercent(item.strategyMaxDrawdown)} | ${formatNullablePercent(item.oracleMaxDrawdown)} | ${formatNullableNumber(item.meanOpportunity, 6)} | ${pct(item.signalScore)} | ${pct(item.exposureAgreement)} | ${round(item.signalsPerDay, 2)} |`)),
      ]
      : [
        "| candidate | scale/window | score | precision | recall | F1 | agreement | cleanliness | noise/signal | signals/day | timing error P50 | timing error P90 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ...test.flatMap((result) => result.cases.map((item) => `| ${result.candidate.id} | ${item.caseId} | ${pct(item.score)} | ${pct(item.precision)} | ${pct(item.recall)} | ${pct(item.f1)} | ${pct(item.exposureAgreement)} | ${pct(item.signalCleanliness)} | ${formatNullableRatio(item.noiseSignalRatio)} | ${round(item.signalsPerDay, 2)} | ${formatNullableDuration(item.lagP50Ms)} | ${formatNullableDuration(item.lagP90Ms)} |`)),
      ]),
    "",
  ];
  return lines.join("\n");
}

function resultTable(results: CandidateResult[], config: Args): string {
  if (config.objective === "value-distillation") {
    return [
      "| family | id | strategy | robust CE | median CE | P90 CE | median KL | exp(-KL) | strategy return | oracle return | strategy DD | oracle DD | signal score | agreement | signals/day |",
      "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
      ...results.map((result) => {
        const score = result.aggregate;
        return `| ${result.candidate.family} | ${result.candidate.id} | ${result.candidate.agreementMode} | ${round(-score.objective, 5)} | ${round(-score.median, 5)} | ${round(-score.p10, 5)} | ${formatNullableNumber(score.valueDistillationKl)} | ${formatNullablePercent(score.valueDistillationScore)} | ${formatNullablePercent(score.strategyReturn)} | ${formatNullablePercent(score.oracleReturn)} | ${formatNullablePercent(score.strategyMaxDrawdown)} | ${formatNullablePercent(score.oracleMaxDrawdown)} | ${pct(score.signalScore)} | ${pct(score.exposureAgreement)} | ${round(score.signalsPerDay, 2)} |`;
      }),
    ].join("\n");
  }
  return [
    "| family | id | strategy | objective | median | P10 | precision | recall | F1 | agreement | cleanliness | noise/signal | signals/day | timing error P50 |",
    "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ...results.map((result) => {
      const score = result.aggregate;
      return `| ${result.candidate.family} | ${result.candidate.id} | ${result.candidate.agreementMode} | ${pct(score.objective)} | ${pct(score.median)} | ${pct(score.p10)} | ${pct(score.precision)} | ${pct(score.recall)} | ${pct(score.f1)} | ${pct(score.exposureAgreement)} | ${pct(score.signalCleanliness)} | ${formatNullableRatio(score.noiseSignalRatio)} | ${round(score.signalsPerDay, 2)} | ${formatNullableDuration(score.lagP50Ms)} |`;
    }),
  ].join("\n");
}

function compactCandidate(candidate: Candidate): Record<string, string | number> {
  return {
    id: candidate.id,
    family: candidate.family,
    efficiency: formatDuration(candidate.efficiencyMs),
    efficiencyVolumeEma: formatDuration(candidate.efficiencyVolumeEmaMs),
    efficiencyVolumePower: round(candidate.efficiencyVolumePower, 5),
    fast: formatDuration(candidate.fastMs),
    slow: formatDuration(candidate.slowMs),
    power: round(candidate.power, 5),
    volume: formatDuration(candidate.volumeMs),
    volumeCap: round(candidate.volumeCap, 5),
    volumePower: round(candidate.volumePower, 5),
    deadbandBpsHour: round(candidate.deadbandBpsHour, 5),
    deadbandMode: candidate.deadbandMode,
    hysteresisReleaseRatio: round(candidate.hysteresisReleaseRatio, 5),
    thresholdLookback: formatDuration(candidate.thresholdLookbackMs),
    thresholdNoiseMultiplier: round(candidate.thresholdNoiseMultiplier, 5),
    buyMaxFraction: round(candidate.buyMaxFraction, 5),
    sellMaxFraction: round(candidate.sellMaxFraction, 5),
    buySizingSigmaBpsHour: round(candidate.buySizingSigmaBpsHour, 5),
    sellSizingSigmaBpsHour: round(candidate.sellSizingSigmaBpsHour, 5),
    agreementMode: candidate.agreementMode,
    confirmationMix: round(candidate.confirmationMix, 5),
    confirmationMinQuality: round(candidate.confirmationMinQuality, 5),
    confirmationAccelerationLookback: formatDuration(candidate.confirmationAccelerationLookbackMs),
    confirmationDistanceLookback: formatDuration(candidate.confirmationDistanceLookbackMs),
    confirmationAccelerationWeight: round(candidate.confirmationAccelerationWeight, 5),
    confirmationDistanceWeight: round(candidate.confirmationDistanceWeight, 5),
    confirmationBias: round(candidate.confirmationBias, 5),
    confirmationEma: formatDuration(candidate.confirmationEmaMs),
    confirmationEmaThresholdBpsHour: round(candidate.confirmationEmaThresholdBpsHour, 5),
    confirmationEmaWeight: round(candidate.confirmationEmaWeight, 5),
    confirmationEmaGateStrength: round(candidate.confirmationEmaGateStrength, 5),
    confirmationRsi: formatDuration(candidate.confirmationRsiMs),
    confirmationRsiThreshold: round(candidate.confirmationRsiThreshold, 5),
    confirmationRsiWeight: round(candidate.confirmationRsiWeight, 5),
    confirmationDmi: formatDuration(candidate.confirmationDmiMs),
    confirmationDmiWeight: round(candidate.confirmationDmiWeight, 5),
    confirmationAdxThreshold: round(candidate.confirmationAdxThreshold, 5),
    meanReversionSuppressionThreshold: round(candidate.meanReversionSuppressionThreshold, 5),
    meanReversionEfficiency: formatDuration(candidate.meanReversionEfficiencyMs),
    meanReversionFast: formatDuration(candidate.meanReversionFastMs),
    meanReversionSlow: formatDuration(candidate.meanReversionSlowMs),
    meanReversionVolatility: formatDuration(candidate.meanReversionVolatilityMs),
    meanReversionReversalThreshold: round(candidate.meanReversionReversalThreshold, 5),
  };
}

function help(): void {
  console.log(`Usage: npm run search:kama -- [options]

  --source-dir PATH                 Daily Binance JSONL shard directory
  --source-interval 1m             Source candle scale (ms, s, m, h, d, w)
  --scales 1m,5m,15m,1h            Causal target aggregations; never finer than source
  --fit-windows START..END,...      Date-only ends are inclusive; timestamps are exclusive
  --validation-windows ...          Must follow fit windows chronologically
  --test-windows ...                Untouched until two family finalists are selected
  --algorithm de --trials 384      Differential evolution (or random) population
  --generations 12 --seed 17       Deterministic evolution controls
  --restarts 2 --full-evaluation-interval 4 --refinement-rounds 3
  --pbest-fraction 0.15             Adaptive current-to-pbest differential evolution
  --confirmation-masks base,acceleration,ema,rsi,dmi,all
  --differential-weight 0.75 --crossover-rate 0.8 --immigrant-rate 0.08
  --workers 12                      Candidate shards for global search; window shards in per-window mode
  --accelerator auto                CUDA for every profitable-size batch (auto, cuda, or cpu)
  --objective signal                signal or value-distillation fitness
  --exposure-grid-size 21 --exposure-min -1 --exposure-max 1
  --value-horizon-mode fixed        fixed or oracle-average-trade
  --value-horizon 1h                Fixed H, or fallback when oracle intervals are unavailable
  --oracle-temperature 0.01         Soft oracle value temperature in log-return units
  --strategy-temperature 0.001      Base temperature; scales by sqrt(H/dt)
  --strategy-volatility-scaling false  Also multiply by trailing-H simple-return stddev
  --opportunity-epsilon 0.000001    Added to max(Q)-min(Q) example weights
  --quote-lend-rate 0 --quote-borrow-rate 0 --asset-borrow-rate 0
  --score-version ${VW_KAMA_SCORE_VERSION}                  Objective formula version
  --seed-candidates FILE,...        Put prior fit JSONL/preset candidates in generation zero
  --screen-windows 1 --screen-scales 1m,15m   Cheap early-pruning stage
  --efficiency-range 1m..2h         Duration ranges become sample counts per scale
  --efficiency-volume-ema-range 1m..12h --efficiency-volume-power-range 0..4
  --fast-range 1s..30m --slow-range 5m..24h --volume-range 1m..12h
  --power-range 0.3..5 --volume-cap-range 1..10 --volume-power-range 0..3
  --deadband-bps-hour-range 0.05..2000
  --threshold-lookback-range 5m..24h --threshold-noise-multiplier-range 0..8
  --buy-max-fraction-range 0.05..1 --sell-max-fraction-range 0.05..1
  --buy-sizing-sigma-range 0.01..300 --sell-sizing-sigma-range 0.01..300
  --agreement-modes sizing,confidence
  --deadband-modes flat,hysteresis,hold
  --confirmation-mix-range 0..1 --confirmation-min-quality-range 0..0.95
  --confirmation-acceleration-lookback-range 1m..6h
  --confirmation-distance-lookback-range 1m..6h
  --confirmation-acceleration-weight-range 0..5
  --confirmation-distance-weight-range 0..5 --confirmation-bias-range -5..5
  --hysteresis-release-ratio-range 0..1
  --confirmation-ema-range 5m..24h --confirmation-ema-threshold-range 0.1..300
  --confirmation-ema-weight-range 0..5 --confirmation-ema-gate-strength-range 0..1
  --confirmation-rsi-range 2m..6h --confirmation-rsi-threshold-range 0..20
  --confirmation-rsi-weight-range 0..5
  --confirmation-dmi-range 2m..6h --confirmation-dmi-weight-range 0..5
  --confirmation-adx-threshold-range 5..50
  --mean-reversion-efficiency-range 1m..24h --mean-reversion-fast-range 1m..6h
  --mean-reversion-slow-range 5m..24h --mean-reversion-volatility-range 1m..24h
  --mean-reversion-suppression-threshold-range 0..3
  --mean-reversion-reversal-threshold-range 0..6
  --mode per-window --preset-window-ids ID,... --preset-output PATH
  --match-window 2h --timing-half-life 10m
  --oracle-friction 0.00175 --warmup-multiple 3 --case-warmup 72h
  --output PATH --report PATH`);
}

function validCandle(candle: TradingCandle): boolean {
  return Number.isFinite(candle.openTime) && Number.isFinite(candle.closeTime)
    && candle.open > 0 && candle.high > 0 && candle.low > 0 && candle.close > 0
    && Number.isFinite(candle.volume) && candle.volume >= 0;
}

function appendJson(file: string, value: unknown): void { fs.appendFileSync(file, `${JSON.stringify(value)}\n`); }
function utcDay(time: number): number { return Math.floor(time / DAY) * DAY; }
function ratio(value: number, total: number): number { return total > 0 ? value / total : 0; }
function eventRatio(value: number, total: number, oppositeTotal: number): number {
  return total > 0 ? value / total : oppositeTotal === 0 ? 1 : 0;
}
function harmonic(a: number, b: number): number { return a + b > 0 ? 2 * a * b / (a + b) : 0; }
function sum(values: number[]): number { return values.reduce((total, value) => total + value, 0); }
function nullableSum(values: Array<number | null>): number | null {
  const finite = values.flatMap((value) => value === null ? [] : [value]);
  return finite.length > 0 ? sum(finite) : null;
}
function compoundEquity(values: number[]): number {
  if (values.some((value) => value <= 0)) return 0;
  return Math.exp(sum(values.map(Math.log)));
}
function mean(values: number[]): number { return values.length ? sum(values) / values.length : 0; }
function unique<T>(values: T[]): T[] { return [...new Set(values)]; }
function median(values: number[]): number { return percentile(values, 0.5); }
function nullableMedian(values: Array<number | null>): number | null { return nullablePercentile(values.filter((value): value is number => value !== null), 0.5); }
function nullablePercentile(values: number[], quantile: number): number | null { return values.length ? percentile(values, quantile) : null; }
function percentile(values: number[], quantile: number): number {
  if (!values.length) return 0;
  const sorted = values.slice().sort((a, b) => a - b);
  const index = (sorted.length - 1) * quantile;
  const lower = Math.floor(index);
  const rate = index - lower;
  return (sorted[lower] ?? 0) * (1 - rate) + (sorted[lower + 1] ?? sorted[lower] ?? 0) * rate;
}
function round(value: number, digits = 4): number { return Number(value.toFixed(digits)); }
function roundObject<T extends object>(value: T): T { return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, typeof item === "number" ? round(item, 6) : item])) as T; }
function pct(value: number): string { return `${round(value * 100, 2)}%`; }
function signedPct(value: number): string { return `${value >= 0 ? "+" : ""}${pct(value)}`; }
function capitalize(value: string): string { return value[0]!.toUpperCase() + value.slice(1); }
function formatWindow(window: Window): string { return `${new Date(window.start).toISOString().slice(0, 10)}..${new Date(window.end - 1).toISOString().slice(0, 10)}`; }
function formatNullableDuration(value: number | null): string { return value === null ? "-" : formatDuration(value); }
function formatNullableRatio(value: number | null): string { return value === null ? "∞" : round(value, 3).toString(); }
function formatNullableNumber(value: number | null, digits = 5): string {
  return value === null ? "-" : round(value, digits).toString();
}
function formatNullablePercent(value: number | null): string { return value === null ? "-" : pct(value); }
function formatSearchFitness(value: number, objective: SearchObjective): string {
  return objective === "value-distillation" ? `CE ${(-value).toFixed(5)}` : `${(value * 100).toFixed(2)}%`;
}
function formatDuration(ms: number): string {
  for (const [suffix, size] of [["w", 7 * DAY], ["d", DAY], ["h", HOUR], ["m", 60_000], ["s", 1_000]] as const) {
    if (ms >= size && ms % size === 0) return `${ms / size}${suffix}`;
  }
  return `${round(ms, 0)}ms`;
}
function formatRange(range: Range, format: (value: number) => string = String): string { return `${format(range.min)}..${format(range.max)}`; }
function positive(value: string, label: string): number { const parsed = Number(value); if (!Number.isFinite(parsed) || parsed <= 0) throw new Error(`--${label} must be positive.`); return parsed; }
function nonNegative(value: string, label: string): number { const parsed = Number(value); if (!Number.isFinite(parsed) || parsed < 0) throw new Error(`--${label} must be non-negative.`); return parsed; }
function bounded(value: string, label: string, minimum: number, maximum: number): number { const parsed = Number(value); if (!Number.isFinite(parsed) || parsed < minimum || parsed > maximum) throw new Error(`--${label} must be between ${minimum} and ${maximum}.`); return parsed; }
function integer(value: string, label: string, minimum = 1): number { const parsed = Number(value); if (!Number.isSafeInteger(parsed) || parsed < minimum) throw new Error(`--${label} must be an integer >= ${minimum}.`); return parsed; }
function booleanValue(value: string, label: string): boolean {
  if (value === "true") return true;
  if (value === "false") return false;
  throw new Error(`--${label} must be true or false.`);
}
function linearRandom(random: () => number, range: Range): number { return range.min + random() * (range.max - range.min); }
function sparseRandom(random: () => number, range: Range, zeroChance = 0.25): number {
  return range.min === 0 && random() < zeroChance ? 0 : linearRandom(random, range);
}
function edgeRandom(random: () => number, range: Range): number {
  const value = random();
  if (range.min === 0 && value < 0.2) return 0;
  if (range.max === 1 && value < 0.3) return 1;
  return linearRandom(random, range);
}
function logRandom(random: () => number, range: Range): number { return range.min <= 0 ? linearRandom(random, range) : Math.exp(Math.log(range.min) + random() * (Math.log(range.max) - Math.log(range.min))); }
function unit(value: number, range: Range): number { return range.max === range.min ? 0 : clamp01((value - range.min) / (range.max - range.min)); }
function fromUnit(value: number, range: Range): number { return range.min + clamp01(value) * (range.max - range.min); }
function sparseUnit(value: number, range: Range, zeroFraction = 0.2): number {
  return range.min !== 0 ? unit(value, range) : value <= 0 ? 0 : zeroFraction + (1 - zeroFraction) * unit(value, range);
}
function sparseRange(value: number, range: Range, zeroFraction = 0.2): number {
  return range.min !== 0
    ? fromUnit(value, range)
    : value < zeroFraction ? 0 : fromUnit((value - zeroFraction) / (1 - zeroFraction), range);
}
function logUnit(value: number, range: Range): number { return range.min <= 0 ? unit(value, range) : range.max === range.min ? 0 : clamp01((Math.log(value) - Math.log(range.min)) / (Math.log(range.max) - Math.log(range.min))); }
function logRange(value: number, range: Range): number { return range.min <= 0 ? fromUnit(value, range) : Math.exp(Math.log(range.min) + clamp01(value) * (Math.log(range.max) - Math.log(range.min))); }
function clamp01(value: number): number { return Math.max(0, Math.min(1, value)); }
function shuffle(values: number[], random: () => number): void { for (let index = values.length - 1; index > 0; index -= 1) { const other = Math.floor(random() * (index + 1)); [values[index], values[other]] = [values[other]!, values[index]!]; } }
function mulberry32(seed: number): () => number { let state = seed >>> 0; return () => { state += 0x6d2b79f5; let value = state; value = Math.imul(value ^ value >>> 15, value | 1); value ^= value + Math.imul(value ^ value >>> 7, value | 61); return ((value ^ value >>> 14) >>> 0) / 4_294_967_296; }; }
