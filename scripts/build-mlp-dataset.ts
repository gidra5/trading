import fs from "node:fs/promises";
import { createHash } from "node:crypto";
import os from "node:os";
import path from "node:path";
import readline from "node:readline";
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { fileURLToPath } from "node:url";
import {
  constants as zlibConstants,
  gunzipSync,
  zstdCompressSync,
} from "node:zlib";
import {
  conditionalCutoffRawParameters,
  exposureHoldingFeasibleInterval,
  MLP_FEATURE_SCHEMA_VERSION,
  MLP_INPUT_FEATURE_COUNT,
  prepareExposureValueOracle,
  prepareExposureValueOracleCuda,
  type Candle,
} from "@trading/bot-algo";
import { fetchBinanceSpotDailyShard } from "../apps/server/src/binance-history-cache.js";
import { MlpFeatureStore } from "../apps/server/src/mlp-feature-store.js";

const DAY_MS = 86_400_000;
const MINUTE_MS = 60_000;
const SECOND_MS = 1_000;
const TEACHER_PARAMETER_COUNT = 8;
const TEACHER_METRIC_NAMES = [
  "crossEntropy",
  "klDivergence",
  "meanSquaredError",
  "iterations",
  "restarts",
  "converged",
  "distanceImbalance",
] as const;
const TEACHER_METRIC_COUNT = TEACHER_METRIC_NAMES.length;
const DISTANCE_IMBALANCE_METRIC_INDEX = 6;
const SPLITS = ["train", "validation", "test"] as const;
type Split = typeof SPLITS[number];

interface TrainingWindow { id: string; start: string; end: string }
interface ExplicitExampleSelection {
  mode: "explicit-time-blocks";
  days: Array<{ date: string; split: Split }>;
  utcBlocks: Array<[number, number]>;
  componentCoveragePredictionDelaysMs: number[];
  sourceDatasetDir?: string;
}
interface FrozenWeightedExampleSelection {
  mode: "frozen-weighted-time-blocks";
  blocks: Array<{
    date: string;
    split: Split;
    startMinute: number;
    endMinute: number;
    sourceMeanWeight: number;
  }>;
  componentCoveragePredictionDelaysMs: number[];
  sourceDatasetDir: string;
  seed: number;
  blockExamples: number;
  sourceCounts: Record<Split, number>;
  selectedCounts: Record<Split, number>;
  weightedSplits: Split[];
}
type ExampleSelection = ExplicitExampleSelection | FrozenWeightedExampleSelection;
interface TrainingPlan {
  version: number;
  id: string;
  label?: string;
  componentStoreId: string;
  dataDir: string;
  datasetDir: string;
  artifactDir: string;
  runDir: string;
  componentSeedDatasetDirs?: string[];
  oraclePreparation: { backend: "cuda" };
  samplingIntervalMs: number;
  predictionDelayMs: number;
  splitAnchorDate?: string;
  latestTestDays: number;
  excludedAggregateWindows: string[];
  windows: TrainingWindow[];
  exampleSelection?: ExampleSelection;
  frozenStudySampling?: {
    sourceDatasetDir: string;
    outputDatasetDir: string;
    outputPlanFile: string;
    blockExamples: number;
    seed: number;
    selectedCounts: Record<Split, number>;
    weightedSplits: Split[];
  };
  productionTraining?: {
    sourceDatasetDir: string;
    outputDatasetDir: string;
    outputPlanFile: string;
  };
  componentCompression?: {
    features?: "zstd";
  };
  featurePreparationWorkers?: number;
  runtimeMinuteOracleTargets?: boolean;
  execution: {
    feeBps: number;
    minimumUsableExposure: number;
    maximumUsableExposure: number;
    minimumEffectiveExposure: number;
    maximumEffectiveExposure: number;
    maintenanceBpsHour: { quoteLend: number; quoteBorrow: number; assetBorrow: number };
    gridSize: number;
    temperature: number;
    holdingPeriodSteps: number;
    valueHorizonSteps: number;
    horizonEndMode: "extend";
  };
  teacherFit: {
    representation?: "fitted-parameters" | "direct-diagnostics";
    backend: "cuda";
    device: "cuda";
    batchSize: number;
    projectionIterations: number;
    maxIterations: number;
    adaptiveIterations: number;
    adaptiveRounds: number;
    restartCount: number;
    sampleStates: number;
    sampleActions: number;
    tolerance: number;
    maxMeanKlDivergence: number;
    maxMeanSquaredError: number;
    lineSearchCandidates: number;
    temporalRefinementRounds: number;
    temporalIterations: number;
    temporalFollowupIterations: number;
    temporalEquivalentLossAbsolute: number;
    temporalEquivalentLossRelative: number;
    optimizerBackend: "pytorch-batched" | "triton-queued";
    optimizerHostCheckInterval: number;
    qualityFallbackIterations: number;
    visibleSampleFraction: number;
    scoreHingeSpan: number;
    compactVisibleInitialization: boolean;
    inputAlignmentFloats: number;
    inputQueueBatches: number;
    pipelinedRefinement: boolean;
    workerJobsBeforeRecycle: number;
  };
  training: {
    targetRepresentation?: "rawOracleProbabilities" | "minuteOracleProbabilities";
    epochs?: number;
    batchSize?: number;
    evaluationBatchSize?: number;
    validationFraction?: number;
    trainingFraction?: number;
    weightedTrainingSample?: boolean;
    selectionMetric?: "loss" | "klDivergence";
    targetValidation?: {
      klDivergence: number;
      klDivergenceStdDev?: number;
    };
    initializeFromCheckpoint?: string;
    inheritedBestEpoch?: number;
    evaluationOnly?: boolean;
    lossWeights?: {
      crossEntropy: number;
      probabilityMse: number;
      excessEntropy: number;
      temporalMutualInformation: number;
      oracleMutualInformation: number;
    };
    timeWeighting: {
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
    };
  };
}

interface TimeRange { start: number; end: number; id: string }
interface DatasetShard {
  split: Split;
  date: string;
  segment: number;
  count: number;
  teacherBackend: "cpu-bfgs" | "cuda";
  features: string;
  featureRowOffset: number;
  featureRowStride: number;
  teacherParameters: string;
  teacherMetrics: string;
  oracleRowOffset: number;
  oracleRowStride: number;
  baseTimeWeights: string;
  timeWeights: string;
  rawOracleProbabilities: string;
  minuteOracleProbabilities: string;
  resolutionDivergence: string;
  predictionTimeStart: number;
  oracleTargetTimeStart: number;
  rejectedCount?: number;
  refinementPass?: number;
  featureSchemaVersion?: number;
  teacherMetricVisibleLower?: number;
  teacherMetricVisibleUpper?: number;
}

interface FeatureComponent {
  date: string;
  count: number;
  features: string;
  featuresCompression?: "zstd";
  featuresUncompressedBytes?: number;
  times: string;
  featureSchemaVersion: number;
  rowSelectionSignature?: string;
  materializedRowRanges?: Array<[number, number]>;
}

interface OracleComponent {
  date: string;
  count: number;
  teacherBackend: "cuda";
  teacherParameters: string;
  teacherMetrics: string;
  rawOracleProbabilities: string;
  times: string;
  rejectedCount?: number;
  refinementPass?: number;
  teacherMetricVisibleLower: number;
  teacherMetricVisibleUpper: number;
  teacherFitSignature?: string;
  rowSelectionSignature?: string;
  materializedRowRanges?: Array<[number, number]>;
}

interface Progress {
  version: 6;
  planId: string;
  updatedAt: string;
  latestCompleteDay: string;
  grid?: number[];
  currentGrid?: number[];
  featureComponents: FeatureComponent[];
  oracleComponents: OracleComponent[];
  shards: DatasetShard[];
}

interface SourceDayRejection {
  date: string;
  splits: Split[];
  reason: "source-read-error" | "incomplete-or-noncontiguous-day";
  detail: string;
  expectedCandles: number;
  observedCandles: number | null;
  firstOpenTime: number | null;
  lastOpenTime: number | null;
  firstUnexpectedIndex: number | null;
  expectedOpenTime: number | null;
  observedOpenTime: number | null;
  updatedAt: string;
}

interface SourceRejectionQueue {
  version: 1;
  planId: string;
  updatedAt: string;
  cases: SourceDayRejection[];
}

interface TeacherResult { rawParameters: Float32Array; metrics: number[] }
interface PackedTeacherInputs {
  data: Float32Array;
  count: number;
  rowStride: number;
}
interface TeacherRefinementCase {
  date: string;
  time: number;
  crossEntropy: number;
  klDivergence: number;
  meanSquaredError: number;
  iterations: number;
  restarts: number;
  converged: boolean;
  refinementPass?: number;
}
interface TeacherRefinementQueue {
  version: 2;
  planId: string;
  updatedAt: string;
  cases: TeacherRefinementCase[];
}
interface GpuTeacherConfig {
  actionGrid: number[];
  currentGrid: number[];
  friction: number;
  transitionLogScale: number;
  latentLower: number;
  latentUpper: number;
  visibleLower: number;
  visibleUpper: number;
  metricVisibleLower: number;
  metricVisibleUpper: number;
  distanceEpsilon: number;
  fit: TrainingPlan["teacherFit"];
}
interface GpuTeacherProgress {
  examplesCompleted: number;
  examplesTotal: number;
  examplesPerSecond: number;
  gpuMemoryMiB: number;
  meanKlDivergence: number;
  meanSquaredError: number;
  metricVisibleLower: number;
  metricVisibleUpper: number;
  metricActionCells: number;
  metricCurrentCells: number;
  temporalWarmSelectedFraction: number;
  temporalWarmEquivalentFraction: number;
  temporalWarmQualityAcceptedFraction: number;
  temporalWarmQualityBetterFraction: number;
  temporalMeanNormalizedStepBefore: number;
  temporalMeanNormalizedStepAfter: number;
  pipelinedRefinement: boolean;
  pipelineWaitFraction: number;
}

async function main(): Promise<void> {
  const {
    planFile,
    outputOverride,
    refinementPass,
    preparationMode,
  } = parseArguments(process.argv.slice(2));
  const plan = JSON.parse(await fs.readFile(planFile, "utf8")) as TrainingPlan;
  validatePlan(plan);
  if (preparationMode === "frozen-study-source" && !plan.frozenStudySampling) {
    throw new Error("Frozen study source preparation requires frozenStudySampling in the plan.");
  }
  const runtimeWorkerJobsBeforeRecycle = runtimePositiveInteger(
    process.env.TRADING_MLP_WORKER_JOBS_BEFORE_RECYCLE,
    plan.teacherFit.workerJobsBeforeRecycle,
  );
  const directDiagnostics = plan.teacherFit.representation === "direct-diagnostics"
    || preparationMode === "frozen-study-source";
  const currentTeacherFitSignature = teacherFitSignature(plan);
  const compatibleTeacherFitSignatures = new Set([
    currentTeacherFitSignature,
    ...(directDiagnostics ? [legacyTeacherFitSignature(plan)] : []),
  ]);
  const componentStoreId = plan.componentStoreId;
  const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
  const dataDir = path.resolve(repoRoot, plan.dataDir);
  const output = path.resolve(
    repoRoot,
    outputOverride
      ?? (preparationMode === "frozen-study-source"
        ? plan.productionTraining?.sourceDatasetDir
          ?? plan.frozenStudySampling!.sourceDatasetDir
        : plan.datasetDir),
  );
  const oneSecondRoot = path.join(dataDir, "historical", "spot-btcusdt", "btcusdt", "1s");
  const oneSecondStore = new OneSecondCandleStore(dataDir);
  const latestCompleteDay = await findLatestCompleteDay(oneSecondRoot);
  const splitAnchorDay = plan.splitAnchorDate
    ? parseDay(plan.splitAnchorDate)
    : plan.exampleSelection
    ? Math.max(...exampleSelectionDays(plan.exampleSelection).map(parseDay))
    : latestCompleteDay;
  const testEnd = splitAnchorDay + DAY_MS;
  const testStart = testEnd - plan.latestTestDays * DAY_MS;
  const { trainRanges, validationRanges } = buildWindowRanges(plan);
  const allPredictionDays = selectedUtcDays(trainRanges, validationRanges, testStart, testEnd);
  const predictionDays = plan.exampleSelection
    ? exampleSelectionDays(plan.exampleSelection).map(parseDay)
      .sort((left, right) => left - right)
    : allPredictionDays;
  const pairingSegments = predictionDays.flatMap((day) => pairedSegmentsForDay(
    day,
    plan.samplingIntervalMs,
    plan.predictionDelayMs,
    trainRanges,
    validationRanges,
    testStart,
    testEnd,
    plan.exampleSelection,
  ));
  const componentCoverageDelays = plan.exampleSelection
    ? plan.exampleSelection.componentCoveragePredictionDelaysMs
    : [plan.predictionDelayMs];
  const componentPairingSegments = componentCoverageDelays.flatMap((predictionDelayMs) =>
    predictionDays.flatMap((day) => pairedSegmentsForDay(
      day,
      plan.samplingIntervalMs,
      predictionDelayMs,
      trainRanges,
      validationRanges,
      testStart,
      testEnd,
      plan.exampleSelection,
    )));
  const selectedDays = [...new Set(componentPairingSegments.map((segment) =>
    utcDay(segment.oracleTargetTimeStart)))].sort((left, right) => left - right);
  const featureRowsByDay = componentRowsByDay(pairingSegments, "feature");
  const oracleRowsByDay = componentRowsByDay(componentPairingSegments, "oracle");
  const exampleSelectionSignature = selectionSignature(plan.exampleSelection);

  await Promise.all([
    fs.mkdir(path.join(output, "components", "inputs"), { recursive: true }),
    fs.mkdir(path.join(output, "components", "oracle"), { recursive: true }),
    fs.mkdir(path.join(output, "pairs"), { recursive: true }),
  ]);
  const progressFile = path.join(output, "progress.json");
  const progress = await loadProgress(progressFile, componentStoreId, splitAnchorDay);
  const componentSeedDatasetDirs = [...new Set([
    ...(plan.componentSeedDatasetDirs ?? []),
    ...(plan.exampleSelection?.sourceDatasetDir
      ? [plan.exampleSelection.sourceDatasetDir]
      : []),
  ])];
  for (const seedDatasetDir of componentSeedDatasetDirs) {
    await importReusableComponents(
      path.resolve(repoRoot, seedDatasetDir),
      output,
      progress,
      predictionDays,
      selectedDays,
      compatibleTeacherFitSignatures,
      plan,
      featureRowsByDay,
      oracleRowsByDay,
    );
  }
  if (componentSeedDatasetDirs.length > 0) await atomicWriteJson(progressFile, progress);
  const refinementQueueFile = path.join(output, "teacher-refinement-queue.json");
  const refinementQueue = await loadRefinementQueue(refinementQueueFile, componentStoreId);
  const refinementCases = new Map(refinementQueue.cases.map((item) => [
    refinementCaseKey(item.date, item.time), item,
  ]));
  const sourceRejectionQueueFile = path.join(output, "source-rejection-queue.json");
  const sourceRejectionQueue = await loadSourceRejectionQueue(
    sourceRejectionQueueFile, componentStoreId,
  );
  const sourceRejectionCases = new Map(sourceRejectionQueue.cases.map((item) => [item.date, item]));
  let existingDataset: {
    version?: number;
    planId?: string;
    counts?: Record<Split, number>;
    grid?: number[];
    currentGrid?: number[];
    predictionDelayMs?: number;
    latestCompleteDay?: string;
    featureSchemaVersion?: number;
    featureCount?: number;
    [key: string]: unknown;
  } | undefined;
  try {
    existingDataset = JSON.parse(await fs.readFile(path.join(output, "dataset.json"), "utf8"));
    if (existingDataset.version === 8 && existingDataset.planId === plan.id
      && (existingDataset.exampleWeighting as { baseFileField?: string } | undefined)
        ?.baseFileField === "baseTimeWeights"
      && SPLITS.every((split) => (existingDataset!.counts?.[split] ?? 0) > 0)
      && refinementPass === 0
      && existingDataset.predictionDelayMs === plan.predictionDelayMs
      && (plan.exampleSelection
        || existingDataset.latestCompleteDay === isoDate(splitAnchorDay))
      && existingDataset.exampleSelectionSignature === exampleSelectionSignature
      && existingDataset.featureSchemaVersion === MLP_FEATURE_SCHEMA_VERSION
      && existingDataset.featureCount === MLP_INPUT_FEATURE_COUNT
      && existingDataset.teacherFitSignature === currentTeacherFitSignature) {
      process.stdout.write(`${JSON.stringify({
        event: "dataset-complete",
        output,
        counts: existingDataset.counts,
        resumed: true,
        predictionDelayMs: plan.predictionDelayMs,
        featureSchemaVersion: MLP_FEATURE_SCHEMA_VERSION,
      })}\n`);
      return;
    }
    if (existingDataset.version !== 7 && existingDataset.version !== 8) {
      throw new Error("Existing dataset manifest uses an incompatible component layout.");
    }
    if (existingDataset.featureSchemaVersion !== MLP_FEATURE_SCHEMA_VERSION
      || existingDataset.featureCount !== MLP_INPUT_FEATURE_COUNT) {
      throw new Error("Existing dataset uses an obsolete MLP feature contract; build the current dataset in a clean output directory.");
    }
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    if (refinementPass > 0) {
      throw new Error("Teacher refinement requires an existing complete dataset manifest.");
    }
  }
  const pendingRefinementGroups = new Set(
    [...refinementCases.values()].map((item) => item.date),
  );
  const featureStore = new MlpFeatureStore(dataDir);
  let teacherFitter: GpuTeacherFitter | undefined;
  let prefetchedSource: { day: number; promise: Promise<Candle[]> } | undefined;
  let expectedGrid: number[] | undefined = progress.grid ?? existingDataset?.grid;
  let expectedCurrentGrid: number[] | undefined = progress.currentGrid
    ?? existingDataset?.currentGrid;

  process.stdout.write(`${JSON.stringify({
    event: "dataset-start",
    planId: plan.id,
    componentStoreId,
    output,
    predictionDelayMs: plan.predictionDelayMs,
    latestCompleteDay: isoDate(splitAnchorDay),
    predictionDays: predictionDays.length,
    days: selectedDays.length,
    selectedExamples: pairingSegments.reduce((sum, segment) => sum + segment.count, 0),
    componentCoveragePredictionDelaysMs: componentCoverageDelays,
    exampleSelectionSignature,
    completedFeatureComponents: progress.featureComponents.length,
    completedOracleComponents: progress.oracleComponents.length,
    teacherBackend: plan.teacherFit.backend,
    oracleBackend: plan.oraclePreparation.backend,
    teacherBatchSize: plan.teacherFit.batchSize,
    refinementPass,
    latestTestRange: [new Date(testStart).toISOString(), new Date(testEnd).toISOString()],
  })}\n`);

  try {
    for (let dayIndex = 0; dayIndex < selectedDays.length; dayIndex += 1) {
      const day = selectedDays[dayIndex]!;
      const date = isoDate(day);
      const requiredOracleRows = oracleRowsByDay.get(date) ?? [];
      const requiredOracleSignature = rowSelectionSignature(requiredOracleRows);
      const neededSplits = [...new Set(componentPairingSegments
        .filter((segment) => utcDay(segment.oracleTargetTimeStart) === day)
        .map((segment) => segment.split))];
      const queuedSource = sourceRejectionCases.get(date);
      const existingComponent = progress.oracleComponents.find((item) => item.date === date);
      const teacherContractChanged = existingComponent?.teacherFitSignature
        ? !compatibleTeacherFitSignatures.has(existingComponent.teacherFitSignature)
        : true;
      const metricScopeChanged = existingComponent?.teacherMetricVisibleLower
          !== plan.execution.minimumUsableExposure
        || existingComponent?.teacherMetricVisibleUpper
          !== plan.execution.maximumUsableExposure;
      const needsRefinement = pendingRefinementGroups.has(date)
        || Boolean(queuedSource)
        || (existingComponent?.rejectedCount ?? 0) > 0;
      const existingRowsReusable = existingComponent
        && !metricScopeChanged
        && (!teacherContractChanged || directDiagnostics)
        && await oracleComponentComplete(
          output,
          existingComponent,
          plan.execution.gridSize,
          existingComponent.rowSelectionSignature ?? "full",
        )
        ? materializedComponentRows(existingComponent)
        : [];
      const reusableRows = new Set(existingRowsReusable);
      const missingOracleRows = requiredOracleRows.filter((row) => !reusableRows.has(row));
      const refitForQuality = refinementPass > 0 && needsRefinement
        && (existingComponent?.refinementPass ?? 0) < refinementPass;
      const fitOracleRows = refitForQuality ? requiredOracleRows : missingOracleRows;
      const shouldBuild = fitOracleRows.length > 0;
      if (!shouldBuild) continue;

      let source: Candle[];
      const oracleEnd = day + DAY_MS + plan.execution.valueHorizonSteps * SECOND_MS;
      try {
        const sourcePromise = prefetchedSource?.day === day
          ? prefetchedSource.promise
          : oneSecondStore.loadRange(day - DAY_MS, oracleEnd);
        prefetchedSource = undefined;
        source = await sourcePromise;
        const nextDay = selectedDays[dayIndex + 1];
        if (nextDay !== undefined) {
          const nextOracleEnd = nextDay + DAY_MS
            + plan.execution.valueHorizonSteps * SECOND_MS;
          const promise = oneSecondStore.loadRange(nextDay - DAY_MS, nextOracleEnd);
          promise.catch(() => undefined);
          prefetchedSource = { day: nextDay, promise };
        }
      } catch (error) {
        await recordSourceRejection({
          date,
          splits: neededSplits,
          reason: "source-read-error",
          detail: error instanceof Error ? error.message : String(error),
          expectedCandles: DAY_MS / SECOND_MS,
          observedCandles: null,
          firstOpenTime: null,
          lastOpenTime: null,
          firstUnexpectedIndex: null,
          expectedOpenTime: null,
          observedOpenTime: null,
          updatedAt: new Date().toISOString(),
        }, sourceRejectionCases, sourceRejectionQueueFile, componentStoreId, dayIndex, selectedDays.length);
        continue;
      }
      const scored = source.filter((candle) => candle.openTime >= day && candle.openTime < day + DAY_MS);
      const sourceIssue = inspectScoredDay(scored, day, date, neededSplits);
      if (sourceIssue) {
        await recordSourceRejection(
          sourceIssue,
          sourceRejectionCases,
          sourceRejectionQueueFile,
          componentStoreId,
          dayIndex,
          selectedDays.length,
        );
        continue;
      }
      const sourceRecovered = sourceRejectionCases.has(date);
      const oracleCandles = source.filter((candle) =>
        candle.openTime >= day
        && candle.openTime < oracleEnd);
      const expectedOracleCandles = DAY_MS / SECOND_MS + plan.execution.valueHorizonSteps;
      const oracleSourceIssue = inspectContinuousSourceRange(
        oracleCandles,
        day,
        expectedOracleCandles,
        date,
        neededSplits,
        "scored day plus future value horizon",
      );
      if (oracleSourceIssue) {
        await recordSourceRejection(
          oracleSourceIssue,
          sourceRejectionCases,
          sourceRejectionQueueFile,
          componentStoreId,
          dayIndex,
          selectedDays.length,
        );
        continue;
      }
      const feeRate = plan.execution.feeBps / 10_000;
      const oracleStarted = performance.now();
      const oraclePrices = Float64Array.from(oracleCandles, (candle) => candle.close);
      const reuseOracleFactor = Boolean(existingComponent)
        && expectedGrid && expectedCurrentGrid
        && await rawOracleFactorComplete(
          output, existingComponent, expectedGrid.length, requiredOracleSignature,
        );
      const preparedOracle = reuseOracleFactor ? undefined : await prepareExposureValueOracleCuda(
        oraclePrices, {
          scoreStartIndex: 0,
          holdingPeriodSteps: plan.execution.holdingPeriodSteps,
          valueHorizonSteps: plan.execution.valueHorizonSteps,
          friction: feeRate,
          gridSize: plan.execution.gridSize,
          // The teacher sees the complete effective range so survival cutoffs
          // remain identifiable even when they sit outside usable leverage.
          minExposure: plan.execution.minimumEffectiveExposure,
          maxExposure: plan.execution.maximumEffectiveExposure,
          maxEffectiveExposure: Math.max(
            Math.abs(plan.execution.minimumEffectiveExposure),
            Math.abs(plan.execution.maximumEffectiveExposure),
          ),
          // The realized path stops at the shard boundary, but rolling value
          // targets use oraclePrices through t + valueHorizonSteps.
          terminalIndex: scored.length - 1,
          temperature: plan.execution.temperature,
          opportunityEpsilon: 0,
          quoteLendRate: bpsHourToPerSecond(plan.execution.maintenanceBpsHour.quoteLend),
          quoteBorrowRate: bpsHourToPerSecond(plan.execution.maintenanceBpsHour.quoteBorrow),
          assetBorrowRate: bpsHourToPerSecond(plan.execution.maintenanceBpsHour.assetBorrow),
          includeActionValues: false,
          includeProbabilities: true,
        },
      );
      const oracle = preparedOracle?.oracle ?? {
        grid: Float64Array.from(expectedGrid!),
        currentGrid: Float64Array.from(expectedCurrentGrid!),
        probabilities: bufferFloat32(await fs.readFile(path.join(
          output,
          existingComponent!.rawOracleProbabilities,
        ))),
      };
      process.stdout.write(`${JSON.stringify({
        event: "dataset-oracle",
        date,
        day: dayIndex + 1,
        days: selectedDays.length,
        backend: reuseOracleFactor ? "stored-raw-grid" : "cuda",
        kernelMs: preparedOracle?.kernelMs ?? 0,
        wallMs: performance.now() - oracleStarted,
        candles: oracleCandles.length,
        scoredCandles: scored.length,
        futureCandles: plan.execution.valueHorizonSteps,
        horizonEndMode: plan.execution.horizonEndMode,
      })}\n`);
      if (!oracle.probabilities) throw new Error("Oracle did not retain teacher probabilities.");
      expectedGrid ??= Array.from(oracle.grid);
      expectedCurrentGrid ??= Array.from(oracle.currentGrid);
      assertSameGrid(expectedGrid, oracle.grid, "action");
      assertSameGrid(expectedCurrentGrid, oracle.currentGrid, "current-exposure");
      progress.grid = expectedGrid;
      progress.currentGrid = expectedCurrentGrid;
      teacherFitter ??= new GpuTeacherFitter(repoRoot, {
        actionGrid: expectedGrid,
        currentGrid: expectedCurrentGrid,
        friction: feeRate,
        transitionLogScale: 1 / plan.execution.temperature,
        latentLower: plan.execution.minimumEffectiveExposure,
        latentUpper: plan.execution.maximumEffectiveExposure,
        visibleLower: plan.execution.minimumEffectiveExposure,
        visibleUpper: plan.execution.maximumEffectiveExposure,
        metricVisibleLower: plan.execution.minimumUsableExposure,
        metricVisibleUpper: plan.execution.maximumUsableExposure,
        distanceEpsilon: plan.training.timeWeighting.distanceEpsilon,
        fit: refinementPass > 0 ? {
          ...plan.teacherFit,
          maxIterations: Math.max(
            plan.teacherFit.maxIterations,
            plan.teacherFit.adaptiveIterations,
          ),
          adaptiveIterations: Math.max(
            plan.teacherFit.adaptiveIterations * 2,
            plan.teacherFit.maxIterations,
          ),
          adaptiveRounds: plan.teacherFit.adaptiveRounds + 1,
          qualityFallbackIterations: Math.max(
            plan.teacherFit.qualityFallbackIterations,
            plan.teacherFit.adaptiveIterations,
          ),
          workerJobsBeforeRecycle: directDiagnostics
            ? Math.max(64, runtimeWorkerJobsBeforeRecycle)
            : runtimeWorkerJobsBeforeRecycle,
        } : {
          ...plan.teacherFit,
          workerJobsBeforeRecycle: directDiagnostics
            ? Math.max(64, runtimeWorkerJobsBeforeRecycle)
            : runtimeWorkerJobsBeforeRecycle,
        },
      }, directDiagnostics);
      {
        const rows = fitOracleRows.map((candleIndex) => ({ candleIndex }));
        const times = rows.map(({ candleIndex }) => scored[candleIndex]!.closeTime);
        const existingFeature = progress.featureComponents.find((item) => item.date === date);
        const requiredFeatureRows = featureRowsByDay.get(date) ?? [];
        const requiredFeatureSignature = rowSelectionSignature(requiredFeatureRows);
        const buildSameDayFeature = preparationMode !== "frozen-study-source"
          && predictionDays.includes(day)
          && sameIntegerRows(requiredFeatureRows, requiredOracleRows)
          && sameIntegerRows(requiredFeatureRows, fitOracleRows)
          && !await featureComponentComplete(
            output, existingFeature, requiredFeatureSignature,
          );
        const features = buildSameDayFeature
          ? await featureStore.prepare(source, times)
          : undefined;
        const cutoffIntervals = rows.map(({ candleIndex }) => exposureHoldingFeasibleInterval(
          oraclePrices,
          candleIndex,
          plan.execution.holdingPeriodSteps,
          {
            friction: feeRate,
            minExposure: plan.execution.minimumUsableExposure,
            maxExposure: plan.execution.maximumUsableExposure,
            maxEffectiveExposure: Math.max(
              Math.abs(plan.execution.minimumEffectiveExposure),
              Math.abs(plan.execution.maximumEffectiveExposure),
            ),
            quoteLendRate: bpsHourToPerSecond(plan.execution.maintenanceBpsHour.quoteLend),
            quoteBorrowRate: bpsHourToPerSecond(plan.execution.maintenanceBpsHour.quoteBorrow),
            assetBorrowRate: bpsHourToPerSecond(plan.execution.maintenanceBpsHour.assetBorrow),
          },
        ));
        const teacherInputs = packTeacherInputs(
          oracle.probabilities,
          oracle.grid,
          rows.map(({ candleIndex }) => candleIndex),
          cutoffIntervals,
          {
            latentLower: plan.execution.minimumEffectiveExposure,
            latentUpper: plan.execution.maximumEffectiveExposure,
          },
          plan.teacherFit.inputAlignmentFloats,
        );
        let encodedFeatureRows: Buffer | undefined;
        let featureEncodingMs = 0;
        const teacherStarted = performance.now();
        const fittedTeacher = await teacherFitter.fit(teacherInputs, (event) => {
          const done = event.examplesCompleted;
          if (done === 1 || done % plan.teacherFit.batchSize === 0 || done === rows.length) {
            process.stdout.write(`${JSON.stringify({
              event: "dataset-progress",
              date,
              component: "oracle",
              examplesCompleted: done,
              examplesTotal: rows.length,
              day: dayIndex + 1,
              days: selectedDays.length,
              examplesPerSecond: event.examplesPerSecond,
              gpuMemoryMiB: event.gpuMemoryMiB,
              meanKlDivergence: event.meanKlDivergence,
              meanSquaredError: event.meanSquaredError,
              metricVisibleLower: event.metricVisibleLower,
              metricVisibleUpper: event.metricVisibleUpper,
              metricActionCells: event.metricActionCells,
              metricCurrentCells: event.metricCurrentCells,
              temporalWarmSelectedFraction: event.temporalWarmSelectedFraction,
              temporalWarmEquivalentFraction: event.temporalWarmEquivalentFraction,
              temporalWarmQualityAcceptedFraction: event.temporalWarmQualityAcceptedFraction,
              temporalWarmQualityBetterFraction: event.temporalWarmQualityBetterFraction,
              temporalMeanNormalizedStepBefore: event.temporalMeanNormalizedStepBefore,
              temporalMeanNormalizedStepAfter: event.temporalMeanNormalizedStepAfter,
              pipelinedRefinement: event.pipelinedRefinement,
              pipelineWaitFraction: event.pipelineWaitFraction,
            })}\n`);
          }
        }, features ? () => {
          const started = performance.now();
          encodedFeatureRows = encodeFeatureRows(times, features);
          featureEncodingMs = performance.now() - started;
        } : undefined);
        if (features && !encodedFeatureRows) {
          throw new Error("MLP feature encoding did not run with the CUDA fit.");
        }
        const teacherWallMs = performance.now() - teacherStarted;
        const persistedRows = requiredOracleRows;
        const persistedTimes = persistedRows.map((candleIndex) => scored[candleIndex]!.closeTime);
        const teacher = refitForQuality || existingRowsReusable.length === 0
          ? fittedTeacher
          : await mergeTeacherResults(
              output,
              existingComponent!,
              existingRowsReusable,
              fitOracleRows,
              fittedTeacher,
              persistedRows,
            );
        if (refinementPass > 0 || teacherContractChanged) {
          for (const [key, item] of refinementCases) {
            if (item.date === date) refinementCases.delete(key);
          }
        }
        const acceptedIndexes: number[] = [];
        let rejectedCount = 0;
        teacher.forEach((fit, index) => {
          const [crossEntropy, klDivergence, meanSquaredError, iterations, restarts, converged] = fit.metrics;
          if (klDivergence! <= plan.teacherFit.maxMeanKlDivergence
            && meanSquaredError! <= plan.teacherFit.maxMeanSquaredError) {
            acceptedIndexes.push(index);
            return;
          }
          rejectedCount += 1;
          const item: TeacherRefinementCase = {
            date,
            time: persistedTimes[index]!,
            crossEntropy: crossEntropy!,
            klDivergence: klDivergence!,
            meanSquaredError: meanSquaredError!,
            iterations: iterations!,
            restarts: restarts!,
            converged: converged! > 0,
            ...(refinementPass > 0 ? { refinementPass } : {}),
          };
          refinementCases.set(refinementCaseKey(date, item.time), item);
        });
        // Rejections are a refinement signal, not a reason to remove a market
        // regime from MLP training. Oracle components remain complete UTC days
        // so alternate delay manifests can reuse them without refitting.
        const persistedIndexes = persistedRows.map((_, index) => index);
        const prefix = path.join(
          "components",
          "oracle",
          `${date}${refinementPass > 0 ? `.refined-${refinementPass}` : ""}`,
        );
        const oracleComponent: OracleComponent = {
          date,
          count: DAY_MS / SECOND_MS,
          teacherBackend: "cuda",
          teacherParameters: `${prefix}.teacher-parameters.f32`,
          teacherMetrics: `${prefix}.teacher-metrics.f32`,
          rawOracleProbabilities: reuseOracleFactor
            ? existingComponent!.rawOracleProbabilities
            : `${prefix}.raw-oracle-probabilities.f32`,
          times: reuseOracleFactor ? existingComponent!.times : `${prefix}.times.i64`,
          teacherMetricVisibleLower: plan.execution.minimumUsableExposure,
          teacherMetricVisibleUpper: plan.execution.maximumUsableExposure,
          teacherFitSignature: currentTeacherFitSignature,
          rowSelectionSignature: requiredOracleSignature,
          ...(requiredOracleSignature === "full"
            ? {}
            : { materializedRowRanges: rowsToRanges(requiredOracleRows) }),
          ...(rejectedCount > 0 ? { rejectedCount } : {}),
          ...(refinementPass > 0 ? { refinementPass } : {}),
        };
        const featureComponent = encodedFeatureRows
          ? {
              ...featureComponentForDate(
                date,
                requiredFeatureSignature,
                plan.componentCompression?.features,
              ),
              ...(requiredFeatureSignature === "full"
                ? {}
                : { materializedRowRanges: rowsToRanges(requiredFeatureRows) }),
            }
          : undefined;
        const persistStarted = performance.now();
        await Promise.all([
          writeTeacherParametersComponentAtomic(
            path.join(output, oracleComponent.teacherParameters), teacher,
            requiredOracleRows,
          ),
          writeTeacherMetricsComponentAtomic(
            path.join(output, oracleComponent.teacherMetrics), teacher, requiredOracleRows,
          ),
          reuseOracleFactor ? Promise.resolve() : writeRawOracleProbabilitiesAtomic(
            path.join(output, oracleComponent.rawOracleProbabilities),
            oracle.probabilities,
            oracle.grid.length,
            persistedIndexes.map((index) => persistedRows[index]!),
            requiredOracleRows,
          ),
          reuseOracleFactor
            ? Promise.resolve()
            : writeTimesComponentAtomic(
              path.join(output, oracleComponent.times), persistedTimes, requiredOracleRows,
            ),
          featureComponent && encodedFeatureRows
            ? Promise.all([
                writeFeatureRowsAtomic(
                  path.join(output, featureComponent.features),
                  encodedFeatureRows,
                  requiredFeatureRows,
                  featureComponent.featuresCompression,
                ),
                writeTimesComponentAtomic(
                  path.join(output, featureComponent.times), times, requiredFeatureRows,
                ),
              ])
            : Promise.resolve(),
          rejectedCount > 0 || refinementPass > 0 || teacherContractChanged
            ? atomicWriteJson(refinementQueueFile, {
                version: 2,
                planId: componentStoreId,
                updatedAt: new Date().toISOString(),
                cases: [...refinementCases.values()].sort((left, right) =>
                  left.time - right.time),
              } satisfies TeacherRefinementQueue)
            : Promise.resolve(),
        ]);
        replaceByDate(progress.oracleComponents, oracleComponent);
        if (featureComponent) replaceByDate(progress.featureComponents, featureComponent);
        progress.updatedAt = new Date().toISOString();
        await atomicWriteJson(progressFile, progress);
        process.stdout.write(`${JSON.stringify({
          event: "dataset-component",
          component: "oracle",
          ...oracleComponent,
        })}\n`);
        if (featureComponent) {
          process.stdout.write(`${JSON.stringify({
            event: "dataset-component",
            component: "inputs",
            ...featureComponent,
          })}\n`);
        }
        process.stdout.write(`${JSON.stringify({
          event: "dataset-stage-timing",
          date,
          component: "oracle",
          examples: persistedRows.length,
          fittedExamples: rows.length,
          acceptedExamples: acceptedIndexes.length,
          teacherWallMs,
          featureEncodingMs,
          featureEncodingOverlapped: Boolean(featureComponent),
          rawOracleGridBytes: rows.length * oracle.grid.length * Float32Array.BYTES_PER_ELEMENT,
          persistMs: performance.now() - persistStarted,
        })}\n`);
      }
      if (sourceRecovered) {
        sourceRejectionCases.delete(date);
        await writeSourceRejectionQueue(
          sourceRejectionQueueFile, componentStoreId, sourceRejectionCases,
        );
        process.stdout.write(`${JSON.stringify({
          event: "dataset-source-recovered",
          date,
          day: dayIndex + 1,
          days: selectedDays.length,
          remainingRejectedDays: sourceRejectionCases.size,
        })}\n`);
      }
    }

    // Inputs are keyed only by causal prediction time and are independent of
    // oracle delay. The frozen-study source pass deliberately omits them:
    // after weighted blocks are frozen, the normal pass materializes only
    // selected rows.
    if (preparationMode !== "frozen-study-source") {
      if ((plan.featurePreparationWorkers ?? 1) > 1
        && plan.componentCompression?.features === "zstd"
        && [...featureRowsByDay.values()].every((rows) =>
          rowSelectionSignature(rows) === "full")) {
        await materializeFullFeatureDaysParallel(
          repoRoot,
          planFile,
          output,
          progress,
          progressFile,
          predictionDays,
          featureRowsByDay,
          plan.featurePreparationWorkers!,
        );
      }
      for (let dayIndex = 0; dayIndex < predictionDays.length; dayIndex += 1) {
      const day = predictionDays[dayIndex]!;
      const date = isoDate(day);
      const requiredFeatureRows = featureRowsByDay.get(date) ?? [];
      const requiredFeatureSignature = rowSelectionSignature(requiredFeatureRows);
      const existing = progress.featureComponents.find((item) => item.date === date);
      if (await featureComponentComplete(output, existing, requiredFeatureSignature)) continue;
      const neededSplits = [...new Set(pairingSegments
        .filter((segment) => segment.date === date)
        .map((segment) => segment.split))];
      let source: Candle[];
      try {
        source = await oneSecondStore.loadRange(day - DAY_MS, day + DAY_MS);
      } catch (error) {
        await recordSourceRejection({
          date,
          splits: neededSplits,
          reason: "source-read-error",
          detail: error instanceof Error ? error.message : String(error),
          expectedCandles: DAY_MS / SECOND_MS,
          observedCandles: null,
          firstOpenTime: null,
          lastOpenTime: null,
          firstUnexpectedIndex: null,
          expectedOpenTime: null,
          observedOpenTime: null,
          updatedAt: new Date().toISOString(),
        }, sourceRejectionCases, sourceRejectionQueueFile,
        componentStoreId, dayIndex, predictionDays.length);
        continue;
      }
      const scored = source.filter((candle) => candle.openTime >= day
        && candle.openTime < day + DAY_MS);
      const sourceIssue = inspectScoredDay(scored, day, date, neededSplits);
      if (sourceIssue) {
        await recordSourceRejection(
          sourceIssue,
          sourceRejectionCases,
          sourceRejectionQueueFile,
          componentStoreId,
          dayIndex,
          predictionDays.length,
        );
        continue;
      }
      const times = requiredFeatureRows.map((index) => scored[index]!.closeTime);
      const features = await featureStore.prepare(source, times);
      const started = performance.now();
      const encoded = encodeFeatureRows(times, features);
      const component = {
        ...featureComponentForDate(
          date,
          requiredFeatureSignature,
          plan.componentCompression?.features,
        ),
        ...(requiredFeatureSignature === "full"
          ? {}
          : { materializedRowRanges: rowsToRanges(requiredFeatureRows) }),
      };
      await Promise.all([
        writeFeatureRowsAtomic(
          path.join(output, component.features),
          encoded,
          requiredFeatureRows,
          component.featuresCompression,
        ),
        writeTimesComponentAtomic(
          path.join(output, component.times), times, requiredFeatureRows,
        ),
      ]);
      replaceByDate(progress.featureComponents, component);
      progress.updatedAt = new Date().toISOString();
      await atomicWriteJson(progressFile, progress);
      if (sourceRejectionCases.delete(date)) {
        await writeSourceRejectionQueue(
          sourceRejectionQueueFile, componentStoreId, sourceRejectionCases,
        );
      }
      process.stdout.write(`${JSON.stringify({
        event: "dataset-component",
        component: "inputs",
        ...component,
        day: dayIndex + 1,
        days: predictionDays.length,
        encodingMs: performance.now() - started,
      })}\n`);
      }
    }
  } finally {
    await teacherFitter?.close();
    teacherFitter = undefined;
  }

  if (!expectedGrid || !expectedCurrentGrid) {
    throw new Error("No MLP oracle components were produced.");
  }
  if (preparationMode === "frozen-study-source") {
    for (const day of selectedDays) {
      const date = isoDate(day);
      const component = progress.oracleComponents.find((item) => item.date === date);
      if (!component || !await oracleComponentComplete(
        output,
        component,
        expectedGrid.length,
        component.rowSelectionSignature ?? "full",
      ) || !componentMaterializesRows(
        component,
        oracleRowsByDay.get(date) ?? [],
      )) {
        throw new Error(`Oracle component ${date} is incomplete; resume preparation.`);
      }
    }
    progress.shards = pairingSegments.map((segment) => {
      const oracleDate = isoDate(utcDay(segment.oracleTargetTimeStart));
      const oracle = progress.oracleComponents.find((item) => item.date === oracleDate)!;
      return datasetShardForSegment(
        segment,
        oracle,
        "",
        plan,
      );
    });
    progress.updatedAt = new Date().toISOString();
    await atomicWriteJson(progressFile, progress);
    await persistMinuteOraclePairs(
      output,
      progress,
      plan,
      oneSecondStore,
      expectedGrid,
      { persistProbabilities: false },
    );
    const sourceCounts = Object.fromEntries(SPLITS.map((split) => [
      split,
      progress.shards.filter((shard) => shard.split === split)
        .reduce((sum, shard) => sum + shard.count, 0),
    ])) as Record<Split, number>;
    await persistExampleTimeWeights(
      output,
      progress,
      plan.samplingIntervalMs,
      plan.training.timeWeighting,
    );
    const generatedPlan = plan.productionTraining
      ? await writeProductionTrainingPlan(
          repoRoot,
          plan,
          sourceCounts,
          isoDate(splitAnchorDay),
        )
      : await freezeWeightedStudyPlan(
          repoRoot,
          output,
          plan,
          sourceCounts,
        );
    progress.updatedAt = new Date().toISOString();
    await atomicWriteJson(progressFile, progress);
    process.stdout.write(`${JSON.stringify({
      event: plan.productionTraining
        ? "production-training-source-complete"
        : "frozen-study-source-complete",
      output,
      sourceCounts,
      ...(!plan.productionTraining && plan.frozenStudySampling
        ? { selectedCounts: plan.frozenStudySampling.selectedCounts }
        : {}),
      generatedPlan,
      diagnosticOnly: directDiagnostics,
    })}\n`);
    return;
  }
  for (const day of predictionDays) {
    const date = isoDate(day);
    const component = progress.featureComponents.find((item) => item.date === date);
    if (!await featureComponentComplete(
      output, component, component?.rowSelectionSignature ?? "full",
    ) || !componentMaterializesRows(
      component,
      featureRowsByDay.get(date) ?? [],
    )) {
      throw new Error(`Input component ${date} is incomplete; resume preparation.`);
    }
  }
  for (const day of selectedDays) {
    const date = isoDate(day);
    const component = progress.oracleComponents.find((item) => item.date === date);
    if (!component || !await oracleComponentComplete(
      output,
      component,
      expectedGrid.length,
      component.rowSelectionSignature ?? "full",
    ) || !componentMaterializesRows(
      component,
      oracleRowsByDay.get(date) ?? [],
    )) {
      throw new Error(`Oracle component ${date} is incomplete; resume preparation.`);
    }
  }
  progress.shards = pairingSegments.map((segment) => {
    const feature = progress.featureComponents.find((item) => item.date === segment.date)!;
    const oracleDate = isoDate(utcDay(segment.oracleTargetTimeStart));
    const oracle = progress.oracleComponents.find((item) => item.date === oracleDate)!;
    return datasetShardForSegment(segment, oracle, feature.features, plan);
  });
  if (plan.runtimeMinuteOracleTargets) {
    for (const seedDatasetDir of componentSeedDatasetDirs) {
      await importReusableResolutionMetadata(
        path.resolve(repoRoot, seedDatasetDir),
        output,
        progress.shards,
      );
    }
  }
  progress.updatedAt = new Date().toISOString();
  await atomicWriteJson(progressFile, progress);
  const resolutionSummary = await persistMinuteOraclePairs(
    output,
    progress,
    plan,
    oneSecondStore,
    expectedGrid,
    { persistProbabilities: !plan.runtimeMinuteOracleTargets },
  );
  const counts = Object.fromEntries(SPLITS.map((split) => [
    split,
    progress.shards.filter((shard) => shard.split === split)
      .reduce((sum, shard) => sum + shard.count, 0),
  ]));
  if (SPLITS.some((split) => counts[split] === 0)) {
    throw new Error(`Dataset split is empty: ${JSON.stringify(counts)}`);
  }
  const timeWeightingSummary = plan.exampleSelection?.mode
    === "frozen-weighted-time-blocks"
    ? await persistFrozenStudyWeights(
        repoRoot,
        output,
        progress,
        plan,
        plan.exampleSelection,
      )
    : await persistExampleTimeWeights(
        output,
        progress,
        plan.samplingIntervalMs,
        plan.training.timeWeighting,
      );
  progress.updatedAt = new Date().toISOString();
  await atomicWriteJson(progressFile, progress);
  const manifest = {
    version: 8,
    createdAt: new Date().toISOString(),
    planId: plan.id,
    planFile: path.relative(output, planFile),
    samplingIntervalMs: plan.samplingIntervalMs,
    predictionDelayMs: plan.predictionDelayMs,
    exampleSelectionSignature,
    ...(plan.exampleSelection ? { exampleSelection: plan.exampleSelection } : {}),
    timestampPairing: {
      predictionTime: "causal input-feature timestamp and execution timestamp",
      oracleTargetTime: "predictionTime - predictionDelayMs",
      splitAssignment: "predictionTime",
      responseLagMs: plan.predictionDelayMs,
    },
    componentLayout: {
      version: 1,
      storeId: componentStoreId,
      compression: {
        features: plan.componentCompression?.features ?? "none",
      },
      inputRows: plan.exampleSelection
        ? "sparse row-addressable UTC day files containing selected prediction-time blocks"
        : "complete UTC days keyed by prediction timestamp",
      oracleRows: plan.exampleSelection
        ? "sparse row-addressable UTC day files containing the union required by every study delay"
        : "complete UTC days keyed by oracle target timestamp",
      pairing: "manifest row offsets and strides; components are reusable across delays",
      inputComponents: progress.featureComponents,
      oracleComponents: progress.oracleComponents,
    },
    featureSchemaVersion: MLP_FEATURE_SCHEMA_VERSION,
    featureCount: MLP_INPUT_FEATURE_COUNT,
    teacherParameterCount: TEACHER_PARAMETER_COUNT,
    teacherMetricCount: TEACHER_METRIC_COUNT,
    teacherMetricNames: TEACHER_METRIC_NAMES,
    exampleWeighting: {
      dtype: "float32",
      layout: "row-major [example]",
      fileField: "timeWeights",
      baseFileField: "baseTimeWeights",
      distanceImbalanceMetadataField: "teacherMetrics.distanceImbalance",
      resolutionDivergenceMetadataField: "resolutionDivergence",
      distanceImbalanceScope: "one signed scalar per complete timestamp example, aggregated across the cutoff-applied raw oracle current-exposure/action surface on teacherMetricSupport",
      timeWeighting: plan.training.timeWeighting,
      combination:
        "distance-imbalance weight * (1 + resolutionDivergenceMultiplier * resolution JSD)",
      persistenceOrder: "strictly chronological within each dataset split",
      storedWeights: "causal unnormalized example weights",
      trainingTransform: "divide each batch by its mean only",
      summaryBySplit: timeWeightingSummary,
    },
    actionCount: expectedGrid.length,
    grid: expectedGrid,
    currentGrid: expectedCurrentGrid,
    rawOracleMap: {
      materializedDtype: "float32",
      materializedLayout: "row-major [example, currentExposure, targetExposure]",
      shape: [expectedCurrentGrid.length, expectedGrid.length],
      currentExposureGrid: "currentGrid",
      targetExposureGrid: "grid",
      normalized: true,
      losslessEncoding: "base-probabilities-plus-deterministic-transaction-transition-v1",
      factorDtype: "float32",
      factorLayout: "row-major [example, targetExposure]",
      factorShape: [expectedGrid.length],
      factorFileField: "rawOracleProbabilities",
      factorBytesPerExample: expectedGrid.length * Float32Array.BYTES_PER_ELEMENT,
      factorBytesBySplit: Object.fromEntries(SPLITS.map((split) => [
        split,
        counts[split] * expectedGrid.length * Float32Array.BYTES_PER_ELEMENT,
      ])),
      denseBytesPerExample: expectedCurrentGrid.length * expectedGrid.length
        * Float32Array.BYTES_PER_ELEMENT,
      materializer: "mlp_model.materialize_raw_oracle_policy_map",
      optionalHardCutoffCoordinates: "teacherParameters[6:8]",
      semantics: "full normalized conditional CUDA oracle map before the exact hard-feasibility cutoff and parametric fitting; stored losslessly as its oracle base row because the current-exposure transaction transition is deterministic from the manifest execution support",
    },
    minuteOracleMap: {
      factorDtype: "float32",
      factorLayout: "row-major [example, targetExposure]",
      factorShape: [expectedGrid.length],
      factorFileField: "minuteOracleProbabilities",
      storage: plan.runtimeMinuteOracleTargets
        ? "computed-directly-at-training-startup"
        : "persisted-per-example",
      sampling:
        "close-only one-minute path ending at the latest completed UTC minute available at the example timestamp",
      timestampAlignment:
        "floor to the latest completed one-minute candle; exact at second 59 and conservatively 1-59 seconds earlier otherwise",
      holdingPeriodSteps: 1,
      valueHorizonSteps: 60,
      normalized: true,
      resolutionDivergence: {
        metric: "Jensen-Shannon divergence",
        range: [0, Math.log(2)],
        support: [
          plan.execution.minimumUsableExposure,
          plan.execution.maximumUsableExposure,
        ],
        fileField: "resolutionDivergence",
        summaryBySplit: resolutionSummary,
      },
    },
    policySupport: {
      latent_lower: plan.execution.minimumEffectiveExposure,
      latent_upper: plan.execution.maximumEffectiveExposure,
      visible_lower: plan.execution.minimumUsableExposure,
      visible_upper: plan.execution.maximumUsableExposure,
      hinge_span: plan.teacherFit.scoreHingeSpan,
      friction: plan.execution.feeBps / 10_000,
      temperature: plan.execution.temperature,
    },
    teacherFitSupport: {
      visible_lower: plan.execution.minimumEffectiveExposure,
      visible_upper: plan.execution.maximumEffectiveExposure,
      purpose: "fit score and hard survival cutoffs before usable-range truncation",
    },
    teacherMetricSupport: {
      visible_lower: plan.execution.minimumUsableExposure,
      visible_upper: plan.execution.maximumUsableExposure,
      actionCells: expectedGrid.filter((value) =>
        value >= plan.execution.minimumUsableExposure
        && value <= plan.execution.maximumUsableExposure).length,
      currentCells: expectedCurrentGrid.filter((value) =>
        value >= plan.execution.minimumUsableExposure
        && value <= plan.execution.maximumUsableExposure).length,
      purpose: "quality metrics, rejection gates, refinement selection, and temporal fit selection",
    },
    splitPolicy: {
      training: plan.exampleSelection
        ? "explicit calendar days selected for the study training split"
        : "first half of every non-aggregate inspector window",
      validation: plan.exampleSelection
        ? "explicit calendar days selected for the study validation split"
        : "second half of every non-aggregate inspector window",
      test: plan.exampleSelection
        ? "explicit calendar days selected for the study test split"
        : `latest ${plan.latestTestDays} complete cached days`,
      overlapPriority: ["test", "validation", "train"],
      excludedAggregateWindows: plan.excludedAggregateWindows,
    },
    execution: plan.execution,
    oraclePreparation: plan.oraclePreparation,
    teacherFit: plan.teacherFit,
    teacherFitSignature: currentTeacherFitSignature,
    teacherBackends: [...new Set(progress.oracleComponents.map((item) => item.teacherBackend))],
    teacherRefinementQueue: {
      file: path.relative(output, refinementQueueFile),
      rejectedExamples: refinementCases.size,
      maximumKlDivergence: plan.teacherFit.maxMeanKlDivergence,
      maximumMeanSquaredError: plan.teacherFit.maxMeanSquaredError,
    },
    sourceRejectionQueue: {
      file: path.relative(output, sourceRejectionQueueFile),
      rejectedDays: sourceRejectionCases.size,
    },
    refinementPass: Math.max(
      refinementPass,
      ...progress.oracleComponents.map((item) => item.refinementPass ?? 0),
    ),
    counts,
    shards: progress.shards.filter((shard) => shard.count > 0).sort((left, right) =>
      left.date.localeCompare(right.date) || left.segment - right.segment
        || left.split.localeCompare(right.split)),
  };
  await atomicWriteJson(path.join(output, "dataset.json"), manifest);
  process.stdout.write(`${JSON.stringify({
    event: refinementPass > 0 ? "dataset-refinement-complete" : "dataset-complete",
    output,
    counts,
    refinementPass,
    predictionDelayMs: plan.predictionDelayMs,
    rejectedExamples: refinementCases.size,
    rejectedSourceDays: sourceRejectionCases.size,
  })}\n`);
}

void main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : error);
  process.exitCode = 1;
});

class GpuTeacherFitter {
  private child?: ChildProcessWithoutNullStreams;
  private workerReady?: Promise<void>;
  private workerExit?: Promise<number>;
  private active?: {
    progress: (event: GpuTeacherProgress) => void;
    resolve: () => void;
    reject: (error: Error) => void;
  };
  private stderr = "";
  private completedWorkerJobs = 0;

  constructor(
    private readonly repoRoot: string,
    private readonly config: GpuTeacherConfig,
    private readonly diagnosticOnly = false,
  ) {}

  async fit(
    inputs: PackedTeacherInputs,
    progress: (event: GpuTeacherProgress) => void,
    overlap?: () => void,
  ): Promise<TeacherResult[]> {
    if (inputs.count === 0
      || inputs.rowStride < this.config.actionGrid.length + 2
      || inputs.data.length !== inputs.count * inputs.rowStride
      || !inputs.data.every(Number.isFinite)) {
      throw new Error("CUDA teacher fitter received invalid base-probability rows.");
    }
    const temporaryDirectory = await fs.mkdtemp(path.join(os.tmpdir(), "trading-mlp-teacher-"));
    const inputFile = path.join(temporaryDirectory, "probabilities.f32");
    const configFile = path.join(temporaryDirectory, "config.json");
    const parameterFile = path.join(temporaryDirectory, "parameters.f32");
    const metricFile = path.join(temporaryDirectory, "metrics.f32");
    try {
      await Promise.all([
        fs.writeFile(inputFile, Buffer.from(
          inputs.data.buffer,
          inputs.data.byteOffset,
          inputs.data.byteLength,
        )),
        fs.writeFile(configFile, `${JSON.stringify({
          action_grid: this.config.actionGrid,
          current_grid: this.config.currentGrid,
          friction: this.config.friction,
          transition_log_scale: this.config.transitionLogScale,
          latent_lower: this.config.latentLower,
          latent_upper: this.config.latentUpper,
          visible_lower: this.config.visibleLower,
          visible_upper: this.config.visibleUpper,
          metric_visible_lower: this.config.metricVisibleLower,
          metric_visible_upper: this.config.metricVisibleUpper,
          distance_epsilon: this.config.distanceEpsilon,
          sample_states: this.config.fit.sampleStates,
          sample_actions: this.config.fit.sampleActions,
          projection_iterations: this.config.fit.projectionIterations,
          iterations: this.config.fit.maxIterations,
          adaptive_iterations: this.config.fit.adaptiveIterations,
          adaptive_rounds: this.config.fit.adaptiveRounds,
          restarts: this.config.fit.restartCount,
          batch_size: this.config.fit.batchSize,
          tolerance: this.config.fit.tolerance,
          max_mean_kl: this.config.fit.maxMeanKlDivergence,
          max_mean_mse: this.config.fit.maxMeanSquaredError,
          line_search_candidates: this.config.fit.lineSearchCandidates,
          temporal_refinement_rounds: this.config.fit.temporalRefinementRounds,
          temporal_iterations: this.config.fit.temporalIterations,
          temporal_followup_iterations: this.config.fit.temporalFollowupIterations,
          temporal_equivalent_loss_absolute:
            this.config.fit.temporalEquivalentLossAbsolute,
          temporal_equivalent_loss_relative:
            this.config.fit.temporalEquivalentLossRelative,
          optimizer_backend: this.config.fit.optimizerBackend,
          optimizer_host_check_interval: this.config.fit.optimizerHostCheckInterval,
          quality_fallback_iterations: this.config.fit.qualityFallbackIterations,
          visible_sample_fraction: this.config.fit.visibleSampleFraction,
          score_hinge_span: this.config.fit.scoreHingeSpan,
          compact_visible_initialization: this.config.fit.compactVisibleInitialization,
          input_row_stride: inputs.rowStride,
          input_queue_batches: this.config.fit.inputQueueBatches,
          pipelined_refinement: this.config.fit.pipelinedRefinement,
        }, null, 2)}\n`),
      ]);
      await this.runPython(
        inputFile,
        configFile,
        parameterFile,
        metricFile,
        inputs.count,
        progress,
        overlap,
      );
      const [parameterBuffer, metricBuffer] = await Promise.all([
        fs.readFile(parameterFile),
        fs.readFile(metricFile),
      ]);
      const expectedParameterBytes = inputs.count * TEACHER_PARAMETER_COUNT * 4;
      const expectedMetricBytes = inputs.count * TEACHER_METRIC_COUNT * 4;
      if (parameterBuffer.byteLength !== expectedParameterBytes
        || metricBuffer.byteLength !== expectedMetricBytes) {
        throw new Error("CUDA teacher fitter returned invalid output shapes.");
      }
      const parameters = bufferFloat32(parameterBuffer);
      const metrics = bufferFloat32(metricBuffer);
      return Array.from({ length: inputs.count }, (_, index) => ({
        rawParameters: parameters.slice(
          index * TEACHER_PARAMETER_COUNT,
          (index + 1) * TEACHER_PARAMETER_COUNT,
        ),
        metrics: Array.from(metrics.slice(
          index * TEACHER_METRIC_COUNT,
          (index + 1) * TEACHER_METRIC_COUNT,
        )),
      }));
    } finally {
      await fs.rm(temporaryDirectory, { recursive: true, force: true });
    }
  }

  async close(): Promise<void> {
    await this.recycleWorker();
  }

  private async runPython(
    inputFile: string,
    configFile: string,
    parameterFile: string,
    metricFile: string,
    count: number,
    progress: (event: GpuTeacherProgress) => void,
    overlap?: () => void,
  ): Promise<void> {
    await this.ensureWorker();
    if (!this.child || this.active) {
      throw new Error("CUDA teacher worker cannot accept another fitting job.");
    }
    const completion = new Promise<void>((resolve, reject) => {
      this.active = { progress, resolve, reject };
      this.child!.stdin.write(`${JSON.stringify({
        input: inputFile,
        count,
        config: configFile,
        parametersOutput: parameterFile,
        metricsOutput: metricFile,
        diagnosticOnly: this.diagnosticOnly,
      })}\n`, (error) => {
        if (!error) return;
        this.active = undefined;
        reject(error);
      });
    });
    overlap?.();
    await completion;
    this.completedWorkerJobs += 1;
    if (this.completedWorkerJobs >= this.config.fit.workerJobsBeforeRecycle) {
      await this.recycleWorker();
    }
  }

  private async recycleWorker(): Promise<void> {
    const child = this.child;
    const workerExit = this.workerExit;
    if (!child || !workerExit) return;
    child.stdin.end();
    await workerExit;
  }

  private async ensureWorker(): Promise<void> {
    if (this.workerReady) return this.workerReady;
    const python = path.join(this.repoRoot, ".venv-ml", "bin", "python");
    const child = spawn(python, [
      path.join(this.repoRoot, "ml", "fit_teacher_cuda.py"),
      "--worker",
      "--device", this.config.fit.device,
    ], {
      cwd: this.repoRoot,
      stdio: ["pipe", "pipe", "pipe"],
    });
    this.child = child;
    this.stderr = "";
    this.completedWorkerJobs = 0;
    child.stderr.setEncoding("utf8");
    child.stderr.on("data", (chunk: string) => {
      this.stderr += chunk;
      process.stderr.write(chunk);
    });
    this.workerExit = new Promise<number>((resolve) => {
      child.once("exit", (exitCode, signal) => {
        const code = exitCode ?? (signal ? 128 : 1);
        if (this.active) {
          this.active.reject(new Error(
            `CUDA teacher worker exited with code ${code}: ${this.stderr.trim()}`,
          ));
          this.active = undefined;
        }
        this.child = undefined;
        this.workerReady = undefined;
        this.workerExit = undefined;
        resolve(code);
      });
    });
    this.workerReady = new Promise<void>((resolve, reject) => {
      child.once("error", reject);
      const lines = readline.createInterface({ input: child.stdout });
      lines.on("line", (line) => {
        process.stdout.write(`${line}\n`);
        try {
          const event = JSON.parse(line) as GpuTeacherProgress & {
            event?: string;
            message?: string;
            traceback?: string;
          };
          if (event.event === "gpu-teacher-worker-ready") {
            resolve();
          } else if (event.event === "gpu-teacher-progress") {
            this.active?.progress(event);
          } else if (event.event === "gpu-teacher-complete") {
            const active = this.active;
            this.active = undefined;
            active?.resolve();
          } else if (event.event === "gpu-teacher-error") {
            const active = this.active;
            this.active = undefined;
            active?.reject(new Error(
              `CUDA teacher fitting failed: ${event.message ?? "unknown error"}`
              + `${event.traceback ? `\n${event.traceback}` : ""}`,
            ));
          }
        } catch {
          // Preserve non-JSON diagnostics without treating them as progress.
        }
      });
      this.workerExit!.then((code) => {
        if (code !== 0) {
          reject(new Error(`CUDA teacher worker exited before ready with code ${code}.`));
        }
      });
    });
    try {
      await this.workerReady;
    } catch (error) {
      this.workerReady = undefined;
      throw error;
    }
  }
}

function bufferFloat32(buffer: Buffer): Float32Array {
  return new Float32Array(
    buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength),
  );
}

function packTeacherInputs(
  probabilities: Float32Array,
  actionGrid: Float64Array,
  rowIndexes: number[],
  cutoffIntervals: readonly { lower: number; upper: number }[],
  support: { latentLower: number; latentUpper: number },
  alignmentFloats: number,
): PackedTeacherInputs {
  const actionCount = actionGrid.length;
  if (cutoffIntervals.length !== rowIndexes.length) {
    throw new Error("Teacher cutoff rows do not match requested oracle rows.");
  }
  const rowStride = Math.ceil((actionCount + 2) / alignmentFloats) * alignmentFloats;
  const data = new Float32Array(rowIndexes.length * rowStride);
  rowIndexes.forEach((sourceRow, outputRow) => {
    const start = sourceRow * actionCount;
    const source = probabilities.subarray(start, start + actionCount);
    if (source.length !== actionCount) {
      throw new Error(`Teacher probability row ${sourceRow} is outside the oracle buffer.`);
    }
    const destination = outputRow * rowStride;
    const cutoff = cutoffIntervals[outputRow]!;
    let total = 0;
    for (let action = 0; action < actionCount; action += 1) {
      const probability = actionGrid[action]! >= cutoff.lower && actionGrid[action]! <= cutoff.upper
        ? source[action]!
        : 0;
      data[destination + action] = probability;
      total += probability;
    }
    if (!(total > 0)) throw new Error(`Teacher cutoff removed every action in row ${sourceRow}.`);
    for (let action = 0; action < actionCount; action += 1) {
      data[destination + action] /= total;
    }
    const raw = conditionalCutoffRawParameters(cutoff.lower, cutoff.upper, support);
    data[destination + actionCount] = raw[0];
    data[destination + actionCount + 1] = raw[1];
  });
  return { data, count: rowIndexes.length, rowStride };
}

function buildWindowRanges(plan: TrainingPlan): { trainRanges: TimeRange[]; validationRanges: TimeRange[] } {
  const excluded = new Set(plan.excludedAggregateWindows);
  const trainRanges: TimeRange[] = [];
  const validationRanges: TimeRange[] = [];
  for (const window of plan.windows) {
    if (excluded.has(window.id)) continue;
    const start = parseDay(window.start);
    const end = parseDay(window.end) + DAY_MS;
    const midpoint = start + Math.floor((end - start) / plan.samplingIntervalMs / 2)
      * plan.samplingIntervalMs;
    trainRanges.push({ id: window.id, start, end: midpoint });
    validationRanges.push({ id: window.id, start: midpoint, end });
  }
  return { trainRanges, validationRanges };
}

function selectedUtcDays(
  train: TimeRange[],
  validation: TimeRange[],
  testStart: number,
  testEnd: number,
): number[] {
  const result = new Set<number>();
  for (const range of [...train, ...validation, { id: "latest", start: testStart, end: testEnd }]) {
    for (let day = utcDay(range.start); day < range.end; day += DAY_MS) result.add(day);
  }
  return [...result].sort((left, right) => left - right);
}

function exampleSelectionDays(selection: ExampleSelection): string[] {
  return [...new Set(
    selection.mode === "explicit-time-blocks"
      ? selection.days.map((item) => item.date)
      : selection.blocks.map((item) => item.date),
  )];
}

interface PairingSegment {
  split: Split;
  date: string;
  segment: number;
  count: number;
  rowStride: number;
  featureRowOffset: number;
  oracleRowOffset: number;
  predictionTimeStart: number;
  oracleTargetTimeStart: number;
}

function datasetShardForSegment(
  segment: PairingSegment,
  oracle: OracleComponent,
  features: string,
  plan: TrainingPlan,
): DatasetShard {
  const pairPrefix = `${segment.split}-${segment.date}-${segment.segment}`;
  return {
    split: segment.split,
    date: segment.date,
    segment: segment.segment,
    count: segment.count,
    teacherBackend: "cuda",
    features,
    featureRowOffset: segment.featureRowOffset,
    featureRowStride: segment.rowStride,
    teacherParameters: oracle.teacherParameters,
    teacherMetrics: oracle.teacherMetrics,
    oracleRowOffset: segment.oracleRowOffset,
    oracleRowStride: segment.rowStride,
    baseTimeWeights: path.join("pairs", `${pairPrefix}.base-time-weights.f32`),
    timeWeights: path.join("pairs", `${pairPrefix}.time-weights.f32`),
    rawOracleProbabilities: oracle.rawOracleProbabilities,
    minuteOracleProbabilities: path.join(
      "pairs",
      `${pairPrefix}.delay-${plan.predictionDelayMs}`
        + ".completed-minute-oracle-probabilities.f32",
    ),
    resolutionDivergence: path.join(
      "pairs",
      `${pairPrefix}.delay-${plan.predictionDelayMs}`
        + ".completed-minute-resolution-jsd.f32",
    ),
    predictionTimeStart: segment.predictionTimeStart,
    oracleTargetTimeStart: segment.oracleTargetTimeStart,
    featureSchemaVersion: MLP_FEATURE_SCHEMA_VERSION,
    teacherMetricVisibleLower: plan.execution.minimumUsableExposure,
    teacherMetricVisibleUpper: plan.execution.maximumUsableExposure,
    ...(oracle.refinementPass !== undefined ? { refinementPass: oracle.refinementPass } : {}),
  };
}

function pairedSegmentsForDay(
  day: number,
  interval: number,
  predictionDelayMs: number,
  trainRanges: TimeRange[],
  validationRanges: TimeRange[],
  testStart: number,
  testEnd: number,
  selection?: ExampleSelection,
): PairingSegment[] {
  const result: PairingSegment[] = [];
  const rowStride = interval / SECOND_MS;
  const date = isoDate(day);
  const selectedDay = selection?.mode === "explicit-time-blocks"
    ? selection.days.find((item) => item.date === date)
    : undefined;
  const frozenBlocks = selection?.mode === "frozen-weighted-time-blocks"
    ? selection.blocks.filter((item) => item.date === date)
    : undefined;
  for (let time = day + interval; time <= day + DAY_MS; time += interval) {
    const predictionTime = time - 1;
    let split: Split | undefined;
    if (selection?.mode === "explicit-time-blocks") split = selectedDay?.split;
    else if (selection?.mode === "frozen-weighted-time-blocks") {
      const minuteOfDay = (predictionTime - day) / MINUTE_MS;
      split = frozenBlocks?.find((block) =>
        minuteOfDay >= block.startMinute && minuteOfDay < block.endMinute)?.split;
    }
    else if (predictionTime >= testStart && predictionTime < testEnd) split = "test";
    else if (validationRanges.some((range) =>
      predictionTime >= range.start && predictionTime < range.end)) split = "validation";
    else if (trainRanges.some((range) =>
      predictionTime >= range.start && predictionTime < range.end)) split = "train";
    if (!split) continue;
    if (selection?.mode === "explicit-time-blocks") {
      const minuteOfDay = (predictionTime - day) / 60_000;
      if (!selection.utcBlocks.some(([start, end]) =>
        minuteOfDay >= start && minuteOfDay < end)) continue;
    }
    const oracleTargetTime = predictionTime - predictionDelayMs;
    const oracleDay = utcDay(oracleTargetTime);
    const featureRowOffset = Math.floor((predictionTime - day) / SECOND_MS);
    const oracleRowOffset = Math.floor((oracleTargetTime - oracleDay) / SECOND_MS);
    const previous = result.at(-1);
    if (previous
      && previous.split === split
      && utcDay(previous.oracleTargetTimeStart) === oracleDay
      && previous.predictionTimeStart + previous.count * interval === predictionTime
      && previous.featureRowOffset + previous.count * rowStride === featureRowOffset
      && previous.oracleRowOffset + previous.count * rowStride === oracleRowOffset) {
      previous.count += 1;
      continue;
    }
    result.push({
      split,
      date,
      segment: result.length,
      count: 1,
      rowStride,
      featureRowOffset,
      oracleRowOffset,
      predictionTimeStart: predictionTime,
      oracleTargetTimeStart: oracleTargetTime,
    });
  }
  return result;
}

function componentRowsByDay(
  segments: readonly PairingSegment[],
  kind: "feature" | "oracle",
): Map<string, number[]> {
  const rows = new Map<string, Set<number>>();
  for (const segment of segments) {
    const time = kind === "feature"
      ? segment.predictionTimeStart
      : segment.oracleTargetTimeStart;
    const date = isoDate(utcDay(time));
    const offset = kind === "feature" ? segment.featureRowOffset : segment.oracleRowOffset;
    const values = rows.get(date) ?? new Set<number>();
    for (let index = 0; index < segment.count; index += 1) {
      values.add(offset + index * segment.rowStride);
    }
    rows.set(date, values);
  }
  return new Map([...rows].map(([date, values]) => [
    date,
    [...values].sort((left, right) => left - right),
  ]));
}

function rowSelectionSignature(rows: readonly number[]): string {
  if (rows.length === DAY_MS / SECOND_MS
    && rows.every((value, index) => value === index)) return "full";
  const hash = createHash("sha256");
  const values = new Uint32Array(rows);
  hash.update(Buffer.from(values.buffer, values.byteOffset, values.byteLength));
  return `sparse-${rows.length}-${hash.digest("hex").slice(0, 16)}`;
}

function rowsToRanges(rows: readonly number[]): Array<[number, number]> {
  const result: Array<[number, number]> = [];
  for (const row of rows) {
    const previous = result.at(-1);
    if (previous && row === previous[1]) previous[1] += 1;
    else result.push([row, row + 1]);
  }
  return result;
}

function materializedComponentRows(
  component: { rowSelectionSignature?: string; materializedRowRanges?: Array<[number, number]> }
    | undefined,
): number[] {
  if (!component || component.rowSelectionSignature === "full"
    || component.rowSelectionSignature === undefined) {
    return component ? Array.from({ length: DAY_MS / SECOND_MS }, (_, index) => index) : [];
  }
  const rows = (component.materializedRowRanges ?? []).flatMap(([start, end]) =>
    Array.from({ length: end - start }, (_, index) => start + index));
  if (rowSelectionSignature(rows) !== component.rowSelectionSignature) {
    throw new Error("Sparse component row ranges do not match their selection signature.");
  }
  return rows;
}

function componentMaterializesRows(
  component: {
    rowSelectionSignature?: string;
    materializedRowRanges?: Array<[number, number]>;
  } | undefined,
  requiredRows: readonly number[],
): boolean {
  if (!component) return false;
  if (component.rowSelectionSignature === "full"
    || component.rowSelectionSignature === undefined) return true;
  const ranges = component.materializedRowRanges ?? [];
  let rangeIndex = 0;
  for (const row of requiredRows) {
    while (rangeIndex < ranges.length && row >= ranges[rangeIndex]![1]) {
      rangeIndex += 1;
    }
    const range = ranges[rangeIndex];
    if (!range || row < range[0] || row >= range[1]) return false;
  }
  return true;
}

function selectionSignature(selection: ExplicitExampleSelection | undefined): string {
  return selection
    ? createHash("sha256").update(JSON.stringify(selection)).digest("hex")
    : "full";
}

function componentCoversRows(
  componentSignature: string | undefined,
  requiredSignature: string,
): boolean {
  // Components written before sparse study support were complete UTC days.
  return componentSignature === "full" || componentSignature === undefined
    || componentSignature === requiredSignature;
}

function sameIntegerRows(left: readonly number[], right: readonly number[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

async function materializeFullFeatureDaysParallel(
  repoRoot: string,
  planFile: string,
  output: string,
  progress: Progress,
  progressFile: string,
  predictionDays: readonly number[],
  featureRowsByDay: ReadonlyMap<string, readonly number[]>,
  requestedWorkers: number,
): Promise<void> {
  const pending: Array<{ date: string; day: number; days: number }> = [];
  for (let index = 0; index < predictionDays.length; index += 1) {
    const date = isoDate(predictionDays[index]!);
    const required = rowSelectionSignature(featureRowsByDay.get(date) ?? []);
    const existing = progress.featureComponents.find((item) => item.date === date);
    if (!await featureComponentComplete(output, existing, required)) {
      pending.push({ date, day: index + 1, days: predictionDays.length });
    }
  }
  if (pending.length === 0) return;
  const workerCount = Math.min(requestedWorkers, pending.length);
  const groups = Array.from({ length: workerCount }, () =>
    [] as Array<{ date: string; day: number; days: number }>);
  pending.forEach((request, index) => groups[index % workerCount]!.push(request));
  process.stdout.write(`${JSON.stringify({
    event: "dataset-feature-workers-start",
    workers: workerCount,
    pendingDays: pending.length,
    completedDays: predictionDays.length - pending.length,
    compression: "zstd",
  })}\n`);

  let commitChain = Promise.resolve();
  const commit = (component: FeatureComponent & {
    event?: string;
    component?: string;
    day?: number;
    days?: number;
    encodingMs?: number;
    compressedBytes?: number;
  }): Promise<void> => {
    commitChain = commitChain.then(async () => {
      if (component.count !== DAY_MS / SECOND_MS
        || component.rowSelectionSignature !== "full"
        || component.featuresCompression !== "zstd"
        || !await featureComponentComplete(output, component, "full")) {
        throw new Error(`Parallel feature worker returned an invalid ${component.date} component.`);
      }
      const {
        event: _event,
        component: _component,
        day,
        days,
        encodingMs,
        compressedBytes,
        ...persisted
      } = component;
      replaceByDate(progress.featureComponents, persisted);
      progress.updatedAt = new Date().toISOString();
      await atomicWriteJson(progressFile, progress);
      process.stdout.write(`${JSON.stringify({
        event: "dataset-component",
        component: "inputs",
        ...persisted,
        day,
        days,
        encodingMs,
        compressedBytes,
        parallelWorkers: workerCount,
      })}\n`);
    });
    return commitChain;
  };

  await Promise.all(groups.map(async (requests) => {
    const child = spawn(process.execPath, [
      path.join(repoRoot, "node_modules/tsx/dist/cli.mjs"),
      path.join(repoRoot, "scripts/build-mlp-feature-days.ts"),
      "--plan", planFile,
      "--output", output,
    ], {
      cwd: repoRoot,
      env: { ...process.env, TMPDIR: "/tmp" },
      stdio: ["pipe", "pipe", "pipe"],
    });
    child.stderr.on("data", (chunk) => process.stderr.write(chunk));
    const exited = new Promise<number>((resolve, reject) => {
      child.once("error", reject);
      child.once("exit", (code) => resolve(code ?? 1));
    });
    child.stdin.end(requests.map((request) => JSON.stringify(request)).join("\n") + "\n");
    const lines = readline.createInterface({ input: child.stdout });
    for await (const line of lines) {
      if (!line) continue;
      const value = JSON.parse(line) as FeatureComponent & {
        event?: string;
        component?: string;
        day?: number;
        days?: number;
        encodingMs?: number;
        compressedBytes?: number;
      };
      if (value.event !== "dataset-component" || value.component !== "inputs") {
        throw new Error(`Unexpected parallel feature-worker output: ${line}`);
      }
      await commit(value);
    }
    const exitCode = await exited;
    if (exitCode !== 0) {
      throw new Error(`Parallel feature worker exited with code ${exitCode}.`);
    }
  }));
  await commitChain;
  process.stdout.write(`${JSON.stringify({
    event: "dataset-feature-workers-complete",
    workers: workerCount,
    preparedDays: pending.length,
  })}\n`);
}

function encodeFeatureRows(
  times: readonly number[],
  features: Awaited<ReturnType<MlpFeatureStore["prepare"]>>,
): Buffer {
  const values = new Float32Array(times.length * MLP_INPUT_FEATURE_COUNT);
  for (let row = 0; row < times.length; row += 1) {
    features.encode(times[row]!, values, row * MLP_INPUT_FEATURE_COUNT);
  }
  const half = new Uint16Array(values.length);
  for (let index = 0; index < values.length; index += 1) {
    half[index] = float32ToFloat16(values[index]!);
  }
  return Buffer.from(half.buffer, half.byteOffset, half.byteLength);
}

function featureComponentForDate(
  date: string,
  rowSelection = "full",
  compression?: "zstd",
): FeatureComponent {
  const prefix = path.join("components", "inputs", date);
  const uncompressedBytes = (DAY_MS / SECOND_MS)
    * MLP_INPUT_FEATURE_COUNT * 2;
  return {
    date,
    count: DAY_MS / SECOND_MS,
    features: `${prefix}.features.f16${compression === "zstd" ? ".zst" : ""}`,
    ...(compression === "zstd" ? {
      featuresCompression: compression,
      featuresUncompressedBytes: uncompressedBytes,
    } : {}),
    times: `${prefix}.times.i64`,
    featureSchemaVersion: MLP_FEATURE_SCHEMA_VERSION,
    rowSelectionSignature: rowSelection,
    ...(rowSelection === "full" ? {} : { materializedRowRanges: [] }),
  };
}

async function featureComponentComplete(
  output: string,
  component: FeatureComponent | undefined,
  requiredRowSelection = "full",
): Promise<boolean> {
  return Boolean(component
    && component.count === DAY_MS / SECOND_MS
    && componentCoversRows(component.rowSelectionSignature, requiredRowSelection)
    && component.featureSchemaVersion === MLP_FEATURE_SCHEMA_VERSION
    && (component.featuresCompression === "zstd"
      ? component.featuresUncompressedBytes
          === component.count * MLP_INPUT_FEATURE_COUNT * 2
        && await nonEmptyFile(path.join(output, component.features))
      : await fileHasBytes(
          path.join(output, component.features),
          component.count * MLP_INPUT_FEATURE_COUNT * 2,
        ))
    && await fileHasBytes(path.join(output, component.times), component.count * 8));
}

async function oracleComponentComplete(
  output: string,
  component: OracleComponent,
  actionCount: number,
  requiredRowSelection = "full",
): Promise<boolean> {
  return component.count === DAY_MS / SECOND_MS
    && componentCoversRows(component.rowSelectionSignature, requiredRowSelection)
    && await fileHasBytes(
      path.join(output, component.teacherParameters),
      component.count * TEACHER_PARAMETER_COUNT * 4,
    )
    && await fileHasBytes(
      path.join(output, component.teacherMetrics),
      component.count * TEACHER_METRIC_COUNT * 4,
    )
    && await fileHasBytes(
      path.join(output, component.rawOracleProbabilities),
      component.count * actionCount * 4,
    )
    && await fileHasBytes(path.join(output, component.times), component.count * 8);
}

async function rawOracleFactorComplete(
  output: string,
  component: OracleComponent,
  actionCount: number,
  requiredRowSelection = "full",
): Promise<boolean> {
  return component.count === DAY_MS / SECOND_MS
    && componentCoversRows(component.rowSelectionSignature, requiredRowSelection)
    && await fileHasBytes(
      path.join(output, component.rawOracleProbabilities),
      component.count * actionCount * 4,
    )
    && await fileHasBytes(path.join(output, component.times), component.count * 8);
}

async function fileHasBytes(file: string, expectedBytes: number): Promise<boolean> {
  try {
    return (await fs.stat(file)).size === expectedBytes;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return false;
    throw error;
  }
}

async function nonEmptyFile(file: string): Promise<boolean> {
  try {
    return (await fs.stat(file)).size > 0;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return false;
    throw error;
  }
}

async function importReusableComponents(
  sourceOutput: string,
  output: string,
  progress: Progress,
  predictionDays: readonly number[],
  oracleDays: readonly number[],
  compatibleTeacherFitSignatures: ReadonlySet<string>,
  plan: TrainingPlan,
  featureRowsByDay: ReadonlyMap<string, readonly number[]>,
  oracleRowsByDay: ReadonlyMap<string, readonly number[]>,
): Promise<void> {
  if (path.resolve(sourceOutput) === path.resolve(output)) return;
  let source: Progress;
  try {
    source = JSON.parse(await fs.readFile(path.join(sourceOutput, "progress.json"), "utf8")) as Progress;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return;
    throw error;
  }
  if (source.planId !== plan.componentStoreId) return;
  let importedFeatures = 0;
  let importedOracles = 0;
  for (const day of predictionDays) {
    const date = isoDate(day);
    const required = rowSelectionSignature(featureRowsByDay.get(date) ?? []);
    const target = progress.featureComponents.find((item) => item.date === date);
    if (await featureComponentComplete(output, target, required)) continue;
    const component = source.featureComponents.find((item) => item.date === date);
    if (!component || !await featureComponentComplete(
      sourceOutput, component, component.rowSelectionSignature ?? "full",
    )) continue;
    const targetRows = materializedComponentRows(target);
    const sourceRows = materializedComponentRows(component);
    if (targetRows.length > 0 && targetRows.length >= sourceRows.length) continue;
    await linkComponentFiles(sourceOutput, output, [component!.features, component!.times]);
    replaceByDate(progress.featureComponents, {
      ...component!,
      rowSelectionSignature: component!.rowSelectionSignature ?? "full",
    });
    importedFeatures += 1;
  }
  const actionCount = source.grid?.length ?? plan.execution.gridSize;
  for (const day of oracleDays) {
    const date = isoDate(day);
    const required = rowSelectionSignature(oracleRowsByDay.get(date) ?? []);
    const target = progress.oracleComponents.find((item) => item.date === date);
    if (target?.teacherFitSignature
      && compatibleTeacherFitSignatures.has(target.teacherFitSignature)
      && target.teacherMetricVisibleLower === plan.execution.minimumUsableExposure
      && target.teacherMetricVisibleUpper === plan.execution.maximumUsableExposure
      && await oracleComponentComplete(output, target, actionCount, required)) continue;
    const component = source.oracleComponents.find((item) => item.date === date);
    if (!component
      || !component.teacherFitSignature
      || !compatibleTeacherFitSignatures.has(component.teacherFitSignature)
      || component.teacherMetricVisibleLower !== plan.execution.minimumUsableExposure
      || component.teacherMetricVisibleUpper !== plan.execution.maximumUsableExposure
      || !await oracleComponentComplete(
        sourceOutput,
        component,
        actionCount,
        component.rowSelectionSignature ?? "full",
      )) continue;
    const targetRows = materializedComponentRows(target);
    const sourceRows = materializedComponentRows(component);
    if (targetRows.length > 0 && targetRows.length >= sourceRows.length) continue;
    await linkComponentFiles(sourceOutput, output, [
      component.teacherParameters,
      component.teacherMetrics,
      component.rawOracleProbabilities,
      component.times,
    ]);
    replaceByDate(progress.oracleComponents, {
      ...component,
      rowSelectionSignature: component.rowSelectionSignature ?? "full",
    });
    importedOracles += 1;
  }
  progress.grid ??= source.grid;
  progress.currentGrid ??= source.currentGrid;
  if (importedFeatures > 0 || importedOracles > 0) {
    progress.updatedAt = new Date().toISOString();
    process.stdout.write(`${JSON.stringify({
      event: "dataset-components-reused",
      sourceOutput,
      importedFeatures,
      importedOracles,
    })}\n`);
  }
}

async function importReusableResolutionMetadata(
  sourceOutput: string,
  output: string,
  targetShards: readonly DatasetShard[],
): Promise<void> {
  if (path.resolve(sourceOutput) === path.resolve(output)) return;
  let source: Progress;
  try {
    source = JSON.parse(
      await fs.readFile(path.join(sourceOutput, "progress.json"), "utf8"),
    ) as Progress;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return;
    throw error;
  }
  const sourceByPairing = new Map(source.shards.map((shard) => [
    shardPairingKey(shard),
    shard,
  ]));
  let imported = 0;
  for (const target of targetShards) {
    const existing = path.join(output, target.resolutionDivergence);
    if (await fileHasBytes(existing, target.count * Float32Array.BYTES_PER_ELEMENT)) {
      continue;
    }
    const reusable = sourceByPairing.get(shardPairingKey(target));
    if (!reusable || reusable.resolutionDivergence !== target.resolutionDivergence
      || !await fileHasBytes(
        path.join(sourceOutput, reusable.resolutionDivergence),
        reusable.count * Float32Array.BYTES_PER_ELEMENT,
      )) continue;
    await linkComponentFiles(
      sourceOutput,
      output,
      [reusable.resolutionDivergence],
    );
    imported += 1;
  }
  if (imported > 0) {
    process.stdout.write(`${JSON.stringify({
      event: "dataset-resolution-metadata-reused",
      sourceOutput,
      importedShards: imported,
      semantics: "stored one-second/one-minute resolution JSD; minute targets remain runtime-only",
    })}\n`);
  }
}

function shardPairingKey(shard: DatasetShard): string {
  return [
    shard.split,
    shard.date,
    shard.segment,
    shard.count,
    shard.featureRowOffset,
    shard.featureRowStride,
    shard.oracleRowOffset,
    shard.oracleRowStride,
    shard.predictionTimeStart,
    shard.oracleTargetTimeStart,
  ].join(":");
}

async function linkComponentFiles(
  sourceOutput: string,
  output: string,
  relativeFiles: readonly string[],
): Promise<void> {
  for (const relativeFile of relativeFiles) {
    const source = path.join(sourceOutput, relativeFile);
    const destination = path.join(output, relativeFile);
    await fs.mkdir(path.dirname(destination), { recursive: true });
    await fs.rm(destination, { force: true });
    try {
      await fs.link(source, destination);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "EXDEV") throw error;
      await fs.copyFile(source, destination);
    }
  }
}

function replaceByDate<T extends { date: string }>(values: T[], value: T): void {
  const index = values.findIndex((item) => item.date === value.date);
  if (index >= 0) values[index] = value;
  else values.push(value);
}

function teacherFitSignature(plan: TrainingPlan): string {
  const execution = plan.execution;
  const contract = {
    teacherFit: plan.teacherFit,
    support: {
      latentLower: execution.minimumEffectiveExposure,
      latentUpper: execution.maximumEffectiveExposure,
      metricVisibleLower: execution.minimumUsableExposure,
      metricVisibleUpper: execution.maximumUsableExposure,
      friction: execution.feeBps / 10_000,
      temperature: execution.temperature,
      gridSize: execution.gridSize,
    },
  };
  return createHash("sha256").update(JSON.stringify(contract)).digest("hex");
}

function legacyTeacherFitSignature(plan: TrainingPlan): string {
  const { representation: _, ...legacyTeacherFit } = plan.teacherFit;
  const contract = {
    teacherFit: legacyTeacherFit,
    support: {
      latentLower: plan.execution.minimumEffectiveExposure,
      latentUpper: plan.execution.maximumEffectiveExposure,
      metricVisibleLower: plan.execution.minimumUsableExposure,
      metricVisibleUpper: plan.execution.maximumUsableExposure,
      friction: plan.execution.feeBps / 10_000,
      temperature: plan.execution.temperature,
      gridSize: plan.execution.gridSize,
    },
  };
  return createHash("sha256").update(JSON.stringify(contract)).digest("hex");
}

async function writeTeacherParametersAtomic(file: string, rows: TeacherResult[]): Promise<void> {
  const values = new Float32Array(rows.length * TEACHER_PARAMETER_COUNT);
  rows.forEach((row, index) => values.set(row.rawParameters, index * TEACHER_PARAMETER_COUNT));
  await writeTypedArrayAtomic(file, Buffer.from(values.buffer));
}

async function writeTeacherParametersComponentAtomic(
  file: string,
  rows: TeacherResult[],
  componentRows: readonly number[],
): Promise<void> {
  const values = new Float32Array(rows.length * TEACHER_PARAMETER_COUNT);
  rows.forEach((row, index) => values.set(row.rawParameters, index * TEACHER_PARAMETER_COUNT));
  await writeComponentRowsAtomic(
    file,
    Buffer.from(values.buffer, values.byteOffset, values.byteLength),
    TEACHER_PARAMETER_COUNT * Float32Array.BYTES_PER_ELEMENT,
    componentRows,
  );
}

async function writeTeacherMetricsAtomic(file: string, rows: TeacherResult[]): Promise<void> {
  const values = new Float32Array(rows.length * TEACHER_METRIC_COUNT);
  rows.forEach((row, index) => values.set(row.metrics, index * TEACHER_METRIC_COUNT));
  await writeTypedArrayAtomic(file, Buffer.from(values.buffer));
}

async function writeTeacherMetricsComponentAtomic(
  file: string,
  rows: TeacherResult[],
  componentRows: readonly number[],
): Promise<void> {
  const values = new Float32Array(rows.length * TEACHER_METRIC_COUNT);
  rows.forEach((row, index) => values.set(row.metrics, index * TEACHER_METRIC_COUNT));
  await writeComponentRowsAtomic(
    file,
    Buffer.from(values.buffer, values.byteOffset, values.byteLength),
    TEACHER_METRIC_COUNT * Float32Array.BYTES_PER_ELEMENT,
    componentRows,
  );
}

async function mergeTeacherResults(
  output: string,
  existing: OracleComponent,
  existingRows: readonly number[],
  fittedRows: readonly number[],
  fitted: readonly TeacherResult[],
  outputRows: readonly number[],
): Promise<TeacherResult[]> {
  if (fittedRows.length !== fitted.length) {
    throw new Error("Fitted teacher rows do not match the returned CUDA results.");
  }
  const [parameterBuffer, metricBuffer] = await Promise.all([
    fs.readFile(path.join(output, existing.teacherParameters)),
    fs.readFile(path.join(output, existing.teacherMetrics)),
  ]);
  const parameters = bufferFloat32(parameterBuffer);
  const metrics = bufferFloat32(metricBuffer);
  const existingSet = new Set(existingRows);
  const fittedIndex = new Map(fittedRows.map((row, index) => [row, index]));
  return outputRows.map((row) => {
    const replacement = fittedIndex.get(row);
    if (replacement !== undefined) return fitted[replacement]!;
    if (!existingSet.has(row)) {
      throw new Error(`Teacher component merge is missing row ${row}.`);
    }
    return {
      rawParameters: parameters.slice(
        row * TEACHER_PARAMETER_COUNT,
        (row + 1) * TEACHER_PARAMETER_COUNT,
      ),
      metrics: Array.from(metrics.slice(
        row * TEACHER_METRIC_COUNT,
        (row + 1) * TEACHER_METRIC_COUNT,
      )),
    };
  });
}

async function persistMinuteOraclePairs(
  output: string,
  progress: Progress,
  plan: TrainingPlan,
  oneSecondStore: OneSecondCandleStore,
  actionGrid: readonly number[],
  options: { persistProbabilities: boolean } = { persistProbabilities: true },
): Promise<Record<Split, {
  examples: number;
  meanResolutionJsd: number;
  maximumResolutionJsd: number;
}>> {
  const visibleIndexes = actionGrid
    .map((action, index) => ({ action, index }))
    .filter(({ action }) =>
      action >= plan.execution.minimumUsableExposure
      && action <= plan.execution.maximumUsableExposure)
    .map(({ index }) => index);
  const byTargetDate = new Map<string, DatasetShard[]>();
  for (const shard of progress.shards) {
    const targetDate = isoDate(utcDay(shard.oracleTargetTimeStart));
    const values = byTargetDate.get(targetDate) ?? [];
    values.push(shard);
    byTargetDate.set(targetDate, values);
  }
  const fineCache = new Map<string, Float32Array>();
  for (const [date, shards] of [...byTargetDate].sort(([left], [right]) =>
    left.localeCompare(right))) {
    const incomplete = await Promise.all(shards.map(async (shard) =>
      (options.persistProbabilities && !await fileHasBytes(
          path.join(output, shard.minuteOracleProbabilities),
          shard.count * actionGrid.length * Float32Array.BYTES_PER_ELEMENT,
        ))
      || !await fileHasBytes(
        path.join(output, shard.resolutionDivergence),
        shard.count * Float32Array.BYTES_PER_ELEMENT,
      )));
    if (incomplete.every((value) => !value)) continue;

    const day = parseDay(date);
    const source = await oneSecondStore.loadRange(
      day - MINUTE_MS,
      day + DAY_MS + plan.execution.valueHorizonSteps * SECOND_MS,
    );
    const expected = DAY_MS / SECOND_MS + plan.execution.valueHorizonSteps
      + MINUTE_MS / SECOND_MS;
    const issue = inspectContinuousSourceRange(
      source,
      day - MINUTE_MS,
      expected,
      date,
      [...new Set(shards.map((shard) => shard.split))],
      "completed one-minute oracle source",
    );
    if (issue) throw new Error(issue.detail);

    const prices = Float64Array.from(
      { length: source.length / 60 },
      (_, index) => source[(index + 1) * 60 - 1]!.close,
    );
    const feeRate = plan.execution.feeBps / 10_000;
    const minuteOracle = prepareExposureValueOracle(prices, {
      scoreStartIndex: 0,
      holdingPeriodSteps: 1,
      valueHorizonSteps: 60,
      friction: feeRate,
      gridSize: plan.execution.gridSize,
      minExposure: plan.execution.minimumEffectiveExposure,
      maxExposure: plan.execution.maximumEffectiveExposure,
      maxEffectiveExposure: Math.max(
        Math.abs(plan.execution.minimumEffectiveExposure),
        Math.abs(plan.execution.maximumEffectiveExposure),
      ),
      terminalIndex: 1_440,
      temperature: plan.execution.temperature,
      opportunityEpsilon: 0,
      quoteLendRate: bpsHourToPerSteps(
        plan.execution.maintenanceBpsHour.quoteLend,
        60,
      ),
      quoteBorrowRate: bpsHourToPerSteps(
        plan.execution.maintenanceBpsHour.quoteBorrow,
        60,
      ),
      assetBorrowRate: bpsHourToPerSteps(
        plan.execution.maintenanceBpsHour.assetBorrow,
        60,
      ),
      includeActionValues: false,
      includeProbabilities: true,
    });
    if (!minuteOracle.probabilities) {
      throw new Error(`One-minute oracle did not retain probabilities for ${date}.`);
    }
    assertSameGrid(actionGrid, minuteOracle.grid, "one-minute action");
    const outputs = new Map<DatasetShard, {
      probabilities?: Float32Array;
      divergence: Float32Array;
    }>();
    shards.forEach((shard, index) => {
      if (incomplete[index]) {
        outputs.set(shard, {
          ...(options.persistProbabilities
            ? { probabilities: new Float32Array(shard.count * actionGrid.length) }
            : {}),
          divergence: new Float32Array(shard.count),
        });
      }
    });
    for (const shard of outputs.keys()) {
      for (let row = 0; row < shard.count; row += 1) {
        const oracleRow = shard.oracleRowOffset + row * shard.oracleRowStride;
        const phase = oracleRow % 60;
        const minute = Math.floor(oracleRow / 60);
        // At second 59 the current UTC minute has just completed. At every
        // earlier second, use the previous completed minute. With a 60-minute
        // prediction delay this entire 60-step coarse path is therefore a
        // deterministic subset of the model's causal one-minute inputs.
        const coarseRow = minute + (phase === 59 ? 1 : 0);
        const coarseStart = coarseRow * actionGrid.length;
        const coarse = minuteOracle.probabilities.subarray(
          coarseStart,
          coarseStart + actionGrid.length,
        );
        const destination = outputs.get(shard)!;
        destination.probabilities?.set(coarse, row * actionGrid.length);
        let fine = fineCache.get(shard.rawOracleProbabilities);
        if (!fine) {
          fine = bufferFloat32(await fs.readFile(path.join(
            output,
            shard.rawOracleProbabilities,
          )));
          fineCache.set(shard.rawOracleProbabilities, fine);
          while (fineCache.size > 4) fineCache.delete(fineCache.keys().next().value!);
        }
        const fineStart = (
          shard.oracleRowOffset + row * shard.oracleRowStride
        ) * actionGrid.length;
        destination.divergence[row] = visibleJensenShannonDivergence(
          fine.subarray(fineStart, fineStart + actionGrid.length),
          coarse,
          visibleIndexes,
        );
      }
    }
    await Promise.all([...outputs].flatMap(([shard, values]) => [
      ...(values.probabilities ? [writeTypedArrayAtomic(
        path.join(output, shard.minuteOracleProbabilities),
        Buffer.from(
          values.probabilities.buffer,
          values.probabilities.byteOffset,
          values.probabilities.byteLength,
        ),
      )] : []),
      writeTypedArrayAtomic(
        path.join(output, shard.resolutionDivergence),
        Buffer.from(
          values.divergence.buffer,
          values.divergence.byteOffset,
          values.divergence.byteLength,
        ),
      ),
    ]));
    process.stdout.write(`${JSON.stringify({
      event: "dataset-minute-oracle",
      date,
      examples: [...outputs.keys()].reduce((sum, shard) => sum + shard.count, 0),
      alignment: "latest-completed-minute",
    })}\n`);
  }

  const summary = {} as Record<Split, {
    examples: number;
    meanResolutionJsd: number;
    maximumResolutionJsd: number;
  }>;
  for (const split of SPLITS) {
    let count = 0;
    let total = 0;
    let maximum = 0;
    for (const shard of progress.shards.filter((item) => item.split === split)) {
      const divergence = bufferFloat32(
        await fs.readFile(path.join(output, shard.resolutionDivergence)),
      );
      if (divergence.length !== shard.count) {
        throw new Error(`Resolution divergence rows are misaligned for ${shard.date}.`);
      }
      for (const value of divergence) {
        if (!Number.isFinite(value) || value < 0 || value > Math.log(2) + 1e-5) {
          throw new Error(`Invalid one-second/one-minute resolution JSD ${value}.`);
        }
        count += 1;
        total += value;
        maximum = Math.max(maximum, value);
      }
    }
    summary[split] = {
      examples: count,
      meanResolutionJsd: total / Math.max(1, count),
      maximumResolutionJsd: maximum,
    };
  }
  return summary;
}

function visibleJensenShannonDivergence(
  fine: Float32Array,
  coarse: Float32Array,
  visibleIndexes: readonly number[],
): number {
  let fineTotal = 0;
  let coarseTotal = 0;
  for (const index of visibleIndexes) {
    fineTotal += fine[index]!;
    coarseTotal += coarse[index]!;
  }
  let result = 0;
  for (const index of visibleIndexes) {
    const p = fine[index]! / Math.max(Number.MIN_VALUE, fineTotal);
    const q = coarse[index]! / Math.max(Number.MIN_VALUE, coarseTotal);
    const mixture = 0.5 * (p + q);
    if (p > 0) result += 0.5 * p * (Math.log(p) - Math.log(mixture));
    if (q > 0) result += 0.5 * q * (Math.log(q) - Math.log(mixture));
  }
  return Math.max(0, Math.min(Math.log(2), result));
}

async function persistExampleTimeWeights(
  output: string,
  progress: Progress,
  samplingIntervalMs: number,
  weighting: TrainingPlan["training"]["timeWeighting"],
): Promise<Record<Split, {
  examples: number;
  meanAbsoluteDistanceImbalance: number;
  meanResolutionJsd: number;
  meanUnnormalizedWeight: number;
  maximumUnnormalizedWeight: number;
  effectiveSampleRatio: number;
}>> {
  const summary = {} as Record<Split, {
    examples: number;
    meanAbsoluteDistanceImbalance: number;
    meanResolutionJsd: number;
    meanUnnormalizedWeight: number;
    maximumUnnormalizedWeight: number;
    effectiveSampleRatio: number;
  }>;
  const decayPerStep = 0.5 ** (1 / weighting.memoryHalfLifeSteps);
  const metricCache = new Map<string, Float32Array>();
  for (const split of SPLITS) {
    const shards = progress.shards
      .filter((shard) => shard.split === split && shard.count > 0)
      .sort((left, right) =>
        left.date.localeCompare(right.date) || left.segment - right.segment);
    let evidence = 0;
    let side = 0;
    let previousTime: number | undefined;
    let count = 0;
    let absoluteAdviceTotal = 0;
    let resolutionDivergenceTotal = 0;
    let weightTotal = 0;
    let weightSquareTotal = 0;
    let maximumWeight = 0;
    for (const shard of shards) {
      const resolutionDivergence = bufferFloat32(
        await fs.readFile(path.join(output, shard.resolutionDivergence)),
      );
      if (resolutionDivergence.length !== shard.count) {
        throw new Error(`Resolution divergence rows are misaligned for ${shard.date}.`);
      }
      let metrics = metricCache.get(shard.teacherMetrics);
      if (!metrics) {
        const metricBuffer = await fs.readFile(path.join(output, shard.teacherMetrics));
        if (metricBuffer.byteLength % (TEACHER_METRIC_COUNT * 4) !== 0) {
          throw new Error(`MLP oracle metric component ${shard.teacherMetrics} is misaligned.`);
        }
        metrics = bufferFloat32(metricBuffer);
        metricCache.set(shard.teacherMetrics, metrics);
        while (metricCache.size > 4) metricCache.delete(metricCache.keys().next().value!);
      }
      const baseWeights = new Float32Array(shard.count);
      const weights = new Float32Array(shard.count);
      for (let index = 0; index < shard.count; index += 1) {
        const time = shard.predictionTimeStart + index * samplingIntervalMs;
        const oracleRow = shard.oracleRowOffset + index * shard.oracleRowStride;
        const advice = metrics[oracleRow * TEACHER_METRIC_COUNT
          + DISTANCE_IMBALANCE_METRIC_INDEX]!;
        if (!Number.isFinite(advice) || Math.abs(advice) > 1.000_001) {
          throw new Error(
            `Invalid raw-oracle distance imbalance ${advice} at ${split}:${shard.date}:${time}.`,
          );
        }
        if (previousTime !== undefined) {
          const elapsed = time - previousTime;
          if (elapsed <= 0) {
            throw new Error(`${split} example timestamps must be globally strictly increasing.`);
          }
          const elapsedSteps = elapsed / samplingIntervalMs;
          if (elapsedSteps > weighting.resetAfterGapSteps) {
            evidence = 0;
            side = 0;
          } else {
            evidence *= decayPerStep ** Math.max(0, elapsedSteps - 1);
          }
        }
        const magnitude = Math.abs(advice);
        let multiplier = 1;
        if (magnitude >= weighting.minimumAdviceMagnitude) {
          const currentSide = advice > 0 ? 1 : -1;
          if (currentSide !== side) {
            evidence = 0;
            side = currentSide;
          }
          multiplier = Math.min(
            weighting.maximumMultiplier,
            1 + weighting.growthPerPriorAdvice * evidence,
          );
          evidence += 1;
        } else {
          evidence *= decayPerStep;
        }
        const exampleWeight = weighting.minimumWeight + magnitude * multiplier;
        const resolutionJsd = resolutionDivergence[index]!;
        baseWeights[index] = exampleWeight;
        const combinedWeight = exampleWeight * (
          1 + weighting.resolutionDivergenceMultiplier * resolutionJsd
        );
        weights[index] = combinedWeight;
        absoluteAdviceTotal += magnitude;
        resolutionDivergenceTotal += resolutionJsd;
        weightTotal += combinedWeight;
        weightSquareTotal += combinedWeight * combinedWeight;
        maximumWeight = Math.max(maximumWeight, combinedWeight);
        count += 1;
        previousTime = time;
      }
      await Promise.all([
        writeTypedArrayAtomic(
          path.join(output, shard.baseTimeWeights),
          Buffer.from(
            baseWeights.buffer,
            baseWeights.byteOffset,
            baseWeights.byteLength,
          ),
        ),
        writeTypedArrayAtomic(
          path.join(output, shard.timeWeights),
          Buffer.from(weights.buffer, weights.byteOffset, weights.byteLength),
        ),
      ]);
    }
    summary[split] = {
      examples: count,
      meanAbsoluteDistanceImbalance: absoluteAdviceTotal / Math.max(1, count),
      meanResolutionJsd: resolutionDivergenceTotal / Math.max(1, count),
      meanUnnormalizedWeight: weightTotal / Math.max(1, count),
      maximumUnnormalizedWeight: maximumWeight,
      effectiveSampleRatio: weightTotal * weightTotal
        / Math.max(Number.EPSILON, count * weightSquareTotal),
    };
    process.stdout.write(`${JSON.stringify({
      event: "dataset-example-weighting",
      split,
      ...summary[split],
      normalized: false,
    })}\n`);
  }
  return summary;
}

async function persistFrozenStudyWeights(
  repoRoot: string,
  output: string,
  progress: Progress,
  plan: TrainingPlan,
  selection: FrozenWeightedExampleSelection,
): Promise<Record<Split, {
  examples: number;
  meanAbsoluteDistanceImbalance: number;
  meanResolutionJsd: number;
  meanUnnormalizedWeight: number;
  maximumUnnormalizedWeight: number;
  effectiveSampleRatio: number;
}>> {
  const sourceRoot = path.resolve(repoRoot, selection.sourceDatasetDir);
  const source = JSON.parse(
    await fs.readFile(path.join(sourceRoot, "progress.json"), "utf8"),
  ) as Progress;
  const summary = {} as Record<Split, {
    examples: number;
    meanAbsoluteDistanceImbalance: number;
    meanResolutionJsd: number;
    meanUnnormalizedWeight: number;
    maximumUnnormalizedWeight: number;
    effectiveSampleRatio: number;
  }>;
  const sourceWeightCache = new Map<string, Float32Array>();
  const metricCache = new Map<string, Float32Array>();
  for (const split of SPLITS) {
    let count = 0;
    let absoluteAdviceTotal = 0;
    let resolutionDivergenceTotal = 0;
    let weightTotal = 0;
    let weightSquareTotal = 0;
    let maximumWeight = 0;
    for (const shard of progress.shards
      .filter((item) => item.split === split)
      .sort((left, right) =>
        left.date.localeCompare(right.date) || left.segment - right.segment)) {
      const sourceShard = source.shards.find((candidate) => {
        if (candidate.split !== split) return false;
        const candidateEnd = candidate.predictionTimeStart
          + (candidate.count - 1) * plan.samplingIntervalMs;
        const shardEnd = shard.predictionTimeStart
          + (shard.count - 1) * plan.samplingIntervalMs;
        return shard.predictionTimeStart >= candidate.predictionTimeStart
          && shardEnd <= candidateEnd;
      });
      if (!sourceShard) {
        throw new Error(
          `Frozen source weights do not cover ${split}:${shard.date}:`
          + `${shard.predictionTimeStart}.`,
        );
      }
      const sourceOffset = (
        shard.predictionTimeStart - sourceShard.predictionTimeStart
      ) / plan.samplingIntervalMs;
      if (!Number.isInteger(sourceOffset)) {
        throw new Error(`Frozen source weight offset is not integral for ${shard.date}.`);
      }
      const readSourceWeights = async (file: string): Promise<Float32Array> => {
        let values = sourceWeightCache.get(file);
        if (!values) {
          values = bufferFloat32(await fs.readFile(path.join(sourceRoot, file)));
          sourceWeightCache.set(file, values);
          while (sourceWeightCache.size > 8) {
            sourceWeightCache.delete(sourceWeightCache.keys().next().value!);
          }
        }
        return values;
      };
      const [sourceBase, sourceCombined] = await Promise.all([
        readSourceWeights(sourceShard.baseTimeWeights),
        readSourceWeights(sourceShard.timeWeights),
      ]);
      const baseWeights = sourceBase.slice(sourceOffset, sourceOffset + shard.count);
      const weights = sourceCombined.slice(sourceOffset, sourceOffset + shard.count);
      if (baseWeights.length !== shard.count || weights.length !== shard.count) {
        throw new Error(`Frozen source weight range is incomplete for ${shard.date}.`);
      }
      await Promise.all([
        writeTypedArrayAtomic(
          path.join(output, shard.baseTimeWeights),
          Buffer.from(baseWeights.buffer, baseWeights.byteOffset, baseWeights.byteLength),
        ),
        writeTypedArrayAtomic(
          path.join(output, shard.timeWeights),
          Buffer.from(weights.buffer, weights.byteOffset, weights.byteLength),
        ),
      ]);
      const divergence = bufferFloat32(
        await fs.readFile(path.join(output, shard.resolutionDivergence)),
      );
      let metrics = metricCache.get(shard.teacherMetrics);
      if (!metrics) {
        metrics = bufferFloat32(
          await fs.readFile(path.join(output, shard.teacherMetrics)),
        );
        metricCache.set(shard.teacherMetrics, metrics);
        while (metricCache.size > 4) metricCache.delete(metricCache.keys().next().value!);
      }
      for (let index = 0; index < shard.count; index += 1) {
        const oracleRow = shard.oracleRowOffset + index * shard.oracleRowStride;
        absoluteAdviceTotal += Math.abs(
          metrics[oracleRow * TEACHER_METRIC_COUNT + DISTANCE_IMBALANCE_METRIC_INDEX]!,
        );
        resolutionDivergenceTotal += divergence[index]!;
        const value = weights[index]!;
        weightTotal += value;
        weightSquareTotal += value * value;
        maximumWeight = Math.max(maximumWeight, value);
        count += 1;
      }
    }
    summary[split] = {
      examples: count,
      meanAbsoluteDistanceImbalance: absoluteAdviceTotal / Math.max(1, count),
      meanResolutionJsd: resolutionDivergenceTotal / Math.max(1, count),
      meanUnnormalizedWeight: weightTotal / Math.max(1, count),
      maximumUnnormalizedWeight: maximumWeight,
      effectiveSampleRatio: weightTotal * weightTotal
        / Math.max(Number.EPSILON, count * weightSquareTotal),
    };
    process.stdout.write(`${JSON.stringify({
      event: "dataset-frozen-example-weighting",
      split,
      ...summary[split],
      sourceDatasetDir: selection.sourceDatasetDir,
      normalized: false,
    })}\n`);
  }
  return summary;
}

interface WeightedStudyCandidate {
  date: string;
  split: Split;
  startMinute: number;
  endMinute: number;
  sourceMeanWeight: number;
  samplingKey: number;
}

async function freezeWeightedStudyPlan(
  repoRoot: string,
  sourceOutput: string,
  plan: TrainingPlan,
  sourceCounts: Record<Split, number>,
): Promise<string> {
  const config = plan.frozenStudySampling;
  if (!config) throw new Error("Frozen study sampling configuration is missing.");
  const blockDurationMs = config.blockExamples * plan.samplingIntervalMs;
  if (blockDurationMs % MINUTE_MS !== 0) {
    throw new Error("Frozen study blocks must cover a whole number of minutes.");
  }
  const blockMinutes = blockDurationMs / MINUTE_MS;
  const selected: WeightedStudyCandidate[] = [];
  for (const split of SPLITS) {
    const candidates: Omit<WeightedStudyCandidate, "samplingKey">[] = [];
    const shards = JSON.parse(
      await fs.readFile(path.join(sourceOutput, "progress.json"), "utf8"),
    ).shards as DatasetShard[];
    for (const shard of shards
      .filter((item) => item.split === split)
      .sort((left, right) =>
        left.date.localeCompare(right.date) || left.segment - right.segment)) {
      const weights = bufferFloat32(
        await fs.readFile(path.join(sourceOutput, shard.timeWeights)),
      );
      for (
        let localStart = 0;
        localStart + config.blockExamples <= shard.count;
        localStart += config.blockExamples
      ) {
        let total = 0;
        for (let index = localStart; index < localStart + config.blockExamples; index += 1) {
          total += weights[index]!;
        }
        const sourceMeanWeight = total / config.blockExamples;
        const featureRow = shard.featureRowOffset + localStart * shard.featureRowStride;
        const startMinute = Math.floor(featureRow * SECOND_MS / MINUTE_MS);
        candidates.push({
          date: shard.date,
          split,
          startMinute,
          endMinute: startMinute + blockMinutes,
          sourceMeanWeight,
        });
      }
    }
    const requestedBlocks = config.selectedCounts[split] / config.blockExamples;
    if (candidates.length < requestedBlocks) {
      throw new Error(
        `Frozen ${split} sample needs ${requestedBlocks} blocks but only `
        + `${candidates.length} non-overlapping source blocks are available.`,
      );
    }
    const random = seededRandom(config.seed ^ stableStringHash(split));
    const weighted = config.weightedSplits.includes(split);
    const ranked = candidates.map((candidate) => ({
      ...candidate,
      // Efraimidis-Spirakis weighted sampling without replacement. Using the
      // whole temporal block's mean keeps temporal objectives meaningful.
      samplingKey: Math.log(Math.max(Number.MIN_VALUE, random()))
        / (weighted ? candidate.sourceMeanWeight : 1),
    })).sort((left, right) => right.samplingKey - left.samplingKey);
    selected.push(...ranked.slice(0, requestedBlocks));
  }
  selected.sort((left, right) =>
    left.date.localeCompare(right.date)
    || left.startMinute - right.startMinute
    || left.split.localeCompare(right.split));
  const selectedCounts = Object.fromEntries(SPLITS.map((split) => [
    split,
    selected.filter((item) => item.split === split).length * config.blockExamples,
  ])) as Record<Split, number>;
  for (const split of SPLITS) {
    if (selectedCounts[split] !== config.selectedCounts[split]) {
      throw new Error(`Frozen ${split} sample count changed during selection.`);
    }
  }
  const generated = structuredClone(plan);
  generated.id = `${plan.id}-minute-oracle-weighted-study`;
  generated.label = `${plan.label ?? plan.id} · one-minute oracle · frozen production-wide sample`;
  generated.datasetDir = config.outputDatasetDir;
  generated.artifactDir = `${plan.artifactDir}-minute-oracle-weighted-study`;
  generated.runDir = `${plan.runDir}-minute-oracle-weighted-study`;
  generated.componentSeedDatasetDirs = [
    config.sourceDatasetDir,
    ...(plan.componentSeedDatasetDirs ?? []),
  ];
  generated.exampleSelection = {
    mode: "frozen-weighted-time-blocks",
    blocks: selected.map((item) => ({
      date: item.date,
      split: item.split,
      startMinute: item.startMinute,
      endMinute: item.endMinute,
      sourceMeanWeight: item.sourceMeanWeight,
    })),
    componentCoveragePredictionDelaysMs: [plan.predictionDelayMs],
    sourceDatasetDir: config.sourceDatasetDir,
    seed: config.seed,
    blockExamples: config.blockExamples,
    sourceCounts,
    selectedCounts,
    weightedSplits: config.weightedSplits,
  };
  generated.training = {
    ...generated.training,
    targetRepresentation: "minuteOracleProbabilities",
    trainingFraction: 1,
    validationFraction: 1,
    weightedTrainingSample: false,
    lossWeights: {
      crossEntropy: 1,
      probabilityMse: 1,
      excessEntropy: 0,
      temporalMutualInformation: 0,
      oracleMutualInformation: 0,
    },
  } as TrainingPlan["training"] & Record<string, unknown>;
  delete generated.frozenStudySampling;
  const outputPlan = path.resolve(repoRoot, config.outputPlanFile);
  await fs.mkdir(path.dirname(outputPlan), { recursive: true });
  await atomicWriteJson(outputPlan, generated);
  process.stdout.write(`${JSON.stringify({
    event: "frozen-study-selection",
    sourceCounts,
    selectedCounts,
    blockExamples: config.blockExamples,
    selectedBlocks: selected.length,
    weightedSplits: config.weightedSplits,
    outputPlan,
  })}\n`);
  return outputPlan;
}

async function writeProductionTrainingPlan(
  repoRoot: string,
  plan: TrainingPlan,
  sourceCounts: Record<Split, number>,
  splitAnchorDate: string,
): Promise<string> {
  const config = plan.productionTraining;
  if (!config) throw new Error("Production training configuration is missing.");
  const generated = structuredClone(plan);
  generated.id = `${plan.id}-full-minute-oracle`;
  generated.label = `${plan.label ?? plan.id} · full 33M production corpus`;
  generated.datasetDir = config.outputDatasetDir;
  generated.artifactDir = `${plan.artifactDir}-full-minute-oracle`;
  generated.runDir = `${plan.runDir}-full-minute-oracle`;
  generated.splitAnchorDate = splitAnchorDate;
  generated.componentSeedDatasetDirs = [
    config.sourceDatasetDir,
    ...(plan.componentSeedDatasetDirs ?? []),
  ];
  generated.componentCompression = { features: "zstd" };
  generated.featurePreparationWorkers = 3;
  generated.runtimeMinuteOracleTargets = true;
  generated.training = {
    ...generated.training,
    targetRepresentation: "minuteOracleProbabilities",
    trainingFraction: 1,
    validationFraction: 1,
    weightedTrainingSample: false,
    lossWeights: {
      crossEntropy: 1,
      probabilityMse: 1,
      excessEntropy: 0,
      temporalMutualInformation: 0,
      oracleMutualInformation: 0,
    },
  } as TrainingPlan["training"] & Record<string, unknown>;
  delete generated.exampleSelection;
  delete generated.frozenStudySampling;
  delete generated.productionTraining;
  const outputPlan = path.resolve(repoRoot, config.outputPlanFile);
  await fs.mkdir(path.dirname(outputPlan), { recursive: true });
  await atomicWriteJson(outputPlan, generated);
  process.stdout.write(`${JSON.stringify({
    event: "production-training-plan",
    sourceCounts,
    splitAnchorDate,
    outputPlan,
    compressedFeatures: true,
    runtimeMinuteOracleTargets: true,
  })}\n`);
  return outputPlan;
}

function stableStringHash(value: string): number {
  let hash = 2_166_136_261;
  for (const character of value) {
    hash ^= character.charCodeAt(0);
    hash = Math.imul(hash, 16_777_619);
  }
  return hash >>> 0;
}

function seededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ value >>> 15, value | 1);
    value ^= value + Math.imul(value ^ value >>> 7, value | 61);
    return ((value ^ value >>> 14) >>> 0) / 4_294_967_296;
  };
}

async function writeRawOracleProbabilitiesAtomic(
  file: string,
  probabilities: Float32Array,
  actionCount: number,
  indexes: readonly number[],
  componentRows?: readonly number[],
): Promise<void> {
  if (!Number.isInteger(actionCount) || actionCount < 1
    || probabilities.length % actionCount !== 0
    || indexes.some((index) => !Number.isInteger(index)
      || index < 0 || (index + 1) * actionCount > probabilities.length)) {
    throw new Error("Raw oracle grid rows do not match the CUDA oracle output.");
  }
  if (componentRows && componentRows.length !== indexes.length) {
    throw new Error("Raw oracle component rows do not match selected probability rows.");
  }
  if (componentRows && rowSelectionSignature(componentRows) !== "full") {
    const values = new Float32Array(indexes.length * actionCount);
    indexes.forEach((sourceIndex, destinationIndex) => {
      const sourceStart = sourceIndex * actionCount;
      values.set(
        probabilities.subarray(sourceStart, sourceStart + actionCount),
        destinationIndex * actionCount,
      );
    });
    await writeComponentRowsAtomic(
      file,
      Buffer.from(values.buffer, values.byteOffset, values.byteLength),
      actionCount * Float32Array.BYTES_PER_ELEMENT,
      componentRows,
    );
    return;
  }
  const temporary = `${file}.tmp`;
  const handle = await fs.open(temporary, "w");
  try {
    const rowsPerChunk = 4_096;
    for (let start = 0; start < indexes.length; start += rowsPerChunk) {
      const end = Math.min(indexes.length, start + rowsPerChunk);
      const values = new Float32Array((end - start) * actionCount);
      for (let destinationIndex = start; destinationIndex < end; destinationIndex += 1) {
        const sourceIndex = indexes[destinationIndex]!;
        const sourceStart = sourceIndex * actionCount;
        values.set(
          probabilities.subarray(sourceStart, sourceStart + actionCount),
          (destinationIndex - start) * actionCount,
        );
      }
      await handle.writeFile(Buffer.from(
        values.buffer,
        values.byteOffset,
        values.byteLength,
      ));
    }
  } catch (error) {
    await handle.close();
    await fs.rm(temporary, { force: true });
    throw error;
  }
  await handle.close();
  await fs.rename(temporary, file);
}

async function writeTimesAtomic(file: string, times: readonly number[]): Promise<void> {
  const buffer = Buffer.allocUnsafe(times.length * 8);
  for (let index = 0; index < times.length; index += 1) {
    buffer.writeBigInt64LE(BigInt(times[index]!), index * 8);
  }
  await writeTypedArrayAtomic(file, buffer);
}

async function writeTimesComponentAtomic(
  file: string,
  times: readonly number[],
  componentRows: readonly number[],
): Promise<void> {
  const buffer = Buffer.allocUnsafe(times.length * 8);
  for (let index = 0; index < times.length; index += 1) {
    buffer.writeBigInt64LE(BigInt(times[index]!), index * 8);
  }
  await writeComponentRowsAtomic(file, buffer, 8, componentRows);
}

async function writeFeatureRowsAtomic(
  file: string,
  rows: Buffer,
  componentRows: readonly number[],
  compression?: "zstd",
): Promise<void> {
  if (compression !== "zstd") {
    await writeComponentRowsAtomic(
      file,
      rows,
      MLP_INPUT_FEATURE_COUNT * 2,
      componentRows,
    );
    return;
  }
  if (rowSelectionSignature(componentRows) !== "full") {
    throw new Error("Zstandard feature components require complete UTC-day rows.");
  }
  const compressed = zstdCompressSync(rows, {
    params: {
      [zlibConstants.ZSTD_c_compressionLevel]: 3,
    },
  });
  await writeTypedArrayAtomic(file, compressed);
}

async function writeComponentRowsAtomic(
  file: string,
  rows: Buffer,
  rowBytes: number,
  componentRows: readonly number[],
): Promise<void> {
  const totalRows = DAY_MS / SECOND_MS;
  if (!Number.isInteger(rowBytes) || rowBytes < 1
    || rows.byteLength !== componentRows.length * rowBytes
    || componentRows.some((row, index) => !Number.isInteger(row)
      || row < 0 || row >= totalRows || (index > 0 && row <= componentRows[index - 1]!))) {
    throw new Error("Sparse dataset component rows are invalid or unsorted.");
  }
  if (rowSelectionSignature(componentRows) === "full") {
    await writeTypedArrayAtomic(file, rows);
    return;
  }
  const temporary = `${file}.tmp`;
  const handle = await fs.open(temporary, "w");
  try {
    await handle.truncate(totalRows * rowBytes);
    let selectedStart = 0;
    while (selectedStart < componentRows.length) {
      let selectedEnd = selectedStart + 1;
      while (selectedEnd < componentRows.length
        && componentRows[selectedEnd] === componentRows[selectedEnd - 1]! + 1) {
        selectedEnd += 1;
      }
      const sourceStart = selectedStart * rowBytes;
      const sourceEnd = selectedEnd * rowBytes;
      await handle.write(
        rows.subarray(sourceStart, sourceEnd),
        0,
        sourceEnd - sourceStart,
        componentRows[selectedStart]! * rowBytes,
      );
      selectedStart = selectedEnd;
    }
  } catch (error) {
    await handle.close();
    await fs.rm(temporary, { force: true });
    throw error;
  }
  await handle.close();
  await fs.rename(temporary, file);
}

async function writeTypedArrayAtomic(file: string, buffer: Buffer): Promise<void> {
  const temporary = `${file}.tmp`;
  await fs.writeFile(temporary, buffer);
  await fs.rename(temporary, file);
}

class OneSecondCandleStore {
  private readonly dataDir: string;
  private readonly root: string;
  private readonly days = new Map<number, Promise<Candle[]>>();

  constructor(dataDir: string) {
    this.dataDir = dataDir;
    this.root = path.join(dataDir, "historical", "spot-btcusdt", "btcusdt", "1s");
  }

  async loadRange(start: number, end: number): Promise<Candle[]> {
    const requested: Array<Promise<Candle[]>> = [];
    for (let day = utcDay(start); day < end; day += DAY_MS) requested.push(this.day(day));
    const result = (await Promise.all(requested)).flat().filter((candle) =>
      candle.openTime >= start && candle.openTime < end);
    result.sort((left, right) => left.openTime - right.openTime);
    return result;
  }

  private day(day: number): Promise<Candle[]> {
    const existing = this.days.get(day);
    if (existing) {
      this.days.delete(day);
      this.days.set(day, existing);
      return existing;
    }
    const pending = this.prepareDay(day);
    this.days.set(day, pending);
    pending.catch(() => {
      if (this.days.get(day) === pending) this.days.delete(day);
    });
    while (this.days.size > 4) this.days.delete(this.days.keys().next().value!);
    return pending;
  }

  private async prepareDay(day: number): Promise<Candle[]> {
    const date = isoDate(day);
    try {
      return await this.readDay(day);
    } catch (localError) {
      const started = performance.now();
      const localDetail = errorMessage(localError);
      process.stdout.write(`${JSON.stringify({
        event: "dataset-source-recovery-start",
        date,
        detail: localDetail,
      })}\n`);
      try {
        await fetchBinanceSpotDailyShard({ dataDir: this.dataDir, date, day });
        const recovered = await this.readDay(day);
        process.stdout.write(`${JSON.stringify({
          event: "dataset-source-recovery-complete",
          date,
          candles: recovered.length,
          elapsedMs: performance.now() - started,
        })}\n`);
        return recovered;
      } catch (recoveryError) {
        throw new Error(
          `${date}: local one-second source is unusable (${localDetail}); `
          + `automatic Binance recovery failed (${errorMessage(recoveryError)}).`,
        );
      }
    }
  }

  private async readDay(day: number): Promise<Candle[]> {
    const date = isoDate(day);
    const failures: string[] = [];
    for (const source of [
      { file: path.join(this.root, `${date}.jsonl`), compressed: false },
      { file: path.join(this.root, `${date}.jsonl.gz`), compressed: true },
    ]) {
      try {
        const bytes = await fs.readFile(source.file);
        const content = source.compressed
          ? gunzipSync(bytes).toString("utf8")
          : bytes.toString("utf8");
        const result: Candle[] = [];
        for (const line of content.split("\n")) {
          if (line) result.push(JSON.parse(line) as Candle);
        }
        const issue = inspectScoredDay(result, day, date, []);
        if (issue) throw new Error(issue.detail);
        return result;
      } catch (error) {
        failures.push(`${path.basename(source.file)}: ${errorMessage(error)}`);
      }
    }
    throw new Error(`${date}: no complete validated local one-second shard; ${failures.join("; ")}`);
  }
}

async function findLatestCompleteDay(root: string): Promise<number> {
  const dates = (await fs.readdir(root))
    .map((file) => /^(\d{4}-\d{2}-\d{2})\.jsonl(?:\.gz)?$/.exec(file)?.[1])
    .filter((value): value is string => Boolean(value))
    .sort();
  if (dates.length === 0) throw new Error("No cached one-second history is available for testing.");
  return parseDay(dates.at(-1)!);
}

async function loadProgress(file: string, planId: string, latestDay: number): Promise<Progress> {
  try {
    const value = JSON.parse(await fs.readFile(file, "utf8")) as Progress;
    if (value.version !== 6 || value.planId !== planId) {
      throw new Error("Existing dataset progress belongs to a different component store.");
    }
    const latestCompleteDay = isoDate(latestDay);
    if (value.latestCompleteDay !== latestCompleteDay) {
      process.stdout.write(`${JSON.stringify({
        event: "dataset-split-range-rebased",
        previousLatestCompleteDay: value.latestCompleteDay,
        latestCompleteDay,
        preservedFeatureComponents: value.featureComponents.length,
        preservedOracleComponents: value.oracleComponents.length,
        discardedPairingShards: value.shards.length,
      })}\n`);
      value.latestCompleteDay = latestCompleteDay;
      value.shards = [];
      value.updatedAt = new Date().toISOString();
    }
    return value;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    return {
      version: 6,
      planId,
      updatedAt: new Date().toISOString(),
      latestCompleteDay: isoDate(latestDay),
      featureComponents: [],
      oracleComponents: [],
      shards: [],
    };
  }
}

async function loadRefinementQueue(
  file: string,
  planId: string,
): Promise<TeacherRefinementQueue> {
  try {
    const value = JSON.parse(await fs.readFile(file, "utf8")) as TeacherRefinementQueue;
    if (value.version !== 2 || value.planId !== planId || !Array.isArray(value.cases)) {
      throw new Error("Existing teacher refinement queue belongs to a different training plan.");
    }
    return value;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    return {
      version: 2,
      planId,
      updatedAt: new Date().toISOString(),
      cases: [],
    };
  }
}

async function loadSourceRejectionQueue(
  file: string,
  planId: string,
): Promise<SourceRejectionQueue> {
  try {
    const value = JSON.parse(await fs.readFile(file, "utf8")) as SourceRejectionQueue;
    if (value.version !== 1 || value.planId !== planId || !Array.isArray(value.cases)) {
      throw new Error("Existing source rejection queue belongs to a different training plan.");
    }
    return value;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    return {
      version: 1,
      planId,
      updatedAt: new Date().toISOString(),
      cases: [],
    };
  }
}

async function writeSourceRejectionQueue(
  file: string,
  planId: string,
  cases: ReadonlyMap<string, SourceDayRejection>,
): Promise<void> {
  await atomicWriteJson(file, {
    version: 1,
    planId,
    updatedAt: new Date().toISOString(),
    cases: [...cases.values()].sort((left, right) => left.date.localeCompare(right.date)),
  } satisfies SourceRejectionQueue);
}

async function recordSourceRejection(
  rejection: SourceDayRejection,
  cases: Map<string, SourceDayRejection>,
  file: string,
  planId: string,
  dayIndex: number,
  dayCount: number,
): Promise<void> {
  cases.set(rejection.date, rejection);
  await writeSourceRejectionQueue(file, planId, cases);
  process.stdout.write(`${JSON.stringify({
    event: "dataset-source-rejected",
    ...rejection,
    day: dayIndex + 1,
    days: dayCount,
    rejectedDays: cases.size,
  })}\n`);
}

function refinementCaseKey(date: string, time: number): string {
  return `${date}:${time}`;
}

function inspectScoredDay(
  candles: readonly Candle[],
  day: number,
  date: string,
  splits: Split[],
): SourceDayRejection | undefined {
  return inspectContinuousSourceRange(
    candles,
    day,
    DAY_MS / SECOND_MS,
    date,
    splits,
    "scored UTC day",
  );
}

function inspectContinuousSourceRange(
  candles: readonly Candle[],
  start: number,
  expectedCandles: number,
  date: string,
  splits: Split[],
  scope: string,
): SourceDayRejection | undefined {
  let firstUnexpectedIndex: number | null = null;
  let expectedOpenTime: number | null = null;
  let observedOpenTime: number | null = null;
  let mismatch: string | null = null;
  const comparedCount = Math.min(candles.length, expectedCandles);
  for (let index = 0; index < comparedCount; index += 1) {
    const expected = start + index * SECOND_MS;
    const candle = candles[index]!;
    if (candle.openTime !== expected) {
      firstUnexpectedIndex = index;
      expectedOpenTime = expected;
      observedOpenTime = candle.openTime;
      mismatch = `first timestamp mismatch at row ${index}`;
      break;
    }
    const invalid = invalidOneSecondCandle(candle);
    if (invalid) {
      firstUnexpectedIndex = index;
      expectedOpenTime = expected;
      observedOpenTime = candle.openTime;
      mismatch = `invalid candle at row ${index}: ${invalid}`;
      break;
    }
  }
  if (firstUnexpectedIndex === null && candles.length < expectedCandles) {
    firstUnexpectedIndex = candles.length;
    expectedOpenTime = start + candles.length * SECOND_MS;
    mismatch = `first timestamp missing at row ${candles.length}`;
  }
  if (candles.length === expectedCandles && firstUnexpectedIndex === null) return undefined;
  mismatch ??= "timestamps extend beyond the expected UTC day";
  return {
    date,
    splits,
    reason: "incomplete-or-noncontiguous-day",
    detail: `${date} ${scope} has ${candles.length}/${expectedCandles} one-second candles; ${mismatch}.`,
    expectedCandles,
    observedCandles: candles.length,
    firstOpenTime: candles[0]?.openTime ?? null,
    lastOpenTime: candles.at(-1)?.openTime ?? null,
    firstUnexpectedIndex,
    expectedOpenTime,
    observedOpenTime,
    updatedAt: new Date().toISOString(),
  };
}

function invalidOneSecondCandle(candle: Candle): string | undefined {
  if (!Number.isSafeInteger(candle.closeTime)
    || candle.closeTime < candle.openTime
    || candle.closeTime >= candle.openTime + SECOND_MS) return "invalid close time";
  if (![candle.open, candle.high, candle.low, candle.close, candle.volume].every(Number.isFinite)) {
    return "non-finite OHLCV value";
  }
  if (candle.volume < 0) return "negative volume";
  if (candle.high < Math.max(candle.open, candle.low, candle.close)) return "high is below OHLC values";
  if (candle.low > Math.min(candle.open, candle.high, candle.close)) return "low is above OHLC values";
  if (!candle.closed) return "candle is not closed";
  return undefined;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function assertSameGrid(expected: readonly number[], observed: ArrayLike<number>, name: string): void {
  if (expected.length !== observed.length
    || expected.some((value, index) => value !== observed[index])) {
    throw new Error(`MLP dataset ${name} grid changed between shards.`);
  }
}

function parseArguments(values: string[]): {
  planFile: string;
  outputOverride?: string;
  refinementPass: number;
  preparationMode: "dataset" | "frozen-study-source";
} {
  const options = new Map<string, string>();
  for (let index = 0; index < values.length; index += 2) {
    const key = values[index];
    const value = values[index + 1];
    if (!key?.startsWith("--") || value === undefined) throw new Error(`Invalid argument near ${key ?? "end"}.`);
    options.set(key.slice(2), value);
  }
  const planFile = path.resolve(options.get("plan") ?? "ml/training-plan.json");
  const refinementPass = Number(options.get("refinement-pass") ?? 0);
  if (!Number.isInteger(refinementPass) || refinementPass < 0) {
    throw new Error("--refinement-pass must be a non-negative integer.");
  }
  const preparationMode = options.get("preparation-mode") ?? "dataset";
  if (preparationMode !== "dataset" && preparationMode !== "frozen-study-source") {
    throw new Error("--preparation-mode must be dataset or frozen-study-source.");
  }
  return {
    planFile,
    outputOverride: options.get("output"),
    refinementPass,
    preparationMode,
  };
}

function validatePlan(plan: TrainingPlan): void {
  const execution = plan.execution;
  const timeWeighting = plan.training?.timeWeighting;
  const selection = plan.exampleSelection;
  const validCoverage = !selection || (
    Array.isArray(selection.componentCoveragePredictionDelaysMs)
    && selection.componentCoveragePredictionDelaysMs.length > 0
    && selection.componentCoveragePredictionDelaysMs.includes(plan.predictionDelayMs)
    && selection.componentCoveragePredictionDelaysMs.every((delay) =>
      Number.isInteger(delay / SECOND_MS) && delay >= 0)
  );
  const validSelection = !selection || validCoverage && (
    selection.mode === "explicit-time-blocks"
      ? Array.isArray(selection.days) && selection.days.length > 0
        && new Set(selection.days.map((item) => item.date)).size === selection.days.length
        && selection.days.every((item) => Number.isFinite(parseDay(item.date))
          && SPLITS.includes(item.split))
        && Array.isArray(selection.utcBlocks) && selection.utcBlocks.length > 0
        && selection.utcBlocks.every(([start, end]) => Number.isInteger(start)
          && Number.isInteger(end) && start >= 0 && start < end && end <= 1_440)
        && selection.utcBlocks.every((block, index) => index === 0
          || block[0] >= selection.utcBlocks[index - 1]![1])
      : selection.mode === "frozen-weighted-time-blocks"
        ? Array.isArray(selection.blocks) && selection.blocks.length > 0
        && typeof selection.sourceDatasetDir === "string"
        && Number.isInteger(selection.seed)
        && Number.isInteger(selection.blockExamples) && selection.blockExamples > 0
        && selection.blocks.every((block) =>
          Number.isFinite(parseDay(block.date))
          && SPLITS.includes(block.split)
          && Number.isInteger(block.startMinute)
          && Number.isInteger(block.endMinute)
          && block.startMinute >= 0
          && block.startMinute < block.endMinute
          && block.endMinute <= 1_440
          && Number.isFinite(block.sourceMeanWeight)
          && block.sourceMeanWeight > 0)
        : false
  );
  const frozenStudy = plan.frozenStudySampling;
  const validFrozenStudy = !frozenStudy || (
    typeof frozenStudy.sourceDatasetDir === "string"
    && typeof frozenStudy.outputDatasetDir === "string"
    && typeof frozenStudy.outputPlanFile === "string"
    && Number.isInteger(frozenStudy.blockExamples)
    && frozenStudy.blockExamples >= 1
    && Number.isInteger(frozenStudy.seed)
    && SPLITS.every((split) =>
      Number.isInteger(frozenStudy.selectedCounts[split])
      && frozenStudy.selectedCounts[split] >= frozenStudy.blockExamples
      && frozenStudy.selectedCounts[split] % frozenStudy.blockExamples === 0)
    && Array.isArray(frozenStudy.weightedSplits)
    && frozenStudy.weightedSplits.every((split) => SPLITS.includes(split))
  );
  if (plan.version !== 4 || !plan.id || !plan.componentStoreId
    || !plan.datasetDir || !Array.isArray(plan.windows)
    || plan.oraclePreparation?.backend !== "cuda"
    || !Number.isInteger(plan.samplingIntervalMs / SECOND_MS)
    || plan.samplingIntervalMs < SECOND_MS
    || !Number.isInteger(plan.predictionDelayMs / SECOND_MS)
    || plan.predictionDelayMs < 0
    || !Number.isInteger(plan.latestTestDays) || plan.latestTestDays < 1
    || plan.featurePreparationWorkers !== undefined
      && (!Number.isInteger(plan.featurePreparationWorkers)
        || plan.featurePreparationWorkers < 1)
    || !["fitted-parameters", "direct-diagnostics"].includes(
      plan.teacherFit.representation ?? "fitted-parameters",
    )
    || plan.teacherFit.backend !== "cuda" || plan.teacherFit.device !== "cuda"
    || !Number.isInteger(plan.teacherFit.batchSize) || plan.teacherFit.batchSize < 1
    || !Number.isInteger(plan.teacherFit.projectionIterations)
    || plan.teacherFit.projectionIterations < 1
    || !Number.isInteger(plan.teacherFit.maxIterations) || plan.teacherFit.maxIterations < 1
    || !Number.isInteger(plan.teacherFit.adaptiveIterations)
    || plan.teacherFit.adaptiveIterations < 1
    || !Number.isInteger(plan.teacherFit.adaptiveRounds) || plan.teacherFit.adaptiveRounds < 0
    || !Number.isInteger(plan.teacherFit.restartCount) || plan.teacherFit.restartCount < 1
    || !isCenteredPowerOfTwoGrid(execution.gridSize)
    || !isCenteredPowerOfTwoGrid(plan.teacherFit.sampleStates)
    || !isCenteredPowerOfTwoGrid(plan.teacherFit.sampleActions)
    || plan.teacherFit.sampleStates > execution.gridSize
    || plan.teacherFit.sampleActions > execution.gridSize
    || !(plan.teacherFit.maxMeanKlDivergence > 0)
    || !(plan.teacherFit.maxMeanSquaredError > 0)
    || !Number.isInteger(plan.teacherFit.lineSearchCandidates)
    || plan.teacherFit.lineSearchCandidates < 1
    || !Number.isInteger(plan.teacherFit.temporalRefinementRounds)
    || plan.teacherFit.temporalRefinementRounds < 1
    || !Number.isInteger(plan.teacherFit.temporalIterations)
    || plan.teacherFit.temporalIterations < 1
    || !Number.isInteger(plan.teacherFit.temporalFollowupIterations)
    || plan.teacherFit.temporalFollowupIterations < 1
    || !(plan.teacherFit.temporalEquivalentLossAbsolute >= 0)
    || !(plan.teacherFit.temporalEquivalentLossRelative >= 0)
    || !["pytorch-batched", "triton-queued"].includes(plan.teacherFit.optimizerBackend)
    || !Number.isInteger(plan.teacherFit.optimizerHostCheckInterval)
    || plan.teacherFit.optimizerHostCheckInterval < 1
    || !Number.isInteger(plan.teacherFit.qualityFallbackIterations)
    || plan.teacherFit.qualityFallbackIterations < 0
    || !(plan.teacherFit.visibleSampleFraction >= 0
      && plan.teacherFit.visibleSampleFraction <= 1)
    || !(plan.teacherFit.scoreHingeSpan > 0)
    || typeof plan.teacherFit.compactVisibleInitialization !== "boolean"
    || !Number.isInteger(plan.teacherFit.inputAlignmentFloats)
    || plan.teacherFit.inputAlignmentFloats < 1
    || !Number.isInteger(plan.teacherFit.inputQueueBatches)
    || plan.teacherFit.inputQueueBatches < 1
    || typeof plan.teacherFit.pipelinedRefinement !== "boolean"
    || !Number.isInteger(plan.teacherFit.workerJobsBeforeRecycle)
    || plan.teacherFit.workerJobsBeforeRecycle < 1
    || !timeWeighting
    || timeWeighting.mode !== "distanceImbalance"
    || timeWeighting.stateAggregation !== "globalDistanceRatio"
    || !(timeWeighting.distanceEpsilon >= 0)
    || !(timeWeighting.minimumWeight > 0)
    || !(timeWeighting.minimumAdviceMagnitude >= 0
      && timeWeighting.minimumAdviceMagnitude <= 1)
    || !(timeWeighting.memoryHalfLifeSteps > 0)
    || !(timeWeighting.growthPerPriorAdvice >= 0)
    || !(timeWeighting.maximumMultiplier >= 1)
    || !(timeWeighting.resetAfterGapSteps >= 1)
    || !(timeWeighting.resolutionDivergenceMultiplier >= 0)
    || !(execution.minimumEffectiveExposure <= execution.minimumUsableExposure)
    || !(execution.maximumEffectiveExposure >= execution.maximumUsableExposure)
    || !(execution.minimumUsableExposure < execution.maximumUsableExposure)
    || execution.horizonEndMode !== "extend"
    || !Number.isInteger(execution.holdingPeriodSteps) || execution.holdingPeriodSteps < 1
    || !Number.isInteger(execution.valueHorizonSteps)
    || execution.valueHorizonSteps < execution.holdingPeriodSteps
    || !(execution.temperature > 0)
    || !validSelection
    || !validFrozenStudy) {
    throw new Error("Invalid MLP v4 training plan.");
  }
}

function isCenteredPowerOfTwoGrid(count: number): boolean {
  return Number.isInteger(count) && count >= 3 && (count & (count + 1)) === 0;
}

function runtimePositiveInteger(value: string | undefined, fallback: number): number {
  if (value === undefined) return fallback;
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 1) {
    throw new Error("TRADING_MLP_WORKER_JOBS_BEFORE_RECYCLE must be a positive integer.");
  }
  return parsed;
}

const FLOAT32_TO_FLOAT16_SCRATCH = new Float32Array(1);
const FLOAT32_TO_FLOAT16_BITS = new Uint32Array(FLOAT32_TO_FLOAT16_SCRATCH.buffer);

function float32ToFloat16(value: number): number {
  FLOAT32_TO_FLOAT16_SCRATCH[0] = value;
  const bits = FLOAT32_TO_FLOAT16_BITS[0]!;
  const sign = bits >>> 16 & 0x8000;
  let exponent = (bits >>> 23 & 0xff) - 127 + 15;
  let mantissa = bits & 0x7fffff;
  if (exponent <= 0) {
    if (exponent < -10) return sign;
    mantissa = (mantissa | 0x800000) >>> (1 - exponent);
    return sign | (mantissa + 0x1000 >>> 13);
  }
  if (exponent >= 31) return sign | 0x7c00;
  mantissa += 0x1000;
  if (mantissa & 0x800000) {
    mantissa = 0;
    exponent += 1;
  }
  return exponent >= 31 ? sign | 0x7c00 : sign | exponent << 10 | mantissa >>> 13;
}

function bpsHourToPerSecond(bps: number): number {
  return Math.expm1(Math.log1p(bps / 10_000) / 3_600);
}

function bpsHourToPerSteps(bps: number, stepsPerHour: number): number {
  return Math.expm1(Math.log1p(bps / 10_000) / stepsPerHour);
}

async function atomicWriteJson(file: string, value: unknown): Promise<void> {
  const temporary = `${file}.tmp`;
  await fs.writeFile(temporary, `${JSON.stringify(value, null, 2)}\n`);
  await fs.rename(temporary, file);
}

function parseDay(value: string): number {
  const time = Date.parse(`${value}T00:00:00.000Z`);
  if (!Number.isFinite(time)) throw new Error(`Invalid UTC date '${value}'.`);
  return time;
}

function isoDate(time: number): string {
  return new Date(time).toISOString().slice(0, 10);
}

function utcDay(time: number): number {
  const date = new Date(time);
  return Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate());
}
