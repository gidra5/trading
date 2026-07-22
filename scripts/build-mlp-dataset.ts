import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import readline from "node:readline";
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { fileURLToPath } from "node:url";
import { gunzipSync } from "node:zlib";
import {
  conditionalCutoffRawParameters,
  exposureHoldingFeasibleInterval,
  MLP_FEATURE_SCHEMA_VERSION,
  MLP_INPUT_FEATURE_COUNT,
  prepareExposureValueOracleCuda,
  type Candle,
  type MlpExposureStateInputs,
} from "@trading/bot-algo";
import { MlpFeatureStore } from "../apps/server/src/mlp-feature-store.js";

const DAY_MS = 86_400_000;
const SECOND_MS = 1_000;
const TEACHER_PARAMETER_COUNT = 8;
const TEACHER_METRIC_COUNT = 6;
const SPLITS = ["train", "validation", "test"] as const;
type Split = typeof SPLITS[number];

interface TrainingWindow { id: string; start: string; end: string }
interface TrainingPlan {
  version: number;
  id: string;
  dataDir: string;
  datasetDir: string;
  oraclePreparation: { backend: "cuda" };
  samplingIntervalMs: number;
  latestTestDays: number;
  excludedAggregateWindows: string[];
  windows: TrainingWindow[];
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
  };
  teacherFit: {
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
    inputAlignmentFloats: number;
    inputQueueBatches: number;
    pipelinedRefinement: boolean;
  };
}

interface TimeRange { start: number; end: number; id: string }
interface DatasetShard {
  split: Split;
  date: string;
  count: number;
  teacherBackend: "cpu-bfgs" | "cuda";
  features: string;
  teacherParameters: string;
  teacherMetrics: string;
  times: string;
  rejectedCount?: number;
  refinementPass?: number;
  featureSchemaVersion?: number;
  teacherMetricVisibleLower?: number;
  teacherMetricVisibleUpper?: number;
}

interface Progress {
  version: 3;
  planId: string;
  updatedAt: string;
  latestCompleteDay: string;
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
  split: Split;
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
  version: 1;
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
  temporalMeanNormalizedStepBefore: number;
  temporalMeanNormalizedStepAfter: number;
  pipelinedRefinement: boolean;
  pipelineWaitFraction: number;
}

async function main(): Promise<void> {
  const { planFile, outputOverride, refinementPass, refreshFeatures } = parseArguments(process.argv.slice(2));
  const plan = JSON.parse(await fs.readFile(planFile, "utf8")) as TrainingPlan;
  validatePlan(plan);
  const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
  const dataDir = path.resolve(repoRoot, plan.dataDir);
  const output = path.resolve(repoRoot, outputOverride ?? plan.datasetDir);
  const oneSecondRoot = path.join(dataDir, "historical", "spot-btcusdt", "btcusdt", "1s");
  const oneSecondStore = new OneSecondCandleStore(dataDir);
  const latestCompleteDay = await findLatestCompleteDay(oneSecondRoot);
  const testEnd = latestCompleteDay + DAY_MS;
  const testStart = testEnd - plan.latestTestDays * DAY_MS;
  const { trainRanges, validationRanges } = buildWindowRanges(plan);
  const selectedDays = selectedUtcDays(trainRanges, validationRanges, testStart, testEnd);

  await fs.mkdir(path.join(output, "shards"), { recursive: true });
  const progressFile = path.join(output, "progress.json");
  const progress = await loadProgress(progressFile, plan.id, latestCompleteDay);
  const refinementQueueFile = path.join(output, "teacher-refinement-queue.json");
  const refinementQueue = await loadRefinementQueue(refinementQueueFile, plan.id);
  const refinementCases = new Map(refinementQueue.cases.map((item) => [
    refinementCaseKey(item.split, item.date, item.time), item,
  ]));
  const sourceRejectionQueueFile = path.join(output, "source-rejection-queue.json");
  const sourceRejectionQueue = await loadSourceRejectionQueue(sourceRejectionQueueFile, plan.id);
  const sourceRejectionCases = new Map(sourceRejectionQueue.cases.map((item) => [item.date, item]));
  let existingDataset: {
    version?: number;
    planId?: string;
    counts?: Record<Split, number>;
    grid?: number[];
    currentGrid?: number[];
    featureSchemaVersion?: number;
    featureCount?: number;
    [key: string]: unknown;
  } | undefined;
  try {
    existingDataset = JSON.parse(await fs.readFile(path.join(output, "dataset.json"), "utf8"));
    if (existingDataset.version === 4 && existingDataset.planId === plan.id
      && SPLITS.every((split) => (existingDataset!.counts?.[split] ?? 0) > 0)
      && refinementPass === 0
      && existingDataset.featureSchemaVersion === MLP_FEATURE_SCHEMA_VERSION
      && existingDataset.featureCount === MLP_INPUT_FEATURE_COUNT) {
      process.stdout.write(`${JSON.stringify({
        event: "dataset-complete",
        output,
        counts: existingDataset.counts,
        resumed: true,
        featureSchemaVersion: MLP_FEATURE_SCHEMA_VERSION,
      })}\n`);
      return;
    }
    if (existingDataset.version !== 4 || existingDataset.planId !== plan.id) {
      throw new Error("Existing dataset manifest is incomplete or belongs to another plan.");
    }
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    if (refinementPass > 0) {
      throw new Error("Teacher refinement requires an existing complete dataset manifest.");
    }
  }
  if (refreshFeatures && !existingDataset) {
    throw new Error("Feature refresh requires an existing complete dataset manifest.");
  }
  if (existingDataset && refinementPass === 0) {
    await refreshDatasetFeatures({
      dataDir,
      output,
      plan,
      progress,
      dataset: existingDataset,
    });
    return;
  }
  const pendingRefinementGroups = new Set(
    [...refinementCases.values()]
      .map((item) => `${item.split}:${item.date}`),
  );
  const featureStore = new MlpFeatureStore(dataDir);
  let teacherFitter: GpuTeacherFitter | undefined;
  let prefetchedSource: { day: number; promise: Promise<Candle[]> } | undefined;
  let expectedGrid: number[] | undefined = existingDataset?.grid;
  let expectedCurrentGrid: number[] | undefined = existingDataset?.currentGrid;

  process.stdout.write(`${JSON.stringify({
    event: "dataset-start",
    planId: plan.id,
    output,
    days: selectedDays.length,
    completedShards: progress.shards.length,
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
      const rowSplits = sampledRowsForDay(
        day,
        plan.samplingIntervalMs,
        trainRanges,
        validationRanges,
        testStart,
        testEnd,
      );
      const queuedSource = sourceRejectionCases.get(date);
      const neededSplits = SPLITS.filter((split) => {
        if (!rowSplits.some((value) => value.split === split)) return false;
        const key = `${split}:${date}`;
        const shard = progress.shards.find((value) => value.split === split && value.date === date);
        const metricScopeChanged = shard?.teacherMetricVisibleLower
            !== plan.execution.minimumUsableExposure
          || shard?.teacherMetricVisibleUpper
            !== plan.execution.maximumUsableExposure;
        if (refinementPass === 0) {
          const expectedCount = rowSplits.filter((value) => value.split === split).length;
          return shard?.count !== expectedCount || metricScopeChanged;
        }
        const needsRefinement = pendingRefinementGroups.has(key)
          || queuedSource?.splits.includes(split)
          || (shard?.rejectedCount ?? 0) > 0;
        return metricScopeChanged
          || (needsRefinement && (shard?.refinementPass ?? 0) < refinementPass);
      });
      if (neededSplits.length === 0) continue;

      let source: Candle[];
      try {
        const sourcePromise = prefetchedSource?.day === day
          ? prefetchedSource.promise
          : oneSecondStore.loadRange(day - DAY_MS, day + DAY_MS);
        prefetchedSource = undefined;
        source = await sourcePromise;
        const nextDay = selectedDays[dayIndex + 1];
        if (nextDay !== undefined) {
          const promise = oneSecondStore.loadRange(nextDay - DAY_MS, nextDay + DAY_MS);
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
        }, sourceRejectionCases, sourceRejectionQueueFile, plan.id, dayIndex, selectedDays.length);
        continue;
      }
      const scored = source.filter((candle) => candle.openTime >= day && candle.openTime < day + DAY_MS);
      const sourceIssue = inspectScoredDay(scored, day, date, neededSplits);
      if (sourceIssue) {
        await recordSourceRejection(
          sourceIssue,
          sourceRejectionCases,
          sourceRejectionQueueFile,
          plan.id,
          dayIndex,
          selectedDays.length,
        );
        continue;
      }
      const sourceRecovered = sourceRejectionCases.has(date);
      const oracleCandles = source.filter((candle) =>
        candle.openTime >= day
        && candle.openTime < day + DAY_MS);
      const feeRate = plan.execution.feeBps / 10_000;
      const oracleStarted = performance.now();
      const oraclePrices = Float64Array.from(oracleCandles, (candle) => candle.close);
      const preparedOracle = await prepareExposureValueOracleCuda(
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
      const oracle = preparedOracle.oracle;
      process.stdout.write(`${JSON.stringify({
        event: "dataset-oracle",
        date,
        day: dayIndex + 1,
        days: selectedDays.length,
        backend: "cuda",
        kernelMs: preparedOracle.kernelMs,
        wallMs: performance.now() - oracleStarted,
        candles: oracleCandles.length,
      })}\n`);
      if (!oracle.probabilities) throw new Error("Oracle did not retain teacher probabilities.");
      expectedGrid ??= Array.from(oracle.grid);
      expectedCurrentGrid ??= Array.from(oracle.currentGrid);
      assertSameGrid(expectedGrid, oracle.grid, "action");
      assertSameGrid(expectedCurrentGrid, oracle.currentGrid, "current-exposure");
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
        } : plan.teacherFit,
      });
      const state: MlpExposureStateInputs = {
        feeRate,
        minimumUsableExposure: plan.execution.minimumUsableExposure,
        maximumUsableExposure: plan.execution.maximumUsableExposure,
        minimumEffectiveExposure: plan.execution.minimumEffectiveExposure,
        maximumEffectiveExposure: plan.execution.maximumEffectiveExposure,
        quoteLendRate: plan.execution.maintenanceBpsHour.quoteLend / 10_000,
        quoteBorrowRate: plan.execution.maintenanceBpsHour.quoteBorrow / 10_000,
        assetBorrowRate: plan.execution.maintenanceBpsHour.assetBorrow / 10_000,
      };

      for (const split of neededSplits) {
        const rows = rowSplits.filter((value) => value.split === split);
        const times = rows.map(({ candleIndex }) => scored[candleIndex]!.closeTime);
        const features = await featureStore.prepare(source, times, state);
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
        const teacher = await teacherFitter.fit(teacherInputs, (event) => {
          const done = event.examplesCompleted;
          if (done === 1 || done % plan.teacherFit.batchSize === 0 || done === rows.length) {
            process.stdout.write(`${JSON.stringify({
              event: "dataset-progress",
              date,
              split,
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
              temporalMeanNormalizedStepBefore: event.temporalMeanNormalizedStepBefore,
              temporalMeanNormalizedStepAfter: event.temporalMeanNormalizedStepAfter,
              pipelinedRefinement: event.pipelinedRefinement,
              pipelineWaitFraction: event.pipelineWaitFraction,
            })}\n`);
          }
        }, () => {
          const started = performance.now();
          encodedFeatureRows = encodeFeatureRows(times, features);
          featureEncodingMs = performance.now() - started;
        });
        if (!encodedFeatureRows) throw new Error("MLP feature encoding did not run with the CUDA fit.");
        const teacherWallMs = performance.now() - teacherStarted;
        if (refinementPass > 0) {
          for (const [key, item] of refinementCases) {
            if (item.split === split && item.date === date) refinementCases.delete(key);
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
            split,
            date,
            time: times[index]!,
            crossEntropy: crossEntropy!,
            klDivergence: klDivergence!,
            meanSquaredError: meanSquaredError!,
            iterations: iterations!,
            restarts: restarts!,
            converged: converged! > 0,
            ...(refinementPass > 0 ? { refinementPass } : {}),
          };
          refinementCases.set(refinementCaseKey(split, date, item.time), item);
        });
        // Rejections are a refinement signal, not a reason to remove a market
        // regime from MLP training. Persist the best available fit for every
        // timestamp and keep the strict quality failures in the resumable queue.
        const persistedIndexes = rows.map((_, index) => index);
        const prefix = path.join(
          "shards",
          `${split}-${date}${refinementPass > 0 ? `.refined-${refinementPass}` : ""}`,
        );
        const shard: DatasetShard = {
          split,
          date,
          count: persistedIndexes.length,
          teacherBackend: "cuda",
          features: `${prefix}.features.f16`,
          teacherParameters: `${prefix}.teacher-parameters.f32`,
          teacherMetrics: `${prefix}.teacher-metrics.f32`,
          times: `${prefix}.times.i64`,
          featureSchemaVersion: MLP_FEATURE_SCHEMA_VERSION,
          teacherMetricVisibleLower: plan.execution.minimumUsableExposure,
          teacherMetricVisibleUpper: plan.execution.maximumUsableExposure,
          ...(rejectedCount > 0 ? { rejectedCount } : {}),
          ...(refinementPass > 0 ? { refinementPass } : {}),
        };
        const persistStarted = performance.now();
        await Promise.all([
          writeSelectedFeatureRowsAtomic(
            path.join(output, shard.features),
            encodedFeatureRows,
            persistedIndexes,
          ),
          writeTeacherParametersAtomic(path.join(output, shard.teacherParameters), teacher),
          writeTeacherMetricsAtomic(path.join(output, shard.teacherMetrics), teacher),
          writeTimesAtomic(path.join(output, shard.times), times),
          rejectedCount > 0 || refinementPass > 0
            ? atomicWriteJson(refinementQueueFile, {
                version: 1,
                planId: plan.id,
                updatedAt: new Date().toISOString(),
                cases: [...refinementCases.values()].sort((left, right) =>
                  left.time - right.time || left.split.localeCompare(right.split)),
              } satisfies TeacherRefinementQueue)
            : Promise.resolve(),
        ]);
        const existingShardIndex = progress.shards.findIndex((item) =>
          item.split === split && item.date === date);
        if (existingShardIndex >= 0) progress.shards[existingShardIndex] = shard;
        else progress.shards.push(shard);
        progress.updatedAt = new Date().toISOString();
        await atomicWriteJson(progressFile, progress);
        process.stdout.write(`${JSON.stringify({ event: "dataset-shard", ...shard })}\n`);
        process.stdout.write(`${JSON.stringify({
          event: "dataset-stage-timing",
          date,
          split,
          examples: rows.length,
          acceptedExamples: acceptedIndexes.length,
          teacherWallMs,
          featureEncodingMs,
          featureEncodingOverlapped: true,
          persistMs: performance.now() - persistStarted,
        })}\n`);
      }
      if (sourceRecovered) {
        sourceRejectionCases.delete(date);
        await writeSourceRejectionQueue(sourceRejectionQueueFile, plan.id, sourceRejectionCases);
        process.stdout.write(`${JSON.stringify({
          event: "dataset-source-recovered",
          date,
          day: dayIndex + 1,
          days: selectedDays.length,
          remainingRejectedDays: sourceRejectionCases.size,
        })}\n`);
      }
    }
  } finally {
    await teacherFitter?.close();
    teacherFitter = undefined;
  }

  if (!expectedGrid || !expectedCurrentGrid || progress.shards.length === 0) {
    throw new Error("No MLP dataset shards were produced.");
  }
  const counts = Object.fromEntries(SPLITS.map((split) => [
    split,
    progress.shards.filter((shard) => shard.split === split)
      .reduce((sum, shard) => sum + shard.count, 0),
  ]));
  if (SPLITS.some((split) => counts[split] === 0)) {
    throw new Error(`Dataset split is empty: ${JSON.stringify(counts)}`);
  }
  const manifest = {
    version: 4,
    createdAt: new Date().toISOString(),
    planId: plan.id,
    planFile: path.relative(output, planFile),
    samplingIntervalMs: plan.samplingIntervalMs,
    featureSchemaVersion: MLP_FEATURE_SCHEMA_VERSION,
    featureCount: MLP_INPUT_FEATURE_COUNT,
    teacherParameterCount: TEACHER_PARAMETER_COUNT,
    teacherMetricCount: TEACHER_METRIC_COUNT,
    teacherMetricNames: ["crossEntropy", "klDivergence", "meanSquaredError", "iterations", "restarts", "converged"],
    actionCount: expectedGrid.length,
    grid: expectedGrid,
    currentGrid: expectedCurrentGrid,
    policySupport: {
      latent_lower: plan.execution.minimumEffectiveExposure,
      latent_upper: plan.execution.maximumEffectiveExposure,
      visible_lower: plan.execution.minimumUsableExposure,
      visible_upper: plan.execution.maximumUsableExposure,
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
      training: "first half of every non-aggregate inspector window",
      validation: "second half of every non-aggregate inspector window",
      test: `latest ${plan.latestTestDays} complete cached days`,
      overlapPriority: ["test", "validation", "train"],
      excludedAggregateWindows: plan.excludedAggregateWindows,
    },
    execution: plan.execution,
    oraclePreparation: plan.oraclePreparation,
    teacherFit: plan.teacherFit,
    teacherBackends: [...new Set(progress.shards.map((shard) => shard.teacherBackend))],
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
      ...progress.shards.map((shard) => shard.refinementPass ?? 0),
    ),
    counts,
    shards: progress.shards.filter((shard) => shard.count > 0).sort((left, right) =>
      left.date.localeCompare(right.date) || left.split.localeCompare(right.split)),
  };
  await atomicWriteJson(path.join(output, "dataset.json"), manifest);
  process.stdout.write(`${JSON.stringify({
    event: refinementPass > 0 ? "dataset-refinement-complete" : "dataset-complete",
    output,
    counts,
    refinementPass,
    rejectedExamples: refinementCases.size,
    rejectedSourceDays: sourceRejectionCases.size,
  })}\n`);
}

async function refreshDatasetFeatures(options: {
  dataDir: string;
  output: string;
  plan: TrainingPlan;
  progress: Progress;
  dataset: Record<string, unknown>;
}): Promise<void> {
  const { dataDir, output, plan, progress, dataset } = options;
  const featureStore = new MlpFeatureStore(dataDir);
  const oneSecondStore = new OneSecondCandleStore(dataDir);
  const state: MlpExposureStateInputs = {
    feeRate: plan.execution.feeBps / 10_000,
    minimumUsableExposure: plan.execution.minimumUsableExposure,
    maximumUsableExposure: plan.execution.maximumUsableExposure,
    minimumEffectiveExposure: plan.execution.minimumEffectiveExposure,
    maximumEffectiveExposure: plan.execution.maximumEffectiveExposure,
    quoteLendRate: plan.execution.maintenanceBpsHour.quoteLend / 10_000,
    quoteBorrowRate: plan.execution.maintenanceBpsHour.quoteBorrow / 10_000,
    assetBorrowRate: plan.execution.maintenanceBpsHour.assetBorrow / 10_000,
  };
  const shardsByDate = new Map<string, DatasetShard[]>();
  for (const shard of progress.shards) {
    const values = shardsByDate.get(shard.date) ?? [];
    values.push(shard);
    shardsByDate.set(shard.date, values);
  }
  const dates = [...shardsByDate.keys()].sort();
  process.stdout.write(`${JSON.stringify({
    event: "dataset-feature-refresh-start",
    featureSchemaVersion: MLP_FEATURE_SCHEMA_VERSION,
    featureCount: MLP_INPUT_FEATURE_COUNT,
    days: dates.length,
    shards: progress.shards.length,
  })}\n`);
  let completedShards = 0;
  for (let dateIndex = 0; dateIndex < dates.length; dateIndex += 1) {
    const date = dates[dateIndex]!;
    const day = parseDay(date);
    const dateShards = shardsByDate.get(date)!;
    const source = dateShards.some((shard) =>
      shard.featureSchemaVersion !== MLP_FEATURE_SCHEMA_VERSION && shard.count > 0)
      ? await oneSecondStore.loadRange(day - DAY_MS, day + DAY_MS)
      : undefined;
    for (const shard of dateShards) {
      if (shard.featureSchemaVersion !== MLP_FEATURE_SCHEMA_VERSION && shard.count > 0) {
        const times = await readTimes(path.join(output, shard.times), shard.count);
        const features = await featureStore.prepare(source!, times, state);
        await writeFeatureRowsAtomic(path.join(output, shard.features), times, features);
      }
      shard.featureSchemaVersion = MLP_FEATURE_SCHEMA_VERSION;
      completedShards += 1;
      progress.updatedAt = new Date().toISOString();
      await atomicWriteJson(path.join(output, "progress.json"), progress);
      process.stdout.write(`${JSON.stringify({
        event: "dataset-feature-refresh-progress",
        date,
        split: shard.split,
        day: dateIndex + 1,
        days: dates.length,
        completedShards,
        shards: progress.shards.length,
        examples: shard.count,
      })}\n`);
    }
  }
  await atomicWriteJson(path.join(output, "dataset.json"), {
      ...dataset,
      createdAt: new Date().toISOString(),
      featureSchemaVersion: MLP_FEATURE_SCHEMA_VERSION,
      featureCount: MLP_INPUT_FEATURE_COUNT,
    });
  process.stdout.write(`${JSON.stringify({
    event: "dataset-feature-refresh-complete",
    featureSchemaVersion: MLP_FEATURE_SCHEMA_VERSION,
    featureCount: MLP_INPUT_FEATURE_COUNT,
    shards: progress.shards.length,
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

  constructor(
    private readonly repoRoot: string,
    private readonly config: GpuTeacherConfig,
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
    if (!this.child) return;
    this.child.stdin.end();
    await this.workerExit;
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
      })}\n`, (error) => {
        if (!error) return;
        this.active = undefined;
        reject(error);
      });
    });
    overlap?.();
    await completion;
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

function sampledRowsForDay(
  day: number,
  interval: number,
  trainRanges: TimeRange[],
  validationRanges: TimeRange[],
  testStart: number,
  testEnd: number,
): Array<{ candleIndex: number; split: Split }> {
  const result: Array<{ candleIndex: number; split: Split }> = [];
  for (let time = day + interval; time <= day + DAY_MS; time += interval) {
    const timestamp = time - 1;
    let split: Split | undefined;
    if (timestamp >= testStart && timestamp < testEnd) split = "test";
    else if (validationRanges.some((range) => timestamp >= range.start && timestamp < range.end)) split = "validation";
    else if (trainRanges.some((range) => timestamp >= range.start && timestamp < range.end)) split = "train";
    if (split) result.push({ candleIndex: Math.floor((timestamp - day) / SECOND_MS), split });
  }
  return result;
}

async function writeFeatureRowsAtomic(
  file: string,
  times: readonly number[],
  features: Awaited<ReturnType<MlpFeatureStore["prepare"]>>,
): Promise<void> {
  await writeTypedArrayAtomic(file, encodeFeatureRows(times, features));
}

function encodeFeatureRows(
  times: readonly number[],
  features: Awaited<ReturnType<MlpFeatureStore["prepare"]>>,
): Buffer {
  const values = new Float32Array(times.length * MLP_INPUT_FEATURE_COUNT);
  for (let row = 0; row < times.length; row += 1) {
    features.encode(times[row]!, values, row * MLP_INPUT_FEATURE_COUNT);
  }
  const buffer = Buffer.allocUnsafe(values.length * 2);
  for (let index = 0; index < values.length; index += 1) {
    buffer.writeUInt16LE(float32ToFloat16(values[index]!), index * 2);
  }
  return buffer;
}

async function writeSelectedFeatureRowsAtomic(
  file: string,
  rows: Buffer,
  indexes: readonly number[],
): Promise<void> {
  const rowBytes = MLP_INPUT_FEATURE_COUNT * 2;
  if (rows.length % rowBytes !== 0
    || indexes.some((index) => !Number.isInteger(index) || index < 0 || (index + 1) * rowBytes > rows.length)) {
    throw new Error("MLP selected feature rows do not match the encoded feature buffer.");
  }
  const selected = Buffer.allocUnsafe(indexes.length * rowBytes);
  indexes.forEach((sourceIndex, destinationIndex) => {
    rows.copy(
      selected,
      destinationIndex * rowBytes,
      sourceIndex * rowBytes,
      (sourceIndex + 1) * rowBytes,
    );
  });
  await writeTypedArrayAtomic(file, selected);
}

async function writeTeacherParametersAtomic(file: string, rows: TeacherResult[]): Promise<void> {
  const values = new Float32Array(rows.length * TEACHER_PARAMETER_COUNT);
  rows.forEach((row, index) => values.set(row.rawParameters, index * TEACHER_PARAMETER_COUNT));
  await writeTypedArrayAtomic(file, Buffer.from(values.buffer));
}

async function writeTeacherMetricsAtomic(file: string, rows: TeacherResult[]): Promise<void> {
  const values = new Float32Array(rows.length * TEACHER_METRIC_COUNT);
  rows.forEach((row, index) => values.set(row.metrics, index * TEACHER_METRIC_COUNT));
  await writeTypedArrayAtomic(file, Buffer.from(values.buffer));
}

async function writeTimesAtomic(file: string, times: readonly number[]): Promise<void> {
  const buffer = Buffer.allocUnsafe(times.length * 8);
  for (let index = 0; index < times.length; index += 1) {
    buffer.writeBigInt64LE(BigInt(times[index]!), index * 8);
  }
  await writeTypedArrayAtomic(file, buffer);
}

async function readTimes(file: string, count: number): Promise<number[]> {
  const buffer = await fs.readFile(file);
  if (buffer.length !== count * 8) {
    throw new Error(`MLP timestamp shard ${file} has ${buffer.length} bytes; expected ${count * 8}.`);
  }
  return Array.from({ length: count }, (_, index) => Number(buffer.readBigInt64LE(index * 8)));
}

async function writeTypedArrayAtomic(file: string, buffer: Buffer): Promise<void> {
  const temporary = `${file}.tmp`;
  await fs.writeFile(temporary, buffer);
  await fs.rename(temporary, file);
}

class OneSecondCandleStore {
  private readonly root: string;
  private readonly days = new Map<number, Promise<Candle[]>>();

  constructor(dataDir: string) {
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
    const pending = this.readDay(day);
    this.days.set(day, pending);
    pending.catch(() => {
      if (this.days.get(day) === pending) this.days.delete(day);
    });
    while (this.days.size > 4) this.days.delete(this.days.keys().next().value!);
    return pending;
  }

  private async readDay(day: number): Promise<Candle[]> {
    const date = isoDate(day);
    let content: string;
    try {
      content = await fs.readFile(path.join(this.root, `${date}.jsonl`), "utf8");
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
      content = gunzipSync(await fs.readFile(path.join(this.root, `${date}.jsonl.gz`))).toString("utf8");
    }
    const result: Candle[] = [];
    for (const line of content.split("\n")) {
      if (line) result.push(JSON.parse(line) as Candle);
    }
    return result;
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
    if (value.version !== 3 || value.planId !== planId
      || value.latestCompleteDay !== isoDate(latestDay)) {
      throw new Error("Existing dataset progress belongs to a different plan or latest-history test range.");
    }
    value.shards = value.shards.map((shard) => ({
      ...shard,
      teacherBackend: shard.teacherBackend ?? "cpu-bfgs",
    }));
    return value;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    return {
      version: 3,
      planId,
      updatedAt: new Date().toISOString(),
      latestCompleteDay: isoDate(latestDay),
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
    if (value.version !== 1 || value.planId !== planId || !Array.isArray(value.cases)) {
      throw new Error("Existing teacher refinement queue belongs to a different training plan.");
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

function refinementCaseKey(split: Split, date: string, time: number): string {
  return `${split}:${date}:${time}`;
}

function inspectScoredDay(
  candles: readonly Candle[],
  day: number,
  date: string,
  splits: Split[],
): SourceDayRejection | undefined {
  const expectedCandles = DAY_MS / SECOND_MS;
  let firstUnexpectedIndex: number | null = null;
  let expectedOpenTime: number | null = null;
  let observedOpenTime: number | null = null;
  const comparedCount = Math.min(candles.length, expectedCandles);
  for (let index = 0; index < comparedCount; index += 1) {
    const expected = day + index * SECOND_MS;
    if (candles[index]!.openTime !== expected) {
      firstUnexpectedIndex = index;
      expectedOpenTime = expected;
      observedOpenTime = candles[index]!.openTime;
      break;
    }
  }
  if (firstUnexpectedIndex === null && candles.length < expectedCandles) {
    firstUnexpectedIndex = candles.length;
    expectedOpenTime = day + candles.length * SECOND_MS;
  }
  if (candles.length === expectedCandles && firstUnexpectedIndex === null) return undefined;
  const mismatch = firstUnexpectedIndex === null
    ? "timestamps extend beyond the expected UTC day"
    : `first timestamp mismatch at row ${firstUnexpectedIndex}`;
  return {
    date,
    splits,
    reason: "incomplete-or-noncontiguous-day",
    detail: `${date} has ${candles.length}/${expectedCandles} one-second candles; ${mismatch}.`,
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
  refreshFeatures: boolean;
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
  const refreshFeatures = options.get("refresh-features") === "true";
  return { planFile, outputOverride: options.get("output"), refinementPass, refreshFeatures };
}

function validatePlan(plan: TrainingPlan): void {
  const execution = plan.execution;
  if (plan.version !== 3 || !plan.id || !plan.datasetDir || !Array.isArray(plan.windows)
    || plan.oraclePreparation?.backend !== "cuda"
    || !Number.isInteger(plan.samplingIntervalMs / SECOND_MS)
    || plan.samplingIntervalMs < SECOND_MS
    || !Number.isInteger(plan.latestTestDays) || plan.latestTestDays < 1
    || plan.teacherFit.backend !== "cuda" || plan.teacherFit.device !== "cuda"
    || !Number.isInteger(plan.teacherFit.batchSize) || plan.teacherFit.batchSize < 1
    || !Number.isInteger(plan.teacherFit.projectionIterations)
    || plan.teacherFit.projectionIterations < 1
    || !Number.isInteger(plan.teacherFit.maxIterations) || plan.teacherFit.maxIterations < 1
    || !Number.isInteger(plan.teacherFit.adaptiveIterations)
    || plan.teacherFit.adaptiveIterations < 1
    || !Number.isInteger(plan.teacherFit.adaptiveRounds) || plan.teacherFit.adaptiveRounds < 0
    || !Number.isInteger(plan.teacherFit.restartCount) || plan.teacherFit.restartCount < 1
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
    || !Number.isInteger(plan.teacherFit.inputAlignmentFloats)
    || plan.teacherFit.inputAlignmentFloats < 1
    || !Number.isInteger(plan.teacherFit.inputQueueBatches)
    || plan.teacherFit.inputQueueBatches < 1
    || typeof plan.teacherFit.pipelinedRefinement !== "boolean"
    || !(execution.minimumEffectiveExposure <= execution.minimumUsableExposure)
    || !(execution.maximumEffectiveExposure >= execution.maximumUsableExposure)
    || !(execution.minimumUsableExposure < execution.maximumUsableExposure)
    || !(execution.temperature > 0)) {
    throw new Error("Invalid MLP v3 training plan.");
  }
}

function float32ToFloat16(value: number): number {
  const bits = new Uint32Array(new Float32Array([value]).buffer)[0]!;
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
