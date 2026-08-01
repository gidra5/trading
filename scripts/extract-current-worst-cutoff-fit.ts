import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { readCandleShardReferenceSync } from "@trading/storage";
import {
  conditionalExposureProbabilities,
  conditionalFourSegmentExposureProbabilities,
  conditionalFourSegmentParametersFromRaw,
  conditionalCutoffRawParameters,
  exposureHoldingFeasibleInterval,
  prepareExposureValueOracleCuda,
  type Candle,
} from "@trading/bot-algo";

const TEACHER_METRIC_COUNT = 7;

interface TrainingPlan {
  id: string;
  dataDir: string;
  datasetDir: string;
  execution: {
    feeBps: number;
    minimumUsableExposure: number;
    maximumUsableExposure: number;
    minimumEffectiveExposure: number;
    maximumEffectiveExposure: number;
    maintenanceBpsHour: { quoteBorrow: number; assetBorrow: number };
    gridSize: number;
    temperature: number;
    holdingPeriodSteps: number;
    decisionDelaySteps?: number;
    valueHorizonSteps: number;
  };
  teacherFit: {
    sampleStates: number;
    sampleActions: number;
    projectionIterations: number;
    maxIterations: number;
    adaptiveIterations: number;
    adaptiveRounds: number;
    restartCount: number;
    batchSize: number;
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
  };
}

interface DatasetShard {
  split: "train" | "validation" | "test";
  date: string;
  count: number;
  teacherParameters: string;
  teacherMetrics: string;
  times: string;
}

interface WorstFit {
  shard: DatasetShard;
  row: number;
  time: number;
  raw: number[];
  metrics: number[];
}

const DAY_MS = 86_400_000;

void main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : error);
  process.exitCode = 1;
});

async function main(): Promise<void> {
  const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
  const planFile = path.resolve(repoRoot, argument("plan") ?? "ml/training-plan.json");
  const plan = JSON.parse(fs.readFileSync(planFile, "utf8")) as TrainingPlan;
  const dataset = path.resolve(repoRoot, argument("dataset") ?? plan.datasetDir);
  const output = path.resolve(
    repoRoot,
    argument("output") ?? "data/training/analysis/current-worst-cutoff-fit.json",
  );
  const progress = JSON.parse(fs.readFileSync(path.join(dataset, "progress.json"), "utf8")) as {
    shards: DatasetShard[];
  };
  const requestedDate = argument("date");
  const requestedTime = argument("time");
  if ((requestedDate === undefined) !== (requestedTime === undefined)) {
    throw new Error("--date and --time must be provided together.");
  }
  const worst = requestedDate && requestedTime
    ? requestedFit(requestedDate, requestedTime)
    : findWorstFit(dataset, progress.shards);
  const sourceRoot = path.resolve(
    repoRoot,
    plan.dataDir,
    "market/immutable/refs/candles/spot-btcusdt/btcusdt/1s",
  );
  const candles = readCandles(sourceRoot, worst.shard.date);
  inspectCompleteDay(candles, worst.shard.date);
  const candleIndex = candles.findIndex((candle) => candle.closeTime === worst.time);
  if (candleIndex < 0) throw new Error(`No source candle closes at ${worst.time}.`);
  const feeRate = plan.execution.feeBps / 10_000;
  const maximumEffectiveExposure = Math.max(
    Math.abs(plan.execution.minimumEffectiveExposure),
    Math.abs(plan.execution.maximumEffectiveExposure),
  );
  const maintenance = {
    quoteBorrowRate: bpsHourToPerSecond(plan.execution.maintenanceBpsHour.quoteBorrow),
    assetBorrowRate: bpsHourToPerSecond(plan.execution.maintenanceBpsHour.assetBorrow),
  };
  const prices = Float64Array.from(candles, (candle) => candle.close);
  const prepared = await prepareExposureValueOracleCuda(prices, {
    scoreStartIndex: 0,
    holdingPeriodSteps: plan.execution.holdingPeriodSteps,
    decisionDelaySteps: plan.execution.decisionDelaySteps ?? 1,
    valueHorizonSteps: plan.execution.valueHorizonSteps,
    friction: feeRate,
    gridSize: plan.execution.gridSize,
    minExposure: plan.execution.minimumEffectiveExposure,
    maxExposure: plan.execution.maximumEffectiveExposure,
    maxEffectiveExposure: maximumEffectiveExposure,
    terminalIndex: candles.length - 1,
    temperature: plan.execution.temperature,
    opportunityEpsilon: 0,
    ...maintenance,
    includeActionValues: false,
    includeProbabilities: true,
  });
  const oracle = prepared.oracle;
  if (!oracle.probabilities) throw new Error("CUDA oracle did not retain probabilities.");
  const cutoff = exposureHoldingFeasibleInterval(
    prices,
    candleIndex,
    plan.execution.holdingPeriodSteps,
    {
      friction: feeRate,
      minExposure: plan.execution.minimumUsableExposure,
      maxExposure: plan.execution.maximumUsableExposure,
      maxEffectiveExposure: maximumEffectiveExposure,
      ...maintenance,
    },
  );
  const base = new Float64Array(oracle.grid.length);
  const sourceOffset = candleIndex * oracle.grid.length;
  let baseTotal = 0;
  for (let action = 0; action < oracle.grid.length; action += 1) {
    const exposure = oracle.grid[action]!;
    const probability = exposure >= cutoff.lower && exposure <= cutoff.upper
      ? oracle.probabilities[sourceOffset + action]!
      : 0;
    base[action] = probability;
    baseTotal += probability;
  }
  if (!(baseTotal > 0)) throw new Error("Exact cutoff removed every oracle action.");
  for (let action = 0; action < base.length; action += 1) base[action] /= baseTotal;

  if (worst.raw.length === 0) {
    const fitted = fitSingleCase(repoRoot, plan, oracle.grid, oracle.currentGrid, base, cutoff);
    worst.raw = fitted.raw;
    worst.metrics = fitted.metrics;
  }

  const parameters = conditionalFourSegmentParametersFromRaw(worst.raw, {
    latentLower: plan.execution.minimumEffectiveExposure,
    latentUpper: plan.execution.maximumEffectiveExposure,
    visibleLower: plan.execution.minimumEffectiveExposure,
    visibleUpper: plan.execution.maximumEffectiveExposure,
    hingeSpan: plan.teacherFit.scoreHingeSpan,
    friction: feeRate,
    temperature: plan.execution.temperature,
  });
  const actions = Array.from(oracle.grid);
  const currents = Array.from(oracle.currentGrid);
  const target = new Float64Array(actions.length * currents.length);
  const fitted = new Float64Array(target.length);
  const residual = new Float64Array(target.length);
  const rowKl = new Float64Array(currents.length);
  let crossEntropy = 0;
  let entropy = 0;
  let meanSquaredError = 0;
  let visibleCrossEntropy = 0;
  let visibleEntropy = 0;
  let visibleMeanSquaredError = 0;
  let visibleCurrentRows = 0;
  let visibleCells = 0;
  let worstCurrentRow = 0;
  for (let currentIndex = 0; currentIndex < currents.length; currentIndex += 1) {
    const current = currents[currentIndex]!;
    const targetRow = conditionalExposureProbabilities(
      base,
      oracle.grid,
      current,
      feeRate,
      1 / plan.execution.temperature,
    );
    const fittedRow = conditionalFourSegmentExposureProbabilities(
      oracle.grid,
      current,
      parameters,
    );
    let currentKl = 0;
    for (let action = 0; action < actions.length; action += 1) {
      const index = currentIndex * actions.length + action;
      const oracleProbability = targetRow[action]!;
      const fittedProbability = fittedRow[action]!;
      target[index] = oracleProbability;
      fitted[index] = fittedProbability;
      residual[index] = fittedProbability - oracleProbability;
      if (oracleProbability > 0) {
        const targetLog = Math.log(oracleProbability);
        const fittedLog = Math.log(Math.max(Number.MIN_VALUE, fittedProbability));
        crossEntropy -= oracleProbability * fittedLog;
        entropy -= oracleProbability * targetLog;
        currentKl += oracleProbability * (targetLog - fittedLog);
      }
      meanSquaredError += (fittedProbability - oracleProbability) ** 2;
    }
    rowKl[currentIndex] = currentKl;
    if (currentKl > rowKl[worstCurrentRow]!) worstCurrentRow = currentIndex;
    if (current >= plan.execution.minimumUsableExposure
      && current <= plan.execution.maximumUsableExposure) {
      let targetVisibleTotal = 0;
      let fittedVisibleTotal = 0;
      for (let action = 0; action < actions.length; action += 1) {
        if (actions[action]! < plan.execution.minimumUsableExposure
          || actions[action]! > plan.execution.maximumUsableExposure) continue;
        const index = currentIndex * actions.length + action;
        targetVisibleTotal += target[index]!;
        fittedVisibleTotal += fitted[index]!;
      }
      visibleCurrentRows += 1;
      for (let action = 0; action < actions.length; action += 1) {
        if (actions[action]! < plan.execution.minimumUsableExposure
          || actions[action]! > plan.execution.maximumUsableExposure) continue;
        const index = currentIndex * actions.length + action;
        const oracleProbability = target[index]! / targetVisibleTotal;
        const fittedProbability = fitted[index]! / fittedVisibleTotal;
        if (oracleProbability > 0) {
          visibleCrossEntropy -= oracleProbability * Math.log(
            Math.max(Number.MIN_VALUE, fittedProbability),
          );
          visibleEntropy -= oracleProbability * Math.log(oracleProbability);
        }
        visibleMeanSquaredError += (fittedProbability - oracleProbability) ** 2;
        visibleCells += 1;
      }
    }
  }
  crossEntropy /= currents.length;
  entropy /= currents.length;
  meanSquaredError /= target.length;
  visibleCrossEntropy /= visibleCurrentRows;
  visibleEntropy /= visibleCurrentRows;
  visibleMeanSquaredError /= visibleCells;

  const result = {
    version: 1,
    planId: plan.id,
    source: {
      split: worst.shard.split,
      date: worst.shard.date,
      time: worst.time,
      isoTime: new Date(worst.time).toISOString(),
      candleIndex,
      price: candles[candleIndex]!.close,
      row: worst.row,
      oracleKernelMs: prepared.kernelMs,
    },
    support: {
      effective: [plan.execution.minimumEffectiveExposure, plan.execution.maximumEffectiveExposure],
      usable: [plan.execution.minimumUsableExposure, plan.execution.maximumUsableExposure],
      exactCutoff: [cutoff.lower, cutoff.upper],
      decodedCutoff: [parameters.cutoffLower, parameters.cutoffUpper],
      holdingPeriodSteps: plan.execution.holdingPeriodSteps,
    },
    rawParameters: worst.raw,
    decodedParameters: parameters,
    storedMetrics: {
      crossEntropy: worst.metrics[0],
      klDivergence: worst.metrics[1],
      meanSquaredError: worst.metrics[2],
      iterations: worst.metrics[3],
      restarts: worst.metrics[4],
      converged: worst.metrics[5] > 0,
    },
    recomputedMetrics: {
      crossEntropy: visibleCrossEntropy,
      entropy: visibleEntropy,
      klDivergence: visibleCrossEntropy - visibleEntropy,
      meanSquaredError: visibleMeanSquaredError,
      support: "usable",
    },
    recomputedFullMetrics: {
      crossEntropy,
      entropy,
      klDivergence: crossEntropy - entropy,
      meanSquaredError,
    },
    grid: { actions, currents },
    matrices: {
      oracle: rounded(target),
      fitted: rounded(fitted),
      residual: rounded(residual),
      rowKl: rounded(rowKl),
    },
    selection: {
      worstCurrentRow,
      worstCurrentExposure: currents[worstCurrentRow],
    },
  };
  fs.mkdirSync(path.dirname(output), { recursive: true });
  fs.writeFileSync(output, `${JSON.stringify(result)}\n`);
  const visualization = argument("visualization");
  if (visualization) {
    const visualizationFile = path.resolve(repoRoot, visualization);
    const template = fs.readFileSync(visualizationFile, "utf8");
    const embedded = template.includes("__FIT_DATA__")
      ? template.replace("__FIT_DATA__", JSON.stringify(result))
      : template.replace(
          /const FIT_DATA = [^\n]*;\n  const root/,
          `const FIT_DATA = ${JSON.stringify(result)};\n  const root`,
        );
    if (embedded === template) {
      throw new Error(`Visualization ${visualizationFile} has no replaceable FIT_DATA value.`);
    }
    fs.writeFileSync(
      visualizationFile,
      embedded,
    );
  }
  process.stdout.write(`${JSON.stringify({
    event: "current-worst-cutoff-fit",
    output,
    date: worst.shard.date,
    time: worst.time,
    storedKl: worst.metrics[1],
    recomputedKl: result.recomputedMetrics.klDivergence,
    mse: result.recomputedMetrics.meanSquaredError,
    cutoff: result.support.decodedCutoff,
    worstCurrentExposure: result.selection.worstCurrentExposure,
    ...(visualization ? { visualization: path.resolve(repoRoot, visualization) } : {}),
  })}\n`);
}

function requestedFit(date: string, time: string): WorstFit {
  const parsedTime = Number(time);
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date) || !Number.isSafeInteger(parsedTime)) {
    throw new Error("Requested fit requires an ISO date and integer millisecond timestamp.");
  }
  return {
    shard: {
      split: "validation",
      date,
      count: 1,
      teacherParameters: "",
      teacherMetrics: "",
      times: "",
    },
    row: 0,
    time: parsedTime,
    raw: [],
    metrics: [],
  };
}

function fitSingleCase(
  repoRoot: string,
  plan: TrainingPlan,
  actionGrid: Float64Array,
  currentGrid: Float64Array,
  base: Float64Array,
  cutoff: { lower: number; upper: number },
): { raw: number[]; metrics: number[] } {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "mlp-cutoff-fit-"));
  try {
    const input = path.join(directory, "input.f32");
    const config = path.join(directory, "config.json");
    const parameters = path.join(directory, "parameters.f32");
    const metrics = path.join(directory, "metrics.f32");
    const rowStride = Math.ceil((actionGrid.length + 2) / plan.teacherFit.inputAlignmentFloats)
      * plan.teacherFit.inputAlignmentFloats;
    const packed = new Float32Array(rowStride);
    packed.set(base);
    const cutoffRaw = conditionalCutoffRawParameters(cutoff.lower, cutoff.upper, {
      latentLower: plan.execution.minimumEffectiveExposure,
      latentUpper: plan.execution.maximumEffectiveExposure,
    });
    packed[actionGrid.length] = cutoffRaw[0];
    packed[actionGrid.length + 1] = cutoffRaw[1];
    fs.writeFileSync(input, Buffer.from(packed.buffer));
    fs.writeFileSync(config, `${JSON.stringify({
      action_grid: Array.from(actionGrid),
      current_grid: Array.from(currentGrid),
      friction: plan.execution.feeBps / 10_000,
      transition_log_scale: 1 / plan.execution.temperature,
      latent_lower: plan.execution.minimumEffectiveExposure,
      latent_upper: plan.execution.maximumEffectiveExposure,
      visible_lower: plan.execution.minimumEffectiveExposure,
      visible_upper: plan.execution.maximumEffectiveExposure,
      metric_visible_lower: plan.execution.minimumUsableExposure,
      metric_visible_upper: plan.execution.maximumUsableExposure,
      sample_states: plan.teacherFit.sampleStates,
      sample_actions: plan.teacherFit.sampleActions,
      projection_iterations: plan.teacherFit.projectionIterations,
      iterations: plan.teacherFit.maxIterations,
      adaptive_iterations: plan.teacherFit.adaptiveIterations,
      adaptive_rounds: plan.teacherFit.adaptiveRounds,
      restarts: plan.teacherFit.restartCount,
      batch_size: 1,
      tolerance: plan.teacherFit.tolerance,
      max_mean_kl: plan.teacherFit.maxMeanKlDivergence,
      max_mean_mse: plan.teacherFit.maxMeanSquaredError,
      line_search_candidates: plan.teacherFit.lineSearchCandidates,
      temporal_refinement_rounds: 0,
      temporal_iterations: plan.teacherFit.temporalIterations,
      temporal_followup_iterations: plan.teacherFit.temporalFollowupIterations,
      temporal_equivalent_loss_absolute: plan.teacherFit.temporalEquivalentLossAbsolute,
      temporal_equivalent_loss_relative: plan.teacherFit.temporalEquivalentLossRelative,
      optimizer_backend: plan.teacherFit.optimizerBackend,
      optimizer_host_check_interval: plan.teacherFit.optimizerHostCheckInterval,
      quality_fallback_iterations: plan.teacherFit.qualityFallbackIterations,
      visible_sample_fraction: plan.teacherFit.visibleSampleFraction,
      score_hinge_span: plan.teacherFit.scoreHingeSpan,
      compact_visible_initialization: plan.teacherFit.compactVisibleInitialization,
      input_row_stride: rowStride,
      input_queue_batches: 1,
      pipelined_refinement: false,
    })}\n`);
    const outcome = spawnSync(path.join(
      repoRoot,
      ".venv-ml",
      process.platform === "win32" ? "Scripts/python.exe" : "bin/python",
    ), [
      path.join(repoRoot, "ml/fit_teacher_cuda.py"),
      "--input", input,
      "--count", "1",
      "--config", config,
      "--parameters-output", parameters,
      "--metrics-output", metrics,
      "--device", "cuda",
    ], {
      cwd: repoRoot,
      env: { ...process.env, PYTHONPATH: path.join(repoRoot, "ml") },
      encoding: "utf8",
      maxBuffer: 16 * 1024 * 1024,
    });
    if (outcome.status !== 0) {
      throw new Error(`Single-case CUDA fit failed: ${outcome.stderr || outcome.stdout}`);
    }
    return {
      raw: Array.from(float32File(parameters)),
      metrics: Array.from(float32File(metrics)),
    };
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
}

function findWorstFit(dataset: string, shards: DatasetShard[]): WorstFit {
  let worst: WorstFit | undefined;
  for (const shard of shards) {
    if (shard.count < 1) continue;
    const metrics = float32File(path.join(dataset, shard.teacherMetrics));
    const raw = float32File(path.join(dataset, shard.teacherParameters));
    const times = int64File(path.join(dataset, shard.times));
    if (metrics.length !== shard.count * TEACHER_METRIC_COUNT || raw.length !== shard.count * 8
      || times.length !== shard.count) {
      throw new Error(`Shard ${shard.split}:${shard.date} has inconsistent array sizes.`);
    }
    for (let row = 0; row < shard.count; row += 1) {
      const kl = metrics[row * TEACHER_METRIC_COUNT + 1]!;
      if (!Number.isFinite(kl) || worst && kl <= worst.metrics[1]!) continue;
      worst = {
        shard,
        row,
        time: Number(times[row]),
        metrics: Array.from(metrics.slice(
          row * TEACHER_METRIC_COUNT,
          (row + 1) * TEACHER_METRIC_COUNT,
        )),
        raw: Array.from(raw.slice(row * 8, row * 8 + 8)),
      };
    }
  }
  if (!worst) throw new Error("The partial dataset contains no finite teacher fit.");
  return worst;
}

function float32File(file: string): Float32Array {
  const buffer = fs.readFileSync(file);
  return new Float32Array(buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength));
}

function int64File(file: string): BigInt64Array {
  const buffer = fs.readFileSync(file);
  return new BigInt64Array(buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength));
}

function rounded(values: ArrayLike<number>): number[] {
  return Array.from(values, (value) => Number(value.toPrecision(7)));
}

function readCandles(root: string, date: string): Candle[] {
  return readCandleShardReferenceSync(path.join(root, `${date}.json`));
}

function inspectCompleteDay(candles: Candle[], date: string): void {
  if (candles.length !== DAY_MS / 1_000) throw new Error(`${date} has ${candles.length} candles.`);
  for (let index = 1; index < candles.length; index += 1) {
    if (candles[index]!.openTime !== candles[index - 1]!.openTime + 1_000) {
      throw new Error(`${date} has a gap at candle ${index}.`);
    }
  }
}

function bpsHourToPerSecond(bps: number): number {
  return Math.expm1(Math.log1p(bps / 10_000) / 3_600);
}

function argument(name: string): string | undefined {
  const index = process.argv.indexOf(`--${name}`);
  return index >= 0 ? process.argv[index + 1] : undefined;
}
