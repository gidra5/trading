import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { gunzipSync } from "node:zlib";
import {
  conditionalCutoffRawParameters,
  conditionalExposureProbabilities,
  conditionalFourSegmentExposureProbabilities,
  conditionalFourSegmentParametersFromRaw,
  exposureHoldingFeasibleInterval,
  prepareExposureValueOracleCuda,
  type Candle,
} from "@trading/bot-algo";

const DAY_MS = 86_400_000;
const PARAMETER_COUNT = 8;
const METRIC_COUNT = 6;

interface SourceCase {
  split: "train" | "validation" | "test";
  date: string;
  time: number;
  klDivergence: number;
  meanSquaredError: number;
  price: number;
  [key: string]: unknown;
}

interface PreparedCase {
  source: SourceCase;
  candleIndex: number;
  base: Float64Array;
  cutoff: { lower: number; upper: number };
}

void main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : error);
  process.exitCode = 1;
});

async function main(): Promise<void> {
  const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
  const planFile = path.resolve(repoRoot, argument("plan") ?? "ml/training-plan.json");
  const sourceFile = path.resolve(
    repoRoot,
    argument("source") ?? "data/ml-analysis/mlp-rejection-oracles-grid151-control/metadata.json",
  );
  const output = path.resolve(
    repoRoot,
    argument("output") ?? "data/ml-analysis/hard-cutoff-rejection-corpus.json",
  );
  const plan = JSON.parse(fs.readFileSync(planFile, "utf8"));
  const source = JSON.parse(fs.readFileSync(sourceFile, "utf8")) as {
    cases: SourceCase[];
    selection: unknown;
  };
  const feeRate = plan.execution.feeBps / 10_000;
  const maximumEffectiveExposure = Math.max(
    Math.abs(plan.execution.minimumEffectiveExposure),
    Math.abs(plan.execution.maximumEffectiveExposure),
  );
  const maintenance = {
    quoteLendRate: bpsHourToPerSecond(plan.execution.maintenanceBpsHour.quoteLend),
    quoteBorrowRate: bpsHourToPerSecond(plan.execution.maintenanceBpsHour.quoteBorrow),
    assetBorrowRate: bpsHourToPerSecond(plan.execution.maintenanceBpsHour.assetBorrow),
  };
  const sourceRoot = path.resolve(
    repoRoot,
    plan.dataDir,
    "historical/spot-btcusdt/btcusdt/1s",
  );
  const grouped = Map.groupBy(source.cases, (item) => item.date);
  const prepared: PreparedCase[] = [];
  let actionGrid: Float64Array | undefined;
  let currentGrid: Float64Array | undefined;
  for (const [dateIndex, [date, cases]] of [...grouped].entries()) {
    const candles = readCandles(sourceRoot, date);
    inspectCompleteDay(candles, date);
    const prices = Float64Array.from(candles, (candle) => candle.close);
    const oracle = await prepareExposureValueOracleCuda(prices, {
      scoreStartIndex: 0,
      holdingPeriodSteps: plan.execution.holdingPeriodSteps,
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
    if (!oracle.oracle.probabilities) throw new Error(`Oracle probabilities missing for ${date}.`);
    actionGrid ??= oracle.oracle.grid;
    currentGrid ??= oracle.oracle.currentGrid;
    assertGrid(actionGrid, oracle.oracle.grid, "action");
    assertGrid(currentGrid, oracle.oracle.currentGrid, "current");
    const indexByTime = new Map(candles.map((candle, index) => [candle.closeTime, index]));
    for (const item of cases) {
      const candleIndex = indexByTime.get(item.time);
      if (candleIndex === undefined) throw new Error(`${date} has no candle closing at ${item.time}.`);
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
      const base = new Float64Array(actionGrid.length);
      const offset = candleIndex * actionGrid.length;
      let total = 0;
      for (let action = 0; action < actionGrid.length; action += 1) {
        const exposure = actionGrid[action]!;
        const probability = exposure >= cutoff.lower && exposure <= cutoff.upper
          ? oracle.oracle.probabilities[offset + action]!
          : 0;
        base[action] = probability;
        total += probability;
      }
      if (!(total > 0)) throw new Error(`${date}:${item.time} has no feasible oracle action.`);
      for (let action = 0; action < base.length; action += 1) base[action] /= total;
      prepared.push({ source: item, candleIndex, base, cutoff });
    }
    process.stdout.write(`${JSON.stringify({
      event: "hard-cutoff-corpus-oracle",
      date,
      day: dateIndex + 1,
      days: grouped.size,
      cases: cases.length,
      kernelMs: oracle.kernelMs,
    })}\n`);
  }
  if (!actionGrid || !currentGrid) throw new Error("The source rejection corpus is empty.");
  const fitted = fitCases(repoRoot, plan, actionGrid, currentGrid, prepared);
  const cases = prepared.map((item, index) => evaluateCase(
    item,
    fitted.raw.slice(index * PARAMETER_COUNT, (index + 1) * PARAMETER_COUNT),
    fitted.metrics.slice(index * METRIC_COUNT, (index + 1) * METRIC_COUNT),
    actionGrid!,
    currentGrid!,
    plan,
  ));
  const criticalIndexes = cases.flatMap((item, index) =>
    item.usable.klDivergence > 0.03 ? [index] : []);
  const deepPrepared = criticalIndexes.map((index) => prepared[index]!);
  const deepFitted = deepPrepared.length > 0
    ? fitCases(repoRoot, plan, actionGrid, currentGrid, deepPrepared, true)
    : undefined;
  const deepCritical = deepPrepared.map((item, index) => evaluateCase(
    item,
    deepFitted!.raw.slice(index * PARAMETER_COUNT, (index + 1) * PARAMETER_COUNT),
    deepFitted!.metrics.slice(index * METRIC_COUNT, (index + 1) * METRIC_COUNT),
    actionGrid!,
    currentGrid!,
    plan,
  ));
  const sorted = [...cases].sort((left, right) =>
    right.usable.klDivergence - left.usable.klDivergence);
  const fullKl = cases.map((item) => item.full.klDivergence);
  const usableKl = cases.map((item) => item.usable.klDivergence);
  const usableMeanRmse = cases.map((item) => item.usable.actionMeanRmse);
  const result = {
    version: 1,
    createdAt: new Date().toISOString(),
    planId: plan.id,
    source: sourceFile,
    sample: source.selection,
    thresholds: {
      strictKl: plan.teacherFit.maxMeanKlDivergence,
      strictMeanSquaredError: plan.teacherFit.maxMeanSquaredError,
      materialShapeKl: 0.01,
      criticalShapeKl: 0.03,
    },
    summary: {
      cases: cases.length,
      dates: grouped.size,
      oldWorstKl: Math.max(...cases.map((item) => item.old.klDivergence)),
      newFullKl: summarize(fullKl),
      newUsableKl: summarize(usableKl),
      newUsableActionMeanRmse: summarize(usableMeanRmse),
      strictAccepted: cases.filter((item) => item.strictAccepted).length,
      strictRejected: cases.filter((item) => !item.strictAccepted).length,
      materialFullShapeCases: cases.filter((item) => item.full.klDivergence > 0.01).length,
      criticalFullShapeCases: cases.filter((item) => item.full.klDivergence > 0.03).length,
      materialUsableShapeCases: cases.filter((item) => item.usable.klDivergence > 0.01).length,
      criticalUsableShapeCases: cases.filter((item) => item.usable.klDivergence > 0.03).length,
      maximumStoredVsRecomputedKlError: Math.max(...cases.map((item) =>
        Math.abs(item.fitter.klDivergence - item.usable.klDivergence))),
      deepCriticalCases: deepCritical.length,
      deepRemainingCriticalUsableShapeCases: deepCritical.filter((item) =>
        item.usable.klDivergence > 0.03).length,
      deepCriticalUsableKl: deepCritical.length > 0
        ? summarize(deepCritical.map((item) => item.usable.klDivergence))
        : null,
    },
    worst: sorted.slice(0, 20),
    deepCritical,
    cases,
  };
  fs.mkdirSync(path.dirname(output), { recursive: true });
  fs.writeFileSync(output, `${JSON.stringify(result, null, 2)}\n`);
  process.stdout.write(`${JSON.stringify({
    event: "hard-cutoff-corpus-complete",
    output,
    summary: result.summary,
    worst: result.worst.slice(0, 5).map((item) => ({
      date: item.date,
      time: item.time,
      oldKl: item.old.klDivergence,
      fullKl: item.full.klDivergence,
      usableKl: item.usable.klDivergence,
      usableActionMeanRmse: item.usable.actionMeanRmse,
      cutoff: item.cutoff,
    })),
  })}\n`);
}

function fitCases(
  repoRoot: string,
  plan: any,
  actionGrid: Float64Array,
  currentGrid: Float64Array,
  cases: PreparedCase[],
  deep = false,
): { raw: Float32Array; metrics: Float32Array } {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "mlp-cutoff-corpus-"));
  try {
    const input = path.join(directory, "input.f32");
    const config = path.join(directory, "config.json");
    const parameters = path.join(directory, "parameters.f32");
    const metrics = path.join(directory, "metrics.f32");
    const rowStride = Math.ceil((actionGrid.length + 2) / plan.teacherFit.inputAlignmentFloats)
      * plan.teacherFit.inputAlignmentFloats;
    const packed = new Float32Array(cases.length * rowStride);
    for (const [index, item] of cases.entries()) {
      const offset = index * rowStride;
      packed.set(item.base, offset);
      const raw = conditionalCutoffRawParameters(item.cutoff.lower, item.cutoff.upper, {
        latentLower: plan.execution.minimumEffectiveExposure,
        latentUpper: plan.execution.maximumEffectiveExposure,
      });
      packed[offset + actionGrid.length] = raw[0];
      packed[offset + actionGrid.length + 1] = raw[1];
    }
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
      iterations: deep
        ? Math.max(plan.teacherFit.maxIterations, plan.teacherFit.adaptiveIterations)
        : plan.teacherFit.maxIterations,
      adaptive_iterations: deep
        ? Math.max(plan.teacherFit.adaptiveIterations * 2, plan.teacherFit.maxIterations)
        : plan.teacherFit.adaptiveIterations,
      adaptive_rounds: deep
        ? plan.teacherFit.adaptiveRounds + 1
        : plan.teacherFit.adaptiveRounds,
      restarts: plan.teacherFit.restartCount,
      batch_size: cases.length,
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
      quality_fallback_iterations: deep
        ? Math.max(plan.teacherFit.qualityFallbackIterations, plan.teacherFit.adaptiveIterations)
        : plan.teacherFit.qualityFallbackIterations,
      input_row_stride: rowStride,
      input_queue_batches: 1,
      pipelined_refinement: false,
    })}\n`);
    const outcome = spawnSync(path.join(repoRoot, ".venv-ml/bin/python"), [
      path.join(repoRoot, "ml/fit_teacher_cuda.py"),
      "--input", input,
      "--count", String(cases.length),
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
      throw new Error(`Cutoff corpus CUDA fit failed: ${outcome.stderr || outcome.stdout}`);
    }
    return { raw: float32File(parameters), metrics: float32File(metrics) };
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
}

function evaluateCase(
  item: PreparedCase,
  raw: Float32Array,
  fitterMetrics: Float32Array,
  actions: Float64Array,
  currents: Float64Array,
  plan: any,
) {
  const parameters = conditionalFourSegmentParametersFromRaw(raw, {
    latentLower: plan.execution.minimumEffectiveExposure,
    latentUpper: plan.execution.maximumEffectiveExposure,
    visibleLower: plan.execution.minimumEffectiveExposure,
    visibleUpper: plan.execution.maximumEffectiveExposure,
    friction: plan.execution.feeBps / 10_000,
    temperature: plan.execution.temperature,
  });
  let fullKl = 0;
  let fullMse = 0;
  let usableKl = 0;
  let usableMse = 0;
  let usableMeanSquare = 0;
  let worstRowKl = 0;
  let worstUsableRowKl = 0;
  let usableCells = 0;
  let usableCurrentCount = 0;
  const usableMask = Array.from(actions, (action) =>
    action >= plan.execution.minimumUsableExposure
    && action <= plan.execution.maximumUsableExposure);
  for (const current of currents) {
    const target = conditionalExposureProbabilities(
      item.base,
      actions,
      current,
      plan.execution.feeBps / 10_000,
      1 / plan.execution.temperature,
    );
    const fitted = conditionalFourSegmentExposureProbabilities(actions, current, parameters);
    let rowKl = 0;
    let targetUsableTotal = 0;
    let fittedUsableTotal = 0;
    for (let action = 0; action < actions.length; action += 1) {
      fullMse += (fitted[action]! - target[action]!) ** 2;
      if (target[action]! > 0) {
        rowKl += target[action]! * Math.log(
          target[action]! / Math.max(Number.MIN_VALUE, fitted[action]!),
        );
      }
      if (usableMask[action]) {
        targetUsableTotal += target[action]!;
        fittedUsableTotal += fitted[action]!;
      }
    }
    fullKl += rowKl;
    worstRowKl = Math.max(worstRowKl, rowKl);
    if (current < plan.execution.minimumUsableExposure
      || current > plan.execution.maximumUsableExposure) continue;
    usableCurrentCount += 1;
    let rowUsableKl = 0;
    let targetMean = 0;
    let fittedMean = 0;
    for (let action = 0; action < actions.length; action += 1) {
      if (!usableMask[action]) continue;
      const targetProbability = target[action]! / targetUsableTotal;
      const fittedProbability = fitted[action]! / fittedUsableTotal;
      usableMse += (fittedProbability - targetProbability) ** 2;
      usableCells += 1;
      if (targetProbability > 0) {
        rowUsableKl += targetProbability * Math.log(
          targetProbability / Math.max(Number.MIN_VALUE, fittedProbability),
        );
      }
      targetMean += targetProbability * actions[action]!;
      fittedMean += fittedProbability * actions[action]!;
    }
    usableKl += rowUsableKl;
    worstUsableRowKl = Math.max(worstUsableRowKl, rowUsableKl);
    usableMeanSquare += (fittedMean - targetMean) ** 2;
  }
  const full = {
    klDivergence: fullKl / currents.length,
    meanSquaredError: fullMse / (currents.length * actions.length),
    worstCurrentKlDivergence: worstRowKl,
  };
  const usable = {
    klDivergence: usableKl / usableCurrentCount,
    meanSquaredError: usableMse / usableCells,
    worstCurrentKlDivergence: worstUsableRowKl,
    actionMeanRmse: Math.sqrt(usableMeanSquare / usableCurrentCount),
  };
  return {
    date: item.source.date,
    time: item.source.time,
    isoTime: new Date(item.source.time).toISOString(),
    price: item.source.price,
    old: {
      klDivergence: item.source.klDivergence,
      meanSquaredError: item.source.meanSquaredError,
    },
    cutoff: [parameters.cutoffLower, parameters.cutoffUpper],
    rawParameters: Array.from(raw),
    fitter: {
      crossEntropy: fitterMetrics[0],
      klDivergence: fitterMetrics[1],
      meanSquaredError: fitterMetrics[2],
      iterations: fitterMetrics[3],
      converged: fitterMetrics[5]! > 0,
    },
    full,
    usable,
    strictAccepted: usable.klDivergence <= plan.teacherFit.maxMeanKlDivergence
      && usable.meanSquaredError <= plan.teacherFit.maxMeanSquaredError,
  };
}

function summarize(values: number[]) {
  const sorted = [...values].sort((left, right) => left - right);
  const quantile = (value: number) => sorted[Math.round(value * (sorted.length - 1))]!;
  return {
    mean: values.reduce((sum, value) => sum + value, 0) / values.length,
    median: quantile(0.5),
    p90: quantile(0.9),
    p99: quantile(0.99),
    maximum: sorted.at(-1),
  };
}

function assertGrid(expected: Float64Array, actual: Float64Array, name: string): void {
  if (expected.length !== actual.length
    || expected.some((value, index) => value !== actual[index])) {
    throw new Error(`Corpus ${name} grids differ between dates.`);
  }
}

function float32File(file: string): Float32Array {
  const buffer = fs.readFileSync(file);
  return new Float32Array(buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength));
}

function readCandles(root: string, date: string): Candle[] {
  const plain = path.join(root, `${date}.jsonl`);
  const content = fs.existsSync(plain)
    ? fs.readFileSync(plain, "utf8")
    : gunzipSync(fs.readFileSync(`${plain}.gz`)).toString("utf8");
  return content.split("\n").filter(Boolean).map((line) => JSON.parse(line) as Candle);
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
