import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { performance } from "node:perf_hooks";
import { fileURLToPath } from "node:url";
import { gunzipSync } from "node:zlib";
import {
  directOracleDiagnosticsCuda,
  exposureHoldingCutoffsCuda,
  prepareExposureValueOracleCuda,
  vwKamaCudaStatus,
  type Candle,
} from "@trading/bot-algo";

const bpsHourToPerSecond = (bps: number): number =>
  Math.expm1(Math.log1p(bps / 10_000) / 3_600);

void main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : error);
  process.exitCode = 1;
});

async function main(): Promise<void> {
  const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
  const planFile = path.resolve(repoRoot, argument("plan") ?? "ml/training-plan.json");
  const plan = JSON.parse(fs.readFileSync(planFile, "utf8"));
  const date = argument("date") ?? "2023-11-30";
  const candleLimit = positiveInteger(argument("candles") ?? "86400", "candles");
  const iterations = positiveInteger(argument("iterations") ?? "1", "iterations");
  const execution = plan.execution;
  const holdingPeriodSteps = positiveInteger(
    argument("holding-steps") ?? String(execution.holdingPeriodSteps),
    "holding-steps",
  );
  const valueHorizonSteps = positiveInteger(
    argument("horizon-steps") ?? String(execution.valueHorizonSteps),
    "horizon-steps",
  );
  const maintenanceBps = argument("maintenance-bps") === undefined
    ? execution.maintenanceBpsHour
    : {
        quoteBorrow: Number(argument("maintenance-bps")),
        assetBorrow: Number(argument("maintenance-bps")),
      };
  if (!Object.values(maintenanceBps).every(Number.isFinite)) {
    throw new Error("--maintenance-bps must be finite.");
  }
  const gridSizes = (argument("grid-sizes") ?? String(execution.gridSize))
    .split(",")
    .map((value) => positiveInteger(value.trim(), "grid-sizes"));
  const sourceRoot = path.resolve(repoRoot, plan.dataDir, "historical/spot-btcusdt/btcusdt/1s");
  const plainFile = path.join(sourceRoot, `${date}.jsonl`);
  const compressedFile = `${plainFile}.gz`;
  const content = fs.existsSync(plainFile)
    ? fs.readFileSync(plainFile, "utf8")
    : gunzipSync(fs.readFileSync(compressedFile)).toString("utf8");
  const candles = content.split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line) as Candle)
    .slice(0, candleLimit);
  if (candles.length !== candleLimit) {
    throw new Error(`${date} supplies ${candles.length}/${candleLimit} requested candles.`);
  }
  for (let index = 1; index < candles.length; index += 1) {
    if (candles[index]!.openTime !== candles[index - 1]!.openTime + 1_000) {
      throw new Error(`${date} has a candle gap within the requested profiling range.`);
    }
  }

  const status = await vwKamaCudaStatus();
  if (!status.available) throw new Error(status.reason);
  const feeRate = execution.feeBps / 10_000;
  const samples: Array<{
    gridSize: number;
    kernelMs: number;
    wallMs: number;
    probabilitiesSha256?: string;
  }> = [];
  const latest = new Map<number, Awaited<ReturnType<typeof prepareExposureValueOracleCuda>>["oracle"]>();
  const firstProbabilities = new Map<number, Float32Array>();
  for (let iteration = 0; iteration < iterations; iteration += 1) {
    const orderedGridSizes = gridSizes.map((_, index) =>
      gridSizes[(index + iteration) % gridSizes.length]!,
    );
    for (const gridSize of orderedGridSizes) {
      const started = performance.now();
      const result = await prepareExposureValueOracleCuda(candles.map((candle) => candle.close), {
        scoreStartIndex: 0,
        holdingPeriodSteps,
        valueHorizonSteps,
        friction: feeRate,
        gridSize,
        minExposure: execution.minimumUsableExposure,
        maxExposure: execution.maximumUsableExposure,
        maxEffectiveExposure: Math.max(
          Math.abs(execution.minimumEffectiveExposure),
          Math.abs(execution.maximumEffectiveExposure),
        ),
        terminalIndex: candles.length - 1,
        temperature: execution.temperature,
        opportunityEpsilon: 0,
        quoteBorrowRate: bpsHourToPerSecond(maintenanceBps.quoteBorrow),
        assetBorrowRate: bpsHourToPerSecond(maintenanceBps.assetBorrow),
        includeActionValues: false,
        includeProbabilities: true,
        includePath: argument("include-path") !== "false",
        distributionOnly: argument("distribution-only") === "true",
      });
      const first = firstProbabilities.get(gridSize);
      const determinism = first && result.oracle.probabilities
        ? compare(first, result.oracle.probabilities)
        : undefined;
      if (!first && result.oracle.probabilities) {
        firstProbabilities.set(gridSize, result.oracle.probabilities.slice());
      }
      samples.push({
        gridSize,
        kernelMs: result.kernelMs,
        wallMs: performance.now() - started,
        probabilitiesSha256: result.oracle.probabilities
          ? sha256(result.oracle.probabilities)
          : undefined,
        ...determinism,
      });
      latest.set(gridSize, result.oracle);
    }
  }
  const referenceGridSize = Math.max(...gridSizes);
  const reference = latest.get(referenceGridSize)!;
  const probabilitiesOutput = argument("probabilities-output");
  if (probabilitiesOutput && reference.probabilities) {
    fs.writeFileSync(
      path.resolve(repoRoot, probabilitiesOutput),
      Buffer.from(
        reference.probabilities.buffer,
        reference.probabilities.byteOffset,
        reference.probabilities.byteLength,
      ),
    );
  }
  let diagnostics;
  if (argument("diagnostics") === "true" && reference.probabilities) {
    const cutoffStarted = performance.now();
    const cutoffs = await exposureHoldingCutoffsCuda(
      candles.map((candle) => candle.close),
      candles.length,
      holdingPeriodSteps,
      reference.execution,
    );
    const diagnosticStarted = performance.now();
    const measured = await directOracleDiagnosticsCuda(
      reference.probabilities,
      reference.grid,
      reference.currentGrid,
      cutoffs.cutoffLowers,
      cutoffs.cutoffUppers,
      {
        visibleLower: execution.minimumUsableExposure,
        visibleUpper: execution.maximumUsableExposure,
        friction: feeRate,
        transitionLogScale: 1 / execution.temperature,
        distanceEpsilon: 1e-6,
      },
    );
    diagnostics = {
      cutoffKernelMs: cutoffs.kernelMs,
      cutoffWallMs: diagnosticStarted - cutoffStarted,
      kernelMs: measured.kernelMs,
      wallMs: performance.now() - diagnosticStarted,
      entropiesSha256: sha256(measured.entropies),
      distanceImbalancesSha256: sha256(measured.distanceImbalances),
    };
  }
  const byGrid = Object.fromEntries(gridSizes.map((gridSize) => {
    const values = samples.filter((sample) => sample.gridSize === gridSize);
    const oracle = latest.get(gridSize)!;
    return [gridSize, {
      minimumKernelMs: Math.min(...values.map((sample) => sample.kernelMs)),
      meanKernelMs: mean(values.map((sample) => sample.kernelMs)),
      meanWallMs: mean(values.map((sample) => sample.wallMs)),
      zeroGridValue: oracle.grid[Math.floor(oracle.grid.length / 2)],
      finalLogReturn: oracle.path.logReturn,
      finalExposure: oracle.path.terminalExposure,
      probabilitiesSha256: oracle.probabilities ? sha256(oracle.probabilities) : undefined,
      versusReference: gridSize === referenceGridSize ? undefined : {
        policyMeanRmse: rmse(oracle.policyMeans, reference.policyMeans),
        modalExposureRmse: rmse(oracle.modalExposures, reference.modalExposures),
        averageRegretRmse: rmse(oracle.averageRegrets, reference.averageRegrets),
        pathExposureRmse: rmse(oracle.path.exposures, reference.path.exposures),
        finalLogReturnDifference: oracle.path.logReturn - reference.path.logReturn,
      },
    }];
  }));
  process.stdout.write(`${JSON.stringify({
    event: "oracle-profile",
    device: status.device,
    date,
    candles: candles.length,
    iterations,
    gridSizes,
    referenceGridSize,
    execution,
    holdingPeriodSteps,
    valueHorizonSteps,
    maintenanceBps,
    samples,
    byGrid,
    diagnostics,
  }, null, 2)}\n`);
}

function mean(values: readonly number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function sha256(values: ArrayBufferView): string {
  return createHash("sha256")
    .update(Buffer.from(values.buffer, values.byteOffset, values.byteLength))
    .digest("hex");
}

function compare(left: Float32Array, right: Float32Array): {
  mismatches: number;
  maximumAbsoluteDifference: number;
  firstMismatch: number;
} {
  let mismatches = 0;
  let maximumAbsoluteDifference = 0;
  let firstMismatch = -1;
  for (let index = 0; index < left.length; index += 1) {
    if (Object.is(left[index], right[index])) continue;
    if (firstMismatch < 0) firstMismatch = index;
    mismatches += 1;
    maximumAbsoluteDifference = Math.max(
      maximumAbsoluteDifference,
      Math.abs(left[index]! - right[index]!),
    );
  }
  return { mismatches, maximumAbsoluteDifference, firstMismatch };
}

function rmse(left: ArrayLike<number>, right: ArrayLike<number>): number {
  if (left.length !== right.length) throw new Error("Oracle comparison lengths differ.");
  let squared = 0;
  for (let index = 0; index < left.length; index += 1) {
    const difference = left[index]! - right[index]!;
    squared += difference * difference;
  }
  return Math.sqrt(squared / left.length);
}

function argument(name: string): string | undefined {
  const index = process.argv.indexOf(`--${name}`);
  return index >= 0 ? process.argv[index + 1] : undefined;
}

function positiveInteger(value: string, name: string): number {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 1) throw new Error(`--${name} must be a positive integer.`);
  return parsed;
}
