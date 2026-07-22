import fs from "node:fs";
import path from "node:path";
import { performance } from "node:perf_hooks";
import { fileURLToPath } from "node:url";
import { gunzipSync } from "node:zlib";
import {
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
  const samples: Array<{ gridSize: number; kernelMs: number; wallMs: number }> = [];
  const latest = new Map<number, Awaited<ReturnType<typeof prepareExposureValueOracleCuda>>["oracle"]>();
  for (let iteration = 0; iteration < iterations; iteration += 1) {
    const orderedGridSizes = gridSizes.map((_, index) =>
      gridSizes[(index + iteration) % gridSizes.length]!,
    );
    for (const gridSize of orderedGridSizes) {
      const started = performance.now();
      const result = await prepareExposureValueOracleCuda(candles.map((candle) => candle.close), {
        scoreStartIndex: 0,
        holdingPeriodSteps: execution.holdingPeriodSteps,
        valueHorizonSteps: execution.valueHorizonSteps,
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
        quoteLendRate: bpsHourToPerSecond(execution.maintenanceBpsHour.quoteLend),
        quoteBorrowRate: bpsHourToPerSecond(execution.maintenanceBpsHour.quoteBorrow),
        assetBorrowRate: bpsHourToPerSecond(execution.maintenanceBpsHour.assetBorrow),
        includeActionValues: false,
        includeProbabilities: true,
      });
      samples.push({ gridSize, kernelMs: result.kernelMs, wallMs: performance.now() - started });
      latest.set(gridSize, result.oracle);
    }
  }
  const referenceGridSize = Math.max(...gridSizes);
  const reference = latest.get(referenceGridSize)!;
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
    samples,
    byGrid,
  }, null, 2)}\n`);
}

function mean(values: readonly number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
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
