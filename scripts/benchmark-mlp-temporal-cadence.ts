import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { readCandleShardReferenceSync } from "@trading/storage";
import {
  conditionalCutoffRawParameters,
  exposureHoldingFeasibleInterval,
  prepareExposureValueOracleCuda,
  type Candle,
} from "@trading/bot-algo";

const SECOND_MS = 1_000;

void main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : error);
  process.exitCode = 1;
});

async function main(): Promise<void> {
  const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
  const planFile = path.resolve(repoRoot, argument("plan") ?? "ml/training-plan.json");
  const plan = JSON.parse(fs.readFileSync(planFile, "utf8"));
  const date = argument("date") ?? "2026-07-09";
  const startSecond = positiveInteger(argument("start-second") ?? "43200", "start-second", true);
  const count = positiveInteger(argument("examples") ?? "4096", "examples");
  const output = path.resolve(
    repoRoot,
    argument("output") ?? "data/training/analysis/mlp-temporal-cadence-1s",
  );
  const execution = plan.execution;
  const sourceRoot = path.resolve(
    repoRoot,
    plan.dataDir,
    "market/immutable/refs/candles/spot-btcusdt/btcusdt/1s",
  );
  const candles = readCandleDay(sourceRoot, date);
  const required = count + execution.valueHorizonSteps;
  const selected = candles.slice(startSecond, startSecond + required);
  if (selected.length !== required) {
    throw new Error(
      `${date} supplies ${selected.length}/${required} candles after second ${startSecond}.`,
    );
  }
  for (let index = 1; index < selected.length; index += 1) {
    if (selected[index]!.openTime !== selected[index - 1]!.openTime + SECOND_MS) {
      throw new Error(`${date} has a candle gap at benchmark row ${index}.`);
    }
  }

  const feeRate = execution.feeBps / 10_000;
  const prices = Float64Array.from(selected, (candle) => candle.close);
  const oracleStarted = performance.now();
  const prepared = await prepareExposureValueOracleCuda(prices, {
    scoreStartIndex: 0,
    holdingPeriodSteps: execution.holdingPeriodSteps,
    decisionDelaySteps: execution.decisionDelaySteps ?? 1,
    valueHorizonSteps: execution.valueHorizonSteps,
    friction: feeRate,
    gridSize: execution.gridSize,
    minExposure: execution.minimumEffectiveExposure,
    maxExposure: execution.maximumEffectiveExposure,
    maxEffectiveExposure: Math.max(
      Math.abs(execution.minimumEffectiveExposure),
      Math.abs(execution.maximumEffectiveExposure),
    ),
    terminalIndex: count - 1,
    temperature: execution.temperature,
    opportunityEpsilon: 0,
    quoteBorrowRate: bpsHourToPerSecond(execution.maintenanceBpsHour.quoteBorrow),
    assetBorrowRate: bpsHourToPerSecond(execution.maintenanceBpsHour.assetBorrow),
    includeActionValues: false,
    includeProbabilities: true,
  });
  const oracle = prepared.oracle;
  if (!oracle.probabilities) throw new Error("CUDA oracle did not retain probabilities.");
  const actionCount = oracle.grid.length;
  if (oracle.probabilities.length < count * actionCount
    || oracle.probabilities.length % actionCount !== 0) {
    throw new Error(
      `CUDA oracle returned ${oracle.probabilities.length} probability values; `
      + `expected at least ${count * actionCount} complete-grid values.`,
    );
  }
  const alignment = plan.teacherFit.inputAlignmentFloats;
  const rowStride = Math.ceil((actionCount + 2) / alignment) * alignment;
  const packed = new Float32Array(count * rowStride);
  for (let row = 0; row < count; row += 1) {
    const cutoff = exposureHoldingFeasibleInterval(
      prices,
      row,
      execution.holdingPeriodSteps,
      {
        friction: feeRate,
        minExposure: execution.minimumUsableExposure,
        maxExposure: execution.maximumUsableExposure,
        maxEffectiveExposure: Math.max(
          Math.abs(execution.minimumEffectiveExposure),
          Math.abs(execution.maximumEffectiveExposure),
        ),
        quoteBorrowRate: bpsHourToPerSecond(execution.maintenanceBpsHour.quoteBorrow),
        assetBorrowRate: bpsHourToPerSecond(execution.maintenanceBpsHour.assetBorrow),
      },
    );
    const destination = row * rowStride;
    let total = 0;
    for (let action = 0; action < actionCount; action += 1) {
      const probability = oracle.grid[action]! >= cutoff.lower
          && oracle.grid[action]! <= cutoff.upper
        ? oracle.probabilities[row * actionCount + action]!
        : 0;
      packed[destination + action] = probability;
      total += probability;
    }
    if (!(total > 0)) throw new Error(`Cutoff removed every action at row ${row}.`);
    for (let action = 0; action < actionCount; action += 1) {
      packed[destination + action] /= total;
    }
    const cutoffRaw = conditionalCutoffRawParameters(cutoff.lower, cutoff.upper, {
      latentLower: execution.minimumEffectiveExposure,
      latentUpper: execution.maximumEffectiveExposure,
    });
    packed[destination + actionCount] = cutoffRaw[0];
    packed[destination + actionCount + 1] = cutoffRaw[1];
  }

  fs.mkdirSync(output, { recursive: true });
  const inputFile = "teacher-inputs.f32";
  fs.writeFileSync(
    path.join(output, inputFile),
    Buffer.from(packed.buffer, packed.byteOffset, packed.byteLength),
  );
  fs.writeFileSync(path.join(output, "metadata.json"), `${JSON.stringify({
    version: 1,
    planId: plan.id,
    date,
    startSecond,
    startTime: selected[0]!.closeTime,
    samplingIntervalMs: SECOND_MS,
    count,
    rowStride,
    inputFile,
    actionGrid: Array.from(oracle.grid),
    currentGrid: Array.from(oracle.currentGrid),
    oracleKernelMs: prepared.kernelMs,
    oracleWallMs: performance.now() - oracleStarted,
  }, null, 2)}\n`);
  process.stdout.write(`${JSON.stringify({
    event: "temporal-cadence-corpus",
    output,
    date,
    startSecond,
    examples: count,
    actionCells: actionCount,
    rowStride,
    oracleKernelMs: prepared.kernelMs,
    oracleWallMs: performance.now() - oracleStarted,
  })}\n`);

  const python = path.join(
    repoRoot,
    ".venv-ml",
    process.platform === "win32" ? "Scripts/python.exe" : "bin/python",
  );
  const result = spawnSync(python, [
    path.join(repoRoot, "ml/benchmark_temporal_cadence.py"),
    "--input", output,
    "--plan", planFile,
    "--cadences", argument("cadences") ?? "1,5,15,30,60,300,1800",
    "--device", argument("device") ?? "cuda",
  ], {
    cwd: repoRoot,
    env: { ...process.env, PYTHONPATH: path.join(repoRoot, "ml") },
    stdio: "inherit",
  });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    throw new Error(`Temporal cadence benchmark exited with code ${result.status}.`);
  }
}

function readCandleDay(root: string, date: string): Candle[] {
  const candles = readCandleShardReferenceSync(path.join(root, `${date}.json`));
  if (candles.length !== 86_400) {
    throw new Error(`${date} has ${candles.length}/86400 one-second candles.`);
  }
  return candles;
}

function bpsHourToPerSecond(bps: number): number {
  return Math.expm1(Math.log1p(bps / 10_000) / 3_600);
}

function argument(name: string): string | undefined {
  const index = process.argv.indexOf(`--${name}`);
  return index >= 0 ? process.argv[index + 1] : undefined;
}

function positiveInteger(value: string, name: string, allowZero = false): number {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < (allowZero ? 0 : 1)) {
    throw new Error(`--${name} must be ${allowZero ? "a non-negative" : "a positive"} integer.`);
  }
  return parsed;
}
