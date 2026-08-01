import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { readCandleShardReferenceSync } from "@trading/storage";
import {
  prepareExposureValueOracleCuda,
  vwKamaCudaStatus,
  type Candle,
} from "@trading/bot-algo";

const DAY_MS = 86_400_000;

interface RejectionCase {
  split: "train" | "validation" | "test";
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

interface RejectionQueue {
  planId: string;
  cases: RejectionCase[];
}

interface TrainingPlan {
  id: string;
  dataDir: string;
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
}

interface SelectedCase extends RejectionCase {
  candleIndex: number;
  price: number;
  return1m: number;
  return1h: number;
  realizedVolatility1h: number;
  dayReturn: number;
  oracleOffset: number;
}

void main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : error);
  process.exitCode = 1;
});

async function main(): Promise<void> {
  const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
  const planFile = path.resolve(repoRoot, argument("plan") ?? "ml/training-plan.json");
  const plan = JSON.parse(fs.readFileSync(planFile, "utf8")) as TrainingPlan;
  const queueFile = path.resolve(
    repoRoot,
    argument("queue") ?? path.join(
      "data/training/datasets", plan.id, "state", "teacher-refinement-queue.json",
    ),
  );
  const output = path.resolve(
    repoRoot,
    argument("output") ?? "data/training/analysis/mlp-rejection-oracles",
  );
  const dateCount = positiveInteger(argument("dates") ?? "12", "dates");
  const casesPerDate = positiveInteger(argument("cases-per-date") ?? "16", "cases-per-date");
  const refinementPass = Number(argument("refinement-pass") ?? "1");
  const gridSize = positiveInteger(
    argument("grid-size") ?? String(plan.execution.gridSize),
    "grid-size",
  );
  const queue = JSON.parse(fs.readFileSync(queueFile, "utf8")) as RejectionQueue;
  if (queue.planId !== plan.id) throw new Error(`Queue ${queue.planId} does not match ${plan.id}.`);
  const candidates = queue.cases.filter((item) => item.refinementPass === refinementPass);
  if (candidates.length === 0) throw new Error(`No refinement-pass ${refinementPass} cases remain.`);
  const selectedDates = selectDates(candidates, dateCount);
  const selected = selectedDates.flatMap((date) =>
    quantileSample(candidates.filter((item) => item.date === date), casesPerDate));
  selected.sort((left, right) => left.time - right.time);

  const status = await vwKamaCudaStatus();
  if (!status.available) throw new Error(status.reason);
  fs.mkdirSync(output, { recursive: true });
  const sourceRoot = path.resolve(
    repoRoot,
    plan.dataDir,
    "market/immutable/refs/candles/spot-btcusdt/btcusdt/1s",
  );
  const probabilityParts: Buffer[] = [];
  const outputCases: SelectedCase[] = [];
  let actionGrid: number[] | undefined;
  let currentGrid: number[] | undefined;
  let oracleOffset = 0;
  const feeRate = plan.execution.feeBps / 10_000;
  for (const [dateIndex, date] of selectedDates.entries()) {
    const candles = readCandles(sourceRoot, date);
    inspectCompleteDay(candles, date);
    const prepared = await prepareExposureValueOracleCuda(candles.map((candle) => candle.close), {
      scoreStartIndex: 0,
      holdingPeriodSteps: plan.execution.holdingPeriodSteps,
      decisionDelaySteps: plan.execution.decisionDelaySteps ?? 1,
      valueHorizonSteps: plan.execution.valueHorizonSteps,
      friction: feeRate,
      gridSize,
      minExposure: plan.execution.minimumUsableExposure,
      maxExposure: plan.execution.maximumUsableExposure,
      maxEffectiveExposure: Math.max(
        Math.abs(plan.execution.minimumEffectiveExposure),
        Math.abs(plan.execution.maximumEffectiveExposure),
      ),
      terminalIndex: candles.length - 1,
      temperature: plan.execution.temperature,
      opportunityEpsilon: 0,
      quoteBorrowRate: bpsHourToPerSecond(plan.execution.maintenanceBpsHour.quoteBorrow),
      assetBorrowRate: bpsHourToPerSecond(plan.execution.maintenanceBpsHour.assetBorrow),
      includeActionValues: false,
      includeProbabilities: true,
    });
    const oracle = prepared.oracle;
    if (!oracle.probabilities) throw new Error(`Oracle probabilities missing for ${date}.`);
    actionGrid ??= Array.from(oracle.grid);
    currentGrid ??= Array.from(oracle.currentGrid);
    const indexByTime = new Map(candles.map((candle, index) => [candle.closeTime, index]));
    const dayCases = selected.filter((item) => item.date === date);
    for (const item of dayCases) {
      const candleIndex = indexByTime.get(item.time);
      if (candleIndex === undefined) {
        throw new Error(`${date} has no source candle whose close time is ${item.time}.`);
      }
      const begin = candleIndex * oracle.grid.length;
      probabilityParts.push(Buffer.from(
        oracle.probabilities.buffer,
        oracle.probabilities.byteOffset + begin * Float32Array.BYTES_PER_ELEMENT,
        oracle.grid.length * Float32Array.BYTES_PER_ELEMENT,
      ));
      const closes = candles.map((candle) => candle.close);
      outputCases.push({
        ...item,
        candleIndex,
        price: closes[candleIndex]!,
        return1m: trailingReturn(closes, candleIndex, 60),
        return1h: trailingReturn(closes, candleIndex, 3_600),
        realizedVolatility1h: trailingVolatility(closes, candleIndex, 3_600),
        dayReturn: trailingReturn(closes, candleIndex, candleIndex),
        oracleOffset,
      });
      oracleOffset += oracle.grid.length;
    }
    process.stdout.write(`${JSON.stringify({
      event: "rejection-oracle-day",
      date,
      day: dateIndex + 1,
      days: selectedDates.length,
      cases: dayCases.length,
      kernelMs: prepared.kernelMs,
    })}\n`);
  }
  const probabilitiesFile = path.join(output, "base-probabilities.f32");
  fs.writeFileSync(probabilitiesFile, Buffer.concat(probabilityParts));
  const global = summarizeQueue(candidates);
  const metadata = {
    version: 1,
    createdAt: new Date().toISOString(),
    planId: plan.id,
    refinementPass,
    device: status.device,
    queueFile,
    queueSummary: global,
    selection: {
      method: "date mean-KL quantiles; within-date KL quantiles",
      requestedDates: dateCount,
      requestedCasesPerDate: casesPerDate,
      selectedDates,
      cases: outputCases.length,
    },
    actionGrid,
    currentGrid,
    execution: { ...plan.execution, gridSize },
    probabilitiesFile: path.basename(probabilitiesFile),
    cases: outputCases,
  };
  fs.writeFileSync(path.join(output, "metadata.json"), `${JSON.stringify(metadata, null, 2)}\n`);
  process.stdout.write(`${JSON.stringify({ event: "rejection-oracles-complete", output, ...global })}\n`);
}

function selectDates(cases: RejectionCase[], requested: number): string[] {
  const grouped = new Map<string, RejectionCase[]>();
  for (const item of cases) grouped.set(item.date, [...(grouped.get(item.date) ?? []), item]);
  const dates = [...grouped.entries()].map(([date, items]) => ({
    date,
    score: items.reduce((sum, item) => sum + item.klDivergence, 0) / items.length,
  })).sort((left, right) => left.score - right.score);
  return quantileSample(dates, Math.min(requested, dates.length)).map((item) => item.date).sort();
}

function quantileSample<T extends { klDivergence?: number; score?: number }>(
  values: T[],
  requested: number,
): T[] {
  if (values.length <= requested) return [...values];
  const sorted = [...values].sort((left, right) =>
    (left.klDivergence ?? left.score ?? 0) - (right.klDivergence ?? right.score ?? 0));
  const indexes = new Set<number>();
  for (let index = 0; index < requested; index += 1) {
    indexes.add(Math.round(index * (sorted.length - 1) / Math.max(1, requested - 1)));
  }
  return [...indexes].map((index) => sorted[index]!);
}

function summarizeQueue(cases: RejectionCase[]): Record<string, number> {
  const kl = cases.map((item) => item.klDivergence).sort((left, right) => left - right);
  const mse = cases.map((item) => item.meanSquaredError).sort((left, right) => left - right);
  const percentile = (values: number[], quantile: number): number =>
    values[Math.round((values.length - 1) * quantile)]!;
  return {
    cases: cases.length,
    dates: new Set(cases.map((item) => item.date)).size,
    convergedFraction: cases.filter((item) => item.converged).length / cases.length,
    meanKlDivergence: kl.reduce((sum, value) => sum + value, 0) / kl.length,
    medianKlDivergence: percentile(kl, 0.5),
    p90KlDivergence: percentile(kl, 0.9),
    p99KlDivergence: percentile(kl, 0.99),
    meanSquaredError: mse.reduce((sum, value) => sum + value, 0) / mse.length,
    medianSquaredError: percentile(mse, 0.5),
    p90SquaredError: percentile(mse, 0.9),
    p99SquaredError: percentile(mse, 0.99),
  };
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

function trailingReturn(prices: number[], index: number, distance: number): number {
  const first = prices[Math.max(0, index - distance)]!;
  return prices[index]! / first - 1;
}

function trailingVolatility(prices: number[], index: number, distance: number): number {
  const start = Math.max(1, index - distance + 1);
  let sum = 0;
  let squareSum = 0;
  let count = 0;
  for (let candle = start; candle <= index; candle += 1) {
    const value = Math.log(prices[candle]! / prices[candle - 1]!);
    sum += value;
    squareSum += value * value;
    count += 1;
  }
  if (count < 2) return 0;
  return Math.sqrt(Math.max(0, squareSum / count - (sum / count) ** 2));
}

function bpsHourToPerSecond(bps: number): number {
  return Math.expm1(Math.log1p(bps / 10_000) / 3_600);
}

function argument(name: string): string | undefined {
  const index = process.argv.indexOf(`--${name}`);
  return index >= 0 ? process.argv[index + 1] : undefined;
}

function positiveInteger(value: string, name: string): number {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 1) throw new Error(`--${name} must be positive.`);
  return parsed;
}
