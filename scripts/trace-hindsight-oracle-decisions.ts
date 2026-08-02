import fs from "node:fs";
import path from "node:path";
import type {
  Candle,
  ExposureValueOracleActionDistribution,
} from "@trading/bot-algo";
import {
  readCandleShardReferenceSync,
  readReferencedPayloadSync,
} from "@trading/storage";
import { appConfig } from "../apps/server/src/config.js";
import {
  runBotBacktestFromCandles,
  type OracleBacktestDecision,
} from "../apps/server/src/bot-backtest.js";

const DAY_MS = 86_400_000;
const TEST_DAYS = 30;
const ORACLE_NAMESPACE = "oracle/1s/hindsight-bot-71391c44b323e044e6ab";
const ORACLE_CONTRACT_HASH = "71391c44b323e044e6ab0015941824ba53e90f5bcaaf9fee48288df3a10ec49b";
const HISTORY_ROOT = path.resolve(
  "data/market/immutable/refs/candles/spot-btcusdt/btcusdt/1s",
);
const TARGET_ROOT = path.resolve(
  "data/training/immutable/refs/oracle/1s/hindsight-bot-71391c44b323e044e6ab",
);

void main();

async function main(): Promise<void> {
  const split = choiceArgument("split", ["validation", "test"] as const, "validation");
  if (split === "test" && !process.argv.includes("--allow-test")) {
    throw new Error("Tracing the locked test split requires explicit --allow-test.");
  }
  const days = positiveIntegerArgument("days", 30);
  if (days > TEST_DAYS) {
    throw new Error(`--days cannot exceed the ${TEST_DAYS}-day split boundary.`);
  }
  const maximumLeverage = optionalPositiveNumberArgument("max-leverage") ?? 100;
  const allTargets = fs.readdirSync(TARGET_ROOT)
    .filter((file) => /^\d{4}-\d{2}-\d{2}\.json$/.test(file))
    .sort();
  const targetFiles = split === "test"
    ? allTargets.slice(-days)
    : allTargets.slice(-(TEST_DAYS + days), -TEST_DAYS);
  if (targetFiles.length !== days) {
    throw new Error(`Target cache has ${targetFiles.length}/${days} requested ${split} days.`);
  }
  const dates = targetFiles.map((file) => file.slice(0, -5));
  const firstDay = Date.parse(`${dates[0]}T00:00:00Z`);
  const warmupDate = new Date(firstDay - DAY_MS).toISOString().slice(0, 10);
  const warmup = loadCandles(warmupDate).slice(-3_600);
  const candles = dates.flatMap(loadCandles);
  if (warmup.length !== 3_600 || candles.length !== days * 86_400) {
    throw new Error("Teacher-trace candle history is incomplete.");
  }

  const decisions: OracleBacktestDecision[] = [];
  const startedAt = new Date().toISOString();
  const result = await runBotBacktestFromCandles(candles, {
    config: appConfig.strategy,
    strategy: "hindsight-oracle-1s",
    warmup,
    hindsightOracleDistributionAt: storedOracleProvider(dates),
    hindsightOracleMaximumLeverage: maximumLeverage,
    summaryOnly: true,
    onOracleDecision: (decision) => decisions.push(decision),
    onProgress: ({ candlesProcessed, totalCandles, elapsedMs }) => {
      if (candlesProcessed % Math.ceil(totalCandles / 20) !== 0
        && candlesProcessed !== totalCandles) return;
      console.log(JSON.stringify({
        event: "teacher-trace-progress",
        candlesProcessed,
        totalCandles,
        elapsedMs,
      }));
    },
  });
  const expectedDecisions = days * 1_440;
  if (decisions.length !== expectedDecisions) {
    throw new Error(`Teacher trace has ${decisions.length}/${expectedDecisions} decisions.`);
  }
  const output = path.resolve(
    "data/training/derived/joint-price-oracle/teacher-traces",
    `${split}-${dates[0]}-${dates.at(-1)}-leverage-${fileNumber(maximumLeverage)}.json`,
  );
  atomicJson(output, {
    version: 2,
    kind: "exact-hindsight-oracle-bot-teacher-trace",
    createdAt: new Date().toISOString(),
    startedAt,
    split,
    dates,
    oracle: {
      valueHorizonSteps: 3_600,
      decisionDelaySteps: 60,
      holdingPeriodSteps: 60,
      maximumLeverage,
    },
    decisions,
    summary: result.summary,
  });
  console.log(JSON.stringify({
    event: "teacher-trace-complete",
    output,
    decisions: decisions.length,
    summary: result.summary,
  }));
}

function loadCandles(date: string): Candle[] {
  const file = path.join(HISTORY_ROOT, `${date}.json`);
  if (!fs.existsSync(file)) throw new Error(`Missing one-second candle day ${date}.`);
  return readCandleShardReferenceSync(file);
}

function storedOracleProvider(
  dates: readonly string[],
): (timestamp: number) => ExposureValueOracleActionDistribution | null {
  const rows = new Map<number, {
    values: Float32Array;
    offset: number;
    grid: number[];
  }>();
  for (const date of dates) {
    const file = path.join(TARGET_ROOT, `${date}.json`);
    const { reference, payload } = readReferencedPayloadSync(file);
    const columns = Number(reference.layout.columns);
    const grid = (reference.metadata as { contract?: { usableGrid?: number[] } } | undefined)
      ?.contract?.usableGrid;
    const metadata = reference.metadata as {
      contractHash?: string;
      contract?: {
        intervalMs?: number;
        decisionIntervalMs?: number;
        options?: {
          holdingPeriodSteps?: number;
          decisionDelaySteps?: number;
          valueHorizonSteps?: number;
          temperature?: number;
        };
      };
    } | undefined;
    const expectedStart = Date.parse(`${date}T00:00:00Z`) + 999;
    const copied = payload.buffer.slice(
      payload.byteOffset,
      payload.byteOffset + payload.byteLength,
    );
    const values = new Float32Array(copied);
    if (reference.kind !== "trading-sequential-shard"
      || reference.namespace !== ORACLE_NAMESPACE
      || reference.key !== date
      || metadata?.contractHash !== ORACLE_CONTRACT_HASH
      || metadata.contract?.intervalMs !== 1_000
      || metadata.contract?.decisionIntervalMs !== 60_000
      || metadata.contract?.options?.holdingPeriodSteps !== 60
      || metadata.contract?.options?.decisionDelaySteps !== 60
      || metadata.contract?.options?.valueHorizonSteps !== 3_600
      || metadata.contract?.options?.temperature !== 0.01
      || reference.sequence.start !== expectedStart
      || reference.sequence.step !== 60_000
      || reference.sequence.count !== 1_440
      || reference.sequence.unit !== "unix-ms"
      || reference.layout.encoding !== "raw-row-major"
      || reference.layout.dtype !== "float32-le"
      || reference.layout.rows !== 1_440
      || columns !== 101
      || values.length !== reference.sequence.count * columns
      || grid?.length !== columns) {
      throw new Error(`Oracle target layout is incompatible: ${file}`);
    }
    for (let row = 0; row < reference.sequence.count; row += 1) {
      const timestamp = reference.sequence.start + row * reference.sequence.step;
      if (rows.has(timestamp)) {
        throw new Error(`Oracle target timeline overlaps at ${timestamp}: ${file}`);
      }
      rows.set(timestamp, {
        values,
        offset: row * columns,
        grid,
      });
    }
  }
  return (timestamp) => {
    const row = rows.get(timestamp);
    return row
      ? probabilityDistribution(
          row.values.subarray(row.offset, row.offset + row.grid.length),
          row.grid,
        )
      : null;
  };
}

function probabilityDistribution(
  source: ArrayLike<number>,
  sourceGrid: readonly number[],
): ExposureValueOracleActionDistribution {
  const probabilities = Float64Array.from(source);
  const grid = Float64Array.from(sourceGrid);
  let total = 0;
  for (const probability of probabilities) {
    if (!Number.isFinite(probability) || probability < 0) {
      throw new Error("Oracle target row contains an invalid probability.");
    }
    total += probability;
  }
  if (!(total > 0) || !Number.isFinite(total) || Math.abs(total - 1) > 1e-3) {
    throw new Error(`Oracle target row has invalid probability mass ${total}.`);
  }
  let modalIndex = 0;
  let mean = 0;
  let secondMoment = 0;
  let entropy = 0;
  let feasibleActionCount = 0;
  for (let index = 0; index < probabilities.length; index += 1) {
    probabilities[index] /= total;
    const probability = probabilities[index]!;
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

function positiveIntegerArgument(name: string, fallback: number): number {
  const prefix = `--${name}=`;
  const value = Number(
    process.argv.find((argument) => argument.startsWith(prefix))?.slice(prefix.length)
      ?? fallback,
  );
  if (!Number.isInteger(value) || value < 1) {
    throw new Error(`--${name} must be a positive integer.`);
  }
  return value;
}

function optionalPositiveNumberArgument(name: string): number | undefined {
  const prefix = `--${name}=`;
  const raw = process.argv.find((argument) => argument.startsWith(prefix))
    ?.slice(prefix.length);
  if (raw === undefined) return undefined;
  const value = Number(raw);
  if (!(value > 0) || !Number.isFinite(value)) {
    throw new Error(`--${name} must be finite and positive.`);
  }
  return value;
}

function choiceArgument<const T extends readonly string[]>(
  name: string,
  choices: T,
  fallback: T[number],
): T[number] {
  const prefix = `--${name}=`;
  const value = process.argv.find((argument) => argument.startsWith(prefix))
    ?.slice(prefix.length) ?? fallback;
  if (!choices.includes(value)) {
    throw new Error(`--${name} must be one of ${choices.join(", ")}.`);
  }
  return value as T[number];
}

function atomicJson(file: string, value: unknown): void {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const temporary = `${file}.${process.pid}.tmp`;
  fs.writeFileSync(temporary, `${JSON.stringify(value)}\n`);
  fs.renameSync(temporary, file);
}

function fileNumber(value: number): string {
  return String(value).replaceAll("-", "minus-").replaceAll(".", "_");
}
