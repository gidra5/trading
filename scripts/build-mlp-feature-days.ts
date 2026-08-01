import fs from "node:fs/promises";
import path from "node:path";
import readline from "node:readline";
import { fileURLToPath } from "node:url";
import {
  MLP_FEATURE_SCHEMA_VERSION,
  MLP_INPUT_FEATURE_COUNT,
  type Candle,
} from "@trading/bot-algo";
import { SequentialShardStore, TradingStorageLayout } from "@trading/storage";
import { MlpFeatureStore } from "../apps/server/src/mlp-feature-store.js";

const DAY_MS = 86_400_000;
const SECOND_MS = 1_000;

interface Plan {
  dataDir: string;
}

interface Request {
  date: string;
  day: number;
  days: number;
}

void main();

async function main(): Promise<void> {
  const planFile = requiredArgument("plan");
  const output = path.resolve(requiredArgument("output"));
  const plan = JSON.parse(await fs.readFile(planFile, "utf8")) as Plan;
  const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
  const dataDir = path.resolve(repoRoot, plan.dataDir);
  const store = new MlpFeatureStore(dataDir);
  const shardStore = new SequentialShardStore(
    new TradingStorageLayout(dataDir).trainingStore,
  );
  const input = readline.createInterface({ input: process.stdin });

  for await (const line of input) {
    if (!line) continue;
    const request = JSON.parse(line) as Request;
    const day = Date.parse(`${request.date}T00:00:00.000Z`);
    if (!Number.isFinite(day) || day % DAY_MS !== 0) {
      throw new Error(`Invalid feature day '${request.date}'.`);
    }
    const source = await store.loadSecondRange(day - DAY_MS, day + DAY_MS);
    validateSource(source, day, request.date);
    const scored = source.slice(DAY_MS / SECOND_MS);
    const times = scored.map((candle) => candle.closeTime);
    const features = await store.prepare(source, times);
    const encodingStarted = performance.now();
    const encoded = encodeFeatureRows(times, features);
    const encodingMs = performance.now() - encodingStarted;
    const compressionStarted = performance.now();
    const prefix = path.join("components", "inputs", request.date);
    const featureFile = `${prefix}.features.json`;
    const stored = await shardStore.put({
      namespace: `features/mlp-${MLP_FEATURE_SCHEMA_VERSION}`,
      key: request.date,
      payload: encoded,
      sequence: {
        start: times[0]!,
        step: SECOND_MS,
        count: times.length,
        unit: "unix-ms",
      },
      layout: {
        encoding: "row-major",
        dtype: "float16-le",
        rows: times.length,
        columns: MLP_INPUT_FEATURE_COUNT,
        featureSchemaVersion: MLP_FEATURE_SCHEMA_VERSION,
      },
      metadata: {
        symbol: "BTCUSDT",
        interval: "1s",
        role: "model-input-features",
      },
      compressionLevel: 9,
    });
    const datasetReference = path.join(output, featureFile);
    await fs.mkdir(path.dirname(datasetReference), { recursive: true });
    await fs.writeFile(
      datasetReference,
      `${JSON.stringify(stored.reference, null, 2)}\n`,
      { flag: "wx" },
    );
    const compressionMs = performance.now() - compressionStarted;
    const persistStarted = performance.now();
    const persistMs = performance.now() - persistStarted;
    process.stdout.write(`${JSON.stringify({
      event: "dataset-component",
      component: "inputs",
      date: request.date,
      count: DAY_MS / SECOND_MS,
      features: featureFile,
      featuresCompression: "zstd",
      featuresUncompressedBytes: encoded.byteLength,
      timeSequence: {
        encoding: "implicit-linear",
        startTimeMs: times[0]!,
        stepMs: SECOND_MS,
        count: times.length,
      },
      featureSchemaVersion: MLP_FEATURE_SCHEMA_VERSION,
      rowSelectionSignature: "full",
      day: request.day,
      days: request.days,
      encodingMs,
      compressionMs,
      persistMs,
      compressedBytes: stored.reference.object.compressedBytes,
      contentHash: stored.reference.object.contentHash,
      canonicalReference: path.relative(dataDir, stored.referenceFile),
    })}\n`);
  }
}

function requiredArgument(name: string): string {
  const index = process.argv.indexOf(`--${name}`);
  const value = index >= 0 ? process.argv[index + 1] : undefined;
  if (!value) {
    throw new Error(
      "Usage: build-mlp-feature-days.ts --plan <plan.json> --output <dataset>",
    );
  }
  return path.resolve(value);
}

function validateSource(
  source: readonly Candle[],
  day: number,
  date: string,
): void {
  const expected = 2 * DAY_MS / SECOND_MS;
  if (source.length !== expected) {
    throw new Error(`${date} feature source has ${source.length}/${expected} seconds.`);
  }
  const start = day - DAY_MS;
  for (let index = 0; index < source.length; index += 1) {
    if (source[index]!.openTime !== start + index * SECOND_MS) {
      throw new Error(`${date} feature source is non-contiguous at row ${index}.`);
    }
  }
}

function encodeFeatureRows(
  times: readonly number[],
  features: Awaited<ReturnType<MlpFeatureStore["prepare"]>>,
): Buffer {
  const half = new Uint16Array(times.length * MLP_INPUT_FEATURE_COUNT);
  if (features.encodeHalfRows) {
    features.encodeHalfRows(times, half);
  } else {
    for (let row = 0; row < times.length; row += 1) {
      features.encodeHalf(times[row]!, half, row * MLP_INPUT_FEATURE_COUNT);
    }
  }
  return Buffer.from(half.buffer, half.byteOffset, half.byteLength);
}
