import fs from "node:fs/promises";
import path from "node:path";
import readline from "node:readline";
import { fileURLToPath } from "node:url";
import { constants as zlibConstants, zstdCompressSync } from "node:zlib";
import {
  MLP_FEATURE_SCHEMA_VERSION,
  MLP_INPUT_FEATURE_COUNT,
  type Candle,
} from "@trading/bot-algo";
import { MlpFeatureStore } from "../apps/server/src/mlp-feature-store.js";

const DAY_MS = 86_400_000;
const SECOND_MS = 1_000;
const FLOAT32_TO_FLOAT16_SCRATCH = new Float32Array(1);
const FLOAT32_TO_FLOAT16_BITS = new Uint32Array(FLOAT32_TO_FLOAT16_SCRATCH.buffer);

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
    const started = performance.now();
    const encoded = encodeFeatureRows(times, features);
    const compressed = zstdCompressSync(encoded, {
      params: {
        [zlibConstants.ZSTD_c_compressionLevel]: 3,
      },
    });
    const prefix = path.join("components", "inputs", request.date);
    const featureFile = `${prefix}.features.f16.zst`;
    const timeFile = `${prefix}.times.i64`;
    await Promise.all([
      writeAtomic(path.join(output, featureFile), compressed),
      writeAtomic(path.join(output, timeFile), encodeTimes(times)),
    ]);
    process.stdout.write(`${JSON.stringify({
      event: "dataset-component",
      component: "inputs",
      date: request.date,
      count: DAY_MS / SECOND_MS,
      features: featureFile,
      featuresCompression: "zstd",
      featuresUncompressedBytes: encoded.byteLength,
      times: timeFile,
      featureSchemaVersion: MLP_FEATURE_SCHEMA_VERSION,
      rowSelectionSignature: "full",
      day: request.day,
      days: request.days,
      encodingMs: performance.now() - started,
      compressedBytes: compressed.byteLength,
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

function encodeTimes(times: readonly number[]): Buffer {
  const buffer = Buffer.allocUnsafe(times.length * 8);
  for (let index = 0; index < times.length; index += 1) {
    buffer.writeBigInt64LE(BigInt(times[index]!), index * 8);
  }
  return buffer;
}

function float32ToFloat16(value: number): number {
  FLOAT32_TO_FLOAT16_SCRATCH[0] = value;
  const raw = FLOAT32_TO_FLOAT16_BITS[0]!;
  const sign = raw >>> 16 & 0x8000;
  let exponent = (raw >>> 23 & 0xff) - 127 + 15;
  let mantissa = raw & 0x7fffff;
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

async function writeAtomic(file: string, value: Uint8Array): Promise<void> {
  await fs.mkdir(path.dirname(file), { recursive: true });
  const temporary = `${file}.${process.pid}.tmp`;
  await fs.writeFile(temporary, value);
  await fs.rename(temporary, file);
}
