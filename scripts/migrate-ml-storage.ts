import fs from "node:fs/promises";
import path from "node:path";
import { gunzipSync, zstdDecompressSync } from "node:zlib";
import { fileURLToPath } from "node:url";
import {
  putCandleShard,
  SequentialShardStore,
  TradingStorageLayout,
  type SequentialCandle,
} from "@trading/storage";

const args = process.argv.slice(2);
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const dataRoot = path.resolve(argument("data-dir") ?? path.join(repoRoot, "data"));
const layout = new TradingStorageLayout(dataRoot);
const apply = args.includes("--apply");
const all = args.includes("--all");
const liveCandles = args.includes("--live-candles");
const requestedDataset = argument("dataset");
const DAY_MS = 86_400_000;
const DATASET_MIGRATION_CONCURRENCY = 3;

void main();

async function main(): Promise<void> {
  if (!all && !liveCandles && !requestedDataset) {
    throw new Error(
      "Select --all, --live-candles, or --dataset <directory>. Migration is a dry run by default.",
    );
  }
  if (apply && !args.includes("--confirm-stopped")) {
    throw new Error(
      "Applying the one-way migration requires --confirm-stopped after every writer covered "
      + "by the selected migration has stopped.",
    );
  }

  const legacy = {
    market: path.join(dataRoot, "historical"),
    datasets: path.join(dataRoot, "ml-datasets"),
    runs: path.join(dataRoot, "ml-runs"),
    cache: path.join(dataRoot, "runtime-cache"),
    jointStudies: path.join(dataRoot, "ml-joint-studies"),
    dynamicStudies: path.join(dataRoot, "ml-dynamic-studies"),
    analysis: path.join(dataRoot, "ml-analysis"),
    replacedTrainingState: path.join(dataRoot, "replaced-training-state"),
    modelArtifacts: path.join(dataRoot, "models", "mlp"),
    state: path.join(dataRoot, "state"),
    backtests: path.join(dataRoot, "backtests"),
    jobs: path.join(dataRoot, "jobs"),
  };
  const datasetSources = requestedDataset
    ? [path.resolve(repoRoot, requestedDataset)]
    : all
      ? uniquePaths([
          ...await incompleteCanonicalDatasets(),
          ...await childDirectories(legacy.datasets),
        ])
      : [];
  const report = {
    mode: apply ? "applied" : "dry-run",
    oneWay: true,
    legacy: {
      market: await selectedTreeStats(all, legacy.market),
      liveCandleStreams: await selectedTreeStats(
        all || liveCandles,
        layout.marketRoot,
        /-candles\.jsonl$/i,
      ),
      datasets: await selectedTreeStats(all || Boolean(requestedDataset), legacy.datasets),
      runs: await selectedTreeStats(all, legacy.runs),
      cache: await selectedTreeStats(all, legacy.cache),
      jointStudies: await selectedTreeStats(all, legacy.jointStudies),
      dynamicStudies: await selectedTreeStats(all, legacy.dynamicStudies),
      analysis: await selectedTreeStats(all, legacy.analysis),
      replacedTrainingState: await selectedTreeStats(all, legacy.replacedTrainingState),
      modelArtifacts: await selectedTreeStats(all, legacy.modelArtifacts, /\.pt$/i),
      state: await selectedTreeStats(all, legacy.state),
      backtests: await selectedTreeStats(all, legacy.backtests),
      jobs: await selectedTreeStats(all, legacy.jobs),
    },
    datasets: datasetSources.map((source) => path.basename(source)),
    migrated: {
      candleShards: 0,
      liveCandleDays: 0,
      datasetPayloads: 0,
      timestampFilesRemoved: 0,
      checkpoints: 0,
      movedRoots: 0,
    },
  };
  if (!apply) {
    process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
    return;
  }

  await Promise.all([
    fs.mkdir(layout.marketStore, { recursive: true }),
    fs.mkdir(layout.trainingStore, { recursive: true }),
    fs.mkdir(layout.trainingDatasets, { recursive: true }),
    fs.mkdir(layout.trainingRuns, { recursive: true }),
    fs.mkdir(layout.trainingCache, { recursive: true }),
    fs.mkdir(layout.trainingAnalysis, { recursive: true }),
    fs.mkdir(layout.marketMutable, { recursive: true }),
    fs.mkdir(layout.runtimeState, { recursive: true }),
    fs.mkdir(layout.runtimeBacktests, { recursive: true }),
    fs.mkdir(layout.runtimeLogs, { recursive: true }),
  ]);
  if (all) report.migrated.candleShards = await migrateMarket(legacy.market);
  for (const source of datasetSources) {
    const migrated = await migrateDataset(source);
    report.migrated.datasetPayloads += migrated.payloads;
    report.migrated.timestampFilesRemoved += migrated.timestamps;
    report.migrated.movedRoots += migrated.moved ? 1 : 0;
  }
  if (all) {
    report.migrated.movedRoots += await moveChildren(
      legacy.jointStudies,
      path.join(layout.trainingRuns, "studies", "joint"),
    );
    report.migrated.movedRoots += await moveChildren(
      legacy.dynamicStudies,
      path.join(layout.trainingRuns, "studies", "dynamic"),
    );
    report.migrated.movedRoots += await moveChildren(
      legacy.analysis,
      layout.trainingAnalysis,
    );
    report.migrated.movedRoots += await moveChildren(legacy.runs, layout.trainingRuns);
    report.migrated.checkpoints = await migrateCheckpoints(layout.trainingRuns);
    report.migrated.checkpoints += await migrateModelCheckpoints(legacy.modelArtifacts);
    report.migrated.movedRoots += await moveChildren(
      legacy.replacedTrainingState,
      path.join(layout.trainingRuns, "recovered"),
    );
    await organizeRunMetadata(layout.trainingRuns);
    report.migrated.movedRoots += await migrateRuntimeCache(legacy.cache);
    report.migrated.liveCandleDays = await migrateLiveCandleStreams();
    report.migrated.movedRoots += await migrateMutableMarket();
    report.migrated.movedRoots += await moveChildren(legacy.state, layout.runtimeState);
    report.migrated.movedRoots += await moveChildren(legacy.backtests, layout.runtimeBacktests);
    report.migrated.movedRoots += await migrateJobLogs(legacy.jobs);
  } else if (liveCandles) {
    report.migrated.liveCandleDays = await migrateLiveCandleStreams();
  }
  process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
}

async function migrateMarket(root: string): Promise<number> {
  const store = new SequentialShardStore(layout.marketStore);
  const files = (await walkFiles(root)).filter((file) => /\.jsonl(?:\.gz)?$/i.test(file));
  let migrated = 0;
  for (const file of files) {
    const relative = slash(path.relative(root, file));
    const parts = relative.split("/");
    if (parts.length !== 4) throw new Error(`Unexpected historical candle path: ${file}.`);
    const [market, symbol, interval, filename] = parts as [string, string, string, string];
    const date = filename.replace(/\.jsonl(?:\.gz)?$/i, "");
    const encoded = await fs.readFile(file);
    const text = file.endsWith(".gz") ? gunzipSync(encoded).toString("utf8") : encoded.toString("utf8");
    const candles = text.split(/\r?\n/).filter(Boolean).map((line) =>
      JSON.parse(line) as SequentialCandle);
    await putCandleShard(store, {
      namespace: `candles/${market}/${symbol}/${interval}`,
      key: date,
      candles,
      stepMs: intervalMilliseconds(interval),
      ...(candles.length > 0 ? {} : {
        empty: {
          symbol: symbol.toUpperCase(),
          interval,
          startTime: Date.parse(`${date}T00:00:00.000Z`),
          closeTimeOffsetMs: intervalMilliseconds(interval) - 1,
          closed: true,
        },
      }),
      metadata: { migratedFrom: slash(path.relative(repoRoot, file)) },
    });
    await fs.unlink(file);
    migrated += 1;
  }
  await removeEmptyDirectories(root);
  return migrated;
}

async function migrateDataset(source: string): Promise<{ payloads: number; timestamps: number; moved: boolean }> {
  const sourceRoot = path.resolve(source);
  const id = path.basename(sourceRoot);
  const destination = path.join(layout.trainingDatasets, safePart(id));
  let root = sourceRoot;
  let moved = false;
  if (!isInside(root, layout.trainingDatasets)) {
    if (await exists(destination)) throw new Error(`Dataset destination already exists: ${destination}.`);
    await fs.rename(root, destination);
    root = destination;
    moved = true;
  }
  await fs.mkdir(path.join(root, "state"), { recursive: true });
  for (const name of ["progress.json", "teacher-refinement-queue.json", "source-oracle-queue.json"]) {
    const file = path.join(root, name);
    if (await exists(file)) await moveFile(file, path.join(root, "state", name));
  }

  const manifest = await optionalJson(path.join(root, "dataset.json")) ?? {};
  const numeric = (await walkFiles(root)).filter((file) =>
    /\.(?:zst|f16|f32)$/i.test(file));
  const replacements = new Map<string, string>();
  const uncompressedBytes = new Map<string, number>();
  await recoverMigratedComponentMappings(root, replacements, uncompressedBytes);
  const store = new SequentialShardStore(layout.trainingStore);
  const migratedPayloads = await mapConcurrent(
    numeric,
    DATASET_MIGRATION_CONCURRENCY,
    async (file) => {
    const oldRelative = slash(path.relative(root, file));
    const encoded = await fs.readFile(file);
    const payload = file.endsWith(".zst") ? zstdDecompressSync(encoded) : encoded;
    const shape = inferShape(file, payload.byteLength, manifest);
    const date = /(\d{4}-\d{2}-\d{2})/.exec(path.basename(file))?.[1];
    const sequence = inferredSequence(date, shape.rows, file);
    const result = await store.put({
      namespace: inferredNamespace(file),
      key: `${safePart(id)}/${safeKey(oldRelative)}`,
      payload,
      sequence,
      layout: {
        encoding: "row-major",
        dtype: shape.dtype,
        rows: shape.rows,
        columns: shape.columns,
      },
      metadata: { migratedFrom: slash(path.relative(repoRoot, file)) },
      compressionLevel: 9,
    });
    const newRelative = canonicalComponentName(oldRelative);
    await writeJsonAtomic(path.join(root, newRelative), result.reference);
    if (inferredNamespace(file).startsWith("training/dataset-pairs/")) {
      await fs.unlink(result.referenceFile);
    }
    await fs.unlink(file);
    replacements.set(oldRelative, newRelative);
    replacements.set(oldRelative.replaceAll("/", "\\"), newRelative.replaceAll("/", "\\"));
    uncompressedBytes.set(slash(newRelative), payload.byteLength);
      return 1;
    },
  );
  const payloads = migratedPayloads.reduce((total, count) => total + count, 0);

  let timestamps = 0;
  for (const file of (await walkFiles(root)).filter((value) => value.endsWith(".times.i64"))) {
    await fs.unlink(file);
    timestamps += 1;
  }
  await rewriteJsonTree(root, (value) =>
    rewriteDatasetJson(value, replacements, uncompressedBytes));
  await removeEmptyDirectories(root);
  return { payloads, timestamps, moved };
}

async function recoverMigratedComponentMappings(
  root: string,
  replacements: Map<string, string>,
  uncompressedBytes: Map<string, number>,
): Promise<void> {
  const datasetId = path.basename(root);
  for (const file of (await walkFiles(root)).filter((value) => value.endsWith(".json"))) {
    const reference = await optionalJson(file) as any;
    if (reference?.kind !== "trading-sequential-shard"
      || typeof reference.metadata?.migratedFrom !== "string"
      || !Number.isSafeInteger(reference.object?.uncompressedBytes)) continue;
    const migratedFrom = slash(reference.metadata.migratedFrom);
    const marker = `/${datasetId}/`;
    const markerIndex = migratedFrom.lastIndexOf(marker);
    if (markerIndex < 0) continue;
    const oldRelative = migratedFrom.slice(markerIndex + marker.length);
    const newRelative = slash(path.relative(root, file));
    replacements.set(oldRelative, newRelative);
    replacements.set(oldRelative.replaceAll("/", "\\"), newRelative.replaceAll("/", "\\"));
    uncompressedBytes.set(newRelative, reference.object.uncompressedBytes);
  }
}

async function incompleteCanonicalDatasets(): Promise<string[]> {
  const result: string[] = [];
  for (const directory of await childDirectories(layout.trainingDatasets)) {
    if ((await walkFiles(directory)).some((file) => /\.(?:zst|f16|f32)$/i.test(file))) {
      result.push(directory);
    }
  }
  return result;
}

async function mapConcurrent<T, R>(
  values: readonly T[],
  concurrency: number,
  transform: (value: T) => Promise<R>,
): Promise<R[]> {
  const results = new Array<R>(values.length);
  let cursor = 0;
  await Promise.all(Array.from(
    { length: Math.min(concurrency, values.length) },
    async () => {
      while (cursor < values.length) {
        const index = cursor;
        cursor += 1;
        results[index] = await transform(values[index]!);
      }
    },
  ));
  return results;
}

function uniquePaths(values: readonly string[]): string[] {
  const seen = new Set<string>();
  return values.filter((value) => {
    const normalized = path.resolve(value).toLowerCase();
    if (seen.has(normalized)) return false;
    seen.add(normalized);
    return true;
  });
}

async function migrateCheckpoints(root: string): Promise<number> {
  const store = new SequentialShardStore(layout.trainingStore);
  const files = (await walkFiles(root)).filter((file) => /\.pt$/i.test(file));
  const replacements = new Map<string, string>();
  for (const file of files) {
    const relative = slash(path.relative(root, file));
    const pointer = checkpointPointer(file);
    const result = await store.putArtifactFile({
      namespace: "training/checkpoints/migrated",
      key: safeKey(relative),
      sourceFile: file,
      mediaType: "application/x-pytorch-checkpoint",
      metadata: { migratedFrom: slash(path.relative(repoRoot, file)) },
    });
    await writeJsonAtomic(pointer, result.reference);
    await fs.unlink(result.referenceFile);
    await fs.unlink(file);
    replacements.set(relative, slash(path.relative(root, pointer)));
  }
  await rewriteJsonTree(root, (value) => rewriteStrings(value, replacements));
  return files.length;
}

async function migrateModelCheckpoints(root: string): Promise<number> {
  const store = new SequentialShardStore(layout.trainingStore);
  const files = (await walkFiles(root)).filter((file) => /\.pt$/i.test(file));
  for (const file of files) {
    const modelId = path.basename(path.dirname(file));
    const name = path.basename(file, ".pt");
    const pointerName = name === "best-model"
      ? "best.json"
      : name === "checkpoint"
        ? "last.json"
        : `${name}.json`;
    const pointer = path.join(layout.run(modelId), "checkpoints", pointerName);
    const result = await store.putArtifactFile({
      namespace: `training/checkpoints/${modelId}`,
      key: safeKey(name),
      sourceFile: file,
      mediaType: "application/x-pytorch-checkpoint",
      metadata: { migratedFrom: slash(path.relative(repoRoot, file)) },
    });
    await writeJsonAtomic(pointer, result.reference);
    await fs.unlink(result.referenceFile);
    await fs.unlink(file);
  }
  return files.length;
}

function checkpointPointer(file: string): string {
  const name = path.basename(file).toLowerCase();
  const directory = path.dirname(file);
  const checkpointDirectory = path.basename(directory).toLowerCase() === "checkpoints"
    ? directory
    : path.join(directory, "checkpoints");
  if (name === "best.pt" || name === "best-model.pt") {
    return path.join(checkpointDirectory, "best.json");
  }
  if (name === "last.pt" || name === "checkpoint.pt") {
    return path.join(checkpointDirectory, "last.json");
  }
  return path.join(checkpointDirectory, `${path.basename(file, ".pt")}.json`);
}

async function migrateMutableMarket(): Promise<number> {
  if (!await exists(layout.marketRoot)) return 0;
  let moved = 0;
  for (const entry of await fs.readdir(layout.marketRoot, { withFileTypes: true })) {
    if (["immutable", "mutable"].includes(entry.name.toLowerCase())) continue;
    const source = path.join(layout.marketRoot, entry.name);
    const destination = entry.name.toLowerCase() === "tmp"
      ? layout.marketTmp
      : path.join(layout.marketMutable, "streams", entry.name);
    if (await exists(destination)) {
      moved += await mergeDirectories(source, destination);
      continue;
    }
    await fs.mkdir(path.dirname(destination), { recursive: true });
    await fs.rename(source, destination);
    moved += 1;
  }
  return moved;
}

async function migrateLiveCandleStreams(): Promise<number> {
  const files = (await walkFiles(layout.marketRoot))
    .filter((file) => /-candles\.jsonl$/i.test(path.basename(file)));
  const store = new SequentialShardStore(layout.marketStore);
  let migratedDays = 0;
  for (const file of files) {
    const match = /^(.+)-(\d+[smhdw])-candles\.jsonl$/i.exec(path.basename(file));
    if (!match) throw new Error(`Unexpected live candle stream name: ${file}.`);
    const market = safePart(path.basename(path.dirname(file)));
    const symbol = safePart(match[1]!);
    const interval = match[2]!.toLowerCase();
    const stepMs = intervalMilliseconds(interval);
    if (DAY_MS % stepMs !== 0 && stepMs % DAY_MS !== 0) {
      throw new Error(`Live candle interval must align to UTC days: ${interval}.`);
    }
    const candles = parseCandleLines(await fs.readFile(file, "utf8"), file);
    const byDate = new Map<string, SequentialCandle[]>();
    for (const candle of candles) {
      validateMigratedCandle(candle, symbol, interval, stepMs, file);
      const date = new Date(candle.openTime).toISOString().slice(0, 10);
      const rows = byDate.get(date) ?? [];
      rows.push(candle);
      byDate.set(date, rows);
    }

    for (const [date, rows] of [...byDate].sort(([left], [right]) => left.localeCompare(right))) {
      const sorted = deduplicateCandles(rows, file);
      const namespace = `candles/${market}/${symbol}/${interval}`;
      const stagingFile = path.join(
        layout.marketMutable,
        "candles",
        market,
        symbol,
        interval,
        `${date}.jsonl`,
      );
      if (isCompleteCandleShard(date, sorted, stepMs)) {
        await putCandleShard(store, {
          namespace,
          key: date,
          candles: sorted,
          stepMs,
          metadata: {
            source: "live-recorder",
            completeUtcDay: stepMs <= DAY_MS,
            completeCandlePeriod: true,
            migratedFrom: slash(path.relative(repoRoot, file)),
          },
        });
        await fs.rm(stagingFile, { force: true });
      } else if (await exists(store.referenceFile(namespace, date))) {
        await fs.rm(stagingFile, { force: true });
      } else {
        const existing = await optionalCandleLines(stagingFile);
        for (const candle of existing) {
          validateMigratedCandle(candle, symbol, interval, stepMs, stagingFile);
        }
        const merged = deduplicateCandles([...existing, ...sorted], stagingFile);
        await writeJsonLinesAtomic(stagingFile, merged);
      }
      migratedDays += 1;
    }
    await fs.unlink(file);
  }
  return migratedDays;
}

function parseCandleLines(content: string, file: string): SequentialCandle[] {
  return content.split(/\r?\n/).filter(Boolean).map((line, index) => {
    try {
      return JSON.parse(line) as SequentialCandle;
    } catch (error) {
      throw new Error(`Invalid candle JSON at ${file}:${index + 1}.`, { cause: error });
    }
  });
}

async function optionalCandleLines(file: string): Promise<SequentialCandle[]> {
  try {
    return parseCandleLines(await fs.readFile(file, "utf8"), file);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return [];
    throw error;
  }
}

function validateMigratedCandle(
  candle: SequentialCandle,
  symbol: string,
  interval: string,
  stepMs: number,
  file: string,
): void {
  if (candle.symbol.toLowerCase() !== symbol
    || candle.interval.toLowerCase() !== interval
    || !candle.closed
    || !Number.isSafeInteger(candle.openTime)
    || candle.openTime % Math.min(stepMs, DAY_MS) !== 0
    || candle.closeTime !== candle.openTime + stepMs - 1
    || ![candle.open, candle.high, candle.low, candle.close, candle.volume].every(Number.isFinite)) {
    throw new Error(`Invalid candle at ${candle.openTime} in ${file}.`);
  }
}

function deduplicateCandles(
  candles: readonly SequentialCandle[],
  file: string,
): SequentialCandle[] {
  const sorted = [...candles].sort((left, right) => left.openTime - right.openTime);
  const result: SequentialCandle[] = [];
  for (const candle of sorted) {
    const previous = result.at(-1);
    if (previous?.openTime === candle.openTime) {
      if (!equalCandle(previous, candle)) {
        throw new Error(`Conflicting candle ${candle.openTime} while migrating ${file}.`);
      }
      continue;
    }
    result.push(candle);
  }
  return result;
}

function equalCandle(left: SequentialCandle, right: SequentialCandle): boolean {
  return left.symbol === right.symbol
    && left.interval === right.interval
    && left.openTime === right.openTime
    && left.closeTime === right.closeTime
    && left.open === right.open
    && left.high === right.high
    && left.low === right.low
    && left.close === right.close
    && left.volume === right.volume
    && left.closed === right.closed;
}

function isCompleteCandleShard(
  date: string,
  candles: readonly SequentialCandle[],
  stepMs: number,
): boolean {
  const dayStart = Date.parse(`${date}T00:00:00.000Z`);
  const expectedRows = stepMs >= DAY_MS ? 1 : DAY_MS / stepMs;
  if (candles.length !== expectedRows) return false;
  return candles.every((candle, index) => candle.openTime === dayStart + index * stepMs);
}

async function migrateJobLogs(root: string): Promise<number> {
  if (!await exists(root)) return 0;
  let moved = 0;
  for (const file of await directFiles(root)) {
    const training = /^(?:mlp|return-oracle|joint-price-oracle)/i.test(path.basename(file));
    const destination = training
      ? path.join(
          layout.trainingRuns,
          "legacy-runtime",
          file.toLowerCase().endsWith(".pid") ? "state" : "logs",
          path.basename(file),
        )
      : path.join(layout.runtimeLogs, path.basename(file));
    await moveFile(file, destination);
    moved += 1;
  }
  for (const directory of await childDirectories(root)) {
    const training = /^(?:mlp|return-oracle|joint-price-oracle)/i.test(path.basename(directory));
    const destination = training
      ? path.join(layout.trainingRuns, "legacy-runtime", "logs", path.basename(directory))
      : path.join(layout.runtimeLogs, path.basename(directory));
    if (await exists(destination)) throw new Error(`Job-log destination exists: ${destination}.`);
    await fs.mkdir(path.dirname(destination), { recursive: true });
    await fs.rename(directory, destination);
    moved += 1;
  }
  await removeEmptyDirectories(root);
  return moved;
}

async function organizeRunMetadata(root: string): Promise<void> {
  for (const directory of await walkDirectories(root)) {
    const directoryName = path.basename(directory).toLowerCase();
    if (directoryName === "logs") {
      const oldMetricLog = path.join(directory, "training.log");
      if (await exists(oldMetricLog)) {
        await moveFile(oldMetricLog, path.join(directory, "training.history.jsonl"));
      }
      continue;
    }
    if (["state", "control", "checkpoints", "tmp"].includes(directoryName)) continue;
    for (const name of ["status.json", "result.json"]) {
      const source = path.join(directory, name);
      if (await exists(source)) await moveFile(source, path.join(directory, "state", name));
    }
    const finalize = path.join(directory, "FINALIZE");
    if (await exists(finalize)) await moveFile(finalize, path.join(directory, "control", "FINALIZE"));
    for (const file of await directFiles(directory)) {
      if (/\.(?:log|jsonl)$/i.test(file)) {
        const destinationName = path.basename(file).toLowerCase() === "training.log"
          ? "training.history.jsonl"
          : path.basename(file);
        await moveFile(file, path.join(directory, "logs", destinationName));
      }
    }
  }
}

async function migrateRuntimeCache(root: string): Promise<number> {
  let moved = 0;
  const legacyLogs = path.join(layout.trainingRuns, "legacy-runtime", "logs");
  if (await exists(root)) {
    for (const file of await directFiles(root)) {
      const destination = /\.(?:log|jsonl)$/i.test(file)
        ? path.join(legacyLogs, path.basename(file))
        : path.join(layout.trainingCache, path.basename(file));
      await moveFile(file, destination);
      moved += 1;
    }
    for (const directory of await childDirectories(root)) {
      const destination = directory.toLowerCase().includes("log")
        ? path.join(legacyLogs, path.basename(directory))
        : path.join(layout.trainingCache, path.basename(directory));
      if (await exists(destination)) {
        moved += await mergeDirectories(directory, destination);
      } else {
        await fs.mkdir(path.dirname(destination), { recursive: true });
        await fs.rename(directory, destination);
        moved += 1;
      }
    }
    await removeEmptyDirectories(root);
  }
  const misplaced = path.join(layout.trainingCache, "legacy");
  if (await exists(misplaced)) moved += await moveChildren(misplaced, layout.trainingCache);
  return moved;
}

function inferShape(file: string, bytes: number, manifest: any): {
  dtype: "float16-le" | "float32-le";
  rows: number;
  columns: number;
} {
  const name = path.basename(file).toLowerCase();
  const dtype = name.includes(".f16") ? "float16-le" as const : "float32-le" as const;
  const itemBytes = dtype === "float16-le" ? 2 : 4;
  const actionCount = Number(manifest.actionCount ?? manifest.actionGrid?.length ?? 255);
  let columns = 1;
  if (name.includes("features")) columns = Number(manifest.featureCount ?? 901);
  else if (name.includes("simple-returns")) columns = Number(manifest.inputReturnCount ?? 60);
  else if (name.includes("raw-oracle") || name.includes("minute-oracle")) columns = actionCount;
  else if (name.includes("teacher-parameters")) columns = Number(manifest.teacherParameterCount ?? 8);
  else if (name.includes("teacher-metrics")) columns = Number(manifest.teacherMetricCount ?? 7);
  const rows = bytes / itemBytes / columns;
  if (!Number.isSafeInteger(rows) || rows < 1) {
    throw new Error(`Cannot infer a row-major shape for ${file} (${bytes} bytes).`);
  }
  return { dtype, rows, columns };
}

function inferredSequence(date: string | undefined, rows: number, file: string) {
  if (date && rows === 86_400) {
    return { start: Date.parse(`${date}T00:00:00.000Z`) + 999, step: 1_000, count: rows, unit: "unix-ms" as const };
  }
  if (date && rows === 1_441) {
    return { start: Date.parse(`${date}T00:00:00.000Z`) - 1, step: 60_000, count: rows, unit: "unix-ms" as const };
  }
  return { start: 0, step: 1, count: rows, unit: "index" as const };
}

function inferredNamespace(file: string): string {
  const name = path.basename(file).toLowerCase();
  if (name.includes("features") || name.includes("simple-returns")) return "features/migrated";
  if (name.includes("raw-oracle")) return "oracle/1s/migrated";
  if (name.includes("minute-oracle")) return "oracle/1m/migrated";
  if (name.includes("teacher-parameters")) return "training/teacher-parameters/migrated";
  if (name.includes("teacher-metrics")) return "training/teacher-metrics/migrated";
  if (name.includes("base-time-weights")) return "training/dataset-pairs/base-time-weights";
  if (name.includes("time-weights")) return "training/dataset-pairs/time-weights";
  if (name.includes("resolution")) return "training/dataset-pairs/resolution-divergence";
  return "training/dataset-components/migrated";
}

function canonicalComponentName(relative: string): string {
  return relative.replace(/\.(?:f16|f32)(?:\.zst)?$/i, ".json").replace(/\.zst$/i, ".json");
}

function rewriteDatasetJson(
  value: unknown,
  replacements: Map<string, string>,
  uncompressedBytes: Map<string, number>,
): unknown {
  if (Array.isArray(value)) {
    return value.map((item) => rewriteDatasetJson(item, replacements, uncompressedBytes));
  }
  if (!value || typeof value !== "object") {
    if (typeof value !== "string") return value;
    return rewritePathString(replacements.get(value) ?? value);
  }
  const result: Record<string, unknown> = {};
  const source = value as Record<string, unknown>;
  for (const [key, item] of Object.entries(source)) {
    if (key === "times" && typeof item === "string") continue;
    if (key === "componentCompression") continue;
    result[key] = rewriteDatasetJson(item, replacements, uncompressedBytes);
  }
  if (typeof source.times === "string" && typeof source.date === "string"
    && Number.isSafeInteger(source.count)) {
    result.timeSequence = {
      encoding: "implicit-linear",
      startTimeMs: Date.parse(`${source.date}T00:00:00.000Z`) + 999,
      stepMs: 1_000,
      count: source.count,
    };
  }
  for (const field of [
    "features",
    "rawOracleProbabilities",
    "teacherParameters",
    "teacherMetrics",
  ]) {
    const file = result[field];
    if (typeof file !== "string") continue;
    const bytes = uncompressedBytes.get(slash(file));
    if (bytes === undefined) continue;
    result[`${field}Compression`] = "zstd";
    result[`${field}UncompressedBytes`] = bytes;
  }
  if (typeof result.storeId === "string" && result.compression && typeof result.compression === "object") {
    result.compression = {
      features: "zstd",
      rawOracleProbabilities: "zstd",
      minuteOracleProbabilities: "zstd",
    };
  }
  return result;
}

function rewriteStrings(value: unknown, replacements: Map<string, string>): unknown {
  if (typeof value === "string") return replacements.get(value) ?? value;
  if (Array.isArray(value)) return value.map((item) => rewriteStrings(item, replacements));
  if (!value || typeof value !== "object") return value;
  return Object.fromEntries(Object.entries(value).map(([key, item]) =>
    [key, rewriteStrings(item, replacements)]));
}

function rewritePathString(value: string): string {
  return value
    .replaceAll("data\\ml-datasets\\", "data\\training\\datasets\\")
    .replaceAll("data/ml-datasets/", "data/training/datasets/")
    .replaceAll("data\\ml-runs\\", "data\\training\\runs\\")
    .replaceAll("data/ml-runs/", "data/training/runs/");
}

async function rewriteJsonTree(root: string, transform: (value: unknown) => unknown): Promise<void> {
  for (const file of (await walkFiles(root)).filter((value) => value.endsWith(".json"))) {
    const value = await optionalJson(file);
    if (value !== undefined) await writeJsonAtomic(file, transform(value));
  }
}

async function moveChildren(source: string, destination: string): Promise<number> {
  if (!await exists(source)) return 0;
  let moved = 0;
  await fs.mkdir(destination, { recursive: true });
  for (const entry of await fs.readdir(source, { withFileTypes: true })) {
    const target = path.join(destination, entry.name);
    const childSource = path.join(source, entry.name);
    if (await exists(target)) {
      if (!entry.isDirectory() || !(await fs.stat(target)).isDirectory()) {
        throw new Error(`Migration destination exists: ${target}.`);
      }
      moved += await mergeDirectories(childSource, target);
    } else {
      await fs.rename(childSource, target);
      moved += 1;
    }
  }
  await removeEmptyDirectories(source);
  return moved;
}

async function mergeDirectories(source: string, destination: string): Promise<number> {
  const [sourceStat, destinationStat] = await Promise.all([
    fs.stat(source),
    fs.stat(destination),
  ]);
  if (!sourceStat.isDirectory() || !destinationStat.isDirectory()) {
    throw new Error(`Cannot merge non-directory market paths: ${source} -> ${destination}.`);
  }
  let moved = 0;
  for (const entry of await fs.readdir(source, { withFileTypes: true })) {
    const childSource = path.join(source, entry.name);
    const childDestination = path.join(destination, entry.name);
    if (!await exists(childDestination)) {
      await fs.rename(childSource, childDestination);
      moved += 1;
      continue;
    }
    if (!entry.isDirectory()) {
      throw new Error(`Market merge destination exists: ${childDestination}.`);
    }
    moved += await mergeDirectories(childSource, childDestination);
  }
  await fs.rmdir(source);
  return moved;
}

async function moveFile(source: string, destination: string): Promise<void> {
  if (await exists(destination)) throw new Error(`Migration destination exists: ${destination}.`);
  await fs.mkdir(path.dirname(destination), { recursive: true });
  await fs.rename(source, destination);
}

async function childDirectories(root: string): Promise<string[]> {
  try {
    return (await fs.readdir(root, { withFileTypes: true }))
      .filter((entry) => entry.isDirectory())
      .map((entry) => path.join(root, entry.name));
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return [];
    throw error;
  }
}

async function directFiles(root: string): Promise<string[]> {
  try {
    return (await fs.readdir(root, { withFileTypes: true }))
      .filter((entry) => entry.isFile())
      .map((entry) => path.join(root, entry.name));
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return [];
    throw error;
  }
}

async function walkFiles(root: string): Promise<string[]> {
  const result: string[] = [];
  for (const directory of await walkDirectories(root)) result.push(...await directFiles(directory));
  return result;
}

async function walkDirectories(root: string): Promise<string[]> {
  if (!await exists(root)) return [];
  const result: string[] = [];
  const pending = [root];
  while (pending.length) {
    const directory = pending.pop()!;
    result.push(directory);
    pending.push(...await childDirectories(directory));
  }
  return result;
}

async function removeEmptyDirectories(root: string): Promise<void> {
  if (!await exists(root)) return;
  const directories = await walkDirectories(root);
  for (const directory of directories.sort((left, right) => right.length - left.length)) {
    try {
      await fs.rmdir(directory);
    } catch (error) {
      if (!['ENOENT', 'ENOTEMPTY'].includes((error as NodeJS.ErrnoException).code ?? "")) throw error;
    }
  }
}

async function treeStats(
  root: string,
  filter?: RegExp,
): Promise<{ files: number; bytes: number }> {
  const files = (await walkFiles(root)).filter((file) => !filter || filter.test(file));
  let bytes = 0;
  for (let offset = 0; offset < files.length; offset += 128) {
    const sizes = await Promise.all(
      files.slice(offset, offset + 128).map(async (file) => (await fs.stat(file)).size),
    );
    bytes += sizes.reduce((total, size) => total + size, 0);
  }
  return { files: files.length, bytes };
}

function selectedTreeStats(
  selected: boolean,
  root: string,
  filter?: RegExp,
): Promise<{ files: number; bytes: number }> {
  return selected ? treeStats(root, filter) : Promise.resolve({ files: 0, bytes: 0 });
}

async function optionalJson(file: string): Promise<any | undefined> {
  try {
    return JSON.parse(await fs.readFile(file, "utf8"));
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return undefined;
    throw error;
  }
}

async function writeJsonAtomic(file: string, value: unknown): Promise<void> {
  await fs.mkdir(path.dirname(file), { recursive: true });
  const temporary = `${file}.${process.pid}.tmp`;
  await fs.writeFile(temporary, `${JSON.stringify(value, null, 2)}\n`, { flag: "wx" });
  await fs.rename(temporary, file);
}

async function writeJsonLinesAtomic(
  file: string,
  values: readonly SequentialCandle[],
): Promise<void> {
  await fs.mkdir(path.dirname(file), { recursive: true });
  const temporary = `${file}.${process.pid}.${Date.now()}.tmp`;
  await fs.writeFile(
    temporary,
    values.map((value) => JSON.stringify(value)).join("\n") + "\n",
    { flag: "wx" },
  );
  await fs.rename(temporary, file);
}

async function exists(file: string): Promise<boolean> {
  try {
    await fs.stat(file);
    return true;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return false;
    throw error;
  }
}

function isInside(candidate: string, root: string): boolean {
  const relative = path.relative(path.resolve(root), path.resolve(candidate));
  return Boolean(relative) && !relative.startsWith("..") && !path.isAbsolute(relative);
}

function intervalMilliseconds(interval: string): number {
  const match = /^(\d+)([smhdw])$/.exec(interval);
  if (!match) throw new Error(`Unsupported candle interval: ${interval}.`);
  return Number(match[1]) * ({
    s: 1_000,
    m: 60_000,
    h: 3_600_000,
    d: 86_400_000,
    w: 604_800_000,
  }[match[2]!]!);
}

function safePart(value: string): string {
  const result = value.toLowerCase().replace(/[^a-z0-9._=-]+/g, "-");
  if (!result) throw new Error(`Unsafe storage identifier: ${value}.`);
  return result;
}

function safeKey(value: string): string {
  return slash(value).split("/").map(safePart).join("/");
}

function slash(value: string): string {
  return value.replaceAll(path.sep, "/");
}

function argument(name: string): string | undefined {
  const index = args.indexOf(`--${name}`);
  return index >= 0 ? args[index + 1] : undefined;
}
