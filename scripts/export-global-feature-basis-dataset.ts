import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { readCandleShardReferenceSync, type SequentialCandle } from "@trading/storage";
import {
  buildSignalDefinitions,
  IndicatorEngine,
} from "./analyze-technical-indicator-predictiveness.ts";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const ONE_SECOND_DIRECTORY = "data/market/immutable/refs/candles/spot-btcusdt/btcusdt/1s";
const ONE_MINUTE_DIRECTORY = "data/market/immutable/refs/candles/spot-btcusdt/btcusdt/1m";
const DEFAULT_OUTPUT = "data/runtime-cache/global-feature-basis";
const START = Date.parse("2021-07-25T00:00:00.000Z");
const TRAIN_END = Date.parse("2024-07-25T00:00:00.000Z");
const PRIMARY_END = Date.parse("2025-07-25T00:00:00.000Z");
const END = Date.parse("2026-07-25T00:00:00.000Z");
const DAY_MS = 86_400_000;

interface FeatureDefinition {
  id: string;
  name: string;
  family: string;
  parameters: string;
  lookback: string;
  delay: string;
  kind?: "binary" | "continuous";
}

interface Pending {
  originTime: number;
  originClose: number;
  features: Float32Array;
  targets: Float32Array;
  remaining: number;
  split: number;
}

class DatasetWriter {
  private featureRows: number[] = [];
  private targetRows: number[] = [];
  private splits: number[] = [];
  private times: number[] = [];
  rows = 0;

  readonly featuresFile: string;
  readonly targetsFile: string;
  readonly splitsFile: string;
  readonly timesFile: string;

  constructor(
    private readonly output: string,
    readonly id: string,
    readonly featureCount: number,
    readonly targetCount: number,
  ) {
    fs.mkdirSync(output, { recursive: true });
    this.featuresFile = path.join(output, `${id}.features.f32`);
    this.targetsFile = path.join(output, `${id}.targets.f32`);
    this.splitsFile = path.join(output, `${id}.splits.u8`);
    this.timesFile = path.join(output, `${id}.times.f64`);
    for (const file of [this.featuresFile, this.targetsFile, this.splitsFile, this.timesFile]) {
      fs.writeFileSync(file, Buffer.alloc(0));
    }
  }

  add(row: Pending) {
    this.featureRows.push(...row.features);
    this.targetRows.push(...row.targets);
    this.splits.push(row.split);
    this.times.push(row.originTime);
    this.rows += 1;
    if (this.splits.length >= 8_192) this.flush();
  }

  close() { this.flush(); }

  private flush() {
    if (this.splits.length === 0) return;
    appendTyped(this.featuresFile, Float32Array.from(this.featureRows));
    appendTyped(this.targetsFile, Float32Array.from(this.targetRows));
    appendTyped(this.splitsFile, Uint8Array.from(this.splits));
    appendTyped(this.timesFile, Float64Array.from(this.times));
    this.featureRows = [];
    this.targetRows = [];
    this.splits = [];
    this.times = [];
  }
}

export function run(args = process.argv.slice(2)) {
  const value = (name: string) => {
    const index = args.indexOf(name);
    return index < 0 ? undefined : args[index + 1];
  };
  const scale = value("--scale") ?? "all";
  if (!new Set(["all", "1s", "1m"]).has(scale)) throw new Error(`Unsupported --scale ${scale}`);
  const output = resolve(value("--output-dir") ?? DEFAULT_OUTPUT);
  const artifacts: any[] = [];
  if (scale === "all" || scale === "1s") artifacts.push(exportOneSecond(output));
  if (scale === "all" || scale === "1m") artifacts.push(exportOneMinute(output));
  const manifestFile = path.join(output, "manifest.json");
  const existingDatasets = scale === "all" || !fs.existsSync(manifestFile)
    ? []
    : JSON.parse(fs.readFileSync(manifestFile, "utf8")).datasets ?? [];
  const datasetById = new Map(existingDatasets.map((dataset: any) => [dataset.id, dataset]));
  for (const artifact of artifacts) datasetById.set(artifact.id, artifact);
  const manifest = {
    version: 1,
    generatedAt: new Date().toISOString(),
    objective: "Joint chronological feature-subset search for BTC return distributions",
    split: {
      start: new Date(START).toISOString(),
      trainEndExclusive: new Date(TRAIN_END).toISOString(),
      primaryEndExclusive: new Date(PRIMARY_END).toISOString(),
      endExclusive: new Date(END).toISOString(),
      labels: ["train", "primary", "transfer"],
    },
    datasets: [...datasetById.values()],
  };
  fs.writeFileSync(manifestFile, `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
  console.log(`Wrote ${path.relative(repoRoot, manifestFile)}`);
  return manifest;
}

function exportOneSecond(output: string) {
  const definitions: FeatureDefinition[] = [
    feature("previous-return-1s", "Previous 1s signed log return", "return history", "lag=1s", "1s", "latest completed second"),
    feature("previous-return-zero", "Previous return is exactly zero", "activity", "binary", "1s", "latest completed second", "binary"),
    feature("active-count-60s", "Active-return count", "activity", "window=60s", "60s", "trailing through origin"),
    feature("realized-volatility-60s", "Realized volatility", "volatility", "sqrt(sum(r^2)), window=60s", "60s", "trailing through origin"),
    feature("rsi-2s", "RSI", "price dynamics", "period=2s, Wilder alpha=1/2", "recursive; effective 2s", "through origin"),
    feature("ema-acceleration-2s-1s", "EMA acceleration", "price dynamics", "EMA period=2s, difference horizon=1s", "2s EMA; two 1s slopes", "through origin"),
    feature("ema-slope-8s-8s", "EMA slope", "price dynamics", "EMA period=8s, difference horizon=8s", "8s EMA and 8s lag", "through origin"),
    feature("completed-1h-log-volume", "Last completed 1h log(1+base volume)", "volume regime", "UTC-aligned 1h", "1h", "updates only after hour close"),
    feature("range-1s", "Completed 1s high-low log range", "candle shape", "10000*log(high/low)", "1s", "latest completed second"),
    feature("close-location-1s", "Completed 1s close location", "candle shape", "(2*close-high-low)/(high-low)", "1s", "latest completed second"),
  ];
  const signalDefinitions = selectSignals(["rsi-2", "ema-acceleration-2-1", "ema-slope-8-8"]);
  let indicator: IndicatorEngine | undefined;
  const indicatorValues = new Float64Array(signalDefinitions.length);
  const ring = new Float64Array(60);
  let ringSeen = 0;
  let active60 = 0;
  let square60 = 0;
  let previousClose = Number.NaN;
  let hourBucket = -1;
  let hourVolume = 0;
  let completedHourLogVolume = Number.NaN;
  let pending: Pending | undefined;
  const writer = new DatasetWriter(output, "1s", definitions.length, 1);
  const files = selectedFiles(resolve(ONE_SECOND_DIRECTORY));
  for (let fileIndex = 0; fileIndex < files.length; fileIndex += 1) {
    const candles = readCandleShardReferenceSync(files[fileIndex]!);
    for (const candle of candles) {
      if (candle.openTime < START) continue;
      if (candle.openTime >= END) break;
      if (pending && candle.openTime === pending.originTime + 1_000) {
        pending.targets[0] = Math.log(candle.close / pending.originClose) * 10_000;
        pending.remaining = 0;
        writer.add(pending);
        pending = undefined;
      }
      const returnBps = Number.isFinite(previousClose) ? Math.log(candle.close / previousClose) * 10_000 : 0;
      const ringIndex = ringSeen % ring.length;
      if (ringSeen >= ring.length) {
        const removed = ring[ringIndex]!;
        square60 -= removed * removed;
        active60 -= removed !== 0 ? 1 : 0;
      }
      ring[ringIndex] = returnBps;
      ringSeen += 1;
      square60 += returnBps * returnBps;
      active60 += returnBps !== 0 ? 1 : 0;
      const bucket = Math.floor(candle.openTime / 3_600_000);
      if (bucket !== hourBucket) {
        hourBucket = bucket;
        hourVolume = 0;
      }
      hourVolume += candle.volume;
      if ((candle.openTime + 1_000) % 3_600_000 === 0) completedHourLogVolume = Math.log1p(hourVolume);
      if (!indicator) indicator = new IndicatorEngine(signalDefinitions, candle.close);
      else indicator.update(candle.close);
      indicator.values(indicatorValues);
      previousClose = candle.close;
      const dayIndex = Math.floor(candle.openTime / DAY_MS);
      const rotatingSecond = dayIndex % 60;
      if (
        Math.floor(candle.openTime / 1_000) % 60 === rotatingSecond
        && ringSeen >= 60
        && Number.isFinite(completedHourLogVolume)
      ) {
        if (pending) throw new Error("Overlapping one-second origin pending unexpectedly.");
        pending = {
          originTime: candle.openTime,
          originClose: candle.close,
          features: Float32Array.from([
            returnBps,
            returnBps === 0 ? 1 : 0,
            active60,
            Math.sqrt(Math.max(0, square60)),
            indicatorValues[0]!,
            indicatorValues[1]!,
            indicatorValues[2]!,
            completedHourLogVolume,
            Math.log(candle.high / candle.low) * 10_000,
            closeLocation(candle),
          ]),
          targets: new Float32Array(1),
          remaining: 1,
          split: splitOf(candle.openTime),
        };
      }
    }
    if ((fileIndex + 1) % 100 === 0) console.error(`1s export: ${fileIndex + 1}/${files.length} days, ${writer.rows.toLocaleString()} rows`);
  }
  writer.close();
  return datasetMetadata(writer, definitions, [{ id: "1s", minutes: 1 / 60 }], "one rotating origin per minute");
}

function exportOneMinute(output: string) {
  const returnWindows = [1, 2, 5, 15, 60];
  const volatilityWindows = [2, 5, 15, 30, 60, 240];
  const rsiIds = [2, 4, 8, 14, 32, 64].map((period) => `rsi-${period}`);
  const slopeIds = ["ema-slope-2-1", "ema-slope-8-8", "ema-slope-32-8", "ema-slope-128-32"];
  const accelerationIds = ["ema-acceleration-2-1", "ema-acceleration-4-2", "ema-acceleration-8-4", "ema-acceleration-32-8"];
  const signalIds = [...rsiIds, ...slopeIds, ...accelerationIds];
  const signalDefinitions = selectSignals(signalIds);
  const definitions: FeatureDefinition[] = [
    ...returnWindows.map((window) => feature(`return-${window}m`, `Trailing signed return`, "return history", `window=${window}m`, `${window}m`, "through origin")),
    ...volatilityWindows.map((window) => feature(`realized-volatility-${window}m`, "Realized volatility", "volatility", `sqrt(sum(r_1m^2)), window=${window}m`, `${window}m`, "through origin")),
    ...rsiIds.map((id, index) => feature(id, "RSI", "price dynamics", `period=${[2, 4, 8, 14, 32, 64][index]}m`, "recursive", "through origin")),
    ...slopeIds.map((id) => feature(id, "EMA slope", "price dynamics", id.replace("ema-slope-", "period/horizon="), "recursive", "through origin")),
    ...accelerationIds.map((id) => feature(id, "EMA acceleration", "price dynamics", id.replace("ema-acceleration-", "period/horizon="), "recursive", "through origin")),
    feature("range-1m", "Completed 1m high-low log range", "candle shape", "10000*log(high/low)", "1m", "latest completed minute"),
    feature("close-location-1m", "Completed 1m close location", "candle shape", "(2*close-high-low)/(high-low)", "1m", "latest completed minute"),
    feature("log-volume-1m", "Completed 1m log(1+base volume)", "volume", "1m bar", "1m", "latest completed minute"),
    feature("relative-log-volume-32m", "1m log-volume surprise", "volume", "log volume minus EMA(32m)", "32m recursive", "through origin"),
    ...[5, 15, 60].map((window) => feature(`completed-${window}m-log-volume`, "Last completed log(1+base volume)", "volume regime", `UTC-aligned ${window}m`, `${window}m`, "updates only after bar close")),
    feature("utc-hour-sin", "UTC hour sine", "calendar", "sin(2*pi*secondOfDay/86400)", "known", "none"),
    feature("utc-hour-cos", "UTC hour cosine", "calendar", "cos(2*pi*secondOfDay/86400)", "known", "none"),
  ];
  const writer = new DatasetWriter(output, "1m", definitions.length, 3);
  const closes = new Float64Array(241);
  const returns = new Float64Array(240);
  let seen = 0;
  let previousClose = Number.NaN;
  const signalValues = new Float64Array(signalDefinitions.length);
  let indicator: IndicatorEngine | undefined;
  let logVolumeEma32 = Number.NaN;
  const completedVolume = new Map<number, number>();
  const volumeBuckets = new Map<number, { bucket: number; volume: number }>();
  const schedule = new Map<number, Array<{ pending: Pending; target: number }>>();
  const horizons = [1, 15, 60];
  const files = selectedFiles(resolve(ONE_MINUTE_DIRECTORY));
  for (let fileIndex = 0; fileIndex < files.length; fileIndex += 1) {
    const candles = readCandleShardReferenceSync(files[fileIndex]!);
    for (const candle of candles) {
      if (candle.openTime < START) continue;
      if (candle.openTime >= END) break;
      const due = schedule.get(candle.openTime);
      if (due) {
        for (const item of due) {
          item.pending.targets[item.target] = Math.log(candle.close / item.pending.originClose) * 10_000;
          item.pending.remaining -= 1;
          if (item.pending.remaining === 0) writer.add(item.pending);
        }
        schedule.delete(candle.openTime);
      }
      const returnBps = Number.isFinite(previousClose) ? Math.log(candle.close / previousClose) * 10_000 : 0;
      closes[seen % closes.length] = candle.close;
      returns[seen % returns.length] = returnBps;
      seen += 1;
      if (!indicator) indicator = new IndicatorEngine(signalDefinitions, candle.close);
      else indicator.update(candle.close);
      indicator.values(signalValues);
      const logVolume = Math.log1p(candle.volume);
      const relativeLogVolume = Number.isFinite(logVolumeEma32) ? logVolume - logVolumeEma32 : 0;
      logVolumeEma32 = Number.isFinite(logVolumeEma32)
        ? logVolumeEma32 + 2 / 33 * (logVolume - logVolumeEma32)
        : logVolume;
      for (const window of [5, 15, 60]) {
        const bucket = Math.floor(candle.openTime / (window * 60_000));
        const state = volumeBuckets.get(window);
        if (!state || state.bucket !== bucket) volumeBuckets.set(window, { bucket, volume: candle.volume });
        else state.volume += candle.volume;
        if ((candle.openTime + 60_000) % (window * 60_000) === 0) {
          completedVolume.set(window, Math.log1p(volumeBuckets.get(window)!.volume));
        }
      }
      previousClose = candle.close;
      const dayIndex = Math.floor(candle.openTime / DAY_MS);
      const minuteOfDay = Math.floor((candle.openTime % DAY_MS) / 60_000);
      if (
        minuteOfDay % 4 === dayIndex % 4
        && seen >= 241
        && [5, 15, 60].every((window) => Number.isFinite(completedVolume.get(window)))
      ) {
        const values: number[] = [];
        for (const window of returnWindows) values.push(trailingReturn(closes, seen, window, candle.close));
        for (const window of volatilityWindows) values.push(trailingVolatility(returns, seen, window));
        values.push(...signalValues);
        values.push(
          Math.log(candle.high / candle.low) * 10_000,
          closeLocation(candle),
          logVolume,
          relativeLogVolume,
          completedVolume.get(5)!,
          completedVolume.get(15)!,
          completedVolume.get(60)!,
        );
        const phase = 2 * Math.PI * ((candle.openTime % DAY_MS) / DAY_MS);
        values.push(Math.sin(phase), Math.cos(phase));
        if (values.length !== definitions.length) throw new Error(`Feature count mismatch: ${values.length} vs ${definitions.length}`);
        const pending: Pending = {
          originTime: candle.openTime,
          originClose: candle.close,
          features: Float32Array.from(values),
          targets: new Float32Array(horizons.length),
          remaining: horizons.length,
          split: splitOf(candle.openTime),
        };
        horizons.forEach((horizon, target) => {
          const time = candle.openTime + horizon * 60_000;
          const list = schedule.get(time) ?? [];
          list.push({ pending, target });
          schedule.set(time, list);
        });
      }
    }
    if ((fileIndex + 1) % 100 === 0) console.error(`1m export: ${fileIndex + 1}/${files.length} days, ${writer.rows.toLocaleString()} completed rows`);
  }
  writer.close();
  return datasetMetadata(
    writer,
    definitions,
    [{ id: "1m", minutes: 1 }, { id: "15m", minutes: 15 }, { id: "1h", minutes: 60 }],
    "one rotating origin per four minutes; scorer thins to non-overlapping outcomes",
  );
}

function datasetMetadata(
  writer: DatasetWriter,
  features: FeatureDefinition[],
  targets: Array<{ id: string; minutes: number }>,
  sampling: string,
) {
  return {
    id: writer.id,
    rows: writer.rows,
    featureCount: writer.featureCount,
    targetCount: writer.targetCount,
    sampling,
    files: {
      features: path.relative(resolve(DEFAULT_OUTPUT), writer.featuresFile).replaceAll("\\", "/"),
      targets: path.relative(resolve(DEFAULT_OUTPUT), writer.targetsFile).replaceAll("\\", "/"),
      splits: path.relative(resolve(DEFAULT_OUTPUT), writer.splitsFile).replaceAll("\\", "/"),
      times: path.relative(resolve(DEFAULT_OUTPUT), writer.timesFile).replaceAll("\\", "/"),
    },
    features,
    targets,
  };
}

function selectedFiles(directory: string) {
  return fs.readdirSync(directory)
    .filter((name) => /^\d{4}-\d{2}-\d{2}\.json$/.test(name))
    .map((name) => ({ name, start: Date.parse(`${name.slice(0, 10)}T00:00:00.000Z`) }))
    .filter((item) => item.start >= Math.floor(START / DAY_MS) * DAY_MS && item.start < END)
    .sort((left, right) => left.start - right.start)
    .map((item) => path.join(directory, item.name));
}

function selectSignals(ids: string[]) {
  const byId = new Map(buildSignalDefinitions().map((definition) => [definition.id, definition]));
  return ids.map((id) => {
    const definition = byId.get(id);
    if (!definition) throw new Error(`Unknown signal ${id}`);
    return definition;
  });
}

function trailingReturn(closes: Float64Array, seen: number, window: number, current: number) {
  if (seen <= window) return 0;
  const lag = closes[(seen - 1 - window + closes.length) % closes.length]!;
  return Math.log(current / lag) * 10_000;
}

function trailingVolatility(returns: Float64Array, seen: number, window: number) {
  let sum = 0;
  const count = Math.min(seen, window);
  for (let lag = 0; lag < count; lag += 1) {
    const value = returns[(seen - 1 - lag + returns.length) % returns.length]!;
    sum += value * value;
  }
  return Math.sqrt(sum);
}

function closeLocation(candle: SequentialCandle) {
  const range = candle.high - candle.low;
  return range > 0 ? (2 * candle.close - candle.high - candle.low) / range : 0;
}

function splitOf(time: number) {
  return time < TRAIN_END ? 0 : time < PRIMARY_END ? 1 : 2;
}

function feature(
  id: string,
  name: string,
  family: string,
  parameters: string,
  lookback: string,
  delay: string,
  kind: FeatureDefinition["kind"] = "continuous",
): FeatureDefinition {
  return { id, name, family, parameters, lookback, delay, kind };
}

function appendTyped(file: string, values: ArrayBufferView) {
  fs.appendFileSync(file, Buffer.from(values.buffer, values.byteOffset, values.byteLength));
}

function resolve(relativeOrAbsolute: string) {
  return path.isAbsolute(relativeOrAbsolute) ? relativeOrAbsolute : path.resolve(repoRoot, relativeOrAbsolute);
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) run();
