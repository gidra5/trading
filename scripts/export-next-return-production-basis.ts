import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import {
  readCandleShardReferenceSync,
  readDerivativesKlinesShardReferenceSync,
  readTradeFlowShardReferenceSync,
  type SequentialCandle,
} from "@trading/storage";
import {
  FuturesMinuteFeatureEngine,
  TradeFlowFeatureEngine,
} from "./analyze-forward-market-return-information.ts";
import {
  buildSignalDefinitions,
  IndicatorEngine,
} from "./analyze-technical-indicator-predictiveness.ts";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const DAY_MS = 86_400_000;
const EXAMPLES_PER_SPLIT = 65_536;
const CALIBRATION_EXAMPLES = 16_384;
const HISTORY_RETURNS = 120;
const MAX_WINDOW = 14_400;
const START = Date.parse("2026-07-18T00:00:00.000Z");
const CALIBRATION_START = Date.parse("2026-08-02T12:00:00.000Z");
const TRAIN_END = Date.parse("2026-08-03T00:00:00.000Z");
const VALIDATION_END = Date.parse("2026-08-10T00:00:00.000Z");
const END = Date.parse("2026-08-17T00:00:00.000Z");
const OUTPUT = "data/training/datasets/next-return-production-basis-4l-65k-v1";
const CALIBRATION_OUTPUT = "data/training/datasets/next-return-production-basis-calibration-16k-v1";
const CALIBRATION_TAIL_OUTPUT = "data/training/datasets/next-return-production-basis-calibration-tail-16k-v1";
const TEMPORAL_OUTPUT = "data/training/datasets/next-return-production-basis-history120-4l-65k-v1";
const TEMPORAL_CALIBRATION_OUTPUT = "data/training/datasets/next-return-production-basis-history120-calibration-16k-v1";
const COMPACT_TEMPORAL_OUTPUT = "data/training/datasets/next-return-production-basis-history120-4l-train512k-v1";
const SPOT_1S = "data/market/immutable/refs/candles/spot-btcusdt/btcusdt/1s";
const SPOT_1M = "data/market/immutable/refs/candles";
const FLOW = "data/market/immutable/refs/research/trade-flow/spot-btcusdt/btcusdt/1s";
const FUTURES = "data/market/immutable/refs/research/derivatives-klines/usdm-futures/btcusdt/1m";
const VOLATILITY_WINDOWS = [5, 15, 60, 300, 900, 1_800, 3_600, 14_400] as const;
const MEAN_ABSOLUTE_WINDOWS = [5, 15, 60, 300, 900, 3_600] as const;
const SPLIT_NAMES = ["train", "validation", "test"] as const;

interface FeatureDefinition {
  id: string;
  label: string;
  family: string;
  construction: string;
  source: string;
  availability: string;
}

interface PendingRow {
  split: number;
  time: number;
  close: number;
  features?: Float32Array;
  originIndex?: number;
}

interface SplitBuffer {
  features?: Float32Array;
  featureFile?: number;
  origins?: Int32Array;
  timelineRows: number;
  timelineInitialized: boolean;
  targets: Float32Array;
  times: Float64Array;
  rows: number;
}

class ReturnRing {
  readonly values = new Float64Array(MAX_WINDOW);
  seen = 0;

  push(value: number): void {
    this.values[this.seen % this.values.length] = value;
    this.seen += 1;
  }

  lag(lag: number): number {
    if (lag < 0 || lag >= Math.min(this.seen, this.values.length)) return 0;
    return this.values[(this.seen - 1 - lag + this.values.length) % this.values.length]!;
  }

  moments(window: number): { meanAbsolute: number; rms: number; active: number } {
    const count = Math.min(window, this.seen, this.values.length);
    let absolute = 0;
    let square = 0;
    let active = 0;
    for (let lag = 0; lag < count; lag += 1) {
      const value = this.lag(lag);
      absolute += Math.abs(value);
      square += value * value;
      active += value !== 0 ? 1 : 0;
    }
    return {
      meanAbsolute: count > 0 ? absolute / count : 0,
      rms: count > 0 ? Math.sqrt(square / count) : 0,
      active,
    };
  }
}

class FeatureHistory {
  readonly values: Float32Array;
  seen = 0;

  constructor(readonly channelCount: number) {
    this.values = new Float32Array(HISTORY_RETURNS * channelCount);
  }

  push(featureRow: Float32Array): void {
    const row = this.seen % HISTORY_RETURNS;
    this.values[row * this.channelCount] = featureRow[HISTORY_RETURNS - 1]!;
    this.values.set(featureRow.subarray(HISTORY_RETURNS), row * this.channelCount + 1);
    this.seen += 1;
  }

  flattenChannelMajor(): Float32Array {
    if (this.seen < HISTORY_RETURNS) throw new Error("feature history is not warm");
    const output = new Float32Array(HISTORY_RETURNS * this.channelCount);
    let offset = 0;
    for (let channel = 0; channel < this.channelCount; channel += 1) {
      for (let lag = HISTORY_RETURNS - 1; lag >= 0; lag -= 1) {
        const row = (this.seen - 1 - lag + HISTORY_RETURNS) % HISTORY_RETURNS;
        output[offset] = this.values[row * this.channelCount + channel]!;
        offset += 1;
      }
    }
    return output;
  }

  flattenTimeMajor(): Float32Array {
    if (this.seen < HISTORY_RETURNS) throw new Error("feature history is not warm");
    const output = new Float32Array(HISTORY_RETURNS * this.channelCount);
    for (let lag = HISTORY_RETURNS - 1, outputRow = 0; lag >= 0; lag -= 1, outputRow += 1) {
      const row = (this.seen - 1 - lag + HISTORY_RETURNS) % HISTORY_RETURNS;
      output.set(
        this.values.subarray(row * this.channelCount, (row + 1) * this.channelCount),
        outputRow * this.channelCount,
      );
    }
    return output;
  }

  flattenRecentTimeMajor(seconds: number): Float32Array {
    if (!Number.isInteger(seconds) || seconds < 1 || seconds > HISTORY_RETURNS) {
      throw new Error(`feature history seconds must be in [1, ${HISTORY_RETURNS}]`);
    }
    if (this.seen < seconds) throw new Error("feature history is not warm");
    const output = new Float32Array(seconds * this.channelCount);
    for (let lag = seconds - 1, outputRow = 0; lag >= 0; lag -= 1, outputRow += 1) {
      const row = (this.seen - 1 - lag + HISTORY_RETURNS) % HISTORY_RETURNS;
      output.set(
        this.values.subarray(row * this.channelCount, (row + 1) * this.channelCount),
        outputRow * this.channelCount,
      );
    }
    return output;
  }

  latest(): Float32Array {
    if (this.seen < 1) throw new Error("feature history is empty");
    const row = (this.seen - 1) % HISTORY_RETURNS;
    return this.values.subarray(row * this.channelCount, (row + 1) * this.channelCount);
  }
}

function definitions(): FeatureDefinition[] {
  const output: FeatureDefinition[] = [];
  for (let lag = HISTORY_RETURNS - 1; lag >= 0; lag -= 1) output.push({
    id: `return-lag-${lag}s`,
    label: `Signed log return ${lag === 0 ? "latest" : `${lag}s before latest`}`,
    family: "raw return sequence",
    construction: "log(close_t / close_t-1)",
    source: "BTCUSDT spot 1s candle",
    availability: "completed second",
  });
  const add = (id: string, label: string, family: string, construction: string,
    source = "BTCUSDT spot 1s candle", availability = "through origin") => output.push({
      id, label, family, construction, source, availability,
    });
  add("return-latest-vol-normalized", "Latest return / RMS(60s)", "return state", "r[t] / exp(v_60s)");
  add("return-lag-1-vol-normalized", "Second-latest return / RMS(60s)", "return state", "r[t-1] / exp(v_60s)");
  add("return-latest-is-zero", "Latest return is exactly zero", "activity", "1[r[t] = 0]");
  add("return-lag-1-is-zero", "Second-latest return is exactly zero", "activity", "1[r[t-1] = 0]");
  add("active-fraction-10s", "Active-return fraction, 10s", "activity", "count(r != 0) / 10");
  add("active-fraction-60s", "Active-return fraction, 60s", "activity", "count(r != 0) / 60");
  add("zero-run-age-log", "Clipped zero-run age", "activity", "log1p(min(consecutive zero returns, 3600))");
  add("log-rms-anchor-1h", "Log-RMS volatility anchor, 1h", "volatility", "v_3600");
  for (const window of VOLATILITY_WINDOWS.filter((value) => value !== 3_600)) {
    add(`log-rms-${window}s-minus-1h`, `Log-RMS ${window}s minus 1h anchor`, "volatility", `v_${window} - v_3600`);
  }
  for (const window of MEAN_ABSOLUTE_WINDOWS) {
    add(`log-mean-absolute-return-${window}s`, `Log mean absolute return, ${window}s`, "volatility", `log(eps + mean(abs(r)), ${window}s)`);
  }
  add("rsi-2-mapped", "RSI(2s), mapped to [-1,1]", "price dynamics", "(RSI(2)-50)/50");
  add("ema-acceleration-2s-1s-vol-normalized", "EMA acceleration(2s,1s) / RMS(5s)", "price dynamics", "EMA acceleration bps / RMS bps(5s)");
  add("ema-slope-8s-8s-vol-normalized", "EMA slope(8s,8s) / RMS(15s)", "price dynamics", "EMA slope bps / RMS bps(15s)");
  add("completed-1h-log-base-volume", "Last completed 1h log base volume", "volume regime", "log1p(base volume)", "BTCUSDT spot 1s candles", "after UTC hour close");
  add("completed-1h-volume-observed", "Completed-hour volume observed", "availability", "Boolean observed flag", "BTCUSDT spot 1s candles", "known at origin");
  add("completed-1h-volume-age-log-seconds", "Completed-hour volume age", "availability", "log1p(seconds since completed hour)", "BTCUSDT spot 1s candles", "known at origin");
  add("range-1s-vol-normalized", "Completed 1s range / RMS(60s)", "candle shape", "log1p(log(high/low) / RMS(60s))");
  add("close-location-1s", "Completed 1s close location", "candle shape", "(2*close-high-low)/(high-low)");
  add("haar-adjacent-contrast-16s", "Normalized Haar adjacent-return contrast", "local multiscale state", "(r[t-1]-r[t])/(sqrt(2)*sqrt(sum_16(r^2)))");
  add("efficiency-absolute-16s", "Signed 16s absolute-return efficiency", "path state", "sum(r)/sum(abs(r))");
  add("efficiency-rms-16s", "Signed 16s variance-normalized efficiency", "path state", "sum(r)/sqrt(sum(r^2))");
  for (const age of [1, 2]) for (const side of ["sell", "no-trade", "buy"]) add(
    `spot-last-aggressor-${side}-${age}s`, `Spot last aggressor ${side}, ${age}s old`, "spot trade flow", "one-hot last aggressor side", "Binance spot aggregate trades", "completed second",
  );
  add("spot-quote-imbalance-1s", "Spot taker quote-volume imbalance, 1s", "spot trade flow", "(buy-sell)/(buy+sell)", "Binance spot aggregate trades", "completed second");
  add("spot-quote-imbalance-ema-2s", "Spot quote imbalance EMA(2s)", "spot trade flow", "separate buy/sell quote EMAs then imbalance", "Binance spot aggregate trades", "completed second");
  add("spot-quote-imbalance-ema-8s", "Spot quote imbalance EMA(8s)", "spot trade flow", "separate buy/sell quote EMAs then imbalance", "Binance spot aggregate trades", "completed second");
  add("spot-aggregate-count-imbalance-1s", "Spot aggregate-trade count imbalance, 1s", "spot trade flow", "(buy-sell)/(buy+sell)", "Binance spot aggregate trades", "completed second");
  add("futures-log-trade-count-1m", "USD-M futures log trade count, completed 1m", "futures activity", "log1p(trade count)", "Binance BTCUSDT USD-M 1m kline", "after minute close");
  add("futures-range-1m", "USD-M futures high-low range, completed 1m", "futures activity", "10000*log(high/low)", "Binance BTCUSDT USD-M 1m kline", "after minute close");
  add("futures-observed", "Futures source observed", "availability", "Boolean observed flag", "Binance BTCUSDT USD-M 1m kline", "known at origin");
  add("futures-age-log-seconds", "Futures feature age", "availability", "log1p(seconds since completed minute)", "Binance BTCUSDT USD-M 1m kline", "known at origin");
  add("eth-minus-btc-log-rms-30m", "ETH minus BTC log-RMS volatility, 30m", "cross-market regime", "v_ETH,30m-v_BTC,30m", "ETHUSDT and BTCUSDT spot 1m candles", "after minute close");
  add("eth-minus-btc-log-rms-60m", "ETH minus BTC log-RMS volatility, 60m", "cross-market regime", "v_ETH,60m-v_BTC,60m", "ETHUSDT and BTCUSDT spot 1m candles", "after minute close");
  add("eth-volatility-observed", "ETH volatility observed", "availability", "Boolean observed flag", "ETHUSDT spot 1m candle", "known at origin");
  add("eth-volatility-age-log-seconds", "ETH volatility feature age", "availability", "log1p(seconds since completed minute)", "ETHUSDT spot 1m candle", "known at origin");
  for (const phase of ["second", "minute", "hour", "day-of-week"]) {
    add(`calendar-${phase}-sin`, `UTC ${phase} sine`, "calendar", `sin(2*pi*${phase}/period)`, "UTC clock", "known before origin");
    add(`calendar-${phase}-cos`, `UTC ${phase} cosine`, "calendar", `cos(2*pi*${phase}/period)`, "UTC clock", "known before origin");
  }
  return output;
}

function temporalDefinitions(
  base: FeatureDefinition[], seconds = HISTORY_RETURNS, timeMajor = false,
): FeatureDefinition[] {
  const latestReturn = base[HISTORY_RETURNS - 1]!;
  const channels = [latestReturn, ...base.slice(HISTORY_RETURNS)];
  const output: FeatureDefinition[] = [];
  const add = (channel: FeatureDefinition, lag: number) => {
      output.push(lag === 0 ? channel : {
        ...channel,
        id: `${channel.id}-lag-${lag}s`,
        label: `${channel.label}, observed ${lag}s earlier`,
        availability: `${channel.availability}; value fixed ${lag}s before origin`,
      });
  };
  if (timeMajor) {
    for (let lag = seconds - 1; lag >= 0; lag -= 1) for (const channel of channels) add(channel, lag);
  } else {
    for (const channel of channels) for (let lag = seconds - 1; lag >= 0; lag -= 1) add(channel, lag);
  }
  return output;
}

function logRms(ring: ReturnRing, window: number): number {
  const rms = ring.moments(window).rms;
  return 0.5 * Math.log(1e-16 + rms * rms);
}

function oneHotSide(value: number): number[] {
  return [value < 0 ? 1 : 0, value === 0 ? 1 : 0, value > 0 ? 1 : 0];
}

function phasePair(value: number, period: number): [number, number] {
  const angle = 2 * Math.PI * value / period;
  return [Math.sin(angle), Math.cos(angle)];
}

function splitOf(time: number): number {
  if (time >= START && time < TRAIN_END) return 0;
  if (time >= TRAIN_END && time < VALIDATION_END) return 1;
  if (time >= VALIDATION_END && time < END) return 2;
  return -1;
}

function selectSignals(ids: string[]) {
  const byId = new Map(buildSignalDefinitions().map((definition) => [definition.id, definition]));
  return ids.map((id) => {
    const definition = byId.get(id);
    if (!definition) throw new Error(`Unknown signal ${id}`);
    return definition;
  });
}

function ref(directory: string, day: string): string {
  return path.resolve(repoRoot, directory, `${day}.json`);
}

function candleRef(symbol: string, day: string): string {
  return ref(path.join(SPOT_1M, `spot-${symbol}`, symbol, "1m"), day);
}

function days(): string[] {
  const result: string[] = [];
  for (let time = START; time < END; time += DAY_MS) result.push(new Date(time).toISOString().slice(0, 10));
  return result;
}

function writeTyped(file: string, values: ArrayBufferView): void {
  fs.writeFileSync(file, Buffer.from(values.buffer, values.byteOffset, values.byteLength));
}

function integerArgument(args: string[], name: string, fallback: number): number {
  const position = args.indexOf(name);
  if (position < 0) return fallback;
  const value = Number(args[position + 1]);
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new Error(`${name} requires a positive integer`);
  }
  return value;
}

function writeFloat32(file: number, values: Float32Array, elementOffset: number): void {
  const bytes = Buffer.from(values.buffer, values.byteOffset, values.byteLength);
  fs.writeSync(
    file, bytes, 0, bytes.byteLength,
    elementOffset * Float32Array.BYTES_PER_ELEMENT,
  );
}

function closeLocation(candle: SequentialCandle): number {
  return candle.high > candle.low
    ? (2 * candle.close - candle.high - candle.low) / (candle.high - candle.low)
    : 0;
}

function featureRow(
  featureDefinitions: FeatureDefinition[],
  ring: ReturnRing,
  indicatorValues: Float64Array,
  completedHourLogVolume: number,
  completedHourBoundary: number,
  candle: SequentialCandle,
  tradeValues: Float64Array,
  futuresValues: Float64Array,
  completedMinuteBoundary: number,
  ethReturns: ReturnRing,
  btcMinuteReturns: ReturnRing,
): Float32Array {
  const values: number[] = [];
  for (let lag = HISTORY_RETURNS - 1; lag >= 0; lag -= 1) values.push(ring.lag(lag));
  const moments5 = ring.moments(5);
  const moments10 = ring.moments(10);
  const moments15 = ring.moments(15);
  const moments16 = ring.moments(16);
  const moments60 = ring.moments(60);
  const rms60 = Math.max(moments60.rms, 1e-8);
  values.push(
    ring.lag(0) / rms60,
    ring.lag(1) / rms60,
    ring.lag(0) === 0 ? 1 : 0,
    ring.lag(1) === 0 ? 1 : 0,
    moments10.active / 10,
    moments60.active / 60,
  );
  let zeroAge = 0;
  while (zeroAge < Math.min(3_600, ring.seen) && ring.lag(zeroAge) === 0) zeroAge += 1;
  values.push(Math.log1p(zeroAge));
  const anchor = logRms(ring, 3_600);
  values.push(anchor);
  for (const window of VOLATILITY_WINDOWS.filter((value) => value !== 3_600)) values.push(logRms(ring, window) - anchor);
  for (const window of MEAN_ABSOLUTE_WINDOWS) values.push(Math.log(1e-12 + ring.moments(window).meanAbsolute));
  values.push(
    (indicatorValues[0]! - 50) / 50,
    indicatorValues[1]! / Math.max(moments5.rms * 10_000, 1e-4),
    indicatorValues[2]! / Math.max(moments15.rms * 10_000, 1e-4),
    completedHourLogVolume,
    1,
    Math.log1p(Math.max(0, (candle.openTime + 1_000 - completedHourBoundary) / 1_000)),
    Math.log1p(Math.log(candle.high / candle.low) / rms60),
    closeLocation(candle),
  );
  let sum16 = 0;
  let absolute16 = 0;
  let square16 = 0;
  for (let lag = 0; lag < Math.min(16, ring.seen); lag += 1) {
    const value = ring.lag(lag);
    sum16 += value;
    absolute16 += Math.abs(value);
    square16 += value * value;
  }
  values.push(
    (ring.lag(1) - ring.lag(0)) / Math.max(Math.SQRT2 * Math.sqrt(square16), 1e-12),
    absolute16 > 0 ? sum16 / absolute16 : 0,
    square16 > 0 ? sum16 / Math.sqrt(square16) : 0,
    ...oneHotSide(tradeValues[15]!),
    ...oneHotSide(tradeValues[29]!),
    tradeValues[0]!,
    tradeValues[21]!,
    tradeValues[22]!,
    tradeValues[3]!,
    futuresValues[12]!,
    futuresValues[7]!,
    1,
    Math.log1p(Math.max(0, (candle.openTime + 1_000 - completedMinuteBoundary) / 1_000)),
    logRms(ethReturns, 30) - logRms(btcMinuteReturns, 30),
    logRms(ethReturns, 60) - logRms(btcMinuteReturns, 60),
    1,
    Math.log1p(Math.max(0, (candle.openTime + 1_000 - completedMinuteBoundary) / 1_000)),
  );
  const origin = new Date(candle.openTime + 1_000);
  values.push(
    ...phasePair(origin.getUTCSeconds(), 60),
    ...phasePair(origin.getUTCMinutes(), 60),
    ...phasePair(origin.getUTCHours(), 24),
    ...phasePair(origin.getUTCDay(), 7),
  );
  if (values.length !== featureDefinitions.length || values.some((value) => !Number.isFinite(value))) {
    throw new Error(`Invalid feature row: ${values.length} values for ${featureDefinitions.length} definitions`);
  }
  return Float32Array.from(values);
}

export function run(args = process.argv.slice(2)): void {
  const calibrationOnly = args.includes("--calibration-only");
  const calibrationTail = args.includes("--calibration-tail");
  const allFeatureHistory = args.includes("--all-feature-history");
  const compactTemporal = args.includes("--compact-temporal");
  const featureHistorySeconds = integerArgument(
    args, "--feature-history-seconds", allFeatureHistory ? HISTORY_RETURNS : 1,
  );
  if (featureHistorySeconds < 1 || featureHistorySeconds > HISTORY_RETURNS) {
    throw new Error(`--feature-history-seconds must be in [1, ${HISTORY_RETURNS}]`);
  }
  if (calibrationTail && !calibrationOnly) {
    throw new Error("--calibration-tail requires --calibration-only export");
  }
  if (featureHistorySeconds > 1 && !allFeatureHistory) {
    throw new Error("--feature-history-seconds greater than 1 requires --all-feature-history");
  }
  if (compactTemporal && (!allFeatureHistory || calibrationOnly)) {
    throw new Error("--compact-temporal requires non-calibration --all-feature-history export");
  }
  const calibrationExamples = integerArgument(
    args, "--calibration-examples", CALIBRATION_EXAMPLES,
  );
  const splitNames: readonly string[] = calibrationOnly ? ["calibration"] : SPLIT_NAMES;
  const exampleCounts = calibrationOnly
    ? [calibrationExamples]
    : [
        integerArgument(args, "--train-examples", EXAMPLES_PER_SPLIT),
        integerArgument(args, "--validation-examples", EXAMPLES_PER_SPLIT),
        integerArgument(args, "--test-examples", EXAMPLES_PER_SPLIT),
      ];
  const outputArg = args.indexOf("--output-dir");
  const defaultOutput = calibrationTail ? CALIBRATION_TAIL_OUTPUT
    : compactTemporal ? COMPACT_TEMPORAL_OUTPUT : allFeatureHistory
    ? (calibrationOnly ? TEMPORAL_CALIBRATION_OUTPUT : TEMPORAL_OUTPUT)
    : (calibrationOnly ? CALIBRATION_OUTPUT : OUTPUT);
  const output = path.resolve(repoRoot, outputArg >= 0 ? args[outputArg + 1]! : defaultOutput);
  const snapshotDefinitions = definitions();
  const featureDefinitions = calibrationTail && !allFeatureHistory
    ? [snapshotDefinitions[HISTORY_RETURNS - 1]!, ...snapshotDefinitions.slice(HISTORY_RETURNS)]
    : allFeatureHistory
    ? temporalDefinitions(snapshotDefinitions, featureHistorySeconds, calibrationTail)
    : snapshotDefinitions;
  fs.mkdirSync(output, { recursive: true });
  const buffers: SplitBuffer[] = splitNames.map((name, split) => ({
    ...(allFeatureHistory && !calibrationTail ? {
      featureFile: fs.openSync(path.join(
        output,
        compactTemporal ? `${name}.timeline-features.f32` : `${name}.features.f32`,
      ), "w"),
      ...(compactTemporal ? { origins: new Int32Array(exampleCounts[split]!) } : {}),
    } : {
      features: new Float32Array(exampleCounts[split]! * featureDefinitions.length),
    }),
    targets: new Float32Array(exampleCounts[split]!),
    times: new Float64Array(exampleCounts[split]!),
    rows: 0,
    timelineRows: 0,
    timelineInitialized: false,
  }));
  if (allFeatureHistory && !calibrationTail && !compactTemporal) for (let split = 0; split < buffers.length; split += 1) {
    fs.ftruncateSync(
      buffers[split]!.featureFile!,
      exampleCounts[split]! * featureDefinitions.length * Float32Array.BYTES_PER_ELEMENT,
    );
  }
  const ring = new ReturnRing();
  const featureHistory = new FeatureHistory(1 + snapshotDefinitions.length - HISTORY_RETURNS);
  const ethReturns = new ReturnRing();
  const btcMinuteReturns = new ReturnRing();
  const trade = new TradeFlowFeatureEngine();
  const futures = new FuturesMinuteFeatureEngine();
  const indicatorValues = new Float64Array(3);
  const signalDefinitions = selectSignals(["rsi-2", "ema-acceleration-2-1", "ema-slope-8-8"]);
  let indicator: IndicatorEngine | undefined;
  let previousClose = Number.NaN;
  let previousEthClose = Number.NaN;
  let previousBtcMinuteClose = Number.NaN;
  let pending: PendingRow | undefined;
  let hourBucket = -1;
  let hourVolume = 0;
  let completedHourLogVolume = Number.NaN;
  let completedHourBoundary = Number.NaN;
  let completedMinuteBoundary = Number.NaN;
  let calibrationIntervalStart = CALIBRATION_START;
  const calibrationScanStart = TRAIN_END - Math.max(
    TRAIN_END - CALIBRATION_START,
    Math.ceil(calibrationExamples / 0.2) * 1_000 + MAX_WINDOW * 1_000,
  );

  for (const day of days()) {
    if (calibrationTail
      && Date.parse(`${day}T00:00:00.000Z`) < calibrationScanStart - DAY_MS) continue;
    const seconds = readCandleShardReferenceSync(ref(SPOT_1S, day));
    const flow = readTradeFlowShardReferenceSync(ref(FLOW, day));
    const btcMinutes = readCandleShardReferenceSync(candleRef("btcusdt", day));
    const ethMinutes = readCandleShardReferenceSync(candleRef("ethusdt", day));
    const futureMinutes = readDerivativesKlinesShardReferenceSync(ref(FUTURES, day));
    if (seconds.length !== 86_400 || flow.length !== 86_400 || btcMinutes.length !== 1_440
      || ethMinutes.length !== 1_440 || futureMinutes.length !== 1_440) {
      throw new Error(`${day}: incomplete source axis`);
    }
    for (let second = 0; second < seconds.length; second += 1) {
      const candle = seconds[second]!;
      const returnValue = Number.isFinite(previousClose) ? Math.log(candle.close / previousClose) : 0;
      if (pending) {
        const target = Math.fround(Math.log(candle.close / pending.close));
        const buffer = buffers[pending.split]!;
        if (target !== 0 && (calibrationTail || buffer.rows < exampleCounts[pending.split]!)) {
          const outputRow = calibrationTail
            ? buffer.rows % exampleCounts[pending.split]!
            : buffer.rows;
          if (compactTemporal) {
            if (pending.originIndex === undefined) throw new Error("compact origin index is missing");
            buffer.origins![outputRow] = pending.originIndex;
          } else if (buffer.featureFile !== undefined) {
            if (!pending.features) throw new Error("flattened temporal features are missing");
            const bytes = Buffer.from(
              pending.features.buffer,
              pending.features.byteOffset,
              pending.features.byteLength,
            );
            fs.writeSync(
              buffer.featureFile,
              bytes,
              0,
              bytes.byteLength,
              outputRow * bytes.byteLength,
            );
          } else {
            buffer.features!.set(pending.features!, outputRow * featureDefinitions.length);
          }
          buffer.targets[outputRow] = target;
          buffer.times[outputRow] = pending.time;
          buffer.rows += 1;
        }
      }
      ring.push(returnValue);
      if (!indicator) indicator = new IndicatorEngine(signalDefinitions, candle.close);
      else indicator.update(candle.close);
      indicator.values(indicatorValues);
      trade.update(flow[second]!);
      const bucket = Math.floor(candle.openTime / 3_600_000);
      if (bucket !== hourBucket) {
        hourBucket = bucket;
        hourVolume = 0;
      }
      hourVolume += candle.volume;
      const boundary = candle.openTime + 1_000;
      if (boundary % 3_600_000 === 0) {
        completedHourLogVolume = Math.log1p(hourVolume);
        completedHourBoundary = boundary;
      }
      if (boundary % 60_000 === 0) {
        const minuteIndex = Math.floor(second / 60);
        const btcMinute = btcMinutes[minuteIndex]!;
        const ethMinute = ethMinutes[minuteIndex]!;
        btcMinuteReturns.push(Number.isFinite(previousBtcMinuteClose)
          ? Math.log(btcMinute.close / previousBtcMinuteClose) : 0);
        ethReturns.push(Number.isFinite(previousEthClose)
          ? Math.log(ethMinute.close / previousEthClose) : 0);
        previousBtcMinuteClose = btcMinute.close;
        previousEthClose = ethMinute.close;
        futures.update(futureMinutes[minuteIndex], btcMinute.close, btcMinute.volume);
        completedMinuteBoundary = boundary;
      }
      previousClose = candle.close;
      const split = calibrationOnly
        ? (candle.openTime >= (calibrationTail ? calibrationScanStart : CALIBRATION_START)
          && candle.openTime < TRAIN_END ? 0 : -1)
        : splitOf(candle.openTime);
      const ready = ring.seen >= MAX_WINDOW && ethReturns.seen >= 60 && futures.valid
        && trade.valid && Number.isFinite(completedHourLogVolume)
        && Number.isFinite(completedHourBoundary)
        && Number.isFinite(completedMinuteBoundary);
      const snapshot = ready ? featureRow(
        snapshotDefinitions, ring, indicatorValues, completedHourLogVolume,
        completedHourBoundary, candle, trade.values(), futures.values(),
        completedMinuteBoundary, ethReturns, btcMinuteReturns,
      ) : undefined;
      if (snapshot && (allFeatureHistory || calibrationTail)) featureHistory.push(snapshot);
      const buffer = split >= 0 ? buffers[split] : undefined;
      const needsOrigin = split >= 0
        && (calibrationTail || buffer!.rows < exampleCounts[split]!) && ready
        && (!allFeatureHistory || featureHistory.seen >= HISTORY_RETURNS);
      let compactOriginIndex: number | undefined;
      if (needsOrigin && compactTemporal) {
        if (!buffer!.timelineInitialized) {
          const initial = featureHistory.flattenTimeMajor();
          writeFloat32(buffer!.featureFile!, initial, 0);
          buffer!.timelineRows = HISTORY_RETURNS;
          buffer!.timelineInitialized = true;
        } else {
          writeFloat32(
            buffer!.featureFile!, featureHistory.latest(),
            buffer!.timelineRows * featureHistory.channelCount,
          );
          buffer!.timelineRows += 1;
        }
        compactOriginIndex = buffer!.timelineRows - 1;
      }
      pending = needsOrigin ? {
            split,
            time: boundary,
            close: candle.close,
            ...(compactTemporal
              ? { originIndex: compactOriginIndex }
              : { features: calibrationTail && allFeatureHistory
                  ? featureHistory.flattenRecentTimeMajor(featureHistorySeconds)
                  : calibrationTail
                  ? featureHistory.latest()
                  : allFeatureHistory
                  ? featureHistory.flattenChannelMajor()
                  : snapshot! }),
          }
        : undefined;
    }
    console.error(`${day}: ${buffers.map((buffer) => buffer.rows.toLocaleString()).join(" / ")} clean ${splitNames.join("/")} rows`);
    if (calibrationTail && Date.parse(`${day}T00:00:00.000Z`) + DAY_MS >= TRAIN_END) break;
    if (!calibrationTail
      && buffers.every((buffer, split) => buffer.rows === exampleCounts[split])) break;
  }
  if (buffers.some((buffer, split) => calibrationTail
    ? buffer.rows < exampleCounts[split]!
    : buffer.rows !== exampleCounts[split]!)) {
    throw new Error(`Insufficient clean rows: ${buffers.map((buffer) => buffer.rows).join(", ")}`);
  }
  buffers.forEach((buffer, split) => {
    const name = splitNames[split]!;
    if (calibrationTail && buffer.rows <= exampleCounts[split]!) {
      calibrationIntervalStart = buffer.times[0]!;
    }
    if (buffer.featureFile !== undefined) fs.closeSync(buffer.featureFile);
    else if (calibrationTail && buffer.rows > exampleCounts[split]!) {
      const count = exampleCounts[split]!;
      const start = buffer.rows % count;
      const orderedFeatures = new Float32Array(count * featureDefinitions.length);
      const orderedTargets = new Float32Array(count);
      const orderedTimes = new Float64Array(count);
      for (let row = 0; row < count; row += 1) {
        const source = (start + row) % count;
        orderedFeatures.set(
          buffer.features!.subarray(
            source * featureDefinitions.length,
            (source + 1) * featureDefinitions.length,
          ),
          row * featureDefinitions.length,
        );
        orderedTargets[row] = buffer.targets[source]!;
        orderedTimes[row] = buffer.times[source]!;
      }
      calibrationIntervalStart = orderedTimes[0]!;
      writeTyped(path.join(output, `${name}.features.f32`), orderedFeatures);
      writeTyped(path.join(output, `${name}.targets.f32`), orderedTargets);
      writeTyped(path.join(output, `${name}.times.f64`), orderedTimes);
      return;
    } else writeTyped(path.join(output, `${name}.features.f32`), buffer.features!);
    if (buffer.origins) writeTyped(path.join(output, `${name}.origins.i32`), buffer.origins);
    writeTyped(path.join(output, `${name}.targets.f32`), buffer.targets);
    writeTyped(path.join(output, `${name}.times.f64`), buffer.times);
  });
  const manifest = calibrationOnly ? {
    version: 1,
    generatedAt: new Date().toISOString(),
    contract: calibrationTail
      ? allFeatureHistory
        ? `next-return-production-basis-history${featureHistorySeconds}-clean-pre-validation-tail-calibration-v1`
        : "next-return-production-basis-clean-immediate-pre-validation-tail-calibration-v1"
      : allFeatureHistory
      ? `next-return-production-basis-history${featureHistorySeconds}-clean-pre-validation-calibration-v1`
      : "next-return-production-basis-clean-pre-validation-calibration-v1",
    featureCount: featureDefinitions.length,
    target: "next completed 1s BTCUSDT signed log return",
    targetFilter: "float32-exact-zero targets excluded; zeros remain in input histories",
    rawHistoryReturns: HISTORY_RETURNS,
    featureHistorySeconds,
    temporalChannelCount: allFeatureHistory ? featureHistory.channelCount : undefined,
    temporalLayout: allFeatureHistory
      ? (calibrationTail ? "time-major" : "channel-major")
      : undefined,
    examples: calibrationExamples,
    interval: {
      start: new Date(calibrationIntervalStart).toISOString(),
      endExclusive: new Date(TRAIN_END).toISOString(),
    },
    relationship: {
      training: "strictly after the source model's retained training examples; evaluators verify timestamps",
      validation: calibrationTail
        ? `the final ${calibrationExamples.toLocaleString("en-US")} clean active-return examples immediately before the 2026-08-03 validation boundary`
        : "strictly before the 2026-08-03 validation boundary",
      test: "untouched",
    },
    features: featureDefinitions,
    files: {
      calibration: {
        features: "calibration.features.f32",
        targets: "calibration.targets.f32",
        times: "calibration.times.f64",
      },
    },
  } : {
    version: 1,
    generatedAt: new Date().toISOString(),
    contract: allFeatureHistory
      ? "next-return-production-basis-history120-clean-chronological-v1"
      : "next-return-production-basis-clean-chronological-v1",
    featureCount: featureDefinitions.length,
    target: "next completed 1s BTCUSDT signed log return",
    targetFilter: "float32-exact-zero targets excluded from every split; zeros remain in input histories",
    rawHistoryReturns: HISTORY_RETURNS,
    featureHistorySeconds,
    temporalChannelCount: allFeatureHistory ? featureHistory.channelCount : undefined,
    storageLayout: compactTemporal ? "temporal-channel-timeline-v1" : "flattened-feature-matrix-v1",
    examplesPerSplit: exampleCounts.every((value) => value === exampleCounts[0])
      ? exampleCounts[0]
      : undefined,
    examplesBySplit: Object.fromEntries(SPLIT_NAMES.map((name, split) => [name, exampleCounts[split]])),
    timelineRowsBySplit: compactTemporal
      ? Object.fromEntries(SPLIT_NAMES.map((name, split) => [name, buffers[split]!.timelineRows]))
      : undefined,
    splits: {
      train: { start: new Date(START).toISOString(), endExclusive: new Date(TRAIN_END).toISOString() },
      validation: { start: new Date(TRAIN_END).toISOString(), endExclusive: new Date(VALIDATION_END).toISOString() },
      test: { start: new Date(VALIDATION_END).toISOString(), endExclusive: new Date(END).toISOString() },
    },
    includedSources: ["BTCUSDT spot 1s/1m", "BTCUSDT spot aggregate trade flow 1s", "BTCUSDT USD-M futures 1m", "ETHUSDT spot 1m", "UTC calendar"],
    omittedForCoverage: [
      "spot order book: only 18,954 fresh recent origins, below 65,536 clean examples per split",
      "cross-exchange books, Deribit, liquidations, GDELT, mempool and macro/ETF feeds: live-only or insufficient point-in-time history",
      "futures positioning/basis/funding and research-only coordinates: explicitly excluded by the recommendation document",
    ],
    features: featureDefinitions,
    files: Object.fromEntries(SPLIT_NAMES.map((name) => [name, compactTemporal ? {
      timelineFeatures: `${name}.timeline-features.f32`, origins: `${name}.origins.i32`,
      targets: `${name}.targets.f32`, times: `${name}.times.f64`,
    } : {
      features: `${name}.features.f32`, targets: `${name}.targets.f32`, times: `${name}.times.f64`,
    }])),
  };
  fs.writeFileSync(path.join(output, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
  console.log(`Wrote ${path.relative(repoRoot, output)}: ${featureDefinitions.length} features x ${exampleCounts.map((value) => value.toLocaleString()).join("/")} clean ${splitNames.join("/")} examples.`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  try { run(); } catch (error) {
    console.error(error instanceof Error ? error.stack ?? error.message : String(error));
    process.exitCode = 1;
  }
}
