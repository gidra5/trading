import fs from "node:fs";
import path from "node:path";
import readline from "node:readline";
import { fileURLToPath, pathToFileURL } from "node:url";
import {
  readCandleShardReferenceSync,
  readDerivativesKlinesShardReferenceSync,
  readDerivativesMetricsShardReferenceSync,
  readTradeFlowShardReferenceSync,
  type SequentialCandle,
} from "@trading/storage";
import {
  FuturesMetricsFeatureEngine,
  FuturesMinuteFeatureEngine,
  TradeFlowFeatureEngine,
} from "./analyze-forward-market-return-information.ts";
import {
  buildSpotBookFeatureDefinitions,
  spotSnapshotValues,
  type SpotBookSnapshot,
  validSnapshot,
} from "./analyze-spot-order-book-return-information.ts";
import {
  buildSignalDefinitions,
  IndicatorEngine,
} from "./analyze-technical-indicator-predictiveness.ts";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const START = Date.parse("2026-07-18T00:00:00.000Z");
const TRAIN_END = Date.parse("2026-08-03T00:00:00.000Z");
const PRIMARY_END = Date.parse("2026-08-10T00:00:00.000Z");
const END = Date.parse("2026-08-17T00:00:00.000Z");
const DAY_MS = 86_400_000;
const OUTPUT = "data/runtime-cache/global-feature-basis-30d";
const BOOK_INPUT = "data/market/mutable/streams/spot-btcusdt/btcusdt-orderbook.jsonl";
const SPOT_1S = "data/market/immutable/refs/candles/spot-btcusdt/btcusdt/1s";
const SPOT_1M = "data/market/immutable/refs/candles";
const FLOW = "data/market/immutable/refs/research/trade-flow/spot-btcusdt/btcusdt/1s";
const FUTURES = "data/market/immutable/refs/research/derivatives-klines/usdm-futures/btcusdt/1m";
const METRICS = "data/market/immutable/refs/research/derivatives-metrics/usdm-futures/btcusdt/5m";
const GDELT = "data/market/immutable/external/gdelt-doc-crypto-15m/2026-07-18_2026-08-16.json";
const HORIZONS = [1, 60, 900, 3_600] as const;

interface Definition {
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
  values: number[];
  targets: Float32Array;
  remaining: number;
  split: number;
}

export async function run(args = process.argv.slice(2)): Promise<void> {
  const value = (name: string) => {
    const index = args.indexOf(name);
    return index < 0 ? undefined : args[index + 1];
  };
  const output = resolve(value("--output-dir") ?? OUTPUT);
  const bookInput = resolve(value("--book-input") ?? BOOK_INPUT);
  const gdeltFile = resolve(GDELT);
  const includeGdelt = fs.existsSync(gdeltFile);
  const definitions = featureDefinitions(includeGdelt);
  const bookAtBoundary = await readBookBoundaries(bookInput);
  const gdeltByTime = includeGdelt ? readGdelt(gdeltFile) : new Map<number, GdeltRow>();
  const features: number[] = [];
  const targets: number[] = [];
  const splits: number[] = [];
  const times: number[] = [];
  const schedule = new Map<number, Pending[]>();
  const signalDefinitions = selectSignals(["rsi-2", "ema-acceleration-2-1", "ema-slope-8-8"]);
  const indicatorValues = new Float64Array(3);
  let indicator: IndicatorEngine | undefined;
  const trade = new TradeFlowFeatureEngine();
  const futures = new FuturesMinuteFeatureEngine();
  const metrics = new FuturesMetricsFeatureEngine();
  const return60 = new Float64Array(60);
  let returnSeen = 0;
  let square60 = 0;
  let active60 = 0;
  let previousClose = Number.NaN;
  let hourBucket = -1;
  let hourVolume = 0;
  let completedHourLogVolume = Number.NaN;
  const minuteReturns = new Float64Array(240);
  let minuteSeen = 0;
  let previousMinuteClose = Number.NaN;
  const altHistories = new Map<string, { closes: Float64Array; returns: Float64Array; seen: number }>();
  for (const symbol of ["ethusdt", "solusdt", "bnbusdt", "dogeusdt"]) {
    altHistories.set(symbol, { closes: new Float64Array(6), returns: new Float64Array(60), seen: 0 });
  }
  const newsIntensityHistory: number[] = [];
  let newsValues = new Float64Array(6);
  let previousNewsTone = Number.NaN;
  let newsToneEma = Number.NaN;
  let latestNewsTime = Number.NaN;
  let lastNewsBucket = -1;

  for (const day of days()) {
    const seconds = readCandleShardReferenceSync(ref(SPOT_1S, day));
    const flow = readTradeFlowShardReferenceSync(ref(FLOW, day));
    const btcMinutes = readCandleShardReferenceSync(candleRef("btcusdt", day));
    const futureMinutes = readDerivativesKlinesShardReferenceSync(ref(FUTURES, day));
    const metricRows = readDerivativesMetricsShardReferenceSync(ref(METRICS, day));
    const altMinutes = new Map<string, SequentialCandle[]>();
    for (const symbol of altHistories.keys()) {
      altMinutes.set(symbol, readCandleShardReferenceSync(candleRef(symbol, day)));
    }
    if (seconds.length !== 86_400 || flow.length !== 86_400 || btcMinutes.length !== 1_440) {
      throw new Error(`${day}: incomplete recent research axis.`);
    }
    for (let second = 0; second < seconds.length; second += 1) {
      const candle = seconds[second]!;
      const due = schedule.get(candle.openTime);
      if (due) {
        for (const pending of due) {
          const targetIndex = HORIZONS.length - pending.remaining;
          pending.targets[targetIndex] = Math.log(candle.close / pending.originClose) * 10_000;
          pending.remaining -= 1;
          if (pending.remaining === 0) {
            features.push(...pending.values);
            targets.push(...pending.targets);
            splits.push(pending.split);
            times.push(pending.originTime);
          }
        }
        schedule.delete(candle.openTime);
      }
      const returnBps = Number.isFinite(previousClose)
        ? Math.log(candle.close / previousClose) * 10_000 : 0;
      const ringIndex = returnSeen % return60.length;
      if (returnSeen >= return60.length) {
        const removed = return60[ringIndex]!;
        square60 -= removed * removed;
        active60 -= removed !== 0 ? 1 : 0;
      }
      return60[ringIndex] = returnBps;
      returnSeen += 1;
      square60 += returnBps * returnBps;
      active60 += returnBps !== 0 ? 1 : 0;
      const bucket = Math.floor(candle.openTime / 3_600_000);
      if (bucket !== hourBucket) {
        hourBucket = bucket;
        hourVolume = 0;
      }
      hourVolume += candle.volume;
      if (!indicator) indicator = new IndicatorEngine(signalDefinitions, candle.close);
      else indicator.update(candle.close);
      indicator.values(indicatorValues);
      trade.update(flow[second]!);
      previousClose = candle.close;
      const boundary = candle.openTime + 1_000;
      if (boundary % 3_600_000 === 0) completedHourLogVolume = Math.log1p(hourVolume);
      if (boundary % 60_000 !== 0) continue;
      const minute = btcMinutes[Math.floor(second / 60)]!;
      const minuteReturn = Number.isFinite(previousMinuteClose)
        ? Math.log(minute.close / previousMinuteClose) * 10_000 : 0;
      minuteReturns[minuteSeen % minuteReturns.length] = minuteReturn;
      minuteSeen += 1;
      previousMinuteClose = minute.close;
      futures.update(futureMinutes[Math.floor(second / 60)], minute.close, minute.volume);
      const availableMetricTime = boundary - 300_000;
      if (availableMetricTime >= Date.parse(`${day}T00:00:00.000Z`)
        && availableMetricTime % 300_000 === 0) {
        metrics.update(metricRows[Math.floor((availableMetricTime % DAY_MS) / 300_000)], minute.close);
      }
      const crossValues: number[] = [];
      for (const [symbol, history] of altHistories) {
        const alt = altMinutes.get(symbol)![Math.floor(second / 60)]!;
        const prior = history.seen > 0 ? history.closes[(history.seen - 1) % history.closes.length]! : alt.close;
        const altReturn = Math.log(alt.close / prior) * 10_000;
        history.closes[history.seen % history.closes.length] = alt.close;
        history.returns[history.seen % history.returns.length] = altReturn;
        history.seen += 1;
        crossValues.push(
          altReturn,
          trailingAltReturn(history, 5, alt.close),
          trailingVolatility(history.returns, history.seen, 30),
          trailingVolatility(history.returns, history.seen, 60),
        );
      }
      if (returnSeen < 60 || minuteSeen < 240 || !Number.isFinite(completedHourLogVolume)
        || !trade.valid || !futures.valid || !metrics.valid) continue;
      const book = bookAtBoundary.get(boundary);
      const bookValues = book?.values ?? new Float64Array(buildSpotBookFeatureDefinitions().length);
      const newsBucket = Math.floor((boundary - 30 * 60_000) / (15 * 60_000)) * (15 * 60_000);
      if (includeGdelt && newsBucket > lastNewsBucket) {
        lastNewsBucket = newsBucket;
        const news = gdeltByTime.get(newsBucket);
        if (news?.observed && news.articleIntensity !== null && news.articleCount !== null) {
          const mean = average(newsIntensityHistory);
          const deviation = standardDeviation(newsIntensityHistory, mean);
          const tone = news.averageTone ?? previousNewsTone;
          const toneChange = Number.isFinite(tone) && Number.isFinite(previousNewsTone)
            ? tone - previousNewsTone : 0;
          const toneShock = Number.isFinite(tone) && Number.isFinite(newsToneEma)
            ? tone - newsToneEma : 0;
          newsValues = Float64Array.from([
            Math.log1p(news.articleCount),
            news.articleIntensity,
            deviation > 0 ? (news.articleIntensity - mean) / deviation : 0,
            Number.isFinite(tone) ? tone : 0,
            toneChange,
            toneShock,
          ]);
          newsIntensityHistory.push(news.articleIntensity);
          if (newsIntensityHistory.length > 96) newsIntensityHistory.shift();
          if (Number.isFinite(tone)) {
            newsToneEma = Number.isFinite(newsToneEma) ? newsToneEma + 2 / 97 * (tone - newsToneEma) : tone;
            previousNewsTone = tone;
          }
          latestNewsTime = news.time;
        }
      }
      const newsObserved = Number.isFinite(latestNewsTime) && boundary - (latestNewsTime + 30 * 60_000) <= 60 * 60_000;
      const values = [
        returnBps,
        returnBps === 0 ? 1 : 0,
        active60,
        Math.sqrt(Math.max(0, square60)),
        ...indicatorValues,
        completedHourLogVolume,
        Math.log(candle.high / candle.low) * 10_000,
        closeLocation(candle),
        trailingVolatility(minuteReturns, minuteSeen, 15),
        trailingVolatility(minuteReturns, minuteSeen, 30),
        trailingVolatility(minuteReturns, minuteSeen, 60),
        trailingVolatility(minuteReturns, minuteSeen, 240),
        Math.log(minute.high / minute.low) * 10_000,
        ...trade.values(),
        ...futures.values(),
        ...metrics.values(),
        ...crossValues,
        ...bookValues,
        book ? 1 : 0,
        book ? Math.max(0, boundary - book.eventTime) : 5_000,
        ...(includeGdelt ? [
          ...newsValues,
          newsObserved ? 1 : 0,
          newsObserved ? Math.max(0, boundary - (latestNewsTime + 30 * 60_000)) / 60_000 : 60,
        ] : []),
      ];
      if (values.length !== definitions.length) {
        throw new Error(`Feature count mismatch: ${values.length} vs ${definitions.length}`);
      }
      const pending: Pending = {
        originTime: candle.openTime,
        originClose: candle.close,
        values,
        targets: new Float32Array(HORIZONS.length),
        remaining: HORIZONS.length,
        split: splitOf(candle.openTime),
      };
      HORIZONS.forEach((horizon) => {
        const dueTime = candle.openTime + horizon * 1_000;
        const list = schedule.get(dueTime) ?? [];
        list.push(pending);
        schedule.set(dueTime, list);
      });
    }
    console.error(`${day}: ${times.length.toLocaleString()} completed rows`);
  }
  fs.mkdirSync(output, { recursive: true });
  writeTyped(path.join(output, "recent.features.f32"), Float32Array.from(features));
  writeTyped(path.join(output, "recent.targets.f32"), Float32Array.from(targets));
  writeTyped(path.join(output, "recent.splits.u8"), Uint8Array.from(splits));
  writeTyped(path.join(output, "recent.times.f64"), Float64Array.from(times));
  const manifest = {
    version: 1,
    generatedAt: new Date().toISOString(),
    objective: "Joint 30-day feature-subset discovery across recent public and local sources",
    split: {
      start: new Date(START).toISOString(),
      trainEndExclusive: new Date(TRAIN_END).toISOString(),
      primaryEndExclusive: new Date(PRIMARY_END).toISOString(),
      endExclusive: new Date(END).toISOString(),
      labels: ["train", "primary", "transfer"],
    },
    datasets: [{
      id: "recent",
      rows: times.length,
      featureCount: definitions.length,
      targetCount: HORIZONS.length,
      sampling: "one origin at every completed UTC minute; all inputs are available by that boundary",
      files: {
        features: "recent.features.f32",
        targets: "recent.targets.f32",
        splits: "recent.splits.u8",
        times: "recent.times.f64",
      },
      features: definitions,
      targets: [
        { id: "1s", minutes: 1 / 60 },
        { id: "1m", minutes: 1 },
        { id: "15m", minutes: 15 },
        { id: "1h", minutes: 60 },
      ],
    }],
  };
  fs.writeFileSync(path.join(output, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
  console.log(`Wrote ${path.relative(root, output)} with ${times.length.toLocaleString()} rows and ${definitions.length} features.`);
}

function featureDefinitions(includeGdelt: boolean): Definition[] {
  const core: Definition[] = [
    feature("previous-return-1s", "Previous 1s signed log return", "return history", "lag=1s", "1s", "latest completed second"),
    feature("previous-return-zero", "Previous return is exactly zero", "activity", "binary", "1s", "latest completed second", "binary"),
    feature("active-count-60s", "Active-return count", "activity", "window=60s", "60s", "through origin"),
    feature("realized-volatility-60s", "Realized volatility", "volatility", "sqrt(sum(r_1s^2))", "60s", "through origin"),
    feature("rsi-2s", "RSI", "price dynamics", "period=2s", "recursive", "through origin"),
    feature("ema-acceleration-2s-1s", "EMA acceleration", "price dynamics", "period=2s,horizon=1s", "recursive", "through origin"),
    feature("ema-slope-8s-8s", "EMA slope", "price dynamics", "period=8s,horizon=8s", "recursive", "through origin"),
    feature("completed-1h-log-volume", "Last completed 1h log volume", "volume regime", "UTC-aligned 1h", "1h", "after hour close"),
    feature("range-1s", "Completed 1s range", "candle shape", "10000*log(high/low)", "1s", "latest completed second"),
    feature("close-location-1s", "Completed 1s close location", "candle shape", "normalized OHLC location", "1s", "latest completed second"),
    ...[15, 30, 60, 240].map((window) => feature(`realized-volatility-${window}m`, "Realized volatility", "minute volatility", `sqrt(sum(r_1m^2)),window=${window}m`, `${window}m`, "through origin")),
    feature("range-1m", "Completed 1m range", "minute candle shape", "10000*log(high/low)", "1m", "latest completed minute"),
  ];
  const forward = [
    ...TradeFlowFeatureEngine.definitions(),
    ...FuturesMinuteFeatureEngine.definitions(),
    ...FuturesMetricsFeatureEngine.definitions(),
  ].map((item) => feature(item.id, item.label, item.family, item.label, inferredLookback(item.id), inferredDelay(item.source)));
  const cross = ["eth", "sol", "bnb", "doge"].flatMap((symbol) => [
    feature(`${symbol}-return-1m`, `${symbol.toUpperCase()} return`, "cross-market return", "window=1m", "1m", "latest completed minute"),
    feature(`${symbol}-return-5m`, `${symbol.toUpperCase()} return`, "cross-market return", "window=5m", "5m", "through origin"),
    feature(`${symbol}-realized-volatility-30m`, `${symbol.toUpperCase()} realized volatility`, "cross-market volatility", "window=30m", "30m", "through origin"),
    feature(`${symbol}-realized-volatility-60m`, `${symbol.toUpperCase()} realized volatility`, "cross-market volatility", "window=60m", "60m", "through origin"),
  ]);
  const book = buildSpotBookFeatureDefinitions().map((item) => feature(
    `spot-book-${item.id}`, item.label, `spot book ${item.family}`, item.label,
    "latest snapshot", "strictly before boundary; maximum age 5s",
  ));
  const gdelt = includeGdelt ? [
    feature("gdelt-log-article-count", "GDELT crypto article count", "news GDELT proxy", "log1p distinct article count", "latest 15m bucket", "conservative 30m publication lag"),
    feature("gdelt-article-intensity", "GDELT crypto article intensity", "news GDELT proxy", "matching / monitored articles", "latest 15m bucket", "conservative 30m publication lag"),
    feature("gdelt-intensity-zscore-24h", "GDELT intensity shock", "news GDELT proxy", "z-score versus trailing 24h", "24h", "conservative 30m publication lag"),
    feature("gdelt-average-tone", "GDELT average article tone", "news GDELT proxy", "document-level average tone", "latest 15m bucket", "conservative 30m publication lag"),
    feature("gdelt-tone-change-15m", "GDELT tone change", "news GDELT proxy", "change from prior observed bucket", "15m", "conservative 30m publication lag"),
    feature("gdelt-tone-shock-24h", "GDELT tone shock", "news GDELT proxy", "tone minus EMA(96 buckets)", "24h recursive", "conservative 30m publication lag"),
    feature("gdelt-observed", "GDELT proxy observed", "source metadata", "binary", "1h freshness", "known at boundary", "binary"),
    feature("gdelt-age-minutes", "GDELT proxy age", "source metadata", "minutes since conservative availability", "1h clipped", "known at boundary"),
  ] : [];
  return [
    ...core,
    ...forward,
    ...cross,
    ...book,
    feature("spot-book-observed", "Fresh spot book observed", "source metadata", "binary", "5s", "known at boundary", "binary"),
    feature("spot-book-age-ms", "Spot book age", "source metadata", "milliseconds", "5s clipped", "known at boundary"),
    ...gdelt,
  ];
}

interface GdeltRow {
  time: number;
  articleCount: number | null;
  articleIntensity: number | null;
  averageTone: number | null;
  observed: boolean;
}

function readGdelt(file: string): Map<number, GdeltRow> {
  if (!fs.existsSync(file)) throw new Error(`Missing GDELT history: ${file}`);
  const artifact = JSON.parse(fs.readFileSync(file, "utf8")) as { stepMs: number; rows: GdeltRow[] };
  if (artifact.stepMs !== 15 * 60_000 || !Array.isArray(artifact.rows)) throw new Error("Invalid GDELT history artifact.");
  return new Map(artifact.rows.map((row) => [row.time, row]));
}

async function readBookBoundaries(input: string): Promise<Map<number, { eventTime: number; values: Float64Array }>> {
  const output = new Map<number, { eventTime: number; values: Float64Array }>();
  if (!fs.existsSync(input)) return output;
  const reader = readline.createInterface({ input: fs.createReadStream(input), crlfDelay: Infinity });
  let previous: { snapshot: SpotBookSnapshot; values: Float64Array } | undefined;
  for await (const line of reader) {
    let snapshot: SpotBookSnapshot;
    try { snapshot = JSON.parse(line) as SpotBookSnapshot; } catch { continue; }
    if (!validSnapshot(snapshot)) continue;
    const values = spotSnapshotValues(snapshot, previous);
    previous = { snapshot, values };
    const boundary = Math.ceil(snapshot.eventTime / 60_000) * 60_000;
    if (boundary < START || boundary >= END || boundary - snapshot.eventTime > 5_000) continue;
    const current = output.get(boundary);
    if (!current || snapshot.eventTime > current.eventTime) output.set(boundary, { eventTime: snapshot.eventTime, values });
  }
  console.error(`Book coverage: ${output.size.toLocaleString()} fresh minute boundaries.`);
  return output;
}

function trailingAltReturn(
  history: { closes: Float64Array; seen: number },
  window: number,
  current: number,
): number {
  if (history.seen <= window) return 0;
  const lag = history.closes[(history.seen - 1 - window + history.closes.length) % history.closes.length]!;
  return Math.log(current / lag) * 10_000;
}

function trailingVolatility(values: Float64Array, seen: number, window: number): number {
  const count = Math.min(seen, window);
  let sum = 0;
  for (let lag = 0; lag < count; lag += 1) {
    const value = values[(seen - 1 - lag + values.length) % values.length]!;
    sum += value * value;
  }
  return Math.sqrt(sum);
}

function average(values: number[]): number {
  return values.length > 0 ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
}

function standardDeviation(values: number[], mean: number): number {
  if (values.length < 2) return 0;
  return Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length);
}

function closeLocation(candle: SequentialCandle): number {
  return candle.high > candle.low
    ? (2 * candle.close - candle.high - candle.low) / (candle.high - candle.low)
    : 0;
}

function inferredLookback(id: string): string {
  const match = /(?:-|^)(\d+)(s|m|h)(?:-|$)/.exec(id);
  return match ? `${match[1]}${match[2]}` : "latest completed observation";
}

function inferredDelay(source: string): string {
  if (source === "spot trade flow") return "latest completed second";
  if (source === "futures basis/flow") return "after 1m candle close";
  return "full 5m publication lag";
}

function feature(
  id: string,
  name: string,
  family: string,
  parameters: string,
  lookback: string,
  delay: string,
  kind: "binary" | "continuous" = "continuous",
): Definition {
  return { id, name, family, parameters, lookback, delay, kind };
}

function splitOf(time: number): number {
  if (time < TRAIN_END) return 0;
  if (time < PRIMARY_END) return 1;
  return 2;
}

function days(): string[] {
  const output: string[] = [];
  for (let time = START; time < END; time += DAY_MS) output.push(new Date(time).toISOString().slice(0, 10));
  return output;
}

function candleRef(symbol: string, day: string): string {
  return ref(path.join(SPOT_1M, `spot-${symbol}`, symbol, "1m"), day);
}

function ref(directory: string, day: string): string {
  return path.join(resolve(directory), `${day}.json`);
}

function selectSignals(ids: string[]) {
  const byId = new Map(buildSignalDefinitions().map((definition) => [definition.id, definition]));
  return ids.map((id) => {
    const definition = byId.get(id);
    if (!definition) throw new Error(`Unknown signal ${id}`);
    return definition;
  });
}

function writeTyped(file: string, array: ArrayBufferView): void {
  fs.writeFileSync(file, Buffer.from(array.buffer, array.byteOffset, array.byteLength));
}

function resolve(file: string): string { return path.resolve(root, file); }

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  run().catch((error: unknown) => {
    console.error(error instanceof Error ? error.stack ?? error.message : String(error));
    process.exitCode = 1;
  });
}
