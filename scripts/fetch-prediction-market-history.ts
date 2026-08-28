import fs from "node:fs/promises";
import { createReadStream, createWriteStream, type WriteStream } from "node:fs";
import path from "node:path";
import { finished } from "node:stream/promises";
import { createGzip, type Gzip } from "node:zlib";
import { fileURLToPath } from "node:url";
import {
  buildCausalTradeStates,
  buildCausalTradeStatesForOrigins,
  normalizeKalshiMinuteCandles,
  normalizeKalshiTrades,
  type KalshiCandlestickRaw,
  type KalshiTradeRaw,
  type PredictionTrade,
} from "./lib/prediction-market-data.ts";

const API = "https://external-api.kalshi.com/trade-api/v2";
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

export const ASSET_SERIES = [
  "KXBTC15M", "KXBTCD", "KXBTC",
  "KXETHD", "KXETH",
  "KXSOL15M",
  "KXXRP15M", "KXXRPD", "KXXRP",
  "KXBTCVSHYPE",
] as const;

export const GLOBAL_SERIES = [
  "KXFEDDECISION", "KXFED", "KXRATECUTCOUNT",
  "KXCPIYOY", "KXCPI", "KXPAYROLLS", "KXGDP", "KXU3", "KXRECSSNBER",
] as const;

const DEFAULT_FAST_SERIES = new Set(["KXBTC15M", "KXBTCD", "KXSOL15M", "KXXRP15M"]);

const SYSTEMIC_CATEGORIES = new Set([
  "Economics",
  "Financials",
  "Politics",
  "Elections",
  "World",
  "Commodities",
]);

const SYSTEMIC_KEYWORDS = /\b(ai|artificial intelligence|semiconductor|chip|nvidia|microsoft|google|apple|tesla|amazon|openai|spacex|pandemic|covid|vaccine|hurricane|earthquake|war|shipping|suez|oil|energy|nuclear|cyber|hack|tariff|sanction)\b/i;
const FAST_CRYPTO_KEYWORDS = /\b(15\s*m(?:in(?:ute)?)?|price|range|directional|above\/?below|up\s*(?:or|\/)\s*down)\b/i;

interface KalshiSeriesRaw {
  ticker: string;
  title?: string;
  category?: string;
  frequency?: string;
}

export interface RelevantSeries {
  ticker: string;
  title: string | null;
  category: string;
  frequency: string | null;
  scope: "asset" | "global";
  fast: boolean;
}

export function classifyRelevantSeries(series: KalshiSeriesRaw): RelevantSeries | null {
  const ticker = series.ticker?.trim();
  const title = series.title?.trim() ?? "";
  const category = series.category?.trim() ?? "";
  const frequency = series.frequency?.trim() ?? "";
  // Kalshi still returns legacy aliases alongside the canonical KX series.
  if (!ticker?.startsWith("KX")) return null;
  if (category === "Crypto") {
    return {
      ticker,
      title: title || null,
      category,
      frequency: frequency || null,
      scope: "asset",
      fast: ["fifteen_min", "hourly"].includes(frequency) && FAST_CRYPTO_KEYWORDS.test(title),
    };
  }
  if (SYSTEMIC_CATEGORIES.has(category)) {
    return { ticker, title: title || null, category, frequency: frequency || null, scope: "global", fast: false };
  }
  if (["Science and Technology", "Companies", "Health", "Climate and Weather", "Transportation"].includes(category)
      && SYSTEMIC_KEYWORDS.test(title)) {
    return { ticker, title: title || null, category, frequency: frequency || null, scope: "global", fast: false };
  }
  return null;
}

interface KalshiMarketRaw {
  ticker: string;
  event_ticker?: string;
  title?: string;
  subtitle?: string;
  yes_sub_title?: string;
  no_sub_title?: string;
  created_time?: string;
  updated_time?: string;
  open_time: string;
  close_time: string;
  expected_expiration_time?: string;
  expiration_time?: string;
  settlement_ts?: string;
  strike_type?: string;
  floor_strike?: number;
  cap_strike?: number;
  functional_strike?: string;
  volume_fp?: string;
}

interface CutoffResponse {
  market_settled_ts?: string;
  trades_created_ts?: string;
}

interface GzipWriter {
  gzip: Gzip;
  file: WriteStream;
  done: Promise<void>;
  rows: number;
  write(row: unknown): Promise<void>;
  close(): Promise<void>;
  abort(): Promise<void>;
}

function argument(args: string[], name: string): string | undefined {
  const index = args.indexOf(name);
  return index < 0 ? undefined : args[index + 1];
}

function parseInstant(value: string, endOfDay = false): number {
  const normalized = /^\d{4}-\d{2}-\d{2}$/.test(value)
    ? `${value}T${endOfDay ? "23:59:59.999" : "00:00:00.000"}Z`
    : value;
  const parsed = Date.parse(normalized);
  if (!Number.isFinite(parsed)) throw new Error(`Invalid time: ${value}`);
  return parsed;
}

function marketTime(value: string | undefined): number | null {
  if (!value) return null;
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : null;
}

async function requestJson<T>(url: URL, allowMissing = false): Promise<T | null> {
  const attempts = 8;
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    let response: Response;
    try {
      response = await fetch(url, { headers: { accept: "application/json", "user-agent": "trading-research/1.0" } });
    } catch (error) {
      if (attempt === attempts - 1) throw error;
      await new Promise((resolve) => setTimeout(resolve, Math.min(8_000, 500 * 2 ** attempt) + Math.random() * 250));
      continue;
    }
    if (allowMissing && (response.status === 400 || response.status === 404)) return null;
    if (response.status === 403) {
      throw new Error(`Kalshi rejected this host with HTTP 403. Run the importer from a jurisdiction/host where the official API is available: ${url.origin}`);
    }
    if (response.ok) return await response.json() as T;
    if (response.status !== 429 && response.status < 500) {
      throw new Error(`${response.status} ${response.statusText} from ${url}`);
    }
    const retryAfterMs = Number(response.headers.get("retry-after")) * 1_000;
    const delay = Math.max(Number.isFinite(retryAfterMs) ? retryAfterMs : 0, Math.min(8_000, 500 * 2 ** attempt));
    await new Promise((resolve) => setTimeout(resolve, delay + Math.random() * 250));
  }
  throw new Error(`Repeated request failure from ${url}`);
}

async function paged<T>(
  route: string,
  collection: string,
  params: Record<string, string>,
  maxPages: number,
  allowTruncated = false,
): Promise<T[]> {
  const rows: T[] = [];
  let cursor = "";
  for (let page = 0; page < maxPages; page += 1) {
    const url = new URL(`${API}/${route}`);
    for (const [name, value] of Object.entries(params)) url.searchParams.set(name, value);
    if (!url.searchParams.has("limit")) url.searchParams.set("limit", "1000");
    if (cursor) url.searchParams.set("cursor", cursor);
    const payload = await requestJson<Record<string, unknown>>(url, true);
    if (!payload) break;
    const batch = payload[collection];
    if (!Array.isArray(batch)) throw new Error(`Missing ${collection} array from ${url}`);
    rows.push(...batch as T[]);
    cursor = typeof payload.cursor === "string" ? payload.cursor : "";
    if (!cursor || batch.length === 0) break;
  }
  if (cursor && !allowTruncated) throw new Error(`Reached --max-pages before exhausting ${route}`);
  return rows;
}

async function fetchSeriesCatalog(): Promise<KalshiSeriesRaw[]> {
  const url = new URL(`${API}/series`);
  url.searchParams.set("limit", "1000");
  const payload = await requestJson<{ series: KalshiSeriesRaw[] }>(url);
  if (!payload || !Array.isArray(payload.series)) throw new Error("Kalshi series catalog is unavailable");
  return payload.series;
}

async function concurrentForEach<T>(
  rows: T[],
  concurrency: number,
  task: (row: T, index: number) => Promise<void>,
): Promise<void> {
  let cursor = 0;
  async function worker() {
    while (true) {
      const index = cursor++;
      if (index >= rows.length) return;
      await task(rows[index]!, index);
    }
  }
  await Promise.all(Array.from({ length: Math.max(1, Math.min(concurrency, rows.length)) }, worker));
}

async function discoverMarkets(
  series: string,
  startMs: number,
  endMs: number,
  maxPages: number,
  marketCutoffMs: number | null,
): Promise<KalshiMarketRaw[]> {
  const useHistorical = marketCutoffMs === null || startMs < marketCutoffMs;
  const useLive = marketCutoffMs === null || endMs >= marketCutoffMs;
  const [live, historical] = await Promise.all([
    useLive
      ? paged<KalshiMarketRaw>("markets", "markets", {
        series_ticker: series,
        min_close_ts: String(Math.floor(startMs / 1000)),
      }, maxPages)
      : Promise.resolve([]),
    useHistorical
      ? paged<KalshiMarketRaw>("historical/markets", "markets", { series_ticker: series }, maxPages)
      : Promise.resolve([]),
  ]);
  const byTicker = new Map<string, KalshiMarketRaw>();
  for (const market of [...historical, ...live]) {
    const open = marketTime(market.open_time);
    const close = marketTime(market.close_time);
    if (open === null || close === null || open >= endMs || close <= startMs) continue;
    byTicker.set(market.ticker, market);
  }
  return [...byTicker.values()].sort((left, right) => left.open_time.localeCompare(right.open_time));
}

async function fetchTrades(
  ticker: string,
  startMs: number,
  endMs: number,
  maxPages: number,
  tradesCutoffMs: number | null,
  includePrior: boolean,
) {
  const params = {
    ticker,
    min_ts: String(Math.max(0, Math.floor((startMs - 1_000) / 1000))),
    max_ts: String(Math.ceil(endMs / 1000)),
  };
  const beforeParams = { ticker, max_ts: String(Math.floor(startMs / 1000)), limit: "1" };
  const useHistorical = tradesCutoffMs === null || startMs < tradesCutoffMs;
  const useLive = tradesCutoffMs === null || endMs >= tradesCutoffMs;
  const [live, historical, priorLive, priorHistorical] = await Promise.all([
    useLive ? paged<KalshiTradeRaw>("markets/trades", "trades", params, maxPages) : Promise.resolve([]),
    useHistorical ? paged<KalshiTradeRaw>("historical/trades", "trades", params, maxPages) : Promise.resolve([]),
    useLive && includePrior ? paged<KalshiTradeRaw>("markets/trades", "trades", beforeParams, 1, true) : Promise.resolve([]),
    useHistorical && includePrior ? paged<KalshiTradeRaw>("historical/trades", "trades", beforeParams, 1, true) : Promise.resolve([]),
  ]);
  const normalized = normalizeKalshiTrades([...priorHistorical, ...priorLive, ...historical, ...live]);
  const firstIntervalStart = Math.ceil(startMs / 1_000) * 1_000 - 1_000;
  const latestEarlier = normalized.filter((trade) => trade.timeMs < firstIntervalStart).at(-1);
  return [
    ...(latestEarlier ? [latestEarlier] : []),
    ...normalized.filter((trade) => trade.timeMs >= firstIntervalStart && trade.timeMs < endMs),
  ];
}

async function fetchMinuteCandles(
  series: string,
  ticker: string,
  startMs: number,
  endMs: number,
  marketCutoffMs: number | null,
) {
  const rows: KalshiCandlestickRaw[] = [];
  // Kalshi caps one response at 5,000 requested candles, including empty
  // periods. Keep each inclusive request slightly below that boundary.
  const chunkMs = 4_999 * 60_000;
  for (let chunkStart = startMs; chunkStart < endMs; chunkStart += chunkMs) {
    const chunkEnd = Math.min(endMs, chunkStart + chunkMs);
    const params = new URLSearchParams({
      start_ts: String(Math.floor(chunkStart / 1000)),
      end_ts: String(Math.ceil(chunkEnd / 1000)),
      period_interval: "1",
    });
    const liveUrl = new URL(`${API}/series/${encodeURIComponent(series)}/markets/${encodeURIComponent(ticker)}/candlesticks?${params}`);
    const historicalUrl = new URL(`${API}/historical/markets/${encodeURIComponent(ticker)}/candlesticks?${params}`);
    const useHistorical = marketCutoffMs === null || chunkStart < marketCutoffMs;
    const useLive = marketCutoffMs === null || chunkEnd >= marketCutoffMs;
    const [live, historical] = await Promise.all([
      useLive ? requestJson<{ candlesticks: KalshiCandlestickRaw[] }>(liveUrl, true) : Promise.resolve(null),
      useHistorical ? requestJson<{ candlesticks: KalshiCandlestickRaw[] }>(historicalUrl, true) : Promise.resolve(null),
    ]);
    rows.push(...(historical?.candlesticks ?? []), ...(live?.candlesticks ?? []));
  }
  return normalizeKalshiMinuteCandles(rows);
}

async function fetchBatchMinuteCandles(
  jobs: Array<{ market: KalshiMarketRaw; id: number }>,
  startMs: number,
  endMs: number,
): Promise<Map<number, ReturnType<typeof normalizeKalshiMinuteCandles>>> {
  const output = new Map<number, ReturnType<typeof normalizeKalshiMinuteCandles>>();
  if (jobs.length === 0) return output;
  const marketIdByTicker = new Map(jobs.map((job) => [job.market.ticker, job.id]));
  const rowsByMarket = new Map<number, KalshiCandlestickRaw[]>();
  const byEvent = new Map<string, typeof jobs>();
  for (const job of jobs) {
    const event = job.market.event_ticker ?? job.market.ticker;
    const group = byEvent.get(event) ?? [];
    group.push(job);
    byEvent.set(event, group);
  }
  const requests: Array<{ group: typeof jobs; chunkStart: number; chunkEnd: number }> = [];
  for (const group of byEvent.values()) {
    // The batch endpoint accepts 100 tickers and returns at most 10,000 candles.
    // Grouping common event strike ladders minimizes the requested time rectangle.
    const groupStart = Math.max(startMs, Math.min(...group.map((job) => marketTime(job.market.open_time)!)));
    const groupEnd = Math.min(endMs, Math.max(...group.map((job) => marketTime(job.market.close_time)!)));
    const chunkMinutes = Math.max(1, Math.floor(8_000 / group.length));
    for (let chunkStart = groupStart; chunkStart < groupEnd; chunkStart += chunkMinutes * 60_000) {
      const chunkEnd = Math.min(groupEnd, chunkStart + chunkMinutes * 60_000);
      requests.push({ group, chunkStart, chunkEnd });
    }
  }
  await concurrentForEach(requests, 8, async ({ group, chunkStart, chunkEnd }) => {
    const url = new URL(`${API}/markets/candlesticks`);
    url.searchParams.set("market_tickers", group.map((job) => job.market.ticker).join(","));
    url.searchParams.set("start_ts", String(Math.floor(chunkStart / 1000)));
    url.searchParams.set("end_ts", String(Math.ceil(chunkEnd / 1000)));
    url.searchParams.set("period_interval", "1");
    const payload = await requestJson<{ markets: Array<{ market_ticker: string; candlesticks: KalshiCandlestickRaw[] }> }>(url);
    for (const market of payload?.markets ?? []) {
      const id = marketIdByTicker.get(market.market_ticker);
      if (id === undefined) continue;
      const rows = rowsByMarket.get(id) ?? [];
      rows.push(...(market.candlesticks ?? []));
      rowsByMarket.set(id, rows);
    }
  });
  for (const job of jobs) output.set(job.id, normalizeKalshiMinuteCandles(rowsByMarket.get(job.id) ?? []));
  return output;
}

async function fetchBatchPriorTrades(
  jobs: Array<{ market: KalshiMarketRaw; id: number }>,
  originMs: number,
): Promise<Map<number, PredictionTrade>> {
  const output = new Map<number, PredictionTrade>();
  if (jobs.length === 0) return output;
  const marketByTicker = new Map(jobs.map((job) => [job.market.ticker, job]));
  const url = new URL(`${API}/markets/candlesticks`);
  url.searchParams.set("market_tickers", jobs.map((job) => job.market.ticker).join(","));
  url.searchParams.set("start_ts", String(Math.floor(originMs / 1000)));
  url.searchParams.set("end_ts", String(Math.ceil((originMs + 60_000) / 1000)));
  url.searchParams.set("period_interval", "1");
  url.searchParams.set("include_latest_before_start", "true");
  const payload = await requestJson<{ markets: Array<{ market_ticker: string; candlesticks: KalshiCandlestickRaw[] }> }>(url);
  for (const market of payload?.markets ?? []) {
    const job = marketByTicker.get(market.market_ticker);
    if (!job) continue;
    const states = normalizeKalshiMinuteCandles(market.candlesticks ?? []);
    const probability = states.find((state) => state.previousTradeProbability !== null)?.previousTradeProbability ?? null;
    if (probability === null) continue;
    output.set(job.id, {
      id: `batch-prior-${job.market.ticker}`,
      ticker: job.market.ticker,
      timeMs: originMs - 1,
      yesPrice: probability,
      count: 0,
      takerSide: null,
      isBlockTrade: false,
    });
  }
  return output;
}

function sanitizedMarket(market: KalshiMarketRaw, series: RelevantSeries, id: number) {
  return {
    id,
    provider: "kalshi",
    series: series.ticker,
    seriesTitle: series.title,
    category: series.category,
    frequency: series.frequency,
    scope: series.scope,
    fast: series.fast,
    ticker: market.ticker,
    eventTicker: market.event_ticker ?? null,
    title: market.title ?? null,
    subtitle: market.subtitle ?? null,
    yesSubtitle: market.yes_sub_title ?? null,
    noSubtitle: market.no_sub_title ?? null,
    createdTime: market.created_time ?? null,
    metadataUpdatedTime: market.updated_time ?? null,
    mutableMetadataAvailableAt: market.updated_time ?? market.created_time ?? null,
    openTime: market.open_time,
    closeTime: market.close_time,
    expectedExpirationTime: market.expected_expiration_time ?? null,
    expirationTime: market.expiration_time ?? null,
    strikeType: market.strike_type ?? null,
    floorStrike: market.floor_strike ?? null,
    capStrike: market.cap_strike ?? null,
    functionalStrike: market.functional_strike ?? null,
  };
}

function backfillMarket(market: KalshiMarketRaw): KalshiMarketRaw {
  return {
    ticker: market.ticker,
    event_ticker: market.event_ticker,
    title: market.title,
    subtitle: market.subtitle,
    yes_sub_title: market.yes_sub_title,
    no_sub_title: market.no_sub_title,
    created_time: market.created_time,
    updated_time: market.updated_time,
    open_time: market.open_time,
    close_time: market.close_time,
    expected_expiration_time: market.expected_expiration_time,
    expiration_time: market.expiration_time,
    strike_type: market.strike_type,
    floor_strike: market.floor_strike,
    cap_strike: market.cap_strike,
    functional_strike: market.functional_strike,
    volume_fp: market.volume_fp,
  };
}

async function writeJsonAtomic(target: string, value: unknown): Promise<void> {
  const temporary = `${target}.writing`;
  await fs.writeFile(temporary, `${JSON.stringify(value)}\n`);
  await fs.rename(temporary, target);
}

async function concatenateFiles(sources: string[], target: string): Promise<void> {
  const output = createWriteStream(target);
  try {
    for (const source of sources) {
      for await (const chunk of createReadStream(source)) {
        if (!output.write(chunk)) await new Promise<void>((resolve) => output.once("drain", resolve));
      }
    }
    output.end();
    await finished(output);
  } catch (error) {
    output.destroy();
    throw error;
  }
}

function gzipWriter(target: string): GzipWriter {
  const gzip = createGzip({ level: 9 });
  const file = createWriteStream(target);
  gzip.pipe(file);
  const writer: GzipWriter = {
    gzip,
    file,
    done: finished(file),
    rows: 0,
    async write(row: unknown) {
      writer.rows += 1;
      if (!gzip.write(`${JSON.stringify(row)}\n`)) await new Promise<void>((resolve) => gzip.once("drain", resolve));
    },
    async close() {
      gzip.end();
      await writer.done;
    },
    async abort() {
      gzip.destroy();
      file.destroy();
      try { await writer.done; } catch { /* expected for an aborted partial file */ }
    },
  };
  return writer;
}

export async function run(args = process.argv.slice(2)): Promise<void> {
  if (args.includes("--help")) {
    console.log(`Usage: npm run fetch:prediction-markets -- [options]

  --start ISO_OR_YYYY-MM-DD        required
  --end ISO_OR_YYYY-MM-DD          required
  --profile asset|global|core|relevant
                                  default: core; relevant discovers all supported
                                  crypto, macro, financial, political, commodity,
                                  geopolitical, and systemic-event series
  --series TICKER,TICKER           override the profile
  --resolution 1s|1m|both          default: both
  --all-series-at-1s               include slow/global series in 1s output
  --sample-1s-at-minute-origins    retain strict preceding-second state only at
                                  minute model origins
  --reuse-discovery                reuse the exact saved series/market discovery
                                  sidecar for this output interval
  --discovery-only                 save discovery sidecar without fetching history
  --market-concurrency N           concurrent market histories, default: 12
  --max-pages N                    safety limit per endpoint, default: 500
  --output-dir PATH                default: data/market/mutable/prediction-markets/kalshi`);
    return;
  }
  const startValue = argument(args, "--start");
  const endValue = argument(args, "--end");
  if (!startValue || !endValue) throw new Error("Both --start and --end are required");
  const startMs = parseInstant(startValue);
  const endMs = parseInstant(endValue, true);
  if (endMs <= startMs) throw new Error("--end must be after --start");
  const profile = argument(args, "--profile") ?? "core";
  if (!new Set(["asset", "global", "core", "relevant"]).has(profile)) throw new Error(`Invalid --profile ${profile}`);
  const explicitSeries = argument(args, "--series")?.split(",").map((item) => item.trim()).filter(Boolean);
  const resolution = argument(args, "--resolution") ?? "both";
  if (!new Set(["1s", "1m", "both"]).has(resolution)) throw new Error(`Invalid --resolution ${resolution}`);
  const include1s = resolution === "1s" || resolution === "both";
  const include1m = resolution === "1m" || resolution === "both";
  const allSeriesAt1s = args.includes("--all-series-at-1s");
  const sampleOneSecondAtMinuteOrigins = args.includes("--sample-1s-at-minute-origins");
  const marketConcurrency = Number(argument(args, "--market-concurrency") ?? 12);
  const maxPages = Number(argument(args, "--max-pages") ?? 500);
  if (!Number.isInteger(marketConcurrency) || marketConcurrency < 1 || marketConcurrency > 64) {
    throw new Error("--market-concurrency must be an integer from 1 through 64");
  }
  const outputDir = path.resolve(repoRoot, argument(args, "--output-dir") ?? "data/market/mutable/prediction-markets/kalshi");
  await fs.mkdir(outputDir, { recursive: true });
  const suffix = `${new Date(startMs).toISOString().replaceAll(":", "-")}_${new Date(endMs).toISOString().replaceAll(":", "-")}`;
  const oneSecondPartial = path.join(outputDir, `${suffix}.1s.ndjson.gz.partial`);
  const oneMinutePartial = path.join(outputDir, `${suffix}.1m.ndjson.gz.partial`);
  const oneSecond = oneSecondPartial.replace(/\.partial$/, "");
  const oneMinute = oneMinutePartial.replace(/\.partial$/, "");
  const discoveryPath = path.join(outputDir, `${suffix}.discovery.json`);
  const discoveryProgressPath = path.join(outputDir, `${suffix}.discovery.partial.json`);
  const reuseDiscovery = args.includes("--reuse-discovery");
  const discoveryOnly = args.includes("--discovery-only");
  const metadata: ReturnType<typeof sanitizedMarket>[] = [];
  const failures: Array<{ series: string; ticker?: string; error: string }> = [];
  try {
    const cutoff = await requestJson<CutoffResponse>(new URL(`${API}/historical/cutoff`), true);
    const marketCutoffMs = marketTime(cutoff?.market_settled_ts);
    const tradesCutoffMs = marketTime(cutoff?.trades_created_ts);
    let catalogSeries: number;
    let relevant: RelevantSeries[];
    let jobs: Array<{ series: RelevantSeries; market: KalshiMarketRaw; id: number }>;
    if (reuseDiscovery) {
      const saved = JSON.parse(await fs.readFile(discoveryPath, "utf8")) as {
        request: { start: string; end: string; profile: string };
        catalogSeries: number;
        series: RelevantSeries[];
        jobs: Array<{ series: RelevantSeries; market: KalshiMarketRaw; id: number }>;
      };
      if (saved.request.start !== new Date(startMs).toISOString()
          || saved.request.end !== new Date(endMs).toISOString()
          || saved.request.profile !== profile) {
        throw new Error("Saved discovery does not match this request interval/profile");
      }
      catalogSeries = saved.catalogSeries;
      relevant = saved.series;
      jobs = saved.jobs;
      console.log(`Reused ${jobs.length.toLocaleString()} discovered markets from ${path.basename(discoveryPath)}.`);
    } else {
      const catalog = await fetchSeriesCatalog();
      catalogSeries = catalog.length;
      const catalogByTicker = new Map(catalog.map((row) => [row.ticker, row]));
      const selectedTickers = explicitSeries ?? (profile === "asset"
        ? [...ASSET_SERIES]
        : profile === "global"
          ? [...GLOBAL_SERIES]
          : profile === "core"
            ? [...ASSET_SERIES, ...GLOBAL_SERIES]
            : []);
      relevant = profile === "relevant" && !explicitSeries
        ? catalog.map(classifyRelevantSeries).filter((row): row is RelevantSeries => row !== null)
        : selectedTickers.map((ticker) => {
          const raw = catalogByTicker.get(ticker) ?? { ticker };
          return classifyRelevantSeries(raw) ?? {
            ticker,
            title: raw.title ?? null,
            category: raw.category ?? "Unclassified",
            frequency: raw.frequency ?? null,
            scope: (ASSET_SERIES as readonly string[]).includes(ticker) ? "asset" as const : "global" as const,
            fast: DEFAULT_FAST_SERIES.has(ticker),
          };
        });
      const request = {
        start: new Date(startMs).toISOString(),
        end: new Date(endMs).toISOString(),
        profile,
        explicitSeries: explicitSeries ?? null,
      };
      const marketsBySeries = new Map<string, KalshiMarketRaw[]>();
      const completedSeries = new Set<string>();
      try {
        const saved = JSON.parse(await fs.readFile(discoveryProgressPath, "utf8")) as {
          request: typeof request;
          series: RelevantSeries[];
          completedSeries: string[];
          marketsBySeries: Array<{ ticker: string; markets: KalshiMarketRaw[] }>;
        };
        const sameRequest = JSON.stringify(saved.request) === JSON.stringify(request);
        const sameSeries = saved.series.map((row) => row.ticker).sort().join("\n")
          === relevant.map((row) => row.ticker).sort().join("\n");
        if (sameRequest && sameSeries) {
          for (const row of saved.marketsBySeries) marketsBySeries.set(row.ticker, row.markets);
          for (const ticker of saved.completedSeries) completedSeries.add(ticker);
          console.log(`Resuming discovery after ${completedSeries.size.toLocaleString()} completed series.`);
        }
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
      }
      let completedSinceCheckpoint = 0;
      let checkpointChain = Promise.resolve();
      const saveDiscoveryProgress = () => {
        const snapshot = {
          version: 1,
          request,
          catalogSeries,
          series: relevant,
          completedSeries: [...completedSeries].sort(),
          marketsBySeries: [...marketsBySeries.entries()].sort(([left], [right]) => left.localeCompare(right))
            .map(([ticker, markets]) => ({ ticker, markets: markets.map(backfillMarket) })),
        };
        checkpointChain = checkpointChain.then(() => writeJsonAtomic(discoveryProgressPath, snapshot));
        return checkpointChain;
      };
      const remaining = relevant.filter((series) => !completedSeries.has(series.ticker));
      await concurrentForEach(remaining, marketConcurrency, async (series) => {
        let lastError: unknown;
        for (let discoveryAttempt = 0; discoveryAttempt < 5; discoveryAttempt += 1) {
          try {
            const discovered = await discoverMarkets(
              series.ticker, startMs, endMs, maxPages, marketCutoffMs,
            );
            marketsBySeries.set(
              series.ticker,
              profile === "relevant"
                ? discovered.filter((market) => Number(market.volume_fp ?? 0) > 0)
                : discovered,
            );
            completedSeries.add(series.ticker);
            completedSinceCheckpoint += 1;
            if (completedSinceCheckpoint >= 100) {
              completedSinceCheckpoint = 0;
              await saveDiscoveryProgress();
              console.log(`Discovery checkpoint: ${completedSeries.size.toLocaleString()}/${relevant.length.toLocaleString()} series.`);
            }
            return;
          } catch (error) {
            if (error instanceof Error && error.message.includes("HTTP 403")) throw error;
            lastError = error;
            await new Promise((resolve) => setTimeout(resolve, 2_000 * (discoveryAttempt + 1)));
          }
        }
        failures.push({ series: series.ticker, error: lastError instanceof Error ? lastError.message : String(lastError) });
      });
      await saveDiscoveryProgress();
      if (failures.length > 0) throw new Error(`Prediction-market discovery was incomplete: ${failures[0]!.error}`);
      jobs = relevant.flatMap((series) => {
        const markets = marketsBySeries.get(series.ticker) ?? [];
        if (markets.length) console.log(`${series.ticker}: ${markets.length.toLocaleString()} overlapping markets`);
        return markets.map((market) => ({ series, market }));
      }).sort((left, right) => (
        left.series.ticker.localeCompare(right.series.ticker)
        || left.market.open_time.localeCompare(right.market.open_time)
        || left.market.ticker.localeCompare(right.market.ticker)
      )).map((job, id) => ({ ...job, id }));
      await writeJsonAtomic(discoveryPath, {
        version: 1,
        request,
        catalogSeries,
        series: relevant,
        jobs: jobs.map((job) => ({ ...job, market: backfillMarket(job.market) })),
      });
      await fs.rm(discoveryProgressPath, { force: true });
    }
    if (discoveryOnly) {
      await Promise.all([fs.rm(oneSecondPartial, { force: true }), fs.rm(oneMinutePartial, { force: true })]);
      console.log(`Saved ${jobs.length.toLocaleString()} markets to ${path.basename(discoveryPath)}.`);
      return;
    }
    for (const job of jobs) metadata.push(sanitizedMarket(job.market, job.series, job.id));
    console.log(`Selected ${relevant.length.toLocaleString()} relevant series and ${jobs.length.toLocaleString()} traded overlapping markets.`);
    const historyPartsDir = path.join(outputDir, `${suffix}.history-parts-v3`);
    await fs.mkdir(historyPartsDir, { recursive: true });
    const historyRequest = {
      version: 1,
      start: new Date(startMs).toISOString(),
      end: new Date(endMs).toISOString(),
      profile,
      resolution,
      allSeriesAt1s,
      sampleOneSecondAtMinuteOrigins,
      marketCount: jobs.length,
    };
    const historyRequestPath = path.join(historyPartsDir, "request.json");
    try {
      const savedRequest = JSON.parse(await fs.readFile(historyRequestPath, "utf8"));
      if (JSON.stringify(savedRequest) !== JSON.stringify(historyRequest)) {
        throw new Error(`History checkpoint at ${historyPartsDir} belongs to a different request`);
      }
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
      await writeJsonAtomic(historyRequestPath, historyRequest);
    }
    const fetchMarketRows = async (
      { series: seriesRow, market, id }: typeof jobs[number],
      priorTrade: PredictionTrade | undefined,
    ) => {
        const marketClose = marketTime(market.close_time)!;
        const activeStart = Math.max(startMs, marketTime(market.open_time)!);
        const activeEnd = Math.min(endMs, marketClose);
        let lastError: unknown;
        for (let marketAttempt = 0; marketAttempt < 3; marketAttempt += 1) {
          try {
            const secondRows: unknown[] = [];
            const minuteRows: unknown[] = [];
            let trades: Awaited<ReturnType<typeof fetchTrades>> | null = null;
            const fastMarket = allSeriesAt1s || seriesRow.fast || DEFAULT_FAST_SERIES.has(seriesRow.ticker);
            if (include1s && fastMarket && profile !== "relevant") {
              const marketOpen = marketTime(market.open_time)!;
              trades = await fetchTrades(
                market.ticker, activeStart, activeEnd, maxPages, tradesCutoffMs,
                profile !== "relevant" && activeStart > marketOpen,
              );
              if (priorTrade) trades.unshift(priorTrade);
              const states = sampleOneSecondAtMinuteOrigins
                ? buildCausalTradeStatesForOrigins(
                  trades,
                  Array.from(
                    { length: Math.max(0, Math.floor((activeEnd - Math.ceil(activeStart / 60_000) * 60_000) / 60_000) + 1) },
                    (_, index) => Math.ceil(activeStart / 60_000) * 60_000 + index * 60_000,
                  ),
                  1_000,
                )
                : buildCausalTradeStates(trades, activeStart, activeEnd, 1_000);
              for (const state of states) {
                secondRows.push({ marketId: id, ...state, timeToCloseMs: marketClose - state.availableAt });
              }
            }
            if (include1m && profile === "relevant" && seriesRow.scope === "asset" && !fastMarket) {
              const marketOpen = marketTime(market.open_time)!;
              trades ??= await fetchTrades(
                market.ticker, activeStart, activeEnd, maxPages, tradesCutoffMs,
                profile !== "relevant" && activeStart > marketOpen,
              );
              if (priorTrade && !trades.some((trade) => trade.id === priorTrade.id)) trades.unshift(priorTrade);
              for (const state of buildCausalTradeStates(trades, activeStart, activeEnd, 60_000)) {
                minuteRows.push({ marketId: id, ...state, timeToCloseMs: marketClose - state.availableAt });
              }
            }
            if (include1m && !(profile === "relevant" && seriesRow.scope === "asset")) {
              if (profile === "relevant" && seriesRow.scope === "global") {
                const marketOpen = marketTime(market.open_time)!;
                trades ??= await fetchTrades(
                  market.ticker, activeStart, activeEnd, maxPages, tradesCutoffMs,
                  profile !== "relevant" && activeStart > marketOpen,
                );
                if (priorTrade && !trades.some((trade) => trade.id === priorTrade.id)) trades.unshift(priorTrade);
                const origins = trades
                  .filter((trade) => trade.timeMs >= activeStart && trade.timeMs < activeEnd)
                  .map((trade) => Math.floor(trade.timeMs / 60_000) * 60_000 + 60_000)
                  .filter((origin) => origin <= activeEnd);
                const firstOrigin = Math.ceil(activeStart / 60_000) * 60_000;
                if (trades.some((trade) => trade.timeMs < firstOrigin)) origins.push(firstOrigin);
                for (const state of buildCausalTradeStatesForOrigins(trades, origins, 60_000)) {
                  minuteRows.push({ marketId: id, ...state, timeToCloseMs: marketClose - state.availableAt });
                }
              } else {
                const candles = await fetchMinuteCandles(seriesRow.ticker, market.ticker, activeStart, activeEnd, marketCutoffMs);
                for (const state of candles) {
                  if (state.availableAt < activeStart || state.availableAt > activeEnd) continue;
                  minuteRows.push({ marketId: id, ...state, timeToCloseMs: marketClose - state.availableAt });
                }
              }
            }
            return { secondRows, minuteRows };
          } catch (error) {
            if (error instanceof Error && error.message.includes("HTTP 403")) throw error;
            lastError = error;
            await new Promise((resolve) => setTimeout(resolve, 2_000 * (marketAttempt + 1)));
          }
        }
        throw new Error(`${seriesRow.ticker}/${market.ticker}: ${lastError instanceof Error ? lastError.message : String(lastError)}`);
    };
    const batchSize = 100;
    const batchCount = Math.ceil(jobs.length / batchSize);
    let oneSecondRows = 0;
    let oneMinuteRows = 0;
    const secondParts: string[] = [];
    const minuteParts: string[] = [];
    for (let batchIndex = 0; batchIndex < batchCount; batchIndex += 1) {
      const batchName = String(batchIndex).padStart(6, "0");
      const batchManifestPath = path.join(historyPartsDir, `${batchName}.json`);
      const secondPart = path.join(historyPartsDir, `${batchName}.1s.ndjson.gz`);
      const minutePart = path.join(historyPartsDir, `${batchName}.1m.ndjson.gz`);
      try {
        const saved = JSON.parse(await fs.readFile(batchManifestPath, "utf8")) as {
          oneSecondRows: number;
          oneMinuteRows: number;
        };
        oneSecondRows += saved.oneSecondRows;
        oneMinuteRows += saved.oneMinuteRows;
        if (include1s) secondParts.push(secondPart);
        if (include1m) minuteParts.push(minutePart);
        continue;
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
      }
      const batch = jobs.slice(batchIndex * batchSize, (batchIndex + 1) * batchSize);
      const results: Array<Awaited<ReturnType<typeof fetchMarketRows>> | undefined> = new Array(batch.length);
      const priorTrades = profile === "relevant"
        ? await fetchBatchPriorTrades(
          batch.filter((job) => marketTime(job.market.open_time)! < startMs),
          startMs,
        )
        : new Map<number, PredictionTrade>();
      await concurrentForEach(batch, marketConcurrency, async (job, index) => {
        results[index] = await fetchMarketRows(job, priorTrades.get(job.id));
      });
      if (include1m && profile === "relevant") {
        const fastAssetJobs = batch.filter((job) => (
          job.series.scope === "asset"
          && (allSeriesAt1s || job.series.fast || DEFAULT_FAST_SERIES.has(job.series.ticker))
        ));
        if (fastAssetJobs.length > 0) {
          const batchStart = Math.min(...fastAssetJobs.map((job) => Math.max(startMs, marketTime(job.market.open_time)!)));
          const batchEnd = Math.max(...fastAssetJobs.map((job) => Math.min(endMs, marketTime(job.market.close_time)!)));
          const candlesByMarket = await fetchBatchMinuteCandles(fastAssetJobs, batchStart, batchEnd);
          for (let index = 0; index < batch.length; index += 1) {
            const job = batch[index]!;
            if (!fastAssetJobs.includes(job)) continue;
            const activeStart = Math.max(startMs, marketTime(job.market.open_time)!);
            const activeEnd = Math.min(endMs, marketTime(job.market.close_time)!);
            const marketClose = marketTime(job.market.close_time)!;
            for (const state of candlesByMarket.get(job.id) ?? []) {
              if (state.availableAt < activeStart || state.availableAt > activeEnd) continue;
              results[index]!.minuteRows.push({ marketId: job.id, ...state, timeToCloseMs: marketClose - state.availableAt });
            }
          }
        }
      }
      const secondPartWriter = include1s ? gzipWriter(`${secondPart}.partial`) : null;
      const minutePartWriter = include1m ? gzipWriter(`${minutePart}.partial`) : null;
      try {
        for (const result of results) {
          for (const row of result?.secondRows ?? []) await secondPartWriter?.write(row);
          for (const row of result?.minuteRows ?? []) await minutePartWriter?.write(row);
        }
        await Promise.all([secondPartWriter?.close(), minutePartWriter?.close()]);
        if (secondPartWriter) await fs.rename(`${secondPart}.partial`, secondPart);
        if (minutePartWriter) await fs.rename(`${minutePart}.partial`, minutePart);
      } catch (error) {
        await Promise.all([secondPartWriter?.abort(), minutePartWriter?.abort()]);
        throw error;
      }
      const batchSummary = {
        firstMarketId: batch[0]?.id ?? null,
        lastMarketId: batch.at(-1)?.id ?? null,
        markets: batch.length,
        oneSecondRows: secondPartWriter?.rows ?? 0,
        oneMinuteRows: minutePartWriter?.rows ?? 0,
      };
      await writeJsonAtomic(batchManifestPath, batchSummary);
      oneSecondRows += batchSummary.oneSecondRows;
      oneMinuteRows += batchSummary.oneMinuteRows;
      if (include1s) secondParts.push(secondPart);
      if (include1m) minuteParts.push(minutePart);
      console.log(`History checkpoint: ${Math.min((batchIndex + 1) * batchSize, jobs.length).toLocaleString()}/${jobs.length.toLocaleString()} markets.`);
    }
    if (include1s) {
      await concatenateFiles(secondParts, oneSecondPartial);
      await fs.rename(oneSecondPartial, oneSecond);
    }
    if (include1m) {
      await concatenateFiles(minuteParts, oneMinutePartial);
      await fs.rename(oneMinutePartial, oneMinute);
    }
    const retrievedAt = new Date().toISOString();
    await fs.writeFile(path.join(outputDir, `${suffix}.markets.json`), `${JSON.stringify({ version: 1, provider: "kalshi", retrievedAt, markets: metadata }, null, 2)}\n`);
    await fs.writeFile(path.join(outputDir, `${suffix}.manifest.json`), `${JSON.stringify({
      version: 1,
      provider: "kalshi",
      retrievedAt,
      request: {
        start: new Date(startMs).toISOString(), end: new Date(endMs).toISOString(), profile,
        explicitSeries: explicitSeries ?? null, selectedSeries: relevant.map((row) => row.ticker),
        resolution, allSeriesAt1s, sampleOneSecondAtMinuteOrigins, marketConcurrency, maxPages,
      },
      discovery: {
        catalogSeries,
        selectedSeries: relevant.length,
        selectedSeriesByScope: Object.fromEntries(["asset", "global"].map((scope) => [
          scope, relevant.filter((row) => row.scope === scope).length,
        ])),
        selectedSeriesByCategory: Object.fromEntries([...new Set(relevant.map((row) => row.category))].sort().map((category) => [
          category, relevant.filter((row) => row.category === category).length,
        ])),
        tradedOverlappingMarkets: jobs.length,
      },
      historicalCutoff: cutoff,
      semantics: {
        oneSecond: profile === "relevant"
          ? "No dedicated preceding-second archive; completed one-minute prediction-market states are evaluated causally for both target horizons."
          : "State at candle origin; only trades with created_time strictly earlier than availableAt. Optional minute-origin sampling preserves the preceding-second aggregation.",
        oneMinute: profile === "relevant"
          ? "Fast asset rows use official completed one-minute candles; other rows use strict causal completed-minute trade state. No settlement fields or post-origin trades are included."
          : "Official completed candle; availableAt equals end_period_ts.",
        excluded: ["settlement result", "settlement value", "post-settlement current price fields"],
        metadata: "Question/strike fields are mutable and must be masked before mutableMetadataAvailableAt; historical field-level revisions are not available.",
      },
      counts: { markets: metadata.length, oneSecondRows, oneMinuteRows, failures: failures.length },
      failures,
    }, null, 2)}\n`);
    await fs.rm(historyPartsDir, { recursive: true, force: true });
    console.log(`Wrote ${oneSecondRows} causal 1s rows and ${oneMinuteRows} causal 1m rows for ${metadata.length} markets.`);
  } catch (error) {
    await Promise.all([fs.rm(oneSecondPartial, { force: true }), fs.rm(oneMinutePartial, { force: true })]);
    throw error;
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  run().catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
