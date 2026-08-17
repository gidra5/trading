import { createHash } from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  deriveDeribitOptionSummary,
  mergeMempoolMiningSeries,
  normalizeCommunityCryptoDaily,
  normalizeCoinMetricsRows,
  normalizeDvolRows,
  normalizeFredVixCsv,
  type DeribitBookSummary,
  type CommunityCryptoMetricPayload,
} from "./lib/external-public-data.ts";

const DAY_MS = 86_400_000;
const DEFAULT_START = "2021-03-24";
const COIN_METRICS = [
  "AdrActCnt", "BlkCnt", "CapMVRVCur", "FeeTotNtv", "FlowInExNtv", "FlowInExUSD",
  "FlowOutExNtv", "FlowOutExUSD", "HashRate", "IssTotNtv", "IssTotUSD", "SplyExNtv",
  "SplyExUSD", "TxCnt", "TxTfrCnt", "volume_reported_spot_usd_1d",
] as const;
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

interface Artifact<T> {
  version: number;
  source: string;
  retrievedAt: string;
  request: Record<string, unknown>;
  sha256: string;
  rows: T;
}

export async function run(args = process.argv.slice(2)): Promise<void> {
  if (args.includes("--help")) {
    console.log(`Usage: npm run fetch:external-public -- [options]

  --source all|deribit-dvol|cboe-vix|deribit-surface|coinmetrics|mempool|community-crypto
  --start YYYY-MM-DD
  --end YYYY-MM-DD
  --output-dir data/market/mutable/external
  --quiet`);
    return;
  }
  const value = (name: string) => {
    const index = args.indexOf(name);
    return index < 0 ? undefined : args[index + 1];
  };
  const source = value("--source") ?? "all";
  const start = value("--start") ?? DEFAULT_START;
  const end = value("--end") ?? new Date().toISOString().slice(0, 10);
  const outputDir = path.resolve(repoRoot, value("--output-dir") ?? "data/market/mutable/external");
  const quiet = args.includes("--quiet");
  await fs.mkdir(outputDir, { recursive: true });
  const selected = source === "all"
    ? ["deribit-dvol", "cboe-vix", "deribit-surface", "coinmetrics", "mempool", "community-crypto"]
    : [source];
  for (const item of selected) {
    if (item === "deribit-dvol") await fetchDvol(outputDir, start, end, quiet);
    else if (item === "cboe-vix") await fetchVix(outputDir, start, end);
    else if (item === "deribit-surface") await fetchDeribitSurface(outputDir, quiet);
    else if (item === "coinmetrics") await fetchCoinMetrics(outputDir, start, end, quiet);
    else if (item === "mempool") await fetchMempool(outputDir, quiet);
    else if (item === "community-crypto") await fetchCommunityCrypto(outputDir, quiet);
    else throw new Error(`Unknown --source ${item}`);
  }
}

async function fetchVix(outputDir: string, start: string, end: string) {
  const url = new URL("https://fred.stlouisfed.org/graph/fredgraph.csv");
  url.searchParams.set("id", "VIXCLS");
  const rows = normalizeFredVixCsv(await requestText(url))
    .filter((row) => row.time >= parseDay(start) && row.time < parseDay(end) + DAY_MS);
  await writeArtifact(path.join(outputDir, "fred-cboe-vix-1d.json"), {
    version: 1,
    source: "Federal Reserve Bank of St. Louis FRED VIXCLS; source series Cboe Market Statistics",
    retrievedAt: new Date().toISOString(),
    request: {
      series: "VIXCLS",
      frequency: "daily close",
      start,
      end,
      causalAvailability: "following UTC day boundary",
      url: "https://fred.stlouisfed.org/series/VIXCLS",
    },
    rows,
  });
  console.log(`Stored ${rows.length.toLocaleString()} daily VIX rows.`);
}

async function fetchCommunityCrypto(outputDir: string, quiet: boolean) {
  const api = new URL("https://api.github.com/repos/ErcinDedeoglu/crypto-market-data/contents/data/daily");
  const listing = await requestJson<Array<{ name: string; download_url: string | null }>>(api);
  const files = listing.filter((item) => item.name.endsWith(".json") && item.download_url);
  const payloads: Record<string, CommunityCryptoMetricPayload> = {};
  for (const file of files) {
    const metric = file.name.replace(/\.json$/i, "");
    payloads[metric] = await requestJson(new URL(file.download_url!));
    if (!quiet) console.log(`Community crypto daily: ${metric}`);
  }
  const rows = normalizeCommunityCryptoDaily(payloads);
  await writeArtifact(path.join(outputDir, "community-crypto-market-daily.json"), {
    version: 1,
    source: "ErcinDedeoglu/crypto-market-data daily archive (CC BY 4.0; upstream-derived and retrospectively revised)",
    retrievedAt: new Date().toISOString(),
    request: {
      repository: "https://github.com/ErcinDedeoglu/crypto-market-data",
      files: files.map((file) => file.name),
      assumedAvailability: "next UTC day; latest upstream modification retained separately",
    },
    rows,
  });
  console.log(`Stored ${rows.length.toLocaleString()} community daily rows across ${files.length} metrics.`);
}

async function fetchDvol(outputDir: string, start: string, end: string, quiet: boolean) {
  const startTime = parseDay(start);
  const endTime = parseDay(end) + DAY_MS - 1;
  const rows = [];
  const chunkMs = 35 * DAY_MS;
  for (let cursor = startTime; cursor <= endTime; cursor += chunkMs) {
    const chunkEnd = Math.min(endTime, cursor + chunkMs - 1);
    const url = new URL("https://www.deribit.com/api/v2/public/get_volatility_index_data");
    url.searchParams.set("currency", "BTC");
    url.searchParams.set("start_timestamp", String(cursor));
    url.searchParams.set("end_timestamp", String(chunkEnd));
    url.searchParams.set("resolution", "3600");
    const payload = await requestJson<{ result: { data: unknown[]; continuation: unknown } }>(url);
    rows.push(...normalizeDvolRows(payload.result.data));
    if (!quiet) console.log(`Deribit DVOL through ${new Date(chunkEnd).toISOString().slice(0, 10)}: ${rows.length} rows`);
  }
  const unique = [...new Map(rows.map((row) => [row.time, row])).values()]
    .sort((left, right) => left.time - right.time);
  await writeArtifact(path.join(outputDir, "deribit-btc-dvol-1h.json"), {
    version: 1,
    source: "Deribit public/get_volatility_index_data",
    retrievedAt: new Date().toISOString(),
    request: { currency: "BTC", resolution: "3600", start, end, causalAvailability: "candle open + 1h" },
    rows: unique,
  });
  console.log(`Stored ${unique.length.toLocaleString()} hourly DVOL rows.`);
}

async function fetchCoinMetrics(outputDir: string, start: string, end: string, quiet: boolean) {
  const rows: Array<Record<string, unknown>> = [];
  let pageToken: string | undefined;
  do {
    const url = new URL("https://community-api.coinmetrics.io/v4/timeseries/asset-metrics");
    url.searchParams.set("assets", "btc");
    url.searchParams.set("metrics", COIN_METRICS.join(","));
    url.searchParams.set("frequency", "1d");
    url.searchParams.set("start_time", start);
    url.searchParams.set("end_time", end);
    url.searchParams.set("page_size", "10000");
    if (pageToken) url.searchParams.set("next_page_token", pageToken);
    const payload = await requestJson<{ data: Array<Record<string, unknown>>; next_page_token?: string }>(url);
    rows.push(...payload.data);
    pageToken = payload.next_page_token || undefined;
    if (!quiet) console.log(`Coin Metrics: ${rows.length} rows`);
  } while (pageToken);
  const normalized = normalizeCoinMetricsRows(rows, [...COIN_METRICS]);
  await writeArtifact(path.join(outputDir, "coinmetrics-btc-network-flows-1d.json"), {
    version: 1,
    source: "Coin Metrics Community API v4 asset-metrics",
    retrievedAt: new Date().toISOString(),
    request: { asset: "btc", frequency: "1d", metrics: COIN_METRICS, start, end },
    rows: normalized,
  });
  console.log(`Stored ${normalized.length.toLocaleString()} Coin Metrics daily rows.`);
}

async function fetchMempool(outputDir: string, quiet: boolean) {
  const endpoints = {
    fees: "https://mempool.space/api/v1/mining/blocks/fees/3y",
    feeRates: "https://mempool.space/api/v1/mining/blocks/fee-rates/3y",
    sizesWeights: "https://mempool.space/api/v1/mining/blocks/sizes-weights/3y",
    rewards: "https://mempool.space/api/v1/mining/blocks/rewards/3y",
  } as const;
  const payloads: Record<string, unknown> = {};
  for (const [name, url] of Object.entries(endpoints)) {
    payloads[name] = await requestJson(new URL(url));
    if (!quiet) console.log(`mempool.space: fetched ${name}`);
  }
  const rows = mergeMempoolMiningSeries(payloads);
  await writeArtifact(path.join(outputDir, "mempool-btc-mining-proxies-3y.json"), {
    version: 1,
    source: "mempool.space public mining REST API",
    retrievedAt: new Date().toISOString(),
    request: { period: "3y", endpoints, causalAvailability: "aggregate timestamp + median bucket gap" },
    rows,
  });
  await appendSnapshot(path.join(outputDir, "mempool-live-snapshots.jsonl"), {
    observedAt: Date.now(),
    mempool: await requestJson(new URL("https://mempool.space/api/mempool")),
    recommendedFees: await requestJson(new URL("https://mempool.space/api/v1/fees/recommended")),
    projectedBlocks: await requestJson(new URL("https://mempool.space/api/v1/fees/mempool-blocks")),
    recent: await requestJson(new URL("https://mempool.space/api/mempool/recent")),
  });
  console.log(`Stored ${rows.length.toLocaleString()} mempool mining-proxy rows and one live snapshot.`);
}

async function fetchDeribitSurface(outputDir: string, quiet: boolean) {
  const observedAt = Date.now();
  const url = new URL("https://www.deribit.com/api/v2/public/get_book_summary_by_currency");
  url.searchParams.set("currency", "BTC");
  url.searchParams.set("kind", "option");
  const payload = await requestJson<{ result: DeribitBookSummary[] }>(url);
  const summary = deriveDeribitOptionSummary(payload.result, observedAt);
  await appendSnapshot(path.join(outputDir, "deribit-btc-option-surface-summaries.jsonl"), summary);
  const rawDirectory = path.join(outputDir, "deribit-option-surface-raw");
  await fs.mkdir(rawDirectory, { recursive: true });
  await writeArtifact(path.join(rawDirectory, `${new Date(observedAt).toISOString().replace(/[:.]/g, "-")}.json`), {
    version: 1,
    source: "Deribit public/get_book_summary_by_currency",
    retrievedAt: new Date(observedAt).toISOString(),
    request: { currency: "BTC", kind: "option" },
    rows: payload.result,
  });
  if (!quiet) console.log(`Option term snapshot: ${JSON.stringify(summary.term)}`);
  console.log(`Stored Deribit option surface with ${payload.result.length.toLocaleString()} instruments.`);
}

async function requestJson<T = unknown>(url: URL, attempts = 5): Promise<T> {
  return JSON.parse(await requestText(url, attempts)) as T;
}

async function requestText(url: URL, attempts = 5): Promise<string> {
  let lastError: unknown;
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 60_000);
    try {
      const response = await fetch(url, {
        signal: controller.signal,
        headers: { "user-agent": "trading-external-feature-audit/1.0" },
      });
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}: ${await response.text()}`);
      return await response.text();
    } catch (error) {
      lastError = error;
      if (attempt + 1 < attempts) await delay(500 * 2 ** attempt);
    } finally {
      clearTimeout(timeout);
    }
  }
  throw lastError;
}

async function writeArtifact<T>(file: string, artifact: Omit<Artifact<T>, "sha256">): Promise<void> {
  const rows = JSON.stringify(artifact.rows);
  const complete = { ...artifact, sha256: createHash("sha256").update(rows).digest("hex") };
  await atomicWrite(file, `${JSON.stringify(complete)}\n`);
}

async function appendSnapshot(file: string, snapshot: unknown): Promise<void> {
  await fs.mkdir(path.dirname(file), { recursive: true });
  await fs.appendFile(file, `${JSON.stringify(snapshot)}\n`, "utf8");
}

async function atomicWrite(file: string, contents: string): Promise<void> {
  await fs.mkdir(path.dirname(file), { recursive: true });
  const temporary = `${file}.${process.pid}.partial`;
  await fs.writeFile(temporary, contents, "utf8");
  await fs.rename(temporary, file);
}

function parseDay(value: string): number {
  const parsed = Date.parse(`${value}T00:00:00.000Z`);
  if (!Number.isFinite(parsed) || new Date(parsed).toISOString().slice(0, 10) !== value) {
    throw new Error(`Invalid date ${value}`);
  }
  return parsed;
}

function delay(milliseconds: number) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  run().catch((error: unknown) => {
    console.error(error instanceof Error ? error.stack ?? error.message : error);
    process.exitCode = 1;
  });
}
