import { createHash } from "node:crypto";
import { Resolver } from "node:dns/promises";
import fs from "node:fs/promises";
import { request as httpsRequest } from "node:https";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  deriveDeribitOptionSummary,
  mergeMempoolMiningSeries,
  normalizeCommunityCryptoDaily,
  normalizeCoinMetricsRows,
  normalizeDvolRows,
  normalizeBinanceFundingRows,
  normalizeMacroCsv,
  normalizeMacroHtmlTable,
  normalizeFredVixCsv,
  type BinanceFundingRateRow,
  type DeribitBookSummary,
  type CommunityCryptoMetricPayload,
  type MacroRow,
  type MacroSeriesDefinition,
} from "./lib/external-public-data.ts";

const DAY_MS = 86_400_000;
const DEFAULT_START = "2021-03-24";
const COIN_METRICS = [
  "AdrActCnt", "BlkCnt", "CapMVRVCur", "FeeTotNtv", "FlowInExNtv", "FlowInExUSD",
  "FlowOutExNtv", "FlowOutExUSD", "HashRate", "IssTotNtv", "IssTotUSD", "SplyExNtv",
  "SplyExUSD", "TxCnt", "TxTfrCnt", "volume_reported_spot_usd_1d",
] as const;
const fred = (
  id: string,
  label: string,
  economy: string,
  frequency: MacroSeriesDefinition["frequency"],
  availabilityLagDays: number,
): MacroSeriesDefinition => ({
  id, label, economy, frequency, availabilityLagDays,
  provider: "Federal Reserve Bank of St. Louis FRED",
  sourceUrl: `https://fred.stlouisfed.org/series/${id}`,
});
const FRED_MACRO_SERIES: MacroSeriesDefinition[] = [
  fred("DFF", "US effective federal funds rate", "United States", "daily", 1),
  fred("ECBDFR", "ECB deposit facility rate", "Euro area", "daily", 1),
  fred("DGS2", "US 2-year Treasury yield", "United States", "daily", 1),
  fred("DGS10", "US 10-year Treasury yield", "United States", "daily", 1),
  fred("T10Y2Y", "US 10-year minus 2-year yield spread", "United States", "daily", 1),
  fred("DTWEXBGS", "Trade-weighted US dollar index", "United States", "daily", 1),
  fred("BAMLH0A0HYM2", "US high-yield credit spread", "United States", "daily", 1),
  fred("CPIAUCSL", "US consumer price index", "United States", "monthly", 45),
  fred("CP0000EZ19M086NEST", "Euro-area consumer price index", "Euro area", "monthly", 45),
  fred("UNRATE", "US unemployment rate", "United States", "monthly", 40),
  fred("PAYEMS", "US nonfarm payroll employment", "United States", "monthly", 40),
  fred("INDPRO", "US industrial production index", "United States", "monthly", 50),
  fred("GDPC1", "US real GDP", "United States", "quarterly", 150),
  fred("CLVMNACSCAB1GQEA19", "Euro-area real GDP", "Euro area", "quarterly", 150),
  fred("JPNRGDPEXP", "Japan real GDP", "Japan", "quarterly", 150),
  fred("DEXCHUS", "Chinese yuan per US dollar", "China", "daily", 1),
  fred("IRSTCI01JPM156N", "Japan short-term interest rate", "Japan", "monthly", 45),
  fred("IRSTCI01INM156N", "India short-term interest rate", "India", "monthly", 45),
];

const ecb = (id: string, label: string): MacroSeriesDefinition => ({
  id, label, economy: "Euro area", frequency: "daily", availabilityLagDays: 1,
  provider: "European Central Bank Data Portal",
  sourceUrl: id === "ECB_ESTR"
    ? "https://data.ecb.europa.eu/data/datasets/EST"
    : "https://data.ecb.europa.eu/data/datasets/YC",
});
const ECB_MACRO_SERIES = [
  { definition: ecb("ECB_ESTR", "Euro short-term rate (€STR)"), key: "EST/B.EU000A2X2A25.WT" },
  { definition: ecb("ECB_YC_2Y", "Euro-area AAA 2-year yield"), key: "YC/B.U2.EUR.4F.G_N_C.SV_C_YM.SR_2Y" },
  { definition: ecb("ECB_YC_10Y", "Euro-area AAA 10-year yield"), key: "YC/B.U2.EUR.4F.G_N_C.SV_C_YM.SR_10Y" },
] as const;

const oecd = (
  id: string,
  label: string,
  economy: string,
  frequency: MacroSeriesDefinition["frequency"],
  availabilityLagDays: number,
  sourceUrl: string,
): MacroSeriesDefinition => ({
  id, label, economy, frequency, availabilityLagDays,
  provider: "OECD Data Explorer SDMX API",
  sourceUrl,
});
const OECD_CPI_FLOW = "OECD.SDD.TPS,DSD_G20_PRICES@DF_G20_PRICES,1.0";
const OECD_GDP_FLOW = "OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA_EXPENDITURE_GROWTH_G20,1.1";
const OECD_INDUSTRIAL_FLOW = "OECD.SDD.STES,DSD_KEI@DF_KEI,4.0";
const OECD_CPI_SERIES = [
  ["CHN", "China"], ["GBR", "United Kingdom"], ["JPN", "Japan"], ["IND", "India"],
].map(([area, economy]) => ({
  definition: oecd(`OECD_CPI_YOY_${area}`, `${economy} CPI year-over-year`, economy!, "monthly", 45,
    `https://data-explorer.oecd.org/vis?fs[0]=Topic%2C1%7CEconomy%23ECON%23%7CPrices%23ECON_PRICES%23`),
  key: `${area}.M...PA...`,
}));
const OECD_GDP_SERIES = [
  ["CHN", "China"], ["GBR", "United Kingdom"], ["IND", "India"],
].map(([area, economy]) => ({
  definition: oecd(`OECD_REAL_GDP_QOQ_${area}`, `${economy} real GDP quarter-over-quarter`, economy!, "quarterly", 150,
    "https://data-explorer.oecd.org/"),
  key: `Q.Y.${area}.S1.S1.B1GQ._Z._Z._Z.PC.L.G1.T0102`,
}));
const OECD_INDUSTRIAL_SERIES = [
  ["EA20", "Euro area"], ["GBR", "United Kingdom"], ["JPN", "Japan"], ["IND", "India"],
].map(([area, economy]) => ({
  definition: oecd(`OECD_INDUSTRIAL_PRODUCTION_${area}`, `${economy} industrial production index`, economy!, "monthly", 50,
    "https://data-explorer.oecd.org/"),
  key: `${area}.M.PRVM.IX.BTE..`,
}));
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

  --source all|deribit-dvol|cboe-vix|global-macro|binance-funding|deribit-surface|coinmetrics|mempool|community-crypto
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
    ? ["deribit-dvol", "cboe-vix", "global-macro", "binance-funding", "deribit-surface", "coinmetrics", "mempool", "community-crypto"]
    : [source];
  for (const item of selected) {
    if (item === "deribit-dvol") await fetchDvol(outputDir, start, end, quiet);
    else if (item === "cboe-vix") await fetchVix(outputDir, start, end);
    else if (item === "global-macro") await fetchGlobalMacro(outputDir, start, end, quiet);
    else if (item === "binance-funding") await fetchBinanceFunding(outputDir, start, end, quiet);
    else if (item === "deribit-surface") await fetchDeribitSurface(outputDir, quiet);
    else if (item === "coinmetrics") await fetchCoinMetrics(outputDir, start, end, quiet);
    else if (item === "mempool") await fetchMempool(outputDir, quiet);
    else if (item === "community-crypto") await fetchCommunityCrypto(outputDir, quiet);
    else throw new Error(`Unknown --source ${item}`);
  }
}

async function fetchGlobalMacro(outputDir: string, start: string, end: string, quiet: boolean) {
  const rows: MacroRow[] = [];
  const failures: Array<{ id: string; provider: string; error: string }> = [];
  const attempt = async (
    definition: MacroSeriesDefinition,
    load: () => Promise<MacroRow[]>,
  ): Promise<MacroRow[]> => {
    try {
      return await load();
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      failures.push({ id: definition.id, provider: definition.provider, error: message });
      console.warn(`Skipped unavailable macro ${definition.id}: ${message}`);
      return [];
    }
  };
  const extendedStart = new Date(parseDay(start) - 400 * DAY_MS).toISOString().slice(0, 10);
  for (const definition of FRED_MACRO_SERIES) {
    const url = new URL("https://fred.stlouisfed.org/graph/fredgraph.csv");
    url.searchParams.set("id", definition.id);
    url.searchParams.set("cosd", extendedStart);
    url.searchParams.set("coed", end);
    const normalized = await attempt(definition, async () => normalizeMacroCsv(await requestText(url), definition)
      .filter((row) => row.time >= parseDay(start) - 400 * DAY_MS && row.time < parseDay(end) + DAY_MS));
    rows.push(...normalized);
    if (!quiet) console.log(`FRED macro ${definition.id}: ${normalized.length.toLocaleString()} rows`);
  }

  for (const { definition, key } of ECB_MACRO_SERIES) {
    const url = new URL(`https://data-api.ecb.europa.eu/service/data/${key}`);
    url.searchParams.set("startPeriod", extendedStart);
    url.searchParams.set("endPeriod", end);
    url.searchParams.set("format", "csvdata");
    const normalized = await attempt(definition, async () => normalizeMacroCsv(await requestText(url), definition, {
      period: "TIME_PERIOD", value: "OBS_VALUE",
    }));
    rows.push(...normalized);
    if (!quiet) console.log(`ECB macro ${definition.id}: ${normalized.length.toLocaleString()} rows`);
  }
  const euroTwoYear = rows.filter((row) => row.id === "ECB_YC_2Y");
  const euroTenYear = new Map(rows.filter((row) => row.id === "ECB_YC_10Y").map((row) => [row.time, row]));
  const euroSpreadDefinition: MacroSeriesDefinition = {
    ...ecb("ECB_YC_10Y2Y", "Euro-area AAA 10-year minus 2-year yield spread"),
    sourceUrl: "https://data.ecb.europa.eu/data/datasets/YC",
  };
  rows.push(...euroTwoYear.flatMap((twoYear) => {
    const tenYear = euroTenYear.get(twoYear.time);
    return tenYear ? [{ ...euroSpreadDefinition, time: twoYear.time, availableAt: Math.max(twoYear.availableAt, tenYear.availableAt), value: tenYear.value - twoYear.value }] : [];
  }));

  for (const { definition, key } of OECD_CPI_SERIES) {
    const url = oecdUrl(OECD_CPI_FLOW, key, extendedStart);
    const normalized = await attempt(definition, async () => normalizeMacroCsv(await requestText(url, 5, oecdHeaders()), definition, {
      period: "TIME_PERIOD", value: "OBS_VALUE",
      filter: { MEASURE: "CPI", UNIT_MEASURE: "PA", TRANSFORMATION: "GY" },
    }));
    rows.push(...normalized);
    if (!quiet) console.log(`OECD macro ${definition.id}: ${normalized.length.toLocaleString()} rows`);
  }
  for (const { definition, key } of OECD_GDP_SERIES) {
    const normalized = await attempt(definition, async () => normalizeMacroCsv(
      await requestText(oecdUrl(OECD_GDP_FLOW, key, extendedStart), 5, oecdHeaders()),
      definition,
      { period: "TIME_PERIOD", value: "OBS_VALUE" },
    ));
    rows.push(...normalized);
    if (!quiet) console.log(`OECD macro ${definition.id}: ${normalized.length.toLocaleString()} rows`);
  }
  for (const { definition, key } of OECD_INDUSTRIAL_SERIES) {
    const normalized = await attempt(definition, async () => normalizeMacroCsv(
      await requestText(oecdUrl(OECD_INDUSTRIAL_FLOW, key, extendedStart), 5, oecdHeaders()),
      definition,
      { period: "TIME_PERIOD", value: "OBS_VALUE" },
    ));
    rows.push(...normalized);
    if (!quiet) console.log(`OECD macro ${definition.id}: ${normalized.length.toLocaleString()} rows`);
  }

  const cbrStart = formatCbrDate(parseDay(start) - 400 * DAY_MS);
  const cbrEnd = formatCbrDate(parseDay(end));
  const cbrKeyRate: MacroSeriesDefinition = {
    id: "CBR_KEY_RATE", label: "Bank of Russia key rate", economy: "Russia", frequency: "event", availabilityLagDays: 1,
    provider: "Bank of Russia", sourceUrl: "https://www.cbr.ru/eng/hd_base/KeyRate/",
  };
  const cbrInflation: MacroSeriesDefinition = {
    id: "CBR_CPI_YOY", label: "Russia inflation year-over-year", economy: "Russia", frequency: "monthly", availabilityLagDays: 45,
    provider: "Bank of Russia", sourceUrl: "https://www.cbr.ru/statistics/ddkp/infl/",
  };
  const cbrKeyUrl = new URL(cbrKeyRate.sourceUrl);
  cbrKeyUrl.searchParams.set("UniDbQuery.Posted", "True");
  cbrKeyUrl.searchParams.set("UniDbQuery.From", cbrStart);
  cbrKeyUrl.searchParams.set("UniDbQuery.To", cbrEnd);
  const cbrInflationUrl = new URL(cbrInflation.sourceUrl);
  cbrInflationUrl.searchParams.set("UniDbQuery.Posted", "True");
  cbrInflationUrl.searchParams.set("UniDbQuery.From", cbrStart);
  cbrInflationUrl.searchParams.set("UniDbQuery.To", cbrEnd);
  const cbrKeyDailyRows = await attempt(cbrKeyRate, async () => normalizeMacroHtmlTable(await requestCbrText(cbrKeyUrl), cbrKeyRate, { period: 0, value: 1 }));
  const cbrKeyRows = cbrKeyDailyRows.filter((row, index) => index === 0 || row.value !== cbrKeyDailyRows[index - 1]!.value);
  const cbrInflationRows = await attempt(cbrInflation, async () => normalizeMacroHtmlTable(await requestCbrText(cbrInflationUrl), cbrInflation, { period: 0, value: 2 }));
  rows.push(...cbrKeyRows, ...cbrInflationRows);
  if (!quiet) console.log(`Bank of Russia macro: ${cbrKeyRows.length.toLocaleString()} key-rate and ${cbrInflationRows.length.toLocaleString()} inflation rows`);

  const boeDefinition: MacroSeriesDefinition = {
    id: "BOE_BANK_RATE", label: "Bank of England Bank Rate", economy: "United Kingdom", frequency: "event", availabilityLagDays: 1,
    provider: "Bank of England", sourceUrl: "https://www.bankofengland.co.uk/boeapps/database/Bank-Rate.asp",
  };
  const boeRows = await attempt(boeDefinition, async () => normalizeMacroHtmlTable(await requestText(new URL(boeDefinition.sourceUrl)), boeDefinition, { period: 0, value: 1 }));
  rows.push(...boeRows);
  if (!quiet) console.log(`Bank of England macro ${boeDefinition.id}: ${boeRows.length.toLocaleString()} rows`);

  const earliestTime = parseDay(start) - 400 * DAY_MS;
  const latestTime = parseDay(end) + DAY_MS;
  rows.splice(0, rows.length, ...rows.filter((row) => (
    row.time >= earliestTime && row.time < latestTime
    && row.availableAt >= earliestTime && row.availableAt < latestTime + 200 * DAY_MS
  )));
  rows.sort((left, right) => left.availableAt - right.availableAt || left.id.localeCompare(right.id));
  const series = [...new Set(rows.map((row) => row.id))];
  await writeArtifact(path.join(outputDir, "global-macro-state.json"), {
    version: 1,
    source: "Official and intergovernmental macro series from FRED, ECB, OECD, Bank of England, and Bank of Russia",
    retrievedAt: new Date().toISOString(),
    request: {
      start,
      end,
      series: [...FRED_MACRO_SERIES, ...ECB_MACRO_SERIES.map((item) => item.definition), ...OECD_CPI_SERIES.map((item) => item.definition), ...OECD_GDP_SERIES.map((item) => item.definition), ...OECD_INDUSTRIAL_SERIES.map((item) => item.definition), cbrKeyRate, cbrInflation, boeDefinition, euroSpreadDefinition],
      failedSeries: failures,
      causalAvailability: "fixed conservative provider/frequency release lags; current revised values, not vintage snapshots",
    },
    rows,
  });
  console.log(`Stored ${rows.length.toLocaleString()} global macro rows across ${series.length} valid series; ${failures.length} unavailable series.`);
}

async function fetchBinanceFunding(outputDir: string, start: string, end: string, quiet: boolean) {
  const startTime = parseDay(start);
  const endTime = parseDay(end) + DAY_MS - 1;
  const rows: BinanceFundingRateRow[] = [];
  let cursor = startTime;
  while (cursor <= endTime) {
    const url = new URL("https://fapi.binance.com/fapi/v1/fundingRate");
    url.searchParams.set("symbol", "BTCUSDT");
    url.searchParams.set("startTime", String(cursor));
    url.searchParams.set("endTime", String(endTime));
    url.searchParams.set("limit", "1000");
    const page = normalizeBinanceFundingRows(await requestJson<unknown[]>(url));
    if (page.length === 0) break;
    rows.push(...page);
    const next = page.at(-1)!.time + 1;
    if (next <= cursor || page.length < 1_000) break;
    cursor = next;
    if (!quiet) console.log(`Binance funding through ${new Date(page.at(-1)!.time).toISOString()}: ${rows.length.toLocaleString()} rows`);
  }
  const unique = [...new Map(rows.map((row) => [row.time, row])).values()]
    .filter((row) => row.time >= startTime && row.time <= endTime)
    .sort((left, right) => left.time - right.time);
  await writeArtifact(path.join(outputDir, "binance-btcusdt-funding-rates.json"), {
    version: 1,
    source: "Binance USD-M Futures public funding-rate history",
    retrievedAt: new Date().toISOString(),
    request: {
      symbol: "BTCUSDT",
      endpoint: "GET /fapi/v1/fundingRate",
      start,
      end,
      causalAvailability: "funding settlement timestamp + 1 minute",
    },
    rows: unique,
  });
  console.log(`Stored ${unique.length.toLocaleString()} BTCUSDT funding settlements.`);
}

async function fetchVix(outputDir: string, start: string, end: string) {
  const url = new URL("https://fred.stlouisfed.org/graph/fredgraph.csv");
  url.searchParams.set("id", "VIXCLS");
  url.searchParams.set("cosd", start);
  url.searchParams.set("coed", end);
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

async function requestText(url: URL, attempts = 5, headers: Record<string, string> = {}): Promise<string> {
  let lastError: unknown;
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 60_000);
    try {
      const response = await fetch(url, {
        signal: controller.signal,
        headers: { "user-agent": "trading-external-feature-audit/1.0", ...headers },
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

function oecdUrl(flow: string, key: string, start: string) {
  const url = new URL(`https://sdmx.oecd.org/public/rest/v1/data/${flow}/${key}`);
  url.searchParams.set("startPeriod", start.slice(0, 7));
  return url;
}

function oecdHeaders() {
  return { accept: "text/csv", "accept-language": "en" };
}

function formatCbrDate(time: number) {
  const date = new Date(time);
  return [String(date.getUTCDate()).padStart(2, "0"), String(date.getUTCMonth() + 1).padStart(2, "0"), date.getUTCFullYear()].join(".");
}

async function requestCbrText(url: URL): Promise<string> {
  try {
    return await requestText(url, 2);
  } catch (originalError) {
    const resolver = new Resolver();
    resolver.setServers(["8.8.8.8"]);
    const addresses = await resolver.resolve4(url.hostname);
    const address = addresses[0];
    if (!address) throw originalError;
    return await new Promise<string>((resolve, reject) => {
      const request = httpsRequest(url, {
        headers: { "user-agent": "trading-external-feature-audit/1.0" },
        lookup: (_hostname, options, callback) => {
          if (typeof options === "object" && options.all) callback(null, [{ address, family: 4 }]);
          else callback(null, address, 4);
        },
      }, (response) => {
        const chunks: Buffer[] = [];
        response.on("data", (chunk: Buffer) => chunks.push(chunk));
        response.on("end", () => {
          const body = Buffer.concat(chunks).toString("utf8");
          if ((response.statusCode ?? 500) >= 400) reject(new Error(`${response.statusCode}: ${body}`));
          else resolve(body);
        });
      });
      request.setTimeout(60_000, () => request.destroy(new Error("Bank of Russia request timed out")));
      request.on("error", reject);
      request.end();
    });
  }
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
