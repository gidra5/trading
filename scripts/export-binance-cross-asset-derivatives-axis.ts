import { createHash } from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import AdmZip from "adm-zip";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const DEFAULT_AXIS = "data/runtime-cache/binance-cross-asset-1m-basis-30d";
const START = Date.parse("2026-07-18T00:00:00.000Z");
const END = Date.parse("2026-08-17T00:00:00.000Z");
const DAY_MS = 86_400_000;
const STEP_MS = 300_000;
const ROWS = (END - START) / STEP_MS;
const COLUMNS = [
  "sumOpenInterest",
  "sumOpenInterestValue",
  "topTraderAccountLongShortRatio",
  "topTraderPositionLongShortRatio",
  "globalLongShortRatio",
  "takerBuySellVolumeRatio",
  "observed",
] as const;
const HEADER = [
  "create_time",
  "symbol",
  "sum_open_interest",
  "sum_open_interest_value",
  "count_toptrader_long_short_ratio",
  "sum_toptrader_long_short_ratio",
  "count_long_short_ratio",
  "sum_taker_long_short_vol_ratio",
].join(",");

interface AxisMarket { venue: "spot" | "usdm-futures"; symbol: string }
interface AxisAsset { asset: string; markets: AxisMarket[] }

export async function run(args = process.argv.slice(2)): Promise<void> {
  const value = (name: string) => {
    const index = args.indexOf(name);
    return index < 0 ? undefined : args[index + 1];
  };
  if (args.includes("--help")) {
    console.log(`Usage: npm run analysis:cross-asset-derivatives:export -- [options]

  --axis-dir DIR                  Completed compact cross-asset minute axis
  --concurrency N                 Concurrent markets (default 12)
  --limit N                       Optional smoke-test market limit

Downloads checksum-verified Binance USD-M 5m positioning/open-interest metrics
and official funding history. Only compact aligned values are retained.`);
    return;
  }
  const axisDir = resolve(value("--axis-dir") ?? DEFAULT_AXIS);
  const concurrency = Number(value("--concurrency") ?? 12);
  const limit = value("--limit") ? Number(value("--limit")) : undefined;
  if (!Number.isInteger(concurrency) || concurrency < 1 || concurrency > 32) {
    throw new Error("--concurrency must be an integer from 1 through 32");
  }
  const manifest = JSON.parse(await fs.readFile(path.join(axisDir, "manifest.json"), "utf8"));
  let assets = (manifest.assets as AxisAsset[])
    .map((asset) => ({
      ...asset,
      market: asset.markets.find((market) => market.venue === "usdm-futures"),
    }))
    .filter((asset) => asset.market);
  if (limit) assets = assets.slice(0, limit);
  let completed = 0;
  const results = await mapConcurrent(assets, concurrency, async (asset) => {
    const directory = path.join(axisDir, "assets", safe(asset.asset));
    const symbol = asset.market!.symbol;
    try {
      const metrics = await exportMetrics(directory, symbol);
      const funding = await exportFunding(directory, symbol);
      return { asset: asset.asset, symbol, metrics, funding, error: null };
    } catch (error) {
      return { asset: asset.asset, symbol, metrics: null, funding: null, error: message(error) };
    } finally {
      completed += 1;
      if (completed % 10 === 0 || completed === assets.length) {
        console.error(`${completed}/${assets.length} USD-M assets exported`);
      }
    }
  });
  const summary = {
    version: 1,
    generatedAt: new Date().toISOString(),
    sourceAxis: relative(path.join(axisDir, "manifest.json")),
    window: { start: new Date(START).toISOString(), endExclusive: new Date(END).toISOString() },
    metrics: { stepMs: STEP_MS, rows: ROWS, columns: COLUMNS, availabilityLagMs: STEP_MS },
    requestedAssets: assets.length,
    completedMetrics: results.filter((row) => row.metrics && row.metrics.coverage >= 0.95).length,
    completedFunding: results.filter((row) => row.funding && row.funding.events > 0).length,
    assets: results,
  };
  await fs.writeFile(path.join(axisDir, "derivatives-manifest.json"), `${JSON.stringify(summary, null, 2)}\n`);
  console.log(`Wrote ${relative(path.join(axisDir, "derivatives-manifest.json"))}`);
}

async function exportMetrics(directory: string, symbol: string): Promise<{ file: string; observedRows: number; coverage: number }> {
  const file = path.join(directory, "usdm-metrics.f32");
  const expectedBytes = ROWS * COLUMNS.length * 4;
  try {
    const stat = await fs.stat(file);
    if (stat.size === expectedBytes) {
      const payload = await fs.readFile(file);
      const values = new Float32Array(
        payload.buffer,
        payload.byteOffset,
        payload.byteLength / Float32Array.BYTES_PER_ELEMENT,
      );
      let observedRows = 0;
      for (let row = 0; row < ROWS; row += 1) observedRows += values[row * COLUMNS.length + 6] === 1 ? 1 : 0;
      return { file: relative(file), observedRows, coverage: observedRows / ROWS };
    }
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
  }
  const values = new Float32Array(ROWS * COLUMNS.length);
  values.fill(Number.NaN);
  for (let row = 0; row < ROWS; row += 1) values[row * COLUMNS.length + 6] = 0;
  for (let day = START; day < END; day += DAY_MS) {
    const date = new Date(day).toISOString().slice(0, 10);
    const archiveName = `${symbol}-metrics-${date}.zip`;
    const url = `https://data.binance.vision/data/futures/um/daily/metrics/${encodeURIComponent(symbol)}/${archiveName}`;
    try {
      parseMetrics(await downloadVerified(url), `${symbol}-metrics-${date}.csv`, symbol, values);
    } catch (error) {
      if (/HTTP 404/.test(message(error))) continue;
      throw error;
    }
  }
  let observedRows = 0;
  for (let row = 0; row < ROWS; row += 1) observedRows += values[row * COLUMNS.length + 6] === 1 ? 1 : 0;
  await fs.mkdir(directory, { recursive: true });
  await fs.writeFile(file, Buffer.from(values.buffer));
  return { file: relative(file), observedRows, coverage: observedRows / ROWS };
}

function parseMetrics(archive: Buffer, expectedCsv: string, symbol: string, output: Float32Array): void {
  const entries = new AdmZip(archive).getEntries().filter((entry) => !entry.isDirectory);
  if (entries.length !== 1 || entries[0]!.entryName !== expectedCsv) {
    throw new Error(`expected one ZIP entry named ${expectedCsv}`);
  }
  const lines = entries[0]!.getData().toString("utf8").replace(/^\ufeff/, "").trimEnd().split(/\r?\n/);
  if (lines[0]?.trim().toLowerCase() !== HEADER) throw new Error(`${symbol}: metrics schema changed`);
  for (const line of lines.slice(1)) {
    const row = line.split(",").map((item) => item.trim().replace(/^"|"$/g, ""));
    if (row.length !== 8 || row[1] !== symbol) continue;
    const time = Date.parse(`${row[0]!.replace(" ", "T")}Z`);
    if (time < START || time >= END || (time - START) % STEP_MS !== 0) continue;
    const values = row.slice(2).map(Number);
    if (values.some((item) => !Number.isFinite(item) || item <= 0)) continue;
    const offset = ((time - START) / STEP_MS) * COLUMNS.length;
    for (let column = 0; column < values.length; column += 1) output[offset + column] = values[column]!;
    output[offset + 6] = 1;
  }
}

async function exportFunding(directory: string, symbol: string): Promise<{ file: string; events: number }> {
  const file = path.join(directory, "usdm-funding.json");
  try {
    const cached = JSON.parse(await fs.readFile(file, "utf8"));
    if (cached.symbol === symbol && cached.start === START && cached.endExclusive === END) {
      return { file: relative(file), events: cached.events.length };
    }
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
  }
  const url = new URL("https://fapi.binance.com/fapi/v1/fundingRate");
  url.searchParams.set("symbol", symbol);
  url.searchParams.set("startTime", String(START));
  url.searchParams.set("endTime", String(END - 1));
  url.searchParams.set("limit", "1000");
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${symbol} funding: HTTP ${response.status}`);
  const payload = await response.json() as Array<{ fundingTime: number; fundingRate: string; markPrice?: string }>;
  const events = payload.map((row) => ({
    time: Number(row.fundingTime),
    rate: Number(row.fundingRate),
    markPrice: row.markPrice === undefined ? null : Number(row.markPrice),
  })).filter((row) => row.time >= START && row.time < END && Number.isFinite(row.rate));
  await fs.mkdir(directory, { recursive: true });
  await fs.writeFile(file, `${JSON.stringify({ version: 1, source: "fapi.binance.com/fapi/v1/fundingRate", symbol, start: START, endExclusive: END, events }, null, 2)}\n`);
  return { file: relative(file), events: events.length };
}

async function downloadVerified(url: string): Promise<Buffer> {
  const [checksumResponse, archiveResponse] = await Promise.all([fetch(`${url}.CHECKSUM`), fetch(url)]);
  if (!archiveResponse.ok) throw new Error(`${path.basename(url)}: HTTP ${archiveResponse.status}`);
  if (!checksumResponse.ok) throw new Error(`${path.basename(url)}.CHECKSUM: HTTP ${checksumResponse.status}`);
  const checksum = /^([a-fA-F0-9]{64})(?:\s|$)/.exec((await checksumResponse.text()).trim())?.[1]?.toLowerCase();
  if (!checksum) throw new Error(`${path.basename(url)}: invalid checksum`);
  const archive = Buffer.from(await archiveResponse.arrayBuffer());
  if (createHash("sha256").update(archive).digest("hex") !== checksum) throw new Error(`${path.basename(url)}: checksum mismatch`);
  return archive;
}

async function mapConcurrent<T, R>(values: readonly T[], concurrency: number, worker: (value: T) => Promise<R>): Promise<R[]> {
  const output = new Array<R>(values.length);
  let next = 0;
  await Promise.all(Array.from({ length: Math.min(concurrency, values.length) }, async () => {
    while (true) {
      const index = next++;
      if (index >= values.length) return;
      output[index] = await worker(values[index]!);
    }
  }));
  return output;
}

function safe(value: string): string { return Buffer.from(value).toString("base64url"); }
function message(error: unknown): string { return error instanceof Error ? error.message : String(error); }
function resolve(file: string): string { return path.resolve(ROOT, file); }
function relative(file: string): string { return path.relative(ROOT, file).replaceAll("\\", "/"); }

const invoked = process.argv[1] ? path.resolve(process.argv[1]) : undefined;
if (invoked === path.resolve(import.meta.filename)) {
  run().catch((error: unknown) => {
    console.error(error instanceof Error ? error.stack ?? error.message : error);
    process.exitCode = 1;
  });
}
