import { createHash } from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import AdmZip from "adm-zip";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const DAY_MS = 86_400_000;
const MINUTE_MS = 60_000;
const START = Date.parse("2026-07-18T00:00:00.000Z");
const END = Date.parse("2026-08-17T00:00:00.000Z");
const ROWS = (END - START) / MINUTE_MS;
const COLUMNS = [
  "open", "high", "low", "close", "baseVolume", "quoteVolume",
  "tradeCount", "takerBuyBaseVolume", "takerBuyQuoteVolume", "observed",
] as const;
const DEFAULT_UNIVERSE = "docs/portfolio/runs/2026-07-22-all-usdt-1m-360c-q4-amp5pct-pearson-k257.json";
const DEFAULT_OUTPUT = "data/runtime-cache/binance-cross-asset-1m-basis-30d";
const EXPLICIT_ASSETS = ["ETH", "SOL", "XRP", "HYPE"];

type Venue = "spot" | "usdm-futures";

interface ProductAsset {
  baseAsset: string;
  priceVenue?: Venue;
  products?: string[];
  productSymbols?: Partial<Record<Venue, string[]>>;
}

interface Market {
  venue: Venue;
  symbol: string;
}

interface AssetResult {
  asset: string;
  rank: number | null;
  requestedExplicitly: boolean;
  markets: Array<Market & { file: string; observedRows: number; coverage: number }>;
  preferredMarket: Market | null;
  errors: string[];
}

export async function run(args = process.argv.slice(2)): Promise<void> {
  const value = (name: string) => {
    const index = args.indexOf(name);
    return index < 0 ? undefined : args[index + 1];
  };
  if (args.includes("--help")) {
    console.log(`Usage: npm run analysis:cross-asset-axis:export -- [options]

  --universe FILE                 Multiscale Binance basis JSON
  --output-dir DIR                Compact derived-source directory
  --concurrency N                 Concurrent asset downloads (default 8)
  --limit N                       Optional smoke-test asset limit

Official monthly/daily ZIPs are checksum verified, converted to a compact aligned
float32 minute axis, and discarded. Both spot and USD-M are retained when available.`);
    return;
  }
  const universeFile = resolve(value("--universe") ?? DEFAULT_UNIVERSE);
  const output = resolve(value("--output-dir") ?? DEFAULT_OUTPUT);
  const concurrency = Number(value("--concurrency") ?? 8);
  const limit = value("--limit") ? Number(value("--limit")) : undefined;
  if (!Number.isInteger(concurrency) || concurrency < 1 || concurrency > 32) {
    throw new Error("--concurrency must be an integer from 1 through 32");
  }
  await fs.mkdir(output, { recursive: true });
  const universe = JSON.parse(await fs.readFile(universeFile, "utf8"));
  const historical = await historicalProducts(universe);
  const current = await currentProducts();
  const basisEntries = (universe.basis?.entries ?? universe.entries) as Array<{
    rank: number;
    asset?: string;
    baseAsset?: string;
  }>;
  const ranked = basisEntries.map((entry) => ({
    asset: entry.asset ?? entry.baseAsset!,
    rank: entry.rank,
    requestedExplicitly: false,
  }));
  for (const asset of EXPLICIT_ASSETS) if (!ranked.some((row) => row.asset === asset)) {
    ranked.push({ asset, rank: null as unknown as number, requestedExplicitly: true });
  }
  const requested = limit ? ranked.slice(0, limit) : ranked;
  let complete = 0;
  const results = await mapConcurrent(requested, concurrency, async (entry, index) => {
    const products = mergeProducts(historical.get(entry.asset), current.get(entry.asset));
    const markets = marketsFor(products);
    const errors: string[] = [];
    const stored: AssetResult["markets"] = [];
    for (const market of markets) {
      try {
        const row = await exportMarket(output, entry.asset, market);
        stored.push(row);
      } catch (error) {
        errors.push(`${market.venue}:${market.symbol}: ${message(error)}`);
      }
    }
    const preferred = stored.find((row) => row.venue === "spot" && row.coverage >= 0.99)
      ?? stored.find((row) => row.venue === "usdm-futures" && row.coverage >= 0.99)
      ?? stored.slice().sort((left, right) => right.coverage - left.coverage)[0]
      ?? null;
    complete += 1;
    if (complete % 10 === 0 || complete === requested.length) {
      console.error(`${complete}/${requested.length} assets exported`);
    }
    return {
      asset: entry.asset,
      rank: entry.rank,
      requestedExplicitly: entry.requestedExplicitly,
      markets: stored,
      preferredMarket: preferred ? { venue: preferred.venue, symbol: preferred.symbol } : null,
      errors,
    } satisfies AssetResult;
  });
  const manifest = {
    version: 1,
    generatedAt: new Date().toISOString(),
    objective: "Compact aligned source axis for the comprehensive Binance cross-asset BTC feature search",
    sourceUniverse: relative(universeFile),
    sourceUniverseSize: universe.universe?.eligibleSymbols ?? universe.basis?.universeSize ?? universe.universeSize,
    sourceBasisSize: basisEntries.length,
    explicitAssets: EXPLICIT_ASSETS,
    requestedAssets: requested.length,
    window: {
      start: new Date(START).toISOString(),
      endExclusive: new Date(END).toISOString(),
      rows: ROWS,
      stepMs: MINUTE_MS,
    },
    layout: { dtype: "little-endian float32", columns: COLUMNS },
    assets: results,
    coverage: {
      preferredAtLeast99Percent: results.filter((row) => row.preferredMarket
        && row.markets.find((market) => market.venue === row.preferredMarket!.venue)?.coverage! >= 0.99).length,
      withSpot: results.filter((row) => row.markets.some((market) => market.venue === "spot")).length,
      withUsdmFutures: results.filter((row) => row.markets.some((market) => market.venue === "usdm-futures")).length,
      withBoth: results.filter((row) => row.markets.length === 2).length,
      failed: results.filter((row) => !row.preferredMarket).length,
    },
  };
  await fs.writeFile(path.join(output, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`);
  console.log(`Wrote ${relative(path.join(output, "manifest.json"))}`);
}

async function exportMarket(
  output: string,
  asset: string,
  market: Market,
): Promise<Market & { file: string; observedRows: number; coverage: number }> {
  const directory = path.join(output, "assets", safe(asset));
  const file = path.join(directory, `${market.venue}.f32`);
  const sidecar = path.join(directory, `${market.venue}.json`);
  const expectedBytes = ROWS * COLUMNS.length * 4;
  try {
    const [stat, cached] = await Promise.all([
      fs.stat(file),
      fs.readFile(sidecar, "utf8").then(JSON.parse),
    ]);
    if (stat.size === expectedBytes && cached.symbol === market.symbol && cached.rows === ROWS) {
      return { ...market, file: relative(file), observedRows: cached.observedRows, coverage: cached.coverage };
    }
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
  }
  const values = new Float32Array(ROWS * COLUMNS.length);
  values.fill(Number.NaN);
  for (let row = 0; row < ROWS; row += 1) values[row * COLUMNS.length + 9] = 0;
  let observedRows = 0;
  for (const archive of archives(market)) {
    try {
      const payload = await downloadVerified(archive.url);
      observedRows += parseKlines(payload, archive.csvName, market.symbol, values);
    } catch (error) {
      if (/HTTP 404/.test(message(error))) continue;
      throw error;
    }
  }
  await fs.mkdir(directory, { recursive: true });
  await fs.writeFile(file, Buffer.from(values.buffer));
  const coverage = observedRows / ROWS;
  await fs.writeFile(sidecar, `${JSON.stringify({
    version: 1,
    source: "data.binance.vision",
    sourceDataset: market.venue === "spot" ? "spot/klines" : "futures/um/klines",
    ...market,
    start: START,
    endExclusive: END,
    rows: ROWS,
    columns: COLUMNS,
    observedRows,
    coverage,
  }, null, 2)}\n`);
  return { ...market, file: relative(file), observedRows, coverage };
}

function parseKlines(
  archive: Buffer,
  expectedCsv: string,
  symbol: string,
  output: Float32Array,
): number {
  const zip = new AdmZip(archive);
  const entries = zip.getEntries().filter((entry) => !entry.isDirectory);
  if (entries.length !== 1 || entries[0]!.entryName !== expectedCsv) {
    throw new Error(`expected one ZIP entry named ${expectedCsv}`);
  }
  let stored = 0;
  const lines = entries[0]!.getData().toString("utf8").replace(/^\ufeff/, "").split(/\r?\n/);
  for (const line of lines) {
    if (!line || /^[A-Za-z_]/.test(line)) continue;
    const row = line.split(",");
    if (row.length < 11) continue;
    let time = Number(row[0]);
    if (time > 100_000_000_000_000) time = Math.floor(time / 1_000);
    if (time < START || time >= END || (time - START) % MINUTE_MS !== 0) continue;
    const index = (time - START) / MINUTE_MS;
    const offset = index * COLUMNS.length;
    const parsed = [row[1], row[2], row[3], row[4], row[5], row[7], row[8], row[9], row[10]]
      .map(Number);
    if (parsed.some((value) => !Number.isFinite(value)) || parsed.slice(0, 4).some((value) => value <= 0)) {
      throw new Error(`${symbol}: invalid kline at ${time}`);
    }
    for (let column = 0; column < parsed.length; column += 1) output[offset + column] = parsed[column]!;
    if (output[offset + 9] !== 1) stored += 1;
    output[offset + 9] = 1;
  }
  return stored;
}

function archives(market: Market): Array<{ url: string; csvName: string }> {
  const root = market.venue === "spot"
    ? "https://data.binance.vision/data/spot"
    : "https://data.binance.vision/data/futures/um";
  const encoded = encodeURIComponent(market.symbol);
  const result = [{
    url: `${root}/monthly/klines/${encoded}/1m/${encoded}-1m-2026-07.zip`,
    csvName: `${market.symbol}-1m-2026-07.csv`,
  }];
  for (let day = Date.parse("2026-08-01T00:00:00.000Z"); day < END; day += DAY_MS) {
    const date = new Date(day).toISOString().slice(0, 10);
    result.push({
      url: `${root}/daily/klines/${encoded}/1m/${encoded}-1m-${date}.zip`,
      csvName: `${market.symbol}-1m-${date}.csv`,
    });
  }
  return result;
}

async function downloadVerified(url: string): Promise<Buffer> {
  const [checksumResponse, archiveResponse] = await Promise.all([
    fetch(`${url}.CHECKSUM`),
    fetch(url),
  ]);
  if (!archiveResponse.ok) throw new Error(`${path.basename(url)}: HTTP ${archiveResponse.status}`);
  if (!checksumResponse.ok) throw new Error(`${path.basename(url)}.CHECKSUM: HTTP ${checksumResponse.status}`);
  const checksum = /^([a-fA-F0-9]{64})(?:\s|$)/.exec((await checksumResponse.text()).trim())?.[1]?.toLowerCase();
  if (!checksum) throw new Error(`${path.basename(url)}: invalid checksum`);
  const archive = Buffer.from(await archiveResponse.arrayBuffer());
  const actual = createHash("sha256").update(archive).digest("hex");
  if (actual !== checksum) throw new Error(`${path.basename(url)}: checksum mismatch`);
  return archive;
}

async function historicalProducts(universe: any): Promise<Map<string, ProductAsset>> {
  const output = new Map<string, ProductAsset>();
  for (const asset of universe.assets ?? []) {
    output.set(asset.baseAsset, mergeProducts(output.get(asset.baseAsset), asset)!);
  }
  for (const source of universe.sources ?? []) {
    const report = JSON.parse(await fs.readFile(resolve(source.file), "utf8"));
    for (const asset of report.assets ?? []) {
      output.set(asset.baseAsset, mergeProducts(output.get(asset.baseAsset), asset));
    }
  }
  return output;
}

async function currentProducts(): Promise<Map<string, ProductAsset>> {
  const output = new Map<string, ProductAsset>();
  const [spotResponse, futuresResponse] = await Promise.all([
    fetch("https://api.binance.com/api/v3/exchangeInfo"),
    fetch("https://fapi.binance.com/fapi/v1/exchangeInfo"),
  ]);
  if (!spotResponse.ok || !futuresResponse.ok) throw new Error("Binance exchange-info request failed");
  const spot = await spotResponse.json() as any;
  const futures = await futuresResponse.json() as any;
  for (const row of spot.symbols ?? []) if (row.quoteAsset === "USDT" && row.status === "TRADING") {
    addCurrent(output, row.baseAsset, "spot", row.symbol);
  }
  for (const row of futures.symbols ?? []) if (row.quoteAsset === "USDT" && row.status === "TRADING") {
    addCurrent(output, row.baseAsset, "usdm-futures", row.symbol);
  }
  return output;
}

function addCurrent(output: Map<string, ProductAsset>, asset: string, venue: Venue, symbol: string): void {
  const current = output.get(asset) ?? { baseAsset: asset, products: [], productSymbols: {} };
  current.products = [...new Set([...(current.products ?? []), venue])];
  current.productSymbols ??= {};
  current.productSymbols[venue] = [...new Set([...(current.productSymbols[venue] ?? []), symbol])];
  output.set(asset, current);
}

function mergeProducts(left?: ProductAsset, right?: ProductAsset): ProductAsset | undefined {
  if (!left) return right ? structuredClone(right) : undefined;
  if (!right) return structuredClone(left);
  const productSymbols: ProductAsset["productSymbols"] = {};
  for (const venue of ["spot", "usdm-futures"] as const) {
    productSymbols[venue] = [...new Set([
      ...(left.productSymbols?.[venue] ?? []),
      ...(right.productSymbols?.[venue] ?? []),
    ])];
  }
  return {
    baseAsset: left.baseAsset || right.baseAsset,
    priceVenue: left.priceVenue ?? right.priceVenue,
    products: [...new Set([...(left.products ?? []), ...(right.products ?? [])])],
    productSymbols,
  };
}

function marketsFor(asset?: ProductAsset): Market[] {
  if (!asset) return [];
  const markets: Market[] = [];
  for (const venue of ["spot", "usdm-futures"] as const) {
    const symbols = asset.productSymbols?.[venue] ?? [];
    const symbol = symbols.find((item) => item === `${asset.baseAsset}USDT`)
      ?? symbols.find((item) => item.endsWith("USDT"));
    if (symbol) markets.push({ venue, symbol });
  }
  return markets;
}

async function mapConcurrent<T, R>(
  values: readonly T[],
  concurrency: number,
  worker: (value: T, index: number) => Promise<R>,
): Promise<R[]> {
  const output = new Array<R>(values.length);
  let next = 0;
  await Promise.all(Array.from({ length: Math.min(concurrency, values.length) }, async () => {
    while (true) {
      const index = next++;
      if (index >= values.length) return;
      output[index] = await worker(values[index]!, index);
    }
  }));
  return output;
}

function safe(value: string): string {
  return Buffer.from(value).toString("base64url");
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function resolve(file: string): string {
  return path.resolve(ROOT, file);
}

function relative(file: string): string {
  return path.relative(ROOT, file).replaceAll("\\", "/");
}

const invoked = process.argv[1] ? path.resolve(process.argv[1]) : undefined;
if (invoked === path.resolve(import.meta.filename)) {
  run().catch((error: unknown) => {
    console.error(error instanceof Error ? error.stack ?? error.message : error);
    process.exitCode = 1;
  });
}
