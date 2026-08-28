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
const MINUTE_MS = 60_000;
const ROWS = (END - START) / MINUTE_MS;
const BANDS = [-5, -4, -3, -2, -1, -0.2, 0.2, 1, 2, 3, 4, 5] as const;
const bandLabel = (band: number) => `${band > 0 ? "Plus" : "Minus"}${String(Math.abs(band)).replace(".", "p")}pct`;
const COLUMNS = [
  ...BANDS.map((band) => `depth${bandLabel(band)}`),
  ...BANDS.map((band) => `notional${bandLabel(band)}`),
  "observed",
];
const HEADER = "timestamp,percentage,depth,notional";

interface AxisMarket { venue: "spot" | "usdm-futures"; symbol: string }
interface AxisAsset { asset: string; markets: AxisMarket[] }

export async function run(args = process.argv.slice(2)): Promise<void> {
  const value = (name: string) => {
    const index = args.indexOf(name);
    return index < 0 ? undefined : args[index + 1];
  };
  if (args.includes("--help")) {
    console.log(`Usage: npm run analysis:cross-asset-book-depth:export -- [options]

  --axis-dir DIR                  Completed compact cross-asset minute axis
  --concurrency N                 Concurrent markets (default 8)
  --limit N                       Optional smoke-test market limit

Downloads checksum-verified Binance USD-M percentage-depth snapshots and retains
only the latest complete 12-band snapshot available inside each completed minute.`);
    return;
  }
  const axisDir = resolve(value("--axis-dir") ?? DEFAULT_AXIS);
  const concurrency = Number(value("--concurrency") ?? 8);
  const limit = value("--limit") ? Number(value("--limit")) : undefined;
  const manifest = JSON.parse(await fs.readFile(path.join(axisDir, "manifest.json"), "utf8"));
  let assets = (manifest.assets as AxisAsset[])
    .map((asset) => ({ ...asset, market: asset.markets.find((market) => market.venue === "usdm-futures") }))
    .filter((asset) => asset.market);
  if (limit) assets = assets.slice(0, limit);
  let completed = 0;
  const results = await mapConcurrent(assets, concurrency, async (asset) => {
    try {
      return { asset: asset.asset, symbol: asset.market!.symbol, ...(await exportAsset(axisDir, asset.asset, asset.market!.symbol)), error: null };
    } catch (error) {
      return { asset: asset.asset, symbol: asset.market!.symbol, file: null, observedRows: 0, coverage: 0, sourceBytes: 0, error: message(error) };
    } finally {
      completed += 1;
      if (completed % 10 === 0 || completed === assets.length) console.error(`${completed}/${assets.length} book-depth assets exported`);
    }
  });
  const output = {
    version: 1,
    generatedAt: new Date().toISOString(),
    sourceAxis: relative(path.join(axisDir, "manifest.json")),
    source: "data.binance.vision futures/um/daily/bookDepth",
    window: { start: new Date(START).toISOString(), endExclusive: new Date(END).toISOString(), rows: ROWS, stepMs: MINUTE_MS },
    columns: COLUMNS,
    requestedAssets: assets.length,
    completedAtLeast95Percent: results.filter((row) => row.coverage >= 0.95).length,
    sourceBytes: results.reduce((sum, row) => sum + row.sourceBytes, 0),
    assets: results,
  };
  await fs.writeFile(path.join(axisDir, "book-depth-manifest.json"), `${JSON.stringify(output, null, 2)}\n`);
  console.log(`Wrote ${relative(path.join(axisDir, "book-depth-manifest.json"))}`);
}

async function exportAsset(axisDir: string, asset: string, symbol: string): Promise<{ file: string; observedRows: number; coverage: number; sourceBytes: number }> {
  const directory = path.join(axisDir, "assets", safe(asset));
  const file = path.join(directory, "usdm-book-depth.f32");
  const sidecar = path.join(directory, "usdm-book-depth.json");
  const expectedBytes = ROWS * COLUMNS.length * 4;
  try {
    const [stat, cached] = await Promise.all([fs.stat(file), fs.readFile(sidecar, "utf8").then(JSON.parse)]);
    if (stat.size === expectedBytes && cached.symbol === symbol) {
      return { file: relative(file), observedRows: cached.observedRows, coverage: cached.coverage, sourceBytes: cached.sourceBytes };
    }
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
  }
  const values = new Float32Array(ROWS * COLUMNS.length);
  values.fill(Number.NaN);
  for (let row = 0; row < ROWS; row += 1) values[row * COLUMNS.length + COLUMNS.length - 1] = 0;
  let sourceBytes = 0;
  for (let day = START; day < END; day += DAY_MS) {
    const date = new Date(day).toISOString().slice(0, 10);
    const archiveName = `${symbol}-bookDepth-${date}.zip`;
    const url = `https://data.binance.vision/data/futures/um/daily/bookDepth/${encodeURIComponent(symbol)}/${archiveName}`;
    try {
      const archive = await downloadVerified(url);
      sourceBytes += archive.byteLength;
      parseArchive(archive, `${symbol}-bookDepth-${date}.csv`, values);
    } catch (error) {
      if (/HTTP 404/.test(message(error))) continue;
      throw error;
    }
  }
  let observedRows = 0;
  for (let row = 0; row < ROWS; row += 1) observedRows += values[row * COLUMNS.length + COLUMNS.length - 1] === 1 ? 1 : 0;
  const coverage = observedRows / ROWS;
  await fs.mkdir(directory, { recursive: true });
  await fs.writeFile(file, Buffer.from(values.buffer));
  await fs.writeFile(sidecar, `${JSON.stringify({ version: 1, symbol, rows: ROWS, columns: COLUMNS, observedRows, coverage, sourceBytes }, null, 2)}\n`);
  return { file: relative(file), observedRows, coverage, sourceBytes };
}

function parseArchive(archive: Buffer, expectedCsv: string, output: Float32Array): void {
  const entries = new AdmZip(archive).getEntries().filter((entry) => !entry.isDirectory);
  if (entries.length !== 1 || entries[0]!.entryName !== expectedCsv) throw new Error(`expected one ZIP entry named ${expectedCsv}`);
  const lines = entries[0]!.getData().toString("utf8").replace(/^\ufeff/, "").trimEnd().split(/\r?\n/);
  if (lines[0]?.trim() !== HEADER) throw new Error(`${expectedCsv}: schema changed`);
  let timestamp = "";
  let currentMinute = -1;
  let depth = new Float64Array(BANDS.length);
  let notional = new Float64Array(BANDS.length);
  depth.fill(Number.NaN);
  notional.fill(Number.NaN);
  const flush = () => {
    if (currentMinute < 0
      || depth.some((value) => !Number.isFinite(value) || value <= 0)
      || notional.some((value) => !Number.isFinite(value) || value <= 0)) return;
    const offset = currentMinute * COLUMNS.length;
    for (let index = 0; index < BANDS.length; index += 1) {
      output[offset + index] = depth[index]!;
      output[offset + BANDS.length + index] = notional[index]!;
    }
    output[offset + COLUMNS.length - 1] = 1;
  };
  for (const line of lines.slice(1)) {
    const row = line.split(",");
    if (row.length !== 4) continue;
    if (row[0] !== timestamp) {
      flush();
      timestamp = row[0]!;
      depth = new Float64Array(BANDS.length);
      notional = new Float64Array(BANDS.length);
      depth.fill(Number.NaN);
      notional.fill(Number.NaN);
      const time = Date.parse(`${timestamp.replace(" ", "T")}Z`);
      currentMinute = time >= START && time < END ? Math.floor((time - START) / MINUTE_MS) : -1;
    }
    const band = Number(row[1]);
    const index = BANDS.indexOf(band as typeof BANDS[number]);
    const depthValue = Number(row[2]);
    const notionalValue = Number(row[3]);
    if (index >= 0 && Number.isFinite(depthValue) && depthValue > 0
      && Number.isFinite(notionalValue) && notionalValue > 0) {
      depth[index] = depthValue;
      notional[index] = notionalValue;
    }
  }
  flush();
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
