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
const COLUMNS = [
  "previousReturn1s", "priorReturn1s", "previousReturnZero",
  "activeCount10s", "activeCount60s", "zeroRunAge",
  "realizedVolatility5s", "realizedVolatility15s", "realizedVolatility60s",
  "rsi2s", "emaAcceleration2s1s", "emaSlope8s8s",
  "range1s", "closeLocation1s", "haarContrast16s",
  "takerQuoteImbalance1s", "takerBaseImbalance1s",
  "takerQuoteImbalanceEma2s", "takerQuoteImbalanceEma8s",
  "logQuoteVolume1s", "logTradeCount1s", "takerVwapGapBps", "observed",
] as const;

interface AxisMarket { venue: "spot" | "usdm-futures"; symbol: string }
interface AxisAsset { asset: string; markets: AxisMarket[] }

class FastState {
  readonly returns = new Float64Array(60);
  readonly ema8History = new Float64Array(9);
  seen = 0;
  previousClose = Number.NaN;
  previousReturn = 0;
  zeroRun = 0;
  gainEma = 0;
  lossEma = 0;
  closeEma2 = Number.NaN;
  closeEma8 = Number.NaN;
  imbalanceEma2 = Number.NaN;
  imbalanceEma8 = Number.NaN;
  previousSlope2 = Number.NaN;
  lastTime = Number.NaN;

  reset(): void {
    this.returns.fill(0);
    this.ema8History.fill(0);
    this.seen = 0;
    this.previousClose = Number.NaN;
    this.previousReturn = 0;
    this.zeroRun = 0;
    this.gainEma = 0;
    this.lossEma = 0;
    this.closeEma2 = Number.NaN;
    this.closeEma8 = Number.NaN;
    this.imbalanceEma2 = Number.NaN;
    this.imbalanceEma8 = Number.NaN;
    this.previousSlope2 = Number.NaN;
  }

  update(time: number, row: string[]): number[] {
    if (Number.isFinite(this.lastTime) && time !== this.lastTime + 1_000) this.reset();
    this.lastTime = time;
    const open = Number(row[1]);
    const high = Number(row[2]);
    const low = Number(row[3]);
    const close = Number(row[4]);
    const baseVolume = Number(row[5]);
    const quoteVolume = Number(row[7]);
    const tradeCount = Number(row[8]);
    const takerBuyBase = Number(row[9]);
    const takerBuyQuote = Number(row[10]);
    const priorReturn = this.previousReturn;
    const currentReturn = Number.isFinite(this.previousClose)
      ? Math.log(close / this.previousClose) * 10_000 : 0;
    this.returns[this.seen % this.returns.length] = currentReturn;
    this.seen += 1;
    this.previousClose = close;
    this.previousReturn = currentReturn;
    this.zeroRun = currentReturn === 0 ? this.zeroRun + 1 : 0;
    const gain = Math.max(currentReturn, 0);
    const loss = Math.max(-currentReturn, 0);
    this.gainEma += 2 / 3 * (gain - this.gainEma);
    this.lossEma += 2 / 3 * (loss - this.lossEma);
    const priorEma2 = this.closeEma2;
    this.closeEma2 = Number.isFinite(this.closeEma2) ? this.closeEma2 + 2 / 3 * (close - this.closeEma2) : close;
    this.closeEma8 = Number.isFinite(this.closeEma8) ? this.closeEma8 + 2 / 9 * (close - this.closeEma8) : close;
    const logEma2 = Math.log(this.closeEma2) * 10_000;
    const priorLogEma2 = Number.isFinite(priorEma2) ? Math.log(priorEma2) * 10_000 : logEma2;
    const slope2 = logEma2 - priorLogEma2;
    const acceleration2 = Number.isFinite(this.previousSlope2) ? slope2 - this.previousSlope2 : 0;
    this.previousSlope2 = slope2;
    const ema8Index = (this.seen - 1) % this.ema8History.length;
    const previousEma8 = this.seen > 8 ? this.ema8History[(this.seen - 9) % this.ema8History.length]! : Math.log(this.closeEma8) * 10_000;
    const logEma8 = Math.log(this.closeEma8) * 10_000;
    this.ema8History[ema8Index] = logEma8;
    const quoteImbalance = quoteVolume > 0 ? (2 * takerBuyQuote - quoteVolume) / quoteVolume : 0;
    const baseImbalance = baseVolume > 0 ? (2 * takerBuyBase - baseVolume) / baseVolume : 0;
    this.imbalanceEma2 = Number.isFinite(this.imbalanceEma2) ? this.imbalanceEma2 + 2 / 3 * (quoteImbalance - this.imbalanceEma2) : quoteImbalance;
    this.imbalanceEma8 = Number.isFinite(this.imbalanceEma8) ? this.imbalanceEma8 + 2 / 9 * (quoteImbalance - this.imbalanceEma8) : quoteImbalance;
    const buyVwap = takerBuyBase > 0 ? takerBuyQuote / takerBuyBase : Number.NaN;
    const sellBase = baseVolume - takerBuyBase;
    const sellQuote = quoteVolume - takerBuyQuote;
    const sellVwap = sellBase > 0 ? sellQuote / sellBase : Number.NaN;
    const vwapGap = buyVwap > 0 && sellVwap > 0 ? Math.log(buyVwap / sellVwap) * 10_000 : 0;
    const range = high > 0 && low > 0 ? Math.log(high / low) * 10_000 : 0;
    const closeLocation = high > low ? (2 * close - high - low) / (high - low) : 0;
    const rsi = this.gainEma + this.lossEma > 0 ? this.gainEma / (this.gainEma + this.lossEma) : 0.5;
    return [
      currentReturn, priorReturn, currentReturn === 0 ? 1 : 0,
      this.active(10), this.active(60), Math.log1p(this.zeroRun),
      this.volatility(5), this.volatility(15), this.volatility(60),
      rsi, acceleration2, logEma8 - previousEma8,
      range, closeLocation, this.haar(16),
      quoteImbalance, baseImbalance, this.imbalanceEma2, this.imbalanceEma8,
      Math.log1p(quoteVolume), Math.log1p(tradeCount), vwapGap,
    ];
  }

  active(window: number): number {
    const count = Math.min(window, this.seen, this.returns.length);
    let active = 0;
    for (let offset = 0; offset < count; offset += 1) active += this.returns[(this.seen - 1 - offset + 60) % 60] !== 0 ? 1 : 0;
    return active;
  }

  volatility(window: number): number {
    const count = Math.min(window, this.seen, this.returns.length);
    let square = 0;
    for (let offset = 0; offset < count; offset += 1) {
      const value = this.returns[(this.seen - 1 - offset + 60) % 60]!;
      square += value * value;
    }
    return Math.sqrt(square);
  }

  haar(window: number): number {
    if (this.seen < window) return 0;
    let recent = 0;
    let prior = 0;
    let square = 0;
    for (let offset = 0; offset < window; offset += 1) {
      const value = this.returns[(this.seen - 1 - offset + 60) % 60]!;
      if (offset < window / 2) recent += value;
      else prior += value;
      square += value * value;
    }
    return square > 0 ? (recent - prior) / Math.sqrt(square) : 0;
  }
}

export async function run(args = process.argv.slice(2)): Promise<void> {
  const value = (name: string) => {
    const index = args.indexOf(name);
    return index < 0 ? undefined : args[index + 1];
  };
  if (args.includes("--help")) {
    console.log(`Usage: npm run analysis:cross-asset-fast:export -- [options]

  --axis-dir DIR                  Completed compact cross-asset minute axis
  --concurrency N                 Concurrent spot markets (default 4)
  --limit N                       Optional smoke-test market limit

Downloads checksum-verified Binance spot 1s klines, computes the fast BTC-family
coordinates online, and retains only one causal feature row per minute.`);
    return;
  }
  const axisDir = resolve(value("--axis-dir") ?? DEFAULT_AXIS);
  const concurrency = Number(value("--concurrency") ?? 4);
  const limit = value("--limit") ? Number(value("--limit")) : undefined;
  const manifest = JSON.parse(await fs.readFile(path.join(axisDir, "manifest.json"), "utf8"));
  let assets = (manifest.assets as AxisAsset[])
    .map((asset) => ({ ...asset, market: asset.markets.find((market) => market.venue === "spot") }))
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
      if (completed % 5 === 0 || completed === assets.length) console.error(`${completed}/${assets.length} fast spot assets exported`);
    }
  });
  const output = {
    version: 1,
    generatedAt: new Date().toISOString(),
    sourceAxis: relative(path.join(axisDir, "manifest.json")),
    source: "data.binance.vision spot/daily/klines/1s",
    window: { start: new Date(START).toISOString(), endExclusive: new Date(END).toISOString(), rows: ROWS, stepMs: MINUTE_MS },
    columns: COLUMNS,
    requestedAssets: assets.length,
    completedAtLeast95Percent: results.filter((row) => row.coverage >= 0.95).length,
    sourceBytes: results.reduce((sum, row) => sum + row.sourceBytes, 0),
    assets: results,
  };
  await fs.writeFile(path.join(axisDir, "fast-manifest.json"), `${JSON.stringify(output, null, 2)}\n`);
  console.log(`Wrote ${relative(path.join(axisDir, "fast-manifest.json"))}`);
}

async function exportAsset(axisDir: string, asset: string, symbol: string): Promise<{ file: string; observedRows: number; coverage: number; sourceBytes: number }> {
  const directory = path.join(axisDir, "assets", safe(asset));
  const file = path.join(directory, "spot-fast.f32");
  const sidecar = path.join(directory, "spot-fast.json");
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
  const state = new FastState();
  let sourceBytes = 0;
  for (let day = START; day < END; day += DAY_MS) {
    const date = new Date(day).toISOString().slice(0, 10);
    const archiveName = `${symbol}-1s-${date}.zip`;
    const url = `https://data.binance.vision/data/spot/daily/klines/${encodeURIComponent(symbol)}/1s/${archiveName}`;
    try {
      const archive = await downloadVerified(url);
      sourceBytes += archive.byteLength;
      parseArchive(archive, `${symbol}-1s-${date}.csv`, state, values);
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

function parseArchive(archive: Buffer, expectedCsv: string, state: FastState, output: Float32Array): void {
  const entries = new AdmZip(archive).getEntries().filter((entry) => !entry.isDirectory);
  if (entries.length !== 1 || entries[0]!.entryName !== expectedCsv) throw new Error(`expected one ZIP entry named ${expectedCsv}`);
  const lines = entries[0]!.getData().toString("utf8").replace(/^\ufeff/, "").split(/\r?\n/);
  for (const line of lines) {
    if (!line || /^[A-Za-z_]/.test(line)) continue;
    const row = line.split(",");
    if (row.length < 11) continue;
    let time = Number(row[0]);
    if (time > 100_000_000_000_000) time = Math.floor(time / 1_000);
    if (time < START || time >= END || time % 1_000 !== 0) continue;
    const features = state.update(time, row);
    if (time % MINUTE_MS !== 59_000 || state.seen < 60) continue;
    const minute = Math.floor((time - START) / MINUTE_MS);
    const offset = minute * COLUMNS.length;
    for (let column = 0; column < features.length; column += 1) output[offset + column] = features[column]!;
    output[offset + COLUMNS.length - 1] = 1;
  }
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
