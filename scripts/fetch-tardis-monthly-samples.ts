import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";
import readline from "node:readline";
import { Readable } from "node:stream";
import { fileURLToPath } from "node:url";
import { createGunzip, gzipSync } from "node:zlib";
import {
  headerIndexes,
  parseLiquidationRow,
  parseQuoteRow,
  retainLastQuotePerSecond,
  type LiquidationEvent,
  type QuoteSecond,
} from "./lib/tardis-monthly-samples.ts";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const DEFAULT_OUTPUT = "data/market/mutable/external/tardis-monthly-samples";
const DEFAULT_DATES = [
  "2025-04-01", "2025-05-01", "2025-06-01", "2025-07-01", "2025-08-01",
  "2025-09-01", "2025-10-01", "2025-11-01", "2026-05-01", "2026-06-01",
];

interface SourceSpec {
  id: string;
  exchange: string;
  dataType: "quotes" | "liquidations";
  symbol: string;
  kind: "quote" | "liquidation";
}

const SOURCES: SourceSpec[] = [
  { id: "binance-spot-btcusdt", exchange: "binance", dataType: "quotes", symbol: "BTCUSDT", kind: "quote" },
  { id: "coinbase-spot-btcusd", exchange: "coinbase", dataType: "quotes", symbol: "BTC-USD", kind: "quote" },
  { id: "kraken-spot-xbtusd", exchange: "kraken", dataType: "quotes", symbol: "XBT-USD", kind: "quote" },
  { id: "deribit-btc-perpetual", exchange: "deribit", dataType: "quotes", symbol: "BTC-PERPETUAL", kind: "quote" },
  { id: "binance-usdm-liquidations", exchange: "binance-futures", dataType: "liquidations", symbol: "PERPETUALS", kind: "liquidation" },
  { id: "deribit-liquidations", exchange: "deribit", dataType: "liquidations", symbol: "PERPETUALS", kind: "liquidation" },
];

export async function run(args = process.argv.slice(2)) {
  const value = (name: string) => {
    const index = args.indexOf(name);
    return index < 0 ? undefined : args[index + 1];
  };
  if (args.includes("--help")) {
    console.log(`Usage: npm run fetch:tardis-samples -- [options]

  --output-dir PATH
  --dates YYYY-MM-DD,YYYY-MM-DD
  --sources source-id,source-id
  --force

Tardis makes the first UTC day of each month available without an API key.
Raw tick CSV is streamed through gzip and reduced to causal one-second quotes or
BTC liquidation events; the multi-gigabyte source files are never retained.`);
    return;
  }
  const output = path.resolve(repoRoot, value("--output-dir") ?? DEFAULT_OUTPUT);
  const dates = value("--dates")?.split(",").filter(Boolean) ?? DEFAULT_DATES;
  const requested = new Set(value("--sources")?.split(",").filter(Boolean) ?? SOURCES.map(({ id }) => id));
  const sources = SOURCES.filter(({ id }) => requested.has(id));
  const unknown = [...requested].filter((id) => !SOURCES.some((source) => source.id === id));
  if (unknown.length > 0) throw new Error(`Unknown source ids: ${unknown.join(", ")}`);
  await fsp.mkdir(output, { recursive: true });
  for (const date of dates) for (const source of sources) {
    const directory = path.join(output, source.id);
    const file = path.join(directory, `${date}.json.gz`);
    if (!args.includes("--force") && fs.existsSync(file)) {
      console.log(`skip ${source.id} ${date}`);
      continue;
    }
    await fsp.mkdir(directory, { recursive: true });
    const url = datasetUrl(source, date);
    console.log(`fetch ${source.id} ${date} ${url}`);
    const artifact = await aggregateSource(url, source, date);
    await fsp.writeFile(file, gzipSync(JSON.stringify(artifact), { level: 9 }));
    console.log(`wrote ${path.relative(repoRoot, file)} (${artifact.rows.length.toLocaleString()} rows)`);
  }
}

async function aggregateSource(url: URL, source: SourceSpec, date: string) {
  const response = await request(url);
  if (!response.body) throw new Error(`${url} returned no response body`);
  const compressed = Readable.fromWeb(response.body as never);
  const lines = readline.createInterface({ input: compressed.pipe(createGunzip()), crlfDelay: Infinity });
  let columns: Map<string, number> | undefined;
  let rawRows = 0;
  const quotes: QuoteSecond[] = [];
  const liquidations: LiquidationEvent[] = [];
  for await (const line of lines) {
    if (!columns) {
      columns = headerIndexes(line);
      continue;
    }
    rawRows += 1;
    if (source.kind === "quote") {
      const row = parseQuoteRow(line, columns);
      if (row) quotes.push(row);
    } else {
      const row = parseLiquidationRow(line, columns);
      if (row && isBitcoinLiquidation(source, row[4])) liquidations.push(row);
    }
  }
  const rows = source.kind === "quote" ? retainLastQuotePerSecond(quotes) : liquidations;
  return {
    version: 1,
    retrievedAt: new Date().toISOString(),
    source: "Tardis downloadable CSV monthly public sample",
    sourceUrl: url.toString(),
    exchange: source.exchange,
    dataType: source.dataType,
    symbol: source.symbol,
    date,
    cadence: source.kind === "quote" ? "last receive-time quote per UTC second" : "every BTC liquidation event",
    timing: "local_timestamp from the Tardis collector; no exchange-clock lookahead",
    rawRows,
    rows,
  };
}

async function request(url: URL, attempts = 4): Promise<Response> {
  let lastError: unknown;
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    try {
      const response = await fetch(url, { headers: { "user-agent": "trading-tardis-sample-research/1.0" } });
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      return response;
    } catch (error) {
      lastError = error;
      if (attempt + 1 < attempts) await new Promise((resolve) => setTimeout(resolve, 1_000 * 2 ** attempt));
    }
  }
  throw lastError;
}

function datasetUrl(source: SourceSpec, date: string) {
  const [year, month, day] = date.split("-");
  return new URL(`https://datasets.tardis.dev/v1/${source.exchange}/${source.dataType}/${year}/${month}/${day}/${source.symbol}.csv.gz`);
}

function isBitcoinLiquidation(source: SourceSpec, symbol: string) {
  return source.exchange === "binance-futures" ? symbol === "BTCUSDT" : symbol === "BTC-PERPETUAL" || symbol.startsWith("BTC-");
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  run().catch((error: unknown) => {
    console.error(error instanceof Error ? error.stack ?? error.message : error);
    process.exitCode = 1;
  });
}
