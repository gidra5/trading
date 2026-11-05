import fs from "node:fs/promises";
import { createReadStream } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createGunzip, gzipSync } from "node:zlib";
import { createInterface } from "node:readline";
import WebSocket from "ws";
import { rateClass } from "./lib/collector-storage.ts";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const value = (args: readonly string[], name: string, fallback: string) => { const index = args.indexOf(name); return index < 0 ? fallback : args[index + 1] ?? fallback; };
const json = async (url: string) => { const r = await fetch(url, { signal: AbortSignal.timeout(25_000) }); if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`); return r.json(); };

export async function measureLive(seconds: number, universeFile: string) {
  if (!(seconds >= 5 && seconds <= 300)) throw new Error("Live sample must be 5..300 seconds");
  const manifest = JSON.parse(await fs.readFile(universeFile, "utf8"));
  const [spotTicker, futuresTicker] = await Promise.all([
    json("https://api.binance.com/api/v3/ticker/24hr"), json("https://fapi.binance.com/fapi/v1/ticker/24hr"),
  ]) as Array<Array<{ symbol: string; count: number; quoteVolume: string; openTime: number; closeTime: number }>>;
  const tickers = { spot: spotTicker!, "usdm-futures": futuresTicker! };
  const markets = (manifest.entries as Array<{ baseAsset: string; productSymbols: Record<string, string[]> }>).flatMap((entry) =>
    Object.entries(entry.productSymbols).flatMap(([venue, symbols]) => symbols.map((symbol) => ({ baseAsset: entry.baseAsset, venue, symbol }))));
  const activity = markets.map((market) => {
    const ticker = tickers[market.venue as keyof typeof tickers]?.find((row) => row.symbol === market.symbol);
    const tickerSeconds = ticker ? (ticker.closeTime - ticker.openTime) / 1000 : null;
    const fillRate = ticker && tickerSeconds! > 0 ? ticker.count / tickerSeconds! : null;
    return { ...market, individualFills24h: ticker?.count ?? null, tickerSeconds, quoteVolume24h: ticker ? Number(ticker.quoteVolume) : null,
      fillsPerSecond24h: fillRate, rateClass: fillRate === null ? null : rateClass(fillRate) };
  });
  const majors = new Set(["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT"]);
  const chosen = new Map<string, typeof activity[number]>();
  for (const venue of ["spot", "usdm-futures"]) {
    const group = activity.filter((row) => row.venue === venue && row.fillsPerSecond24h !== null).sort((a, b) => a.fillsPerSecond24h! - b.fillsPerSecond24h!);
    for (const row of group.filter((row) => majors.has(row.symbol))) chosen.set(`${venue}:${row.symbol}`, row);
    for (const c of ["quiet", "medium", "busy", "very-busy"]) {
      const tier = group.filter((row) => row.rateClass === c);
      const row = tier[Math.floor(tier.length / 2)]; if (row) chosen.set(`${venue}:${row.symbol}`, row);
    }
    for (const symbol of ["XAUUSDT", "XAGUSDT", "CLUSDT", "BRENTUSDT", "SPXUSDT", "NDXUSDT", "EURUSDT", "TSLAUSDT"]) {
      const row = group.find((row) => row.symbol === symbol);
      if (row) chosen.set(`${venue}:${row.symbol}`, row);
      else {
        const ticker = tickers[venue as keyof typeof tickers]?.find((row) => row.symbol === symbol);
        if (ticker) {
          const tickerSeconds = (ticker.closeTime - ticker.openTime) / 1000, fillsPerSecond24h = ticker.count / tickerSeconds;
          chosen.set(`${venue}:${symbol}`, { venue, symbol, baseAsset: symbol.replace(/USDT$/, ""), individualFills24h: ticker.count, tickerSeconds,
            quoteVolume24h: Number(ticker.quoteVolume), fillsPerSecond24h, rateClass: rateClass(fillsPerSecond24h) });
        }
      }
    }
  }
  const sampleRows: unknown[] = [];
  await Promise.all([...chosen.values()].map(async (market) => {
    const streams = market.venue === "spot"
      ? [`${market.symbol.toLowerCase()}@trade`, `${market.symbol.toLowerCase()}@depth@100ms`, `${market.symbol.toLowerCase()}@bookTicker`]
      : [`${market.symbol.toLowerCase()}@depth@100ms`, `${market.symbol.toLowerCase()}@bookTicker`];
    await capture(market, market.venue === "spot" ? "wss://stream.binance.com:9443/stream?streams=" : "wss://fstream.binance.com/public/stream?streams=", streams);
  }));
  // Futures trades use the separate market-stream route.
  await Promise.all([...chosen.values()].filter((row) => row.venue === "usdm-futures").map((market) =>
    capture(market, "wss://fstream.binance.com/market/stream?streams=", [`${market.symbol.toLowerCase()}@aggTrade`])));
  return { version: 1, measuredAt: new Date().toISOString(), secondsRequested: seconds, universeFile: path.relative(ROOT, universeFile),
    tickerSemantics: "24h count is individual fills. Futures aggregate-message rates use sampled fill/message ratio; it is not count/86400.", activity, samples: sampleRows };

  async function capture(market: typeof activity[number], prefix: string, streams: string[]) {
    const stats = new Map<string, { count: number; rawBytes: number; lines: string[]; fills: number; levels: number }>();
    let openedAt = 0, endedAt = 0, failure: string | null = null;
    await new Promise<void>((resolve) => {
      const socket = new WebSocket(prefix + streams.join("/"));
      const timeout = setTimeout(() => { endedAt = Date.now(); socket.terminate(); resolve(); }, (seconds + 15) * 1000);
      socket.on("open", () => { openedAt = Date.now(); setTimeout(() => { endedAt = Date.now(); socket.close(); setTimeout(() => socket.terminate(), 1000).unref(); }, seconds * 1000).unref(); });
      socket.on("message", (bytes) => {
        try {
          const parsed = JSON.parse(bytes.toString()); const payload = parsed.data ?? parsed;
          const stream = parsed.stream ?? "unknown";
          const row = stats.get(stream) ?? { count: 0, rawBytes: 0, lines: [], fills: 0, levels: 0 };
          const line = `${JSON.stringify({ recordedAt: Date.now(), payload: { receivedAt: Date.now(), monotonicNs: process.hrtime.bigint().toString(), message: payload } })}\n`;
          row.count++; row.rawBytes += Buffer.byteLength(line); row.lines.push(line);
          row.fills += payload.e === "aggTrade" ? Math.max(1, Number(payload.l) - Number(payload.f) + 1) : payload.e === "trade" ? 1 : 0;
          row.levels += (payload.b?.length ?? 0) + (payload.a?.length ?? 0);
          stats.set(stream, row);
        } catch (error) { failure = String(error); }
      });
      socket.on("error", (error) => { failure = error.message; });
      socket.on("close", () => { endedAt ||= Date.now(); clearTimeout(timeout); resolve(); });
    });
    const observedSeconds = openedAt ? Math.max(0, (endedAt - openedAt) / 1000) : 0;
    for (const stream of streams) {
      const row = stats.get(stream); const lines = row?.lines ?? [];
      const compressed1s = compressBatches(lines, 1000), compressed60s = compressBatches(lines, 60_000);
      sampleRows.push({ ...market, stream, observedSeconds, failure, messages: row?.count ?? 0,
        messageRate: observedSeconds ? (row?.count ?? 0) / observedSeconds : null, rawBytes: row?.rawBytes ?? 0,
        gzipOneSecondBytes: compressed1s, gzipSixtySecondBytes: compressed60s, individualFills: row?.fills ?? 0,
        changedLevels: row?.levels ?? 0, gzipLevel: 6, compressionPolicy: "synthetic receive-time 1s/60s gzip members, envelope matches raw collector fields" });
    }
  }
}

export function compressBatches(lines: readonly string[], intervalMs: number): number {
  const batches = new Map<number, string[]>();
  for (const line of lines) {
    const bucket = Math.floor(JSON.parse(line).recordedAt / intervalMs);
    const rows = batches.get(bucket) ?? []; rows.push(line); batches.set(bucket, rows);
  }
  return [...batches.values()].reduce((sum, rows) => sum + gzipSync(rows.join(""), { level: 6 }).length, 0);
}

export async function auditArchives(directory: string) {
  const sources = (await fs.readdir(directory, { withFileTypes: true })).filter((row) => row.isDirectory() && !row.name.endsWith("-session") && row.name !== "collector-session");
  const measurements = [];
  for (const source of sources) {
    const sourceDir = path.join(directory, source.name);
    const files = await fs.readdir(sourceDir, { withFileTypes: true });
    const candidates = await Promise.all(files.filter((row) => row.isFile() && row.name.endsWith(".jsonl.gz")).map(async (row) => ({ file: path.join(sourceDir, row.name), bytes: (await fs.stat(path.join(sourceDir, row.name))).size })));
    candidates.sort((a, b) => b.bytes - a.bytes);
    const sample = candidates[0]; if (!sample) continue;
    let count = 0, rawBytes = 0, firstAt = 0, lastAt = 0, clippedMs = 0, gaps = 0, invalid = 0;
    const periodic = /option-summary|mempool|premium-index/.test(source.name) ? 60_000 : source.name === "gdelt-crypto-news" ? 900_000 : source.name === "cross-exchange-book-1s" || source.name === "binance-usdm-mark-price" ? 1000 : 0;
    const gapLimit = periodic ? periodic * 2.5 : 60_000;
    for await (const line of createInterface({ input: createReadStream(sample.file).pipe(createGunzip()), crlfDelay: Infinity })) {
      if (!line) continue; rawBytes += Buffer.byteLength(line) + 1;
      try {
        const row = JSON.parse(line); const at = Number(row.recordedAt);
        if (!Number.isFinite(at)) { invalid++; continue; }
        if (!firstAt) firstAt = at;
        if (lastAt && at >= lastAt) { const delta = at - lastAt; if (delta <= gapLimit) clippedMs += delta; else gaps++; }
        else if (lastAt) invalid++;
        lastAt = at; count++;
      } catch { invalid++; }
    }
    const observedSeconds = (clippedMs + (count ? periodic : 0)) / 1000;
    measurements.push({ source: source.name, sampleFile: path.relative(ROOT, sample.file).replaceAll("\\", "/"),
      selection: "largest archived JSONL gzip file per source; representative active-day sample, not all history", filesAvailable: candidates.length,
      compressedBytes: sample.bytes, uncompressedBytes: rawBytes, rows: count, invalidRows: invalid,
      firstAt: new Date(firstAt).toISOString(), lastAt: new Date(lastAt).toISOString(), calendarSpanSeconds: (lastAt - firstAt) / 1000,
      observedSeconds, excludedLongGaps: gaps, coverageMethod: `receive-time gap heuristic excludes entire deltas >${gapLimit / 1000}s; not certified connectivity or packet completeness`,
      compressedBytesPerRow: count ? sample.bytes / count : null, uncompressedBytesPerRow: count ? rawBytes / count : null,
      compressedBytesPerObservedSecond: observedSeconds ? sample.bytes / observedSeconds : null, rowsPerObservedSecond: observedSeconds ? count / observedSeconds : null });
    console.log(`${source.name}: ${count} rows, ${(observedSeconds / 3600).toFixed(2)} observed h, ${(sample.bytes / Math.max(1, count)).toFixed(1)} compressed B/row`);
  }
  return { version: 1, measuredAt: new Date().toISOString(), root: path.relative(ROOT, directory).replaceAll("\\", "/"), measurements };
}

export async function measureRest(universeFile: string) {
  const universe = JSON.parse(await fs.readFile(universeFile, "utf8"));
  const tickers = await Promise.all([json("https://api.binance.com/api/v3/ticker/24hr"), json("https://fapi.binance.com/fapi/v1/ticker/24hr")]);
  const snapshots: unknown[] = [];
  for (const venue of ["spot", "usdm-futures"]) {
    const ticker = tickers[venue === "spot" ? 0 : 1] as Array<{ symbol: string; count: number }>;
    const symbols = new Set<string>(["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT"]);
    for (const c of ["quiet", "medium", "busy", "very-busy"]) {
      const candidates = (universe.entries as Array<{ productSymbols: Record<string, string[]> }>).flatMap((entry) => entry.productSymbols[venue] ?? [])
        .filter((symbol) => { const row = ticker.find((row) => row.symbol === symbol); return row && rateClass(row.count / 86_400) === c; });
      const symbol = candidates[Math.floor(candidates.length / 2)]; if (symbol) symbols.add(symbol);
    }
    for (const symbol of ["XAUUSDT", "XAGUSDT", "CLUSDT", "BRENTUSDT", "SPXUSDT", "NDXUSDT", "EURUSDT", "TSLAUSDT"]) {
      if (ticker.some((row) => row.symbol === symbol)) symbols.add(symbol);
    }
    for (const symbol of symbols) {
      const limit = venue === "spot" ? 5000 : 1000;
      const url = venue === "spot" ? `https://api.binance.com/api/v3/depth?symbol=${symbol}&limit=${limit}` : `https://fapi.binance.com/fapi/v1/depth?symbol=${symbol}&limit=${limit}`;
      try {
        const payload = await json(url) as { bids: unknown[]; asks: unknown[] };
        const data = `${JSON.stringify({ recordedAt: Date.now(), payload })}\n`;
        snapshots.push({ venue, symbol, limitPerSide: limit, levels: payload.bids.length + payload.asks.length,
          uncompressedBytes: Buffer.byteLength(data), gzipBytes: gzipSync(data, { level: 6 }).length, source: url });
      } catch (error) { snapshots.push({ venue, symbol, error: String(error) }); }
    }
  }
  return { version: 1, measuredAt: new Date().toISOString(), universeFile: path.relative(ROOT, universeFile),
    tickers: { spot: tickers[0], "usdm-futures": tickers[1] }, snapshots };
}

export async function run(args = process.argv.slice(2)) {
  if (args.includes("--help")) {
    console.log(`Bounded public measurements for the storage estimator.
  --sample-seconds N       Capture public WebSockets for 5..300 seconds (default 60)
  --rest                   Measure public 24h activity and representative REST depth seeds
  --audit-archives         Read existing gzip archives; no network
  --universe FILE          Default docs/portfolio/restored-market-universe.json
  --archive-dir DIR        Default data/market/mutable/external-live
  --output FILE            Defaults to distinct live/rest/archive-measurements-2026-11-01.json under data/research/collector-storage
No persistent collector is started. --help opens no connections.`);
    return;
  }
  const mode = args.includes("--audit-archives") ? "archive" : args.includes("--rest") ? "rest" : "live";
  const output = path.resolve(ROOT, value(args, "--output", `data/research/collector-storage/${mode}-measurements-2026-11-01.json`));
  const result = args.includes("--audit-archives") ? await auditArchives(path.resolve(ROOT, value(args, "--archive-dir", "data/market/mutable/external-live")))
    : args.includes("--rest") ? await measureRest(path.resolve(ROOT, value(args, "--universe", "docs/portfolio/restored-market-universe.json")))
      : await measureLive(Number(value(args, "--sample-seconds", "60")), path.resolve(ROOT, value(args, "--universe", "docs/portfolio/restored-market-universe.json")));
  await fs.mkdir(path.dirname(output), { recursive: true }); await fs.writeFile(output, JSON.stringify(result, null, 2) + "\n");
  console.log(output);
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) run().catch((error) => { console.error(error); process.exitCode = 1; });
