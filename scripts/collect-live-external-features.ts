import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { gunzipSync, gzipSync, gzip } from "node:zlib";
import { promisify } from "node:util";
import AdmZip from "adm-zip";
import WebSocket from "ws";
import { deriveDeribitOptionSummary, type DeribitBookSummary } from "./lib/external-public-data.ts";
import { summarizeGdeltGkg, summarizeGdeltNgrams } from "./lib/external-live-data.ts";
import { LiveBookCompactor } from "./lib/live-book-compactor.ts";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const DEFAULT_OUTPUT = "data/market/mutable/external-live";
const USER_AGENT = "trading-live-external-collector/1.0";
const gzipBuffer = promisify(gzip);

interface WebSocketSpec {
  id: string;
  url: string;
  subscribe?: unknown;
  onOpen?: () => Promise<void>;
  minRecordIntervalMs?: number;
  staleAfterMs?: number;
}

const COMPACTED_BOOK_SOURCES = new Set([
  "binance-spot-depth-diff",
  "binance-usdm-book-ticker",
  "coinbase-btcusd-level2",
  "kraken-btcusd-book",
  "deribit-btc-perpetual-book",
]);

class RotatingJsonlSink {
  private streams = new Map<string, {
    date: string;
    file: string;
    chunks: string[];
    bytes: number;
    completed: Promise<void>;
  }>();
  private pendingWrites = new Set<Promise<void>>();
  private writtenBytesBySource = new Map<string, number>();
  private readonly flushTimer: NodeJS.Timeout;

  constructor(private readonly root: string) {
    this.flushTimer = setInterval(() => {
      for (const source of this.streams.keys()) void this.flush(source);
    }, 1_000);
  }

  write(source: string, payload: unknown, recordedAt = Date.now()) {
    const date = new Date(recordedAt).toISOString().slice(0, 10);
    let current = this.streams.get(source);
    if (!current || current.date !== date) {
      if (current) void this.flushState(current, source);
      const directory = path.join(this.root, source);
      fs.mkdirSync(directory, { recursive: true });
      current = {
        date,
        file: path.join(directory, `${date}.jsonl.gz`),
        chunks: [],
        bytes: 0,
        completed: Promise.resolve(),
      };
      this.streams.set(source, current);
    }
    const line = `${JSON.stringify({ recordedAt, payload })}\n`;
    current.chunks.push(line);
    current.bytes += Buffer.byteLength(line);
    if (current.bytes >= 256 * 1_024) void this.flush(source);
  }

  async close() {
    clearInterval(this.flushTimer);
    for (const [source, stream] of this.streams) await this.flushState(stream, source);
    while (this.pendingWrites.size > 0) await Promise.all(this.pendingWrites);
    this.streams.clear();
  }

  bytesWritten() {
    return Object.fromEntries(this.writtenBytesBySource);
  }

  private flush(source: string) {
    const current = this.streams.get(source);
    return current ? this.flushState(current, source) : Promise.resolve();
  }

  private flushState(current: {
    file: string;
    chunks: string[];
    bytes: number;
    completed: Promise<void>;
  }, source: string) {
    if (current.chunks.length === 0) return current.completed;
    const contents = current.chunks.join("");
    current.chunks = [];
    current.bytes = 0;
    const write = current.completed.then(async () => {
      const compressed = await gzipBuffer(contents, { level: 6 });
      await fsp.appendFile(current.file, compressed);
      this.writtenBytesBySource.set(source, (this.writtenBytesBySource.get(source) ?? 0) + compressed.length);
    }).catch((error: unknown) => {
      console.error(`${source} compressed sink error: ${error instanceof Error ? error.message : error}`);
    });
    current.completed = write;
    this.pendingWrites.add(write);
    void write.then(
      () => this.pendingWrites.delete(write),
      () => this.pendingWrites.delete(write),
    );
    return write;
  }
}

export async function run(args = process.argv.slice(2)) {
  const value = (name: string) => {
    const index = args.indexOf(name);
    return index < 0 ? undefined : args[index + 1];
  };
  if (args.includes("--help")) {
    console.log(`Usage: npm run collect:external-live -- [options]

  --output-dir data/market/mutable/external-live
  --duration-seconds N       Stop after N seconds; omit to run indefinitely
  --rest-only                Skip WebSocket streams
  --raw-books               Also retain full high-frequency order-book messages
  --status-file FILE         Atomically write collector health every 5 seconds

Optional environment variables:
  TRADING_ECONOMICS_API_KEY  Point-in-time macro calendar actual/forecast feed
  BLOCKWORKS_API_KEY         Daily ETF flow feed`);
    return;
  }
  const output = path.resolve(repoRoot, value("--output-dir") ?? DEFAULT_OUTPUT);
  const durationSeconds = Number(value("--duration-seconds") ?? 0);
  const restOnly = args.includes("--rest-only");
  const rawBooks = args.includes("--raw-books");
  const statusFile = value("--status-file") ? path.resolve(repoRoot, value("--status-file")!) : undefined;
  await fsp.mkdir(output, { recursive: true });
  const sink = new RotatingJsonlSink(output);
  const bookCompactor = new LiveBookCompactor();
  let stopping = false;
  const sockets = new Set<WebSocket>();
  const timers = new Set<NodeJS.Timeout>();
  let optionSnapshotCount = 0;
  let binanceSnapshotRunning = false;
  const startedAt = Date.now();
  const sourceHealth = new Map<string, { messages: number; lastMessageAt: number; opens: number; closes: number }>();

  const stop = async () => {
    if (stopping) return;
    stopping = true;
    for (const timer of timers) clearInterval(timer);
    for (const socket of sockets) socket.close();
  };
  process.once("SIGINT", () => void stop());
  process.once("SIGTERM", () => void stop());
  if (durationSeconds > 0) setTimeout(() => void stop(), durationSeconds * 1_000).unref();

  const compactTimer = setInterval(() => {
    sink.write("cross-exchange-book-1s", bookCompactor.summarize(), Date.now());
  }, 1_000);
  compactTimer.unref();
  timers.add(compactTimer);
  if (statusFile) {
    const statusTimer = setInterval(() => void writeStatus(), 5_000);
    statusTimer.unref();
    timers.add(statusTimer);
    await writeStatus();
  }

  sink.write("collector-session", {
    event: "start",
    pid: process.pid,
    startedAt: new Date(startedAt).toISOString(),
    sources: ["binance", "coinbase", "kraken", "deribit", "mempool.space", "GDELT"],
  });

  schedule("deribit options", 60_000, async () => {
    const observedAt = Date.now();
    const url = new URL("https://www.deribit.com/api/v2/public/get_book_summary_by_currency");
    url.searchParams.set("currency", "BTC");
    url.searchParams.set("kind", "option");
    const payload = await requestJson<{ result: DeribitBookSummary[] }>(url);
    sink.write("deribit-btc-option-summary", deriveDeribitOptionSummary(payload.result, observedAt), observedAt);
    optionSnapshotCount += 1;
    if (optionSnapshotCount % 15 === 1) {
      const directory = path.join(output, "deribit-btc-option-surface-raw", new Date(observedAt).toISOString().slice(0, 10));
      await fsp.mkdir(directory, { recursive: true });
      const file = path.join(directory, `${new Date(observedAt).toISOString().replace(/[:.]/g, "-")}.json.gz`);
      await fsp.writeFile(file, gzipSync(JSON.stringify({ observedAt, rows: payload.result })));
    }
  });
  schedule("mempool", 60_000, async () => sink.write("mempool-live", {
    observedAt: Date.now(),
    mempool: await requestJson(new URL("https://mempool.space/api/mempool")),
    recommendedFees: await requestJson(new URL("https://mempool.space/api/v1/fees/recommended")),
    projectedBlocks: await requestJson(new URL("https://mempool.space/api/v1/fees/mempool-blocks")),
    recent: await requestJson(new URL("https://mempool.space/api/mempool/recent")),
  }));
  schedule("Binance futures premium", 60_000, async () => sink.write(
    "binance-usdm-premium-index",
    await requestJson(new URL("https://fapi.binance.com/fapi/v1/premiumIndex?symbol=BTCUSDT")),
  ));
  schedule("GDELT", 15 * 60_000, async () => sink.write("gdelt-crypto-news", await collectGdelt()));

  const tradingEconomicsKey = process.env.TRADING_ECONOMICS_API_KEY;
  if (tradingEconomicsKey) schedule("Trading Economics", 60_000, async () => {
    const url = new URL("https://api.tradingeconomics.com/calendar/country/united%20states");
    url.searchParams.set("c", tradingEconomicsKey);
    url.searchParams.set("importance", "2");
    sink.write("macro-calendar-trading-economics", await requestJson(url));
  });
  else console.log("TRADING_ECONOMICS_API_KEY is not configured; macro consensus collection is disabled.");

  const blockworksKey = process.env.BLOCKWORKS_API_KEY;
  if (blockworksKey) schedule("Blockworks ETF", 60 * 60_000, async () => {
    const url = new URL("https://api.blockworks.com/v1/metrics/etf-flows-total-usd");
    url.searchParams.set("project", "bitcoin");
    sink.write("bitcoin-etf-flows", await requestJson(url, { "x-api-key": blockworksKey }));
  });
  else console.log("BLOCKWORKS_API_KEY is not configured; point-in-time ETF flow collection is disabled.");

  const websocketTasks: Promise<void>[] = [];
  if (!restOnly) {
    const specs: WebSocketSpec[] = [
      {
        id: "binance-spot-depth-diff",
        url: "wss://stream.binance.com:9443/ws/btcusdt@depth@100ms",
        onOpen: refreshBinanceSpotSnapshot,
        staleAfterMs: 15_000,
      },
      { id: "binance-spot-aggtrade", url: "wss://stream.binance.com:9443/ws/btcusdt@aggTrade", staleAfterMs: 30_000 },
      { id: "binance-usdm-liquidations", url: "wss://fstream.binance.com/market/ws/!forceOrder@arr" },
      { id: "binance-usdm-mark-price", url: "wss://fstream.binance.com/market/ws/btcusdt@markPrice@1s", staleAfterMs: 10_000 },
      { id: "binance-usdm-book-ticker", url: "wss://fstream.binance.com/public/ws/btcusdt@bookTicker", minRecordIntervalMs: 100, staleAfterMs: 15_000 },
      {
        id: "coinbase-btcusd-level2",
        url: "wss://ws-feed.exchange.coinbase.com",
        subscribe: { type: "subscribe", product_ids: ["BTC-USD"], channels: ["level2_batch", "heartbeat"] },
      },
      {
        id: "kraken-btcusd-book",
        url: "wss://ws.kraken.com/v2",
        subscribe: { method: "subscribe", params: { channel: "book", symbol: ["BTC/USD"], depth: 100, snapshot: true } },
      },
      {
        id: "deribit-btc-perpetual-book",
        url: "wss://www.deribit.com/ws/api/v2",
        subscribe: { jsonrpc: "2.0", id: 1, method: "public/subscribe", params: { channels: ["book.BTC-PERPETUAL.100ms", "trades.BTC-PERPETUAL.100ms", "trades.option.BTC.100ms"] } },
      },
    ];
    for (const spec of specs) websocketTasks.push(superviseWebSocket(spec));
  }

  while (!stopping) await delay(250);
  await Promise.allSettled(websocketTasks);
  sink.write("collector-session", { event: "stop", stoppedAt: new Date().toISOString() });
  await sink.close();
  if (statusFile) await writeStatus();
  console.log("External live collection stopped cleanly.");

  async function refreshBinanceSpotSnapshot() {
    if (binanceSnapshotRunning) return;
    binanceSnapshotRunning = true;
    try {
      const observedAt = Date.now();
      const snapshot = await requestJson(new URL("https://api.binance.com/api/v3/depth?symbol=BTCUSDT&limit=5000"));
      bookCompactor.consume("binance-spot-depth-snapshot", snapshot, observedAt);
      if (rawBooks) sink.write("binance-spot-depth-snapshot", snapshot, observedAt);
    } finally {
      binanceSnapshotRunning = false;
    }
  }

  async function writeStatus() {
    if (!statusFile) return;
    const now = Date.now();
    const status = {
      version: 1,
      pid: process.pid,
      startedAt: new Date(startedAt).toISOString(),
      updatedAt: new Date(now).toISOString(),
      stopping,
      output,
      rawBooks,
      compressedBytesBySource: sink.bytesWritten(),
      compressedBytesTotal: Object.values(sink.bytesWritten()).reduce((sum, bytes) => sum + bytes, 0),
      sources: Object.fromEntries([...sourceHealth.entries()].map(([source, health]) => [source, {
        ...health,
        lastMessageAt: health.lastMessageAt > 0 ? new Date(health.lastMessageAt).toISOString() : null,
        ageSeconds: health.lastMessageAt > 0 ? (now - health.lastMessageAt) / 1_000 : null,
      }])),
    };
    await fsp.mkdir(path.dirname(statusFile), { recursive: true });
    const temporary = `${statusFile}.${process.pid}.tmp`;
    await fsp.writeFile(temporary, `${JSON.stringify(status, null, 2)}\n`, "utf8");
    await fsp.rename(temporary, statusFile);
  }

  function schedule(label: string, interval: number, task: () => Promise<void>) {
    let running = false;
    const execute = async () => {
      if (stopping || running) return;
      running = true;
      try {
        await task();
      } catch (error) {
        console.error(`${label}: ${error instanceof Error ? error.message : error}`);
      } finally {
        running = false;
      }
    };
    void execute();
    const timer = setInterval(() => void execute(), interval);
    timer.unref();
    timers.add(timer);
  }

  async function superviseWebSocket(spec: WebSocketSpec) {
    let backoff = 1_000;
    while (!stopping) {
      try {
        await connectOnce(spec);
        backoff = 1_000;
      } catch (error) {
        if (!stopping) console.error(`${spec.id}: ${error instanceof Error ? error.message : error}`);
      }
      if (!stopping) {
        await delay(backoff);
        backoff = Math.min(30_000, backoff * 2);
      }
    }
  }

  function connectOnce(spec: WebSocketSpec) {
    return new Promise<void>((resolve, reject) => {
      const socket = new WebSocket(spec.url, { headers: { "user-agent": USER_AGENT } });
      sockets.add(socket);
      let opened = false;
      let lastRecordedAt = 0;
      let lastMessageAt = Date.now();
      const health = sourceHealth.get(spec.id) ?? { messages: 0, lastMessageAt: 0, opens: 0, closes: 0 };
      sourceHealth.set(spec.id, health);
      const ping = setInterval(() => {
        if (socket.readyState === WebSocket.OPEN) socket.ping();
      }, 30_000);
      ping.unref();
      const staleCheck = spec.staleAfterMs ? setInterval(() => {
        if (socket.readyState === WebSocket.OPEN && Date.now() - lastMessageAt > spec.staleAfterMs!) {
          console.error(`${spec.id}: no messages for ${Date.now() - lastMessageAt}ms; reconnecting`);
          socket.close(4_000, "stale stream");
        }
      }, Math.min(5_000, spec.staleAfterMs)) : undefined;
      staleCheck?.unref();
      socket.on("open", () => {
        opened = true;
        health.opens += 1;
        sink.write(`${spec.id}-session`, { event: "open", url: spec.url });
        if (spec.subscribe) socket.send(JSON.stringify(spec.subscribe));
        void spec.onOpen?.().catch((error) => console.error(`${spec.id} snapshot: ${error.message}`));
      });
      socket.on("message", (data) => {
        const receivedAt = Date.now();
        lastMessageAt = receivedAt;
        health.messages += 1;
        health.lastMessageAt = receivedAt;
        const text = data.toString();
        let payload: unknown = text;
        try { payload = JSON.parse(text); } catch { /* retain raw text */ }
        bookCompactor.consume(spec.id, payload, receivedAt);
        if (spec.id === "binance-spot-depth-diff" && bookCompactor.needsBinanceSnapshot()) {
          void refreshBinanceSpotSnapshot().catch((error) => console.error(`${spec.id} resnapshot: ${error.message}`));
        }
        if (spec.id === "kraken-btcusd-book" && bookCompactor.needsKrakenSnapshot()) {
          console.error(`${spec.id}: reconstructed book crossed; reconnecting for a fresh snapshot`);
          socket.close(4_001, "invalid reconstructed book");
          return;
        }
        if (spec.minRecordIntervalMs && receivedAt - lastRecordedAt < spec.minRecordIntervalMs) return;
        lastRecordedAt = receivedAt;
        const deribitChannel = String((payload as any)?.params?.channel ?? "");
        const routedSource = spec.id === "deribit-btc-perpetual-book" && deribitChannel.startsWith("trades.")
          ? deribitChannel.includes("option") ? "deribit-btc-option-trades" : "deribit-btc-perpetual-trades"
          : spec.id;
        if (rawBooks || !COMPACTED_BOOK_SOURCES.has(spec.id) || routedSource !== spec.id) {
          sink.write(routedSource, { receivedAt, monotonicNs: process.hrtime.bigint().toString(), message: payload }, receivedAt);
        }
      });
      socket.on("error", (error) => {
        if (!opened) reject(error);
        else console.error(`${spec.id} websocket: ${error.message}`);
      });
      socket.on("close", (code, reason) => {
        clearInterval(ping);
        if (staleCheck) clearInterval(staleCheck);
        sockets.delete(socket);
        health.closes += 1;
        sink.write(`${spec.id}-session`, { event: "close", code, reason: reason.toString() });
        resolve();
      });
    });
  }
}

async function collectGdelt() {
  const ngramList = await requestText(new URL("http://data.gdeltproject.org/gdeltv3/web/ngrams/LASTUPDATE.TXT"));
  const ngramUrl = ngramList.split(/\r?\n/).find((line) => line.includes(".1gram.txt.gz"));
  const gkgList = await requestText(new URL("http://data.gdeltproject.org/gdeltv2/lastupdate.txt"));
  const gkgUrl = gkgList.split(/\r?\n/).map((line) => line.trim().split(/\s+/).at(-1))
    .find((line) => line?.includes(".gkg.csv.zip"));
  if (!ngramUrl || !gkgUrl) throw new Error("GDELT update manifests did not contain expected files");
  const [ngramBytes, gkgBytes] = await Promise.all([
    requestBytes(new URL(ngramUrl)),
    requestBytes(new URL(gkgUrl)),
  ]);
  const ngrams = summarizeGdeltNgrams(gunzipSync(ngramBytes).toString("utf8"));
  const zip = new AdmZip(gkgBytes);
  const entries = zip.getEntries().filter((entry) => !entry.isDirectory);
  if (entries.length !== 1) throw new Error(`GDELT GKG ZIP has ${entries.length} entries`);
  const gkg = summarizeGdeltGkg(entries[0]!.getData().toString("utf8"));
  return { observedAt: Date.now(), ngramUrl, gkgUrl, ngrams, gkg };
}

async function requestJson<T = unknown>(url: URL, headers: Record<string, string> = {}): Promise<T> {
  return JSON.parse(await requestText(url, headers)) as T;
}

async function requestText(url: URL, headers: Record<string, string> = {}) {
  return (await requestBytes(url, headers)).toString("utf8");
}

async function requestBytes(url: URL, headers: Record<string, string> = {}, attempts = 5): Promise<Buffer> {
  let lastError: unknown;
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 60_000);
    try {
      const response = await fetch(url, { signal: controller.signal, headers: { "user-agent": USER_AGENT, ...headers } });
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      return Buffer.from(await response.arrayBuffer());
    } catch (error) {
      lastError = error;
      if (attempt + 1 < attempts) await delay(500 * 2 ** attempt);
    } finally {
      clearTimeout(timeout);
    }
  }
  throw lastError;
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
