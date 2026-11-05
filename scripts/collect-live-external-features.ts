import fsp from "node:fs/promises";
import path from "node:path";
import { randomUUID, createHash } from "node:crypto";
import { fileURLToPath } from "node:url";
import { gunzipSync, gzipSync } from "node:zlib";
import AdmZip from "adm-zip";
import WebSocket from "ws";
import { deriveDeribitOptionSummary, type DeribitBookSummary } from "./lib/external-public-data.ts";
import { summarizeGdeltGkg, summarizeGdeltNgrams } from "./lib/external-live-data.ts";
import { LiveBookCompactor } from "./lib/live-book-compactor.ts";
import { RotatingJsonlSink } from "./lib/live-jsonl-sink.ts";
import { BinanceEventCollector, selectCaptureMarkets, type CaptureUniverse, type BinanceVenue } from "./lib/binance-event-collection.ts";
import { createHyperliquidCaptureSpecs } from "./lib/hyperliquid-collection.ts";
import { runBlockchainCapture, validateChainConfigs, type ChainConfig, type ChainHealth } from "./lib/blockchain-events.ts";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const DEFAULT_OUTPUT = "data/market/mutable/external-live";
const USER_AGENT = "trading-live-external-collector/1.0";

interface WebSocketSpec {
  readonly id: string;
  readonly url: string;
  readonly subscribe?: unknown;
  readonly subscriptions?: readonly unknown[];
  readonly heartbeatMessage?: unknown;
  readonly onOpen?: () => Promise<void>;
  readonly minRecordIntervalMs?: number;
  readonly staleAfterMs?: number;
}

const COMPACTED_BOOK_SOURCES = new Set([
  "binance-spot-depth-diff",
  "binance-usdm-book-ticker",
  "coinbase-btcusd-level2",
  "kraken-btcusd-book",
  "deribit-btc-perpetual-book",
]);

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
  --quality compact|full    full retains all published Binance event messages
  --universe FILE           Explicit economic identities + physical routes; implies full
  --symbols BTCUSDT,ETHUSDT  Filter explicit physical market symbols
  --venues spot,usdm-futures Filter Binance product venues
  --markets-per-socket N    Multiplex batch size, 1..100 (default 20)
  --snapshot-interval-ms N  Global REST snapshot start spacing >=3000 (default 3000)
  --checkpoint-seconds N   Local reconstructed book checkpoint cadence (default 300)
  --derived-rows true|false Store per-event derived BBO/trade alignment (default false)
  --compression gzip|none  Event JSONL compression (default gzip)
  --gzip-level N           Compression level 0..9 (default 6)
  --fsync true|false        Sync each flushed chunk (default true in full mode)
  --max-pending-mib N       Stop on sink backlog overflow (default 256)
  --binance-only            Skip BTC non-Binance supplemental feeds
  --hyperliquid             Also retain public address-attributed observations
  --hyperliquid-coins CSV   Exact provider coin IDs (default BTC)
  --hyperliquid-users CSV   Optional explicit watched addresses (max 10)
  --blockchain-config FILE Bitcoin, EVM, Solana, TRON or XRP Ledger blocks
  --status-file FILE         Atomically write collector health every 5 seconds

Optional environment variables:
  TRADING_ECONOMICS_API_KEY  Point-in-time macro calendar actual/forecast feed
  BLOCKWORKS_API_KEY         Daily ETF flow feed`);
    return;
  }
  const output = path.resolve(repoRoot, value("--output-dir") ?? DEFAULT_OUTPUT);
  const durationSeconds = Number(value("--duration-seconds") ?? 0);
  if (!Number.isFinite(durationSeconds) || durationSeconds < 0) throw new Error("duration-seconds must be nonnegative");
  const restOnly = args.includes("--rest-only");
  const universeFile = value("--universe");
  const quality = value("--quality") ?? (universeFile ? "full" : "compact");
  if (quality !== "full" && quality !== "compact") throw new Error("quality must be compact or full");
  const fullQuality = quality === "full";
  if (universeFile && !fullQuality) throw new Error("A universe requires --quality full; compact mode is the BTC supplemental feed");
  if (fullQuality && restOnly) throw new Error("Full event recording requires WebSockets");
  const rawBooks = fullQuality || args.includes("--raw-books");
  const binanceOnly = args.includes("--binance-only");
  const compression = value("--compression") ?? "gzip";
  if (compression !== "gzip" && compression !== "none") throw new Error("compression must be gzip or none");
  const fsyncValue = value("--fsync") ?? String(fullQuality);
  if (fsyncValue !== "true" && fsyncValue !== "false") throw new Error("fsync must be true or false");
  const blockchainFile = value("--blockchain-config");
  const blockchainConfig = blockchainFile ? JSON.parse(await fsp.readFile(path.resolve(repoRoot, blockchainFile), "utf8")) : { chains: [] };
  if (!Array.isArray(blockchainConfig.chains)) throw new Error("Blockchain config requires a chains array");
  const chains = blockchainConfig.chains as readonly ChainConfig[];
  validateChainConfigs(chains);
  if (chains.length && fsyncValue !== "true") throw new Error("Blockchain cursor durability requires --fsync true");
  const derivedValue = value("--derived-rows") ?? "false";
  if (derivedValue !== "true" && derivedValue !== "false") throw new Error("derived-rows must be true or false");
  const derivedRows = derivedValue === "true";
  const positive = (name: string, fallback: number) => {
    const number = Number(value(name) ?? fallback);
    if (!Number.isFinite(number) || number <= 0) throw new Error(`${name} must be positive`);
    return number;
  };
  const csv = (name: string) => value(name)?.split(",").map((part) => part.trim()).filter(Boolean);
  const hyperliquidSpecs = args.includes("--hyperliquid") ? createHyperliquidCaptureSpecs(csv("--hyperliquid-coins") ?? ["BTC"], csv("--hyperliquid-users") ?? []) : [];
  const venues = csv("--venues") ?? ["spot", "usdm-futures"];
  if (venues.some((venue) => venue !== "spot" && venue !== "usdm-futures")) throw new Error("venues must be spot and/or usdm-futures");
  const defaultUniverse: CaptureUniverse = {
    entries: [{ baseAsset: "BTC", assetId: "crypto:BTC", routes: [
      { venue: "spot", symbol: "BTCUSDT", baseAsset: "BTC", quoteAsset: "USDT", priceOrientation: "direct" },
      { venue: "usdm-futures", symbol: "BTCUSDT", baseAsset: "BTC", quoteAsset: "USDT", priceOrientation: "direct" },
    ] }],
  };
  const universeBytes = universeFile ? await fsp.readFile(path.resolve(repoRoot, universeFile)) : Buffer.from(JSON.stringify(defaultUniverse));
  const universe: CaptureUniverse = JSON.parse(universeBytes.toString("utf8"));
  const universeSha256 = createHash("sha256").update(universeBytes).digest("hex");
  const markets = fullQuality ? selectCaptureMarkets(universe, csv("--symbols"), venues as readonly BinanceVenue[]) : [];
  const marketsPerSocket = positive("--markets-per-socket", 20);
  const snapshotIntervalMs = positive("--snapshot-interval-ms", 3_000);
  const checkpointSeconds = positive("--checkpoint-seconds", 300);
  const gzipLevel = Number(value("--gzip-level") ?? 6);
  if (!Number.isInteger(marketsPerSocket) || marketsPerSocket > 100) throw new Error("markets-per-socket must be 1..100");
  if (snapshotIntervalMs < 3_000) throw new Error("snapshot-interval-ms must be >=3000");
  if (!Number.isInteger(gzipLevel) || gzipLevel < 0 || gzipLevel > 9) throw new Error("gzip-level must be 0..9");
  const statusFile = value("--status-file") ? path.resolve(repoRoot, value("--status-file")!) : undefined;
  await fsp.mkdir(output, { recursive: true });
  const universeSnapshot = path.join("collector-universes", `${universeSha256}.json`);
  if (fullQuality) {
    await fsp.mkdir(path.join(output, "collector-universes"), { recursive: true });
    try { await fsp.writeFile(path.join(output, universeSnapshot), universeBytes, { flag: "wx" }); }
    catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error;
      const previous = await fsp.readFile(path.join(output, universeSnapshot));
      if (!previous.equals(universeBytes)) throw new Error("The saved universe snapshot does not match its content hash");
    }
  }
  const sessionId = randomUUID();
  let fatalError: Error | undefined;
  let binanceCollector: BinanceEventCollector | undefined;
  const fail = (error: Error) => { fatalError ??= error; console.error(`Collector fatal error: ${error.message}`); void stop(); };
  const sink = new RotatingJsonlSink(output, { compression, gzipLevel, fsync: fsyncValue === "true", sessionId, sessionFiles: fullQuality || chains.length > 0,
    maxPendingBytes: positive("--max-pending-mib", 256) * 1_024 * 1_024, onError: fail });
  const bookCompactor = new LiveBookCompactor();
  let stopping = false;
  const sockets = new Set<WebSocket>();
  const timers = new Set<NodeJS.Timeout>();
  const restJobs = new Set<Promise<void>>();
  const restStop = new AbortController();
  let blockchainHealth: readonly ChainHealth[] = [];
  const getJson = <T = unknown>(url: URL, headers: Record<string, string> = {}) => requestJson<T>(url, headers, restStop.signal);
  let optionSnapshotCount = 0;
  let binanceSnapshotRunning = false;
  const startedAt = Date.now();
  const sourceHealth = new Map<string, { messages: number; lastMessageAt: number; opens: number; closes: number }>();

  const stop = async () => {
    if (stopping) return;
    stopping = true;
    restStop.abort();
    binanceCollector?.stop();
    for (const timer of timers) clearInterval(timer);
    for (const socket of sockets) socket.terminate();
  };
  process.once("SIGINT", stop);
  process.once("SIGTERM", stop);
  const durationTimer = durationSeconds > 0 ? setTimeout(() => void stop(), durationSeconds * 1_000) : undefined;
  durationTimer?.unref();

  const compactTimer = setInterval(() => {
    try { sink.write("cross-exchange-book-1s", bookCompactor.summarize(), Date.now()); } catch (error) { fail(error instanceof Error ? error : new Error(String(error))); }
  }, 1_000);
  compactTimer.unref();
  timers.add(compactTimer);
  if (statusFile) {
    const statusTimer = setInterval(() => void writeStatus().catch(fail), 5_000);
    statusTimer.unref();
    timers.add(statusTimer);
    await writeStatus();
  }

  sink.write("collector-session", {
    event: "start",
    pid: process.pid,
    sessionId,
    quality,
    derivedRows,
    universeFile: universeFile ?? null,
    universeSha256: fullQuality ? universeSha256 : null,
    universeSnapshot: fullQuality ? universeSnapshot : null,
    captureOptions: { marketsPerSocket, snapshotIntervalMs, checkpointSeconds, compression, gzipLevel,
      fsync: fsyncValue === "true", derivedRows, maxPendingMiB: positive("--max-pending-mib", 256), binanceOnly },
    markets,
    startedAt: new Date(startedAt).toISOString(),
    sources: ["binance", ...(!binanceOnly ? ["coinbase", "kraken", "deribit", "mempool.space", "GDELT"] : []),
      ...(hyperliquidSpecs.length > 0 ? ["hyperliquid"] : []), ...chains.map((chain) => `blockchain:${chain.id}`)],
    blockchainConfigSha256: chains.length ? createHash("sha256").update(JSON.stringify(chains)).digest("hex") : null,
  });

  if (!binanceOnly) {
  schedule("deribit options", 60_000, async () => {
    const observedAt = Date.now();
    const url = new URL("https://www.deribit.com/api/v2/public/get_book_summary_by_currency");
    url.searchParams.set("currency", "BTC");
    url.searchParams.set("kind", "option");
    const payload = await getJson<{ result: DeribitBookSummary[] }>(url);
    if (stopping) return;
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
    mempool: await getJson(new URL("https://mempool.space/api/mempool")),
    recommendedFees: await getJson(new URL("https://mempool.space/api/v1/fees/recommended")),
    projectedBlocks: await getJson(new URL("https://mempool.space/api/v1/fees/mempool-blocks")),
    recent: await getJson(new URL("https://mempool.space/api/mempool/recent")),
  }));
  schedule("Binance futures premium", 60_000, async () => sink.write(
    "binance-usdm-premium-index",
    await getJson(new URL("https://fapi.binance.com/fapi/v1/premiumIndex?symbol=BTCUSDT")),
  ));
  schedule("GDELT", 15 * 60_000, async () => sink.write("gdelt-crypto-news", await collectGdelt(restStop.signal)));

  const tradingEconomicsKey = process.env.TRADING_ECONOMICS_API_KEY;
  if (tradingEconomicsKey) schedule("Trading Economics", 60_000, async () => {
    const url = new URL("https://api.tradingeconomics.com/calendar/country/united%20states");
    url.searchParams.set("c", tradingEconomicsKey);
    url.searchParams.set("importance", "2");
    sink.write("macro-calendar-trading-economics", await getJson(url));
  });
  else console.log("TRADING_ECONOMICS_API_KEY is not configured; macro consensus collection is disabled.");

  const blockworksKey = process.env.BLOCKWORKS_API_KEY;
  if (blockworksKey) schedule("Blockworks ETF", 60 * 60_000, async () => {
    const url = new URL("https://api.blockworks.com/v1/metrics/etf-flows-total-usd");
    url.searchParams.set("project", "bitcoin");
    sink.write("bitcoin-etf-flows", await getJson(url, { "x-api-key": blockworksKey }));
  });
  else console.log("BLOCKWORKS_API_KEY is not configured; point-in-time ETF flow collection is disabled.");
  }

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
    for (const spec of [...specs.filter((spec) => !binanceOnly || spec.id.startsWith("binance-")), ...hyperliquidSpecs]) websocketTasks.push(superviseWebSocket(spec));
  }
  if (fullQuality) {
    binanceCollector = new BinanceEventCollector({ markets, sink, sessionId,
      marketsPerSocket, snapshotIntervalMs, checkpointSeconds, derivedRows, onFatal: fail });
    websocketTasks.push(binanceCollector.run().catch(fail));
  }
  if (chains.length) websocketTasks.push(runBlockchainCapture(chains, sink,
    path.join(output, "blockchain-cursors"), restStop.signal, (health) => { blockchainHealth = health; }).catch(fail));

  while (!stopping) await delay(250);
  await Promise.allSettled(websocketTasks);
  await Promise.allSettled(restJobs);
  if (durationTimer) clearTimeout(durationTimer);
  process.removeListener("SIGINT", stop);
  process.removeListener("SIGTERM", stop);
  if (fatalError) {
    await sink.close().catch(() => {});
    if (statusFile) await writeStatus();
    throw fatalError;
  }
  sink.write("collector-session", { event: "stop", stoppedAt: new Date().toISOString() });
  await sink.close();
  if (statusFile) await writeStatus();
  console.log("External live collection stopped cleanly.");

  async function refreshBinanceSpotSnapshot() {
    if (binanceSnapshotRunning) return;
    binanceSnapshotRunning = true;
    const task = (async () => {
      const observedAt = Date.now();
      const snapshot = await getJson(new URL("https://api.binance.com/api/v3/depth?symbol=BTCUSDT&limit=5000"));
      if (stopping) return;
      bookCompactor.consume("binance-spot-depth-snapshot", snapshot, observedAt);
      if (rawBooks) sink.write("binance-spot-depth-snapshot", snapshot, observedAt);
    })();
    restJobs.add(task);
    try { await task;
    } finally {
      restJobs.delete(task);
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
      fatalError: fatalError?.message ?? null,
      sessionId,
      quality,
      derivedRows,
      durability: sink.health(),
      blockchainHealth,
      binanceMarkets: binanceCollector?.health() ?? [],
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
        if (!stopping) console.error(`${label}: ${error instanceof Error ? error.message : error}`);
        if (["ENOSPC", "EIO", "EROFS", "EACCES"].includes((error as NodeJS.ErrnoException).code ?? "")) fail(error instanceof Error ? error : new Error(String(error)));
      } finally {
        running = false;
      }
    };
    const launch = () => {
      const job = execute(); restJobs.add(job);
      void job.finally(() => restJobs.delete(job));
    };
    launch();
    const timer = setInterval(launch, interval);
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
        await delay(backoff, restStop.signal).catch(() => {});
        backoff = Math.min(30_000, backoff * 2);
      }
    }
  }

  function connectOnce(spec: WebSocketSpec) {
    return new Promise<void>((resolve, reject) => {
      const connectionId = randomUUID();
      const socket = new WebSocket(spec.url, { headers: { "user-agent": USER_AGENT } });
      sockets.add(socket);
      let opened = false;
      let lastRecordedAt = 0;
      let lastMessageAt = Date.now();
      const health = sourceHealth.get(spec.id) ?? { messages: 0, lastMessageAt: 0, opens: 0, closes: 0 };
      sourceHealth.set(spec.id, health);
      const ping = setInterval(() => {
        if (socket.readyState === WebSocket.OPEN) {
          if (spec.heartbeatMessage) socket.send(JSON.stringify(spec.heartbeatMessage));
          else socket.ping();
        }
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
        try {
        opened = true;
        health.opens += 1;
        sink.write(`${spec.id}-session`, { event: "open", connectionId, url: spec.url });
        if (spec.subscribe) socket.send(JSON.stringify(spec.subscribe));
        for (const subscription of spec.subscriptions ?? []) socket.send(JSON.stringify(subscription));
        void spec.onOpen?.().catch((error) => console.error(`${spec.id} snapshot: ${error.message}`));
        } catch (error) { fail(error instanceof Error ? error : new Error(String(error))); }
      });
      socket.on("message", (data) => {
        try {
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
        if (!fullQuality && spec.minRecordIntervalMs && receivedAt - lastRecordedAt < spec.minRecordIntervalMs) return;
        lastRecordedAt = receivedAt;
        const deribitChannel = String((payload as any)?.params?.channel ?? "");
        const routedSource = spec.id === "deribit-btc-perpetual-book" && deribitChannel.startsWith("trades.")
          ? deribitChannel.includes("option") ? "deribit-btc-option-trades" : "deribit-btc-perpetual-trades"
          : spec.id;
        if (rawBooks || !COMPACTED_BOOK_SOURCES.has(spec.id) || routedSource !== spec.id) {
          sink.write(routedSource, { receivedAt, connectionId, monotonicNs: process.hrtime.bigint().toString(), message: payload,
            ...(fullQuality ? { raw: text } : {}) }, receivedAt);
        }
        } catch (error) { fail(error instanceof Error ? error : new Error(String(error))); }
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
        try { sink.write(`${spec.id}-session`, { event: "close", connectionId, code, reason: reason.toString() }); }
        catch (error) { fail(error instanceof Error ? error : new Error(String(error))); }
        resolve();
      });
    });
  }
}

async function collectGdelt(signal?: AbortSignal) {
  const ngramList = await requestText(new URL("http://data.gdeltproject.org/gdeltv3/web/ngrams/LASTUPDATE.TXT"), {}, signal);
  const ngramUrl = ngramList.split(/\r?\n/).find((line) => line.includes(".1gram.txt.gz"));
  const gkgList = await requestText(new URL("http://data.gdeltproject.org/gdeltv2/lastupdate.txt"), {}, signal);
  const gkgUrl = gkgList.split(/\r?\n/).map((line) => line.trim().split(/\s+/).at(-1))
    .find((line) => line?.includes(".gkg.csv.zip"));
  if (!ngramUrl || !gkgUrl) throw new Error("GDELT update manifests did not contain expected files");
  const [ngramBytes, gkgBytes] = await Promise.all([
    requestBytes(new URL(ngramUrl), {}, 5, signal),
    requestBytes(new URL(gkgUrl), {}, 5, signal),
  ]);
  const ngrams = summarizeGdeltNgrams(gunzipSync(ngramBytes).toString("utf8"));
  const zip = new AdmZip(gkgBytes);
  const entries = zip.getEntries().filter((entry) => !entry.isDirectory);
  if (entries.length !== 1) throw new Error(`GDELT GKG ZIP has ${entries.length} entries`);
  const gkg = summarizeGdeltGkg(entries[0]!.getData().toString("utf8"));
  return { observedAt: Date.now(), ngramUrl, gkgUrl, ngrams, gkg };
}

async function requestJson<T = unknown>(url: URL, headers: Record<string, string> = {}, signal?: AbortSignal): Promise<T> {
  return JSON.parse(await requestText(url, headers, signal)) as T;
}

async function requestText(url: URL, headers: Record<string, string> = {}, signal?: AbortSignal) {
  return (await requestBytes(url, headers, 5, signal)).toString("utf8");
}

async function requestBytes(url: URL, headers: Record<string, string> = {}, attempts = 5, signal?: AbortSignal): Promise<Buffer> {
  let lastError: unknown;
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    signal?.throwIfAborted();
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 60_000);
    try {
      const response = await fetch(url, { signal: signal ? AbortSignal.any([signal, controller.signal]) : controller.signal, headers: { "user-agent": USER_AGENT, ...headers } });
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      return Buffer.from(await response.arrayBuffer());
    } catch (error) {
      lastError = error;
      signal?.throwIfAborted();
      if (attempt + 1 < attempts) await delay(500 * 2 ** attempt, signal);
    } finally {
      clearTimeout(timeout);
    }
  }
  throw lastError;
}

function delay(milliseconds: number, signal?: AbortSignal) {
  signal?.throwIfAborted();
  return new Promise<void>((resolve, reject) => {
    const complete = () => { signal?.removeEventListener("abort", abort); resolve(); };
    const timer = setTimeout(complete, milliseconds);
    const abort = () => { clearTimeout(timer); signal?.removeEventListener("abort", abort); reject(signal?.reason); };
    signal?.addEventListener("abort", abort, { once: true });
  });
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  run().catch((error: unknown) => {
    console.error(error instanceof Error ? error.stack ?? error.message : error);
    process.exitCode = 1;
  });
}
