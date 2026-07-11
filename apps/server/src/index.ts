import fs from "node:fs";
import { monitorEventLoopDelay, performance } from "node:perf_hooks";
import Fastify from "fastify";
import cors from "@fastify/cors";
import staticFiles from "@fastify/static";
import { WebSocketServer, type WebSocket } from "ws";
import type {
  BacktestPreset,
  ManualTradeInput,
  PartialStrategyConfig,
} from "@trading/bot-algo";
import { appConfig } from "./config.js";
import { BinanceMarketStream } from "./binance-stream.js";
import { BinanceExchangeUserDataStream } from "./binance-user-data-stream.js";
import {
  BinanceMarketCatalog,
  getMarketStorageKey,
  isStreamVenue,
  type BinanceMarketCatalogSnapshot,
  type BinanceMarketListing,
  type MarketGroup,
} from "./binance-markets.js";
import { TradingRuntime } from "./runtime.js";
import { TradingStorage } from "./storage.js";
import type { HistoricalBacktestMarket } from "./historical-backtest.js";
import { CorrelationService } from "./correlation-service.js";
import {
  BinanceExchangeTrading,
  type BinanceExchangeCancelOrderInput,
  type BinanceExchangePlaceOrderInput,
} from "./binance-exchange.js";

const server = Fastify({
  logger: {
    level: process.env.LOG_LEVEL ?? "info",
  },
});

const MAX_RANDOM_PAIR_COUNT = 25;
const MARKET_CATALOG_RESPONSE_TIMEOUT_MS = 5_000;
const DASHBOARD_WS_HEARTBEAT_MS = 30_000;
const HEARTBEAT_LOG_MS = Math.max(0, Number(process.env.TRADING_HEARTBEAT_LOG_MS ?? 60_000));
const eventLoopDelay = monitorEventLoopDelay({ resolution: 20 });
let tcpConnectionCount = 0;

configureProcessReports();
eventLoopDelay.enable();
server.server.on("connection", (socket) => {
  tcpConnectionCount += 1;
  socket.on("close", () => {
    tcpConnectionCount = Math.max(0, tcpConnectionCount - 1);
  });
});

await server.register(cors, {
  origin: true,
});

if (fs.existsSync(appConfig.webDistDir)) {
  await server.register(staticFiles, {
    root: appConfig.webDistDir,
    prefix: "/",
  });
}

const marketCatalog = new BinanceMarketCatalog({
  apiKey: appConfig.binanceApiKey,
  apiSecret: appConfig.binanceApiSecret,
});
const initialMarket = await resolveInitialMarket();
let activeMarket = initialMarket;
const exchangeTrading = new BinanceExchangeTrading(appConfig.binanceExchange);
const runtime = new TradingRuntime(
  createStorage(initialMarket),
  initialMarket,
  appConfig.strategy,
  appConfig.interval,
  {
    dataDir: appConfig.dataDir,
    maxBytes: appConfig.historicalCache.maxBytes,
    minFreeBytes: appConfig.historicalCache.minFreeBytes,
  },
  exchangeTrading,
  {
    hardStop: appConfig.exchangeAccountGuard.hardStop,
    onWarning: (message) =>
      server.log.warn({ subsystem: "exchange-account-guard" }, message),
  },
);
await runtime.init();
const correlationService = new CorrelationService({
  dataDir: appConfig.dataDir,
  interval: appConfig.interval,
  lookbackMs: appConfig.correlations.lookbackDays * 24 * 60 * 60 * 1000,
  maxMarkets: appConfig.correlations.maxMarkets,
  historicalCache: {
    maxBytes: appConfig.historicalCache.maxBytes,
    minFreeBytes: appConfig.historicalCache.minFreeBytes,
  },
});

server.get("/health", async () => ({
  ok: true,
  environment: appConfig.environment,
  market: appConfig.market.id,
  symbol: appConfig.market.symbol,
  dataDir: appConfig.dataDir,
}));

server.get("/api/diagnostics", async () => diagnosticsSnapshot());

server.get("/api/state", async () => publicSnapshot());

server.get("/api/markets", async (request) => {
  const query = request.query as { refresh?: string };
  const refresh = query.refresh === "1" || query.refresh === "true";
  const catalogRequest = marketCatalog.list(refresh).catch((error) => {
    const message = error instanceof Error ? error.message : "Binance market catalog failed.";
    server.log.warn({ error: message }, "Binance market catalog request failed");
    return fallbackMarketCatalog(activeMarket, message);
  });

  return Promise.race([
    catalogRequest,
    delay(MARKET_CATALOG_RESPONSE_TIMEOUT_MS).then(() =>
      fallbackMarketCatalog(
        activeMarket,
        `Binance market catalog did not respond within ${MARKET_CATALOG_RESPONSE_TIMEOUT_MS}ms.`,
      ),
    ),
  ]);
});

server.post("/api/market", async (request, reply) => {
  const body = (request.body ?? {}) as { marketId?: string };
  if (!body.marketId) {
    return reply.code(400).send({ error: "marketId is required." });
  }

  const market = await marketCatalog.find(body.marketId);
  if (!market) {
    return reply.code(404).send({ error: "Market is not listed by Binance." });
  }
  if (!market.supportsLiveStream || !isStreamVenue(market.venue)) {
    return reply.code(400).send({
      error: market.unavailableReason ?? "This market is not supported by the live dashboard yet.",
    });
  }

  streamGeneration += 1;
  stream.stop();
  userDataStream.stop();
  await runtime.switchMarket(market, createStorage(market));
  activeMarket = market;
  stream = createStream(market, streamGeneration);
  userDataStream = createUserDataStream(market, streamGeneration);
  stream.start();
  userDataStream.start();
  broadcastState();
  return publicSnapshot();
});

server.get("/api/history", async (request) => {
  const query = request.query as { limit?: string };
  const limit = clampInt(Number(query.limit ?? 500), 1, 2_000);
  return {
    candles: runtime.snapshot().market.candles.slice(-limit),
  };
});

server.post("/api/bot/start", async () => {
  await runtime.startBot();
  broadcastState();
  return publicSnapshot();
});

server.post("/api/bot/stop", async () => {
  await runtime.stopBot();
  broadcastState();
  return publicSnapshot();
});

server.post("/api/bot/reset", async () => {
  await runtime.resetBot();
  broadcastState();
  return publicSnapshot();
});

server.post("/api/bot/close-positions", async (request, reply) => {
  try {
    const body = (request.body ?? {}) as { includeUnprofitable?: boolean };
    await runtime.closePositions({ includeUnprofitable: body.includeUnprofitable === true });
    broadcastState();
    return publicSnapshot();
  } catch (error) {
    const message = error instanceof Error ? error.message : "Position close failed";
    return reply.code(400).send({ error: message });
  }
});

server.put("/api/bot/config", async (request) => {
  const body = (request.body ?? {}) as PartialStrategyConfig;
  await runtime.updateBotConfig(body);
  broadcastState();
  return publicSnapshot();
});

server.put("/api/bot/execution", async (request, reply) => {
  try {
    const body = (request.body ?? {}) as { mode?: string };
    await runtime.setExecutionMode(body.mode === "binance" ? "binance" : "simulated");
    broadcastState();
    return publicSnapshot();
  } catch (error) {
    const message = error instanceof Error ? error.message : "Execution mode update failed";
    return reply.code(400).send({ error: message });
  }
});

server.post("/api/bot/manual-trade", async (request, reply) => {
  try {
    const body = (request.body ?? {}) as ManualTradeInput;
    await runtime.recordManualTrade(body);
    broadcastState();
    return publicSnapshot();
  } catch (error) {
    const message = error instanceof Error ? error.message : "Manual trade failed";
    return reply.code(400).send({ error: message });
  }
});

server.post("/api/exchange/sync", async (_request, reply) => {
  try {
    await runtime.syncExchange();
    broadcastState();
    return publicSnapshot();
  } catch (error) {
    const message = error instanceof Error ? error.message : "Binance exchange sync failed";
    return reply.code(400).send({ error: message });
  }
});

server.put("/api/exchange/credentials", async (request, reply) => {
  try {
    const body = (request.body ?? {}) as {
      mode?: string;
      apiKey?: string;
      apiSecret?: string;
    };
    await runtime.setExchangeCredentials(body);
    broadcastState();
    return publicSnapshot();
  } catch (error) {
    const message = error instanceof Error ? error.message : "Binance credential update failed";
    return reply.code(400).send({ error: message });
  }
});

server.post("/api/exchange/order", async (request, reply) => {
  try {
    const body = (request.body ?? {}) as BinanceExchangePlaceOrderInput;
    await runtime.placeExchangeOrder(normalizeExchangeOrderInput(body));
    broadcastState();
    return publicSnapshot();
  } catch (error) {
    const message = error instanceof Error ? error.message : "Binance exchange order failed";
    return reply.code(400).send({ error: message });
  }
});

server.delete("/api/exchange/order", async (request, reply) => {
  try {
    const body = (request.body ?? {}) as BinanceExchangeCancelOrderInput;
    await runtime.cancelExchangeOrder(body);
    broadcastState();
    return publicSnapshot();
  } catch (error) {
    const message = error instanceof Error ? error.message : "Binance exchange cancel failed";
    return reply.code(400).send({ error: message });
  }
});

server.delete("/api/exchange/open-orders", async (_request, reply) => {
  try {
    await runtime.cancelAllExchangeOrders();
    broadcastState();
    return publicSnapshot();
  } catch (error) {
    const message = error instanceof Error ? error.message : "Binance exchange cancel-all failed";
    return reply.code(400).send({ error: message });
  }
});

server.post("/api/exchange/leverage", async (request, reply) => {
  try {
    const body = (request.body ?? {}) as { leverage?: number };
    const leverage = Number(body.leverage);
    if (!Number.isFinite(leverage) || leverage <= 0) {
      return reply.code(400).send({ error: "Positive leverage is required." });
    }
    await runtime.setExchangeLeverage(leverage);
    broadcastState();
    return publicSnapshot();
  } catch (error) {
    const message = error instanceof Error ? error.message : "Binance exchange leverage update failed";
    return reply.code(400).send({ error: message });
  }
});

server.post("/api/correlations", async (request, reply) => {
  try {
    const body = (request.body ?? {}) as { refresh?: boolean };
    const markets = await selectCorrelationMarkets(activeMarket, body.refresh === true);
    correlationService.startVector(
      activeMarket,
      markets,
      broadcastState,
      body.refresh === true,
    );
    broadcastState();
    return publicSnapshot();
  } catch (error) {
    const message = error instanceof Error ? error.message : "Correlation computation failed";
    return reply.code(400).send({ error: message });
  }
});

server.post("/api/backtest", async (request, reply) => {
  const body = (request.body ?? {}) as {
    preset?: BacktestPreset;
    source?: "candles" | "orderbook-mid";
    limit?: number;
    historicalStartTime?: number;
    startingQuote?: number;
    historicalDays?: number;
    randomSampleCount?: number;
    randomWindowDays?: number;
    randomMinWindowDays?: number;
    randomMaxWindowDays?: number;
    randomLookbackDays?: number;
    randomPairCount?: number;
  };
  const preset =
    body.preset ??
    (body.source === "orderbook-mid" ? "saved-orderbook" : "saved-candles");

  try {
    const randomPairCount =
      body.randomPairCount === undefined
        ? undefined
        : clampInt(Number(body.randomPairCount), 0, MAX_RANDOM_PAIR_COUNT);
    const randomMarkets =
      isRandomBacktestPreset(preset) && randomPairCount !== undefined && randomPairCount > 0
        ? await selectRandomBacktestMarkets(randomPairCount)
        : undefined;

    runtime.startBacktest({
      preset,
      limit: clampInt(Number(body.limit ?? 1_000), 10, 10_000),
      historicalStartTime:
        body.historicalStartTime === undefined || !Number.isFinite(Number(body.historicalStartTime))
          ? undefined
          : Number(body.historicalStartTime),
      startingQuote: body.startingQuote,
      historicalDays:
        body.historicalDays === undefined
          ? undefined
          : clampInt(Number(body.historicalDays), 1, 3650),
      randomSampleCount:
        body.randomSampleCount === undefined
          ? undefined
          : clampInt(Number(body.randomSampleCount), 1, 200),
      randomWindowDays:
        body.randomWindowDays === undefined
          ? undefined
          : clampInt(Number(body.randomWindowDays), 1, 365),
      randomMinWindowDays:
        body.randomMinWindowDays === undefined
          ? undefined
          : clampInt(Number(body.randomMinWindowDays), 1, 365),
      randomMaxWindowDays:
        body.randomMaxWindowDays === undefined
          ? undefined
          : clampInt(Number(body.randomMaxWindowDays), 1, 365),
      randomLookbackDays:
        body.randomLookbackDays === undefined
          ? undefined
          : clampInt(Number(body.randomLookbackDays), 1, 3650),
      randomPairCount: randomMarkets?.length ?? randomPairCount,
      randomMarkets,
    }, scheduleBroadcast);
    broadcastState();
    return publicSnapshot();
  } catch (error) {
    const message = error instanceof Error ? error.message : "Backtest failed";
    return reply.code(400).send({ error: message });
  }
});

server.post("/api/backtest/stop", async () => {
  runtime.stopBacktest(broadcastState);
  broadcastState();
  return publicSnapshot();
});

server.setErrorHandler((error, _request, reply) => {
  server.log.error(error);
  reply.code(500).send({
    error: error instanceof Error ? error.message : "Internal server error",
  });
});

let streamGeneration = 0;
let stream = createStream(activeMarket, streamGeneration);
let userDataStream = createUserDataStream(activeMarket, streamGeneration);

function createStream(market: BinanceMarketListing, generation: number): BinanceMarketStream {
  if (!isStreamVenue(market.venue)) {
    throw new Error(`${market.displaySymbol} is not a streamable Binance market.`);
  }

  return new BinanceMarketStream({
    symbol: market.symbol,
    venue: market.venue,
    interval: appConfig.interval,
    environment: exchangeTrading.streamEnvironmentFor(market),
    handlers: {
      onStatus: (status) => {
        if (generation !== streamGeneration) {
          return;
        }
        runtime.handleStatus(status);
        scheduleBroadcast();
      },
      onTick: async (tick) => {
        if (generation !== streamGeneration) {
          return;
        }
        const events = await runtime.handleTick(tick);
        if (events.length > 0) {
          broadcastState();
        } else {
          scheduleBroadcast();
        }
      },
      onCandle: async (candle) => {
        if (generation !== streamGeneration) {
          return;
        }
        await runtime.handleCandle(candle);
        if (correlationService.handleCandle(candle)) {
          broadcastState();
        } else {
          scheduleBroadcast();
        }
      },
      onOrderBook: async (snapshot) => {
        if (generation !== streamGeneration) {
          return;
        }
        await runtime.handleOrderBook(snapshot);
        scheduleBroadcast();
      },
    },
  });
}

function createUserDataStream(
  market: BinanceMarketListing,
  generation: number,
): BinanceExchangeUserDataStream {
  return new BinanceExchangeUserDataStream({
    market,
    exchangeTrading,
    handlers: {
      onStatus: (status) => {
        if (generation !== streamGeneration) {
          return;
        }
        exchangeTrading.updateUserDataStreamStatus(market, status);
        scheduleBroadcast();
      },
      onUserData: async (payload) => {
        if (generation !== streamGeneration) {
          return;
        }
        const events = await runtime.handleExchangeUserData(payload);
        if (generation !== streamGeneration) {
          return;
        }
        if (events.length > 0) {
          broadcastState();
        } else {
          scheduleBroadcast();
        }
      },
    },
  });
}

function normalizeExchangeOrderInput(
  input: BinanceExchangePlaceOrderInput,
): BinanceExchangePlaceOrderInput {
  const side = input.side === "sell" ? "sell" : "buy";
  const type =
    input.type === "market"
      ? "market"
      : input.type === "stop-market"
        ? "stop-market"
        : "limit";
  const quantity = Number(input.quantity);
  if (!Number.isFinite(quantity) || quantity <= 0) {
    throw new Error("Positive order quantity is required.");
  }
  const price = input.price === undefined ? undefined : Number(input.price);
  if (type === "limit" && (!Number.isFinite(price) || (price as number) <= 0)) {
    throw new Error("Positive limit order price is required.");
  }
  const stopPrice =
    input.stopPrice === undefined
      ? type === "stop-market"
        ? price
        : undefined
      : Number(input.stopPrice);
  if (type === "stop-market" && (!Number.isFinite(stopPrice) || (stopPrice as number) <= 0)) {
    throw new Error("Positive stop-market stop price is required.");
  }
  return {
    ...input,
    side,
    type,
    quantity,
    price: type === "limit" ? price : undefined,
    stopPrice,
    timeInForce: type === "limit" ? input.timeInForce ?? "GTC" : undefined,
  };
}

function createStorage(market: BinanceMarketListing): TradingStorage {
  return new TradingStorage(
    appConfig.dataDir,
    getMarketStorageKey(market),
    market.symbol,
    appConfig.interval,
  );
}

async function resolveInitialMarket(): Promise<BinanceMarketListing> {
  try {
    return (await marketCatalog.find(appConfig.market.id)) ?? appConfig.market;
  } catch (error) {
    server.log.warn(
      error instanceof Error ? error.message : "Initial Binance market lookup failed.",
    );
    return appConfig.market;
  }
}

async function selectRandomBacktestMarkets(
  count: number,
): Promise<HistoricalBacktestMarket[]> {
  const activeMarket = runtime.snapshot().market;
  const catalog = await marketCatalog.list();
  const candidates = catalog.markets.filter(
    (market) =>
      market.id !== activeMarket.id &&
      market.group === activeMarket.group &&
      market.venue === activeMarket.venue &&
      market.quoteAsset === activeMarket.quoteAsset &&
      market.supportsHistoricalCandles &&
      market.supportsLiveStream &&
      isStreamVenue(market.venue),
  );
  const selected = shuffle(candidates).slice(0, count);

  if (selected.length === 0) {
    throw new Error(
      `No eligible random ${activeMarket.quoteAsset} pairs found for ${activeMarket.displaySymbol}.`,
    );
  }

  return selected.map(toHistoricalBacktestMarket);
}

async function selectCorrelationMarkets(
  market: BinanceMarketListing,
  forceRefresh = false,
): Promise<BinanceMarketListing[]> {
  if (!market.supportsHistoricalCandles || !isStreamVenue(market.venue)) {
    throw new Error(`${market.displaySymbol} does not support candle correlations.`);
  }

  const catalog = await marketCatalog.list(forceRefresh);
  const candidates = catalog.markets.filter(
    (item) =>
      item.quoteAsset === market.quoteAsset &&
      item.supportsHistoricalCandles &&
      item.supportsLiveStream &&
      isStreamVenue(item.venue) &&
      isCorrelationMarketGroup(item.group),
  );

  if (!candidates.some((item) => item.id === market.id)) {
    candidates.unshift(market);
  }

  return candidates;
}

function isCorrelationMarketGroup(group: BinanceMarketListing["group"]): boolean {
  return (
    group === "spot" ||
    group === "bstocks" ||
    group === "futures" ||
    group === "tradfi" ||
    group === "options"
  );
}

function toHistoricalBacktestMarket(
  market: BinanceMarketListing,
): HistoricalBacktestMarket {
  if (!isStreamVenue(market.venue)) {
    throw new Error(`${market.displaySymbol} is not a candle backtest market.`);
  }

  return {
    marketId: market.id,
    marketKey: getMarketStorageKey(market),
    venue: market.venue,
    symbol: market.symbol,
    displaySymbol: market.displaySymbol,
    baseAsset: market.baseAsset,
    quoteAsset: market.quoteAsset,
    maxLeverage: market.maxLeverage,
  };
}

function isRandomBacktestPreset(preset: BacktestPreset): boolean {
  return preset === "random-windows" || preset === "random-length-windows";
}

function shuffle<T>(items: T[]): T[] {
  const shuffled = [...items];
  for (let index = shuffled.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(Math.random() * (index + 1));
    [shuffled[index], shuffled[swapIndex]] = [shuffled[swapIndex], shuffled[index]];
  }
  return shuffled;
}

function fallbackMarketCatalog(
  market: BinanceMarketListing,
  warning: string,
): BinanceMarketCatalogSnapshot {
  const markets = [market];
  return {
    markets,
    counts: countCatalogGroups(markets),
    sources: [
      {
        source: market.venue,
        status: "failed",
        count: 0,
        message: warning,
      },
    ],
    warnings: [warning],
    refreshedAt: Date.now(),
  };
}

function countCatalogGroups(
  markets: readonly BinanceMarketListing[],
): Record<MarketGroup, number> {
  return {
    spot: 0,
    bstocks: 0,
    futures: 0,
    tradfi: 0,
    options: 0,
    predictions: 0,
    ...markets.reduce<Partial<Record<MarketGroup, number>>>((counts, market) => {
      counts[market.group] = (counts[market.group] ?? 0) + 1;
      return counts;
    }, {}),
  };
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function configureProcessReports(): void {
  if (!process.report) {
    return;
  }

  try {
    fs.mkdirSync(appConfig.dataDir, { recursive: true });
    process.report.directory = appConfig.dataDir;
    process.report.reportOnFatalError = true;
    process.report.reportOnSignal = true;
    process.report.reportOnUncaughtException = true;
    process.report.signal = "SIGUSR2";
  } catch (error) {
    server.log.warn(
      {
        error: error instanceof Error ? error.message : "Unknown process report setup error",
      },
      "Process report setup failed",
    );
  }
}

let broadcastTimer: NodeJS.Timeout | undefined;
let broadcastImmediate: NodeJS.Immediate | undefined;
let broadcastRunning = false;
let broadcastDirty = false;
let lastBroadcastAt = 0;
let broadcastTimerDueAt = 0;
let snapshotSeq = 0;

const address = await server.listen({
  host: appConfig.host,
  port: appConfig.port,
});

const sockets = new Set<WebSocket>();
const socketAlive = new WeakMap<WebSocket, boolean>();
const wss = new WebSocketServer({
  server: server.server,
  path: "/ws",
});
const snapshotSource = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
const MAX_SOCKET_BUFFER_BYTES = 1_000_000;
const DASHBOARD_BROADCAST_MIN_INTERVAL_MS = 500;

wss.on("connection", (socket) => {
  sockets.add(socket);
  socketAlive.set(socket, true);
  server.log.info({ clients: sockets.size }, "Dashboard websocket connected");
  sendSnapshot(socket, publicSnapshot());
  socket.on("pong", () => {
    socketAlive.set(socket, true);
  });
  socket.on("close", (code, reason) => {
    sockets.delete(socket);
    socketAlive.delete(socket);
    server.log.info(
      {
        clients: sockets.size,
        code,
        reason: reason.toString(),
      },
      "Dashboard websocket closed",
    );
  });
  socket.on("error", (error) => {
    sockets.delete(socket);
    socketAlive.delete(socket);
    server.log.warn({ error: error.message }, "Dashboard websocket error");
  });
});

const dashboardSocketHeartbeat = setInterval(() => {
  for (const socket of sockets) {
    if (socket.readyState !== socket.OPEN) {
      sockets.delete(socket);
      socketAlive.delete(socket);
      continue;
    }

    if (socketAlive.get(socket) === false) {
      sockets.delete(socket);
      socketAlive.delete(socket);
      server.log.warn({ clients: sockets.size }, "Dashboard websocket heartbeat missed");
      socket.terminate();
      continue;
    }

    socketAlive.set(socket, false);
    socket.ping();
  }
}, DASHBOARD_WS_HEARTBEAT_MS);
dashboardSocketHeartbeat.unref();

stream.start();
userDataStream.start();
server.log.info(`Trading server listening at ${address}`);
server.log.info(`Dashboard websocket available at ws://localhost:${appConfig.port}/ws`);

const heartbeatLogTimer =
  HEARTBEAT_LOG_MS > 0
    ? setInterval(() => {
        const diagnostics = diagnosticsSnapshot();
        server.log.info(
          {
            uptimeSec: diagnostics.uptimeSec,
            tcpConnections: diagnostics.connections.tcp,
            dashboardWebSockets: diagnostics.connections.dashboardWebSockets,
            eventLoopDelayMaxMs: diagnostics.eventLoopDelayMs.max,
            eventLoopDelayP99Ms: diagnostics.eventLoopDelayMs.p99,
            rssBytes: diagnostics.memory.rss,
            correlationStatus: diagnostics.correlations.status,
            correlationMessage: diagnostics.correlations.message,
          },
          "Trading server heartbeat",
        );
      }, HEARTBEAT_LOG_MS)
    : undefined;
heartbeatLogTimer?.unref();

function scheduleBroadcast(): void {
  queueBroadcast(500);
}

function broadcastState(): void {
  queueBroadcast(0);
}

function queueBroadcast(delayMs: number): void {
  if (sockets.size === 0) {
    return;
  }

  broadcastDirty = true;
  const now = Date.now();
  const minimumDelay = Math.max(0, lastBroadcastAt + DASHBOARD_BROADCAST_MIN_INTERVAL_MS - now);
  const effectiveDelayMs = Math.max(delayMs, minimumDelay);

  if (effectiveDelayMs <= 0) {
    if (broadcastTimer) {
      clearTimeout(broadcastTimer);
      broadcastTimer = undefined;
      broadcastTimerDueAt = 0;
    }
    queueBroadcastFlush();
    return;
  }

  const dueAt = now + effectiveDelayMs;
  if (broadcastTimer || broadcastImmediate) {
    if (broadcastTimer && broadcastTimerDueAt > dueAt) {
      clearTimeout(broadcastTimer);
      broadcastTimer = undefined;
      broadcastTimerDueAt = 0;
    } else {
      return;
    }
  }

  if (broadcastImmediate) {
    return;
  }

  broadcastTimer = setTimeout(() => {
    broadcastTimer = undefined;
    broadcastTimerDueAt = 0;
    queueBroadcastFlush();
  }, effectiveDelayMs);
  broadcastTimerDueAt = dueAt;
}

function queueBroadcastFlush(): void {
  if (broadcastImmediate) {
    return;
  }

  broadcastImmediate = setImmediate(flushBroadcast);
}

function flushBroadcast(): void {
  broadcastImmediate = undefined;
  if (broadcastRunning) {
    broadcastDirty = true;
    return;
  }
  if (!broadcastDirty || sockets.size === 0) {
    broadcastDirty = false;
    return;
  }

  broadcastRunning = true;
  broadcastDirty = false;
  void Promise.resolve()
    .then(() => {
      const snapshot = publicSnapshot();
      const message = JSON.stringify({
        type: "snapshot",
        sequence: snapshot.snapshotSeq,
        sentAt: Date.now(),
        payload: snapshot,
      });
      for (const socket of sockets) {
        if (socket.readyState !== socket.OPEN) {
          continue;
        }
        if (socket.bufferedAmount > MAX_SOCKET_BUFFER_BYTES) {
          continue;
        }
        sendMessage(socket, message);
      }
      lastBroadcastAt = Date.now();
    })
    .catch((error) => {
      server.log.warn(
        error instanceof Error ? error.message : "Dashboard broadcast failed.",
      );
    })
    .finally(() => {
      broadcastRunning = false;
      if (broadcastDirty && sockets.size > 0) {
        queueBroadcast(0);
      }
    });
}

function sendSnapshot(socket: WebSocket, snapshot: ReturnType<typeof publicSnapshot>): void {
  sendMessage(
    socket,
    JSON.stringify({
      type: "snapshot",
      sequence: snapshot.snapshotSeq,
      sentAt: Date.now(),
      payload: snapshot,
    }),
  );
}

function sendMessage(socket: WebSocket, message: string): void {
  try {
    socket.send(message, (error) => {
      if (error) {
        sockets.delete(socket);
        server.log.warn({ error: error.message }, "Dashboard websocket send failed");
        socket.close();
      }
    });
  } catch {
    sockets.delete(socket);
    socket.close();
  }
}

function publicSnapshot() {
  snapshotSeq += 1;
  const snapshot = runtime.snapshot();
  return {
    ...snapshot,
    snapshotSource,
    snapshotSeq,
    snapshotAt: Date.now(),
    correlations: correlationService.snapshotForMarket(snapshot.market.id),
  };
}

function diagnosticsSnapshot() {
  const elu = performance.eventLoopUtilization();
  const delayStats = eventLoopDelayStats();
  return {
    ok: true,
    at: Date.now(),
    pid: process.pid,
    uptimeSec: process.uptime(),
    environment: appConfig.environment,
    market: activeMarket.id,
    symbol: activeMarket.symbol,
    memory: process.memoryUsage(),
    cpuUsage: process.cpuUsage(),
    eventLoopUtilization: elu,
    eventLoopDelayMs: delayStats,
    connections: {
      tcp: tcpConnectionCount,
      dashboardWebSockets: sockets.size,
      dashboardWebSocketStates: countSocketStates(),
    },
    broadcast: {
      running: broadcastRunning,
      dirty: broadcastDirty,
      timerActive: Boolean(broadcastTimer),
      timerDueAt: broadcastTimerDueAt || undefined,
      immediateActive: Boolean(broadcastImmediate),
      lastBroadcastAt: lastBroadcastAt || undefined,
      snapshotSeq,
    },
    correlations: correlationService.snapshotForMarket(activeMarket.id),
  };
}

function eventLoopDelayStats() {
  return {
    min: nanosecondsToMilliseconds(eventLoopDelay.min),
    mean: nanosecondsToMilliseconds(eventLoopDelay.mean),
    max: nanosecondsToMilliseconds(eventLoopDelay.max),
    p95: nanosecondsToMilliseconds(eventLoopDelay.percentile(95)),
    p99: nanosecondsToMilliseconds(eventLoopDelay.percentile(99)),
  };
}

function nanosecondsToMilliseconds(value: number): number {
  return Number.isFinite(value) ? value / 1_000_000 : 0;
}

function countSocketStates(): Record<string, number> {
  const counts: Record<string, number> = {
    connecting: 0,
    open: 0,
    closing: 0,
    closed: 0,
    other: 0,
  };
  for (const socket of sockets) {
    if (socket.readyState === socket.CONNECTING) {
      counts.connecting += 1;
    } else if (socket.readyState === socket.OPEN) {
      counts.open += 1;
    } else if (socket.readyState === socket.CLOSING) {
      counts.closing += 1;
    } else if (socket.readyState === socket.CLOSED) {
      counts.closed += 1;
    } else {
      counts.other += 1;
    }
  }
  return counts;
}

function clampInt(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) {
    return min;
  }

  return Math.max(min, Math.min(max, Math.round(value)));
}

async function shutdown(): Promise<void> {
  streamGeneration += 1;
  if (heartbeatLogTimer) {
    clearInterval(heartbeatLogTimer);
  }
  clearInterval(dashboardSocketHeartbeat);
  stream.stop();
  await correlationService.flush();
  correlationService.stop();
  await runtime.flushState();
  await server.close();
}

process.on("SIGINT", () => {
  void shutdown().then(() => process.exit(0));
});

process.on("SIGTERM", () => {
  void shutdown().then(() => process.exit(0));
});
