import fs from "node:fs";
import Fastify from "fastify";
import cors from "@fastify/cors";
import staticFiles from "@fastify/static";
import { WebSocketServer, type WebSocket } from "ws";
import type { BacktestPreset, ManualTradeInput, StrategyConfig } from "@trading/bot-algo";
import { appConfig } from "./config.js";
import { BinanceMarketStream } from "./binance-stream.js";
import {
  BinanceMarketCatalog,
  getMarketStorageKey,
  isStreamVenue,
  type BinanceMarketListing,
} from "./binance-markets.js";
import { TradingRuntime } from "./runtime.js";
import { TradingStorage } from "./storage.js";
import type { HistoricalBacktestMarket } from "./historical-backtest.js";
import { CorrelationService } from "./correlation-service.js";

const server = Fastify({
  logger: {
    level: process.env.LOG_LEVEL ?? "info",
  },
});

const MAX_RANDOM_PAIR_COUNT = 25;

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
  market: appConfig.market.id,
  symbol: appConfig.market.symbol,
  dataDir: appConfig.dataDir,
}));

server.get("/api/state", async () => publicSnapshot());

server.get("/api/markets", async (request) => {
  const query = request.query as { refresh?: string };
  return marketCatalog.list(query.refresh === "1" || query.refresh === "true");
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
  await runtime.switchMarket(market, createStorage(market));
  activeMarket = market;
  stream = createStream(market, streamGeneration);
  stream.start();
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

server.put("/api/bot/config", async (request) => {
  const body = (request.body ?? {}) as Partial<StrategyConfig>;
  await runtime.updateBotConfig(body);
  broadcastState();
  return publicSnapshot();
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

function createStream(market: BinanceMarketListing, generation: number): BinanceMarketStream {
  if (!isStreamVenue(market.venue)) {
    throw new Error(`${market.displaySymbol} is not a streamable Binance market.`);
  }

  return new BinanceMarketStream({
    symbol: market.symbol,
    venue: market.venue,
    interval: appConfig.interval,
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

const address = await server.listen({
  host: appConfig.host,
  port: appConfig.port,
});

const sockets = new Set<WebSocket>();
const wss = new WebSocketServer({
  server: server.server,
  path: "/ws",
});

wss.on("connection", (socket) => {
  sockets.add(socket);
  socket.send(
    JSON.stringify({
      type: "snapshot",
      payload: publicSnapshot(),
    }),
  );
  socket.on("close", () => sockets.delete(socket));
});

stream.start();
server.log.info(`Trading server listening at ${address}`);
server.log.info(`Dashboard websocket available at ws://localhost:${appConfig.port}/ws`);

let broadcastTimer: NodeJS.Timeout | undefined;

function scheduleBroadcast(): void {
  if (broadcastTimer) {
    return;
  }

  broadcastTimer = setTimeout(() => {
    broadcastTimer = undefined;
    broadcastState();
  }, 500);
}

function broadcastState(): void {
  if (sockets.size === 0) {
    return;
  }

  const message = JSON.stringify({
    type: "snapshot",
    payload: publicSnapshot(),
  });

  for (const socket of sockets) {
    if (socket.readyState === socket.OPEN) {
      socket.send(message);
    }
  }
}

function publicSnapshot() {
  const snapshot = runtime.snapshot();
  return {
    ...snapshot,
    correlations: correlationService.snapshotForMarket(snapshot.market.id),
  };
}

function clampInt(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) {
    return min;
  }

  return Math.max(min, Math.min(max, Math.round(value)));
}

async function shutdown(): Promise<void> {
  streamGeneration += 1;
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
