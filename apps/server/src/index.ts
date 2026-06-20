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

const server = Fastify({
  logger: {
    level: process.env.LOG_LEVEL ?? "info",
  },
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
});
const runtime = new TradingRuntime(
  createStorage(appConfig.market),
  appConfig.market,
  appConfig.strategy,
  appConfig.interval,
  {
    dataDir: appConfig.dataDir,
    maxBytes: appConfig.historicalCache.maxBytes,
    minFreeBytes: appConfig.historicalCache.minFreeBytes,
  },
);
await runtime.init();

server.get("/health", async () => ({
  ok: true,
  market: appConfig.market.id,
  symbol: appConfig.market.symbol,
  dataDir: appConfig.dataDir,
}));

server.get("/api/state", async () => runtime.snapshot());

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
  stream = createStream(market, streamGeneration);
  stream.start();
  broadcastState();
  return runtime.snapshot();
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
  return runtime.snapshot();
});

server.post("/api/bot/stop", async () => {
  await runtime.stopBot();
  broadcastState();
  return runtime.snapshot();
});

server.post("/api/bot/reset", async () => {
  await runtime.resetBot();
  broadcastState();
  return runtime.snapshot();
});

server.put("/api/bot/config", async (request) => {
  const body = (request.body ?? {}) as Partial<StrategyConfig>;
  await runtime.updateBotConfig(body);
  broadcastState();
  return runtime.snapshot();
});

server.post("/api/bot/manual-trade", async (request, reply) => {
  try {
    const body = (request.body ?? {}) as ManualTradeInput;
    await runtime.recordManualTrade(body);
    broadcastState();
    return runtime.snapshot();
  } catch (error) {
    const message = error instanceof Error ? error.message : "Manual trade failed";
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
  };
  const preset =
    body.preset ??
    (body.source === "orderbook-mid" ? "saved-orderbook" : "saved-candles");

  try {
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
    }, broadcastState);
    broadcastState();
    return runtime.snapshot();
  } catch (error) {
    const message = error instanceof Error ? error.message : "Backtest failed";
    return reply.code(400).send({ error: message });
  }
});

server.setErrorHandler((error, _request, reply) => {
  server.log.error(error);
  reply.code(500).send({
    error: error instanceof Error ? error.message : "Internal server error",
  });
});

let streamGeneration = 0;
let stream = createStream(appConfig.market, streamGeneration);

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
        scheduleBroadcast();
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
      payload: runtime.snapshot(),
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
  const message = JSON.stringify({
    type: "snapshot",
    payload: runtime.snapshot(),
  });

  for (const socket of sockets) {
    if (socket.readyState === socket.OPEN) {
      socket.send(message);
    }
  }
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
  await runtime.flushState();
  await server.close();
}

process.on("SIGINT", () => {
  void shutdown().then(() => process.exit(0));
});

process.on("SIGTERM", () => {
  void shutdown().then(() => process.exit(0));
});
