import fs from "node:fs";
import Fastify from "fastify";
import cors from "@fastify/cors";
import staticFiles from "@fastify/static";
import { WebSocketServer, type WebSocket } from "ws";
import type { BacktestPreset, StrategyConfig } from "@trading/bot-algo";
import { appConfig } from "./config.js";
import { BinanceMarketStream } from "./binance-stream.js";
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

const storage = new TradingStorage(
  appConfig.dataDir,
  appConfig.symbol,
  appConfig.interval,
);
const runtime = new TradingRuntime(storage, appConfig.strategy, appConfig.interval, {
  dataDir: appConfig.dataDir,
  maxBytes: appConfig.historicalCache.maxBytes,
  minFreeBytes: appConfig.historicalCache.minFreeBytes,
});
await runtime.init();

server.get("/health", async () => ({
  ok: true,
  symbol: appConfig.symbol,
  dataDir: appConfig.dataDir,
}));

server.get("/api/state", async () => runtime.snapshot());

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

server.post("/api/backtest", async (request, reply) => {
  const body = (request.body ?? {}) as {
    preset?: BacktestPreset;
    source?: "candles" | "orderbook-mid";
    limit?: number;
    startingQuote?: number;
  };
  const preset =
    body.preset ??
    (body.source === "orderbook-mid" ? "saved-orderbook" : "saved-candles");

  try {
    runtime.startBacktest({
      preset,
      limit: clampInt(Number(body.limit ?? 1_000), 10, 10_000),
      startingQuote: body.startingQuote,
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

const stream = new BinanceMarketStream(
  appConfig.streamSymbol,
  appConfig.interval,
  {
    onStatus: (status) => {
      runtime.handleStatus(status);
      scheduleBroadcast();
    },
    onTick: async (tick) => {
      const events = await runtime.handleTick(tick);
      if (events.length > 0) {
        broadcastState();
      } else {
        scheduleBroadcast();
      }
    },
    onCandle: async (candle) => {
      await runtime.handleCandle(candle);
      scheduleBroadcast();
    },
    onOrderBook: async (snapshot) => {
      await runtime.handleOrderBook(snapshot);
      scheduleBroadcast();
    },
  },
);

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
