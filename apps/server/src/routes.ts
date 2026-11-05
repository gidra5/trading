import type { FastifyPluginAsyncZod } from "fastify-type-provider-zod";
import type { ApiContext } from "./app.js";
import {
  BotConfigureRouteSchema,
  BotStartRouteSchema,
  BotStopRouteSchema,
  BotResetRouteSchema,
  BotOpenPositionRouteSchema,
  BotClosePositionRouteSchema,
  HealthRouteSchema,
} from "./schemas.js";
import { WebClient } from "./webclient.js";
import {
  TrainingMetricsReader,
  TrainingRunNotFoundError,
} from "./training-metrics.js";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "../../..");
const trainingMetrics = new TrainingMetricsReader(repoRoot);

export const routes: FastifyPluginAsyncZod<{ context: ApiContext }> = async (
  server,
  { context },
) => {
  server.get("/health", { schema: HealthRouteSchema }, async () => ({
    status: "ok" as const,
  }));
  server.get("/api/mlp-training/metrics", async (request, reply) => {
    const query = request.query as { cursor?: string; run?: string };
    const cursor = Number(query.cursor ?? 0);
    if (!Number.isSafeInteger(cursor) || cursor < 0) {
      return reply.code(400).send({ error: "cursor must be a non-negative safe integer" });
    }
    try {
      return await trainingMetrics.read(cursor, query.run);
    } catch (error) {
      if (error instanceof TrainingRunNotFoundError) {
        return reply.code(400).send({ error: error.message });
      }
      throw error;
    }
  });
  server.post("/api/mlp-training/comparison", async (request, reply) => {
    const values = (request.body as { runs?: unknown } | undefined)?.runs;
    if (!Array.isArray(values)
        || values.length < 1
        || values.some((value) => typeof value !== "string" || !value)
        || new Set(values).size !== values.length) {
      return reply.code(400).send({ error: "runs must contain distinct run keys" });
    }
    try {
      return await trainingMetrics.compare(values as string[]);
    } catch (error) {
      if (error instanceof TrainingRunNotFoundError) {
        return reply.code(400).send({ error: error.message });
      }
      throw error;
    }
  });
  server.get("/ws", { websocket: true }, (socket, request) => {
    const client = new WebClient(socket, (error) => {
      request.log.error({ err: error }, "WebSocket client failed");
    });
    const removeClient = context.addWebClient(client);
    socket.once("close", () => {
      try {
        removeClient();
      } catch (error) {
        request.log.error({ err: error }, "WebSocket cleanup failed");
      }
    });
  });

  server.post(
    "/bot/start",
    { schema: BotStartRouteSchema },
    async (request) => {
      return context.bot.start(request.body.symbol);
    },
  );
  server.post("/bot/stop", { schema: BotStopRouteSchema }, async (request) => {
    return context.bot.stop(request.body.symbol);
  });
  server.post(
    "/bot/reset",
    { schema: BotResetRouteSchema },
    async (request) => {
      return context.bot.reset(request.body.symbol);
    },
  );
  server.post(
    "/bot/configure",
    { schema: BotConfigureRouteSchema },
    async (request) => {
      return context.bot.configure(request.body);
    },
  );

  server.post(
    "/bot/position/open",
    { schema: BotOpenPositionRouteSchema },
    async (request) => {
      return context.bot.openPosition(request.body);
    },
  );

  server.post(
    "/bot/position/close",
    { schema: BotClosePositionRouteSchema },
    async (request) => {
      return context.bot.closePosition(request.body);
    },
  );
};
