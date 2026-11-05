import { STATUS_CODES } from "node:http";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import websocket from "@fastify/websocket";
import Fastify, { type FastifyServerOptions } from "fastify";
import {
  serializerCompiler,
  validatorCompiler,
  type ZodTypeProvider,
} from "fastify-type-provider-zod";
import {
  BotControlsUnavailableError,
  createContext,
  type ApiContext,
} from "./app.js";
import { openDatabase } from "./db/index.js";
import { createBotStore } from "./db/bot-store.js";
import { routes } from "./routes.js";
import { OraclePaperService, type OraclePaperOptions } from "./oracle-paper-service.js";
import { oraclePaperRoutes } from "./oracle-paper-routes.js";

export async function createServer(
  options: FastifyServerOptions & { databasePath?: string; oraclePaper?: OraclePaperOptions } = {},
  context?: ApiContext,
) {
  const { databasePath, oraclePaper, ...fastifyOptions } = options;
  const server = Fastify(fastifyOptions).withTypeProvider<ZodTypeProvider>();
  server.setValidatorCompiler(validatorCompiler);
  server.setSerializerCompiler(serializerCompiler);

  server.setErrorHandler((error, request, reply) => {
    const errorStatus =
      error instanceof Error && "statusCode" in error
        ? error.statusCode
        : undefined;
    const statusCode =
      typeof errorStatus === "number" &&
      Number.isInteger(errorStatus) &&
      errorStatus >= 400 &&
      errorStatus <= 599
        ? errorStatus
        : 500;
    if (statusCode >= 500)
      request.log.error({ err: error }, "API request failed");
    return reply.code(statusCode).send({
      statusCode,
      error: STATUS_CODES[statusCode] ?? "Error",
      message:
        statusCode >= 500 && !(error instanceof BotControlsUnavailableError)
          ? "Internal Server Error"
          : error instanceof Error
            ? error.message
            : "Request failed",
    });
  });

  const database = context ? undefined : openDatabase(databasePath);
  if (database) server.addHook("onClose", async () => database.close());
  try {
    const apiContext = context ?? createContext(createBotStore(database!.db));
    await server.register(websocket);
    await server.register(routes, { context: apiContext });
    if (oraclePaper) {
      const service = new OraclePaperService(oraclePaper);
      server.addHook("onClose", async () => service.close());
      await server.register(oraclePaperRoutes, { service });
    }
    return server;
  } catch (error) {
    await server.close();
    database?.close();
    throw error;
  }
}

async function startServer(): Promise<void> {
  const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "../../..");
  const configPath = process.env.ORACLE_DENSE_CONFIG;
  const server = await createServer({ logger: true, ...(configPath ? { oraclePaper: {
    symbol: process.env.ORACLE_DENSE_SYMBOL ?? "BTCUSDT",
    worker: { pythonExecutable: process.env.ORACLE_DENSE_PYTHON ?? resolve(repoRoot,
      process.platform === "win32" ? ".venv-ml/Scripts/python.exe" : ".venv-ml/bin/python"),
      configPath: resolve(repoRoot, configPath), cwd: repoRoot,
      stateDirectory: resolve(repoRoot, process.env.ORACLE_DENSE_STATE ?? "data/paper/oracle-dense") },
  } } : {}) });

  async function shutdown(signal: NodeJS.Signals): Promise<void> {
    server.log.info({ signal }, "Shutting down server");
    try {
      await server.close();
    } catch (error) {
      server.log.error(error);
      process.exitCode = 1;
    }
  }

  process.once("SIGINT", () => void shutdown("SIGINT"));
  process.once("SIGTERM", () => void shutdown("SIGTERM"));

  try {
    const host = process.env.HOST ?? "0.0.0.0";
    const port = Number(process.env.PORT ?? 3001);

    if (!Number.isInteger(port) || port < 1 || port > 65535) {
      throw new Error("PORT must be an integer between 1 and 65535");
    }

    await server.listen({ host, port });
  } catch (error) {
    server.log.error(error);
    await server.close();
    process.exitCode = 1;
  }
}

// Imports can construct a server for tests without starting a listener or signal handlers.
if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(resolve(process.argv[1])).href
) {
  await startServer();
}
