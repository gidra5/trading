import type { FastifyPluginAsyncZod } from "fastify-type-provider-zod";
import { z } from "zod";
import type { OraclePaperService } from "./oracle-paper-service.js";

const Features = z.array(z.number()).length(59);
const Observation = z.strictObject({ features: z.tuple([Features, Features]),
  prices: z.tuple([z.number().positive(), z.number().positive()]),
  timestamps: z.tuple([z.number().int(), z.number().int()]) });
const Initial = z.strictObject({ observation: Observation,
  account: z.strictObject({ quote: z.number(), asset: z.number(), previousPrice: z.number().positive(),
    orders: z.array(z.strictObject({ side: z.enum(["buy", "sell"]), price: z.number().positive(),
      notional: z.number().positive() })).max(3) }),
  terminalSeconds: z.number().int(), consumedQuote: z.number().nonnegative(), filled: z.boolean() });

/** Dedicated paper endpoints; generic exchange bot controls retain their own semantics. */
export const oraclePaperRoutes: FastifyPluginAsyncZod<{ service: OraclePaperService }> = async (server, { service }) => {
  server.get("/bot/oracle-paper", async () => service.status());
  server.post("/bot/oracle-paper/start", { schema: { body: Initial } }, async request => service.start(request.body));
  server.post("/bot/oracle-paper/tick", { schema: { body: Observation } }, async request => service.tick(request.body));
  server.post("/bot/oracle-paper/resume", async () => service.resume());
  server.post("/bot/oracle-paper/checkpoint", async () => service.checkpoint());
  server.post("/bot/oracle-paper/stop", async () => service.stop());
  server.post("/bot/oracle-paper/reset", async () => service.reset());
};
