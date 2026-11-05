import type { Immutable } from "@trading/bot-algo/trading-api";
import { z } from "zod";

export const SymbolSchema = z
  .string()
  .min(1)
  .max(32)
  .regex(/^[A-Z0-9_-]+$/);
export const NetworkSchema = z.enum(["live", "demo", "testnet"]);

const CredentialSchema = z
  .string()
  .min(1)
  .refine((value) => value.trim().length > 0, "Credential must not be blank");
export const CredentialsSchema = z.strictObject({
  apiKey: CredentialSchema,
  apiSecret: CredentialSchema,
});

export const ApiErrorResponseSchema = z.object({
  statusCode: z.int().min(400).max(599),
  error: z.string(),
  message: z.string(),
});

export const HealthResponseSchema = z.object({ status: z.literal("ok") });

export const BotStartRequestSchema = z.strictObject({ symbol: SymbolSchema });
export const BotStartResponseSchema = z.object({
  symbol: SymbolSchema,
  status: z.literal("running"),
});

export const BotStopRequestSchema = z.strictObject({ symbol: SymbolSchema });
export const BotStopResponseSchema = z.object({
  symbol: SymbolSchema,
  status: z.literal("stopped"),
});

export const BotResetRequestSchema = z.strictObject({ symbol: SymbolSchema });
export const BotResetResponseSchema = z.object({
  symbol: SymbolSchema,
  status: z.literal("reset"),
});

const PositionSideSchema = z.enum(["long", "short"]);
export const BotOpenPositionRequestSchema = z.strictObject({
  symbol: SymbolSchema,
  side: PositionSideSchema,
  quantity: z.number().positive(),
  lifecycle: z
    .strictObject({
      lifetimeMs: z.number().int().positive().optional(),
      stopLossPrice: z.number().positive().optional(),
      takeProfitPrice: z.number().positive().optional(),
    })
    .optional(),
});
export const BotClosePositionRequestSchema = z.strictObject({
  symbol: SymbolSchema,
  side: PositionSideSchema,
  quantity: z.number().positive(),
  positionId: z.string().trim().min(1).optional(),
});
// Acceptance acknowledges the control request; execution/fills arrive separately.
export const BotPositionResponseSchema = z.object({
  symbol: SymbolSchema,
  status: z.literal("accepted"),
});

export const BotConfigureRequestSchema = z.strictObject({
  network: NetworkSchema,
  credentials: CredentialsSchema,
});
export const BotConfigureResponseSchema = z.object({
  network: NetworkSchema,
  status: z.literal("configured"),
});

// Persisted records use the same configuration contract as the API.
export const BotSettingsSchema = z.strictObject({
  id: z.literal(1),
  configuration: BotConfigureRequestSchema,
  updatedAt: z.int().nonnegative(),
});
export const BotStateSchema = z.strictObject({
  symbol: SymbolSchema,
  /** Desired activity; this does not imply a bot process is currently running. */
  active: z.boolean(),
  updatedAt: z.int().nonnegative(),
});
export type BotSettings = Immutable<z.infer<typeof BotSettingsSchema>>;
export type BotState = Immutable<z.infer<typeof BotStateSchema>>;

export const ErrorResponseSchemas = {
  "4xx": ApiErrorResponseSchema,
  "5xx": ApiErrorResponseSchema,
} as const;

export const HealthRouteSchema = {
  response: { 200: HealthResponseSchema, ...ErrorResponseSchemas },
};
export const BotStartRouteSchema = {
  body: BotStartRequestSchema,
  response: { 200: BotStartResponseSchema, ...ErrorResponseSchemas },
};
export const BotStopRouteSchema = {
  body: BotStopRequestSchema,
  response: { 200: BotStopResponseSchema, ...ErrorResponseSchemas },
};
export const BotConfigureRouteSchema = {
  body: BotConfigureRequestSchema,
  response: { 200: BotConfigureResponseSchema, ...ErrorResponseSchemas },
};

export const BotResetRouteSchema = {
  body: BotResetRequestSchema,
  response: { 200: BotResetResponseSchema, ...ErrorResponseSchemas },
};
export const BotOpenPositionRouteSchema = {
  body: BotOpenPositionRequestSchema,
  response: { 200: BotPositionResponseSchema, ...ErrorResponseSchemas },
};
export const BotClosePositionRouteSchema = {
  body: BotClosePositionRequestSchema,
  response: { 200: BotPositionResponseSchema, ...ErrorResponseSchemas },
};

export type ApiErrorResponse = Immutable<
  z.infer<typeof ApiErrorResponseSchema>
>;
export type HealthResponse = Immutable<z.infer<typeof HealthResponseSchema>>;
export type BotStartRequest = Immutable<z.infer<typeof BotStartRequestSchema>>;
export type BotStartResponse = Immutable<
  z.infer<typeof BotStartResponseSchema>
>;
export type BotStopRequest = Immutable<z.infer<typeof BotStopRequestSchema>>;
export type BotStopResponse = Immutable<z.infer<typeof BotStopResponseSchema>>;
export type BotConfigureRequest = Immutable<
  z.infer<typeof BotConfigureRequestSchema>
>;
export type BotConfigureResponse = Immutable<
  z.infer<typeof BotConfigureResponseSchema>
>;
export type BotResetResponse = Immutable<
  z.infer<typeof BotResetResponseSchema>
>;
export type BotOpenPositionRequest = Immutable<
  z.infer<typeof BotOpenPositionRequestSchema>
>;
export type BotClosePositionRequest = Immutable<
  z.infer<typeof BotClosePositionRequestSchema>
>;
export type BotPositionResponse = Immutable<
  z.infer<typeof BotPositionResponseSchema>
>;
