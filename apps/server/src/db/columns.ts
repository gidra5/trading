import type { Immutable } from "@trading/bot-algo/trading-api";
import { customType } from "drizzle-orm/sqlite-core";
import type { z } from "zod";

// Zod supplies each column's TypeScript type and validates reads and writes.
export function zodText<T extends string>(schema: z.ZodType<T>) {
  return customType<{ data: T; driverData: string }>({
    dataType: () => "text",
    toDriver: (value) => schema.parse(value),
    fromDriver: (value) => schema.parse(value),
  });
}

export function zodInteger<T extends number>(schema: z.ZodType<T>) {
  return customType<{ data: T; driverData: number }>({
    dataType: () => "integer",
    toDriver: (value) => schema.parse(value),
    fromDriver: (value) => schema.parse(value),
  });
}

export function zodBoolean<T extends boolean>(schema: z.ZodType<T>) {
  return customType<{ data: T; driverData: number }>({
    dataType: () => "integer",
    toDriver: (value) => (schema.parse(value) ? 1 : 0),
    fromDriver: (value) =>
      schema.parse(value === 0 ? false : value === 1 ? true : value),
  });
}

export function zodJson<T>(schema: z.ZodType<T>) {
  return customType<{ data: Immutable<T>; driverData: string }>({
    dataType: () => "text",
    toDriver: (value) => JSON.stringify(schema.parse(value)),
    fromDriver: (value) => schema.parse(JSON.parse(value)),
  });
}
