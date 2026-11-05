import { asc, eq } from "drizzle-orm";
import {
  BotConfigureRequestSchema,
  BotSettingsSchema,
  BotStateSchema,
  SymbolSchema,
  type BotConfigureRequest,
  type BotState,
} from "../schemas.js";
import type { Database } from "./index.js";
import { botSettings, botStates } from "./schema.js";

export function createBotStore(db: Database) {
  return {
    getConfig(): BotConfigureRequest | null {
      return (
        db.select().from(botSettings).where(eq(botSettings.id, 1)).get()
          ?.configuration ?? null
      );
    },
    setConfig(input: BotConfigureRequest): BotConfigureRequest {
      const configuration = BotConfigureRequestSchema.parse(input);
      const record = BotSettingsSchema.parse({
        id: 1,
        configuration,
        updatedAt: Date.now(),
      });
      db.insert(botSettings)
        .values(record)
        .onConflictDoUpdate({
          target: botSettings.id,
          set: { configuration, updatedAt: record.updatedAt },
        })
        .run();
      return configuration;
    },
    clearConfig() {
      db.delete(botSettings).where(eq(botSettings.id, 1)).run();
    },
    getState(symbol: string): BotState | null {
      return (
        db
          .select()
          .from(botStates)
          .where(eq(botStates.symbol, SymbolSchema.parse(symbol)))
          .get() ?? null
      );
    },
    listStates(): readonly BotState[] {
      return db.select().from(botStates).orderBy(asc(botStates.symbol)).all();
    },
    setActive(symbol: string, active: boolean): BotState {
      const record = BotStateSchema.parse({
        symbol,
        active,
        updatedAt: Date.now(),
      });
      db.insert(botStates)
        .values(record)
        .onConflictDoUpdate({
          target: botStates.symbol,
          set: { active: record.active, updatedAt: record.updatedAt },
        })
        .run();
      return record;
    },
  };
}

export type BotStore = ReturnType<typeof createBotStore>;
