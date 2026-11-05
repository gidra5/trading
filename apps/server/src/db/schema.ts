import { sql } from "drizzle-orm";
import { check, sqliteTable } from "drizzle-orm/sqlite-core";
import { BotSettingsSchema, BotStateSchema } from "../schemas.js";
import { zodBoolean, zodInteger, zodJson, zodText } from "./columns.js";

const settings = BotSettingsSchema.shape;
export const botSettings = sqliteTable(
  "bot_settings",
  {
    id: zodInteger(settings.id)("id").primaryKey().notNull(),
    configuration: zodJson(settings.configuration)("configuration").notNull(),
    updatedAt: zodInteger(settings.updatedAt)("updated_at").notNull(),
  },
  (table) => [
    check("bot_settings_singleton", sql`${table.id} = 1`),
    check(
      "bot_settings_configuration_json",
      sql`json_valid(${table.configuration})`,
    ),
    check("bot_settings_updated_at", sql`${table.updatedAt} >= 0`),
  ],
);

const state = BotStateSchema.shape;
export const botStates = sqliteTable(
  "bot_states",
  {
    symbol: zodText(state.symbol)("symbol").primaryKey().notNull(),
    active: zodBoolean(state.active)("active").notNull(),
    updatedAt: zodInteger(state.updatedAt)("updated_at").notNull(),
  },
  (table) => [
    check("bot_states_active", sql`${table.active} IN (0, 1)`),
    check("bot_states_updated_at", sql`${table.updatedAt} >= 0`),
  ],
);
