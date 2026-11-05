import { closeSync, mkdirSync, openSync } from "node:fs";
import { dirname } from "node:path";
import SQLite from "better-sqlite3";
import { drizzle } from "drizzle-orm/better-sqlite3";
import { migrate } from "drizzle-orm/better-sqlite3/migrator";
import * as schema from "./schema.js";
import { databasePath, migrationsFolder } from "./paths.js";

/** Opens the local database and applies committed migrations before it is used. */
export function openDatabase(filename?: string) {
  const path = databasePath(filename);
  if (path !== ":memory:") {
    mkdirSync(dirname(path), { recursive: true, mode: 0o700 });
    closeSync(openSync(path, "a", 0o600));
  }
  const sqlite = new SQLite(path);
  try {
    sqlite.pragma("busy_timeout = 5000");
    sqlite.pragma("journal_mode = WAL");
    sqlite.pragma("foreign_keys = ON");
    sqlite.pragma("synchronous = FULL");
    const db = drizzle(sqlite, { schema });
    migrate(db, { migrationsFolder });
    return {
      db,
      path,
      close() {
        if (sqlite.open) sqlite.close();
      },
    };
  } catch (error) {
    sqlite.close();
    throw error;
  }
}

export type Database = ReturnType<typeof openDatabase>["db"];
export { botSettings, botStates } from "./schema.js";
