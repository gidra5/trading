import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

export const migrationsFolder = fileURLToPath(
  new URL("../../drizzle/", import.meta.url),
);
const defaultPath = fileURLToPath(
  new URL("../../../../data/trading.sqlite", import.meta.url),
);

export function databasePath(path = process.env.DATABASE_PATH ?? defaultPath) {
  if (path === ":memory:") return path;
  if (!path.trim()) throw new Error("DATABASE_PATH must not be empty");
  return resolve(path);
}
