import { defineConfig } from "drizzle-kit";
import { fileURLToPath } from "node:url";
import { relative } from "node:path";
import { databasePath, migrationsFolder } from "./src/db/paths.js";

export default defineConfig({
  dialect: "sqlite",
  schema: fileURLToPath(
    new URL("./src/db/schema.ts", import.meta.url),
  ).replaceAll("\\", "/"),
  out: relative(process.cwd(), migrationsFolder).replaceAll("\\", "/"),
  dbCredentials: { url: databasePath() },
});
