import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const webRoot = resolve(__dirname, "..");

const runtimeApiUrl =
  process.env.TRADING_API_URL ||
  process.env.API_URL ||
  process.env.BACKEND_URL ||
  process.env.VITE_API_URL ||
  "";

const outputDirs = [resolve(webRoot, "dist")];

for (const dir of outputDirs) {
  mkdirSync(dir, { recursive: true });
  writeFileSync(resolve(dir, "config.json"), JSON.stringify({ apiUrl: runtimeApiUrl }), "utf8");
}
