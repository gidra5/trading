import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { readCandleShardReferenceSync } from "@trading/storage";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const INPUT = "data/market/immutable/refs/candles/spot-btcusdt/btcusdt/1m";
const OUTPUT = "data/runtime-cache/dense-lagged-indicator-audit";
const WARMUP_START = Date.parse("2021-07-01T00:00:00.000Z");
const END = Date.parse("2026-07-25T00:00:00.000Z");
const DAY_MS = 86_400_000;

export function run(args = process.argv.slice(2)): void {
  const value = (name: string) => {
    const index = args.indexOf(name);
    return index < 0 ? undefined : args[index + 1];
  };
  const input = resolve(value("--input-dir") ?? INPUT);
  const output = resolve(value("--output-dir") ?? OUTPUT);
  fs.mkdirSync(output, { recursive: true });
  const timesFile = path.join(output, "minute-times.f64");
  const closesFile = path.join(output, "minute-closes.f64");
  fs.writeFileSync(timesFile, Buffer.alloc(0));
  fs.writeFileSync(closesFile, Buffer.alloc(0));

  const files = fs.readdirSync(input, { withFileTypes: true })
    .filter((entry) => entry.isFile() && /^\d{4}-\d{2}-\d{2}\.json$/.test(entry.name))
    .map((entry) => ({
      file: path.join(input, entry.name),
      day: entry.name.slice(0, 10),
      start: Date.parse(`${entry.name.slice(0, 10)}T00:00:00.000Z`),
    }))
    .filter((entry) => entry.start >= WARMUP_START && entry.start < END)
    .sort((left, right) => left.start - right.start);
  if (files.length === 0) throw new Error("No one-minute candle shards selected.");

  let rows = 0;
  let previousTime = Number.NaN;
  for (let index = 0; index < files.length; index += 1) {
    const entry = files[index]!;
    if (index > 0 && entry.start !== files[index - 1]!.start + DAY_MS) {
      throw new Error(`Missing daily shard before ${entry.day}.`);
    }
    const candles = readCandleShardReferenceSync(entry.file);
    if (candles.length !== 1_440) throw new Error(`${entry.day}: expected 1,440 candles, got ${candles.length}.`);
    const times = new Float64Array(candles.length);
    const closes = new Float64Array(candles.length);
    for (let row = 0; row < candles.length; row += 1) {
      const candle = candles[row]!;
      if (Number.isFinite(previousTime) && candle.openTime !== previousTime + 60_000) {
        throw new Error(`${entry.day}: discontinuity at ${new Date(candle.openTime).toISOString()}.`);
      }
      times[row] = candle.openTime;
      closes[row] = candle.close;
      previousTime = candle.openTime;
    }
    fs.appendFileSync(timesFile, Buffer.from(times.buffer));
    fs.appendFileSync(closesFile, Buffer.from(closes.buffer));
    rows += candles.length;
    if ((index + 1) % 200 === 0) console.error(`Indicator axis: ${index + 1}/${files.length} days`);
  }

  const manifest = {
    version: 1,
    generatedAt: new Date().toISOString(),
    source: path.relative(repoRoot, input).replaceAll("\\", "/"),
    warmupStart: new Date(WARMUP_START).toISOString(),
    endExclusive: new Date(END).toISOString(),
    rows,
    stepMs: 60_000,
    files: {
      times: path.basename(timesFile),
      closes: path.basename(closesFile),
    },
  };
  fs.writeFileSync(path.join(output, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
  console.log(`Wrote ${path.relative(repoRoot, output)} (${rows.toLocaleString()} minutes)`);
}

function resolve(value: string): string {
  return path.isAbsolute(value) ? value : path.resolve(repoRoot, value);
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) run();
