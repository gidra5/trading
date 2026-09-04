import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { readCandleShardReferenceSync } from "../packages/storage/src/candles.js";
import { EVENT_SECOND_DYNAMICS, eventSecondDynamics } from "../packages/bot-algo/src/event-second-features.js";
export { EVENT_SECOND_DYNAMICS, eventSecondDynamics } from "../packages/bot-algo/src/event-second-features.js";

type SecondClose = { openTime: number; close: number };
export interface EventSecondObservation { availableAt: number; close: number; values: number[]; }

export function eventSecondDynamicsAt(rows: ReadonlyMap<number, EventSecondObservation>, candle: SecondClose): number[] {
  const time = candle.openTime + 60000, row = rows.get(time);
  if (!row || row.availableAt !== time || row.close !== candle.close || row.values.length !== EVENT_SECOND_DYNAMICS.length
    || row.values.some(v => !Number.isFinite(v))) throw new Error("Missing or misaligned exact-second dynamics");
  return row.values;
}

/** Cache only deterministic minute-boundary summaries of immutable 1s shards. */
export function loadEventSecondDynamics(start: number, end: number) {
  const root = path.resolve(__dirname, ".."), DAY = 86400000;
  const refs = path.join(root, "data/market/immutable/refs/candles/spot-btcusdt/btcusdt/1s");
  const cache = path.join(root, "data/market/derived/event-second-dynamics-v1");
  const rows = new Map<number, EventSecondObservation>(), missing: string[] = [], references: string[] = [];
  const unavailableTimes: number[] = [];
  const fingerprint = createHash("sha256"), started = performance.now();
  let built = 0, reused = 0;
  const schema = { contract: "event-second-dynamics-v1", names: EVENT_SECOND_DYNAMICS, completedCloses: 64,
    emaAlpha: 2 / 3, rsiAlpha: 0.5, initialization: "EMA=oldest close, RSI gain/loss=0; process the next 63 changes" };
  const codeHash = createHash("sha256").update(fs.readFileSync(__filename))
    .update(fs.readFileSync(path.join(root, "packages/bot-algo/src/event-second-features.ts"))).digest("hex");
  fs.mkdirSync(cache, { recursive: true });
  for (let day = Math.floor(start / DAY) * DAY; day < end; day += DAY) {
    const date = new Date(day).toISOString().slice(0, 10), previousDate = new Date(day - DAY).toISOString().slice(0, 10);
    const file = path.join(refs, `${date}.json`), previousFile = path.join(refs, `${previousDate}.json`);
    if (!fs.existsSync(file)) { missing.push(date); continue; }
    const sourceHash = createHash("sha256").update(JSON.stringify(schema)).update(codeHash).update(fs.readFileSync(file));
    const hasPrevious = fs.existsSync(previousFile);
    if (hasPrevious) sourceHash.update(fs.readFileSync(previousFile));
    const signature = sourceHash.digest("hex"), cached = path.join(cache, `${date}-${signature}.json`);
    let observations: EventSecondObservation[];
    if (fs.existsSync(cached)) {
      const saved = JSON.parse(fs.readFileSync(cached, "utf8"));
      if (saved.sourceHash !== signature || JSON.stringify(saved.schema) !== JSON.stringify(schema)) throw new Error("Invalid second-dynamics cache");
      observations = saved.observations; reused++;
    } else {
      const seconds = [...(hasPrevious ? readCandleShardReferenceSync(previousFile).slice(-63) : []), ...readCandleShardReferenceSync(file)];
      // A malformed or partial source candle invalidates every feature window
      // that touches it. Preserve the source; never fill it as a complete second.
      const completed = seconds.map(row => ({ openTime: row.openTime,
        close: row.closed && row.closeTime === row.openTime + 999 ? row.close : NaN }));
      observations = [];
      for (let i = 0; i < seconds.length; i++) {
        const row = seconds[i], availableAt = row.openTime + 1000;
        if (row.openTime < day || availableAt % 60000 || !row.closed || row.closeTime !== availableAt - 1) continue;
        const values = eventSecondDynamics(completed, i);
        if (values) observations.push({ availableAt, close: row.close, values });
      }
      fs.writeFileSync(cached, JSON.stringify({ schema, sourceHash: signature, observations })); built++;
    }
    let last = day;
    const available = new Set<number>();
    for (const row of observations) {
      if (!Number.isSafeInteger(row.availableAt) || row.availableAt <= last || row.availableAt > day + DAY || row.availableAt % 60000
        || !(row.close > 0) || !Number.isFinite(row.close) || row.values.length !== EVENT_SECOND_DYNAMICS.length || row.values.some(v => !Number.isFinite(v)))
        throw new Error("Malformed second-dynamics observation");
      last = row.availableAt; rows.set(row.availableAt, row); available.add(row.availableAt);
    }
    for (let time = day + 60000; time <= day + DAY; time += 60000) if (!available.has(time)) unavailableTimes.push(time);
    references.push(cached); fingerprint.update(fs.readFileSync(cached));
  }
  return { rows, missing, unavailableTimes, references, fingerprint: fingerprint.digest("hex"), built, reused, elapsedSec: (performance.now() - started) / 1000 };
}
