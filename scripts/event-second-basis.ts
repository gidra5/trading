import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { EVENT_SECOND_INPUTS, type EventCandle } from "../packages/bot-algo/src/event-distribution.js";

function lowerBound(values: Float64Array, value: number): number {
  let lo = 0, hi = values.length;
  while (lo < hi) { const mid = (lo + hi) >>> 1; if (values[mid] < value) lo = mid + 1; else hi = mid; }
  return lo;
}

/** Read the existing rotating-second cache, joining only completed observations.
 * These five coordinates have at most 60s of input support; no recursively
 * carried indicator state crosses the research runner's one-day purge. */
export class EventSecondBasis {
  private readonly times: Float64Array;
  private readonly width: number;
  private readonly columns: number[];
  private readonly featureFile: string;
  readonly fingerprint: string;

  constructor(directory: string) {
    const manifestBytes = fs.readFileSync(path.join(directory, "manifest.json"));
    const manifest = JSON.parse(manifestBytes.toString("utf8"));
    const dataset = manifest.datasets.find((d: { id: string }) => d.id === "1s");
    if (!dataset) throw new Error("No one-second feature dataset");
    this.width = dataset.featureCount;
    this.columns = EVENT_SECOND_INPUTS.map(id => dataset.features.findIndex((f: { id: string }) => f.id === id));
    if (this.columns.some(i => i < 0)) throw new Error("Incomplete one-second feature basis");
    const timeBytes = fs.readFileSync(path.join(directory, dataset.files.times));
    // Copy avoids assumptions about Buffer allocation alignment.
    this.times = new Float64Array(timeBytes.buffer.slice(timeBytes.byteOffset, timeBytes.byteOffset + timeBytes.byteLength));
    this.featureFile = path.join(directory, dataset.files.features);
    if (this.times.length !== dataset.rows || fs.statSync(this.featureFile).size !== dataset.rows * this.width * 4)
      throw new Error("One-second feature dataset dimensions changed");
    for (let i = 0; i < this.times.length; i++) if (!Number.isFinite(this.times[i])
      || (i && this.times[i] <= this.times[i - 1])) throw new Error("Unordered one-second feature timestamps");
    this.fingerprint = createHash("sha256").update(manifestBytes).update(timeBytes).update(fs.readFileSync(this.featureFile)).digest("hex");
  }

  attach(candles: EventCandle[]): void {
    if (!candles.length) return;
    const first = Math.max(0, lowerBound(this.times, candles[0].openTime) - 1);
    const last = lowerBound(this.times, candles.at(-1)!.openTime + 60_000);
    const buffer = Buffer.alloc((last - first) * this.width * 4);
    const fd = fs.openSync(this.featureFile, "r");
    try {
      let read = 0;
      while (read < buffer.length) {
        const n = fs.readSync(fd, buffer, read, buffer.length - read, first * this.width * 4 + read);
        if (!n) throw new Error("Truncated one-second feature dataset");
        read += n;
      }
    } finally { fs.closeSync(fd); }
    let row = first - 1;
    for (const candle of candles) {
      const decisionTime = candle.openTime + 60_000;
      while (row + 1 < last && this.times[row + 1] + 1_000 <= decisionTime) row++;
      delete candle.secondBasis;
      if (row < first || decisionTime - (this.times[row] + 1_000) >= 60_000) continue;
      candle.secondBasis = { availableAt: this.times[row] + 1_000,
        values: this.columns.map(column => buffer.readFloatLE(((row - first) * this.width + column) * 4)) };
    }
  }
}
