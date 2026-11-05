/** Storage arithmetic uses decimal bytes and explicit observation support. */
export const SECONDS_PER_DAY = 86_400;
export const MONTH_SECONDS = 30 * SECONDS_PER_DAY;

export type Interval = Readonly<{ startMs: number; endMs: number }>;
export type RateClass = "quiet" | "medium" | "busy" | "very-busy";
export type StreamBudget = Readonly<{
  source: string;
  recordsPerSecond: number;
  uncompressedBytesPerRecord: number;
  compressedBytesPerRecord: number;
  evidence: string;
}>;

export function rateClass(fillsPerSecond: number): RateClass {
  if (!Number.isFinite(fillsPerSecond) || fillsPerSecond < 0) throw new Error("Invalid fill rate");
  return fillsPerSecond < .1 ? "quiet" : fillsPerSecond < 1 ? "medium" : fillsPerSecond < 10 ? "busy" : "very-busy";
}

/** Merge support before summing, so overlapping/replayed sessions never add time twice. */
export function unionMilliseconds(intervals: readonly Interval[]): number {
  const sorted = intervals.map((interval) => {
    if (!Number.isFinite(interval.startMs) || !Number.isFinite(interval.endMs) || interval.endMs < interval.startMs) {
      throw new Error("Invalid observation interval");
    }
    return interval;
  }).sort((a, b) => a.startMs - b.startMs);
  let total = 0, start = 0, end = 0, open = false;
  for (const interval of sorted) {
    if (!open) { start = interval.startMs; end = interval.endMs; open = true; }
    else if (interval.startMs <= end) end = Math.max(end, interval.endMs);
    else { total += end - start; start = interval.startMs; end = interval.endMs; }
  }
  return total + (open ? end - start : 0);
}

/** Explicit heuristic for archives without a reliable observation/session mask.
 * A long silence is excluded in full, not silently counted as an observed quiet market.
 * This cannot detect short packet losses and is never certified coverage.
 */
export function gapClippedMilliseconds(times: readonly number[], maximumGapMs: number, finalCadenceMs = 0): number {
  if (!(maximumGapMs > 0) || !(finalCadenceMs >= 0)) throw new Error("Invalid cadence");
  const intervals: Interval[] = [];
  for (let i = 1; i < times.length; i++) {
    const previous = times[i - 1]!, current = times[i]!;
    if (current < previous) throw new Error("Nonmonotonic receive timestamps");
    if (current - previous <= maximumGapMs) intervals.push({ startMs: previous, endMs: current });
  }
  if (times.length && finalCadenceMs) intervals.push({ startMs: times[times.length - 1]!, endMs: times[times.length - 1]! + finalCadenceMs });
  return unionMilliseconds(intervals);
}

export function budgetStreams(streams: readonly StreamBudget[], seconds = MONTH_SECONDS, multiplier = 1) {
  if (!(seconds >= 0) || !Number.isFinite(seconds) || !(multiplier >= 0) || !Number.isFinite(multiplier)) throw new Error("Invalid duration/multiplier");
  const rows = streams.map((stream) => {
    for (const value of [stream.recordsPerSecond, stream.uncompressedBytesPerRecord, stream.compressedBytesPerRecord]) {
      if (!(value >= 0) || !Number.isFinite(value)) throw new Error(`Invalid budget: ${stream.source}`);
    }
    const records = stream.recordsPerSecond * seconds * multiplier;
    return { ...stream, records, uncompressedBytes: records * stream.uncompressedBytesPerRecord,
      compressedBytes: records * stream.compressedBytesPerRecord };
  });
  return { seconds, multiplier, rows, uncompressedBytes: rows.reduce((sum, row) => sum + row.uncompressedBytes, 0),
    compressedBytes: rows.reduce((sum, row) => sum + row.compressedBytes, 0) };
}

export function formatBytes(bytes: number): string {
  return bytes >= 1e12 ? `${(bytes / 1e12).toFixed(2)} TB (${(bytes / 2 ** 40).toFixed(2)} TiB)`
    : `${(bytes / 1e9).toFixed(2)} GB (${(bytes / 2 ** 30).toFixed(2)} GiB)`;
}
