import type { EventCandle } from "./event-distribution.js";

/** Explicit research assumptions, not inferred from zero volume. End is exclusive. */
export interface EventMarketClosure { start: number; end: number; reason: string; }
export function validateEventMarketClosures(closures: readonly EventMarketClosure[]): void {
  for (let i = 0; i < closures.length; i++) {
    const c = closures[i];
    if (!Number.isSafeInteger(c.start) || !Number.isSafeInteger(c.end) || c.start % 1000 || c.end % 1000
      || c.end <= c.start || !c.reason || (i > 0 && c.start < closures[i - 1].end))
      throw new Error("Invalid, overlapping or unordered market closures");
  }
}
/** Use at the execution timestamp. Do not pass the future schedule to a predictor. */
export const eventMarketUnavailable = (closures: readonly EventMarketClosure[], time: number): boolean =>
  closures.some(c => time >= c.start && time < c.end);
export function eventAvailabilitySourceStart(start: number, closures: readonly EventMarketClosure[]): number {
  for (let i = closures.length - 1; i >= 0; i--) if (start >= closures[i].start && start < closures[i].end) start = closures[i].start - 1000;
  return start;
}

export type EventSourceSecond = EventCandle & { closed: boolean; closeTime: number };
export function validateEventSourceSecond(row: EventSourceSecond): void {
  if (!row.closed || !Number.isSafeInteger(row.openTime) || row.openTime % 1000 || row.closeTime !== row.openTime + 999
    || ![row.open, row.high, row.low, row.close, row.volume].every(Number.isFinite)
    || row.low <= 0 || row.volume < 0 || row.high < Math.max(row.open, row.close)
    || row.low > Math.min(row.open, row.close) || row.carriedMark)
    throw new Error(`Malformed or incomplete native second candle: ${JSON.stringify(row)}`);
}

/** Replay-only marked view. Sources stay immutable; every unexpected gap still fails.
 * Rows may include history before start to anchor a closure. A partial source row
 * within a declared closure is accepted only as evidence of an unchanged mark,
 * never as a complete observed candle. */
export function markUnavailableEventSeconds(rows: readonly EventSourceSecond[], start: number, end: number,
  closures: readonly EventMarketClosure[]): EventCandle[] {
  validateEventMarketClosures(closures);
  if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start % 1000 || end % 1000 || end <= start)
    throw new Error("Invalid marked-view boundaries");
  for (let i = 0; i < rows.length; i++) if (!Number.isSafeInteger(rows[i].openTime) || rows[i].openTime % 1000
    || (i > 0 && rows[i].openTime <= rows[i - 1].openTime)) throw new Error("Unordered/duplicate native source seconds");
  const requiredStart = eventAvailabilitySourceStart(start, closures);
  let index = 0;
  while (index < rows.length && rows[index].openTime < requiredStart) index++;
  let mark: number | undefined;
  const output: EventCandle[] = [];
  for (let time = requiredStart; time < end; time += 1000) {
    const row = rows[index]?.openTime === time ? rows[index++] : undefined;
    if (eventMarketUnavailable(closures, time)) {
      if (mark === undefined) throw new Error("Unavailable interval requires a preceding observed mark");
      if (row && (!row.closed || !Number.isSafeInteger(row.closeTime) || row.closeTime < time || row.closeTime > time + 999
        || row.volume !== 0 || ![row.open, row.high, row.low, row.close].every(p => p === mark) || row.carriedMark))
        throw new Error(`Source contradicts declared unavailable interval at ${time}`);
      if (time >= start) output.push({ openTime: time, open: mark, high: mark, low: mark, close: mark, volume: 0, carriedMark: true });
    } else {
      if (!row) throw new Error(`Gap outside declared unavailable interval at ${time}`);
      validateEventSourceSecond(row); mark = row.close;
      if (time >= start) output.push(row);
    }
  }
  return output;
}
