const DAY = 86400000;
type Excluded = { id: string; startTime: number; endTime: number };
export type EventSourceRange = { start: number; end: number };

/** Merge actual requested source intervals; never fill a gap between them. */
export function mergeEventSourceRanges(ranges: readonly EventSourceRange[]): EventSourceRange[] {
  if (ranges.some(r => !Number.isSafeInteger(r.start) || !Number.isSafeInteger(r.end) || r.end <= r.start))
    throw new Error("Invalid event source range");
  const merged: EventSourceRange[] = [];
  for (const row of ranges.slice().sort((a, b) => a.start - b.start)) {
    const previous = merged.at(-1);
    if (previous && row.start <= previous.end) previous.end = Math.max(previous.end, row.end);
    else merged.push({ ...row });
  }
  return merged;
}

export function eventSourceDays(ranges: readonly EventSourceRange[]): number[] {
  const days = new Set<number>();
  for (const range of mergeEventSourceRanges(ranges))
    for (let day = Math.floor(range.start / DAY) * DAY; day < range.end; day += DAY) days.add(day);
  return [...days].sort((a, b) => a - b);
}

/** Latest whole-day fit/calibration block before a target, with its complete
 * feature support outside every excluded interval. Dates alone choose the
 * block; neither returns nor sample/model performance enter this rule. */
export function eventFitPeriods(targetStart: number, fitDays: number, historyMs: number, excluded: readonly Excluded[]) {
  if (!Number.isSafeInteger(targetStart) || targetStart % DAY !== 0 || !Number.isInteger(fitDays) || fitDays < 1
    || !Number.isSafeInteger(historyMs) || historyMs < 0 || excluded.some(r =>
      !Number.isSafeInteger(r.startTime) || !Number.isSafeInteger(r.endTime) || r.endTime <= r.startTime))
    throw new Error("Invalid event fitting periods");
  let anchor = targetStart;
  const shifts: Array<{ from: number; to: number; excludedIds: string[] }> = [];
  for (;;) {
    const fitEnd = anchor - DAY, fitStart = fitEnd - fitDays * DAY, sourceStart = fitStart - historyMs;
    const conflicts = excluded.filter(r => r.startTime < anchor && r.endTime > sourceStart);
    if (!conflicts.length) return { fitStart, fitEnd, calibrationStart: fitEnd, calibrationEnd: anchor,
      sourceStart, testGapDays: (targetStart - anchor) / DAY, shifts };
    const earlier = Math.floor(Math.min(...conflicts.map(r => r.startTime)) / DAY) * DAY;
    if (!(earlier < anchor)) throw new Error("Event fitting anchor did not move backwards");
    shifts.push({ from: anchor, to: earlier, excludedIds: conflicts.map(r => r.id) });
    anchor = earlier;
  }
}
