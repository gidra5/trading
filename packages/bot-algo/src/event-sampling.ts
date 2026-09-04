/** Average inverse label concurrency, for one price series per call.
 * A label from candle i to j uses returns i->i+1 through j-1->j, represented
 * by [i,j). Shared endpoints alone are not overlapping returns. Only completed
 * estimation labels belong here; these weights are unavailable at prediction
 * time and are neither independent-sample counts nor an unbiasedness guarantee.
 * Endpoint integration avoids allocating an array over potentially long gaps. */
export function eventAverageUniqueness(rows: readonly { start: number; end: number }[]): number[] {
  if (rows.some(r => !Number.isSafeInteger(r.start) || !Number.isSafeInteger(r.end)
    || r.start < 0 || r.end <= r.start)) throw new Error("Invalid event label interval");
  const changes = new Map<number, number>();
  for (const row of rows) {
    changes.set(row.start, (changes.get(row.start) ?? 0) + 1);
    changes.set(row.end, (changes.get(row.end) ?? 0) - 1);
  }
  const times = [...changes.keys()].sort((a, b) => a - b), integral = new Map<number, number>();
  let count = 0, value = 0;
  for (let i = 0; i < times.length; i++) {
    integral.set(times[i], value); count += changes.get(times[i])!;
    if (count > 0 && i + 1 < times.length) value += (times[i + 1] - times[i]) / count;
  }
  return rows.map(row => {
    const weight = (integral.get(row.end)! - integral.get(row.start)!) / (row.end - row.start);
    if (!(weight > 0 && weight <= 1 + 1e-10)) throw new Error("Invalid event uniqueness precision");
    return Math.min(1, weight);
  });
}
