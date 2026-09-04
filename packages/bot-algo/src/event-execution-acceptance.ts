export interface EventAcceptanceChoice {
  accepted: number;
  rejected: number;
  /** Unavailable/size-filtered or possible recovery exceptions do not impose
   * a position in the ordinary opening-price acceptance sequence. */
  free: boolean;
}

/** Maximum value of an accepted prefix, suffix or interval in ascending
 * opening-price order. Identical openings must already share one choice.
 * -Infinity prohibits a branch. Free rows preserve independent exceptions. */
export function maximizeEventAcceptanceSequence(rows: readonly EventAcceptanceChoice[], direction: "prefix" | "suffix" | "interval") {
  const add = (a: number, b: number) => a === -Infinity || b === -Infinity ? -Infinity : a + b;
  let before = direction === "prefix" ? -Infinity : 0;
  let inside = direction === "prefix" ? 0 : -Infinity, after = -Infinity;
  for (const row of rows) {
    if (row.free) {
      const value = Math.max(row.accepted, row.rejected);
      before = add(before, value); inside = add(inside, value); after = add(after, value);
      continue;
    }
    const nextBefore = add(before, row.rejected);
    const nextInside = add(Math.max(before, inside), row.accepted);
    const nextAfter = direction === "suffix" ? -Infinity : add(Math.max(inside, after), row.rejected);
    before = nextBefore; inside = nextInside; after = nextAfter;
  }
  return Math.max(before, inside, after);
}
