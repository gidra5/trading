export const EVENT_COMPLETED_HISTORY_INPUTS = ["event-history-coverage-16", "previous-event-sign", "log1p-previous-event-abs-bps",
  "log1p-previous-event-minutes", "log1p-event-abs-bps-4", "log1p-event-minutes-4", "event-efficiency-4",
  "log1p-event-abs-bps-16", "log1p-event-minutes-16", "event-efficiency-16"] as const;
export interface CompletedHistoryEvent { originTime: number; availableAt: number; return: number; duration: number; }

/** Endpoint-chain memory of at most 16 complete moves within one day. The
 * entire event must start inside the lookback, so the existing one-day feature
 * purge still covers these inputs. Episode/gap resets are explicit. */
export class EventCompletedHistory {
  private history: CompletedHistoryEvent[] = [];
  private lastAvailable: number;
  private lastForecast: number;
  constructor(readonly after: number) {
    if (!Number.isFinite(after)) throw new Error("Invalid event-history start");
    this.lastAvailable = this.lastForecast = after;
  }
  features(at: number): number[] {
    if (!Number.isFinite(at) || at < this.lastAvailable || at < this.lastForecast) throw new Error("Event history is not yet available");
    this.lastForecast = at;
    this.history = this.history.filter(e => e.originTime >= at - 86_400_000);
    const previous = this.history.at(-1);
    const out = [this.history.length / 16, previous ? Math.sign(previous.return) : 0,
      previous ? Math.log1p(Math.abs(Math.log1p(previous.return)) * 1e4) : 0, previous ? Math.log1p(previous.duration) : 0];
    for (const length of [4, 16]) {
      const rows = this.history.slice(-length), n = rows.length;
      const signed = rows.reduce((s, e) => s + Math.log1p(e.return), 0);
      const absolute = rows.reduce((s, e) => s + Math.abs(Math.log1p(e.return)), 0);
      out.push(n ? Math.log1p(absolute * 1e4 / n) : 0,
        n ? Math.log1p(rows.reduce((s, e) => s + e.duration, 0) / n) : 0, absolute ? signed / absolute : 0);
    }
    return out;
  }
  observe(event: CompletedHistoryEvent, now: number): void {
    if (!Number.isFinite(event.originTime) || !Number.isFinite(event.availableAt) || !Number.isFinite(now)
      || event.originTime < this.lastAvailable || event.originTime < this.lastForecast || event.availableAt > now
      || !Number.isInteger(event.duration) || event.duration < 1 || event.availableAt !== event.originTime + event.duration * 60_000
      || !(event.return > -1) || !Number.isFinite(event.return)) throw new Error("Event history requires ordered completed outcomes");
    if (event.originTime > this.lastAvailable) this.history = [];
    this.history.push({ ...event });
    if (this.history.length > 16) this.history.shift();
    this.lastAvailable = event.availableAt;
  }
}
