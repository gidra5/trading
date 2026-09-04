/** Lifecycle decomposition of one physical, netted asset position.
 * Account feasibility and log utility remain joint. Virtual pro-rata pieces
 * need not be exchange lots: precision/minimums apply to the physical fill. */
export interface EventPosition {
  id: number; side: -1 | 1; entryTime: number; entryPrice: number;
  initialQuantity: number; remainingQuantity: number;
  entryCost: number; exitCost: number; borrowing: number;
  realizedPnl: number; equityAdjustment: number; closedAt?: number;
  origin: "fill" | "initial-mark";
  closeReason?: "fill" | "terminal" | "liquidation";
}
export interface EventPositionChange {
  id: number; operation: "open" | "reduce" | "close";
  quantity: number; remainingQuantity: number; cost: number; realizedPnl: number;
}
export class EventPositionLedger {
  private active: EventPosition[] = [];
  private closed: EventPosition[] = [];
  private nextId = 1;
  private netQuantity = 0;
  private lastTime = -Infinity;
  private totals = { realizedPnl: 0, entryCost: 0, exitCost: 0, borrowing: 0, equityAdjustment: 0,
    created: 0, closed: 0, longPnl: 0, shortPnl: 0 };
  constructor(private readonly retainClosed = true) {}

  private at(time: number) {
    if (!Number.isSafeInteger(time) || time < this.lastTime) throw new Error("Unordered lifecycle event");
    this.lastTime = time;
  }
  quantity() { return this.netQuantity; }
  private collapsedQuantity() { return this.active.reduce((sum, p) => sum + p.side * p.remainingQuantity, 0); }
  /** Exit-only local domains; new exposure always receives a new position ID. */
  closeDomains() {
    return this.active.map(p => ({ id: p.id, side: p.side, minimumFraction: 0, maximumFraction: 1,
      remainingQuantity: p.remainingQuantity }));
  }
  positions() { return structuredClone([...this.closed, ...this.active].sort((a, b) => a.id - b.id)); }

  /** Reconcile the confirmed post-fill asset balance, never a requested target.
   * Keeping that physical balance explicit avoids accumulating virtual-lot
   * summation error into phantom positions after a complete exchange close. */
  fill(input: { time: number; price: number; quantityAfter: number; cost: number;
    origin?: "fill" | "initial-mark"; reason?: "fill" | "terminal" | "liquidation" }): EventPositionChange[] {
    const { time, price, quantityAfter, cost } = input, quantity = quantityAfter - this.netQuantity;
    if (!(price > 0) || ![price, quantity, cost].every(Number.isFinite) || cost < 0
      || quantity === 0 && cost !== 0 || input.origin === "initial-mark" && (this.nextId !== 1 || cost !== 0))
      throw new Error("Invalid lifecycle fill");
    this.at(time);
    if (!quantity) return [];
    const previous = this.quantity(), side = Math.sign(quantity) as -1 | 1, total = Math.abs(quantity);
    const reduction = previous * quantity < 0 ? Math.min(total, Math.abs(previous)) : 0;
    const changes: EventPositionChange[] = [];
    const closeCost = cost * reduction / total;
    if (reduction > 0) {
      const full = quantityAfter === 0 || quantityAfter * previous < 0;
      const size = Math.abs(this.collapsedQuantity());
      let remainingReduction = reduction, remainingCost = closeCost;
      const survivors: EventPosition[] = [];
      for (let i = 0; i < this.active.length; i++) {
        const p = this.active[i], last = i === this.active.length - 1;
        const closed = full ? p.remainingQuantity : last ? remainingReduction : reduction * p.remainingQuantity / size;
        const fee = last ? remainingCost : closeCost * p.remainingQuantity / size;
        const pnl = p.side * closed * (price - p.entryPrice);
        p.remainingQuantity = full ? 0 : Math.max(0, p.remainingQuantity - closed);
        p.exitCost += fee; p.realizedPnl += pnl;
        remainingReduction -= closed; remainingCost -= fee;
        if (p.remainingQuantity === 0) {
          p.closedAt = time; p.closeReason = input.reason ?? "fill";
          this.totals.closed++; if (this.retainClosed) this.closed.push(p);
        } else survivors.push(p);
        this.totals.realizedPnl += pnl;
        if (p.side > 0) this.totals.longPnl += pnl; else this.totals.shortPnl += pnl;
        changes.push({ id: p.id, operation: p.remainingQuantity ? "reduce" : "close", quantity: closed,
          remainingQuantity: p.remainingQuantity, cost: fee, realizedPnl: pnl });
      }
      this.active = survivors; this.totals.exitCost += closeCost;
    }
    const added = total - reduction;
    if (added > 0) {
      const p: EventPosition = { id: this.nextId++, side, entryTime: time, entryPrice: price,
        initialQuantity: added, remainingQuantity: added, entryCost: cost - closeCost, exitCost: 0,
        borrowing: 0, realizedPnl: 0, equityAdjustment: 0, origin: input.origin ?? "fill" };
      this.active.push(p); this.totals.entryCost += p.entryCost; this.totals.created++;
      changes.push({ id: p.id, operation: "open", quantity: added, remainingQuantity: added,
        cost: p.entryCost, realizedPnl: 0 });
    }
    this.netQuantity = quantityAfter;
    if (Math.abs(this.collapsedQuantity() - quantityAfter) > 1e-12 * Math.max(1, Math.abs(quantityAfter), total))
      throw new Error("Lifecycle fill failed to preserve net asset units");
    return changes;
  }

  /** Allocate the actual account charge, including any funding credit. */
  charge(time: number, amount: number, kind: "borrowing" | "equityAdjustment" = "borrowing") {
    if (!Number.isFinite(amount)) throw new Error("Invalid lifecycle charge");
    this.at(time);
    if (!amount) return;
    const size = Math.abs(this.quantity());
    if (!size) throw new Error("Cannot attribute a position charge without an active position");
    let remainder = amount;
    this.active.forEach((p, i) => {
      const share = i === this.active.length - 1 ? remainder : amount * p.remainingQuantity / size;
      p[kind] += share; remainder -= share;
    });
    this.totals[kind] += amount;
  }

  /** The simulator writes equity to zero at an accounting mark. This is not
   * represented as an observed exchange fill or a favorable liquidation price. */
  liquidate(time: number, price: number, equityAtMark: number) {
    this.charge(time, -equityAtMark, "equityAdjustment");
    return this.fill({ time, price, quantityAfter: 0, cost: 0, reason: "liquidation" });
  }

  summary(price: number) {
    if (!(price > 0) || !Number.isFinite(price)) throw new Error("Invalid lifecycle mark");
    let unrealizedPnl = 0, longUnrealized = 0, shortUnrealized = 0;
    for (const p of this.active) {
      const pnl = p.side * p.remainingQuantity * (price - p.entryPrice);
      unrealizedPnl += pnl;
      if (p.side > 0) longUnrealized += pnl; else shortUnrealized += pnl;
    }
    return { ...this.totals, active: this.active.length, quantity: this.quantity(), collapsedQuantity: this.collapsedQuantity(), unrealizedPnl,
      longPnl: this.totals.longPnl + longUnrealized, shortPnl: this.totals.shortPnl + shortUnrealized,
      costs: this.totals.entryCost + this.totals.exitCost,
      equityChange: this.totals.realizedPnl + unrealizedPnl - this.totals.entryCost - this.totals.exitCost
        - this.totals.borrowing + this.totals.equityAdjustment };
  }
}
