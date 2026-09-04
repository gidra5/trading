import type { EventAccount, EventCosts } from "../src/event-log-policy.js";
import type { EventDistribution } from "../src/event-distribution.js";

/** Small exhaustive stochastic Bellman reference. No production trade,
 * interpolation, holding or optimizer helpers are used. Prices fill at the
 * event decision price because the supplied law has no opening-gap variable.
 * This is a model oracle, never a search over a realized future price path. */
export function eventBellmanReference(model: EventDistribution, costs: EventCosts,
  options: { maxNodes?: number; maxOrderLots?: number; terminal?: "marked" | "friction" | "market" } = {}) {
  const fee = (costs.feeBps + costs.slippageBps) / 10000;
  const cache = new Map<string, { value: number; quantity: number }>();
  const stats = { nodes: 0, actions: 0, outcomes: 0, cacheHits: 0 };
  const terminal = (a: EventAccount) => {
    if (options.terminal === "marked") return 0;
    const quantity = a.exposure * a.equity / a.price, notional = Math.abs(quantity) * a.price;
    if (options.terminal === "market" && (notional < costs.minNotional - 1e-8 || Math.abs(quantity) < costs.minQuantity - 1e-12)) return 0;
    const factor = 1 - notional * fee / a.equity;
    return factor > 0 ? Math.log(factor) : -Infinity;
  };
  const trade = (a: EventAccount, quantity: number): EventAccount | null => {
    if (!quantity) return Math.abs(a.exposure) <= costs.maxLeverage + 1e-9 ? a : null;
    const lots = quantity / costs.quantityStep, notional = Math.abs(quantity) * a.price;
    if (Math.abs(lots - Math.round(lots)) > 1e-7 || Math.abs(quantity) < costs.minQuantity - 1e-12
      || notional < costs.minNotional - 1e-8 || notional > costs.maxNotional + 1e-8) return null;
    const equity = a.equity - fee * notional;
    if (!(equity > 0)) return null;
    const exposure = (a.exposure * a.equity + quantity * a.price) / equity;
    if (Math.abs(exposure) > costs.maxLeverage + 1e-8
      && !(Math.abs(a.exposure) > costs.maxLeverage && Math.abs(exposure) < Math.abs(a.exposure) - 1e-9
        && notional >= costs.maxNotional - costs.quantityStep * a.price - 1e-8)) return null;
    return { equity, price: a.price, exposure };
  };
  const expectation = (leaf: number, a: EventAccount, depth: number,
    continuation: (leaf: number, a: EventAccount, depth: number) => number): number => {
    let result = 0;
    for (const atom of model.kernels[leaf]) if (atom.probability > 0) {
      stats.outcomes++;
      const quantity = a.exposure * a.equity / a.price;
      const debt = quantity >= 0 ? Math.max(0, quantity * a.price - a.equity) : -quantity * a.price;
      const borrowing = debt * (quantity >= 0 ? costs.longBorrowBpsPerDay : costs.shortBorrowBpsPerDay) / 10000 * atom.duration / 1440;
      const price = a.price * (1 + atom.return), equity = a.equity + quantity * (price - a.price) - borrowing;
      const worstPrice = a.price * (1 + (quantity >= 0 ? atom.low : atom.high));
      const minimum = a.equity + quantity * (worstPrice - a.price) - borrowing;
      if (equity <= 0 || minimum <= costs.maintenanceMargin * Math.abs(quantity) * worstPrice) return -Infinity;
      const next = { equity, price, exposure: quantity * price / equity };
      const value = Math.log(equity / a.equity) + (depth === 1 ? terminal(next) : continuation(atom.next, next, depth - 1));
      if (!Number.isFinite(value)) return -Infinity;
      result += atom.probability * value;
    }
    return result;
  };
  const holdValue = (leaf: number, a: EventAccount, depth: number) => expectation(leaf, a, depth, (l, next, d) => decide(l, next, d).value);
  const actionValue = (leaf: number, a: EventAccount, depth: number, quantity: number) => {
    const next = trade(a, quantity);
    return next ? Math.log(next.equity / a.equity) + holdValue(leaf, next, depth) : -Infinity;
  };
  const decide = (leaf: number, a: EventAccount, depth: number): { value: number; quantity: number } => {
    if (!Number.isInteger(depth) || depth < 1 || !model.kernels[leaf] || !(a.equity > 0 && a.price > 0)
      || !Number.isFinite(a.exposure)) throw new Error("Invalid reference query");
    const key = [leaf, depth, a.equity, a.price, a.exposure].join(":");
    const cached = cache.get(key);
    if (cached) { stats.cacheHits++; return cached; }
    if (++stats.nodes > (options.maxNodes ?? 100000)) throw new Error("Reference node budget exceeded");
    const maximum = Math.floor(costs.maxNotional / a.price / costs.quantityStep + 1e-8);
    if (maximum > (options.maxOrderLots ?? 1000)) throw new Error("Reference lot budget exceeded");
    let best = { value: actionValue(leaf, a, depth, 0), quantity: 0 };
    for (let lots = -maximum; lots <= maximum; lots++) if (lots) {
      stats.actions++;
      const quantity = lots * costs.quantityStep, value = actionValue(leaf, a, depth, quantity);
      if (value > best.value + 1e-14 || (value === best.value && Math.abs(quantity) < Math.abs(best.quantity)))
        best = { value, quantity };
    }
    cache.set(key, best); return best;
  };
  const evaluatePolicy = (leaf: number, a: EventAccount, depth: number,
    select: (leaf: number, a: EventAccount, depth: number) => number): number => {
    const next = trade(a, select(leaf, a, depth));
    return next ? Math.log(next.equity / a.equity) + expectation(leaf, next, depth,
      (l, account, d) => evaluatePolicy(l, account, d, select)) : -Infinity;
  };
  return { decide, holdValue, actionValue, evaluatePolicy, stats };
}
