import type { EventCandle } from "./event-distribution.js";
import { eventCapAllowsTrade, validateEventCosts, type EventAccount, type EventCosts } from "./event-log-policy.js";

/** Controlled sufficient statistics for the RESEARCH replay's constant inventory
 * between events. Prices are ratios to the decision close. Funding coefficients
 * bind this compression to its recorded costs; this is not an exchange fill law. */
export interface EventExecutionPath {
  version: 1; costs: EventCosts; seconds: number;
  openRatio: number; closeRatio: number; lowRatio: number; highRatio: number;
  openingAvailable: boolean; terminalAvailable: boolean;
  longDebtGrowth: number; minimumDiscountedLongLow: number;
  shortBorrowPriceIntegral: number; maximumShortMaintenancePrice: number;
}

/** Compile a realized path or a training outcome, never feed a future realized
 * path to a decision. Indices describe (origin, end], one-second bars only. */
export function summarizeEventExecutionPath(c: readonly EventCandle[], origin: number, end: number, costs: EventCosts,
  terminalAvailable = !c[end]?.carriedMark): EventExecutionPath {
  validateEventCosts(costs);
  if (!Number.isInteger(origin) || !Number.isInteger(end) || origin < 0 || end <= origin || end >= c.length
    || !(c[origin].close > 0) || !Number.isFinite(c[origin].close)) throw new Error("Invalid execution path boundaries");
  const price = c[origin].close, longRate = costs.longBorrowBpsPerDay / 10000 / 86400;
  const shortRate = costs.shortBorrowBpsPerDay / 10000 / 86400;
  let growth = 1, low = Infinity, high = 0, discountedLow = Infinity, integral = 0, shortRisk = -Infinity;
  for (let i = origin + 1; i <= end; i++) {
    const row = c[i];
    if (row.openTime - c[i - 1].openTime !== 1000 || ![row.open, row.low, row.high, row.close].every(Number.isFinite)
      || row.low <= 0 || row.low > Math.min(row.open, row.close) || row.high < Math.max(row.open, row.close))
      throw new Error("Invalid or discontinuous execution path");
    const lo = row.low / price, hi = row.high / price;
    growth *= 1 + longRate; integral += row.open / price * shortRate;
    low = Math.min(low, lo); high = Math.max(high, hi);
    discountedLow = Math.min(discountedLow, lo / growth);
    shortRisk = Math.max(shortRisk, integral + (1 + costs.maintenanceMargin) * hi);
  }
  return { version: 1, costs: { ...costs }, seconds: end - origin, openRatio: c[origin + 1].open / price,
    closeRatio: c[end].close / price, lowRatio: low, highRatio: high,
    openingAvailable: !c[origin + 1].carriedMark, terminalAvailable,
    longDebtGrowth: growth, minimumDiscountedLongLow: discountedLow,
    shortBorrowPriceIntegral: integral, maximumShortMaintenancePrice: shortRisk };
}

/** Evaluate one committed base-quantity request against a forecast outcome or
 * realized diagnostic path. Failed next-open requests leave existing holdings;
 * no trade is admissible above the entry leverage cap until maintenance fails.
 * A log-utility liquidation is exact, but this summary deliberately does not
 * claim to recover its intra-event timestamp or accrued charges before ruin. */
export function evaluateEventExecutionPath(path: EventExecutionPath, account: EventAccount, requestedQuantity: number,
  terminal: "marked" | "market" = "marked") {
  const c = path.costs, { equity: initialEquity, price, exposure } = account;
  if (!(initialEquity > 0 && price > 0) || ![initialEquity, price, exposure, requestedQuantity].every(Number.isFinite)
    || Math.abs(requestedQuantity / c.quantityStep - Math.round(requestedQuantity / c.quantityStep)) > 1e-7
    || !["marked", "market"].includes(terminal)) throw new Error("Invalid execution-path account or request");
  let quantity = exposure * initialEquity / price, equity = initialEquity + quantity * price * (path.openRatio - 1);
  const open = price * path.openRatio, close = price * path.closeRatio;
  const feeRate = (c.feeBps + c.slippageBps) / 10000;
  let filledQuantity = 0, fee = 0;
  const ruin = (phase: "opening" | "holding") => ({ equity: 0, price: close, exposure: 0, quantity: 0,
    requestedQuantity, filledQuantity, canceled: phase === "holding" && requestedQuantity !== 0 && filledQuantity === 0,
    fee, borrowing: phase === "opening" ? 0 : null, liquidated: true, liquidationPhase: phase,
    logGrowth: -Infinity, terminalOrders: 0, terminalDust: 0, unsettledNotional: 0 });
  if (equity <= c.maintenanceMargin * Math.abs(quantity) * open) return ruin("opening");
  if (requestedQuantity && path.openingAvailable) {
    const turnover = Math.abs(requestedQuantity) * open, cost = turnover * feeRate;
    if (Math.abs(requestedQuantity) >= c.minQuantity - 1e-12 && turnover >= c.minNotional - 1e-8
      && turnover <= c.maxNotional + 1e-8 && equity > cost && eventCapAllowsTrade(quantity * open / equity,
        (quantity + requestedQuantity) * open / (equity - cost), turnover, open, c)) {
      filledQuantity = requestedQuantity; fee = cost; equity -= cost;
      quantity = Math.round((quantity + requestedQuantity) / c.quantityStep) * c.quantityStep;
    }
  }
  const cash = equity - quantity * open;
  const borrowedLong = quantity > 0 && cash < 0;
  const riskCash = quantity >= 0 ? cash + quantity * price * (1 - c.maintenanceMargin)
    * (borrowedLong ? path.minimumDiscountedLongLow : path.lowRatio)
    : cash + quantity * price * path.maximumShortMaintenancePrice;
  if (!(riskCash > 0)) return ruin("holding");
  const borrowing = borrowedLong ? -cash * (path.longDebtGrowth - 1)
    : quantity < 0 ? -quantity * price * path.shortBorrowPriceIntegral : 0;
  equity = cash + quantity * close - borrowing;
  if (!(equity > 0)) return ruin("holding");
  const notional = Math.abs(quantity) * close;
  let terminalOrders = 0, terminalDust = 0, unsettledNotional = 0;
  if (terminal === "market") {
    if (!path.terminalAvailable) unsettledNotional = notional;
    else if (quantity && Math.abs(quantity) >= c.minQuantity - 1e-12 && notional >= c.minNotional - 1e-8) {
      const cost = notional * feeRate; equity -= cost; fee += cost;
      terminalOrders = Math.ceil(notional / c.maxNotional); quantity = 0;
    } else terminalDust = notional;
  }
  return { equity, price: close, exposure: quantity * close / equity, quantity, requestedQuantity, filledQuantity,
    canceled: requestedQuantity !== 0 && filledQuantity === 0, fee, borrowing, liquidated: false,
    liquidationPhase: null, logGrowth: equity > 0 ? Math.log(equity / initialEquity) : -Infinity,
    terminalOrders, terminalDust, unsettledNotional };
}
