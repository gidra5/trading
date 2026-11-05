import type {
  TradingMarketRules,
  TradingOrderStatus,
  TradingQuantityRules,
} from "@trading/bot-algo/trading-api";
import { combineSteps, positive, withinRules } from "./common.js";

export class BinanceUnsupportedOperationError extends Error {
  constructor(product: string, operation: string) {
    super(`${product} does not support ${operation}`);
    this.name = "BinanceUnsupportedOperationError";
  }
}

/** A lost response may conceal an accepted order. Never retry automatically. */
export class BinanceOrderSubmissionError extends Error {
  constructor(
    readonly clientOrderId: string,
    readonly conditional: boolean,
    cause: unknown,
  ) {
    super(
      `Order submission failed or is uncertain; reconcile client ID ${clientOrderId} before retrying`,
      { cause },
    );
    this.name = "BinanceOrderSubmissionError";
  }
}

export function errorCode(error: unknown): number | undefined {
  return typeof error === "object" && error !== null && "code" in error
    ? Number(error.code)
    : undefined;
}

export function orderStatus(status: string): TradingOrderStatus {
  switch (status) {
    case "NEW":
    case "ACCEPTED":
      return "open";
    case "PARTIALLY_FILLED":
      return "partially-filled";
    case "FILLED":
      return "filled";
    case "CANCELED":
    case "CANCELLED":
      return "canceled";
    case "EXPIRED":
    case "EXPIRED_IN_MATCH":
      return "expired";
    case "REJECTED":
      return "rejected";
    default:
      return "pending";
  }
}

export const enabled = (value: unknown): number | null =>
  Number(value) > 0 ? Number(value) : null;

export interface ExchangeFilter {
  filterType: string;
  minPrice?: string | number;
  maxPrice?: string | number;
  tickSize?: string | number;
  minQty?: string | number;
  maxQty?: string | number;
  stepSize?: string | number;
  minNotional?: string | number;
  maxNotional?: string | number;
}

export function cashMarketRules(
  filters: readonly ExchangeFilter[],
): TradingMarketRules {
  const price = filters.find((f) => f.filterType === "PRICE_FILTER");
  const lot = filters.find((f) => f.filterType === "LOT_SIZE");
  const market = filters.find((f) => f.filterType === "MARKET_LOT_SIZE");
  const notionals = filters.filter((f) =>
    ["MIN_NOTIONAL", "NOTIONAL"].includes(f.filterType),
  );
  const limitQuantity = {
    min: enabled(lot?.minQty),
    max: enabled(lot?.maxQty),
    step: enabled(lot?.stepSize),
  };
  const maxQuantity = Math.min(
    enabled(market?.maxQty) ?? Infinity,
    limitQuantity.max ?? Infinity,
  );
  return {
    price: {
      min: enabled(price?.minPrice),
      max: enabled(price?.maxPrice),
      step: enabled(price?.tickSize),
    },
    limitQuantity,
    marketQuantity: {
      min: enabled(
        Math.max(enabled(market?.minQty) ?? 0, limitQuantity.min ?? 0),
      ),
      max: maxQuantity === Infinity ? null : maxQuantity,
      step: combineSteps(enabled(market?.stepSize), limitQuantity.step),
    },
    minNotional: enabled(
      Math.max(0, ...notionals.map((f) => enabled(f.minNotional) ?? 0)),
    ),
    maxNotional:
      notionals.map((f) => enabled(f.maxNotional)).find((v) => v !== null) ??
      null,
    maxLeverage: 1,
  };
}

export interface Book {
  readonly bids: readonly (readonly [string | number, string | number])[];
  readonly asks: readonly (readonly [string | number, string | number])[];
}

export const DEFAULT_FRICTION_DECAY_RATE = 5;

export function validateFrictionDecayRate(rate: number): void {
  if (!Number.isFinite(rate) || rate < 0)
    throw new Error("frictionDecayRate must be finite and nonnegative");
}

// todo: review, cleanup
/**
 * Expected worst-side VWAP cost relative to mid, including spread.
 * Size weights fall exponentially by exp(-decay) across the observable quantity range.
 * Sum small lot grids exactly; otherwise use 256 deterministic probability quantiles.
 * A supplied size bypasses averaging and walks depth for that quantity without rounding.
 */
export function estimateBookCost(
  book: Book,
  rules: TradingQuantityRules,
  size?: number,
  decay = DEFAULT_FRICTION_DECAY_RATE,
): number {
  validateFrictionDecayRate(decay);
  if (size !== undefined) positive(size, "size");
  if (!book.bids.length || !book.asks.length)
    throw new Error("Cannot estimate friction from an empty order book");
  const levels = (rows: Book["bids"], side: "buy" | "sell") =>
    rows.map(([rawPrice, rawQuantity], index) => {
      const price = Number(rawPrice),
        quantity = Number(rawQuantity);
      positive(price, "book price");
      positive(quantity, "book quantity");
      const previous = index > 0 ? Number(rows[index - 1][0]) : price;
      if (side === "buy" ? price < previous : price > previous)
        throw new Error("Unsorted order book");
      return { price, quantity };
    });
  const bids = levels(book.bids, "sell"),
    asks = levels(book.asks, "buy");
  if (bids[0].price > asks[0].price) throw new Error("Crossed order book");
  const mid = (bids[0].price + asks[0].price) / 2;
  const depth = (rows: typeof bids) =>
    rows.reduce((sum, row) => sum + row.quantity, 0);
  const available = Math.min(depth(bids), depth(asks));
  positive(available, "available book depth");
  for (const [name, value] of [
    ["min", rules.min],
    ["max", rules.max],
  ] as const) {
    if (value !== null && (!Number.isFinite(value) || value < 0))
      throw new Error(`Invalid quantity ${name}`);
  }
  if (rules.step !== null) positive(rules.step, "quantity step");

  const minimum = Math.max(rules.min ?? 0, rules.step ?? 0);
  const maximum = Math.min(rules.max ?? available, available);
  const step = rules.step;
  const min =
    step === null ? minimum : Math.ceil(minimum / step - 1e-10) * step;
  const max =
    step === null ? maximum : Math.floor(maximum / step + 1e-10) * step;
  if (max <= 0 || min > max)
    throw new Error(
      "No valid order sizes within visible book depth and quantity limits",
    );

  const cost = (size: number) => {
    const average = (rows: typeof bids) => {
      let remaining = size,
        quote = 0;
      for (const { price, quantity } of rows) {
        const take = Math.min(remaining, quantity);
        quote += take * price;
        remaining -= take;
        if (remaining <= size * 1e-12) return quote / size;
      }
      throw new Error("Insufficient order-book depth");
    };
    return Math.max(average(asks) / mid - 1, 1 - average(bids) / mid, 0);
  };
  if (size !== undefined) {
    if (!withinRules(size, rules))
      throw new Error("size must satisfy the market quantity limits and step");
    return cost(size);
  }
  if (min === max) return cost(min);

  const samples = 256;
  if (step !== null) {
    const count = Math.round((max - min) / step) + 1;
    if (!Number.isSafeInteger(count))
      throw new Error("Quantity grid exceeds supported precision");
    if (count === 1) return cost(min);
    const rate = decay / (count - 1);
    if (count <= samples) {
      let total = 0,
        weightSum = 0;
      for (let index = 0; index < count; index++) {
        const weight = Math.exp(-rate * index);
        total += weight * cost(min + index * step);
        weightSum += weight;
      }
      return total / weightSum;
    }
    const mass = -Math.expm1(-rate * count);
    let total = 0;
    for (let index = 0; index < samples; index++) {
      const u = (index + 0.5) / samples;
      const gridIndex = Math.min(
        count - 1,
        Math.floor(decay < 1e-8 ? u * count : -Math.log1p(-u * mass) / rate),
      );
      total += cost(min + gridIndex * step);
    }
    return total / samples;
  }

  const mass = -Math.expm1(-decay);
  let total = 0;
  for (let index = 0; index < samples; index++) {
    const u = (index + 0.5) / samples;
    const fraction = decay < 1e-8 ? u : -Math.log1p(-u * mass) / decay;
    const size = min + (max - min) * fraction;
    total += cost(size);
  }
  return total / samples;
}
