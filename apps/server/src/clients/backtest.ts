import assert from "node:assert/strict";
import type {
  TradingCandle,
  TradingTick,
  TradingHistoryRequest,
  TradingMarketRules,
  TradingOrderCapacityRequest,
  TradingOrderSnapshot,
} from "@trading/bot-algo/trading-api";
import { BaseTradingApi } from "./base-trading-api.js";
import { BacktestAccount } from "./backtest-account.js";
import {
  positive,
  validateHistoryRequest,
  validateLeverage,
} from "./common.js";

export interface BacktestTradingApiConfig {
  readonly initialQuote: number;
  readonly leverage: number;
  /** Number of completed history candles available for warmup before replay starts. */
  readonly startIndex?: number;
  readonly feeRate: number;
  readonly slippageRate: number;
  readonly maintenanceMarginRate: number;
  readonly liquidationFeeRate: number;
  readonly marketRules: TradingMarketRules;
  /** OHLC does not reveal intrabar order. Choose the assumed path explicitly. */
  readonly candlePath: "open-high-low-close" | "open-low-high-close";
  /** Omitted funding means no funding charges are modeled. */
  readonly funding?: readonly {
    readonly timestamp: number;
    readonly rate: number;
    readonly markPrice: number;
  }[];
}

/** A single-symbol, linear USD-M futures account with deterministic candle replay. */
export class BacktestTradingApi extends BaseTradingApi {
  private readonly config: BacktestTradingApiConfig;
  private readonly account: BacktestAccount;
  private readonly history: readonly TradingCandle[];
  private readonly intervalMs: number | undefined;
  private cursor: number;

  private get finished() {
    return this.cursor === this.history.length;
  }

  constructor(
    candles: readonly TradingCandle[],
    config: BacktestTradingApiConfig,
  ) {
    super();
    validateConfig(config);
    this.config = config;
    const source = candles;
    this.intervalMs = source[0] && source[0].closeTime - source[0].openTime + 1;
    for (const [i, candle] of source.entries()) {
      validateCandle(candle);
      assert(
        candle.closeTime - candle.openTime + 1 === this.intervalMs,
        "History candles must have the same duration",
      );
      assert(
        i === 0 || candle.openTime > source[i - 1].closeTime,
        "History candles must be chronological and non-overlapping",
      );
    }
    const cursor = config.startIndex ?? 0;
    assert(
      Number.isSafeInteger(cursor) && cursor >= 0 && cursor <= source.length,
      "Invalid startIndex",
    );
    this.history = source;
    this.cursor = cursor;
    const last = source[cursor - 1];
    this.account = new BacktestAccount(
      this.config,
      last?.close ?? null,
      last?.closeTime ?? -1,
    );
  }

  get price(): number | null {
    return this.account.price;
  }

  advance(): TradingTick | null {
    if (this.finished) return null;
    const candle = this.history[this.cursor];
    assert(candle, "Missing replay candle");
    this.account.advance(candle.open, candle.openTime, true);
    const path =
      this.config.candlePath === "open-high-low-close"
        ? [candle.high, candle.low, candle.close]
        : [candle.low, candle.high, candle.close];
    for (const [i, price] of path.entries()) {
      this.account.advance(
        price,
        candle.openTime + ((candle.closeTime - candle.openTime) * (i + 1)) / 3,
      );
    }
    this.cursor++;
    if (this.finished) this.account.cancelAll();
    return {
      timestamp: candle.closeTime,
      price: candle.close,
      quantity: candle.volume,
      candle,
    };
  }

  async getFriction(size?: number): Promise<number> {
    if (size !== undefined) positive(size, "size");
    return this.config.feeRate + this.config.slippageRate;
  }
  async getEquity() {
    return this.account.equity;
  }
  async getOrderCapacity(input: TradingOrderCapacityRequest) {
    return this.account.getOrderCapacity(input, this.finished);
  }
  async getMarketRules() {
    return this.config.marketRules;
  }

  async getHistory(
    input: TradingHistoryRequest,
  ): Promise<readonly TradingCandle[]> {
    validateHistoryRequest(input);
    if (this.intervalMs === undefined) return [];
    assert(
      input.intervalMs % this.intervalMs === 0,
      "History interval must be a multiple of the source candle interval",
    );
    if (input.count === 0) return [];
    if (input.intervalMs === this.intervalMs)
      return this.history.slice(
        Math.max(0, this.cursor - input.count),
        this.cursor,
      );
    const result: TradingCandle[] = [];
    // This accumulator is private and mutable until its completed value is published.
    let bucket:
      { -readonly [K in keyof TradingCandle]: TradingCandle[K] } | undefined;
    for (let index = 0; index < this.cursor; index++) {
      const candle = this.history[index];
      const start =
        Math.floor(candle.openTime / input.intervalMs) * input.intervalMs;
      if (candle.openTime === start) bucket = { ...candle };
      else if (
        bucket?.openTime === start &&
        bucket.closeTime + 1 === candle.openTime
      ) {
        bucket.high = Math.max(bucket.high, candle.high);
        bucket.low = Math.min(bucket.low, candle.low);
        bucket.close = candle.close;
        bucket.closeTime = candle.closeTime;
        bucket.volume += candle.volume;
        bucket.aggressiveBuyVolume = addOptional(
          bucket.aggressiveBuyVolume,
          candle.aggressiveBuyVolume,
        );
        bucket.aggressiveSellVolume = addOptional(
          bucket.aggressiveSellVolume,
          candle.aggressiveSellVolume,
        );
      } else bucket = undefined;
      if (bucket?.closeTime === start + input.intervalMs - 1) {
        result.push(bucket);
        bucket = undefined;
      }
    }
    return result.slice(-input.count);
  }
  async submit(order: TradingOrderSnapshot) {
    return this.account.submit(order, this.finished);
  }
  async cancelOrder(id: string) {
    return this.account.cancelOrder(id);
  }
  getOrder(id: string) {
    return this.account.getOrder(id);
  }
  snapshot() {
    return this.account.snapshot();
  }
  drainOrderEvents() {
    return this.account.drainOrderEvents();
  }
}

function addOptional(a: number | undefined, b: number | undefined) {
  return a === undefined || b === undefined ? undefined : a + b;
}

function validateConfig(config: BacktestTradingApiConfig): void {
  assert(
    Number.isFinite(config.initialQuote) && config.initialQuote >= 0,
    "Invalid initialQuote",
  );
  validateLeverage(config.leverage);
  assert(
    ["open-high-low-close", "open-low-high-close"].includes(config.candlePath),
    "Invalid candlePath",
  );
  for (const name of [
    "feeRate",
    "slippageRate",
    "maintenanceMarginRate",
    "liquidationFeeRate",
  ] as const) {
    assert(
      Number.isFinite(config[name]) && config[name] >= 0 && config[name] < 1,
      `${name} must be in [0, 1)`,
    );
  }
  const rules = config.marketRules;
  assert(rules, "marketRules are required");
  validateLeverage(rules.maxLeverage);
  assert(
    config.leverage <= rules.maxLeverage,
    "Configured leverage exceeds market maximum",
  );
  for (const rule of [rules.price, rules.limitQuantity, rules.marketQuantity]) {
    for (const value of [rule.min, rule.max, rule.step])
      if (value !== null) positive(value, "market rule");
    assert(
      rule.min === null || rule.max === null || rule.min <= rule.max,
      "Invalid market rule range",
    );
  }
  for (const value of [rules.minNotional, rules.maxNotional])
    if (value !== null) positive(value, "notional rule");
  assert(
    rules.minNotional === null ||
      rules.maxNotional === null ||
      rules.minNotional <= rules.maxNotional,
    "Invalid notional range",
  );
  for (const [i, funding] of (config.funding ?? []).entries()) {
    assert(
      Number.isSafeInteger(funding.timestamp) &&
        funding.timestamp >= 0 &&
        Number.isFinite(funding.rate),
      "Invalid funding record",
    );
    positive(funding.markPrice, "funding markPrice");
    assert(
      i === 0 || funding.timestamp > config.funding![i - 1].timestamp,
      "Funding records must be chronological and unique",
    );
  }
}

function validateCandle(candle: TradingCandle): void {
  for (const value of [candle.open, candle.high, candle.low, candle.close])
    positive(value, "candle price");
  assert(
    Number.isSafeInteger(candle.openTime) &&
      candle.openTime >= 0 &&
      Number.isSafeInteger(candle.closeTime) &&
      candle.closeTime >= candle.openTime &&
      candle.low <= Math.min(candle.open, candle.close) &&
      candle.high >= Math.max(candle.open, candle.close) &&
      Number.isFinite(candle.volume) &&
      candle.volume >= 0,
    "Invalid candle",
  );
  for (const volume of [
    candle.aggressiveBuyVolume,
    candle.aggressiveSellVolume,
  ]) {
    assert(
      volume === undefined ||
        (Number.isFinite(volume) && volume >= 0 && volume <= candle.volume),
      "Invalid aggressive volume",
    );
  }
}
