import { BaseTradingApi } from "./base-trading-api.js";
import { randomUUID } from "node:crypto";
import type {
  TradingOrderSnapshot,
  TradingOrderResult,
  TradingOrderCapacityRequest,
  TradingOrderCapacity,
} from "@trading/bot-algo/trading-api";
import { floorToStep, positive, validateOrder } from "./common.js";
import {
  BinanceOrderSubmissionError,
  BinanceUnsupportedOperationError,
  errorCode,
  orderStatus,
} from "./binance-common.js";

/** Cash capacity and submission rules shared by Spot and options. */
export abstract class BinanceCashApi extends BaseTradingApi {
  abstract initialize(): Promise<void>;
  protected abstract send(
    order: TradingOrderSnapshot,
    clientId: string,
  ): Promise<{ status: string }>;
  protected abstract readonly product: "spot" | "options";

  protected async submit(
    order: TradingOrderSnapshot,
  ): Promise<TradingOrderResult> {
    validateOrder(order);
    if (this.product === "spot" && order.reduceOnly)
      throw new BinanceUnsupportedOperationError("spot", "reduceOnly");
    if (this.product === "options" && order.type !== "limit")
      throw new BinanceUnsupportedOperationError(
        "options",
        `${order.type} orders; use limit orders`,
      );
    if (
      this.product === "options" &&
      order.side === "sell" &&
      !order.reduceOnly
    )
      throw new BinanceUnsupportedOperationError(
        "options",
        "uncovered option writing; sell with reduceOnly",
      );
    await this.initialize();
    const clientId = randomUUID();
    try {
      const response = await this.send(order, clientId);
      const completed: TradingOrderSnapshot = {
        ...order,
        id: `${this.product}:${clientId}`,
        leverage: 1,
        status: orderStatus(response.status),
      };
      return {
        accepted: !["rejected", "canceled", "expired"].includes(
          completed.status,
        ),
        order: completed,
      };
    } catch (error) {
      if ([-2010, -2019, -2021, -2022, -1013].includes(errorCode(error) ?? 0))
        return {
          accepted: false,
          order: {
            ...order,
            id: `${this.product}:${clientId}`,
            leverage: 1,
            status: "rejected",
          },
        };
      throw new BinanceOrderSubmissionError(
        clientId,
        order.stopPrice !== null,
        error,
      );
    }
  }

  async getOrderCapacity(
    input: TradingOrderCapacityRequest,
  ): Promise<TradingOrderCapacity> {
    positive(input.price, "price");
    positive(input.leverage, "leverage");
    if (input.side !== "buy" && input.side !== "sell")
      throw new Error("Invalid order side");
    const [equity, rules, friction] = await Promise.all([
      this.getEquity(),
      this.getMarketRules(),
      this.getFriction(),
    ]);
    const quantityRules =
      this.product === "options" ? rules.limitQuantity : rules.marketQuantity;
    const available =
      input.side === "buy"
        ? equity.quoteAvailable / (input.price * (1 + friction))
        : equity.assetAvailable;
    const size = floorToStep(
      Math.max(
        0,
        Math.min(
          available,
          quantityRules.max ?? Infinity,
          (rules.maxNotional ?? Infinity) / input.price,
        ),
      ),
      quantityRules.step,
    );
    const quote = size * input.price;
    return {
      quote:
        size < (quantityRules.min ?? 0) || quote < (rules.minNotional ?? 0)
          ? 0
          : quote,
      leverage: 1,
    };
  }

  protected clientId(id: string): string {
    const match = new RegExp(`^${this.product}:([a-zA-Z0-9_-]{1,36})$`).exec(
      id,
    );
    if (!match) throw new Error(`Invalid ${this.product} order ID`);
    return match[1];
  }
}
