import { z } from "zod";
import type { SignedTransport } from "./binance-types.js";
import { BinanceOrderSubmissionError } from "./binance-common.js";

export interface BinancePredictionConfig {
  readonly product: "predictions";
  readonly walletAddress: string;
  readonly walletId: string;
  readonly tokenId: string;
  readonly marketId: number;
  readonly vendor: string;
  readonly accountType: "SPOT" | "FUNDING";
  readonly fundingSource: "MPC" | "CEX";
}

/** BUY spends collateral; SELL spends shares. Decimal strings avoid wei rounding. */
export type PredictionOrderInput = {
  readonly side: "buy" | "sell";
  readonly amount: string;
} & (
  | { readonly type: "market" }
  | { readonly type: "limit"; readonly price: string }
);

const wei = z.string().regex(/^\d+$/);
const quoteSchema = z.object({
  quoteId: z.string().min(1),
  tokenId: z.string(),
  walletAddress: z.string(),
  side: z.enum(["BUY", "SELL"]),
  orderType: z.enum(["MARKET", "LIMIT"]),
  amountIn: wei,
  amountOut: wei,
  feeAmount: wei,
  minReceive: wei,
  averagePrice: z.number().positive(),
  priceImpact: z.number().finite(),
  expireAt: z.number().int(),
  priceLimit: z.string().nullable(),
});
export type PredictionQuote = Readonly<z.infer<typeof quoteSchema>>;

export function predictionAmountToWei(value: string): string {
  if (!/^(0|[1-9]\d*)(\.\d{1,18})?$/.test(value))
    throw new Error(
      "Prediction amount must be a positive decimal with at most 18 places",
    );
  const [whole, fraction = ""] = value.split(".");
  const result = BigInt(whole) * 10n ** 18n + BigInt(fraction.padEnd(18, "0"));
  if (!result) throw new Error("Prediction amount must be positive");
  return result.toString();
}
const prefix = "sapi/v1/w3w/wallet/prediction/";

/** Explicit quote API: exact collateral spend cannot promise exact base size. */
export class BinancePredictionApi {
  private readonly quotes = new Map<
    string,
    { quote: PredictionQuote; price?: string }
  >();
  constructor(
    private readonly config: BinancePredictionConfig,
    private readonly client: SignedTransport,
  ) {
    if (
      !config.walletAddress.trim() ||
      !config.walletId.trim() ||
      !config.tokenId.trim() ||
      !config.vendor.trim()
    )
      throw new Error("Prediction wallet, token and vendor are required");
    if (!Number.isSafeInteger(config.marketId) || config.marketId < 1)
      throw new Error("Invalid prediction marketId");

    if (
      !["SPOT", "FUNDING"].includes(config.accountType) ||
      !["MPC", "CEX"].includes(config.fundingSource)
    )
      throw new Error("Invalid prediction funding configuration");
  }
  async initialize(): Promise<void> {
    const result = z
      .object({
        wallets: z.array(
          z.object({ walletId: z.string(), walletAddress: z.string() }),
        ),
      })
      .parse(await this.client.getPrivate(prefix + "wallet/list"));
    if (
      !result.wallets.some(
        (w) =>
          w.walletId === this.config.walletId &&
          w.walletAddress.toLowerCase() ===
            this.config.walletAddress.toLowerCase(),
      )
    )
      throw new Error(
        "Prediction wallet is not registered for these credentials",
      );
  }
  async getQuote(input: PredictionOrderInput): Promise<PredictionQuote> {
    if (input.side !== "buy" && input.side !== "sell")
      throw new Error("Invalid order side");
    if (input.type !== "market" && input.type !== "limit")
      throw new Error("Predictions support market and limit orders");
    const amountIn = predictionAmountToWei(input.amount);
    if (input.type === "limit") {
      predictionAmountToWei(input.price);
      if (Number(input.price) >= 1)
        throw new Error("Prediction price must be between 0 and 1");
    }
    const quote = quoteSchema.parse(
      await this.client.postPrivate(prefix + "trade/get-quote", {
        walletAddress: this.config.walletAddress,
        tokenId: this.config.tokenId,
        side: input.side === "buy" ? "BUY" : "SELL",
        amountIn,
        orderType: input.type === "market" ? "MARKET" : "LIMIT",
        ...(input.type === "limit" ? { priceLimit: input.price } : {}),
      }),
    );
    if (
      quote.tokenId !== this.config.tokenId ||
      quote.walletAddress.toLowerCase() !==
        this.config.walletAddress.toLowerCase() ||
      quote.side !== input.side.toUpperCase() ||
      quote.orderType !== input.type.toUpperCase() ||
      quote.amountIn !== amountIn ||
      (input.type === "limit" &&
        (quote.priceLimit === null ||
          predictionAmountToWei(quote.priceLimit) !==
            predictionAmountToWei(input.price)))
    )
      throw new Error("Prediction quote does not match the request");
    for (const [id, stored] of this.quotes)
      if (stored.quote.expireAt <= Date.now()) this.quotes.delete(id);
    this.quotes.set(quote.quoteId, {
      quote,
      ...(input.type === "limit" ? { price: input.price } : {}),
    });
    return quote;
  }
  /** One submission attempt per locally obtained quote, even after an uncertain response. */
  async placeQuote(
    quoteId: string,
  ): Promise<{ id: string; status: "pending" }> {
    const stored = this.quotes.get(quoteId);
    if (!stored)
      throw new Error("Unknown or already submitted prediction quote");
    const { quote, price } = stored;
    if (quote.expireAt <= Date.now()) {
      this.quotes.delete(quoteId);
      throw new Error("Prediction quote expired");
    }
    this.quotes.delete(quoteId);
    try {
      const result = z.object({ orderId: z.string().min(1) }).parse(
        await this.client.postPrivate(prefix + "trade/place-order-bundle", {
          walletAddress: this.config.walletAddress,
          walletId: this.config.walletId,
          quoteId,
          orderType: quote.orderType,
          timeInForce: quote.orderType === "MARKET" ? "FOK" : "GTC",
          accountType: this.config.accountType,
          fundingSource: this.config.fundingSource,
          ...(price === undefined ? {} : { priceLimit: price }),
        }),
      );
      return { id: `prediction:${result.orderId}`, status: "pending" };
    } catch (error) {
      // The quote ID is the reconciliation key; this venue has no client-order-ID field.
      throw new BinanceOrderSubmissionError(quoteId, false, error);
    }
  }
  async cancelOrder(id: string): Promise<boolean> {
    const match = /^prediction:([a-zA-Z0-9_-]+)$/.exec(id);
    if (!match) throw new Error("Invalid prediction order ID");
    const result = z
      .object({
        canceled: z.array(z.string()),
        failed: z.array(z.object({ orderId: z.string(), reason: z.string() })),
      })
      .parse(
        await this.client.postPrivate(prefix + "trade/batch-cancel", {
          walletAddress: this.config.walletAddress,
          walletId: this.config.walletId,
          "cancelInfoList[0].orderId": match[1],
        }),
      );
    if (result.canceled.includes(match[1])) return true;
    const failure = result.failed.find((f) => f.orderId === match[1]);
    if (failure?.reason === "ORDER_NOT_FOUND") return false;
    throw new Error(
      `Prediction cancellation failed: ${failure?.reason ?? "missing result"}`,
    );
  }
  async getOrderBook() {
    return z
      .object({
        tokenId: z.string(),
        timestamp: z.number(),
        bids: z.array(z.object({ price: z.string(), size: z.string() })),
        asks: z.array(z.object({ price: z.string(), size: z.string() })),
      })
      .parse(
        await this.client.getPrivate(prefix + "order-book", {
          vendor: this.config.vendor,
          marketId: this.config.marketId,
          tokenId: this.config.tokenId,
        }),
      );
  }
  async getPosition() {
    return z
      .object({
        position: z
          .object({
            tokenId: z.string(),
            shares: z.string(),
            avgPrice: z.string(),
            value: z.string(),
            currentPrice: z.string(),
            positionStatus: z.string(),
          })
          .nullable(),
      })
      .parse(
        await this.client.getPrivate(prefix + "position/token", {
          walletAddress: this.config.walletAddress,
          tokenId: this.config.tokenId,
        }),
      ).position;
  }
  /** CEX payment balances; these do not describe MPC cash or reserved positions. */
  async getPaymentBalances() {
    return z
      .object({
        items: z.array(
          z.object({
            accountType: z.string(),
            availableBalanceDisplay: z.string(),
            enabled: z.boolean(),
          }),
        ),
      })
      .parse(await this.client.getPrivate(prefix + "balance/payment-options"))
      .items;
  }
  async getActiveOrders(offset = 0, limit = 100) {
    if (
      !Number.isSafeInteger(offset) ||
      offset < 0 ||
      !Number.isInteger(limit) ||
      limit < 1 ||
      limit > 100
    )
      throw new Error("Invalid prediction order pagination");
    return z
      .object({
        total: z.number().int(),
        orders: z.array(
          z.object({
            orderId: z.string(),
            marketId: z.number(),
            outcomeIndex: z.number(),
            status: z.string(),
            side: z.string(),
            makerUsdtAmount: z.string(),
            makerShareQty: z.string(),
            filledUsdtAmount: z.string(),
            filledShareQty: z.string(),
          }),
        ),
      })
      .parse(
        await this.client.getPrivate(prefix + "order/list", {
          walletAddress: this.config.walletAddress,
          marketId: this.config.marketId,
          offset,
          limit,
        }),
      );
  }
}
