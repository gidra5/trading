import type { MainClient, USDMClient } from "binance";
import type { TradingApi } from "@trading/bot-algo/trading-api";

export type BinanceEnvironment = "live" | "testnet" | "demo";
export type BinanceProduct = "spot" | "futures" | "options" | "predictions";

export interface BinanceClientConfig {
  readonly environment: BinanceEnvironment;
  readonly apiKey: string;
  readonly apiSecret: string;
  readonly timeoutMs?: number;
}

export interface BinanceMarketConfig {
  readonly symbol: string;
  readonly product: BinanceProduct;
  /** Exponential decay across the size range: 0 is uniform; positive favors smaller sizes. Default 5. */
  readonly frictionDecayRate?: number;
}

export type FuturesTransport = Pick<
  USDMClient,
  | "getExchangeInfo"
  | "getKlines"
  | "getAggregateTrades"
  | "getOrderBook"
  | "getFuturesAccountConfig"
  | "getFuturesSymbolConfig"
  | "setLeverage"
  | "getNotionalAndLeverageBrackets"
  | "getAccountInformationV3"
  | "getAccountCommissionRate"
  | "submitNewOrder"
  | "submitNewAlgoOrder"
  | "cancelOrder"
  | "cancelAlgoOrder"
>;

export type SpotTransport = Pick<
  MainClient,
  | "getExchangeInfo"
  | "getKlines"
  | "getAggregateTrades"
  | "getOrderBook"
  | "getAccountInformation"
  | "getCommissionRates"
  | "submitNewOrder"
  | "cancelOrder"
>;

/** SDK generic methods provide signing/transport for products without named methods. */
export type SignedTransport = Pick<
  MainClient,
  "get" | "getPrivate" | "postPrivate" | "deletePrivate"
>;

export interface BinanceTransport {
  readonly futures?: FuturesTransport;
  readonly spot?: SpotTransport;
  readonly options?: SignedTransport;
  readonly predictions?: SignedTransport;
}

export interface BinanceMarketApi extends TradingApi {
  initialize(): Promise<void>;
}
