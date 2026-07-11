import type {
  TradingApi,
  TradingOrderEvent,
  TradingTick,
} from "@trading/bot-algo";

export interface EquitySnapshot {
  quoteAvailable: number;
  quoteReserved: number;
  quoteUnleveraged: number;
  assetAvailable: number;
  assetReserved: number;
  assetUnleveraged: number;
}

export interface RuntimeTradingApi extends TradingApi {
  getEquity(): Promise<EquitySnapshot>;
  onTick(tick: TradingTick): Promise<void>;
  drainEvents(): TradingOrderEvent[];
}
