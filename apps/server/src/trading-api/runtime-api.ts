import type {
  TradingApi,
  TradingEquitySnapshot,
  TradingOrderEvent,
  TradingTick,
} from "@trading/bot-algo";

export type EquitySnapshot = TradingEquitySnapshot;

export interface RuntimeTradingApi extends TradingApi {
  onTick(tick: TradingTick): Promise<void>;
  drainEvents(): TradingOrderEvent[];
}
