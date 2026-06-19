import type {
  BacktestProgressSnapshot,
  BacktestPreset,
  BotEvent,
  Candle,
  OrderBookSnapshot,
  PaperBotState,
  PositionLedger,
} from "@trading/bot-algo";

export interface RuntimeSnapshot {
  market: {
    symbol: string;
    interval: string;
    connected: boolean;
    statusMessage: string;
    lastEventAt: number;
    lastPrice: number;
    candles: Candle[];
    orderBook?: OrderBookSnapshot;
  };
  bot: PaperBotState;
  positions: PositionLedger;
  recentEvents: BotEvent[];
  backtest: BacktestProgressSnapshot;
}

export type BacktestSelection = BacktestPreset;
export type { BacktestProgressSnapshot };
