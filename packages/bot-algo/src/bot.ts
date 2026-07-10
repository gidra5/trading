type Candle = {
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
};
type CreateStopLimitOrderInput = {
  side: TradingSide;
  size: number;
  stopPrice: number;
  limitPrice: number;
};

type CreateStopMarketOrderInput = {
  side: TradingSide;
  size: number;
  price: number;
};

type CreateLimitOrderInput = {
  side: TradingSide;
  size: number;
  price: number;
};

type CreateMarketOrderInput = {
  side: TradingSide;
  size: number;
};

type TradingHistoryRequest = {
  intervalMs: number;
  count: number;
};

export interface TradingApi {
  createStopLimitOrder(input: CreateStopLimitOrderInput): Promise<TradingOrder>;
  createStopMarketOrder(input: CreateStopMarketOrderInput): Promise<TradingOrder>;
  createLimitOrder(input: CreateLimitOrderInput): Promise<TradingOrder>;
  createMarketOrder(input: CreateMarketOrderInput): Promise<TradingOrder>;
  cancelOrder(id: string): Promise<boolean>;
  getHistory(input: TradingHistoryRequest): Promise<Candle[]>;
  getEquity(): Promise<EquitySnapshot>;
  getFriction(): Promise<number>;
}

export interface BotMetricsReporter<TSnapshot extends StrategySnapshot = StrategySnapshot> {
  onOrderCreated?(order: TradingOrder): void | Promise<void>;
  onOrderEvent?(event: ExecutionEvent): void | Promise<void>;
  onSnapshot?(snapshot: BotRuntimeSnapshot<TSnapshot>): void | Promise<void>;
}

export interface BotOptions<TSnapshot extends StrategySnapshot = StrategySnapshot> {
  config: TradingBotConfig;
  reporter: BotMetricsReporter<TSnapshot>;
}

export interface BotLotBorrow {
  lotId: string;
  amountBorrowed: number;
}

export interface BotLotGridOrder {
  id: string;
  size: number;
  price: number;
  filled: number;
}

export interface BotLotGrid {
  orders: BotLotGridOrder[];
  priceAnchor: number;
}

export interface BotLot {
  id: string;
  side: TradingSide;
  quote: number;
  asset: number;
  internalBorrow: BotLotBorrow[];
  externalBorrow: number;
  entryGrid: BotLotGrid | null;
  exitGrid: BotLotGrid | null;
}

class Bot implements TradingBot {
  private tradingApi: TradingApi;
  private tradingStrategy: TradingStrategy<unknown>;
  private reporter: BotMetricsReporter;
  private lots: BotLot[] = [];
  private config: BotConfig;
  private indicators = {};

  constructor(options: BotOptions<unknown>) {
    this.tradingApi = tradingApi;
    this.tradingStrategy = new Strategy();
    this.reporter = reporter;
  }

  warmup(): void {}

  async onTick(price: number, quantity: number): Promise<void> {
    await this.tradingStrategy.onTick(price, quantity);
  }

  async onOrder(id: string, status: TradingOrderStatus): Promise<void> {
    await this.tradingStrategy.onOrder(id, status);
  }

  async snapshot(): Promise<T> {
    return this.tradingStrategy.snapshot();
  }

  async restore(snapshot: T | null): Promise<void> {
    await this.tradingStrategy.restore(snapshot);
  }
}
