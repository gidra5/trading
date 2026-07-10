type TradingStrategyEntrySignal = {
  side: TradingSide;
  size: number;

  // if present - the price that is anticipated by the entry signal.
  price: number | null;
  // if present - skewness of the price distribution around the anticipated price.
  confidence: number | null;
};
type TradingStrategyExitSignal = {
  side: TradingSide;

  // if present - the price that is anticipated by the entry signal.
  price: number | null;
  // if present - skewness of the price distribution around the anticipated price.
  // 0 - uniform, 1 - everything at the target price
  confidence: number | null;
};

interface TradingStrategy<T> {
  onTick: (price: number, quantity: number) => Promise<void>;
  onOrder: (id: string, status: TradingOrderStatus) => Promise<void>;

  snapshot: () => Promise<T>;
  restore: (snapshot: T | null) => Promise<void>;

  entrySignal: () => Promise<TradingStrategyEntrySignal | null>;
  exitSignal: () => Promise<TradingStrategyExitSignal | null>;
}

class Strategy implements TradingStrategy<unknown> {}
