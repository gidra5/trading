import {
  putCandleShard,
  SequentialShardStore,
  TradingStorageLayout,
  type SequentialCandle,
} from "@trading/storage";

export async function writeCanonicalCandleDay(
  dataDir: string,
  candles: readonly SequentialCandle[],
  options: {
    date?: string;
    marketKey?: string;
    stepMs?: number;
    symbol?: string;
    interval?: string;
  } = {},
): Promise<string> {
  const first = candles[0];
  const symbol = first?.symbol ?? options.symbol ?? "BTCUSDT";
  const interval = first?.interval ?? options.interval ?? "1s";
  const date = first
    ? new Date(first.openTime).toISOString().slice(0, 10)
    : options.date;
  if (!date) throw new Error("An empty canonical candle fixture requires its date.");
  const layout = new TradingStorageLayout(dataDir);
  const store = new SequentialShardStore(layout.marketStore);
  const result = await putCandleShard(store, {
    namespace: [
      "candles",
      options.marketKey ?? `spot-${symbol.toLowerCase()}`,
      symbol.toLowerCase(),
      interval,
    ].join("/"),
    key: date,
    candles,
    stepMs: options.stepMs,
    ...(first ? {} : {
      empty: {
        symbol,
        interval,
        startTime: Date.parse(`${date}T00:00:00.000Z`),
        closeTimeOffsetMs: (options.stepMs ?? 1_000) - 1,
        closed: true,
      },
    }),
    metadata: { source: "test-fixture", completeUtcDay: false },
  });
  return result.referenceFile;
}
