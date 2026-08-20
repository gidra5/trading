type Side = "bid" | "ask";

interface FlowCounters {
  bidAddedQuote: number;
  bidRemovedQuote: number;
  askAddedQuote: number;
  askRemovedQuote: number;
}

interface BookView {
  valid: boolean;
  ageMs: number | null;
  bid: number | null;
  ask: number | null;
  bidQty: number | null;
  askQty: number | null;
  mid: number | null;
  spreadBps: number | null;
  l1Imbalance: number | null;
  top5Imbalance: number | null;
}

class OrderBook {
  readonly bids = new Map<number, number>();
  readonly asks = new Map<number, number>();
  valid = false;
  lastReceivedAt = 0;

  replace(bids: unknown, asks: unknown, receivedAt: number) {
    this.bids.clear();
    this.asks.clear();
    this.applyLevels("bid", bids, undefined);
    this.applyLevels("ask", asks, undefined);
    this.valid = this.bids.size > 0 && this.asks.size > 0;
    this.lastReceivedAt = receivedAt;
  }

  update(side: Side, price: unknown, quantity: unknown, receivedAt: number, flow?: FlowCounters) {
    const numericPrice = Number(price);
    const numericQuantity = Number(quantity);
    if (!(numericPrice > 0) || !(numericQuantity >= 0)) return;
    const levels = side === "bid" ? this.bids : this.asks;
    const previous = levels.get(numericPrice) ?? 0;
    if (numericQuantity === 0) levels.delete(numericPrice);
    else levels.set(numericPrice, numericQuantity);
    if (flow && numericQuantity !== previous) {
      const quoteChange = Math.abs(numericQuantity - previous) * numericPrice;
      if (side === "bid") {
        if (numericQuantity > previous) flow.bidAddedQuote += quoteChange;
        else flow.bidRemovedQuote += quoteChange;
      } else if (numericQuantity > previous) flow.askAddedQuote += quoteChange;
      else flow.askRemovedQuote += quoteChange;
    }
    this.lastReceivedAt = receivedAt;
    this.valid = this.bids.size > 0 && this.asks.size > 0;
  }

  invalidate() {
    this.valid = false;
  }

  truncate(depth: number) {
    for (const [price] of this.sorted("bid", Number.POSITIVE_INFINITY).slice(depth)) this.bids.delete(price);
    for (const [price] of this.sorted("ask", Number.POSITIVE_INFINITY).slice(depth)) this.asks.delete(price);
    this.valid = this.bids.size > 0 && this.asks.size > 0;
  }

  isStructurallyValid() {
    const bid = this.sorted("bid", 1)[0]?.[0];
    const ask = this.sorted("ask", 1)[0]?.[0];
    return this.valid && bid !== undefined && ask !== undefined && bid < ask;
  }

  view(now: number): BookView {
    const bids = this.sorted("bid", 5);
    const asks = this.sorted("ask", 5);
    const bestBid = bids[0];
    const bestAsk = asks[0];
    const structurallyValid = this.valid && Boolean(bestBid && bestAsk && bestBid[0] < bestAsk[0]);
    if (!structurallyValid) return emptyView(this.lastReceivedAt > 0 ? now - this.lastReceivedAt : null);
    const bid = bestBid![0];
    const ask = bestAsk![0];
    const bidQty = bestBid![1];
    const askQty = bestAsk![1];
    const mid = (bid + ask) / 2;
    const top5Bid = bids.reduce((sum, [, quantity]) => sum + quantity, 0);
    const top5Ask = asks.reduce((sum, [, quantity]) => sum + quantity, 0);
    return {
      valid: true,
      ageMs: Math.max(0, now - this.lastReceivedAt),
      bid,
      ask,
      bidQty,
      askQty,
      mid,
      spreadBps: 10_000 * (ask - bid) / mid,
      l1Imbalance: imbalance(bidQty, askQty),
      top5Imbalance: imbalance(top5Bid, top5Ask),
    };
  }

  private applyLevels(side: Side, input: unknown, flow?: FlowCounters) {
    if (!Array.isArray(input)) return;
    for (const level of input) {
      if (Array.isArray(level)) this.update(side, level[0], level[1], this.lastReceivedAt, flow);
      else if (level && typeof level === "object") {
        const row = level as Record<string, unknown>;
        this.update(side, row.price, row.qty ?? row.quantity, this.lastReceivedAt, flow);
      }
    }
  }

  private sorted(side: Side, limit: number) {
    return [...(side === "bid" ? this.bids : this.asks).entries()]
      .sort((left, right) => side === "bid" ? right[0] - left[0] : left[0] - right[0])
      .slice(0, limit);
  }
}

function emptyFlow(): FlowCounters {
  return { bidAddedQuote: 0, bidRemovedQuote: 0, askAddedQuote: 0, askRemovedQuote: 0 };
}

function emptyView(ageMs: number | null): BookView {
  return {
    valid: false,
    ageMs,
    bid: null,
    ask: null,
    bidQty: null,
    askQty: null,
    mid: null,
    spreadBps: null,
    l1Imbalance: null,
    top5Imbalance: null,
  };
}

function imbalance(bid: number, ask: number) {
  const total = bid + ask;
  return total > 0 ? (bid - ask) / total : null;
}

export class LiveBookCompactor {
  private readonly binanceSpot = new OrderBook();
  private readonly coinbase = new OrderBook();
  private readonly kraken = new OrderBook();
  private readonly deribit = new OrderBook();
  private readonly binanceFutures = new OrderBook();
  private readonly pendingBinanceDiffs: Array<{ message: any; receivedAt: number }> = [];
  private binanceLastUpdateId: number | undefined;
  private binanceFlow = emptyFlow();
  private intervalStart = Date.now();

  needsBinanceSnapshot() {
    return this.binanceLastUpdateId === undefined;
  }

  needsKrakenSnapshot() {
    return this.kraken.lastReceivedAt > 0 && !this.kraken.isStructurallyValid();
  }

  consume(source: string, message: any, receivedAt = Date.now()) {
    if (!message || typeof message !== "object") return;
    if (source === "binance-spot-depth-snapshot") this.consumeBinanceSnapshot(message, receivedAt);
    else if (source === "binance-spot-depth-diff") this.consumeBinanceDiff(message, receivedAt);
    else if (source === "coinbase-btcusd-level2") this.consumeCoinbase(message, receivedAt);
    else if (source === "kraken-btcusd-book") this.consumeKraken(message, receivedAt);
    else if (source === "deribit-btc-perpetual-book") this.consumeDeribit(message, receivedAt);
    else if (source === "binance-usdm-book-ticker") this.consumeFuturesBbo(message, receivedAt);
  }

  summarize(observedAt = Date.now()) {
    const venues = {
      binanceSpot: this.binanceSpot.view(observedAt),
      coinbaseSpot: this.coinbase.view(observedAt),
      krakenSpot: this.kraken.view(observedAt),
      deribitPerpetual: this.deribit.view(observedAt),
      binancePerpetual: this.binanceFutures.view(observedAt),
    };
    const freshSpot = [venues.binanceSpot, venues.coinbaseSpot, venues.krakenSpot]
      .filter((row) => row.valid && row.ageMs !== null && row.ageMs <= 5_000 && row.mid !== null);
    const mids = freshSpot.map((row) => row.mid!);
    const center = mids.length > 0 ? mids.reduce((sum, value) => sum + value, 0) / mids.length : null;
    const bestBid = maximum(freshSpot.map((row) => row.bid));
    const bestAsk = minimum(freshSpot.map((row) => row.ask));
    const futures = venues.binancePerpetual;
    const result = {
      schemaVersion: 1,
      intervalStart: this.intervalStart,
      observedAt,
      venues,
      binanceSpotFlow: {
        ...this.binanceFlow,
        netBidQuote: this.binanceFlow.bidAddedQuote - this.binanceFlow.bidRemovedQuote,
        netAskQuote: this.binanceFlow.askAddedQuote - this.binanceFlow.askRemovedQuote,
        note: "Removed depth includes cancellations and trade depletion; L2 updates cannot separate them.",
      },
      crossVenue: {
        freshSpotVenueCount: freshSpot.length,
        spotMidDispersionBps: center && mids.length > 1
          ? 10_000 * (Math.max(...mids) - Math.min(...mids)) / center
          : null,
        bestExecutableSpreadBps: center && bestBid !== null && bestAsk !== null
          ? 10_000 * (bestBid - bestAsk) / center
          : null,
        binancePerpetualBasisBps: center && futures.valid && futures.mid !== null && futures.ageMs !== null && futures.ageMs <= 5_000
          ? 10_000 * (futures.mid - center) / center
          : null,
      },
    };
    this.intervalStart = observedAt;
    this.binanceFlow = emptyFlow();
    return result;
  }

  private consumeBinanceSnapshot(message: any, receivedAt: number) {
    const lastUpdateId = Number(message.lastUpdateId);
    if (!Number.isFinite(lastUpdateId)) return;
    this.binanceSpot.replace(message.bids, message.asks, receivedAt);
    this.binanceLastUpdateId = lastUpdateId;
    const buffered = this.pendingBinanceDiffs.splice(0);
    for (const row of buffered) this.applyBinanceDiff(row.message, row.receivedAt);
  }

  private consumeBinanceDiff(message: any, receivedAt: number) {
    if (this.binanceLastUpdateId === undefined) {
      this.pendingBinanceDiffs.push({ message, receivedAt });
      if (this.pendingBinanceDiffs.length > 20_000) this.pendingBinanceDiffs.shift();
      return;
    }
    this.applyBinanceDiff(message, receivedAt);
  }

  private applyBinanceDiff(message: any, receivedAt: number) {
    const first = Number(message.U);
    const last = Number(message.u);
    if (!Number.isFinite(first) || !Number.isFinite(last) || this.binanceLastUpdateId === undefined) return;
    if (last <= this.binanceLastUpdateId) return;
    const next = this.binanceLastUpdateId + 1;
    if (!(first <= next && next <= last)) {
      this.binanceSpot.invalidate();
      this.binanceLastUpdateId = undefined;
      this.pendingBinanceDiffs.length = 0;
      this.pendingBinanceDiffs.push({ message, receivedAt });
      return;
    }
    for (const level of Array.isArray(message.b) ? message.b : []) {
      this.binanceSpot.update("bid", level[0], level[1], receivedAt, this.binanceFlow);
    }
    for (const level of Array.isArray(message.a) ? message.a : []) {
      this.binanceSpot.update("ask", level[0], level[1], receivedAt, this.binanceFlow);
    }
    this.binanceLastUpdateId = last;
  }

  private consumeCoinbase(message: any, receivedAt: number) {
    if (message.type === "snapshot") this.coinbase.replace(message.bids, message.asks, receivedAt);
    else if (message.type === "l2update") {
      for (const change of Array.isArray(message.changes) ? message.changes : []) {
        if (!Array.isArray(change)) continue;
        this.coinbase.update(change[0] === "buy" ? "bid" : "ask", change[1], change[2], receivedAt);
      }
    }
  }

  private consumeKraken(message: any, receivedAt: number) {
    if (message.channel !== "book" || !Array.isArray(message.data)) return;
    const data = message.data[0];
    if (!data) return;
    if (message.type === "snapshot") this.kraken.replace(data.bids, data.asks, receivedAt);
    else if (message.type === "update") {
      for (const level of Array.isArray(data.bids) ? data.bids : []) this.kraken.update("bid", level.price, level.qty, receivedAt);
      for (const level of Array.isArray(data.asks) ? data.asks : []) this.kraken.update("ask", level.price, level.qty, receivedAt);
      // Kraken does not send qty=0 for levels that merely fall outside the
      // subscribed depth, so retaining more than 100 levels eventually makes
      // a reversed market path expose stale levels as the top of book.
      this.kraken.truncate(100);
    }
  }

  private consumeDeribit(message: any, receivedAt: number) {
    if (!String(message?.params?.channel ?? "").startsWith("book.")) return;
    const data = message.params.data;
    if (!data) return;
    if (data.type === "snapshot") {
      this.deribit.replace(actionLevels(data.bids), actionLevels(data.asks), receivedAt);
      return;
    }
    for (const level of Array.isArray(data.bids) ? data.bids : []) this.deribit.update("bid", level[1], level[0] === "delete" ? 0 : level[2], receivedAt);
    for (const level of Array.isArray(data.asks) ? data.asks : []) this.deribit.update("ask", level[1], level[0] === "delete" ? 0 : level[2], receivedAt);
  }

  private consumeFuturesBbo(message: any, receivedAt: number) {
    const bid = Number(message.b);
    const ask = Number(message.a);
    if (!(bid > 0) || !(ask > bid)) return;
    this.binanceFutures.replace([[bid, Number(message.B)]], [[ask, Number(message.A)]], receivedAt);
  }
}

function actionLevels(levels: unknown) {
  if (!Array.isArray(levels)) return [];
  return levels.filter(Array.isArray).map((level) => [level[1], level[0] === "delete" ? 0 : level[2]]);
}

function maximum(values: Array<number | null>) {
  const finite = values.filter((value): value is number => Number.isFinite(value));
  return finite.length > 0 ? Math.max(...finite) : null;
}

function minimum(values: Array<number | null>) {
  const finite = values.filter((value): value is number => Number.isFinite(value));
  return finite.length > 0 ? Math.min(...finite) : null;
}
