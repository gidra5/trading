import WebSocket from "ws";
import type { Candle, OrderBookSnapshot, PriceTick } from "@trading/bot-algo";

export interface MarketStreamStatus {
  connected: boolean;
  message: string;
  lastEventAt: number;
  reconnectAttempt: number;
}

export interface MarketStreamHandlers {
  onTick: (tick: PriceTick) => void | Promise<void>;
  onCandle: (candle: Candle) => void | Promise<void>;
  onOrderBook: (snapshot: OrderBookSnapshot) => void | Promise<void>;
  onStatus: (status: MarketStreamStatus) => void;
}

export class BinanceMarketStream {
  private socket?: WebSocket;
  private reconnectTimer?: NodeJS.Timeout;
  private reconnectAttempt = 0;
  private stopped = true;

  constructor(
    private readonly streamSymbol: string,
    private readonly interval: string,
    private readonly handlers: MarketStreamHandlers,
  ) {}

  start(): void {
    if (!this.stopped) {
      return;
    }

    this.stopped = false;
    this.connect();
  }

  stop(): void {
    this.stopped = true;
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
    }

    this.socket?.close();
    this.socket = undefined;
  }

  private connect(): void {
    const streams = [
      `${this.streamSymbol}@trade`,
      `${this.streamSymbol}@kline_${this.interval}`,
      `${this.streamSymbol}@depth10@1000ms`,
    ].join("/");
    const url = `wss://stream.binance.com:9443/stream?streams=${streams}`;

    this.socket = new WebSocket(url);

    this.socket.on("open", () => {
      this.reconnectAttempt = 0;
      this.handlers.onStatus({
        connected: true,
        message: "Connected to Binance market streams",
        lastEventAt: Date.now(),
        reconnectAttempt: this.reconnectAttempt,
      });
    });

    this.socket.on("message", (raw) => {
      this.handleMessage(raw.toString());
    });

    this.socket.on("close", () => {
      this.handlers.onStatus({
        connected: false,
        message: "Binance stream closed",
        lastEventAt: Date.now(),
        reconnectAttempt: this.reconnectAttempt,
      });
      this.scheduleReconnect();
    });

    this.socket.on("error", (error) => {
      this.handlers.onStatus({
        connected: false,
        message: `Binance stream error: ${error.message}`,
        lastEventAt: Date.now(),
        reconnectAttempt: this.reconnectAttempt,
      });
    });
  }

  private handleMessage(raw: string): void {
    const payload = JSON.parse(raw) as { stream?: string; data?: unknown };
    const stream = payload.stream ?? "";
    const data = payload.data as Record<string, unknown> | undefined;

    if (!data) {
      return;
    }

    if (stream.includes("@trade")) {
      const tick = parseTrade(data);
      if (tick) {
        void this.handlers.onTick(tick);
      }
      return;
    }

    if (stream.includes("@kline")) {
      const candle = parseKline(data);
      if (candle) {
        void this.handlers.onCandle(candle);
      }
      return;
    }

    if (stream.includes("@depth")) {
      const snapshot = parseDepth(this.streamSymbol.toUpperCase(), data);
      if (snapshot) {
        void this.handlers.onOrderBook(snapshot);
      }
    }
  }

  private scheduleReconnect(): void {
    if (this.stopped) {
      return;
    }

    this.reconnectAttempt += 1;
    const delay = Math.min(30_000, 1_000 * 2 ** Math.min(5, this.reconnectAttempt));
    this.reconnectTimer = setTimeout(() => this.connect(), delay);
  }
}

function parseTrade(data: Record<string, unknown>): PriceTick | undefined {
  const symbol = String(data.s ?? "");
  const price = Number(data.p);

  if (!symbol || !Number.isFinite(price)) {
    return undefined;
  }

  return {
    symbol,
    eventTime: Number(data.E ?? data.T ?? Date.now()),
    price,
    quantity: Number(data.q ?? 0),
  };
}

function parseKline(data: Record<string, unknown>): Candle | undefined {
  const kline = data.k as Record<string, unknown> | undefined;
  if (!kline) {
    return undefined;
  }

  const symbol = String(kline.s ?? data.s ?? "");
  const close = Number(kline.c);

  if (!symbol || !Number.isFinite(close)) {
    return undefined;
  }

  return {
    symbol,
    interval: String(kline.i ?? "1m"),
    openTime: Number(kline.t),
    closeTime: Number(kline.T),
    open: Number(kline.o),
    high: Number(kline.h),
    low: Number(kline.l),
    close,
    volume: Number(kline.v),
    closed: Boolean(kline.x),
  };
}

function parseDepth(
  symbol: string,
  data: Record<string, unknown>,
): OrderBookSnapshot | undefined {
  const rawBids = data.bids as [string, string][] | undefined;
  const rawAsks = data.asks as [string, string][] | undefined;

  if (!rawBids?.length || !rawAsks?.length) {
    return undefined;
  }

  return {
    symbol,
    eventTime: Number(data.E ?? Date.now()),
    bids: rawBids.map(([price, quantity]) => ({
      price: Number(price),
      quantity: Number(quantity),
    })),
    asks: rawAsks.map(([price, quantity]) => ({
      price: Number(price),
      quantity: Number(quantity),
    })),
  };
}
