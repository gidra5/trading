import WebSocket from "ws";
import type { Candle, OrderBookSnapshot, PriceTick } from "@trading/bot-algo";
import type { StreamVenue } from "./binance-markets.js";
import type { BinancePaperStreamEnvironment } from "./binance-paper.js";

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

export interface BinanceMarketStreamOptions {
  symbol: string;
  venue: StreamVenue;
  interval: string;
  environment?: BinancePaperStreamEnvironment;
  handlers: MarketStreamHandlers;
}

interface StreamSocketSpec {
  label: string;
  url: string;
}

export class BinanceMarketStream {
  private sockets = new Map<string, WebSocket>();
  private reconnectTimers = new Map<string, NodeJS.Timeout>();
  private reconnectAttempts = new Map<string, number>();
  private openSockets = new Set<string>();
  private stopped = true;
  private readonly symbol: string;
  private readonly streamSymbol: string;
  private readonly venue: StreamVenue;
  private readonly interval: string;
  private readonly environment: BinancePaperStreamEnvironment;
  private readonly handlers: MarketStreamHandlers;

  constructor(options: BinanceMarketStreamOptions) {
    this.symbol = options.symbol.toUpperCase();
    this.streamSymbol = options.symbol.toLowerCase();
    this.venue = options.venue;
    this.interval = options.interval;
    this.environment = options.environment ?? "live";
    this.handlers = options.handlers;
  }

  start(): void {
    if (!this.stopped) {
      return;
    }

    this.stopped = false;
    for (const spec of this.buildSocketSpecs()) {
      this.connect(spec);
    }
  }

  stop(): void {
    this.stopped = true;
    for (const timer of this.reconnectTimers.values()) {
      clearTimeout(timer);
    }

    this.reconnectTimers.clear();
    this.reconnectAttempts.clear();
    this.openSockets.clear();

    for (const socket of this.sockets.values()) {
      socket.close();
    }
    this.sockets.clear();
  }

  private connect(spec: StreamSocketSpec): void {
    const socket = new WebSocket(spec.url);
    this.sockets.set(spec.label, socket);

    socket.on("open", () => {
      this.openSockets.add(spec.label);
      this.reconnectAttempts.set(spec.label, 0);
      this.emitStatus(`Connected to Binance ${this.venue} ${this.environment} ${spec.label} stream`);
    });

    socket.on("message", (raw) => {
      this.handleMessage(raw.toString());
    });

    socket.on("close", () => {
      this.openSockets.delete(spec.label);
      this.emitStatus(`Binance ${this.venue} ${this.environment} ${spec.label} stream closed`);
      this.scheduleReconnect(spec);
    });

    socket.on("error", (error) => {
      this.openSockets.delete(spec.label);
      this.emitStatus(
        `Binance ${this.venue} ${this.environment} ${spec.label} stream error: ${error.message}`,
      );
    });
  }

  private handleMessage(raw: string): void {
    const payload = JSON.parse(raw) as { stream?: string; data?: unknown };
    const stream = (payload.stream ?? "").toLowerCase();
    const data = payload.data as Record<string, unknown> | undefined;

    if (!data) {
      return;
    }

    if (
      stream.includes("@trade") ||
      stream.includes("@aggtrade") ||
      stream.includes("@optiontrade")
    ) {
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
      const snapshot = parseDepth(this.symbol, data);
      if (snapshot) {
        void this.handlers.onOrderBook(snapshot);
      }
    }
  }

  private scheduleReconnect(spec: StreamSocketSpec): void {
    if (this.stopped) {
      return;
    }

    const reconnectAttempt = (this.reconnectAttempts.get(spec.label) ?? 0) + 1;
    this.reconnectAttempts.set(spec.label, reconnectAttempt);
    const delay = Math.min(30_000, 1_000 * 2 ** Math.min(5, reconnectAttempt));
    const timer = setTimeout(() => {
      this.reconnectTimers.delete(spec.label);
      this.connect(spec);
    }, delay);
    this.reconnectTimers.set(spec.label, timer);
  }

  private emitStatus(message: string): void {
    const expectedSockets = this.buildSocketSpecs().length;
    const reconnectAttempt = Math.max(0, ...this.reconnectAttempts.values());
    this.handlers.onStatus({
      connected: this.openSockets.size > 0 && this.openSockets.size === expectedSockets,
      message,
      lastEventAt: Date.now(),
      reconnectAttempt,
    });
  }

  private buildSocketSpecs(): StreamSocketSpec[] {
    const depthStream = `${this.streamSymbol}@depth10@500ms`;
    const spotDepthStream = `${this.streamSymbol}@depth10@1000ms`;
    const klineStream = `${this.streamSymbol}@kline_${this.interval}`;

    if (this.environment === "spot-testnet") {
      return [
        {
          label: "combined",
          url: combinedStreamUrl("wss://stream.testnet.binance.vision", [
            `${this.streamSymbol}@trade`,
            klineStream,
            spotDepthStream,
          ]),
        },
      ];
    }

    if (this.environment === "spot-demo") {
      return [
        {
          label: "combined",
          url: combinedStreamUrl("wss://demo-stream.binance.com:9443", [
            `${this.streamSymbol}@trade`,
            klineStream,
            spotDepthStream,
          ]),
        },
      ];
    }

    if (this.environment === "usdm-futures-testnet") {
      return [
        {
          label: "combined",
          url: combinedStreamUrl("wss://demo-fstream.binance.com", [
            `${this.streamSymbol}@aggTrade`,
            klineStream,
            depthStream,
          ]),
        },
      ];
    }

    if (this.environment === "coinm-futures-testnet") {
      return [
        {
          label: "combined",
          url: combinedStreamUrl("wss://demo-dstream.binance.com", [
            `${this.streamSymbol}@aggTrade`,
            klineStream,
            depthStream,
          ]),
        },
      ];
    }

    if (this.venue === "spot") {
      return [
        {
          label: "combined",
          url: combinedStreamUrl("wss://stream.binance.com:9443", [
            `${this.streamSymbol}@trade`,
            klineStream,
            spotDepthStream,
          ]),
        },
      ];
    }

    if (this.venue === "usdm-futures") {
      return [
        {
          label: "market",
          url: combinedStreamUrl("wss://fstream.binance.com/market", [
            `${this.streamSymbol}@aggTrade`,
            klineStream,
          ]),
        },
        {
          label: "public",
          url: combinedStreamUrl("wss://fstream.binance.com/public", [depthStream]),
        },
      ];
    }

    if (this.venue === "coinm-futures") {
      return [
        {
          label: "combined",
          url: combinedStreamUrl("wss://dstream.binance.com", [
            `${this.streamSymbol}@aggTrade`,
            klineStream,
            depthStream,
          ]),
        },
      ];
    }

    return [
      {
        label: "market",
        url: combinedStreamUrl("wss://fstream.binance.com/market", [klineStream]),
      },
      {
        label: "public",
        url: combinedStreamUrl("wss://fstream.binance.com/public", [
          `${this.streamSymbol}@optionTrade`,
          depthStream,
        ]),
      },
    ];
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
  const rawBids = (data.bids ?? data.b) as [string, string][] | undefined;
  const rawAsks = (data.asks ?? data.a) as [string, string][] | undefined;

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

function combinedStreamUrl(baseUrl: string, streams: string[]): string {
  return `${baseUrl}/stream?streams=${streams.join("/")}`;
}
