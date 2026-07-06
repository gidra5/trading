import WebSocket from "ws";
import type { BinanceMarketListing } from "./binance-markets.js";
import {
  BinancePaperTrading,
  type BinancePaperUserDataStreamStatus,
} from "./binance-paper.js";

const KEEPALIVE_INTERVAL_MS = 30 * 60 * 1000;
const EVENT_SYNC_DELAY_MS = 150;

export interface BinancePaperUserDataStreamHandlers {
  onStatus: (status: BinancePaperUserDataStreamStatus) => void;
  onUserData: (payload: unknown) => void | Promise<void>;
}

export interface BinancePaperUserDataStreamOptions {
  market: BinanceMarketListing;
  paperTrading: BinancePaperTrading;
  handlers: BinancePaperUserDataStreamHandlers;
}

export class BinancePaperUserDataStream {
  private socket?: WebSocket;
  private reconnectTimer?: NodeJS.Timeout;
  private keepAliveTimer?: NodeJS.Timeout;
  private eventTimer?: NodeJS.Timeout;
  private pendingPayloads: unknown[] = [];
  private reconnectAttempt = 0;
  private listenKey?: string;
  private stopped = true;

  constructor(private readonly options: BinancePaperUserDataStreamOptions) {}

  start(): void {
    if (!this.stopped) {
      return;
    }
    this.stopped = false;
    if (!this.options.paperTrading.canStreamUserData(this.options.market)) {
      this.emitStatus(false, "Binance user-data stream unavailable for this market");
      return;
    }
    void this.openAndConnect();
  }

  stop(): void {
    this.stopped = true;
    this.clearTimers();
    const listenKey = this.listenKey;
    this.listenKey = undefined;
    this.socket?.close();
    this.socket = undefined;
    if (listenKey) {
      void this.options.paperTrading
        .closeUserDataStream(this.options.market, listenKey)
        .catch(() => undefined);
    }
  }

  private async openAndConnect(): Promise<void> {
    if (this.stopped) {
      return;
    }

    try {
      const session = await this.options.paperTrading.openUserDataStream(this.options.market);
      if (this.stopped) {
        await this.options.paperTrading.closeUserDataStream(this.options.market, session.listenKey);
        return;
      }
      this.listenKey = session.listenKey;
      this.connect(session.url, session.mode);
      this.startKeepAlive();
    } catch (error) {
      this.emitStatus(
        false,
        `Binance user-data stream failed: ${errorMessage(error)}`,
      );
      this.scheduleReconnect();
    }
  }

  private connect(url: string, mode: string): void {
    this.socket?.close();
    const socket = new WebSocket(url);
    this.socket = socket;

    socket.on("open", () => {
      this.reconnectAttempt = 0;
      this.emitStatus(true, `Connected to Binance ${mode} user-data stream`);
    });

    socket.on("message", (raw) => {
      this.handleMessage(raw.toString());
    });

    socket.on("close", () => {
      if (this.stopped) {
        return;
      }
      this.emitStatus(false, `Binance ${mode} user-data stream closed`);
      this.scheduleReconnect();
    });

    socket.on("error", (error) => {
      this.emitStatus(false, `Binance ${mode} user-data stream error: ${error.message}`);
    });
  }

  private handleMessage(raw: string): void {
    let payload: unknown;
    try {
      const parsed = JSON.parse(raw) as { data?: unknown };
      payload = parsed.data ?? parsed;
    } catch {
      return;
    }

    if (isListenKeyExpired(payload)) {
      this.emitStatus(false, "Binance user-data listenKey expired");
      this.recreateListenKey();
      return;
    }

    this.pendingPayloads.push(payload);
    if (this.eventTimer) {
      return;
    }
    this.eventTimer = setTimeout(() => {
      this.eventTimer = undefined;
      void this.flushPayloads();
    }, EVENT_SYNC_DELAY_MS);
  }

  private async flushPayloads(): Promise<void> {
    const payloads = this.pendingPayloads;
    this.pendingPayloads = [];
    for (const payload of payloads) {
      if (this.stopped) {
        return;
      }
      await this.options.handlers.onUserData(payload);
    }
  }

  private startKeepAlive(): void {
    if (this.keepAliveTimer) {
      clearInterval(this.keepAliveTimer);
    }
    this.keepAliveTimer = setInterval(() => {
      void this.keepAlive();
    }, KEEPALIVE_INTERVAL_MS);
  }

  private async keepAlive(): Promise<void> {
    const listenKey = this.listenKey;
    if (!listenKey || this.stopped) {
      return;
    }
    try {
      await this.options.paperTrading.keepAliveUserDataStream(this.options.market, listenKey);
      this.emitStatus(true, "Binance user-data stream keepalive sent");
    } catch (error) {
      this.emitStatus(
        false,
        `Binance user-data keepalive failed: ${errorMessage(error)}`,
      );
      this.recreateListenKey();
    }
  }

  private recreateListenKey(): void {
    this.socket?.close();
    this.socket = undefined;
    this.listenKey = undefined;
    this.scheduleReconnect(0);
  }

  private scheduleReconnect(delayOverride?: number): void {
    if (this.stopped || this.reconnectTimer) {
      return;
    }
    this.reconnectAttempt += 1;
    const delay =
      delayOverride ??
      Math.min(30_000, 1_000 * 2 ** Math.min(5, this.reconnectAttempt));
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = undefined;
      void this.openAndConnect();
    }, delay);
  }

  private clearTimers(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = undefined;
    }
    if (this.keepAliveTimer) {
      clearInterval(this.keepAliveTimer);
      this.keepAliveTimer = undefined;
    }
    if (this.eventTimer) {
      clearTimeout(this.eventTimer);
      this.eventTimer = undefined;
    }
    this.pendingPayloads = [];
  }

  private emitStatus(connected: boolean, message: string): void {
    this.options.handlers.onStatus({
      connected,
      message,
      lastEventAt: Date.now(),
      reconnectAttempt: this.reconnectAttempt,
    });
  }
}

function isListenKeyExpired(payload: unknown): boolean {
  if (!payload || typeof payload !== "object") {
    return false;
  }
  return (payload as { e?: unknown }).e === "listenKeyExpired";
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
