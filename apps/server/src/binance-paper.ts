import { createHmac } from "node:crypto";
import type { TradingOrder } from "@trading/bot-algo";
import type { BinanceMarketListing } from "./binance-markets.js";

export type BinancePaperMode =
  | "auto"
  | "spot-testnet"
  | "spot-demo"
  | "usdm-futures-testnet"
  | "coinm-futures-testnet";

export interface BinancePaperConfig {
  enabled: boolean;
  mode: BinancePaperMode;
  apiKey?: string;
  apiSecret?: string;
  recvWindowMs: number;
  autoSubmit: boolean;
  baseUrlOverride?: string;
}

export interface BinancePaperBalance {
  asset: string;
  free: number;
  locked: number;
  walletBalance?: number;
  availableBalance?: number;
  unrealizedPnl?: number;
}

export interface BinancePaperPosition {
  symbol: string;
  positionSide?: string;
  positionAmt: number;
  entryPrice?: number;
  markPrice?: number;
  unrealizedPnl?: number;
  notional?: number;
  leverage?: number;
  marginType?: string;
  isolatedMargin?: number;
  updateTime?: number;
}

export interface BinancePaperOrder {
  symbol: string;
  orderId: string;
  clientOrderId: string;
  side: string;
  type: string;
  status: string;
  price: number;
  originalQuantity: number;
  executedQuantity: number;
  cumulativeQuoteQuantity?: number;
  avgPrice?: number;
  timeInForce?: string;
  reduceOnly?: boolean;
  positionSide?: string;
  createdAt?: number;
  updatedAt?: number;
}

export interface BinancePaperSnapshot {
  enabled: boolean;
  configured: boolean;
  compatible: boolean;
  mode: BinancePaperMode;
  resolvedMode?: Exclude<BinancePaperMode, "auto">;
  baseUrl?: string;
  streamEnvironment?: BinancePaperStreamEnvironment;
  autoSubmit: boolean;
  connected: boolean;
  lastSyncAt?: number;
  lastSubmitAt?: number;
  message: string;
  error?: string;
  balances: BinancePaperBalance[];
  positions: BinancePaperPosition[];
  openOrders: BinancePaperOrder[];
  lastOrder?: BinancePaperOrder;
}

export interface BinancePaperPlaceOrderInput {
  symbol?: string;
  side: "buy" | "sell";
  type: "limit" | "market";
  quantity: number;
  price?: number;
  timeInForce?: "GTC" | "IOC" | "FOK" | "GTX";
  reduceOnly?: boolean;
  positionSide?: "BOTH" | "LONG" | "SHORT";
  clientOrderId?: string;
}

export interface BinancePaperCancelOrderInput {
  symbol?: string;
  orderId?: string | number;
  clientOrderId?: string;
}

export type BinancePaperStreamEnvironment =
  | "live"
  | "spot-testnet"
  | "spot-demo"
  | "usdm-futures-testnet"
  | "coinm-futures-testnet";

interface ResolvedPaperEnvironment {
  mode: Exclude<BinancePaperMode, "auto">;
  product: "spot" | "usdm-futures" | "coinm-futures";
  baseUrl: string;
  restPrefix: "/api/v3" | "/fapi/v1" | "/dapi/v1";
  accountPath: string;
  balancePath?: string;
  streamEnvironment: BinancePaperStreamEnvironment;
}

interface BinanceApiErrorPayload {
  code?: unknown;
  msg?: unknown;
}

const EMPTY_SNAPSHOT: Pick<
  BinancePaperSnapshot,
  "balances" | "positions" | "openOrders" | "connected"
> = {
  connected: false,
  balances: [],
  positions: [],
  openOrders: [],
};

export class BinancePaperTrading {
  private snapshots = new Map<string, BinancePaperSnapshot>();
  private timeOffsets = new Map<string, number>();

  constructor(private readonly config: BinancePaperConfig) {}

  streamEnvironmentFor(market: BinanceMarketListing): BinancePaperStreamEnvironment {
    const environment = this.resolveEnvironment(market);
    if (!environment || !this.config.enabled) {
      return "live";
    }
    return environment.streamEnvironment;
  }

  snapshot(market: BinanceMarketListing): BinancePaperSnapshot {
    const environment = this.resolveEnvironment(market);
    const key = environment ? snapshotKey(environment, market.symbol) : market.id;
    const cached = this.snapshots.get(key);
    if (cached) {
      return cached;
    }

    if (!this.config.enabled) {
      return {
        ...EMPTY_SNAPSHOT,
        enabled: false,
        configured: false,
        compatible: true,
        mode: this.config.mode,
        autoSubmit: this.config.autoSubmit,
        message: "Binance paper trading disabled",
      };
    }

    if (!environment) {
      return {
        ...EMPTY_SNAPSHOT,
        enabled: true,
        configured: Boolean(this.config.apiKey && this.config.apiSecret),
        compatible: false,
        mode: this.config.mode,
        autoSubmit: this.config.autoSubmit,
        message: `Binance paper mode ${this.config.mode} is not compatible with ${market.venue}`,
      };
    }

    return {
      ...EMPTY_SNAPSHOT,
      enabled: true,
      configured: Boolean(this.config.apiKey && this.config.apiSecret),
      compatible: true,
      mode: this.config.mode,
      resolvedMode: environment.mode,
      baseUrl: environment.baseUrl,
      streamEnvironment: environment.streamEnvironment,
      autoSubmit: this.config.autoSubmit,
      message:
        this.config.apiKey && this.config.apiSecret
          ? "Binance paper trading ready; sync not run yet"
          : "Set BINANCE_PAPER_API_KEY and BINANCE_PAPER_API_SECRET to enable signed requests",
    };
  }

  async sync(market: BinanceMarketListing): Promise<BinancePaperSnapshot> {
    const environment = this.requireReadyEnvironment(market);
    const [balances, account, openOrders] = await Promise.all([
      this.fetchBalances(environment),
      this.fetchAccount(environment),
      this.fetchOpenOrders(environment, market.symbol),
    ]);
    const positions = extractPositions(environment, account, market.symbol);
    const snapshot: BinancePaperSnapshot = {
      ...this.snapshot(market),
      configured: true,
      compatible: true,
      connected: true,
      lastSyncAt: Date.now(),
      message: "Binance paper account synced",
      error: undefined,
      balances,
      positions,
      openOrders,
    };
    this.snapshots.set(snapshotKey(environment, market.symbol), snapshot);
    return snapshot;
  }

  async placeOrder(
    market: BinanceMarketListing,
    input: BinancePaperPlaceOrderInput,
  ): Promise<BinancePaperSnapshot> {
    const environment = this.requireReadyEnvironment(market);
    const payload = await this.signedRequest<Record<string, unknown>>(
      environment,
      "POST",
      `${orderRestPrefix(environment)}/order`,
      orderParams(environment, market.symbol, input),
    );
    const lastOrder = normalizeOrder(payload);
    const synced = await this.sync(market);
    const snapshot: BinancePaperSnapshot = {
      ...synced,
      lastSubmitAt: Date.now(),
      lastOrder,
      message: `${input.side.toUpperCase()} ${input.type.toUpperCase()} order submitted to ${environment.mode}`,
    };
    this.snapshots.set(snapshotKey(environment, market.symbol), snapshot);
    return snapshot;
  }

  async submitBotOrder(
    market: BinanceMarketListing,
    order: TradingOrder,
  ): Promise<BinancePaperSnapshot | undefined> {
    if (!this.config.autoSubmit || order.status !== "open") {
      return undefined;
    }
    const environment = this.resolveEnvironment(market);
    if (
      !environment ||
      (environment.product === "spot" && order.positionEffect === "open" && order.side === "sell")
    ) {
      return undefined;
    }

    return this.placeOrder(market, {
      symbol: market.symbol,
      side: order.side,
      type: order.type,
      quantity: order.quantity,
      price: order.type === "limit" ? order.price : undefined,
      timeInForce: "GTC",
      reduceOnly: environment.product !== "spot" && order.positionEffect === "close",
      clientOrderId: clientOrderIdForBotOrder(order.id),
    });
  }

  async cancelOrder(
    market: BinanceMarketListing,
    input: BinancePaperCancelOrderInput,
  ): Promise<BinancePaperSnapshot> {
    const environment = this.requireReadyEnvironment(market);
    const params: Record<string, string> = {
      symbol: (input.symbol ?? market.symbol).toUpperCase(),
    };
    if (input.orderId !== undefined) {
      params.orderId = String(input.orderId);
    }
    if (input.clientOrderId) {
      params.origClientOrderId = input.clientOrderId;
    }
    if (!params.orderId && !params.origClientOrderId) {
      throw new Error("Provide orderId or clientOrderId to cancel a Binance paper order.");
    }

    const payload = await this.signedRequest<Record<string, unknown>>(
      environment,
      "DELETE",
      `${orderRestPrefix(environment)}/order`,
      params,
    );
    const lastOrder = normalizeOrder(payload);
    const synced = await this.sync(market);
    const snapshot: BinancePaperSnapshot = {
      ...synced,
      lastSubmitAt: Date.now(),
      lastOrder,
      message: `Order ${lastOrder.orderId || input.orderId || input.clientOrderId} cancelled on ${environment.mode}`,
    };
    this.snapshots.set(snapshotKey(environment, market.symbol), snapshot);
    return snapshot;
  }

  async cancelAllOpenOrders(market: BinanceMarketListing): Promise<BinancePaperSnapshot> {
    const environment = this.requireReadyEnvironment(market);
    await this.signedRequest<unknown>(
      environment,
      "DELETE",
      cancelAllOpenOrdersPath(environment),
      { symbol: market.symbol },
    );
    const synced = await this.sync(market);
    const snapshot: BinancePaperSnapshot = {
      ...synced,
      lastSubmitAt: Date.now(),
      message: `All open ${market.symbol} orders cancelled on ${environment.mode}`,
    };
    this.snapshots.set(snapshotKey(environment, market.symbol), snapshot);
    return snapshot;
  }

  async changeLeverage(
    market: BinanceMarketListing,
    leverage: number,
  ): Promise<BinancePaperSnapshot> {
    const environment = this.requireReadyEnvironment(market);
    if (environment.product === "spot") {
      throw new Error("Leverage is only available for futures paper modes.");
    }
    const nextLeverage = Math.max(1, Math.min(125, Math.round(leverage)));
    await this.signedRequest<unknown>(environment, "POST", `${orderRestPrefix(environment)}/leverage`, {
      symbol: market.symbol,
      leverage: String(nextLeverage),
    });
    const synced = await this.sync(market);
    const snapshot: BinancePaperSnapshot = {
      ...synced,
      lastSubmitAt: Date.now(),
      message: `${market.symbol} leverage set to ${nextLeverage}x on ${environment.mode}`,
    };
    this.snapshots.set(snapshotKey(environment, market.symbol), snapshot);
    return snapshot;
  }

  private requireReadyEnvironment(market: BinanceMarketListing): ResolvedPaperEnvironment {
    if (!this.config.enabled) {
      throw new Error("Binance paper trading is disabled.");
    }
    if (!this.config.apiKey || !this.config.apiSecret) {
      throw new Error("BINANCE_PAPER_API_KEY and BINANCE_PAPER_API_SECRET are required.");
    }
    const environment = this.resolveEnvironment(market);
    if (!environment) {
      throw new Error(
        `Binance paper mode ${this.config.mode} is not compatible with ${market.venue}.`,
      );
    }
    return environment;
  }

  private resolveEnvironment(
    market: BinanceMarketListing,
  ): ResolvedPaperEnvironment | undefined {
    const mode = this.config.mode === "auto" ? defaultModeForVenue(market.venue) : this.config.mode;
    if (!mode) {
      return undefined;
    }
    if (!modeIsCompatibleWithVenue(mode, market.venue)) {
      return undefined;
    }
    return environmentForMode(mode, this.config.baseUrlOverride);
  }

  private async fetchBalances(
    environment: ResolvedPaperEnvironment,
  ): Promise<BinancePaperBalance[]> {
    if (environment.balancePath) {
      const payload = await this.signedRequest<unknown[]>(
        environment,
        "GET",
        environment.balancePath,
      );
      return payload.map(normalizeFuturesBalance).filter(isUsefulBalance);
    }

    const payload = await this.signedRequest<Record<string, unknown>>(
      environment,
      "GET",
      environment.accountPath,
      { omitZeroBalances: "true" },
    );
    const balances = Array.isArray(payload.balances) ? payload.balances : [];
    return balances.map(normalizeSpotBalance).filter(isUsefulBalance);
  }

  private async fetchAccount(
    environment: ResolvedPaperEnvironment,
  ): Promise<Record<string, unknown>> {
    if (environment.product === "spot") {
      return {};
    }
    return this.signedRequest<Record<string, unknown>>(environment, "GET", environment.accountPath);
  }

  private async fetchOpenOrders(
    environment: ResolvedPaperEnvironment,
    symbol: string,
  ): Promise<BinancePaperOrder[]> {
    const payload = await this.signedRequest<unknown[]>(
      environment,
      "GET",
      `${orderRestPrefix(environment)}/openOrders`,
      { symbol },
    );
    return payload.map(normalizeOrder);
  }

  private async signedRequest<T>(
    environment: ResolvedPaperEnvironment,
    method: "GET" | "POST" | "DELETE",
    path: string,
    params: Record<string, string | number | boolean | undefined> = {},
  ): Promise<T> {
    await this.syncTime(environment);
    const search = new URLSearchParams();
    for (const [key, value] of Object.entries(params)) {
      if (value !== undefined && value !== "") {
        search.set(key, String(value));
      }
    }
    search.set("recvWindow", String(this.config.recvWindowMs));
    search.set("timestamp", String(Date.now() + (this.timeOffsets.get(environment.mode) ?? 0)));
    const signature = createHmac("sha256", this.config.apiSecret ?? "")
      .update(search.toString())
      .digest("hex");
    search.set("signature", signature);

    return requestJson<T>(new URL(`${path}?${search.toString()}`, environment.baseUrl), {
      method,
      headers: {
        "X-MBX-APIKEY": this.config.apiKey ?? "",
      },
    });
  }

  private async syncTime(environment: ResolvedPaperEnvironment): Promise<void> {
    if (this.timeOffsets.has(environment.mode)) {
      return;
    }
    const timePath =
      environment.product === "spot"
        ? "/api/v3/time"
        : environment.product === "usdm-futures"
          ? "/fapi/v1/time"
          : "/dapi/v1/time";
    const payload = await requestJson<{ serverTime?: unknown }>(
      new URL(timePath, environment.baseUrl),
      { method: "GET" },
    );
    const serverTime = Number(payload.serverTime);
    this.timeOffsets.set(
      environment.mode,
      Number.isFinite(serverTime) ? serverTime - Date.now() : 0,
    );
  }
}

function defaultModeForVenue(
  venue: BinanceMarketListing["venue"],
): Exclude<BinancePaperMode, "auto"> | undefined {
  if (venue === "spot") {
    return "spot-testnet";
  }
  if (venue === "usdm-futures") {
    return "usdm-futures-testnet";
  }
  if (venue === "coinm-futures") {
    return "coinm-futures-testnet";
  }
  return undefined;
}

function modeIsCompatibleWithVenue(
  mode: Exclude<BinancePaperMode, "auto">,
  venue: BinanceMarketListing["venue"],
): boolean {
  if (mode === "spot-testnet" || mode === "spot-demo") {
    return venue === "spot";
  }
  if (mode === "usdm-futures-testnet") {
    return venue === "usdm-futures";
  }
  if (mode === "coinm-futures-testnet") {
    return venue === "coinm-futures";
  }
  return false;
}

function environmentForMode(
  mode: Exclude<BinancePaperMode, "auto">,
  baseUrlOverride: string | undefined,
): ResolvedPaperEnvironment {
  if (mode === "spot-testnet") {
    return {
      mode,
      product: "spot",
      baseUrl: baseUrlOverride ?? "https://testnet.binance.vision",
      restPrefix: "/api/v3",
      accountPath: "/api/v3/account",
      streamEnvironment: "spot-testnet",
    };
  }
  if (mode === "spot-demo") {
    return {
      mode,
      product: "spot",
      baseUrl: baseUrlOverride ?? "https://demo-api.binance.com",
      restPrefix: "/api/v3",
      accountPath: "/api/v3/account",
      streamEnvironment: "spot-demo",
    };
  }
  if (mode === "usdm-futures-testnet") {
    return {
      mode,
      product: "usdm-futures",
      baseUrl: baseUrlOverride ?? "https://demo-fapi.binance.com",
      restPrefix: "/fapi/v1",
      accountPath: "/fapi/v3/account",
      balancePath: "/fapi/v3/balance",
      streamEnvironment: "usdm-futures-testnet",
    };
  }
  return {
    mode,
    product: "coinm-futures",
    baseUrl: baseUrlOverride ?? "https://demo-dapi.binance.com",
    restPrefix: "/dapi/v1",
    accountPath: "/dapi/v1/account",
    balancePath: "/dapi/v1/balance",
    streamEnvironment: "coinm-futures-testnet",
  };
}

function orderRestPrefix(
  environment: ResolvedPaperEnvironment,
): "/api/v3" | "/fapi/v1" | "/dapi/v1" {
  if (environment.product === "spot") {
    return "/api/v3";
  }
  return environment.restPrefix === "/dapi/v1" ? "/dapi/v1" : "/fapi/v1";
}

function cancelAllOpenOrdersPath(environment: ResolvedPaperEnvironment): string {
  return environment.product === "spot"
    ? "/api/v3/openOrders"
    : `${orderRestPrefix(environment)}/allOpenOrders`;
}

function orderParams(
  environment: ResolvedPaperEnvironment,
  symbol: string,
  input: BinancePaperPlaceOrderInput,
): Record<string, string | boolean | undefined> {
  const type = input.type.toUpperCase();
  const params: Record<string, string | boolean | undefined> = {
    symbol: (input.symbol ?? symbol).toUpperCase(),
    side: input.side.toUpperCase(),
    type,
    quantity: decimalParam(input.quantity),
    newClientOrderId: input.clientOrderId ?? createClientOrderId(),
  };

  if (type === "LIMIT") {
    params.timeInForce = input.timeInForce ?? "GTC";
    params.price = decimalParam(input.price);
    if (!params.price) {
      throw new Error("Limit paper orders require price.");
    }
  }

  if (environment.product !== "spot") {
    if (input.reduceOnly !== undefined) {
      params.reduceOnly = input.reduceOnly ? "true" : "false";
    }
    if (input.positionSide) {
      params.positionSide = input.positionSide;
    }
  }

  return params;
}

function decimalParam(value: number | undefined): string | undefined {
  if (value === undefined) {
    return undefined;
  }
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error("Binance order quantity and price must be positive numbers.");
  }
  return value.toLocaleString("en-US", {
    useGrouping: false,
    maximumFractionDigits: 12,
  });
}

function createClientOrderId(): string {
  return `trd_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`.slice(
    0,
    36,
  );
}

function clientOrderIdForBotOrder(orderId: string): string {
  return `bot_${orderId.replace(/[^.A-Z:/a-z0-9_-]/g, "_")}`.slice(0, 36);
}

function normalizeSpotBalance(raw: unknown): BinancePaperBalance {
  const item = asRecord(raw);
  return {
    asset: stringValue(item.asset),
    free: numberValue(item.free),
    locked: numberValue(item.locked),
  };
}

function normalizeFuturesBalance(raw: unknown): BinancePaperBalance {
  const item = asRecord(raw);
  return {
    asset: stringValue(item.asset),
    free: numberValue(item.availableBalance),
    locked: Math.max(0, numberValue(item.balance) - numberValue(item.availableBalance)),
    walletBalance: numberValue(item.balance),
    availableBalance: numberValue(item.availableBalance),
    unrealizedPnl: numberValue(item.crossUnPnl),
  };
}

function isUsefulBalance(balance: BinancePaperBalance): boolean {
  return (
    Boolean(balance.asset) &&
    (balance.free !== 0 || balance.locked !== 0 || balance.walletBalance !== 0)
  );
}

function extractPositions(
  environment: ResolvedPaperEnvironment,
  account: Record<string, unknown>,
  symbol: string,
): BinancePaperPosition[] {
  if (environment.product === "spot") {
    return [];
  }
  const positions = Array.isArray(account.positions) ? account.positions : [];
  return positions
    .map((raw) => {
      const item = asRecord(raw);
      return {
        symbol: stringValue(item.symbol),
        positionSide: stringValue(item.positionSide),
        positionAmt: numberValue(item.positionAmt),
        entryPrice: numberValue(item.entryPrice),
        markPrice: numberValue(item.markPrice),
        unrealizedPnl: numberValue(item.unrealizedProfit),
        notional: numberValue(item.notional),
        leverage: numberValue(item.leverage),
        marginType: stringValue(item.marginType),
        isolatedMargin: numberValue(item.isolatedMargin),
        updateTime: numberValue(item.updateTime),
      };
    })
    .filter(
      (position) =>
        position.symbol === symbol &&
        (position.positionAmt !== 0 ||
          position.notional !== 0 ||
          position.unrealizedPnl !== 0),
    );
}

function normalizeOrder(raw: unknown): BinancePaperOrder {
  const item = asRecord(raw);
  return {
    symbol: stringValue(item.symbol),
    orderId: stringValue(item.orderId),
    clientOrderId: stringValue(item.clientOrderId),
    side: stringValue(item.side),
    type: stringValue(item.type),
    status: stringValue(item.status),
    price: numberValue(item.price),
    originalQuantity: numberValue(item.origQty),
    executedQuantity: numberValue(item.executedQty),
    cumulativeQuoteQuantity: numberValue(item.cummulativeQuoteQty ?? item.cumQuote),
    avgPrice: numberValue(item.avgPrice),
    timeInForce: stringValue(item.timeInForce),
    reduceOnly: booleanValue(item.reduceOnly),
    positionSide: stringValue(item.positionSide),
    createdAt: numberValue(item.time),
    updatedAt: numberValue(item.updateTime ?? item.transactTime),
  };
}

async function requestJson<T>(url: URL, init: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  const text = await response.text();
  const payload = text ? JSON.parse(text) as unknown : {};
  if (!response.ok) {
    const errorPayload = asRecord(payload) as BinanceApiErrorPayload;
    const code = errorPayload.code === undefined ? "" : ` ${String(errorPayload.code)}`;
    const message =
      typeof errorPayload.msg === "string"
        ? errorPayload.msg
        : `HTTP ${response.status} ${text.slice(0, 240)}`;
    throw new Error(`Binance paper request failed:${code} ${message}`.trim());
  }
  return payload as T;
}

function snapshotKey(environment: ResolvedPaperEnvironment, symbol: string): string {
  return `${environment.mode}:${symbol.toUpperCase()}`;
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" ? value as Record<string, unknown> : {};
}

function stringValue(value: unknown): string {
  if (value === undefined || value === null) {
    return "";
  }
  return String(value);
}

function numberValue(value: unknown): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function booleanValue(value: unknown): boolean | undefined {
  if (value === true || value === false) {
    return value;
  }
  if (value === "true") {
    return true;
  }
  if (value === "false") {
    return false;
  }
  return undefined;
}
