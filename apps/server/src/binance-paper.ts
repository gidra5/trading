import { createHmac } from "node:crypto";
import type {
  ExchangeReconciliationInput,
  ExchangeTradeFill,
  TradingOrder,
} from "@trading/bot-algo";
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
  localOrderId?: string;
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

export interface BinancePaperTrade {
  id: string;
  symbol: string;
  orderId: string;
  clientOrderId?: string;
  localOrderId?: string;
  side: "buy" | "sell";
  price: number;
  quantity: number;
  quoteQuantity: number;
  commission: number;
  commissionAsset: string;
  feeQuote: number;
  realizedPnl?: number;
  positionSide?: string;
  time: number;
  maker?: boolean;
}

export interface BinancePaperSymbolFilters {
  symbol: string;
  pricePrecision?: number;
  quantityPrecision?: number;
  tickSize?: number;
  minPrice?: number;
  maxPrice?: number;
  stepSize?: number;
  marketStepSize?: number;
  minQuantity?: number;
  maxQuantity?: number;
  minMarketQuantity?: number;
  maxMarketQuantity?: number;
  minNotional?: number;
  maxNotional?: number;
}

export interface BinancePaperCommission {
  makerFeeBps?: number;
  takerFeeBps?: number;
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
  userDataStreamConnected?: boolean;
  lastUserDataStreamAt?: number;
  userDataStreamMessage?: string;
  lastSyncAt?: number;
  lastSubmitAt?: number;
  message: string;
  error?: string;
  maxLeverage?: number;
  symbolFilters?: BinancePaperSymbolFilters;
  commission?: BinancePaperCommission;
  feeBps?: number;
  estimatedSlippageBps?: number;
  balances: BinancePaperBalance[];
  positions: BinancePaperPosition[];
  openOrders: BinancePaperOrder[];
  recentOrders: BinancePaperOrder[];
  recentTrades: BinancePaperTrade[];
  lastOrder?: BinancePaperOrder;
}

export interface BinancePaperUserDataStreamSession {
  listenKey: string;
  url: string;
  mode: Exclude<BinancePaperMode, "auto">;
}

export interface BinancePaperUserDataStreamStatus {
  connected: boolean;
  message: string;
  lastEventAt: number;
  reconnectAttempt: number;
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
  "balances" | "positions" | "openOrders" | "recentOrders" | "recentTrades" | "connected"
> = {
  connected: false,
  balances: [],
  positions: [],
  openOrders: [],
  recentOrders: [],
  recentTrades: [],
};

export class BinancePaperTrading {
  private snapshots = new Map<string, BinancePaperSnapshot>();
  private timeOffsets = new Map<string, number>();
  private symbolFilters = new Map<string, BinancePaperSymbolFilters>();
  private maxLeverageBySymbol = new Map<string, number>();

  constructor(private readonly config: BinancePaperConfig) {}

  drivesOrderExecution(market: BinanceMarketListing): boolean {
    return Boolean(
      this.config.enabled &&
        this.config.autoSubmit &&
        this.config.apiKey &&
        this.config.apiSecret &&
        this.resolveEnvironment(market),
    );
  }

  canSubmitOrders(market: BinanceMarketListing): boolean {
    return Boolean(
      this.config.enabled &&
        this.config.apiKey &&
        this.config.apiSecret &&
        this.resolveEnvironment(market),
    );
  }

  canStreamUserData(market: BinanceMarketListing): boolean {
    const environment = this.resolveEnvironment(market);
    return Boolean(
      this.config.enabled &&
        this.config.apiKey &&
        this.config.apiSecret &&
        environment &&
        environment.product !== "spot",
    );
  }

  klineEndpointFor(market: BinanceMarketListing): string | undefined {
    const environment = this.resolveEnvironment(market);
    if (!this.config.enabled || !environment) {
      return undefined;
    }
    return new URL(exchangeKlinePath(environment), environment.baseUrl).toString();
  }

  async openUserDataStream(
    market: BinanceMarketListing,
  ): Promise<BinancePaperUserDataStreamSession> {
    const environment = this.requireReadyEnvironment(market);
    if (environment.product === "spot") {
      throw new Error("Spot paper user-data streams require WebSocket API auth and are not enabled yet.");
    }
    const payload = await this.apiKeyRequest<Record<string, unknown>>(
      environment,
      "POST",
      userDataStreamPath(environment),
    );
    const listenKey = stringValue(payload.listenKey);
    if (!listenKey) {
      throw new Error("Binance paper user-data stream did not return a listenKey.");
    }
    return {
      listenKey,
      url: userDataStreamWebSocketUrl(environment, listenKey),
      mode: environment.mode,
    };
  }

  async keepAliveUserDataStream(
    market: BinanceMarketListing,
    _listenKey: string,
  ): Promise<void> {
    const environment = this.requireReadyEnvironment(market);
    if (environment.product === "spot") {
      return;
    }
    await this.apiKeyRequest<Record<string, unknown>>(
      environment,
      "PUT",
      userDataStreamPath(environment),
    );
  }

  async closeUserDataStream(
    market: BinanceMarketListing,
    _listenKey: string,
  ): Promise<void> {
    const environment = this.requireReadyEnvironment(market);
    if (environment.product === "spot") {
      return;
    }
    await this.apiKeyRequest<Record<string, unknown>>(
      environment,
      "DELETE",
      userDataStreamPath(environment),
    );
  }

  updateUserDataStreamStatus(
    market: BinanceMarketListing,
    status: BinancePaperUserDataStreamStatus,
  ): BinancePaperSnapshot {
    const environment = this.resolveEnvironment(market);
    const snapshot = {
      ...this.snapshot(market),
      userDataStreamConnected: status.connected,
      lastUserDataStreamAt: status.lastEventAt,
      userDataStreamMessage: status.message,
    };
    this.snapshots.set(environment ? snapshotKey(environment, market.symbol) : market.id, snapshot);
    return snapshot;
  }

  reconciliationFromUserDataEvent(
    market: BinanceMarketListing,
    payload: unknown,
  ): ExchangeReconciliationInput | undefined {
    const environment = this.resolveEnvironment(market);
    if (!environment || environment.product === "spot") {
      return undefined;
    }
    return futuresReconciliationFromUserDataEvent(market, payload);
  }

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
    const [
      balances,
      account,
      openOrders,
      recentOrders,
      recentTrades,
      symbolFilters,
      maxLeverage,
      commission,
      estimatedSlippageBps,
    ] = await Promise.all([
      this.fetchBalances(environment),
      this.fetchAccount(environment),
      this.fetchOpenOrders(environment, market.symbol),
      this.fetchRecentOrders(environment, market.symbol),
      this.fetchRecentTrades(environment, market),
      this.fetchSymbolFilters(environment, market.symbol),
      this.fetchMaxLeverage(environment, market.symbol),
      this.fetchCommission(environment, market.symbol),
      this.fetchEstimatedSlippageBps(environment, market.symbol),
    ]);
    const ordersById = new Map(recentOrders.map((order) => [order.orderId, order]));
    const trades = recentTrades.map((trade) => ({
      ...trade,
      clientOrderId: trade.clientOrderId || ordersById.get(trade.orderId)?.clientOrderId,
      localOrderId: trade.localOrderId || ordersById.get(trade.orderId)?.localOrderId,
    }));
    const positions = extractPositions(environment, account, market.symbol);
    const snapshot: BinancePaperSnapshot = {
      ...this.snapshot(market),
      configured: true,
      compatible: true,
      connected: true,
      lastSyncAt: Date.now(),
      message: "Binance paper account synced",
      error: undefined,
      maxLeverage,
      symbolFilters,
      commission,
      feeBps: commission?.takerFeeBps,
      estimatedSlippageBps,
      balances,
      positions,
      openOrders,
      recentOrders,
      recentTrades: trades,
    };
    this.snapshots.set(snapshotKey(environment, market.symbol), snapshot);
    return snapshot;
  }

  async placeOrder(
    market: BinanceMarketListing,
    input: BinancePaperPlaceOrderInput,
  ): Promise<BinancePaperSnapshot> {
    const environment = this.requireReadyEnvironment(market);
    const normalizedInput = await this.normalizeOrderInput(environment, market, input);
    const payload = await this.signedRequest<Record<string, unknown>>(
      environment,
      "POST",
      `${orderRestPrefix(environment)}/order`,
      orderParams(environment, market.symbol, normalizedInput),
    );
    const lastOrder = normalizeOrder(payload);
    const synced = await this.sync(market);
    const snapshot: BinancePaperSnapshot = {
      ...synced,
      lastSubmitAt: Date.now(),
      lastOrder,
      message: `${normalizedInput.side.toUpperCase()} ${normalizedInput.type.toUpperCase()} order submitted to ${environment.mode}`,
    };
    this.snapshots.set(snapshotKey(environment, market.symbol), snapshot);
    return snapshot;
  }

  async submitBotOrder(
    market: BinanceMarketListing,
    order: TradingOrder,
    options: { force?: boolean } = {},
  ): Promise<BinancePaperSnapshot | undefined> {
    if ((!options.force && !this.config.autoSubmit) || order.status !== "open") {
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
      price: order.price,
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

  async closeOpenPositions(
    market: BinanceMarketListing,
    options: { includeUnprofitable?: boolean } = {},
  ): Promise<BinancePaperSnapshot> {
    const environment = this.requireReadyEnvironment(market);
    if (environment.product === "spot") {
      throw new Error("Exchange position close is only available for futures paper modes.");
    }

    const initial = await this.sync(market);
    const positions = initial.positions.filter((position) =>
      shouldClosePosition(position, options.includeUnprofitable === true),
    );
    if (positions.length === 0) {
      return {
        ...initial,
        message: "No open Binance paper positions to close",
      };
    }

    let closedCount = 0;
    for (const position of positions) {
      const quantity = Math.abs(position.positionAmt);
      if (!Number.isFinite(quantity) || quantity <= 0) {
        continue;
      }
      const positionSide = normalizePositionSideForOrder(position.positionSide);
      const isHedgePosition = positionSide === "LONG" || positionSide === "SHORT";
      await this.placeOrder(market, {
        symbol: market.symbol,
        side: position.positionAmt > 0 ? "sell" : "buy",
        type: "market",
        quantity,
        reduceOnly: isHedgePosition ? undefined : true,
        positionSide,
        clientOrderId: createClientOrderId("close"),
      });
      closedCount += 1;
    }

    const synced = await this.sync(market);
    const snapshot: BinancePaperSnapshot = {
      ...synced,
      lastSubmitAt: Date.now(),
      message:
        closedCount === 1
          ? `Closed 1 open ${market.symbol} Binance paper position`
          : `Closed ${closedCount} open ${market.symbol} Binance paper positions`,
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
    const maxLeverage = await this.fetchMaxLeverage(environment, market.symbol);
    const leverageCap = maxLeverage && maxLeverage > 0 ? maxLeverage : 125;
    const nextLeverage = Math.max(1, Math.min(leverageCap, Math.round(leverage)));
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

  private async fetchRecentOrders(
    environment: ResolvedPaperEnvironment,
    symbol: string,
  ): Promise<BinancePaperOrder[]> {
    try {
      const payload = await this.signedRequest<unknown[]>(
        environment,
        "GET",
        `${orderRestPrefix(environment)}/allOrders`,
        { symbol, limit: 100 },
      );
      return payload.map(normalizeOrder);
    } catch {
      return [];
    }
  }

  private async fetchRecentTrades(
    environment: ResolvedPaperEnvironment,
    market: BinanceMarketListing,
  ): Promise<BinancePaperTrade[]> {
    try {
      const payload = await this.signedRequest<unknown[]>(
        environment,
        "GET",
        `${orderRestPrefix(environment)}/userTrades`,
        { symbol: market.symbol, limit: 100 },
      );
      return payload.map((trade) => normalizeTrade(trade, market));
    } catch {
      return [];
    }
  }

  private async fetchSymbolFilters(
    environment: ResolvedPaperEnvironment,
    symbol: string,
  ): Promise<BinancePaperSymbolFilters | undefined> {
    const key = `${environment.mode}:${symbol}`;
    const cached = this.symbolFilters.get(key);
    if (cached) {
      return cached;
    }

    const payload = await requestJson<Record<string, unknown>>(
      new URL(`${exchangeInfoPath(environment)}?symbol=${encodeURIComponent(symbol)}`, environment.baseUrl),
      { method: "GET" },
    );
    const symbols = Array.isArray(payload.symbols) ? payload.symbols : [payload.symbols ?? payload];
    const raw = symbols.map(asRecord).find((item) => stringValue(item.symbol) === symbol);
    if (!raw) {
      return undefined;
    }

    const filters = normalizeSymbolFilters(raw);
    this.symbolFilters.set(key, filters);
    return filters;
  }

  private async fetchMaxLeverage(
    environment: ResolvedPaperEnvironment,
    symbol: string,
  ): Promise<number | undefined> {
    if (environment.product === "spot") {
      return undefined;
    }
    const key = `${environment.mode}:${symbol}`;
    if (this.maxLeverageBySymbol.has(key)) {
      return this.maxLeverageBySymbol.get(key);
    }

    try {
      const payload = await this.signedRequest<unknown>(
        environment,
        "GET",
        leverageBracketPath(environment),
        { symbol },
      );
      const maxLeverage = parseMaxLeverage(payload, symbol);
      if (maxLeverage) {
        this.maxLeverageBySymbol.set(key, maxLeverage);
      }
      return maxLeverage;
    } catch {
      return undefined;
    }
  }

  private async fetchCommission(
    environment: ResolvedPaperEnvironment,
    symbol: string,
  ): Promise<BinancePaperCommission | undefined> {
    try {
      const payload = await this.signedRequest<Record<string, unknown>>(
        environment,
        "GET",
        commissionPath(environment),
        { symbol },
      );
      return normalizeCommission(payload, environment.product);
    } catch {
      return undefined;
    }
  }

  private async fetchEstimatedSlippageBps(
    environment: ResolvedPaperEnvironment,
    symbol: string,
  ): Promise<number | undefined> {
    try {
      const payload = await requestJson<Record<string, unknown>>(
        new URL(`${bookTickerPath(environment)}?symbol=${encodeURIComponent(symbol)}`, environment.baseUrl),
        { method: "GET" },
      );
      const bid = numberValue(payload.bidPrice);
      const ask = numberValue(payload.askPrice);
      const mid = (bid + ask) / 2;
      if (bid <= 0 || ask <= 0 || mid <= 0 || ask < bid) {
        return undefined;
      }
      return ((ask - bid) / mid) * 5_000;
    } catch {
      return undefined;
    }
  }

  private async normalizeOrderInput(
    environment: ResolvedPaperEnvironment,
    market: BinanceMarketListing,
    input: BinancePaperPlaceOrderInput,
  ): Promise<BinancePaperPlaceOrderInput> {
    const filters = await this.fetchSymbolFilters(environment, market.symbol);
    if (!filters) {
      return input;
    }

    const price =
      input.type === "limit"
        ? normalizePrice(input.price, input.side, filters)
        : input.price;
    const quantity = normalizeQuantity(input.quantity, input.type, filters);
    const effectivePrice =
      input.type === "limit"
        ? price
        : Number.isFinite(Number(input.price)) && Number(input.price) > 0
          ? Number(input.price)
          : undefined;
    const minNotional = filters.minNotional ?? 0;
    let adjustedQuantity = quantity;

    if (minNotional > 0 && effectivePrice && effectivePrice * adjustedQuantity < minNotional) {
      adjustedQuantity = normalizeQuantity(
        minNotional / effectivePrice,
        input.type,
        filters,
        "ceil",
      );
    }

    validateNormalizedOrder(input, adjustedQuantity, effectivePrice, filters);
    return {
      ...input,
      price: input.type === "limit" ? effectivePrice : input.price,
      quantity: adjustedQuantity,
    };
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

  private async apiKeyRequest<T>(
    environment: ResolvedPaperEnvironment,
    method: "POST" | "PUT" | "DELETE",
    path: string,
    params: Record<string, string | number | boolean | undefined> = {},
  ): Promise<T> {
    const search = new URLSearchParams();
    for (const [key, value] of Object.entries(params)) {
      if (value !== undefined && value !== "") {
        search.set(key, String(value));
      }
    }
    const query = search.toString();
    const pathWithQuery = query ? `${path}?${query}` : path;
    return requestJson<T>(new URL(pathWithQuery, environment.baseUrl), {
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

function exchangeInfoPath(environment: ResolvedPaperEnvironment): string {
  if (environment.product === "spot") {
    return "/api/v3/exchangeInfo";
  }
  return `${orderRestPrefix(environment)}/exchangeInfo`;
}

function exchangeKlinePath(environment: ResolvedPaperEnvironment): string {
  return `${orderRestPrefix(environment)}/klines`;
}

function bookTickerPath(environment: ResolvedPaperEnvironment): string {
  return `${orderRestPrefix(environment)}/ticker/bookTicker`;
}

function commissionPath(environment: ResolvedPaperEnvironment): string {
  if (environment.product === "spot") {
    return "/api/v3/account/commission";
  }
  return environment.product === "coinm-futures"
    ? "/dapi/v1/commissionRate"
    : "/fapi/v1/commissionRate";
}

function leverageBracketPath(environment: ResolvedPaperEnvironment): string {
  return environment.product === "coinm-futures"
    ? "/dapi/v2/leverageBracket"
    : "/fapi/v1/leverageBracket";
}

function userDataStreamPath(environment: ResolvedPaperEnvironment): string {
  return `${orderRestPrefix(environment)}/listenKey`;
}

function userDataStreamWebSocketUrl(
  environment: ResolvedPaperEnvironment,
  listenKey: string,
): string {
  if (environment.product === "usdm-futures") {
    const base =
      environment.mode === "usdm-futures-testnet"
        ? "wss://demo-fstream.binance.com/private"
        : "wss://fstream.binance.com/private";
    return `${base}/ws/${encodeURIComponent(listenKey)}`;
  }

  const base =
    environment.mode === "coinm-futures-testnet"
      ? "wss://demo-dstream.binance.com"
      : "wss://dstream.binance.com";
  return `${base}/ws/${encodeURIComponent(listenKey)}`;
}

function normalizePrice(
  value: number | undefined,
  side: "buy" | "sell",
  filters: BinancePaperSymbolFilters,
): number | undefined {
  if (value === undefined) {
    return undefined;
  }
  const price = Number(value);
  if (!Number.isFinite(price) || price <= 0) {
    throw new Error("Limit paper orders require a positive price.");
  }
  const tickSize = filters.tickSize ?? 0;
  if (tickSize <= 0) {
    return price;
  }
  return side === "buy"
    ? roundToStep(price, tickSize, "floor")
    : roundToStep(price, tickSize, "ceil");
}

function normalizeQuantity(
  value: number,
  type: "limit" | "market",
  filters: BinancePaperSymbolFilters,
  mode: "floor" | "ceil" = "floor",
): number {
  const quantity = Number(value);
  if (!Number.isFinite(quantity) || quantity <= 0) {
    throw new Error("Binance order quantity must be positive.");
  }
  const stepSize =
    type === "market" && filters.marketStepSize
      ? filters.marketStepSize
      : filters.stepSize;
  if (!stepSize || stepSize <= 0) {
    return quantity;
  }
  return roundToStep(quantity, stepSize, mode);
}

function validateNormalizedOrder(
  input: BinancePaperPlaceOrderInput,
  quantity: number,
  price: number | undefined,
  filters: BinancePaperSymbolFilters,
): void {
  const minQuantity =
    input.type === "market" && filters.minMarketQuantity
      ? filters.minMarketQuantity
      : filters.minQuantity;
  const maxQuantity =
    input.type === "market" && filters.maxMarketQuantity
      ? filters.maxMarketQuantity
      : filters.maxQuantity;
  if (minQuantity && quantity < minQuantity) {
    throw new Error(`Normalized Binance order quantity ${quantity} is below minQty ${minQuantity}.`);
  }
  if (maxQuantity && maxQuantity > 0 && quantity > maxQuantity) {
    throw new Error(`Normalized Binance order quantity ${quantity} is above maxQty ${maxQuantity}.`);
  }
  if (price !== undefined) {
    if (filters.minPrice && price < filters.minPrice) {
      throw new Error(`Normalized Binance order price ${price} is below minPrice ${filters.minPrice}.`);
    }
    if (filters.maxPrice && filters.maxPrice > 0 && price > filters.maxPrice) {
      throw new Error(`Normalized Binance order price ${price} is above maxPrice ${filters.maxPrice}.`);
    }
    const notional = price * quantity;
    if (filters.minNotional && notional < filters.minNotional) {
      throw new Error(`Normalized Binance order notional ${notional} is below minNotional ${filters.minNotional}.`);
    }
    if (filters.maxNotional && filters.maxNotional > 0 && notional > filters.maxNotional) {
      throw new Error(`Normalized Binance order notional ${notional} is above maxNotional ${filters.maxNotional}.`);
    }
  }
}

function roundToStep(value: number, step: number, mode: "floor" | "ceil"): number {
  const precision = decimalPrecision(step);
  const scaled = value / step;
  const rounded = mode === "ceil"
    ? Math.ceil(scaled - 1e-12)
    : Math.floor(scaled + 1e-12);
  return Number((rounded * step).toFixed(precision));
}

function decimalPrecision(step: number): number {
  const text = step.toLocaleString("en-US", {
    useGrouping: false,
    maximumFractionDigits: 18,
  });
  const dotIndex = text.indexOf(".");
  if (dotIndex < 0) {
    return 0;
  }
  return text.length - dotIndex - 1;
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

function createClientOrderId(prefix = "trd"): string {
  return `${prefix}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`.slice(
    0,
    36,
  );
}

function clientOrderIdForBotOrder(orderId: string): string {
  return `bot_${orderId.replace(/[^.A-Z:/a-z0-9_-]/g, "_")}`.slice(0, 36);
}

function localOrderIdFromClientOrderId(clientOrderId: string): string | undefined {
  if (!clientOrderId.startsWith("bot_")) {
    return undefined;
  }
  const orderId = clientOrderId.slice(4);
  return orderId || undefined;
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

function shouldClosePosition(
  position: BinancePaperPosition,
  includeUnprofitable: boolean,
): boolean {
  if (position.positionAmt === 0) {
    return false;
  }
  if (includeUnprofitable) {
    return true;
  }
  if (Number.isFinite(position.unrealizedPnl) && (position.unrealizedPnl as number) > 0) {
    return true;
  }

  const entryPrice = position.entryPrice ?? 0;
  const markPrice = position.markPrice ?? 0;
  if (entryPrice <= 0 || markPrice <= 0) {
    return false;
  }
  return position.positionAmt > 0 ? markPrice > entryPrice : markPrice < entryPrice;
}

function normalizePositionSideForOrder(
  positionSide: string | undefined,
): "BOTH" | "LONG" | "SHORT" | undefined {
  const normalized = stringValue(positionSide).toUpperCase();
  if (normalized === "LONG" || normalized === "SHORT") {
    return normalized;
  }
  return undefined;
}

function normalizeOrder(raw: unknown): BinancePaperOrder {
  const item = asRecord(raw);
  const clientOrderId = stringValue(item.clientOrderId);
  return {
    symbol: stringValue(item.symbol),
    orderId: stringValue(item.orderId),
    clientOrderId,
    localOrderId: localOrderIdFromClientOrderId(clientOrderId),
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

function normalizeTrade(raw: unknown, market: BinanceMarketListing): BinancePaperTrade {
  const item = asRecord(raw);
  const side = tradeSide(item);
  const price = numberValue(item.price);
  const quantity = numberValue(item.qty);
  const quoteQuantity = numberValue(item.quoteQty);
  const commission = numberValue(item.commission);
  const commissionAsset = stringValue(item.commissionAsset);
  const feeQuote =
    commissionAsset === market.quoteAsset
      ? commission
      : commissionAsset === market.baseAsset
        ? commission * price
        : 0;
  const clientOrderId = stringValue(item.clientOrderId);
  const id = stringValue(item.id);
  return {
    id: `${market.id}:trade:${id || `${stringValue(item.orderId)}:${stringValue(item.time)}`}`,
    symbol: stringValue(item.symbol),
    orderId: stringValue(item.orderId),
    clientOrderId,
    localOrderId: localOrderIdFromClientOrderId(clientOrderId),
    side,
    price,
    quantity,
    quoteQuantity: quoteQuantity || price * quantity,
    commission,
    commissionAsset,
    feeQuote,
    realizedPnl: optionalNumber(item.realizedPnl),
    positionSide: stringValue(item.positionSide),
    time: numberValue(item.time),
    maker: booleanValue(item.maker),
  };
}

function normalizeCommission(
  payload: Record<string, unknown>,
  product: ResolvedPaperEnvironment["product"],
): BinancePaperCommission {
  if (product === "spot") {
    const standard = asRecord(payload.standardCommission);
    const maker = optionalNumber(standard.maker ?? payload.makerCommission);
    const taker = optionalNumber(standard.taker ?? payload.takerCommission);
    return {
      makerFeeBps: rateToBps(maker),
      takerFeeBps: rateToBps(taker),
    };
  }

  return {
    makerFeeBps: rateToBps(optionalNumber(payload.makerCommissionRate)),
    takerFeeBps: rateToBps(optionalNumber(payload.takerCommissionRate)),
  };
}

function rateToBps(value: number | undefined): number | undefined {
  if (value === undefined || !Number.isFinite(value) || value < 0) {
    return undefined;
  }
  return value > 1 ? value : value * 10_000;
}

function normalizeSymbolFilters(raw: Record<string, unknown>): BinancePaperSymbolFilters {
  const filters = Array.isArray(raw.filters) ? raw.filters.map(asRecord) : [];
  const priceFilter = filterByType(filters, "PRICE_FILTER");
  const lotSize = filterByType(filters, "LOT_SIZE");
  const marketLotSize = filterByType(filters, "MARKET_LOT_SIZE");
  const minNotional = filterByType(filters, "MIN_NOTIONAL");
  const notional = filterByType(filters, "NOTIONAL");
  return {
    symbol: stringValue(raw.symbol),
    pricePrecision: optionalNumber(raw.pricePrecision),
    quantityPrecision: optionalNumber(raw.quantityPrecision),
    tickSize: optionalNumber(priceFilter.tickSize),
    minPrice: optionalNumber(priceFilter.minPrice),
    maxPrice: optionalNumber(priceFilter.maxPrice),
    stepSize: optionalNumber(lotSize.stepSize),
    marketStepSize: optionalNumber(marketLotSize.stepSize),
    minQuantity: optionalNumber(lotSize.minQty),
    maxQuantity: optionalNumber(lotSize.maxQty),
    minMarketQuantity: optionalNumber(marketLotSize.minQty),
    maxMarketQuantity: optionalNumber(marketLotSize.maxQty),
    minNotional: optionalNumber(
      notional.minNotional ??
        notional.notional ??
        minNotional.minNotional ??
        minNotional.notional,
    ),
    maxNotional: optionalNumber(notional.maxNotional),
  };
}

function filterByType(
  filters: Array<Record<string, unknown>>,
  filterType: string,
): Record<string, unknown> {
  return filters.find((filter) => stringValue(filter.filterType) === filterType) ?? {};
}

function parseMaxLeverage(payload: unknown, symbol: string): number | undefined {
  const rows = Array.isArray(payload) ? payload : [payload];
  let maxLeverage = 0;
  for (const row of rows.map(asRecord)) {
    const rowSymbol = stringValue(row.symbol) || stringValue(row.pair);
    if (rowSymbol && rowSymbol !== symbol) {
      continue;
    }
    const brackets = Array.isArray(row.brackets) ? row.brackets.map(asRecord) : [];
    for (const bracket of brackets) {
      maxLeverage = Math.max(maxLeverage, numberValue(bracket.initialLeverage));
    }
  }
  return maxLeverage > 0 ? maxLeverage : undefined;
}

function tradeSide(item: Record<string, unknown>): "buy" | "sell" {
  const side = stringValue(item.side).toUpperCase();
  if (side === "BUY") {
    return "buy";
  }
  if (side === "SELL") {
    return "sell";
  }
  return booleanValue(item.isBuyer) ? "buy" : "sell";
}

function futuresReconciliationFromUserDataEvent(
  market: BinanceMarketListing,
  payload: unknown,
): ExchangeReconciliationInput | undefined {
  const event = asRecord(payload);
  if (stringValue(event.e) !== "ORDER_TRADE_UPDATE") {
    return undefined;
  }

  const order = asRecord(event.o);
  if (stringValue(order.s) !== market.symbol) {
    return undefined;
  }

  const clientOrderId = stringValue(order.c);
  const localOrderId = localOrderIdFromClientOrderId(clientOrderId);
  if (!localOrderId) {
    return undefined;
  }

  const side = normalizeUserDataSide(order.S);
  const status = normalizeUserDataOrderStatus(order.X);
  const executionType = stringValue(order.x).toUpperCase();
  const externalOrderId = stringValue(order.i);
  const transactionTime = numberValue(event.T) || numberValue(order.T) || numberValue(event.E);
  const orderPrice = numberValue(order.ap) || numberValue(order.p) || numberValue(order.L);
  const filledQuantity = numberValue(order.z);
  const quoteQuantity = orderPrice > 0 ? orderPrice * filledQuantity : undefined;
  const reduceOnly = booleanValue(order.R);
  const positionEffect = reduceOnly ? "close" as const : undefined;
  const orders = [{
    localOrderId,
    externalOrderId,
    clientOrderId,
    side,
    type: normalizeUserDataOrderType(order.o),
    status,
    price: orderPrice,
    quantity: numberValue(order.q),
    filledQuantity,
    quoteQuantity,
    createdAt: transactionTime,
    updatedAt: transactionTime,
    positionEffect,
    reason: `user-data ${executionType || status}`,
  }];

  const lastQuantity = numberValue(order.l);
  const lastPrice = numberValue(order.L) || orderPrice;
  const tradeId = stringValue(order.t);
  const fills: ExchangeTradeFill[] = [];
  if (executionType === "TRADE" && lastQuantity > 0 && lastPrice > 0) {
    const commission = numberValue(order.n);
    const commissionAsset = stringValue(order.N);
    const feeQuote =
      commissionAsset === market.quoteAsset
        ? commission
        : commissionAsset === market.baseAsset
          ? commission * lastPrice
          : 0;
    fills.push({
      id: `${market.id}:trade:${tradeId || `${externalOrderId}:${transactionTime}`}`,
      localOrderId,
      externalOrderId,
      clientOrderId,
      side,
      price: lastPrice,
      quantity: lastQuantity,
      quoteQuantity: lastPrice * lastQuantity,
      feeQuote,
      realizedPnl: optionalNumber(order.rp),
      filledAt: transactionTime,
      reason: "user-data exchange fill",
      positionEffect,
    });
  }

  return { orders, fills };
}

function normalizeUserDataSide(value: unknown): "buy" | "sell" {
  return stringValue(value).toUpperCase() === "SELL" ? "sell" : "buy";
}

function normalizeUserDataOrderType(value: unknown): "limit" | "market" {
  return stringValue(value).toUpperCase() === "MARKET" ? "market" : "limit";
}

function normalizeUserDataOrderStatus(value: unknown): "open" | "filled" | "cancelled" {
  const status = stringValue(value).toUpperCase();
  if (status === "FILLED") {
    return "filled";
  }
  if (
    status === "CANCELED" ||
    status === "CANCELLED" ||
    status === "EXPIRED" ||
    status === "EXPIRED_IN_MATCH" ||
    status === "REJECTED"
  ) {
    return "cancelled";
  }
  return "open";
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

function optionalNumber(value: unknown): number | undefined {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
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
