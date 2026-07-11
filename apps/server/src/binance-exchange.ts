import { createHmac } from "node:crypto";
import type {
  ExchangeReconciliationInput,
  ExchangeTradeFill,
  TradingOrder,
} from "@trading/bot-algo";
import type { BinanceMarketListing } from "./binance-markets.js";

export type BinanceExchangeMode =
  | "auto"
  | "live"
  | "spot-live"
  | "usdm-futures-live"
  | "coinm-futures-live"
  | "spot-testnet"
  | "spot-demo"
  | "usdm-futures-testnet"
  | "coinm-futures-testnet";

export type ResolvedBinanceExchangeMode = Exclude<BinanceExchangeMode, "auto" | "live">;

export interface BinanceExchangeConfig {
  enabled: boolean;
  mode: BinanceExchangeMode;
  apiKey?: string;
  apiSecret?: string;
  liveApiKey?: string;
  liveApiSecret?: string;
  recvWindowMs: number;
  autoSubmit: boolean;
  baseUrlOverride?: string;
}

export interface BinanceExchangeBalance {
  asset: string;
  free: number;
  locked: number;
  walletBalance?: number;
  availableBalance?: number;
  unrealizedPnl?: number;
}

export interface BinanceExchangePosition {
  symbol: string;
  positionSide?: string;
  positionAmt: number;
  entryPrice?: number;
  markPrice?: number;
  unrealizedPnl?: number;
  notional?: number;
  maxNotional?: number;
  maxQuantity?: number;
  leverage?: number;
  marginType?: string;
  isolatedMargin?: number;
  updateTime?: number;
}

export type BinanceExchangePositionMode = "one-way" | "hedge";

export interface BinanceExchangeOrder {
  symbol: string;
  orderId: string;
  clientOrderId: string;
  localOrderId?: string;
  algo?: boolean;
  side: string;
  type: string;
  status: string;
  price: number;
  stopPrice?: number;
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

export interface BinanceExchangeTrade {
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

export interface BinanceExchangeSymbolFilters {
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

export interface BinanceExchangeCommission {
  makerFeeBps?: number;
  takerFeeBps?: number;
}

export interface BinanceExchangeSnapshot {
  enabled: boolean;
  configured: boolean;
  compatible: boolean;
  mode: BinanceExchangeMode;
  resolvedMode?: ResolvedBinanceExchangeMode;
  live: boolean;
  baseUrl?: string;
  streamEnvironment?: BinanceExchangeStreamEnvironment;
  autoSubmit: boolean;
  connected: boolean;
  userDataStreamConnected?: boolean;
  lastUserDataStreamAt?: number;
  userDataStreamMessage?: string;
  lastSyncAt?: number;
  lastSubmitAt?: number;
  positionMode?: BinanceExchangePositionMode;
  message: string;
  error?: string;
  maxLeverage?: number;
  symbolFilters?: BinanceExchangeSymbolFilters;
  commission?: BinanceExchangeCommission;
  feeBps?: number;
  estimatedSlippageBps?: number;
  balances: BinanceExchangeBalance[];
  positions: BinanceExchangePosition[];
  openOrders: BinanceExchangeOrder[];
  recentOrders: BinanceExchangeOrder[];
  recentTrades: BinanceExchangeTrade[];
  lastOrder?: BinanceExchangeOrder;
}

export interface BinanceExchangeUserDataStreamSession {
  listenKey: string;
  url: string;
  mode: ResolvedBinanceExchangeMode;
}

export interface BinanceExchangeUserDataStreamStatus {
  connected: boolean;
  message: string;
  lastEventAt: number;
  reconnectAttempt: number;
}

export interface BinanceExchangePlaceOrderInput {
  symbol?: string;
  side: "buy" | "sell";
  type: "limit" | "market" | "stop-market";
  quantity: number;
  price?: number;
  stopPrice?: number;
  timeInForce?: "GTC" | "IOC" | "FOK" | "GTX";
  reduceOnly?: boolean;
  positionSide?: "BOTH" | "LONG" | "SHORT";
  clientOrderId?: string;
}

export interface BinanceExchangeCancelOrderInput {
  symbol?: string;
  orderId?: string | number;
  clientOrderId?: string;
  algo?: boolean;
}

export interface BinanceExchangePositionContext {
  positions: BinanceExchangePosition[];
  positionMode?: BinanceExchangePositionMode;
}

export interface BinanceExchangeOrderCapacityState extends BinanceExchangePositionContext {
  availableBalanceQuote: number;
  openOrders: BinanceExchangeOrder[];
}

export interface BinanceExchangeOrderUpdates {
  openOrders: BinanceExchangeOrder[];
  recentTrades: BinanceExchangeTrade[];
}

export interface BinanceExchangeMarketInfo {
  filters?: BinanceExchangeSymbolFilters;
  maxLeverage?: number;
}

export interface BinanceExchangeFriction {
  feeBps?: number;
  estimatedSlippageBps?: number;
}

export class BinanceExchangeOrderSubmissionSkipped extends Error {
  readonly recoverable = true;

  constructor(message: string) {
    super(message);
    this.name = "BinanceExchangeOrderSubmissionSkipped";
  }
}

export type BinanceExchangeStreamEnvironment =
  | "live"
  | "spot-testnet"
  | "spot-demo"
  | "usdm-futures-testnet"
  | "coinm-futures-testnet";

interface ResolvedExchangeEnvironment {
  mode: ResolvedBinanceExchangeMode;
  product: "spot" | "usdm-futures" | "coinm-futures";
  live: boolean;
  baseUrl: string;
  restPrefix: "/api/v3" | "/fapi/v1" | "/dapi/v1";
  accountPath: string;
  balancePath?: string;
  streamEnvironment: BinanceExchangeStreamEnvironment;
}

interface BinanceApiErrorPayload {
  code?: unknown;
  msg?: unknown;
}

const EMPTY_SNAPSHOT: Pick<
  BinanceExchangeSnapshot,
  "balances" | "positions" | "openOrders" | "recentOrders" | "recentTrades" | "connected"
> = {
  connected: false,
  balances: [],
  positions: [],
  openOrders: [],
  recentOrders: [],
  recentTrades: [],
};

export class BinanceExchangeTrading {
  private snapshots = new Map<string, BinanceExchangeSnapshot>();
  private timeOffsets = new Map<string, number>();
  private symbolFilters = new Map<string, BinanceExchangeSymbolFilters>();
  private maxLeverageBySymbol = new Map<string, number>();

  constructor(private readonly config: BinanceExchangeConfig) {}

  updateConfig(patch: Partial<BinanceExchangeConfig>): void {
    Object.assign(this.config, patch);
    this.snapshots.clear();
  }

  drivesOrderExecution(market: BinanceMarketListing): boolean {
    return Boolean(
      this.config.enabled &&
        this.config.autoSubmit &&
        this.canSubmitOrders(market),
    );
  }

  canSubmitOrders(market: BinanceMarketListing): boolean {
    const environment = this.resolveEnvironment(market);
    const credentials = environment ? this.credentialsFor(environment) : undefined;
    return Boolean(
      this.config.enabled &&
        environment &&
        credentials?.apiKey &&
        credentials?.apiSecret,
    );
  }

  canStreamUserData(market: BinanceMarketListing): boolean {
    const environment = this.resolveEnvironment(market);
    const credentials = environment ? this.credentialsFor(environment) : undefined;
    return Boolean(
      this.config.enabled &&
        environment &&
        credentials?.apiKey &&
        credentials?.apiSecret &&
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
  ): Promise<BinanceExchangeUserDataStreamSession> {
    const environment = this.requireReadyEnvironment(market);
    if (environment.product === "spot") {
      throw new Error("Spot user-data streams require WebSocket API auth and are not enabled yet.");
    }
    const payload = await this.apiKeyRequest<Record<string, unknown>>(
      environment,
      "POST",
      userDataStreamPath(environment),
    );
    const listenKey = stringValue(payload.listenKey);
    if (!listenKey) {
      throw new Error("Binance user-data stream did not return a listenKey.");
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
    status: BinanceExchangeUserDataStreamStatus,
  ): BinanceExchangeSnapshot {
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

  streamEnvironmentFor(market: BinanceMarketListing): BinanceExchangeStreamEnvironment {
    const environment = this.resolveEnvironment(market);
    if (!environment || !this.config.enabled) {
      return "live";
    }
    return environment.streamEnvironment;
  }

  snapshot(market: BinanceMarketListing): BinanceExchangeSnapshot {
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
        live: false,
        autoSubmit: this.config.autoSubmit,
        message: "Binance exchange trading disabled",
      };
    }

    if (!environment) {
      return {
        ...EMPTY_SNAPSHOT,
        enabled: true,
        configured: false,
        compatible: false,
        mode: this.config.mode,
        live: false,
        autoSubmit: this.config.autoSubmit,
        message: `Binance exchange mode ${this.config.mode} is not compatible with ${market.venue}`,
      };
    }

    const credentials = this.credentialsFor(environment);
    return {
      ...EMPTY_SNAPSHOT,
      enabled: true,
      configured: Boolean(credentials.apiKey && credentials.apiSecret),
      compatible: true,
      mode: this.config.mode,
      resolvedMode: environment.mode,
      live: environment.live,
      baseUrl: environment.baseUrl,
      streamEnvironment: environment.streamEnvironment,
      autoSubmit: this.config.autoSubmit,
      message:
        credentials.apiKey && credentials.apiSecret
          ? `${environment.live ? "Live" : "Sandbox"} Binance exchange ready; sync not run yet`
          : environment.live
            ? "Add live Binance tokens in the dashboard to enable signed requests"
            : "Add sandbox Binance tokens in the dashboard to enable signed requests",
    };
  }

  async fetchAccountBalances(market: BinanceMarketListing): Promise<BinanceExchangeBalance[]> {
    const environment = this.requireReadyEnvironment(market);
    return this.fetchBalances(environment);
  }

  async fetchPositionContext(market: BinanceMarketListing): Promise<BinanceExchangePositionContext> {
    const environment = this.requireReadyEnvironment(market);
    if (environment.product === "spot") return { positions: [] };
    const [account, positionMode] = await Promise.all([
      this.fetchAccount(environment),
      this.fetchPositionMode(environment),
    ]);
    return { positions: extractPositions(environment, account, market.symbol), positionMode };
  }

  async fetchOrderCapacityState(
    market: BinanceMarketListing,
    price: number,
  ): Promise<BinanceExchangeOrderCapacityState> {
    const environment = this.requireReadyEnvironment(market);
    if (environment.product === "spot") {
      throw new Error("Spot order capacity uses account balances directly.");
    }
    const [account, openOrders, positionMode] = await Promise.all([
      this.fetchAccount(environment),
      this.fetchOpenOrders(environment, market.symbol),
      this.fetchPositionMode(environment),
    ]);
    const positions = extractPositions(environment, account, market.symbol, false);
    return {
      availableBalanceQuote: availableBalanceQuote(environment, account, market, price),
      positions,
      openOrders,
      positionMode: positionMode ?? (
        positions.some((position) => position.positionSide === "LONG" || position.positionSide === "SHORT")
          ? "hedge"
          : "one-way"
      ),
    };
  }

  async fetchOrderUpdates(market: BinanceMarketListing): Promise<BinanceExchangeOrderUpdates> {
    const environment = this.requireReadyEnvironment(market);
    const [openOrders, recentOrders, recentTrades] = await Promise.all([
      this.fetchOpenOrders(environment, market.symbol),
      this.fetchRecentOrders(environment, market.symbol),
      this.fetchRecentTrades(environment, market),
    ]);
    const ordersById = new Map(recentOrders.map((order) => [order.orderId, order]));
    return {
      openOrders,
      recentTrades: recentTrades.map((trade) => ({
        ...trade,
        clientOrderId: trade.clientOrderId || ordersById.get(trade.orderId)?.clientOrderId,
        localOrderId: trade.localOrderId || ordersById.get(trade.orderId)?.localOrderId,
      })),
    };
  }

  async fetchMarketInfo(market: BinanceMarketListing): Promise<BinanceExchangeMarketInfo> {
    const environment = this.requireReadyEnvironment(market);
    const [filters, maxLeverage] = await Promise.all([
      this.fetchSymbolFilters(environment, market.symbol),
      this.fetchMaxLeverage(environment, market.symbol),
    ]);
    return { filters, maxLeverage };
  }

  async fetchFriction(market: BinanceMarketListing): Promise<BinanceExchangeFriction> {
    const environment = this.requireReadyEnvironment(market);
    const [commission, estimatedSlippageBps] = await Promise.all([
      this.fetchCommission(environment, market.symbol),
      this.fetchEstimatedSlippageBps(environment, market.symbol),
    ]);
    return { feeBps: commission?.takerFeeBps, estimatedSlippageBps };
  }

  async submitOrderDirect(
    market: BinanceMarketListing,
    input: BinanceExchangePlaceOrderInput,
  ): Promise<BinanceExchangeOrder> {
    const environment = this.requireReadyEnvironment(market);
    const normalizedInput = await this.normalizeOrderInput(environment, market, input);
    const payload = await this.signedRequest<Record<string, unknown>>(
      environment,
      "POST",
      orderSubmissionPath(environment, normalizedInput),
      orderParams(environment, market.symbol, normalizedInput),
    );
    return normalizeOrder(payload);
  }

  async cancelOrderDirect(
    market: BinanceMarketListing,
    input: BinanceExchangeCancelOrderInput,
  ): Promise<BinanceExchangeOrder> {
    const environment = this.requireReadyEnvironment(market);
    const algo = input.algo === true && environment.product !== "spot";
    const params: Record<string, string> = algo ? {} : {
      symbol: (input.symbol ?? market.symbol).toUpperCase(),
    };
    if (input.orderId !== undefined) params[algo ? "algoId" : "orderId"] = String(input.orderId);
    if (input.clientOrderId) params[algo ? "clientAlgoId" : "origClientOrderId"] = input.clientOrderId;
    if (
      (!algo && !params.orderId && !params.origClientOrderId) ||
      (algo && !params.algoId && !params.clientAlgoId)
    ) throw new Error("Provide orderId or clientOrderId to cancel a Binance order.");
    return normalizeOrder(await this.signedRequest<Record<string, unknown>>(
      environment,
      "DELETE",
      algo ? `${orderRestPrefix(environment)}/algoOrder` : `${orderRestPrefix(environment)}/order`,
      params,
    ));
  }

  async sync(market: BinanceMarketListing): Promise<BinanceExchangeSnapshot> {
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
      positionMode,
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
      this.fetchPositionMode(environment),
    ]);
    const ordersById = new Map(recentOrders.map((order) => [order.orderId, order]));
    const trades = recentTrades.map((trade) => ({
      ...trade,
      clientOrderId: trade.clientOrderId || ordersById.get(trade.orderId)?.clientOrderId,
      localOrderId: trade.localOrderId || ordersById.get(trade.orderId)?.localOrderId,
    }));
    const positions = extractPositions(environment, account, market.symbol);
    const snapshot: BinanceExchangeSnapshot = {
      ...this.snapshot(market),
      configured: true,
      compatible: true,
      connected: true,
      lastSyncAt: Date.now(),
      positionMode,
      message: `${environment.live ? "Live" : "Sandbox"} Binance account synced`,
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
    input: BinanceExchangePlaceOrderInput,
  ): Promise<BinanceExchangeSnapshot> {
    const lastOrder = await this.submitOrderDirect(market, input);
    const synced = await this.sync(market);
    const snapshot: BinanceExchangeSnapshot = {
      ...synced,
      lastSubmitAt: Date.now(),
      lastOrder,
      message: `${input.side.toUpperCase()} ${input.type.toUpperCase()} order submitted to ${synced.resolvedMode ?? synced.mode}`,
    };
    const environment = this.requireReadyEnvironment(market);
    this.snapshots.set(snapshotKey(environment, market.symbol), snapshot);
    return snapshot;
  }

  async submitBotOrder(
    market: BinanceMarketListing,
    order: TradingOrder,
    options: { force?: boolean } = {},
  ): Promise<BinanceExchangeSnapshot | undefined> {
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

    const reduceOnly = order.positionEffect === "close" ? true : undefined;
    const positionSide =
      environment.product === "spot"
        ? undefined
        : futuresPositionSideForBotOrder(order);

    if (environment.product !== "spot" && order.positionEffect === "close") {
      await this.assertReducibleFuturesPosition(environment, market, order);
    }

    return this.placeOrder(market, {
      symbol: market.symbol,
      side: order.side,
      type: order.type,
      quantity: order.quantity,
      price: order.type === "stop-market" ? undefined : order.price,
      stopPrice: order.type === "stop-market" ? order.price : undefined,
      timeInForce: order.type === "limit" ? "GTC" : undefined,
      reduceOnly,
      positionSide,
      clientOrderId: clientOrderIdForBotOrder(order.id),
    });
  }

  private async assertReducibleFuturesPosition(
    environment: ResolvedExchangeEnvironment,
    market: BinanceMarketListing,
    order: TradingOrder,
  ): Promise<void> {
    const positionMode = await this.fetchPositionMode(environment);
    const snapshot = await this.latestSnapshotForSubmission(environment, market);
    const availableQuantity = reducibleFuturesQuantityForOrder(
      snapshot,
      order,
      positionMode,
    );
    const tolerance = Math.max(0.00000001, order.quantity * 0.000001);
    if (order.quantity <= availableQuantity + tolerance) {
      return;
    }

    const targetSide = order.side === "buy" ? "short" : "long";
    throw new BinanceExchangeOrderSubmissionSkipped(
      `Binance submit skipped: ${order.side.toUpperCase()} close order ${order.id} requires ${formatQuantity(order.quantity)} ${market.baseAsset}, but only ${formatQuantity(availableQuantity)} ${targetSide} ${market.baseAsset} is reducible after open close orders.`,
    );
  }

  private async latestSnapshotForSubmission(
    environment: ResolvedExchangeEnvironment,
    market: BinanceMarketListing,
  ): Promise<BinanceExchangeSnapshot> {
    const cached = this.snapshots.get(snapshotKey(environment, market.symbol));
    if (cached?.connected) {
      return cached;
    }
    return this.sync(market);
  }

  async cancelOrder(
    market: BinanceMarketListing,
    input: BinanceExchangeCancelOrderInput,
  ): Promise<BinanceExchangeSnapshot> {
    const environment = this.requireReadyEnvironment(market);
    const lastOrder = await this.cancelOrderDirect(market, input);
    const synced = await this.sync(market);
    const snapshot: BinanceExchangeSnapshot = {
      ...synced,
      lastSubmitAt: Date.now(),
      lastOrder,
      message: `Order ${lastOrder.orderId || input.orderId || input.clientOrderId} cancelled on ${environment.mode}`,
    };
    this.snapshots.set(snapshotKey(environment, market.symbol), snapshot);
    return snapshot;
  }

  async cancelAllOpenOrders(market: BinanceMarketListing): Promise<BinanceExchangeSnapshot> {
    const environment = this.requireReadyEnvironment(market);
    const symbol = market.symbol;
    if (environment.product === "spot") {
      await this.signedRequest<unknown>(
        environment,
        "DELETE",
        "/api/v3/openOrders",
        { symbol },
      );
    } else {
      await Promise.all([
        this.signedRequest<unknown>(
          environment,
          "DELETE",
          `${orderRestPrefix(environment)}/allOpenOrders`,
          { symbol },
        ),
        this.signedRequest<unknown>(
          environment,
          "DELETE",
          `${orderRestPrefix(environment)}/algoOpenOrders`,
          { symbol },
        ),
      ]);
    }
    const synced = await this.sync(market);
    const snapshot: BinanceExchangeSnapshot = {
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
  ): Promise<BinanceExchangeSnapshot> {
    const environment = this.requireReadyEnvironment(market);
    if (environment.product === "spot") {
      throw new Error("Exchange position close is only available for futures modes.");
    }

    const initial = await this.sync(market);
    const positions = initial.positions.filter((position) =>
      shouldClosePosition(position, options.includeUnprofitable === true),
    );
    if (positions.length === 0) {
      return {
        ...initial,
        message: "No open Binance positions to close",
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
    const snapshot: BinanceExchangeSnapshot = {
      ...synced,
      lastSubmitAt: Date.now(),
      message:
        closedCount === 1
          ? `Closed 1 open ${market.symbol} Binance position`
          : `Closed ${closedCount} open ${market.symbol} Binance positions`,
    };
    this.snapshots.set(snapshotKey(environment, market.symbol), snapshot);
    return snapshot;
  }

  async changeLeverage(
    market: BinanceMarketListing,
    leverage: number,
  ): Promise<BinanceExchangeSnapshot> {
    const environment = this.requireReadyEnvironment(market);
    if (environment.product === "spot") {
      throw new Error("Leverage is only available for futures modes.");
    }
    const maxLeverage = await this.fetchMaxLeverage(environment, market.symbol);
    const leverageCap = maxLeverage && maxLeverage > 0 ? maxLeverage : 125;
    const nextLeverage = Math.max(1, Math.min(leverageCap, Math.round(leverage)));
    await this.signedRequest<unknown>(environment, "POST", `${orderRestPrefix(environment)}/leverage`, {
      symbol: market.symbol,
      leverage: String(nextLeverage),
    });
    const synced = await this.sync(market);
    const snapshot: BinanceExchangeSnapshot = {
      ...synced,
      lastSubmitAt: Date.now(),
      message: `${market.symbol} leverage set to ${nextLeverage}x on ${environment.mode}`,
    };
    this.snapshots.set(snapshotKey(environment, market.symbol), snapshot);
    return snapshot;
  }

  private requireReadyEnvironment(market: BinanceMarketListing): ResolvedExchangeEnvironment {
    if (!this.config.enabled) {
      throw new Error("Binance exchange trading is disabled.");
    }
    const environment = this.resolveEnvironment(market);
    if (!environment) {
      throw new Error(
        `Binance exchange mode ${this.config.mode} is not compatible with ${market.venue}.`,
      );
    }
    const credentials = this.credentialsFor(environment);
    if (!credentials.apiKey || !credentials.apiSecret) {
      throw new Error(
        environment.live
          ? "BINANCE_API_KEY and BINANCE_API_SECRET are required for live Binance exchange requests."
          : "BINANCE_PAPER_API_KEY and BINANCE_PAPER_API_SECRET are required for sandbox Binance exchange requests.",
      );
    }
    return environment;
  }

  private resolveEnvironment(
    market: BinanceMarketListing,
  ): ResolvedExchangeEnvironment | undefined {
    const mode =
      this.config.mode === "auto"
        ? defaultModeForVenue(market.venue, "sandbox")
        : this.config.mode === "live"
          ? defaultModeForVenue(market.venue, "live")
          : this.config.mode;
    if (!mode) {
      return undefined;
    }
    if (!modeIsCompatibleWithVenue(mode, market.venue)) {
      return undefined;
    }
    return environmentForMode(mode, this.config.baseUrlOverride);
  }

  private credentialsFor(
    environment: ResolvedExchangeEnvironment,
  ): { apiKey?: string; apiSecret?: string } {
    return environment.live
      ? {
          apiKey: this.config.liveApiKey,
          apiSecret: this.config.liveApiSecret,
        }
      : {
          apiKey: this.config.apiKey,
          apiSecret: this.config.apiSecret,
        };
  }

  private async fetchBalances(
    environment: ResolvedExchangeEnvironment,
  ): Promise<BinanceExchangeBalance[]> {
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
    environment: ResolvedExchangeEnvironment,
  ): Promise<Record<string, unknown>> {
    if (environment.product === "spot") {
      return {};
    }
    return this.signedRequest<Record<string, unknown>>(environment, "GET", environment.accountPath);
  }

  private async fetchOpenOrders(
    environment: ResolvedExchangeEnvironment,
    symbol: string,
  ): Promise<BinanceExchangeOrder[]> {
    const payload = await this.signedRequest<unknown[]>(
      environment,
      "GET",
      `${orderRestPrefix(environment)}/openOrders`,
      { symbol },
    );
    const orders = payload.map(normalizeOrder);
    if (environment.product === "spot") {
      return orders;
    }
    return [...orders, ...(await this.fetchOpenAlgoOrders(environment, symbol))];
  }

  private async fetchRecentOrders(
    environment: ResolvedExchangeEnvironment,
    symbol: string,
  ): Promise<BinanceExchangeOrder[]> {
    try {
      const payload = await this.signedRequest<unknown[]>(
        environment,
        "GET",
        `${orderRestPrefix(environment)}/allOrders`,
        { symbol, limit: 100 },
      );
      const orders = payload.map(normalizeOrder);
      if (environment.product === "spot") {
        return orders;
      }
      return [...orders, ...(await this.fetchRecentAlgoOrders(environment, symbol))];
    } catch {
      return [];
    }
  }

  private async fetchOpenAlgoOrders(
    environment: ResolvedExchangeEnvironment,
    symbol: string,
  ): Promise<BinanceExchangeOrder[]> {
    try {
      const payload = await this.signedRequest<unknown[]>(
        environment,
        "GET",
        `${orderRestPrefix(environment)}/openAlgoOrders`,
        { symbol },
      );
      return payload.map(normalizeOrder);
    } catch {
      return [];
    }
  }

  private async fetchRecentAlgoOrders(
    environment: ResolvedExchangeEnvironment,
    symbol: string,
  ): Promise<BinanceExchangeOrder[]> {
    try {
      const payload = await this.signedRequest<unknown[]>(
        environment,
        "GET",
        `${orderRestPrefix(environment)}/allAlgoOrders`,
        { symbol, limit: 100 },
      );
      return payload.map(normalizeOrder);
    } catch {
      return [];
    }
  }

  private async fetchRecentTrades(
    environment: ResolvedExchangeEnvironment,
    market: BinanceMarketListing,
  ): Promise<BinanceExchangeTrade[]> {
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
    environment: ResolvedExchangeEnvironment,
    symbol: string,
  ): Promise<BinanceExchangeSymbolFilters | undefined> {
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
    environment: ResolvedExchangeEnvironment,
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
    environment: ResolvedExchangeEnvironment,
    symbol: string,
  ): Promise<BinanceExchangeCommission | undefined> {
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
    environment: ResolvedExchangeEnvironment,
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
    environment: ResolvedExchangeEnvironment,
    market: BinanceMarketListing,
    input: BinanceExchangePlaceOrderInput,
  ): Promise<BinanceExchangePlaceOrderInput> {
    const positionMode = input.positionSide === "BOTH"
      ? "one-way"
      : input.positionSide === "LONG" || input.positionSide === "SHORT"
        ? "hedge"
        : await this.fetchPositionMode(environment);
    const positionInput = normalizeFuturesPositionModeInput(input, positionMode, environment);
    const filters = await this.fetchSymbolFilters(environment, market.symbol);
    if (!filters) {
      return positionInput;
    }

    const price =
      positionInput.type === "limit"
        ? normalizePrice(positionInput.price, positionInput.side, filters)
        : positionInput.price;
    const stopPrice =
      positionInput.type === "stop-market"
        ? normalizeStopPrice(positionInput.stopPrice ?? positionInput.price, positionInput.side, filters)
        : positionInput.stopPrice;
    const quantity = normalizeQuantity(positionInput.quantity, positionInput.type, filters);
    const effectivePrice =
      positionInput.type === "limit"
        ? price
        : positionInput.type === "stop-market"
          ? stopPrice
          : Number.isFinite(Number(positionInput.price)) && Number(positionInput.price) > 0
            ? Number(positionInput.price)
            : undefined;
    const minNotional = filters.minNotional ?? 0;
    let adjustedQuantity = quantity;

    if (minNotional > 0 && effectivePrice && effectivePrice * adjustedQuantity < minNotional) {
      adjustedQuantity = normalizeQuantity(
        minNotional / effectivePrice,
        positionInput.type,
        filters,
        "ceil",
      );
    }

    validateNormalizedOrder(positionInput, adjustedQuantity, effectivePrice, filters);
    return {
      ...positionInput,
      price: positionInput.type === "limit" ? effectivePrice : positionInput.price,
      stopPrice,
      quantity: adjustedQuantity,
    };
  }

  private async fetchPositionMode(
    environment: ResolvedExchangeEnvironment,
  ): Promise<BinanceExchangePositionMode | undefined> {
    if (environment.product === "spot") {
      return undefined;
    }

    try {
      const payload = await this.signedRequest<Record<string, unknown>>(
        environment,
        "GET",
        positionSideModePath(environment),
      );
      return booleanValue(payload.dualSidePosition) ? "hedge" : "one-way";
    } catch {
      return undefined;
    }
  }

  private async signedRequest<T>(
    environment: ResolvedExchangeEnvironment,
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
    const credentials = this.credentialsFor(environment);
    const signature = createHmac("sha256", credentials.apiSecret ?? "")
      .update(search.toString())
      .digest("hex");
    search.set("signature", signature);

    return requestJson<T>(new URL(`${path}?${search.toString()}`, environment.baseUrl), {
      method,
      headers: {
        "X-MBX-APIKEY": credentials.apiKey ?? "",
      },
    });
  }

  private async apiKeyRequest<T>(
    environment: ResolvedExchangeEnvironment,
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
    const credentials = this.credentialsFor(environment);
    return requestJson<T>(new URL(pathWithQuery, environment.baseUrl), {
      method,
      headers: {
        "X-MBX-APIKEY": credentials.apiKey ?? "",
      },
    });
  }

  private async syncTime(environment: ResolvedExchangeEnvironment): Promise<void> {
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
  environment: "sandbox" | "live",
): ResolvedBinanceExchangeMode | undefined {
  if (environment === "live") {
    if (venue === "spot") {
      return "spot-live";
    }
    if (venue === "usdm-futures") {
      return "usdm-futures-live";
    }
    if (venue === "coinm-futures") {
      return "coinm-futures-live";
    }
    return undefined;
  }

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
  mode: ResolvedBinanceExchangeMode,
  venue: BinanceMarketListing["venue"],
): boolean {
  if (mode === "spot-testnet" || mode === "spot-demo" || mode === "spot-live") {
    return venue === "spot";
  }
  if (mode === "usdm-futures-testnet" || mode === "usdm-futures-live") {
    return venue === "usdm-futures";
  }
  if (mode === "coinm-futures-testnet" || mode === "coinm-futures-live") {
    return venue === "coinm-futures";
  }
  return false;
}

function environmentForMode(
  mode: ResolvedBinanceExchangeMode,
  baseUrlOverride: string | undefined,
): ResolvedExchangeEnvironment {
  if (mode === "spot-live") {
    return {
      mode,
      product: "spot",
      live: true,
      baseUrl: baseUrlOverride ?? "https://api.binance.com",
      restPrefix: "/api/v3",
      accountPath: "/api/v3/account",
      streamEnvironment: "live",
    };
  }
  if (mode === "spot-testnet") {
    return {
      mode,
      product: "spot",
      live: false,
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
      live: false,
      baseUrl: baseUrlOverride ?? "https://demo-api.binance.com",
      restPrefix: "/api/v3",
      accountPath: "/api/v3/account",
      streamEnvironment: "spot-demo",
    };
  }
  if (mode === "usdm-futures-live") {
    return {
      mode,
      product: "usdm-futures",
      live: true,
      baseUrl: baseUrlOverride ?? "https://fapi.binance.com",
      restPrefix: "/fapi/v1",
      accountPath: "/fapi/v3/account",
      balancePath: "/fapi/v3/balance",
      streamEnvironment: "live",
    };
  }
  if (mode === "usdm-futures-testnet") {
    return {
      mode,
      product: "usdm-futures",
      live: false,
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
    live: mode === "coinm-futures-live",
    baseUrl: baseUrlOverride ?? (mode === "coinm-futures-live"
      ? "https://dapi.binance.com"
      : "https://demo-dapi.binance.com"),
    restPrefix: "/dapi/v1",
    accountPath: "/dapi/v1/account",
    balancePath: "/dapi/v1/balance",
    streamEnvironment: mode === "coinm-futures-live" ? "live" : "coinm-futures-testnet",
  };
}

function orderRestPrefix(
  environment: ResolvedExchangeEnvironment,
): "/api/v3" | "/fapi/v1" | "/dapi/v1" {
  if (environment.product === "spot") {
    return "/api/v3";
  }
  return environment.restPrefix === "/dapi/v1" ? "/dapi/v1" : "/fapi/v1";
}

function exchangeInfoPath(environment: ResolvedExchangeEnvironment): string {
  if (environment.product === "spot") {
    return "/api/v3/exchangeInfo";
  }
  return `${orderRestPrefix(environment)}/exchangeInfo`;
}

function exchangeKlinePath(environment: ResolvedExchangeEnvironment): string {
  return `${orderRestPrefix(environment)}/klines`;
}

function bookTickerPath(environment: ResolvedExchangeEnvironment): string {
  return `${orderRestPrefix(environment)}/ticker/bookTicker`;
}

function commissionPath(environment: ResolvedExchangeEnvironment): string {
  if (environment.product === "spot") {
    return "/api/v3/account/commission";
  }
  return environment.product === "coinm-futures"
    ? "/dapi/v1/commissionRate"
    : "/fapi/v1/commissionRate";
}

function leverageBracketPath(environment: ResolvedExchangeEnvironment): string {
  return environment.product === "coinm-futures"
    ? "/dapi/v2/leverageBracket"
    : "/fapi/v1/leverageBracket";
}

function positionSideModePath(environment: ResolvedExchangeEnvironment): string {
  return `${orderRestPrefix(environment)}/positionSide/dual`;
}

function userDataStreamPath(environment: ResolvedExchangeEnvironment): string {
  return `${orderRestPrefix(environment)}/listenKey`;
}

function userDataStreamWebSocketUrl(
  environment: ResolvedExchangeEnvironment,
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
  filters: BinanceExchangeSymbolFilters,
): number | undefined {
  if (value === undefined) {
    return undefined;
  }
  const price = Number(value);
  if (!Number.isFinite(price) || price <= 0) {
    throw new Error("Limit Binance orders require a positive price.");
  }
  const tickSize = filters.tickSize ?? 0;
  if (tickSize <= 0) {
    return price;
  }
  return side === "buy"
    ? roundToStep(price, tickSize, "floor")
    : roundToStep(price, tickSize, "ceil");
}

function normalizeStopPrice(
  value: number | undefined,
  side: "buy" | "sell",
  filters: BinanceExchangeSymbolFilters,
): number | undefined {
  if (value === undefined) {
    return undefined;
  }
  const price = Number(value);
  if (!Number.isFinite(price) || price <= 0) {
    throw new Error("Stop-market Binance orders require a positive stop price.");
  }
  const tickSize = filters.tickSize ?? 0;
  if (tickSize <= 0) {
    return price;
  }
  return side === "buy"
    ? roundToStep(price, tickSize, "ceil")
    : roundToStep(price, tickSize, "floor");
}

function normalizeQuantity(
  value: number,
  type: BinanceExchangePlaceOrderInput["type"],
  filters: BinanceExchangeSymbolFilters,
  mode: "floor" | "ceil" = "floor",
): number {
  const quantity = Number(value);
  if (!Number.isFinite(quantity) || quantity <= 0) {
    throw new Error("Binance order quantity must be positive.");
  }
  const stepSize =
    isMarketLikeOrderType(type) && filters.marketStepSize
      ? filters.marketStepSize
      : filters.stepSize;
  if (!stepSize || stepSize <= 0) {
    return quantity;
  }
  return roundToStep(quantity, stepSize, mode);
}

function validateNormalizedOrder(
  input: BinanceExchangePlaceOrderInput,
  quantity: number,
  price: number | undefined,
  filters: BinanceExchangeSymbolFilters,
): void {
  const minQuantity =
    isMarketLikeOrderType(input.type) && filters.minMarketQuantity
      ? filters.minMarketQuantity
      : filters.minQuantity;
  const maxQuantity =
    isMarketLikeOrderType(input.type) && filters.maxMarketQuantity
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

function normalizeFuturesPositionModeInput(
  input: BinanceExchangePlaceOrderInput,
  positionMode: BinanceExchangePositionMode | undefined,
  environment: ResolvedExchangeEnvironment,
): BinanceExchangePlaceOrderInput {
  if (environment.product === "spot") {
    return {
      ...input,
      reduceOnly: undefined,
      positionSide: undefined,
    };
  }

  if (positionMode === "hedge") {
    return {
      ...input,
      reduceOnly: undefined,
      positionSide: input.positionSide ?? inferHedgePositionSide(input),
    };
  }

  return {
    ...input,
    reduceOnly: input.reduceOnly ? true : undefined,
    positionSide: input.positionSide === "BOTH" ? "BOTH" : undefined,
  };
}

function inferHedgePositionSide(
  input: BinanceExchangePlaceOrderInput,
): "LONG" | "SHORT" {
  if (input.reduceOnly) {
    return input.side === "sell" ? "LONG" : "SHORT";
  }

  return input.side === "buy" ? "LONG" : "SHORT";
}

function futuresPositionSideForBotOrder(
  order: TradingOrder,
): "LONG" | "SHORT" | undefined {
  if (order.positionEffect === "close") {
    return order.side === "sell" ? "LONG" : "SHORT";
  }
  if (order.positionEffect === "open") {
    return order.side === "buy" ? "LONG" : "SHORT";
  }
  return undefined;
}

function reducibleFuturesQuantityForOrder(
  snapshot: BinanceExchangeSnapshot,
  order: TradingOrder,
  positionMode: BinanceExchangePositionMode | undefined,
): number {
  const mode = positionMode ?? snapshot.positionMode ?? "one-way";
  const positionQuantity = futuresPositionQuantityForClose(snapshot, order, mode);
  const openCloseQuantity = openFuturesCloseOrderQuantity(snapshot, order, mode);
  return Math.max(0, positionQuantity - openCloseQuantity);
}

function futuresPositionQuantityForClose(
  snapshot: BinanceExchangeSnapshot,
  order: TradingOrder,
  positionMode: BinanceExchangePositionMode,
): number {
  let quantity = 0;
  for (const position of snapshot.positions) {
    const positionSide = stringValue(position.positionSide).toUpperCase();
    const positionAmt = position.positionAmt;
    if (positionMode === "hedge") {
      if (order.side === "buy" && positionSide === "SHORT") {
        quantity += Math.abs(positionAmt);
      } else if (order.side === "sell" && positionSide === "LONG") {
        quantity += Math.abs(positionAmt);
      }
      continue;
    }

    if (order.side === "buy" && positionAmt < 0) {
      quantity += -positionAmt;
    } else if (order.side === "sell" && positionAmt > 0) {
      quantity += positionAmt;
    }
  }
  return quantity;
}

function openFuturesCloseOrderQuantity(
  snapshot: BinanceExchangeSnapshot,
  order: TradingOrder,
  positionMode: BinanceExchangePositionMode,
): number {
  let quantity = 0;
  for (const openOrder of snapshot.openOrders) {
    if (!isOpenFuturesCloseOrderForSide(openOrder, order.side, positionMode)) {
      continue;
    }
    quantity += Math.max(0, openOrder.originalQuantity - openOrder.executedQuantity);
  }
  return quantity;
}

function isOpenFuturesCloseOrderForSide(
  order: BinanceExchangeOrder,
  side: TradingOrder["side"],
  positionMode: BinanceExchangePositionMode,
): boolean {
  const orderSide = normalizeOrderSide(order.side);
  if (orderSide !== side) {
    return false;
  }
  if (positionMode === "hedge") {
    const positionSide = stringValue(order.positionSide).toUpperCase();
    return (
      order.reduceOnly === true ||
      (orderSide === "buy" && positionSide === "SHORT") ||
      (orderSide === "sell" && positionSide === "LONG")
    );
  }
  return order.reduceOnly === true;
}

function normalizeOrderSide(side: string): TradingOrder["side"] {
  return side.toUpperCase() === "SELL" ? "sell" : "buy";
}

function isMarketLikeOrderType(type: BinanceExchangePlaceOrderInput["type"]): boolean {
  return type === "market" || type === "stop-market";
}

function formatQuantity(value: number): string {
  if (!Number.isFinite(value)) {
    return "0";
  }
  return Number(value.toFixed(8)).toString();
}

function orderSubmissionPath(
  environment: ResolvedExchangeEnvironment,
  input: BinanceExchangePlaceOrderInput,
): string {
  if (input.type === "stop-market" && environment.product !== "spot") {
    return `${orderRestPrefix(environment)}/algoOrder`;
  }
  return `${orderRestPrefix(environment)}/order`;
}

function orderParams(
  environment: ResolvedExchangeEnvironment,
  symbol: string,
  input: BinanceExchangePlaceOrderInput,
): Record<string, string | boolean | undefined> {
  const orderType = binanceOrderType(environment, input);
  const clientOrderId = input.clientOrderId ?? createClientOrderId();
  const params: Record<string, string | boolean | undefined> = {
    symbol: (input.symbol ?? symbol).toUpperCase(),
    side: input.side.toUpperCase(),
    type: orderType,
    quantity: decimalParam(input.quantity),
  };

  if (input.type === "stop-market" && environment.product !== "spot") {
    params.algoType = "CONDITIONAL";
    params.triggerPrice = decimalParam(input.stopPrice ?? input.price);
    params.clientAlgoId = clientOrderId;
    if (!params.triggerPrice) {
      throw new Error("Stop-market Binance orders require stopPrice.");
    }
  } else {
    params.newClientOrderId = clientOrderId;
  }

  if (orderType === "LIMIT") {
    params.timeInForce = input.timeInForce ?? "GTC";
    params.price = decimalParam(input.price);
    if (!params.price) {
      throw new Error("Limit Binance orders require price.");
    }
  }

  if (orderType === "STOP_LOSS") {
    params.stopPrice = decimalParam(input.stopPrice ?? input.price);
    if (!params.stopPrice) {
      throw new Error("Stop-market Binance orders require stopPrice.");
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

function binanceOrderType(
  environment: ResolvedExchangeEnvironment,
  input: BinanceExchangePlaceOrderInput,
): string {
  if (input.type === "stop-market") {
    return environment.product === "spot" ? "STOP_LOSS" : "STOP_MARKET";
  }
  return input.type.toUpperCase();
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
  if (clientOrderId.startsWith("bot-")) {
    return clientOrderId;
  }
  if (!clientOrderId.startsWith("bot_")) {
    return undefined;
  }
  const orderId = clientOrderId.slice(4);
  return orderId || undefined;
}

function normalizeSpotBalance(raw: unknown): BinanceExchangeBalance {
  const item = asRecord(raw);
  return {
    asset: stringValue(item.asset),
    free: numberValue(item.free),
    locked: numberValue(item.locked),
  };
}

function normalizeFuturesBalance(raw: unknown): BinanceExchangeBalance {
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

function isUsefulBalance(balance: BinanceExchangeBalance): boolean {
  return (
    Boolean(balance.asset) &&
    (balance.free !== 0 || balance.locked !== 0 || balance.walletBalance !== 0)
  );
}

function extractPositions(
  environment: ResolvedExchangeEnvironment,
  account: Record<string, unknown>,
  symbol: string,
  activeOnly = true,
): BinanceExchangePosition[] {
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
        maxNotional: optionalNumber(item.maxNotional ?? item.maxNotionalValue),
        maxQuantity: optionalNumber(item.maxQty),
        leverage: numberValue(item.leverage),
        marginType: stringValue(item.marginType),
        isolatedMargin: numberValue(item.isolatedMargin),
        updateTime: numberValue(item.updateTime),
      };
    })
    .filter((position) => position.symbol === symbol && (!activeOnly || (
      position.positionAmt !== 0 || position.notional !== 0 || position.unrealizedPnl !== 0
    )));
}

function availableBalanceQuote(
  environment: ResolvedExchangeEnvironment,
  account: Record<string, unknown>,
  market: BinanceMarketListing,
  price: number,
): number {
  const direct = optionalNumber(account.availableBalance);
  if (direct !== undefined) {
    return Math.max(0, environment.product === "coinm-futures" ? direct * price : direct);
  }
  const asset = environment.product === "coinm-futures" ? market.baseAsset : market.quoteAsset;
  const row = (Array.isArray(account.assets) ? account.assets : [])
    .map(asRecord)
    .find((item) => stringValue(item.asset) === asset);
  const available = optionalNumber(row?.availableBalance) ?? 0;
  return Math.max(0, environment.product === "coinm-futures" ? available * price : available);
}

function shouldClosePosition(
  position: BinanceExchangePosition,
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

function normalizeOrder(raw: unknown): BinanceExchangeOrder {
  const item = asRecord(raw);
  const clientOrderId = stringValue(
    item.clientOrderId ??
      item.clientAlgoId ??
      item.newClientStrategyId,
  );
  const stopPrice = optionalNumber(item.stopPrice ?? item.triggerPrice);
  return {
    symbol: stringValue(item.symbol),
    orderId: stringValue(item.orderId ?? item.algoId ?? item.strategyId),
    clientOrderId,
    localOrderId: localOrderIdFromClientOrderId(clientOrderId),
    algo: item.algoId !== undefined || item.clientAlgoId !== undefined || item.strategyId !== undefined,
    side: stringValue(item.side),
    type: stringValue(item.type ?? item.orderType ?? item.strategyType),
    status: stringValue(item.status ?? item.algoStatus ?? item.strategyStatus),
    price: numberValue(item.price) || stopPrice || 0,
    stopPrice,
    originalQuantity: numberValue(item.origQty ?? item.quantity),
    executedQuantity: numberValue(item.executedQty),
    cumulativeQuoteQuantity: numberValue(item.cummulativeQuoteQty ?? item.cumQuote),
    avgPrice: numberValue(item.avgPrice),
    timeInForce: stringValue(item.timeInForce),
    reduceOnly: booleanValue(item.reduceOnly),
    positionSide: stringValue(item.positionSide),
    createdAt: numberValue(item.time ?? item.createTime ?? item.bookTime),
    updatedAt: numberValue(item.updateTime ?? item.transactTime ?? item.createTime ?? item.bookTime),
  };
}

function normalizeTrade(raw: unknown, market: BinanceMarketListing): BinanceExchangeTrade {
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
  product: ResolvedExchangeEnvironment["product"],
): BinanceExchangeCommission {
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

function normalizeSymbolFilters(raw: Record<string, unknown>): BinanceExchangeSymbolFilters {
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

function futuresPositionEffectFromFields(
  side: "buy" | "sell",
  reduceOnly: boolean,
  positionSide: string | undefined,
): "open" | "close" | undefined {
  if (reduceOnly) {
    return "close";
  }

  const normalizedPositionSide = stringValue(positionSide).toUpperCase();
  if (normalizedPositionSide === "LONG") {
    return side === "buy" ? "open" : "close";
  }
  if (normalizedPositionSide === "SHORT") {
    return side === "sell" ? "open" : "close";
  }
  return undefined;
}

function futuresReconciliationFromUserDataEvent(
  market: BinanceMarketListing,
  payload: unknown,
): ExchangeReconciliationInput | undefined {
  const event = asRecord(payload);
  const eventType = stringValue(event.e);
  if (eventType === "ALGO_UPDATE") {
    return futuresAlgoReconciliationFromUserDataEvent(market, event);
  }
  if (eventType !== "ORDER_TRADE_UPDATE") {
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
  const reduceOnly = booleanValue(order.R) === true;
  const positionSide = stringValue(order.ps);
  const positionEffect = futuresPositionEffectFromFields(side, reduceOnly, positionSide);
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

function futuresAlgoReconciliationFromUserDataEvent(
  market: BinanceMarketListing,
  event: Record<string, unknown>,
): ExchangeReconciliationInput | undefined {
  const order = asRecord(event.o);
  if (stringValue(order.s) !== market.symbol) {
    return undefined;
  }

  const clientOrderId = stringValue(order.caid);
  const localOrderId = localOrderIdFromClientOrderId(clientOrderId);
  if (!localOrderId) {
    return undefined;
  }

  const side = normalizeUserDataSide(order.S);
  const reduceOnly = booleanValue(order.R) === true;
  const positionSide = stringValue(order.ps);
  const transactionTime = numberValue(event.T) || numberValue(event.E);
  const avgPrice = numberValue(order.ap);
  const orderPrice = avgPrice || numberValue(order.p) || numberValue(order.tp);
  const filledQuantity = numberValue(order.aq);
  return {
    orders: [{
      localOrderId,
      externalOrderId: stringValue(order.aid) || stringValue(order.ai),
      clientOrderId,
      side,
      type: normalizeUserDataOrderType(order.o),
      status: normalizeUserDataAlgoOrderStatus(order.X, filledQuantity),
      price: orderPrice,
      quantity: numberValue(order.q),
      filledQuantity,
      quoteQuantity: orderPrice > 0 && filledQuantity > 0
        ? orderPrice * filledQuantity
        : undefined,
      createdAt: transactionTime,
      updatedAt: transactionTime,
      positionEffect: futuresPositionEffectFromFields(side, reduceOnly, positionSide),
      reason: `user-data algo ${stringValue(order.X) || "update"}`,
    }],
    fills: [],
  };
}

function normalizeUserDataSide(value: unknown): "buy" | "sell" {
  return stringValue(value).toUpperCase() === "SELL" ? "sell" : "buy";
}

function normalizeUserDataOrderType(value: unknown): "limit" | "market" | "stop-market" {
  const type = stringValue(value).toUpperCase();
  if (type === "MARKET") {
    return "market";
  }
  if (type === "STOP_MARKET" || type === "STOP_LOSS") {
    return "stop-market";
  }
  return "limit";
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

function normalizeUserDataAlgoOrderStatus(
  value: unknown,
  filledQuantity: number,
): "open" | "filled" | "cancelled" {
  const status = stringValue(value).toUpperCase();
  if (status === "FINISHED") {
    return filledQuantity > 0 ? "filled" : "cancelled";
  }
  if (status === "CANCELED" || status === "EXPIRED" || status === "REJECTED") {
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
    throw new Error(`Binance request failed:${code} ${message}`.trim());
  }
  return payload as T;
}

function snapshotKey(environment: ResolvedExchangeEnvironment, symbol: string): string {
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
