import { For, Show, createEffect, createMemo, createSignal, onCleanup, onMount } from "solid-js";
import { createStore, reconcile, unwrap } from "solid-js/store";
import {
  Activity,
  ChevronDown,
  ChevronRight,
  Check,
  MinusCircle,
  Play,
  Plus,
  RefreshCw,
  RotateCcw,
  Save,
  Search,
  Square,
  X,
} from "lucide-solid";
import type {
  BacktestCandleChart,
  BacktestChartAnnotation,
  BacktestChartSmaSeries,
  BacktestExtremumTrace,
  BacktestOraclePoint,
  BacktestOrderTrace,
  BacktestPositionStateTrace,
  BacktestPositionTrace,
  BacktestTrace,
  BacktestTraceFrame,
  BacktestReplayFrame,
  BacktestExtremaOrderMassSideSummary,
  BotEvent,
  Candle,
  LongPositionLot,
  ManualTradeInput,
  PeakValleyBotConfig,
  PeakValleyAverageDiagnostics,
  PeakValleyStrategyDiagnostics,
  PositionLedger,
  ShortPositionLot,
  EquityPoint,
  StrategyConfig,
  TradeFill,
  TradingFill,
  TradingOrder,
  TradingOrderSnapshot,
  TradingPosition,
} from "@trading/bot-algo";
import {
  CandleChart,
  type CandleChartOverlayVisibility,
  type CandleChartPriceDisplay,
  type CandleChartViewport,
} from "./components/CandleChart";
import { EquityChart } from "./components/EquityChart";
import {
  formatAsset,
  formatBytes,
  formatDateTime,
  formatDuration,
  formatPercent,
  formatQuote,
  formatTime,
} from "./format";
import type {
  BacktestProgressSnapshot,
  BacktestSelection,
  BotExecutionMode,
  BinanceExchangeMode,
  BinanceExchangeOrder,
  BinanceExchangeSnapshot,
  BinanceMarketCatalog,
  BinanceMarketListing,
  CorrelationEntry,
  CorrelationSnapshot,
  MarketGroup,
  RuntimeSnapshot,
} from "./types";

const apiBase = "/backend";
const wsUrl = websocketUrl(apiBase, "/ws");
const SOCKET_SNAPSHOT_APPLY_MS = 500;
const RELATIVE_RATE_TO_BPS_HOUR = 36_000_000;
const buttonBaseClass =
  "inline-flex min-h-9 select-none items-center justify-center gap-2 whitespace-nowrap rounded-2 px-3 py-2 text-sm font-semibold transition active:translate-y-px focus-visible:outline-none focus-visible:ring-2 disabled:cursor-not-allowed disabled:opacity-45 disabled:active:translate-y-0";
const buttonPrimaryClass =
  `${buttonBaseClass} border border-accent bg-accent text-ink-950 shadow-[0_10px_24px_rgba(56,189,248,0.18)] hover:bg-accent/88 focus-visible:ring-accent/55`;
const buttonPanelClass =
  `${buttonBaseClass} border border-line bg-ink-800 text-ink-100 hover:bg-ink-700 focus-visible:ring-ink-600/55`;
const buttonDangerClass =
  `${buttonBaseClass} border border-loss/55 bg-loss/14 text-loss hover:border-loss hover:bg-loss/22 focus-visible:ring-loss/40`;

function websocketUrl(basePath: string, socketPath: string): string {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const normalizedBase = basePath.replace(/\/$/, "");
  const normalizedSocketPath = socketPath.startsWith("/") ? socketPath : `/${socketPath}`;

  return `${protocol}//${window.location.host}${normalizedBase}${normalizedSocketPath}`;
}

interface BacktestSettings {
  extremaSmaWindowMinutes: number;
  historicalDays: number;
  randomSampleCount: number;
  randomWindowDays: number;
  randomMinWindowDays: number;
  randomMaxWindowDays: number;
  randomLookbackDays: number;
  randomPairCount: number;
}

type CorrelationSortMode = "abs-desc" | "abs-asc" | "value-desc" | "value-asc";

const defaultBacktestSettings: BacktestSettings = {
  extremaSmaWindowMinutes: 30,
  historicalDays: 30,
  randomSampleCount: 40,
  randomWindowDays: 7,
  randomMinWindowDays: 1,
  randomMaxWindowDays: 30,
  randomLookbackDays: 365,
  randomPairCount: 0,
};

export function App() {
  const [snapshotStore, setSnapshotStore] = createStore<{ snapshot?: RuntimeSnapshot }>({});
  const snapshot = () => snapshotStore.snapshot;
  const [now, setNow] = createSignal(Date.now());
  const [connection, setConnection] = createSignal<"connecting" | "live" | "offline">(
    "connecting",
  );
  const [backtestPreset, setBacktestPreset] =
    createSignal<BacktestSelection>("saved-candles");
  const [backtestSettings, setBacktestSettings] = createSignal<BacktestSettings>({
    ...defaultBacktestSettings,
  });
  const [backtestError, setBacktestError] = createSignal<string>();
  const [configDraft, setConfigDraft] = createSignal<PeakValleyBotConfig>();
  const [configError, setConfigError] = createSignal<string>();
  const [manualTradeError, setManualTradeError] = createSignal<string>();
  const [exchangeError, setExchangeError] = createSignal<string>();
  const [marketCatalog, setMarketCatalog] = createSignal<BinanceMarketCatalog>();
  const [marketError, setMarketError] = createSignal<string>();
  const [switchingMarketId, setSwitchingMarketId] = createSignal<string>();
  const [correlationSortMode, setCorrelationSortMode] =
    createSignal<CorrelationSortMode>("abs-desc");
  const [correlationError, setCorrelationError] = createSignal<string>();
  let socket: WebSocket | undefined;
  let reconnectTimer: number | undefined;
  let runClockTimer: number | undefined;
  let pendingSocketSnapshot: RuntimeSnapshot | undefined;
  let socketSnapshotTimer: number | undefined;
  let disposed = false;
  let requestedCorrelationMarketId: string | undefined;
  let lastSnapshotSource: string | undefined;
  let lastSnapshotSeq = 0;

  const bot = createMemo(() => snapshot()?.bot);
  const market = createMemo(() => snapshot()?.market);
  const metrics = createMemo(() => bot()?.metrics);
  const openOrders = createMemo(() =>
    (bot()?.diagnostics.plannedOrders ?? [])
      .filter((order) => order.status === "pending" || order.status === "open" || order.status === "partially-filled")
      .slice()
      .reverse(),
  );
  const fills = createMemo(() =>
    (snapshot()?.recentEvents ?? [])
      .filter((event): event is typeof event & { fill: TradingFill } => Boolean(event.fill))
      .slice(0, 12),
  );
  const events = createMemo(() => snapshot()?.recentEvents ?? []);
  const backtest = createMemo(() => snapshot()?.backtest);
  const correlations = createMemo(() => snapshot()?.correlations);
  const exchange = createMemo(() => snapshot()?.exchange);
  const hasClosablePositions = createMemo(() => {
    return (bot()?.state.positions ?? []).some((position) => position.asset > 0.00000001);
  });
  const botRunStartedAt = createMemo(() => {
    if (bot()?.status !== "running") {
      return undefined;
    }
    return bot()?.runStartedAt;
  });
  const botRunDurationMs = createMemo(() => {
    const startedAt = botRunStartedAt();
    if (!startedAt) {
      return undefined;
    }
    const elapsed = now() - startedAt;
    return elapsed > 0 ? elapsed : 0;
  });
  const liveEquityCurve = () => snapshot()?.equityCurve ?? [];
  const isBotRunning = () => bot()?.status === "running";

  createEffect(() => {
    const config = bot()?.config;
    if (config && !configDraft()) {
      setConfigDraft(structuredClone(unwrap(config)));
    }
  });

  const applySnapshot = (next: RuntimeSnapshot) => {
    if (next.snapshotSource !== lastSnapshotSource) {
      lastSnapshotSource = next.snapshotSource;
      lastSnapshotSeq = 0;
    }
    if (next.snapshotSeq <= lastSnapshotSeq) {
      return false;
    }

    lastSnapshotSeq = next.snapshotSeq;
    setSnapshotStore("snapshot", reconcile(next, { merge: true }));
    return true;
  };

  const loadInitial = async () => {
    const response = await fetch(`${apiBase}/api/state`);
    if (!response.ok) {
      throw new Error(`State request failed: ${response.status}`);
    }
    applySnapshot((await response.json()) as RuntimeSnapshot);
  };

  const loadMarkets = async (refresh = false) => {
    setMarketError(undefined);
    const response = await fetch(`${apiBase}/api/markets${refresh ? "?refresh=1" : ""}`);
    const payload = await response.json();
    if (!response.ok) {
      setMarketError(payload.error ?? "Market list request failed");
      return;
    }

    setMarketCatalog(payload as BinanceMarketCatalog);
  };

  const flushSocketSnapshot = () => {
    socketSnapshotTimer = undefined;
    const next = pendingSocketSnapshot;
    pendingSocketSnapshot = undefined;
    if (next) {
      applySnapshot(next);
    }
  };

  const queueSocketSnapshot = (next: RuntimeSnapshot) => {
    if (!snapshot()) {
      applySnapshot(next);
      return;
    }

    pendingSocketSnapshot = next;
    if (socketSnapshotTimer !== undefined) {
      return;
    }
    socketSnapshotTimer = window.setTimeout(flushSocketSnapshot, SOCKET_SNAPSHOT_APPLY_MS);
  };

  const connect = () => {
    if (disposed) {
      return;
    }
    setConnection("connecting");
    const nextSocket = new WebSocket(wsUrl);
    socket = nextSocket;

    nextSocket.onopen = () => setConnection("live");
    nextSocket.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data) as {
          type: "snapshot";
          sequence?: number;
          sentAt?: number;
          payload: RuntimeSnapshot;
        };
        if (message.type === "snapshot") {
          queueSocketSnapshot(message.payload);
        }
      } catch {
        nextSocket.close();
      }
    };
    nextSocket.onclose = () => {
      if (socket !== nextSocket || disposed) {
        return;
      }
      setConnection("offline");
      reconnectTimer = window.setTimeout(connect, 1_500);
    };
    nextSocket.onerror = () => {
      nextSocket.close();
    };
  };

  const controlBot = async (action: "start" | "stop" | "reset") => {
    const response = await fetch(`${apiBase}/api/bot/${action}`, {
      method: "POST",
    });
    if (response.ok) {
      applySnapshot((await response.json()) as RuntimeSnapshot);
    }
  };

  const closeBotPositions = async (includeUnprofitable: boolean) => {
    setExchangeError(undefined);
    const response = await fetch(`${apiBase}/api/bot/close-positions`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
      },
      body: JSON.stringify({ includeUnprofitable }),
    });
    const payload = await response.json();
    if (!response.ok) {
      setExchangeError(payload.error ?? "Position close failed");
      return;
    }
    applySnapshot(payload as RuntimeSnapshot);
  };

  const selectMarket = async (marketId: string) => {
    if (!marketId || marketId === market()?.id || switchingMarketId()) {
      return;
    }

    const listing = marketCatalog()?.markets.find((item) => item.id === marketId);
    if (listing && !listing.supportsLiveStream) {
      setMarketError(listing.unavailableReason ?? "This market is not live-streamable yet.");
      return;
    }

    setMarketError(undefined);
    setSwitchingMarketId(marketId);
    const response = await fetch(`${apiBase}/api/market`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
      },
      body: JSON.stringify({ marketId }),
    });
    const payload = await response.json();
    setSwitchingMarketId(undefined);
    if (!response.ok) {
      setMarketError(payload.error ?? "Market switch failed");
      return;
    }

    const nextSnapshot = payload as RuntimeSnapshot;
    if (applySnapshot(nextSnapshot)) {
      setConfigDraft(structuredClone(nextSnapshot.bot.config));
    }
  };

  const runBacktest = async (historicalStartTime?: number) => {
    setBacktestError(undefined);
    const preset = backtestPreset();
    const settings = backtestSettings();
    const body: {
      preset: BacktestSelection;
      limit: number;
      historicalStartTime?: number;
      historicalDays?: number;
      randomSampleCount?: number;
      randomWindowDays?: number;
      randomMinWindowDays?: number;
      randomMaxWindowDays?: number;
      randomLookbackDays?: number;
      randomPairCount?: number;
      extremaSmaWindowMinutes?: number;
    } = {
      preset,
      limit: preset === "saved-orderbook" ? 3_000 : 1_000,
      extremaSmaWindowMinutes: settings.extremaSmaWindowMinutes,
    };

    if (preset === "last-x") {
      body.historicalDays = settings.historicalDays;
    }

    if (preset === "random-windows") {
      body.randomSampleCount = settings.randomSampleCount;
      body.randomWindowDays = settings.randomWindowDays;
      body.randomLookbackDays = settings.randomLookbackDays;
      body.randomPairCount = settings.randomPairCount;
    }

    if (preset === "random-length-windows") {
      body.randomSampleCount = settings.randomSampleCount;
      body.randomMinWindowDays = settings.randomMinWindowDays;
      body.randomMaxWindowDays = settings.randomMaxWindowDays;
      body.randomLookbackDays = settings.randomLookbackDays;
      body.randomPairCount = settings.randomPairCount;
    }
    if (historicalStartTime && historicalStartTime > 0) {
      body.historicalStartTime = historicalStartTime;
    }

    const response = await fetch(`${apiBase}/api/backtest`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
      },
      body: JSON.stringify(body),
    });
    const payload = await response.json();
    if (!response.ok) {
      setBacktestError(payload.error ?? "Backtest failed");
      return;
    }

    applySnapshot(payload as RuntimeSnapshot);
  };

  const stopBacktest = async () => {
    setBacktestError(undefined);
    const response = await fetch(`${apiBase}/api/backtest/stop`, {
      method: "POST",
    });
    const payload = await response.json();
    if (!response.ok) {
      setBacktestError(payload.error ?? "Backtest stop failed");
      return;
    }

    applySnapshot(payload as RuntimeSnapshot);
  };

  const loadCorrelations = async (refresh = false) => {
    setCorrelationError(undefined);
    const response = await fetch(`${apiBase}/api/correlations`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
      },
      body: JSON.stringify({ refresh }),
    });
    const payload = await response.json();
    if (!response.ok) {
      setCorrelationError(payload.error ?? "Correlation request failed");
      return;
    }

    applySnapshot(payload as RuntimeSnapshot);
  };

  createEffect(() => {
    const marketId = market()?.id;
    if (!marketId || requestedCorrelationMarketId === marketId) {
      return;
    }

    requestedCorrelationMarketId = marketId;
    void loadCorrelations();
  });

  const applyConfig = async () => {
    const config = configDraft();
    if (!config) {
      return;
    }

    setConfigError(undefined);
    const response = await fetch(`${apiBase}/api/bot/config`, {
      method: "PUT",
      headers: {
        "content-type": "application/json",
      },
      body: JSON.stringify(config),
    });
    const payload = await response.json();
    if (!response.ok) {
      setConfigError(payload.error ?? "Config update failed");
      return;
    }

    const nextSnapshot = payload as RuntimeSnapshot;
    if (applySnapshot(nextSnapshot)) {
      setConfigDraft(structuredClone(nextSnapshot.bot.config));
    }
  };

  const recordManualTrade = async (input: ManualTradeInput): Promise<boolean> => {
    setManualTradeError(undefined);
    const response = await fetch(`${apiBase}/api/bot/manual-trade`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
      },
      body: JSON.stringify(input),
    });
    const payload = await response.json();
    if (!response.ok) {
      setManualTradeError(payload.error ?? "Manual trade failed");
      return false;
    }

    applySnapshot(payload as RuntimeSnapshot);
    return true;
  };

  const syncExchange = async () => {
    setExchangeError(undefined);
    const response = await fetch(`${apiBase}/api/exchange/sync`, {
      method: "POST",
    });
    const payload = await response.json();
    if (!response.ok) {
      setExchangeError(payload.error ?? "Exchange sync failed");
      return;
    }
    applySnapshot(payload as RuntimeSnapshot);
  };

  const placeExchangeOrder = async (input: {
    side: "buy" | "sell";
    type: "limit" | "market" | "stop-market";
    quantity: number;
    price?: number;
    stopPrice?: number;
    reduceOnly?: boolean;
  }) => {
    setExchangeError(undefined);
    const response = await fetch(`${apiBase}/api/exchange/order`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
      },
      body: JSON.stringify(input),
    });
    const payload = await response.json();
    if (!response.ok) {
      setExchangeError(payload.error ?? "Exchange order failed");
      return;
    }
    applySnapshot(payload as RuntimeSnapshot);
  };

  const cancelExchangeOrder = async (order: BinanceExchangeOrder) => {
    setExchangeError(undefined);
    const response = await fetch(`${apiBase}/api/exchange/order`, {
      method: "DELETE",
      headers: {
        "content-type": "application/json",
      },
      body: JSON.stringify({
        orderId: order.orderId,
        clientOrderId: order.clientOrderId,
        algo: order.algo,
      }),
    });
    const payload = await response.json();
    if (!response.ok) {
      setExchangeError(payload.error ?? "Exchange cancel failed");
      return;
    }
    applySnapshot(payload as RuntimeSnapshot);
  };

  const cancelAllExchangeOrders = async () => {
    setExchangeError(undefined);
    const response = await fetch(`${apiBase}/api/exchange/open-orders`, {
      method: "DELETE",
    });
    const payload = await response.json();
    if (!response.ok) {
      setExchangeError(payload.error ?? "Exchange cancel-all failed");
      return;
    }
    applySnapshot(payload as RuntimeSnapshot);
  };

  const setExchangeLeverage = async (leverage: number) => {
    setExchangeError(undefined);
    const response = await fetch(`${apiBase}/api/exchange/leverage`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
      },
      body: JSON.stringify({ leverage }),
    });
    const payload = await response.json();
    if (!response.ok) {
      setExchangeError(payload.error ?? "Exchange leverage update failed");
      return;
    }
    applySnapshot(payload as RuntimeSnapshot);
  };

  const setBotExecutionMode = async (mode: BotExecutionMode) => {
    setExchangeError(undefined);
    const response = await fetch(`${apiBase}/api/bot/execution`, {
      method: "PUT",
      headers: {
        "content-type": "application/json",
      },
      body: JSON.stringify({ mode }),
    });
    const payload = await response.json();
    if (!response.ok) {
      setExchangeError(payload.error ?? "Execution mode update failed");
      return;
    }
    applySnapshot(payload as RuntimeSnapshot);
  };

  const saveExchangeCredentials = async (input: {
    mode: BinanceExchangeMode;
    apiKey?: string;
    apiSecret?: string;
  }): Promise<boolean> => {
    setExchangeError(undefined);
    const response = await fetch(`${apiBase}/api/exchange/credentials`, {
      method: "PUT",
      headers: {
        "content-type": "application/json",
      },
      body: JSON.stringify(input),
    });
    const payload = await response.json();
    if (!response.ok) {
      setExchangeError(payload.error ?? "Credential update failed");
      return false;
    }
    applySnapshot(payload as RuntimeSnapshot);
    return true;
  };

  const updateBacktestSetting = <K extends keyof BacktestSettings>(
    key: K,
    value: BacktestSettings[K],
  ) => {
    setBacktestSettings((current) => {
      const next = {
        ...current,
        [key]: value,
      };

      if (key === "randomMinWindowDays" && next.randomMaxWindowDays < value) {
        next.randomMaxWindowDays = value;
      }
      if (
        (key === "randomMinWindowDays" || key === "randomMaxWindowDays") &&
        next.randomLookbackDays < next.randomMaxWindowDays
      ) {
        next.randomLookbackDays = next.randomMaxWindowDays;
      }

      return next;
    });
  };

  onMount(() => {
    void loadInitial().catch(() => setConnection("offline"));
    void loadMarkets().catch((error) =>
      setMarketError(error instanceof Error ? error.message : "Market list request failed"),
    );
    connect();
    runClockTimer = window.setInterval(() => setNow(Date.now()), 1_000);
  });

  onCleanup(() => {
    disposed = true;
    socket?.close();
    if (reconnectTimer) {
      window.clearTimeout(reconnectTimer);
    }
    if (runClockTimer) {
      window.clearInterval(runClockTimer);
    }
    if (socketSnapshotTimer) {
      window.clearTimeout(socketSnapshotTimer);
    }
  });

  return (
    <main class="min-h-screen bg-ink-950 text-ink-100">
      <div class="mx-auto flex w-full max-w-7xl flex-col gap-4 px-4 py-4 lg:px-6">
        <header class="flex flex-col gap-3 border-b border-line pb-4 lg:flex-row lg:items-center lg:justify-between">
          <div class="flex min-w-0 flex-col gap-3">
            <AssetSelector
              catalog={marketCatalog()}
              selectedMarketId={market()?.id}
              selectedSymbol={market()?.symbol ?? "BTCUSDT"}
              disabled={Boolean(switchingMarketId())}
              error={marketError()}
              onSelect={(marketId) => void selectMarket(marketId)}
              onRefresh={() => void loadMarkets(true)}
            />
            <div class="flex flex-wrap items-center gap-2">
              <StatusPill label={market()?.venue ?? "spot"} active />
              <StatusPill label={connection()} active={connection() === "live"} />
              <StatusPill
                label={market()?.connected ? "Binance live" : "Binance offline"}
                active={Boolean(market()?.connected)}
              />
              <StatusPill
                label={exchangeStatusLabel(exchange())}
                active={Boolean(exchange()?.connected)}
              />
              <StatusPill
                label={snapshot()?.execution?.exchangeDriven ? "Binance execution" : "Simulated execution"}
                active={Boolean(snapshot()?.execution?.exchangeDriven)}
              />
              <StatusPill label={bot()?.status ?? "starting"} active={bot()?.status === "running"} />
            </div>
            <Show when={isBotRunning()} fallback={<div class="text-sm text-ink-400">Bot stopped</div>}>
              <div class="text-sm text-ink-300">
                Start: {formatDateTime(botRunStartedAt())}
                <span class="mx-2">·</span>
                Duration: {formatDuration(botRunDurationMs())}
              </div>
            </Show>
          </div>

          <div class="flex flex-wrap items-center gap-2">
            <a class={buttonPanelClass} href="#/kama-inspector">
              <Search size={16} />
              KAMA Inspector
            </a>
            <button class={buttonPrimaryClass} type="button" onClick={() => void controlBot("start")}>
              <Play size={16} />
              Start
            </button>
            <button class={buttonDangerClass} type="button" onClick={() => void controlBot("stop")}>
              <Square size={16} />
              Stop
            </button>
            <button class="btn" type="button" onClick={() => void controlBot("reset")}>
              <RotateCcw size={16} />
              Reset
            </button>
            <button
              class="btn"
              type="button"
              disabled={!hasClosablePositions()}
              onClick={() => void closeBotPositions(false)}
            >
              <MinusCircle size={16} />
              Close Profitable + Pause
            </button>
            <button
              class={buttonDangerClass}
              type="button"
              disabled={!hasClosablePositions()}
              onClick={() => void closeBotPositions(true)}
            >
              <MinusCircle size={16} />
              Force Close All
            </button>
          </div>
        </header>

        <section class="grid grid-cols-2 gap-3 lg:grid-cols-6">
          <MetricCard
            label="Last Price"
            value={`${formatQuote(market()?.lastPrice, 2)} ${market()?.quoteAsset ?? "USDT"}`}
          />
          <MetricCard label="Equity" value={`$${formatQuote(metrics()?.equity, 2)}`} />
          <MetricCard
            label="Return"
            value={formatPercent(metrics()?.returnPct)}
            tone={(metrics()?.returnPct ?? 0) >= 0 ? "gain" : "loss"}
          />
          <MetricCard label="Quote Free" value={`$${formatQuote(bot()?.equity.quoteAvailable, 2)}`} />
          <MetricCard
            label={`${market()?.baseAsset ?? "Base"} Free`}
            value={formatAsset(bot()?.equity.assetAvailable)}
          />
          <MetricCard label="Open Orders" value={openOrders().length.toString()} />
        </section>

        <LiveEquityPanel points={liveEquityCurve()} />

        <AlgorithmPanel
          config={configDraft()}
          marketMaxLeverage={market()?.maxLeverage}
          error={configError()}
          onChange={setConfigDraft}
          onApply={() => void applyConfig()}
        />

        <ExchangePaperPanel
          market={market()}
          snapshot={exchange()}
          execution={snapshot()?.execution}
          botRunning={isBotRunning()}
          error={exchangeError()}
          onSetExecutionMode={(mode) => void setBotExecutionMode(mode)}
          onSaveCredentials={saveExchangeCredentials}
          onSync={() => void syncExchange()}
          onPlaceOrder={(input) => void placeExchangeOrder(input)}
          onCancelOrder={(order) => void cancelExchangeOrder(order)}
          onCancelAll={() => void cancelAllExchangeOrders()}
          onSetLeverage={(leverage) => void setExchangeLeverage(leverage)}
        />

        <section class="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1fr)_360px]">
          <div class="panel min-h-105">
            <div class="mb-3 flex items-center justify-between">
              <div>
                <div class="muted-label">Live Candles</div>
                <h2 class="text-lg font-semibold">{market()?.interval ?? "1m"} chart</h2>
              </div>
              <div class="text-right text-sm text-ink-300">
                Updated {formatTime(market()?.lastEventAt)}
              </div>
            </div>
            <div class="h-88 lg:h-100">
              <CandleChart
                candles={market()?.candles ?? []}
                orders={bot()?.diagnostics.plannedOrders ?? []}
                lastPrice={market()?.lastPrice ?? 0}
              />
            </div>
          </div>

          <div class="grid grid-cols-1 gap-4">
            <PerformancePanel snapshot={snapshot()} />
            <StrategyStatePanel bot={bot()} />
            <OrderBookPanel snapshot={snapshot()} />
          </div>
        </section>

        <CorrelationPanel
          snapshot={correlations()}
          sortMode={correlationSortMode()}
          error={correlationError()}
          onSortChange={setCorrelationSortMode}
          onRefresh={(refresh) => void loadCorrelations(refresh)}
        />

        <section class="grid min-w-0 grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1fr)_420px]">
          <div class="grid min-w-0 grid-cols-1 gap-4 lg:grid-cols-2">
            <OrdersPanel title="Open Orders" orders={openOrders()} />
            <FillsPanel fills={fills()} />
          </div>
          <EventsPanel events={events()} />
        </section>

        <TradingPositionsPanel
          positions={bot()?.state.positions ?? []}
          baseAsset={market()?.baseAsset ?? "Base"}
          quoteAsset={market()?.quoteAsset ?? "USDT"}
          error={manualTradeError()}
          onRecordTrade={recordManualTrade}
        />

        <BacktestPanel
          preset={backtestPreset()}
          onPresetChange={setBacktestPreset}
          settings={backtestSettings()}
          onSettingChange={updateBacktestSetting}
          progress={backtest()}
          error={backtestError()}
          liveStartAt={botRunStartedAt()}
          onRun={() => void runBacktest()}
          onRunFromLiveStart={() => void runBacktest(botRunStartedAt())}
          onStop={() => void stopBacktest()}
        />
      </div>
    </main>
  );
}

function MetricCard(props: {
  label: string;
  value: string;
  tone?: "gain" | "loss" | "neutral";
}) {
  return (
    <div class="panel-tight">
      <div class="muted-label">{props.label}</div>
      <div
        class="metric-value mt-1 truncate"
        classList={{
          "text-gain": props.tone === "gain",
          "text-loss": props.tone === "loss",
        }}
      >
        {props.value}
      </div>
    </div>
  );
}

function StatusPill(props: { label: string; active: boolean }) {
  return (
    <span
      class="inline-flex items-center gap-2 rounded-2 border px-2.5 py-1 text-xs font-medium"
      classList={{
        "border-gain/50 bg-gain/12 text-gain": props.active,
        "border-line bg-ink-800 text-ink-300": !props.active,
      }}
    >
      <span
        class="h-2 w-2 rounded-full"
        classList={{
          "bg-gain": props.active,
          "bg-ink-600": !props.active,
        }}
      />
      {props.label}
    </span>
  );
}

function ExchangeRule(props: { label: string; value: string }) {
  return (
    <div class="rounded-2 bg-ink-800 p-3">
      <div class="muted-label">{props.label}</div>
      <div class="mt-1 truncate text-sm font-semibold tabular-nums text-ink-100">
        {props.value}
      </div>
    </div>
  );
}

function exchangeStatusLabel(exchange: BinanceExchangeSnapshot | undefined): string {
  if (!exchange?.enabled) {
    return "Exchange off";
  }
  if (!exchange.configured) {
    return "Exchange keys missing";
  }
  if (!exchange.compatible) {
    return "Exchange incompatible";
  }
  const prefix = exchange.live ? "Live" : "Sandbox";
  if (exchange.userDataStreamConnected) {
    return `${prefix} streaming`;
  }
  return exchange.connected
    ? `${prefix} synced`
    : `${prefix} ready`;
}

function ExchangePaperPanel(props: {
  market?: RuntimeSnapshot["market"];
  snapshot?: BinanceExchangeSnapshot;
  execution?: RuntimeSnapshot["execution"];
  botRunning: boolean;
  error?: string;
  onSetExecutionMode: (mode: BotExecutionMode) => void;
  onSaveCredentials: (input: {
    mode: BinanceExchangeMode;
    apiKey?: string;
    apiSecret?: string;
  }) => Promise<boolean>;
  onSync: () => void;
  onPlaceOrder: (input: {
    side: "buy" | "sell";
    type: "limit" | "market" | "stop-market";
    quantity: number;
    price?: number;
    stopPrice?: number;
    reduceOnly?: boolean;
  }) => void;
  onCancelOrder: (order: BinanceExchangeOrder) => void;
  onCancelAll: () => void;
  onSetLeverage: (leverage: number) => void;
}) {
  const [side, setSide] = createSignal<"buy" | "sell">("buy");
  const [type, setType] = createSignal<"limit" | "market" | "stop-market">("limit");
  const [quantity, setQuantity] = createSignal(0.001);
  const [price, setPrice] = createSignal(0);
  const [reduceOnly, setReduceOnly] = createSignal(false);
  const [leverage, setLeverage] = createSignal(1);
  const [credentialMode, setCredentialMode] = createSignal<BinanceExchangeMode>("live");
  const [credentialApiKey, setCredentialApiKey] = createSignal("");
  const [credentialApiSecret, setCredentialApiSecret] = createSignal("");
  let initializedPrice = false;
  let observedExchangeMode: string | undefined;
  const exchange = () => props.snapshot;
  const execution = () => props.execution;

  createEffect(() => {
    const lastPrice = props.market?.lastPrice;
    if (!initializedPrice && lastPrice && lastPrice > 0) {
      initializedPrice = true;
      setPrice(Number(lastPrice.toFixed(2)));
    }
  });

  createEffect(() => {
    const mode = exchange()?.mode;
    if (mode && mode !== observedExchangeMode) {
      observedExchangeMode = mode;
      setCredentialMode(normalizeBinanceCredentialMode(mode));
    }
  });

  const executionMode = () => execution()?.mode ?? "simulated";
  const canChangeExecution = () => !props.botRunning;
  const canTrade = () =>
    Boolean(exchange()?.enabled && exchange()?.configured && exchange()?.compatible);
  const isFutures = () =>
    exchange()?.resolvedMode === "usdm-futures-testnet" ||
    exchange()?.resolvedMode === "coinm-futures-testnet" ||
    exchange()?.resolvedMode === "usdm-futures-live" ||
    exchange()?.resolvedMode === "coinm-futures-live";
  const quoteAsset = () => props.market?.quoteAsset ?? "USDT";

  const submitOrder = () => {
    props.onPlaceOrder({
      side: side(),
      type: type(),
      quantity: quantity(),
      price: type() === "limit" ? price() : undefined,
      stopPrice: type() === "stop-market" ? price() : undefined,
      reduceOnly: reduceOnly(),
    });
  };

  const saveCredentials = async () => {
    const saved = await props.onSaveCredentials({
      mode: credentialMode(),
      apiKey: credentialApiKey().trim() || undefined,
      apiSecret: credentialApiSecret().trim() || undefined,
    });
    if (saved) {
      setCredentialApiKey("");
      setCredentialApiSecret("");
    }
  };

  return (
    <section class="panel min-w-0 overflow-hidden">
      <div class="mb-4 flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <div class="muted-label">Binance Exchange</div>
          <h2 class="text-lg font-semibold">
            {exchange()?.resolvedMode ?? exchange()?.mode ?? "disabled"}
          </h2>
          <div class="mt-1 text-xs text-ink-300">
            {exchange()?.userDataStreamMessage ?? exchange()?.message}
          </div>
        </div>
        <div class="flex flex-wrap gap-2">
          <button class="btn" type="button" onClick={props.onSync} disabled={!canTrade()}>
            <RefreshCw size={16} />
            Sync
          </button>
          <Show when={isFutures()}>
            <button
              class="btn"
              type="button"
              onClick={() => props.onSetLeverage(leverage())}
              disabled={!canTrade()}
            >
              <Activity size={16} />
              {formatLeverage(leverage())}
            </button>
          </Show>
        </div>
      </div>

      <Show when={props.error ?? exchange()?.error}>
        {(message) => (
          <div class="mb-3 rounded-2 bg-loss/12 p-3 text-sm text-loss">{message()}</div>
        )}
      </Show>

      <div class="mb-4 rounded-2 bg-ink-800 p-3">
        <div class="mb-3 flex flex-col gap-1">
          <div class="muted-label">Binance Tokens</div>
          <div class="text-xs text-ink-300">
            {exchange()?.configured ? "Credentials saved for this endpoint" : "Credentials missing"}
          </div>
        </div>
        <div class="grid grid-cols-1 gap-3 lg:grid-cols-[180px_minmax(0,1fr)_minmax(0,1fr)_auto] lg:items-end">
          <SelectField
            label="Endpoint"
            value={credentialMode()}
            options={[
              { value: "live", label: "Live" },
              { value: "auto", label: "Sandbox" },
            ]}
            onInput={(value) => setCredentialMode(normalizeBinanceCredentialMode(value))}
          />
          <TextField
            label="API Key"
            value={credentialApiKey()}
            placeholder={exchange()?.configured ? "Saved" : "Required"}
            autocomplete="off"
            onInput={setCredentialApiKey}
          />
          <TextField
            label="API Secret"
            value={credentialApiSecret()}
            placeholder={exchange()?.configured ? "Saved" : "Required"}
            type="password"
            autocomplete="new-password"
            onInput={setCredentialApiSecret}
          />
          <button
            class={buttonPrimaryClass}
            type="button"
            disabled={props.botRunning}
            onClick={() => void saveCredentials()}
          >
            <Save size={16} />
            Save
          </button>
        </div>
      </div>

      <div class="mb-4 rounded-2 bg-ink-800 p-3">
        <div class="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <div class="muted-label">Bot Execution</div>
            <div class="mt-1 text-sm font-semibold text-ink-100">
              {executionMode() === "binance" ? "Binance account" : "Local simulator"}
            </div>
            <div class="mt-1 text-xs text-ink-300">
              {execution()?.message ?? "Bot orders execute in the local simulator."}
            </div>
          </div>
          <div class="grid grid-cols-2 gap-2 sm:w-80">
            <button
              class={executionMode() === "simulated" ? buttonPrimaryClass : buttonPanelClass}
              type="button"
              disabled={!canChangeExecution() || executionMode() === "simulated"}
              onClick={() => props.onSetExecutionMode("simulated")}
            >
              <Check size={16} />
              Simulated
            </button>
            <button
              class={executionMode() === "binance" ? buttonDangerClass : buttonPanelClass}
              type="button"
              disabled={
                !canChangeExecution() ||
                executionMode() === "binance" ||
                !execution()?.canUseExchange
              }
              onClick={() => props.onSetExecutionMode("binance")}
            >
              <Activity size={16} />
              Binance
            </button>
          </div>
        </div>
      </div>

      <div class="mb-4 grid grid-cols-2 gap-2 lg:grid-cols-6">
        <ExchangeRule
          label="User Stream"
          value={exchange()?.userDataStreamConnected ? "Connected" : "Offline"}
        />
        <ExchangeRule label="Taker Fee" value={formatBps(exchange()?.feeBps)} />
        <ExchangeRule
          label="Spread Slip"
          value={formatBps(exchange()?.estimatedSlippageBps)}
        />
        <ExchangeRule
          label="Min Notional"
          value={formatOptionalQuote(exchange()?.symbolFilters?.minNotional, quoteAsset())}
        />
        <ExchangeRule
          label="Max Notional"
          value={formatOptionalQuote(exchange()?.symbolFilters?.maxNotional, quoteAsset())}
        />
        <ExchangeRule
          label="Qty Step"
          value={formatOptionalAsset(exchange()?.symbolFilters?.stepSize)}
        />
      </div>

      <div class="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(280px,380px)]">
        <div class="grid grid-cols-1 gap-4 lg:grid-cols-2">
          <div class="rounded-2 bg-ink-800 p-3">
            <div class="muted-label">Balances</div>
            <div class="mt-3 space-y-2">
              <Show
                when={(exchange()?.balances.length ?? 0) > 0}
                fallback={<div class="text-sm text-ink-400">No synced balances</div>}
              >
                <For each={exchange()?.balances.slice(0, 8) ?? []}>
                  {(balance) => (
                    <div class="flex items-center justify-between gap-3 rounded-2 bg-ink-900 px-3 py-2 text-sm">
                      <span class="font-semibold text-ink-100">{balance.asset}</span>
                      <span class="text-right tabular-nums text-ink-200">
                        {formatAsset(balance.availableBalance ?? balance.free)}
                      </span>
                    </div>
                  )}
                </For>
              </Show>
            </div>
          </div>

          <div class="rounded-2 bg-ink-800 p-3">
            <div class="muted-label">Futures Positions</div>
            <div class="mt-3 space-y-2">
              <Show
                when={(exchange()?.positions.length ?? 0) > 0}
                fallback={<div class="text-sm text-ink-400">No synced positions</div>}
              >
                <For each={exchange()?.positions ?? []}>
                  {(position) => (
                    <div class="rounded-2 bg-ink-900 px-3 py-2 text-sm">
                      <div class="flex items-center justify-between gap-3">
                        <span class="font-semibold text-ink-100">
                          {position.symbol} {position.positionSide}
                        </span>
                        <span
                          class="tabular-nums"
                          classList={{
                            "text-gain": position.positionAmt > 0,
                            "text-loss": position.positionAmt < 0,
                          }}
                        >
                          {formatAsset(position.positionAmt)}
                        </span>
                      </div>
                      <div class="mt-1 flex justify-between gap-3 text-xs text-ink-400">
                        <span>Entry {formatQuote(position.entryPrice, 2)}</span>
                        <span>PnL {formatQuote(position.unrealizedPnl, 2)} {quoteAsset()}</span>
                      </div>
                    </div>
                  )}
                </For>
              </Show>
            </div>
          </div>
        </div>

        <div class="rounded-2 bg-ink-800 p-3">
          <div class="muted-label">Place Exchange Order</div>
          <div class="mt-3 grid grid-cols-2 gap-3">
            <SelectField
              label="Side"
              value={side()}
              options={[
                { value: "buy", label: "Buy" },
                { value: "sell", label: "Sell" },
              ]}
              onInput={(value) => setSide(value === "sell" ? "sell" : "buy")}
            />
            <SelectField
              label="Type"
              value={type()}
              options={[
                { value: "limit", label: "Limit" },
                { value: "market", label: "Market" },
                { value: "stop-market", label: "Stop Market" },
              ]}
              onInput={(value) =>
                setType(
                  value === "market"
                    ? "market"
                    : value === "stop-market"
                      ? "stop-market"
                      : "limit",
                )
              }
            />
            <NumberField
              label="Quantity"
              value={quantity()}
              min={0}
              step={0.000001}
              onInput={setQuantity}
            />
            <Show when={type() === "limit" || type() === "stop-market"}>
              <NumberField
                label={`${type() === "stop-market" ? "Stop" : "Price"} ${quoteAsset()}`}
                value={price()}
                min={0}
                step={0.01}
                onInput={setPrice}
              />
            </Show>
            <Show when={isFutures()}>
              <NumberField
                label="Leverage"
                value={leverage()}
                min={1}
                max={125}
                step={1}
                onInput={(value) => setLeverage(Math.round(value))}
              />
              <BooleanField
                label="Reduce Only"
                checked={reduceOnly()}
                onInput={setReduceOnly}
              />
            </Show>
          </div>
          <button
            class={`${buttonPrimaryClass} mt-3 w-full`}
            type="button"
            disabled={!canTrade()}
            onClick={submitOrder}
          >
            <Plus size={16} />
            Submit Exchange Order
          </button>
        </div>
      </div>

      <div class="mt-4 min-w-0 overflow-x-auto">
        <div class="mb-2 flex items-center justify-between gap-3">
          <div class="muted-label">Exchange Open Orders</div>
          <button
            class="btn px-2 py-1 text-xs"
            type="button"
            disabled={!canTrade() || (exchange()?.openOrders.length ?? 0) === 0}
            onClick={props.onCancelAll}
          >
            Cancel All
          </button>
        </div>
        <table class="w-full min-w-180 text-left text-sm">
          <thead class="text-xs uppercase text-ink-400">
            <tr>
              <th class="py-2 pr-3">Order</th>
              <th class="py-2 pr-3">Side</th>
              <th class="py-2 pr-3">Type</th>
              <th class="py-2 pr-3 text-right">Price</th>
              <th class="py-2 pr-3 text-right">Qty</th>
              <th class="py-2 pr-3">Updated</th>
              <th class="py-2 text-right">Action</th>
            </tr>
          </thead>
          <tbody>
            <For each={exchange()?.openOrders ?? []}>
              {(order) => (
                <tr class="border-t border-line">
                  <td class="py-2 pr-3 font-mono text-xs text-ink-300">{order.orderId}</td>
                  <td
                    class="py-2 pr-3 font-semibold"
                    classList={{
                      "text-gain": order.side === "BUY",
                      "text-loss": order.side === "SELL",
                    }}
                  >
                    {order.side}
                  </td>
                  <td class="py-2 pr-3 text-ink-300">{order.type}</td>
                  <td class="py-2 pr-3 text-right tabular-nums">
                    {formatQuote(order.price, 2)}
                  </td>
                  <td class="py-2 pr-3 text-right tabular-nums">
                    {formatAsset(order.originalQuantity)}
                  </td>
                  <td class="py-2 pr-3 text-ink-400">
                    {formatDateTime(order.updatedAt || order.createdAt)}
                  </td>
                  <td class="py-2 text-right">
                    <button
                      class="btn px-2 py-1 text-xs"
                      type="button"
                      onClick={() => props.onCancelOrder(order)}
                    >
                      <X size={14} />
                    </button>
                  </td>
                </tr>
              )}
            </For>
          </tbody>
        </table>
        <Show when={(exchange()?.openOrders.length ?? 0) === 0}>
          <div class="border-t border-line py-3 text-sm text-ink-400">
            No synced exchange open orders
          </div>
        </Show>
      </div>
    </section>
  );
}

const marketGroupFilters: Array<MarketGroup | "all"> = [
  "all",
  "spot",
  "bstocks",
  "futures",
  "tradfi",
  "options",
  "predictions",
];

function AssetSelector(props: {
  catalog?: BinanceMarketCatalog;
  selectedMarketId?: string;
  selectedSymbol: string;
  disabled?: boolean;
  error?: string;
  onSelect: (marketId: string) => void;
  onRefresh: () => void;
}) {
  const [query, setQuery] = createSignal("");
  const [group, setGroup] = createSignal<MarketGroup | "all">("all");
  const selected = createMemo(() =>
    props.catalog?.markets.find((market) => market.id === props.selectedMarketId),
  );
  const filteredMarkets = createMemo(() => {
    const normalizedQuery = query().trim().toLowerCase();
    const selectedMarket = selected();
    const filtered =
      props.catalog?.markets.filter((market) => {
        if (group() !== "all" && market.group !== group()) {
          return false;
        }
        if (normalizedQuery && !market.searchable.includes(normalizedQuery)) {
          return false;
        }
        return true;
      }) ?? [];

    if (selectedMarket && !filtered.some((market) => market.id === selectedMarket.id)) {
      return [selectedMarket, ...filtered];
    }

    return filtered;
  });
  const countForGroup = (value: MarketGroup | "all") =>
    value === "all"
      ? props.catalog?.markets.length ?? 0
      : props.catalog?.counts[value] ?? 0;

  return (
    <div class="min-w-0 lg:min-w-150">
      <div class="flex items-center justify-between gap-3">
        <div class="min-w-0">
          <div class="muted-label">Asset</div>
          <h1 class="truncate text-2xl font-semibold tabular-nums">
            {selected()?.displaySymbol ?? props.selectedSymbol}
          </h1>
        </div>
        <button
          class="btn px-2.5"
          onClick={props.onRefresh}
          disabled={props.disabled}
          type="button"
        >
          <RefreshCw size={16} />
        </button>
      </div>

      <div class="mt-2 flex flex-wrap gap-1.5">
        <For each={marketGroupFilters}>
          {(item) => (
            <button
              class="rounded-2 border px-2.5 py-1 text-xs uppercase tracking-wide transition"
              classList={{
                "border-accent bg-accent/18 text-ink-100": group() === item,
                "border-line bg-ink-800 text-ink-300 hover:border-accent": group() !== item,
              }}
              onClick={() => setGroup(item)}
              type="button"
            >
              {marketGroupLabel(item)} {countForGroup(item)}
            </button>
          )}
        </For>
      </div>

      <div class="mt-2 grid grid-cols-1 gap-2 md:grid-cols-[minmax(0,240px)_minmax(0,1fr)]">
        <label class="relative min-w-0">
          <Search size={15} class="pointer-events-none absolute left-2.5 top-2.5 text-ink-300" />
          <input
            class="w-full rounded-2 border border-line bg-ink-800 py-2 pl-8 pr-2 text-sm text-ink-100"
            aria-label="Search markets"
            value={query()}
            placeholder="Search BTC, TSLA, XAU..."
            onInput={(event) => setQuery(event.currentTarget.value)}
          />
        </label>
        <select
          class="w-full rounded-2 border border-line bg-ink-800 px-3 py-2 text-sm text-ink-100 disabled:opacity-60"
          value={props.selectedMarketId ?? ""}
          disabled={props.disabled || !props.catalog}
          onInput={(event) => props.onSelect(event.currentTarget.value)}
        >
          <Show when={props.catalog} fallback={<option value="">Loading Binance markets...</option>}>
            <For each={filteredMarkets()}>
              {(market) => (
                <option
                  value={market.id}
                  disabled={!market.supportsLiveStream}
                  title={market.unavailableReason}
                >
                  {marketOptionLabel(market)}
                </option>
              )}
            </For>
          </Show>
        </select>
      </div>

      <Show when={props.error ?? firstCatalogWarning(props.catalog)}>
        {(message) => <div class="mt-2 text-xs text-warn">{message()}</div>}
      </Show>
    </div>
  );
}

function firstCatalogWarning(catalog: BinanceMarketCatalog | undefined): string | undefined {
  return catalog?.warnings[0];
}

function marketGroupLabel(group: MarketGroup | "all"): string {
  if (group === "all") {
    return "All";
  }
  if (group === "tradfi") {
    return "TradFi";
  }
  if (group === "bstocks") {
    return "bStocks";
  }

  return group[0].toUpperCase() + group.slice(1);
}

function marketOptionLabel(market: BinanceMarketListing): string {
  const unavailable = market.supportsLiveStream ? "" : " unavailable";
  const contract = market.contractType ? ` ${market.contractType.replace("_", " ")}` : "";
  const maxLeverage = market.maxLeverage ? ` | max ${formatLeverage(market.maxLeverage)}` : "";
  return `${market.displaySymbol} | ${venueLabel(market.venue)} | ${market.symbol}${contract}${maxLeverage}${unavailable}`;
}

function venueLabel(venue: BinanceMarketListing["venue"]): string {
  if (venue === "usdm-futures") {
    return "USD-M";
  }
  if (venue === "coinm-futures") {
    return "COIN-M";
  }
  if (venue === "predictions") {
    return "Prediction";
  }

  return venue[0].toUpperCase() + venue.slice(1);
}

function AlgorithmPanel(props: {
  config?: PeakValleyBotConfig;
  marketMaxLeverage?: number;
  error?: string;
  onChange: (config: PeakValleyBotConfig) => void;
  onApply: () => void;
}) {
  const marketMaxLeverage = () =>
    Number.isFinite(props.marketMaxLeverage) && (props.marketMaxLeverage as number) >= 1
      ? props.marketMaxLeverage
      : undefined;
  const update = <K extends keyof PeakValleyBotConfig>(
    key: K,
    value: PeakValleyBotConfig[K],
  ) => {
    if (!props.config) {
      return;
    }

    props.onChange({
      ...props.config,
      [key]: value,
    });
  };
  const updateValleyPeak = <K extends keyof PeakValleyBotConfig["strategy"]>(
    key: K,
    value: PeakValleyBotConfig["strategy"][K],
  ) => {
    if (!props.config) {
      return;
    }

    props.onChange({
      ...props.config,
      strategy: {
        ...props.config.strategy,
        [key]: value,
      },
    });
  };
  const updateValleyPeakOffsets = (
    key:
      | "buyConfirmationOffsets"
      | "sellConfirmationOffsets"
      | "buyExitConfirmationOffsets"
      | "sellExitConfirmationOffsets",
    value: number[],
  ) => updateValleyPeak(key, value);
  const updateGrid = (
    key: "entryGrid" | "exitGrid",
    patch: Partial<PeakValleyBotConfig[typeof key]>,
  ) => {
    if (props.config) {
      props.onChange({
        ...props.config,
        [key]: { ...props.config[key], ...patch },
      });
    }
  };
  const updateBorrow = (patch: Partial<PeakValleyBotConfig["internalBorrow"]>) => {
    if (props.config) {
      props.onChange({
        ...props.config,
        internalBorrow: { ...props.config.internalBorrow, ...patch },
      });
    }
  };
  const signalTimingOptions = [
    { value: "start", label: "Start" },
    { value: "end", label: "End" },
  ];

  return (
    <section class="panel">
      <div class="mb-4 flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <div class="muted-label">Algorithm</div>
          <h2 class="text-lg font-semibold">
            Peak/Valley
          </h2>
        </div>
        <button
          class={buttonPrimaryClass}
          disabled={!props.config}
          onClick={props.onApply}
          type="button"
        >
          <Save size={16} />
          Apply
        </button>
      </div>

      <Show when={props.error}>
        {(message) => (
          <div class="mb-3 rounded-2 bg-loss/12 p-3 text-sm text-loss">{message()}</div>
        )}
      </Show>

      <Show when={props.config}>
        {(config) => (
          <div class="grid grid-cols-1 gap-4 xl:grid-cols-3">
            <div class="rounded-2 bg-ink-800 p-3">
              <div class="muted-label">Strategy</div>
              <div class="mt-2 rounded-2 bg-ink-900 px-3 py-2 text-sm font-semibold text-ink-100">
                Peak/Valley
              </div>
              <div class="mt-3 grid grid-cols-2 gap-3">
                <NumberField
                  label="Max Trade"
                  value={config().maxTradeQuote}
                  min={1}
                  onInput={(value) => update("maxTradeQuote", value)}
                />
                <NumberField
                  label="Max Lev"
                  value={config().maxTargetLeverage}
                  min={1}
                  max={marketMaxLeverage()}
                  step={0.25}
                  onInput={(value) =>
                    update("maxTargetLeverage", clampNumber(value, 1, marketMaxLeverage() ?? 999))
                  }
                />
                <NumberField
                  label="Min Trade"
                  value={config().minTradeQuote}
                  min={0}
                  onInput={(value) => update("minTradeQuote", value)}
                />
                <Show when={marketMaxLeverage()}>
                  {(value) => (
                    <div class="rounded-2 bg-ink-900 px-2 py-2">
                      <div class="muted-label">Exchange Max</div>
                      <div class="mt-1 text-sm font-semibold tabular-nums text-ink-100">
                        {formatLeverage(value())}
                      </div>
                    </div>
                  )}
                </Show>
                <NumberField
                  label="Cooldown sec"
                  value={config().cooldownMs / 1000}
                  min={0}
                  onInput={(value) => update("cooldownMs", value * 1000)}
                />
                <NumberField
                  label="Profit Share"
                  value={config().internalBorrow.borrowerProfitShare}
                  min={0}
                  max={1}
                  step={0.05}
                  onInput={(value) =>
                    updateBorrow({ borrowerProfitShare: clampNumber(value, 0, 1) })
                  }
                />
                <BooleanField
                  label="Internal Borrow"
                  checked={config().internalBorrow.enabled}
                  onInput={(enabled) => updateBorrow({ enabled })}
                />
                <BooleanField
                  label="Lock Lenders"
                  checked={config().internalBorrow.lockLenderAmounts}
                  onInput={(lockLenderAmounts) => updateBorrow({ lockLenderAmounts })}
                />
              </div>
            </div>

            <div class="rounded-2 bg-ink-800 p-3">
              <div class="muted-label">Peak/Valley Indicators</div>
              <div class="mt-3 grid grid-cols-2 gap-3">
                <BooleanField
                  label="Relative Rates"
                  checked={config().strategy.relativeRateEnabled}
                  onInput={(value) => updateValleyPeak("relativeRateEnabled", value)}
                />
                <SelectField
                  label="KAMA Rate"
                  value={config().strategy.kamaRateMode}
                  options={[
                    { value: "relative", label: "Relative derivative" },
                    { value: "log", label: "Log-relative rate" },
                  ]}
                  onInput={(value) =>
                    updateValleyPeak(
                      "kamaRateMode",
                      value as PeakValleyBotConfig["strategy"]["kamaRateMode"],
                    )
                  }
                />
                <SelectField
                  label="Average Type"
                  value={config().strategy.movingAverageType}
                  options={[
                    { value: "sma", label: "SMA" },
                    { value: "ema", label: "EMA" },
                  ]}
                  onInput={(value) =>
                    updateValleyPeak(
                      "movingAverageType",
                      value as PeakValleyBotConfig["strategy"]["movingAverageType"],
                    )
                  }
                />
                <SelectField
                  label="Derivative Source"
                  value={config().strategy.derivativeSource}
                  options={[
                    { value: "price", label: "Price" },
                    { value: "kama", label: "KAMA" },
                  ]}
                  onInput={(value) =>
                    updateValleyPeak(
                      "derivativeSource",
                      value as PeakValleyBotConfig["strategy"]["derivativeSource"],
                    )
                  }
                />
                <SelectField
                  label="Clamp Mode"
                  value={config().strategy.derivativeClampMode}
                  options={[
                    { value: "deadband", label: "Deadband" },
                    { value: "hysteresis", label: "Hysteresis" },
                    { value: "hold", label: "Hold" },
                  ]}
                  onInput={(value) =>
                    updateValleyPeak(
                      "derivativeClampMode",
                      value as PeakValleyBotConfig["strategy"]["derivativeClampMode"],
                    )
                  }
                />
                <NumberField
                  label="Clamp Inner"
                  value={config().strategy.derivativeClampInnerThresholdRatio}
                  min={0}
                  max={1}
                  step={0.05}
                  onInput={(value) =>
                    updateValleyPeak(
                      "derivativeClampInnerThresholdRatio",
                      value,
                    )
                  }
                />
                <SelectField
                  label="Buy Entry Edge"
                  value={config().strategy.buyEntrySignalTiming}
                  options={signalTimingOptions}
                  onInput={(value) =>
                    updateValleyPeak(
                      "buyEntrySignalTiming",
                      value as PeakValleyBotConfig["strategy"]["buyEntrySignalTiming"],
                    )
                  }
                />
                <SelectField
                  label="Sell Entry Edge"
                  value={config().strategy.sellEntrySignalTiming}
                  options={signalTimingOptions}
                  onInput={(value) =>
                    updateValleyPeak(
                      "sellEntrySignalTiming",
                      value as PeakValleyBotConfig["strategy"]["sellEntrySignalTiming"],
                    )
                  }
                />
                <SelectField
                  label="Buy Exit Edge"
                  value={config().strategy.buyExitSignalTiming}
                  options={signalTimingOptions}
                  onInput={(value) =>
                    updateValleyPeak(
                      "buyExitSignalTiming",
                      value as PeakValleyBotConfig["strategy"]["buyExitSignalTiming"],
                    )
                  }
                />
                <SelectField
                  label="Sell Exit Edge"
                  value={config().strategy.sellExitSignalTiming}
                  options={signalTimingOptions}
                  onInput={(value) =>
                    updateValleyPeak(
                      "sellExitSignalTiming",
                      value as PeakValleyBotConfig["strategy"]["sellExitSignalTiming"],
                    )
                  }
                />
                <NumberField
                  label="Buy Rate"
                  value={config().strategy.buySpendRate}
                  min={0}
                  step={0.05}
                  onInput={(value) => updateValleyPeak("buySpendRate", value)}
                />
                <NumberField
                  label="Sell Rate"
                  value={config().strategy.sellAmountRate}
                  min={0}
                  step={0.05}
                  onInput={(value) => updateValleyPeak("sellAmountRate", value)}
                />
                <BooleanField
                  label="Long Side"
                  checked={config().strategy.longSideEnabled}
                  onInput={(value) => updateValleyPeak("longSideEnabled", value)}
                />
                <BooleanField
                  label="Short Side"
                  checked={config().strategy.shortSideEnabled}
                  onInput={(value) => updateValleyPeak("shortSideEnabled", value)}
                />
                <SelectField
                  label="Sigma Mode"
                  value={config().strategy.sigmaMode}
                  options={[
                    { value: "trend", label: "Trend" },
                    { value: "static", label: "Static" },
                    { value: "sigmoid-trend", label: "Sigmoid" },
                  ]}
                  onInput={(value) =>
                    updateValleyPeak(
                      "sigmaMode",
                      value as PeakValleyBotConfig["strategy"]["sigmaMode"],
                    )
                  }
                />
                <NumberField
                  label="Buy Sigma"
                  value={config().strategy.buySigma}
                  min={0.000001}
                  step={0.01}
                  onInput={(value) => updateValleyPeak("buySigma", value)}
                />
                <NumberField
                  label="Sell Sigma"
                  value={config().strategy.sellSigma}
                  min={0.000001}
                  step={0.01}
                  onInput={(value) => updateValleyPeak("sellSigma", value)}
                />
                <NumberField
                  label="Sigma A"
                  value={config().strategy.trendSigmaA}
                  min={0.000001}
                  step={0.01}
                  onInput={(value) => updateValleyPeak("trendSigmaA", value)}
                />
                <NumberField
                  label="Sell b1"
                  value={config().strategy.trendSigmaSellB1}
                  min={0}
                  step={0.01}
                  onInput={(value) => updateValleyPeak("trendSigmaSellB1", value)}
                />
                <NumberField
                  label="Buy b2"
                  value={config().strategy.trendSigmaBuyB2}
                  min={0}
                  step={0.01}
                  onInput={(value) => updateValleyPeak("trendSigmaBuyB2", value)}
                />
                <NumberField
                  label="Trend Window"
                  value={config().strategy.trendSigmaWindowSec / 60}
                  min={1}
                  step={1}
                  onInput={(value) => updateValleyPeak("trendSigmaWindowSec", value * 60)}
                />
                <NumberField
                  label="Sigmoid Low"
                  value={config().strategy.sigmoidSigmaLow}
                  min={0.000001}
                  step={0.01}
                  onInput={(value) => updateValleyPeak("sigmoidSigmaLow", value)}
                />
                <NumberField
                  label="Sigmoid High"
                  value={config().strategy.sigmoidSigmaHigh}
                  min={0.000001}
                  step={0.01}
                  onInput={(value) => updateValleyPeak("sigmoidSigmaHigh", value)}
                />
                <NumberField
                  label="KAMA ER"
                  value={config().strategy.kamaErLen}
                  min={1}
                  step={1}
                  onInput={(value) => updateValleyPeak("kamaErLen", Math.round(value))}
                />
                <NumberField
                  label="ER Volume EMA"
                  value={config().strategy.kamaErVolumeLen}
                  min={1}
                  step={1}
                  onInput={(value) => updateValleyPeak("kamaErVolumeLen", Math.round(value))}
                />
                <NumberField
                  label="ER Volume Power"
                  value={config().strategy.kamaErVolumePower}
                  min={0}
                  step={0.1}
                  onInput={(value) => updateValleyPeak("kamaErVolumePower", value)}
                />
                <NumberField
                  label="KAMA Fast"
                  value={config().strategy.kamaFastLen}
                  min={1}
                  step={1}
                  onInput={(value) => updateValleyPeak("kamaFastLen", Math.round(value))}
                />
                <NumberField
                  label="KAMA Slow"
                  value={config().strategy.kamaSlowLen}
                  min={1}
                  step={1}
                  onInput={(value) => updateValleyPeak("kamaSlowLen", Math.round(value))}
                />
                <NumberField
                  label="KAMA Power"
                  value={config().strategy.kamaPower}
                  min={0.1}
                  step={0.1}
                  onInput={(value) => updateValleyPeak("kamaPower", value)}
                />
                <NumberField
                  label="KAMA Volume"
                  value={config().strategy.kamaVolumeLen}
                  min={1}
                  step={1}
                  onInput={(value) => updateValleyPeak("kamaVolumeLen", Math.round(value))}
                />
                <NumberField
                  label="KAMA Rate EMA"
                  value={config().strategy.kamaRateEmaLen}
                  min={1}
                  step={1}
                  onInput={(value) => updateValleyPeak("kamaRateEmaLen", Math.round(value))}
                />
                <NumberField
                  label="Volume Cap"
                  value={config().strategy.kamaVolumeCap}
                  min={1}
                  step={0.1}
                  onInput={(value) => updateValleyPeak("kamaVolumeCap", value)}
                />
                <NumberField
                  label="Volume Power"
                  value={config().strategy.kamaVolumePower}
                  min={0}
                  step={0.1}
                  onInput={(value) => updateValleyPeak("kamaVolumePower", value)}
                />
                <NumberField
                  label="KAMA Rate Low (bps/h)"
                  value={config().strategy.kamaRateThresholdLow * RELATIVE_RATE_TO_BPS_HOUR}
                  min={0}
                  step={1}
                  onInput={(value) => updateValleyPeak(
                    "kamaRateThresholdLow",
                    value / RELATIVE_RATE_TO_BPS_HOUR,
                  )}
                />
                <NumberField
                  label="KAMA Rate High (bps/h)"
                  value={config().strategy.kamaRateThresholdHigh * RELATIVE_RATE_TO_BPS_HOUR}
                  min={0}
                  step={1}
                  onInput={(value) => updateValleyPeak(
                    "kamaRateThresholdHigh",
                    value / RELATIVE_RATE_TO_BPS_HOUR,
                  )}
                />
                <NumberField
                  label="Threshold Noise Lookback (sec)"
                  value={config().strategy.kamaThresholdLookbackSec}
                  min={1}
                  step={60}
                  onInput={(value) => updateValleyPeak("kamaThresholdLookbackSec", value)}
                />
                <SelectField
                  label="Threshold Noise Response"
                  value={config().strategy.kamaThresholdNoiseResponse}
                  options={[
                    { value: "proportional", label: "Proportional" },
                    { value: "inverse", label: "Inverse" },
                  ]}
                  onInput={(value) => updateValleyPeak(
                    "kamaThresholdNoiseResponse",
                    value as PeakValleyBotConfig["strategy"]["kamaThresholdNoiseResponse"],
                  )}
                />
                <Show when={config().strategy.kamaThresholdNoiseResponse === "proportional"}>
                  <NumberField
                    label="Threshold Noise Multiplier"
                    value={config().strategy.kamaThresholdNoiseMultiplier}
                    min={0}
                    step={0.1}
                    onInput={(value) => updateValleyPeak("kamaThresholdNoiseMultiplier", value)}
                  />
                </Show>
                <Show when={config().strategy.kamaThresholdNoiseResponse === "inverse"}>
                  <NumberField
                    label="Inverse Threshold Max (bps/h)"
                    value={config().strategy.kamaThresholdInverseMax * RELATIVE_RATE_TO_BPS_HOUR}
                    min={0}
                    step={1}
                    onInput={(value) => updateValleyPeak(
                      "kamaThresholdInverseMax",
                      value / RELATIVE_RATE_TO_BPS_HOUR,
                    )}
                  />
                  <NumberField
                    label="Inverse Noise Half-Decay (bps/h)"
                    value={config().strategy.kamaThresholdInverseNoiseScale * RELATIVE_RATE_TO_BPS_HOUR}
                    min={0.000001}
                    step={1}
                    onInput={(value) => updateValleyPeak(
                      "kamaThresholdInverseNoiseScale",
                      value / RELATIVE_RATE_TO_BPS_HOUR,
                    )}
                  />
                </Show>
                <NumberField
                  label="KAMA Signal Friction (bps)"
                  value={config().strategy.kamaSignalFriction * 10_000}
                  min={0}
                  step={1}
                  onInput={(value) => updateValleyPeak("kamaSignalFriction", value / 10_000)}
                />
                <NumberField
                  label="Warmup min"
                  value={config().strategy.saturationSec / 60}
                  min={0}
                  onInput={(value) => updateValleyPeak("saturationSec", value * 60)}
                />
                <NumberListField
                  label="Buy Confirms"
                  value={config().strategy.buyConfirmationOffsets}
                  onInput={(value) => updateValleyPeakOffsets("buyConfirmationOffsets", value)}
                />
                <NumberListField
                  label="Sell Confirms"
                  value={config().strategy.sellConfirmationOffsets}
                  onInput={(value) => updateValleyPeakOffsets("sellConfirmationOffsets", value)}
                />
                <NumberListField
                  label="Buy Exit Confirms"
                  value={config().strategy.buyExitConfirmationOffsets}
                  onInput={(value) =>
                    updateValleyPeakOffsets("buyExitConfirmationOffsets", value)
                  }
                />
                <NumberListField
                  label="Sell Exit Confirms"
                  value={config().strategy.sellExitConfirmationOffsets}
                  onInput={(value) =>
                    updateValleyPeakOffsets("sellExitConfirmationOffsets", value)
                  }
                />
              </div>
            </div>

            <div class="rounded-2 bg-ink-800 p-3">
              <div class="muted-label">Grids & Lifecycle</div>
              <div class="mt-3 grid grid-cols-2 gap-3">
                <NumberField
                  label="Entry Orders"
                  value={config().entryGrid.orderCount}
                  min={1}
                  onInput={(orderCount) => updateGrid("entryGrid", { orderCount: Math.round(orderCount) })}
                />
                <NumberField
                  label="Entry Step %"
                  value={config().entryGrid.maxPriceStep * 100}
                  min={0}
                  step={0.01}
                  onInput={(value) => updateGrid("entryGrid", { maxPriceStep: value / 100 })}
                />
                <NumberField
                  label="Exit Orders"
                  value={config().exitGrid.orderCount}
                  min={1}
                  onInput={(orderCount) => updateGrid("exitGrid", { orderCount: Math.round(orderCount) })}
                />
                <NumberField
                  label="Exit Step %"
                  value={config().exitGrid.maxPriceStep * 100}
                  min={0}
                  step={0.01}
                  onInput={(value) => updateGrid("exitGrid", { maxPriceStep: value / 100 })}
                />
                <SelectField
                  label="Entry Sizes"
                  value={config().entryGrid.sizeDistribution}
                  options={[
                    { value: "linear", label: "Linear" },
                    { value: "geometric", label: "Geometric" },
                  ]}
                  onInput={(sizeDistribution) => updateGrid("entryGrid", {
                    sizeDistribution: sizeDistribution as "linear" | "geometric",
                  })}
                />
                <SelectField
                  label="Exit Sizes"
                  value={config().exitGrid.sizeDistribution}
                  options={[
                    { value: "linear", label: "Linear" },
                    { value: "geometric", label: "Geometric" },
                  ]}
                  onInput={(sizeDistribution) => updateGrid("exitGrid", {
                    sizeDistribution: sizeDistribution as "linear" | "geometric",
                  })}
                />
                <NumberField
                  label="Entry Fraction"
                  value={config().entryGrid.sizeFraction}
                  min={0.01}
                  max={1}
                  step={0.01}
                  onInput={(sizeFraction) => updateGrid("entryGrid", { sizeFraction })}
                />
                <NumberField
                  label="Exit Fraction"
                  value={config().exitGrid.sizeFraction}
                  min={0.01}
                  max={1}
                  step={0.01}
                  onInput={(sizeFraction) => updateGrid("exitGrid", { sizeFraction })}
                />
                <SelectField
                  label="Grid Reset"
                  value={config().exitGrid.reset}
                  options={[
                    { value: "previous-anchor", label: "Previous Anchor" },
                    { value: "last-filled-order", label: "Last Fill" },
                  ]}
                  onInput={(reset) => updateGrid("exitGrid", {
                    reset: reset as "previous-anchor" | "last-filled-order",
                  })}
                />
                <NumberField
                  label="Lifetime min"
                  value={(config().positionLifetimeMs ?? 0) > 0 ? (config().positionLifetimeMs ?? 0) / 60_000 : null}
                  min={0}
                  onInput={(value) => update("positionLifetimeMs", value > 0 ? value * 60_000 : null)}
                />
                <NumberField
                  label="Stop Loss %"
                  value={(config().stopLossRate ?? 0) > 0 ? (config().stopLossRate ?? 0) * 100 : null}
                  min={0}
                  step={0.1}
                  onInput={(value) => update("stopLossRate", value > 0 ? value / 100 : null)}
                />
                <NumberField
                  label="Take Profit %"
                  value={(config().takeProfitRate ?? 0) > 0 ? (config().takeProfitRate ?? 0) * 100 : null}
                  min={0}
                  step={0.1}
                  onInput={(value) => update("takeProfitRate", value > 0 ? value / 100 : null)}
                />
              </div>
            </div>
          </div>
        )}
      </Show>
    </section>
  );
}

function NumberField(props: {
  label: string;
  value: number | null | undefined;
  min?: number;
  max?: number;
  step?: number;
  placeholder?: string;
  onInput: (value: number) => void;
}) {
  const value = () =>
    typeof props.value === "number" && Number.isFinite(props.value)
      ? props.value
      : "";
  return (
    <label class="block">
      <span class="muted-label">{props.label}</span>
      <input
        class="mt-1 w-full rounded-2 border border-line bg-ink-900 px-2 py-2 text-sm text-ink-100 tabular-nums"
        type="number"
        min={props.min}
        max={props.max}
        step={props.step ?? 1}
        placeholder={props.placeholder}
        value={value()}
        onInput={(event) => props.onInput(Number(event.currentTarget.value))}
      />
    </label>
  );
}

function NumberListField(props: {
  label: string;
  value: number[];
  onInput: (value: number[]) => void;
}) {
  const parseOffsets = (rawValue: string) =>
    rawValue
      .split(",")
      .map((part) => Number(part.trim()))
      .filter((value) => Number.isFinite(value) && value >= 0)
      .map((value) => Math.round(value));

  return (
    <label class="block">
      <span class="muted-label">{props.label}</span>
      <input
        class="mt-1 w-full rounded-2 border border-line bg-ink-900 px-2 py-2 text-sm text-ink-100 tabular-nums"
        type="text"
        value={props.value.join(",")}
        onInput={(event) => props.onInput(parseOffsets(event.currentTarget.value))}
      />
    </label>
  );
}

function TextField(props: {
  label: string;
  value: string;
  type?: "text" | "password";
  placeholder?: string;
  autocomplete?: string;
  onInput: (value: string) => void;
}) {
  return (
    <label class="block">
      <span class="muted-label">{props.label}</span>
      <input
        class="mt-1 w-full rounded-2 border border-line bg-ink-900 px-2 py-2 text-sm text-ink-100"
        type={props.type ?? "text"}
        value={props.value}
        placeholder={props.placeholder}
        autocomplete={props.autocomplete}
        spellcheck={false}
        onInput={(event) => props.onInput(event.currentTarget.value)}
      />
    </label>
  );
}

function SelectField(props: {
  label: string;
  value: string;
  options: Array<{ value: string; label: string }>;
  onInput: (value: string) => void;
}) {
  return (
    <label class="block">
      <span class="muted-label">{props.label}</span>
      <select
        class="mt-1 w-full rounded-2 border border-line bg-ink-900 px-2 py-2 text-sm font-semibold text-ink-100"
        value={props.value}
        onInput={(event) => props.onInput(event.currentTarget.value)}
      >
        <For each={props.options}>
          {(option) => <option value={option.value}>{option.label}</option>}
        </For>
      </select>
    </label>
  );
}

function BooleanField(props: {
  label: string;
  checked: boolean;
  onInput: (value: boolean) => void;
}) {
  return (
    <label class="block rounded-2 border border-line bg-ink-900 px-2 py-2">
      <span class="muted-label">{props.label}</span>
      <span class="mt-2 flex items-center gap-2 text-sm font-semibold text-ink-100">
        <input
          class="h-4 w-4 accent-accent"
          type="checkbox"
          checked={props.checked}
          onInput={(event) => props.onInput(event.currentTarget.checked)}
        />
        Enabled
      </span>
    </label>
  );
}

function PerformancePanel(props: { snapshot?: RuntimeSnapshot }) {
  const metrics = () => props.snapshot?.bot.metrics;

  return (
    <div class="panel">
      <div class="mb-3 flex items-center justify-between">
        <div>
          <div class="muted-label">Performance</div>
          <h2 class="text-lg font-semibold">
            {props.snapshot?.execution?.exchangeDriven ? "Binance Bot" : "Paper Bot"}
          </h2>
        </div>
        <Activity size={18} class="text-accent" />
      </div>
      <div class="grid grid-cols-2 gap-3">
        <SmallMetric label="Net PnL" value={`$${formatQuote(metrics()?.netPnl, 2)}`} />
        <SmallMetric label="Return" value={formatPercent(metrics()?.returnPct)} />
        <SmallMetric label="Positions" value={formatQuote(metrics()?.positionCount, 0)} />
        <SmallMetric label="Orders" value={formatQuote(metrics()?.orderCount, 0)} />
        <SmallMetric label="Long Asset" value={formatAsset(metrics()?.longAsset)} />
        <SmallMetric label="Short Asset" value={formatAsset(metrics()?.shortAsset)} />
      </div>
    </div>
  );
}

function LiveEquityPanel(props: { points: EquityPoint[] }) {
  return (
    <div class="panel">
      <div class="mb-3 flex items-center justify-between">
        <div>
          <div class="muted-label">Live Performance</div>
          <h2 class="text-lg font-semibold">Equity Curve</h2>
        </div>
      </div>
      <div class="h-48">
        <EquityChart points={props.points} emptyText="No live equity samples yet" />
      </div>
    </div>
  );
}

function StrategyStatePanel(props: { bot?: RuntimeSnapshot["bot"] }) {
  const diagnostics = () => props.bot?.diagnostics.strategy;
  const risk = (side: "long" | "short") =>
    [...(props.bot?.diagnostics.entryRisk ?? [])].reverse().find((item) => item.side === side);

  return (
    <div class="panel">
      <div class="mb-3 flex items-center justify-between">
        <div>
          <div class="muted-label">Decision State</div>
          <h2 class="text-lg font-semibold">Peak/Valley</h2>
        </div>
        <Activity size={18} class="text-accent" />
      </div>
      <Show when={diagnostics()} fallback={<div class="text-sm text-ink-300">Waiting for strategy evaluation</div>}>
        {(state) => (
          <>
            <div class="mb-3 flex flex-wrap items-center gap-2 text-xs">
              <span class={state().ready ? "rounded-2 bg-gain/12 px-2 py-1 text-gain" : "rounded-2 bg-warn/12 px-2 py-1 text-warn"}>
                {state().ready ? "ready" : `${formatDuration(state().warmupRemainingMs)} warmup`}
              </span>
              <span class="rounded-2 bg-ink-800 px-2 py-1 text-ink-300">
                {state().movingAverageType.toUpperCase()} · source {state().derivativeSource.toUpperCase()}
              </span>
              <Show when={state().latestTick}>
                {(tick) => <span class="text-ink-400">${formatQuote(tick().price, 4)} · {formatTime(tick().timestamp)}</span>}
              </Show>
            </div>
            <StrategyMarketState diagnostics={state()} />
            <div class="grid grid-cols-2 gap-3">
              <SmallMetric
                label="Last Signal"
                value={state().lastSignal ? `${state().lastSignal?.type} ${state().lastSignal?.side}` : "-"}
              />
              <SmallMetric label="Reason" value={state().lastSignal?.reason ?? "-"} />
              <SmallMetric label="Blockers" value={state().blockers.join(", ") || "none"} />
              <SmallMetric
                label="Gates"
                value={`${state().gates.filter((gate) => gate.passed).length}/${state().gates.length}`}
              />
              <SmallMetric label="Long Risk" value={entryRiskLabel(risk("long"))} />
              <SmallMetric label="Short Risk" value={entryRiskLabel(risk("short"))} />
              <SmallMetric label="Buy Size / Sigma" value={`${formatPercent(state().sizing.buy.size * 100)} / ${formatQuote(state().sizing.buy.sigma, 7)}`} />
              <SmallMetric label="Sell Size / Sigma" value={`${formatPercent(state().sizing.sell.size * 100)} / ${formatQuote(state().sizing.sell.sigma, 7)}`} />
            </div>

            <div class="mt-3 grid grid-cols-1 gap-2">
              <StrategyDecisionCheck label="Entry Long" code="long.entry" diagnostics={state()} />
              <StrategyDecisionCheck label="Entry Short" code="short.entry" diagnostics={state()} />
              <StrategyDecisionCheck label="Exit Short" code="short.exit" diagnostics={state()} />
              <StrategyDecisionCheck label="Exit Long" code="long.exit" diagnostics={state()} />
            </div>

            <div class="mt-3 overflow-auto rounded-2 bg-ink-800 p-3">
              <div class="mb-2 flex items-center justify-between gap-3">
                <div class="muted-label">Average Roles & Derivatives</div>
                <Show when={state().kama}>
                  {(kama) => (
                    <span class="text-xs tabular-nums text-ink-300">
                      KAMA ${formatQuote(kama().value, 4)} · rate {formatQuote(kama().rawRate, 8)} → {formatQuote(kama().clampedRate, 8)} · ER {formatQuote(kama().efficiencyRatio, 3)} · RV {formatQuote(kama().relativeVolume, 2)} · α {formatQuote(kama().alpha, 4)}
                    </span>
                  )}
                </Show>
              </div>
              <table class="w-full min-w-160 text-sm">
                <thead>
                  <tr>
                    <th class="table-head pb-2">Role</th>
                    <th class="table-head pb-2">Window</th>
                    <th class="table-head pb-2">Average</th>
                    <th class="table-head pb-2">Raw rate</th>
                    <th class="table-head pb-2">Previous</th>
                    <th class="table-head pb-2">Clamped</th>
                    <th class="table-head pb-2">Thresholds</th>
                    <th class="table-head pb-2">Shape</th>
                  </tr>
                </thead>
                <tbody>
                  <For each={state().averages}>
                    {(average) => (
                      <tr>
                        <td class="td-cell"><AverageRoleBadges average={average} /></td>
                        <td class="td-cell">{formatDuration(average.windowSec * 1_000)}</td>
                        <td class="td-cell tabular-nums">${formatQuote(average.value, 4)}</td>
                        <td class="td-cell tabular-nums">{formatQuote(average.rawRate, 9)}</td>
                        <td class="td-cell tabular-nums">{formatQuote(average.previousClampedRate, 9)}</td>
                        <td class="td-cell tabular-nums">{formatQuote(average.clampedRate, 9)}</td>
                        <td class="td-cell tabular-nums">{formatQuote(average.thresholdLow, 7)} / {formatQuote(average.thresholdHigh, 7)}</td>
                        <td class="td-cell">{averageShape(average)}</td>
                      </tr>
                    )}
                  </For>
                </tbody>
              </table>
            </div>
          </>
        )}
      </Show>
    </div>
  );
}

function StrategyMarketState(props: { diagnostics: PeakValleyStrategyDiagnostics }) {
  const average = () => props.diagnostics.averages.find((item) => item.roles.trendSigma)
    ?? props.diagnostics.averages.at(-1);
  const state = () => (average()?.clampedRate ?? 0) > 0
    ? "rising"
    : (average()?.clampedRate ?? 0) < 0
      ? "falling"
      : "sideways";
  return (
    <div
      class="mb-3 rounded-2 border p-3"
      classList={{
        "border-gain/35 bg-gain/10": state() === "rising",
        "border-loss/35 bg-loss/10": state() === "falling",
        "border-line bg-ink-800": state() === "sideways",
      }}
    >
      <div class="flex items-center justify-between gap-3">
        <div>
          <div class="muted-label">Market State</div>
          <div class="mt-1 text-lg font-semibold capitalize">{state()}</div>
        </div>
        <div class="text-right text-xs tabular-nums text-ink-300">
          <div>{formatDuration((average()?.windowSec ?? 0) * 1_000)} window</div>
          <div>{formatQuote(average()?.rawRate, 9)} → {formatQuote(average()?.clampedRate, 9)}</div>
        </div>
      </div>
    </div>
  );
}

function StrategyDecisionCheck(props: {
  label: string;
  code: string;
  diagnostics: PeakValleyStrategyDiagnostics;
}) {
  const result = () => props.diagnostics.gates.find((gate) => gate.code === props.code);
  const details = () => props.diagnostics.gates.filter((gate) => gate.code.startsWith(`${props.code}.`));
  return (
    <div class="rounded-2 bg-ink-800 p-3">
      <div class="flex items-center justify-between gap-3">
        <span class="text-sm font-semibold text-ink-100">{props.label}</span>
        <span class={result()?.passed ? "text-sm font-semibold text-gain" : "text-sm text-ink-400"}>
          {result()?.passed ? "active" : "blocked"}
        </span>
      </div>
      <div class="mt-2 flex flex-wrap gap-2">
        <For each={details()} fallback={<span class="text-xs text-ink-400">No checks yet</span>}>
          {(gate) => (
            <span class={gate.passed ? "rounded-2 bg-gain/12 px-2 py-1 text-xs text-gain" : "rounded-2 bg-loss/10 px-2 py-1 text-xs text-loss"}>
              {gate.code.slice(props.code.length + 1)} · {formatQuote(gate.value, 8)} / {formatQuote(gate.threshold, 8)}
            </span>
          )}
        </For>
      </div>
    </div>
  );
}

function AverageRoleBadges(props: { average: PeakValleyAverageDiagnostics }) {
  const roles = () => [
    props.average.roles.buyPrimary && ["buy", "bg-gain/12 text-gain"],
    props.average.roles.sellPrimary && ["sell", "bg-loss/12 text-loss"],
    props.average.roles.buyEntryConfirmation && ["buy entry confirm", "bg-ink-700 text-ink-200"],
    props.average.roles.sellEntryConfirmation && ["sell entry confirm", "bg-ink-700 text-ink-200"],
    props.average.roles.buyExitConfirmation && ["buy exit confirm", "bg-ink-700 text-ink-200"],
    props.average.roles.sellExitConfirmation && ["sell exit confirm", "bg-ink-700 text-ink-200"],
    props.average.roles.sizing && ["sizing", "bg-accent/12 text-accent"],
    props.average.roles.trendSigma && ["sigma", "bg-warn/12 text-warn"],
  ].filter((item): item is string[] => Boolean(item));
  return (
    <div class="flex min-w-40 flex-wrap gap-1">
      <For each={roles()} fallback={<span class="text-xs text-ink-400">-</span>}>
        {(role) => <span class={`rounded-2 px-2 py-1 text-xs ${role[1]}`}>{role[0]}</span>}
      </For>
    </div>
  );
}

function averageShape(average: PeakValleyAverageDiagnostics): string {
  if (average.previousClampedRate < 0 && average.clampedRate >= 0) return "valley";
  if (average.previousClampedRate > 0 && average.clampedRate <= 0) return "peak";
  return average.clampedRate > 0 ? "rising" : average.clampedRate < 0 ? "falling" : "flat";
}

function entryRiskLabel(report: RuntimeSnapshot["bot"]["diagnostics"]["entryRisk"][number] | undefined): string {
  if (!report) return "-";
  return `${report.blocker ?? "ready"} · $${formatQuote(report.size, 2)} · ${formatLeverage(report.leverage)}`;
}

function SmallMetric(props: { label: string; value: string }) {
  return (
    <div class="rounded-2 bg-ink-800 p-3">
      <div class="muted-label">{props.label}</div>
      <div class="mt-1 text-base font-semibold tabular-nums text-ink-100">{props.value}</div>
    </div>
  );
}

function formatLeverage(value: number | undefined): string {
  if (!Number.isFinite(value)) {
    return "-";
  }
  if ((value as number) >= 999) {
    return ">999x";
  }

  return `${formatQuote(value, 2)}x`;
}

function formatBps(value: number | undefined): string {
  if (!Number.isFinite(value)) {
    return "-";
  }
  return `${formatQuote(value, 3)} bps`;
}

function formatOptionalQuote(value: number | undefined, quoteAsset: string): string {
  if (!Number.isFinite(value)) {
    return "-";
  }
  return `${formatQuote(value, 2)} ${quoteAsset}`;
}

function formatOptionalAsset(value: number | undefined): string {
  if (!Number.isFinite(value)) {
    return "-";
  }
  return formatAsset(value);
}

function formatRatio(value: number | undefined): string {
  if (!Number.isFinite(value)) {
    return "-";
  }

  return formatQuote(value, 3);
}

function OrderBookPanel(props: { snapshot?: RuntimeSnapshot }) {
  const book = () => props.snapshot?.market.orderBook;
  const asks = () => (book()?.asks ?? []).slice(0, 6).reverse();
  const bids = () => (book()?.bids ?? []).slice(0, 6);

  return (
    <div class="panel">
      <div class="mb-3 flex items-center justify-between">
        <div>
          <div class="muted-label">Order Book</div>
          <h2 class="text-lg font-semibold">Top Depth</h2>
        </div>
        <div class="text-xs text-ink-300">{formatTime(book()?.eventTime)}</div>
      </div>
      <div class="grid grid-cols-[1fr_1fr] gap-3 text-sm">
        <div>
          <div class="mb-2 grid grid-cols-2 text-xs uppercase text-ink-300">
            <span>Ask</span>
            <span class="text-right">Size</span>
          </div>
          <For each={asks()}>
            {(level) => (
              <div class="grid grid-cols-2 border-t border-line py-1.5">
                <span class="text-loss tabular-nums">{formatQuote(level.price, 2)}</span>
                <span class="text-right text-ink-200 tabular-nums">{formatAsset(level.quantity)}</span>
              </div>
            )}
          </For>
        </div>
        <div>
          <div class="mb-2 grid grid-cols-2 text-xs uppercase text-ink-300">
            <span>Bid</span>
            <span class="text-right">Size</span>
          </div>
          <For each={bids()}>
            {(level) => (
              <div class="grid grid-cols-2 border-t border-line py-1.5">
                <span class="text-gain tabular-nums">{formatQuote(level.price, 2)}</span>
                <span class="text-right text-ink-200 tabular-nums">{formatAsset(level.quantity)}</span>
              </div>
            )}
          </For>
        </div>
      </div>
    </div>
  );
}

function CorrelationPanel(props: {
  snapshot?: CorrelationSnapshot;
  sortMode: CorrelationSortMode;
  error?: string;
  onSortChange: (mode: CorrelationSortMode) => void;
  onRefresh: (refresh: boolean) => void;
}) {
  const [isExpanded, setIsExpanded] = createSignal(false);
  const snapshot = () => props.snapshot;
  const entries = createMemo(() =>
    sortCorrelationEntries(snapshot()?.entries ?? [], props.sortMode),
  );
  const isRunning = () => snapshot()?.status === "running";
  const progress = () => {
    const current = snapshot()?.calculatedPairs ?? 0;
    const total = snapshot()?.expectedPairs ?? 0;
    if (total <= 0) {
      return isRunning() ? 0 : 100;
    }
    return Math.max(0, Math.min(100, (current / total) * 100));
  };
  const message = () =>
    props.error ?? snapshot()?.error ?? snapshot()?.message ?? "Correlations have not been computed yet";

  return (
    <section class="panel min-w-0 overflow-hidden">
      <div class="mb-4 flex flex-col gap-3 xl:flex-row xl:items-center xl:justify-between">
        <div>
          <div class="muted-label">Correlations</div>
          <h2 class="text-lg font-semibold">
            {snapshot()?.focalDisplaySymbol ?? "Asset"} vector
          </h2>
        </div>
        <div class="flex flex-wrap items-center gap-2">
          <Show when={isExpanded()}>
            <For each={correlationSortModes}>
              {(mode) => (
                <button
                  class="rounded-2 border px-2.5 py-1.5 text-xs font-semibold transition"
                  classList={{
                    "border-accent bg-accent text-ink-950": props.sortMode === mode.value,
                    "border-line bg-ink-800 text-ink-300 hover:border-accent hover:text-ink-100":
                      props.sortMode !== mode.value,
                  }}
                  onClick={() => props.onSortChange(mode.value)}
                  type="button"
                >
                  {mode.label}
                </button>
              )}
            </For>
          </Show>
          <button
            class="btn px-2.5"
            disabled={isRunning()}
            onClick={() => props.onRefresh(true)}
            type="button"
          >
            <RefreshCw size={16} class={isRunning() ? "animate-spin" : ""} />
          </button>
          <button
            aria-controls="correlation-panel-content"
            aria-expanded={isExpanded()}
            aria-label={isExpanded() ? "Collapse correlations" : "Expand correlations"}
            class="btn px-2.5"
            onClick={() => setIsExpanded((value) => !value)}
            title={isExpanded() ? "Collapse correlations" : "Expand correlations"}
            type="button"
          >
            <Show when={isExpanded()} fallback={<ChevronRight size={16} />}>
              <ChevronDown size={16} />
            </Show>
          </button>
        </div>
      </div>

      <Show when={isExpanded()}>
        <div id="correlation-panel-content">
          <Show when={props.error ?? snapshot()?.error}>
            {(error) => <div class="mb-3 rounded-2 bg-loss/12 p-3 text-sm text-loss">{error()}</div>}
          </Show>

          <div class="mb-4 grid grid-cols-2 gap-3 lg:grid-cols-6">
            <SmallMetric
              label="Pairs"
              value={`${formatQuote(snapshot()?.calculatedPairs, 0)} / ${formatQuote(
                snapshot()?.expectedPairs,
                0,
              )}`}
            />
            <SmallMetric label="Markets" value={formatQuote(snapshot()?.marketCount, 0)} />
            <SmallMetric label="Lookback" value={formatDuration(snapshot()?.lookbackMs)} />
            <SmallMetric label="Requests" value={formatQuote(snapshot()?.requests, 0)} />
            <SmallMetric
              label="Cache"
              value={snapshot()?.cacheLoaded ? "Vector" : formatQuote(snapshot()?.cacheFetchedCandles, 0)}
            />
            <SmallMetric label="Stream" value={snapshot()?.streamConnected ? "Live" : "Idle"} />
          </div>

          <div class="mb-4 rounded-2 bg-ink-800 p-3">
            <div class="mb-2 flex items-center justify-between gap-3">
              <div class="min-w-0 truncate text-sm text-ink-100">
                {message()}
                <Show when={snapshot()?.truncated}>
                  <span class="ml-2 text-warn">max {formatQuote(snapshot()?.marketCount, 0)} markets</span>
                </Show>
              </div>
              <div class="shrink-0 text-sm tabular-nums text-ink-300">
                {formatPercent(progress())}
              </div>
            </div>
            <div class="h-2 overflow-hidden rounded-full bg-ink-700">
              <div
                class="h-full bg-accent transition-all"
                style={{ width: `${progress()}%` }}
              />
            </div>
          </div>

          <div class="max-w-full overflow-x-auto">
            <table class="w-full min-w-180">
              <thead>
                <tr>
                  <th class="table-head pb-2">Asset</th>
                  <th class="table-head pb-2">Correlation</th>
                  <th class="table-head pb-2">Abs</th>
                  <th class="table-head pb-2">Samples</th>
                  <th class="table-head pb-2">Window</th>
                  <th class="table-head pb-2">Updated</th>
                </tr>
              </thead>
              <tbody>
                <For each={entries()} fallback={<EmptyRow columns={6} label="No correlations yet" />}>
                  {(entry) => (
                    <tr>
                      <td class="td-cell">
                        <div class="font-semibold text-ink-100">{entry.displaySymbol}</div>
                        <div class="mt-1 text-xs text-ink-300">{entry.symbol}</div>
                      </td>
                      <td
                        class="td-cell font-semibold tabular-nums"
                        classList={{
                          "text-gain": (entry.correlation ?? 0) > 0,
                          "text-loss": (entry.correlation ?? 0) < 0,
                          "text-ink-300": entry.correlation === undefined,
                        }}
                      >
                        {formatCorrelation(entry.correlation)}
                      </td>
                      <td class="td-cell tabular-nums">
                        {formatCorrelation(
                          entry.correlation === undefined ? undefined : Math.abs(entry.correlation),
                        )}
                      </td>
                      <td class="td-cell tabular-nums">{formatQuote(entry.samples, 0)}</td>
                      <td class="td-cell text-ink-300">
                        {formatDateTime(entry.startTime)} - {formatDateTime(entry.endTime)}
                      </td>
                      <td class="td-cell text-ink-300">{formatTime(entry.updatedAt)}</td>
                    </tr>
                  )}
                </For>
              </tbody>
            </table>
          </div>
        </div>
      </Show>
    </section>
  );
}

const correlationSortModes: Array<{ value: CorrelationSortMode; label: string }> = [
  { value: "abs-desc", label: "|r| desc" },
  { value: "abs-asc", label: "|r| asc" },
  { value: "value-desc", label: "r desc" },
  { value: "value-asc", label: "r asc" },
];

function sortCorrelationEntries(
  entries: CorrelationEntry[],
  mode: CorrelationSortMode,
): CorrelationEntry[] {
  return [...entries].sort((a, b) => {
    const aValue = sortableCorrelationValue(a, mode);
    const bValue = sortableCorrelationValue(b, mode);
    const aMissing = aValue === undefined;
    const bMissing = bValue === undefined;
    if (aMissing || bMissing) {
      if (aMissing && bMissing) {
        return a.displaySymbol.localeCompare(b.displaySymbol);
      }
      return aMissing ? 1 : -1;
    }

    const direction = mode === "abs-asc" || mode === "value-asc" ? 1 : -1;
    return (aValue - bValue) * direction || a.displaySymbol.localeCompare(b.displaySymbol);
  });
}

function sortableCorrelationValue(
  entry: CorrelationEntry,
  mode: CorrelationSortMode,
): number | undefined {
  if (!Number.isFinite(entry.correlation)) {
    return undefined;
  }

  return mode.startsWith("abs") ? Math.abs(entry.correlation as number) : entry.correlation;
}

function formatCorrelation(value: number | undefined): string {
  if (!Number.isFinite(value)) {
    return "-";
  }

  return formatQuote(value, 3);
}

function OrdersPanel(props: { title: string; orders: readonly TradingOrderSnapshot[] }) {
  return (
    <div class="panel min-w-0 overflow-hidden">
      <div class="mb-3">
        <div class="muted-label">Orders</div>
        <h2 class="text-lg font-semibold">{props.title}</h2>
      </div>
      <div class="max-w-full overflow-x-auto">
        <table class="w-full min-w-130">
          <thead>
            <tr>
              <th class="table-head pb-2">Side</th>
              <th class="table-head pb-2">Price</th>
              <th class="table-head pb-2">Qty</th>
              <th class="table-head pb-2">Type</th>
              <th class="table-head pb-2">Status</th>
            </tr>
          </thead>
          <tbody>
            <For each={props.orders} fallback={<EmptyRow columns={5} label="No open orders" />}>
              {(order) => (
                <tr>
                  <td class="td-cell">
                    <Side side={order.side} />
                  </td>
                  <td class="td-cell">${formatQuote(order.price ?? order.stopPrice ?? undefined, 2)}</td>
                  <td class="td-cell">{formatAsset(order.size)}</td>
                  <td class="td-cell text-ink-300">{order.type}</td>
                  <td class="td-cell text-ink-300">{order.status}</td>
                </tr>
              )}
            </For>
          </tbody>
        </table>
      </div>
    </div>
  );
}

function FillsPanel(props: {
  fills: Array<RuntimeSnapshot["recentEvents"][number] & { fill: TradingFill }>;
}) {
  return (
    <div class="panel min-w-0 overflow-hidden">
      <div class="mb-3">
        <div class="muted-label">Executions</div>
        <h2 class="text-lg font-semibold">Recent Fills</h2>
      </div>
      <div class="max-w-full overflow-x-auto">
        <table class="w-full min-w-130">
          <thead>
            <tr>
              <th class="table-head pb-2">Order</th>
              <th class="table-head pb-2">Price</th>
              <th class="table-head pb-2">Qty</th>
              <th class="table-head pb-2">Quote</th>
              <th class="table-head pb-2">Filled</th>
            </tr>
          </thead>
          <tbody>
            <For each={props.fills} fallback={<EmptyRow columns={5} label="No fills yet" />}>
              {(fill) => (
                <tr>
                  <td class="td-cell text-ink-300">{fill.orderId ?? "-"}</td>
                  <td class="td-cell">${formatQuote(fill.fill.filledQuote / fill.fill.filledAsset, 2)}</td>
                  <td class="td-cell">{formatAsset(fill.fill.filledAsset)}</td>
                  <td class="td-cell">${formatQuote(fill.fill.filledQuote, 2)}</td>
                  <td class="td-cell text-ink-300">{formatTime(fill.at)}</td>
                </tr>
              )}
            </For>
          </tbody>
        </table>
      </div>
    </div>
  );
}

function TradingPositionsPanel(props: {
  positions: readonly TradingPosition[];
  baseAsset: string;
  quoteAsset: string;
  error?: string;
  onRecordTrade: (input: ManualTradeInput) => Promise<boolean>;
}) {
  const [quantity, setQuantity] = createSignal(0);
  const open = (side: "buy" | "sell") => {
    if (quantity() > 0) {
      void props.onRecordTrade({ side, quantity: quantity(), positionEffect: "open" });
    }
  };
  return (
    <section class="panel min-w-0 overflow-hidden">
      <div class="mb-4 flex flex-wrap items-end justify-between gap-3">
        <div>
          <div class="muted-label">Positions</div>
          <h2 class="text-lg font-semibold">Position Grids</h2>
        </div>
        <div class="flex flex-wrap items-end gap-2">
          <NumberField
            label={`${props.baseAsset} quantity`}
            value={quantity()}
            min={0}
            step={0.000001}
            onInput={setQuantity}
          />
          <button class={buttonPrimaryClass} type="button" onClick={() => open("buy")}>Open Long</button>
          <button class={buttonDangerClass} type="button" onClick={() => open("sell")}>Open Short</button>
        </div>
      </div>
      <Show when={props.error}>
        {(error) => <div class="mb-3 rounded-2 bg-loss/12 p-3 text-sm text-loss">{error()}</div>}
      </Show>
      <div class="max-w-full overflow-x-auto">
        <table class="w-full min-w-160">
          <thead>
            <tr>
              <th class="table-head pb-2">Side</th>
              <th class="table-head pb-2">{props.baseAsset}</th>
              <th class="table-head pb-2">{props.quoteAsset}</th>
              <th class="table-head pb-2">Entry</th>
              <th class="table-head pb-2">Exit</th>
              <th class="table-head pb-2">Borrowed</th>
              <th class="table-head pb-2">Action</th>
            </tr>
          </thead>
          <tbody>
            <For each={props.positions} fallback={<EmptyRow columns={7} label="No positions" />}>
              {(position) => (
                <tr>
                  <td class="td-cell"><Side side={position.side === "long" ? "buy" : "sell"} /></td>
                  <td class="td-cell">{formatAsset(position.asset)}</td>
                  <td class="td-cell">${formatQuote(position.quote, 2)}</td>
                  <td class="td-cell">{position.entryGrid?.orders.length ?? 0} orders</td>
                  <td class="td-cell">{position.exitGrid?.orders.length ?? 0} orders</td>
                  <td class="td-cell text-ink-300">
                    {formatAsset(position.externalBorrow.asset)} / ${formatQuote(position.externalBorrow.quote, 2)}
                  </td>
                  <td class="td-cell">
                    <button
                      class={buttonPanelClass}
                      type="button"
                      onClick={() => void props.onRecordTrade({
                        side: position.side === "long" ? "sell" : "buy",
                        quantity: position.asset,
                        targetPositionId: position.id,
                        positionEffect: "close",
                      })}
                    >
                      Close
                    </button>
                  </td>
                </tr>
              )}
            </For>
          </tbody>
        </table>
      </div>
    </section>
  );
}

function PositionLedgerPanel(props: {
  ledger?: PositionLedger;
  baseAsset: string;
  quoteAsset: string;
  currentPrice: number;
  error?: string;
  onRecordTrade: (input: ManualTradeInput) => Promise<boolean>;
}) {
  const summary = () => props.ledger?.summary;
  const longs = () => (props.ledger?.longs ?? []).filter((lot) => lot.status !== "closed");
  const shorts = () => (props.ledger?.shorts ?? []).filter((lot) => lot.status !== "closed");
  const [draft, setDraft] = createSignal<ManualTradeDraft>();
  const [submitting, setSubmitting] = createSignal(false);
  const [lotViewMode, setLotViewMode] = createSignal<LotViewMode>("tables");
  const currentPrice = () => props.currentPrice || summary()?.currentPrice || 0;
  const openDraft = (draft: ManualTradeDraft) => {
    setDraft({
      ...draft,
      price: draft.price || currentPrice(),
      priceMode: draft.priceMode ?? "current",
      positionEffect: draft.positionEffect ?? (draft.targetPositionId ? "close" : "open"),
      lifetimeMinutes: draft.lifetimeMinutes ?? 0,
      stopLossPrice: draft.stopLossPrice ?? 0,
      takeProfitPrice: draft.takeProfitPrice ?? 0,
    });
  };
  const submitDraft = async () => {
    const value = draft();
    if (!value || submitting()) {
      return;
    }

    const price = manualTradePrice(value, currentPrice());
    const quantity = manualTradeQuantity(value, currentPrice());
    if (!Number.isFinite(price) || price <= 0 || !Number.isFinite(quantity) || quantity <= 0) {
      return;
    }

    setSubmitting(true);
    const ok = await props.onRecordTrade({
      side: value.side,
      price,
      quantity,
      targetPositionId: value.targetPositionId,
      positionEffect: value.positionEffect,
      lifetimeMs:
        !value.targetPositionId && value.lifetimeMinutes > 0
          ? value.lifetimeMinutes * 60_000
          : undefined,
      stopLossPrice:
        !value.targetPositionId && value.stopLossPrice > 0 ? value.stopLossPrice : undefined,
      takeProfitPrice:
        !value.targetPositionId && value.takeProfitPrice > 0 ? value.takeProfitPrice : undefined,
    });
    setSubmitting(false);
    if (ok) {
      setDraft(undefined);
    }
  };

  return (
    <section class="panel min-w-0 overflow-hidden">
      <div class="mb-4 flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <div class="muted-label">Positions</div>
          <h2 class="text-lg font-semibold">Break-even Ledger</h2>
        </div>
        <div class="flex flex-wrap items-center gap-2">
          <div class="mr-1 text-sm text-ink-300">
            Fee + slip {formatPercent((summary()?.feeAndSlippageRate ?? 0) * 100)}
          </div>
          <div class="grid grid-cols-2 gap-1 rounded-2 border border-line bg-ink-900 p-1">
            <For each={lotViewModes}>
              {(mode) => (
                <button
                  class="rounded-2 px-2.5 py-1.5 text-sm font-semibold transition"
                  classList={{
                    "bg-accent text-ink-950": lotViewMode() === mode.value,
                    "text-ink-300 hover:text-ink-100": lotViewMode() !== mode.value,
                  }}
                  onClick={() => setLotViewMode(mode.value)}
                  type="button"
                >
                  {mode.label}
                </button>
              )}
            </For>
          </div>
          <button
            class="btn"
            onClick={() =>
              openDraft({
                title: "Add Long",
                side: "buy",
                quantity: 0,
                quoteAmount: 0,
                price: currentPrice(),
                priceMode: "current",
                positionEffect: "open",
                lifetimeMinutes: 0,
                stopLossPrice: 0,
                takeProfitPrice: 0,
              })
            }
            type="button"
          >
            <Plus size={16} />
            Long
          </button>
          <button
            class="btn"
            onClick={() =>
              openDraft({
                title: "Add Short",
                side: "sell",
                quantity: 0,
                quoteAmount: 0,
                price: currentPrice(),
                priceMode: "current",
                positionEffect: "open",
                lifetimeMinutes: 0,
                stopLossPrice: 0,
                takeProfitPrice: 0,
              })
            }
            type="button"
          >
            <Plus size={16} />
            Short
          </button>
        </div>
      </div>

      <Show when={props.error}>
        {(message) => <div class="mb-3 rounded-2 bg-loss/12 p-3 text-sm text-loss">{message()}</div>}
      </Show>

      <div class="mb-4 grid grid-cols-2 gap-3 lg:grid-cols-4 xl:grid-cols-8">
        <SmallMetric label="Net Sell" value={`$${formatQuote(summary()?.netMarketSellPrice, 2)}`} />
        <SmallMetric label="Gross Buy" value={`$${formatQuote(summary()?.grossMarketBuyPrice, 2)}`} />
        <SmallMetric label="Gross Exp" value={`$${formatQuote(summary()?.grossExposureQuote, 2)}`} />
        <SmallMetric label="Eff Lev" value={formatLeverage(summary()?.effectiveLeverage)} />
        <SmallMetric label="Long Left" value={formatAsset(summary()?.longQuantity)} />
        <SmallMetric label="Short Left" value={formatAsset(summary()?.shortQuantity)} />
        <SmallMetric
          label="Base Debt"
          value={`${formatAsset(summary()?.externalBorrowedBaseQuantity)} ${props.baseAsset}`}
        />
        <SmallMetric label="Debt Value" value={`$${formatQuote(summary()?.externalBorrowedQuote, 2)}`} />
      </div>

      <Show when={draft()}>
        {(value) => (
          <ManualTradeForm
            draft={value()}
            baseAsset={props.baseAsset}
            quoteAsset={props.quoteAsset}
            currentPrice={currentPrice()}
            submitting={submitting()}
            onChange={setDraft}
            onCancel={() => setDraft(undefined)}
            onSubmit={() => void submitDraft()}
          />
        )}
      </Show>

      <Show
        when={lotViewMode() === "tree"}
        fallback={
          <div class="grid min-w-0 grid-cols-1 gap-4 2xl:grid-cols-2">
            <PositionLongTable
              lots={longs()}
              baseAsset={props.baseAsset}
              quoteAsset={props.quoteAsset}
              onClose={(lot) =>
                openDraft({
                  title: "Close Long",
                  side: "sell",
                  quantity: lot.remainingQuantity,
                  quoteAmount: lot.remainingQuantity * currentPrice(),
                  price: currentPrice(),
                  priceMode: "current",
                  targetPositionId: lot.id,
                  positionEffect: "close",
                  lifetimeMinutes: 0,
                  stopLossPrice: 0,
                  takeProfitPrice: 0,
                })
              }
            />
            <PositionShortTable
              lots={shorts()}
              baseAsset={props.baseAsset}
              quoteAsset={props.quoteAsset}
              onClose={(lot) =>
                openDraft({
                  title: "Close Short",
                  side: "buy",
                  quantity: lot.remainingQuantity,
                  quoteAmount: lot.remainingQuantity * currentPrice(),
                  price: currentPrice(),
                  priceMode: "current",
                  targetPositionId: lot.id,
                  positionEffect: "close",
                  lifetimeMinutes: 0,
                  stopLossPrice: 0,
                  takeProfitPrice: 0,
                })
              }
            />
          </div>
        }
      >
        <LotTreeView
          longs={longs()}
          shorts={shorts()}
          baseAsset={props.baseAsset}
          quoteAsset={props.quoteAsset}
        />
      </Show>
    </section>
  );
}

type LotViewMode = "tables" | "tree";

const lotViewModes: Array<{ value: LotViewMode; label: string }> = [
  { value: "tables", label: "Tables" },
  { value: "tree", label: "Tree" },
];

type ManualTradePriceMode = "current" | "limit";

type ManualTradeDraft = {
  title: string;
  side: ManualTradeInput["side"];
  price: number;
  priceMode: ManualTradePriceMode;
  quantity: number;
  quoteAmount: number;
  targetPositionId?: string;
  positionEffect: NonNullable<ManualTradeInput["positionEffect"]>;
  lifetimeMinutes: number;
  stopLossPrice: number;
  takeProfitPrice: number;
};

function manualTradePrice(draft: ManualTradeDraft, currentPrice: number): number {
  return draft.priceMode === "current" ? currentPrice : draft.price;
}

function manualTradeQuantity(draft: ManualTradeDraft, currentPrice: number): number {
  if (draft.targetPositionId) {
    return draft.quantity;
  }

  const price = manualTradePrice(draft, currentPrice);
  if (price <= 0) {
    return 0;
  }

  return draft.quoteAmount / price;
}

function ManualTradeForm(props: {
  draft: ManualTradeDraft;
  baseAsset: string;
  quoteAsset: string;
  currentPrice: number;
  submitting: boolean;
  onChange: (draft: ManualTradeDraft) => void;
  onCancel: () => void;
  onSubmit: () => void;
}) {
  const update = <K extends keyof ManualTradeDraft>(key: K, value: ManualTradeDraft[K]) => {
    props.onChange({
      ...props.draft,
      [key]: value,
    });
  };
  const isClose = () => Boolean(props.draft.targetPositionId);
  const effectivePrice = () => manualTradePrice(props.draft, props.currentPrice);
  const estimatedQuantity = () => manualTradeQuantity(props.draft, props.currentPrice);
  const estimatedQuote = () => estimatedQuantity() * effectivePrice();
  const actionVerb = () => (props.draft.side === "buy" ? "Buy" : "Sell");
  const canSubmit = () =>
    !props.submitting &&
    Number.isFinite(effectivePrice()) &&
    effectivePrice() > 0 &&
    Number.isFinite(estimatedQuantity()) &&
    estimatedQuantity() > 0;
  const setPriceMode = (priceMode: ManualTradePriceMode) => {
    props.onChange({
      ...props.draft,
      priceMode,
      price: priceMode === "limit" && props.draft.price > 0 ? props.draft.price : props.currentPrice,
    });
  };

  return (
    <form
      class="mb-4 rounded-2 bg-ink-800 p-3"
      onSubmit={(event) => {
        event.preventDefault();
        props.onSubmit();
      }}
    >
      <div class="mb-3 flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <div class="muted-label">Manual Fill</div>
          <div class="flex items-center gap-2">
            <h3 class="text-base font-semibold">{props.draft.title}</h3>
            <Side side={props.draft.side} />
          </div>
        </div>
        <div class="flex flex-wrap gap-2">
          <button class={buttonPrimaryClass} type="submit" disabled={!canSubmit()}>
            <Check size={16} />
            Record
          </button>
          <button class="btn" type="button" onClick={props.onCancel}>
            <X size={16} />
            Cancel
          </button>
        </div>
      </div>
      <div class="grid grid-cols-1 gap-3 md:grid-cols-3">
        <Show
          when={!isClose()}
          fallback={
            <div class="block">
              <span class="muted-label">{props.baseAsset} Amount</span>
              <div class="mt-1 rounded-2 border border-line bg-ink-900 px-2 py-2 text-sm text-ink-100 tabular-nums">
                {formatAsset(props.draft.quantity)}
              </div>
            </div>
          }
        >
          <NumberField
            label={`${props.quoteAsset} Amount`}
            value={props.draft.quoteAmount}
            min={0}
            step={0.01}
            onInput={(value) => update("quoteAmount", value)}
          />
        </Show>
        <label class="block">
          <span class="muted-label">Price</span>
          <div class="mt-1 grid grid-cols-2 gap-1 rounded-2 border border-line bg-ink-900 p-1">
            <button
              class="rounded-2 px-2 py-1.5 text-sm font-semibold transition"
              classList={{
                "bg-accent text-ink-950": props.draft.priceMode === "current",
                "text-ink-300 hover:text-ink-100": props.draft.priceMode !== "current",
              }}
              onClick={() => setPriceMode("current")}
              type="button"
            >
              Current
            </button>
            <button
              class="rounded-2 px-2 py-1.5 text-sm font-semibold transition"
              classList={{
                "bg-accent text-ink-950": props.draft.priceMode === "limit",
                "text-ink-300 hover:text-ink-100": props.draft.priceMode !== "limit",
              }}
              onClick={() => setPriceMode("limit")}
              type="button"
            >
              Limit
            </button>
          </div>
        </label>
        <Show
          when={props.draft.priceMode === "limit"}
          fallback={
            <div class="block">
              <span class="muted-label">{props.quoteAsset} Current</span>
              <div class="mt-1 rounded-2 border border-line bg-ink-900 px-2 py-2 text-sm text-ink-100 tabular-nums">
                {formatQuote(props.currentPrice, 4)}
              </div>
            </div>
          }
        >
          <NumberField
            label={`${props.quoteAsset} Limit`}
            value={props.draft.price}
            min={0}
            step={0.01}
            onInput={(value) => update("price", value)}
          />
        </Show>
      </div>
      <Show when={!isClose()}>
        <div class="mt-3 grid grid-cols-1 gap-3 md:grid-cols-3">
          <NumberField
            label="Lifetime Min"
            value={props.draft.lifetimeMinutes}
            min={0}
            step={1}
            onInput={(value) => update("lifetimeMinutes", value)}
          />
          <NumberField
            label="Stop Loss"
            value={props.draft.stopLossPrice}
            min={0}
            step={0.01}
            onInput={(value) => update("stopLossPrice", value)}
          />
          <NumberField
            label="Take Profit"
            value={props.draft.takeProfitPrice}
            min={0}
            step={0.01}
            onInput={(value) => update("takeProfitPrice", value)}
          />
        </div>
      </Show>
      <div class="mt-3 rounded-2 border border-line bg-ink-900 p-3">
        <div class="muted-label">{actionVerb()} Preview</div>
        <div class="mt-1 text-base font-semibold tabular-nums text-ink-100">
          {formatAsset(estimatedQuantity())} {props.baseAsset}
        </div>
        <div class="mt-1 text-sm text-ink-300 tabular-nums">
          at {formatQuote(effectivePrice(), 4)} {props.quoteAsset} for {formatQuote(estimatedQuote(), 2)}{" "}
          {props.quoteAsset}
        </div>
      </div>
    </form>
  );
}

function LotTreeView(props: {
  longs: LongPositionLot[];
  shorts: ShortPositionLot[];
  baseAsset: string;
  quoteAsset: string;
}) {
  const lots = () =>
    [...props.longs, ...props.shorts].sort((left, right) => right.openedAt - left.openedAt);

  return (
    <div class="mb-4 rounded-2 bg-ink-800 p-3">
      <div class="mb-3 flex items-center justify-between gap-3">
        <div>
          <div class="muted-label">Lot Tree</div>
          <h3 class="text-base font-semibold">Position Structure</h3>
        </div>
        <div class="text-sm text-ink-300">
          {formatQuote(props.longs.length + props.shorts.length, 0)} lots
        </div>
      </div>
      <div class="grid grid-cols-1 gap-2">
        <For each={lots()} fallback={<div class="text-sm text-ink-300">No lots yet</div>}>
          {(lot) => (
            <details class="rounded-2 border border-line bg-ink-900 p-3" open>
              <summary class="cursor-pointer list-none">
                <div class="flex flex-wrap items-center justify-between gap-2">
                  <div class="flex min-w-0 items-center gap-2">
                    <span
                      class="rounded-2 px-2 py-1 text-xs font-semibold uppercase"
                      classList={{
                        "bg-gain/12 text-gain": lot.side === "long",
                        "bg-loss/12 text-loss": lot.side === "short",
                      }}
                    >
                      {lot.side}
                    </span>
                    <StatusBadge status={lot.status} />
                    <span class="truncate text-sm font-semibold text-ink-100">{lot.id}</span>
                  </div>
                  <div class="text-sm tabular-nums text-ink-300">
                    {formatAsset(lot.remainingQuantity)} {props.baseAsset} ·{" "}
                    {formatLeverage(lot.leverage)}
                  </div>
                </div>
              </summary>
              <div class="mt-3 border-l border-line pl-3">
                <LotTreeBranch label="Opened" value={formatDateTime(lot.openedAt)} />
                <LotTreeBranch
                  label="Entry"
                  value={`$${formatQuote(lot.averagePrice, 4)} · ${formatAsset(
                    lot.originalQuantity,
                  )} ${props.baseAsset}`}
                />
                <LotTreeBranch
                  label="Exposure"
                  value={`$${formatQuote(lot.exposureQuote, 2)} ${props.quoteAsset}`}
                />
                <LotTreeBranch
                  label="Break-even"
                  value={`$${formatQuote(lotBreakEvenPrice(lot), 4)}`}
                />
                <LotTreeBranch
                  label="Max-loss"
                  value={`$${formatQuote(lotMaxLossPrice(lot), 4)}`}
                />
                <LotTreeBranch
                  label="Recommended"
                  value={lotRecommendedAction(lot, props.quoteAsset)}
                />
                <LotTreeBranch
                  label="Borrow"
                  value={lotBorrowLabel(lot, props.baseAsset, props.quoteAsset)}
                />
                <Show when={lot.pendingQuantity > 0}>
                  <LotTreeBranch
                    label="Pending close"
                    value={`${formatAsset(lot.pendingQuantity)} @ $${formatQuote(
                      lot.pendingLimitPrice,
                      4,
                    )}`}
                  />
                </Show>
                <Show when={lot.closedQuantity > 0}>
                  <LotTreeBranch
                    label="Closed"
                    value={`${formatAsset(lot.closedQuantity)} for $${formatQuote(
                      lot.closedQuote,
                      2,
                    )}`}
                  />
                </Show>
                <Show when={lot.lifetimeMs || lot.stopLossPrice || lot.takeProfitPrice || lot.borrowLocked}>
                  <div class="mt-2">
                    <LotRulesCell lot={lot} />
                  </div>
                </Show>
              </div>
            </details>
          )}
        </For>
      </div>
    </div>
  );
}

function LotTreeBranch(props: { label: string; value: string }) {
  return (
    <div class="grid grid-cols-[110px_minmax(0,1fr)] gap-3 border-t border-line py-2 text-sm">
      <div class="text-ink-300">{props.label}</div>
      <div class="min-w-0 break-words text-ink-100 tabular-nums">{props.value}</div>
    </div>
  );
}

function lotBreakEvenPrice(lot: LongPositionLot | ShortPositionLot): number {
  return lot.side === "long" ? lot.breakEvenSellPrice : lot.breakEvenBuyPrice;
}

function lotMaxLossPrice(lot: LongPositionLot | ShortPositionLot): number {
  return lot.side === "long" ? lot.maxLossSellPrice : lot.maxLossBuyPrice;
}

function lotRecommendedAction(
  lot: LongPositionLot | ShortPositionLot,
  quoteAsset: string,
): string {
  if (lot.side === "long") {
    return `${formatAsset(lot.recommendedSellQuantity)} sell · ${formatQuote(
      lot.recommendedSellQuote,
      2,
    )} ${quoteAsset}`;
  }

  return `${formatAsset(lot.recommendedBuyQuantity)} buy · ${formatQuote(
    lot.recommendedBuyQuote,
    2,
  )} ${quoteAsset}`;
}

function lotBorrowLabel(
  lot: LongPositionLot | ShortPositionLot,
  baseAsset: string,
  quoteAsset: string,
): string {
  if (lot.borrowedQuote <= 0 && lot.borrowedQuantity <= 0) {
    return "-";
  }
  if (lot.side === "long") {
    return `${formatQuote(lot.borrowedQuote, 2)} ${quoteAsset} · int ${formatQuote(
      lot.internalBorrowedQuote,
      2,
    )} / ext ${formatQuote(lot.externalBorrowedQuote, 2)}`;
  }

  return `${formatAsset(lot.borrowedQuantity)} ${baseAsset} · int ${formatAsset(
    lot.internalBorrowedQuantity,
  )} / ext ${formatAsset(lot.externalBorrowedQuantity)}`;
}

function PositionLongTable(props: {
  lots: LongPositionLot[];
  baseAsset: string;
  quoteAsset: string;
  onClose: (lot: LongPositionLot) => void;
}) {
  return (
    <div class="min-w-0 overflow-hidden rounded-2 bg-ink-800 p-3">
      <div class="mb-3">
        <div class="muted-label">Long Lots</div>
        <h3 class="text-base font-semibold">Buy Positions</h3>
      </div>
      <div class="max-w-full overflow-x-auto">
        <table class="min-w-full w-max">
          <thead>
            <tr>
              <th class="table-head pb-2">Status</th>
              <th class="table-head pb-2">Order</th>
              <th class="table-head pb-2">Left/Bought</th>
              <th class="table-head pb-2">Lev</th>
              <th class="table-head pb-2">Borrowed</th>
              <th class="table-head pb-2">Pending</th>
              <th class="table-head pb-2">Closed</th>
              <th class="table-head pb-2">Avg</th>
              <th class="table-head pb-2">Break Even</th>
              <th class="table-head pb-2">Max Loss</th>
              <th class="table-head pb-2">Sell Now</th>
              <th class="table-head pb-2">Possible</th>
              <th class="table-head pb-2">Rules</th>
              <th class="table-head pb-2">Actions</th>
            </tr>
          </thead>
          <tbody>
            <For each={props.lots} fallback={<EmptyRow columns={14} label="No long lots" />}>
              {(lot) => (
                <tr>
                  <td class="td-cell">
                    <StatusBadge status={lot.status} />
                  </td>
                  <td class="td-cell text-ink-300">{lot.sourceOrderId}</td>
                  <td class="td-cell">
                    <QuantityValueRatioCell
                      quantity={lot.remainingQuantity}
                      totalQuantity={lot.filledQuantity || lot.originalQuantity}
                      quote={lot.remainingCostQuote}
                      totalQuote={lot.costQuote}
                      quoteAsset={props.quoteAsset}
                    />
                  </td>
                  <td class="td-cell">
                    <LeverageCell leverage={lot.leverage} />
                  </td>
                  <td class="td-cell">
                    <BorrowedCell
                      lot={lot}
                      baseAsset={props.baseAsset}
                      quoteAsset={props.quoteAsset}
                    />
                  </td>
                  <td class="td-cell">
                    <PendingCell
                      quantity={lot.pendingQuantity}
                      quote={lot.pendingQuote}
                      price={lot.pendingLimitPrice}
                      quoteAsset={props.quoteAsset}
                    />
                  </td>
                  <td class="td-cell">
                    <ActionAmount quantity={lot.closedQuantity} quote={lot.closedQuote} />
                  </td>
                  <td class="td-cell">${formatQuote(lot.averagePrice, 4)}</td>
                  <td class="td-cell">${formatQuote(lot.breakEvenSellPrice, 4)}</td>
                  <td class="td-cell">${formatQuote(lot.maxLossSellPrice, 4)}</td>
                  <td class="td-cell">
                    <ActionAmount quantity={lot.recommendedSellQuantity} quote={lot.recommendedSellQuote} />
                  </td>
                  <td class="td-cell">
                    <PossibleBadge possible={lot.canReachLowerBaseline} />
                  </td>
                  <td class="td-cell">
                    <LotRulesCell lot={lot} />
                  </td>
                  <td class="td-cell">
                    <button
                      class="btn px-2 py-1 text-xs"
                      disabled={lot.status === "pending" || lot.remainingQuantity <= 0}
                      onClick={() => props.onClose(lot)}
                      type="button"
                    >
                      <MinusCircle size={14} />
                      Close
                    </button>
                  </td>
                </tr>
              )}
            </For>
          </tbody>
        </table>
      </div>
    </div>
  );
}

function PositionShortTable(props: {
  lots: ShortPositionLot[];
  baseAsset: string;
  quoteAsset: string;
  onClose: (lot: ShortPositionLot) => void;
}) {
  return (
    <div class="min-w-0 overflow-hidden rounded-2 bg-ink-800 p-3">
      <div class="mb-3">
        <div class="muted-label">Short Lots</div>
        <h3 class="text-base font-semibold">Sell Positions</h3>
      </div>
      <div class="max-w-full overflow-x-auto">
        <table class="min-w-full w-max">
          <thead>
            <tr>
              <th class="table-head pb-2">Status</th>
              <th class="table-head pb-2">Order</th>
              <th class="table-head pb-2">Left/Sold</th>
              <th class="table-head pb-2">Lev</th>
              <th class="table-head pb-2">Borrowed</th>
              <th class="table-head pb-2">Pending</th>
              <th class="table-head pb-2">Closed</th>
              <th class="table-head pb-2">Avg</th>
              <th class="table-head pb-2">Break Even</th>
              <th class="table-head pb-2">Max Loss</th>
              <th class="table-head pb-2">Buy Now</th>
              <th class="table-head pb-2">Possible</th>
              <th class="table-head pb-2">Rules</th>
              <th class="table-head pb-2">Actions</th>
            </tr>
          </thead>
          <tbody>
            <For each={props.lots} fallback={<EmptyRow columns={14} label="No short lots" />}>
              {(lot) => (
                <tr>
                  <td class="td-cell">
                    <StatusBadge status={lot.status} />
                  </td>
                  <td class="td-cell text-ink-300">{lot.sourceOrderId}</td>
                  <td class="td-cell">
                    <QuantityValueRatioCell
                      quantity={lot.remainingQuantity}
                      totalQuantity={lot.filledQuantity || lot.originalQuantity}
                      quote={lot.remainingProceedsQuote}
                      totalQuote={lot.proceedsQuote}
                      quoteAsset={props.quoteAsset}
                    />
                  </td>
                  <td class="td-cell">
                    <LeverageCell leverage={lot.leverage} />
                  </td>
                  <td class="td-cell">
                    <BorrowedCell
                      lot={lot}
                      baseAsset={props.baseAsset}
                      quoteAsset={props.quoteAsset}
                    />
                  </td>
                  <td class="td-cell">
                    <PendingCell
                      quantity={lot.pendingQuantity}
                      quote={lot.pendingQuote}
                      price={lot.pendingLimitPrice}
                      quoteAsset={props.quoteAsset}
                    />
                  </td>
                  <td class="td-cell">
                    <ActionAmount quantity={lot.closedQuantity} quote={lot.closedQuote} />
                  </td>
                  <td class="td-cell">${formatQuote(lot.averagePrice, 4)}</td>
                  <td class="td-cell">${formatQuote(lot.breakEvenBuyPrice, 4)}</td>
                  <td class="td-cell">${formatQuote(lot.maxLossBuyPrice, 4)}</td>
                  <td class="td-cell">
                    <ActionAmount quantity={lot.recommendedBuyQuantity} quote={lot.recommendedBuyQuote} />
                  </td>
                  <td class="td-cell">
                    <PossibleBadge possible={lot.canReachUpperBaseline} />
                  </td>
                  <td class="td-cell">
                    <LotRulesCell lot={lot} />
                  </td>
                  <td class="td-cell">
                    <button
                      class="btn px-2 py-1 text-xs"
                      disabled={lot.status === "pending" || lot.remainingQuantity <= 0}
                      onClick={() => props.onClose(lot)}
                      type="button"
                    >
                      <MinusCircle size={14} />
                      Close
                    </button>
                  </td>
                </tr>
              )}
            </For>
          </tbody>
        </table>
      </div>
    </div>
  );
}

function LotRulesCell(props: { lot: LongPositionLot | ShortPositionLot }) {
  if (
    !props.lot.borrowLocked &&
    !props.lot.lifetimeMs &&
    !props.lot.stopLossPrice &&
    !props.lot.takeProfitPrice
  ) {
    return <span class="text-ink-300">-</span>;
  }

  return (
    <div class="flex max-w-52 flex-wrap gap-1">
      <Show when={props.lot.borrowLocked}>
        <span class="rounded-2 bg-warn/12 px-2 py-1 text-xs font-semibold uppercase text-warn">
          no borrow
        </span>
      </Show>
      <Show when={props.lot.lifetimeMs}>
        <span class="rounded-2 bg-ink-700 px-2 py-1 text-xs text-ink-100">
          Life {formatDuration(props.lot.lifetimeMs)}
        </span>
      </Show>
      <Show when={props.lot.expiresAt}>
        <span class="rounded-2 bg-ink-700 px-2 py-1 text-xs text-ink-100">
          Exp {formatTime(props.lot.expiresAt)}
        </span>
      </Show>
      <Show when={props.lot.stopLossPrice}>
        <span class="rounded-2 bg-loss/12 px-2 py-1 text-xs font-semibold text-loss">
          SL ${formatQuote(props.lot.stopLossPrice, 4)}
        </span>
      </Show>
      <Show when={props.lot.takeProfitPrice}>
        <span class="rounded-2 bg-gain/12 px-2 py-1 text-xs font-semibold text-gain">
          TP ${formatQuote(props.lot.takeProfitPrice, 4)}
        </span>
      </Show>
    </div>
  );
}

function LeverageCell(props: { leverage: number }) {
  return <span class="font-semibold tabular-nums text-ink-100">{formatLeverage(props.leverage)}</span>;
}

function BorrowedCell(props: {
  lot: LongPositionLot | ShortPositionLot;
  baseAsset: string;
  quoteAsset: string;
}) {
  if (props.lot.borrowedQuote <= 0 && props.lot.borrowedQuantity <= 0) {
    return <span class="text-ink-300">-</span>;
  }

  if (props.lot.side === "long") {
    return (
      <div class="tabular-nums">
        <div class="whitespace-nowrap">
          {formatQuote(props.lot.borrowedQuote, 2)} {props.quoteAsset}
        </div>
        <div class="mt-1 whitespace-nowrap text-xs text-ink-300">
          int {formatQuote(props.lot.internalBorrowedQuote, 2)} / ext{" "}
          {formatQuote(props.lot.externalBorrowedQuote, 2)}
          <Show when={props.lot.borrowedFromPositionCount > 0}>
            {" "}
            ({props.lot.borrowedFromPositionCount} pos)
          </Show>
        </div>
      </div>
    );
  }

  return (
    <div class="tabular-nums">
      <div class="whitespace-nowrap">
        {formatAsset(props.lot.borrowedQuantity)} {props.baseAsset}
      </div>
      <div class="mt-1 whitespace-nowrap text-xs text-ink-300">
        int {formatAsset(props.lot.internalBorrowedQuantity)} / ext{" "}
        {formatAsset(props.lot.externalBorrowedQuantity)}
        <Show when={props.lot.borrowedFromPositionCount > 0}>
          {" "}
          ({props.lot.borrowedFromPositionCount} pos)
        </Show>
      </div>
    </div>
  );
}

function QuantityValueRatioCell(props: {
  quantity: number;
  totalQuantity: number;
  quote: number;
  totalQuote: number;
  quoteAsset: string;
}) {
  return (
    <div class="tabular-nums">
      <div class="whitespace-nowrap">
        {formatAsset(props.quantity)} / {formatAsset(props.totalQuantity)}
      </div>
      <div class="mt-1 whitespace-nowrap text-xs text-ink-300">
        {formatQuote(props.quote, 2)} / {formatQuote(props.totalQuote, 2)} {props.quoteAsset}
      </div>
    </div>
  );
}

function PendingCell(props: {
  quantity: number;
  quote: number;
  price: number;
  quoteAsset: string;
}) {
  if (props.quantity <= 0) {
    return <span class="text-ink-300">-</span>;
  }

  return (
    <span class="whitespace-nowrap">
      {formatAsset(props.quantity)} @ ${formatQuote(props.price, 4)}
      <span class="ml-1 text-ink-300">
        {formatQuote(props.quote, 2)} {props.quoteAsset}
      </span>
    </span>
  );
}

function ActionAmount(props: { quantity: number; quote: number }) {
  if (props.quantity <= 0 || props.quote <= 0) {
    return <span class="text-ink-300">-</span>;
  }

  return (
    <span class="whitespace-nowrap">
      {formatAsset(props.quantity)}
      <span class="ml-1 text-ink-300">${formatQuote(props.quote, 2)}</span>
    </span>
  );
}

function StatusBadge(props: { status: string }) {
  return (
    <span
      class="rounded-2 px-2 py-1 text-xs font-semibold uppercase"
      classList={{
        "bg-accent/12 text-accent": props.status === "open",
        "bg-warn/12 text-warn": props.status === "pending",
        "bg-gain/12 text-gain": props.status === "closed",
        "bg-ink-700 text-ink-200": props.status === "partially-closed",
      }}
    >
      {props.status.replace("-", " ")}
    </span>
  );
}

function PossibleBadge(props: { possible: boolean }) {
  return (
    <span
      class="rounded-2 px-2 py-1 text-xs font-semibold uppercase"
      classList={{
        "bg-gain/12 text-gain": props.possible,
        "bg-loss/12 text-loss": !props.possible,
      }}
    >
      {props.possible ? "yes" : "no"}
    </span>
  );
}

function EventsPanel(props: { events: RuntimeSnapshot["recentEvents"] }) {
  return (
    <div class="panel">
      <div class="mb-3">
        <div class="muted-label">Bot Log</div>
        <h2 class="text-lg font-semibold">Recent Events</h2>
      </div>
      <div class="max-h-74 overflow-auto pr-1">
        <For each={props.events} fallback={<div class="text-sm text-ink-300">No bot events yet</div>}>
          {(event) => (
            <div class="border-t border-line py-2">
              <div class="flex items-center justify-between gap-3">
                <span class="text-sm text-ink-100">{event.message}</span>
                <span class="shrink-0 text-xs text-ink-300">{formatTime(event.at)}</span>
              </div>
              <Show when={event.order}>
                {(order) => (
                  <div class="mt-1 text-xs text-ink-300">
                    {order().id} at ${formatQuote(order().price ?? order().stopPrice ?? undefined, 2)}
                  </div>
                )}
              </Show>
            </div>
          )}
        </For>
      </div>
    </div>
  );
}

function BacktestPanel(props: {
  preset: BacktestSelection;
  onPresetChange: (preset: BacktestSelection) => void;
  settings: BacktestSettings;
  onSettingChange: <K extends keyof BacktestSettings>(
    key: K,
    value: BacktestSettings[K],
  ) => void;
  progress?: BacktestProgressSnapshot;
  error?: string;
  liveStartAt?: number;
  onRun: () => void;
  onRunFromLiveStart: () => void;
  onStop: () => void;
}) {
  const result = () => props.progress?.result;
  const summary = () => result()?.summary;
  const extremaOrderMass = () => summary()?.extremaOrderMass;
  const isRunning = () => props.progress?.status === "running";
  const progressPercent = () => Math.max(0, Math.min(100, props.progress?.percent ?? 0));
  const cacheMissCandles = () =>
    props.progress?.cacheMissCandles ?? summary()?.cacheMissCandles ?? 0;
  const cacheFetchedCandles = () =>
    props.progress?.cacheFetchedCandles ?? summary()?.cacheFetchedCandles ?? 0;
  const cacheProgressPercent = () => {
    const missing = cacheMissCandles();
    if (missing <= 0) {
      return undefined;
    }

    return Math.max(0, Math.min(100, (cacheFetchedCandles() / missing) * 100));
  };
  const showCacheProgress = () =>
    props.progress?.status === "running" &&
    cacheProgressPercent() !== undefined &&
    cacheFetchedCandles() < cacheMissCandles();
  const formatExtremaThresholdMass = (
    side: BacktestExtremaOrderMassSideSummary | undefined,
  ) =>
    side
      ? `${formatPercent(side.thresholdMassPct)} / $${formatQuote(side.thresholdQuote, 0)}`
      : "-";
  const formatExtremaAverageDistance = (
    side: BacktestExtremaOrderMassSideSummary | undefined,
  ) =>
    side
      ? `${formatDuration(side.weightedAvgTimeDistanceMs)} / ${formatPercent(
          side.weightedAvgPriceDistancePct,
        )}`
      : "-";
  const formatExtremaP99Frame = (
    side: BacktestExtremaOrderMassSideSummary | undefined,
  ) =>
    side
      ? `${formatDuration(side.massP99JointTimeDistanceMs)} / ${formatPercent(
          side.massP99JointPriceDistancePct,
        )}`
      : "-";
  const error = () => props.error ?? props.progress?.error;
  const canRunFromLiveStart = () =>
    Boolean(
      props.liveStartAt &&
        (props.preset === "last-x" ||
          props.preset === "week" ||
          props.preset === "month" ||
          props.preset === "year"),
    );
  const processedLabel = () => {
    const processed = props.progress?.processedCandles ?? summary()?.candlesProcessed ?? 0;
    const estimated = props.progress?.estimatedCandles ?? 0;
    return estimated > 0
      ? `${formatQuote(processed, 0)} / ${formatQuote(estimated, 0)}`
      : formatQuote(processed, 0);
  };

  return (
    <section class="panel">
      <div class="mb-4 flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <div class="muted-label">Backtest</div>
          <h2 class="text-lg font-semibold">Market Replay</h2>
        </div>
        <div class="flex flex-wrap gap-2">
          <button
            class={buttonPanelClass}
            disabled={isRunning() || !canRunFromLiveStart()}
            onClick={props.onRunFromLiveStart}
            title={
              canRunFromLiveStart()
                ? "Run from live bot start date to now"
                : "Enable with a live bot and Last X / week / month / year preset"
            }
            type="button"
          >
            <Play size={16} />
            From Live Start
          </button>
          <select
            class="rounded-2 border border-line bg-ink-800 px-3 py-2 text-sm text-ink-100"
            value={props.preset}
            onInput={(event) =>
              props.onPresetChange(event.currentTarget.value as BacktestSelection)
            }
            disabled={isRunning()}
          >
            <option value="saved-candles">Saved candles</option>
            <option value="saved-orderbook">Saved order book</option>
            <option value="last-x">Last X days</option>
            <option value="week">Last week</option>
            <option value="month">Last month</option>
            <option value="year">Last year</option>
            <option value="random-windows">Random weeks</option>
            <option value="random-length-windows">Random lengths</option>
          </select>
          <button
            class={buttonPrimaryClass}
            disabled={isRunning()}
            onClick={props.onRun}
            type="button"
          >
            <RefreshCw size={16} class={isRunning() ? "animate-spin" : ""} />
            Run
          </button>
          <Show when={isRunning()}>
            <button
              class={buttonDangerClass}
              onClick={props.onStop}
              type="button"
            >
              <Square size={16} />
              Stop
            </button>
          </Show>
        </div>
      </div>

      <Show when={props.preset === "last-x"}>
        <div class="mb-4 grid grid-cols-1 gap-3 rounded-2 bg-ink-800 p-3 sm:grid-cols-3">
          <BacktestNumberInput
            label="Days"
            value={props.settings.historicalDays}
            min={1}
            max={3650}
            disabled={isRunning()}
            onChange={(value) => props.onSettingChange("historicalDays", value)}
          />
        </div>
      </Show>

      <div class="mb-4 grid grid-cols-1 gap-3 rounded-2 bg-ink-800 p-3 sm:max-w-xs">
        <BacktestNumberInput
          label="Extrema SMA Minutes"
          value={props.settings.extremaSmaWindowMinutes}
          min={1}
          max={1440}
          disabled={isRunning()}
          onChange={(value) => props.onSettingChange("extremaSmaWindowMinutes", value)}
        />
      </div>

      <Show when={props.preset === "random-windows"}>
        <div class="mb-4 grid grid-cols-1 gap-3 rounded-2 bg-ink-800 p-3 sm:grid-cols-2 xl:grid-cols-4">
          <BacktestNumberInput
            label="Samples"
            value={props.settings.randomSampleCount}
            min={1}
            max={200}
            disabled={isRunning()}
            onChange={(value) => props.onSettingChange("randomSampleCount", value)}
          />
          <BacktestNumberInput
            label="Window Days"
            value={props.settings.randomWindowDays}
            min={1}
            max={365}
            disabled={isRunning()}
            onChange={(value) => props.onSettingChange("randomWindowDays", value)}
          />
          <BacktestNumberInput
            label="Lookback Days"
            value={props.settings.randomLookbackDays}
            min={1}
            max={3650}
            disabled={isRunning()}
            onChange={(value) => props.onSettingChange("randomLookbackDays", value)}
          />
          <BacktestNumberInput
            label="Extra Pairs"
            value={props.settings.randomPairCount}
            min={0}
            max={25}
            disabled={isRunning()}
            onChange={(value) => props.onSettingChange("randomPairCount", value)}
          />
        </div>
      </Show>

      <Show when={props.preset === "random-length-windows"}>
        <div class="mb-4 grid grid-cols-1 gap-3 rounded-2 bg-ink-800 p-3 sm:grid-cols-2 xl:grid-cols-5">
          <BacktestNumberInput
            label="Samples"
            value={props.settings.randomSampleCount}
            min={1}
            max={200}
            disabled={isRunning()}
            onChange={(value) => props.onSettingChange("randomSampleCount", value)}
          />
          <BacktestNumberInput
            label="Min Days"
            value={props.settings.randomMinWindowDays}
            min={1}
            max={365}
            disabled={isRunning()}
            onChange={(value) => props.onSettingChange("randomMinWindowDays", value)}
          />
          <BacktestNumberInput
            label="Max Days"
            value={props.settings.randomMaxWindowDays}
            min={Math.max(1, props.settings.randomMinWindowDays)}
            max={365}
            disabled={isRunning()}
            onChange={(value) => props.onSettingChange("randomMaxWindowDays", value)}
          />
          <BacktestNumberInput
            label="Lookback Days"
            value={props.settings.randomLookbackDays}
            min={Math.max(1, props.settings.randomMaxWindowDays)}
            max={3650}
            disabled={isRunning()}
            onChange={(value) => props.onSettingChange("randomLookbackDays", value)}
          />
          <BacktestNumberInput
            label="Extra Pairs"
            value={props.settings.randomPairCount}
            min={0}
            max={25}
            disabled={isRunning()}
            onChange={(value) => props.onSettingChange("randomPairCount", value)}
          />
        </div>
      </Show>

      <Show when={error()}>
        {(message) => (
          <div class="mb-3 rounded-2 bg-loss/12 p-3 text-sm text-loss">{message()}</div>
        )}
      </Show>

      <Show when={props.progress && props.progress.status !== "idle"}>
        <div class="mb-4 rounded-2 bg-ink-800 p-3">
          <div class="mb-2 flex items-center justify-between gap-3">
            <div class="text-sm text-ink-100">{props.progress?.message}</div>
            <div class="text-sm tabular-nums text-ink-300">{formatPercent(progressPercent())}</div>
          </div>
          <div class="h-2 overflow-hidden rounded-full bg-ink-700">
            <div
              class="h-full bg-accent transition-all"
              style={{ width: `${progressPercent()}%` }}
            />
          </div>
          <Show when={showCacheProgress()}>
            <div class="mt-3">
              <div class="mb-2 flex items-center justify-between gap-3">
                <div class="text-xs uppercase tracking-wide text-ink-300">Candle Cache</div>
                <div class="text-xs tabular-nums text-ink-300">
                  {formatQuote(Math.min(cacheFetchedCandles(), cacheMissCandles()), 0)} /{" "}
                  {formatQuote(cacheMissCandles(), 0)} ·{" "}
                  {formatPercent(cacheProgressPercent())}
                </div>
              </div>
              <div class="h-2 overflow-hidden rounded-full bg-ink-700">
                <div
                  class="h-full bg-warn transition-all"
                  style={{ width: `${cacheProgressPercent() ?? 0}%` }}
                />
              </div>
            </div>
          </Show>
        </div>
      </Show>

      <div class="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,1fr)_360px]">
        <div class="h-48">
          <EquityChart points={result()?.equityCurve ?? []} />
        </div>
        <div class="grid grid-cols-2 gap-3">
          <SmallMetric
            label="Equity"
            value={`$${formatQuote(props.progress?.equity ?? summary()?.finalEquity, 2)}`}
          />
          <SmallMetric
            label="Return"
            value={formatPercent(props.progress?.returnPct ?? summary()?.returnPct)}
          />
          <SmallMetric label="Risk Ret" value={formatRatio(summary()?.riskAdjustedReturn)} />
          <SmallMetric label="Sharpe (Annual)" value={formatRatio(summary()?.sharpeRatio)} />
          <SmallMetric
            label="Sharpe (BT Length)"
            value={formatRatio(summary()?.backtestSharpeRatio)}
          />
          <SmallMetric
            label="Perfect Ret"
            value={formatPercent(summary()?.perfectMarginReturnPct)}
          />
          <SmallMetric
            label="Capture"
            value={formatPercent(summary()?.perfectMarginCapturePct)}
          />
          <Show when={extremaOrderMass()}>
            {(mass) => (
              <>
                <SmallMetric
                  label="Buy Valley Mass"
                  value={formatExtremaThresholdMass(mass().buy)}
                />
                <SmallMetric
                  label="Sell Peak Mass"
                  value={formatExtremaThresholdMass(mass().sell)}
                />
                <SmallMetric
                  label="Buy Avg Dist"
                  value={formatExtremaAverageDistance(mass().buy)}
                />
                <SmallMetric
                  label="Sell Avg Dist"
                  value={formatExtremaAverageDistance(mass().sell)}
                />
                <SmallMetric
                  label="Buy 99% Frame"
                  value={formatExtremaP99Frame(mass().buy)}
                />
                <SmallMetric
                  label="Sell 99% Frame"
                  value={formatExtremaP99Frame(mass().sell)}
                />
              </>
            )}
          </Show>
          <SmallMetric
            label="Perfect PnL"
            value={`$${formatQuote(summary()?.perfectMarginNetPnl, 2)}`}
          />
          <SmallMetric
            label="Reinvest Ret"
            value={formatPercent(summary()?.perfectMarginCompoundedReturnPct)}
          />
          <SmallMetric
            label="Reinvest Cap"
            value={formatPercent(summary()?.perfectMarginCompoundedCapturePct)}
          />
          <SmallMetric
            label="Max Acct Lev"
            value={formatLeverage(summary()?.maxEffectiveLeverage ?? summary()?.perfectMarginLeverage)}
          />
          <SmallMetric label="Trades" value={formatQuote(summary()?.tradeCount, 0)} />
          <SmallMetric label="Win Rate" value={formatPercent(summary()?.winRate)} />
          <SmallMetric
            label="Prof Pos"
            value={`${formatQuote(summary()?.profitableClosedPositionCount, 0)} / ${formatQuote(
              summary()?.closedPositionCount,
              0,
            )}`}
          />
          <SmallMetric
            label="Pos Win"
            value={formatPercent(summary()?.profitableClosedPositionRate)}
          />
          <SmallMetric
            label="Liq Pos"
            value={formatQuote(summary()?.liquidatedPositionCount, 0)}
          />
          <SmallMetric label="Candles" value={processedLabel()} />
          <SmallMetric
            label="Speed"
            value={`${formatQuote(
              props.progress?.candlesPerSecond ?? summary()?.candlesPerSecond,
              0,
            )}/s`}
          />
          <SmallMetric label="Drawdown" value={formatPercent(summary()?.maxDrawdownPct)} />
          <Show when={summary()?.sampleCount ?? props.progress?.sampleCount}>
            <SmallMetric
              label="Samples"
              value={`${formatQuote(
                summary()?.samplesProcessed ?? props.progress?.currentSample,
                0,
              )} / ${formatQuote(summary()?.sampleCount ?? props.progress?.sampleCount, 0)}`}
            />
            <Show when={(summary()?.marketCount ?? props.progress?.marketCount ?? 0) > 1}>
              <SmallMetric
                label="Pairs"
                value={`${formatQuote(
                  summary()?.marketCount ?? props.progress?.marketCount,
                  0,
                )} (${formatQuote(
                  summary()?.randomPairCount ?? props.progress?.randomPairCount,
                  0,
                )} extra)`}
              />
            </Show>
            <SmallMetric label="Best" value={formatPercent(summary()?.bestReturnPct)} />
            <SmallMetric label="Worst" value={formatPercent(summary()?.worstReturnPct)} />
            <SmallMetric
              label="Profit/day"
              value={`$${formatQuote(
                summary()?.netPnlPerDay ?? props.progress?.netPnlPerDay,
                2,
              )}`}
            />
            <SmallMetric
              label="Return/day"
              value={formatPercent(
                summary()?.returnPctPerDay ?? props.progress?.returnPctPerDay,
              )}
            />
            <SmallMetric
              label="Profitable"
              value={`${formatQuote(summary()?.profitableSamples, 0)} / ${formatQuote(
                summary()?.sampleCount,
                0,
              )}`}
            />
          </Show>
          <SmallMetric label="Requests" value={formatQuote(props.progress?.requests, 0)} />
          <SmallMetric
            label="Survived"
            value={formatDuration(props.progress?.survivedMs ?? summary()?.survivedMs)}
          />
          <SmallMetric
            label="Cache Hits"
            value={formatQuote(
              props.progress?.cacheHitCandles ?? summary()?.cacheHitCandles,
              0,
            )}
          />
          <SmallMetric
            label="Fetched"
            value={formatQuote(
              props.progress?.cacheFetchedCandles ?? summary()?.cacheFetchedCandles,
              0,
            )}
          />
          <SmallMetric
            label="Cache Size"
            value={formatBytes(props.progress?.cacheSizeBytes ?? summary()?.cacheSizeBytes)}
          />
          <SmallMetric
            label="Evicted"
            value={formatQuote(
              props.progress?.cacheEvictedFiles ?? summary()?.cacheEvictedFiles,
              0,
            )}
          />
        </div>
      </div>

      <Show when={result()?.candleChart}>
        {(chart) => <BacktestReplayChart chart={chart()} />}
      </Show>

      <Show when={summary()}>
        {(item) => (
          <div class="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-300">
            <span>
              {formatDateTime(item().startTime)} to {formatDateTime(item().endTime)}
            </span>
            <Show when={item().stopReason}>
              <span>{item().stopReason}</span>
            </Show>
            <Show when={item().sampleCount}>
              <span>
                avg of {formatQuote(item().sampleCount, 0)} x{" "}
                {item().sampleWindowMs
                  ? formatDuration(item().sampleWindowMs)
                  : `${formatDuration(item().sampleMinWindowMs)}-${formatDuration(
                      item().sampleMaxWindowMs,
                    )}`}{" "}
                {(item()?.marketCount ?? 0) > 1 ? "market-window samples" : "windows"}
              </span>
            </Show>
          </div>
        )}
      </Show>
    </section>
  );
}

type BacktestReplayMetricKey =
  | "price"
  | "equity"
  | "netPnl"
  | "returnPct"
  | "realizedPnl"
  | "unrealizedPnl"
  | "drawdownPct"
  | "exposurePct"
  | "maxEffectiveLeverage"
  | "feesPaid"
  | "tradeCount"
  | "winRate"
  | "quoteFree"
  | "quoteReserved"
  | "baseFree"
  | "baseReserved"
  | "openOrderCount"
  | "longLotCount"
  | "shortLotCount"
  | "longQuantity"
  | "shortQuantity"
  | "netExposureQuote"
  | "grossExposureQuote"
  | "effectiveLeverage"
  | "longExposureQuote"
  | "shortExposureQuote"
  | "pendingLongQuote"
  | "pendingShortQuote";

interface BacktestReplayMetricDefinition {
  key: BacktestReplayMetricKey;
  label: string;
  group: "account" | "risk" | "balances" | "activity" | "positions";
  color: string;
  value: (frame: BacktestMetricFrame) => number | undefined;
  format: (value: number | undefined) => string;
}

type BacktestMetricFrame = BacktestReplayFrame | BacktestTraceFrame;

const backtestReplayMetrics: BacktestReplayMetricDefinition[] = [
  {
    key: "price",
    label: "Price",
    group: "account",
    color: "#38bdf8",
    value: (frame) => frame.price,
    format: (value) => `$${formatQuote(value, 4)}`,
  },
  {
    key: "equity",
    label: "Equity",
    group: "account",
    color: "#22c55e",
    value: (frame) => frame.metrics.equity,
    format: (value) => `$${formatQuote(value, 2)}`,
  },
  {
    key: "netPnl",
    label: "Net PnL",
    group: "account",
    color: "#f5b84b",
    value: (frame) => frame.metrics.netPnl,
    format: (value) => `$${formatQuote(value, 2)}`,
  },
  {
    key: "returnPct",
    label: "Return",
    group: "account",
    color: "#a78bfa",
    value: (frame) => frame.metrics.returnPct,
    format: formatPercent,
  },
  {
    key: "realizedPnl",
    label: "Realized",
    group: "account",
    color: "#14b8a6",
    value: (frame) => frame.metrics.realizedPnl,
    format: (value) => `$${formatQuote(value, 2)}`,
  },
  {
    key: "unrealizedPnl",
    label: "Unrealized",
    group: "account",
    color: "#eab308",
    value: (frame) => frame.metrics.unrealizedPnl,
    format: (value) => `$${formatQuote(value, 2)}`,
  },
  {
    key: "drawdownPct",
    label: "Drawdown",
    group: "risk",
    color: "#f05252",
    value: (frame) => frame.metrics.maxDrawdownPct,
    format: formatPercent,
  },
  {
    key: "exposurePct",
    label: "Exposure",
    group: "risk",
    color: "#fb7185",
    value: (frame) => frame.metrics.exposurePct,
    format: formatPercent,
  },
  {
    key: "maxEffectiveLeverage",
    label: "Max Eff Lev",
    group: "risk",
    color: "#f97316",
    value: (frame) => frame.metrics.maxEffectiveLeverage,
    format: formatLeverage,
  },
  {
    key: "feesPaid",
    label: "Fees",
    group: "account",
    color: "#94a3b8",
    value: (frame) => frame.metrics.feesPaid,
    format: (value) => `$${formatQuote(value, 2)}`,
  },
  {
    key: "tradeCount",
    label: "Trades",
    group: "activity",
    color: "#60a5fa",
    value: (frame) => frame.metrics.tradeCount,
    format: (value) => formatQuote(value, 0),
  },
  {
    key: "winRate",
    label: "Win Rate",
    group: "activity",
    color: "#4ade80",
    value: (frame) => frame.metrics.winRate,
    format: formatPercent,
  },
  {
    key: "quoteFree",
    label: "Quote Free",
    group: "balances",
    color: "#22c55e",
    value: (frame) => frame.quoteFree,
    format: (value) => `$${formatQuote(value, 2)}`,
  },
  {
    key: "quoteReserved",
    label: "Quote Reserved",
    group: "balances",
    color: "#84cc16",
    value: (frame) => frame.quoteReserved,
    format: (value) => `$${formatQuote(value, 2)}`,
  },
  {
    key: "baseFree",
    label: "Base Free",
    group: "balances",
    color: "#2dd4bf",
    value: (frame) => frame.baseFree,
    format: formatAsset,
  },
  {
    key: "baseReserved",
    label: "Base Reserved",
    group: "balances",
    color: "#67e8f9",
    value: (frame) => frame.baseReserved,
    format: formatAsset,
  },
  {
    key: "openOrderCount",
    label: "Open Orders",
    group: "activity",
    color: "#38bdf8",
    value: (frame) => frame.openOrderCount,
    format: (value) => formatQuote(value, 0),
  },
  {
    key: "longLotCount",
    label: "Long Lots",
    group: "activity",
    color: "#22c55e",
    value: (frame) => frame.longLotCount,
    format: (value) => formatQuote(value, 0),
  },
  {
    key: "shortLotCount",
    label: "Short Lots",
    group: "activity",
    color: "#f05252",
    value: (frame) => frame.shortLotCount,
    format: (value) => formatQuote(value, 0),
  },
  {
    key: "longQuantity",
    label: "Long Qty",
    group: "positions",
    color: "#22c55e",
    value: (frame) => frame.positions.summary.longQuantity,
    format: formatAsset,
  },
  {
    key: "shortQuantity",
    label: "Short Qty",
    group: "positions",
    color: "#f05252",
    value: (frame) => frame.positions.summary.shortQuantity,
    format: formatAsset,
  },
  {
    key: "netExposureQuote",
    label: "Net Exposure",
    group: "positions",
    color: "#f5b84b",
    value: (frame) => frame.positions.summary.netExposureQuote,
    format: (value) => `$${formatQuote(value, 2)}`,
  },
  {
    key: "grossExposureQuote",
    label: "Gross Exposure",
    group: "positions",
    color: "#a78bfa",
    value: (frame) => frame.positions.summary.grossExposureQuote,
    format: (value) => `$${formatQuote(value, 2)}`,
  },
  {
    key: "effectiveLeverage",
    label: "Effective Lev",
    group: "risk",
    color: "#f97316",
    value: (frame) => frame.positions.summary.effectiveLeverage,
    format: formatLeverage,
  },
  {
    key: "longExposureQuote",
    label: "Long Exposure",
    group: "positions",
    color: "#16a34a",
    value: (frame) => frame.positions.summary.longExposureQuote,
    format: (value) => `$${formatQuote(value, 2)}`,
  },
  {
    key: "shortExposureQuote",
    label: "Short Exposure",
    group: "positions",
    color: "#dc2626",
    value: (frame) => frame.positions.summary.shortExposureQuote,
    format: (value) => `$${formatQuote(value, 2)}`,
  },
  {
    key: "pendingLongQuote",
    label: "Pending Long",
    group: "positions",
    color: "#86efac",
    value: (frame) => frame.positions.summary.pendingLongQuote,
    format: (value) => `$${formatQuote(value, 2)}`,
  },
  {
    key: "pendingShortQuote",
    label: "Pending Short",
    group: "positions",
    color: "#fca5a5",
    value: (frame) => frame.positions.summary.pendingShortQuote,
    format: (value) => `$${formatQuote(value, 2)}`,
  },
];

const backtestReplayMetricByKey = new Map(
  backtestReplayMetrics.map((metric) => [metric.key, metric]),
);

function averageSeriesKey(series: BacktestChartSmaSeries): string {
  return `${series.index}:${series.windowSec}:${series.label}`;
}

function BacktestReplayChart(props: { chart: BacktestCandleChart }) {
  const [selectedTime, setSelectedTime] = createSignal<number>();
  const [selectedPositionId, setSelectedPositionId] = createSignal<string>();
  const [selectedExtremumId, setSelectedExtremumId] = createSignal<string>();
  const [hoveredOracle, setHoveredOracle] = createSignal<BacktestOraclePoint>();
  const [hiddenAverageKeys, setHiddenAverageKeys] = createSignal<ReadonlySet<string>>(new Set());
  const [overlays, setOverlays] = createSignal<CandleChartOverlayVisibility>({
    averages: true,
    signals: true,
    orders: true,
    fills: true,
    positions: true,
    extrema: true,
    oracle: true,
  });
  const [chartViewport, setChartViewport] = createSignal<CandleChartViewport>();
  const [selectedMetricKey, setSelectedMetricKey] =
    createSignal<BacktestReplayMetricKey>("equity");
  const [priceDisplay, setPriceDisplay] =
    createSignal<CandleChartPriceDisplay>("candles");
  let chartKey = "";

  createEffect(() => {
    const first = props.chart.candles[0]?.openTime ?? 0;
    const last = props.chart.candles.at(-1)?.closeTime ?? 0;
    const nextKey = `${props.chart.candles.length}:${first}:${last}:${
      props.chart.frames?.length ?? 0
    }:${props.chart.trace?.positions.length ?? 0}:${props.chart.trace?.positions.at(-1)?.id ?? ""}:${
      props.chart.smaSeries.map(averageSeriesKey).join(",")
    }`;
    if (nextKey === chartKey) {
      return;
    }

    chartKey = nextKey;
    setSelectedTime(props.chart.trace?.frames.at(-1)?.time ?? props.chart.frames?.at(-1)?.time ?? (last || undefined));
    setSelectedPositionId(props.chart.trace?.positions.at(-1)?.id);
    setSelectedExtremumId(undefined);
    setHoveredOracle(undefined);
    setHiddenAverageKeys(() => new Set<string>());
    setChartViewport(undefined);
  });

  const selectedMetric = createMemo(
    () =>
      backtestReplayMetricByKey.get(selectedMetricKey()) ??
      backtestReplayMetrics[0],
  );
  const selectedCandle = createMemo(() =>
    findReplayCandle(props.chart.candles, selectedTime()),
  );
  const replayFrames = createMemo<BacktestMetricFrame[]>(() =>
    props.chart.trace?.frames ?? props.chart.frames ?? []);
  const selectedFrame = createMemo(() => findReplayFrame(replayFrames(), selectedTime()));
  const selectedAnnotations = createMemo(() => {
    const candle = selectedCandle();
    return candle ? replayAnnotationsForCandle(props.chart.annotations, candle) : [];
  });
  const selectedExtremum = createMemo(() =>
    props.chart.trace?.extrema.find((item) => item.id === selectedExtremumId()),
  );
  const visibleAverageSeries = createMemo(() =>
    props.chart.smaSeries.filter((series) => !hiddenAverageKeys().has(averageSeriesKey(series))));
  const toggleAverageSeries = (series: BacktestChartSmaSeries) => {
    const key = averageSeriesKey(series);
    setHiddenAverageKeys((current) => {
      const next = new Set(current);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };
  const selectReplayTime = (time: number) => {
    setSelectedTime(time);
    const current = chartViewport();
    const index = props.chart.candles.findIndex((candle) =>
      time >= candle.openTime && time <= candle.closeTime);
    if (!current || index < 0 || (index >= current.start && index < current.end)) return;
    const visible = Math.max(1, current.end - current.start);
    const start = clampNumber(
      index - Math.floor(visible / 2),
      0,
      Math.max(0, props.chart.candles.length - visible),
    );
    setChartViewport({ start, end: start + visible });
  };

  return (
    <div class="mt-4 rounded-2 bg-ink-800 p-3">
      <div class="mb-3 flex flex-col gap-2 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <div class="muted-label">Replay Mode</div>
          <h3 class="text-base font-semibold">Backtest Behavior Replay</h3>
          <div class="mt-1 text-xs text-ink-300">
            {formatQuote(props.chart.candles.length, 0)} candles ·{" "}
            {formatQuote(props.chart.annotations.length, 0)} events ·{" "}
            {formatQuote(props.chart.trace?.positions.length ?? props.chart.frames?.length, 0)} position traces
          </div>
        </div>
        <div class="flex flex-wrap items-center gap-2">
          <div class="inline-grid grid-cols-2 overflow-hidden rounded-2 border border-line bg-ink-900">
            <button
              class="px-3 py-1.5 text-xs font-semibold transition"
              classList={{
                "bg-accent text-ink-950": priceDisplay() === "candles",
                "text-ink-300 hover:text-ink-100": priceDisplay() !== "candles",
              }}
              type="button"
              onClick={() => setPriceDisplay("candles")}
            >
              Candles
            </button>
            <button
              class="border-l border-line px-3 py-1.5 text-xs font-semibold transition"
              classList={{
                "bg-accent text-ink-950": priceDisplay() === "line",
                "text-ink-300 hover:text-ink-100": priceDisplay() !== "line",
              }}
              type="button"
              onClick={() => setPriceDisplay("line")}
            >
              Line
            </button>
          </div>
          <For each={[
            ["averages", "Averages"],
            ["signals", "Signals"],
            ["orders", "Orders"],
            ["fills", "Fills"],
            ["positions", "Positions"],
            ["extrema", "Extrema"],
            ["oracle", "Oracle"],
          ] as Array<[keyof CandleChartOverlayVisibility, string]>}>
            {([key, label]) => (
              <button
                class={buttonPanelClass}
                classList={{ "border-accent text-accent": overlays()[key] }}
                type="button"
                aria-pressed={overlays()[key]}
                onClick={() => {
                  if (key === "extrema" && overlays().extrema) setSelectedExtremumId(undefined);
                  if (key === "oracle" && overlays().oracle) setHoveredOracle(undefined);
                  setOverlays((current) => ({ ...current, [key]: !current[key] }));
                }}
              >
                {label}
              </button>
            )}
          </For>
          <Show when={selectedCandle()}>
            {(candle) => (
              <span class="rounded-2 border border-line bg-ink-900 px-2.5 py-1 text-xs tabular-nums text-ink-200">
                {formatDateTime(candle().openTime)} · close ${formatQuote(candle().close, 4)}
              </span>
            )}
          </Show>
          <Show when={overlays().averages}>
            <div class="flex flex-wrap items-center gap-1" role="group" aria-label="Displayed moving averages">
              <For each={props.chart.smaSeries}>
                {(series) => (
                  <button
                    class="inline-flex min-h-7 items-center gap-1.5 rounded-2 border bg-ink-900 px-2 py-1 text-xs transition"
                    classList={{
                      "text-ink-200": !hiddenAverageKeys().has(averageSeriesKey(series)),
                      "border-line text-ink-500": hiddenAverageKeys().has(averageSeriesKey(series)),
                    }}
                    style={{
                      "border-color": hiddenAverageKeys().has(averageSeriesKey(series))
                        ? undefined
                        : series.color,
                    }}
                    type="button"
                    aria-pressed={!hiddenAverageKeys().has(averageSeriesKey(series))}
                    title={`Toggle ${series.label}`}
                    onClick={() => toggleAverageSeries(series)}
                  >
                    <span
                      class="h-2 w-4 rounded-full"
                      classList={{ "opacity-25": hiddenAverageKeys().has(averageSeriesKey(series)) }}
                      style={{ "background-color": series.color }}
                    />
                    {series.label}
                  </button>
                )}
              </For>
            </div>
          </Show>
        </div>
      </div>
      <div class="h-110 lg:h-[560px]">
        <CandleChart
          candles={props.chart.candles}
          orders={[]}
          lastPrice={selectedFrame()?.price ?? props.chart.candles.at(-1)?.close ?? 0}
          smaSeries={visibleAverageSeries()}
          annotations={props.chart.annotations}
          trace={props.chart.trace}
          overlays={overlays()}
          highlightedPositionId={selectedPositionId()}
          selectedTime={selectedTime()}
          viewport={chartViewport()}
          priceDisplay={priceDisplay()}
          onSelectionChange={(selection) => {
            setSelectedTime(selection?.time);
            if (selection?.extremum) {
              setSelectedPositionId(selection.extremum.orders[0]?.positionId ?? selectedPositionId());
            } else {
              const positionId = selection?.annotations.find((item) => item.targetPositionId)?.targetPositionId;
              if (positionId) setSelectedPositionId(positionId);
            }
          }}
          onExtremumHoverChange={(extremum) => setSelectedExtremumId(extremum?.id)}
          onOracleHoverChange={setHoveredOracle}
          onViewportChange={setChartViewport}
          maxCandles={0}
          interactive
          emptyLabel="No replay candles"
        />
      </div>

      <div class="mt-3">
        <BacktestReplayMetricPicker frame={selectedFrame()} selectedKey={selectedMetric().key} onSelect={setSelectedMetricKey} />
      </div>
      <div class="mt-3 h-48 lg:h-56">
        <BacktestReplayMetricChart
          candles={props.chart.candles}
          frames={replayFrames()}
          metric={selectedMetric()}
          selectedTime={selectedTime()}
          viewport={chartViewport()}
          onSelectionTimeChange={setSelectedTime}
        />
      </div>
      <div class="mt-3 grid grid-cols-1 gap-3 xl:grid-cols-[minmax(0,320px)_minmax(0,1fr)_minmax(0,1fr)]">
        <BacktestReplayCandlePanel candle={selectedCandle()} annotations={selectedAnnotations()} />
        <BacktestReplayStatePanel frame={selectedFrame()} selectedMetricKey={selectedMetric().key} onMetricSelect={setSelectedMetricKey} />
        <BacktestReplayPositionPanel frame={selectedFrame()} selectedMetricKey={selectedMetric().key} onMetricSelect={setSelectedMetricKey} />
      </div>

      <Show when={props.chart.trace} fallback={
        <div class="mt-3 grid grid-cols-1 gap-3 xl:grid-cols-2">
          <BacktestReplayEventsPanel annotations={selectedAnnotations()} />
          <BacktestReplayOrdersAndLotsPanel frame={props.chart.frames ? findReplayFrame(props.chart.frames, selectedTime()) : undefined} />
        </div>
      }>
        {(trace) => (
          <>
            <div class="mt-3 grid grid-cols-1 gap-3 xl:grid-cols-2">
              <BacktestTraceSignalPanel trace={trace()} time={selectedTime()} />
              <BacktestTraceExtremumPanel extremum={selectedExtremum()} />
            </div>
            <div class="mt-3 grid grid-cols-1 gap-3 xl:grid-cols-[minmax(240px,0.32fr)_minmax(0,0.68fr)]">
              <BacktestTracePositionList
                positions={trace().positions}
                selectedId={selectedPositionId()}
                onSelect={setSelectedPositionId}
              />
              <BacktestTracePositionPanel
                trace={trace()}
                positionId={selectedPositionId()}
                time={selectedTime()}
                onSelectPosition={setSelectedPositionId}
              />
            </div>
            <div class="mt-3 grid grid-cols-1 gap-3 xl:grid-cols-3">
              <BacktestReplayEventsPanel annotations={selectedAnnotations()} />
              <BacktestTraceOrdersPanel trace={trace()} time={selectedTime()} onSelectPosition={setSelectedPositionId} />
              <BacktestOraclePanel
                trace={trace()}
                selected={hoveredOracle()}
                time={selectedTime()}
                onSelectTime={selectReplayTime}
              />
            </div>
          </>
        )}
      </Show>
    </div>
  );
}

function BacktestReplayCandlePanel(props: {
  candle?: Candle;
  annotations: BacktestChartAnnotation[];
}) {
  return (
    <div class="rounded-2 bg-ink-900 p-3">
      <div class="mb-3 flex items-center justify-between gap-3">
        <div>
          <div class="muted-label">Selected Candle</div>
          <div class="mt-1 text-sm font-semibold text-ink-100">
            {formatDateTime(props.candle?.openTime)}
          </div>
        </div>
        <span class="text-xs tabular-nums text-ink-300">
          {formatQuote(props.annotations.length, 0)} events
        </span>
      </div>
      <div class="grid grid-cols-2 gap-2">
        <SmallMetric label="Open" value={`$${formatQuote(props.candle?.open, 4)}`} />
        <SmallMetric label="Close" value={`$${formatQuote(props.candle?.close, 4)}`} />
        <SmallMetric label="High" value={`$${formatQuote(props.candle?.high, 4)}`} />
        <SmallMetric label="Low" value={`$${formatQuote(props.candle?.low, 4)}`} />
      </div>
    </div>
  );
}

function BacktestTraceSignalPanel(props: { trace: BacktestTrace; time?: number }) {
  const signal = createMemo(() => nearestByTime(props.trace.signals, props.time));
  return (
    <div class="rounded-2 bg-ink-900 p-3">
      <div class="mb-3 flex items-center justify-between gap-3">
        <div>
          <div class="muted-label">Strategy Activity</div>
          <div class="mt-1 text-sm font-semibold text-ink-100">
            {signal() ? formatDateTime(signal()?.time) : "No active signal"}
          </div>
        </div>
        <Show when={signal()}>
          {(item) => <span class="text-xs text-accent">source {item().source}</span>}
        </Show>
      </div>
      <Show when={signal()} fallback={<div class="text-sm text-ink-300">No strategy decision near this point</div>}>
        {(item) => (
          <>
            <div class="mb-2 flex flex-wrap gap-2">
              <For each={item().active} fallback={<span class="rounded-2 bg-ink-800 px-2 py-1 text-xs text-ink-300">confirmation active</span>}>
                {(active) => (
                  <span class="rounded-2 bg-accent/12 px-2 py-1 text-xs font-semibold text-accent">
                    {active.side} {active.type}
                  </span>
                )}
              </For>
            </div>
            <div class="max-h-44 space-y-1 overflow-auto pr-1 text-xs">
              <For each={item().gates}>
                {(gate) => (
                  <div class="flex items-center justify-between gap-3 rounded-2 bg-ink-800 px-2 py-1.5">
                    <span class={gate.passed ? "text-gain" : "text-ink-300"}>{gate.code}</span>
                    <span class="tabular-nums text-ink-400">
                      {formatQuote(gate.value, 6)} / {formatQuote(gate.threshold, 6)}
                    </span>
                  </div>
                )}
              </For>
            </div>
          </>
        )}
      </Show>
    </div>
  );
}

function BacktestTraceExtremumPanel(props: { extremum?: BacktestExtremumTrace }) {
  const box = () => props.extremum?.errorBox;
  return (
    <div class="rounded-2 bg-ink-900 p-3">
      <div class="mb-3">
        <div class="muted-label">Hovered Real Extremum</div>
        <div class="mt-1 text-sm font-semibold text-ink-100">
          {props.extremum
            ? `${props.extremum.kind} at $${formatQuote(props.extremum.price, 4)}`
            : "Hover a peak or valley dot"}
        </div>
      </div>
      <Show when={props.extremum}>
        {(extremum) => (
          <>
            <div class="grid grid-cols-2 gap-2">
              <SmallMetric label="Centered SMA" value={formatDuration(extremum().smaWindowMs)} />
              <SmallMetric label="Matched fills" value={formatQuote(extremum().orders.length, 0)} />
              <SmallMetric label="P99 time box" value={`±${formatDuration(extremum().p99TimeDistanceMs)}`} />
              <SmallMetric label="P99 price box" value={`±${formatPercent(extremum().p99PriceDistancePct)}`} />
              <SmallMetric label="Matched quote" value={`$${formatQuote(box()?.quote, 2)}`} />
              <SmallMetric
                label="Inside threshold"
                value={box()?.quote ? formatPercent((box()!.withinThresholdQuote / box()!.quote) * 100) : "-"}
              />
            </div>
            <div class="mt-2 max-h-28 space-y-1 overflow-auto pr-1">
              <For each={extremum().orders}>
                {(order) => (
                  <div class="rounded-2 bg-ink-800 px-2 py-1.5 text-xs text-ink-300">
                    {order.grid} {shortId(order.orderId)} · {formatDuration(Math.abs(order.timeErrorMs))} · {formatPercent(order.priceErrorPct)} · ${formatQuote(order.quote, 2)}
                  </div>
                )}
              </For>
            </div>
          </>
        )}
      </Show>
    </div>
  );
}

function BacktestTracePositionList(props: {
  positions: BacktestPositionTrace[];
  selectedId?: string;
  onSelect: (id: string) => void;
}) {
  return (
    <div class="rounded-2 bg-ink-900 p-3">
      <div class="mb-3">
        <div class="muted-label">Concrete Positions</div>
        <div class="mt-1 text-sm font-semibold text-ink-100">
          {formatQuote(props.positions.length, 0)} position lifecycles
        </div>
      </div>
      <div class="max-h-132 space-y-2 overflow-auto pr-1">
        <For each={[...props.positions].sort((left, right) => right.createdAt - left.createdAt)}>
          {(position) => (
            <button
              class="w-full rounded-2 border bg-ink-800 p-2 text-left transition hover:border-accent/60"
              classList={{ "border-accent": props.selectedId === position.id, "border-transparent": props.selectedId !== position.id }}
              type="button"
              onClick={() => props.onSelect(position.id)}
            >
              <div class="flex items-center justify-between gap-2">
                <span class={position.side === "long" ? "text-gain" : "text-loss"}>{position.side}</span>
                <span class="text-xs text-ink-400">{formatTime(position.createdAt)}</span>
              </div>
              <div class="mt-1 text-xs text-ink-300">
                {shortId(position.id)} · {formatLeverage(position.leverage)} · {position.closedAt ? "closed" : "open"}
              </div>
            </button>
          )}
        </For>
      </div>
    </div>
  );
}

function BacktestTracePositionPanel(props: {
  trace: BacktestTrace;
  positionId?: string;
  time?: number;
  onSelectPosition: (id: string) => void;
}) {
  const position = createMemo(() => props.trace.positions.find((item) => item.id === props.positionId));
  const state = createMemo(() => positionStateAt(position(), props.time));
  const orders = createMemo(() => props.trace.orders.filter((item) => item.positionId === props.positionId));
  const grids = createMemo(() => props.trace.grids.filter((item) => item.positionId === props.positionId));
  return (
    <div class="rounded-2 bg-ink-900 p-3">
      <Show when={position()} fallback={<div class="text-sm text-ink-300">Select a position or one of its graph events</div>}>
        {(item) => (
          <>
            <div class="mb-3 flex flex-col gap-2 lg:flex-row lg:items-start lg:justify-between">
              <div>
                <div class="muted-label">Position Dynamics</div>
                <div class="mt-1 text-sm font-semibold text-ink-100">
                  {item().side} {shortId(item().id)} · {formatLeverage(item().leverage)}
                </div>
              </div>
              <div class="text-right text-xs text-ink-300">
                <div>created {formatDateTime(item().createdAt)}</div>
                <div>opened {formatDateTime(item().openedAt ?? undefined)}</div>
                <div>closed {formatDateTime(item().closedAt ?? undefined)}</div>
              </div>
            </div>
            <div class="grid grid-cols-2 gap-2 md:grid-cols-4">
              <SmallMetric label="Asset" value={formatAsset(state()?.asset)} />
              <SmallMetric label="Quote" value={`$${formatQuote(state()?.quote, 2)}`} />
              <SmallMetric label="External asset debt" value={formatAsset(state()?.externalBorrow.asset)} />
              <SmallMetric label="External quote debt" value={`$${formatQuote(state()?.externalBorrow.quote, 2)}`} />
            </div>
            <div class="mt-3 grid grid-cols-1 gap-3 lg:grid-cols-2">
              <BacktestBorrowLinks
                label="Borrowed from"
                borrows={state()?.internalBorrow ?? []}
                onSelect={props.onSelectPosition}
              />
              <BacktestBorrowLinks
                label="Positions borrowing from it"
                borrows={state()?.lentTo ?? []}
                onSelect={props.onSelectPosition}
              />
            </div>
            <div class="mt-3 grid grid-cols-1 gap-3 2xl:grid-cols-2">
              <div>
                <div class="mb-2 text-xs uppercase tracking-wide text-ink-300">How it opened and closed</div>
                <div class="max-h-80 space-y-2 overflow-auto pr-1">
                  <For each={orders()}>
                    {(order) => <BacktestTraceOrderRow order={order} />}
                  </For>
                </div>
              </div>
              <div>
                <div class="mb-2 text-xs uppercase tracking-wide text-ink-300">Amount / debt changes</div>
                <div class="max-h-80 space-y-2 overflow-auto pr-1">
                  <For each={item().states}>
                    {(next, index) => <BacktestPositionStateRow state={next} previous={item().states[index() - 1]} />}
                  </For>
                </div>
              </div>
            </div>
            <div class="mt-3 text-xs text-ink-400">
              {grids().map((grid) => `${grid.kind}/${grid.cause} ${shortId(grid.id)} @ $${formatQuote(grid.creationPrice, 4)} (${grid.orderIds.length})`).join(" · ")}
            </div>
          </>
        )}
      </Show>
    </div>
  );
}

function BacktestTraceOrdersPanel(props: {
  trace: BacktestTrace;
  time?: number;
  onSelectPosition: (id: string) => void;
}) {
  const orders = createMemo(() => props.trace.orders.filter((order) =>
    (props.time === undefined || order.createdAt <= props.time)
    && (order.endedAt === null || props.time === undefined || order.endedAt > props.time)));
  const positions = createMemo(() => props.trace.positions.filter((position) =>
    (props.time === undefined || position.createdAt <= props.time)
    && (position.closedAt === null || props.time === undefined || position.closedAt > props.time)));
  return (
    <div class="rounded-2 bg-ink-900 p-3">
      <div class="mb-3 flex items-center justify-between gap-3">
        <div>
          <div class="muted-label">Orders & Positions at Time</div>
          <div class="mt-1 text-sm font-semibold text-ink-100">
            {formatQuote(orders().length, 0)} orders · {formatQuote(positions().length, 0)} positions
          </div>
        </div>
        <span class="text-xs text-ink-400">{formatTime(props.time)}</span>
      </div>
      <div class="max-h-72 space-y-2 overflow-auto pr-1">
        <For each={orders()} fallback={<div class="text-sm text-ink-300">No active orders</div>}>
          {(order) => (
            <button class="block w-full text-left" type="button" onClick={() => props.onSelectPosition(order.positionId)}>
              <BacktestTraceOrderRow order={order} />
            </button>
          )}
        </For>
      </div>
    </div>
  );
}

function BacktestBorrowLinks(props: {
  label: string;
  borrows: Array<{ positionId: string; asset: number; quote: number }>;
  onSelect: (id: string) => void;
}) {
  return (
    <div class="rounded-2 bg-ink-800 p-2">
      <div class="mb-2 text-xs uppercase tracking-wide text-ink-300">{props.label}</div>
      <For each={props.borrows} fallback={<div class="text-xs text-ink-400">None</div>}>
        {(borrow) => (
          <button class="mb-1 block w-full rounded-2 bg-ink-900 px-2 py-1.5 text-left text-xs text-accent" type="button" onClick={() => props.onSelect(borrow.positionId)}>
            {shortId(borrow.positionId)} · {formatAsset(borrow.asset)} / ${formatQuote(borrow.quote, 2)}
          </button>
        )}
      </For>
    </div>
  );
}

function BacktestTraceOrderRow(props: { order: BacktestOrderTrace }) {
  return (
    <div class="rounded-2 bg-ink-800 p-2 text-xs">
      <div class="flex items-center justify-between gap-3">
        <span class={props.order.side === "buy" ? "text-gain" : "text-loss"}>
          {props.order.grid} {props.order.side} {props.order.type}
        </span>
        <span class="text-ink-400">{props.order.outcome}</span>
      </div>
      <div class="mt-1 text-ink-300">
        {shortId(props.order.id)} · {formatAsset(props.order.size)} @ ${formatQuote(props.order.price ?? props.order.stopPrice ?? undefined, 4)} · {formatTime(props.order.createdAt)}
      </div>
      <For each={props.order.fills}>
        {(fill) => (
          <div class="mt-1 rounded-2 bg-ink-900 px-2 py-1 text-ink-300">
            fill {formatAsset(fill.asset)} @ ${formatQuote(fill.price, 4)} · ${formatQuote(fill.quote, 2)} · fee ${formatQuote(fill.feeQuote, 4)} · {formatTime(fill.time)}
          </div>
        )}
      </For>
    </div>
  );
}

function BacktestPositionStateRow(props: { state: BacktestPositionStateTrace; previous?: BacktestPositionStateTrace }) {
  const assetDelta = () => props.state.asset - (props.previous?.asset ?? 0);
  const quoteDelta = () => props.state.quote - (props.previous?.quote ?? 0);
  return (
    <div class="rounded-2 bg-ink-800 p-2 text-xs text-ink-300">
      <div class="flex items-center justify-between gap-3">
        <span>{formatTime(props.state.time)}</span>
        <span class="tabular-nums">Δ {formatAsset(assetDelta())} / ${formatQuote(quoteDelta(), 2)}</span>
      </div>
      <div class="mt-1 tabular-nums">
        balance {formatAsset(props.state.asset)} / ${formatQuote(props.state.quote, 2)} · debt {formatAsset(props.state.externalBorrow.asset)} / ${formatQuote(props.state.externalBorrow.quote, 2)}
      </div>
    </div>
  );
}

function BacktestOraclePanel(props: {
  trace: BacktestTrace;
  selected?: BacktestOraclePoint;
  time?: number;
  onSelectTime: (time: number) => void;
}) {
  const transitions = createMemo(() =>
    props.trace.oracle.points.filter((point) => point.action !== "hold"));
  const selectedIndex = createMemo(() => {
    const values = transitions();
    if (values.length === 0) return -1;
    if (props.selected) {
      const exact = values.findIndex((point) =>
        point.time === props.selected?.time && point.state === props.selected.state);
      if (exact >= 0) return exact;
    }
    return nearestOracleTransitionIndex(values, props.time);
  });
  const selected = createMemo(() => transitions()[selectedIndex()]);
  const visibleTransitions = createMemo(() => {
    const values = transitions();
    const count = Math.min(80, values.length);
    const start = clampNumber(
      selectedIndex() - Math.floor(count / 2),
      0,
      Math.max(0, values.length - count),
    );
    return values.slice(start, start + count).map((point, index) => ({ point, index: start + index }));
  });
  const selectOffset = (offset: number) => {
    const values = transitions();
    const point = values[clampNumber(selectedIndex() + offset, 0, values.length - 1)];
    if (point) props.onSelectTime(point.time);
  };
  return (
    <div class="rounded-2 bg-ink-900 p-3">
      <div class="mb-3 flex items-center justify-between gap-3">
        <div>
          <div class="muted-label">Optimal Trading Path</div>
          <div class="mt-1 text-sm font-semibold text-ink-100">
            Three-state DP oracle · {props.trace.oracle.mode} · {props.trace.oracle.eventMode}
          </div>
        </div>
        <span class="text-xs text-ink-300">
          {formatLeverage(props.trace.oracle.leverage)} · friction {formatPercent(props.trace.oracle.friction * 100)}
        </span>
      </div>
      <div class="mb-2 flex flex-wrap gap-3 text-xs">
        <span class="text-gain">▲ L long</span>
        <span class="text-loss">▼ S short</span>
        <span class="text-ink-300">◆ F flat / wait</span>
        <span class="text-ink-400">{formatQuote(transitions().length, 0)} exact transitions</span>
      </div>
      <Show when={selected()}>
        {(point) => (
          <button
            class="mb-2 block w-full rounded-2 border border-accent/35 bg-accent/8 p-2 text-left"
            type="button"
            onClick={() => props.onSelectTime(point().time)}
          >
            <div class="flex items-center justify-between gap-3 text-xs">
              <span class={oracleStateClass(point().state)}>
                {oracleTransitionLabel(point())} · {point().fromState} → {point().state}
              </span>
              <span class="text-ink-400">#{selectedIndex() + 1}</span>
            </div>
            <div class="mt-1 text-xs tabular-nums text-ink-200">
              ${formatQuote(point().price, 4)} · {formatDateTime(point().time)}
            </div>
          </button>
        )}
      </Show>
      <div class="mb-2 flex items-center justify-between gap-2">
        <button class={buttonPanelClass} type="button" disabled={selectedIndex() <= 0} onClick={() => selectOffset(-1)}>
          Previous
        </button>
        <span class="text-center text-xs text-ink-400">
          {formatQuote(visibleTransitions().length, 0)} around selected
        </span>
        <button class={buttonPanelClass} type="button" disabled={selectedIndex() >= transitions().length - 1} onClick={() => selectOffset(1)}>
          Next
        </button>
      </div>
      <div class="max-h-48 space-y-1 overflow-auto pr-1">
        <For each={visibleTransitions()} fallback={<div class="text-sm text-ink-300">No state changes</div>}>
          {({ point, index }) => (
            <button
              class="flex w-full items-center justify-between gap-3 rounded-2 bg-ink-800 px-2 py-1 text-left text-xs"
              classList={{ "ring-1 ring-accent/55": index === selectedIndex() }}
              type="button"
              onClick={() => props.onSelectTime(point.time)}
            >
              <span class={oracleStateClass(point.state)}>
                {index + 1}. {oracleTransitionLabel(point)} · {point.fromState} → {point.state}
              </span>
              <span class="text-ink-400">${formatQuote(point.price, 4)} · {formatTime(point.time)}</span>
            </button>
          )}
        </For>
      </div>
      <div class="mt-2 text-xs text-ink-400">
        The bottom strip is the held state; hold means remain long/short/flat. {props.trace.oracle.eventMode === "ohlc"
          ? "Events use synthetic open → high → low → close order."
          : "Events use completed candle closes only."}
      </div>
    </div>
  );
}

function nearestOracleTransitionIndex(points: BacktestOraclePoint[], time: number | undefined): number {
  if (points.length === 0) return -1;
  if (time === undefined) return points.length - 1;
  let low = 0;
  let high = points.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if ((points[middle]?.time ?? Number.POSITIVE_INFINITY) < time) low = middle + 1;
    else high = middle;
  }
  if (low === 0) return 0;
  if (low >= points.length) return points.length - 1;
  return time - points[low - 1]!.time <= points[low]!.time - time ? low - 1 : low;
}

function oracleTransitionLabel(point: BacktestOraclePoint): string {
  if (point.state === "flat") return "GO FLAT";
  return point.fromState === "flat"
    ? `GO ${point.state.toUpperCase()}`
    : `SWITCH TO ${point.state.toUpperCase()}`;
}

function oracleStateClass(state: BacktestOraclePoint["state"]): string {
  return state === "long" ? "text-gain" : state === "short" ? "text-loss" : "text-ink-300";
}

function positionStateAt(position: BacktestPositionTrace | undefined, time: number | undefined): BacktestPositionStateTrace | undefined {
  if (!position) return undefined;
  if (time !== undefined && time < position.createdAt) return undefined;
  return position.states.filter((state) => time === undefined || state.time <= time).at(-1) ?? position.states[0];
}

function nearestByTime<T extends { time: number }>(values: T[], time: number | undefined): T | undefined {
  if (values.length === 0) return undefined;
  if (time === undefined) return values.at(-1);
  return values.reduce((nearest, value) =>
    Math.abs(value.time - time) < Math.abs(nearest.time - time) ? value : nearest);
}

function shortId(value: string): string {
  return value.length > 12 ? `${value.slice(0, 6)}…${value.slice(-4)}` : value;
}

function BacktestReplayMetricPicker(props: {
  frame?: BacktestMetricFrame;
  selectedKey: BacktestReplayMetricKey;
  onSelect: (key: BacktestReplayMetricKey) => void;
}) {
  const groups: Array<BacktestReplayMetricDefinition["group"]> = [
    "account",
    "risk",
    "positions",
    "balances",
    "activity",
  ];

  return (
    <div class="rounded-2 bg-ink-900 p-3">
      <div class="mb-3 flex items-center justify-between gap-3">
        <div>
          <div class="muted-label">Metric Chart</div>
          <div class="mt-1 text-sm font-semibold text-ink-100">
            {backtestReplayMetricByKey.get(props.selectedKey)?.label ?? "Metric"}
          </div>
        </div>
        <span class="text-xs text-ink-300">Click a metric to plot it</span>
      </div>
      <div class="flex flex-col gap-2">
        <For each={groups}>
          {(group) => (
            <div class="flex min-w-0 flex-wrap items-center gap-2">
              <span class="w-18 text-xs uppercase tracking-wide text-ink-400">{group}</span>
              <For each={backtestReplayMetrics.filter((metric) => metric.group === group)}>
                {(metric) => (
                  <button
                    class="inline-flex min-h-8 items-center gap-2 rounded-2 border px-2.5 py-1 text-xs font-semibold tabular-nums transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/45"
                    classList={{
                      "border-accent bg-accent/14 text-ink-100": props.selectedKey === metric.key,
                      "border-line bg-ink-800 text-ink-300 hover:border-accent/70 hover:text-ink-100":
                        props.selectedKey !== metric.key,
                    }}
                    type="button"
                    onClick={() => props.onSelect(metric.key)}
                  >
                    <span
                      class="h-2 w-2 rounded-full"
                      style={{ "background-color": metric.color }}
                    />
                    {metric.label}
                    <span class="font-normal text-ink-400">
                      {metric.format(props.frame ? metric.value(props.frame) : undefined)}
                    </span>
                  </button>
                )}
              </For>
            </div>
          )}
        </For>
      </div>
    </div>
  );
}

function BacktestReplayMetricButton(props: {
  metricKey: BacktestReplayMetricKey;
  frame?: BacktestMetricFrame;
  selected: boolean;
  onSelect: (key: BacktestReplayMetricKey) => void;
}) {
  const metric = () => backtestReplayMetricByKey.get(props.metricKey);
  const value = () => {
    const item = metric();
    return item?.format(props.frame ? item.value(props.frame) : undefined) ?? "-";
  };

  return (
    <button
      class="rounded-2 border bg-ink-800 p-3 text-left transition hover:border-accent/70 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/45"
      classList={{
        "border-accent shadow-[inset_0_0_0_1px_rgba(56,189,248,0.32)]": props.selected,
        "border-transparent": !props.selected,
      }}
      type="button"
      onClick={() => props.onSelect(props.metricKey)}
    >
      <div class="flex items-center gap-2">
        <span
          class="h-2 w-2 rounded-full"
          style={{ "background-color": metric()?.color ?? "#38bdf8" }}
        />
        <div class="muted-label">{metric()?.label ?? props.metricKey}</div>
      </div>
      <div class="mt-1 text-base font-semibold tabular-nums text-ink-100">{value()}</div>
    </button>
  );
}

function BacktestReplayMetricChart(props: {
  candles: Candle[];
  frames: BacktestMetricFrame[];
  metric: BacktestReplayMetricDefinition;
  selectedTime?: number;
  viewport?: CandleChartViewport;
  onSelectionTimeChange: (time: number) => void;
}) {
  let canvas!: HTMLCanvasElement;
  let observer: ResizeObserver | undefined;

  const draw = () => {
    if (!canvas) {
      return;
    }

    const parent = canvas.parentElement;
    const width = Math.max(320, parent?.clientWidth ?? canvas.clientWidth);
    const height = Math.max(160, parent?.clientHeight ?? canvas.clientHeight);
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.floor(width * ratio);
    canvas.height = Math.floor(height * ratio);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;

    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }

    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#101217";
    ctx.fillRect(0, 0, width, height);

    const range = replayMetricTimeRange(props.candles, props.viewport);
    if (!range || props.frames.length < 2) {
      drawReplayMetricEmpty(ctx, width, height, "No replay metric samples");
      return;
    }

    const points = props.frames
      .map((frame) => ({
        time: frame.time,
        value: props.metric.value(frame),
      }))
      .filter(
        (point): point is { time: number; value: number } =>
          point.time >= range.startTime &&
          point.time <= range.endTime &&
          Number.isFinite(point.value),
      );

    if (points.length < 2) {
      drawReplayMetricEmpty(ctx, width, height, "No samples in visible range");
      return;
    }

    const values = points.map((point) => point.value);
    let min = Math.min(...values);
    let max = Math.max(...values);
    if (min === max) {
      const pad = Math.max(Math.abs(max) * 0.01, 1);
      min -= pad;
      max += pad;
    }

    const padding = Math.max((max - min) * 0.08, Math.abs(max) * 0.0005, 0.000001);
    const low = min - padding;
    const high = max + padding;
    const plot = replayMetricPlotBounds(width, height);
    const xFor = (time: number) =>
      plot.left + ((time - range.startTime) / Math.max(1, range.endTime - range.startTime)) *
        (plot.right - plot.left);
    const yFor = (value: number) =>
      plot.top + ((high - value) / Math.max(0.000001, high - low)) *
        (plot.bottom - plot.top);

    drawReplayMetricGrid(ctx, width, height, plot, low, high, props.metric.format, yFor);
    drawReplayMetricLine(ctx, points, plot, xFor, yFor, props.metric.color);
    drawReplayMetricSelection(ctx, props.metric, points, plot, xFor, yFor, props.selectedTime);
  };

  createEffect(() => {
    props.frames.length;
    props.metric.key;
    props.selectedTime;
    props.viewport?.start;
    props.viewport?.end;
    props.candles.length;
    draw();
  });

  onMount(() => {
    observer = new ResizeObserver(draw);
    observer.observe(canvas.parentElement ?? canvas);
    draw();
  });

  onCleanup(() => observer?.disconnect());

  const selectAtOffset = (offsetX: number) => {
    const range = replayMetricTimeRange(props.candles, props.viewport);
    if (!range) {
      return;
    }

    const plot = replayMetricPlotBounds(canvas.clientWidth, canvas.clientHeight);
    const normalized = clampNumber(
      (offsetX - plot.left) / Math.max(1, plot.right - plot.left),
      0,
      1,
    );
    const targetTime = range.startTime + normalized * (range.endTime - range.startTime);
    const frame = findReplayFrame(props.frames, targetTime);
    if (frame) {
      props.onSelectionTimeChange(frame.time);
    }
  };

  return (
    <canvas
      ref={canvas}
      class="h-full min-h-40 w-full rounded-2 outline-none focus-visible:ring-2 focus-visible:ring-accent/45"
      tabIndex={0}
      title="Click or move across the chart to inspect that timestamp."
      onPointerDown={(event) => selectAtOffset(event.offsetX)}
      onPointerMove={(event) => {
        if (event.buttons === 1) {
          selectAtOffset(event.offsetX);
        }
      }}
    />
  );
}

function BacktestReplayStatePanel(props: {
  frame?: BacktestMetricFrame;
  selectedMetricKey: BacktestReplayMetricKey;
  onMetricSelect: (key: BacktestReplayMetricKey) => void;
}) {
  const frame = () => props.frame;

  return (
    <div class="rounded-2 bg-ink-900 p-3">
      <div class="mb-3">
        <div class="muted-label">Bot State</div>
        <div class="mt-1 text-sm font-semibold text-ink-100">
          {formatDateTime(frame()?.time)}
        </div>
      </div>
      <div class="grid grid-cols-2 gap-2 lg:grid-cols-3">
        <BacktestReplayMetricButton
          metricKey="equity"
          frame={frame()}
          selected={props.selectedMetricKey === "equity"}
          onSelect={props.onMetricSelect}
        />
        <BacktestReplayMetricButton
          metricKey="returnPct"
          frame={frame()}
          selected={props.selectedMetricKey === "returnPct"}
          onSelect={props.onMetricSelect}
        />
        <BacktestReplayMetricButton
          metricKey="exposurePct"
          frame={frame()}
          selected={props.selectedMetricKey === "exposurePct"}
          onSelect={props.onMetricSelect}
        />
        <BacktestReplayMetricButton
          metricKey="drawdownPct"
          frame={frame()}
          selected={props.selectedMetricKey === "drawdownPct"}
          onSelect={props.onMetricSelect}
        />
        <SmallMetric label="Entry" value={replaySignalLabel(frame()?.entrySignal)} />
        <SmallMetric label="Exit" value={replaySignalLabel(frame()?.exitSignal)} />
        <BacktestReplayMetricButton
          metricKey="quoteFree"
          frame={frame()}
          selected={props.selectedMetricKey === "quoteFree"}
          onSelect={props.onMetricSelect}
        />
        <BacktestReplayMetricButton
          metricKey="baseFree"
          frame={frame()}
          selected={props.selectedMetricKey === "baseFree"}
          onSelect={props.onMetricSelect}
        />
        <SmallMetric label="Market" value={replayMarketState(frame())} />
      </div>
    </div>
  );
}

function BacktestReplayPositionPanel(props: {
  frame?: BacktestMetricFrame;
  selectedMetricKey: BacktestReplayMetricKey;
  onMetricSelect: (key: BacktestReplayMetricKey) => void;
}) {
  return (
    <div class="rounded-2 bg-ink-900 p-3">
      <div class="mb-3">
        <div class="muted-label">Positions</div>
        <div class="mt-1 text-sm font-semibold text-ink-100">
          {formatQuote(props.frame?.longLotCount, 0)} long lots ·{" "}
          {formatQuote(props.frame?.shortLotCount, 0)} short lots
        </div>
      </div>
      <div class="grid grid-cols-2 gap-2 lg:grid-cols-3">
        <BacktestReplayMetricButton
          metricKey="longQuantity"
          frame={props.frame}
          selected={props.selectedMetricKey === "longQuantity"}
          onSelect={props.onMetricSelect}
        />
        <BacktestReplayMetricButton
          metricKey="shortQuantity"
          frame={props.frame}
          selected={props.selectedMetricKey === "shortQuantity"}
          onSelect={props.onMetricSelect}
        />
        <BacktestReplayMetricButton
          metricKey="netExposureQuote"
          frame={props.frame}
          selected={props.selectedMetricKey === "netExposureQuote"}
          onSelect={props.onMetricSelect}
        />
        <BacktestReplayMetricButton
          metricKey="grossExposureQuote"
          frame={props.frame}
          selected={props.selectedMetricKey === "grossExposureQuote"}
          onSelect={props.onMetricSelect}
        />
        <BacktestReplayMetricButton
          metricKey="effectiveLeverage"
          frame={props.frame}
          selected={props.selectedMetricKey === "effectiveLeverage"}
          onSelect={props.onMetricSelect}
        />
        <BacktestReplayMetricButton
          metricKey="openOrderCount"
          frame={props.frame}
          selected={props.selectedMetricKey === "openOrderCount"}
          onSelect={props.onMetricSelect}
        />
      </div>
    </div>
  );
}

function BacktestReplayEventsPanel(props: { annotations: BacktestChartAnnotation[] }) {
  return (
    <div class="rounded-2 bg-ink-900 p-3">
      <div class="mb-3 flex items-center justify-between gap-3">
        <div>
          <div class="muted-label">Selected Events</div>
          <div class="mt-1 text-sm font-semibold text-ink-100">
            Signals, orders, fills
          </div>
        </div>
        <span class="text-xs tabular-nums text-ink-300">
          {formatQuote(props.annotations.length, 0)}
        </span>
      </div>
      <div class="max-h-72 space-y-2 overflow-auto pr-1">
        <For
          each={props.annotations}
          fallback={<div class="text-sm text-ink-300">No bot events at this candle</div>}
        >
          {(annotation) => <BacktestAnnotationRow annotation={annotation} />}
        </For>
      </div>
    </div>
  );
}

function BacktestReplayOrdersAndLotsPanel(props: { frame?: BacktestReplayFrame }) {
  const lots = () => [
    ...(props.frame?.positions.longs ?? []),
    ...(props.frame?.positions.shorts ?? []),
  ].sort((left, right) => right.openedAt - left.openedAt);

  return (
    <div class="rounded-2 bg-ink-900 p-3">
      <div class="mb-3">
        <div class="muted-label">Open Orders & Lots</div>
        <div class="mt-1 text-sm font-semibold text-ink-100">
          {formatQuote(props.frame?.openOrderCount, 0)} orders ·{" "}
          {formatQuote(lots().length, 0)} shown lots
        </div>
      </div>
      <div class="grid grid-cols-1 gap-3 lg:grid-cols-2">
        <div class="min-w-0">
          <div class="mb-2 text-xs uppercase tracking-wide text-ink-300">Orders</div>
          <div class="max-h-72 space-y-2 overflow-auto pr-1">
            <For
              each={props.frame?.openOrders ?? []}
              fallback={<div class="text-sm text-ink-300">No open orders</div>}
            >
              {(order) => <BacktestReplayOrderRow order={order} />}
            </For>
            <Show when={props.frame?.truncatedOpenOrderCount}>
              {(count) => (
                <div class="text-xs text-ink-300">
                  {formatQuote(count(), 0)} older open orders hidden
                </div>
              )}
            </Show>
          </div>
        </div>
        <div class="min-w-0">
          <div class="mb-2 text-xs uppercase tracking-wide text-ink-300">Lots</div>
          <div class="max-h-72 space-y-2 overflow-auto pr-1">
            <For
              each={lots()}
              fallback={<div class="text-sm text-ink-300">No active lots</div>}
            >
              {(lot) => <BacktestReplayLotRow lot={lot} />}
            </For>
            <Show when={(props.frame?.truncatedLongLotCount ?? 0) + (props.frame?.truncatedShortLotCount ?? 0)}>
              {(count) => (
                <div class="text-xs text-ink-300">
                  {formatQuote(count(), 0)} older active lots hidden
                </div>
              )}
            </Show>
          </div>
        </div>
      </div>
    </div>
  );
}

function BacktestAnnotationRow(props: { annotation: BacktestChartAnnotation }) {
  const isBuy = () => props.annotation.kind.startsWith("buy");
  return (
    <div class="rounded-2 bg-ink-900 p-2 text-sm">
      <div class="flex items-center justify-between gap-3">
        <div class="flex min-w-0 items-center gap-2">
          <span
            class="rounded-2 px-2 py-1 text-xs font-semibold uppercase"
            classList={{
              "bg-gain/12 text-gain": isBuy(),
              "bg-loss/12 text-loss": !isBuy(),
            }}
          >
            {props.annotation.kind.replace("-", " ")}
          </span>
          <span class="truncate text-ink-100">{props.annotation.label}</span>
        </div>
        <span class="shrink-0 text-xs text-ink-300">{formatTime(props.annotation.time)}</span>
      </div>
      <div class="mt-1 text-xs text-ink-300 tabular-nums">
        ${formatQuote(props.annotation.price, 4)}
        <Show when={props.annotation.reason}>
          {(reason) => <span> · {reason()}</span>}
        </Show>
        <Show when={props.annotation.targetPositionId}>
          {(target) => <span> · {target()}</span>}
        </Show>
        <Show when={props.annotation.gridId}>
          {(grid) => <span> · {props.annotation.gridKind} grid {shortId(grid())}</span>}
        </Show>
      </div>
    </div>
  );
}

function BacktestReplayOrderRow(props: { order: TradingOrder }) {
  return (
    <div class="rounded-2 bg-ink-800 p-2 text-sm">
      <div class="flex items-center justify-between gap-3">
        <div class="flex min-w-0 items-center gap-2">
          <Side side={props.order.side} />
          <span class="truncate text-ink-100">{props.order.type}</span>
        </div>
        <span class="shrink-0 text-xs text-ink-300">{formatTime(props.order.createdAt)}</span>
      </div>
      <div class="mt-1 text-xs text-ink-300 tabular-nums">
        ${formatQuote(props.order.price, 4)} · {formatAsset(props.order.quantity)} ·{" "}
        {props.order.positionEffect ?? "auto"}
      </div>
    </div>
  );
}

function BacktestReplayLotRow(props: { lot: LongPositionLot | ShortPositionLot }) {
  const breakEven =
    props.lot.side === "long" ? props.lot.breakEvenSellPrice : props.lot.breakEvenBuyPrice;
  const remaining =
    props.lot.side === "long" ? props.lot.remainingCostQuote : props.lot.remainingProceedsQuote;

  return (
    <div class="rounded-2 bg-ink-800 p-2 text-sm">
      <div class="flex items-center justify-between gap-3">
        <Side side={props.lot.side === "long" ? "buy" : "sell"} />
        <span class="shrink-0 text-xs text-ink-300">{props.lot.status}</span>
      </div>
      <div class="mt-1 grid grid-cols-2 gap-x-3 gap-y-1 text-xs tabular-nums text-ink-300">
        <span>qty {formatAsset(props.lot.remainingQuantity)}</span>
        <span>avg ${formatQuote(props.lot.averagePrice, 4)}</span>
        <span>rem ${formatQuote(remaining, 2)}</span>
        <span>be ${formatQuote(breakEven, 4)}</span>
        <span>lev {formatLeverage(props.lot.leverage)}</span>
        <span>{formatTime(props.lot.openedAt)}</span>
      </div>
    </div>
  );
}

function replayMetricTimeRange(
  candles: Candle[],
  viewport: CandleChartViewport | undefined,
): { startTime: number; endTime: number } | undefined {
  if (candles.length === 0) {
    return undefined;
  }

  const start = clampNumber(Math.round(viewport?.start ?? 0), 0, candles.length - 1);
  const end = clampNumber(
    Math.round(viewport?.end ?? candles.length),
    start + 1,
    candles.length,
  );
  const first = candles[start];
  const last = candles[end - 1];
  if (!first || !last) {
    return undefined;
  }

  return {
    startTime: first.openTime,
    endTime: last.closeTime,
  };
}

function replayMetricPlotBounds(width: number, height: number): {
  left: number;
  right: number;
  top: number;
  bottom: number;
} {
  return {
    left: 18,
    right: width - 84,
    top: 18,
    bottom: height - 30,
  };
}

function drawReplayMetricEmpty(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  label: string,
): void {
  ctx.fillStyle = "#aeb6c8";
  ctx.font = "13px Inter, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(label, width / 2, height / 2);
}

function drawReplayMetricGrid(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  plot: { left: number; right: number; top: number; bottom: number },
  low: number,
  high: number,
  format: (value: number | undefined) => string,
  yFor: (value: number) => number,
): void {
  ctx.strokeStyle = "#242833";
  ctx.lineWidth = 1;
  ctx.font = "11px Inter, sans-serif";
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";

  for (let index = 0; index <= 3; index += 1) {
    const value = low + ((high - low) * index) / 3;
    const y = yFor(value);
    ctx.beginPath();
    ctx.moveTo(plot.left, y);
    ctx.lineTo(plot.right, y);
    ctx.stroke();
    ctx.fillStyle = "#aeb6c8";
    ctx.fillText(format(value), plot.right + 10, y);
  }

  ctx.strokeStyle = "#2b303b";
  ctx.strokeRect(plot.left, plot.top, plot.right - plot.left, plot.bottom - plot.top);
  ctx.fillStyle = "#101217";
  ctx.fillRect(0, height - 24, width, 24);
}

function drawReplayMetricLine(
  ctx: CanvasRenderingContext2D,
  points: Array<{ time: number; value: number }>,
  plot: { left: number; right: number; top: number; bottom: number },
  xFor: (time: number) => number,
  yFor: (value: number) => number,
  color: string,
): void {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  points.forEach((point, index) => {
    const x = clampNumber(xFor(point.time), plot.left, plot.right);
    const y = clampNumber(yFor(point.value), plot.top, plot.bottom);
    if (index === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });
  ctx.stroke();
  ctx.restore();
}

function drawReplayMetricSelection(
  ctx: CanvasRenderingContext2D,
  metric: BacktestReplayMetricDefinition,
  points: Array<{ time: number; value: number }>,
  plot: { left: number; right: number; top: number; bottom: number },
  xFor: (time: number) => number,
  yFor: (value: number) => number,
  selectedTime: number | undefined,
): void {
  if (!selectedTime || points.length === 0) {
    return;
  }

  let nearest = points[0];
  let distance = Number.POSITIVE_INFINITY;
  for (const point of points) {
    const nextDistance = Math.abs(point.time - selectedTime);
    if (nextDistance < distance) {
      nearest = point;
      distance = nextDistance;
    }
  }

  const x = clampNumber(xFor(nearest.time), plot.left, plot.right);
  const y = clampNumber(yFor(nearest.value), plot.top, plot.bottom);

  ctx.save();
  ctx.strokeStyle = "#d6dbea";
  ctx.globalAlpha = 0.7;
  ctx.setLineDash([4, 5]);
  ctx.beginPath();
  ctx.moveTo(x, plot.top);
  ctx.lineTo(x, plot.bottom);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.globalAlpha = 1;
  ctx.fillStyle = metric.color;
  ctx.strokeStyle = "#090a0d";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(x, y, 4, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = "#f4f6fb";
  ctx.font = "12px Inter, sans-serif";
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  ctx.fillText(
    `${metric.label}: ${metric.format(nearest.value)}`,
    plot.left + 8,
    plot.top + 8,
  );
  ctx.restore();
}

function findReplayCandle(candles: Candle[], time: number | undefined): Candle | undefined {
  if (candles.length === 0) {
    return undefined;
  }
  if (!time) {
    return candles.at(-1);
  }

  let nearest = candles[0];
  let nearestDistance = Number.POSITIVE_INFINITY;
  for (const candle of candles) {
    if (time >= candle.openTime && time <= candle.closeTime) {
      return candle;
    }

    const distance = Math.min(
      Math.abs(time - candle.openTime),
      Math.abs(time - candle.closeTime),
    );
    if (distance < nearestDistance) {
      nearest = candle;
      nearestDistance = distance;
    }
  }

  return nearest;
}

function findReplayFrame<T extends { time: number }>(
  frames: T[],
  time: number | undefined,
): T | undefined {
  if (frames.length === 0) {
    return undefined;
  }
  if (time === undefined) {
    return frames.at(-1);
  }

  let nearest = frames[0];
  let nearestDistance = Number.POSITIVE_INFINITY;
  for (const frame of frames) {
    const distance = Math.abs(frame.time - time);
    if (distance < nearestDistance) {
      nearest = frame;
      nearestDistance = distance;
    }
  }

  return nearest;
}

function replayMarketState(frame: BacktestMetricFrame | undefined): string {
  return frame && "marketState" in frame ? frame.marketState?.state ?? "-" : "-";
}

function replayAnnotationsForCandle(
  annotations: BacktestChartAnnotation[],
  candle: Candle,
): BacktestChartAnnotation[] {
  return annotations.filter(
    (annotation) => annotation.time >= candle.openTime && annotation.time <= candle.closeTime,
  );
}

function replaySignalLabel(signal: "buy" | "sell" | "hold" | undefined): string {
  return signal ? signal.toUpperCase() : "-";
}

function BacktestNumberInput(props: {
  label: string;
  value: number;
  min: number;
  max: number;
  disabled?: boolean;
  onChange: (value: number) => void;
}) {
  const handleInput = (rawValue: string) => {
    const value = Number(rawValue);
    if (!Number.isFinite(value)) {
      return;
    }

    props.onChange(clampNumber(Math.round(value), props.min, props.max));
  };

  return (
    <label class="block min-w-0">
      <span class="muted-label">{props.label}</span>
      <input
        class="mt-1 w-full rounded-2 border border-line bg-ink-900 px-2 py-2 text-sm text-ink-100 tabular-nums disabled:opacity-60"
        type="number"
        min={props.min}
        max={props.max}
        step={1}
        value={props.value}
        disabled={props.disabled}
        onInput={(event) => handleInput(event.currentTarget.value)}
      />
    </label>
  );
}

function clampNumber(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, Number.isFinite(value) ? value : min));
}

function normalizeBinanceCredentialMode(value: string | undefined): BinanceExchangeMode {
  return value === "live" || value?.endsWith("-live") ? "live" : "auto";
}

function Side(props: { side: "buy" | "sell" }) {
  return (
    <span
      class="rounded-2 px-2 py-1 text-xs font-semibold uppercase"
      classList={{
        "bg-gain/12 text-gain": props.side === "buy",
        "bg-loss/12 text-loss": props.side === "sell",
      }}
    >
      {props.side}
    </span>
  );
}

function EmptyRow(props: { columns: number; label: string }) {
  return (
    <tr>
      <td class="border-t border-line py-4 text-center text-sm text-ink-300" colSpan={props.columns}>
        {props.label}
      </td>
    </tr>
  );
}
