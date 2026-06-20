import { For, Show, createEffect, createMemo, createSignal, onCleanup, onMount } from "solid-js";
import {
  Activity,
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
  BotEvent,
  LongPositionLot,
  ManualTradeInput,
  PositionLedger,
  ShortPositionLot,
  StrategyConfig,
  TradeFill,
  TradingOrder,
} from "@trading/bot-algo";
import { CandleChart } from "./components/CandleChart";
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
  BinanceMarketCatalog,
  BinanceMarketListing,
  MarketGroup,
  RuntimeSnapshot,
} from "./types";

const apiBase =
  import.meta.env.VITE_API_URL ??
  (window.location.port === "5173" ? "http://localhost:3001" : window.location.origin);
const wsUrl = apiBase.replace(/^http/, "ws").replace(/\/$/, "") + "/ws";

interface BacktestSettings {
  historicalDays: number;
  randomSampleCount: number;
  randomWindowDays: number;
  randomMinWindowDays: number;
  randomMaxWindowDays: number;
  randomLookbackDays: number;
}

const defaultBacktestSettings: BacktestSettings = {
  historicalDays: 30,
  randomSampleCount: 40,
  randomWindowDays: 7,
  randomMinWindowDays: 1,
  randomMaxWindowDays: 30,
  randomLookbackDays: 365,
};

export function App() {
  const [snapshot, setSnapshot] = createSignal<RuntimeSnapshot>();
  const [connection, setConnection] = createSignal<"connecting" | "live" | "offline">(
    "connecting",
  );
  const [backtestPreset, setBacktestPreset] =
    createSignal<BacktestSelection>("saved-candles");
  const [backtestSettings, setBacktestSettings] = createSignal<BacktestSettings>({
    ...defaultBacktestSettings,
  });
  const [backtestError, setBacktestError] = createSignal<string>();
  const [configDraft, setConfigDraft] = createSignal<StrategyConfig>();
  const [configError, setConfigError] = createSignal<string>();
  const [manualTradeError, setManualTradeError] = createSignal<string>();
  const [marketCatalog, setMarketCatalog] = createSignal<BinanceMarketCatalog>();
  const [marketError, setMarketError] = createSignal<string>();
  const [switchingMarketId, setSwitchingMarketId] = createSignal<string>();
  let socket: WebSocket | undefined;
  let reconnectTimer: number | undefined;

  const bot = createMemo(() => snapshot()?.bot);
  const market = createMemo(() => snapshot()?.market);
  const metrics = createMemo(() => bot()?.metrics);
  const openOrders = createMemo(() =>
    (bot()?.orders ?? []).filter((order) => order.status === "open").slice().reverse(),
  );
  const fills = createMemo(() => (bot()?.fills ?? []).slice(-12).reverse());
  const events = createMemo(() => snapshot()?.recentEvents ?? []);
  const backtest = createMemo(() => snapshot()?.backtest);

  createEffect(() => {
    const config = bot()?.config;
    if (config && (!configDraft() || configDraft()?.symbol !== config.symbol)) {
      setConfigDraft(structuredClone(config));
    }
  });

  const loadInitial = async () => {
    const response = await fetch(`${apiBase}/api/state`);
    if (!response.ok) {
      throw new Error(`State request failed: ${response.status}`);
    }
    setSnapshot((await response.json()) as RuntimeSnapshot);
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

  const connect = () => {
    setConnection("connecting");
    socket = new WebSocket(wsUrl);

    socket.onopen = () => setConnection("live");
    socket.onmessage = (event) => {
      const message = JSON.parse(event.data) as {
        type: "snapshot";
        payload: RuntimeSnapshot;
      };
      if (message.type === "snapshot") {
        setSnapshot(message.payload);
      }
    };
    socket.onclose = () => {
      setConnection("offline");
      reconnectTimer = window.setTimeout(connect, 1_500);
    };
    socket.onerror = () => {
      socket?.close();
    };
  };

  const controlBot = async (action: "start" | "stop" | "reset") => {
    const response = await fetch(`${apiBase}/api/bot/${action}`, {
      method: "POST",
    });
    if (response.ok) {
      setSnapshot((await response.json()) as RuntimeSnapshot);
    }
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
    setSnapshot(nextSnapshot);
    setConfigDraft(structuredClone(nextSnapshot.bot.config));
  };

  const runBacktest = async () => {
    setBacktestError(undefined);
    const preset = backtestPreset();
    const settings = backtestSettings();
    const body: {
      preset: BacktestSelection;
      limit: number;
      historicalDays?: number;
      randomSampleCount?: number;
      randomWindowDays?: number;
      randomMinWindowDays?: number;
      randomMaxWindowDays?: number;
      randomLookbackDays?: number;
    } = {
      preset,
      limit: preset === "saved-orderbook" ? 3_000 : 1_000,
    };

    if (preset === "last-x") {
      body.historicalDays = settings.historicalDays;
    }

    if (preset === "random-windows") {
      body.randomSampleCount = settings.randomSampleCount;
      body.randomWindowDays = settings.randomWindowDays;
      body.randomLookbackDays = settings.randomLookbackDays;
    }

    if (preset === "random-length-windows") {
      body.randomSampleCount = settings.randomSampleCount;
      body.randomMinWindowDays = settings.randomMinWindowDays;
      body.randomMaxWindowDays = settings.randomMaxWindowDays;
      body.randomLookbackDays = settings.randomLookbackDays;
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

    setSnapshot(payload as RuntimeSnapshot);
  };

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

    setSnapshot(payload as RuntimeSnapshot);
    setConfigDraft(structuredClone((payload as RuntimeSnapshot).bot.config));
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

    setSnapshot(payload as RuntimeSnapshot);
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
  });

  onCleanup(() => {
    socket?.close();
    if (reconnectTimer) {
      window.clearTimeout(reconnectTimer);
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
              <StatusPill label={bot()?.status ?? "starting"} active={bot()?.status === "running"} />
            </div>
          </div>

          <div class="flex flex-wrap items-center gap-2">
            <button class="btn-primary" onClick={() => void controlBot("start")}>
              <Play size={16} />
              Start
            </button>
            <button class="btn" onClick={() => void controlBot("stop")}>
              <Square size={16} />
              Stop
            </button>
            <button class="btn" onClick={() => void controlBot("reset")}>
              <RotateCcw size={16} />
              Reset
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
          <MetricCard label="Quote Free" value={`$${formatQuote(bot()?.quoteFree, 2)}`} />
          <MetricCard
            label={`${bot()?.baseAsset ?? "Base"} Free`}
            value={formatAsset(bot()?.baseFree)}
          />
          <MetricCard label="Open Orders" value={openOrders().length.toString()} />
        </section>

        <AlgorithmPanel
          config={configDraft()}
          error={configError()}
          onChange={setConfigDraft}
          onApply={() => void applyConfig()}
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
                orders={bot()?.orders ?? []}
                lastPrice={market()?.lastPrice ?? 0}
              />
            </div>
          </div>

          <div class="grid grid-cols-1 gap-4">
            <PerformancePanel snapshot={snapshot()} />
            <OrderBookPanel snapshot={snapshot()} />
          </div>
        </section>

        <section class="grid min-w-0 grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1fr)_420px]">
          <div class="grid min-w-0 grid-cols-1 gap-4 lg:grid-cols-2">
            <OrdersPanel title="Open Orders" orders={openOrders()} />
            <FillsPanel fills={fills()} />
          </div>
          <EventsPanel events={events()} />
        </section>

        <PositionLedgerPanel
          ledger={snapshot()?.positions}
          baseAsset={bot()?.baseAsset ?? "Base"}
          quoteAsset={bot()?.quoteAsset ?? "USDT"}
          currentPrice={market()?.lastPrice ?? bot()?.lastPrice ?? 0}
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
          onRun={() => void runBacktest()}
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
        <button class="btn px-2.5" onClick={props.onRefresh} disabled={props.disabled}>
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
          <span class="sr-only">Search markets</span>
          <Search size={15} class="pointer-events-none absolute left-2.5 top-2.5 text-ink-300" />
          <input
            class="w-full rounded-2 border border-line bg-ink-800 py-2 pl-8 pr-2 text-sm text-ink-100"
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
  return `${market.displaySymbol} | ${venueLabel(market.venue)} | ${market.symbol}${contract}${unavailable}`;
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
  config?: StrategyConfig;
  error?: string;
  onChange: (config: StrategyConfig) => void;
  onApply: () => void;
}) {
  const update = <K extends keyof StrategyConfig>(key: K, value: StrategyConfig[K]) => {
    if (!props.config) {
      return;
    }

    props.onChange({
      ...props.config,
      [key]: value,
    });
  };
  const updateLegacy = <K extends keyof StrategyConfig["legacyValleyPeak"]>(
    key: K,
    value: StrategyConfig["legacyValleyPeak"][K],
  ) => {
    if (!props.config) {
      return;
    }

    props.onChange({
      ...props.config,
      legacyValleyPeak: {
        ...props.config.legacyValleyPeak,
        [key]: value,
      },
    });
  };
  const updateRisk = <K extends keyof StrategyConfig["positionRisk"]>(
    key: K,
    value: StrategyConfig["positionRisk"][K],
  ) => {
    if (!props.config) {
      return;
    }

    props.onChange({
      ...props.config,
      positionRisk: {
        ...props.config.positionRisk,
        [key]: value,
      },
    });
  };

  return (
    <section class="panel">
      <div class="mb-4 flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <div class="muted-label">Algorithm</div>
          <h2 class="text-lg font-semibold">
            {props.config?.algorithm === "legacy-valley-peak"
              ? "Legacy Valley/Peak"
              : "Moving Average"}
          </h2>
        </div>
        <button class="btn-primary" disabled={!props.config} onClick={props.onApply}>
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
          <div class="grid grid-cols-1 gap-4 xl:grid-cols-4">
            <div class="rounded-2 bg-ink-800 p-3">
              <label class="muted-label block" for="algorithm-select">
                Strategy
              </label>
              <select
                id="algorithm-select"
                class="mt-2 w-full rounded-2 border border-line bg-ink-900 px-3 py-2 text-sm text-ink-100"
                value={config().algorithm}
                onInput={(event) =>
                  update("algorithm", event.currentTarget.value as StrategyConfig["algorithm"])
                }
              >
                <option value="moving-average">Moving average</option>
                <option value="legacy-valley-peak">Legacy valley/peak</option>
              </select>
              <div class="mt-3 grid grid-cols-2 gap-3">
                <NumberField
                  label="Order USDT"
                  value={config().orderQuoteSize}
                  min={1}
                  onInput={(value) => update("orderQuoteSize", value)}
                />
                <NumberField
                  label="Max Position"
                  value={config().maxPositionQuote}
                  min={1}
                  onInput={(value) => update("maxPositionQuote", value)}
                />
                <NumberField
                  label="Max Orders"
                  value={config().maxOpenOrders}
                  min={1}
                  step={1}
                  onInput={(value) => update("maxOpenOrders", value)}
                />
                <NumberField
                  label="Cooldown sec"
                  value={config().cooldownMs / 1000}
                  min={0}
                  onInput={(value) => update("cooldownMs", value * 1000)}
                />
              </div>
            </div>

            <div class="rounded-2 bg-ink-800 p-3">
              <div class="muted-label">Moving Average</div>
              <div class="mt-3 grid grid-cols-2 gap-3">
                <NumberField
                  label="Fast Window"
                  value={config().fastWindow}
                  min={2}
                  step={1}
                  onInput={(value) => update("fastWindow", value)}
                />
                <NumberField
                  label="Slow Window"
                  value={config().slowWindow}
                  min={3}
                  step={1}
                  onInput={(value) => update("slowWindow", value)}
                />
                <NumberField
                  label="Signal bps"
                  value={config().signalThresholdBps}
                  min={0}
                  onInput={(value) => update("signalThresholdBps", value)}
                />
                <NumberField
                  label="Limit bps"
                  value={config().limitOffsetBps}
                  min={0}
                  onInput={(value) => update("limitOffsetBps", value)}
                />
              </div>
            </div>

            <div class="rounded-2 bg-ink-800 p-3">
              <div class="muted-label">Legacy Valley/Peak</div>
              <div class="mt-3 grid grid-cols-2 gap-3">
                <NumberField
                  label="Buy Rate"
                  value={config().legacyValleyPeak.buySpendRate}
                  min={0}
                  step={0.05}
                  onInput={(value) => updateLegacy("buySpendRate", value)}
                />
                <NumberField
                  label="Sell Rate"
                  value={config().legacyValleyPeak.sellAmountRate}
                  min={0}
                  step={0.05}
                  onInput={(value) => updateLegacy("sellAmountRate", value)}
                />
                <NumberField
                  label="Buy Sigma"
                  value={config().legacyValleyPeak.buySigma}
                  min={0.000001}
                  step={0.01}
                  onInput={(value) => updateLegacy("buySigma", value)}
                />
                <NumberField
                  label="Sell Sigma"
                  value={config().legacyValleyPeak.sellSigma}
                  min={0.000001}
                  step={0.01}
                  onInput={(value) => updateLegacy("sellSigma", value)}
                />
                <NumberField
                  label="Min USDT"
                  value={config().legacyValleyPeak.minTradeQuote}
                  min={0}
                  onInput={(value) => updateLegacy("minTradeQuote", value)}
                />
                <NumberField
                  label="Max USDT"
                  value={config().legacyValleyPeak.maxTradeQuote}
                  min={1}
                  onInput={(value) => updateLegacy("maxTradeQuote", value)}
                />
                <NumberField
                  label="Warmup min"
                  value={config().legacyValleyPeak.saturationSec / 60}
                  min={0}
                  onInput={(value) => updateLegacy("saturationSec", value * 60)}
                />
                <NumberField
                  label="Buy Confirm"
                  value={config().legacyValleyPeak.buyConfirmationOffset}
                  min={0}
                  step={1}
                  onInput={(value) => updateLegacy("buyConfirmationOffset", value)}
                />
              </div>
            </div>

            <div class="rounded-2 bg-ink-800 p-3">
              <div class="muted-label">Position Risk</div>
              <div class="mt-3 grid grid-cols-2 gap-3">
                <NumberField
                  label="Lower Range"
                  value={config().positionRisk.lowerPriceExpectation}
                  min={0}
                  step={0.01}
                  onInput={(value) => updateRisk("lowerPriceExpectation", value)}
                />
                <NumberField
                  label="Long Base"
                  value={config().positionRisk.lowerBaselinePrice}
                  min={0}
                  step={0.01}
                  onInput={(value) => updateRisk("lowerBaselinePrice", value)}
                />
                <NumberField
                  label="Upper Range"
                  value={config().positionRisk.upperPriceExpectation}
                  min={0}
                  step={0.01}
                  onInput={(value) => updateRisk("upperPriceExpectation", value)}
                />
                <NumberField
                  label="Short Base"
                  value={config().positionRisk.upperBaselinePrice}
                  min={0}
                  step={0.01}
                  onInput={(value) => updateRisk("upperBaselinePrice", value)}
                />
                <NumberField
                  label="Max Loss %"
                  value={config().positionRisk.maxLossPct * 100}
                  min={0}
                  step={0.1}
                  onInput={(value) => updateRisk("maxLossPct", value / 100)}
                />
                <NumberField
                  label="Slip bps"
                  value={config().positionRisk.marketSlippageBps}
                  min={0}
                  step={0.1}
                  onInput={(value) => updateRisk("marketSlippageBps", value)}
                />
                <NumberField
                  label="Qty Floor"
                  value={config().positionRisk.quantityFloor}
                  min={0}
                  step={0.00000001}
                  onInput={(value) => updateRisk("quantityFloor", value)}
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
  value: number;
  min?: number;
  step?: number;
  onInput: (value: number) => void;
}) {
  return (
    <label class="block">
      <span class="muted-label">{props.label}</span>
      <input
        class="mt-1 w-full rounded-2 border border-line bg-ink-900 px-2 py-2 text-sm text-ink-100 tabular-nums"
        type="number"
        min={props.min}
        step={props.step ?? 1}
        value={Number.isFinite(props.value) ? props.value : 0}
        onInput={(event) => props.onInput(Number(event.currentTarget.value))}
      />
    </label>
  );
}

function PerformancePanel(props: { snapshot?: RuntimeSnapshot }) {
  const metrics = () => props.snapshot?.bot.metrics;
  const bot = () => props.snapshot?.bot;

  return (
    <div class="panel">
      <div class="mb-3 flex items-center justify-between">
        <div>
          <div class="muted-label">Performance</div>
          <h2 class="text-lg font-semibold">Paper Bot</h2>
        </div>
        <Activity size={18} class="text-accent" />
      </div>
      <div class="grid grid-cols-2 gap-3">
        <SmallMetric label="Realized PnL" value={`$${formatQuote(metrics()?.realizedPnl, 2)}`} />
        <SmallMetric label="Unrealized" value={`$${formatQuote(metrics()?.unrealizedPnl, 2)}`} />
        <SmallMetric label="Win Rate" value={formatPercent(metrics()?.winRate)} />
        <SmallMetric label="Max Drawdown" value={formatPercent(metrics()?.maxDrawdownPct)} />
        <SmallMetric label="Fees" value={`$${formatQuote(metrics()?.feesPaid, 2)}`} />
        <SmallMetric label="Exposure" value={formatPercent(metrics()?.exposurePct)} />
      </div>
      <div class="mt-3 rounded-2 bg-ink-800 p-3 text-sm text-ink-300">
        Avg entry{" "}
        <span class="text-ink-100 tabular-nums">${formatQuote(bot()?.avgEntryPrice, 2)}</span>
      </div>
    </div>
  );
}

function SmallMetric(props: { label: string; value: string }) {
  return (
    <div class="rounded-2 bg-ink-800 p-3">
      <div class="muted-label">{props.label}</div>
      <div class="mt-1 text-base font-semibold tabular-nums text-ink-100">{props.value}</div>
    </div>
  );
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

function OrdersPanel(props: { title: string; orders: TradingOrder[] }) {
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
              <th class="table-head pb-2">Reason</th>
              <th class="table-head pb-2">Created</th>
            </tr>
          </thead>
          <tbody>
            <For each={props.orders} fallback={<EmptyRow columns={5} label="No open orders" />}>
              {(order) => (
                <tr>
                  <td class="td-cell">
                    <Side side={order.side} />
                  </td>
                  <td class="td-cell">${formatQuote(order.price, 2)}</td>
                  <td class="td-cell">{formatAsset(order.quantity)}</td>
                  <td class="td-cell text-ink-300">{order.reason}</td>
                  <td class="td-cell text-ink-300">{formatTime(order.createdAt)}</td>
                </tr>
              )}
            </For>
          </tbody>
        </table>
      </div>
    </div>
  );
}

function FillsPanel(props: { fills: TradeFill[] }) {
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
              <th class="table-head pb-2">Side</th>
              <th class="table-head pb-2">Price</th>
              <th class="table-head pb-2">Qty</th>
              <th class="table-head pb-2">PnL</th>
              <th class="table-head pb-2">Filled</th>
            </tr>
          </thead>
          <tbody>
            <For each={props.fills} fallback={<EmptyRow columns={5} label="No fills yet" />}>
              {(fill) => (
                <tr>
                  <td class="td-cell">
                    <Side side={fill.side} />
                  </td>
                  <td class="td-cell">${formatQuote(fill.price, 2)}</td>
                  <td class="td-cell">{formatAsset(fill.quantity)}</td>
                  <td
                    class="td-cell"
                    classList={{
                      "text-gain": fill.realizedPnl > 0,
                      "text-loss": fill.realizedPnl < 0,
                      "text-ink-300": fill.realizedPnl === 0,
                    }}
                  >
                    ${formatQuote(fill.realizedPnl, 2)}
                  </td>
                  <td class="td-cell text-ink-300">{formatTime(fill.filledAt)}</td>
                </tr>
              )}
            </For>
          </tbody>
        </table>
      </div>
    </div>
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
  const longs = () => props.ledger?.longs ?? [];
  const shorts = () => props.ledger?.shorts ?? [];
  const [draft, setDraft] = createSignal<ManualTradeDraft>();
  const [submitting, setSubmitting] = createSignal(false);
  const currentPrice = () => props.currentPrice || summary()?.currentPrice || 0;
  const openDraft = (draft: ManualTradeDraft) => {
    setDraft({
      ...draft,
      price: draft.price || currentPrice(),
    });
  };
  const submitDraft = async () => {
    const value = draft();
    if (!value || submitting()) {
      return;
    }

    setSubmitting(true);
    const ok = await props.onRecordTrade({
      side: value.side,
      price: value.price,
      quantity: value.quantity,
      reason: value.reason,
      targetPositionId: value.targetPositionId,
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
          <button
            class="btn"
            onClick={() =>
              openDraft({
                title: "Add Long",
                side: "buy",
                quantity: 0,
                price: currentPrice(),
                reason: "manual long position",
              })
            }
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
                price: currentPrice(),
                reason: "manual short position",
              })
            }
          >
            <Plus size={16} />
            Short
          </button>
        </div>
      </div>

      <Show when={props.error}>
        {(message) => (
          <div class="mb-3 rounded-2 bg-loss/12 p-3 text-sm text-loss">{message()}</div>
        )}
      </Show>

      <div class="mb-4 grid grid-cols-2 gap-3 lg:grid-cols-4 xl:grid-cols-8">
        <SmallMetric label="Net Sell" value={`$${formatQuote(summary()?.netMarketSellPrice, 2)}`} />
        <SmallMetric label="Gross Buy" value={`$${formatQuote(summary()?.grossMarketBuyPrice, 2)}`} />
        <SmallMetric label="Long Base" value={`$${formatQuote(summary()?.lowerBaselinePrice, 2)}`} />
        <SmallMetric label="Short Base" value={`$${formatQuote(summary()?.upperBaselinePrice, 2)}`} />
        <SmallMetric label="Long Left" value={formatAsset(summary()?.longQuantity)} />
        <SmallMetric label="Short Left" value={formatAsset(summary()?.shortQuantity)} />
        <SmallMetric label="Pending Buy" value={formatAsset(summary()?.pendingLongQuantity)} />
        <SmallMetric label="Pending Sell" value={formatAsset(summary()?.pendingShortQuantity)} />
      </div>

      <Show when={draft()}>
        {(value) => (
          <ManualTradeForm
            draft={value()}
            baseAsset={props.baseAsset}
            quoteAsset={props.quoteAsset}
            submitting={submitting()}
            onChange={setDraft}
            onCancel={() => setDraft(undefined)}
            onSubmit={() => void submitDraft()}
          />
        )}
      </Show>

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
              price: currentPrice(),
              targetPositionId: lot.id,
              reason: `manual close ${lot.sourceOrderId}`,
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
              price: currentPrice(),
              targetPositionId: lot.id,
              reason: `manual close ${lot.sourceOrderId}`,
            })
          }
        />
      </div>
    </section>
  );
}

type ManualTradeDraft = ManualTradeInput & {
  title: string;
  price: number;
  reason: string;
};

function ManualTradeForm(props: {
  draft: ManualTradeDraft;
  baseAsset: string;
  quoteAsset: string;
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
          <button class="btn-primary" type="submit" disabled={props.submitting}>
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
        <NumberField
          label={`${props.baseAsset} Qty`}
          value={props.draft.quantity}
          min={0}
          step={0.00000001}
          onInput={(value) => update("quantity", value)}
        />
        <NumberField
          label={`${props.quoteAsset} Price`}
          value={props.draft.price}
          min={0}
          step={0.01}
          onInput={(value) => update("price", value)}
        />
        <label class="block">
          <span class="muted-label">Reason</span>
          <input
            class="mt-1 w-full rounded-2 border border-line bg-ink-900 px-2 py-2 text-sm text-ink-100"
            value={props.draft.reason}
            onInput={(event) => update("reason", event.currentTarget.value)}
          />
        </label>
      </div>
    </form>
  );
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
        <table class="w-full min-w-280">
          <thead>
            <tr>
              <th class="table-head pb-2">Status</th>
              <th class="table-head pb-2">Order</th>
              <th class="table-head pb-2">Bought</th>
              <th class="table-head pb-2">Pending</th>
              <th class="table-head pb-2">Closed</th>
              <th class="table-head pb-2">Cost</th>
              <th class="table-head pb-2">Avg</th>
              <th class="table-head pb-2">Left Sell</th>
              <th class="table-head pb-2">Left Cost</th>
              <th class="table-head pb-2">Break Even</th>
              <th class="table-head pb-2">Max Loss</th>
              <th class="table-head pb-2">Sell Now</th>
              <th class="table-head pb-2">Possible</th>
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
                  <td class="td-cell">{formatAsset(lot.filledQuantity || lot.originalQuantity)}</td>
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
                  <td class="td-cell">${formatQuote(lot.costQuote, 2)}</td>
                  <td class="td-cell">${formatQuote(lot.averagePrice, 4)}</td>
                  <td class="td-cell">{formatAsset(lot.remainingQuantity)}</td>
                  <td class="td-cell">${formatQuote(lot.remainingCostQuote, 2)}</td>
                  <td class="td-cell">${formatQuote(lot.breakEvenSellPrice, 4)}</td>
                  <td class="td-cell">${formatQuote(lot.maxLossSellPrice, 4)}</td>
                  <td class="td-cell">
                    <ActionAmount quantity={lot.recommendedSellQuantity} quote={lot.recommendedSellQuote} />
                  </td>
                  <td class="td-cell">
                    <PossibleBadge possible={lot.canReachLowerBaseline} />
                  </td>
                  <td class="td-cell">
                    <button
                      class="btn px-2 py-1 text-xs"
                      disabled={lot.status === "pending" || lot.remainingQuantity <= 0}
                      onClick={() => props.onClose(lot)}
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
        <table class="w-full min-w-280">
          <thead>
            <tr>
              <th class="table-head pb-2">Status</th>
              <th class="table-head pb-2">Order</th>
              <th class="table-head pb-2">Sold</th>
              <th class="table-head pb-2">Pending</th>
              <th class="table-head pb-2">Closed</th>
              <th class="table-head pb-2">Proceeds</th>
              <th class="table-head pb-2">Avg</th>
              <th class="table-head pb-2">Left Buy</th>
              <th class="table-head pb-2">Left Proceeds</th>
              <th class="table-head pb-2">Break Even</th>
              <th class="table-head pb-2">Max Loss</th>
              <th class="table-head pb-2">Buy Now</th>
              <th class="table-head pb-2">Possible</th>
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
                  <td class="td-cell">{formatAsset(lot.filledQuantity || lot.originalQuantity)}</td>
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
                  <td class="td-cell">${formatQuote(lot.proceedsQuote, 2)}</td>
                  <td class="td-cell">${formatQuote(lot.averagePrice, 4)}</td>
                  <td class="td-cell">{formatAsset(lot.remainingQuantity)}</td>
                  <td class="td-cell">${formatQuote(lot.remainingProceedsQuote, 2)}</td>
                  <td class="td-cell">${formatQuote(lot.breakEvenBuyPrice, 4)}</td>
                  <td class="td-cell">${formatQuote(lot.maxLossBuyPrice, 4)}</td>
                  <td class="td-cell">
                    <ActionAmount quantity={lot.recommendedBuyQuantity} quote={lot.recommendedBuyQuote} />
                  </td>
                  <td class="td-cell">
                    <PossibleBadge possible={lot.canReachUpperBaseline} />
                  </td>
                  <td class="td-cell">
                    <button
                      class="btn px-2 py-1 text-xs"
                      disabled={lot.status === "pending" || lot.remainingQuantity <= 0}
                      onClick={() => props.onClose(lot)}
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

function PendingCell(props: {
  quantity: number;
  quote: number;
  price: number;
  quoteAsset: string;
}) {
  if (props.quantity <= 0) {
    return <span class="text-ink-500">-</span>;
  }

  return (
    <span>
      {formatAsset(props.quantity)} @ ${formatQuote(props.price, 4)}
      <span class="ml-1 text-ink-400">
        {formatQuote(props.quote, 2)} {props.quoteAsset}
      </span>
    </span>
  );
}

function ActionAmount(props: { quantity: number; quote: number }) {
  if (props.quantity <= 0 || props.quote <= 0) {
    return <span class="text-ink-500">-</span>;
  }

  return (
    <span>
      {formatAsset(props.quantity)}
      <span class="ml-1 text-ink-400">${formatQuote(props.quote, 2)}</span>
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

function EventsPanel(props: { events: BotEvent[] }) {
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
                    {order().id} at ${formatQuote(order().price, 2)}
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
  onRun: () => void;
}) {
  const result = () => props.progress?.result;
  const summary = () => result()?.summary;
  const isRunning = () => props.progress?.status === "running";
  const progressPercent = () => Math.max(0, Math.min(100, props.progress?.percent ?? 0));
  const error = () => props.error ?? props.progress?.error;
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
          <button class="btn-primary" disabled={isRunning()} onClick={props.onRun}>
            <RefreshCw size={16} class={isRunning() ? "animate-spin" : ""} />
            Run
          </button>
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

      <Show when={props.preset === "random-windows"}>
        <div class="mb-4 grid grid-cols-1 gap-3 rounded-2 bg-ink-800 p-3 sm:grid-cols-3">
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
        </div>
      </Show>

      <Show when={props.preset === "random-length-windows"}>
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
          <SmallMetric label="Trades" value={formatQuote(summary()?.tradeCount, 0)} />
          <SmallMetric label="Win Rate" value={formatPercent(summary()?.winRate)} />
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
                windows
              </span>
            </Show>
          </div>
        )}
      </Show>
    </section>
  );
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
  return Math.max(min, Math.min(max, value));
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
