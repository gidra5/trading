import { For, Show, createEffect, createMemo, createSignal, onCleanup, onMount } from "solid-js";
import { Activity, Play, RefreshCw, RotateCcw, Save, Square } from "lucide-solid";
import type { BotEvent, StrategyConfig, TradeFill, TradingOrder } from "@trading/bot-algo";
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
  RuntimeSnapshot,
} from "./types";

const apiBase =
  import.meta.env.VITE_API_URL ??
  (window.location.port === "5173" ? "http://localhost:3001" : window.location.origin);
const wsUrl = apiBase.replace(/^http/, "ws").replace(/\/$/, "") + "/ws";

export function App() {
  const [snapshot, setSnapshot] = createSignal<RuntimeSnapshot>();
  const [connection, setConnection] = createSignal<"connecting" | "live" | "offline">(
    "connecting",
  );
  const [backtestPreset, setBacktestPreset] =
    createSignal<BacktestSelection>("saved-candles");
  const [backtestError, setBacktestError] = createSignal<string>();
  const [configDraft, setConfigDraft] = createSignal<StrategyConfig>();
  const [configError, setConfigError] = createSignal<string>();
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
    if (config && !configDraft()) {
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

  const runBacktest = async () => {
    setBacktestError(undefined);
    const response = await fetch(`${apiBase}/api/backtest`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
      },
      body: JSON.stringify({
        preset: backtestPreset(),
        limit: backtestPreset() === "saved-orderbook" ? 3_000 : 1_000,
      }),
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

  onMount(() => {
    void loadInitial().catch(() => setConnection("offline"));
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
          <div class="flex flex-wrap items-center gap-3">
            <div>
              <div class="muted-label">Trading Pair</div>
              <h1 class="text-2xl font-semibold tabular-nums">{market()?.symbol ?? "BTCUSDT"}</h1>
            </div>
            <StatusPill label={connection()} active={connection() === "live"} />
            <StatusPill
              label={market()?.connected ? "Binance live" : "Binance offline"}
              active={Boolean(market()?.connected)}
            />
            <StatusPill label={bot()?.status ?? "starting"} active={bot()?.status === "running"} />
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
          <MetricCard label="Last Price" value={`$${formatQuote(market()?.lastPrice, 2)}`} />
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

        <section class="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1fr)_420px]">
          <div class="grid grid-cols-1 gap-4 lg:grid-cols-2">
            <OrdersPanel title="Open Orders" orders={openOrders()} />
            <FillsPanel fills={fills()} />
          </div>
          <EventsPanel events={events()} />
        </section>

        <BacktestPanel
          preset={backtestPreset()}
          onPresetChange={setBacktestPreset}
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
          <div class="grid grid-cols-1 gap-4 xl:grid-cols-3">
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
    <div class="panel overflow-hidden">
      <div class="mb-3">
        <div class="muted-label">Orders</div>
        <h2 class="text-lg font-semibold">{props.title}</h2>
      </div>
      <div class="overflow-x-auto">
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
    <div class="panel overflow-hidden">
      <div class="mb-3">
        <div class="muted-label">Executions</div>
        <h2 class="text-lg font-semibold">Recent Fills</h2>
      </div>
      <div class="overflow-x-auto">
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
            <option value="week">Last week</option>
            <option value="month">Last month</option>
            <option value="year">Last year</option>
          </select>
          <button class="btn-primary" disabled={isRunning()} onClick={props.onRun}>
            <RefreshCw size={16} class={isRunning() ? "animate-spin" : ""} />
            Run
          </button>
        </div>
      </div>

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
          <SmallMetric label="Drawdown" value={formatPercent(summary()?.maxDrawdownPct)} />
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
          </div>
        )}
      </Show>
    </section>
  );
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
