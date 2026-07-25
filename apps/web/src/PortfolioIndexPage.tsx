import {
  For,
  Show,
  createEffect,
  createMemo,
  createSignal,
  onCleanup,
  onMount,
} from "solid-js";
import {
  Activity,
  ArrowLeft,
  BarChart3,
  RefreshCw,
  Search,
} from "lucide-solid";

const apiBase = import.meta.env.DEV ? "/backend" : "";
const POLL_MS = 2_000;
const DAY_MS = 86_400_000;

type ScenarioKey = "gross" | "feeOnly" | "baseline" | "conservative";
type RangeKey = "1m" | "3m" | "6m" | "1y";

interface PerformanceSummary {
  initialLevel: number;
  finalLevel: number;
  totalReturn: number;
  annualizedVolatility: number;
  annualizedSharpe: number;
  maximumDrawdown: number;
  cumulativeTransactionCost: number;
  cumulativeFundingCashflow: number;
  cumulativeTradedNotional: number;
}

interface ScaleReport {
  id: string;
  label: string;
  events: number;
  meanEligibleAssets: number;
  meanBasisSize: number;
  coverageTargetReachedRatio: number;
  meanSleeveExposure: number;
  sleeveWeight: number;
}

interface MonthlyPath {
  month: string;
  gross: MonthlyScenario;
  feeOnly: MonthlyScenario;
  baseline: MonthlyScenario;
  conservative: MonthlyScenario;
}

interface MonthlyScenario {
  startLevel: number;
  endLevel: number;
  return: number;
}

interface SelectedMarket {
  market: string;
  asset: string;
  venue: string;
  selectionEvents: number;
  selectionEventShare: number;
}

interface PortfolioIndexReport {
  generatedAt: string;
  window: {
    start: string;
    end: string;
    minuteCandles: number;
  };
  methodology: {
    name: string;
    decisionTiming: string;
    exposureRange: [number, number];
    borrowing: boolean;
    shorting: boolean;
    leverage: boolean;
    perpetualFunding: string;
    optionsTreatment: string;
    friction: {
      chargedOn: string;
      spotFeeBps: number;
      futuresFeeBps: number;
      baselineAdditionalExecutionBps: number;
      conservativeAdditionalExecutionBps: number;
      capacityImpact: string;
    };
    selection: {
      sampleCount: number;
      targetMedianRSquared: number;
      targetP10RSquared: number;
      maximumConstituentWeight: number;
      amplitudePriority: string;
      sizeMeasure: string;
    };
  };
  performance: Record<ScenarioKey, PerformanceSummary>;
  scales: ScaleReport[];
  monthly: MonthlyPath[];
  portfolio: {
    meanExposure: number;
    minimumExposure: number;
    maximumExposure: number;
    meanActiveConstituents: number;
    medianGrossTradedNotionalPerMinute: number;
    meanGrossTradedNotionalPerMinute: number;
    summedGrossTurnover: number;
    meanMissingNextReturnWeight: number;
    maximumMissingNextReturnWeight: number;
  };
  universe: {
    continuousMarkets: number;
    economicAssets: number;
    markets: Record<string, number>;
    catalog: {
      allProductRows: number;
      currentOptionListings: number;
      currentOptionUnderlyings: number;
    };
    mostFrequentlySelectedMarkets: SelectedMarket[];
  };
  benchmark?: {
    available: boolean;
    finalLevel?: number;
    totalReturn?: number;
    maximumDrawdown?: number;
  };
}

interface CompletedScale {
  scale: {
    id: string;
    label: string;
  };
  events: number;
  meanBasisSize: number;
  meanEligibleAssets: number;
  targetReachedRatio: number;
  meanExposure: number;
}

interface Overview {
  status: "missing" | "building" | "processing" | "complete";
  progress?: {
    scale: string;
    completedEvents: number;
    totalEvents: number;
    ratio: number;
    batchSize: number;
    updatedAt: string;
  };
  completedScales: CompletedScale[];
  report?: PortfolioIndexReport;
}

interface IndexPoint {
  time: number;
  baseline: number;
  gross: number;
  feeOnly: number;
  conservative: number;
  open: number;
  high: number;
  low: number;
  exposure: number;
  cash: number;
  activeConstituents: number;
  turnover: number;
  transactionCost: number;
  fundingCashflow: number;
}

interface IndexSeries {
  sourceUpdatedAt: string;
  from: number;
  to: number;
  sourceMinutes: number;
  sampledPoints: number;
  points: IndexPoint[];
}

const scenarios: Array<{
  key: ScenarioKey;
  label: string;
  shortLabel: string;
  color: string;
  description: string;
}> = [
  {
    key: "gross",
    label: "Gross",
    shortLabel: "Gross",
    color: "#38bdf8",
    description: "Price return only; no fees, execution loss, or funding.",
  },
  {
    key: "feeOnly",
    label: "Fee only",
    shortLabel: "Fees",
    color: "#f5b84b",
    description: "Spot/futures fees and actual funding; no execution loss.",
  },
  {
    key: "baseline",
    label: "Baseline net",
    shortLabel: "Baseline",
    color: "#22c55e",
    description: "Fees, actual funding, and 5 bp execution loss.",
  },
  {
    key: "conservative",
    label: "Conservative",
    shortLabel: "Conservative",
    color: "#f05252",
    description: "Fees, actual funding, and 20 bp execution loss.",
  },
];

export function PortfolioIndexPage() {
  const [overview, setOverview] = createSignal<Overview>();
  const [series, setSeries] = createSignal<IndexSeries>();
  const [scenario, setScenario] = createSignal<ScenarioKey>("baseline");
  const [range, setRange] = createSignal<RangeKey>("1y");
  const [error, setError] = createSignal<string>();
  const [seriesError, setSeriesError] = createSignal<string>();
  const [seriesLoading, setSeriesLoading] = createSignal(false);
  const [now, setNow] = createSignal(Date.now());
  let pollTimer: number | undefined;
  let clockTimer: number | undefined;
  let disposed = false;
  let seriesRequest = 0;

  const report = createMemo(() => overview()?.report);
  const selectedPerformance = createMemo(
    () => report()?.performance[scenario()],
  );
  const selectedScenario = createMemo(
    () => scenarios.find((value) => value.key === scenario()) ?? scenarios[2],
  );

  const loadOverview = async () => {
    try {
      const response = await fetch(`${apiBase}/api/portfolio-index/overview`, {
        cache: "no-store",
      });
      const payload = (await response.json()) as Overview & { error?: string };
      if (!response.ok) {
        throw new Error(payload.error ?? `Overview request failed: ${response.status}`);
      }
      setOverview(payload);
      setError(undefined);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Index overview failed.");
    } finally {
      if (!disposed && overview()?.status !== "complete") {
        pollTimer = window.setTimeout(() => void loadOverview(), POLL_MS);
      }
    }
  };

  const loadSeries = async (activeRange: RangeKey) => {
    const current = report();
    if (!current || overview()?.status !== "complete") return;
    const request = ++seriesRequest;
    const bounds = rangeBounds(current, activeRange);
    setSeriesLoading(true);
    setSeriesError(undefined);
    try {
      const parameters = new URLSearchParams({
        from: String(bounds.from),
        to: String(bounds.to),
        maxPoints: "1800",
      });
      const response = await fetch(
        `${apiBase}/api/portfolio-index/series?${parameters}`,
        { cache: "no-store" },
      );
      const payload = (await response.json()) as IndexSeries & { error?: string };
      if (!response.ok) {
        throw new Error(payload.error ?? `Series request failed: ${response.status}`);
      }
      if (request === seriesRequest) setSeries(payload);
    } catch (reason) {
      if (request === seriesRequest) {
        setSeriesError(
          reason instanceof Error ? reason.message : "Index series request failed.",
        );
      }
    } finally {
      if (request === seriesRequest) setSeriesLoading(false);
    }
  };

  createEffect(() => {
    const currentRange = range();
    const ready = overview()?.status === "complete" && Boolean(report());
    if (ready) void loadSeries(currentRange);
  });

  onMount(() => {
    void loadOverview();
    clockTimer = window.setInterval(() => setNow(Date.now()), 1_000);
  });
  onCleanup(() => {
    disposed = true;
    if (pollTimer !== undefined) window.clearTimeout(pollTimer);
    if (clockTimer !== undefined) window.clearInterval(clockTimer);
  });

  return (
    <main class="min-h-screen bg-ink-950 text-ink-100">
      <div class="mx-auto flex w-full max-w-[96rem] flex-col gap-4 px-4 py-4 lg:px-6">
        <header class="flex flex-col gap-3 border-b border-line pb-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <div class="muted-label">Research / portfolio basis</div>
            <div class="mt-1 flex flex-wrap items-center gap-3">
              <h1 class="text-2xl font-semibold">Binance multiscale index</h1>
              <StatusPill status={overview()?.status ?? "missing"} />
            </div>
            <p class="mt-1 max-w-4xl text-sm text-ink-300">
              Point-in-time orthogonal bases, liquidity-capped sleeves, and
              self-financing long-only portfolio paths with friction.
            </p>
          </div>
          <div class="flex flex-wrap gap-2">
            <button class="btn" type="button" onClick={() => void loadOverview()}>
              <RefreshCw size={16} /> Refresh
            </button>
            <a class="btn" href="#/kama-inspector">
              <Search size={16} /> KAMA Inspector
            </a>
            <a class="btn" href="#/">
              <ArrowLeft size={16} /> Dashboard
            </a>
          </div>
        </header>

        <Show when={error()}>
          {(message) => (
            <div class="rounded-2 border border-loss/50 bg-loss/10 px-4 py-3 text-sm text-loss">
              {message()} Retrying automatically.
            </div>
          )}
        </Show>

        <BuildStatus overview={overview()} now={now()} />

        <Show when={report()} fallback={<PreResult overview={overview()} />}>
          {(value) => (
            <>
              <section class="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                <For each={scenarios}>
                  {(item) => (
                    <ScenarioCard
                      scenario={item}
                      performance={value().performance[item.key]}
                      selected={scenario() === item.key}
                      onSelect={() => setScenario(item.key)}
                    />
                  )}
                </For>
              </section>

              <section class="panel min-w-0">
                <div class="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <div class="muted-label">Portfolio comparison</div>
                    <h2 class="mt-1 text-lg font-semibold">
                      Index level — {selectedScenario().label}
                    </h2>
                    <p class="text-sm text-ink-300">
                      All paths rebalance to the same point-in-time target; only
                      friction and funding treatment differ.
                    </p>
                  </div>
                  <div class="flex flex-wrap gap-1">
                    <For each={(["1m", "3m", "6m", "1y"] as RangeKey[])}>
                      {(item) => (
                        <button
                          type="button"
                          class={range() === item ? "btn border-accent bg-accent/10" : "btn"}
                          onClick={() => setRange(item)}
                        >
                          {item.toUpperCase()}
                        </button>
                      )}
                    </For>
                  </div>
                </div>
                <Show
                  when={!seriesError()}
                  fallback={
                    <div class="mt-3 rounded-2 border border-loss/40 bg-loss/10 p-4 text-sm text-loss">
                      {seriesError()}
                    </div>
                  }
                >
                  <IndexComparisonChart
                    points={series()?.points ?? []}
                    selected={scenario()}
                    loading={seriesLoading()}
                    onSelect={setScenario}
                  />
                </Show>
                <div class="mt-2 flex flex-wrap items-center justify-between gap-2 text-xs text-ink-300">
                  <span>
                    {series()
                      ? `${integer(series()!.sourceMinutes)} source minutes → ${integer(series()!.sampledPoints)} plotted points`
                      : "Waiting for minute candles"}
                  </span>
                  <span>
                    {series()
                      ? `${dateTime(series()!.from)} — ${dateTime(series()!.to)}`
                      : ""}
                  </span>
                </div>
              </section>

              <section class="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
                <MetricCard
                  label="Selected final level"
                  value={indexLevel(selectedPerformance()?.finalLevel)}
                  detail={signedPercent(selectedPerformance()?.totalReturn)}
                  tone={(selectedPerformance()?.totalReturn ?? 0) >= 0 ? "gain" : "loss"}
                />
                <MetricCard
                  label="Maximum drawdown"
                  value={percent(selectedPerformance()?.maximumDrawdown)}
                  detail={`${percent(selectedPerformance()?.annualizedVolatility)} annualized volatility`}
                  tone="loss"
                />
                <MetricCard
                  label="Annualized Sharpe"
                  value={decimal(selectedPerformance()?.annualizedSharpe, 2)}
                  detail={`${money(selectedPerformance()?.cumulativeTransactionCost)} transaction cost`}
                />
                <MetricCard
                  label="Funding cashflow"
                  value={money(selectedPerformance()?.cumulativeFundingCashflow)}
                  detail="Positive is cost; negative is income"
                  tone={(selectedPerformance()?.cumulativeFundingCashflow ?? 0) <= 0 ? "gain" : "warn"}
                />
              </section>

              <section class="grid gap-4 xl:grid-cols-[1.2fr_0.8fr]">
                <ScaleTable scales={value().scales} />
                <PortfolioDiagnostics report={value()} />
              </section>

              <MonthlyTable rows={value().monthly} selected={scenario()} />

              <section class="grid gap-4 xl:grid-cols-[0.85fr_1.15fr]">
                <FrictionPanel report={value()} />
                <ConstituentTable rows={value().universe.mostFrequentlySelectedMarkets} />
              </section>

              <MinuteTable points={series()?.points ?? []} />
            </>
          )}
        </Show>
      </div>
    </main>
  );
}

function BuildStatus(props: { overview?: Overview; now: number }) {
  const progress = () => props.overview?.progress;
  const ratio = () =>
    progress()?.ratio ??
    (props.overview?.status === "complete" ||
    props.overview?.status === "processing"
      ? 1
      : 0);
  return (
    <section class="panel flex flex-col gap-3">
      <div class="flex flex-wrap items-center justify-between gap-3">
        <div>
          <div class="muted-label">Artifact state</div>
          <div class="mt-1 text-lg font-semibold">
            {statusLabel(props.overview?.status ?? "missing")}
          </div>
        </div>
        <div class="text-right text-sm text-ink-300">
          <Show when={progress()}>
            {(value) => (
              <>
                <div>
                  {integer(value().completedEvents)} / {integer(value().totalEvents)}{" "}
                  {value().scale} evaluations
                </div>
                <div>Checkpointed {ago(value().updatedAt, props.now)}</div>
              </>
            )}
          </Show>
          <Show when={props.overview?.status === "complete"}>
            <div>Minute candles and report ready</div>
          </Show>
        </div>
      </div>
      <div class="h-2 overflow-hidden rounded-full bg-ink-800">
        <div
          class="h-full rounded-full bg-accent transition-[width] duration-500"
          style={{ width: `${Math.max(0, Math.min(100, ratio() * 100))}%` }}
        />
      </div>
      <div class="flex flex-wrap justify-between gap-2 text-xs text-ink-300">
        <span>{percent(ratio())}</span>
        <span>
          Completed sleeves:{" "}
          {props.overview?.completedScales.map((scale) => scale.scale.id).join(", ") || "none"}
        </span>
      </div>
    </section>
  );
}

function PreResult(props: { overview?: Overview }) {
  return (
    <section class="panel flex min-h-60 items-center justify-center text-center">
      <div class="max-w-2xl">
        <Activity class="mx-auto mb-3 animate-pulse text-accent" size={28} />
        <h2 class="text-lg font-semibold">
          {props.overview?.status === "processing"
            ? "Constructing funding and minute index paths"
            : "Building point-in-time sleeves"}
        </h2>
        <p class="mt-2 text-sm text-ink-300">
          This page polls the durable artifacts. Portfolio comparisons, friction
          attribution, monthly returns, constituents, and minute candles will
          appear automatically when the annual run completes.
        </p>
      </div>
    </section>
  );
}

function ScenarioCard(props: {
  scenario: (typeof scenarios)[number];
  performance: PerformanceSummary;
  selected: boolean;
  onSelect: () => void;
}) {
  return (
    <button
      type="button"
      class={`panel min-w-0 text-left transition ${
        props.selected
          ? "border-accent ring-1 ring-accent/35"
          : "hover:border-ink-600"
      }`}
      onClick={props.onSelect}
    >
      <div class="flex items-center justify-between gap-2">
        <div class="muted-label">{props.scenario.label}</div>
        <span
          class="h-2.5 w-2.5 rounded-full"
          style={{ background: props.scenario.color }}
        />
      </div>
      <div class="mt-2 flex items-baseline justify-between gap-3">
        <span class="metric-value">{indexLevel(props.performance.finalLevel)}</span>
        <span
          class={`text-sm font-semibold ${
            props.performance.totalReturn >= 0 ? "text-gain" : "text-loss"
          }`}
        >
          {signedPercent(props.performance.totalReturn)}
        </span>
      </div>
      <p class="mt-2 text-xs leading-5 text-ink-300">
        {props.scenario.description}
      </p>
    </button>
  );
}

function IndexComparisonChart(props: {
  points: IndexPoint[];
  selected: ScenarioKey;
  loading: boolean;
  onSelect: (scenario: ScenarioKey) => void;
}) {
  const width = 1_200;
  const height = 390;
  const margin = { top: 22, right: 24, bottom: 38, left: 70 };
  const usable = createMemo(() =>
    scenarios.map((scenario) => ({
      ...scenario,
      values: props.points
        .map((point) => ({ x: point.time, y: point[scenario.key] }))
        .filter((point) => point.y > 0 && Number.isFinite(point.y)),
    })),
  );
  const bounds = createMemo(() => {
    const values = usable().flatMap((item) => item.values.map((point) => point.y));
    if (values.length < 2 || props.points.length < 2) return undefined;
    const logs = values.map(Math.log);
    let low = Math.min(...logs);
    let high = Math.max(...logs);
    const padding = Math.max((high - low) * 0.08, 0.02);
    low -= padding;
    high += padding;
    return {
      xMin: props.points[0]!.time,
      xMax: props.points.at(-1)!.time,
      low,
      high,
    };
  });
  const x = (value: number) => {
    const box = bounds()!;
    return margin.left +
      ((value - box.xMin) / Math.max(1, box.xMax - box.xMin)) *
        (width - margin.left - margin.right);
  };
  const y = (value: number) => {
    const box = bounds()!;
    return margin.top +
      ((box.high - Math.log(Math.max(value, 1e-12))) / (box.high - box.low)) *
        (height - margin.top - margin.bottom);
  };
  const ticks = createMemo(() => {
    const box = bounds();
    if (!box) return [];
    return Array.from({ length: 5 }, (_, index) => {
      const ratio = index / 4;
      const log = box.high - ratio * (box.high - box.low);
      return {
        value: Math.exp(log),
        y:
          margin.top +
          ratio * (height - margin.top - margin.bottom),
      };
    });
  });
  return (
    <Show
      when={bounds()}
      fallback={
        <div class="mt-3 flex h-80 items-center justify-center rounded-2 bg-ink-950/60 text-sm text-ink-300">
          {props.loading ? "Loading minute index data…" : "Minute index data is not ready."}
        </div>
      }
    >
      <svg
        class="mt-3 block h-auto w-full rounded-2 bg-ink-950/60"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Portfolio index comparison"
      >
        <For each={ticks()}>
          {(tick) => (
            <g>
              <line
                class="training-chart-grid"
                x1={margin.left}
                x2={width - margin.right}
                y1={tick.y}
                y2={tick.y}
              />
              <text
                class="training-chart-axis"
                x={margin.left - 10}
                y={tick.y + 4}
                text-anchor="end"
              >
                {compactNumber(tick.value)}
              </text>
            </g>
          )}
        </For>
        <line
          class="training-chart-axis-line"
          x1={margin.left}
          x2={margin.left}
          y1={margin.top}
          y2={height - margin.bottom}
        />
        <line
          class="training-chart-axis-line"
          x1={margin.left}
          x2={width - margin.right}
          y1={height - margin.bottom}
          y2={height - margin.bottom}
        />
        <For each={usable()}>
          {(item) => (
            <polyline
              points={item.values.map((point) => `${x(point.x)},${y(point.y)}`).join(" ")}
              fill="none"
              stroke={item.color}
              stroke-width={item.key === props.selected ? 3 : 1.5}
              stroke-opacity={item.key === props.selected ? 1 : 0.55}
              stroke-linecap="round"
              stroke-linejoin="round"
              vector-effect="non-scaling-stroke"
            />
          )}
        </For>
        <Show when={bounds()}>
          {(box) => (
            <>
              <text
                class="training-chart-axis"
                x={margin.left}
                y={height - 13}
                text-anchor="start"
              >
                {shortDate(box().xMin)}
              </text>
              <text
                class="training-chart-axis"
                x={width - margin.right}
                y={height - 13}
                text-anchor="end"
              >
                {shortDate(box().xMax)}
              </text>
              <text
                class="training-chart-axis"
                x={width / 2}
                y={height - 13}
                text-anchor="middle"
              >
                UTC · log index scale
              </text>
            </>
          )}
        </Show>
      </svg>
      <div class="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs">
        <For each={scenarios}>
          {(item) => (
            <button
              type="button"
              class={`inline-flex items-center gap-1.5 ${
                item.key === props.selected ? "text-ink-100" : "text-ink-300"
              }`}
              onClick={() => props.onSelect(item.key)}
            >
              <span
                class="h-2 w-2 rounded-full"
                style={{ background: item.color }}
              />
              {item.label}
            </button>
          )}
        </For>
      </div>
    </Show>
  );
}

function ScaleTable(props: { scales: ScaleReport[] }) {
  return (
    <section class="panel min-w-0 overflow-hidden">
      <div class="muted-label">Independent sleeve construction</div>
      <h2 class="mt-1 text-lg font-semibold">Scale diagnostics</h2>
      <div class="mt-3 overflow-x-auto">
        <table class="w-full">
          <thead>
            <tr>
              <th class="table-head">Scale</th>
              <th class="table-head text-right">Evaluations</th>
              <th class="table-head text-right">Eligible</th>
              <th class="table-head text-right">Basis</th>
              <th class="table-head text-right">Coverage</th>
              <th class="table-head text-right">Exposure</th>
            </tr>
          </thead>
          <tbody>
            <For each={props.scales}>
              {(scale) => (
                <tr>
                  <td class="td-cell">
                    <div class="font-semibold">{scale.id}</div>
                    <div class="text-xs text-ink-300">{scale.label}</div>
                  </td>
                  <td class="td-cell text-right">{integer(scale.events)}</td>
                  <td class="td-cell text-right">{decimal(scale.meanEligibleAssets, 1)}</td>
                  <td class="td-cell text-right">{decimal(scale.meanBasisSize, 1)}</td>
                  <td class="td-cell text-right">{percent(scale.coverageTargetReachedRatio)}</td>
                  <td class="td-cell text-right">{percent(scale.meanSleeveExposure)}</td>
                </tr>
              )}
            </For>
          </tbody>
        </table>
      </div>
    </section>
  );
}

function PortfolioDiagnostics(props: { report: PortfolioIndexReport }) {
  const portfolio = () => props.report.portfolio;
  const universe = () => props.report.universe;
  return (
    <section class="panel">
      <div class="muted-label">Investability</div>
      <h2 class="mt-1 text-lg font-semibold">Portfolio diagnostics</h2>
      <div class="mt-3 grid grid-cols-2 gap-3">
        <MetricCard
          compact
          label="Mean exposure"
          value={percent(portfolio().meanExposure)}
          detail={`${percent(portfolio().minimumExposure)} minimum`}
        />
        <MetricCard
          compact
          label="Active constituents"
          value={decimal(portfolio().meanActiveConstituents, 1)}
          detail="Mean across minutes"
        />
        <MetricCard
          compact
          label="Turnover / minute"
          value={percent(portfolio().meanGrossTradedNotionalPerMinute)}
          detail={`${decimal(portfolio().summedGrossTurnover, 1)}× summed`}
        />
        <MetricCard
          compact
          label="Missing return weight"
          value={percent(portfolio().meanMissingNextReturnWeight)}
          detail={`${percent(portfolio().maximumMissingNextReturnWeight)} maximum`}
        />
        <MetricCard
          compact
          label="Product rows"
          value={integer(universe().catalog.allProductRows)}
          detail={`${integer(universe().catalog.currentOptionListings)} options`}
        />
        <MetricCard
          compact
          label="Economic universe"
          value={integer(universe().economicAssets)}
          detail={`${integer(universe().continuousMarkets)} continuous markets`}
        />
      </div>
    </section>
  );
}

function MonthlyTable(props: { rows: MonthlyPath[]; selected: ScenarioKey }) {
  return (
    <section class="panel min-w-0 overflow-hidden">
      <div class="flex flex-wrap items-end justify-between gap-2">
        <div>
          <div class="muted-label">Compounded path</div>
          <h2 class="mt-1 text-lg font-semibold">Monthly portfolio returns</h2>
        </div>
        <div class="text-xs text-ink-300">
          Selected: {scenarioLabel(props.selected)}
        </div>
      </div>
      <div class="mt-3 overflow-x-auto">
        <table class="w-full">
          <thead>
            <tr>
              <th class="table-head">Month</th>
              <th class="table-head text-right">Start</th>
              <th class="table-head text-right">End</th>
              <For each={scenarios}>
                {(item) => (
                  <th class="table-head text-right">{item.shortLabel}</th>
                )}
              </For>
            </tr>
          </thead>
          <tbody>
            <For each={props.rows}>
              {(row) => {
                const selected = () => row[props.selected];
                return (
                  <tr>
                    <td class="td-cell font-semibold">{row.month}</td>
                    <td class="td-cell text-right">{indexLevel(selected().startLevel)}</td>
                    <td class="td-cell text-right">{indexLevel(selected().endLevel)}</td>
                    <For each={scenarios}>
                      {(item) => (
                        <td
                          class={`td-cell text-right ${
                            row[item.key].return >= 0 ? "text-gain" : "text-loss"
                          }`}
                        >
                          {signedPercent(row[item.key].return)}
                        </td>
                      )}
                    </For>
                  </tr>
                );
              }}
            </For>
          </tbody>
        </table>
      </div>
    </section>
  );
}

function FrictionPanel(props: { report: PortfolioIndexReport }) {
  const friction = () => props.report.methodology.friction;
  const selection = () => props.report.methodology.selection;
  return (
    <section class="panel">
      <div class="muted-label">Methodology</div>
      <h2 class="mt-1 text-lg font-semibold">Friction and constraints</h2>
      <div class="mt-3 space-y-3 text-sm">
        <Definition
          label="Spot baseline"
          value={`${decimal(friction().spotFeeBps, 1)} bp fee + ${decimal(
            friction().baselineAdditionalExecutionBps,
            1,
          )} bp execution`}
        />
        <Definition
          label="Futures baseline"
          value={`${decimal(friction().futuresFeeBps, 1)} bp fee + ${decimal(
            friction().baselineAdditionalExecutionBps,
            1,
          )} bp execution`}
        />
        <Definition
          label="Conservative execution"
          value={`${decimal(friction().conservativeAdditionalExecutionBps, 1)} bp per dollar bought or sold`}
        />
        <Definition label="Charged on" value={friction().chargedOn} />
        <Definition
          label="Portfolio"
          value="Long-only exposure 0–1; no borrowing, shorting, leverage, interest, or maintenance cost"
        />
        <Definition label="Perpetual funding" value={props.report.methodology.perpetualFunding} />
        <Definition
          label="Basis target"
          value={`${integer(selection().sampleCount)} returns; ${percent(
            selection().targetMedianRSquared,
          )} median R² and ${percent(selection().targetP10RSquared)} p10 R²`}
        />
        <Definition
          label="Weight cap"
          value={`${percent(selection().maximumConstituentWeight)} per constituent`}
        />
        <p class="rounded-xl border border-warn/30 bg-warn/8 p-3 text-xs leading-5 text-ink-300">
          {friction().capacityImpact}
        </p>
      </div>
    </section>
  );
}

function ConstituentTable(props: { rows: SelectedMarket[] }) {
  return (
    <section class="panel min-w-0 overflow-hidden">
      <div class="muted-label">Selection recurrence</div>
      <h2 class="mt-1 text-lg font-semibold">Most frequently selected markets</h2>
      <div class="mt-3 max-h-[34rem] overflow-auto">
        <table class="w-full">
          <thead class="sticky top-0 bg-ink-900">
            <tr>
              <th class="table-head">Asset</th>
              <th class="table-head">Market</th>
              <th class="table-head">Venue</th>
              <th class="table-head text-right">Events</th>
              <th class="table-head text-right">Share</th>
            </tr>
          </thead>
          <tbody>
            <For each={props.rows}>
              {(row) => (
                <tr>
                  <td class="td-cell font-semibold">{row.asset}</td>
                  <td class="td-cell">{row.market}</td>
                  <td class="td-cell text-ink-300">{venueLabel(row.venue)}</td>
                  <td class="td-cell text-right">{integer(row.selectionEvents)}</td>
                  <td class="td-cell text-right">{percent(row.selectionEventShare)}</td>
                </tr>
              )}
            </For>
          </tbody>
        </table>
      </div>
    </section>
  );
}

function MinuteTable(props: { points: IndexPoint[] }) {
  const recent = createMemo(() => props.points.slice(-12).reverse());
  return (
    <section class="panel min-w-0 overflow-hidden">
      <div class="muted-label">Downsampled candle inspection</div>
      <h2 class="mt-1 text-lg font-semibold">Recent plotted index data</h2>
      <div class="mt-3 overflow-x-auto">
        <table class="w-full">
          <thead>
            <tr>
              <th class="table-head">UTC time</th>
              <th class="table-head text-right">Open</th>
              <th class="table-head text-right">High</th>
              <th class="table-head text-right">Low</th>
              <th class="table-head text-right">Baseline</th>
              <th class="table-head text-right">Gross</th>
              <th class="table-head text-right">Fees</th>
              <th class="table-head text-right">Conservative</th>
              <th class="table-head text-right">Exposure</th>
              <th class="table-head text-right">Assets</th>
              <th class="table-head text-right">Turnover</th>
            </tr>
          </thead>
          <tbody>
            <For each={recent()}>
              {(point) => (
                <tr>
                  <td class="td-cell">{dateTime(point.time)}</td>
                  <td class="td-cell text-right">{indexLevel(point.open)}</td>
                  <td class="td-cell text-right">{indexLevel(point.high)}</td>
                  <td class="td-cell text-right">{indexLevel(point.low)}</td>
                  <td class="td-cell text-right">{indexLevel(point.baseline)}</td>
                  <td class="td-cell text-right">{indexLevel(point.gross)}</td>
                  <td class="td-cell text-right">{indexLevel(point.feeOnly)}</td>
                  <td class="td-cell text-right">{indexLevel(point.conservative)}</td>
                  <td class="td-cell text-right">{percent(point.exposure)}</td>
                  <td class="td-cell text-right">{integer(point.activeConstituents)}</td>
                  <td class="td-cell text-right">{percent(point.turnover)}</td>
                </tr>
              )}
            </For>
          </tbody>
        </table>
      </div>
    </section>
  );
}

function MetricCard(props: {
  label: string;
  value: string;
  detail: string;
  tone?: "gain" | "loss" | "warn";
  compact?: boolean;
}) {
  const tone = () =>
    props.tone === "gain"
      ? "text-gain"
      : props.tone === "loss"
        ? "text-loss"
        : props.tone === "warn"
          ? "text-warn"
          : "text-ink-100";
  return (
    <div class={props.compact ? "panel-tight min-w-0" : "panel min-w-0"}>
      <div class="muted-label truncate">{props.label}</div>
      <div class={`mt-1 truncate text-xl font-semibold tabular-nums ${tone()}`}>
        {props.value}
      </div>
      <div class="mt-1 truncate text-xs text-ink-300">{props.detail}</div>
    </div>
  );
}

function Definition(props: { label: string; value: string }) {
  return (
    <div class="grid gap-1 border-b border-line pb-2 sm:grid-cols-[10rem_1fr]">
      <span class="text-ink-300">{props.label}</span>
      <span>{props.value}</span>
    </div>
  );
}

function StatusPill(props: { status: Overview["status"] }) {
  const active = () => props.status === "building" || props.status === "processing";
  return (
    <span
      class={`inline-flex items-center gap-2 rounded-full border px-2.5 py-1 text-xs font-semibold ${
        props.status === "complete"
          ? "border-gain/50 bg-gain/10 text-gain"
          : active()
            ? "border-accent/50 bg-accent/10 text-accent"
            : "border-line bg-ink-800 text-ink-300"
      }`}
    >
      {props.status === "complete" ? (
        <BarChart3 size={13} />
      ) : (
        <Activity size={13} class={active() ? "animate-pulse" : ""} />
      )}
      {statusLabel(props.status)}
    </span>
  );
}

function rangeBounds(report: PortfolioIndexReport, range: RangeKey) {
  const start = Date.parse(`${report.window.start}T00:00:00Z`);
  const to = Date.parse(`${report.window.end}T00:00:00Z`) + DAY_MS;
  const duration =
    range === "1m"
      ? 31 * DAY_MS
      : range === "3m"
        ? 92 * DAY_MS
        : range === "6m"
          ? 183 * DAY_MS
          : 366 * DAY_MS;
  return { from: Math.max(start, to - duration), to };
}

function statusLabel(status: Overview["status"]): string {
  if (status === "building") return "Building point-in-time sleeves";
  if (status === "processing") return "Computing index paths";
  if (status === "complete") return "Complete";
  return "No index artifact";
}

function scenarioLabel(key: ScenarioKey): string {
  return scenarios.find((item) => item.key === key)?.label ?? key;
}

function venueLabel(value: string): string {
  if (value === "usdm-futures") return "USD-M";
  if (value === "coinm-futures") return "COIN-M";
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function percent(value: number | undefined): string {
  return value === undefined || !Number.isFinite(value)
    ? "—"
    : `${(value * 100).toFixed(Math.abs(value) < 0.001 ? 2 : 1)}%`;
}

function signedPercent(value: number | undefined): string {
  return value === undefined || !Number.isFinite(value)
    ? "—"
    : `${value >= 0 ? "+" : ""}${(value * 100).toFixed(1)}%`;
}

function indexLevel(value: number | undefined): string {
  if (value === undefined || !Number.isFinite(value)) return "—";
  if (Math.abs(value) >= 100) return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
  if (Math.abs(value) >= 1) return value.toFixed(3);
  return value.toPrecision(4);
}

function money(value: number | undefined): string {
  if (value === undefined || !Number.isFinite(value)) return "—";
  return `${value < 0 ? "−" : ""}${Math.abs(value).toLocaleString(undefined, {
    maximumFractionDigits: 2,
  })}`;
}

function decimal(value: number | undefined, digits: number): string {
  return value === undefined || !Number.isFinite(value) ? "—" : value.toFixed(digits);
}

function integer(value: number | undefined): string {
  return value === undefined || !Number.isFinite(value)
    ? "—"
    : Math.round(value).toLocaleString();
}

function compactNumber(value: number): string {
  const absolute = Math.abs(value);
  if (absolute >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}m`;
  if (absolute >= 1_000) return `${(value / 1_000).toFixed(1)}k`;
  if (absolute >= 1) return value.toFixed(2);
  return value.toPrecision(2);
}

function dateTime(value: number): string {
  return new Date(value).toISOString().replace("T", " ").slice(0, 16);
}

function shortDate(value: number): string {
  return new Date(value).toISOString().slice(0, 10);
}

function ago(value: string, now: number): string {
  const timestamp = Date.parse(value);
  if (!Number.isFinite(timestamp)) return "unknown";
  const seconds = Math.max(0, Math.floor((now - timestamp) / 1_000));
  return seconds < 60 ? `${seconds}s ago` : `${Math.floor(seconds / 60)}m ago`;
}
