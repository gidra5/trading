import fs from "node:fs";
import path from "node:path";
import { readCandleShardReferenceSync } from "@trading/storage";
import type { BacktestStrategy, BacktestSummary, Candle } from "@trading/bot-algo";
import { appConfig } from "../apps/server/src/config.js";
import { runBotBacktestFromCandles } from "../apps/server/src/bot-backtest.js";
import { historicalWarmupSamples } from "../apps/server/src/historical-backtest.js";
import { KamaInspector } from "../apps/server/src/kama-inspector.js";
import { withStoredAggressorVolume } from "../apps/server/src/trade-flow-backtest.js";

const DAY_MS = 86_400_000;
const INTERVAL_MS = 60_000;
const STRATEGY = suiteStrategy(argument("strategy"));
const HISTORY_ROOT = path.resolve(
  "data/market/immutable/refs/candles/spot-btcusdt/btcusdt/1m",
);
const OUTPUT = path.resolve(
  argument("output")
    ?? `data/benchmarks/${reportName(STRATEGY)}-bot-suite-1m-2026-08-10.json`,
);

interface SuiteWindow {
  id: string;
  startTime: number;
  endTime: number;
}

interface SuiteResult extends SuiteWindow {
  candles: number;
  warmupCandles: number;
  wallDurationMs: number;
  summary: BacktestSummary;
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.stack ?? error.message : String(error));
  process.exitCode = 1;
});

async function main(): Promise<void> {
  const inspector = new KamaInspector(path.resolve("data"));
  const staticWindows = inspector.catalog().windows
    .filter((window) => window.id !== "latest" && !window.id.startsWith("fit-"))
    .map(({ id, startTime, endTime }) => ({ id, startTime, endTime }));
  if (staticWindows.length !== 28) {
    throw new Error(`Expected 28 non-fit inspector windows; found ${staticWindows.length}.`);
  }

  const latestDay = latestContiguousAvailableDay(93);
  if (!latestDay) throw new Error(`No one-minute history found in ${HISTORY_ROOT}.`);
  const latestEndTime = parseDay(latestDay) + DAY_MS;
  const windows: SuiteWindow[] = [
    ...staticWindows,
    ...(STRATEGY === "volume-imbalance" ? [] : [{
      id: "latest-3m",
      startTime: subtractUtcMonths(latestEndTime, 3),
      endTime: latestEndTime,
    }]),
  ];
  const only = new Set(
    (argument("only") ?? "")
      .split(",")
      .map((value) => value.trim())
      .filter(Boolean),
  );
  const selected = only.size > 0
    ? windows.filter((window) => only.has(window.id))
    : windows;
  if (selected.length === 0) throw new Error("No peak/valley suite windows selected.");

  const warmupMs = historicalWarmupSamples(appConfig.strategy, INTERVAL_MS) * INTERVAL_MS;
  const report = {
    strategy: STRATEGY,
    interval: "1m",
    generatedAt: new Date().toISOString(),
    historyRoot: HISTORY_ROOT,
    latestAvailableDay: latestDay,
    config: appConfig.strategy,
    methodology: STRATEGY === "macd"
      ? "12/26/9 crossover events on UTC-aligned one-hour bars; entries require an RSI(14) 30/70 extreme within the preceding six bars; crossover exits are unfiltered"
      : STRATEGY === "volume-imbalance"
        ? "peak/valley entries gated by same-direction +/-15% Binance aggressor-volume imbalance from the current one-minute candle; exits are unfiltered"
        : "peak/valley",
    skipped: STRATEGY === "volume-imbalance"
      ? [{ id: "latest-3m", reason: "stored aggressor-volume coverage ends before this interval" }]
      : [],
    results: [] as SuiteResult[],
  };

  for (const [index, window] of selected.entries()) {
    console.log(
      `START ${window.id} (${index + 1}/${selected.length}) `
        + `${isoDay(window.startTime)}..${isoDay(window.endTime - 1)}`,
    );
    const loaded = await loadWindow(window, warmupMs);
    const startedAt = Date.now();
    const result = await runBotBacktestFromCandles(loaded.candles, {
      config: appConfig.strategy,
      strategy: STRATEGY,
      warmup: loaded.warmup,
      maxEquityPoints: 800,
      maxChartCandles: 2_000,
      summaryOnly: true,
    });
    const suiteResult: SuiteResult = {
      ...window,
      candles: loaded.candles.length,
      warmupCandles: loaded.warmup.length,
      wallDurationMs: Date.now() - startedAt,
      summary: result.summary,
    };
    report.results.push(suiteResult);
    report.generatedAt = new Date().toISOString();
    writeReport(report);
    console.log(`RESULT ${JSON.stringify(compact(suiteResult))}`);
  }

  console.log(`AGGREGATE ${JSON.stringify(aggregate(report.results))}`);
  console.log(`REPORT ${OUTPUT}`);
}

async function loadWindow(window: SuiteWindow, warmupMs: number): Promise<{
  warmup: Candle[];
  candles: Candle[];
}> {
  const loadStart = window.startTime - warmupMs;
  const warmup: Candle[] = [];
  const candles: Candle[] = [];
  for (let day = utcDay(loadStart); day < window.endTime; day += DAY_MS) {
    const date = isoDay(day);
    const daily = readReferenceDay(date);
    if (!daily) throw new Error(`Missing candle history for ${date}.`);
    for (const candle of daily) {
      if (candle.openTime < loadStart || candle.openTime >= window.endTime) continue;
      if (candle.openTime < window.startTime) warmup.push(candle);
      else candles.push(candle);
    }
  }
  if (candles.length === 0) throw new Error(`No measured candles for ${window.id}.`);
  if (STRATEGY !== "volume-imbalance") return { warmup, candles };
  return {
    warmup: await withStoredAggressorVolume(warmup, {
      dataDir: path.resolve("data"),
      venue: "spot",
      symbol: "BTCUSDT",
      requiredFrom: window.startTime,
    }),
    candles: await withStoredAggressorVolume(candles, {
      dataDir: path.resolve("data"),
      venue: "spot",
      symbol: "BTCUSDT",
      requiredFrom: window.startTime,
    }),
  };
}

function readReferenceDay(date: string): Candle[] | undefined {
  const file = path.join(HISTORY_ROOT, `${date}.json`);
  return fs.existsSync(file) ? readCandleShardReferenceSync(file) : undefined;
}

function availableDays(): string[] {
  return fs.readdirSync(HISTORY_ROOT)
    .map((file) => /^(\d{4}-\d{2}-\d{2})\.json$/.exec(file)?.[1])
    .filter((date): date is string => Boolean(date))
    .sort();
}

function latestContiguousAvailableDay(minimumDays: number): string | undefined {
  const dates = availableDays();
  const available = new Set(dates);
  for (let candidate = dates.length - 1; candidate >= 0; candidate -= 1) {
    const end = parseDay(dates[candidate]!);
    let complete = true;
    for (let offset = 1; offset < minimumDays; offset += 1) {
      if (!available.has(isoDay(end - offset * DAY_MS))) {
        complete = false;
        break;
      }
    }
    if (complete) return dates[candidate];
  }
  return undefined;
}

function compact(result: SuiteResult): Record<string, unknown> {
  return {
    id: result.id,
    returnPct: result.summary.returnPct,
    maxInitialBalanceDrawdownPct: result.summary.maxInitialBalanceDrawdownPct,
    maxDrawdownPct: result.summary.maxDrawdownPct,
    tradeCount: result.summary.tradeCount,
    winRate: result.summary.winRate,
    liquidations: result.summary.liquidatedPositionCount,
    wallDurationMs: result.wallDurationMs,
  };
}

function aggregate(results: readonly SuiteResult[]): Record<string, unknown> {
  const returns = results.map((result) => result.summary.returnPct);
  const sorted = [...returns].sort((a, b) => a - b);
  const mean = returns.reduce((sum, value) => sum + value, 0) / returns.length;
  const median = sorted.length % 2 === 1
    ? sorted[Math.floor(sorted.length / 2)]!
    : (sorted[sorted.length / 2 - 1]! + sorted[sorted.length / 2]!) / 2;
  const geometric = returns.some((value) => value <= -100)
    ? -100
    : (Math.exp(
        returns.reduce((sum, value) => sum + Math.log1p(value / 100), 0)
          / returns.length,
      ) - 1) * 100;
  const worst = results.reduce((current, result) =>
    result.summary.returnPct < current.summary.returnPct ? result : current
  );
  const best = results.reduce((current, result) =>
    result.summary.returnPct > current.summary.returnPct ? result : current
  );
  return {
    windows: results.length,
    profitable: returns.filter((value) => value > 0).length,
    liquidated: results.filter((result) => result.summary.liquidatedPositionCount > 0).length,
    meanReturnPct: mean,
    medianReturnPct: median,
    geometricMeanReturnPct: geometric,
    meanInitialBalanceDrawdownPct:
      results.reduce(
        (sum, result) => sum + result.summary.maxInitialBalanceDrawdownPct,
        0,
      ) / results.length,
    meanPeakDrawdownPct:
      results.reduce((sum, result) => sum + result.summary.maxDrawdownPct, 0)
        / results.length,
    worst: { id: worst.id, returnPct: worst.summary.returnPct },
    best: { id: best.id, returnPct: best.summary.returnPct },
  };
}

function writeReport(value: unknown): void {
  fs.mkdirSync(path.dirname(OUTPUT), { recursive: true });
  fs.writeFileSync(OUTPUT, `${JSON.stringify(value, null, 2)}\n`);
}

function argument(name: string): string | undefined {
  const prefix = `--${name}=`;
  return process.argv.find((value) => value.startsWith(prefix))?.slice(prefix.length);
}

function suiteStrategy(value: string | undefined): BacktestStrategy {
  if (value === undefined || value === "peak-valley") return "peak-valley";
  if (value === "macd" || value === "volume-imbalance") return value;
  throw new Error(`Unsupported technical suite strategy: ${value}.`);
}

function reportName(strategy: BacktestStrategy): string {
  if (strategy === "macd") return "macd-rsi";
  if (strategy === "volume-imbalance") return "peak-valley-aggressor-filter";
  return strategy;
}

function parseDay(value: string): number {
  return Date.parse(`${value}T00:00:00.000Z`);
}

function isoDay(timestamp: number): string {
  return new Date(timestamp).toISOString().slice(0, 10);
}

function utcDay(timestamp: number): number {
  const date = new Date(timestamp);
  return Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate());
}

function subtractUtcMonths(timestamp: number, months: number): number {
  const date = new Date(timestamp);
  return Date.UTC(
    date.getUTCFullYear(),
    date.getUTCMonth() - months,
    date.getUTCDate(),
  );
}
