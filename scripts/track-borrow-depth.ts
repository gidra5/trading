import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  SimulatedTradingBot,
  createInitialBotState,
  createStrategyConfig,
  summarizeClosedPositions,
  type Candle,
} from "../packages/bot-algo/src/index.js";

const DAY_MS = 24 * 60 * 60 * 1000;
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

interface Args {
  days: number;
  symbol: string;
  interval: string;
}

interface DepthTracker {
  maxLongDepth: number;
  maxShortDepth: number;
}

interface LegacyPositionLotCacheView {
  longs: Map<string, { borrowDepthRemaining: number }>;
  shorts: Map<string, { borrowDepthRemaining: number }>;
}

type TrackableBot = SimulatedTradingBot & {
  updateLegacyPositionLotCache: () => void;
  legacyPositionLotCache?: LegacyPositionLotCacheView;
};

const args = parseArgs(process.argv.slice(2));
const startedAt = Date.now();
const candles = loadHistoricalCandles(args);
if (candles.length === 0) {
  throw new Error(`No candles found for ${args.symbol.toUpperCase()} ${args.interval}.`);
}

const config = createStrategyConfig({
  symbol: args.symbol.toUpperCase(),
});
const bot = new SimulatedTradingBot(createInitialBotState(config));
const depth: DepthTracker = { maxLongDepth: 0, maxShortDepth: 0 };
const sampleEvery = Math.max(1, Math.ceil(candles.length / 800));

for (let index = 0; index < candles.length; index += 1) {
  const candle = candles[index];
  const liquidatedBefore = bot.liquidatedPositionCount();
  replayCandle(bot, candle, depth);
  const liquidated = bot.liquidatedPositionCount() > liquidatedBefore;
  if (index % sampleEvery === 0 || index === candles.length - 1 || liquidated) {
    bot.markToMarket();
  }
  if (liquidated) {
    break;
  }
}

bot.markToMarket();
observeDepth(bot, depth);
const state = bot.view();
const closed = summarizeClosedPositions(state);
const elapsedMs = Date.now() - startedAt;

console.log(
  [
    `Borrow depth tracking: last-${args.days}-days`,
    `${args.symbol.toUpperCase()} ${args.interval}`,
    `${candles.length.toLocaleString()} candles`,
    `${formatDate(candles[0].openTime)} to ${formatDate(candles[candles.length - 1].closeTime)}`,
    `${state.config.maxLeverage}x max leverage`,
    `${state.config.shortMarginModel} short margin`,
    `borrow depth L${state.config.longBorrowDepth}/S${state.config.shortBorrowDepth}`,
    `borrow lock ${state.config.lockBorrowedLenderCollateral ? "on" : "off"}`,
    `target cap ${formatMoney(state.config.maxPositionQuote)}`,
    `${state.config.cooldownMs / 1000}s cooldown`,
    `elapsed ${elapsedMs.toLocaleString()}ms`,
  ].join(", "),
);
console.log("");
console.log(
  markdownTable(
    [
      "Return",
      "Net PnL",
      "Max DD",
      "Trades",
      "Win Rate",
      "Prof Pos",
      "Liq Pos",
      "Max Long Depth",
      "Max Short Depth",
    ],
    [
      [
        `${formatNumber(state.metrics.returnPct, 2)}%`,
        formatMoney(state.metrics.netPnl),
        `${formatNumber(state.metrics.maxDrawdownPct, 2)}%`,
        state.metrics.tradeCount.toLocaleString(),
        `${formatNumber(state.metrics.winRate, 1)}%`,
        `${closed.profitableClosedPositionCount.toLocaleString()}/${closed.closedPositionCount.toLocaleString()} (${formatNumber(closed.profitableClosedPositionRate, 1)}%)`,
        closed.liquidatedPositionCount.toLocaleString(),
        depth.maxLongDepth.toLocaleString(),
        depth.maxShortDepth.toLocaleString(),
      ],
    ],
  ),
);

function replayCandle(
  bot: SimulatedTradingBot,
  candle: Candle,
  depth: DepthTracker,
): void {
  const duration = Math.max(1, candle.closeTime - candle.openTime);
  const highTime = candle.openTime + duration * 0.33;
  const lowTime = candle.openTime + duration * 0.66;

  replayTick(bot, candle.openTime, candle.open, depth);
  replayTick(bot, highTime, candle.high, depth);
  replayTick(bot, lowTime, candle.low, depth);
  replayTick(bot, candle.closeTime, candle.close, depth);
}

function replayTick(
  bot: SimulatedTradingBot,
  eventTime: number,
  price: number,
  depth: DepthTracker,
): void {
  bot.onReplayPriceTick(eventTime, price);
  observeDepth(bot, depth);
}

function observeDepth(bot: SimulatedTradingBot, depth: DepthTracker): void {
  const trackable = bot as TrackableBot;
  trackable.updateLegacyPositionLotCache();
  const cache = trackable.legacyPositionLotCache;
  if (!cache) {
    return;
  }

  const config = bot.view().config;
  for (const lot of cache.longs.values()) {
    depth.maxLongDepth = Math.max(
      depth.maxLongDepth,
      Math.max(0, config.longBorrowDepth - lot.borrowDepthRemaining),
    );
  }
  for (const lot of cache.shorts.values()) {
    depth.maxShortDepth = Math.max(
      depth.maxShortDepth,
      Math.max(0, config.shortBorrowDepth - lot.borrowDepthRemaining),
    );
  }
}

function loadHistoricalCandles(args: Args): Candle[] {
  const dir = path.join(
    repoRoot,
    "data",
    "historical",
    "spot-btcusdt",
    args.symbol.toLowerCase(),
    args.interval,
  );
  const files = fs
    .readdirSync(dir, { withFileTypes: true })
    .filter((entry) => entry.isFile() && entry.name.endsWith(".jsonl"))
    .map((entry) => entry.name)
    .sort()
    .slice(-args.days);

  const candles: Candle[] = [];
  for (const file of files) {
    const content = fs.readFileSync(path.join(dir, file), "utf8");
    for (const line of content.split("\n")) {
      const trimmed = line.trim();
      if (trimmed) {
        candles.push(JSON.parse(trimmed) as Candle);
      }
    }
  }

  return candles.sort((left, right) => left.openTime - right.openTime);
}

function parseArgs(argv: string[]): Args {
  const values = new Map<string, string>();
  for (let index = 0; index < argv.length; index += 1) {
    const key = argv[index];
    if (!key.startsWith("--")) {
      continue;
    }
    const next = argv[index + 1];
    if (next && !next.startsWith("--")) {
      values.set(key.slice(2), next);
      index += 1;
    } else {
      values.set(key.slice(2), "true");
    }
  }

  return {
    days: parsePositiveInt(values.get("days"), 30),
    symbol: values.get("symbol") ?? "BTCUSDT",
    interval: values.get("interval") ?? "1m",
  };
}

function parsePositiveInt(value: string | undefined, fallback: number): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? Math.round(parsed) : fallback;
}

function markdownTable(header: string[], rows: string[][]): string {
  const separator = header.map(() => "---");
  return [header, separator, ...rows]
    .map((row) => `| ${row.join(" | ")} |`)
    .join("\n");
}

function formatDate(value: number): string {
  return new Date(value).toISOString().slice(0, 10);
}

function formatMoney(value: number): string {
  return `$${formatNumber(value, 2)}`;
}

function formatNumber(value: number, digits: number): string {
  return value.toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}
