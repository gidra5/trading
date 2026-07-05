import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import * as esbuild from "esbuild";
import {
  type Candle,
} from "../packages/bot-algo/src/index.js";

interface CaseRow {
  label: string;
  group: string;
  intervalLabel: string;
  startDate: string;
  endDate: string;
  tableRet: string;
  tableSpan: string;
  tableBias: string;
  tableTurns05: string;
  tableLow: string;
  tableHigh: string;
  note?: string;
}

interface LinePoint {
  time: number;
  value: number;
}

interface SmaConfig {
  label: string;
  periods: number;
  color: string;
}

interface RenderedCase {
  row: CaseRow;
  candles: Candle[];
  smaCandles: Candle[];
  replayCandles: Candle[];
}

interface BrowserCase {
  label: string;
  intervalLabel: string;
  startTime: number;
  endExclusive: number;
  candles: BrowserCandle[];
  smaCandles: BrowserCandle[];
  replayCandles: BrowserCandle[];
}

interface BrowserCandle {
  symbol: string;
  interval: string;
  openTime: number;
  closeTime: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  closed: boolean;
}

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const inputPath = path.join(repoRoot, "tasks.md");
const outputPath = path.join(repoRoot, "docs", "interval-price-movement.html");
const symbol = "BTCUSDT";
const chartInterval = "15m";
const chartIntervalMs = 15 * 60 * 1000;
const strategyInterval = "1m";
const strategyIntervalMs = 60 * 1000;
const smaConfigs: SmaConfig[] = [
  { label: "1h SMA", periods: 4, color: "var(--sma-1h)" },
  { label: "4h SMA", periods: 16, color: "var(--sma-4h)" },
  { label: "12h SMA", periods: 48, color: "var(--sma-12h)" },
];
const smaWarmupCandles = Math.max(...smaConfigs.map((series) => series.periods)) - 1;
const centeredSmaLookaheadCandles = Math.ceil(
  Math.max(...smaConfigs.map((series) => series.periods)) / 2,
);
const strategyWarmupMs = 12 * 60 * 60 * 1000;
const strategyWarmupCandles = Math.ceil(strategyWarmupMs / strategyIntervalMs);

main().catch((error: unknown) => {
  console.error(error instanceof Error ? (error.stack ?? error.message) : String(error));
  process.exitCode = 1;
});

async function main(): Promise<void> {
  const markdown = await fs.readFile(inputPath, "utf8");
  const cases = parseCaseRows(markdown);
  if (cases.length === 0) {
    throw new Error("Could not find the interval table in tasks.md.");
  }

  const rendered: RenderedCase[] = [];
  for (const [index, row] of cases.entries()) {
    console.error(
      `Fetching ${index + 1}/${cases.length}: ${row.label} ${row.intervalLabel} ${symbol}`,
    );
    const chartCandles = await fetchCaseCandles(row, {
      interval: chartInterval,
      intervalMs: chartIntervalMs,
      warmupCandles: smaWarmupCandles,
      lookaheadCandles: centeredSmaLookaheadCandles,
    });
    const replayCandles = await fetchCaseCandles(row, {
      interval: strategyInterval,
      intervalMs: strategyIntervalMs,
      warmupCandles: strategyWarmupCandles,
    });
    const { startTime, endExclusive } = caseTimeRange(row);
    const candles = chartCandles.filter(
      (candle) => candle.openTime >= startTime && candle.openTime < endExclusive,
    );
    const intervalReplayCandles = replayCandles.filter(
      (candle) => candle.openTime >= startTime && candle.openTime < endExclusive,
    );
    if (candles.length < 2) {
      throw new Error(`Not enough candles for ${row.label} ${row.intervalLabel}.`);
    }
    if (intervalReplayCandles.length < 2) {
      throw new Error(`Not enough strategy candles for ${row.label} ${row.intervalLabel}.`);
    }
    rendered.push({
      row: withComputedMetrics(row, candles),
      candles,
      smaCandles: chartCandles,
      replayCandles,
    });
  }

  const browserBundle = await createBrowserStrategyBundle();
  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  await fs.writeFile(outputPath, renderHtml(rendered, browserBundle), "utf8");
  console.log(path.relative(repoRoot, outputPath));
}

async function createBrowserStrategyBundle(): Promise<string> {
  const source = `
    import { runBacktestFromCandles } from "./packages/bot-algo/src/index.js";

    export function runEntryOverlay(cases, settings) {
      return cases.map((testCase) => runEntryOverlayCase(testCase, settings));
    }

    export function runEntryOverlayCase(testCase, settings) {
      const legacyValleyPeak = legacyConfigFromSettings(settings);
      const result = runBacktestFromCandles(testCase.replayCandles, {
        config: {
          symbol: "BTCUSDT",
          algorithm: "legacy-valley-peak",
          startingQuote: 10_000,
          maxLeverage: 1,
          shortMarginModel: "futures-margin",
          longBorrowDepth: 999,
          shortBorrowDepth: 999,
          legacyValleyPeak,
        },
        maxReturnedOrders: 0,
        maxReturnedFills: Number.POSITIVE_INFINITY,
        maxEquityPoints: 1,
        maxChartCandles: 1,
      });

      return aggregateEntryFills(result.fills, testCase.startTime, testCase.endExclusive);
    }

    function legacyConfigFromSettings(settings) {
      const mode = settings.sigmaMode === "sigmoid-trend" ? "sigmoid-trend" : "static";
      if (mode === "sigmoid-trend") {
        return {
          sigmaMode: "sigmoid-trend",
          sigmoidSigmaLow: positive(settings.sigmoidSigmaLow, 0.05),
          sigmoidSigmaHigh: positive(settings.sigmoidSigmaHigh, 0.3),
          trendSigmaWindowSec: positive(settings.trendSigmaWindowSec, 43_200),
          trendSigmaSellB1: finite(settings.trendSigmaSellB1, 15),
          trendSigmaBuyB2: finite(settings.trendSigmaBuyB2, 300),
          longSideEnabled: true,
          shortSideEnabled: true,
        };
      }

      return {
        sigmaMode: "static",
        buySigma: positive(settings.buySigma, 0.1),
        sellSigma: positive(settings.sellSigma, 0.1),
        longSideEnabled: true,
        shortSideEnabled: true,
      };
    }

    function aggregateEntryFills(fills, startTime, endExclusive) {
      const entries = new Map();
      for (const fill of fills) {
        if (
          fill.positionEffect !== "open" ||
          fill.filledAt < startTime ||
          fill.filledAt >= endExclusive ||
          fill.quoteQuantity <= 0
        ) {
          continue;
        }

        const bucketTime = Math.floor(fill.filledAt / ${chartIntervalMs}) * ${chartIntervalMs};
        const key = bucketTime + ":" + fill.side;
        const existing = entries.get(key);
        if (existing) {
          existing.quote += fill.quoteQuantity;
        } else {
          entries.set(key, {
            time: bucketTime,
            side: fill.side,
            quote: fill.quoteQuantity,
          });
        }
      }

      return [...entries.values()].sort(
        (left, right) => left.time - right.time || left.side.localeCompare(right.side),
      );
    }

    function positive(value, fallback) {
      const number = Number(value);
      return Number.isFinite(number) && number > 0 ? number : fallback;
    }

    function finite(value, fallback) {
      const number = Number(value);
      return Number.isFinite(number) ? number : fallback;
    }
  `;

  const result = await esbuild.build({
    stdin: {
      contents: source,
      resolveDir: repoRoot,
      sourcefile: "interval-price-movement-browser.ts",
      loader: "ts",
    },
    bundle: true,
    write: false,
    format: "iife",
    globalName: "TradingStrategyBrowser",
    platform: "browser",
    target: "es2022",
    minify: true,
    logLevel: "silent",
  });

  return result.outputFiles[0]?.text ?? "";
}

function parseCaseRows(markdown: string): CaseRow[] {
  return uniqueRows([
    ...parseChoppyWindowRows(markdown),
    ...parseRegimeRows(markdown),
    ...parseSelectedMarkdownRows(markdown),
    ...parseSharpeRows(markdown),
    ...parseStressRows(markdown),
  ]);
}

function parseChoppyWindowRows(markdown: string): CaseRow[] {
  return parseTsvBlock(markdown, "Window\tClose/Open\tLow\tHigh\tSpan\tMovement score").flatMap(
    (columns) => {
      if (columns.length < 6) {
        return [];
      }
      const [startDate, endDate] = columns[0].split("..");
      if (!isIsoDate(startDate) || !isIsoDate(endDate)) {
        return [];
      }
      return [
        {
          label: `sideways churn candidate, ${columns[5]}`,
          group: "7d choppy candidates",
          intervalLabel: columns[0],
          startDate,
          endDate,
          tableRet: columns[1],
          tableSpan: columns[4],
          tableBias: "",
          tableTurns05: "",
          tableLow: columns[2],
          tableHigh: columns[3],
          note: columns[5],
        },
      ];
    },
  );
}

function parseRegimeRows(markdown: string): CaseRow[] {
  return parseTsvBlock(
    markdown,
    "Regime\tWindow\tMarket move\tStatic sigmas\tReturn\tMax DD\tTrades",
  ).flatMap((columns) => {
    if (columns.length < 7) {
      return [];
    }
    const [startDate, endDate] = columns[1].split("..");
    if (!isIsoDate(startDate) || !isIsoDate(endDate)) {
      return [];
    }
    return [
      {
        label: `${columns[0].toLowerCase()} static benchmark`,
        group: "7d static benchmarks",
        intervalLabel: columns[1],
        startDate,
        endDate,
        tableRet: columns[2],
        tableSpan: "",
        tableBias: "",
        tableTurns05: "",
        tableLow: "",
        tableHigh: "",
        note: `${columns[3]}, bot ${columns[4]}, max DD ${columns[5]}, trades ${columns[6]}`,
      },
    ];
  });
}

function parseSelectedMarkdownRows(markdown: string): CaseRow[] {
  const start = markdown.indexOf("| case");
  if (start === -1) {
    return [];
  }

  const end = markdown.indexOf("\n\n", start);
  const table = markdown.slice(start, end === -1 ? undefined : end);
  const rows: CaseRow[] = [];

  for (const line of table.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed.startsWith("|") || trimmed.includes("---") || trimmed.includes("| case")) {
      continue;
    }

    const columns = trimmed
      .split("|")
      .slice(1, -1)
      .map((column) => stripMarkdownCode(column.trim()));
    if (columns.length < 8) {
      continue;
    }

    const [startDate, endDate] = columns[1].split("..");
    if (!isIsoDate(startDate) || !isIsoDate(endDate)) {
      continue;
    }

    rows.push({
      label: columns[0],
      group: "3d selected cases",
      intervalLabel: columns[1],
      startDate,
      endDate,
      tableRet: columns[2],
      tableSpan: columns[3],
      tableBias: columns[4],
      tableTurns05: columns[5],
      tableLow: columns[6],
      tableHigh: columns[7],
    });
  }

  return rows;
}

function parseSharpeRows(markdown: string): CaseRow[] {
  return parseTsvBlock(
    markdown,
    "duration\ttrend\tinterval\tmarket Sharpe\tmarket move\tbot return\tbot max DD\tbot ann. Sharpe",
  ).flatMap((columns) => {
    if (columns.length < 8) {
      return [];
    }
    const [startDate, endDate] = columns[2].split("..");
    if (!isIsoDate(startDate) || !isIsoDate(endDate)) {
      return [];
    }
    return [
      {
        label: `${columns[0]} ${columns[1]} Sharpe trend, market ${columns[4]}`,
        group: "Sharpe trend cases",
        intervalLabel: columns[2],
        startDate,
        endDate,
        tableRet: columns[4],
        tableSpan: "",
        tableBias: "",
        tableTurns05: "",
        tableLow: "",
        tableHigh: "",
        note: `market Sharpe ${columns[3]}, bot ${columns[5]}, max DD ${columns[6]}, bot ann. Sharpe ${columns[7]}`,
      },
    ];
  });
}

function parseStressRows(markdown: string): CaseRow[] {
  return parseTsvBlock(
    markdown,
    "duration\tinterval\tmarket Sharpe\tmarket move\tbot return",
  ).flatMap((columns) => {
    if (columns.length < 2) {
      return [];
    }
    const [startDate, endDate] = columns[1].split("..");
    if (!isIsoDate(startDate) || !isIsoDate(endDate)) {
      return [];
    }
    const marketMove = columns[3] ?? "";
    const botReturn = columns[4] ?? "";
    return [
      {
        label: `${columns[0]} downtrend stress${marketMove ? `, market ${marketMove}` : ""}`,
        group: "Downtrend stress cases",
        intervalLabel: columns[1],
        startDate,
        endDate,
        tableRet: marketMove,
        tableSpan: "",
        tableBias: "",
        tableTurns05: "",
        tableLow: "",
        tableHigh: "",
        note: [
          columns[2] ? `market Sharpe ${columns[2]}` : "",
          botReturn ? `bot ${botReturn}` : "",
        ]
          .filter(Boolean)
          .join(", "),
      },
    ];
  });
}

function parseTsvBlock(markdown: string, header: string): string[][] {
  const start = markdown.indexOf(header);
  if (start === -1) {
    return [];
  }

  const blockStart = start + header.length;
  const end = markdown.indexOf("\n\n", blockStart);
  const block = markdown.slice(blockStart, end === -1 ? undefined : end);
  return block
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => line.split("\t").map((column) => column.trim()));
}

function uniqueRows(rows: CaseRow[]): CaseRow[] {
  const seen = new Set<string>();
  const unique: CaseRow[] = [];
  for (const row of rows) {
    const key = `${row.group}|${row.label}|${row.intervalLabel}`;
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    unique.push(row);
  }
  return unique;
}

function stripMarkdownCode(value: string): string {
  return value.replace(/^`|`$/g, "");
}

function isIsoDate(value: string | undefined): value is string {
  return /^\d{4}-\d{2}-\d{2}$/.test(value ?? "");
}

async function fetchCaseCandles(
  row: CaseRow,
  options: {
    interval: string;
    intervalMs: number;
    warmupCandles: number;
    lookaheadCandles?: number;
  },
): Promise<Candle[]> {
  const { startTime, endExclusive } = caseTimeRange(row);
  const fetchStartTime = startTime - options.warmupCandles * options.intervalMs;
  const fetchEndExclusive =
    endExclusive + (options.lookaheadCandles ?? 0) * options.intervalMs;
  const candles: Candle[] = [];
  let cursor = fetchStartTime;

  while (cursor < fetchEndExclusive) {
    const requestEnd = fetchEndExclusive - 1;
    const url = new URL("https://api.binance.com/api/v3/klines");
    url.search = new URLSearchParams({
      symbol,
      interval: options.interval,
      startTime: String(cursor),
      endTime: String(requestEnd),
      limit: "1000",
    }).toString();

    const response = await fetch(url, { signal: AbortSignal.timeout(20_000) });
    if (!response.ok) {
      const body = await response.text();
      throw new Error(`Binance klines failed: HTTP ${response.status} ${body.slice(0, 240)}`);
    }

    const rows = (await response.json()) as unknown[][];
    if (rows.length === 0) {
      break;
    }

    const parsed = rows
      .map((item) => parseKline(item, options.interval))
      .filter((candle) => candle.openTime < fetchEndExclusive);
    candles.push(...parsed);
    const last = parsed.at(-1);
    if (!last || last.openTime < cursor) {
      break;
    }
    cursor = last.openTime + options.intervalMs;
  }

  return uniqueCandles(candles).filter(
    (candle) => candle.openTime >= fetchStartTime && candle.openTime < fetchEndExclusive,
  );
}

function caseTimeRange(row: CaseRow): { startTime: number; endExclusive: number } {
  return {
    startTime: Date.parse(`${row.startDate}T00:00:00.000Z`),
    endExclusive: Date.parse(`${addUtcDays(row.endDate, 1)}T00:00:00.000Z`),
  };
}

function parseKline(row: unknown[], interval: string): Candle {
  return {
    symbol,
    interval,
    openTime: Number(row[0]),
    open: Number(row[1]),
    high: Number(row[2]),
    low: Number(row[3]),
    close: Number(row[4]),
    volume: Number(row[5]),
    closeTime: Number(row[6]),
    closed: true,
  };
}

function uniqueCandles(candles: Candle[]): Candle[] {
  return [...new Map(candles.map((candle) => [candle.openTime, candle])).values()].sort(
    (left, right) => left.openTime - right.openTime,
  );
}

function browserCaseFromRenderedCase(item: RenderedCase): BrowserCase {
  const { startTime, endExclusive } = caseTimeRange(item.row);
  return {
    label: item.row.label,
    intervalLabel: item.row.intervalLabel,
    startTime,
    endExclusive,
    candles: item.candles.map(browserCandle),
    smaCandles: item.smaCandles.map(browserCandle),
    replayCandles: item.replayCandles.map(browserCandle),
  };
}

function browserCandle(candle: Candle): BrowserCandle {
  return {
    symbol: candle.symbol,
    interval: candle.interval,
    openTime: candle.openTime,
    closeTime: candle.closeTime,
    open: candle.open,
    high: candle.high,
    low: candle.low,
    close: candle.close,
    volume: candle.volume,
    closed: candle.closed,
  };
}

function withComputedMetrics(row: CaseRow, candles: Candle[]): CaseRow {
  const first = candles[0];
  const last = candles.at(-1);
  if (!first || !last) {
    return row;
  }

  const low = Math.min(...candles.map((candle) => candle.low));
  const high = Math.max(...candles.map((candle) => candle.high));
  const returnPct = ((last.close - first.open) / first.open) * 100;
  const spanPct = ((high - low) / first.open) * 100;
  const midpoint = (first.open + last.close) / 2;
  const averagePrice =
    candles.reduce((sum, candle) => sum + (candle.open + candle.high + candle.low + candle.close) / 4, 0) /
    candles.length;
  const biasPct = midpoint > 0 ? ((averagePrice - midpoint) / midpoint) * 100 : 0;

  return {
    ...row,
    tableRet: row.tableRet || formatPercent(returnPct, 3),
    tableSpan: row.tableSpan || formatPercent(spanPct, 3),
    tableBias: row.tableBias || formatPercent(biasPct, 3),
    tableTurns05: row.tableTurns05 || String(countZigzagTurns(candles, 0.5)),
    tableLow: row.tableLow || formatPrice(low),
    tableHigh: row.tableHigh || formatPrice(high),
  };
}

function countZigzagTurns(candles: Candle[], thresholdPct: number): number {
  const closes = candles.map((candle) => candle.close);
  if (closes.length < 2) {
    return 0;
  }

  let anchor = closes[0];
  let direction = 0;
  let turns = 0;

  for (const close of closes.slice(1)) {
    const movePct = anchor > 0 ? ((close - anchor) / anchor) * 100 : 0;
    if (direction === 0) {
      if (Math.abs(movePct) >= thresholdPct) {
        direction = movePct > 0 ? 1 : -1;
        anchor = close;
      }
      continue;
    }

    if (direction > 0) {
      if (close > anchor) {
        anchor = close;
      } else if (((anchor - close) / anchor) * 100 >= thresholdPct) {
        turns += 1;
        direction = -1;
        anchor = close;
      }
    } else if (close < anchor) {
      anchor = close;
    } else if (((close - anchor) / anchor) * 100 >= thresholdPct) {
      turns += 1;
      direction = 1;
      anchor = close;
    }
  }

  return turns;
}

function renderHtml(cases: RenderedCase[], browserBundle: string): string {
  const generatedAt = new Date().toISOString();
  const sections = cases.map((item, index) => renderCaseSection(item, index)).join("\n");
  const browserCases = safeJson(cases.map(browserCaseFromRenderedCase));
  const overviewRows = cases
    .map(
      ({ row }, index) => {
        const sectionId = caseSectionId(row, index);
        return `
        <tr>
          <td>${escapeHtml(row.group)}</td>
          <td><a class="case-link" href="#${sectionId}">${escapeHtml(row.label)}</a></td>
          <td><a class="case-link muted" href="#${sectionId}">${escapeHtml(row.intervalLabel)}</a></td>
          <td class="${row.tableRet.startsWith("-") ? "negative" : "positive"}">${escapeHtml(
            row.tableRet,
          )}</td>
          <td>${escapeHtml(row.tableSpan)}</td>
          <td>${escapeHtml(row.tableLow)}</td>
          <td>${escapeHtml(row.tableHigh)}</td>
        </tr>`;
      },
    )
    .join("");

  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BTCUSDT Interval Price Movement</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0b0d12;
      --panel: #121620;
      --panel-2: #171c27;
      --line: #2c3342;
      --grid: #273041;
      --text: #edf2ff;
      --muted: #9aa7bd;
      --green: #20c978;
      --red: #ff5c6c;
      --blue: #49a7ff;
      --amber: #f5b84b;
      --sma-1h: #facc15;
      --sma-4h: #38bdf8;
      --sma-12h: #c084fc;
      --entry-buy: #15803d;
      --entry-sell: #b45309;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family:
        Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    html {
      scroll-behavior: smooth;
    }

    a {
      color: inherit;
    }

    main {
      max-width: 1240px;
      margin: 0 auto;
      padding: 28px 20px 48px;
    }

    header {
      display: grid;
      gap: 10px;
      margin-bottom: 22px;
    }

    h1 {
      margin: 0;
      font-size: clamp(28px, 4vw, 46px);
      line-height: 1;
      letter-spacing: 0;
    }

    p {
      margin: 0;
      color: var(--muted);
      line-height: 1.55;
      max-width: 860px;
    }

    .overview {
      overflow-x: auto;
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      margin: 24px 0;
    }

    .controls {
      display: flex;
      flex-wrap: wrap;
      align-items: end;
      gap: 12px;
      padding: 14px 16px;
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
    }

    .control {
      display: grid;
      gap: 6px;
      min-width: 180px;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }

    .control.inline {
      display: inline-flex;
      align-items: center;
      min-width: auto;
      min-height: 38px;
      text-transform: none;
      letter-spacing: 0;
      font-size: 13px;
    }

    .preset-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }

    button {
      min-height: 32px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #0f131c;
      color: var(--text);
      font: inherit;
      cursor: pointer;
    }

    button:hover {
      border-color: var(--blue);
    }

    select,
    input[type="range"],
    input[type="number"] {
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #0f131c;
      color: var(--text);
      font: inherit;
    }

    select {
      min-width: 160px;
      padding: 0 10px;
    }

    input[type="number"] {
      width: 110px;
      padding: 0 10px;
    }

    input[type="range"] {
      width: 150px;
      accent-color: var(--blue);
    }

    .entry-status {
      flex: 1 1 280px;
      min-height: 38px;
      display: flex;
      align-items: center;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.4;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      min-width: 860px;
    }

    th,
    td {
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: right;
      font-variant-numeric: tabular-nums;
      white-space: nowrap;
    }

    th:first-child,
    td:first-child,
    th:nth-child(2),
    td:nth-child(2),
    th:nth-child(3),
    td:nth-child(3) {
      text-align: left;
    }

    tr:last-child td {
      border-bottom: 0;
    }

    th {
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }

    .charts {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(min(100%, 540px), 1fr));
      gap: 16px;
    }

    section {
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      overflow: hidden;
      scroll-margin-top: 16px;
    }

    section:target {
      border-color: var(--blue);
      box-shadow: 0 0 0 1px rgba(73, 167, 255, 0.35);
    }

    .section-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 14px 16px 10px;
      background: var(--panel-2);
      border-bottom: 1px solid var(--line);
    }

    h2 {
      margin: 0;
      font-size: 16px;
      line-height: 1.25;
      letter-spacing: 0;
    }

    .case-link {
      color: var(--text);
      text-decoration: none;
    }

    .case-link:hover {
      color: var(--blue);
      text-decoration: underline;
      text-underline-offset: 3px;
    }

    .case-link.muted {
      color: var(--muted);
    }

    .title-stack {
      display: grid;
      gap: 4px;
    }

    .case-group {
      color: var(--muted);
      font-size: 11px;
      line-height: 1.2;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }

    .interval {
      color: var(--muted);
      font-size: 13px;
      font-variant-numeric: tabular-nums;
      white-space: nowrap;
    }

    .metrics {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      padding: 10px 16px 0;
    }

    .case-note {
      padding: 8px 16px 0;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }

    .metric {
      display: grid;
      gap: 2px;
      min-width: 86px;
    }

    .metric span {
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }

    .metric strong {
      font-size: 13px;
      font-variant-numeric: tabular-nums;
    }

    .legend {
      display: flex;
      flex-wrap: wrap;
      gap: 8px 14px;
      padding: 10px 16px 0;
      color: var(--muted);
      font-size: 12px;
      font-variant-numeric: tabular-nums;
    }

    .legend-item {
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }

    .legend-swatch {
      width: 22px;
      height: 0;
      border-top: 2px solid currentColor;
    }

    .entry-summary {
      display: flex;
      flex-wrap: wrap;
      gap: 8px 14px;
      min-height: 24px;
      padding: 8px 16px 0;
      color: var(--muted);
      font-size: 12px;
      font-variant-numeric: tabular-nums;
    }

    .entry-summary span {
      white-space: nowrap;
    }

    svg {
      display: block;
      width: 100%;
      height: auto;
    }

    .positive {
      color: var(--green);
    }

    .negative {
      color: var(--red);
    }

    .axis {
      fill: var(--muted);
      font-size: 11px;
      font-variant-numeric: tabular-nums;
    }

    .entry-layer {
      display: none;
    }

    .sma-centered {
      display: none;
    }

    .source {
      margin-top: 22px;
      font-size: 13px;
    }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>BTCUSDT interval price movement</h1>
      <p>Each panel renders one UTC interval from the interval blocks in <code>tasks.md</code>, using Binance spot ${chartInterval} klines. The close path is overlaid on the intraperiod high/low envelope, with the dashed line marking the opening price. SMA overlays use 1h, 4h, and 12h close windows with pre-interval warmup candles; the centered SMA toggle shifts each average left by half its window and uses post-interval SMA candles to preserve coverage. Entry thickness overlays are simulated from ${strategyInterval} strategy replays.</p>
    </header>
    <div class="controls">
      <label class="control">
        Sigma mode
        <select id="sigmaMode">
          <option value="static">static</option>
          <option value="sigmoid-trend">sigmoid-trend</option>
        </select>
      </label>
      <label class="control static-control">
        Buy sigma
        <input id="buySigma" type="number" min="0.001" step="0.01" value="0.1">
      </label>
      <label class="control static-control">
        Sell sigma
        <input id="sellSigma" type="number" min="0.001" step="0.01" value="0.1">
      </label>
      <label class="control sigmoid-control">
        Sigmoid low
        <input id="sigmoidLow" type="number" min="0.001" step="0.01" value="0.05">
      </label>
      <label class="control sigmoid-control">
        Sigmoid high
        <input id="sigmoidHigh" type="number" min="0.001" step="0.01" value="0.3">
      </label>
      <label class="control sigmoid-control">
        Trend window
        <select id="trendWindowSec">
          <option value="3600">1h</option>
          <option value="14400">4h</option>
          <option value="43200" selected>12h</option>
        </select>
      </label>
      <label class="control sigmoid-control">
        Sell slope
        <input id="trendSellB1" type="number" step="1" value="15">
      </label>
      <label class="control sigmoid-control">
        Buy slope
        <input id="trendBuyB2" type="number" step="1" value="300">
      </label>
      <label class="control inline">
        <input id="entryToggle" type="checkbox">
        Show entries
      </label>
      <label class="control inline">
        <input id="centerSmaToggle" type="checkbox">
        Center SMA
      </label>
      <label class="control">
        Thickness scale
        <input id="entryScale" type="range" min="0.4" max="2.2" step="0.1" value="1">
      </label>
      <div class="control">
        Static presets
        <div class="preset-row">
          <button type="button" data-static-preset="0.1/0.1">0.1/0.1</button>
          <button type="button" data-static-preset="0.1/0.3">0.1/0.3</button>
          <button type="button" data-static-preset="0.3/0.1">0.3/0.1</button>
          <button type="button" data-static-preset="0.3/0.3">0.3/0.3</button>
        </div>
      </div>
      <div id="entryStatus" class="entry-status"></div>
    </div>
    <div class="overview">
      <table>
        <thead>
          <tr>
            <th>Group</th>
            <th>Case</th>
            <th>Interval</th>
            <th>Return</th>
            <th>Span</th>
            <th>Low</th>
            <th>High</th>
          </tr>
        </thead>
        <tbody>${overviewRows}
        </tbody>
      </table>
    </div>
    <div class="charts">
      ${sections}
    </div>
    <p class="source">Generated ${generatedAt}. Source: Binance public spot klines endpoint, symbol ${symbol}, chart interval ${chartInterval}, strategy interval ${strategyInterval}. Intervals are inclusive by date: chart candles run from 00:00 UTC through the last ${chartInterval} candle of the final date, and strategy replay uses every ${strategyInterval} candle in the same UTC dates.</p>
  </main>
  <script>
    const chartCases = ${browserCases};
  </script>
  <script>${browserBundle}</script>
  <script>
    const sigmaMode = document.getElementById("sigmaMode");
    const buySigma = document.getElementById("buySigma");
    const sellSigma = document.getElementById("sellSigma");
    const sigmoidLow = document.getElementById("sigmoidLow");
    const sigmoidHigh = document.getElementById("sigmoidHigh");
    const trendWindowSec = document.getElementById("trendWindowSec");
    const trendSellB1 = document.getElementById("trendSellB1");
    const trendBuyB2 = document.getElementById("trendBuyB2");
    const entryToggle = document.getElementById("entryToggle");
    const centerSmaToggle = document.getElementById("centerSmaToggle");
    const entryScale = document.getElementById("entryScale");
    const entryStatus = document.getElementById("entryStatus");
    let renderTimer;
    let renderVersion = 0;
    let latestEntries = [];

    const plot = { left: 74, right: 916, top: 28, bottom: 288 };
    const chartIntervalMs = ${chartIntervalMs};

    function settingsFromControls() {
      return {
        sigmaMode: sigmaMode.value,
        buySigma: Number(buySigma.value),
        sellSigma: Number(sellSigma.value),
        sigmoidSigmaLow: Number(sigmoidLow.value),
        sigmoidSigmaHigh: Number(sigmoidHigh.value),
        trendSigmaWindowSec: Number(trendWindowSec.value),
        trendSigmaSellB1: Number(trendSellB1.value),
        trendSigmaBuyB2: Number(trendBuyB2.value),
      };
    }

    function updateModeControls() {
      const sigmoid = sigmaMode.value === "sigmoid-trend";
      document.querySelectorAll(".static-control").forEach((node) => {
        node.style.display = sigmoid ? "none" : "grid";
      });
      document.querySelectorAll(".sigmoid-control").forEach((node) => {
        node.style.display = sigmoid ? "grid" : "none";
      });
    }

    function updateSmaMode() {
      const centered = centerSmaToggle.checked;
      document.querySelectorAll(".sma-trailing").forEach((node) => {
        node.style.display = centered ? "none" : "inline";
      });
      document.querySelectorAll(".sma-centered").forEach((node) => {
        node.style.display = centered ? "inline" : "none";
      });
    }

    function scheduleRerender() {
      clearTimeout(renderTimer);
      const version = ++renderVersion;
      updateModeControls();
      if (!entryToggle.checked) {
        latestEntries = chartCases.map(() => []);
        drawLatestEntries();
        entryStatus.textContent = "Entry overlay hidden.";
        return;
      }
      entryStatus.textContent = "Computing strategy overlay...";
      renderTimer = setTimeout(() => {
        void rerenderEntryLayers(version);
      }, 30);
    }

    async function rerenderEntryLayers(version) {
      const startedAt = performance.now();
      const settings = settingsFromControls();
      latestEntries = chartCases.map(() => []);
      drawLatestEntries();

      for (const [index, testCase] of chartCases.entries()) {
        if (version !== renderVersion) {
          return;
        }
        entryStatus.textContent =
          "Computing strategy overlay " + (index + 1).toLocaleString() + "/" + chartCases.length.toLocaleString() + "...";
        latestEntries[index] = TradingStrategyBrowser.runEntryOverlayCase(testCase, settings);
        drawLatestEntries();
        await new Promise((resolve) => setTimeout(resolve, 0));
      }

      if (version !== renderVersion) {
        return;
      }
      const summary = summarizeEntries(latestEntries);
      entryStatus.textContent = [
        "Rendered " + summary.totalCount.toLocaleString() + " entry buckets",
        "buy " + summary.buyCount.toLocaleString() + " / " + formatQuote(summary.buyQuote),
        "sell " + summary.sellCount.toLocaleString() + " / " + formatQuote(summary.sellQuote),
        Math.round(performance.now() - startedAt).toLocaleString() + "ms",
      ].join(" | ");
    }

    function drawLatestEntries() {
      const enabled = entryToggle.checked;
      const scale = Number(entryScale.value) || 1;
      document.querySelectorAll(".entry-layer").forEach((layer) => {
        const caseIndex = Number(layer.dataset.caseIndex);
        layer.replaceChildren();
        layer.style.display = enabled ? "inline" : "none";
        if (!enabled) {
          return;
        }

        const testCase = chartCases[caseIndex];
        const entries = latestEntries[caseIndex] || [];
        const segments = entries
          .map((entry) => entrySegmentForEntry(testCase, entry))
          .filter(Boolean);
        const maxQuote = Math.max(1, ...segments.map((segment) => segment.quote));
        for (const segment of segments) {
          const baseWidth = 2 + Math.sqrt(segment.quote / maxQuote) * 12;
          const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
          line.setAttribute("x1", round(segment.x1));
          line.setAttribute("y1", round(segment.y1));
          line.setAttribute("x2", round(segment.x2));
          line.setAttribute("y2", round(segment.y2));
          line.setAttribute("stroke", segment.side === "buy" ? "var(--entry-buy)" : "var(--entry-sell)");
          line.setAttribute("stroke-width", String(Math.max(1, baseWidth * scale)));
          line.setAttribute("stroke-linecap", "round");
          line.setAttribute("opacity", "0.72");
          const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
          title.textContent = segment.side + " entry " + formatQuote(segment.quote);
          line.appendChild(title);
          layer.appendChild(line);
        }
      });
      updateEntrySummaries();
    }

    function updateEntrySummaries() {
      document.querySelectorAll(".entry-summary").forEach((summaryNode) => {
        const caseIndex = Number(summaryNode.dataset.caseIndex);
        const summary = summarizeEntries([latestEntries[caseIndex] || []]);
        summaryNode.innerHTML =
          '<span style="color: var(--entry-buy)">buy ' +
          summary.buyCount.toLocaleString() +
          ' / ' +
          formatQuote(summary.buyQuote) +
          '</span><span style="color: var(--entry-sell)">sell ' +
          summary.sellCount.toLocaleString() +
          ' / ' +
          formatQuote(summary.sellQuote) +
          '</span>';
      });
    }

    function entrySegmentForEntry(testCase, entry) {
      const index = testCase.candles.findIndex((candle) => candle.openTime === entry.time);
      if (index < 0) {
        return undefined;
      }
      const geometry = chartGeometry(testCase);
      const candle = testCase.candles[index];
      const previous = testCase.candles[index - 1];
      const x1 = previous ? xFor(geometry, previous.openTime) : xFor(geometry, candle.openTime);
      const y1 = previous ? yFor(geometry, previous.close) : yFor(geometry, candle.close);
      const x2 = previous ? xFor(geometry, candle.openTime) : xFor(geometry, candle.closeTime);
      const y2 = yFor(geometry, candle.close);
      return { x1, y1, x2, y2, side: entry.side, quote: entry.quote };
    }

    function chartGeometry(testCase) {
      if (testCase.geometry) {
        return testCase.geometry;
      }
      const firstTime = testCase.candles[0].openTime;
      const lastOpenTime = testCase.candles[testCase.candles.length - 1].openTime;
      const lastTime = testCase.candles[testCase.candles.length - 1].closeTime;
      const smaSeries = [4, 16, 48].flatMap((periods) =>
        [
          ...movingAverage(testCase.smaCandles, periods),
          ...shiftPoints(movingAverage(testCase.smaCandles, periods), centeredSmaShiftMs(periods)),
        ].filter((point) => point.time >= firstTime && point.time <= lastOpenTime),
      );
      const prices = [
        ...testCase.candles.flatMap((candle) => [candle.low, candle.high]),
        ...smaSeries.map((point) => point.value),
      ];
      const min = Math.min(...prices);
      const max = Math.max(...prices);
      const pad = Math.max((max - min) * 0.08, max * 0.0005);
      testCase.geometry = { firstTime, lastTime, low: min - pad, high: max + pad };
      return testCase.geometry;
    }

    function movingAverage(candles, periods) {
      const points = [];
      let sum = 0;
      for (let index = 0; index < candles.length; index += 1) {
        sum += candles[index].close;
        if (index >= periods) {
          sum -= candles[index - periods].close;
        }
        if (index >= periods - 1) {
          points.push({ time: candles[index].openTime, value: sum / periods });
        }
      }
      return points;
    }

    function shiftPoints(points, shiftMs) {
      return points.map((point) => ({ time: point.time - shiftMs, value: point.value }));
    }

    function centeredSmaShiftMs(periods) {
      return (periods * chartIntervalMs) / 2;
    }

    function xFor(geometry, time) {
      return plot.left + ((time - geometry.firstTime) / Math.max(1, geometry.lastTime - geometry.firstTime)) * (plot.right - plot.left);
    }

    function yFor(geometry, price) {
      return plot.top + ((geometry.high - price) / Math.max(1, geometry.high - geometry.low)) * (plot.bottom - plot.top);
    }

    function round(value) {
      return String(Math.round(value * 100) / 100);
    }

    function formatQuote(value) {
      return value.toLocaleString("en-US", { maximumFractionDigits: 2 });
    }

    function summarizeEntries(entrySets) {
      const summary = {
        totalCount: 0,
        buyCount: 0,
        sellCount: 0,
        buyQuote: 0,
        sellQuote: 0,
      };
      for (const entries of entrySets) {
        for (const entry of entries) {
          summary.totalCount += 1;
          if (entry.side === "buy") {
            summary.buyCount += 1;
            summary.buyQuote += entry.quote;
          } else {
            summary.sellCount += 1;
            summary.sellQuote += entry.quote;
          }
        }
      }
      return summary;
    }

    document.querySelectorAll("[data-static-preset]").forEach((button) => {
      button.addEventListener("click", () => {
        const [buy, sell] = button.dataset.staticPreset.split("/");
        sigmaMode.value = "static";
        buySigma.value = buy;
        sellSigma.value = sell;
        scheduleRerender();
      });
    });

    [
      sigmaMode,
      buySigma,
      sellSigma,
      sigmoidLow,
      sigmoidHigh,
      trendWindowSec,
      trendSellB1,
      trendBuyB2,
    ].forEach((control) => control.addEventListener("input", scheduleRerender));
    entryToggle.addEventListener("change", scheduleRerender);
    centerSmaToggle.addEventListener("change", updateSmaMode);
    entryScale.addEventListener("input", drawLatestEntries);
    updateModeControls();
    updateSmaMode();
    scheduleRerender();
  </script>
</body>
</html>
`;
}

function renderCaseSection({
  row,
  candles,
  smaCandles,
}: RenderedCase, caseIndex: number): string {
  const stroke = row.tableRet.startsWith("-") ? "var(--red)" : "var(--green)";
  const note = row.note ? `<div class="case-note">${escapeHtml(row.note)}</div>` : "";
  const sectionId = caseSectionId(row, caseIndex);
  return `<section id="${sectionId}">
  <div class="section-head">
    <div class="title-stack">
      <div class="case-group">${escapeHtml(row.group)}</div>
      <h2><a class="case-link" href="#${sectionId}">${escapeHtml(row.label)}</a></h2>
    </div>
    <div class="interval">${escapeHtml(row.intervalLabel)}</div>
  </div>
  <div class="metrics">
    ${metric("Ret", row.tableRet, row.tableRet.startsWith("-") ? "negative" : "positive")}
    ${metric("Span", row.tableSpan, "")}
    ${metric("Bias", row.tableBias, row.tableBias.startsWith("-") ? "negative" : "positive")}
    ${metric("Turns", row.tableTurns05, "")}
    ${metric("Low", row.tableLow, "")}
    ${metric("High", row.tableHigh, "")}
  </div>
  ${note}
  ${renderLegend()}
  <div class="entry-summary" data-case-index="${caseIndex}"></div>
  ${renderSvg(candles, smaCandles, stroke, caseIndex)}
</section>`;
}

function caseSectionId(row: CaseRow, index: number): string {
  return `case-${index + 1}-${slugify(row.intervalLabel)}-${slugify(row.label)}`;
}

function slugify(value: string): string {
  const slug = value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return slug || "interval";
}

function metric(label: string, value: string, className: string): string {
  return `<div class="metric">
    <span>${label}</span>
    <strong class="${className}">${escapeHtml(value)}</strong>
  </div>`;
}

function renderLegend(): string {
  return `<div class="legend">
    ${smaConfigs
      .map(
        (series) => `<span class="legend-item" style="color: ${series.color}">
          <span class="legend-swatch"></span>
          <span>${escapeHtml(series.label)}</span>
        </span>`,
      )
      .join("")}
    <span class="legend-item" style="color: var(--entry-buy)">
      <span class="legend-swatch"></span>
      <span>Buy entry size</span>
    </span>
    <span class="legend-item" style="color: var(--entry-sell)">
      <span class="legend-swatch"></span>
      <span>Sell entry size</span>
    </span>
  </div>`;
}

function renderSvg(
  candles: Candle[],
  smaCandles: Candle[],
  stroke: string,
  caseIndex: number,
): string {
  const width = 940;
  const height = 330;
  const plot = {
    left: 74,
    right: width - 24,
    top: 28,
    bottom: height - 42,
  };
  const firstTime = candles[0].openTime;
  const lastOpenTime = candles[candles.length - 1].openTime;
  const lastTime = candles[candles.length - 1].closeTime;
  const trailingSmaSeries = smaConfigs.map((series) => ({
    ...series,
    points: visibleSmaPoints(smaCandles, series.periods, firstTime, lastOpenTime, 0),
  }));
  const centeredSmaSeries = smaConfigs.map((series) => ({
    ...series,
    points: visibleSmaPoints(
      smaCandles,
      series.periods,
      firstTime,
      lastOpenTime,
      centeredSmaShiftMs(series.periods),
    ),
  }));
  const prices = [
    ...candles.flatMap((candle) => [candle.low, candle.high]),
    ...trailingSmaSeries.flatMap((series) => series.points.map((point) => point.value)),
    ...centeredSmaSeries.flatMap((series) => series.points.map((point) => point.value)),
  ];
  const min = Math.min(...prices);
  const max = Math.max(...prices);
  const pad = Math.max((max - min) * 0.08, max * 0.0005);
  const low = min - pad;
  const high = max + pad;
  const openPrice = candles[0].open;
  const yTicks = createPriceTicks(low, high, 5);
  const xTicks = createDayTicks(firstTime, lastTime);

  const xFor = (time: number) =>
    plot.left + ((time - firstTime) / Math.max(1, lastTime - firstTime)) * (plot.right - plot.left);
  const yFor = (price: number) =>
    plot.top + ((high - price) / Math.max(1, high - low)) * (plot.bottom - plot.top);

  const closePath = linePath(candles.map((candle) => [xFor(candle.openTime), yFor(candle.close)]));
  const trailingSmaPaths = renderSmaPaths(trailingSmaSeries, xFor, yFor);
  const centeredSmaPaths = renderSmaPaths(centeredSmaSeries, xFor, yFor);
  const highPath = linePath(candles.map((candle) => [xFor(candle.openTime), yFor(candle.high)]));
  const lowPath = linePath(
    [...candles].reverse().map((candle) => [xFor(candle.openTime), yFor(candle.low)]),
    false,
  );
  const envelopePath = `${highPath} ${lowPath.replace(/^M /, "L ")} Z`;
  const lowCandle = candles.reduce((best, candle) => (candle.low < best.low ? candle : best));
  const highCandle = candles.reduce((best, candle) => (candle.high > best.high ? candle : best));

  return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Price movement chart">
    <rect x="0" y="0" width="${width}" height="${height}" fill="transparent" />
    ${yTicks
      .map((tick) => {
        const y = yFor(tick);
        return `<line x1="${plot.left}" x2="${plot.right}" y1="${round(y)}" y2="${round(
          y,
        )}" stroke="var(--grid)" stroke-width="1" />
        <text class="axis" x="${plot.left - 10}" y="${round(y + 4)}" text-anchor="end">${escapeHtml(
          formatUsd(tick),
        )}</text>`;
      })
      .join("")}
    ${xTicks
      .map((tick) => {
        const x = xFor(tick);
        return `<line x1="${round(x)}" x2="${round(x)}" y1="${plot.top}" y2="${plot.bottom}" stroke="var(--grid)" stroke-width="1" />
        <text class="axis" x="${round(x)}" y="${height - 18}" text-anchor="middle">${escapeHtml(
          formatDay(tick),
        )}</text>`;
      })
      .join("")}
    <line x1="${plot.left}" x2="${plot.right}" y1="${round(yFor(openPrice))}" y2="${round(
      yFor(openPrice),
    )}" stroke="var(--muted)" stroke-width="1" stroke-dasharray="5 5" opacity="0.75" />
    <path d="${envelopePath}" fill="rgba(73, 167, 255, 0.12)" stroke="none" />
    <g class="sma-layer sma-trailing">
      ${trailingSmaPaths}
    </g>
    <g class="sma-layer sma-centered">
      ${centeredSmaPaths}
    </g>
    <path d="${closePath}" fill="none" stroke="${stroke}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round" />
    <g class="entry-layer" data-case-index="${caseIndex}"></g>
    <circle cx="${round(xFor(candles[0].openTime))}" cy="${round(
      yFor(candles[0].open),
    )}" r="4" fill="var(--blue)" />
    <circle cx="${round(xFor(candles[candles.length - 1].openTime))}" cy="${round(
      yFor(candles[candles.length - 1].close),
    )}" r="4" fill="${stroke}" />
    <circle cx="${round(xFor(lowCandle.openTime))}" cy="${round(
      yFor(lowCandle.low),
    )}" r="3.5" fill="var(--amber)" />
    <circle cx="${round(xFor(highCandle.openTime))}" cy="${round(
      yFor(highCandle.high),
    )}" r="3.5" fill="var(--amber)" />
  </svg>`;
}

function movingAverage(candles: Candle[], periods: number): LinePoint[] {
  const points: LinePoint[] = [];
  let sum = 0;

  for (let index = 0; index < candles.length; index += 1) {
    sum += candles[index].close;
    if (index >= periods) {
      sum -= candles[index - periods].close;
    }
    if (index >= periods - 1) {
      points.push({
        time: candles[index].openTime,
        value: sum / periods,
      });
    }
  }

  return points;
}

function visibleSmaPoints(
  candles: Candle[],
  periods: number,
  firstTime: number,
  lastOpenTime: number,
  shiftMs: number,
): LinePoint[] {
  return movingAverage(candles, periods)
    .map((point) => ({ time: point.time - shiftMs, value: point.value }))
    .filter((point) => point.time >= firstTime && point.time <= lastOpenTime);
}

function centeredSmaShiftMs(periods: number): number {
  return (periods * chartIntervalMs) / 2;
}

function renderSmaPaths(
  series: Array<SmaConfig & { points: LinePoint[] }>,
  xFor: (time: number) => number,
  yFor: (price: number) => number,
): string {
  return series
    .filter((item) => item.points.length > 1)
    .map((item) => {
      const path = linePath(item.points.map((point) => [xFor(point.time), yFor(point.value)]));
      return `<path d="${path}" fill="none" stroke="${item.color}" stroke-width="1.8" stroke-linejoin="round" stroke-linecap="round" opacity="0.92" />`;
    })
    .join("");
}

function linePath(points: Array<[number, number]>, includeMove = true): string {
  return points
    .map(([x, y], index) => `${index === 0 && includeMove ? "M" : "L"} ${round(x)} ${round(y)}`)
    .join(" ");
}

function createPriceTicks(low: number, high: number, count: number): number[] {
  if (count <= 1) {
    return [low];
  }
  return Array.from({ length: count }, (_, index) => low + ((high - low) * index) / (count - 1));
}

function createDayTicks(firstTime: number, lastTime: number): number[] {
  const ticks: number[] = [];
  const dayMs = 24 * 60 * 60 * 1000;
  let cursor = Date.UTC(
    new Date(firstTime).getUTCFullYear(),
    new Date(firstTime).getUTCMonth(),
    new Date(firstTime).getUTCDate(),
  );

  while (cursor <= lastTime) {
    if (cursor >= firstTime) {
      ticks.push(cursor);
    }
    cursor += dayMs;
  }

  return ticks;
}

function addUtcDays(date: string, days: number): string {
  const time = Date.parse(`${date}T00:00:00.000Z`) + days * 24 * 60 * 60 * 1000;
  return new Date(time).toISOString().slice(0, 10);
}

function formatDay(time: number): string {
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "2-digit",
    timeZone: "UTC",
  }).format(time);
}

function formatUsd(value: number): string {
  return value.toLocaleString("en-US", {
    maximumFractionDigits: value >= 10_000 ? 0 : 2,
    minimumFractionDigits: 0,
  });
}

function formatPercent(value: number, digits: number): string {
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(digits)}%`;
}

function formatPrice(value: number): string {
  return value.toLocaleString("en-US", {
    maximumFractionDigits: value >= 10_000 ? 2 : 4,
    minimumFractionDigits: 0,
    useGrouping: false,
  });
}

function round(value: number): number {
  return Math.round(value * 100) / 100;
}

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function safeJson(value: unknown): string {
  return JSON.stringify(value).replace(/</g, "\\u003c");
}
