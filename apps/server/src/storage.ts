import fs from "node:fs/promises";
import path from "node:path";
import {
  DailyCandleRecorder,
  SequentialShardStore,
  TradingStorageLayout,
} from "@trading/storage";
import type {
  BacktestResult,
  Candle,
  OrderBookSnapshot,
  PaperBotState,
} from "@trading/bot-algo";

export type TradingExecutionMode = "simulated" | "binance";
export type TradingExchangeMode =
  | "auto"
  | "live"
  | "spot-live"
  | "usdm-futures-live"
  | "coinm-futures-live"
  | "spot-testnet"
  | "spot-demo"
  | "usdm-futures-testnet"
  | "coinm-futures-testnet";

export interface TradingExchangeCredentials {
  mode: TradingExchangeMode;
  sandboxApiKey?: string;
  sandboxApiSecret?: string;
  liveApiKey?: string;
  liveApiSecret?: string;
  updatedAt: number;
}

export interface TradingRuntimeSettings {
  executionMode: TradingExecutionMode;
  exchange?: TradingExchangeCredentials;
  updatedAt: number;
}

export class TradingStorage {
  private readonly layout: TradingStorageLayout;
  private readonly candleRecorder: DailyCandleRecorder;

  constructor(
    dataDir: string,
    private readonly marketKey: string,
    private readonly symbol: string,
    interval: string,
  ) {
    this.layout = new TradingStorageLayout(dataDir);
    const market = safePathPart(marketKey);
    const normalizedSymbol = symbol.toLowerCase();
    this.candleRecorder = new DailyCandleRecorder({
      store: new SequentialShardStore(this.layout.marketStore),
      namespace: `candles/${market}/${normalizedSymbol}/${safePathPart(interval)}`,
      stagingDirectory: path.join(
        this.layout.marketMutable,
        "candles",
        market,
        normalizedSymbol,
        safePathPart(interval),
      ),
      symbol,
      interval,
      stepMs: candleIntervalMilliseconds(interval),
      metadata: { market },
    });
  }

  async ensureReady(): Promise<void> {
    await Promise.all([
      fs.mkdir(this.marketDir, { recursive: true }),
      fs.mkdir(this.stateDir, { recursive: true }),
      fs.mkdir(this.backtestDir, { recursive: true }),
      this.candleRecorder.ensureReady(),
    ]);
  }

  async loadBotState(): Promise<PaperBotState | undefined> {
    return unwrapStateFile(
      await readJson<PaperBotState | StateFile<PaperBotState>>(this.botStatePath),
    );
  }

  async saveBotState(state: PaperBotState): Promise<void> {
    await writeJsonAtomic(this.botStatePath, state);
  }

  async loadLiveBotState(): Promise<PaperBotState | undefined> {
    return unwrapStateFile(
      await readJson<PaperBotState | StateFile<PaperBotState>>(this.liveBotStatePath),
    );
  }

  async saveLiveBotState(state: PaperBotState): Promise<void> {
    await writeJsonAtomic(this.liveBotStatePath, state);
  }

  async loadTradingState<T>(): Promise<T | undefined> {
    return unwrapStateFile(await readJson<T | StateFile<T>>(this.tradingStatePath));
  }

  async saveTradingState(state: unknown): Promise<void> {
    await writeJsonAtomic(this.tradingStatePath, state);
  }

  async loadRuntimeSettings(): Promise<TradingRuntimeSettings | undefined> {
    return unwrapStateFile(
      await readJson<TradingRuntimeSettings | StateFile<TradingRuntimeSettings>>(
        this.runtimeSettingsPath,
      ),
    );
  }

  async saveRuntimeSettings(settings: TradingRuntimeSettings): Promise<void> {
    await writeJsonAtomic(this.runtimeSettingsPath, settings);
  }

  async loadCandles(limit = 500): Promise<Candle[]> {
    return this.candleRecorder.readRecent(limit);
  }

  async appendCandle(candle: Candle): Promise<void> {
    await this.candleRecorder.append(candle);
  }

  async loadOrderBookSnapshots(limit = 2_000): Promise<OrderBookSnapshot[]> {
    return readJsonLines<OrderBookSnapshot>(this.orderBookPath, limit);
  }

  async appendOrderBookSnapshot(snapshot: OrderBookSnapshot): Promise<void> {
    await appendJsonLine(this.orderBookPath, snapshot);
  }

  async saveBacktest(result: BacktestResult): Promise<string> {
    const fileName = `backtest-${Date.now()}-${result.summary.source}.json`;
    const target = path.join(this.backtestDir, fileName);
    await writeJsonAtomic(target, result);
    return target;
  }

  private get marketDir(): string {
    return path.join(this.layout.marketMutable, "streams", safePathPart(this.marketKey));
  }

  private get stateDir(): string {
    return this.layout.runtimeState;
  }

  private get backtestDir(): string {
    return path.join(this.layout.runtimeBacktests, safePathPart(this.marketKey));
  }

  private get orderBookPath(): string {
    return path.join(this.marketDir, `${this.symbol.toLowerCase()}-orderbook.jsonl`);
  }

  private get botStatePath(): string {
    return path.join(
      this.stateDir,
      `paper-bot-${safePathPart(this.marketKey)}-${this.symbol.toLowerCase()}.json`,
    );
  }

  private get liveBotStatePath(): string {
    return path.join(
      this.stateDir,
      `live-bot-${safePathPart(this.marketKey)}-${this.symbol.toLowerCase()}.json`,
    );
  }

  private get tradingStatePath(): string {
    return path.join(
      this.stateDir,
      `trading-bot-${safePathPart(this.marketKey)}-${this.symbol.toLowerCase()}.json`,
    );
  }

  private get runtimeSettingsPath(): string {
    return path.join(
      this.stateDir,
      `runtime-${safePathPart(this.marketKey)}-${this.symbol.toLowerCase()}.json`,
    );
  }
}

function candleIntervalMilliseconds(interval: string): number {
  const match = /^(\d+)([smhdw])$/.exec(interval);
  if (!match) throw new Error(`Unsupported live candle interval: ${interval}.`);
  const value = Number(match[1]);
  const units: Record<string, number> = {
    s: 1_000,
    m: 60_000,
    h: 3_600_000,
    d: 86_400_000,
    w: 604_800_000,
  };
  const unit = units[match[2]!]!;
  const milliseconds = value * unit;
  if (!Number.isSafeInteger(milliseconds) || milliseconds < 1) {
    throw new Error(`Invalid live candle interval: ${interval}.`);
  }
  return milliseconds;
}

interface StateFile<T> {
  state?: T;
}

function safePathPart(value: string): string {
  return value.replace(/[^a-z0-9_-]+/gi, "-").replace(/^-+|-+$/g, "").toLowerCase();
}

function unwrapStateFile<T>(value: T | StateFile<T> | undefined): T | undefined {
  if (!value) {
    return undefined;
  }
  if (isRecord(value)) {
    const record = value as Record<string, unknown>;
    const state = record.state;
    if (isRecord(state)) {
      return state as T;
    }
  }
  return value as T;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

async function readJson<T>(filePath: string): Promise<T | undefined> {
  try {
    const content = await fs.readFile(filePath, "utf8");
    return JSON.parse(content) as T;
  } catch (error) {
    if (isMissingFile(error)) {
      return undefined;
    }

    throw error;
  }
}

async function writeJsonAtomic(filePath: string, value: unknown): Promise<void> {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  const tempPath = `${filePath}.${process.pid}.${Date.now()}.${Math.random().toString(36).slice(2)}.tmp`;
  await fs.writeFile(tempPath, `${JSON.stringify(value, null, 2)}\n`);
  await fs.rename(tempPath, filePath);
}

async function readJsonLines<T>(filePath: string, limit: number): Promise<T[]> {
  try {
    const content = await fs.readFile(filePath, "utf8");
    return parseJsonLines<T>(content, limit);
  } catch (error) {
    if (isMissingFile(error)) {
      return [];
    }

    throw error;
  }
}

function parseJsonLines<T>(content: string, limit: number): T[] {
  const lines = content.split("\n");
  const parsed: T[] = [];

  if (limit > 0) {
    for (let index = lines.length - 1; index >= 0 && parsed.length < limit; index -= 1) {
      const value = parseJsonLine<T>(lines[index]);
      if (value !== undefined) {
        parsed.unshift(value);
      }
    }
    return parsed;
  }

  for (const line of lines) {
    const value = parseJsonLine<T>(line);
    if (value !== undefined) {
      parsed.push(value);
    }
  }
  return parsed;
}

function parseJsonLine<T>(line: string): T | undefined {
  const trimmed = line.trim();
  if (!trimmed || trimmed.includes("\0")) {
    return undefined;
  }

  try {
    return JSON.parse(trimmed) as T;
  } catch {
    return undefined;
  }
}

async function appendJsonLine(filePath: string, value: unknown): Promise<void> {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  const line = `${JSON.stringify(value)}\n`;
  const retryDelaysMs = [10, 25, 50, 100, 200, 400, 800];
  for (let attempt = 0; ; attempt += 1) {
    try {
      await fs.appendFile(filePath, line);
      return;
    } catch (error) {
      if (!isTransientFileContention(error) || attempt >= retryDelaysMs.length) {
        throw error;
      }
      await new Promise((resolve) => setTimeout(resolve, retryDelaysMs[attempt]));
    }
  }
}

function isTransientFileContention(error: unknown): boolean {
  if (!(error instanceof Error) || !("code" in error)) return false;
  const code = (error as NodeJS.ErrnoException).code;
  return code === "EBUSY" || code === "EPERM" || code === "EACCES";
}

function isMissingFile(error: unknown): boolean {
  return (
    error instanceof Error &&
    "code" in error &&
    (error as NodeJS.ErrnoException).code === "ENOENT"
  );
}
