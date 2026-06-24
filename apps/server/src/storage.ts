import fs from "node:fs/promises";
import path from "node:path";
import type {
  BacktestResult,
  Candle,
  OrderBookSnapshot,
  PaperBotState,
} from "@trading/bot-algo";

export class TradingStorage {
  constructor(
    private readonly dataDir: string,
    private readonly marketKey: string,
    private readonly symbol: string,
    private readonly interval: string,
  ) {}

  async ensureReady(): Promise<void> {
    await Promise.all([
      fs.mkdir(this.marketDir, { recursive: true }),
      fs.mkdir(this.stateDir, { recursive: true }),
      fs.mkdir(this.backtestDir, { recursive: true }),
    ]);
  }

  async loadBotState(): Promise<PaperBotState | undefined> {
    return readJson<PaperBotState>(this.botStatePath);
  }

  async saveBotState(state: PaperBotState): Promise<void> {
    await writeJsonAtomic(this.botStatePath, state);
  }

  async loadCandles(limit = 500): Promise<Candle[]> {
    return readJsonLines<Candle>(this.candlePath, limit);
  }

  async appendCandle(candle: Candle): Promise<void> {
    await appendJsonLine(this.candlePath, candle);
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
    return path.join(this.dataDir, "market", safePathPart(this.marketKey));
  }

  private get stateDir(): string {
    return path.join(this.dataDir, "state");
  }

  private get backtestDir(): string {
    return path.join(this.dataDir, "backtests", safePathPart(this.marketKey));
  }

  private get candlePath(): string {
    return path.join(
      this.marketDir,
      `${this.symbol.toLowerCase()}-${this.interval}-candles.jsonl`,
    );
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
}

function safePathPart(value: string): string {
  return value.replace(/[^a-z0-9_-]+/gi, "-").replace(/^-+|-+$/g, "").toLowerCase();
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
  await fs.appendFile(filePath, `${JSON.stringify(value)}\n`);
}

function isMissingFile(error: unknown): boolean {
  return (
    error instanceof Error &&
    "code" in error &&
    (error as NodeJS.ErrnoException).code === "ENOENT"
  );
}
