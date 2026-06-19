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
    return path.join(this.dataDir, "market");
  }

  private get stateDir(): string {
    return path.join(this.dataDir, "state");
  }

  private get backtestDir(): string {
    return path.join(this.dataDir, "backtests");
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
    return path.join(this.stateDir, `paper-bot-${this.symbol.toLowerCase()}.json`);
  }
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
  const tempPath = `${filePath}.tmp`;
  await fs.writeFile(tempPath, `${JSON.stringify(value, null, 2)}\n`);
  await fs.rename(tempPath, filePath);
}

async function readJsonLines<T>(filePath: string, limit: number): Promise<T[]> {
  try {
    const content = await fs.readFile(filePath, "utf8");
    const lines = content.trim().split("\n").filter(Boolean);
    const selectedLines = limit > 0 ? lines.slice(-limit) : lines;
    return selectedLines.map((line) => JSON.parse(line) as T);
  } catch (error) {
    if (isMissingFile(error)) {
      return [];
    }

    throw error;
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
