import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";
import readline from "node:readline";
import { createGunzip } from "node:zlib";

const MINUTE_MS = 60_000;
const SCALE_ORDER = ["1d", "4h", "1h", "15m", "1m"] as const;

interface SleeveMetadata {
  complete: true;
  scale: {
    id: string;
    label: string;
    minutes: number;
    sleeve_weight?: number;
    sleeveWeight?: number;
  };
  events: number;
  meanBasisSize: number;
  meanEligibleAssets: number;
  targetReachedRatio: number;
  meanExposure: number;
}

interface SleeveProgress {
  complete: false;
  events: number;
  completedEvents: number;
  batchSize: number;
}

export interface PortfolioIndexPoint {
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

export interface PortfolioIndexOverview {
  status: "missing" | "building" | "processing" | "complete";
  progress?: {
    scale: string;
    completedEvents: number;
    totalEvents: number;
    ratio: number;
    batchSize: number;
    updatedAt: string;
  };
  completedScales: SleeveMetadata[];
  report?: Record<string, unknown>;
  artifacts: {
    report?: string;
    candles?: string;
  };
}

export interface PortfolioIndexSeries {
  source: string;
  sourceUpdatedAt: string;
  from: number;
  to: number;
  sourceMinutes: number;
  sampledPoints: number;
  points: PortfolioIndexPoint[];
}

export class PortfolioIndexResultsReader {
  private readonly historyDir: string;
  private readonly sleevesDir: string;
  private readonly seriesCache = new Map<string, Promise<PortfolioIndexSeries>>();

  constructor(private readonly root: string) {
    this.historyDir = path.join(root, "index-history");
    this.sleevesDir = path.join(root, "walk-forward-index", "sleeves");
  }

  async overview(): Promise<PortfolioIndexOverview> {
    const reportFile = path.join(this.historyDir, "latest.json");
    const candlesFile = path.join(this.historyDir, "latest.candles.csv.gz");
    const [progress, completedScales, report] = await Promise.all([
      this.latestProgress(),
      this.completedScaleMetadata(),
      readJsonIfExists<Record<string, unknown>>(reportFile),
    ]);
    const hasCandles = await exists(candlesFile);
    const status = progress
      ? "building"
      : report && hasCandles
        ? "complete"
        : completedScales.some((scale) => scale.scale.id === "1m")
          ? "processing"
          : "missing";

    return {
      status,
      ...(progress ? { progress } : {}),
      completedScales,
      ...(report ? { report } : {}),
      artifacts: {
        ...(report ? { report: reportFile } : {}),
        ...(hasCandles ? { candles: candlesFile } : {}),
      },
    };
  }

  async series(input: {
    from?: number;
    to?: number;
    maxPoints?: number;
  }): Promise<PortfolioIndexSeries> {
    const file = path.join(this.historyDir, "latest.candles.csv.gz");
    const stat = await fsp.stat(file);
    const report = await readJsonIfExists<Record<string, unknown>>(
      path.join(this.historyDir, "latest.json"),
    );
    const window = recordValue(report?.window);
    const defaultFrom = Date.parse(stringValue(window?.start) ?? "");
    const defaultEndDay = Date.parse(stringValue(window?.end) ?? "");
    const defaultTo = Number.isFinite(defaultEndDay)
      ? defaultEndDay + 24 * 60 * MINUTE_MS
      : Date.now();
    const from = finiteTimestamp(input.from) ?? defaultFrom;
    const to = finiteTimestamp(input.to) ?? defaultTo;
    if (!Number.isFinite(from) || !Number.isFinite(to) || from >= to) {
      throw new Error("A valid from/to time range is required.");
    }
    const maxPoints = clampInteger(input.maxPoints ?? 2_000, 100, 5_000);
    const key = `${stat.mtimeMs}:${from}:${to}:${maxPoints}`;
    const cached = this.seriesCache.get(key);
    if (cached) return cached;

    const result = this.readSeriesFile(file, stat.mtime.toISOString(), from, to, maxPoints);
    this.seriesCache.clear();
    this.seriesCache.set(key, result);
    return result;
  }

  private async latestProgress(): Promise<
    PortfolioIndexOverview["progress"] | undefined
  > {
    const entries = await readdirIfExists(this.sleevesDir);
    const candidates: Array<{
      progress: SleeveProgress;
      updatedAt: string;
      scale: string;
    }> = [];
    for (const entry of entries) {
      if (!entry.isDirectory()) continue;
      const file = path.join(this.sleevesDir, entry.name, "progress.json");
      const progress = await readJsonIfExists<SleeveProgress>(file);
      if (!progress || progress.complete !== false) continue;
      const stat = await fsp.stat(file);
      const scale = /-(1d|4h|1h|15m|1m)-/.exec(entry.name)?.[1] ?? "unknown";
      candidates.push({ progress, updatedAt: stat.mtime.toISOString(), scale });
    }
    const latest = candidates.sort((left, right) =>
      right.updatedAt.localeCompare(left.updatedAt),
    )[0];
    if (!latest) return undefined;
    return {
      scale: latest.scale,
      completedEvents: latest.progress.completedEvents,
      totalEvents: latest.progress.events,
      ratio:
        latest.progress.events > 0
          ? latest.progress.completedEvents / latest.progress.events
          : 0,
      batchSize: latest.progress.batchSize,
      updatedAt: latest.updatedAt,
    };
  }

  private async completedScaleMetadata(): Promise<SleeveMetadata[]> {
    const entries = await readdirIfExists(this.sleevesDir);
    const latest = new Map<string, { metadata: SleeveMetadata; updatedAt: number }>();
    for (const entry of entries) {
      if (!entry.isDirectory()) continue;
      const file = path.join(this.sleevesDir, entry.name, "metadata.json");
      const metadata = await readJsonIfExists<SleeveMetadata>(file);
      if (!metadata || metadata.complete !== true || !metadata.scale?.id) continue;
      const stat = await fsp.stat(file);
      const existing = latest.get(metadata.scale.id);
      if (!existing || stat.mtimeMs > existing.updatedAt) {
        latest.set(metadata.scale.id, { metadata, updatedAt: stat.mtimeMs });
      }
    }
    return SCALE_ORDER.flatMap((scale) => {
      const value = latest.get(scale);
      return value ? [value.metadata] : [];
    });
  }

  private async readSeriesFile(
    file: string,
    updatedAt: string,
    from: number,
    to: number,
    maxPoints: number,
  ): Promise<PortfolioIndexSeries> {
    const estimatedMinutes = Math.max(1, Math.ceil((to - from) / MINUTE_MS));
    const stride = Math.max(1, Math.ceil(estimatedMinutes / maxPoints));
    const input = fs.createReadStream(file).pipe(createGunzip());
    const lines = readline.createInterface({ input, crlfDelay: Infinity });
    let header: string[] | undefined;
    let indexes: ReturnType<typeof candleColumns> | undefined;
    let qualifying = 0;
    let sourceMinutes = 0;
    let last: PortfolioIndexPoint | undefined;
    const points: PortfolioIndexPoint[] = [];
    for await (const line of lines) {
      if (!header) {
        header = line.split(",");
        indexes = candleColumns(header);
        continue;
      }
      const columns = line.split(",");
      const time = Date.parse(columns[indexes!.openTime] ?? "");
      if (!Number.isFinite(time) || time < from || time >= to) continue;
      const point = parsePoint(indexes!, columns, time);
      if (qualifying % stride === 0) points.push(point);
      qualifying += 1;
      sourceMinutes += 1;
      last = point;
    }
    if (last && points.at(-1)?.time !== last.time) points.push(last);
    return {
      source: file,
      sourceUpdatedAt: updatedAt,
      from,
      to,
      sourceMinutes,
      sampledPoints: points.length,
      points,
    };
  }
}

function parsePoint(
  indexes: ReturnType<typeof candleColumns>,
  columns: string[],
  time: number,
): PortfolioIndexPoint {
  const value = (index: number) => {
    const parsed = Number(columns[index]);
    return Number.isFinite(parsed) ? parsed : 0;
  };
  return {
    time,
    open: value(indexes.open),
    high: value(indexes.high),
    low: value(indexes.low),
    baseline: value(indexes.baseline),
    gross: value(indexes.gross),
    feeOnly: value(indexes.feeOnly),
    conservative: value(indexes.conservative),
    exposure: value(indexes.exposure),
    cash: value(indexes.cash),
    activeConstituents: value(indexes.activeConstituents),
    turnover: value(indexes.turnover),
    transactionCost: value(indexes.transactionCost),
    fundingCashflow: value(indexes.fundingCashflow),
  };
}

function candleColumns(header: string[]) {
  return {
    openTime: column(header, "open_time"),
    open: column(header, "open"),
    high: column(header, "high"),
    low: column(header, "low"),
    baseline: column(header, "close"),
    gross: column(header, "gross_close"),
    feeOnly: column(header, "fee_only_close"),
    conservative: column(header, "conservative_close"),
    exposure: column(header, "target_exposure"),
    cash: column(header, "cash_weight"),
    activeConstituents: column(header, "active_constituents"),
    turnover: column(header, "gross_traded_notional_ratio"),
    transactionCost: column(header, "baseline_transaction_cost"),
    fundingCashflow: column(header, "baseline_funding_cashflow"),
  };
}

function column(header: string[], name: string): number {
  const index = header.indexOf(name);
  if (index < 0) throw new Error(`Index candle file is missing ${name}.`);
  return index;
}

async function exists(file: string): Promise<boolean> {
  try {
    await fsp.access(file, fs.constants.R_OK);
    return true;
  } catch {
    return false;
  }
}

async function readdirIfExists(directory: string): Promise<fs.Dirent[]> {
  try {
    return await fsp.readdir(directory, { withFileTypes: true });
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return [];
    throw error;
  }
}

async function readJsonIfExists<T>(file: string): Promise<T | undefined> {
  try {
    return JSON.parse(await fsp.readFile(file, "utf8")) as T;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return undefined;
    throw error;
  }
}

function finiteTimestamp(value: number | undefined): number | undefined {
  return value !== undefined && Number.isFinite(value) ? value : undefined;
}

function clampInteger(value: number, minimum: number, maximum: number): number {
  if (!Number.isFinite(value)) return minimum;
  return Math.max(minimum, Math.min(maximum, Math.trunc(value)));
}

function recordValue(value: unknown): Record<string, unknown> | undefined {
  return value && typeof value === "object"
    ? (value as Record<string, unknown>)
    : undefined;
}

function stringValue(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}
