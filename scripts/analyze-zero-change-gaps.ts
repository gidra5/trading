import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { readCandleShardReferenceSync } from "@trading/storage";

const DAY_MS = 86_400_000;

interface AnalysisWindow {
  id: string;
  label: string;
  startTime: string;
  endTime: string;
}

interface AnalysisReport {
  version: number;
  source: {
    symbol: string;
    commonAnalysisEndTime: string;
    oneSecond: { referenceDirectory: string };
  };
  fullHistory: AnalysisWindow;
  trailingWindows: AnalysisWindow[];
}

interface GapState {
  definition: AnalysisWindow;
  returns: number;
  zeroReturns: number;
  nonzeroReturns: number;
  seenNonzero: boolean;
  currentRun: number;
  currentRunStartTime: number | null;
  leadingCensoredSeconds: number;
  counts: Map<number, number>;
  longestGaps: Array<{ seconds: number; startTime: number; endTime: number }>;
}

interface Options {
  analysisPath: string;
  outputPath: string;
}

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

try {
  main();
} catch (error: unknown) {
  console.error(error instanceof Error ? error.stack ?? error.message : String(error));
  process.exitCode = 1;
}

function main(): void {
  const options = parseOptions(process.argv.slice(2));
  const analysis = JSON.parse(fs.readFileSync(options.analysisPath, "utf8")) as AnalysisReport;
  validateAnalysis(analysis);
  const definitions = [analysis.fullHistory, ...analysis.trailingWindows]
    .sort((left, right) => windowRank(left.id) - windowRank(right.id));
  const states = definitions.map(createState);
  const directory = path.resolve(repoRoot, analysis.source.oneSecond.referenceDirectory);
  const fullStart = Date.parse(analysis.fullHistory.startTime);
  const analysisEnd = Date.parse(analysis.fullHistory.endTime);
  const files = fs.readdirSync(directory, { withFileTypes: true })
    .filter((entry) => entry.isFile() && /^\d{4}-\d{2}-\d{2}\.json$/.test(entry.name))
    .map((entry) => ({
      file: path.join(directory, entry.name),
      dayStart: Date.parse(`${entry.name.slice(0, 10)}T00:00:00.000Z`),
    }))
    .filter((entry) => entry.dayStart >= fullStart && entry.dayStart + DAY_MS <= analysisEnd)
    .sort((left, right) => left.dayStart - right.dayStart);
  if (files.length === 0) throw new Error("No one-second shards overlap the analysis window.");

  let previousClose = Number.NaN;
  let previousOpenTime = Number.NaN;
  for (const [fileIndex, entry] of files.entries()) {
    if (fileIndex > 0 && entry.dayStart !== files[fileIndex - 1]!.dayStart + DAY_MS) {
      throw new Error(`Missing one-second day before ${new Date(entry.dayStart).toISOString()}.`);
    }
    if (fileIndex % 100 === 0) {
      console.error(`Scanning one-second zero-change gaps ${fileIndex}/${files.length}...`);
    }
    const dayEnd = entry.dayStart + DAY_MS;
    const active = states.filter((state) => {
      const start = Date.parse(state.definition.startTime);
      const end = Date.parse(state.definition.endTime);
      return entry.dayStart >= start && dayEnd <= end;
    });
    const candles = readCandleShardReferenceSync(entry.file);
    if (candles.length !== 86_400) throw new Error(`${path.basename(entry.file)} is incomplete.`);
    for (let index = 0; index < candles.length; index += 1) {
      const candle = candles[index]!;
      if (candle.openTime !== entry.dayStart + index * 1_000
        || candle.closed === false
        || !Number.isFinite(candle.close)
        || candle.close <= 0) {
        throw new Error(`${path.basename(entry.file)} has an invalid candle at ${index}.`);
      }
      if (previousOpenTime === candle.openTime - 1_000) {
        const unchanged = candle.close === previousClose;
        for (const state of active) addReturn(state, unchanged, candle.openTime);
      }
      previousClose = candle.close;
      previousOpenTime = candle.openTime;
    }
  }

  const result = {
    version: 1,
    generatedAt: new Date().toISOString(),
    symbol: analysis.source.symbol,
    commonEndTime: analysis.source.commonAnalysisEndTime,
    definition: "A gap is a maximal positive run of adjacent one-second close-to-close returns equal to zero, bounded by nonzero returns inside the selected window. Boundary-censored runs are excluded from the gap distribution.",
    windows: states.map(finishState),
  };
  fs.mkdirSync(path.dirname(options.outputPath), { recursive: true });
  fs.writeFileSync(options.outputPath, `${JSON.stringify(result, null, 2)}\n`, "utf8");
  console.log(`Wrote ${path.relative(repoRoot, options.outputPath)}`);
  for (const window of result.windows) {
    console.log(
      `${window.id}: ${window.gaps.toLocaleString("en-US")} gaps, median ${window.quantiles.p50}s, p99 ${window.quantiles.p99}s, max ${window.maximumSeconds}s`,
    );
  }
}

function createState(definition: AnalysisWindow): GapState {
  return {
    definition,
    returns: 0,
    zeroReturns: 0,
    nonzeroReturns: 0,
    seenNonzero: false,
    currentRun: 0,
    currentRunStartTime: null,
    leadingCensoredSeconds: 0,
    counts: new Map(),
    longestGaps: [],
  };
}

function addReturn(state: GapState, unchanged: boolean, returnTime: number): void {
  state.returns += 1;
  if (unchanged) {
    state.zeroReturns += 1;
    if (state.seenNonzero) {
      if (state.currentRun === 0) state.currentRunStartTime = returnTime;
      state.currentRun += 1;
    }
    else state.leadingCensoredSeconds += 1;
    return;
  }
  state.nonzeroReturns += 1;
  if (state.seenNonzero && state.currentRun > 0) {
    state.counts.set(state.currentRun, (state.counts.get(state.currentRun) ?? 0) + 1);
    if (state.currentRunStartTime === null) throw new Error("Gap start time is missing.");
    if (state.longestGaps.length < 10
      || state.currentRun > state.longestGaps.at(-1)!.seconds) {
      state.longestGaps.push({
        seconds: state.currentRun,
        startTime: state.currentRunStartTime,
        endTime: returnTime,
      });
      state.longestGaps.sort((left, right) => right.seconds - left.seconds);
      state.longestGaps.length = Math.min(state.longestGaps.length, 10);
    }
  }
  state.seenNonzero = true;
  state.currentRun = 0;
  state.currentRunStartTime = null;
}

function finishState(state: GapState) {
  const entries = [...state.counts.entries()].sort((left, right) => left[0] - right[0]);
  const gaps = entries.reduce((sum, entry) => sum + entry[1], 0);
  const completedZeroSeconds = entries.reduce((sum, entry) => sum + entry[0] * entry[1], 0);
  if (gaps === 0) throw new Error(`Window ${state.definition.id} contains no completed gaps.`);
  return {
    id: state.definition.id,
    label: state.definition.label,
    startTime: state.definition.startTime,
    endTime: state.definition.endTime,
    returns: state.returns,
    zeroReturns: state.zeroReturns,
    nonzeroReturns: state.nonzeroReturns,
    zeroReturnProbability: state.zeroReturns / state.returns,
    gaps,
    completedZeroSeconds,
    leadingCensoredSeconds: state.leadingCensoredSeconds,
    trailingCensoredSeconds: state.currentRun,
    meanSeconds: completedZeroSeconds / gaps,
    quantiles: {
      p50: weightedQuantile(entries, gaps, 0.5),
      p90: weightedQuantile(entries, gaps, 0.9),
      p95: weightedQuantile(entries, gaps, 0.95),
      p99: weightedQuantile(entries, gaps, 0.99),
      p999: weightedQuantile(entries, gaps, 0.999),
    },
    maximumSeconds: entries.at(-1)![0],
    longestGaps: state.longestGaps.map((gap) => ({
      seconds: gap.seconds,
      startTime: new Date(gap.startTime).toISOString(),
      endTime: new Date(gap.endTime).toISOString(),
    })),
    tailProbability: {
      atLeast2s: probabilityAtLeast(entries, gaps, 2),
      atLeast5s: probabilityAtLeast(entries, gaps, 5),
      atLeast10s: probabilityAtLeast(entries, gaps, 10),
      atLeast30s: probabilityAtLeast(entries, gaps, 30),
      atLeast60s: probabilityAtLeast(entries, gaps, 60),
    },
    pmf: entries.map(([seconds, count]) => [seconds, count, count / gaps]),
  };
}

function weightedQuantile(entries: Array<[number, number]>, total: number, probability: number): number {
  const target = Math.max(1, Math.ceil(total * probability));
  let cumulative = 0;
  for (const [value, count] of entries) {
    cumulative += count;
    if (cumulative >= target) return value;
  }
  return entries.at(-1)![0];
}

function probabilityAtLeast(entries: Array<[number, number]>, total: number, threshold: number): number {
  return entries.reduce((sum, [value, count]) => sum + (value >= threshold ? count : 0), 0) / total;
}

function validateAnalysis(analysis: AnalysisReport): void {
  if (analysis.version < 2
    || analysis.source.symbol !== "BTCUSDT"
    || !analysis.source.oneSecond?.referenceDirectory
    || !Array.isArray(analysis.trailingWindows)) {
    throw new Error("Input is not the expected BTCUSDT log-return analysis.");
  }
}

function parseOptions(args: string[]): Options {
  const values = new Map<string, string>();
  for (let index = 0; index < args.length; index += 1) {
    const key = args[index];
    const value = args[index + 1];
    if (!key?.startsWith("--") || !value || value.startsWith("--")) {
      throw new Error(`Invalid argument near ${key ?? "end"}.`);
    }
    values.set(key.slice(2), value);
    index += 1;
  }
  return {
    analysisPath: path.resolve(
      repoRoot,
      values.get("analysis") ?? "data/benchmarks/log-return-distributions.json",
    ),
    outputPath: path.resolve(
      repoRoot,
      values.get("output") ?? "data/benchmarks/zero-change-gaps.json",
    ),
  };
}

function windowRank(id: string): number {
  return ({ full: 0, "365d": 1, "90d": 2, "30d": 3, "7d": 4 } as Record<string, number>)[id]
    ?? Number.MAX_SAFE_INTEGER;
}
