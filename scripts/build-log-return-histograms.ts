import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  readCandleShardReferenceSync,
  type SequentialCandle,
} from "@trading/storage";
import {
  aggregateLogReturns,
  returnsInWindow,
  type TimedReturn,
} from "./lib/log-return-distribution.js";
import {
  createLogReturnHistogramSpec,
  LogReturnHistogramCounter,
} from "./lib/log-return-histogram.js";

const DAY_MS = 86_400_000;
const SCALES = [
  { id: "1s", label: "1 second", intervalMs: 1_000 },
  { id: "1m", label: "1 minute", intervalMs: 60_000 },
  { id: "15m", label: "15 minutes", intervalMs: 15 * 60_000 },
  { id: "1h", label: "1 hour", intervalMs: 60 * 60_000 },
  { id: "4h", label: "4 hours", intervalMs: 4 * 60 * 60_000 },
  { id: "1d", label: "1 day", intervalMs: DAY_MS },
] as const;

interface Options {
  analysisPath: string;
  outputPath: string;
}

interface AnalysisScale {
  id: string;
  observations: number;
  standardDeviationBps: number;
  zeroFraction: number;
}

interface AnalysisWindow {
  id: string;
  label: string;
  startTime: string;
  endTime: string;
  scales: AnalysisScale[];
}

interface AnalysisReport {
  version: number;
  generatedAt: string;
  source: {
    symbol: string;
    commonAnalysisEndTime: string;
    oneSecond: { referenceDirectory: string };
    oneMinute: { referenceDirectory: string };
  };
  fullHistory: AnalysisWindow;
  trailingWindows: AnalysisWindow[];
}

interface WindowHistogram {
  id: string;
  label: string;
  startTime: string;
  endTime: string;
  zeroProbability: number;
  histogram: {
    observations: number;
    binWidthBps: number;
    lowerBps: number;
    upperBps: number;
    binCount: number;
    underflowProbability: number;
    overflowProbability: number;
    nonzeroBins: Array<[index: number, probability: number]>;
  };
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
  const windows = [analysis.fullHistory, ...analysis.trailingWindows];
  const counters = createCounters(analysis, windows);
  countOneSecondReturns(analysis, windows, counters);
  countSlowerReturns(analysis, windows, counters);

  const result = {
    version: 1,
    generatedAt: new Date().toISOString(),
    sourceAnalysis: path.relative(repoRoot, options.analysisPath).replaceAll("\\", "/"),
    sourceAnalysisGeneratedAt: analysis.generatedAt,
    symbol: analysis.source.symbol,
    commonEndTime: analysis.source.commonAnalysisEndTime,
    binning: {
      quantity: "empirical-probability-per-bin",
      binWidthFullHistorySigma: 0.1,
      binCentersFromSigma: -512,
      binCentersToSigma: 512,
      note: "Every scale uses one fixed native-bps bin grid across all windows. Bin width is 0.1 times that scale's full-history standard deviation, zero is the center of the middle bin, and only nonzero bins are serialized.",
    },
    scales: SCALES.map((scale) => ({
      id: scale.id,
      label: scale.label,
      intervalMs: scale.intervalMs,
      windows: windows.map((window) => {
        const summary = scaleSummary(window, scale.id);
        const histogram = counters.get(scale.id)!.get(window.id)!.finish();
        if (histogram.observations !== summary.observations) {
          throw new Error(
            `${scale.id}/${window.id}: histogram has ${histogram.observations} observations, expected ${summary.observations}.`,
          );
        }
        return {
          id: window.id,
          label: window.label,
          startTime: window.startTime,
          endTime: window.endTime,
          zeroProbability: summary.zeroFraction,
          histogram: {
            observations: histogram.observations,
            binWidthBps: histogram.binWidthBps,
            lowerBps: histogram.lowerBps,
            upperBps: histogram.upperBps,
            binCount: histogram.bins.length,
            underflowProbability: histogram.underflowProbability,
            overflowProbability: histogram.overflowProbability,
            nonzeroBins: histogram.bins.flatMap((bin, index) =>
              bin.probability > 0 ? [[index, bin.probability] as [number, number]] : []),
          },
        } satisfies WindowHistogram;
      }),
    })),
  };
  fs.mkdirSync(path.dirname(options.outputPath), { recursive: true });
  fs.writeFileSync(options.outputPath, `${JSON.stringify(result, null, 2)}\n`, "utf8");
  console.log(`Wrote ${path.relative(repoRoot, options.outputPath)}`);
}

function createCounters(
  analysis: AnalysisReport,
  windows: readonly AnalysisWindow[],
): Map<string, Map<string, LogReturnHistogramCounter>> {
  return new Map(SCALES.map((scale) => {
    const fullSigma = scaleSummary(analysis.fullHistory, scale.id).standardDeviationBps;
    const spec = createLogReturnHistogramSpec(fullSigma, 0.1, 512);
    return [
      scale.id,
      new Map(windows.map((window) => [
        window.id,
        new LogReturnHistogramCounter(spec),
      ])),
    ];
  }));
}

function countOneSecondReturns(
  analysis: AnalysisReport,
  windows: readonly AnalysisWindow[],
  counters: ReadonlyMap<string, ReadonlyMap<string, LogReturnHistogramCounter>>,
): void {
  const directory = path.resolve(repoRoot, analysis.source.oneSecond.referenceDirectory);
  const fullStart = Date.parse(analysis.fullHistory.startTime);
  const analysisEnd = Date.parse(analysis.fullHistory.endTime);
  const definitions = windows.map((window) => ({
    id: window.id,
    startTime: Date.parse(window.startTime),
    endTime: Date.parse(window.endTime),
  }));
  const files = fs.readdirSync(directory, { withFileTypes: true })
    .filter((entry) => entry.isFile() && /^\d{4}-\d{2}-\d{2}\.json$/.test(entry.name))
    .map((entry) => path.join(directory, entry.name))
    .sort();
  let previousClose = Number.NaN;
  let previousOpenTime = Number.NaN;
  for (const [index, file] of files.entries()) {
    const dayStart = Date.parse(`${path.basename(file, ".json")}T00:00:00.000Z`);
    const dayEnd = dayStart + DAY_MS;
    if (dayStart < fullStart || dayEnd > analysisEnd) continue;
    if (index % 100 === 0) console.error(`Histogramming one-second history ${index}/${files.length}...`);
    const active = definitions
      .filter((window) => dayStart >= window.startTime && dayEnd <= window.endTime)
      .map((window) => counters.get("1s")!.get(window.id)!);
    const candles = readCandleShardReferenceSync(file);
    if (candles.length !== 86_400) throw new Error(`${path.basename(file)} is incomplete.`);
    for (const [candleIndex, candle] of candles.entries()) {
      if (candle.openTime !== dayStart + candleIndex * 1_000) {
        throw new Error(`${path.basename(file)} has a one-second gap at ${candleIndex}.`);
      }
      if (previousOpenTime === candle.openTime - 1_000) {
        const value = Math.log(candle.close / previousClose);
        for (const counter of active) counter.addLogReturn(value);
      }
      previousClose = candle.close;
      previousOpenTime = candle.openTime;
    }
  }
}

function countSlowerReturns(
  analysis: AnalysisReport,
  windows: readonly AnalysisWindow[],
  counters: ReadonlyMap<string, ReadonlyMap<string, LogReturnHistogramCounter>>,
): void {
  const directory = path.resolve(repoRoot, analysis.source.oneMinute.referenceDirectory);
  const fullStart = Date.parse(analysis.fullHistory.startTime);
  const analysisEnd = Date.parse(analysis.fullHistory.endTime);
  const files = fs.readdirSync(directory, { withFileTypes: true })
    .filter((entry) => entry.isFile() && /^\d{4}-\d{2}-\d{2}\.json$/.test(entry.name))
    .map((entry) => ({
      file: path.join(directory, entry.name),
      dayStart: Date.parse(`${entry.name.slice(0, 10)}T00:00:00.000Z`),
    }))
    .filter((entry) => entry.dayStart >= fullStart - DAY_MS && entry.dayStart < analysisEnd)
    .sort((left, right) => left.dayStart - right.dayStart);
  const candles: SequentialCandle[] = [];
  for (const [index, entry] of files.entries()) {
    if (index % 250 === 0) console.error(`Loading minute history ${index}/${files.length}...`);
    candles.push(...readCandleShardReferenceSync(entry.file));
  }
  candles.sort((left, right) => left.openTime - right.openTime);
  for (const scale of SCALES.filter((item) => item.id !== "1s")) {
    console.error(`Histogramming ${scale.id} returns...`);
    const returns = aggregateLogReturns(candles, scale.intervalMs);
    for (const window of windows) {
      const values = returnsInWindow(
        returns,
        Date.parse(window.startTime),
        Date.parse(window.endTime),
      );
      const counter = counters.get(scale.id)!.get(window.id)!;
      for (const value of values) counter.addLogReturn(value);
    }
  }
}

function scaleSummary(window: AnalysisWindow, scaleId: string): AnalysisScale {
  const summary = window.scales.find((scale) => scale.id === scaleId);
  if (!summary) throw new Error(`Analysis window ${window.id} is missing scale ${scaleId}.`);
  return summary;
}

function validateAnalysis(analysis: AnalysisReport): void {
  if (analysis.version < 2
    || analysis.source.symbol !== "BTCUSDT"
    || !analysis.source.oneSecond?.referenceDirectory
    || !analysis.source.oneMinute?.referenceDirectory
    || !Array.isArray(analysis.trailingWindows)) {
    throw new Error("Log-return analysis report is missing the required 1s/1m sources.");
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
      values.get("output") ?? "data/benchmarks/log-return-histograms.json",
    ),
  };
}
