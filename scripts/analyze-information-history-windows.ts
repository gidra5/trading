import fs from "node:fs";
import path from "node:path";
import { deserialize } from "node:v8";
import { fileURLToPath, pathToFileURL } from "node:url";
import { readCandleShardReferenceSync } from "@trading/storage";
import {
  buildSignalDefinitions,
  IndicatorEngine,
} from "./analyze-technical-indicator-predictiveness.ts";
import {
  buildMarketFeatureDefinitions,
  CausalMarketFeatureEngine,
} from "./analyze-volume-multiscale-information.ts";

const DAY_MS = 86_400_000;
const HISTORY_STATES = 33;
const FEATURE_BINS = 4;
const BASIS_DIMENSIONS = 5;
const BASIS_STATES = FEATURE_BINS ** BASIS_DIMENSIONS;
const MAGNITUDE_BINS = 16;
const TARGET_CLASSES = 33;
const JOINT_LENGTH = HISTORY_STATES * BASIS_STATES * TARGET_CLASSES;
const CONTEXT_LENGTH = HISTORY_STATES * BASIS_STATES;
const HISTORY_LENGTH = HISTORY_STATES * TARGET_CLASSES;
const EVALUATION_STRIDE = 4;
const WARMUP_RETURNS = 12_000;
const SMOOTHING = 0.5;
const COMMON_HISTORY_DAYS = 730;
const WINDOW_DAYS = [7, 30, 90, 180, 365, 730] as const;
const PRICE_BASIS_IDS = ["rsi-2", "ema-acceleration-2-1", "ema-slope-8-8"] as const;
const MARKET_BASIS_IDS = ["1h-log-volume", "1s-range"] as const;
const DEFAULT_ANALYSIS = "data/benchmarks/log-return-distributions.json";
const DEFAULT_PRICE_CALIBRATION = "data/runtime-cache/technical-indicator-calibration.bin";
const DEFAULT_MARKET_CALIBRATION = "data/runtime-cache/volume-multiscale-calibration.bin";
const DEFAULT_OUTPUT = "data/benchmarks/information-history-windows.json";
const DEFAULT_REPORT = "docs/experiments/information-history-windows-2026-08-16.md";

interface AnalysisReport {
  generatedAt: string;
  source: { symbol: string; oneSecond: { referenceDirectory: string } };
  fullHistory: { startTime: string; endTime: string };
}

interface Calibration {
  featureEdges: number[][];
  magnitudeEdges: number[];
}

interface SparseDay {
  indices: Uint32Array;
  counts: Uint32Array;
  observations: number;
}

interface Score {
  observations: number;
  modelLogLossBits: number;
  historyLogLossBits: number;
  informationGainBits: number;
  unseenContextObservations: number;
}

interface HistoryWindow {
  id: string;
  label: string;
  days: number | null;
  joint: Float64Array;
  contextTotals: Float64Array;
  history: Float64Array;
  historyTotals: Float64Array;
  queue: SparseDay[];
  overall: Score;
  annual: Map<number, Score>;
}

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function main(): void {
  const argumentsMap = parseArguments(process.argv.slice(2));
  const analysis = readJson<AnalysisReport>(resolve(
    argumentsMap.get("analysis") ?? DEFAULT_ANALYSIS,
  ));
  const priceCalibration = (deserialize(fs.readFileSync(resolve(
    argumentsMap.get("price-calibration") ?? DEFAULT_PRICE_CALIBRATION,
  ))) as { calibration: Calibration }).calibration;
  const marketEdges = (deserialize(fs.readFileSync(resolve(
    argumentsMap.get("market-calibration") ?? DEFAULT_MARKET_CALIBRATION,
  ))) as { marketEdges: number[][] }).marketEdges;
  const outputPath = resolve(argumentsMap.get("output") ?? DEFAULT_OUTPUT);
  const reportPath = resolve(argumentsMap.get("report") ?? DEFAULT_REPORT);
  const priceDefinitionsAll = buildSignalDefinitions();
  const priceDefinitionIndex = new Map(
    priceDefinitionsAll.map((definition, index) => [definition.id, index]),
  );
  const priceDefinitions = PRICE_BASIS_IDS.map((id) => priceDefinitionsAll[
    priceDefinitionIndex.get(id)!
  ]!);
  const priceEdges = PRICE_BASIS_IDS.map((id) => quartileEdges(
    priceCalibration.featureEdges[priceDefinitionIndex.get(id)!]!,
  ));
  const marketDefinitions = buildMarketFeatureDefinitions();
  const marketDefinitionById = new Map(marketDefinitions.map((definition) => [
    definition.id,
    definition,
  ]));
  const marketIndices = MARKET_BASIS_IDS.map((id) => marketDefinitionById.get(id)!.sourceIndex);
  const selectedMarketEdges = marketIndices.map((index) => marketEdges[index]!);
  const files = selectedFiles(
    path.resolve(repoRoot, analysis.source.oneSecond.referenceDirectory),
    Date.parse(analysis.fullHistory.startTime),
    Date.parse(analysis.fullHistory.endTime),
  );
  const windows: HistoryWindow[] = [
    ...WINDOW_DAYS.map((days) => createWindow(`${days}d`, `${days} trailing days`, days)),
    createWindow("expanding", "all available prior history", null),
  ];
  const priceValues = new Float64Array(priceDefinitions.length);
  const marketEngine = new CausalMarketFeatureEngine();
  let priceEngine: IndicatorEngine | undefined;
  let previousClose = Number.NaN;
  let previousReturnState = 0;
  let returnsSeen = 0;
  let daysSeen = 0;
  const start = Date.parse(analysis.fullHistory.startTime);
  for (const [fileIndex, entry] of files.entries()) {
    if (fileIndex % 50 === 0) console.error(`History windows ${fileIndex}/${files.length}...`);
    const candles = readCandleShardReferenceSync(entry.file);
    if (candles.length !== 86_400) throw new Error(`${path.basename(entry.file)} is incomplete.`);
    const counts = new Map<number, number>();
    let observations = 0;
    for (const candle of candles) {
      if (!priceEngine) {
        priceEngine = new IndicatorEngine(priceDefinitions, candle.close);
        marketEngine.update(candle);
      } else {
        const returnBps = candle.close === previousClose
          ? 0
          : Math.log(candle.close / previousClose) * 10_000;
        const active = returnBps !== 0;
        const sign = returnBps > 0 ? 1 : 0;
        const magnitudeBin = active
          ? upperBound(priceCalibration.magnitudeEdges, Math.abs(returnBps))
          : -1;
        const targetClass = active ? 1 + sign * MAGNITUDE_BINS + magnitudeBin : 0;
        if (returnsSeen >= WARMUP_RETURNS && isEvaluationTarget(returnsSeen)) {
          const marketValid = marketEngine.valid();
          if (marketIndices.every((index) => marketValid[index] === 1)) {
            priceEngine.values(priceValues);
            let basisState = 0;
            for (let index = 0; index < priceEdges.length; index += 1) {
              basisState = basisState * FEATURE_BINS
                + upperBound(priceEdges[index]!, priceValues[index]!);
            }
            const marketValues = marketEngine.values();
            for (let index = 0; index < marketIndices.length; index += 1) {
              basisState = basisState * FEATURE_BINS + upperBound(
                selectedMarketEdges[index]!,
                marketValues[marketIndices[index]!]!,
              );
            }
            const jointIndex = (
              previousReturnState * BASIS_STATES + basisState
            ) * TARGET_CLASSES + targetClass;
            counts.set(jointIndex, (counts.get(jointIndex) ?? 0) + 1);
            observations += 1;
          }
        }
        previousReturnState = active ? 1 + sign * MAGNITUDE_BINS + magnitudeBin : 0;
        priceEngine.update(candle.close);
        marketEngine.update(candle);
        returnsSeen += 1;
      }
      previousClose = candle.close;
    }
    const day = sparseDay(counts, observations);
    if (daysSeen >= COMMON_HISTORY_DAYS) {
      const epoch = anniversaryIndex(start, entry.dayStart);
      for (const window of windows) scoreDay(window, day, epoch);
    }
    for (const window of windows) updateWindow(window, day);
    daysSeen += 1;
  }
  const results = windows.map((window) => result(window));
  const ranked = [...results].sort((left, right) =>
    left.modelLogLossBits - right.modelLogLossBits);
  const artifact = {
    version: 1,
    generatedAt: new Date().toISOString(),
    symbol: analysis.source.symbol,
    target: "next 1s return, 33 cells",
    basis: [...PRICE_BASIS_IDS, ...MARKET_BASIS_IDS],
    evaluation: {
      commonStartTime: new Date(start + COMMON_HISTORY_DAYS * DAY_MS).toISOString(),
      endTime: analysis.fullHistory.endTime,
      targetStride: EVALUATION_STRIDE,
      fitting: "For each UTC test day, counts use only preceding complete days in the specified trailing window.",
      smoothing: SMOOTHING,
      fixedQuantization: "Price and market quartile edges remain fixed from the original year-1 calibration.",
    },
    bestWindow: ranked[0],
    results,
    limitations: [
      "This isolates the history used for probability-table estimation; it does not refit the indicator or quantization transforms for every window.",
      "The exact five-coordinate quartile table is deliberately high-dimensional, so short histories expose the real sparsity cost of this representation.",
      "A hierarchical-backoff or learned continuous model can use short recent windows more efficiently than exact cells.",
    ],
  };
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, `${JSON.stringify(artifact, null, 2)}\n`, "utf8");
  fs.mkdirSync(path.dirname(reportPath), { recursive: true });
  fs.writeFileSync(reportPath, renderReport(artifact), "utf8");
  console.log(`Wrote ${path.relative(repoRoot, outputPath)}`);
  console.log(`Wrote ${path.relative(repoRoot, reportPath)}`);
}

function createWindow(id: string, label: string, days: number | null): HistoryWindow {
  return {
    id,
    label,
    days,
    joint: new Float64Array(JOINT_LENGTH),
    contextTotals: new Float64Array(CONTEXT_LENGTH),
    history: new Float64Array(HISTORY_LENGTH),
    historyTotals: new Float64Array(HISTORY_STATES),
    queue: [],
    overall: emptyScore(),
    annual: new Map(),
  };
}

function sparseDay(counts: Map<number, number>, observations: number): SparseDay {
  const entries = [...counts.entries()].sort((left, right) => left[0] - right[0]);
  return {
    indices: Uint32Array.from(entries, (entry) => entry[0]),
    counts: Uint32Array.from(entries, (entry) => entry[1]),
    observations,
  };
}

function scoreDay(window: HistoryWindow, day: SparseDay, epoch: number): void {
  const annual = window.annual.get(epoch) ?? emptyScore();
  for (let item = 0; item < day.indices.length; item += 1) {
    const jointIndex = day.indices[item]!;
    const count = day.counts[item]!;
    const context = Math.floor(jointIndex / TARGET_CLASSES);
    const target = jointIndex % TARGET_CLASSES;
    const historyState = Math.floor(context / BASIS_STATES);
    const modelProbability = (
      window.joint[jointIndex]! + SMOOTHING
    ) / (
      window.contextTotals[context]! + TARGET_CLASSES * SMOOTHING
    );
    const historyIndex = historyState * TARGET_CLASSES + target;
    const historyProbability = (
      window.history[historyIndex]! + SMOOTHING
    ) / (
      window.historyTotals[historyState]! + TARGET_CLASSES * SMOOTHING
    );
    const modelLoss = -Math.log2(modelProbability);
    const historyLoss = -Math.log2(historyProbability);
    for (const score of [window.overall, annual]) {
      score.observations += count;
      score.modelLogLossBits += count * modelLoss;
      score.historyLogLossBits += count * historyLoss;
      score.informationGainBits += count * Math.log2(modelProbability / historyProbability);
      if (window.contextTotals[context] === 0) score.unseenContextObservations += count;
    }
  }
  window.annual.set(epoch, annual);
}

function updateWindow(window: HistoryWindow, day: SparseDay): void {
  addDay(window, day, 1);
  if (window.days !== null) {
    window.queue.push(day);
    if (window.queue.length > window.days) addDay(window, window.queue.shift()!, -1);
  }
}

function addDay(window: HistoryWindow, day: SparseDay, direction: 1 | -1): void {
  for (let item = 0; item < day.indices.length; item += 1) {
    const jointIndex = day.indices[item]!;
    const delta = direction * day.counts[item]!;
    const context = Math.floor(jointIndex / TARGET_CLASSES);
    const target = jointIndex % TARGET_CLASSES;
    const historyState = Math.floor(context / BASIS_STATES);
    window.joint[jointIndex] += delta;
    window.contextTotals[context] += delta;
    window.history[historyState * TARGET_CLASSES + target] += delta;
    window.historyTotals[historyState] += delta;
  }
}

function emptyScore(): Score {
  return {
    observations: 0,
    modelLogLossBits: 0,
    historyLogLossBits: 0,
    informationGainBits: 0,
    unseenContextObservations: 0,
  };
}

function result(window: HistoryWindow) {
  return {
    id: window.id,
    label: window.label,
    days: window.days,
    ...normalizedScore(window.overall),
    annual: [...window.annual.entries()].map(([epoch, score]) => ({
      epoch,
      ...normalizedScore(score),
    })),
  };
}

function normalizedScore(score: Score) {
  return {
    observations: score.observations,
    modelLogLossBits: score.modelLogLossBits / score.observations,
    historyLogLossBits: score.historyLogLossBits / score.observations,
    informationGainBits: score.informationGainBits / score.observations,
    geometricProbabilityImprovement: 2 ** (score.informationGainBits / score.observations) - 1,
    unseenContextFraction: score.unseenContextObservations / score.observations,
  };
}

function renderReport(artifact: any): string {
  const best = artifact.bestWindow;
  const lines = [
    "# Historical-window sensitivity of the next-1s distribution model",
    "",
    `Generated ${artifact.generatedAt} for ${artifact.symbol}. All windows score exactly the same future targets from ${artifact.evaluation.commonStartTime} through ${artifact.evaluation.endTime}.`,
    "",
    "## Result",
    "",
    `The lowest held-out log loss uses **${best.label}**: ${metric(best.modelLogLossBits)} bits/target. It retains ${metric(best.informationGainBits)} bits/target beyond a previous-return-only model and encounters an unseen five-feature context for ${percent(best.unseenContextFraction)} of targets.`,
    "",
    "| training history | model log loss | history-only log loss | gain beyond history | geometric probability gain | unseen context targets |",
    "|---|---:|---:|---:|---:|---:|",
  ];
  for (const row of artifact.results) {
    lines.push(`| ${row.label} | ${metric(row.modelLogLossBits)} | ${metric(row.historyLogLossBits)} | ${metric(row.informationGainBits)} | ${percent(row.geometricProbabilityImprovement)} | ${percent(row.unseenContextFraction)} |`);
  }
  lines.push(
    "",
    "Lower log loss is better. The information-gain column compares the complete five-feature table against a previous-return-only table trained on the same amount of history, so it does not reward a longer window merely for estimating the unconditional distribution more accurately.",
    "",
    "## By future epoch",
    "",
  );
  const epochs = artifact.results[0].annual.map((row: any) => row.epoch);
  for (const epoch of epochs) {
    lines.push(
      `### Epoch ${epoch}`,
      "",
      "| history | log loss | gain beyond history | unseen contexts |",
      "|---|---:|---:|---:|",
    );
    for (const row of artifact.results) {
      const annual = row.annual.find((entry: any) => entry.epoch === epoch);
      lines.push(`| ${row.label} | ${metric(annual.modelLogLossBits)} | ${metric(annual.informationGainBits)} | ${percent(annual.unseenContextFraction)} |`);
    }
    lines.push("");
  }
  lines.push(
    "## Method limits",
    "",
    ...artifact.limitations.map((limitation: string) => `- ${limitation}`),
    "",
    "Complete values are stored in `data/benchmarks/information-history-windows.json`.",
  );
  return `${lines.join("\n")}\n`;
}

function quartileEdges(edges: number[]): number[] {
  if (edges.length !== 15) throw new Error("Expected 15 calibration edges.");
  return [edges[3]!, edges[7]!, edges[11]!];
}

function upperBound(sorted: number[], value: number): number {
  let low = 0;
  let high = sorted.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (value < sorted[middle]!) high = middle;
    else low = middle + 1;
  }
  return low;
}

function isEvaluationTarget(index: number): boolean {
  let mixed = index ^ (index >>> 16);
  mixed = Math.imul(mixed, 0x45d9f3b);
  mixed ^= mixed >>> 16;
  return (mixed >>> 0) % EVALUATION_STRIDE === 0;
}

function selectedFiles(directory: string, start: number, end: number) {
  return fs.readdirSync(directory, { withFileTypes: true })
    .filter((entry) => entry.isFile() && /^\d{4}-\d{2}-\d{2}\.json$/.test(entry.name))
    .map((entry) => ({
      file: path.join(directory, entry.name),
      dayStart: Date.parse(`${entry.name.slice(0, 10)}T00:00:00.000Z`),
    }))
    .filter((entry) => entry.dayStart >= start && entry.dayStart < end)
    .sort((left, right) => left.dayStart - right.dayStart);
}

function anniversary(startMs: number, years: number): number {
  const start = new Date(startMs);
  return Date.UTC(start.getUTCFullYear() + years, start.getUTCMonth(), start.getUTCDate());
}

function anniversaryIndex(startMs: number, dayMs: number): number {
  const start = new Date(startMs);
  const day = new Date(dayMs);
  let index = day.getUTCFullYear() - start.getUTCFullYear();
  if (dayMs < anniversary(startMs, index)) index -= 1;
  return Math.max(0, index);
}

function parseArguments(args: string[]): Map<string, string> {
  const values = new Map<string, string>();
  for (let index = 0; index < args.length; index += 2) {
    const key = args[index];
    const value = args[index + 1];
    if (!key?.startsWith("--") || !value) throw new Error(`Invalid argument near ${key}.`);
    values.set(key.slice(2), value);
  }
  return values;
}

function resolve(file: string): string {
  return path.resolve(repoRoot, file);
}

function readJson<T>(file: string): T {
  return JSON.parse(fs.readFileSync(file, "utf8")) as T;
}

function metric(value: number): string {
  return value === 0 ? "0" : value.toPrecision(8).replace(/\.0+(?=e|$)/, "");
}

function percent(value: number): string {
  return `${(value * 100).toFixed(3)}%`;
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  try {
    main();
  } catch (error: unknown) {
    console.error(error instanceof Error ? error.stack ?? error.message : String(error));
    process.exitCode = 1;
  }
}
