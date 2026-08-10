import fs from "node:fs";
import path from "node:path";
import {
  kronosInspectorWindows,
  parseKronosForecastArtifact,
  validateDenseForecastCoverage,
} from "./backtest-kronos.js";

const STEP_MS = 60_000;
const HORIZON_CANDLES = 15;
const DEFAULT_PROGRESS =
  "data/benchmarks/kronos-base-ensemble-dense-execution-metrics."
  + "base-tuned-pretrained-ensemble.progress.json";

function argument(name: string, fallback?: string): string {
  const prefix = `--${name}=`;
  const inline = process.argv.slice(2).find((value) => value.startsWith(prefix));
  if (inline) return inline.slice(prefix.length);
  const index = process.argv.indexOf(`--${name}`);
  if (index >= 0 && process.argv[index + 1]) return process.argv[index + 1]!;
  if (fallback !== undefined) return fallback;
  throw new Error(`Missing --${name}.`);
}

function finiteArgument(name: string, fallback: number): number {
  const value = Number(argument(name, String(fallback)));
  if (!Number.isFinite(value)) throw new Error(`--${name} must be finite.`);
  return value;
}

function integerArgument(name: string, fallback: number): number {
  const value = finiteArgument(name, fallback);
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new Error(`--${name} must be a non-negative safe integer.`);
  }
  return value;
}

function objectValue(value: unknown, label: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${label} must be an object.`);
  }
  return value as Record<string, unknown>;
}

function atomicJson(file: string, value: unknown): void {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const temporary = `${file}.${process.pid}.tmp`;
  fs.writeFileSync(temporary, `${JSON.stringify(value, null, 2)}\n`);
  fs.renameSync(temporary, file);
}

const repoRoot = path.resolve(import.meta.dirname, "..");
const progressFile = path.resolve(repoRoot, argument("progress", DEFAULT_PROGRESS));
const windowId = argument("window-id");
const output = path.resolve(
  repoRoot,
  argument(
    "output",
    `data/benchmarks/kronos-progress-${windowId}-forecasts.json`,
  ),
);
const progress = objectValue(
  JSON.parse(fs.readFileSync(progressFile, "utf8")),
  "Kronos progress",
);
if (progress.contract !== "kronos-probabilistic-model-progress-v1") {
  throw new Error("Unsupported Kronos progress contract.");
}
if (!Array.isArray(progress.forecastRows) || progress.forecastRows.length === 0) {
  throw new Error("Kronos progress contains no forecast rows.");
}
const window = kronosInspectorWindows().find((candidate) => candidate.id === windowId);
if (!window) throw new Error(`Unknown non-fit inspector window: ${windowId}.`);
const matchingRows = progress.forecastRows.filter((candidate) => {
  const row = objectValue(candidate, "forecast row");
  return Array.isArray(row.windowIds) && row.windowIds.includes(windowId);
}).sort((left, right) => Number(
  objectValue(left, "forecast row").targetStartTime,
) - Number(objectValue(right, "forecast row").targetStartTime));
const skipRows = integerArgument("skip-rows", 0);
const maxRows = integerArgument("max-rows", matchingRows.length);
if (skipRows >= matchingRows.length || maxRows < 1) {
  throw new Error(
    `Requested row slice ${skipRows}+${maxRows} is outside ${matchingRows.length} matching rows.`,
  );
}
const rows = matchingRows.slice(skipRows, skipRows + maxRows);
const allowPartial = process.argv.includes("--allow-partial");
const grid = Array.from({ length: 101 }, (_, index) => -5 + index / 10);
const oracleGrid = Array.from({ length: 101 }, (_, index) => -100 + index * 2);
const sourceArtifact = {
  version: 2,
  contract: "kronos-causal-15x1m-forecast-v2",
  generatedAt: new Date().toISOString(),
  runSignature: progress.runSignature,
  modelId: progress.modelId,
  market: "Binance spot BTCUSDT",
  intervalMs: STEP_MS,
  lookbackCandles: finiteArgument("lookback", 512),
  horizonCandles: HORIZON_CANDLES,
  originStrideCandles: HORIZON_CANDLES,
  temperature: finiteArgument("temperature", 0.8),
  topP: finiteArgument("top-p", 0.9),
  sampleCount: finiteArgument("sample-count", 20),
  oracleGrid,
  executionOracleGrid: grid,
  executionOracle: {
    holding_period_steps: HORIZON_CANDLES,
    decision_delay_steps: HORIZON_CANDLES,
    value_horizon_steps: HORIZON_CANDLES,
    friction: 0.00175,
    grid_size: 101,
    temperature: 0.01,
    min_exposure: -5,
    max_exposure: 5,
    max_effective_exposure: 12.5,
    quote_borrow_rate: 0.000016658479699709332,
    asset_borrow_rate: 0.000016658479699709332,
  },
  rows,
};
const artifact = parseKronosForecastArtifact(sourceArtifact);
let coverageWindow = window;
if (allowPartial) {
  const firstTargetStart = Math.min(...artifact.rows.map((row) => row.targetStartTime));
  const lastTargetStart = Math.max(...artifact.rows.map((row) => row.targetStartTime));
  coverageWindow = {
    ...window,
    label: `${window.label} (partial progress)`,
    startTime: firstTargetStart,
    endTime: lastTargetStart + HORIZON_CANDLES * STEP_MS,
  };
}
validateDenseForecastCoverage(artifact, [coverageWindow]);
atomicJson(output, sourceArtifact);
console.log(JSON.stringify({
  output,
  windowId,
  rows: artifact.rows.length,
  coverageEndTime: coverageWindow.endTime,
  partial: allowPartial,
  skipRows,
  requestedMaxRows: maxRows,
  progressCompletedOrigins: progress.completedOrigins,
  progressTotalOrigins: progress.totalOrigins,
}, null, 2));
