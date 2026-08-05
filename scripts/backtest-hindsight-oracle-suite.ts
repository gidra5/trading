import fs from "node:fs";
import { createHash } from "node:crypto";
import { availableParallelism } from "node:os";
import path from "node:path";
import { Worker } from "node:worker_threads";
import {
  readCandleShardReferenceSync,
  readReferencedPayload,
  SequentialShardStore,
} from "@trading/storage";
import type {
  BacktestSummary,
  Candle,
  ExposureValueOracleActionDistribution,
  ExposureValueOracleOptions,
  StrategyConfig,
} from "@trading/bot-algo";
import {
  prepareExposureValueOracleCuda,
  vwKamaCudaStatus,
} from "@trading/bot-algo";
import { appConfig } from "../apps/server/src/config.js";
import {
  HINDSIGHT_ORACLE_GRID_SIZE,
  HINDSIGHT_ORACLE_CONFIDENCE_EXPOSURE_POWER,
  HINDSIGHT_ORACLE_CONFIDENCE_LEVERAGE_FLOOR,
  HINDSIGHT_ORACLE_DECISION_DELAY_MS,
  HINDSIGHT_ORACLE_HOLDING_PERIOD_MS,
  HINDSIGHT_ORACLE_MAINTENANCE_BPS_HOUR,
  HINDSIGHT_ORACLE_MAX_EFFECTIVE_EXPOSURE,
  HINDSIGHT_ORACLE_MAX_EXPOSURE,
  HINDSIGHT_ORACLE_TEMPERATURE,
  HINDSIGHT_ORACLE_VALUE_HORIZON_MS,
  runBotBacktestFromCandles,
} from "../apps/server/src/bot-backtest.js";
import { historicalWarmupSamples } from "../apps/server/src/historical-backtest.js";

const DAY_MS = 86_400_000;
const INTERVAL_MS = 1_000;
const HISTORY_ROOT = path.resolve(
  "data/market/immutable/refs/candles/spot-btcusdt/btcusdt/1s",
);
const DEFAULT_OUTPUT = path.resolve(
  "data/benchmarks/hindsight-oracle-bot-suite-2026-07-31.json",
);
const STORED_ORACLE_DATASET_ID =
  "mlp-direct-oracle-temporal-v12-delay-3600s-full-minute-oracle";
const STORED_ORACLE_DATASET = path.resolve(
  "data/training/datasets",
  STORED_ORACLE_DATASET_ID,
  "dataset.json",
);
const STORED_ORACLE_REFERENCES = path.resolve(
  "data/training/immutable/refs/oracle/1s/migrated",
  STORED_ORACLE_DATASET_ID,
  "components/oracle",
);
const CUDA_ORACLE_SLOT_ROOT = path.resolve(
  "data/training/cache/tmp/hindsight-oracle-cuda-slots",
);
const CUDA_ORACLE_SLOT_COUNT = 1;

interface SuiteWindow {
  id: string;
  startTime: number;
  endTime: number;
}

interface SuiteResult {
  id: string;
  startTime: number;
  endTime: number;
  loadedCandles: number;
  warmupCandles: number;
  oracleFutureCandles: number;
  wallDurationMs: number;
  summary: BacktestSummary;
}

interface SuiteReport {
  strategy: "hindsight-oracle-1s";
  summaryOnly: true;
  oracle: {
    intervalMs: number;
    holdingPeriodMs: number;
    decisionDelayMs: number;
    valueHorizonMs: number;
    maximumExposure: number;
    confidenceExposurePower: number;
    confidenceLeverageFloor: number;
  };
  generatedAt: string;
  historyRoot: string;
  latestAvailableDay: string;
  results: SuiteResult[];
}

type OracleBackend = "cpu" | "cuda";

interface OracleCacheDefinition {
  backend: OracleBackend;
  namespace: string;
  contractHash: string;
  metadata: Record<string, unknown>;
}

const STATIC_WINDOWS = [
  range("fit-full", "2025-03-19", "2025-11-13"),
  range("fit-1", "2025-03-19", "2025-05-17"),
  range("fit-2", "2025-05-18", "2025-07-16"),
  range("fit-3", "2025-07-17", "2025-09-14"),
  range("fit-4", "2025-09-15", "2025-11-13"),
  range("sideways-churn-2022-07", "2022-07-28", "2022-08-03"),
  range("sideways-churn-2022-05", "2022-05-14", "2022-05-20"),
  range("sideways-churn-2021-12", "2021-12-14", "2021-12-20"),
  range("sideways-churn-2021-09", "2021-09-08", "2021-09-14"),
  range("sideways-churn-2023-03", "2023-03-18", "2023-03-24"),
  range("regime-up-2023-03", "2023-03-11", "2023-03-17"),
  range("regime-flat-2026-04", "2026-04-22", "2026-04-28"),
  range("regime-down-2022-06", "2022-06-12", "2022-06-18"),
  range("shape-up-low-2024-02", "2024-02-24", "2024-02-26"),
  range("shape-up-high-2022-06", "2022-06-19", "2022-06-21"),
  range("shape-down-low-2023-06", "2023-06-03", "2023-06-05"),
  range("shape-down-high-2022-06", "2022-06-13", "2022-06-15"),
  range("shape-flat-high-bias-2021-10", "2021-10-19", "2021-10-21"),
  range("shape-flat-high-bias-low-2025-02", "2025-02-14", "2025-02-16"),
  range("shape-flat-low-bias-2024-07", "2024-07-07", "2024-07-09"),
  range("shape-flat-low-bias-low-2025-07", "2025-07-04", "2025-07-06"),
  range("shape-flat-mid-bias-2024-01", "2024-01-02", "2024-01-04"),
  range("shape-flat-mid-bias-low-2023-09", "2023-09-15", "2023-09-17"),
  range("sharpe-up-3d-2024-11", "2024-11-09", "2024-11-11"),
  range("sharpe-up-3d-2023-12", "2023-12-03", "2023-12-05"),
  range("sharpe-down-3d-2026-06", "2026-06-01", "2026-06-03"),
  range("sharpe-down-3d-2023-03", "2023-03-07", "2023-03-09"),
  range("sharpe-up-7d-2023-12", "2023-11-29", "2023-12-05"),
  range("sharpe-up-7d-2024-11", "2024-11-05", "2024-11-11"),
  range("sharpe-down-7d-2023-03", "2023-03-03", "2023-03-09"),
  range("sharpe-down-7d-2026-06", "2026-05-27", "2026-06-02"),
  range("failure-down-3d-2022-06", "2022-06-11", "2022-06-13"),
  range("failure-down-7d-2022-06", "2022-06-07", "2022-06-13"),
] satisfies SuiteWindow[];

async function main(): Promise<void> {
  const valueHorizonMinutes = integerArgument(
    "value-horizon-minutes",
    HINDSIGHT_ORACLE_VALUE_HORIZON_MS / 60_000,
  );
  if (valueHorizonMinutes < HINDSIGHT_ORACLE_HOLDING_PERIOD_MS / 60_000) {
    throw new Error("--value-horizon-minutes must cover at least one holding period.");
  }
  const valueHorizonMs = valueHorizonMinutes * 60_000;
  const latestAvailableDay = availableDays().at(-1);
  if (!latestAvailableDay) throw new Error(`No one-second history found in ${HISTORY_ROOT}.`);
  const latestEndTime = parseDay(latestAvailableDay) + DAY_MS;
  const latestMeasuredEndTime = latestEndTime - valueHorizonMs;
  const windows = [
    ...STATIC_WINDOWS,
    {
      id: "latest-3m",
      startTime: subtractUtcMonths(latestEndTime, 3),
      endTime: latestMeasuredEndTime,
    },
  ];
  const only = argument("only")?.split(",").map((value) => value.trim()).filter(Boolean);
  const excludePrefixes = argument("exclude-prefix")
    ?.split(",")
    .map((value) => value.trim())
    .filter(Boolean) ?? [];
  const included = only?.length
    ? windows.filter((window) => only.includes(window.id))
    : windows;
  const selected = included.filter((window) =>
    !excludePrefixes.some((prefix) => window.id.startsWith(prefix)));
  if (selected.length === 0) throw new Error(`No suite windows matched --only ${only?.join(",")}.`);

  const output = path.resolve(argument("output") ?? DEFAULT_OUTPUT);
  const oracleOnly = flag("oracle-only");
  const oracleWorkers = Math.max(1, Math.min(
    availableParallelism(),
    Number.parseInt(argument("oracle-workers") ?? "1", 10) || 1,
  ));
  const requestedOracleBackend = argument("oracle-backend") ?? "cpu";
  if (!['cpu', 'cuda', 'auto'].includes(requestedOracleBackend)) {
    throw new Error("--oracle-backend must be cpu, cuda, or auto.");
  }
  const cudaStatus = requestedOracleBackend === "cpu"
    ? undefined
    : await vwKamaCudaStatus();
  if (requestedOracleBackend === "cuda" && !cudaStatus?.available) {
    throw new Error(cudaStatus?.reason ?? "CUDA oracle backend is unavailable.");
  }
  const oracleBackend: OracleBackend = requestedOracleBackend === "cuda"
    || requestedOracleBackend === "auto" && cudaStatus?.available
    ? "cuda"
    : "cpu";
  console.log(
    `BACKEND oracle=${oracleBackend} horizonMinutes=${valueHorizonMinutes}`
    + `${cudaStatus?.device ? ` device=${cudaStatus.device}` : ""}`,
  );
  const existing = readReport(output, valueHorizonMs);
  const report: SuiteReport = existing ?? {
    strategy: "hindsight-oracle-1s",
    summaryOnly: true,
    oracle: {
      intervalMs: INTERVAL_MS,
      holdingPeriodMs: HINDSIGHT_ORACLE_HOLDING_PERIOD_MS,
      decisionDelayMs: HINDSIGHT_ORACLE_DECISION_DELAY_MS,
      valueHorizonMs,
      maximumExposure: HINDSIGHT_ORACLE_MAX_EXPOSURE,
      confidenceExposurePower: HINDSIGHT_ORACLE_CONFIDENCE_EXPOSURE_POWER,
      confidenceLeverageFloor: HINDSIGHT_ORACLE_CONFIDENCE_LEVERAGE_FLOOR,
    },
    generatedAt: new Date().toISOString(),
    historyRoot: HISTORY_ROOT,
    latestAvailableDay,
    results: [],
  };
  report.summaryOnly = true;
  const completed = new Set(report.results.map((result) => result.id));
  const warmupMs = historicalWarmupSamples(appConfig.strategy, INTERVAL_MS) * INTERVAL_MS;
  let reusableFitOracle: PrecomputedOracleDistributions | undefined;

  for (const [index, window] of selected.entries()) {
    if (completed.has(window.id)) {
      console.log(`SKIP ${window.id} (${index + 1}/${selected.length})`);
      continue;
    }
    console.log(
      `START ${window.id} (${index + 1}/${selected.length}) `
      + `${isoDay(window.startTime)}..${isoDay(window.endTime - 1)}`,
    );
    const loaded = loadWindow(window, warmupMs, valueHorizonMs);
    console.log(
      `LOADED ${window.id} measured=${loaded.candles.length} `
      + `warmup=${loaded.warmup.length} future=${loaded.oracleFuture.length}`,
    );
    const startedAt = Date.now();
    const precomputed = reusableFitOracle?.covers(loaded.candles)
      ? reusableFitOracle
      : await precomputeOracleDistributions(
          loaded.candles,
          loaded.oracleFuture,
          appConfig.strategy,
          oracleWorkers,
          window.id,
          oracleBackend,
          valueHorizonMs,
        );
    if (precomputed === reusableFitOracle) {
      console.log(`REUSE ${window.id} oracle=fit-full`);
    } else if (window.id === "fit-full") {
      reusableFitOracle = precomputed;
    }
    if (oracleOnly) {
      console.log(`ORACLE-ONLY ${window.id} complete`);
      if (window.id === "fit-4") reusableFitOracle = undefined;
      global.gc?.();
      continue;
    }
    const result = await runBotBacktestFromCandles(loaded.candles, {
      config: appConfig.strategy,
      strategy: "hindsight-oracle-1s",
      warmup: loaded.warmup,
      oracleFuture: loaded.oracleFuture,
      maxEquityPoints: 800,
      maxChartCandles: 2_000,
      summaryOnly: true,
      hindsightOracleDistributionAt: precomputed?.distributionAt,
      onProgress: ({ candlesProcessed, totalCandles, elapsedMs }) => {
        console.log(
          `REPLAY ${window.id} candles=${candlesProcessed}/${totalCandles} `
          + `pct=${(candlesProcessed / totalCandles * 100).toFixed(0)} `
          + `durationMs=${elapsedMs}`,
        );
      },
    });
    const suiteResult: SuiteResult = {
      id: window.id,
      startTime: window.startTime,
      endTime: window.endTime,
      loadedCandles: loaded.candles.length,
      warmupCandles: loaded.warmup.length,
      oracleFutureCandles: loaded.oracleFuture.length,
      wallDurationMs: Date.now() - startedAt,
      summary: result.summary,
    };
    report.results.push(suiteResult);
    report.generatedAt = new Date().toISOString();
    writeReport(output, report);
    console.log(`RESULT ${JSON.stringify(compactResult(suiteResult))}`);
    if (window.id === "fit-4") reusableFitOracle = undefined;
    global.gc?.();
  }

  if (oracleOnly) {
    console.log(`ORACLE-ONLY-COMPLETE horizonMinutes=${valueHorizonMinutes}`);
    return;
  }
  console.log(`AGGREGATE ${JSON.stringify(aggregate(report.results))}`);
  console.log(`REPORT ${output}`);
}

interface PrecomputedOracleDistributions {
  distributionAt(timestamp: number): ExposureValueOracleActionDistribution | null;
  covers(candles: readonly Candle[]): boolean;
}

async function precomputeOracleDistributions(
  candles: readonly Candle[],
  future: readonly Candle[],
  config: StrategyConfig,
  workerCount: number,
  windowId: string,
  backend: OracleBackend,
  valueHorizonMs: number,
): Promise<PrecomputedOracleDistributions> {
  const holdingPeriodSteps = Math.max(1, Math.round(
    HINDSIGHT_ORACLE_HOLDING_PERIOD_MS / INTERVAL_MS,
  ));
  const decisionDelaySteps = Math.max(1, Math.round(
    HINDSIGHT_ORACLE_DECISION_DELAY_MS / INTERVAL_MS,
  ));
  const valueHorizonSteps = Math.max(holdingPeriodSteps, Math.round(
    valueHorizonMs / INTERVAL_MS,
  ));
  const priceCount = candles.length + future.length;
  const prices = new Float64Array(new SharedArrayBuffer(priceCount * Float64Array.BYTES_PER_ELEMENT));
  for (let index = 0; index < candles.length; index += 1) prices[index] = candles[index]!.close;
  for (let index = 0; index < future.length; index += 1) {
    prices[candles.length + index] = future[index]!.close;
  }
  const decisionCount = candles.length > 1
    ? Math.floor((candles.length - 2) / holdingPeriodSteps) + 1
    : 0;
  const storedOracle = storedOracleDefinition(config, valueHorizonSteps);
  const outputColumns = storedOracle.grid.length;
  const probabilities = new Float32Array(new SharedArrayBuffer(
    decisionCount * outputColumns * Float32Array.BYTES_PER_ELEMENT,
  ));
  const feasibleActionCounts = new Uint16Array(new SharedArrayBuffer(
    decisionCount * Uint16Array.BYTES_PER_ELEMENT,
  ));
  const loaded = new Uint8Array(new SharedArrayBuffer(decisionCount));
  const cursor = new Int32Array(new SharedArrayBuffer(Int32Array.BYTES_PER_ELEMENT));
  const completed = new Int32Array(new SharedArrayBuffer(Int32Array.BYTES_PER_ELEMENT));
  const quoteBorrowRate = Math.expm1(
    Math.log1p(HINDSIGHT_ORACLE_MAINTENANCE_BPS_HOUR / 10_000)
      * INTERVAL_MS / 3_600_000,
  );
  const cache = oracleCacheDefinition(backend, workerDataOptions({
    holdingPeriodSteps,
    decisionDelaySteps,
    valueHorizonSteps,
    config,
    quoteBorrowRate,
  }), storedOracle.grid);
  const cachedRows = await loadCachedOracleRows({
    windowId,
    candles,
    probabilities,
    feasibleActionCounts,
    loaded,
    decisionCount,
    storedOracle,
    cache,
  });
  const storedRows = cachedRows + await loadStoredOracleRows({
    windowId,
    candles,
    probabilities,
    feasibleActionCounts,
    loaded,
    decisionCount,
    holdingPeriodSteps,
    storedOracle,
  });
  completed[0] = storedRows;
  console.log(
    `STORED ${windowId} rows=${storedRows}/${decisionCount} cache=${cachedRows} `
    + `days=${storedOracle.availableDays}`,
  );
  const options = workerDataOptions({
    holdingPeriodSteps,
    decisionDelaySteps,
    valueHorizonSteps,
    config,
    quoteBorrowRate,
  });
  const workerData = {
    prices: prices.buffer,
    priceCount,
    scoredLength: candles.length,
    probabilities: probabilities.buffer,
    feasibleActionCounts: feasibleActionCounts.buffer,
    loaded: loaded.buffer,
    cursor: cursor.buffer,
    completed: completed.buffer,
    decisionCount,
    holdingPeriodSteps,
    valueHorizonSteps,
    usableIndexes: storedOracle.usableIndexes,
    outputColumns,
    options,
  };
  const startedAt = Date.now();
  const progress = setInterval(() => {
    const count = Atomics.load(completed, 0);
    const elapsedSec = Math.max(1, (Date.now() - startedAt) / 1_000);
    const rate = count / elapsedSec;
    const remainingSec = rate > 0 ? (decisionCount - count) / rate : Infinity;
    console.log(
      `ORACLE ${windowId} ${count}/${decisionCount} `
      + `(${(count / Math.max(1, decisionCount) * 100).toFixed(1)}%) `
      + `eta=${Number.isFinite(remainingSec) ? Math.round(remainingSec) : "?"}s`,
    );
  }, 60_000);
  progress.unref();
  try {
    if (backend === "cuda") {
      await populateCudaOracleRows({
        prices,
        scoredLength: candles.length,
        probabilities,
        feasibleActionCounts,
        loaded,
        completed,
        decisionCount,
        holdingPeriodSteps,
        valueHorizonSteps,
        storedOracle,
        options: workerData.options,
        windowId,
      });
    } else {
      await Promise.all(Array.from({
        length: Math.min(workerCount, decisionCount - storedRows),
      }, () =>
        runOracleWorker(workerData)));
    }
  } finally {
    clearInterval(progress);
  }
  await persistCachedOracleRows({
    candles,
    probabilities,
    loaded,
    decisionCount,
    holdingPeriodSteps,
    storedOracle,
    cache,
  });
  console.log(
    `ORACLE ${windowId} complete rows=${decisionCount} backend=${backend} `
    + `workers=${backend === "cpu" ? workerCount : 1} `
    + `durationMs=${Date.now() - startedAt}`,
  );

  const grid = storedOracle.grid;
  const firstCloseTime = candles[0]!.closeTime;
  const lastCloseTime = candles.at(-1)!.closeTime;
  const scoredLength = candles.length;
  return {
    covers(candidate) {
      const first = candidate[0]?.closeTime;
      const last = candidate.at(-1)?.closeTime;
      return first !== undefined
        && last !== undefined
        && first >= firstCloseTime
        && last <= lastCloseTime
        && (first - firstCloseTime) % INTERVAL_MS === 0;
    },
    distributionAt(timestamp) {
      const candleIndex = Math.round((timestamp - firstCloseTime) / INTERVAL_MS);
      if (
        candleIndex < 0
        || candleIndex >= scoredLength - 1
        || candleIndex % holdingPeriodSteps !== 0
        || firstCloseTime + candleIndex * INTERVAL_MS !== timestamp
      ) return null;
      const row = candleIndex / holdingPeriodSteps;
      const offset = row * grid.length;
      return {
        grid,
        probabilities: probabilities.subarray(offset, offset + grid.length),
        mean: 0,
        secondMoment: 0,
        modalExposure: 0,
        entropy: 0,
        opportunity: 0,
        feasibleActionCount: feasibleActionCounts[row]!,
      };
    },
  };
}

function workerDataOptions(input: {
  holdingPeriodSteps: number;
  decisionDelaySteps: number;
  valueHorizonSteps: number;
  config: StrategyConfig;
  quoteBorrowRate: number;
}): ExposureValueOracleOptions {
  return {
    scoreStartIndex: 0,
    holdingPeriodSteps: input.holdingPeriodSteps,
    decisionDelaySteps: input.decisionDelaySteps,
    valueHorizonSteps: input.valueHorizonSteps,
    friction: (
      input.config.feeBps + input.config.positionRisk.marketSlippageBps
    ) / 10_000,
    gridSize: HINDSIGHT_ORACLE_GRID_SIZE,
    minExposure: -HINDSIGHT_ORACLE_MAX_EFFECTIVE_EXPOSURE,
    maxExposure: HINDSIGHT_ORACLE_MAX_EFFECTIVE_EXPOSURE,
    maxEffectiveExposure: HINDSIGHT_ORACLE_MAX_EFFECTIVE_EXPOSURE,
    temperature: HINDSIGHT_ORACLE_TEMPERATURE,
    opportunityEpsilon: 0,
    quoteBorrowRate: input.quoteBorrowRate,
    assetBorrowRate: input.quoteBorrowRate,
    distributionOnly: true,
    includePath: false,
  };
}

function oracleCacheDefinition(
  backend: OracleBackend,
  options: ExposureValueOracleOptions,
  grid: Float64Array,
): OracleCacheDefinition {
  const metadata = {
    version: 1,
    numericBackend: backend,
    intervalMs: INTERVAL_MS,
    decisionIntervalMs: HINDSIGHT_ORACLE_HOLDING_PERIOD_MS,
    options: {
      holdingPeriodSteps: options.holdingPeriodSteps,
      decisionDelaySteps: options.decisionDelaySteps,
      valueHorizonSteps: options.valueHorizonSteps,
      friction: options.friction,
      gridSize: options.gridSize,
      minExposure: options.minExposure,
      maxExposure: options.maxExposure,
      maxEffectiveExposure: options.maxEffectiveExposure,
      temperature: options.temperature,
      quoteBorrowRate: options.quoteBorrowRate,
      assetBorrowRate: options.assetBorrowRate,
    },
    usableGrid: Array.from(grid),
  };
  const contractHash = createHash("sha256")
    .update(JSON.stringify(metadata))
    .digest("hex");
  return {
    backend,
    namespace: `oracle/1s/hindsight-bot-${contractHash.slice(0, 20)}`,
    contractHash,
    metadata,
  };
}

async function loadCachedOracleRows(input: {
  windowId: string;
  candles: readonly Candle[];
  probabilities: Float32Array;
  feasibleActionCounts: Uint16Array;
  loaded: Uint8Array;
  decisionCount: number;
  storedOracle: StoredOracleDefinition;
  cache: OracleCacheDefinition;
}): Promise<number> {
  const store = oracleCacheStore();
  const firstCloseTime = input.candles[0]!.closeTime;
  const firstDay = utcDay(input.candles[0]!.openTime);
  const lastDay = utcDay(input.candles.at(-1)!.openTime);
  let loadedRows = 0;
  for (let day = firstDay; day <= lastDay; day += DAY_MS) {
    const referenceFile = store.referenceFile(input.cache.namespace, isoDay(day));
    if (!fs.existsSync(referenceFile)) continue;
    const { reference, payload } = await readReferencedPayload(referenceFile, false);
    if (
      reference.metadata?.contractHash !== input.cache.contractHash
      || reference.sequence.step !== HINDSIGHT_ORACLE_HOLDING_PERIOD_MS
      || reference.layout.rows !== reference.sequence.count
      || reference.layout.columns !== input.storedOracle.grid.length
      || reference.layout.dtype !== "float32-le"
      || payload.byteLength !== reference.sequence.count
        * input.storedOracle.grid.length * Float32Array.BYTES_PER_ELEMENT
    ) throw new Error(`Cached oracle shard is incompatible: ${referenceFile}`);
    const source = payload.byteOffset % Float32Array.BYTES_PER_ELEMENT === 0
      ? new Float32Array(
          payload.buffer,
          payload.byteOffset,
          payload.byteLength / Float32Array.BYTES_PER_ELEMENT,
        )
      : new Float32Array(payload.buffer.slice(
          payload.byteOffset,
          payload.byteOffset + payload.byteLength,
        ));
    for (let sourceRow = 0; sourceRow < reference.sequence.count; sourceRow += 1) {
      const timestamp = reference.sequence.start
        + sourceRow * HINDSIGHT_ORACLE_HOLDING_PERIOD_MS;
      const elapsed = timestamp - firstCloseTime;
      if (elapsed < 0 || elapsed % HINDSIGHT_ORACLE_HOLDING_PERIOD_MS !== 0) continue;
      const targetRow = elapsed / HINDSIGHT_ORACLE_HOLDING_PERIOD_MS;
      if (targetRow >= input.decisionCount || input.loaded[targetRow]) continue;
      const sourceOffset = sourceRow * input.storedOracle.grid.length;
      const targetOffset = targetRow * input.storedOracle.grid.length;
      input.probabilities.set(
        source.subarray(sourceOffset, sourceOffset + input.storedOracle.grid.length),
        targetOffset,
      );
      let feasibleActionCount = 0;
      for (let column = 0; column < input.storedOracle.grid.length; column += 1) {
        if (input.probabilities[targetOffset + column]! > 0) feasibleActionCount += 1;
      }
      input.feasibleActionCounts[targetRow] = feasibleActionCount;
      input.loaded[targetRow] = 1;
      loadedRows += 1;
    }
  }
  if (loadedRows > 0) console.log(`CACHE ${input.windowId} rows=${loadedRows}`);
  return loadedRows;
}

async function persistCachedOracleRows(input: {
  candles: readonly Candle[];
  probabilities: Float32Array;
  loaded: Uint8Array;
  decisionCount: number;
  holdingPeriodSteps: number;
  storedOracle: StoredOracleDefinition;
  cache: OracleCacheDefinition;
}): Promise<void> {
  const decisionsPerDay = DAY_MS / HINDSIGHT_ORACLE_HOLDING_PERIOD_MS;
  const store = oracleCacheStore();
  for (let firstRow = 0; firstRow < input.decisionCount; firstRow += decisionsPerDay) {
    const endRow = Math.min(input.decisionCount, firstRow + decisionsPerDay);
    if (endRow - firstRow !== decisionsPerDay) continue;
    const firstCandle = firstRow * input.holdingPeriodSteps;
    const candle = input.candles[firstCandle];
    if (!candle || candle.openTime !== utcDay(candle.openTime)) continue;
    if (!input.loaded.subarray(firstRow, endRow).every((value) => value !== 0)) continue;
    const key = isoDay(candle.openTime);
    const referenceFile = store.referenceFile(input.cache.namespace, key);
    if (fs.existsSync(referenceFile)) continue;
    const firstCell = firstRow * input.storedOracle.grid.length;
    const endCell = endRow * input.storedOracle.grid.length;
    const rows = input.probabilities.subarray(firstCell, endCell);
    const payload = new Uint8Array(rows.buffer, rows.byteOffset, rows.byteLength);
    const result = await store.put({
      namespace: input.cache.namespace,
      key,
      payload,
      sequence: {
        start: candle.closeTime,
        step: HINDSIGHT_ORACLE_HOLDING_PERIOD_MS,
        count: decisionsPerDay,
        unit: "unix-ms",
      },
      layout: {
        encoding: "raw-row-major",
        dtype: "float32-le",
        rows: decisionsPerDay,
        columns: input.storedOracle.grid.length,
      },
      metadata: {
        contractHash: input.cache.contractHash,
        contract: input.cache.metadata,
      },
      compressionLevel: 3,
    });
    console.log(
      `CACHE-WRITE ${key} backend=${input.cache.backend} `
      + `created=${result.objectCreated} bytes=${result.reference.object.compressedBytes}`,
    );
  }
}

function oracleCacheStore(): SequentialShardStore {
  return new SequentialShardStore(path.resolve("data/training/immutable"));
}

async function populateCudaOracleRows(input: {
  prices: Float64Array;
  scoredLength: number;
  probabilities: Float32Array;
  feasibleActionCounts: Uint16Array;
  loaded: Uint8Array;
  completed: Int32Array;
  decisionCount: number;
  holdingPeriodSteps: number;
  valueHorizonSteps: number;
  storedOracle: StoredOracleDefinition;
  options: ExposureValueOracleOptions;
  windowId: string;
}): Promise<void> {
  const decisionsPerChunk = Math.max(
    1,
    Math.floor(DAY_MS / (input.holdingPeriodSteps * INTERVAL_MS)),
  );
  for (let firstRow = 0; firstRow < input.decisionCount; firstRow += decisionsPerChunk) {
    const endRow = Math.min(input.decisionCount, firstRow + decisionsPerChunk);
    if (input.loaded.subarray(firstRow, endRow).every((value) => value !== 0)) continue;
    const firstCandle = firstRow * input.holdingPeriodSteps;
    const lastScoredCandle = (endRow - 1) * input.holdingPeriodSteps;
    const priceEnd = Math.min(
      input.prices.length,
      lastScoredCandle + input.valueHorizonSteps + 1,
    );
    const chunkPrices = input.prices.subarray(firstCandle, priceEnd);
    const releaseCudaSlot = await acquireCudaOracleSlot();
    const prepared = await (async () => {
      try {
        return await prepareExposureValueOracleCuda(chunkPrices, {
          ...input.options,
          scoreStartIndex: 0,
          terminalIndex: chunkPrices.length - 1,
          includeProbabilities: true,
          includePath: false,
          distributionOnly: true,
        });
      } finally {
        releaseCudaSlot();
      }
    })();
    const sourceProbabilities = prepared.oracle.probabilities;
    if (!sourceProbabilities) throw new Error("CUDA oracle did not return probabilities.");
    if (prepared.oracle.grid.length !== HINDSIGHT_ORACLE_GRID_SIZE) {
      throw new Error("CUDA oracle returned an unexpected action grid.");
    }
    for (let row = firstRow; row < endRow; row += 1) {
      if (input.loaded[row]) continue;
      const localCandle = (row - firstRow) * input.holdingPeriodSteps;
      const sourceOffset = localCandle * HINDSIGHT_ORACLE_GRID_SIZE;
      const targetOffset = row * input.storedOracle.grid.length;
      let total = 0;
      for (let column = 0; column < input.storedOracle.usableIndexes.length; column += 1) {
        const probability = sourceProbabilities[
          sourceOffset + input.storedOracle.usableIndexes[column]!
        ]!;
        input.probabilities[targetOffset + column] = probability;
        total += probability;
      }
      if (!(total > 0)) throw new Error(`CUDA oracle row ${row} has no usable probability mass.`);
      let feasibleActionCount = 0;
      for (let column = 0; column < input.storedOracle.grid.length; column += 1) {
        const cell = targetOffset + column;
        input.probabilities[cell] /= total;
        if (input.probabilities[cell]! > 0) feasibleActionCount += 1;
      }
      input.feasibleActionCounts[row] = feasibleActionCount;
      input.loaded[row] = 1;
      Atomics.add(input.completed, 0, 1);
    }
    console.log(
      `ORACLE-CUDA ${input.windowId} rows=${endRow}/${input.decisionCount} `
      + `kernelMs=${prepared.kernelMs.toFixed(1)}`,
    );
  }
}

async function acquireCudaOracleSlot(): Promise<() => void> {
  fs.mkdirSync(CUDA_ORACLE_SLOT_ROOT, { recursive: true });
  while (true) {
    for (let slot = 0; slot < CUDA_ORACLE_SLOT_COUNT; slot += 1) {
      const lockFile = path.join(CUDA_ORACLE_SLOT_ROOT, `${slot}.lock`);
      try {
        const descriptor = fs.openSync(lockFile, "wx");
        fs.writeFileSync(descriptor, `${process.pid}\n`, "utf8");
        return () => {
          fs.closeSync(descriptor);
          fs.rmSync(lockFile, { force: true });
        };
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error;
        let owner = Number.NaN;
        try {
          owner = Number.parseInt(fs.readFileSync(lockFile, "utf8"), 10);
        } catch (readError) {
          if ((readError as NodeJS.ErrnoException).code !== "ENOENT") throw readError;
          continue;
        }
        if (Number.isInteger(owner) && !processExists(owner)) {
          fs.rmSync(lockFile, { force: true });
        }
      }
    }
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
}

function processExists(processId: number): boolean {
  try {
    process.kill(processId, 0);
    return true;
  } catch {
    return false;
  }
}

interface StoredOracleDefinition {
  grid: Float64Array;
  usableIndexes: number[];
  availableDays: number;
  compatible: boolean;
}

function storedOracleDefinition(
  config: StrategyConfig,
  valueHorizonSteps: number,
): StoredOracleDefinition {
  const manifest = JSON.parse(fs.readFileSync(STORED_ORACLE_DATASET, "utf8"));
  const execution = manifest.execution;
  const frictionBps = config.feeBps + config.positionRisk.marketSlippageBps;
  const compatible = !(
    manifest.samplingIntervalMs !== INTERVAL_MS
    || execution?.feeBps !== frictionBps
    || execution?.gridSize !== HINDSIGHT_ORACLE_GRID_SIZE
    || execution?.minimumUsableExposure !== -HINDSIGHT_ORACLE_MAX_EXPOSURE
    || execution?.maximumUsableExposure !== HINDSIGHT_ORACLE_MAX_EXPOSURE
    || execution?.minimumEffectiveExposure !== -HINDSIGHT_ORACLE_MAX_EFFECTIVE_EXPOSURE
    || execution?.maximumEffectiveExposure !== HINDSIGHT_ORACLE_MAX_EFFECTIVE_EXPOSURE
    || execution?.temperature !== HINDSIGHT_ORACLE_TEMPERATURE
    || execution?.holdingPeriodSteps !== HINDSIGHT_ORACLE_HOLDING_PERIOD_MS / INTERVAL_MS
    || execution?.decisionDelaySteps !== HINDSIGHT_ORACLE_DECISION_DELAY_MS / INTERVAL_MS
    || execution?.valueHorizonSteps !== valueHorizonSteps
    || execution?.maintenanceBpsHour?.quoteBorrow !== HINDSIGHT_ORACLE_MAINTENANCE_BPS_HOUR
    || execution?.maintenanceBpsHour?.assetBorrow !== HINDSIGHT_ORACLE_MAINTENANCE_BPS_HOUR
  );
  const sourceGrid = Float64Array.from(manifest.grid);
  if (sourceGrid.length !== HINDSIGHT_ORACLE_GRID_SIZE) {
    throw new Error("Stored one-second oracle grid has an invalid size.");
  }
  const usableIndexes = Array.from(sourceGrid, (exposure, index) => ({ exposure, index }))
    .filter(({ exposure }) => Math.abs(exposure) <= HINDSIGHT_ORACLE_MAX_EXPOSURE)
    .map(({ index }) => index);
  const grid = Float64Array.from(usableIndexes, (index) => sourceGrid[index]!);
  return {
    grid,
    usableIndexes,
    compatible,
    availableDays: compatible && fs.existsSync(STORED_ORACLE_REFERENCES)
      ? fs.readdirSync(STORED_ORACLE_REFERENCES)
          .filter((file) => file.endsWith(".raw-oracle-probabilities.f32.zst.json")).length
      : 0,
  };
}

async function loadStoredOracleRows(input: {
  windowId: string;
  candles: readonly Candle[];
  probabilities: Float32Array;
  feasibleActionCounts: Uint16Array;
  loaded: Uint8Array;
  decisionCount: number;
  holdingPeriodSteps: number;
  storedOracle: StoredOracleDefinition;
}): Promise<number> {
  if (!input.storedOracle.compatible) return 0;
  const firstCloseTime = input.candles[0]!.closeTime;
  const lastCloseTime = input.candles.at(-1)!.closeTime;
  const firstDay = utcDay(input.candles[0]!.openTime);
  const lastDay = utcDay(input.candles.at(-1)!.openTime);
  let loadedRows = 0;
  let lastProgressAt = Date.now();
  for (let day = firstDay; day <= lastDay; day += DAY_MS) {
    const date = isoDay(day);
    const referenceFile = path.join(
      STORED_ORACLE_REFERENCES,
      `${date}.raw-oracle-probabilities.f32.zst.json`,
    );
    if (!fs.existsSync(referenceFile)) continue;
    const { reference, payload } = await readReferencedPayload(referenceFile, false);
    if (
      reference.sequence.step !== INTERVAL_MS
      || reference.layout.rows !== reference.sequence.count
      || reference.layout.columns !== HINDSIGHT_ORACLE_GRID_SIZE
      || reference.layout.dtype !== "float32-le"
      || payload.byteLength
        !== reference.sequence.count * HINDSIGHT_ORACLE_GRID_SIZE * Float32Array.BYTES_PER_ELEMENT
    ) throw new Error(`Stored oracle shard has an invalid layout: ${referenceFile}`);
    const source = payload.byteOffset % Float32Array.BYTES_PER_ELEMENT === 0
      ? new Float32Array(
          payload.buffer,
          payload.byteOffset,
          payload.byteLength / Float32Array.BYTES_PER_ELEMENT,
        )
      : new Float32Array(payload.buffer.slice(
          payload.byteOffset,
          payload.byteOffset + payload.byteLength,
        ));
    for (let sourceRow = 0; sourceRow < reference.sequence.count; sourceRow += input.holdingPeriodSteps) {
      const timestamp = reference.sequence.start + sourceRow * reference.sequence.step;
      if (timestamp < firstCloseTime || timestamp > lastCloseTime) continue;
      const elapsed = timestamp - firstCloseTime;
      if (elapsed % HINDSIGHT_ORACLE_HOLDING_PERIOD_MS !== 0) continue;
      const targetRow = elapsed / HINDSIGHT_ORACLE_HOLDING_PERIOD_MS;
      if (targetRow < 0 || targetRow >= input.decisionCount || input.loaded[targetRow]) continue;
      const sourceOffset = sourceRow * HINDSIGHT_ORACLE_GRID_SIZE;
      const targetOffset = targetRow * input.storedOracle.grid.length;
      let total = 0;
      for (let column = 0; column < input.storedOracle.usableIndexes.length; column += 1) {
        const probability = source[sourceOffset + input.storedOracle.usableIndexes[column]!]!;
        input.probabilities[targetOffset + column] = probability;
        total += probability;
      }
      if (!(total > 0)) throw new Error(`Stored oracle row ${date}:${sourceRow} has no usable mass.`);
      let feasibleActionCount = 0;
      for (let column = 0; column < input.storedOracle.grid.length; column += 1) {
        const cell = targetOffset + column;
        input.probabilities[cell] /= total;
        if (input.probabilities[cell]! > 0) feasibleActionCount += 1;
      }
      input.feasibleActionCounts[targetRow] = feasibleActionCount;
      input.loaded[targetRow] = 1;
      loadedRows += 1;
    }
    if (Date.now() - lastProgressAt >= 10_000) {
      console.log(
        `STORED-LOAD ${input.windowId} rows=${loadedRows}/${input.decisionCount} date=${date}`,
      );
      lastProgressAt = Date.now();
    }
  }
  return loadedRows;
}

function runOracleWorker(workerData: Record<string, unknown>): Promise<void> {
  return new Promise((resolve, reject) => {
    const worker = new Worker(
      path.resolve("scripts/hindsight-oracle-precompute-worker.ts"),
      {
        workerData,
        execArgv: process.execArgv.filter((value) =>
          !value.startsWith("--max-old-space-size") && value !== "--expose-gc"),
      },
    );
    worker.once("error", reject);
    worker.once("exit", (code) => {
      if (code === 0) resolve();
      else reject(new Error(`Hindsight oracle precompute worker exited with code ${code}.`));
    });
  });
}

function range(id: string, start: string, end: string): SuiteWindow {
  return {
    id,
    startTime: parseDay(start),
    endTime: parseDay(end) + DAY_MS,
  };
}

function loadWindow(window: SuiteWindow, warmupMs: number, valueHorizonMs: number): {
  warmup: Candle[];
  candles: Candle[];
  oracleFuture: Candle[];
} {
  const loadStart = window.startTime - warmupMs;
  const loadEnd = window.endTime + valueHorizonMs;
  const warmup: Candle[] = [];
  const candles: Candle[] = [];
  const oracleFuture: Candle[] = [];
  for (let day = utcDay(loadStart); day < loadEnd; day += DAY_MS) {
    const date = isoDay(day);
    const dailyCandles = readReferenceDay(date);
    if (!dailyCandles) {
      if (day >= window.endTime) continue;
      throw new Error(`Missing measured or warmup history for ${date} in ${HISTORY_ROOT}.`);
    }
    for (const candle of dailyCandles) {
      if (candle.openTime < loadStart || candle.openTime >= loadEnd) continue;
      if (candle.openTime < window.startTime) warmup.push(candle);
      else if (candle.openTime < window.endTime) candles.push(candle);
      else oracleFuture.push(candle);
    }
  }
  if (candles.length === 0) throw new Error(`No measured candles loaded for ${window.id}.`);
  return { warmup, candles, oracleFuture };
}

function availableDays(): string[] {
  return fs.readdirSync(HISTORY_ROOT)
    .map((file) => /^(\d{4}-\d{2}-\d{2})\.json$/.exec(file)?.[1])
    .filter((date): date is string => Boolean(date))
    .sort();
}

function readReferenceDay(date: string): Candle[] | undefined {
  const file = path.join(HISTORY_ROOT, `${date}.json`);
  return fs.existsSync(file) ? readCandleShardReferenceSync(file) : undefined;
}

function compactResult(result: SuiteResult): Record<string, unknown> {
  const summary = result.summary;
  return {
    id: result.id,
    returnPct: summary.returnPct,
    netPnl: summary.netPnl,
    finalEquity: summary.finalEquity,
    maxDrawdownPct: summary.maxDrawdownPct,
    sharpeRatio: summary.sharpeRatio,
    tradeCount: summary.tradeCount,
    maxEffectiveLeverage: summary.maxEffectiveLeverage,
    perfectMarginCapturePct: summary.perfectMarginCapturePct,
    wallDurationMs: result.wallDurationMs,
  };
}

function aggregate(results: readonly SuiteResult[]): Record<string, unknown> {
  const returns = results.map((result) => result.summary.returnPct).sort((a, b) => a - b);
  const mean = returns.length > 0
    ? returns.reduce((sum, value) => sum + value, 0) / returns.length
    : 0;
  const median = returns.length === 0
    ? 0
    : returns.length % 2 === 1
      ? returns[Math.floor(returns.length / 2)]!
      : (returns[returns.length / 2 - 1]! + returns[returns.length / 2]!) / 2;
  const best = results.reduce<SuiteResult | undefined>(
    (current, result) => !current || result.summary.returnPct > current.summary.returnPct
      ? result
      : current,
    undefined,
  );
  const worst = results.reduce<SuiteResult | undefined>(
    (current, result) => !current || result.summary.returnPct < current.summary.returnPct
      ? result
      : current,
    undefined,
  );
  return {
    windows: results.length,
    profitableWindows: results.filter((result) => result.summary.netPnl > 0).length,
    meanReturnPct: mean,
    medianReturnPct: median,
    best: best ? { id: best.id, returnPct: best.summary.returnPct } : undefined,
    worst: worst ? { id: worst.id, returnPct: worst.summary.returnPct } : undefined,
    totalTrades: results.reduce((sum, result) => sum + result.summary.tradeCount, 0),
    totalCandles: results.reduce((sum, result) => sum + result.loadedCandles, 0),
    wallDurationMs: results.reduce((sum, result) => sum + result.wallDurationMs, 0),
  };
}

function readReport(file: string, valueHorizonMs: number): SuiteReport | undefined {
  if (!fs.existsSync(file)) return undefined;
  const parsed = JSON.parse(fs.readFileSync(file, "utf8")) as SuiteReport;
  if (
    parsed.strategy !== "hindsight-oracle-1s"
    || parsed.oracle?.intervalMs !== INTERVAL_MS
    || parsed.oracle?.holdingPeriodMs !== HINDSIGHT_ORACLE_HOLDING_PERIOD_MS
    || parsed.oracle?.decisionDelayMs !== HINDSIGHT_ORACLE_DECISION_DELAY_MS
    || parsed.oracle?.valueHorizonMs !== valueHorizonMs
    || parsed.oracle?.maximumExposure !== HINDSIGHT_ORACLE_MAX_EXPOSURE
    || parsed.oracle?.confidenceExposurePower !== HINDSIGHT_ORACLE_CONFIDENCE_EXPOSURE_POWER
    || parsed.oracle?.confidenceLeverageFloor !== HINDSIGHT_ORACLE_CONFIDENCE_LEVERAGE_FLOOR
    || !Array.isArray(parsed.results)
  ) {
    throw new Error(`Existing suite report is incompatible: ${file}`);
  }
  return parsed;
}

function writeReport(file: string, report: SuiteReport): void {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const temporary = `${file}.${process.pid}.tmp`;
  fs.writeFileSync(temporary, `${JSON.stringify(report, null, 2)}\n`, "utf8");
  fs.renameSync(temporary, file);
}

function argument(name: string): string | undefined {
  const prefix = `--${name}=`;
  return process.argv.find((value) => value.startsWith(prefix))?.slice(prefix.length);
}

function flag(name: string): boolean {
  return process.argv.includes(`--${name}`);
}

function integerArgument(name: string, fallback: number): number {
  const raw = argument(name);
  if (raw === undefined) return fallback;
  if (!/^\d+$/.test(raw)) throw new Error(`--${name} must be an integer.`);
  const value = Number(raw);
  if (!Number.isSafeInteger(value)) throw new Error(`--${name} is out of range.`);
  return value;
}

function parseDay(value: string): number {
  return Date.parse(`${value}T00:00:00.000Z`);
}

function isoDay(timestamp: number): string {
  return new Date(timestamp).toISOString().slice(0, 10);
}

function utcDay(timestamp: number): number {
  const date = new Date(timestamp);
  return Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate());
}

function subtractUtcMonths(timestamp: number, months: number): number {
  const date = new Date(timestamp);
  return Date.UTC(
    date.getUTCFullYear(),
    date.getUTCMonth() - months,
    date.getUTCDate(),
  );
}

void main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : String(error));
  process.exitCode = 1;
});
