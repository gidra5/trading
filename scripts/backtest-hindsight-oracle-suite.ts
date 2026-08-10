import fs from "node:fs";
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
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
  LEARNED_ORACLE_DEFAULT_MAXIMUM_LEVERAGE,
  runBotBacktestFromCandles,
} from "../apps/server/src/bot-backtest.js";
import { historicalWarmupSamples } from "../apps/server/src/historical-backtest.js";
import {
  fillOracleClosePrices,
  type OracleReturnNoiseConfig,
  validateOracleReturnNoise,
} from "./hindsight-oracle-return-noise.js";

const DAY_MS = 86_400_000;
const MODEL_PLAN = path.resolve(
  argument("model-plan")
    ?? "ml/training-plans/oracle-distribution-path-15m-two-layer-glu-mean-p50-v1.json",
);
const ORACLE_SOURCE = argument("oracle-source") ?? "hindsight";
if (ORACLE_SOURCE !== "hindsight" && ORACLE_SOURCE !== "model") {
  throw new Error("--oracle-source must be hindsight or model.");
}
const CANDLE_INTERVAL = argument("interval") ?? "1s";
const INTERVAL_MS = candleIntervalMs(CANDLE_INTERVAL);
if (ORACLE_SOURCE === "model" && CANDLE_INTERVAL !== "1m") {
  throw new Error("The oracle distribution-path model requires --interval=1m.");
}
const HISTORY_ROOT = path.resolve(
  `data/market/immutable/refs/candles/spot-btcusdt/btcusdt/${CANDLE_INTERVAL}`,
);
const DEFAULT_OUTPUT = path.resolve(
  ORACLE_SOURCE === "model"
    ? "data/benchmarks/oracle-distribution-path-1m-suite-diagnostics-2026-08-06.json"
    : CANDLE_INTERVAL === "1s"
    ? "data/benchmarks/hindsight-oracle-bot-suite-2026-07-31.json"
    : `data/benchmarks/hindsight-oracle-bot-suite-${CANDLE_INTERVAL}-2026-07-31.json`,
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
  inferenceDurationMs?: number;
  policyDiagnostics?: {
    decisions: number;
    rawNonzero: number;
    conditionedNonzero: number;
    targetNonzero: number;
    signalsEmitted: number;
    meanConfidence: number;
  };
  wallDurationMs: number;
  summary: BacktestSummary;
}

interface SuiteReport {
  strategy: "hindsight-oracle-1s" | "oracle-distribution-path-1m";
  summaryOnly: true;
  oracle: {
    intervalMs: number;
    holdingPeriodMs: number;
    decisionDelayMs: number;
    valueHorizonMs: number;
    maximumExposure: number;
    executionMaximumLeverage?: number;
    confidenceExposurePower: number;
    confidenceLeverageFloor: number;
    confidenceLeverageFloorScaling?: "quadratic-static-confidence";
    expansionConfirmationMass?: number;
    expansionConfirmationBasis?: "distribution-confidence-mass";
    expansionDeltaCapFraction?: number;
    staticConfidenceScale: number;
    distributionConfidenceGate?: "disabled";
    returnCorrelation?: number;
    returnNoiseSeed?: number;
    returnCorrelationBasis?: "rolling-value-horizon-log-return";
    source?: "hindsight" | "model";
    modelId?: string;
  };
  generatedAt: string;
  historyRoot: string;
  latestAvailableDay: string;
  modelWorkerStartupMs?: number;
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
  const modelPlan = ORACLE_SOURCE === "model"
    ? JSON.parse(fs.readFileSync(MODEL_PLAN, "utf8")) as { id: string }
    : undefined;
  const valueHorizonMinutes = integerArgument(
    "value-horizon-minutes",
    ORACLE_SOURCE === "model" ? 15 : HINDSIGHT_ORACLE_VALUE_HORIZON_MS / 60_000,
  );
  if (valueHorizonMinutes < HINDSIGHT_ORACLE_HOLDING_PERIOD_MS / 60_000) {
    throw new Error("--value-horizon-minutes must cover at least one holding period.");
  }
  const valueHorizonMs = valueHorizonMinutes * 60_000;
  const returnCorrelation = finiteArgument("oracle-return-correlation", 1);
  if (ORACLE_SOURCE === "model" && returnCorrelation !== 1) {
    throw new Error("Return-noise perturbation is only available for hindsight inference.");
  }
  const staticConfidenceScale = finiteArgument(
    "static-confidence",
    ORACLE_SOURCE === "model" ? 0.75 : returnCorrelation,
  );
  if (staticConfidenceScale < 0 || staticConfidenceScale > 1) {
    throw new Error("--static-confidence must be in [0, 1].");
  }
  const returnNoiseSeed = integerArgument("oracle-noise-seed", 0);
  const expansionConfirmationMass = finiteArgument("oracle-expansion-confirmation-mass", 1);
  if (expansionConfirmationMass < 0) {
    throw new Error("--oracle-expansion-confirmation-mass must be non-negative.");
  }
  const expansionDeltaCapFraction = finiteArgument(
    "oracle-expansion-delta-cap-fraction",
    HINDSIGHT_ORACLE_CONFIDENCE_LEVERAGE_FLOOR,
  );
  if (!(expansionDeltaCapFraction >= 0)) {
    throw new Error("--oracle-expansion-delta-cap-fraction must be non-negative.");
  }
  const returnNoise = returnCorrelation < 1
    ? {
        correlation: returnCorrelation,
        seed: returnNoiseSeed,
        rollingHorizonSteps: Math.round(valueHorizonMs / INTERVAL_MS),
      }
    : undefined;
  const oracleNoiseSourceHorizonMs = ORACLE_SOURCE === "model"
    ? 0
    : returnNoise
    ? Math.max(valueHorizonMs, 60 * 60_000)
    : valueHorizonMs;
  validateOracleReturnNoise({
    correlation: returnCorrelation,
    seed: returnNoiseSeed,
    rollingHorizonSteps: Math.round(valueHorizonMs / INTERVAL_MS),
  });
  const latestAvailableDay = ORACLE_SOURCE === "model"
    ? latestContiguousAvailableDay(93)
    : availableDays().at(-1);
  if (!latestAvailableDay) throw new Error(`No one-second history found in ${HISTORY_ROOT}.`);
  const latestEndTime = parseDay(latestAvailableDay) + DAY_MS;
  const latestMeasuredEndTime = ORACLE_SOURCE === "model"
    ? latestEndTime
    : latestEndTime - valueHorizonMs;
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
    `BACKEND source=${ORACLE_SOURCE} oracle=${oracleBackend} horizonMinutes=${valueHorizonMinutes}`
    + ` returnCorrelation=${returnCorrelation} noiseSeed=${returnNoiseSeed}`
    + ` staticConfidence=${staticConfidenceScale}`
    + `${cudaStatus?.device ? ` device=${cudaStatus.device}` : ""}`,
  );
  const existing = readReport(
    output,
    valueHorizonMs,
    returnCorrelation,
    returnNoiseSeed,
    expansionConfirmationMass,
    expansionDeltaCapFraction,
    ORACLE_SOURCE,
    modelPlan?.id,
    staticConfidenceScale,
  );
  const report: SuiteReport = existing ?? {
    strategy: ORACLE_SOURCE === "model"
      ? "oracle-distribution-path-1m"
      : "hindsight-oracle-1s",
    summaryOnly: true,
    oracle: {
      intervalMs: INTERVAL_MS,
      holdingPeriodMs: HINDSIGHT_ORACLE_HOLDING_PERIOD_MS,
      decisionDelayMs: HINDSIGHT_ORACLE_DECISION_DELAY_MS,
      valueHorizonMs,
      maximumExposure: HINDSIGHT_ORACLE_MAX_EXPOSURE,
      executionMaximumLeverage: ORACLE_SOURCE === "model"
        ? LEARNED_ORACLE_DEFAULT_MAXIMUM_LEVERAGE
        : HINDSIGHT_ORACLE_MAX_EXPOSURE,
      confidenceExposurePower: HINDSIGHT_ORACLE_CONFIDENCE_EXPOSURE_POWER,
      confidenceLeverageFloor: HINDSIGHT_ORACLE_CONFIDENCE_LEVERAGE_FLOOR,
      confidenceLeverageFloorScaling: "quadratic-static-confidence",
      expansionConfirmationMass,
      expansionConfirmationBasis: "distribution-confidence-mass",
      expansionDeltaCapFraction,
      staticConfidenceScale,
      distributionConfidenceGate: "disabled",
      returnCorrelation,
      returnNoiseSeed,
      returnCorrelationBasis: "rolling-value-horizon-log-return",
      source: ORACLE_SOURCE,
      ...(modelPlan ? { modelId: modelPlan.id } : {}),
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
  const modelWorker = ORACLE_SOURCE === "model"
    ? new OracleDistributionPathWorker(MODEL_PLAN)
    : undefined;

  for (const [index, window] of selected.entries()) {
    if (completed.has(window.id)) {
      console.log(`SKIP ${window.id} (${index + 1}/${selected.length})`);
      continue;
    }
    console.log(
      `START ${window.id} (${index + 1}/${selected.length}) `
      + `${isoDay(window.startTime)}..${isoDay(window.endTime - 1)}`,
    );
    const loaded = loadWindow(window, warmupMs, oracleNoiseSourceHorizonMs);
    console.log(
      `LOADED ${window.id} measured=${loaded.candles.length} `
      + `warmup=${loaded.warmup.length} future=${loaded.oracleFuture.length}`,
    );
    const startedAt = Date.now();
    const inferenceStartedAt = Date.now();
    const reusedFitOracle = reusableFitOracle?.covers(loaded.candles) ?? false;
    const precomputed = reusedFitOracle
      ? reusableFitOracle
      : modelWorker
        ? await modelWorker.predict(window, loaded.candles)
        : await precomputeOracleDistributions(
          loaded.candles,
          loaded.oracleFuture,
          appConfig.strategy,
          oracleWorkers,
          window.id,
          oracleBackend,
          valueHorizonMs,
          returnNoise,
        );
    if (reusedFitOracle) {
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
    const inferenceDurationMs = reusedFitOracle
      ? 0
      : Date.now() - inferenceStartedAt;
    const decisionDiagnostics = {
      decisions: 0,
      rawNonzero: 0,
      conditionedNonzero: 0,
      targetNonzero: 0,
      signalsEmitted: 0,
      confidenceSum: 0,
    };
    const result = await runBotBacktestFromCandles(loaded.candles, {
      config: appConfig.strategy,
      strategy: ORACLE_SOURCE === "model"
        ? "learned-oracle-1m"
        : "hindsight-oracle-1s",
      warmup: loaded.warmup,
      oracleFuture: loaded.oracleFuture,
      maxEquityPoints: 800,
      maxChartCandles: 2_000,
      summaryOnly: true,
      hindsightOracleDistributionAt: ORACLE_SOURCE === "hindsight"
        ? precomputed?.distributionAt
        : undefined,
      learnedOracleDistributionAt: ORACLE_SOURCE === "model"
        ? precomputed?.distributionAt
        : undefined,
      oracleStaticConfidenceScale: staticConfidenceScale,
      oracleExpansionConfirmationMass: expansionConfirmationMass,
      oracleExpansionDeltaCapFraction: expansionDeltaCapFraction,
      onOracleDecision: (decision) => {
        decisionDiagnostics.decisions += 1;
        decisionDiagnostics.rawNonzero += Number(decision.rawModalExposure !== 0);
        decisionDiagnostics.conditionedNonzero += Number(
          decision.conditionedModalExposure !== 0,
        );
        decisionDiagnostics.targetNonzero += Number(decision.targetExposure !== 0);
        decisionDiagnostics.signalsEmitted += Number(decision.signalEmitted);
        decisionDiagnostics.confidenceSum += decision.confidence;
      },
      onProgress: flag("quiet-replay")
        ? undefined
        : ({ candlesProcessed, totalCandles, elapsedMs }) => {
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
      inferenceDurationMs,
      policyDiagnostics: {
        decisions: decisionDiagnostics.decisions,
        rawNonzero: decisionDiagnostics.rawNonzero,
        conditionedNonzero: decisionDiagnostics.conditionedNonzero,
        targetNonzero: decisionDiagnostics.targetNonzero,
        signalsEmitted: decisionDiagnostics.signalsEmitted,
        meanConfidence: decisionDiagnostics.decisions > 0
          ? decisionDiagnostics.confidenceSum / decisionDiagnostics.decisions
          : 0,
      },
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
    await modelWorker?.close();
    return;
  }
  console.log(`AGGREGATE ${JSON.stringify(aggregate(report.results))}`);
  console.log(`REPORT ${output}`);
  if (modelWorker) {
    report.modelWorkerStartupMs = modelWorker.startupDurationMs;
    writeReport(output, report);
    await modelWorker.close();
  }
}

interface PrecomputedOracleDistributions {
  distributionAt(timestamp: number): ExposureValueOracleActionDistribution | null;
  covers(candles: readonly Candle[]): boolean;
}

class OracleDistributionPathWorker {
  readonly child: ChildProcessWithoutNullStreams;
  startupDurationMs = 0;
  private readonly startedAt = Date.now();
  private pending?: {
    expectedRows: number;
    buffer: Buffer;
    offset: number;
    resolve: (value: Buffer) => void;
    reject: (error: Error) => void;
  };

  constructor(planFile: string) {
    const workerPlan = JSON.parse(fs.readFileSync(planFile, "utf8")) as {
      serving?: { type?: string };
    };
    const workerScript = workerPlan.serving?.type === "causal-forward-direct"
      ? "ml/serve_forward_market_oracle.py"
      : workerPlan.serving?.type === "causal-multiscale-direct"
        ? "ml/serve_causal_multiscale_oracle.py"
        : "ml/serve_oracle_distribution_path.py";
    const python = path.resolve(
      ".venv-ml",
      process.platform === "win32" ? "Scripts/python.exe" : "bin/python",
    );
    this.child = spawn(python, [
      path.resolve(workerScript),
      "--plan",
      planFile,
      "--history-dir",
      HISTORY_ROOT,
      "--device",
      argument("model-device") ?? "auto",
      "--batch-size",
      argument("model-batch-size") ?? "4096",
    ], {
      cwd: path.resolve("."),
      env: {
        ...process.env,
        PYTHONUTF8: process.env.PYTHONUTF8 ?? "1",
        PYTHONIOENCODING: process.env.PYTHONIOENCODING ?? "utf-8",
        PYTHONPATH: [path.resolve("ml"), process.env.PYTHONPATH]
          .filter(Boolean).join(path.delimiter),
      },
      stdio: ["pipe", "pipe", "pipe"],
    });
    this.child.stdout.on("data", (chunk: Buffer) => this.accept(chunk));
    this.child.stderr.on("data", (chunk: Buffer) => {
      process.stderr.write(chunk);
    });
    this.child.once("error", (error) => this.fail(error));
    this.child.once("exit", (code) => {
      if (code && this.pending) {
        this.fail(new Error(`Oracle distribution-path worker exited with code ${code}.`));
      }
    });
  }

  async predict(
    window: SuiteWindow,
    candles: readonly Candle[],
  ): Promise<PrecomputedOracleDistributions> {
    if (this.pending) throw new Error("Oracle model worker already has an active request.");
    const expectedRows = Math.round((window.endTime - window.startTime) / INTERVAL_MS);
    if (candles.length !== expectedRows
      || candles[0]?.openTime !== window.startTime
      || candles.at(-1)?.openTime !== window.endTime - INTERVAL_MS) {
      throw new Error(`Model window ${window.id} is not a complete minute range.`);
    }
    const actionCount = 101;
    const expectedBytes = 16 + expectedRows * actionCount * Float32Array.BYTES_PER_ELEMENT;
    const startedAt = Date.now();
    const response = await new Promise<Buffer>((resolve, reject) => {
      this.pending = {
        expectedRows,
        buffer: Buffer.allocUnsafe(expectedBytes),
        offset: 0,
        resolve,
        reject,
      };
      this.child.stdin.write(`${JSON.stringify({
        id: window.id,
        startTime: window.startTime,
        endTime: window.endTime,
      })}\n`);
    });
    const rows = response.readUInt32LE(0);
    const columns = response.readUInt32LE(4);
    const workerDurationMs = Number(response.readBigUInt64LE(8));
    if (rows !== expectedRows || columns !== actionCount) {
      throw new Error(
        `Model response for ${window.id} has ${rows}x${columns}; expected `
        + `${expectedRows}x${actionCount}.`,
      );
    }
    if (this.startupDurationMs === 0) {
      this.startupDurationMs = Math.max(0, Date.now() - startedAt - workerDurationMs);
    }
    const probabilities = new Float32Array(
      response.buffer,
      response.byteOffset + 16,
      rows * columns,
    );
    const grid = Float64Array.from(
      { length: columns },
      (_, index) => -100 + index * 200 / (columns - 1),
    );
    const firstCloseTime = candles[0]!.closeTime;
    const lastCloseTime = candles.at(-1)!.closeTime;
    console.log(
      `MODEL ${window.id} rows=${rows} durationMs=${workerDurationMs} `
      + `rowsPerSecond=${Math.round(rows / Math.max(workerDurationMs / 1_000, 0.001))}`,
    );
    return {
      covers(candidate) {
        const first = candidate[0]?.closeTime;
        const last = candidate.at(-1)?.closeTime;
        return first !== undefined && last !== undefined
          && first >= firstCloseTime && last <= lastCloseTime
          && (first - firstCloseTime) % INTERVAL_MS === 0;
      },
      distributionAt(timestamp) {
        const elapsed = timestamp - firstCloseTime;
        if (elapsed < 0 || elapsed % INTERVAL_MS !== 0 || timestamp > lastCloseTime) {
          return null;
        }
        const row = elapsed / INTERVAL_MS;
        const values = probabilities.subarray(row * columns, (row + 1) * columns);
        let total = 0;
        let mean = 0;
        let secondMoment = 0;
        let entropy = 0;
        let modalIndex = 0;
        let feasibleActionCount = 0;
        for (let index = 0; index < columns; index += 1) {
          const probability = values[index]!;
          total += probability;
          if (probability > values[modalIndex]!) modalIndex = index;
          if (probability > 0) {
            feasibleActionCount += 1;
            entropy -= probability * Math.log(probability);
          }
          mean += probability * grid[index]!;
          secondMoment += probability * grid[index]! ** 2;
        }
        if (!(total > 0) || !Number.isFinite(total)) {
          throw new Error(`Model distribution is invalid at ${timestamp}.`);
        }
        return {
          grid,
          probabilities: values,
          mean: mean / total,
          secondMoment: secondMoment / total,
          modalExposure: grid[modalIndex]!,
          entropy: entropy / total + Math.log(total),
          opportunity: 0,
          feasibleActionCount,
        };
      },
    };
  }

  async close(): Promise<void> {
    if (this.child.exitCode !== null) return;
    this.child.stdin.end();
    await new Promise<void>((resolve, reject) => {
      this.child.once("exit", (code) => code === 0
        ? resolve()
        : reject(new Error(`Oracle distribution-path worker exited with code ${code}.`)));
    });
  }

  private accept(chunk: Buffer): void {
    const pending = this.pending;
    if (!pending) {
      this.child.kill();
      throw new Error("Oracle model worker produced an unsolicited response.");
    }
    if (chunk.length > pending.buffer.length - pending.offset) {
      this.fail(new Error("Oracle model worker response exceeds its declared window."));
      return;
    }
    chunk.copy(pending.buffer, pending.offset);
    pending.offset += chunk.length;
    if (pending.offset === pending.buffer.length) {
      this.pending = undefined;
      pending.resolve(pending.buffer);
    }
  }

  private fail(error: Error): void {
    const pending = this.pending;
    this.pending = undefined;
    pending?.reject(error);
  }
}

async function precomputeOracleDistributions(
  candles: readonly Candle[],
  future: readonly Candle[],
  config: StrategyConfig,
  workerCount: number,
  windowId: string,
  backend: OracleBackend,
  valueHorizonMs: number,
  returnNoise?: OracleReturnNoiseConfig,
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
  fillOracleClosePrices(prices, candles, future, returnNoise);
  const decisionCount = candles.length > 1
    ? Math.floor((candles.length - 2) / holdingPeriodSteps) + 1
    : 0;
  const storedOracle = storedOracleDefinition(config, valueHorizonSteps, !returnNoise);
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
  }), storedOracle.grid, returnNoise);
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
  returnNoise?: OracleReturnNoiseConfig,
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
    ...(returnNoise ? {
      returnNoise: {
        kind: "orthogonalized-rolling-horizon-return-correlation-v4",
        correlation: returnNoise.correlation,
        seed: returnNoise.seed,
        rollingHorizonSteps: returnNoise.rollingHorizonSteps,
      },
    } : {}),
    usableGrid: Array.from(grid),
  };
  const contractHash = createHash("sha256")
    .update(JSON.stringify(metadata))
    .digest("hex");
  return {
    backend,
    namespace: `oracle/${CANDLE_INTERVAL}/hindsight-bot-${contractHash.slice(0, 20)}`,
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
        const openCode = (error as NodeJS.ErrnoException).code;
        // Windows can report an existing lock opened with `wx` as EPERM/EACCES/EBUSY
        // while another process still owns the descriptor, rather than EEXIST.
        if (openCode !== "EEXIST" && openCode !== "EPERM"
          && openCode !== "EACCES" && openCode !== "EBUSY") {
          throw error;
        }
        let owner = Number.NaN;
        try {
          owner = Number.parseInt(fs.readFileSync(lockFile, "utf8"), 10);
        } catch (readError) {
          const code = (readError as NodeJS.ErrnoException).code;
          if (code !== "ENOENT" && code !== "EPERM" && code !== "EACCES" && code !== "EBUSY") {
            throw readError;
          }
          continue;
        }
        const stale = Number.isInteger(owner)
          ? !processExists(owner)
          : malformedLockIsStale(lockFile);
        if (stale) {
          fs.rmSync(lockFile, { force: true });
        }
      }
    }
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
}

function malformedLockIsStale(lockFile: string): boolean {
  try {
    return Date.now() - fs.statSync(lockFile).mtimeMs >= 30_000;
  } catch (error) {
    const code = (error as NodeJS.ErrnoException).code;
    if (code === "ENOENT" || code === "EPERM" || code === "EACCES" || code === "EBUSY") {
      return false;
    }
    throw error;
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
  allowStoredRows: boolean,
): StoredOracleDefinition {
  const manifest = JSON.parse(fs.readFileSync(STORED_ORACLE_DATASET, "utf8"));
  const execution = manifest.execution;
  const frictionBps = config.feeBps + config.positionRisk.marketSlippageBps;
  const compatible = allowStoredRows && !(
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

function latestContiguousAvailableDay(minimumDays: number): string | undefined {
  const dates = availableDays();
  const available = new Set(dates);
  for (let candidate = dates.length - 1; candidate >= 0; candidate -= 1) {
    const end = parseDay(dates[candidate]!);
    let complete = true;
    for (let offset = 1; offset < minimumDays; offset += 1) {
      if (!available.has(isoDay(end - offset * DAY_MS))) {
        complete = false;
        break;
      }
    }
    if (complete) return dates[candidate];
  }
  return undefined;
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
    inferenceDurationMs: results.reduce(
      (sum, result) => sum + (result.inferenceDurationMs ?? 0), 0,
    ),
    wallDurationMs: results.reduce((sum, result) => sum + result.wallDurationMs, 0),
  };
}

function readReport(
  file: string,
  valueHorizonMs: number,
  returnCorrelation: number,
  returnNoiseSeed: number,
  expansionConfirmationMass: number,
  expansionDeltaCapFraction: number,
  source: "hindsight" | "model",
  modelId: string | undefined,
  staticConfidenceScale: number,
): SuiteReport | undefined {
  if (!fs.existsSync(file)) return undefined;
  const parsed = JSON.parse(fs.readFileSync(file, "utf8")) as SuiteReport;
  if (
    parsed.strategy !== (source === "model"
      ? "oracle-distribution-path-1m"
      : "hindsight-oracle-1s")
    || parsed.oracle?.intervalMs !== INTERVAL_MS
    || parsed.oracle?.holdingPeriodMs !== HINDSIGHT_ORACLE_HOLDING_PERIOD_MS
    || parsed.oracle?.decisionDelayMs !== HINDSIGHT_ORACLE_DECISION_DELAY_MS
    || parsed.oracle?.valueHorizonMs !== valueHorizonMs
    || parsed.oracle?.maximumExposure !== HINDSIGHT_ORACLE_MAX_EXPOSURE
    || parsed.oracle?.confidenceExposurePower !== HINDSIGHT_ORACLE_CONFIDENCE_EXPOSURE_POWER
    || parsed.oracle?.confidenceLeverageFloor !== HINDSIGHT_ORACLE_CONFIDENCE_LEVERAGE_FLOOR
    || parsed.oracle?.confidenceLeverageFloorScaling !== "quadratic-static-confidence"
    || parsed.oracle?.expansionConfirmationMass !== expansionConfirmationMass
    || parsed.oracle?.expansionConfirmationBasis !== "distribution-confidence-mass"
    || parsed.oracle?.expansionDeltaCapFraction !== expansionDeltaCapFraction
    || parsed.oracle?.staticConfidenceScale !== staticConfidenceScale
    || parsed.oracle?.distributionConfidenceGate !== "disabled"
    || (parsed.oracle?.returnCorrelation ?? 1) !== returnCorrelation
    || (parsed.oracle?.returnNoiseSeed ?? 0) !== returnNoiseSeed
    || parsed.oracle?.returnCorrelationBasis !== "rolling-value-horizon-log-return"
    || (parsed.oracle?.source ?? "hindsight") !== source
    || parsed.oracle?.modelId !== modelId
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

function finiteArgument(name: string, fallback: number): number {
  const raw = argument(name);
  if (raw === undefined) return fallback;
  const value = Number(raw);
  if (!Number.isFinite(value)) throw new Error(`--${name} must be finite.`);
  return value;
}

function candleIntervalMs(interval: string): number {
  if (interval === "1s") return 1_000;
  if (interval === "1m") return 60_000;
  throw new Error("--interval must be 1s or 1m.");
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
