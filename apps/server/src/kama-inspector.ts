import fs from "node:fs/promises";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Worker } from "node:worker_threads";
import {
  VolumeWeightedKAMAIndicator,
  evaluateVwKamaOracle,
  noiseSignalRatio,
  perfectMarginOracle,
  volumeWeightedKamaWarmupSamples,
  vwKamaScore,
  VW_KAMA_SCORE_VERSION,
  type Candle,
  type BacktestOraclePoint,
  type BacktestChartSmaSeries,
  type PerfectMarginOracleResult,
  type TradingApi,
  type VwKamaAccuracyMetrics,
  type VwKamaCandleRangeRequest,
  type VwKamaCandleRangeResponse,
  type VwKamaEvaluation,
  type VwKamaInspectorCatalog,
  type VwKamaInspectorRequest,
  type VwKamaInspectorResponse,
  type VwKamaInspectorWindow,
  type VwKamaPreset,
  type VwKamaTransition,
} from "@trading/bot-algo";

const DAY_MS = 86_400_000;
const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../..");
const MAX_WARMUP_MS = 3 * DAY_MS;
const MAX_CHART_CANDLES = 2_000;
const MIN_RANGE_CANDLES = 100;
const MAX_RANGE_CANDLES = 5_000;
const SCALES = [1_000, 5_000, 15_000, 60_000, 300_000, 900_000, 3_600_000]
  .map((intervalMs) => ({ label: duration(intervalMs), intervalMs }));
const WINDOWS = [
  window("sideways-churn-2022-07", "Sideways · highest OHLC churn", "Sideways / choppy", "2022-07-28", "2022-08-03"),
  window("sideways-churn-2022-05", "Sideways · highest close churn", "Sideways / choppy", "2022-05-14", "2022-05-20"),
  window("sideways-churn-2021-12", "Sideways · very choppy (Dec 2021)", "Sideways / choppy", "2021-12-14", "2021-12-20"),
  window("sideways-churn-2021-09", "Sideways · very choppy (Sep 2021)", "Sideways / choppy", "2021-09-08", "2021-09-14"),
  window("sideways-churn-2023-03", "Sideways · choppy near-flat", "Sideways / choppy", "2023-03-18", "2023-03-24"),
  window("regime-up-2023-03", "Uptrend · +35.95%", "Regime", "2023-03-11", "2023-03-17"),
  window("regime-flat-2026-04", "Sideways · +0.009%", "Regime", "2026-04-22", "2026-04-28"),
  window("regime-down-2022-06", "Downtrend · -33.26%", "Regime", "2022-06-12", "2022-06-18"),
  window("shape-up-low-2024-02", "Uptrend · low churn", "3-day shape", "2024-02-24", "2024-02-26"),
  window("shape-up-high-2022-06", "Uptrend · high churn", "3-day shape", "2022-06-19", "2022-06-21"),
  window("shape-down-low-2023-06", "Downtrend · low churn", "3-day shape", "2023-06-03", "2023-06-05"),
  window("shape-down-high-2022-06", "Downtrend · high churn", "3-day shape", "2022-06-13", "2022-06-15"),
  window("shape-flat-high-bias-2021-10", "Sideways · high bias / churn", "3-day shape", "2021-10-19", "2021-10-21"),
  window("shape-flat-high-bias-low-2025-02", "Sideways · high bias / low churn", "3-day shape", "2025-02-14", "2025-02-16"),
  window("shape-flat-low-bias-2024-07", "Sideways · low bias / churn", "3-day shape", "2024-07-07", "2024-07-09"),
  window("shape-flat-low-bias-low-2025-07", "Sideways · low bias / low churn", "3-day shape", "2025-07-04", "2025-07-06"),
  window("shape-flat-mid-bias-2024-01", "Sideways · mid bias / churn", "3-day shape", "2024-01-02", "2024-01-04"),
  window("shape-flat-mid-bias-low-2023-09", "Sideways · mid bias / low churn", "3-day shape", "2023-09-15", "2023-09-17"),
  window("sharpe-up-3d-2024-11", "Uptrend · high Sharpe (3d, Nov 2024)", "Sharpe", "2024-11-09", "2024-11-11"),
  window("sharpe-up-3d-2023-12", "Uptrend · high Sharpe (3d, Dec 2023)", "Sharpe", "2023-12-03", "2023-12-05"),
  window("sharpe-down-3d-2026-06", "Downtrend · high Sharpe (3d, Jun 2026)", "Sharpe", "2026-06-01", "2026-06-03"),
  window("sharpe-down-3d-2023-03", "Downtrend · high Sharpe (3d, Mar 2023)", "Sharpe", "2023-03-07", "2023-03-09"),
  window("sharpe-up-7d-2023-12", "Uptrend · high Sharpe (7d, Dec 2023)", "Sharpe", "2023-11-29", "2023-12-05"),
  window("sharpe-up-7d-2024-11", "Uptrend · high Sharpe (7d, Nov 2024)", "Sharpe", "2024-11-05", "2024-11-11"),
  window("sharpe-down-7d-2023-03", "Downtrend · high Sharpe (7d, Mar 2023)", "Sharpe", "2023-03-03", "2023-03-09"),
  window("sharpe-down-7d-2026-06", "Downtrend · high Sharpe (7d, Jun 2026)", "Sharpe", "2026-05-27", "2026-06-02"),
  window("failure-down-3d-2022-06", "Known miss · -22.7% (3d)", "Known miss", "2022-06-11", "2022-06-13"),
  window("failure-down-7d-2022-06", "Known miss · June selloff (7d)", "Known miss", "2022-06-07", "2022-06-13"),
] satisfies VwKamaInspectorWindow[];

const BASELINE_GLOBAL_PARAMETERS = {
  efficiencyMs: 265_846,
  fastMs: 1_113_418,
  slowMs: 64_800_000,
  power: 0.62547,
  volumeMs: 283_525,
  volumeCap: 5.08117,
  volumePower: 0.48543,
  deadbandBpsHour: 30,
  deadbandMode: "hold" as const,
  hysteresisReleaseRatio: 0.25,
  thresholdMode: "adaptive" as const,
  thresholdLookbackMs: 2_457_073,
  thresholdNoiseMultiplier: 0,
  buyMaxFraction: 1,
  sellMaxFraction: 1,
  buySizingSigmaBpsHour: 1e12,
  sellSizingSigmaBpsHour: 1e12,
  agreementMode: "sizing" as const,
  confirmationMix: 0,
  confirmationMinQuality: 0,
  confirmationAccelerationLookbackMs: 60 * 60_000,
  confirmationDistanceLookbackMs: 60 * 60_000,
  confirmationAccelerationWeight: 1,
  confirmationDistanceWeight: 1,
  confirmationEmaMs: 60 * 60_000,
  confirmationEmaThresholdBpsHour: 0,
  confirmationEmaWeight: 0,
  confirmationEmaGateStrength: 0,
  confirmationRsiMs: 14 * 60_000,
  confirmationRsiThreshold: 0,
  confirmationRsiWeight: 0,
  confirmationDmiMs: 14 * 60_000,
  confirmationDmiWeight: 0,
  confirmationAdxThreshold: 20,
  confirmationBias: 0,
  meanReversionMix: 0,
  meanReversionMeanMs: 60 * 60_000,
  meanReversionVolatilityMs: 60 * 60_000,
  meanReversionThreshold: 2,
};

const DEFAULT_REQUEST: VwKamaInspectorRequest = {
  windowId: "shape-up-low-2024-02",
  intervalMs: 1_000,
  parameters: structuredClone(BASELINE_GLOBAL_PARAMETERS),
  oracleFriction: 0.00175,
  matchWindowMs: 2 * 3_600_000,
  timingHalfLifeMs: 10 * 60_000,
  warmupMultiple: 3,
};

const GLOBAL_PRESETS: VwKamaPreset[] = [
  {
    id: "global-clean-v0182",
    label: "Global · score-v2 v0182",
    scope: "global",
    windowId: null,
    parameters: structuredClone(BASELINE_GLOBAL_PARAMETERS),
    source: "Score-v2 chronological validation winner; full sizing",
  },
  {
    id: "global-clean-k0050",
    label: "Global · score-v2 k0050",
    scope: "global",
    windowId: null,
    parameters: {
      efficiencyMs: 1_420_250,
      fastMs: 976_600,
      slowMs: 976_600,
      power: 0.79612,
      volumeMs: 82_113,
      volumeCap: 4.27354,
      volumePower: 0,
      deadbandBpsHour: 21.21468,
      deadbandMode: "hold",
      thresholdMode: "static",
      thresholdLookbackMs: 900_000,
      thresholdNoiseMultiplier: 5.60731,
      buyMaxFraction: 1,
      sellMaxFraction: 1,
      buySizingSigmaBpsHour: 1e12,
      sellSizingSigmaBpsHour: 1e12,
    },
    source: "Score-v2 chronological canonical finalist; full sizing",
  },
  {
    id: "global-runtime-k0021",
    label: "Global · runtime k0021",
    scope: "global",
    windowId: null,
    parameters: {
      efficiencyMs: 14 * 60_000,
      fastMs: 28 * 60_000,
      slowMs: 153 * 60_000,
      power: 0.49045,
      volumeMs: 130 * 60_000,
      volumeCap: 2.65003,
      volumePower: 0,
      deadbandBpsHour: 67.56654,
      deadbandMode: "flat",
      thresholdMode: "static",
      thresholdLookbackMs: 60 * 60_000,
      thresholdNoiseMultiplier: 0,
    },
    source: "Validation-selected multi-scale baseline",
  },
  {
    id: "global-refined-v0016",
    label: "Global · refined v0016",
    scope: "global",
    windowId: null,
    parameters: {
      efficiencyMs: 1_856_036,
      fastMs: 1_873_481,
      slowMs: 3_701_806,
      power: 0.61357,
      volumeMs: 440_028,
      volumeCap: 1.97271,
      volumePower: 1.72738,
      deadbandBpsHour: 0.44351,
      deadbandMode: "hold",
      thresholdMode: "static",
      thresholdLookbackMs: 60 * 60_000,
      thresholdNoiseMultiplier: 0,
    },
    source: "Refined one-to-one 60/40 validation winner",
  },
];

interface CachedWindow {
  id: string;
  source: Promise<Candle[]>;
  scales: Map<number, Promise<Candle[][]>>;
  oracles: Map<string, Promise<PerfectMarginOracleResult>>;
  kama: Map<string, Promise<Float64Array[]>>;
}

type InspectorResult = VwKamaInspectorResponse | VwKamaCandleRangeResponse;

interface PendingRequest {
  resolve: (result: InspectorResult) => void;
  reject: (error: Error) => void;
}

interface InspectorWorkerResponse {
  id: number;
  result?: InspectorResult;
  error?: string;
}

export class KamaInspector {
  private worker: Worker | null = null;
  private nextId = 1;
  private readonly pending = new Map<number, PendingRequest>();

  constructor(private readonly dataDir: string) {}

  catalog(): VwKamaInspectorCatalog {
    return inspectorCatalog(this.dataDir);
  }

  analyze(input: VwKamaInspectorRequest): Promise<VwKamaInspectorResponse> {
    return this.request("analyze", input);
  }

  candles(input: VwKamaCandleRangeRequest): Promise<VwKamaCandleRangeResponse> {
    return this.request("candles", input);
  }

  private request<T extends InspectorResult>(
    type: "analyze" | "candles",
    input: VwKamaInspectorRequest | VwKamaCandleRangeRequest,
  ): Promise<T> {
    const worker = this.getWorker();
    const id = this.nextId++;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve: (result) => resolve(result as T), reject });
      worker.postMessage({ type, id, input });
    });
  }

  async close(): Promise<void> {
    const worker = this.worker;
    this.worker = null;
    const error = new Error("VW-KAMA inspector stopped");
    for (const pending of this.pending.values()) pending.reject(error);
    this.pending.clear();
    if (worker) await worker.terminate();
  }

  private getWorker(): Worker {
    if (this.worker) return this.worker;
    const source = import.meta.url.endsWith(".ts")
      ? new URL("./kama-inspector-worker.ts", import.meta.url)
      : new URL("./kama-inspector-worker.js", import.meta.url);
    const worker = import.meta.url.endsWith(".ts")
      ? new Worker(
          `import("tsx/esm/api").then(({tsImport}) => tsImport(${JSON.stringify(source.href)}, {parentURL:${JSON.stringify(import.meta.url)}}));`,
          { eval: true },
        )
      : new Worker(source, { execArgv: [] });
    worker.on("message", (message: InspectorWorkerResponse) => {
      const pending = this.pending.get(message.id);
      if (!pending) return;
      this.pending.delete(message.id);
      if (message.result) pending.resolve(message.result);
      else pending.reject(new Error(message.error ?? "VW-KAMA worker failed"));
    });
    worker.on("error", (error) => this.failWorker(worker, error));
    worker.on("exit", (code) => {
      if (code !== 0) this.failWorker(worker, new Error(`VW-KAMA worker exited with code ${code}`));
      else if (this.worker === worker) this.worker = null;
    });
    worker.postMessage({ type: "init", dataDir: this.dataDir });
    this.worker = worker;
    return worker;
  }

  private failWorker(worker: Worker, error: Error): void {
    if (this.worker !== worker) return;
    this.worker = null;
    for (const pending of this.pending.values()) pending.reject(error);
    this.pending.clear();
  }
}

export class KamaInspectorEngine {
  private cache: CachedWindow | null = null;

  constructor(private readonly dataDir: string) {}

  catalog(): VwKamaInspectorCatalog {
    return inspectorCatalog(this.dataDir);
  }

  async analyze(input: VwKamaInspectorRequest): Promise<VwKamaInspectorResponse> {
    const startedAt = performance.now();
    const request = normalizeRequest(input);
    const selected = WINDOWS.find((item) => item.id === request.windowId);
    if (!selected) throw new Error(`Unknown VW-KAMA window: ${request.windowId}`);
    const cached = this.cachedWindow(selected);
    const segments = await this.scaledSegments(cached, request.intervalMs);
    const warmupMs = candidateWarmupMs(request);
    const evaluated: Array<{ result: VwKamaEvaluation; candles: Candle[]; scoreStart: number }> = [];

    for (const [index, candles] of segments.entries()) {
      const scoreStart = Math.max(selected.startTime, candles[0]!.openTime + (index > 0 ? warmupMs : 0));
      if (scoreStart >= selected.endTime || candles.at(-1)!.openTime < scoreStart) continue;
      const oracle = await this.oracle(cached, request, candles);
      evaluated.push({
        candles,
        scoreStart,
        result: evaluateVwKamaOracle(candles, {
          ...request,
          scoreStartTime: scoreStart,
          maxPoints: Math.max(100, Math.floor(MAX_CHART_CANDLES / segments.length)),
          oracleResult: oracle,
        }),
      });
    }
    if (evaluated.length === 0) throw new Error("No continuous candles remain after VW-KAMA warmup.");

    const result = combineEvaluations(evaluated, request);
    return {
      window: { ...selected },
      intervalMs: request.intervalMs,
      sourceSegmentCount: segments.length,
      scoredSegmentCount: evaluated.length,
      elapsedMs: Math.round(performance.now() - startedAt),
      ...result,
    };
  }

  async candles(input: VwKamaCandleRangeRequest): Promise<VwKamaCandleRangeResponse> {
    const { request, selected } = normalizeCandleRangeRequest(input);
    const cached = this.cachedWindow(selected);
    const segments = await this.scaledSegments(cached, request.intervalMs);
    const values = await this.kamaValues(cached, request, selected, segments);
    const rendered = renderCandleRange(
      segments,
      request.intervalMs,
      request.startTime,
      request.endTime,
      request.maxCandles,
    );
    return {
      windowId: selected.id,
      intervalMs: request.intervalMs,
      renderIntervalMs: rendered.intervalMs,
      startTime: request.startTime,
      endTime: request.endTime,
      sourceCandleCount: rendered.sourceCount,
      candles: rendered.candles,
      kamaSeries: renderKamaSeries(segments, values, rendered.candles, request),
    };
  }

  private cachedWindow(selected: VwKamaInspectorWindow): CachedWindow {
    if (this.cache?.id === selected.id) return this.cache;
    const source = this.loadSource(selected);
    const cached = {
      id: selected.id,
      source,
      scales: new Map<number, Promise<Candle[][]>>(),
      oracles: new Map<string, Promise<PerfectMarginOracleResult>>(),
      kama: new Map<string, Promise<Float64Array[]>>(),
    };
    source.catch(() => {
      if (this.cache === cached) this.cache = null;
    });
    this.cache = cached;
    return cached;
  }

  private scaledSegments(cached: CachedWindow, intervalMs: number): Promise<Candle[][]> {
    const existing = cached.scales.get(intervalMs);
    if (existing) return existing;
    const pending = cached.source.then((candles) =>
      continuousSegments(aggregateCandles(candles, 1_000, intervalMs), intervalMs));
    cached.scales.set(intervalMs, pending);
    return pending;
  }

  private oracle(
    cached: CachedWindow,
    request: VwKamaInspectorRequest,
    candles: Candle[],
  ): Promise<PerfectMarginOracleResult> {
    const key = `${request.intervalMs}:${candles[0]!.openTime}:${request.oracleFriction}`;
    const existing = cached.oracles.get(key);
    if (existing) return existing;
    const pending = Promise.resolve(perfectMarginOracle(candles, {
      startingQuote: 1,
      leverage: 1,
      friction: request.oracleFriction,
      eventMode: "close",
      maxPathCandles: MAX_CHART_CANDLES,
    }));
    cached.oracles.set(key, pending);
    while (cached.oracles.size > 4) {
      const oldest = cached.oracles.keys().next().value as string | undefined;
      if (oldest === undefined) break;
      cached.oracles.delete(oldest);
    }
    return pending;
  }

  private kamaValues(
    cached: CachedWindow,
    request: VwKamaInspectorRequest,
    selected: VwKamaInspectorWindow,
    segments: Candle[][],
  ): Promise<Float64Array[]> {
    const parameters = request.parameters;
    const key = JSON.stringify([
      request.intervalMs,
      parameters.efficiencyMs,
      parameters.efficiencyVolumeEmaMs,
      parameters.efficiencyVolumePower,
      parameters.fastMs,
      parameters.slowMs,
      parameters.power,
      parameters.volumeMs,
      parameters.volumeCap,
      parameters.volumePower,
      request.warmupMultiple,
    ]);
    const existing = cached.kama.get(key);
    if (existing) return existing;
    const warmupMs = candidateWarmupMs(request);
    const pending = Promise.resolve(segments.map((candles, index) => calculateKama(
      candles,
      request,
      Math.max(selected.startTime, candles[0]!.openTime + (index > 0 ? warmupMs : 0)),
    )));
    cached.kama.set(key, pending);
    while (cached.kama.size > 2) {
      const oldest = cached.kama.keys().next().value as string | undefined;
      if (oldest === undefined) break;
      cached.kama.delete(oldest);
    }
    return pending;
  }

  private async loadSource(selected: VwKamaInspectorWindow): Promise<Candle[]> {
    const start = selected.startTime - MAX_WARMUP_MS;
    const root = path.join(this.dataDir, "historical", "spot-btcusdt", "btcusdt", "1s");
    const candles: Candle[] = [];
    for (let day = utcDay(start); day < selected.endTime; day += DAY_MS) {
      const date = new Date(day).toISOString().slice(0, 10);
      let content: string;
      try {
        content = await fs.readFile(path.join(root, `${date}.jsonl`), "utf8");
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code === "ENOENT") {
          throw new Error(`Missing BTCUSDT 1s shard ${date}; representative fetch is not complete.`);
        }
        throw error;
      }
      for (const line of content.split("\n")) {
        if (!line) continue;
        const candle = JSON.parse(line) as Candle;
        if (candle.openTime >= start && candle.openTime < selected.endTime) candles.push(candle);
      }
    }
    candles.sort((left, right) => left.openTime - right.openTime);
    if (candles.length === 0) throw new Error(`No BTCUSDT 1s candles for ${selected.label}.`);
    return candles;
  }
}

function inspectorCatalog(dataDir: string): VwKamaInspectorCatalog {
  return {
    windows: WINDOWS.map((item) => ({ ...item })),
    scales: SCALES.map((item) => ({ ...item })),
    defaults: structuredClone(DEFAULT_REQUEST),
    presets: [
      ...GLOBAL_PRESETS.map((item) => structuredClone(item)),
      ...loadGlobalPresets(dataDir),
      ...loadWindowPresets(dataDir),
    ],
  };
}

function loadGlobalPresets(dataDir: string): VwKamaPreset[] {
  return loadPresets(dataDir, "vw-kama-global-presets.json").filter((preset) =>
    preset.scope === "global" && preset.parameters).map((preset) => ({
      ...preset,
      historicalScore: preset.scoreVersion === VW_KAMA_SCORE_VERSION
        ? undefined
        : preset.score,
      score: preset.scoreVersion === VW_KAMA_SCORE_VERSION ? preset.score : undefined,
    }));
}

function loadWindowPresets(dataDir: string): VwKamaPreset[] {
  return loadPresets(dataDir, "vw-kama-window-presets.json").flatMap((preset) => {
    const window = WINDOWS.find((item) => item.id === preset.windowId);
    const scale = preset.intervalMs == null
      ? null
      : SCALES.find((item) => item.intervalMs === preset.intervalMs);
    return preset.scope === "window" && window && preset.parameters && (preset.intervalMs == null || scale)
      ? [{
          ...preset,
          historicalScore: preset.scoreVersion === VW_KAMA_SCORE_VERSION
            ? undefined
            : preset.score,
          score: preset.scoreVersion === VW_KAMA_SCORE_VERSION ? preset.score : undefined,
          label: `Window best found · ${window.label}${scale ? ` · ${scale.label}` : ""}`,
        }]
      : [];
  });
}

function loadPresets(dataDir: string, name: string): VwKamaPreset[] {
  const files = [path.join(dataDir, "benchmarks", name), path.join(REPO_ROOT, "data", "benchmarks", name)];
  for (const file of new Set(files)) try {
    const parsed = JSON.parse(readFileSync(file, "utf8"));
    return Array.isArray(parsed) ? parsed as VwKamaPreset[] : [];
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") continue;
    throw error;
  }
  return [];
}

function combineEvaluations(
  evaluated: Array<{ result: VwKamaEvaluation; candles: Candle[]; scoreStart: number }>,
  request: VwKamaInspectorRequest,
): Omit<
  VwKamaInspectorResponse,
  "window" | "intervalMs" | "sourceSegmentCount" | "scoredSegmentCount" | "elapsedMs"
> {
  const results = evaluated.map((item) => item.result);
  const candidateTransitions = results.flatMap((item) => item.candidateTransitions);
  const oracleTransitions = results.flatMap((item) => item.oracleTransitions);
  const candleCount = results.reduce((sum, item) => sum + item.candleCount, 0);
  const metrics = combineMetrics(results, candidateTransitions, oracleTransitions, candleCount, request.intervalMs);
  const scored = evaluated.flatMap((item) =>
    item.candles.filter((candle) => candle.openTime >= item.scoreStart));
  const statePoints = results.flatMap((item) => item.statePoints);
  const indicatorPoints = results.flatMap((item) => item.indicatorPoints);
  const rendered = renderCandleRange(
    continuousSegments(scored, request.intervalMs),
    request.intervalMs,
    scored[0]!.openTime,
    scored.at(-1)!.closeTime + 1,
    MAX_CHART_CANDLES,
  );
  return {
    candleCount,
    metrics,
    renderIntervalMs: rendered.intervalMs,
    candles: rendered.candles,
    kamaSeries: {
      index: -1,
      windowSec: request.parameters.slowMs / 1_000,
      label: "VW-KAMA",
      color: "#f472b6",
      points: results.flatMap((item) => item.kamaSeries.points),
    },
    annotations: sampleEven(results.flatMap((item) => item.annotations), 5_000),
    oracle: {
      mode: "fixed-notional",
      eventMode: "close",
      leverage: 1,
      friction: request.oracleFriction,
      points: results.flatMap((item) => item.oracle.points),
    },
    candidatePath: { points: candidatePath(evaluated) },
    statePoints,
    indicatorPoints,
    candidateTransitions,
    oracleTransitions,
  };
}

function candidatePath(
  evaluated: Array<{ result: VwKamaEvaluation; scoreStart: number }>,
): VwKamaInspectorResponse["candidatePath"]["points"] {
  return evaluated.flatMap(({ result, scoreStart }) => {
    const first = result.statePoints[0];
    const firstTransition = result.candidateTransitions[0];
    const initialState = firstTransition?.time === first?.time
      ? firstTransition.fromState
      : first?.candidate;
    const initial: BacktestOraclePoint[] = initialState ? [{
      time: scoreStart,
      price: 0,
      fromState: initialState,
      state: initialState,
      action: "hold" as const,
    }] : [];
    return [...initial, ...result.candidateTransitions.map((point): BacktestOraclePoint => ({
      time: point.time,
      price: point.price,
      fromState: point.fromState,
      state: point.state,
      action: point.fromState === "flat" ? "open" as const
        : point.state === "flat" ? "close" as const
        : "switch" as const,
    }))];
  }).sort((left, right) => left.time - right.time);
}

function sampleEven<T>(values: T[], limit: number): T[] {
  if (values.length <= limit) return values;
  const result: T[] = [];
  const step = (values.length - 1) / (limit - 1);
  for (let index = 0; index < limit; index += 1) {
    result.push(values[Math.round(index * step)]!);
  }
  return result;
}

function combineMetrics(
  results: VwKamaEvaluation[],
  candidates: VwKamaTransition[],
  oracle: VwKamaTransition[],
  candleCount: number,
  intervalMs: number,
): VwKamaAccuracyMetrics {
  const signalCount = candidates.length;
  const oracleCount = oracle.length;
  const matchedCount = results.reduce((sum, item) => sum + item.metrics.matchedCount, 0);
  const extraSignalCount = signalCount - matchedCount;
  const signalCleanliness = signalCount > 0 ? matchedCount / signalCount : 1;
  const timingCredit = candidates.reduce((sum, item) => sum + item.timingCredit, 0);
  const precision = eventRatio(timingCredit, signalCount, oracleCount);
  const recall = eventRatio(timingCredit, oracleCount, signalCount);
  const f1 = harmonic(precision, recall);
  const exposureAgreement = results.reduce(
    (sum, item) => sum + item.metrics.exposureAgreement * item.candleCount,
    0,
  ) / candleCount;
  const lags = candidates.flatMap((item) => item.lagMs === null ? [] : [item.lagMs]);
  const absoluteLags = lags.map(Math.abs);
  return {
    score: vwKamaScore(f1, exposureAgreement, signalCleanliness),
    precision,
    recall,
    f1,
    rawPrecision: eventRatio(matchedCount, signalCount, oracleCount),
    rawRecall: eventRatio(matchedCount, oracleCount, signalCount),
    exposureAgreement,
    noiseSignalRatio: noiseSignalRatio(extraSignalCount, matchedCount),
    signalCleanliness,
    signalsPerDay: signalCount / Math.max(intervalMs / DAY_MS, candleCount * intervalMs / DAY_MS),
    signalCount,
    oracleCount,
    matchedCount,
    extraSignalCount,
    missedOracleCount: oracleCount - matchedCount,
    lagP50Ms: percentile(absoluteLags, 0.5),
    lagP90Ms: percentile(absoluteLags, 0.9),
    lagP95Ms: percentile(absoluteLags, 0.95),
    lagMedianSignedMs: percentile(lags, 0.5),
  };
}

function normalizeRequest(input: VwKamaInspectorRequest): VwKamaInspectorRequest {
  if (!SCALES.some((item) => item.intervalMs === input.intervalMs)) {
    throw new Error(`Unsupported VW-KAMA scale: ${input.intervalMs}`);
  }
  const request = structuredClone(input);
  if (!["flat", "hold", "hysteresis"].includes(request.parameters.deadbandMode)) {
    throw new Error("VW-KAMA deadband mode must be flat, hold, or hysteresis.");
  }
  if (request.parameters.agreementMode !== undefined
    && request.parameters.agreementMode !== "sizing"
    && request.parameters.agreementMode !== "confidence") {
    throw new Error("VW-KAMA agreement mode must be sizing or confidence.");
  }
  request.parameters.agreementMode ??= "sizing";
  request.parameters.efficiencyVolumeEmaMs ??= request.parameters.volumeMs;
  request.parameters.efficiencyVolumePower ??= 0;
  request.parameters.thresholdMode = request.parameters.thresholdMode === "adaptive" ? "adaptive" : "static";
  request.parameters.thresholdLookbackMs ??= 60 * 60_000;
  request.parameters.thresholdNoiseMultiplier ??= 0;
  request.parameters.buyMaxFraction ??= 1;
  request.parameters.sellMaxFraction ??= 1;
  request.parameters.buySizingSigmaBpsHour ??= 1e12;
  request.parameters.sellSizingSigmaBpsHour ??= 1e12;
  request.parameters.hysteresisReleaseRatio ??= 0.25;
  request.parameters.confirmationMix ??= 0;
  request.parameters.confirmationMinQuality ??= 0;
  request.parameters.confirmationAccelerationLookbackMs ??= 60 * 60_000;
  request.parameters.confirmationDistanceLookbackMs ??= 60 * 60_000;
  request.parameters.confirmationAccelerationWeight ??= 1;
  request.parameters.confirmationDistanceWeight ??= 1;
  request.parameters.confirmationEmaMs ??= 60 * 60_000;
  request.parameters.confirmationEmaThresholdBpsHour ??= 0;
  request.parameters.confirmationEmaWeight ??= 0;
  request.parameters.confirmationEmaGateStrength ??= 0;
  request.parameters.confirmationRsiMs ??= 14 * 60_000;
  request.parameters.confirmationRsiThreshold ??= 0;
  request.parameters.confirmationRsiWeight ??= 0;
  request.parameters.confirmationDmiMs ??= 14 * 60_000;
  request.parameters.confirmationDmiWeight ??= 0;
  request.parameters.confirmationAdxThreshold ??= 20;
  request.parameters.confirmationBias ??= 0;
  request.parameters.meanReversionMix ??= 0;
  request.parameters.meanReversionMeanMs ??= 60 * 60_000;
  request.parameters.meanReversionVolatilityMs ??= 60 * 60_000;
  request.parameters.meanReversionThreshold ??= 2;
  const values = [
    request.parameters.efficiencyMs,
    ...(request.parameters.efficiencyVolumePower > 0
      ? [request.parameters.efficiencyVolumeEmaMs]
      : []),
    request.parameters.fastMs,
    request.parameters.slowMs,
    request.parameters.power,
    request.parameters.volumeMs,
    request.parameters.volumeCap,
    ...(request.parameters.thresholdMode === "adaptive"
      ? [request.parameters.thresholdLookbackMs]
      : []),
    ...(request.parameters.confirmationMix > 0
      ? [
        request.parameters.confirmationAccelerationLookbackMs,
        request.parameters.confirmationDistanceLookbackMs,
      ]
      : []),
    ...((request.parameters.confirmationEmaWeight > 0
      || request.parameters.confirmationEmaGateStrength > 0)
      ? [request.parameters.confirmationEmaMs]
      : []),
    ...(request.parameters.confirmationRsiWeight > 0
      ? [request.parameters.confirmationRsiMs]
      : []),
    ...(request.parameters.confirmationDmiWeight > 0
      ? [request.parameters.confirmationDmiMs]
      : []),
    ...(request.parameters.meanReversionMix > 0
      ? [
        request.parameters.meanReversionMeanMs,
        request.parameters.meanReversionVolatilityMs,
        request.parameters.meanReversionThreshold,
      ]
      : []),
    request.timingHalfLifeMs,
    request.warmupMultiple,
  ];
  if (values.some((value) => !Number.isFinite(value) || value <= 0)) {
    throw new Error("VW-KAMA durations and powers must be positive.");
  }
  const nonNegative = [
    request.parameters.volumePower,
    request.parameters.efficiencyVolumePower,
    request.parameters.deadbandBpsHour,
    request.parameters.thresholdNoiseMultiplier,
    request.parameters.buyMaxFraction,
    request.parameters.sellMaxFraction,
    request.parameters.confirmationAccelerationWeight,
    request.parameters.confirmationDistanceWeight,
    request.parameters.confirmationEmaThresholdBpsHour,
    request.parameters.confirmationEmaWeight,
    request.parameters.confirmationRsiThreshold,
    request.parameters.confirmationRsiWeight,
    request.parameters.confirmationDmiWeight,
    request.parameters.confirmationAdxThreshold,
    request.parameters.meanReversionMix,
    request.oracleFriction,
    request.matchWindowMs,
  ];
  if (nonNegative.some((value) => !Number.isFinite(value) || value < 0)) {
    throw new Error("VW-KAMA thresholds and friction cannot be negative.");
  }
  if (request.parameters.buyMaxFraction > 1 || request.parameters.sellMaxFraction > 1) {
    throw new Error("VW-KAMA sizing fractions cannot exceed one.");
  }
  if (![
    request.parameters.confirmationMix,
    request.parameters.confirmationMinQuality,
    request.parameters.hysteresisReleaseRatio,
    request.parameters.confirmationEmaGateStrength,
    request.parameters.meanReversionMix,
  ]
    .every((value) => Number.isFinite(value) && value >= 0 && value <= 1)) {
    throw new Error("VW-KAMA confirmation and hysteresis fractions must be between zero and one.");
  }
  if (request.parameters.confirmationRsiThreshold > 50
    || request.parameters.confirmationAdxThreshold > 100) {
    throw new Error("VW-KAMA RSI and ADX thresholds exceed their oscillator ranges.");
  }
  if (!Number.isFinite(request.parameters.confirmationBias)) {
    throw new Error("VW-KAMA confirmation bias must be finite.");
  }
  if (![request.parameters.buySizingSigmaBpsHour, request.parameters.sellSizingSigmaBpsHour]
    .every((value) => Number.isFinite(value) && value > 0)) {
    throw new Error("VW-KAMA sizing sigmas must be positive.");
  }
  if (candidateWarmupMs(request) > MAX_WARMUP_MS) {
    throw new Error("VW-KAMA warmup exceeds the cached three-day lead-in.");
  }
  return request;
}

function normalizeCandleRangeRequest(input: VwKamaCandleRangeRequest): {
  request: Omit<VwKamaCandleRangeRequest, "maxCandles"> & { maxCandles: number };
  selected: VwKamaInspectorWindow;
} {
  const analysis = normalizeRequest(input);
  const selected = WINDOWS.find((item) => item.id === analysis.windowId);
  if (!selected) throw new Error(`Unknown VW-KAMA window: ${input.windowId}`);
  const requestedStart = Number(input.startTime);
  const requestedEnd = Number(input.endTime);
  if (!Number.isFinite(requestedStart) || !Number.isFinite(requestedEnd)) {
    throw new Error("VW-KAMA candle range timestamps must be finite.");
  }
  const startTime = clamp(Math.floor(requestedStart), selected.startTime, selected.endTime);
  const endTime = clamp(Math.ceil(requestedEnd), selected.startTime, selected.endTime);
  if (endTime <= startTime) throw new Error("VW-KAMA candle range must overlap the selected window.");
  const requestedLimit = Number(input.maxCandles ?? MAX_CHART_CANDLES);
  if (!Number.isFinite(requestedLimit) || requestedLimit <= 0) {
    throw new Error("VW-KAMA candle range limit must be positive.");
  }
  return {
    selected,
    request: {
      ...analysis,
      startTime,
      endTime,
      maxCandles: clamp(Math.round(requestedLimit), MIN_RANGE_CANDLES, MAX_RANGE_CANDLES),
    },
  };
}

function calculateKama(
  candles: Candle[],
  request: VwKamaInspectorRequest,
  scoreStartTime: number,
): Float64Array {
  const intervalMs = request.intervalMs;
  const parameters = request.parameters;
  const periods = {
    efficiencyPeriod: samples(parameters.efficiencyMs, intervalMs),
    efficiencyVolumePeriod: samples(
      parameters.efficiencyVolumeEmaMs ?? parameters.volumeMs,
      intervalMs,
    ),
    efficiencyVolumePower: parameters.efficiencyVolumePower ?? 0,
    fastPeriod: samples(parameters.fastMs, intervalMs),
    slowPeriod: samples(parameters.slowMs, intervalMs),
    volumePeriod: samples(parameters.volumeMs, intervalMs),
  };
  const indicator = new VolumeWeightedKAMAIndicator({} as TradingApi, {
    ...periods,
    power: parameters.power,
    volumeCap: parameters.volumeCap,
    volumePower: parameters.volumePower,
  });
  const values = new Float64Array(candles.length);
  values.fill(Number.NaN);
  const warmup = volumeWeightedKamaWarmupSamples(periods, request.warmupMultiple);
  const scoreStart = candleLowerBound(candles, scoreStartTime);
  const feedStart = Math.max(0, scoreStart - warmup);
  for (let index = feedStart; index < candles.length; index += 1) {
    const candle = candles[index]!;
    indicator.onTick({ eventTime: candle.closeTime, candle });
    values[index] = indicator.indicator();
  }
  return values;
}

function renderKamaSeries(
  segments: Candle[][],
  values: Float64Array[],
  candles: Candle[],
  request: VwKamaCandleRangeRequest,
): BacktestChartSmaSeries {
  const requested = new Set(candles.map((candle) => candle.closeTime));
  const points: BacktestChartSmaSeries["points"] = [];
  for (let segmentIndex = 0; segmentIndex < segments.length; segmentIndex += 1) {
    const segment = segments[segmentIndex]!;
    const segmentValues = values[segmentIndex]!;
    const from = candleLowerBound(segment, request.startTime);
    const to = candleLowerBound(segment, request.endTime);
    for (let index = from; index < to; index += 1) {
      const candle = segment[index]!;
      if (requested.has(candle.closeTime) && Number.isFinite(segmentValues[index])) {
        points.push({ time: candle.closeTime, value: segmentValues[index]! });
      }
    }
  }
  points.sort((left, right) => left.time - right.time);
  return {
    index: -1,
    windowSec: request.parameters.slowMs / 1_000,
    label: "VW-KAMA",
    color: "#f472b6",
    points,
  };
}

function candidateWarmupMs(request: VwKamaInspectorRequest): number {
  const longest = Math.max(
    request.parameters.efficiencyMs,
    request.parameters.efficiencyVolumePower
      ? request.parameters.efficiencyVolumeEmaMs ?? request.parameters.volumeMs
      : 0,
    request.parameters.slowMs,
    request.parameters.volumeMs,
    request.parameters.thresholdMode === "adaptive"
      ? request.parameters.thresholdLookbackMs ?? 0
      : 0,
    request.parameters.confirmationMix
      ? request.parameters.confirmationAccelerationLookbackMs ?? 0
      : 0,
    request.parameters.confirmationMix
      ? request.parameters.confirmationDistanceLookbackMs ?? 0
      : 0,
    (request.parameters.confirmationMix && request.parameters.confirmationEmaWeight)
      || request.parameters.confirmationEmaGateStrength
      ? request.parameters.confirmationEmaMs ?? 0
      : 0,
    request.parameters.confirmationMix && request.parameters.confirmationRsiWeight
      ? request.parameters.confirmationRsiMs ?? 0
      : 0,
    request.parameters.confirmationMix && request.parameters.confirmationDmiWeight
      ? (request.parameters.confirmationDmiMs ?? 0) * 2
      : 0,
    request.parameters.meanReversionMix
      ? Math.max(
        request.parameters.meanReversionMeanMs ?? 0,
        request.parameters.meanReversionVolatilityMs ?? 0,
      )
      : 0,
  );
  return Math.ceil(longest / request.intervalMs) * request.intervalMs * request.warmupMultiple;
}

function samples(durationMs: number, intervalMs: number): number {
  return Math.max(1, Math.round(durationMs / intervalMs));
}

function aggregateCandles(candles: Candle[], sourceMs: number, targetMs: number): Candle[] {
  if (sourceMs === targetMs) return candles.slice();
  const result: Candle[] = [];
  const expected = targetMs / sourceMs;
  let current: Candle | undefined;
  let bucket = -1;
  let count = 0;
  let contiguous = false;
  let previous = -1;
  const flush = () => {
    if (current && contiguous && count === expected) result.push(current);
  };
  for (const candle of candles) {
    const nextBucket = Math.floor(candle.openTime / targetMs) * targetMs;
    if (nextBucket !== bucket) {
      flush();
      bucket = nextBucket;
      count = 1;
      contiguous = candle.openTime === bucket;
      previous = candle.openTime;
      current = { ...candle, interval: duration(targetMs), openTime: bucket, closeTime: bucket + targetMs - 1 };
      continue;
    }
    if (!current) continue;
    contiguous &&= candle.openTime === previous + sourceMs;
    previous = candle.openTime;
    current.high = Math.max(current.high, candle.high);
    current.low = Math.min(current.low, candle.low);
    current.close = candle.close;
    current.volume += candle.volume;
    count += 1;
  }
  flush();
  return result;
}

function continuousSegments(candles: Candle[], intervalMs: number): Candle[][] {
  const result: Candle[][] = [];
  let current: Candle[] = [];
  for (const candle of candles) {
    if (current.length && candle.openTime !== current.at(-1)!.openTime + intervalMs) {
      result.push(current);
      current = [];
    }
    current.push(candle);
  }
  if (current.length) result.push(current);
  return result;
}

function renderCandleRange(
  segments: Candle[][],
  sourceIntervalMs: number,
  startTime: number,
  endTime: number,
  maxCandles: number,
): { intervalMs: number; sourceCount: number; candles: Candle[] } {
  const sourceSlots = Math.max(1, Math.ceil((endTime - startTime) / sourceIntervalMs));
  let multiplier = Math.max(1, Math.ceil(sourceSlots / maxCandles));
  let intervalMs = sourceIntervalMs * multiplier;
  while (Math.ceil(endTime / intervalMs) - Math.floor(startTime / intervalMs) > maxCandles) {
    multiplier += 1;
    intervalMs = sourceIntervalMs * multiplier;
  }
  const candles: Candle[] = [];
  let sourceCount = 0;

  for (const segment of segments) {
    const from = candleLowerBound(segment, startTime);
    const to = candleLowerBound(segment, endTime);
    sourceCount += to - from;
    if (from >= to) continue;
    candles.push(...renderSegmentCandles(segment.slice(from, to), intervalMs));
  }
  candles.sort((left, right) => left.openTime - right.openTime);
  return { intervalMs, sourceCount, candles };
}

function renderSegmentCandles(candles: Candle[], intervalMs: number): Candle[] {
  const result: Candle[] = [];
  let current: Candle | undefined;
  let bucket = -1;
  const flush = () => {
    if (!current) return;
    result.push({
      ...current,
      interval: duration(current.closeTime - current.openTime + 1),
    });
  };
  for (const candle of candles) {
    const nextBucket = Math.floor(candle.openTime / intervalMs) * intervalMs;
    if (nextBucket !== bucket) {
      flush();
      bucket = nextBucket;
      current = { ...candle };
      continue;
    }
    if (!current) continue;
    current.high = Math.max(current.high, candle.high);
    current.low = Math.min(current.low, candle.low);
    current.close = candle.close;
    current.closeTime = candle.closeTime;
    current.volume += candle.volume;
  }
  flush();
  return result;
}

function candleLowerBound(candles: Candle[], time: number): number {
  let low = 0;
  let high = candles.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (candles[middle]!.openTime < time) low = middle + 1;
    else high = middle;
  }
  return low;
}

function window(id: string, label: string, group: string, start: string, end: string): VwKamaInspectorWindow {
  return {
    id,
    label,
    group,
    startTime: Date.parse(`${start}T00:00:00.000Z`),
    endTime: Date.parse(`${end}T00:00:00.000Z`) + DAY_MS,
  };
}

function duration(ms: number): string {
  if (ms % 3_600_000 === 0) return `${ms / 3_600_000}h`;
  if (ms % 60_000 === 0) return `${ms / 60_000}m`;
  return `${ms / 1_000}s`;
}

function utcDay(time: number): number {
  return Math.floor(time / DAY_MS) * DAY_MS;
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

function eventRatio(value: number, total: number, opposite: number): number {
  return total > 0 ? value / total : opposite === 0 ? 1 : 0;
}

function harmonic(left: number, right: number): number {
  return left + right > 0 ? 2 * left * right / (left + right) : 0;
}

function percentile(values: number[], quantile: number): number | null {
  if (!values.length) return null;
  const sorted = values.slice().sort((left, right) => left - right);
  const index = (sorted.length - 1) * quantile;
  const lower = Math.floor(index);
  const fraction = index - lower;
  return sorted[lower]! * (1 - fraction) + (sorted[lower + 1] ?? sorted[lower]!) * fraction;
}
