import fs from "node:fs";
import path from "node:path";
import readline from "node:readline";
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
const TARGET_CLASSES = 33;
const MAGNITUDE_BINS = 16;
const HISTORY_STATES = 33;
const BASE_FEATURE_BINS = 2;
const BASE_DIMENSIONS = 5;
const BASE_STATES = BASE_FEATURE_BINS ** BASE_DIMENSIONS;
const BASE_CONTEXTS = HISTORY_STATES * BASE_STATES;
const BOOK_BINS = 4;
const BOOK_CONTEXTS = BASE_CONTEXTS * BOOK_BINS;
const SMOOTHING = 0.5;
const MAX_SNAPSHOT_AGE_MS = 5_000;
const MAX_DELTA_GAP_MS = 5_000;
const MINIMUM_AGE_SENSITIVITY_MS = [100, 250, 500, 1_000] as const;
const WARMUP_START = "2026-07-22";
const ANALYSIS_END = "2026-08-15";
const TRAIN_CUTOFF_MS = Date.parse("2026-07-31T00:00:00.000Z");
const PRIMARY_END_MS = Date.parse("2026-08-06T00:00:00.000Z");
const TRANSFER_START_MS = Date.parse("2026-08-10T00:00:00.000Z");
const TRANSFER_END_MS = Date.parse("2026-08-16T00:00:00.000Z");
const PRICE_BASIS_IDS = ["rsi-2", "ema-acceleration-2-1", "ema-slope-8-8"] as const;
const MARKET_BASIS_IDS = ["1h-log-volume", "1s-range"] as const;
const TRAINING_WINDOWS = [
  { id: "24h", label: "24 hours", milliseconds: DAY_MS },
  { id: "72h", label: "72 hours", milliseconds: 3 * DAY_MS },
  { id: "all", label: "all pre-cutoff spot-book history", milliseconds: Infinity },
] as const;
const DEFAULT_INPUT = "data/market/mutable/streams/spot-btcusdt/btcusdt-orderbook.jsonl";
const CANDLE_DIRECTORY = "data/market/immutable/refs/candles/spot-btcusdt/btcusdt/1s";
const DEFAULT_PRICE_CALIBRATION = "data/runtime-cache/technical-indicator-calibration.bin";
const DEFAULT_MARKET_CALIBRATION = "data/runtime-cache/volume-multiscale-calibration.bin";
const DEFAULT_FUTURES_RESULT = "data/benchmarks/order-book-return-information.json";
const DEFAULT_AUDIT = "data/benchmarks/spot-order-book-stream-audit.json";
const DEFAULT_OUTPUT = "data/benchmarks/spot-order-book-return-information.json";
const DEFAULT_REPORT = "docs/experiments/spot-order-book-return-information-2026-08-16.md";

interface Calibration {
  featureEdges: number[][];
  magnitudeEdges: number[];
}

interface Level {
  price: number;
  quantity: number;
}

export interface SpotBookSnapshot {
  symbol: string;
  eventTime: number;
  bids: Level[];
  asks: Level[];
}

interface SpotBookPoint {
  eventTime: number;
  values: Float64Array;
}

interface FeatureDefinition {
  id: string;
  label: string;
  family: string;
}

interface InformationScore {
  observations: number;
  bits: number;
  zeroBits: number;
  activeObservations: number;
  signBits: number;
}

class ProbabilityTable {
  readonly joint: Float64Array;
  readonly totals: Float64Array;
  readonly zero: Float64Array;
  readonly positive: Float64Array;

  constructor(readonly contexts: number) {
    this.joint = new Float64Array(contexts * TARGET_CLASSES);
    this.totals = new Float64Array(contexts);
    this.zero = new Float64Array(contexts);
    this.positive = new Float64Array(contexts);
  }

  add(context: number, target: number): void {
    this.joint[context * TARGET_CLASSES + target] += 1;
    this.totals[context] += 1;
    if (target === 0) this.zero[context] += 1;
    else if (target > MAGNITUDE_BINS) this.positive[context] += 1;
  }
}

interface WindowModel {
  id: string;
  label: string;
  milliseconds: number;
  trainingObservations: number;
  base: ProbabilityTable;
  candidates: ProbabilityTable[];
  regions: Map<string, InformationScore[]>;
}

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

async function main(): Promise<void> {
  const values = parseArguments(process.argv.slice(2));
  const input = resolve(values.get("input") ?? DEFAULT_INPUT);
  const priceCalibration = (deserialize(fs.readFileSync(resolve(
    values.get("price-calibration") ?? DEFAULT_PRICE_CALIBRATION,
  ))) as { calibration: Calibration }).calibration;
  const marketEdges = (deserialize(fs.readFileSync(resolve(
    values.get("market-calibration") ?? DEFAULT_MARKET_CALIBRATION,
  ))) as { marketEdges: number[][] }).marketEdges;
  const output = resolve(values.get("output") ?? DEFAULT_OUTPUT);
  const report = resolve(values.get("report") ?? DEFAULT_REPORT);
  const definitions = buildSpotBookFeatureDefinitions();
  const stateConfiguration = createStateConfiguration(priceCalibration, marketEdges);
  const bookEdges = await calibrateBookEdges(input, definitions.length);
  const models: WindowModel[] = TRAINING_WINDOWS.map((window) => ({
    ...window,
    trainingObservations: 0,
    base: new ProbabilityTable(BASE_CONTEXTS),
    candidates: definitions.map(() => new ProbabilityTable(BOOK_CONTEXTS)),
    regions: new Map(),
  }));
  const scan = await scanCandles(input, bookEdges, stateConfiguration, models);
  const windowResults = models.map((model) => windowResult(model, definitions));
  const flattened = windowResults.flatMap((window) => window.candidates.map((candidate: any) => ({
    ...candidate,
    windowId: window.id,
    windowLabel: window.label,
    trainingObservations: window.trainingObservations,
  })));
  const rankedFull = [...flattened].sort((left, right) =>
    right.primary.fullBits - left.primary.fullBits);
  const rankedSign = [...flattened].sort((left, right) =>
    signStability(right) - signStability(left)
      || right.primary.activeSignBits - left.primary.activeSignBits);
  const stableFull = flattened.filter((candidate) => fullStability(candidate) === 4)
    .sort((left, right) => right.primary.fullBits - left.primary.fullBits);
  const bestFullModel = models.find((model) => model.id === rankedFull[0]!.windowId)!;
  const bestFullFeatureIndex = definitions.findIndex((definition) =>
    definition.id === rankedFull[0]!.id);
  const futures = readJsonIfPresent(resolve(values.get("futures-result") ?? DEFAULT_FUTURES_RESULT));
  const audit = readJsonIfPresent(resolve(values.get("audit") ?? DEFAULT_AUDIT));
  const artifact = {
    version: 1,
    generatedAt: new Date().toISOString(),
    source: {
      book: "locally recorded Binance spot BTCUSDT top-10 snapshots",
      input: path.relative(repoRoot, input),
      validSnapshotsRead: scan.validSnapshotsRead,
      invalidLines: scan.invalidLines,
      firstSnapshotTime: scan.firstSnapshotTime,
      lastSnapshotTime: scan.lastSnapshotTime,
      audit,
      target: "Binance spot BTCUSDT next-1s close return",
      candleDirectory: CANDLE_DIRECTORY,
    },
    split: {
      trainEndExclusive: new Date(TRAIN_CUTOFF_MS).toISOString(),
      primary: {
        start: new Date(TRAIN_CUTOFF_MS).toISOString(),
        endExclusive: new Date(PRIMARY_END_MS).toISOString(),
      },
      transfer: {
        start: new Date(TRANSFER_START_MS).toISOString(),
        endExclusive: new Date(TRANSFER_END_MS).toISOString(),
      },
    },
    method: {
      snapshotPolicy: "Only snapshots strictly earlier than a target second are used; snapshots older than 5 seconds are discarded.",
      deltaPolicy: "Snapshot-change features reset after gaps longer than 5 seconds.",
      targetStride: 1,
      baseConditioning: "Previous-return state plus median splits of RSI(2), EMA acceleration(2,1), EMA slope(8,8), last completed 1h log-volume, and prior 1s range.",
      bookQuantization: "Four cells at training-only empirical quartiles; edges are frozen at the July 31 cutoff.",
      smoothing: SMOOTHING,
    },
    calibration: {
      featureDefinitions: definitions,
      bookEdges,
    },
    coverage: scan.coverage,
    stableFullCandidateCount: stableFull.length,
    bestFullPrimary: rankedFull[0],
    bestStableFull: stableFull[0] ?? null,
    bestActiveSign: rankedSign[0],
    bestFullTrainingProfile: trainingBinProfile(
      bestFullModel.candidates[bestFullFeatureIndex]!,
    ),
    rankedFull,
    windows: windowResults,
    futuresComparison: futures ? {
      caveat: "The futures study uses 30-second cumulative percentage-depth and a much longer history, so this is a directional rather than controlled source comparison.",
      bestFullPrimary: futures.bestFullPrimary,
      bestActiveSign: futures.bestActiveSign,
    } : null,
    limitations: [
      "The local spot stream spans only about three weeks and contains multi-hour and multi-day gaps; usable observations are limited to fresh-snapshot periods.",
      "This is top-10 depth, not a complete depth ladder, and it records snapshots rather than every exchange depth update.",
      "Only one training cutoff and two later calendar blocks are available; positive results need confirmation on a longer independently recorded spot history.",
      "The comparison with archived futures percentage-depth is not apples-to-apples because source cadence, depth representation, dates, and history length differ.",
      "No latency beyond causal timestamp ordering, fees, queue position, spread crossing, fills, or market impact are modeled.",
    ],
  };
  fs.mkdirSync(path.dirname(output), { recursive: true });
  fs.writeFileSync(output, `${JSON.stringify(artifact, null, 2)}\n`, "utf8");
  fs.mkdirSync(path.dirname(report), { recursive: true });
  fs.writeFileSync(report, renderReport(artifact), "utf8");
  console.log(`Wrote ${path.relative(repoRoot, output)}`);
  console.log(`Wrote ${path.relative(repoRoot, report)}`);
}

export function buildSpotBookFeatureDefinitions(): FeatureDefinition[] {
  return [
    { id: "spread-bps", label: "bid-ask spread", family: "top of book" },
    { id: "microprice-offset-bps", label: "microprice offset from midpoint", family: "top of book" },
    { id: "quantity-imbalance-l1", label: "L1 quantity imbalance", family: "queue imbalance" },
    { id: "quantity-imbalance-l2", label: "top-2 quantity imbalance", family: "queue imbalance" },
    { id: "quantity-imbalance-l5", label: "top-5 quantity imbalance", family: "queue imbalance" },
    { id: "quantity-imbalance-l10", label: "top-10 quantity imbalance", family: "queue imbalance" },
    { id: "notional-imbalance-l1", label: "L1 notional imbalance", family: "queue imbalance" },
    { id: "notional-imbalance-l2", label: "top-2 notional imbalance", family: "queue imbalance" },
    { id: "notional-imbalance-l5", label: "top-5 notional imbalance", family: "queue imbalance" },
    { id: "notional-imbalance-l10", label: "top-10 notional imbalance", family: "queue imbalance" },
    { id: "log-quantity-l1", label: "log total L1 quantity", family: "liquidity" },
    { id: "log-quantity-l2", label: "log total top-2 quantity", family: "liquidity" },
    { id: "log-quantity-l5", label: "log total top-5 quantity", family: "liquidity" },
    { id: "log-quantity-l10", label: "log total top-10 quantity", family: "liquidity" },
    { id: "bid-l1-concentration", label: "bid L1/top-10 concentration", family: "depth shape" },
    { id: "ask-l1-concentration", label: "ask L1/top-10 concentration", family: "depth shape" },
    { id: "inner-outer-imbalance", label: "L1 minus top-10 imbalance", family: "depth shape" },
    { id: "mean-depth-distance-bps", label: "quantity-weighted depth distance", family: "depth shape" },
    { id: "depth-distance-asymmetry-bps", label: "ask minus bid depth distance", family: "depth shape" },
    { id: "delta-imbalance-l1", label: "snapshot change in L1 imbalance", family: "book change" },
    { id: "delta-imbalance-l5", label: "snapshot change in top-5 imbalance", family: "book change" },
    { id: "delta-imbalance-l10", label: "snapshot change in top-10 imbalance", family: "book change" },
    { id: "delta-microprice-bps", label: "snapshot change in microprice offset", family: "book change" },
    { id: "delta-spread-bps", label: "snapshot change in spread", family: "book change" },
    { id: "delta-log-quantity-l10", label: "snapshot change in log top-10 quantity", family: "book change" },
    { id: "midpoint-return-bps", label: "snapshot-to-snapshot midpoint return", family: "book change" },
    { id: "normalized-ofi-l1", label: "normalized L1 order-flow imbalance", family: "order flow" },
  ];
}

export function spotSnapshotValues(
  snapshot: SpotBookSnapshot,
  previous: { snapshot: SpotBookSnapshot; values: Float64Array } | undefined,
): Float64Array {
  const bid = snapshot.bids[0]!;
  const ask = snapshot.asks[0]!;
  const midpoint = (bid.price + ask.price) / 2;
  const spreadBps = (ask.price - bid.price) / midpoint * 10_000;
  const microprice = (ask.price * bid.quantity + bid.price * ask.quantity)
    / (bid.quantity + ask.quantity);
  const micropriceOffsetBps = (microprice / midpoint - 1) * 10_000;
  const ks = [1, 2, 5, 10];
  const quantityImbalances: number[] = [];
  const notionalImbalances: number[] = [];
  const logQuantities: number[] = [];
  let bidQuantity10 = 0;
  let askQuantity10 = 0;
  for (const k of ks) {
    let bidQuantity = 0;
    let askQuantity = 0;
    let bidNotional = 0;
    let askNotional = 0;
    for (let level = 0; level < k; level += 1) {
      const bidLevel = snapshot.bids[level]!;
      const askLevel = snapshot.asks[level]!;
      bidQuantity += bidLevel.quantity;
      askQuantity += askLevel.quantity;
      bidNotional += bidLevel.price * bidLevel.quantity;
      askNotional += askLevel.price * askLevel.quantity;
    }
    if (k === 10) {
      bidQuantity10 = bidQuantity;
      askQuantity10 = askQuantity;
    }
    quantityImbalances.push(imbalance(bidQuantity, askQuantity));
    notionalImbalances.push(imbalance(bidNotional, askNotional));
    logQuantities.push(Math.log(bidQuantity + askQuantity));
  }
  let bidDistance = 0;
  let askDistance = 0;
  for (let level = 0; level < 10; level += 1) {
    const bidLevel = snapshot.bids[level]!;
    const askLevel = snapshot.asks[level]!;
    bidDistance += bidLevel.quantity * (midpoint - bidLevel.price) / midpoint * 10_000;
    askDistance += askLevel.quantity * (askLevel.price - midpoint) / midpoint * 10_000;
  }
  bidDistance /= bidQuantity10;
  askDistance /= askQuantity10;
  const raw = [
    spreadBps,
    micropriceOffsetBps,
    ...quantityImbalances,
    ...notionalImbalances,
    ...logQuantities,
    snapshot.bids[0]!.quantity / bidQuantity10,
    snapshot.asks[0]!.quantity / askQuantity10,
    quantityImbalances[0]! - quantityImbalances[3]!,
    (bidDistance + askDistance) / 2,
    askDistance - bidDistance,
  ];
  const recentPrevious = previous
    && snapshot.eventTime - previous.snapshot.eventTime <= MAX_DELTA_GAP_MS
    && snapshot.eventTime > previous.snapshot.eventTime
    ? previous
    : undefined;
  const priorValues = recentPrevious?.values ?? Float64Array.from(raw);
  const previousMidpoint = recentPrevious
    ? (recentPrevious.snapshot.bids[0]!.price + recentPrevious.snapshot.asks[0]!.price) / 2
    : midpoint;
  raw.push(
    raw[2]! - priorValues[2]!,
    raw[4]! - priorValues[4]!,
    raw[5]! - priorValues[5]!,
    raw[1]! - priorValues[1]!,
    raw[0]! - priorValues[0]!,
    raw[13]! - priorValues[13]!,
    Math.log(midpoint / previousMidpoint) * 10_000,
    recentPrevious ? normalizedOrderFlowImbalance(snapshot, recentPrevious.snapshot) : 0,
  );
  return Float64Array.from(raw);
}

function normalizedOrderFlowImbalance(current: SpotBookSnapshot, previous: SpotBookSnapshot): number {
  const bid = current.bids[0]!;
  const ask = current.asks[0]!;
  const previousBid = previous.bids[0]!;
  const previousAsk = previous.asks[0]!;
  const flow = (bid.price >= previousBid.price ? bid.quantity : 0)
    - (bid.price <= previousBid.price ? previousBid.quantity : 0)
    - (ask.price <= previousAsk.price ? ask.quantity : 0)
    + (ask.price >= previousAsk.price ? previousAsk.quantity : 0);
  const scale = (bid.quantity + ask.quantity + previousBid.quantity + previousAsk.quantity) / 2;
  return scale > 0 ? flow / scale : 0;
}

async function calibrateBookEdges(input: string, featureCount: number): Promise<number[][]> {
  console.error("Calibrating spot-book quartiles on snapshots strictly before the cutoff...");
  const samples = Array.from({ length: featureCount }, () => [] as number[]);
  let retained = 0;
  for await (const point of readSpotBookPoints(input, TRAIN_CUTOFF_MS)) {
    for (let feature = 0; feature < featureCount; feature += 1) {
      samples[feature]!.push(point.values[feature]!);
    }
    retained += 1;
    if (retained % 100_000 === 0) console.error(`Calibrated ${retained.toLocaleString()} snapshots...`);
  }
  if (retained === 0) throw new Error("No pre-cutoff spot-book snapshots were available.");
  return samples.map((sample) => {
    sample.sort((left, right) => left - right);
    return [0.25, 0.5, 0.75].map((quantile) => sample[
      Math.min(sample.length - 1, Math.floor(sample.length * quantile))
    ]!);
  });
}

function createStateConfiguration(priceCalibration: Calibration, marketEdges: number[][]) {
  const priceDefinitions = buildSignalDefinitions();
  const priceIndex = new Map(priceDefinitions.map((definition, index) => [definition.id, index]));
  const selectedPriceDefinitions = PRICE_BASIS_IDS.map((id) => priceDefinitions[priceIndex.get(id)!]!);
  const priceCuts = PRICE_BASIS_IDS.map((id) => priceCalibration.featureEdges[priceIndex.get(id)!]![7]!);
  const marketDefinitions = buildMarketFeatureDefinitions();
  const marketById = new Map(marketDefinitions.map((definition) => [definition.id, definition]));
  const marketIndices = MARKET_BASIS_IDS.map((id) => marketById.get(id)!.sourceIndex);
  const marketCuts = marketIndices.map((index) => marketEdges[index]![1]!);
  return { priceCalibration, selectedPriceDefinitions, priceCuts, marketIndices, marketCuts };
}

async function scanCandles(
  input: string,
  bookEdges: number[][],
  configuration: ReturnType<typeof createStateConfiguration>,
  models: WindowModel[],
) {
  pointReaderDiagnostics.invalidLines = 0;
  console.error("Scanning spot candles and causally aligning fresh top-10 snapshots...");
  const iterator = readSpotBookPoints(input, TRANSFER_END_MS)[Symbol.asyncIterator]();
  let nextPointResult = await iterator.next();
  let currentPoint: SpotBookPoint | undefined;
  let firstSnapshotTime: string | null = null;
  let lastSnapshotTime: string | null = null;
  let validSnapshotsRead = 0;
  const marketEngine = new CausalMarketFeatureEngine();
  const priceValues = new Float64Array(configuration.selectedPriceDefinitions.length);
  let priceEngine: IndicatorEngine | undefined;
  let previousClose = Number.NaN;
  let previousReturnState = 0;
  let invalidLines = 0;
  const coverage = {
    trainingTargets: 0,
    primaryTargets: 0,
    primaryFirstTargets: 0,
    primarySecondTargets: 0,
    transferTargets: 0,
    transferFirstTargets: 0,
    transferSecondTargets: 0,
    staleOrMissingTargets: 0,
  };
  for (const day of inclusiveDays(WARMUP_START, ANALYSIS_END)) {
    console.error(`Spot-book alignment ${day}...`);
    const candles = readCandleShardReferenceSync(candleFile(day));
    if (candles.length !== 86_400) throw new Error(`${day}: spot candle day is incomplete.`);
    for (const candle of candles) {
      while (!nextPointResult.done && nextPointResult.value.eventTime < candle.openTime) {
        currentPoint = nextPointResult.value;
        validSnapshotsRead += 1;
        firstSnapshotTime ??= new Date(currentPoint.eventTime).toISOString();
        lastSnapshotTime = new Date(currentPoint.eventTime).toISOString();
        nextPointResult = await iterator.next();
      }
      if (!priceEngine) {
        priceEngine = new IndicatorEngine(configuration.selectedPriceDefinitions, candle.close);
        marketEngine.update(candle);
        previousClose = candle.close;
        continue;
      }
      const returnBps = candle.close === previousClose ? 0 : Math.log(candle.close / previousClose) * 10_000;
      const active = returnBps !== 0;
      const sign = returnBps > 0 ? 1 : 0;
      const magnitudeBin = active
        ? upperBound(configuration.priceCalibration.magnitudeEdges, Math.abs(returnBps))
        : -1;
      const target = active ? 1 + sign * MAGNITUDE_BINS + magnitudeBin : 0;
      const pointAge = currentPoint ? candle.openTime - currentPoint.eventTime : Infinity;
      const inAnalyzedPeriod = candle.openTime >= Date.parse("2026-07-25T00:00:00.000Z");
      const bookUsable = currentPoint && pointAge > 0 && pointAge <= MAX_SNAPSHOT_AGE_MS;
      const marketUsable = configuration.marketIndices.every((index) => marketEngine.valid()[index] === 1);
      if (inAnalyzedPeriod && bookUsable && marketUsable) {
        priceEngine.values(priceValues);
        let basisState = 0;
        for (let index = 0; index < configuration.priceCuts.length; index += 1) {
          basisState = basisState * BASE_FEATURE_BINS
            + (priceValues[index]! >= configuration.priceCuts[index]! ? 1 : 0);
        }
        const marketValues = marketEngine.values();
        for (let index = 0; index < configuration.marketIndices.length; index += 1) {
          basisState = basisState * BASE_FEATURE_BINS + (
            marketValues[configuration.marketIndices[index]!]! >= configuration.marketCuts[index]! ? 1 : 0
          );
        }
        const baseContext = previousReturnState * BASE_STATES + basisState;
        const candidateContexts = bookEdges.map((edges, feature) =>
          baseContext * BOOK_BINS + upperBound(edges, currentPoint!.values[feature]!));
        if (candle.openTime < TRAIN_CUTOFF_MS) {
          coverage.trainingTargets += 1;
          for (const model of models) {
            if (candle.openTime >= TRAIN_CUTOFF_MS - model.milliseconds) {
              model.base.add(baseContext, target);
              model.candidates.forEach((table, feature) => table.add(candidateContexts[feature]!, target));
              model.trainingObservations += 1;
            }
          }
        } else if (candle.openTime < PRIMARY_END_MS) {
          coverage.primaryTargets += 1;
          const half = candle.openTime < Date.parse("2026-08-03T00:00:00.000Z")
            ? "primaryFirst" : "primarySecond";
          if (half === "primaryFirst") coverage.primaryFirstTargets += 1;
          else coverage.primarySecondTargets += 1;
          scoreModels(models, "primary", baseContext, candidateContexts, target);
          scoreModels(models, half, baseContext, candidateContexts, target);
          for (const minimumAge of MINIMUM_AGE_SENSITIVITY_MS) {
            if (pointAge >= minimumAge) {
              scoreModels(models, `primaryLag${minimumAge}`, baseContext, candidateContexts, target);
            }
          }
        } else if (candle.openTime >= TRANSFER_START_MS && candle.openTime < TRANSFER_END_MS) {
          coverage.transferTargets += 1;
          const half = candle.openTime < Date.parse("2026-08-13T00:00:00.000Z")
            ? "transferFirst" : "transferSecond";
          if (half === "transferFirst") coverage.transferFirstTargets += 1;
          else coverage.transferSecondTargets += 1;
          scoreModels(models, "transfer", baseContext, candidateContexts, target);
          scoreModels(models, half, baseContext, candidateContexts, target);
          for (const minimumAge of MINIMUM_AGE_SENSITIVITY_MS) {
            if (pointAge >= minimumAge) {
              scoreModels(models, `transferLag${minimumAge}`, baseContext, candidateContexts, target);
            }
          }
        }
      } else if (inAnalyzedPeriod && (!bookUsable || !marketUsable)) {
        coverage.staleOrMissingTargets += 1;
      }
      previousReturnState = target;
      priceEngine.update(candle.close);
      marketEngine.update(candle);
      previousClose = candle.close;
    }
  }
  if (typeof iterator.return === "function") await iterator.return();
  invalidLines = pointReaderDiagnostics.invalidLines;
  return { coverage, firstSnapshotTime, lastSnapshotTime, validSnapshotsRead, invalidLines };
}

const pointReaderDiagnostics = { invalidLines: 0 };

async function* readSpotBookPoints(input: string, endExclusive: number): AsyncGenerator<SpotBookPoint> {
  let previous: { snapshot: SpotBookSnapshot; values: Float64Array } | undefined;
  const reader = readline.createInterface({
    input: fs.createReadStream(input, { encoding: "utf8" }),
    crlfDelay: Infinity,
  });
  for await (const line of reader) {
    let snapshot: SpotBookSnapshot;
    try {
      snapshot = JSON.parse(line) as SpotBookSnapshot;
    } catch {
      pointReaderDiagnostics.invalidLines += 1;
      continue;
    }
    if (!validSnapshot(snapshot)) {
      pointReaderDiagnostics.invalidLines += 1;
      continue;
    }
    if (snapshot.eventTime >= endExclusive) break;
    const values = spotSnapshotValues(snapshot, previous);
    previous = { snapshot, values };
    yield { eventTime: snapshot.eventTime, values };
  }
  reader.close();
}

export function validSnapshot(snapshot: SpotBookSnapshot): boolean {
  if (snapshot.symbol !== "BTCUSDT" || !Number.isSafeInteger(snapshot.eventTime)
    || snapshot.bids?.length !== 10 || snapshot.asks?.length !== 10) return false;
  for (let index = 0; index < 10; index += 1) {
    const bid = snapshot.bids[index]!;
    const ask = snapshot.asks[index]!;
    if (!(bid.price > 0) || !(bid.quantity > 0) || !(ask.price > 0) || !(ask.quantity > 0)) return false;
    if (index > 0 && (bid.price > snapshot.bids[index - 1]!.price
      || ask.price < snapshot.asks[index - 1]!.price)) return false;
  }
  return snapshot.bids[0]!.price < snapshot.asks[0]!.price;
}

function scoreModels(
  models: WindowModel[],
  region: string,
  baseContext: number,
  candidateContexts: number[],
  target: number,
): void {
  for (const model of models) {
    let scores = model.regions.get(region);
    if (!scores) {
      scores = model.candidates.map(emptyScore);
      model.regions.set(region, scores);
    }
    model.candidates.forEach((candidate, feature) => {
      scoreObservation(scores![feature]!, candidate, model.base, candidateContexts[feature]!, baseContext, target);
    });
  }
}

function scoreObservation(
  score: InformationScore,
  candidate: ProbabilityTable,
  base: ProbabilityTable,
  candidateContext: number,
  baseContext: number,
  target: number,
): void {
  const candidateProbability = smoothed(
    candidate.joint[candidateContext * TARGET_CLASSES + target]!,
    candidate.totals[candidateContext]!, TARGET_CLASSES,
  );
  const baseProbability = smoothed(
    base.joint[baseContext * TARGET_CLASSES + target]!,
    base.totals[baseContext]!, TARGET_CLASSES,
  );
  score.observations += 1;
  score.bits += Math.log2(candidateProbability / baseProbability);
  const candidateZero = target === 0
    ? candidate.zero[candidateContext]!
    : candidate.totals[candidateContext]! - candidate.zero[candidateContext]!;
  const baseZero = target === 0
    ? base.zero[baseContext]!
    : base.totals[baseContext]! - base.zero[baseContext]!;
  score.zeroBits += Math.log2(
    smoothed(candidateZero, candidate.totals[candidateContext]!, 2)
      / smoothed(baseZero, base.totals[baseContext]!, 2),
  );
  if (target === 0) return;
  const positive = target > MAGNITUDE_BINS;
  const candidateActive = candidate.totals[candidateContext]! - candidate.zero[candidateContext]!;
  const baseActive = base.totals[baseContext]! - base.zero[baseContext]!;
  const candidateSign = positive
    ? candidate.positive[candidateContext]!
    : candidateActive - candidate.positive[candidateContext]!;
  const baseSign = positive
    ? base.positive[baseContext]!
    : baseActive - base.positive[baseContext]!;
  score.activeObservations += 1;
  score.signBits += Math.log2(
    smoothed(candidateSign, candidateActive, 2) / smoothed(baseSign, baseActive, 2),
  );
}

function windowResult(model: WindowModel, definitions: FeatureDefinition[]) {
  const region = (id: string, feature: number) => normalizedScore(
    model.regions.get(id)?.[feature] ?? emptyScore(),
  );
  return {
    id: model.id,
    label: model.label,
    trainingObservations: model.trainingObservations,
    candidates: definitions.map((definition, feature) => ({
      ...definition,
      primary: region("primary", feature),
      primaryFirst: region("primaryFirst", feature),
      primarySecond: region("primarySecond", feature),
      transfer: region("transfer", feature),
      transferFirst: region("transferFirst", feature),
      transferSecond: region("transferSecond", feature),
      minimumSnapshotAge: Object.fromEntries(MINIMUM_AGE_SENSITIVITY_MS.map((milliseconds) => [
        `${milliseconds}ms`,
        {
          primary: region(`primaryLag${milliseconds}`, feature),
          transfer: region(`transferLag${milliseconds}`, feature),
        },
      ])),
    })),
  };
}

function trainingBinProfile(table: ProbabilityTable) {
  return Array.from({ length: BOOK_BINS }, (_, bin) => {
    let observations = 0;
    let zero = 0;
    let positive = 0;
    for (let context = bin; context < table.contexts; context += BOOK_BINS) {
      observations += table.totals[context]!;
      zero += table.zero[context]!;
      positive += table.positive[context]!;
    }
    const active = observations - zero;
    return {
      bin,
      observations,
      zeroProbability: zero / observations,
      positiveGivenActive: positive / active,
    };
  });
}

function fullStability(candidate: any): number {
  return [candidate.primaryFirst.fullBits, candidate.primarySecond.fullBits,
    candidate.transferFirst.fullBits, candidate.transferSecond.fullBits]
    .filter((value) => value > 0).length;
}

function signStability(candidate: any): number {
  return [candidate.primaryFirst.activeSignBits, candidate.primarySecond.activeSignBits,
    candidate.transferFirst.activeSignBits, candidate.transferSecond.activeSignBits]
    .filter((value) => value > 0).length;
}

function emptyScore(): InformationScore {
  return { observations: 0, bits: 0, zeroBits: 0, activeObservations: 0, signBits: 0 };
}

function normalizedScore(score: InformationScore) {
  return {
    observations: score.observations,
    fullBits: score.observations > 0 ? score.bits / score.observations : Number.NaN,
    zeroGateBits: score.observations > 0 ? score.zeroBits / score.observations : Number.NaN,
    activeObservations: score.activeObservations,
    activeSignBits: score.activeObservations > 0 ? score.signBits / score.activeObservations : Number.NaN,
  };
}

function renderReport(artifact: any): string {
  const best = artifact.bestFullPrimary;
  const stable = artifact.bestStableFull;
  const bestSign = artifact.bestActiveSign;
  const futuresFull = artifact.futuresComparison?.bestFullPrimary;
  const futuresSign = artifact.futuresComparison?.bestActiveSign;
  const lines = [
    "# Spot top-10 order-book information about the next 1s return",
    "",
    `Generated ${artifact.generatedAt}. This study uses ${artifact.source.validSnapshotsRead.toLocaleString()} locally recorded BTCUSDT spot top-10 snapshots and matching official spot 1s candles.`,
    "",
    "## Result",
    "",
    `The best primary full-distribution result is **${best.label}** with the **${best.windowLabel}** fit: ${metric(best.primary.fullBits)} bits/target in July 31–August 5 and ${metric(best.transfer.fullBits)} in the later August 10–15 block. Its four chronological sub-block scores are ${[best.primaryFirst.fullBits, best.primarySecond.fullBits, best.transferFirst.fullBits, best.transferSecond.fullBits].map(metric).join(", ")}.`,
    "",
    stable
      ? `There are ${artifact.stableFullCandidateCount} feature/window combinations positive in all four chronological sub-blocks. The strongest is **${stable.label}** (${stable.windowLabel}), with ${metric(stable.primary.fullBits)} primary and ${metric(stable.transfer.fullBits)} transfer bits/target.`
      : "No feature/window combination is positive in all four chronological sub-blocks, so the current spot sample does not establish a stable full-distribution improvement.",
    "",
    `For active-return sign, the most stable result is **${bestSign.label}** (${bestSign.windowLabel}): ${metric(bestSign.primary.activeSignBits)} primary and ${metric(bestSign.transfer.activeSignBits)} transfer bits per active target.`,
    "",
    `For the winning L1 feature's training quartiles, P(positive | active) runs ${artifact.bestFullTrainingProfile.map((row: any) => `${(100 * row.positiveGivenActive).toFixed(3)}%`).join(" → ")} from the most ask-heavy to the most bid-heavy cell. The zero-return probabilities are ${artifact.bestFullTrainingProfile.map((row: any) => `${(100 * row.zeroProbability).toFixed(3)}%`).join(" → ")}.`,
    "",
    "## Primary ranking",
    "",
    "| rank | feature | history | family | primary full bits | transfer full bits | primary sign bits | transfer sign bits | positive sub-blocks |",
    "|---:|---|---:|---|---:|---:|---:|---:|---:|",
  ];
  artifact.rankedFull.slice(0, 30).forEach((row: any, index: number) => {
    lines.push(`| ${index + 1} | ${row.label} | ${row.windowLabel} | ${row.family} | ${metric(row.primary.fullBits)} | ${metric(row.transfer.fullBits)} | ${metric(row.primary.activeSignBits)} | ${metric(row.transfer.activeSignBits)} | ${fullStability(row)}/4 |`);
  });
  lines.push(
    "",
    "## Timestamp-lag sensitivity",
    "",
    "To allow for exchange-to-recorder delivery delay, the winning feature was rescored after requiring the latest snapshot to predate the target boundary by at least the stated amount. The probability table remains the same frozen fit.",
    "",
    "| minimum snapshot age | primary observations | primary bits | transfer observations | transfer bits |",
    "|---:|---:|---:|---:|---:|",
  );
  for (const [age, row] of Object.entries(best.minimumSnapshotAge) as Array<[string, any]>) {
    lines.push(`| ${age} | ${row.primary.observations.toLocaleString()} | ${metric(row.primary.fullBits)} | ${row.transfer.observations.toLocaleString()} | ${metric(row.transfer.fullBits)} |`);
  }
  lines.push(
    "",
    "## Coverage and causal split",
    "",
    `The frozen cutoff is ${artifact.split.trainEndExclusive}. The primary test has ${artifact.coverage.primaryTargets.toLocaleString()} fresh-book targets (${artifact.coverage.primaryFirstTargets.toLocaleString()} / ${artifact.coverage.primarySecondTargets.toLocaleString()}); transfer has ${artifact.coverage.transferTargets.toLocaleString()} (${artifact.coverage.transferFirstTargets.toLocaleString()} / ${artifact.coverage.transferSecondTargets.toLocaleString()}). ${artifact.coverage.staleOrMissingTargets.toLocaleString()} seconds are excluded because the book is missing/stale or the candle basis is not ready.`,
    "",
    artifact.method.snapshotPolicy,
    artifact.method.deltaPolicy,
    "All book quartiles and probability tables are fitted only from pre-cutoff data and then frozen.",
    "",
    "## Contrast with futures percentage-depth",
    "",
    futuresFull
      ? `The earlier futures result had ${metric(futuresFull.fullBits)} bits/target for its numerically best full-distribution feature; its best active-sign result was ${metric(futuresSign.activeSignBits)} bits/active target while remaining ${metric(futuresSign.fullBits)} on the full distribution.`
      : "The earlier futures artifact was not available for an automatic comparison.",
    artifact.futuresComparison?.caveat ?? "",
    "",
    "## Limits",
    "",
    ...artifact.limitations.map((limitation: string) => `- ${limitation}`),
    "",
    "Complete values are stored in `data/benchmarks/spot-order-book-return-information.json`.",
  );
  return `${lines.join("\n")}\n`;
}

function inclusiveDays(start: string, end: string): string[] {
  const result: string[] = [];
  for (let time = Date.parse(`${start}T00:00:00.000Z`);
    time <= Date.parse(`${end}T00:00:00.000Z`); time += DAY_MS) {
    result.push(new Date(time).toISOString().slice(0, 10));
  }
  return result;
}

function candleFile(day: string): string {
  return path.join(resolve(CANDLE_DIRECTORY), `${day}.json`);
}

function imbalance(bid: number, ask: number): number {
  return (bid - ask) / (bid + ask);
}

function smoothed(count: number, total: number, classes: number): number {
  return (count + SMOOTHING) / (total + classes * SMOOTHING);
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

function readJsonIfPresent(file: string): any {
  return fs.existsSync(file) ? JSON.parse(fs.readFileSync(file, "utf8")) : null;
}

function metric(value: number): string {
  return Number.isFinite(value)
    ? value === 0 ? "0" : value.toPrecision(8).replace(/\.0+(?=e|$)/, "")
    : "n/a";
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  main().catch((error: unknown) => {
    console.error(error instanceof Error ? error.stack ?? error.message : String(error));
    process.exitCode = 1;
  });
}
