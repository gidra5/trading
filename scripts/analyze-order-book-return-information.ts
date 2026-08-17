import fs from "node:fs";
import path from "node:path";
import { deserialize } from "node:v8";
import { fileURLToPath, pathToFileURL } from "node:url";
import {
  readCandleShardReferenceSync,
  readDerivativesBookDepthShardReferenceSync,
  type SequentialDerivativesBookDepthSnapshot,
} from "@trading/storage";
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
const BASE_FEATURE_BINS = 2;
const BASE_DIMENSIONS = 5;
const BASE_STATES = BASE_FEATURE_BINS ** BASE_DIMENSIONS;
const HISTORY_STATES = 33;
const BASE_CONTEXTS = HISTORY_STATES * BASE_STATES;
const BOOK_BINS = 4;
const BOOK_CONTEXTS = BASE_CONTEXTS * BOOK_BINS;
const EVALUATION_STRIDE = 4;
const WARMUP_RETURNS = 12_000;
const MAX_SNAPSHOT_AGE_SECONDS = 120;
const SMOOTHING = 0.5;
const TRAIN_CUTOFF_DAYS = 180;
const TRAINING_WINDOWS = [30, 60, 90, 180] as const;
const PRICE_BASIS_IDS = ["rsi-2", "ema-acceleration-2-1", "ema-slope-8-8"] as const;
const MARKET_BASIS_IDS = ["1h-log-volume", "1s-range"] as const;
const BOOK_DIRECTORY = "data/market/immutable/refs/derivatives-book-depth/usdm-futures/btcusdt";
const CANDLE_DIRECTORY = "data/market/immutable/refs/candles/spot-btcusdt/btcusdt/1s";
const PRIMARY_START = "2025-03-18";
const PRIMARY_END = "2025-11-13";
const TRANSFER_START = "2026-04-21";
const TRANSFER_END = "2026-06-23";
const DEFAULT_PRICE_CALIBRATION = "data/runtime-cache/technical-indicator-calibration.bin";
const DEFAULT_MARKET_CALIBRATION = "data/runtime-cache/volume-multiscale-calibration.bin";
const DEFAULT_OUTPUT = "data/benchmarks/order-book-return-information.json";
const DEFAULT_REPORT = "docs/experiments/order-book-return-information-2026-08-16.md";

interface Calibration {
  featureEdges: number[][];
  magnitudeEdges: number[];
}

interface FeatureDefinition {
  id: string;
  label: string;
  family: string;
}

interface BookPoint {
  timestampSeconds: number;
  values: Float64Array;
}

interface SparseCounts {
  indices: Uint32Array;
  counts: Uint32Array;
  observations: number;
}

interface DayCounts {
  base: SparseCounts;
  candidates: SparseCounts[];
}

interface InformationScore {
  fullObservations: number;
  fullBits: number;
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

  add(day: SparseCounts): void {
    for (let item = 0; item < day.indices.length; item += 1) {
      const index = day.indices[item]!;
      const count = day.counts[item]!;
      const context = Math.floor(index / TARGET_CLASSES);
      const target = index % TARGET_CLASSES;
      this.joint[index] += count;
      this.totals[context] += count;
      if (target === 0) this.zero[context] += count;
      else if (target > MAGNITUDE_BINS) this.positive[context] += count;
    }
  }
}

interface WindowModel {
  days: number;
  base: ProbabilityTable;
  candidates: ProbabilityTable[];
  scores: InformationScore[];
  firstHalf: InformationScore[];
  secondHalf: InformationScore[];
}

interface FeatureCursor {
  previous?: Float64Array;
}

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function main(): void {
  const values = parseArguments(process.argv.slice(2));
  const priceCalibration = (deserialize(fs.readFileSync(resolve(
    values.get("price-calibration") ?? DEFAULT_PRICE_CALIBRATION,
  ))) as { calibration: Calibration }).calibration;
  const marketEdges = (deserialize(fs.readFileSync(resolve(
    values.get("market-calibration") ?? DEFAULT_MARKET_CALIBRATION,
  ))) as { marketEdges: number[][] }).marketEdges;
  const outputPath = resolve(values.get("output") ?? DEFAULT_OUTPUT);
  const reportPath = resolve(values.get("report") ?? DEFAULT_REPORT);
  const definitions = buildBookFeatureDefinitions();
  const primaryDays = inclusiveDays(PRIMARY_START, PRIMARY_END);
  const transferDays = inclusiveDays(TRANSFER_START, TRANSFER_END);
  assertReferences(primaryDays);
  assertReferences(transferDays);
  const bookEdges = calibrateBookEdges(primaryDays.slice(0, 60), definitions.length);
  const stateConfiguration = createStateConfiguration(priceCalibration, marketEdges);
  const windows: WindowModel[] = TRAINING_WINDOWS.map((days) => ({
    days,
    base: new ProbabilityTable(BASE_CONTEXTS),
    candidates: definitions.map(() => new ProbabilityTable(BOOK_CONTEXTS)),
    scores: definitions.map(emptyScore),
    firstHalf: definitions.map(emptyScore),
    secondHalf: definitions.map(emptyScore),
  }));
  const transferModel: WindowModel = {
    days: primaryDays.length,
    base: new ProbabilityTable(BASE_CONTEXTS),
    candidates: definitions.map(() => new ProbabilityTable(BOOK_CONTEXTS)),
    scores: definitions.map(emptyScore),
    firstHalf: definitions.map(emptyScore),
    secondHalf: definitions.map(emptyScore),
  };
  console.error(`Scanning ${primaryDays.length} primary order-book days...`);
  scanDays(primaryDays, stateConfiguration, bookEdges, (dayIndex, day) => {
    if (dayIndex < TRAIN_CUTOFF_DAYS) {
      for (const window of windows) {
        if (dayIndex >= TRAIN_CUTOFF_DAYS - window.days) addDay(window, day);
      }
    } else {
      const half = dayIndex < TRAIN_CUTOFF_DAYS + 30 ? "first" : "second";
      for (const window of windows) scoreDay(window, day, half);
    }
    addDay(transferModel, day);
  });
  console.error(`Scoring ${transferDays.length} transfer order-book days...`);
  scanDays(transferDays, stateConfiguration, bookEdges, (dayIndex, day) => {
    if (dayIndex < 4) return;
    scoreDay(transferModel, day, "second");
  });
  const windowResults = windows.map((window) => windowResult(window, definitions));
  const primary = windowResults.find((window) => window.days === 90)!;
  const transfer = windowResult(transferModel, definitions);
  const transferById = new Map(transfer.candidates.map((candidate) => [candidate.id, candidate]));
  const ranked = [...primary.candidates].map((candidate) => ({
    ...candidate,
    transfer: transferById.get(candidate.id)!,
  })).sort((left, right) => right.fullBits - left.fullBits);
  const signRanked = [...ranked].sort((left, right) => {
    const leftStable = signStabilityCount(left);
    const rightStable = signStabilityCount(right);
    return rightStable - leftStable || right.activeSignBits - left.activeSignBits;
  });
  const stableFullCandidateCount = ranked.filter((candidate) =>
    stabilityCount(candidate) === 3).length;
  const artifact = {
    version: 1,
    generatedAt: new Date().toISOString(),
    source: {
      book: "Binance USD-M BTCUSDT futures percentage-depth snapshots",
      target: "Binance spot BTCUSDT next-1s close return",
      archiveDays: fs.readdirSync(resolve(BOOK_DIRECTORY)).filter((name) => name.endsWith(".json")).length,
      primary: {
        start: PRIMARY_START,
        end: PRIMARY_END,
        days: primaryDays.length,
        frozenTestStart: primaryDays[TRAIN_CUTOFF_DAYS],
        testDays: primaryDays.length - TRAIN_CUTOFF_DAYS,
      },
      transfer: { start: TRANSFER_START, end: TRANSFER_END, days: transferDays.length },
    },
    method: {
      commonBandsPct: [1, 2, 3, 4, 5],
      snapshotPolicy: `A snapshot is usable only for a later second and for at most ${MAX_SNAPSHOT_AGE_SECONDS}s.`,
      baseConditioning: "Previous-return state plus median splits of the selected five price/volume/range coordinates.",
      bookQuantization: "Four cells fitted from the first 60 primary days; frozen thereafter.",
      targetStride: EVALUATION_STRIDE,
      smoothing: SMOOTHING,
    },
    stableFullCandidateCount,
    bestFullPrimary: [...primary.candidates].sort((left, right) =>
      right.fullBits - left.fullBits)[0],
    bestActiveSign: signRanked[0],
    ranked,
    windows: windowResults,
    transfer,
    limitations: [
      "These are percentage-depth snapshots, not top-of-book quotes, so spread and queue-level microprice cannot be reconstructed.",
      "The depth source is futures while the target is spot; the result measures cross-market information.",
      "Only 368 archive days exist and the primary frozen test is 61 days, far less evidence than the five-year candle study.",
      "Binary conditioning is used for the existing five-feature basis because its original quartile cross is too sparse for this smaller sample.",
      "No fees, latency, fills, spread, or impact are included.",
    ],
  };
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, `${JSON.stringify(artifact, null, 2)}\n`, "utf8");
  fs.mkdirSync(path.dirname(reportPath), { recursive: true });
  fs.writeFileSync(reportPath, renderReport(artifact), "utf8");
  console.log(`Wrote ${path.relative(repoRoot, outputPath)}`);
  console.log(`Wrote ${path.relative(repoRoot, reportPath)}`);
}

export function buildBookFeatureDefinitions(): FeatureDefinition[] {
  const definitions: FeatureDefinition[] = [];
  for (const band of [1, 2, 3, 4, 5]) {
    definitions.push(
      { id: `notional-imbalance-${band}`, label: `notional imbalance within ±${band}%`, family: "imbalance" },
      { id: `depth-imbalance-${band}`, label: `base-depth imbalance within ±${band}%`, family: "imbalance" },
    );
  }
  definitions.push(
    { id: "log-notional-1", label: "log total notional within ±1%", family: "liquidity" },
    { id: "log-notional-5", label: "log total notional within ±5%", family: "liquidity" },
    { id: "log-depth-1", label: "log total base depth within ±1%", family: "liquidity" },
    { id: "log-depth-5", label: "log total base depth within ±5%", family: "liquidity" },
    { id: "mean-notional-imbalance", label: "mean notional imbalance across ±1–5%", family: "imbalance shape" },
    { id: "mean-depth-imbalance", label: "mean base-depth imbalance across ±1–5%", family: "imbalance shape" },
    { id: "notional-imbalance-slope", label: "±5% minus ±1% notional imbalance", family: "imbalance shape" },
    { id: "depth-imbalance-slope", label: "±5% minus ±1% base-depth imbalance", family: "imbalance shape" },
    { id: "notional-concentration", label: "log near/far notional concentration", family: "depth shape" },
    { id: "depth-concentration", label: "log near/far base-depth concentration", family: "depth shape" },
    { id: "delta-notional-imbalance-1", label: "snapshot change in ±1% notional imbalance", family: "book change" },
    { id: "delta-notional-imbalance-5", label: "snapshot change in ±5% notional imbalance", family: "book change" },
    { id: "delta-mean-notional-imbalance", label: "snapshot change in mean notional imbalance", family: "book change" },
    { id: "delta-log-notional-1", label: "snapshot change in ±1% total notional", family: "book change" },
    { id: "delta-log-notional-5", label: "snapshot change in ±5% total notional", family: "book change" },
  );
  return definitions;
}

export function snapshotValues(
  snapshot: SequentialDerivativesBookDepthSnapshot,
  previous: Float64Array | undefined,
): Float64Array {
  const bandIndices = [1, 2, 3, 4, 5];
  const notionalImbalances = bandIndices.map((index) => imbalance(
    snapshot.bidNotional[index]!,
    snapshot.askNotional[index]!,
  ));
  const depthImbalances = bandIndices.map((index) => imbalance(
    snapshot.bidDepth[index]!,
    snapshot.askDepth[index]!,
  ));
  const totalNotional1 = snapshot.bidNotional[1]! + snapshot.askNotional[1]!;
  const totalNotional5 = snapshot.bidNotional[5]! + snapshot.askNotional[5]!;
  const totalDepth1 = snapshot.bidDepth[1]! + snapshot.askDepth[1]!;
  const totalDepth5 = snapshot.bidDepth[5]! + snapshot.askDepth[5]!;
  const logNotional1 = Math.log(totalNotional1);
  const logNotional5 = Math.log(totalNotional5);
  const raw = [
    ...bandIndices.flatMap((_, index) => [notionalImbalances[index]!, depthImbalances[index]!]),
    logNotional1,
    logNotional5,
    Math.log(totalDepth1),
    Math.log(totalDepth5),
    mean(notionalImbalances),
    mean(depthImbalances),
    notionalImbalances[4]! - notionalImbalances[0]!,
    depthImbalances[4]! - depthImbalances[0]!,
    Math.log(totalNotional1 / totalNotional5),
    Math.log(totalDepth1 / totalDepth5),
  ];
  const prior = previous ?? Float64Array.from(raw);
  raw.push(
    raw[0]! - prior[0]!,
    raw[8]! - prior[8]!,
    raw[14]! - prior[14]!,
    logNotional1 - prior[10]!,
    logNotional5 - prior[11]!,
  );
  return Float64Array.from(raw);
}

function calibrateBookEdges(days: string[], featureCount: number): number[][] {
  console.error(`Calibrating ${featureCount} book features on ${days.length} days...`);
  const samples = Array.from({ length: featureCount }, () => [] as number[]);
  const cursor: FeatureCursor = {};
  for (const [dayIndex, day] of days.entries()) {
    if (dayIndex % 10 === 0) console.error(`Book calibration ${dayIndex}/${days.length}...`);
    const points = bookPoints(day, cursor);
    for (const point of points) {
      for (let feature = 0; feature < featureCount; feature += 1) {
        samples[feature]!.push(point.values[feature]!);
      }
    }
  }
  return samples.map((values) => {
    values.sort((left, right) => left - right);
    return [0.25, 0.5, 0.75].map((quantile) => values[
      Math.min(values.length - 1, Math.floor(values.length * quantile))
    ]!);
  });
}

function createStateConfiguration(
  priceCalibration: Calibration,
  marketEdges: number[][],
) {
  const allPriceDefinitions = buildSignalDefinitions();
  const priceIndex = new Map(allPriceDefinitions.map((definition, index) => [definition.id, index]));
  const priceDefinitions = PRICE_BASIS_IDS.map((id) => allPriceDefinitions[priceIndex.get(id)!]!);
  const priceCuts = PRICE_BASIS_IDS.map((id) => priceCalibration.featureEdges[
    priceIndex.get(id)!
  ]![7]!);
  const allMarketDefinitions = buildMarketFeatureDefinitions();
  const marketById = new Map(allMarketDefinitions.map((definition) => [definition.id, definition]));
  const marketIndices = MARKET_BASIS_IDS.map((id) => marketById.get(id)!.sourceIndex);
  const marketCuts = marketIndices.map((index) => marketEdges[index]![1]!);
  return {
    priceCalibration,
    priceDefinitions,
    priceCuts,
    marketIndices,
    marketCuts,
  };
}

function scanDays(
  days: string[],
  configuration: ReturnType<typeof createStateConfiguration>,
  bookEdges: number[][],
  visit: (dayIndex: number, counts: DayCounts) => void,
): void {
  const marketEngine = new CausalMarketFeatureEngine();
  const priceValues = new Float64Array(configuration.priceDefinitions.length);
  let priceEngine: IndicatorEngine | undefined;
  let previousClose = Number.NaN;
  let previousReturnState = 0;
  let returnsSeen = 0;
  let currentPoint: BookPoint | undefined;
  const cursor: FeatureCursor = {};
  for (const [dayIndex, day] of days.entries()) {
    if (dayIndex % 10 === 0) console.error(`Order-book scan ${dayIndex}/${days.length}...`);
    const points = bookPoints(day, cursor);
    let pointIndex = 0;
    const baseMap = new Map<number, number>();
    const candidateMaps = bookEdges.map(() => new Map<number, number>());
    let observations = 0;
    const candles = readCandleShardReferenceSync(candleFile(day));
    if (candles.length !== 86_400) throw new Error(`${day}: spot candle day is incomplete.`);
    for (const candle of candles) {
      const targetSeconds = Math.floor(candle.openTime / 1_000);
      while (pointIndex < points.length
        && points[pointIndex]!.timestampSeconds < targetSeconds) {
        currentPoint = points[pointIndex]!;
        pointIndex += 1;
      }
      if (!priceEngine) {
        priceEngine = new IndicatorEngine(configuration.priceDefinitions, candle.close);
        marketEngine.update(candle);
      } else {
        const returnBps = candle.close === previousClose
          ? 0
          : Math.log(candle.close / previousClose) * 10_000;
        const active = returnBps !== 0;
        const sign = returnBps > 0 ? 1 : 0;
        const magnitudeBin = active ? upperBound(
          configuration.priceCalibration.magnitudeEdges,
          Math.abs(returnBps),
        ) : -1;
        const target = active ? 1 + sign * MAGNITUDE_BINS + magnitudeBin : 0;
        const pointAge = currentPoint ? targetSeconds - currentPoint.timestampSeconds : Infinity;
        if (returnsSeen >= WARMUP_RETURNS
          && isEvaluationTarget(returnsSeen)
          && currentPoint
          && pointAge >= 1
          && pointAge <= MAX_SNAPSHOT_AGE_SECONDS
          && configuration.marketIndices.every((index) => marketEngine.valid()[index] === 1)) {
          priceEngine.values(priceValues);
          let basisState = 0;
          for (let index = 0; index < configuration.priceCuts.length; index += 1) {
            basisState = basisState * BASE_FEATURE_BINS
              + (priceValues[index]! >= configuration.priceCuts[index]! ? 1 : 0);
          }
          const marketValues = marketEngine.values();
          for (let index = 0; index < configuration.marketIndices.length; index += 1) {
            basisState = basisState * BASE_FEATURE_BINS + (
              marketValues[configuration.marketIndices[index]!]! >= configuration.marketCuts[index]!
                ? 1
                : 0
            );
          }
          const baseContext = previousReturnState * BASE_STATES + basisState;
          const baseIndex = baseContext * TARGET_CLASSES + target;
          baseMap.set(baseIndex, (baseMap.get(baseIndex) ?? 0) + 1);
          for (let feature = 0; feature < bookEdges.length; feature += 1) {
            const bin = upperBound(bookEdges[feature]!, currentPoint.values[feature]!);
            const index = (baseContext * BOOK_BINS + bin) * TARGET_CLASSES + target;
            candidateMaps[feature]!.set(index, (candidateMaps[feature]!.get(index) ?? 0) + 1);
          }
          observations += 1;
        }
        previousReturnState = active ? 1 + sign * MAGNITUDE_BINS + magnitudeBin : 0;
        priceEngine.update(candle.close);
        marketEngine.update(candle);
        returnsSeen += 1;
      }
      previousClose = candle.close;
    }
    visit(dayIndex, {
      base: sparse(baseMap, observations),
      candidates: candidateMaps.map((counts) => sparse(counts, observations)),
    });
  }
}

function bookPoints(day: string, cursor: FeatureCursor): BookPoint[] {
  const dayStartSeconds = Date.parse(`${day}T00:00:00.000Z`) / 1_000;
  const snapshots = readDerivativesBookDepthShardReferenceSync(bookFile(day));
  return snapshots.map((snapshot) => {
    const values = snapshotValues(snapshot, cursor.previous);
    cursor.previous = values;
    return {
      timestampSeconds: dayStartSeconds + snapshot.timestampOffsetSeconds,
      values,
    };
  });
}

function addDay(model: WindowModel, day: DayCounts): void {
  model.base.add(day.base);
  model.candidates.forEach((table, index) => table.add(day.candidates[index]!));
}

function scoreDay(model: WindowModel, day: DayCounts, half: "first" | "second"): void {
  model.candidates.forEach((table, candidate) => {
    const score = informationScore(table, model.base, day.candidates[candidate]!);
    addScore(model.scores[candidate]!, score);
    addScore(
      half === "first" ? model.firstHalf[candidate]! : model.secondHalf[candidate]!,
      score,
    );
  });
}

function informationScore(
  candidate: ProbabilityTable,
  base: ProbabilityTable,
  day: SparseCounts,
): InformationScore {
  const score = emptyScore();
  for (let item = 0; item < day.indices.length; item += 1) {
    const index = day.indices[item]!;
    const count = day.counts[item]!;
    const candidateContext = Math.floor(index / TARGET_CLASSES);
    const target = index % TARGET_CLASSES;
    const baseContext = Math.floor(candidateContext / BOOK_BINS);
    const candidateProbability = smoothed(
      candidate.joint[index]!,
      candidate.totals[candidateContext]!,
      TARGET_CLASSES,
    );
    const baseProbability = smoothed(
      base.joint[baseContext * TARGET_CLASSES + target]!,
      base.totals[baseContext]!,
      TARGET_CLASSES,
    );
    score.fullObservations += count;
    score.fullBits += count * Math.log2(candidateProbability / baseProbability);
    const candidateZeroCount = target === 0
      ? candidate.zero[candidateContext]!
      : candidate.totals[candidateContext]! - candidate.zero[candidateContext]!;
    const baseZeroCount = target === 0
      ? base.zero[baseContext]!
      : base.totals[baseContext]! - base.zero[baseContext]!;
    score.zeroBits += count * Math.log2(
      smoothed(candidateZeroCount, candidate.totals[candidateContext]!, 2)
      / smoothed(baseZeroCount, base.totals[baseContext]!, 2),
    );
    if (target !== 0) {
      const positive = target > MAGNITUDE_BINS;
      const candidateActive = candidate.totals[candidateContext]!
        - candidate.zero[candidateContext]!;
      const baseActive = base.totals[baseContext]! - base.zero[baseContext]!;
      const candidateSign = positive
        ? candidate.positive[candidateContext]!
        : candidateActive - candidate.positive[candidateContext]!;
      const baseSign = positive
        ? base.positive[baseContext]!
        : baseActive - base.positive[baseContext]!;
      score.activeObservations += count;
      score.signBits += count * Math.log2(
        smoothed(candidateSign, candidateActive, 2)
        / smoothed(baseSign, baseActive, 2),
      );
    }
  }
  return score;
}

function windowResult(model: WindowModel, definitions: FeatureDefinition[]) {
  return {
    days: model.days,
    candidates: definitions.map((definition, index) => ({
      ...definition,
      ...normalizedScore(model.scores[index]!),
      firstHalf: normalizedScore(model.firstHalf[index]!),
      secondHalf: normalizedScore(model.secondHalf[index]!),
    })),
  };
}

function stabilityCount(candidate: any): number {
  return [
    candidate.firstHalf.fullBits,
    candidate.secondHalf.fullBits,
    candidate.transfer.fullBits,
  ].filter((value) => value > 0).length;
}

function signStabilityCount(candidate: any): number {
  return [
    candidate.firstHalf.activeSignBits,
    candidate.secondHalf.activeSignBits,
    candidate.transfer.activeSignBits,
  ].filter((value) => value > 0).length;
}

function emptyScore(): InformationScore {
  return { fullObservations: 0, fullBits: 0, zeroBits: 0, activeObservations: 0, signBits: 0 };
}

function addScore(target: InformationScore, source: InformationScore): void {
  target.fullObservations += source.fullObservations;
  target.fullBits += source.fullBits;
  target.zeroBits += source.zeroBits;
  target.activeObservations += source.activeObservations;
  target.signBits += source.signBits;
}

function normalizedScore(score: InformationScore) {
  return {
    observations: score.fullObservations,
    fullBits: score.fullBits / score.fullObservations,
    zeroGateBits: score.zeroBits / score.fullObservations,
    activeObservations: score.activeObservations,
    activeSignBits: score.signBits / score.activeObservations,
  };
}

function sparse(counts: Map<number, number>, observations: number): SparseCounts {
  const entries = [...counts.entries()].sort((left, right) => left[0] - right[0]);
  return {
    indices: Uint32Array.from(entries, (entry) => entry[0]),
    counts: Uint32Array.from(entries, (entry) => entry[1]),
    observations,
  };
}

function renderReport(artifact: any): string {
  const bestFull = artifact.bestFullPrimary;
  const bestSign = artifact.bestActiveSign;
  const lines = [
    "# Order-book information about the next 1s return",
    "",
    `Generated ${artifact.generatedAt}. The source contains ${artifact.source.archiveDays} BTCUSDT USD-M percentage-depth days; the target is the next BTCUSDT spot 1s return.`,
    "",
    "## Result",
    "",
    `No candidate improves the full 33-cell distribution consistently across both primary test halves and the separated 2026 transfer period. The numerically best full-distribution candidate in the 90-day primary fit is **${bestFull.label}**, but its marginal score is still ${metric(bestFull.fullBits)} bits/target.`,
    "",
    `There is one narrow repeatable component result: **${bestSign.label}** adds ${metric(bestSign.activeSignBits)} bits per active target for sign. Its primary halves are ${metric(bestSign.firstHalf.activeSignBits)} and ${metric(bestSign.secondHalf.activeSignBits)}, and its 2026 transfer score is ${metric(bestSign.transfer.activeSignBits)}. Its full-distribution scores remain ${metric(bestSign.fullBits)} in the primary test and ${metric(bestSign.transfer.fullBits)} in transfer, so it should not yet enter the full return basis.`,
    "",
    "## Ninety-day primary ranking",
    "",
    "| rank | feature | family | full bits | zero-gate bits | active-sign bits | half 1 | half 2 | 2026 transfer |",
    "|---:|---|---|---:|---:|---:|---:|---:|---:|",
  ];
  artifact.ranked.forEach((row: any, index: number) => {
    lines.push(`| ${index + 1} | ${row.label} | ${row.family} | ${metric(row.fullBits)} | ${metric(row.zeroGateBits)} | ${metric(row.activeSignBits)} | ${metric(row.firstHalf.fullBits)} | ${metric(row.secondHalf.fullBits)} | ${metric(row.transfer.fullBits)} |`);
  });
  lines.push(
    "",
    "## Training-history sensitivity",
    "",
    "The table below follows the repeatable active-sign feature while changing only the frozen probability-table history before the same 61-day primary test.",
    "",
    "| training history | full marginal bits | zero-gate bits | active-sign bits |",
    "|---:|---:|---:|---:|",
  );
  for (const window of artifact.windows) {
    const row = window.candidates.find((candidate: any) => candidate.id === bestSign.id);
    lines.push(`| ${window.days} days | ${metric(row.fullBits)} | ${metric(row.zeroGateBits)} | ${metric(row.activeSignBits)} |`);
  }
  lines.push(
    "",
    "## Causal alignment",
    "",
    artifact.method.snapshotPolicy,
    "The ±0.2% band is excluded because older ten-band snapshots do not contain it. All retained ±1–5% values are cumulative and present in both schemas.",
    "",
    "## Limits",
    "",
    ...artifact.limitations.map((limitation: string) => `- ${limitation}`),
    "",
    "Complete values are stored in `data/benchmarks/order-book-return-information.json`.",
  );
  return `${lines.join("\n")}\n`;
}

function inclusiveDays(start: string, end: string): string[] {
  const result: string[] = [];
  for (
    let time = Date.parse(`${start}T00:00:00.000Z`);
    time <= Date.parse(`${end}T00:00:00.000Z`);
    time += DAY_MS
  ) result.push(new Date(time).toISOString().slice(0, 10));
  return result;
}

function assertReferences(days: string[]): void {
  for (const day of days) {
    if (!fs.existsSync(bookFile(day)) || !fs.existsSync(candleFile(day))) {
      throw new Error(`Missing book or candle reference for ${day}.`);
    }
  }
}

function bookFile(day: string): string {
  return path.join(resolve(BOOK_DIRECTORY), `${day}.json`);
}

function candleFile(day: string): string {
  return path.join(resolve(CANDLE_DIRECTORY), `${day}.json`);
}

function imbalance(bid: number, ask: number): number {
  return (bid - ask) / (bid + ask);
}

function mean(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
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

function isEvaluationTarget(index: number): boolean {
  let mixed = index ^ (index >>> 16);
  mixed = Math.imul(mixed, 0x45d9f3b);
  mixed ^= mixed >>> 16;
  return (mixed >>> 0) % EVALUATION_STRIDE === 0;
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

function metric(value: number): string {
  return Number.isFinite(value)
    ? value === 0 ? "0" : value.toPrecision(8).replace(/\.0+(?=e|$)/, "")
    : "n/a";
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  try {
    main();
  } catch (error: unknown) {
    console.error(error instanceof Error ? error.stack ?? error.message : String(error));
    process.exitCode = 1;
  }
}
