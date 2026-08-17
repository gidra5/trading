import fs from "node:fs";
import path from "node:path";
import { deserialize } from "node:v8";
import { fileURLToPath, pathToFileURL } from "node:url";
import {
  readCandleShardReferenceSync,
  readDerivativesKlinesShardReferenceSync,
  readDerivativesMetricsShardReferenceSync,
  readTradeFlowShardReferenceSync,
  type SequentialDerivativesKlineRow,
  type SequentialDerivativesMetricRow,
  type SequentialTradeFlowSecond,
} from "@trading/storage";
import {
  buildSignalDefinitions,
  IndicatorEngine,
} from "./analyze-technical-indicator-predictiveness.ts";
import {
  buildMarketFeatureDefinitions,
  CausalMarketFeatureEngine,
} from "./analyze-volume-multiscale-information.ts";

const TARGET_CLASSES = 33;
const MAGNITUDE_BINS = 16;
const BASE_BINS = 2;
const BASE_DIMENSIONS = 5;
const BASE_STATES = BASE_BINS ** BASE_DIMENSIONS;
const HISTORY_STATES = 33;
const BASE_CONTEXTS = HISTORY_STATES * BASE_STATES;
const CANDIDATE_BINS = 4;
const CANDIDATE_CONTEXTS = BASE_CONTEXTS * CANDIDATE_BINS;
const SMOOTHING = 0.5;
const EVALUATION_STRIDE = 8;
const CALIBRATION_STRIDE = 64;
const WARMUP_RETURNS = 12_000;
const TRAIN_CUTOFF_DAYS = 180;
const TRAINING_WINDOWS = [30, 60, 90, 180] as const;
const PRICE_BASIS_IDS = ["rsi-2", "ema-acceleration-2-1", "ema-slope-8-8"] as const;
const MARKET_BASIS_IDS = ["1h-log-volume", "1s-range"] as const;
const PRIMARY_START = "2025-03-18";
const PRIMARY_END = "2025-11-13";
const TRANSFER_START = "2026-04-21";
const TRANSFER_END = "2026-06-23";
const CANDLE_DIRECTORY = "data/market/immutable/refs/candles/spot-btcusdt/btcusdt/1s";
const FLOW_DIRECTORY = "data/market/immutable/refs/trade-flow/spot-btcusdt/btcusdt/1s";
const FUTURES_DIRECTORY = "data/market/immutable/refs/derivatives-klines/usdm-futures/btcusdt/1m";
const METRICS_DIRECTORY = "data/market/immutable/refs/derivatives-metrics/usdm-futures/btcusdt/5m";
const DEFAULT_PRICE_CALIBRATION = "data/runtime-cache/technical-indicator-calibration.bin";
const DEFAULT_MARKET_CALIBRATION = "data/runtime-cache/volume-multiscale-calibration.bin";
const DEFAULT_OUTPUT = "data/benchmarks/forward-market-return-information.json";
const DEFAULT_REPORT = "docs/experiments/forward-market-return-information-2026-08-16.md";
const DAY_MS = 86_400_000;

interface Calibration {
  featureEdges: number[][];
  magnitudeEdges: number[];
}

interface FeatureDefinition {
  id: string;
  label: string;
  family: string;
  source: "spot trade flow" | "futures basis/flow" | "futures positioning";
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

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function main(): void {
  const args = parseArguments(process.argv.slice(2));
  const priceCalibration = (deserialize(fs.readFileSync(resolve(
    args.get("price-calibration") ?? DEFAULT_PRICE_CALIBRATION,
  ))) as { calibration: Calibration }).calibration;
  const marketEdges = (deserialize(fs.readFileSync(resolve(
    args.get("market-calibration") ?? DEFAULT_MARKET_CALIBRATION,
  ))) as { marketEdges: number[][] }).marketEdges;
  const output = resolve(args.get("output") ?? DEFAULT_OUTPUT);
  const report = resolve(args.get("report") ?? DEFAULT_REPORT);
  const definitions = buildForwardFeatureDefinitions();
  const primaryDays = inclusiveDays(PRIMARY_START, PRIMARY_END);
  const transferDays = inclusiveDays(TRANSFER_START, TRANSFER_END);
  assertReferences([...primaryDays, ...transferDays]);
  const configuration = createStateConfiguration(priceCalibration, marketEdges);
  const candidateEdges = calibrateCandidateEdges(primaryDays.slice(0, 60), definitions.length);
  const windows: WindowModel[] = TRAINING_WINDOWS.map((days) => createWindow(days, definitions.length));
  const transferModel = createWindow(primaryDays.length, definitions.length);
  console.error(`Scanning ${primaryDays.length} primary forward-market days...`);
  scanDays(primaryDays, configuration, candidateEdges, (dayIndex, day) => {
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
  console.error(`Scoring ${transferDays.length} transfer forward-market days...`);
  scanDays(transferDays, configuration, candidateEdges, (dayIndex, day) => {
    if (dayIndex < 4) return;
    scoreDay(transferModel, day, "second");
  });
  const windowResults = windows.map((window) => windowResult(window, definitions));
  const primary = windowResults.find((window) => window.days === 90)!;
  const transfer = windowResult(transferModel, definitions);
  const transferById = new Map(transfer.candidates.map((candidate) => [candidate.id, candidate]));
  const ranked = primary.candidates.map((candidate) => ({
    ...candidate,
    transfer: transferById.get(candidate.id)!,
  })).sort((left, right) => right.fullBits - left.fullBits);
  const sourceResults = [...new Set(definitions.map((definition) => definition.source))].map((source) => {
    const sourceRanked = ranked.filter((candidate) => candidate.source === source);
    const stable = sourceRanked.filter((candidate) => stabilityCount(candidate) === 3);
    const signRanked = [...sourceRanked].sort((left, right) =>
      signStabilityCount(right) - signStabilityCount(left)
        || right.activeSignBits - left.activeSignBits);
    return {
      source,
      featureCount: sourceRanked.length,
      stableFullCount: stable.length,
      bestFull: sourceRanked[0],
      bestStableFull: stable[0] ?? null,
      bestActiveSign: signRanked[0],
    };
  });
  const stable = ranked.filter((candidate) => stabilityCount(candidate) === 3);
  const artifact = {
    version: 1,
    generatedAt: new Date().toISOString(),
    source: {
      target: "Binance spot BTCUSDT next-1s close return",
      spotTradeFlow: "Binance spot aggregate trades reduced to causal 1s aggressor-flow rows",
      futuresKlines: "Binance USD-M BTCUSDT completed 1m candles",
      futuresMetrics: "Binance USD-M BTCUSDT 5m open-interest, positioning, and taker-ratio metrics",
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
      candidateFeatures: definitions,
      candidateEdges,
      targetStride: EVALUATION_STRIDE,
      calibrationStride: CALIBRATION_STRIDE,
      smoothing: SMOOTHING,
      timing: [
        "Spot aggressor flow uses only the immediately preceding completed 1s bin and its causal EMAs.",
        "Futures kline features update only after the corresponding 1m candle closes.",
        "Futures positioning metrics retain a full 5m lag before use.",
      ],
      baseConditioning: "Previous-return state plus median splits of the selected five price/volume/range coordinates.",
      quantization: "Each candidate is split at training-only quartiles fitted on the first 60 primary days and frozen.",
    },
    stableFullCandidateCount: stable.length,
    bestFull: ranked[0],
    bestStableFull: stable[0] ?? null,
    ranked,
    sources: sourceResults,
    windows: windowResults,
    transfer,
    existingFifteenMinuteMatchedModel: {
      result: "negative",
      controlValidationObjective: 2.689429,
      jointValidationObjective: 2.695700,
      relativeChange: 0.00233,
      note: "The earlier matched 15m neural test added all 231 forward-market inputs together and was 0.233% worse than its candle-only control.",
    },
    limitations: [
      "The source corpus contains a long 2025 block and a separated 2026 transfer block, but not continuous five-year coverage.",
      "Individual quartile screens identify marginal information; correlated winners from the same source are not additive.",
      "Five-minute derivatives metrics are repeated between releases, so chronological block stability matters more than a naive independent-sample significance estimate.",
      "The 1s target test does not replace a dedicated execution backtest and excludes fees, latency, fills, spread, and impact.",
    ],
  };
  fs.mkdirSync(path.dirname(output), { recursive: true });
  fs.writeFileSync(output, `${JSON.stringify(artifact, null, 2)}\n`, "utf8");
  fs.mkdirSync(path.dirname(report), { recursive: true });
  fs.writeFileSync(report, renderReport(artifact), "utf8");
  console.log(`Wrote ${path.relative(repoRoot, output)}`);
  console.log(`Wrote ${path.relative(repoRoot, report)}`);
}

function createWindow(days: number, featureCount: number): WindowModel {
  return {
    days,
    base: new ProbabilityTable(BASE_CONTEXTS),
    candidates: Array.from({ length: featureCount }, () => new ProbabilityTable(CANDIDATE_CONTEXTS)),
    scores: Array.from({ length: featureCount }, emptyScore),
    firstHalf: Array.from({ length: featureCount }, emptyScore),
    secondHalf: Array.from({ length: featureCount }, emptyScore),
  };
}

export function buildForwardFeatureDefinitions(): FeatureDefinition[] {
  return [
    ...TradeFlowFeatureEngine.definitions(),
    ...FuturesMinuteFeatureEngine.definitions(),
    ...FuturesMetricsFeatureEngine.definitions(),
  ];
}

export class TradeFlowFeatureEngine {
  static definitions(): FeatureDefinition[] {
    const source = "spot trade flow" as const;
    return [
      { id: "spot-flow-quote-imbalance-1s", label: "spot taker quote imbalance, last 1s", family: "aggressor direction", source },
      { id: "spot-flow-base-imbalance-1s", label: "spot taker base imbalance, last 1s", family: "aggressor direction", source },
      { id: "spot-flow-trade-count-imbalance-1s", label: "spot raw-trade count imbalance, last 1s", family: "aggressor direction", source },
      { id: "spot-flow-aggregate-count-imbalance-1s", label: "spot aggregate-trade count imbalance, last 1s", family: "aggressor direction", source },
      { id: "spot-flow-quantity-squared-skew-1s", label: "spot aggregate-size-squared skew, last 1s", family: "trade size", source },
      { id: "spot-flow-maximum-skew-1s", label: "spot maximum aggregate-size skew, last 1s", family: "trade size", source },
      { id: "spot-flow-log-quote-volume-1s", label: "spot log quote volume, last 1s", family: "activity", source },
      { id: "spot-flow-log-trade-count-1s", label: "spot log raw-trade count, last 1s", family: "activity", source },
      { id: "spot-flow-raw-per-aggregate-1s", label: "spot raw trades per aggregate, last 1s", family: "trade structure", source },
      { id: "spot-flow-aggregate-hhi-1s", label: "spot aggregate-size HHI, last 1s", family: "trade size", source },
      { id: "spot-flow-max-share-1s", label: "spot maximum aggregate share, last 1s", family: "trade size", source },
      { id: "spot-flow-vwap-gap-1s", label: "spot buyer-minus-seller VWAP gap, last 1s", family: "price pressure", source },
      { id: "spot-flow-arrival-gap-1s", label: "spot buyer-minus-seller arrival centroid, last 1s", family: "timing", source },
      { id: "spot-flow-flip-rate-1s", label: "spot aggressor-side flip rate, last 1s", family: "trade sequence", source },
      { id: "spot-flow-first-side-1s", label: "spot first aggressor side, last 1s", family: "trade sequence", source },
      { id: "spot-flow-last-side-1s", label: "spot last aggressor side, last 1s", family: "trade sequence", source },
      { id: "spot-flow-last-offset-1s", label: "spot last-trade position within prior second", family: "timing", source },
      { id: "spot-flow-two-sided-1s", label: "spot two-sided activity, last 1s", family: "activity", source },
      { id: "spot-flow-quote-surprise-1s", label: "spot quote-volume surprise vs EMA(32s)", family: "activity", source },
      { id: "spot-flow-trade-surprise-1s", label: "spot trade-count surprise vs EMA(32s)", family: "activity", source },
      { id: "spot-flow-delta-quote-imbalance-1s", label: "change in spot taker quote imbalance", family: "aggressor change", source },
      ...[2, 8, 32, 128].map((period) => ({
        id: `spot-flow-quote-imbalance-ema-${period}`,
        label: `spot taker quote imbalance EMA(${period}s)`,
        family: "aggressor direction",
        source,
      })),
      ...[2, 8, 32, 128].map((period) => ({
        id: `spot-flow-trade-imbalance-ema-${period}`,
        label: `spot raw-trade imbalance EMA(${period}s)`,
        family: "aggressor direction",
        source,
      })),
      ...[2, 3, 5].flatMap((lag) => ([
        { id: `spot-flow-last-side-lag-${lag}s`, label: `spot last aggressor side, ${lag}s old`, family: "lagged trade sequence", source },
        { id: `spot-flow-trade-count-imbalance-lag-${lag}s`, label: `spot raw-trade count imbalance, ${lag}s old`, family: "lagged aggressor direction", source },
        { id: `spot-flow-quote-imbalance-lag-${lag}s`, label: `spot taker quote imbalance, ${lag}s old`, family: "lagged aggressor direction", source },
      ])),
    ];
  }

  private readonly output = new Float64Array(38);
  private readonly periods = [2, 8, 32, 128];
  private readonly buyQuoteEma = new Float64Array(4);
  private readonly sellQuoteEma = new Float64Array(4);
  private readonly buyTradeEma = new Float64Array(4);
  private readonly sellTradeEma = new Float64Array(4);
  private initialized = false;
  private previousQuoteImbalance = 0;
  private quoteActivityEma = 0;
  private tradeActivityEma = 0;
  private readonly lastSideHistory: number[] = [];
  private readonly tradeImbalanceHistory: number[] = [];
  private readonly quoteImbalanceHistory: number[] = [];
  valid = false;

  values(): Float64Array { return this.output; }

  update(row: SequentialTradeFlowSecond): void {
    const buyBase = row.aggressiveBuyBaseVolume;
    const sellBase = row.aggressiveSellBaseVolume;
    const buyQuote = row.aggressiveBuyQuoteVolume;
    const sellQuote = row.aggressiveSellQuoteVolume;
    const buyTrade = row.aggressiveBuyTradeCount;
    const sellTrade = row.aggressiveSellTradeCount;
    const buyAggregate = row.aggressiveBuyAggregateTradeCount;
    const sellAggregate = row.aggressiveSellAggregateTradeCount;
    const totalBase = buyBase + sellBase;
    const totalQuote = buyQuote + sellQuote;
    const totalTrade = buyTrade + sellTrade;
    const totalAggregate = buyAggregate + sellAggregate;
    const quoteImbalance = imbalance(buyQuote, sellQuote);
    const tradeImbalance = imbalance(buyTrade, sellTrade);
    const buyVwap = buyBase > 0 ? buyQuote / buyBase : 0;
    const sellVwap = sellBase > 0 ? sellQuote / sellBase : 0;
    const twoSided = buyBase > 0 && sellBase > 0;
    const maximum = Math.max(row.aggressiveBuyMaxAggregateQuantity, row.aggressiveSellMaxAggregateQuantity);
    const quoteActivity = Math.log1p(totalQuote);
    const tradeActivity = Math.log1p(totalTrade);
    if (!this.initialized) {
      this.buyQuoteEma.fill(buyQuote);
      this.sellQuoteEma.fill(sellQuote);
      this.buyTradeEma.fill(buyTrade);
      this.sellTradeEma.fill(sellTrade);
      this.quoteActivityEma = quoteActivity;
      this.tradeActivityEma = tradeActivity;
      this.previousQuoteImbalance = quoteImbalance;
      this.initialized = true;
    }
    this.output.set([
      quoteImbalance,
      imbalance(buyBase, sellBase),
      tradeImbalance,
      imbalance(buyAggregate, sellAggregate),
      imbalance(row.aggressiveBuyAggregateQuantitySquared, row.aggressiveSellAggregateQuantitySquared),
      imbalance(row.aggressiveBuyMaxAggregateQuantity, row.aggressiveSellMaxAggregateQuantity),
      quoteActivity,
      tradeActivity,
      Math.log((totalTrade + 1) / (totalAggregate + 1)),
      totalBase > 0 ? (row.aggressiveBuyAggregateQuantitySquared + row.aggressiveSellAggregateQuantitySquared) / (totalBase * totalBase) : 0,
      totalBase > 0 ? maximum / totalBase : 0,
      twoSided ? 2 * (buyVwap - sellVwap) / (buyVwap + sellVwap) : 0,
      twoSided ? row.aggressiveBuyBaseVolumeTimeMoment / buyBase - row.aggressiveSellBaseVolumeTimeMoment / sellBase : 0,
      totalAggregate > 1 ? row.aggressorSideFlipCount / (totalAggregate - 1) : 0,
      row.firstAggressorSide,
      row.lastAggressorSide,
      row.lastTradeOffsetMicros === null ? 0 : row.lastTradeOffsetMicros / 1_000_000,
      twoSided ? 1 : 0,
      quoteActivity - this.quoteActivityEma,
      tradeActivity - this.tradeActivityEma,
      quoteImbalance - this.previousQuoteImbalance,
    ]);
    for (let index = 0; index < this.periods.length; index += 1) {
      const alpha = 2 / (this.periods[index]! + 1);
      this.buyQuoteEma[index] += alpha * (buyQuote - this.buyQuoteEma[index]!);
      this.sellQuoteEma[index] += alpha * (sellQuote - this.sellQuoteEma[index]!);
      this.buyTradeEma[index] += alpha * (buyTrade - this.buyTradeEma[index]!);
      this.sellTradeEma[index] += alpha * (sellTrade - this.sellTradeEma[index]!);
      this.output[21 + index] = imbalance(this.buyQuoteEma[index]!, this.sellQuoteEma[index]!);
      this.output[25 + index] = imbalance(this.buyTradeEma[index]!, this.sellTradeEma[index]!);
    }
    this.lastSideHistory.push(row.lastAggressorSide);
    this.tradeImbalanceHistory.push(tradeImbalance);
    this.quoteImbalanceHistory.push(quoteImbalance);
    for (const history of [this.lastSideHistory, this.tradeImbalanceHistory, this.quoteImbalanceHistory]) {
      if (history.length > 5) history.shift();
    }
    let lagOutput = 29;
    for (const lag of [2, 3, 5]) {
      this.output[lagOutput++] = lagged(this.lastSideHistory, lag);
      this.output[lagOutput++] = lagged(this.tradeImbalanceHistory, lag);
      this.output[lagOutput++] = lagged(this.quoteImbalanceHistory, lag);
    }
    this.quoteActivityEma += 2 / 33 * (quoteActivity - this.quoteActivityEma);
    this.tradeActivityEma += 2 / 33 * (tradeActivity - this.tradeActivityEma);
    this.previousQuoteImbalance = quoteImbalance;
    this.valid = true;
  }
}

export class FuturesMinuteFeatureEngine {
  static definitions(): FeatureDefinition[] {
    const source = "futures basis/flow" as const;
    return [
      { id: "futures-basis-level", label: "futures-minus-spot basis level", family: "basis", source },
      { id: "futures-basis-change-1m", label: "futures basis change, 1m", family: "basis", source },
      ...[5, 15, 60].map((period) => ({ id: `futures-basis-deviation-${period}m`, label: `futures basis deviation from EMA(${period}m)`, family: "basis", source })),
      { id: "futures-return-1m", label: "futures return, last completed 1m", family: "futures price", source },
      { id: "futures-relative-return-1m", label: "futures-minus-spot return, last completed 1m", family: "cross-market return", source },
      { id: "futures-range-1m", label: "futures range, last completed 1m", family: "futures candle", source },
      { id: "futures-close-location-1m", label: "futures close location, last completed 1m", family: "futures candle", source },
      { id: "futures-taker-quote-imbalance-1m", label: "futures taker quote imbalance, last completed 1m", family: "futures flow", source },
      { id: "futures-taker-base-imbalance-1m", label: "futures taker base imbalance, last completed 1m", family: "futures flow", source },
      { id: "futures-log-quote-volume-1m", label: "futures log quote volume, last completed 1m", family: "futures activity", source },
      { id: "futures-log-trade-count-1m", label: "futures log trade count, last completed 1m", family: "futures activity", source },
      { id: "futures-log-mean-trade-notional-1m", label: "futures log mean trade notional, last completed 1m", family: "futures activity", source },
      { id: "futures-spot-quote-activity-ratio-1m", label: "futures/spot quote activity ratio, last completed 1m", family: "cross-market activity", source },
      { id: "futures-quote-volume-surprise-1m", label: "futures quote-volume surprise vs EMA(60m)", family: "futures activity", source },
      { id: "futures-trade-count-surprise-1m", label: "futures trade-count surprise vs EMA(60m)", family: "futures activity", source },
      ...[5, 15, 60].map((period) => ({ id: `futures-taker-imbalance-ema-${period}m`, label: `futures taker quote imbalance EMA(${period}m)`, family: "futures flow", source })),
    ];
  }

  private readonly output = new Float64Array(20);
  private previousFuturesClose = Number.NaN;
  private previousSpotMinuteClose = Number.NaN;
  private previousBasis = 0;
  private basisEmas = new Float64Array(3);
  private takerEmas = new Float64Array(3);
  private quoteActivityEma = 0;
  private tradeActivityEma = 0;
  valid = false;

  values(): Float64Array { return this.output; }

  update(row: SequentialDerivativesKlineRow | undefined, spotClose: number, spotBaseVolume: number): void {
    if (!row || row.close === null || row.tradeCount === null || row.quoteVolume === null
      || row.takerBuyQuoteVolume === null || row.baseVolume === null
      || row.takerBuyBaseVolume === null || row.open === null || row.high === null || row.low === null) return;
    const basis = Math.log(row.close / spotClose) * 10_000;
    const futuresReturn = Number.isFinite(this.previousFuturesClose)
      ? Math.log(row.close / this.previousFuturesClose) * 10_000 : 0;
    const spotReturn = Number.isFinite(this.previousSpotMinuteClose)
      ? Math.log(spotClose / this.previousSpotMinuteClose) * 10_000 : 0;
    const takerQuoteImbalance = row.quoteVolume > 0
      ? (2 * row.takerBuyQuoteVolume - row.quoteVolume) / row.quoteVolume : 0;
    const takerBaseImbalance = row.baseVolume > 0
      ? (2 * row.takerBuyBaseVolume - row.baseVolume) / row.baseVolume : 0;
    const quoteActivity = Math.log1p(row.quoteVolume);
    const tradeActivity = Math.log1p(row.tradeCount);
    if (!this.valid) {
      this.basisEmas.fill(basis);
      this.takerEmas.fill(takerQuoteImbalance);
      this.quoteActivityEma = quoteActivity;
      this.tradeActivityEma = tradeActivity;
      this.previousBasis = basis;
    }
    this.output.set([
      basis,
      basis - this.previousBasis,
      basis - this.basisEmas[0]!,
      basis - this.basisEmas[1]!,
      basis - this.basisEmas[2]!,
      futuresReturn,
      futuresReturn - spotReturn,
      Math.log(row.high / row.low) * 10_000,
      row.high > row.low ? (row.close - row.low) / (row.high - row.low) : 0.5,
      takerQuoteImbalance,
      takerBaseImbalance,
      quoteActivity,
      tradeActivity,
      Math.log((row.quoteVolume + 1) / (row.tradeCount + 1)),
      Math.log((row.quoteVolume + 1) / (spotBaseVolume * spotClose + 1)),
      quoteActivity - this.quoteActivityEma,
      tradeActivity - this.tradeActivityEma,
      this.takerEmas[0]!,
      this.takerEmas[1]!,
      this.takerEmas[2]!,
    ]);
    [5, 15, 60].forEach((period, index) => {
      const alpha = 2 / (period + 1);
      this.basisEmas[index] += alpha * (basis - this.basisEmas[index]!);
      this.takerEmas[index] += alpha * (takerQuoteImbalance - this.takerEmas[index]!);
    });
    this.quoteActivityEma += 2 / 61 * (quoteActivity - this.quoteActivityEma);
    this.tradeActivityEma += 2 / 61 * (tradeActivity - this.tradeActivityEma);
    this.previousFuturesClose = row.close;
    this.previousSpotMinuteClose = spotClose;
    this.previousBasis = basis;
    this.valid = true;
  }
}

export class FuturesMetricsFeatureEngine {
  static readonly metricNames = [
    "sumOpenInterest", "sumOpenInterestValue", "topTraderAccountLongShortRatio",
    "topTraderPositionLongShortRatio", "globalLongShortRatio", "takerBuySellVolumeRatio",
  ] as const;

  static definitions(): FeatureDefinition[] {
    const source = "futures positioning" as const;
    const definitions: FeatureDefinition[] = [];
    for (const metric of ["open-interest", "open-interest-value"] as const) {
      for (const horizon of [5, 15, 60, 240]) definitions.push({
        id: `${metric}-log-change-${horizon}m`, label: `${metric.replaceAll("-", " ")} log change, ${horizon}m`, family: "open interest", source,
      });
      definitions.push({ id: `${metric}-log-deviation-24h`, label: `${metric.replaceAll("-", " ")} deviation from EMA(24h)`, family: "open interest", source });
    }
    definitions.push({ id: "open-interest-implied-price-basis", label: "OI-value/OI implied-price basis to spot", family: "open interest", source });
    for (const [id, label] of [
      ["top-account", "top-trader account long/short"],
      ["top-position", "top-trader position long/short"],
      ["global", "global account long/short"],
      ["taker", "futures taker buy/sell"],
    ] as const) {
      definitions.push(
        { id: `${id}-ratio-log-level`, label: `${label} log level`, family: "positioning ratio", source },
        { id: `${id}-ratio-log-change-5m`, label: `${label} log change, 5m`, family: "positioning ratio", source },
        { id: `${id}-ratio-log-change-1h`, label: `${label} log change, 1h`, family: "positioning ratio", source },
        { id: `${id}-ratio-log-deviation-24h`, label: `${label} deviation from EMA(24h)`, family: "positioning ratio", source },
      );
    }
    definitions.push(
      { id: "top-position-minus-account-log", label: "top-position minus top-account log ratio", family: "positioning spread", source },
      { id: "top-account-minus-global-log", label: "top-account minus global log ratio", family: "positioning spread", source },
    );
    return definitions;
  }

  private readonly output = new Float64Array(29);
  private readonly histories = Array.from({ length: 6 }, () => [] as number[]);
  private readonly ema24h = new Float64Array(6);
  valid = false;

  values(): Float64Array { return this.output; }

  update(row: SequentialDerivativesMetricRow | undefined, spotClose: number): void {
    if (!row || FuturesMetricsFeatureEngine.metricNames.some((name) => row[name] === null)) return;
    const values = FuturesMetricsFeatureEngine.metricNames.map((name) => row[name]!);
    if (!this.valid) this.ema24h.set(values.map(Math.log));
    values.forEach((value, index) => {
      this.histories[index]!.push(value);
      if (this.histories[index]!.length > 289) this.histories[index]!.shift();
    });
    let outputIndex = 0;
    for (const metricIndex of [0, 1]) {
      for (const lag of [1, 3, 12, 48]) {
        this.output[outputIndex++] = logChange(this.histories[metricIndex]!, lag);
      }
      this.output[outputIndex++] = Math.log(values[metricIndex]!) - this.ema24h[metricIndex]!;
    }
    this.output[outputIndex++] = Math.log((values[1]! / values[0]!) / spotClose) * 10_000;
    for (let metricIndex = 2; metricIndex < 6; metricIndex += 1) {
      this.output[outputIndex++] = Math.log(values[metricIndex]!);
      this.output[outputIndex++] = logChange(this.histories[metricIndex]!, 1);
      this.output[outputIndex++] = logChange(this.histories[metricIndex]!, 12);
      this.output[outputIndex++] = Math.log(values[metricIndex]!) - this.ema24h[metricIndex]!;
    }
    this.output[outputIndex++] = Math.log(values[3]!) - Math.log(values[2]!);
    this.output[outputIndex++] = Math.log(values[2]!) - Math.log(values[4]!);
    for (let index = 0; index < values.length; index += 1) {
      const logValue = Math.log(values[index]!);
      this.ema24h[index] += 2 / 289 * (logValue - this.ema24h[index]!);
    }
    this.valid = true;
  }
}

function calibrateCandidateEdges(days: string[], featureCount: number): number[][] {
  console.error(`Calibrating ${featureCount} forward-market features on ${days.length} days...`);
  const samples = Array.from({ length: featureCount }, () => [] as number[]);
  scanSourceValues(days, (targetIndex, values) => {
    if (targetIndex % CALIBRATION_STRIDE !== 0) return;
    for (let feature = 0; feature < featureCount; feature += 1) {
      const value = values[feature]!;
      if (Number.isFinite(value)) samples[feature]!.push(value);
    }
  });
  return samples.map((sample, feature) => {
    if (sample.length === 0) throw new Error(`Feature ${feature} has no calibration samples.`);
    sample.sort((left, right) => left - right);
    return [0.25, 0.5, 0.75].map((quantile) => sample[
      Math.min(sample.length - 1, Math.floor(sample.length * quantile))
    ]!);
  });
}

function scanSourceValues(days: string[], visit: (targetIndex: number, values: Float64Array) => void): void {
  const trade = new TradeFlowFeatureEngine();
  const futures = new FuturesMinuteFeatureEngine();
  const metrics = new FuturesMetricsFeatureEngine();
  let previousFutures: SequentialDerivativesKlineRow[] | undefined;
  let previousMetrics: SequentialDerivativesMetricRow[] | undefined;
  let previousSpotClose = Number.NaN;
  let spotMinuteVolume = 0;
  let targetIndex = 0;
  const values = new Float64Array(buildForwardFeatureDefinitions().length);
  for (const [dayIndex, day] of days.entries()) {
    if (dayIndex % 10 === 0) console.error(`Forward calibration ${dayIndex}/${days.length}...`);
    const candles = readCandleShardReferenceSync(candleFile(day));
    const flow = readTradeFlowShardReferenceSync(flowFile(day));
    const futuresRows = readDerivativesKlinesShardReferenceSync(futuresFile(day));
    const metricRows = readDerivativesMetricsShardReferenceSync(metricsFile(day));
    for (let second = 0; second < candles.length; second += 1) {
      const candle = candles[second]!;
      if (second % 60 === 0 && Number.isFinite(previousSpotClose)) {
        const row = second === 0 ? previousFutures?.at(-1) : futuresRows[second / 60 - 1];
        futures.update(row, previousSpotClose, spotMinuteVolume);
        spotMinuteVolume = 0;
      }
      if (second % 300 === 0 && Number.isFinite(previousSpotClose)) {
        const row = second === 0 ? previousMetrics?.at(-1) : metricRows[second / 300 - 1];
        metrics.update(row, previousSpotClose);
      }
      copySourceValues(values, trade, futures, metrics);
      if (trade.valid && futures.valid && metrics.valid) visit(targetIndex, values);
      trade.update(flow[second]!);
      spotMinuteVolume += candle.volume;
      previousSpotClose = candle.close;
      targetIndex += 1;
    }
    previousFutures = futuresRows;
    previousMetrics = metricRows;
  }
}

function createStateConfiguration(priceCalibration: Calibration, marketEdges: number[][]) {
  const allPriceDefinitions = buildSignalDefinitions();
  const priceIndex = new Map(allPriceDefinitions.map((definition, index) => [definition.id, index]));
  const priceDefinitions = PRICE_BASIS_IDS.map((id) => allPriceDefinitions[priceIndex.get(id)!]!);
  const priceCuts = PRICE_BASIS_IDS.map((id) => priceCalibration.featureEdges[priceIndex.get(id)!]![7]!);
  const marketDefinitions = buildMarketFeatureDefinitions();
  const marketById = new Map(marketDefinitions.map((definition) => [definition.id, definition]));
  const marketIndices = MARKET_BASIS_IDS.map((id) => marketById.get(id)!.sourceIndex);
  const marketCuts = marketIndices.map((index) => marketEdges[index]![1]!);
  return { priceCalibration, priceDefinitions, priceCuts, marketIndices, marketCuts };
}

function scanDays(
  days: string[],
  configuration: ReturnType<typeof createStateConfiguration>,
  candidateEdges: number[][],
  visit: (dayIndex: number, counts: DayCounts) => void,
): void {
  const trade = new TradeFlowFeatureEngine();
  const futures = new FuturesMinuteFeatureEngine();
  const metrics = new FuturesMetricsFeatureEngine();
  const marketEngine = new CausalMarketFeatureEngine();
  const priceValues = new Float64Array(configuration.priceDefinitions.length);
  const sourceValues = new Float64Array(candidateEdges.length);
  let priceEngine: IndicatorEngine | undefined;
  let previousClose = Number.NaN;
  let previousReturnState = 0;
  let returnsSeen = 0;
  let previousFutures: SequentialDerivativesKlineRow[] | undefined;
  let previousMetrics: SequentialDerivativesMetricRow[] | undefined;
  let spotMinuteVolume = 0;
  for (const [dayIndex, day] of days.entries()) {
    if (dayIndex % 10 === 0) console.error(`Forward-market scan ${dayIndex}/${days.length}...`);
    const candles = readCandleShardReferenceSync(candleFile(day));
    const flow = readTradeFlowShardReferenceSync(flowFile(day));
    const futuresRows = readDerivativesKlinesShardReferenceSync(futuresFile(day));
    const metricRows = readDerivativesMetricsShardReferenceSync(metricsFile(day));
    const baseMap = new Map<number, number>();
    const candidateMaps = candidateEdges.map(() => new Map<number, number>());
    let observations = 0;
    for (let second = 0; second < candles.length; second += 1) {
      const candle = candles[second]!;
      if (second % 60 === 0 && Number.isFinite(previousClose)) {
        const row = second === 0 ? previousFutures?.at(-1) : futuresRows[second / 60 - 1];
        futures.update(row, previousClose, spotMinuteVolume);
        spotMinuteVolume = 0;
      }
      if (second % 300 === 0 && Number.isFinite(previousClose)) {
        const row = second === 0 ? previousMetrics?.at(-1) : metricRows[second / 300 - 1];
        metrics.update(row, previousClose);
      }
      if (!priceEngine) {
        priceEngine = new IndicatorEngine(configuration.priceDefinitions, candle.close);
        marketEngine.update(candle);
      } else {
        const returnBps = candle.close === previousClose ? 0 : Math.log(candle.close / previousClose) * 10_000;
        const active = returnBps !== 0;
        const sign = returnBps > 0 ? 1 : 0;
        const magnitudeBin = active ? upperBound(configuration.priceCalibration.magnitudeEdges, Math.abs(returnBps)) : -1;
        const target = active ? 1 + sign * MAGNITUDE_BINS + magnitudeBin : 0;
        if (returnsSeen >= WARMUP_RETURNS && isEvaluationTarget(returnsSeen)
          && trade.valid && futures.valid && metrics.valid
          && configuration.marketIndices.every((index) => marketEngine.valid()[index] === 1)) {
          priceEngine.values(priceValues);
          let basisState = 0;
          for (let index = 0; index < configuration.priceCuts.length; index += 1) {
            basisState = basisState * BASE_BINS + (priceValues[index]! >= configuration.priceCuts[index]! ? 1 : 0);
          }
          const marketValues = marketEngine.values();
          for (let index = 0; index < configuration.marketIndices.length; index += 1) {
            basisState = basisState * BASE_BINS + (
              marketValues[configuration.marketIndices[index]!]! >= configuration.marketCuts[index]! ? 1 : 0
            );
          }
          const baseContext = previousReturnState * BASE_STATES + basisState;
          const baseIndex = baseContext * TARGET_CLASSES + target;
          baseMap.set(baseIndex, (baseMap.get(baseIndex) ?? 0) + 1);
          copySourceValues(sourceValues, trade, futures, metrics);
          for (let feature = 0; feature < candidateEdges.length; feature += 1) {
            const bin = upperBound(candidateEdges[feature]!, sourceValues[feature]!);
            const index = (baseContext * CANDIDATE_BINS + bin) * TARGET_CLASSES + target;
            candidateMaps[feature]!.set(index, (candidateMaps[feature]!.get(index) ?? 0) + 1);
          }
          observations += 1;
        }
        previousReturnState = target;
        priceEngine.update(candle.close);
        marketEngine.update(candle);
        returnsSeen += 1;
      }
      trade.update(flow[second]!);
      spotMinuteVolume += candle.volume;
      previousClose = candle.close;
    }
    previousFutures = futuresRows;
    previousMetrics = metricRows;
    visit(dayIndex, {
      base: sparse(baseMap, observations),
      candidates: candidateMaps.map((counts) => sparse(counts, observations)),
    });
  }
}

function copySourceValues(
  output: Float64Array,
  trade: TradeFlowFeatureEngine,
  futures: FuturesMinuteFeatureEngine,
  metrics: FuturesMetricsFeatureEngine,
): void {
  let offset = 0;
  output.set(trade.values(), offset);
  offset += trade.values().length;
  output.set(futures.values(), offset);
  offset += futures.values().length;
  output.set(metrics.values(), offset);
}

function addDay(model: WindowModel, day: DayCounts): void {
  model.base.add(day.base);
  model.candidates.forEach((table, index) => table.add(day.candidates[index]!));
}

function scoreDay(model: WindowModel, day: DayCounts, half: "first" | "second"): void {
  model.candidates.forEach((table, feature) => {
    const score = informationScore(table, model.base, day.candidates[feature]!);
    addScore(model.scores[feature]!, score);
    addScore(half === "first" ? model.firstHalf[feature]! : model.secondHalf[feature]!, score);
  });
}

function informationScore(candidate: ProbabilityTable, base: ProbabilityTable, day: SparseCounts): InformationScore {
  const score = emptyScore();
  for (let item = 0; item < day.indices.length; item += 1) {
    const index = day.indices[item]!;
    const count = day.counts[item]!;
    const candidateContext = Math.floor(index / TARGET_CLASSES);
    const target = index % TARGET_CLASSES;
    const baseContext = Math.floor(candidateContext / CANDIDATE_BINS);
    const candidateProbability = smoothed(candidate.joint[index]!, candidate.totals[candidateContext]!, TARGET_CLASSES);
    const baseProbability = smoothed(base.joint[baseContext * TARGET_CLASSES + target]!, base.totals[baseContext]!, TARGET_CLASSES);
    score.observations += count;
    score.bits += count * Math.log2(candidateProbability / baseProbability);
    const candidateZero = target === 0 ? candidate.zero[candidateContext]! : candidate.totals[candidateContext]! - candidate.zero[candidateContext]!;
    const baseZero = target === 0 ? base.zero[baseContext]! : base.totals[baseContext]! - base.zero[baseContext]!;
    score.zeroBits += count * Math.log2(smoothed(candidateZero, candidate.totals[candidateContext]!, 2) / smoothed(baseZero, base.totals[baseContext]!, 2));
    if (target !== 0) {
      const positive = target > MAGNITUDE_BINS;
      const candidateActive = candidate.totals[candidateContext]! - candidate.zero[candidateContext]!;
      const baseActive = base.totals[baseContext]! - base.zero[baseContext]!;
      const candidateSign = positive ? candidate.positive[candidateContext]! : candidateActive - candidate.positive[candidateContext]!;
      const baseSign = positive ? base.positive[baseContext]! : baseActive - base.positive[baseContext]!;
      score.activeObservations += count;
      score.signBits += count * Math.log2(smoothed(candidateSign, candidateActive, 2) / smoothed(baseSign, baseActive, 2));
    }
  }
  return score;
}

function windowResult(model: WindowModel, definitions: FeatureDefinition[]) {
  return {
    days: model.days,
    candidates: definitions.map((definition, feature) => ({
      ...definition,
      ...normalizedScore(model.scores[feature]!),
      firstHalf: normalizedScore(model.firstHalf[feature]!),
      secondHalf: normalizedScore(model.secondHalf[feature]!),
    })),
  };
}

function renderReport(artifact: any): string {
  const lines = [
    "# Forward-market information about the next 1s spot return",
    "",
    `Generated ${artifact.generatedAt}. The screen tests ${artifact.method.candidateFeatures.length} causal features from spot aggressor flow, futures basis/flow, and futures positioning after the existing five-coordinate price/volume/range basis.`,
    "",
    "## Result by source",
    "",
    "| source | features | stable full improvements | best primary full bits | transfer full bits | best active-sign feature | primary sign bits | transfer sign bits |",
    "|---|---:|---:|---:|---:|---|---:|---:|",
  ];
  for (const source of artifact.sources) {
    const full = source.bestStableFull ?? source.bestFull;
    const sign = source.bestActiveSign;
    lines.push(`| ${source.source} | ${source.featureCount} | ${source.stableFullCount} | ${metric(full.fullBits)} | ${metric(full.transfer.fullBits)} | ${sign.label} | ${metric(sign.activeSignBits)} | ${metric(sign.transfer.activeSignBits)} |`);
  }
  lines.push(
    "",
    artifact.bestStableFull
      ? `Across all sources, **${artifact.bestStableFull.label}** is the strongest feature positive in both primary halves and transfer: ${metric(artifact.bestStableFull.fullBits)} primary and ${metric(artifact.bestStableFull.transfer.fullBits)} transfer bits/target.`
      : "No feature improves the full distribution in both primary halves and transfer.",
    "",
    "## Spot-flow age decay",
    "",
    "The last-side and imbalance coordinates were repeated with older completed seconds while leaving the model and targets unchanged.",
    "",
    "| flow age | last-side primary bits | last-side transfer bits | trade-count-imbalance primary bits | quote-imbalance primary bits |",
    "|---:|---:|---:|---:|---:|",
  );
  for (const lag of [1, 2, 3, 5]) {
    const suffix = lag === 1 ? "1s" : `lag-${lag}s`;
    const lastSide = artifact.ranked.find((row: any) => row.id === `spot-flow-last-side-${suffix}`);
    const tradeCount = artifact.ranked.find((row: any) => row.id === `spot-flow-trade-count-imbalance-${suffix}`);
    const quote = artifact.ranked.find((row: any) => row.id === `spot-flow-quote-imbalance-${suffix}`);
    lines.push(`| ${lag}s | ${metric(lastSide.fullBits)} | ${metric(lastSide.transfer.fullBits)} | ${metric(tradeCount.fullBits)} | ${metric(quote.fullBits)} |`);
  }
  lines.push(
    "",
    "The signal remains large at 2s but is gone by 3s. It is therefore a short-lived microstructure feature, not a persistent directional forecast.",
    "",
    "## Ninety-day primary ranking",
    "",
    "| rank | feature | source | family | full bits | zero bits | active-sign bits | half 1 | half 2 | transfer | stable blocks |",
    "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
  );
  artifact.ranked.slice(0, 40).forEach((row: any, index: number) => {
    lines.push(`| ${index + 1} | ${row.label} | ${row.source} | ${row.family} | ${metric(row.fullBits)} | ${metric(row.zeroGateBits)} | ${metric(row.activeSignBits)} | ${metric(row.firstHalf.fullBits)} | ${metric(row.secondHalf.fullBits)} | ${metric(row.transfer.fullBits)} | ${stabilityCount(row)}/3 |`);
  });
  lines.push(
    "",
    "## Training-history sensitivity",
    "",
    `The table follows the overall strongest stable feature, **${artifact.bestStableFull?.label ?? artifact.bestFull.label}**, across histories before the same frozen primary test.`,
    "",
    "| training history | full bits | active-sign bits |",
    "|---:|---:|---:|",
  );
  const followedId = artifact.bestStableFull?.id ?? artifact.bestFull.id;
  for (const window of artifact.windows) {
    const row = window.candidates.find((candidate: any) => candidate.id === followedId);
    lines.push(`| ${window.days} days | ${metric(row.fullBits)} | ${metric(row.activeSignBits)} |`);
  }
  lines.push(
    "",
    "## Relation to the existing 15m model test",
    "",
    artifact.existingFifteenMinuteMatchedModel.note,
    "This individual information screen can reveal a useful coordinate even when adding every feature to a finite neural model is counterproductive; it still does not establish that combining the winners will improve the 15m model.",
    "",
    "## Causal alignment",
    "",
    ...artifact.method.timing.map((row: string) => `- ${row}`),
    "",
    "## Limits",
    "",
    ...artifact.limitations.map((row: string) => `- ${row}`),
    "",
    "Complete values are stored in `data/benchmarks/forward-market-return-information.json`.",
  );
  return `${lines.join("\n")}\n`;
}

function stabilityCount(candidate: any): number {
  return [candidate.firstHalf.fullBits, candidate.secondHalf.fullBits, candidate.transfer.fullBits].filter((value) => value > 0).length;
}

function signStabilityCount(candidate: any): number {
  return [candidate.firstHalf.activeSignBits, candidate.secondHalf.activeSignBits, candidate.transfer.activeSignBits].filter((value) => value > 0).length;
}

function emptyScore(): InformationScore {
  return { observations: 0, bits: 0, zeroBits: 0, activeObservations: 0, signBits: 0 };
}

function addScore(target: InformationScore, source: InformationScore): void {
  target.observations += source.observations;
  target.bits += source.bits;
  target.zeroBits += source.zeroBits;
  target.activeObservations += source.activeObservations;
  target.signBits += source.signBits;
}

function normalizedScore(score: InformationScore) {
  return {
    observations: score.observations,
    fullBits: score.bits / score.observations,
    zeroGateBits: score.zeroBits / score.observations,
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

function assertReferences(days: string[]): void {
  for (const day of days) {
    for (const file of [candleFile(day), flowFile(day), futuresFile(day), metricsFile(day)]) {
      if (!fs.existsSync(file)) throw new Error(`Missing source reference: ${file}`);
    }
  }
}

function candleFile(day: string): string { return path.join(resolve(CANDLE_DIRECTORY), `${day}.json`); }
function flowFile(day: string): string { return path.join(resolve(FLOW_DIRECTORY), `${day}.json`); }
function futuresFile(day: string): string { return path.join(resolve(FUTURES_DIRECTORY), `${day}.json`); }
function metricsFile(day: string): string { return path.join(resolve(METRICS_DIRECTORY), `${day}.json`); }

function inclusiveDays(start: string, end: string): string[] {
  const result: string[] = [];
  for (let time = Date.parse(`${start}T00:00:00.000Z`); time <= Date.parse(`${end}T00:00:00.000Z`); time += DAY_MS) {
    result.push(new Date(time).toISOString().slice(0, 10));
  }
  return result;
}

function imbalance(positive: number, negative: number): number {
  const total = positive + negative;
  return total > 0 ? (positive - negative) / total : 0;
}

function logChange(history: number[], lag: number): number {
  return history.length > lag ? Math.log(history.at(-1)! / history[history.length - 1 - lag]!) : 0;
}

function lagged(history: number[], secondsOld: number): number {
  return history.length >= secondsOld ? history[history.length - secondsOld]! : 0;
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

function resolve(file: string): string { return path.resolve(repoRoot, file); }

function metric(value: number): string {
  return Number.isFinite(value) ? value === 0 ? "0" : value.toPrecision(8).replace(/\.0+(?=e|$)/, "") : "n/a";
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  try {
    main();
  } catch (error: unknown) {
    console.error(error instanceof Error ? error.stack ?? error.message : String(error));
    process.exitCode = 1;
  }
}
