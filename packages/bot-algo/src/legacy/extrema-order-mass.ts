import type {
  BacktestExtremaOrderMassSideSummary,
  BacktestExtremaOrderMassSummary,
  Candle,
  TradeFill,
} from "./types.js";

const DEFAULT_SMA_WINDOW_MS = 30 * 60 * 1000;
const DEFAULT_THRESHOLD_TIME_MS = 5 * 60 * 1000;
const DEFAULT_THRESHOLD_PRICE_DISTANCE_PCT = 0.02;

export type ExtremaKind = "peak" | "valley";

interface RollingClose {
  time: number;
  price: number;
}

interface CenteredSmaPoint {
  time: number;
  price: number;
}

export interface ExtremaPoint extends CenteredSmaPoint {
  kind: ExtremaKind;
}

interface MatchedMassDistance {
  quote: number;
  quantity: number;
  timeDistanceMs: number;
  priceDistancePct: number;
  jointScale: number;
}

export interface ExtremaOrderMassCollector {
  smaWindowMs: number;
  smaShiftMs: number;
  thresholdTimeMs: number;
  thresholdPriceDistancePct: number;
  sampleIntervalMs?: number;
  sampleCount?: number;
  rollingCloses: RollingClose[];
  rollingSum: number;
  previousSmaPoint?: CenteredSmaPoint;
  candidateSmaPoint?: CenteredSmaPoint;
  peaks: ExtremaPoint[];
  valleys: ExtremaPoint[];
}

export function createExtremaOrderMassCollector(options: {
  smaWindowMs?: number;
  thresholdTimeMs?: number;
  thresholdPriceDistancePct?: number;
} = {}): ExtremaOrderMassCollector {
  const smaWindowMs = Math.max(1, options.smaWindowMs ?? DEFAULT_SMA_WINDOW_MS);
  return {
    smaWindowMs,
    smaShiftMs: smaWindowMs / 2,
    thresholdTimeMs: Math.max(0, options.thresholdTimeMs ?? DEFAULT_THRESHOLD_TIME_MS),
    thresholdPriceDistancePct: Math.max(
      0,
      options.thresholdPriceDistancePct ?? DEFAULT_THRESHOLD_PRICE_DISTANCE_PCT,
    ),
    rollingCloses: [],
    rollingSum: 0,
    peaks: [],
    valleys: [],
  };
}

export function observeExtremaOrderMassCandle(
  collector: ExtremaOrderMassCollector,
  candle: Readonly<Candle>,
): void {
  if (
    !Number.isFinite(candle.closeTime) ||
    !Number.isFinite(candle.close) ||
    candle.close <= 0
  ) {
    return;
  }

  const sampleCount = extremaSmaSampleCount(collector, candle);
  collector.rollingCloses.push({
    time: candle.closeTime,
    price: candle.close,
  });
  collector.rollingSum += candle.close;

  while (collector.rollingCloses.length > sampleCount) {
    const removed = collector.rollingCloses.shift();
    if (removed) {
      collector.rollingSum -= removed.price;
    }
  }

  if (collector.rollingCloses.length < sampleCount) {
    return;
  }

  observeCenteredSmaPoint(collector, {
    time: candle.closeTime - collector.smaShiftMs,
    price: collector.rollingSum / collector.rollingCloses.length,
  });
}

export function summarizeExtremaOrderMass(
  collector: Readonly<ExtremaOrderMassCollector>,
  fills: readonly TradeFill[],
): BacktestExtremaOrderMassSummary {
  return {
    smaWindowMs: collector.smaWindowMs,
    smaShiftMs: collector.smaShiftMs,
    thresholdTimeMs: collector.thresholdTimeMs,
    thresholdPriceDistancePct: collector.thresholdPriceDistancePct,
    peakCount: collector.peaks.length,
    valleyCount: collector.valleys.length,
    buy: summarizeSideMass("valley", collector.valleys, fills, collector),
    sell: summarizeSideMass("peak", collector.peaks, fills, collector),
  };
}

export function aggregateExtremaOrderMassSummaries(
  summaries: Array<BacktestExtremaOrderMassSummary | undefined>,
): BacktestExtremaOrderMassSummary | undefined {
  const available = summaries.filter(
    (summary): summary is BacktestExtremaOrderMassSummary => summary !== undefined,
  );
  const first = available[0];
  if (!first) {
    return undefined;
  }

  return {
    smaWindowMs: first.smaWindowMs,
    smaShiftMs: first.smaShiftMs,
    thresholdTimeMs: first.thresholdTimeMs,
    thresholdPriceDistancePct: first.thresholdPriceDistancePct,
    peakCount: sum(available.map((summary) => summary.peakCount)),
    valleyCount: sum(available.map((summary) => summary.valleyCount)),
    buy: aggregateSideSummaries(available.map((summary) => summary.buy), first.buy.target),
    sell: aggregateSideSummaries(available.map((summary) => summary.sell), first.sell.target),
  };
}

function extremaSmaSampleCount(
  collector: ExtremaOrderMassCollector,
  candle: Readonly<Candle>,
): number {
  const inferredIntervalMs = inferCandleIntervalMs(candle);
  if (collector.sampleIntervalMs === undefined && inferredIntervalMs > 0) {
    collector.sampleIntervalMs = inferredIntervalMs;
  }

  const sampleIntervalMs = collector.sampleIntervalMs ?? 60 * 1000;
  collector.sampleCount = Math.max(
    1,
    Math.round(collector.smaWindowMs / sampleIntervalMs),
  );
  return collector.sampleCount;
}

function inferCandleIntervalMs(candle: Readonly<Candle>): number {
  const durationMs = candle.closeTime - candle.openTime + 1;
  if (Number.isFinite(durationMs) && durationMs > 0) {
    return durationMs;
  }
  return 60 * 1000;
}

function observeCenteredSmaPoint(
  collector: ExtremaOrderMassCollector,
  point: CenteredSmaPoint,
): void {
  const left = collector.previousSmaPoint;
  const middle = collector.candidateSmaPoint;
  if (left && middle) {
    const leftDelta = middle.price - left.price;
    const rightDelta = point.price - middle.price;
    const epsilon = Math.max(Math.abs(middle.price) * 1e-12, 1e-9);
    if (leftDelta > epsilon && rightDelta < -epsilon) {
      collector.peaks.push({ ...middle, kind: "peak" });
    } else if (leftDelta < -epsilon && rightDelta > epsilon) {
      collector.valleys.push({ ...middle, kind: "valley" });
    }
  }

  collector.previousSmaPoint = middle;
  collector.candidateSmaPoint = point;
}

function summarizeSideMass(
  target: ExtremaKind,
  extrema: readonly ExtremaPoint[],
  fills: readonly TradeFill[],
  collector: Readonly<ExtremaOrderMassCollector>,
): BacktestExtremaOrderMassSideSummary {
  const side = target === "peak" ? "sell" : "buy";
  const sideFills = fills.filter((fill) => fill.side === side);
  let totalQuote = 0;
  let totalQuantity = 0;
  let matchedQuote = 0;
  let matchedQuantity = 0;
  let thresholdQuote = 0;
  let thresholdQuantity = 0;
  let matchedFillCount = 0;
  let thresholdFillCount = 0;
  const distances: MatchedMassDistance[] = [];

  for (const fill of sideFills) {
    const quote = fillQuoteMass(fill);
    const quantity = fillQuantityMass(fill);
    totalQuote += quote;
    totalQuantity += quantity;
    if (quote <= 0 || fill.price <= 0 || !Number.isFinite(fill.filledAt)) {
      continue;
    }

    const matched = nearestExtremum(extrema, fill.filledAt);
    if (!matched || matched.price <= 0) {
      continue;
    }

    const timeDistanceMs = Math.abs(fill.filledAt - matched.time);
    const priceDistancePct = Math.abs((fill.price - matched.price) / matched.price) * 100;
    const jointScale = Math.max(
      collector.thresholdTimeMs > 0 ? timeDistanceMs / collector.thresholdTimeMs : 0,
      collector.thresholdPriceDistancePct > 0
        ? priceDistancePct / collector.thresholdPriceDistancePct
        : 0,
    );
    matchedFillCount += 1;
    matchedQuote += quote;
    matchedQuantity += quantity;
    distances.push({
      quote,
      quantity,
      timeDistanceMs,
      priceDistancePct,
      jointScale,
    });

    if (
      timeDistanceMs <= collector.thresholdTimeMs &&
      priceDistancePct <= collector.thresholdPriceDistancePct
    ) {
      thresholdFillCount += 1;
      thresholdQuote += quote;
      thresholdQuantity += quantity;
    }
  }

  const p99JointScale = weightedQuantile(distances, 0.99, (item) => item.jointScale);

  return {
    target,
    targetExtremaCount: extrema.length,
    fillCount: sideFills.length,
    matchedFillCount,
    totalQuote,
    totalQuantity,
    matchedQuote,
    matchedQuantity,
    matchedMassPct: percent(matchedQuote, totalQuote),
    thresholdFillCount,
    thresholdQuote,
    thresholdQuantity,
    thresholdMassPct: percent(thresholdQuote, totalQuote),
    weightedAvgTimeDistanceMs: weightedAverage(
      distances,
      (item) => item.timeDistanceMs,
    ),
    weightedAvgPriceDistancePct: weightedAverage(
      distances,
      (item) => item.priceDistancePct,
    ),
    massP50TimeDistanceMs: weightedQuantile(
      distances,
      0.5,
      (item) => item.timeDistanceMs,
    ),
    massP90TimeDistanceMs: weightedQuantile(
      distances,
      0.9,
      (item) => item.timeDistanceMs,
    ),
    massP99TimeDistanceMs: weightedQuantile(
      distances,
      0.99,
      (item) => item.timeDistanceMs,
    ),
    massP50PriceDistancePct: weightedQuantile(
      distances,
      0.5,
      (item) => item.priceDistancePct,
    ),
    massP90PriceDistancePct: weightedQuantile(
      distances,
      0.9,
      (item) => item.priceDistancePct,
    ),
    massP99PriceDistancePct: weightedQuantile(
      distances,
      0.99,
      (item) => item.priceDistancePct,
    ),
    massP99JointScale: p99JointScale,
    massP99JointTimeDistanceMs:
      p99JointScale === undefined ? undefined : p99JointScale * collector.thresholdTimeMs,
    massP99JointPriceDistancePct:
      p99JointScale === undefined
        ? undefined
        : p99JointScale * collector.thresholdPriceDistancePct,
  };
}

function aggregateSideSummaries(
  summaries: BacktestExtremaOrderMassSideSummary[],
  target: ExtremaKind,
): BacktestExtremaOrderMassSideSummary {
  const totalQuote = sum(summaries.map((summary) => summary.totalQuote));
  const matchedQuote = sum(summaries.map((summary) => summary.matchedQuote));
  const thresholdQuote = sum(summaries.map((summary) => summary.thresholdQuote));

  return {
    target,
    targetExtremaCount: sum(summaries.map((summary) => summary.targetExtremaCount)),
    fillCount: sum(summaries.map((summary) => summary.fillCount)),
    matchedFillCount: sum(summaries.map((summary) => summary.matchedFillCount)),
    totalQuote,
    totalQuantity: sum(summaries.map((summary) => summary.totalQuantity)),
    matchedQuote,
    matchedQuantity: sum(summaries.map((summary) => summary.matchedQuantity)),
    matchedMassPct: percent(matchedQuote, totalQuote),
    thresholdFillCount: sum(summaries.map((summary) => summary.thresholdFillCount)),
    thresholdQuote,
    thresholdQuantity: sum(summaries.map((summary) => summary.thresholdQuantity)),
    thresholdMassPct: percent(thresholdQuote, totalQuote),
    weightedAvgTimeDistanceMs: weightedAverageSummaries(
      summaries,
      (summary) => summary.weightedAvgTimeDistanceMs,
    ),
    weightedAvgPriceDistancePct: weightedAverageSummaries(
      summaries,
      (summary) => summary.weightedAvgPriceDistancePct,
    ),
    massP50TimeDistanceMs: weightedAverageSummaries(
      summaries,
      (summary) => summary.massP50TimeDistanceMs,
    ),
    massP90TimeDistanceMs: weightedAverageSummaries(
      summaries,
      (summary) => summary.massP90TimeDistanceMs,
    ),
    massP99TimeDistanceMs: weightedAverageSummaries(
      summaries,
      (summary) => summary.massP99TimeDistanceMs,
    ),
    massP50PriceDistancePct: weightedAverageSummaries(
      summaries,
      (summary) => summary.massP50PriceDistancePct,
    ),
    massP90PriceDistancePct: weightedAverageSummaries(
      summaries,
      (summary) => summary.massP90PriceDistancePct,
    ),
    massP99PriceDistancePct: weightedAverageSummaries(
      summaries,
      (summary) => summary.massP99PriceDistancePct,
    ),
    massP99JointScale: weightedAverageSummaries(
      summaries,
      (summary) => summary.massP99JointScale,
    ),
    massP99JointTimeDistanceMs: weightedAverageSummaries(
      summaries,
      (summary) => summary.massP99JointTimeDistanceMs,
    ),
    massP99JointPriceDistancePct: weightedAverageSummaries(
      summaries,
      (summary) => summary.massP99JointPriceDistancePct,
    ),
  };
}

function nearestExtremum(
  extrema: readonly ExtremaPoint[],
  time: number,
): ExtremaPoint | undefined {
  if (extrema.length === 0) {
    return undefined;
  }

  let low = 0;
  let high = extrema.length;
  while (low < high) {
    const mid = Math.floor((low + high) / 2);
    if (extrema[mid].time < time) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }

  const next = extrema[low];
  const previous = extrema[low - 1];
  if (!previous) {
    return next;
  }
  if (!next) {
    return previous;
  }
  return Math.abs(previous.time - time) <= Math.abs(next.time - time) ? previous : next;
}

function fillQuoteMass(fill: Readonly<TradeFill>): number {
  if (Number.isFinite(fill.quoteQuantity) && fill.quoteQuantity > 0) {
    return fill.quoteQuantity;
  }
  if (
    Number.isFinite(fill.price) &&
    fill.price > 0 &&
    Number.isFinite(fill.quantity) &&
    fill.quantity > 0
  ) {
    return fill.price * fill.quantity;
  }
  return 0;
}

function fillQuantityMass(fill: Readonly<TradeFill>): number {
  return Number.isFinite(fill.quantity) && fill.quantity > 0 ? fill.quantity : 0;
}

function weightedAverage(
  distances: readonly MatchedMassDistance[],
  value: (item: MatchedMassDistance) => number,
): number | undefined {
  let weightedTotal = 0;
  let totalWeight = 0;
  for (const item of distances) {
    const current = value(item);
    if (!Number.isFinite(current) || item.quote <= 0) {
      continue;
    }
    weightedTotal += current * item.quote;
    totalWeight += item.quote;
  }
  return totalWeight > 0 ? weightedTotal / totalWeight : undefined;
}

function weightedQuantile(
  distances: readonly MatchedMassDistance[],
  ratio: number,
  value: (item: MatchedMassDistance) => number,
): number | undefined {
  const sorted = distances
    .map((item) => ({
      value: value(item),
      weight: item.quote,
    }))
    .filter((item) => Number.isFinite(item.value) && item.weight > 0)
    .sort((left, right) => left.value - right.value);

  const totalWeight = sum(sorted.map((item) => item.weight));
  if (totalWeight <= 0) {
    return undefined;
  }

  const targetWeight = totalWeight * ratio;
  let cumulative = 0;
  for (const item of sorted) {
    cumulative += item.weight;
    if (cumulative >= targetWeight) {
      return item.value;
    }
  }
  return sorted.at(-1)?.value;
}

function weightedAverageSummaries(
  summaries: readonly BacktestExtremaOrderMassSideSummary[],
  value: (summary: BacktestExtremaOrderMassSideSummary) => number | undefined,
): number | undefined {
  let weightedTotal = 0;
  let totalWeight = 0;
  for (const summary of summaries) {
    const current = value(summary);
    if (!Number.isFinite(current) || summary.matchedQuote <= 0) {
      continue;
    }
    weightedTotal += (current as number) * summary.matchedQuote;
    totalWeight += summary.matchedQuote;
  }
  return totalWeight > 0 ? weightedTotal / totalWeight : undefined;
}

function percent(numerator: number, denominator: number): number {
  return denominator > 0 ? (numerator / denominator) * 100 : 0;
}

function sum(values: readonly number[]): number {
  return values.reduce((total, value) => total + (Number.isFinite(value) ? value : 0), 0);
}
