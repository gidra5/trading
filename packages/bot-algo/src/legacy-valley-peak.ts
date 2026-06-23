import type {
  LegacyValleyPeakConfig,
  LegacyValleyPeakMemory,
  RollingAverageMemory,
  RollingAveragePoint,
} from "./types.js";

export interface LegacyValleyPeakInput {
  eventTime: number;
  price: number;
  feeRate: number;
  buyingPowerQuote: number;
  baseFree: number;
  positionQuote: number;
  maxPositionQuote: number;
}

export type LegacyValleyPeakDecision =
  | { signal: "hold" }
  | { signal: "buy"; reason: string; quoteSize: number }
  | { signal: "sell"; reason: string; quantity: number };

export const defaultLegacyValleyPeakConfig: LegacyValleyPeakConfig = {
  averagingRangesSec: [1, 60, 600, 1800, 3600, 3600 * 4, 3600 * 12],
  rateRatios: [0.5, 0.5, 0.1, 0.05, 0.01, 0.01, 0.001],
  rateThresholdsLow: [0.25, 0.25, 0.25, 0.25, 0.15, 0.05, 0.05],
  rateThresholdsHigh: [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
  buyDataIndex: 0,
  sellDataIndex: 1,
  buyConfirmationOffset: 6,
  sellConfirmationOffsets: [2, 1],
  saturationSec: 3600,
  buySpendRate: 1,
  sellAmountRate: 1,
  buySigma: 0.3,
  sellSigma: 0.1,
  minTradeQuote: 25,
  maxTradeQuote: 50_000,
  exitGridEnabled: true,
  exitGridMarketEntry: true,
  exitGridOrderCount: 6,
  exitGridPriceDistribution: "uniform",
  exitGridSizeDistribution: "geometric",
  exitGridSellFraction: 0.35,
  exitGridMinProfitBps: 20,
  exitGridResetBps: 10,
  exitGridPositionMode: "per-lot",
  exitGridResetMode: "filled-grid",
};

const ROLLING_AVERAGE_COMPACT_EXPIRED = 2048;

export function createLegacyValleyPeakConfig(
  overrides: Partial<LegacyValleyPeakConfig> = {},
): LegacyValleyPeakConfig {
  const config: LegacyValleyPeakConfig = {
    ...defaultLegacyValleyPeakConfig,
    ...overrides,
    averagingRangesSec:
      overrides.averagingRangesSec ?? defaultLegacyValleyPeakConfig.averagingRangesSec,
    rateRatios: overrides.rateRatios ?? defaultLegacyValleyPeakConfig.rateRatios,
    rateThresholdsLow:
      overrides.rateThresholdsLow ?? defaultLegacyValleyPeakConfig.rateThresholdsLow,
    rateThresholdsHigh:
      overrides.rateThresholdsHigh ?? defaultLegacyValleyPeakConfig.rateThresholdsHigh,
    sellConfirmationOffsets:
      overrides.sellConfirmationOffsets ??
      defaultLegacyValleyPeakConfig.sellConfirmationOffsets,
  };

  const rangeCount = Math.max(1, config.averagingRangesSec.length);
  config.rateRatios = padNumbers(config.rateRatios, rangeCount, 0.5);
  config.rateThresholdsLow = padNumbers(config.rateThresholdsLow, rangeCount, 0.25);
  config.rateThresholdsHigh = padNumbers(config.rateThresholdsHigh, rangeCount, 0.25);
  config.buyDataIndex = clampInt(config.buyDataIndex, 0, rangeCount - 1);
  config.sellDataIndex = clampInt(config.sellDataIndex, 0, rangeCount - 1);
  config.buyConfirmationOffset = Math.max(0, Math.round(config.buyConfirmationOffset));
  config.sellConfirmationOffsets = config.sellConfirmationOffsets.map((offset) =>
    Math.max(0, Math.round(offset)),
  );
  config.saturationSec = Math.max(0, config.saturationSec);
  config.buySpendRate = Math.max(0, config.buySpendRate);
  config.sellAmountRate = Math.max(0, config.sellAmountRate);
  config.buySigma = Math.max(0.000001, config.buySigma);
  config.sellSigma = Math.max(0.000001, config.sellSigma);
  config.minTradeQuote = Math.max(0, config.minTradeQuote);
  config.maxTradeQuote = Math.max(config.minTradeQuote, config.maxTradeQuote);
  config.exitGridOrderCount = clampInt(config.exitGridOrderCount, 1, 24);
  if (
    config.exitGridPriceDistribution !== "uniform" &&
    config.exitGridPriceDistribution !== "geometric"
  ) {
    config.exitGridPriceDistribution =
      defaultLegacyValleyPeakConfig.exitGridPriceDistribution;
  }
  if (
    config.exitGridSizeDistribution !== "geometric" &&
    config.exitGridSizeDistribution !== "linear" &&
    config.exitGridSizeDistribution !== "constant"
  ) {
    config.exitGridSizeDistribution =
      defaultLegacyValleyPeakConfig.exitGridSizeDistribution;
  }
  config.exitGridSellFraction = clamp(config.exitGridSellFraction, 0.01, 1);
  config.exitGridMinProfitBps = Math.max(0, config.exitGridMinProfitBps);
  config.exitGridResetBps = Math.max(0, config.exitGridResetBps);
  if (
    config.exitGridPositionMode !== "aggregate" &&
    config.exitGridPositionMode !== "per-lot"
  ) {
    config.exitGridPositionMode = defaultLegacyValleyPeakConfig.exitGridPositionMode;
  }
  if (
    config.exitGridResetMode !== "higher-peak" &&
    config.exitGridResetMode !== "filled-grid"
  ) {
    config.exitGridResetMode = defaultLegacyValleyPeakConfig.exitGridResetMode;
  }

  return config;
}

export function createLegacyValleyPeakMemory(
  config: LegacyValleyPeakConfig,
): LegacyValleyPeakMemory {
  const averages = config.averagingRangesSec.map(createRollingAverageMemory);
  return {
    buyAverages: averages,
    sellAverages: averages,
  };
}

export function normalizeLegacyValleyPeakMemory(
  memory: LegacyValleyPeakMemory | undefined,
  config: LegacyValleyPeakConfig,
): LegacyValleyPeakMemory {
  if (!memory) {
    return createLegacyValleyPeakMemory(config);
  }

  const averages = normalizeAverageList(memory.buyAverages ?? memory.sellAverages, config);
  return {
    startedAt: memory.startedAt,
    buyAverages: averages,
    sellAverages: averages,
    exitGrids: memory.exitGrids,
  };
}

export function evaluateLegacyValleyPeak(
  memory: LegacyValleyPeakMemory,
  config: LegacyValleyPeakConfig,
  input: LegacyValleyPeakInput,
): LegacyValleyPeakDecision {
  const tsSec = input.eventTime / 1000;
  memory.startedAt ??= input.eventTime;

  const averages = memory.buyAverages;
  memory.sellAverages = averages;
  for (let index = 0; index < config.averagingRangesSec.length; index += 1) {
    updateRollingAverage(
      averages[index],
      input.price,
      tsSec,
      config.averagingRangesSec[index],
      config.rateRatios[index],
      config.rateThresholdsLow[index],
      config.rateThresholdsHigh[index],
    );
  }

  if (input.eventTime - memory.startedAt < config.saturationSec * 1000) {
    return { signal: "hold" };
  }

  const feeAdjustedBuyRate = input.price / (1 - input.feeRate);
  const feeAdjustedSellRate = input.price * (1 - input.feeRate);

  if (shouldBuy(memory, config)) {
    const quoteSize = buyQuoteSize(memory, config, input, feeAdjustedBuyRate);
    if (quoteSize >= config.minTradeQuote) {
      return {
        signal: "buy",
        reason: "legacy valley detected",
        quoteSize,
      };
    }
  }

  if (shouldSell(memory, config)) {
    const quantity = sellQuantity(memory, config, input, feeAdjustedSellRate);
    if (quantity * feeAdjustedSellRate >= config.minTradeQuote) {
      return {
        signal: "sell",
        reason: "legacy peak detected",
        quantity,
      };
    }
  }

  return { signal: "hold" };
}

function shouldBuy(
  memory: LegacyValleyPeakMemory,
  config: LegacyValleyPeakConfig,
): boolean {
  const primary = memory.buyAverages[config.buyDataIndex];
  if (!isValley(primary)) {
    return false;
  }

  const confirmationIndex = config.buyDataIndex + config.buyConfirmationOffset;
  const confirmation = memory.buyAverages[confirmationIndex];
  const confirmationPoint = latestPoint(confirmation);
  if (confirmationPoint && confirmationPoint.rateClamped >= 0) {
    return false;
  }

  return true;
}

function shouldSell(
  memory: LegacyValleyPeakMemory,
  config: LegacyValleyPeakConfig,
): boolean {
  const primary = memory.sellAverages[config.sellDataIndex];
  if (!isPeak(primary)) {
    return false;
  }

  for (const offset of config.sellConfirmationOffsets) {
    const confirmation = memory.sellAverages[config.sellDataIndex + offset];
    const confirmationPoint = latestPoint(confirmation);
    if (confirmationPoint && confirmationPoint.rateClamped <= 0) {
      return false;
    }
  }

  return true;
}

function buyQuoteSize(
  memory: LegacyValleyPeakMemory,
  config: LegacyValleyPeakConfig,
  input: LegacyValleyPeakInput,
  rate: number,
): number {
  const data = memory.buyAverages[Math.min(3, memory.buyAverages.length - 1)];
  const derivative = latestPoint(data)?.rate ?? 0;
  const buyingPowerQuote = Math.max(0, input.buyingPowerQuote);
  const desired =
    buyingPowerQuote *
    config.buySpendRate *
    gaussian(derivative, 0, config.buySigma);
  const remainingPositionQuote = Math.max(0, input.maxPositionQuote - input.positionQuote);

  return clamp(
    desired,
    config.minTradeQuote,
    Math.min(config.maxTradeQuote, buyingPowerQuote, remainingPositionQuote, rate * 10_000),
  );
}

function sellQuantity(
  memory: LegacyValleyPeakMemory,
  config: LegacyValleyPeakConfig,
  input: LegacyValleyPeakInput,
  rate: number,
): number {
  const data = memory.sellAverages[Math.min(3, memory.sellAverages.length - 1)];
  const derivative = latestPoint(data)?.rate ?? 0;
  const desired =
    input.baseFree * config.sellAmountRate * gaussian(derivative, 0, config.sellSigma);
  const minQuantity = config.minTradeQuote / rate;
  const maxQuantity = Math.min(config.maxTradeQuote / rate, input.baseFree);

  return clamp(desired, minQuantity, maxQuantity);
}

function updateRollingAverage(
  memory: RollingAverageMemory,
  value: number,
  tsSec: number,
  rangeSec: number,
  derivativeRatio: number,
  thresholdLow: number,
  thresholdHigh: number,
): void {
  let startIndex = memory.startIndex ?? 0;
  startIndex = clampInt(startIndex, 0, memory.timestamps.length);

  while (
    startIndex < memory.timestamps.length &&
    memory.timestamps[startIndex] + rangeSec < tsSec
  ) {
    memory.sum -= memory.entries[startIndex] ?? 0;
    startIndex += 1;
  }
  memory.startIndex = startIndex;

  memory.timestamps.push(tsSec);
  memory.entries.push(value);
  memory.sum += value;
  const activeCount = memory.timestamps.length - startIndex;
  const avg = memory.sum / activeCount;
  memory.averages.push(avg);

  if (activeCount < 2) {
    recordPoint(memory, avg, 0, 0);
    compactRollingAverage(memory);
    return;
  }

  const lastIndex = memory.timestamps.length - 1;
  const windowSize = memory.timestamps[lastIndex] - memory.timestamps[startIndex];
  const pointTs = tsSec - windowSize * derivativeRatio;
  const sampleValue = sampleAverage(memory, pointTs, startIndex);

  const delta = tsSec - pointTs;
  if (delta === 0) {
    recordPoint(memory, avg, 0, 0);
    compactRollingAverage(memory);
    return;
  }

  const derivative = (avg - sampleValue) / delta;
  let rateClamped = 0;
  if (derivative >= thresholdHigh) {
    rateClamped = derivative;
  } else if (derivative <= -thresholdLow) {
    rateClamped = derivative;
  }

  recordPoint(memory, avg, derivative, rateClamped);
  compactRollingAverage(memory);
}

function sampleAverage(
  memory: RollingAverageMemory,
  ts: number,
  startIndex = 0,
): number {
  const data = memory.averages;
  const timestamps = memory.timestamps;
  const lastIndex = timestamps.length - 1;
  if (lastIndex <= startIndex || ts <= timestamps[startIndex]) {
    memory.previousSampleIndex = startIndex;
    return data[startIndex];
  }

  let index =
    memory.previousSampleIndex ??
    binarySearchLeft(timestamps, ts, startIndex, timestamps.length) - 1;
  index = clampInt(index, startIndex, Math.max(startIndex, lastIndex - 1));

  while (index > startIndex && timestamps[index] >= ts) {
    index -= 1;
  }
  while (index < lastIndex - 1 && timestamps[index + 1] < ts) {
    index += 1;
  }

  const prev = data[index];
  const next = data[index + 1];
  const delta = timestamps[index + 1] - timestamps[index];
  if (delta === 0) {
    memory.previousSampleIndex = index;
    return prev;
  }

  const nextDelta = timestamps[index + 1] - ts;
  const fraction = nextDelta / delta;
  memory.previousSampleIndex = index;
  return interpolate(prev, next, fraction);
}

function isValley(memory: RollingAverageMemory | undefined): boolean {
  const latest = latestPoint(memory);
  const previous = previousPoint(memory);
  return Boolean(latest && previous && latest.rateClamped >= 0 && previous.rateClamped < 0);
}

function isPeak(memory: RollingAverageMemory | undefined): boolean {
  const latest = latestPoint(memory);
  const previous = previousPoint(memory);
  return Boolean(latest && previous && latest.rateClamped <= 0 && previous.rateClamped > 0);
}

function latestPoint(memory: RollingAverageMemory | undefined): RollingAveragePoint | undefined {
  return memory?.points.at(-1);
}

function previousPoint(memory: RollingAverageMemory | undefined): RollingAveragePoint | undefined {
  return memory?.points.at(-2);
}

function recordPoint(
  memory: RollingAverageMemory,
  avg: number,
  rate: number,
  rateClamped: number,
): void {
  const points = memory.points;
  if (points.length === 0) {
    points.push({ avg, rate, rateClamped });
    return;
  }
  if (points.length === 1) {
    points.push({ avg, rate, rateClamped });
    return;
  }
  if (points.length > 2) {
    points[1] = points[points.length - 1];
    points.length = 2;
  }

  const previous = points[0];
  const latest = points[1];
  previous.avg = latest.avg;
  previous.rate = latest.rate;
  previous.rateClamped = latest.rateClamped;
  latest.avg = avg;
  latest.rate = rate;
  latest.rateClamped = rateClamped;
}

function createRollingAverageMemory(): RollingAverageMemory {
  return {
    entries: [],
    averages: [],
    timestamps: [],
    sum: 0,
    startIndex: 0,
    points: [],
  };
}

function normalizeAverageList(
  list: RollingAverageMemory[] | undefined,
  config: LegacyValleyPeakConfig,
): RollingAverageMemory[] {
  return config.averagingRangesSec.map((_, index) => normalizeAverage(list?.[index]));
}

function normalizeAverage(memory: RollingAverageMemory | undefined): RollingAverageMemory {
  const entries = memory?.entries ?? [];
  const averages = memory?.averages ?? [];
  const timestamps = memory?.timestamps ?? [];
  const maxStartIndex = Math.min(entries.length, averages.length, timestamps.length);
  const startIndex = clampInt(memory?.startIndex ?? 0, 0, maxStartIndex);

  return {
    entries,
    averages,
    timestamps,
    sum: memory?.sum ?? 0,
    startIndex,
    previousSampleIndex: memory?.previousSampleIndex,
    points: memory?.points ?? [],
  };
}

function gaussian(x: number, mu: number, sigma: number): number {
  const w = (x - mu) / sigma;
  return Math.exp(-(w ** 2) / 2);
}

function interpolate(nextValue: number, prevValue: number, fraction: number): number {
  return nextValue * (1 - fraction) + prevValue * fraction;
}

function compactRollingAverage(memory: RollingAverageMemory): void {
  const startIndex = memory.startIndex ?? 0;
  if (
    startIndex < ROLLING_AVERAGE_COMPACT_EXPIRED ||
    startIndex < memory.timestamps.length / 2
  ) {
    return;
  }

  memory.entries.splice(0, startIndex);
  memory.averages.splice(0, startIndex);
  memory.timestamps.splice(0, startIndex);
  if (memory.previousSampleIndex !== undefined) {
    memory.previousSampleIndex = Math.max(0, memory.previousSampleIndex - startIndex);
  }
  memory.startIndex = 0;
}

function binarySearchLeft(
  values: number[],
  target: number,
  low = 0,
  high = values.length,
): number {
  while (low < high) {
    const mid = Math.floor((low + high) / 2);
    if (values[mid] < target) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }

  return low;
}

function clamp(value: number, min: number, max: number): number {
  if (max < min) {
    return 0;
  }

  return Math.min(Math.max(value, min), max);
}

function clampInt(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) {
    return min;
  }

  return Math.max(min, Math.min(max, Math.round(value)));
}

function padNumbers(values: number[], length: number, fallback: number): number[] {
  const next = values.slice(0, length);
  while (next.length < length) {
    next.push(next.at(-1) ?? fallback);
  }

  return next;
}
