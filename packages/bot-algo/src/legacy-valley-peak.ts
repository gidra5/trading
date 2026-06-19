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
  quoteFree: number;
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
  buySigma: 0.1,
  sellSigma: 0.1,
  minTradeQuote: 25,
  maxTradeQuote: 750,
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

  return config;
}

export function createLegacyValleyPeakMemory(
  config: LegacyValleyPeakConfig,
): LegacyValleyPeakMemory {
  return {
    buyAverages: config.averagingRangesSec.map(createRollingAverageMemory),
    sellAverages: config.averagingRangesSec.map(createRollingAverageMemory),
  };
}

export function normalizeLegacyValleyPeakMemory(
  memory: LegacyValleyPeakMemory | undefined,
  config: LegacyValleyPeakConfig,
): LegacyValleyPeakMemory {
  if (!memory) {
    return createLegacyValleyPeakMemory(config);
  }

  return {
    startedAt: memory.startedAt,
    buyAverages: normalizeAverageList(memory.buyAverages, config),
    sellAverages: normalizeAverageList(memory.sellAverages, config),
  };
}

export function evaluateLegacyValleyPeak(
  memory: LegacyValleyPeakMemory,
  config: LegacyValleyPeakConfig,
  input: LegacyValleyPeakInput,
): LegacyValleyPeakDecision {
  const tsSec = input.eventTime / 1000;
  memory.startedAt ??= input.eventTime;

  for (let index = 0; index < config.averagingRangesSec.length; index += 1) {
    updateRollingAverage(memory.buyAverages[index], {
      value: input.price,
      tsSec,
      rangeSec: config.averagingRangesSec[index],
      derivativeRatio: config.rateRatios[index],
      thresholdLow: config.rateThresholdsLow[index],
      thresholdHigh: config.rateThresholdsHigh[index],
    });
    updateRollingAverage(memory.sellAverages[index], {
      value: input.price,
      tsSec,
      rangeSec: config.averagingRangesSec[index],
      derivativeRatio: config.rateRatios[index],
      thresholdLow: config.rateThresholdsLow[index],
      thresholdHigh: config.rateThresholdsHigh[index],
    });
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
  const desired = input.quoteFree * config.buySpendRate * gaussian(derivative, 0, config.buySigma);
  const remainingPositionQuote = Math.max(0, input.maxPositionQuote - input.positionQuote);

  return clamp(
    desired,
    config.minTradeQuote,
    Math.min(config.maxTradeQuote, input.quoteFree * 0.98, remainingPositionQuote, rate * 10_000),
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
  options: {
    value: number;
    tsSec: number;
    rangeSec: number;
    derivativeRatio: number;
    thresholdLow: number;
    thresholdHigh: number;
  },
): void {
  let startIndex = memory.startIndex ?? 0;
  startIndex = clampInt(startIndex, 0, memory.timestamps.length);

  while (
    startIndex < memory.timestamps.length &&
    memory.timestamps[startIndex] + options.rangeSec < options.tsSec
  ) {
    memory.sum -= memory.entries[startIndex] ?? 0;
    startIndex += 1;
  }
  memory.startIndex = startIndex;

  memory.timestamps.push(options.tsSec);
  memory.entries.push(options.value);
  memory.sum += options.value;
  const activeCount = memory.timestamps.length - startIndex;
  const avg = memory.sum / activeCount;
  memory.averages.push(avg);

  if (activeCount < 2) {
    recordPoint(memory, { avg, rate: 0, rateClamped: 0 });
    compactRollingAverage(memory);
    return;
  }

  const lastIndex = memory.timestamps.length - 1;
  const windowSize = memory.timestamps[lastIndex] - memory.timestamps[startIndex];
  const pointTs = options.tsSec - windowSize * options.derivativeRatio;
  const sample = sampleData(
    memory.averages,
    memory.timestamps,
    pointTs,
    memory.previousSampleIndex,
    startIndex,
  );
  memory.previousSampleIndex = sample.index;

  const delta = options.tsSec - pointTs;
  if (delta === 0) {
    recordPoint(memory, { avg, rate: 0, rateClamped: 0 });
    compactRollingAverage(memory);
    return;
  }

  const derivative = (avg - sample.value) / delta;
  let rateClamped = 0;
  if (derivative >= options.thresholdHigh) {
    rateClamped = derivative;
  } else if (derivative <= -options.thresholdLow) {
    rateClamped = derivative;
  }

  recordPoint(memory, { avg, rate: derivative, rateClamped });
  compactRollingAverage(memory);
}

function sampleData(
  data: number[],
  timestamps: number[],
  ts: number,
  previousIndex?: number,
  startIndex = 0,
): { index: number; value: number } {
  const lastIndex = timestamps.length - 1;
  if (lastIndex <= startIndex || ts <= timestamps[startIndex]) {
    return { index: startIndex, value: data[startIndex] };
  }

  let index = previousIndex ?? binarySearchLeft(timestamps, ts, startIndex, timestamps.length) - 1;
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
    return { index, value: prev };
  }

  const nextDelta = timestamps[index + 1] - ts;
  const fraction = nextDelta / delta;
  return {
    index,
    value: interpolate(prev, next, fraction),
  };
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

function recordPoint(memory: RollingAverageMemory, point: RollingAveragePoint): void {
  memory.points.push(point);
  if (memory.points.length > 3) {
    memory.points.splice(0, memory.points.length - 3);
  }
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
