import type {
  PutSequentialShardResult,
  SequentialShardReference,
  ShardLayout,
  StorageMetadata,
} from "./types.js";
import {
  readReferencedPayload,
  readReferencedPayloadSync,
  SequentialShardStore,
} from "./store.js";

export type AggressorSide = -1 | 0 | 1;

export interface SequentialTradeFlowSecond {
  openTime: number;
  aggressiveBuyBaseVolume: number;
  aggressiveSellBaseVolume: number;
  aggressiveBuyQuoteVolume: number;
  aggressiveSellQuoteVolume: number;
  aggressiveBuyAggregateQuantitySquared: number;
  aggressiveSellAggregateQuantitySquared: number;
  aggressiveBuyMaxAggregateQuantity: number;
  aggressiveSellMaxAggregateQuantity: number;
  aggressiveBuyBaseVolumeTimeMoment: number;
  aggressiveSellBaseVolumeTimeMoment: number;
  aggressiveBuyAggregateTradeCount: number;
  aggressiveSellAggregateTradeCount: number;
  aggressiveBuyTradeCount: number;
  aggressiveSellTradeCount: number;
  aggressorSideFlipCount: number;
  firstAggressorSide: AggressorSide;
  lastAggressorSide: AggressorSide;
  firstTradeOffsetMicros: number | null;
  lastTradeOffsetMicros: number | null;
}

export interface TradeFlowAggregate {
  /** Unix timestamp with microsecond precision. */
  timeMicros: number;
  price: number;
  quantity: number;
  /** Number of raw trades represented by this aggregate trade. */
  tradeCount: number;
  /** Buyer aggression is +1; seller aggression is -1. */
  aggressorSide: Exclude<AggressorSide, 0>;
}

interface TradeFlowColumn {
  name: keyof Omit<SequentialTradeFlowSecond, "openTime">;
  encoding: "float64-le" | "uint32-le" | "int8";
  offset: number;
  bytes: number;
  nullValue?: number;
}

interface TradeFlowLayout extends ShardLayout {
  encoding: "trade-flow-columnar-v1";
  columns: TradeFlowColumn[];
}

export interface PutTradeFlowShardRequest {
  namespace: string;
  key: string;
  seconds: readonly SequentialTradeFlowSecond[];
  stepMs?: number;
  metadata?: StorageMetadata;
}

const FLOAT_COLUMNS = [
  "aggressiveBuyBaseVolume",
  "aggressiveSellBaseVolume",
  "aggressiveBuyQuoteVolume",
  "aggressiveSellQuoteVolume",
  "aggressiveBuyAggregateQuantitySquared",
  "aggressiveSellAggregateQuantitySquared",
  "aggressiveBuyMaxAggregateQuantity",
  "aggressiveSellMaxAggregateQuantity",
  "aggressiveBuyBaseVolumeTimeMoment",
  "aggressiveSellBaseVolumeTimeMoment",
] as const;

const COUNT_COLUMNS = [
  "aggressiveBuyAggregateTradeCount",
  "aggressiveSellAggregateTradeCount",
  "aggressiveBuyTradeCount",
  "aggressiveSellTradeCount",
  "aggressorSideFlipCount",
] as const;

const SIDE_COLUMNS = ["firstAggressorSide", "lastAggressorSide"] as const;
const OFFSET_COLUMNS = ["firstTradeOffsetMicros", "lastTradeOffsetMicros"] as const;
const NULL_OFFSET = 0xffff_ffff;

export class DailyTradeFlowAccumulator {
  readonly dayStart: number;
  readonly stepMs: number;
  readonly count: number;
  private readonly seconds: SequentialTradeFlowSecond[];
  private previousTimeMicros = -1;

  constructor(dayStart: number, stepMs = 1_000) {
    if (!Number.isSafeInteger(dayStart)
      || !Number.isSafeInteger(stepMs)
      || stepMs < 1
      || 86_400_000 % stepMs !== 0) {
      throw new Error("Trade-flow UTC-day axis is invalid.");
    }
    this.dayStart = dayStart;
    this.stepMs = stepMs;
    this.count = 86_400_000 / stepMs;
    this.seconds = Array.from(
      { length: this.count },
      (_, index) => emptyTradeFlowSecond(dayStart + index * stepMs),
    );
  }

  append(aggregate: TradeFlowAggregate): void {
    if (!Number.isSafeInteger(aggregate.timeMicros)
      || aggregate.timeMicros < this.dayStart * 1_000
      || aggregate.timeMicros >= (this.dayStart + 86_400_000) * 1_000
      || aggregate.timeMicros < this.previousTimeMicros) {
      throw new Error("Aggregate-trade timestamp is outside the UTC day or out of order.");
    }
    if (!Number.isFinite(aggregate.price) || aggregate.price <= 0
      || !Number.isFinite(aggregate.quantity) || aggregate.quantity <= 0
      || !Number.isSafeInteger(aggregate.tradeCount) || aggregate.tradeCount < 1
      || (aggregate.aggressorSide !== -1 && aggregate.aggressorSide !== 1)) {
      throw new Error("Aggregate trade is invalid.");
    }
    const elapsedMicros = aggregate.timeMicros - this.dayStart * 1_000;
    const index = Math.floor(elapsedMicros / (this.stepMs * 1_000));
    const offsetMicros = elapsedMicros - index * this.stepMs * 1_000;
    const second = this.seconds[index]!;
    const buy = aggregate.aggressorSide === 1;
    const prefix = buy ? "aggressiveBuy" : "aggressiveSell";
    const baseVolumeKey = `${prefix}BaseVolume` as const;
    const quoteVolumeKey = `${prefix}QuoteVolume` as const;
    const squaredKey = `${prefix}AggregateQuantitySquared` as const;
    const maximumKey = `${prefix}MaxAggregateQuantity` as const;
    const timeMomentKey = `${prefix}BaseVolumeTimeMoment` as const;
    const aggregateCountKey = `${prefix}AggregateTradeCount` as const;
    const tradeCountKey = `${prefix}TradeCount` as const;
    second[baseVolumeKey] += aggregate.quantity;
    second[quoteVolumeKey] += aggregate.price * aggregate.quantity;
    second[squaredKey] += aggregate.quantity * aggregate.quantity;
    second[maximumKey] = Math.max(second[maximumKey], aggregate.quantity);
    second[timeMomentKey] += aggregate.quantity * offsetMicros / 1_000_000;
    second[aggregateCountKey] += 1;
    second[tradeCountKey] += aggregate.tradeCount;
    if (second.lastAggressorSide !== 0
      && second.lastAggressorSide !== aggregate.aggressorSide) {
      second.aggressorSideFlipCount += 1;
    }
    if (second.firstAggressorSide === 0) {
      second.firstAggressorSide = aggregate.aggressorSide;
      second.firstTradeOffsetMicros = offsetMicros;
    }
    second.lastAggressorSide = aggregate.aggressorSide;
    second.lastTradeOffsetMicros = offsetMicros;
    this.previousTimeMicros = aggregate.timeMicros;
  }

  finish(): readonly SequentialTradeFlowSecond[] {
    return this.seconds;
  }
}

export async function putTradeFlowShard(
  store: SequentialShardStore,
  request: PutTradeFlowShardRequest,
): Promise<PutSequentialShardResult> {
  const encoded = encodeTradeFlow(request.seconds, request.stepMs);
  return store.put({
    namespace: request.namespace,
    key: request.key,
    payload: encoded.payload,
    sequence: encoded.sequence,
    layout: encoded.layout,
    ...(request.metadata ? { metadata: request.metadata } : {}),
    compressionLevel: 9,
  });
}

export function encodeTradeFlow(
  seconds: readonly SequentialTradeFlowSecond[],
  requestedStepMs?: number,
): {
  payload: Buffer;
  sequence: SequentialShardReference["sequence"];
  layout: TradeFlowLayout;
} {
  if (seconds.length === 0) throw new Error("Cannot encode an empty trade-flow shard.");
  const first = seconds[0]!;
  const step = requestedStepMs
    ?? (seconds.length > 1 ? seconds[1]!.openTime - first.openTime : undefined);
  if (!Number.isSafeInteger(step) || step! < 1) {
    throw new Error("Trade-flow shard interval is missing or invalid.");
  }
  seconds.forEach((second, index) => validateSecond(
    second,
    first.openTime + index * step!,
    step!,
    index,
  ));

  const chunks: Buffer[] = [];
  const columns: TradeFlowColumn[] = [];
  const add = (
    name: TradeFlowColumn["name"],
    encoding: TradeFlowColumn["encoding"],
    bytes: Buffer,
    nullValue?: number,
  ): void => {
    const offset = chunks.reduce((total, chunk) => total + chunk.byteLength, 0);
    chunks.push(bytes);
    columns.push({
      name,
      encoding,
      offset,
      bytes: bytes.byteLength,
      ...(nullValue === undefined ? {} : { nullValue }),
    });
  };
  for (const name of FLOAT_COLUMNS) {
    const bytes = Buffer.allocUnsafe(seconds.length * 8);
    seconds.forEach((second, index) => bytes.writeDoubleLE(second[name], index * 8));
    add(name, "float64-le", bytes);
  }
  for (const name of COUNT_COLUMNS) {
    const bytes = Buffer.allocUnsafe(seconds.length * 4);
    seconds.forEach((second, index) => bytes.writeUInt32LE(second[name], index * 4));
    add(name, "uint32-le", bytes);
  }
  for (const name of SIDE_COLUMNS) {
    const bytes = Buffer.allocUnsafe(seconds.length);
    seconds.forEach((second, index) => bytes.writeInt8(second[name], index));
    add(name, "int8", bytes);
  }
  for (const name of OFFSET_COLUMNS) {
    const bytes = Buffer.allocUnsafe(seconds.length * 4);
    seconds.forEach((second, index) => bytes.writeUInt32LE(
      second[name] ?? NULL_OFFSET,
      index * 4,
    ));
    add(name, "uint32-le", bytes, NULL_OFFSET);
  }
  return {
    payload: Buffer.concat(chunks),
    sequence: {
      start: first.openTime,
      step: step!,
      count: seconds.length,
      unit: "unix-ms",
    },
    layout: { encoding: "trade-flow-columnar-v1", columns },
  };
}

export function decodeTradeFlow(
  reference: SequentialShardReference,
  payload: Uint8Array,
): SequentialTradeFlowSecond[] {
  if (reference.layout.encoding !== "trade-flow-columnar-v1") {
    throw new Error(`Unsupported trade-flow encoding: ${reference.layout.encoding}.`);
  }
  const layout = reference.layout as TradeFlowLayout;
  const expectedNames = [...FLOAT_COLUMNS, ...COUNT_COLUMNS, ...SIDE_COLUMNS, ...OFFSET_COLUMNS];
  if (!Array.isArray(layout.columns)
    || layout.columns.length !== expectedNames.length
    || layout.columns.some((column, index) => column.name !== expectedNames[index])) {
    throw new Error("Trade-flow shard layout is invalid.");
  }
  const decoded = new Map<TradeFlowColumn["name"], Array<number | null>>();
  for (const column of layout.columns) {
    const width = column.encoding === "float64-le" ? 8 : column.encoding === "uint32-le" ? 4 : 1;
    const end = column.offset + column.bytes;
    if (!Number.isSafeInteger(column.offset)
      || column.offset < 0
      || column.bytes !== reference.sequence.count * width
      || end > payload.byteLength) {
      throw new Error(`Trade-flow ${column.name} column bounds are invalid.`);
    }
    const bytes = Buffer.from(payload.buffer, payload.byteOffset + column.offset, column.bytes);
    decoded.set(column.name, Array.from({ length: reference.sequence.count }, (_, index) => {
      const value = column.encoding === "float64-le"
        ? bytes.readDoubleLE(index * 8)
        : column.encoding === "uint32-le"
          ? bytes.readUInt32LE(index * 4)
          : bytes.readInt8(index);
      return value === column.nullValue ? null : value;
    }));
  }
  return Array.from({ length: reference.sequence.count }, (_, index) => {
    const value = <Name extends TradeFlowColumn["name"]>(name: Name) =>
      decoded.get(name)![index] as SequentialTradeFlowSecond[Name];
    return {
      openTime: reference.sequence.start + index * reference.sequence.step,
      aggressiveBuyBaseVolume: value("aggressiveBuyBaseVolume"),
      aggressiveSellBaseVolume: value("aggressiveSellBaseVolume"),
      aggressiveBuyQuoteVolume: value("aggressiveBuyQuoteVolume"),
      aggressiveSellQuoteVolume: value("aggressiveSellQuoteVolume"),
      aggressiveBuyAggregateQuantitySquared: value("aggressiveBuyAggregateQuantitySquared"),
      aggressiveSellAggregateQuantitySquared: value("aggressiveSellAggregateQuantitySquared"),
      aggressiveBuyMaxAggregateQuantity: value("aggressiveBuyMaxAggregateQuantity"),
      aggressiveSellMaxAggregateQuantity: value("aggressiveSellMaxAggregateQuantity"),
      aggressiveBuyBaseVolumeTimeMoment: value("aggressiveBuyBaseVolumeTimeMoment"),
      aggressiveSellBaseVolumeTimeMoment: value("aggressiveSellBaseVolumeTimeMoment"),
      aggressiveBuyAggregateTradeCount: value("aggressiveBuyAggregateTradeCount"),
      aggressiveSellAggregateTradeCount: value("aggressiveSellAggregateTradeCount"),
      aggressiveBuyTradeCount: value("aggressiveBuyTradeCount"),
      aggressiveSellTradeCount: value("aggressiveSellTradeCount"),
      aggressorSideFlipCount: value("aggressorSideFlipCount"),
      firstAggressorSide: value("firstAggressorSide"),
      lastAggressorSide: value("lastAggressorSide"),
      firstTradeOffsetMicros: value("firstTradeOffsetMicros"),
      lastTradeOffsetMicros: value("lastTradeOffsetMicros"),
    };
  });
}

export async function readTradeFlowShardReference(
  referenceFile: string,
): Promise<SequentialTradeFlowSecond[]> {
  const { reference, payload } = await readReferencedPayload(referenceFile);
  return decodeTradeFlow(reference, payload);
}

export function readTradeFlowShardReferenceSync(
  referenceFile: string,
): SequentialTradeFlowSecond[] {
  const { reference, payload } = readReferencedPayloadSync(referenceFile);
  return decodeTradeFlow(reference, payload);
}

export function emptyTradeFlowSecond(openTime: number): SequentialTradeFlowSecond {
  return {
    openTime,
    aggressiveBuyBaseVolume: 0,
    aggressiveSellBaseVolume: 0,
    aggressiveBuyQuoteVolume: 0,
    aggressiveSellQuoteVolume: 0,
    aggressiveBuyAggregateQuantitySquared: 0,
    aggressiveSellAggregateQuantitySquared: 0,
    aggressiveBuyMaxAggregateQuantity: 0,
    aggressiveSellMaxAggregateQuantity: 0,
    aggressiveBuyBaseVolumeTimeMoment: 0,
    aggressiveSellBaseVolumeTimeMoment: 0,
    aggressiveBuyAggregateTradeCount: 0,
    aggressiveSellAggregateTradeCount: 0,
    aggressiveBuyTradeCount: 0,
    aggressiveSellTradeCount: 0,
    aggressorSideFlipCount: 0,
    firstAggressorSide: 0,
    lastAggressorSide: 0,
    firstTradeOffsetMicros: null,
    lastTradeOffsetMicros: null,
  };
}

function validateSecond(
  second: SequentialTradeFlowSecond,
  expectedTime: number,
  stepMs: number,
  index: number,
): void {
  if (second.openTime !== expectedTime) {
    throw new Error(`Trade-flow shard time axis is invalid at row ${index}.`);
  }
  if (!FLOAT_COLUMNS.every((name) => Number.isFinite(second[name]) && second[name] >= 0)
    || !COUNT_COLUMNS.every((name) => Number.isSafeInteger(second[name])
      && second[name] >= 0 && second[name] <= 0xffff_ffff)
    || !SIDE_COLUMNS.every((name) => second[name] === -1
      || second[name] === 0 || second[name] === 1)
    || !OFFSET_COLUMNS.every((name) => second[name] === null
      || (Number.isSafeInteger(second[name]) && second[name]! >= 0
        && second[name]! < stepMs * 1_000))) {
    throw new Error(`Trade-flow shard contains an invalid value at row ${index}.`);
  }
  const aggregateCount = second.aggressiveBuyAggregateTradeCount
    + second.aggressiveSellAggregateTradeCount;
  const tradeCount = second.aggressiveBuyTradeCount + second.aggressiveSellTradeCount;
  if ((aggregateCount === 0) !== (tradeCount === 0)
    || (aggregateCount === 0) !== (second.firstAggressorSide === 0)
    || (aggregateCount === 0) !== (second.lastAggressorSide === 0)
    || (aggregateCount === 0) !== (second.firstTradeOffsetMicros === null)
    || (aggregateCount === 0) !== (second.lastTradeOffsetMicros === null)
    || second.aggressorSideFlipCount > Math.max(0, aggregateCount - 1)) {
    throw new Error(`Trade-flow no-trade invariants are invalid at row ${index}.`);
  }
}
