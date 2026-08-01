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

export interface SequentialCandle {
  symbol: string;
  interval: string;
  openTime: number;
  closeTime: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  closed: boolean;
}

interface CandleColumn {
  name: "open" | "high" | "low" | "close" | "volume";
  encoding: "scaled-delta-zigzag-varint" | "float64-le";
  offset: number;
  bytes: number;
  scale?: number;
}

interface CandleTimeJump {
  index: number;
  deltaMs: number;
}

interface CandleCloseTimeOffsetOverride {
  index: number;
  offsetMs: number;
}

interface CandleClosedOverride {
  index: number;
  closed: boolean;
}

interface CandleLayout extends ShardLayout {
  encoding: "candle-columnar-delta-v1";
  columns: CandleColumn[];
  constants: {
    symbol: string;
    interval: string;
    closeTimeOffsetMs: number;
    closed: boolean;
  };
  timeJumps?: CandleTimeJump[];
  closeTimeOffsetOverrides?: CandleCloseTimeOffsetOverride[];
  closedOverrides?: CandleClosedOverride[];
}

export interface PutCandleShardRequest {
  namespace: string;
  key: string;
  candles: readonly SequentialCandle[];
  stepMs?: number;
  empty?: {
    symbol: string;
    interval: string;
    startTime: number;
    closeTimeOffsetMs: number;
    closed: boolean;
  };
  metadata?: StorageMetadata;
}

export async function putCandleShard(
  store: SequentialShardStore,
  request: PutCandleShardRequest,
): Promise<PutSequentialShardResult> {
  const encoded = request.candles.length > 0
    ? encodeCandles(request.candles, request.stepMs)
    : encodeEmptyCandleShard(request.stepMs, request.empty);
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

function encodeEmptyCandleShard(
  stepMs: number | undefined,
  descriptor: PutCandleShardRequest["empty"],
): ReturnType<typeof encodeCandles> {
  if (!descriptor || !Number.isSafeInteger(stepMs) || stepMs! < 1) {
    throw new Error("An empty candle shard requires its candle descriptor and interval.");
  }
  if (!descriptor.symbol || !descriptor.interval
    || !Number.isSafeInteger(descriptor.startTime)
    || !Number.isSafeInteger(descriptor.closeTimeOffsetMs)
    || descriptor.closeTimeOffsetMs < 0
    || descriptor.closeTimeOffsetMs >= stepMs!) {
    throw new Error("Empty candle shard descriptor is invalid.");
  }
  return {
    payload: Buffer.alloc(0),
    sequence: {
      start: descriptor.startTime,
      step: stepMs!,
      count: 0,
      unit: "unix-ms",
    },
    layout: {
      encoding: "candle-columnar-delta-v1",
      columns: ["open", "high", "low", "close", "volume"].map((name) => ({
        name,
        encoding: "float64-le",
        offset: 0,
        bytes: 0,
      })) as CandleColumn[],
      constants: {
        symbol: descriptor.symbol,
        interval: descriptor.interval,
        closeTimeOffsetMs: descriptor.closeTimeOffsetMs,
        closed: descriptor.closed,
      },
    },
  };
}

export function encodeCandles(
  candles: readonly SequentialCandle[],
  requestedStepMs?: number,
): {
  payload: Buffer;
  sequence: SequentialShardReference["sequence"];
  layout: CandleLayout;
} {
  if (candles.length === 0) throw new Error("Cannot encode an empty candle shard.");
  const first = candles[0]!;
  const step = requestedStepMs
    ?? (candles.length > 1 ? candles[1]!.openTime - first.openTime : undefined);
  if (!Number.isSafeInteger(step) || step! < 1) {
    throw new Error("Candle shard interval is missing or invalid.");
  }
  const closeTimeOffsetMs = first.closeTime - first.openTime;
  if (!Number.isSafeInteger(closeTimeOffsetMs)
    || closeTimeOffsetMs < 0
    || closeTimeOffsetMs >= step!) {
    throw new Error("Candle close-time offset is invalid.");
  }
  const timeJumps: CandleTimeJump[] = [];
  const closeTimeOffsetOverrides: CandleCloseTimeOffsetOverride[] = [];
  const closedOverrides: CandleClosedOverride[] = [];
  let cumulativeTimeJumpMs = 0;
  for (let index = 0; index < candles.length; index += 1) {
    const candle = candles[index]!;
    const expectedOpenTime = first.openTime + index * step! + cumulativeTimeJumpMs;
    if (index > 0 && candle.openTime !== expectedOpenTime) {
      const deltaMs = candle.openTime - expectedOpenTime;
      if (!Number.isSafeInteger(deltaMs)
        || candle.openTime <= candles[index - 1]!.openTime) {
        throw new Error(`Candle shard time axis is invalid at row ${index}.`);
      }
      timeJumps.push({ index, deltaMs });
      cumulativeTimeJumpMs += deltaMs;
    }
    const candleCloseTimeOffsetMs = candle.closeTime - candle.openTime;
    if (!Number.isSafeInteger(candleCloseTimeOffsetMs)
      || candleCloseTimeOffsetMs < 0
      || candleCloseTimeOffsetMs >= step!) {
      throw new Error(`Candle close-time offset is invalid at row ${index}.`);
    }
    if (candle.symbol !== first.symbol
      || candle.interval !== first.interval) {
      throw new Error(`Candle shard constants changed at row ${index}.`);
    }
    if (candleCloseTimeOffsetMs !== closeTimeOffsetMs) {
      closeTimeOffsetOverrides.push({ index, offsetMs: candleCloseTimeOffsetMs });
    }
    if (candle.closed !== first.closed) {
      closedOverrides.push({ index, closed: candle.closed });
    }
    if (![candle.open, candle.high, candle.low, candle.close, candle.volume]
      .every(Number.isFinite)) {
      throw new Error(`Candle shard contains a non-finite value at row ${index}.`);
    }
  }

  const chunks: Buffer[] = [];
  const columns: CandleColumn[] = [];
  for (const name of ["open", "high", "low", "close", "volume"] as const) {
    const values = candles.map((candle) => candle[name]);
    const encoded = encodeNumericColumn(values);
    const offset = chunks.reduce((total, chunk) => total + chunk.byteLength, 0);
    chunks.push(encoded.bytes);
    columns.push({
      name,
      encoding: encoded.encoding,
      offset,
      bytes: encoded.bytes.byteLength,
      ...(encoded.scale === undefined ? {} : { scale: encoded.scale }),
    });
  }
  return {
    payload: Buffer.concat(chunks),
    sequence: {
      start: first.openTime,
      step: step!,
      count: candles.length,
      unit: "unix-ms",
    },
    layout: {
      encoding: "candle-columnar-delta-v1",
      columns,
      constants: {
        symbol: first.symbol,
        interval: first.interval,
        closeTimeOffsetMs,
        closed: first.closed,
      },
      ...(timeJumps.length > 0 ? { timeJumps } : {}),
      ...(closeTimeOffsetOverrides.length > 0 ? { closeTimeOffsetOverrides } : {}),
      ...(closedOverrides.length > 0 ? { closedOverrides } : {}),
    },
  };
}

export function decodeCandles(
  reference: SequentialShardReference,
  payload: Uint8Array,
): SequentialCandle[] {
  if (reference.layout.encoding !== "candle-columnar-delta-v1") {
    throw new Error(`Unsupported candle encoding: ${reference.layout.encoding}.`);
  }
  const layout = reference.layout as CandleLayout;
  if (!Array.isArray(layout.columns) || layout.columns.length !== 5 || !layout.constants) {
    throw new Error("Candle shard layout is invalid.");
  }
  const values = new Map<CandleColumn["name"], number[]>();
  for (const column of layout.columns) {
    const end = column.offset + column.bytes;
    if (!Number.isSafeInteger(column.offset)
      || !Number.isSafeInteger(column.bytes)
      || column.offset < 0
      || column.bytes < 0
      || end > payload.byteLength) {
      throw new Error(`Candle ${column.name} column bounds are invalid.`);
    }
    values.set(
      column.name,
      decodeNumericColumn(payload.subarray(column.offset, end), column, reference.sequence.count),
    );
  }
  const timeJumps = new Map<number, number>();
  for (const jump of layout.timeJumps ?? []) {
    if (!Number.isSafeInteger(jump.index)
      || jump.index < 1
      || jump.index >= reference.sequence.count
      || !Number.isSafeInteger(jump.deltaMs)
      || timeJumps.has(jump.index)) {
      throw new Error("Candle shard time-jump metadata is invalid.");
    }
    timeJumps.set(jump.index, jump.deltaMs);
  }
  const closeTimeOffsetOverrides = new Map<number, number>();
  for (const override of layout.closeTimeOffsetOverrides ?? []) {
    if (!Number.isSafeInteger(override.index)
      || override.index < 1
      || override.index >= reference.sequence.count
      || !Number.isSafeInteger(override.offsetMs)
      || override.offsetMs < 0
      || override.offsetMs >= reference.sequence.step
      || closeTimeOffsetOverrides.has(override.index)) {
      throw new Error("Candle shard close-time override metadata is invalid.");
    }
    closeTimeOffsetOverrides.set(override.index, override.offsetMs);
  }
  const closedOverrides = new Map<number, boolean>();
  for (const override of layout.closedOverrides ?? []) {
    if (!Number.isSafeInteger(override.index)
      || override.index < 1
      || override.index >= reference.sequence.count
      || typeof override.closed !== "boolean"
      || closedOverrides.has(override.index)) {
      throw new Error("Candle shard closed override metadata is invalid.");
    }
    closedOverrides.set(override.index, override.closed);
  }
  let cumulativeTimeJumpMs = 0;
  return Array.from({ length: reference.sequence.count }, (_, index) => {
    cumulativeTimeJumpMs += timeJumps.get(index) ?? 0;
    const openTime = reference.sequence.start
      + index * reference.sequence.step
      + cumulativeTimeJumpMs;
    return {
      symbol: layout.constants.symbol,
      interval: layout.constants.interval,
      openTime,
      closeTime: openTime + (
        closeTimeOffsetOverrides.get(index) ?? layout.constants.closeTimeOffsetMs
      ),
      open: values.get("open")![index]!,
      high: values.get("high")![index]!,
      low: values.get("low")![index]!,
      close: values.get("close")![index]!,
      volume: values.get("volume")![index]!,
      closed: closedOverrides.get(index) ?? layout.constants.closed,
    };
  });
}

export async function readCandleShardReference(
  referenceFile: string,
): Promise<SequentialCandle[]> {
  const { reference, payload } = await readReferencedPayload(referenceFile);
  return decodeCandles(reference, payload);
}

export function readCandleShardReferenceSync(referenceFile: string): SequentialCandle[] {
  const { reference, payload } = readReferencedPayloadSync(referenceFile);
  return decodeCandles(reference, payload);
}

function encodeNumericColumn(values: readonly number[]): {
  bytes: Buffer;
  encoding: CandleColumn["encoding"];
  scale?: number;
} {
  const scale = exactDecimalScale(values);
  if (scale === undefined) {
    const bytes = Buffer.allocUnsafe(values.length * 8);
    values.forEach((value, index) => bytes.writeDoubleLE(value, index * 8));
    return { bytes, encoding: "float64-le" };
  }
  const output: number[] = [];
  let previous = 0n;
  values.forEach((value, index) => {
    const integer = BigInt(Math.round(value * scale));
    const delta = index === 0 ? integer : integer - previous;
    writeUnsignedVarint(zigZag(delta), output);
    previous = integer;
  });
  return {
    bytes: Buffer.from(output),
    encoding: "scaled-delta-zigzag-varint",
    scale,
  };
}

function decodeNumericColumn(
  bytes: Uint8Array,
  column: CandleColumn,
  count: number,
): number[] {
  if (column.encoding === "float64-le") {
    if (bytes.byteLength !== count * 8) throw new Error(`${column.name} float column is truncated.`);
    const view = Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    return Array.from({ length: count }, (_, index) => view.readDoubleLE(index * 8));
  }
  if (!Number.isSafeInteger(column.scale) || column.scale! < 1) {
    throw new Error(`${column.name} scaled column has an invalid scale.`);
  }
  const values: number[] = [];
  let offset = 0;
  let previous = 0n;
  for (let index = 0; index < count; index += 1) {
    const decoded = readUnsignedVarint(bytes, offset);
    offset = decoded.offset;
    const integer = (index === 0 ? 0n : previous) + unZigZag(decoded.value);
    values.push(Number(integer) / column.scale!);
    previous = integer;
  }
  if (offset !== bytes.byteLength) throw new Error(`${column.name} column has trailing bytes.`);
  return values;
}

function exactDecimalScale(values: readonly number[]): number | undefined {
  for (let decimals = 0, scale = 1; decimals <= 12; decimals += 1, scale *= 10) {
    let valid = true;
    for (const value of values) {
      const scaled = value * scale;
      if (!Number.isSafeInteger(Math.round(scaled))
        || Math.round(scaled) / scale !== value) {
        valid = false;
        break;
      }
    }
    if (valid) return scale;
  }
  return undefined;
}

function zigZag(value: bigint): bigint {
  return value >= 0n ? value * 2n : -value * 2n - 1n;
}

function unZigZag(value: bigint): bigint {
  return value % 2n === 0n ? value / 2n : -(value + 1n) / 2n;
}

function writeUnsignedVarint(value: bigint, output: number[]): void {
  let remaining = value;
  while (remaining >= 0x80n) {
    output.push(Number((remaining & 0x7fn) | 0x80n));
    remaining >>= 7n;
  }
  output.push(Number(remaining));
}

function readUnsignedVarint(
  bytes: Uint8Array,
  start: number,
): { value: bigint; offset: number } {
  let value = 0n;
  let shift = 0n;
  let offset = start;
  while (offset < bytes.byteLength && shift <= 70n) {
    const byte = bytes[offset++]!;
    value |= BigInt(byte & 0x7f) << shift;
    if ((byte & 0x80) === 0) return { value, offset };
    shift += 7n;
  }
  throw new Error("Invalid or truncated unsigned varint.");
}
