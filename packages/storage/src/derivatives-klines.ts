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

export const DERIVATIVES_KLINE_COLUMNS = [
  "open",
  "high",
  "low",
  "close",
  "baseVolume",
  "quoteVolume",
  "tradeCount",
  "takerBuyBaseVolume",
  "takerBuyQuoteVolume",
] as const;

export const DERIVATIVES_KLINE_ENCODING = "derivatives-klines-columnar-v1";
export const DERIVATIVES_KLINE_STEP_MS = 60_000;
export const DERIVATIVES_KLINE_CLOSE_TIME_OFFSET_MS = 59_999;

export type DerivativesKlineName = typeof DERIVATIVES_KLINE_COLUMNS[number];

export interface SequentialDerivativesKlineRow
  extends Record<DerivativesKlineName, number | null> {
  openTime: number;
}

export interface DerivativesKlineColumn {
  name: DerivativesKlineName | "validMask";
  encoding: "float64-le" | "uint64-le" | "uint8";
  offset: number;
  bytes: number;
}

export interface DerivativesKlinesLayout extends ShardLayout {
  encoding: typeof DERIVATIVES_KLINE_ENCODING;
  columns: DerivativesKlineColumn[];
  closeTimeOffsetMs: typeof DERIVATIVES_KLINE_CLOSE_TIME_OFFSET_MS;
  closed: true;
}

export interface PutDerivativesKlinesShardRequest {
  namespace: string;
  key: string;
  rows: readonly SequentialDerivativesKlineRow[];
  metadata?: StorageMetadata;
}

export async function putDerivativesKlinesShard(
  store: SequentialShardStore,
  request: PutDerivativesKlinesShardRequest,
): Promise<PutSequentialShardResult> {
  const encoded = encodeDerivativesKlines(request.rows);
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

export function encodeDerivativesKlines(
  rows: readonly SequentialDerivativesKlineRow[],
): {
  payload: Buffer;
  sequence: SequentialShardReference["sequence"];
  layout: DerivativesKlinesLayout;
} {
  if (rows.length === 0) throw new Error("Cannot encode an empty derivatives-klines shard.");
  const start = rows[0]!.openTime;
  rows.forEach((row, index) => validateRow(
    row,
    start + index * DERIVATIVES_KLINE_STEP_MS,
    index,
  ));

  const chunks: Buffer[] = [];
  const columns: DerivativesKlineColumn[] = [];
  let offset = 0;
  for (const name of DERIVATIVES_KLINE_COLUMNS) {
    const encoding = name === "tradeCount" ? "uint64-le" : "float64-le";
    const bytes = Buffer.alloc(rows.length * 8);
    rows.forEach((row, index) => {
      const value = row[name] ?? 0;
      if (encoding === "uint64-le") {
        bytes.writeBigUInt64LE(BigInt(value), index * 8);
      } else {
        bytes.writeDoubleLE(value, index * 8);
      }
    });
    chunks.push(bytes);
    columns.push({ name, encoding, offset, bytes: bytes.byteLength });
    offset += bytes.byteLength;
  }
  const validMask = Buffer.alloc(rows.length);
  rows.forEach((row, index) => validMask.writeUInt8(row.open === null ? 0 : 1, index));
  columns.push({
    name: "validMask",
    encoding: "uint8",
    offset,
    bytes: validMask.byteLength,
  });
  chunks.push(validMask);
  return {
    payload: Buffer.concat(chunks),
    sequence: {
      start,
      step: DERIVATIVES_KLINE_STEP_MS,
      count: rows.length,
      unit: "unix-ms",
    },
    layout: {
      encoding: DERIVATIVES_KLINE_ENCODING,
      columns,
      closeTimeOffsetMs: DERIVATIVES_KLINE_CLOSE_TIME_OFFSET_MS,
      closed: true,
    },
  };
}

export function decodeDerivativesKlines(
  reference: SequentialShardReference,
  payload: Uint8Array,
): SequentialDerivativesKlineRow[] {
  const layout = validateLayout(reference, payload.byteLength);
  const bytes = Buffer.from(payload.buffer, payload.byteOffset, payload.byteLength);
  const maskOffset = layout.columns.at(-1)!.offset;
  const rows = Array.from({ length: reference.sequence.count }, (_, index) => {
    const valid = bytes.readUInt8(maskOffset + index);
    if (valid !== 0 && valid !== 1) {
      throw new Error(`Derivatives-klines validity mask is invalid at row ${index}.`);
    }
    const row = {
      openTime: reference.sequence.start + index * reference.sequence.step,
    } as SequentialDerivativesKlineRow;
    layout.columns.slice(0, -1).forEach((column) => {
      const valueOffset = column.offset + index * 8;
      let value: number;
      if (column.encoding === "uint64-le") {
        const integer = bytes.readBigUInt64LE(valueOffset);
        if (integer > BigInt(Number.MAX_SAFE_INTEGER)) {
          throw new Error(`Derivatives-klines trade count exceeds safe integers at row ${index}.`);
        }
        value = Number(integer);
      } else {
        value = bytes.readDoubleLE(valueOffset);
      }
      if (valid === 0 && value !== 0) {
        throw new Error(`Derivatives-klines missing row ${index} contains nonzero data.`);
      }
      row[column.name as DerivativesKlineName] = valid === 0 ? null : value;
    });
    return row;
  });
  rows.forEach((row, index) => validateRow(row, row.openTime, index));
  return rows;
}

export async function readDerivativesKlinesShardReference(
  referenceFile: string,
): Promise<SequentialDerivativesKlineRow[]> {
  const { reference, payload } = await readReferencedPayload(referenceFile);
  return decodeDerivativesKlines(reference, payload);
}

export function readDerivativesKlinesShardReferenceSync(
  referenceFile: string,
): SequentialDerivativesKlineRow[] {
  const { reference, payload } = readReferencedPayloadSync(referenceFile);
  return decodeDerivativesKlines(reference, payload);
}

function validateLayout(
  reference: SequentialShardReference,
  payloadBytes: number,
): DerivativesKlinesLayout {
  if (reference.layout.encoding !== DERIVATIVES_KLINE_ENCODING) {
    throw new Error(`Unsupported derivatives-klines encoding: ${reference.layout.encoding}.`);
  }
  const layout = reference.layout as DerivativesKlinesLayout;
  const count = reference.sequence.count;
  let expectedOffset = 0;
  if (reference.sequence.unit !== "unix-ms"
    || reference.sequence.step !== DERIVATIVES_KLINE_STEP_MS
    || layout.closeTimeOffsetMs !== DERIVATIVES_KLINE_CLOSE_TIME_OFFSET_MS
    || layout.closed !== true
    || !Array.isArray(layout.columns)
    || layout.columns.length !== DERIVATIVES_KLINE_COLUMNS.length + 1) {
    throw new Error("Derivatives-klines shard layout is invalid.");
  }
  for (let index = 0; index < DERIVATIVES_KLINE_COLUMNS.length; index += 1) {
    const name = DERIVATIVES_KLINE_COLUMNS[index]!;
    const column = layout.columns[index];
    const encoding = name === "tradeCount" ? "uint64-le" : "float64-le";
    if (column?.name !== name
      || column.encoding !== encoding
      || column.offset !== expectedOffset
      || column.bytes !== count * 8) {
      throw new Error("Derivatives-klines shard layout is invalid.");
    }
    expectedOffset += count * 8;
  }
  const validMask = layout.columns.at(-1);
  if (validMask?.name !== "validMask"
    || validMask.encoding !== "uint8"
    || validMask.offset !== expectedOffset
    || validMask.bytes !== count
    || payloadBytes !== expectedOffset + count) {
    throw new Error("Derivatives-klines shard layout is invalid.");
  }
  return layout;
}

function validateRow(
  row: SequentialDerivativesKlineRow,
  expectedTime: number,
  index: number,
): void {
  if (!Number.isSafeInteger(row.openTime) || row.openTime !== expectedTime) {
    throw new Error(`Derivatives-klines shard time axis is invalid at row ${index}.`);
  }
  const nullCount = DERIVATIVES_KLINE_COLUMNS.reduce(
    (count, name) => count + (row[name] === null ? 1 : 0),
    0,
  );
  if (nullCount !== 0 && nullCount !== DERIVATIVES_KLINE_COLUMNS.length) {
    throw new Error(`Derivatives-klines row ${index} is only partially missing.`);
  }
  if (nullCount !== 0) return;
  const {
    open,
    high,
    low,
    close,
    baseVolume,
    quoteVolume,
    tradeCount,
    takerBuyBaseVolume,
    takerBuyQuoteVolume,
  } = row as Record<DerivativesKlineName, number> & { openTime: number };
  if (![open, high, low, close].every((value) => Number.isFinite(value) && value > 0)
    || ![
      baseVolume,
      quoteVolume,
      takerBuyBaseVolume,
      takerBuyQuoteVolume,
    ].every((value) => Number.isFinite(value) && value >= 0)
    || !Number.isSafeInteger(tradeCount)
    || tradeCount < 0
    || high < Math.max(open, close)
    || low > Math.min(open, close)
    || low > high
    || takerBuyBaseVolume > baseVolume
    || takerBuyQuoteVolume > quoteVolume
    || (tradeCount === 0 && (
      open !== high
      || open !== low
      || open !== close
      || baseVolume !== 0
      || quoteVolume !== 0
      || takerBuyBaseVolume !== 0
      || takerBuyQuoteVolume !== 0
    ))
    || (tradeCount > 0 && (baseVolume === 0 || quoteVolume === 0))) {
    throw new Error(`Derivatives-klines shard contains an invalid row at index ${index}.`);
  }
}
