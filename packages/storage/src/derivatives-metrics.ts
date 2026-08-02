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

export const DERIVATIVES_METRIC_COLUMNS = [
  "sumOpenInterest",
  "sumOpenInterestValue",
  "topTraderAccountLongShortRatio",
  "topTraderPositionLongShortRatio",
  "globalLongShortRatio",
  "takerBuySellVolumeRatio",
] as const;

export type DerivativesMetricName = typeof DERIVATIVES_METRIC_COLUMNS[number];

export interface SequentialDerivativesMetricRow
  extends Record<DerivativesMetricName, number | null> {
  openTime: number;
}

interface DerivativesMetricColumn {
  name: DerivativesMetricName | "validMask";
  encoding: "float64-le" | "uint8";
  offset: number;
  bytes: number;
}

interface DerivativesMetricsLayout extends ShardLayout {
  encoding: "derivatives-metrics-columnar-v1";
  columns: DerivativesMetricColumn[];
}

export interface PutDerivativesMetricsShardRequest {
  namespace: string;
  key: string;
  rows: readonly SequentialDerivativesMetricRow[];
  stepMs?: number;
  metadata?: StorageMetadata;
}

export async function putDerivativesMetricsShard(
  store: SequentialShardStore,
  request: PutDerivativesMetricsShardRequest,
): Promise<PutSequentialShardResult> {
  const encoded = encodeDerivativesMetrics(request.rows, request.stepMs);
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

export function encodeDerivativesMetrics(
  rows: readonly SequentialDerivativesMetricRow[],
  requestedStepMs?: number,
): {
  payload: Buffer;
  sequence: SequentialShardReference["sequence"];
  layout: DerivativesMetricsLayout;
} {
  if (rows.length === 0) throw new Error("Cannot encode an empty derivatives-metrics shard.");
  const first = rows[0]!;
  const step = requestedStepMs
    ?? (rows.length > 1 ? rows[1]!.openTime - first.openTime : undefined);
  if (!Number.isSafeInteger(step) || step! < 1) {
    throw new Error("Derivatives-metrics shard interval is missing or invalid.");
  }
  rows.forEach((row, index) => validateRow(
    row,
    first.openTime + index * step!,
    index,
  ));

  const chunks: Buffer[] = [];
  const columns: DerivativesMetricColumn[] = [];
  for (const name of DERIVATIVES_METRIC_COLUMNS) {
    const bytes = Buffer.allocUnsafe(rows.length * 8);
    rows.forEach((row, index) => bytes.writeDoubleLE(row[name] ?? 0, index * 8));
    const offset = chunks.reduce((total, chunk) => total + chunk.byteLength, 0);
    chunks.push(bytes);
    columns.push({ name, encoding: "float64-le", offset, bytes: bytes.byteLength });
  }
  const validMask = Buffer.alloc(rows.length);
  rows.forEach((row, rowIndex) => {
    let mask = 0;
    DERIVATIVES_METRIC_COLUMNS.forEach((name, columnIndex) => {
      if (row[name] !== null) mask |= 1 << columnIndex;
    });
    validMask.writeUInt8(mask, rowIndex);
  });
  columns.push({
    name: "validMask",
    encoding: "uint8",
    offset: chunks.reduce((total, chunk) => total + chunk.byteLength, 0),
    bytes: validMask.byteLength,
  });
  chunks.push(validMask);
  return {
    payload: Buffer.concat(chunks),
    sequence: {
      start: first.openTime,
      step: step!,
      count: rows.length,
      unit: "unix-ms",
    },
    layout: { encoding: "derivatives-metrics-columnar-v1", columns },
  };
}

export function decodeDerivativesMetrics(
  reference: SequentialShardReference,
  payload: Uint8Array,
): SequentialDerivativesMetricRow[] {
  if (reference.layout.encoding !== "derivatives-metrics-columnar-v1") {
    throw new Error(
      `Unsupported derivatives-metrics encoding: ${reference.layout.encoding}.`,
    );
  }
  const layout = reference.layout as DerivativesMetricsLayout;
  if (!Array.isArray(layout.columns)
    || layout.columns.length !== DERIVATIVES_METRIC_COLUMNS.length + 1
    || layout.columns.slice(0, -1).some((column, index) => (
      column.name !== DERIVATIVES_METRIC_COLUMNS[index]
      || column.encoding !== "float64-le"
      || column.offset !== index * reference.sequence.count * 8
      || column.bytes !== reference.sequence.count * 8
    ))
    || layout.columns.at(-1)?.name !== "validMask"
    || layout.columns.at(-1)?.encoding !== "uint8"
    || layout.columns.at(-1)?.offset !== reference.sequence.count
      * DERIVATIVES_METRIC_COLUMNS.length * 8
    || layout.columns.at(-1)?.bytes !== reference.sequence.count
    || payload.byteLength !== reference.sequence.count
      * (DERIVATIVES_METRIC_COLUMNS.length * 8 + 1)) {
    throw new Error("Derivatives-metrics shard layout is invalid.");
  }
  const bytes = Buffer.from(payload.buffer, payload.byteOffset, payload.byteLength);
  const validMask = layout.columns.at(-1)!.offset;
  const rows = Array.from({ length: reference.sequence.count }, (_, index) => {
    const row = {
      openTime: reference.sequence.start + index * reference.sequence.step,
    } as SequentialDerivativesMetricRow;
    const mask = bytes.readUInt8(validMask + index);
    layout.columns.slice(0, -1).forEach((column, columnIndex) => {
      row[column.name as DerivativesMetricName] = (mask & (1 << columnIndex)) === 0
        ? null
        : bytes.readDoubleLE(column.offset + index * 8);
    });
    return row;
  });
  rows.forEach((row, index) => validateRow(row, row.openTime, index));
  return rows;
}

export async function readDerivativesMetricsShardReference(
  referenceFile: string,
): Promise<SequentialDerivativesMetricRow[]> {
  const { reference, payload } = await readReferencedPayload(referenceFile);
  return decodeDerivativesMetrics(reference, payload);
}

export function readDerivativesMetricsShardReferenceSync(
  referenceFile: string,
): SequentialDerivativesMetricRow[] {
  const { reference, payload } = readReferencedPayloadSync(referenceFile);
  return decodeDerivativesMetrics(reference, payload);
}

function validateRow(
  row: SequentialDerivativesMetricRow,
  expectedTime: number,
  index: number,
): void {
  if (!Number.isSafeInteger(row.openTime) || row.openTime !== expectedTime) {
    throw new Error(`Derivatives-metrics shard time axis is invalid at row ${index}.`);
  }
  if (!DERIVATIVES_METRIC_COLUMNS.every((name) => (
    row[name] === null || (Number.isFinite(row[name]) && row[name]! > 0)
  ))) {
    throw new Error(`Derivatives-metrics shard contains an invalid value at row ${index}.`);
  }
}
