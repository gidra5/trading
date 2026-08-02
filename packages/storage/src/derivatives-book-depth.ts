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

export const DERIVATIVES_BOOK_DEPTH_ENCODING = "derivatives-book-depth-columnar-v1";
export const DERIVATIVES_BOOK_DEPTH_TIMESTAMP_RESOLUTION_MS = 1_000;
export const DERIVATIVES_BOOK_DEPTH_DAY_SECONDS = 86_400;
export const DERIVATIVES_BOOK_DEPTH_BANDS = [0.2, 1, 2, 3, 4, 5] as const;
export const DERIVATIVES_BOOK_DEPTH_MATRIX_COLUMNS = [
  "bidDepth",
  "askDepth",
  "bidNotional",
  "askNotional",
] as const;
export const DERIVATIVES_BOOK_DEPTH_TEN_BAND_MASK = 0b11_1110;
export const DERIVATIVES_BOOK_DEPTH_TWELVE_BAND_MASK = 0b11_1111;

export type DerivativesBookDepthBand = typeof DERIVATIVES_BOOK_DEPTH_BANDS[number];
export type DerivativesBookDepthMatrixName =
  typeof DERIVATIVES_BOOK_DEPTH_MATRIX_COLUMNS[number];
export type DerivativesBookDepthSchemaBandCount = 10 | 12;
export type DerivativesBookDepthValueVector = readonly [
  number | null,
  number | null,
  number | null,
  number | null,
  number | null,
  number | null,
];
export type DerivativesBookDepthAvailabilityVector = readonly [
  boolean,
  boolean,
  boolean,
  boolean,
  boolean,
  boolean,
];

/**
 * Logical decoded row. Arrays use DERIVATIVES_BOOK_DEPTH_BANDS as their fixed
 * second axis, so a later Python reader can stack these rows directly into
 * [snapshot, band] matrices with the same public field names.
 */
export interface SequentialDerivativesBookDepthSnapshot {
  timestampOffsetSeconds: number;
  schemaBandCount: DerivativesBookDepthSchemaBandCount;
  bidDepth: DerivativesBookDepthValueVector;
  askDepth: DerivativesBookDepthValueVector;
  bidNotional: DerivativesBookDepthValueVector;
  askNotional: DerivativesBookDepthValueVector;
  bandAvailable: DerivativesBookDepthAvailabilityVector;
}

export type DerivativesBookDepthColumnName =
  | "timestampOffsetSeconds"
  | "schemaBandCount"
  | DerivativesBookDepthMatrixName
  | "bandAvailable";

export interface DerivativesBookDepthColumn {
  name: DerivativesBookDepthColumnName;
  encoding: "uint32-le" | "uint8" | "float64-le";
  offset: number;
  bytes: number;
  shape: [number] | [number, number];
}

export interface DerivativesBookDepthLayout extends ShardLayout {
  encoding: typeof DERIVATIVES_BOOK_DEPTH_ENCODING;
  columns: DerivativesBookDepthColumn[];
  utcDayStartMs: number;
  timestampResolutionMs: typeof DERIVATIVES_BOOK_DEPTH_TIMESTAMP_RESOLUTION_MS;
  timestampStorage: "utc-day-second-offset";
  bandPercentages: [...typeof DERIVATIVES_BOOK_DEPTH_BANDS];
  percentageSemantics: "negative-bid-positive-ask";
  valuesAreCumulative: true;
  matrixOrder: "snapshot-major-band-minor";
}

export interface PutDerivativesBookDepthShardRequest {
  namespace: string;
  key: string;
  utcDayStartMs: number;
  snapshots: readonly SequentialDerivativesBookDepthSnapshot[];
  metadata?: StorageMetadata;
}

export async function putDerivativesBookDepthShard(
  store: SequentialShardStore,
  request: PutDerivativesBookDepthShardRequest,
): Promise<PutSequentialShardResult> {
  const encoded = encodeDerivativesBookDepth(request.snapshots, request.utcDayStartMs);
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

export function encodeDerivativesBookDepth(
  snapshots: readonly SequentialDerivativesBookDepthSnapshot[],
  utcDayStartMs: number,
): {
  payload: Buffer;
  sequence: SequentialShardReference["sequence"];
  layout: DerivativesBookDepthLayout;
} {
  validateUtcDayStart(utcDayStartMs);
  if (snapshots.length === 0) {
    throw new Error("Cannot encode an empty derivatives-book-depth shard.");
  }
  snapshots.forEach((snapshot, index) => validateSnapshot(
    snapshot,
    index,
    index === 0 ? undefined : snapshots[index - 1]!.timestampOffsetSeconds,
  ));

  const chunks: Buffer[] = [];
  const columns: DerivativesBookDepthColumn[] = [];
  let offset = 0;
  const append = (
    name: DerivativesBookDepthColumnName,
    encoding: DerivativesBookDepthColumn["encoding"],
    bytes: Buffer,
    shape: DerivativesBookDepthColumn["shape"],
  ): void => {
    chunks.push(bytes);
    columns.push({ name, encoding, offset, bytes: bytes.byteLength, shape });
    offset += bytes.byteLength;
  };

  const timestampOffsets = Buffer.alloc(snapshots.length * 4);
  snapshots.forEach((snapshot, index) => timestampOffsets.writeUInt32LE(
    snapshot.timestampOffsetSeconds,
    index * 4,
  ));
  append("timestampOffsetSeconds", "uint32-le", timestampOffsets, [snapshots.length]);

  const schemaBandCounts = Buffer.alloc(snapshots.length);
  snapshots.forEach((snapshot, index) => schemaBandCounts.writeUInt8(
    snapshot.schemaBandCount,
    index,
  ));
  append("schemaBandCount", "uint8", schemaBandCounts, [snapshots.length]);

  for (const name of DERIVATIVES_BOOK_DEPTH_MATRIX_COLUMNS) {
    const values = Buffer.alloc(
      snapshots.length * DERIVATIVES_BOOK_DEPTH_BANDS.length * 8,
    );
    snapshots.forEach((snapshot, snapshotIndex) => {
      snapshot[name].forEach((value, bandIndex) => values.writeDoubleLE(
        value ?? 0,
        (snapshotIndex * DERIVATIVES_BOOK_DEPTH_BANDS.length + bandIndex) * 8,
      ));
    });
    append(name, "float64-le", values, [
      snapshots.length,
      DERIVATIVES_BOOK_DEPTH_BANDS.length,
    ]);
  }

  const availability = Buffer.alloc(
    snapshots.length * DERIVATIVES_BOOK_DEPTH_BANDS.length,
  );
  snapshots.forEach((snapshot, snapshotIndex) => {
    snapshot.bandAvailable.forEach((available, bandIndex) => availability.writeUInt8(
      available ? 1 : 0,
      snapshotIndex * DERIVATIVES_BOOK_DEPTH_BANDS.length + bandIndex,
    ));
  });
  append("bandAvailable", "uint8", availability, [
    snapshots.length,
    DERIVATIVES_BOOK_DEPTH_BANDS.length,
  ]);

  return {
    payload: Buffer.concat(chunks),
    sequence: { start: 0, step: 1, count: snapshots.length, unit: "index" },
    layout: {
      encoding: DERIVATIVES_BOOK_DEPTH_ENCODING,
      columns,
      utcDayStartMs,
      timestampResolutionMs: DERIVATIVES_BOOK_DEPTH_TIMESTAMP_RESOLUTION_MS,
      timestampStorage: "utc-day-second-offset",
      bandPercentages: [...DERIVATIVES_BOOK_DEPTH_BANDS],
      percentageSemantics: "negative-bid-positive-ask",
      valuesAreCumulative: true,
      matrixOrder: "snapshot-major-band-minor",
    },
  };
}

export function decodeDerivativesBookDepth(
  reference: SequentialShardReference,
  payload: Uint8Array,
): SequentialDerivativesBookDepthSnapshot[] {
  const layout = validateLayout(reference, payload.byteLength);
  const bytes = Buffer.from(payload.buffer, payload.byteOffset, payload.byteLength);
  const columns = new Map(layout.columns.map((column) => [column.name, column]));
  const timestampColumn = columns.get("timestampOffsetSeconds")!;
  const schemaColumn = columns.get("schemaBandCount")!;
  const availabilityColumn = columns.get("bandAvailable")!;
  const snapshots = Array.from({ length: reference.sequence.count }, (_, snapshotIndex) => {
    const timestampOffsetSeconds = bytes.readUInt32LE(
      timestampColumn.offset + snapshotIndex * 4,
    );
    const rawSchema = bytes.readUInt8(schemaColumn.offset + snapshotIndex);
    if (rawSchema !== 10 && rawSchema !== 12) {
      throw new Error(
        `Derivatives-book-depth schema code is invalid at row ${snapshotIndex}.`,
      );
    }
    const schemaBandCount: DerivativesBookDepthSchemaBandCount = rawSchema;
    const expectedMask = availabilityMask(schemaBandCount);
    const bandAvailable = DERIVATIVES_BOOK_DEPTH_BANDS.map((_, bandIndex) => {
      const raw = bytes.readUInt8(
        availabilityColumn.offset
          + snapshotIndex * DERIVATIVES_BOOK_DEPTH_BANDS.length
          + bandIndex,
      );
      if (raw !== 0 && raw !== 1) {
        throw new Error(
          `Derivatives-book-depth availability value is invalid at row ${snapshotIndex}.`,
        );
      }
      const expected = (expectedMask & (1 << bandIndex)) !== 0;
      if ((raw === 1) !== expected) {
        throw new Error(
          `Derivatives-book-depth availability schema differs at row ${snapshotIndex}.`,
        );
      }
      return raw === 1;
    }) as unknown as DerivativesBookDepthAvailabilityVector;
    const matrices = Object.fromEntries(
      DERIVATIVES_BOOK_DEPTH_MATRIX_COLUMNS.map((name) => {
        const column = columns.get(name)!;
        const values = DERIVATIVES_BOOK_DEPTH_BANDS.map((_, bandIndex) => {
          const raw = bytes.readDoubleLE(
            column.offset
              + (snapshotIndex * DERIVATIVES_BOOK_DEPTH_BANDS.length + bandIndex) * 8,
          );
          if (!bandAvailable[bandIndex] && raw !== 0) {
            throw new Error(
              `Derivatives-book-depth unavailable ${name} is nonzero at row `
              + `${snapshotIndex}.`,
            );
          }
          return bandAvailable[bandIndex] ? raw : null;
        }) as unknown as DerivativesBookDepthValueVector;
        return [name, values];
      }),
    ) as Record<DerivativesBookDepthMatrixName, DerivativesBookDepthValueVector>;
    return {
      timestampOffsetSeconds,
      schemaBandCount,
      ...matrices,
      bandAvailable,
    };
  });
  snapshots.forEach((snapshot, index) => validateSnapshot(
    snapshot,
    index,
    index === 0 ? undefined : snapshots[index - 1]!.timestampOffsetSeconds,
  ));
  return snapshots;
}

export async function readDerivativesBookDepthShardReference(
  referenceFile: string,
): Promise<SequentialDerivativesBookDepthSnapshot[]> {
  const { reference, payload } = await readReferencedPayload(referenceFile);
  return decodeDerivativesBookDepth(reference, payload);
}

export function readDerivativesBookDepthShardReferenceSync(
  referenceFile: string,
): SequentialDerivativesBookDepthSnapshot[] {
  const { reference, payload } = readReferencedPayloadSync(referenceFile);
  return decodeDerivativesBookDepth(reference, payload);
}

function validateLayout(
  reference: SequentialShardReference,
  payloadBytes: number,
): DerivativesBookDepthLayout {
  if (reference.layout.encoding !== DERIVATIVES_BOOK_DEPTH_ENCODING) {
    throw new Error(
      `Unsupported derivatives-book-depth encoding: ${reference.layout.encoding}.`,
    );
  }
  const layout = reference.layout as DerivativesBookDepthLayout;
  validateUtcDayStart(layout.utcDayStartMs);
  const count = reference.sequence.count;
  const bandCount = DERIVATIVES_BOOK_DEPTH_BANDS.length;
  const expected = [
    {
      name: "timestampOffsetSeconds",
      encoding: "uint32-le",
      bytes: count * 4,
      shape: [count],
    },
    {
      name: "schemaBandCount",
      encoding: "uint8",
      bytes: count,
      shape: [count],
    },
    ...DERIVATIVES_BOOK_DEPTH_MATRIX_COLUMNS.map((name) => ({
      name,
      encoding: "float64-le",
      bytes: count * bandCount * 8,
      shape: [count, bandCount],
    })),
    {
      name: "bandAvailable",
      encoding: "uint8",
      bytes: count * bandCount,
      shape: [count, bandCount],
    },
  ] as const;
  let expectedOffset = 0;
  if (reference.sequence.start !== 0
    || reference.sequence.step !== 1
    || reference.sequence.unit !== "index"
    || count < 1
    || layout.timestampResolutionMs !== DERIVATIVES_BOOK_DEPTH_TIMESTAMP_RESOLUTION_MS
    || layout.timestampStorage !== "utc-day-second-offset"
    || layout.percentageSemantics !== "negative-bid-positive-ask"
    || layout.valuesAreCumulative !== true
    || layout.matrixOrder !== "snapshot-major-band-minor"
    || !Array.isArray(layout.bandPercentages)
    || layout.bandPercentages.length !== bandCount
    || layout.bandPercentages.some((band, index) => (
      band !== DERIVATIVES_BOOK_DEPTH_BANDS[index]
    ))
    || !Array.isArray(layout.columns)
    || layout.columns.length !== expected.length) {
    throw new Error("Derivatives-book-depth shard layout is invalid.");
  }
  expected.forEach((item, index) => {
    const column = layout.columns[index];
    if (column?.name !== item.name
      || column.encoding !== item.encoding
      || column.offset !== expectedOffset
      || column.bytes !== item.bytes
      || !sameShape(column.shape, item.shape)) {
      throw new Error("Derivatives-book-depth shard layout is invalid.");
    }
    expectedOffset += item.bytes;
  });
  if (payloadBytes !== expectedOffset) {
    throw new Error("Derivatives-book-depth shard layout is invalid.");
  }
  return layout;
}

function validateSnapshot(
  snapshot: SequentialDerivativesBookDepthSnapshot,
  index: number,
  previousTimestampOffsetSeconds?: number,
): void {
  if (!Number.isSafeInteger(snapshot.timestampOffsetSeconds)
    || snapshot.timestampOffsetSeconds < 0
    || snapshot.timestampOffsetSeconds >= DERIVATIVES_BOOK_DEPTH_DAY_SECONDS) {
    throw new Error(`Derivatives-book-depth timestamp offset is invalid at row ${index}.`);
  }
  if (previousTimestampOffsetSeconds !== undefined
    && snapshot.timestampOffsetSeconds <= previousTimestampOffsetSeconds) {
    throw new Error(`Derivatives-book-depth timestamps are not increasing at row ${index}.`);
  }
  if (snapshot.schemaBandCount !== 10 && snapshot.schemaBandCount !== 12) {
    throw new Error(`Derivatives-book-depth schema is invalid at row ${index}.`);
  }
  const expectedMask = availabilityMask(snapshot.schemaBandCount);
  if (snapshot.bandAvailable.length !== DERIVATIVES_BOOK_DEPTH_BANDS.length
    || snapshot.bandAvailable.some((available, bandIndex) => (
      typeof available !== "boolean"
      || available !== ((expectedMask & (1 << bandIndex)) !== 0)
    ))) {
    throw new Error(`Derivatives-book-depth availability schema differs at row ${index}.`);
  }

  for (const name of DERIVATIVES_BOOK_DEPTH_MATRIX_COLUMNS) {
    const values = snapshot[name];
    if (values.length !== DERIVATIVES_BOOK_DEPTH_BANDS.length) {
      throw new Error(`Derivatives-book-depth ${name} width differs at row ${index}.`);
    }
    values.forEach((value, bandIndex) => {
      if (snapshot.bandAvailable[bandIndex]) {
        if (!Number.isFinite(value) || value! <= 0) {
          throw new Error(
            `Derivatives-book-depth ${name} is missing or nonpositive at row ${index}.`,
          );
        }
      } else if (value !== null) {
        throw new Error(
          `Derivatives-book-depth unavailable ${name} has data at row ${index}.`,
        );
      }
    });
  }

  for (const name of DERIVATIVES_BOOK_DEPTH_MATRIX_COLUMNS) {
    let previous = -Infinity;
    snapshot[name].forEach((value, bandIndex) => {
      if (!snapshot.bandAvailable[bandIndex]) return;
      if (value! < previous) {
        throw new Error(
          `Derivatives-book-depth cumulative ${name} is non-monotone at row ${index}.`,
        );
      }
      previous = value!;
    });
  }
}

function validateUtcDayStart(value: number): void {
  if (!Number.isSafeInteger(value)
    || value % (DERIVATIVES_BOOK_DEPTH_DAY_SECONDS
      * DERIVATIVES_BOOK_DEPTH_TIMESTAMP_RESOLUTION_MS) !== 0) {
    throw new Error("Derivatives-book-depth UTC day start is invalid.");
  }
}

function availabilityMask(
  schemaBandCount: DerivativesBookDepthSchemaBandCount,
): number {
  return schemaBandCount === 12
    ? DERIVATIVES_BOOK_DEPTH_TWELVE_BAND_MASK
    : DERIVATIVES_BOOK_DEPTH_TEN_BAND_MASK;
}

function sameShape(
  actual: readonly number[],
  expected: readonly number[],
): boolean {
  return Array.isArray(actual)
    && actual.length === expected.length
    && actual.every((value, index) => value === expected[index]);
}
