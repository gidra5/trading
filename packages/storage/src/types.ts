export interface StorageMetadata {
  [key: string]: unknown;
}

export interface SequentialAxis {
  /** Timestamp or sequence number represented by row zero. */
  start: number;
  /** Default distance between adjacent rows; a codec may record sparse exceptions. */
  step: number;
  count: number;
  unit: "unix-ms" | "index";
}

export interface ShardLayout extends StorageMetadata {
  encoding: string;
}

export interface StoredObject {
  algorithm: "sha256";
  /** Hash of the uncompressed canonical payload. */
  contentHash: string;
  file: string;
  compression: "zstd" | "none";
  compressionLevel?: number;
  uncompressedBytes: number;
  compressedBytes: number;
}

export interface SequentialShardReference {
  version: 1;
  kind: "trading-sequential-shard";
  namespace: string;
  key: string;
  createdAt: string;
  object: StoredObject;
  sequence: SequentialAxis;
  layout: ShardLayout;
  metadata?: StorageMetadata;
}

export interface ImmutableArtifactReference {
  version: 1;
  kind: "trading-immutable-artifact";
  namespace: string;
  key: string;
  createdAt: string;
  object: StoredObject;
  mediaType: string;
  metadata?: StorageMetadata;
}

export type StorageReference = SequentialShardReference | ImmutableArtifactReference;

export interface PutImmutableArtifactRequest {
  namespace: string;
  key: string;
  payload: Uint8Array;
  mediaType: string;
  metadata?: StorageMetadata;
  compression?: "zstd" | "none";
  compressionLevel?: number;
}

export interface PutImmutableArtifactFileRequest {
  namespace: string;
  key: string;
  sourceFile: string;
  mediaType: string;
  metadata?: StorageMetadata;
}

export interface PutSequentialShardRequest {
  namespace: string;
  key: string;
  payload: Uint8Array;
  sequence: SequentialAxis;
  layout: ShardLayout;
  metadata?: StorageMetadata;
  compressionLevel?: number;
}

export interface PutSequentialShardResult {
  reference: SequentialShardReference;
  referenceFile: string;
  objectFile: string;
  objectCreated: boolean;
}

export interface PutImmutableArtifactResult {
  reference: ImmutableArtifactReference;
  referenceFile: string;
  objectFile: string;
  objectCreated: boolean;
}

export interface StorageAudit {
  references: number;
  objects: number;
  referencedObjects: number;
  orphanObjects: Array<{
    file: string;
    bytes: number;
    modifiedAtMs: number;
  }>;
  invalidReferences: Array<{ file: string; error: string }>;
  referencedBytes: number;
  orphanBytes: number;
  temporaryFiles: Array<{ file: string; bytes: number; modifiedAtMs: number }>;
}
