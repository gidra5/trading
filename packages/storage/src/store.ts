import { createHash, randomUUID } from "node:crypto";
import {
  constants as fsConstants,
  createReadStream,
  readFileSync,
} from "node:fs";
import fs from "node:fs/promises";
import path from "node:path";
import { promisify } from "node:util";
import {
  constants as zlibConstants,
  zstdCompress,
  zstdDecompressSync,
} from "node:zlib";
import type {
  PutSequentialShardRequest,
  PutSequentialShardResult,
  PutImmutableArtifactRequest,
  PutImmutableArtifactFileRequest,
  PutImmutableArtifactResult,
  ImmutableArtifactReference,
  SequentialShardReference,
  StorageReference,
  StorageAudit,
} from "./types.js";

const HASH_PATTERN = /^[a-f0-9]{64}$/;
const SAFE_SEGMENT = /^[a-zA-Z0-9][a-zA-Z0-9._=-]*$/;
const zstdCompressAsync = promisify(zstdCompress);

export class SequentialShardStore {
  readonly root: string;

  constructor(root: string) {
    this.root = path.resolve(root);
  }

  async put(request: PutSequentialShardRequest): Promise<PutSequentialShardResult> {
    validateSequence(request.sequence);
    validateLayout(request.layout);
    const namespace = storageIdentifier(request.namespace);
    const key = storageIdentifier(request.key);
    const payload = Buffer.from(
      request.payload.buffer,
      request.payload.byteOffset,
      request.payload.byteLength,
    );
    const contentHash = createHash("sha256").update(payload).digest("hex");
    const compressionLevel = request.compressionLevel ?? 9;
    if (!Number.isInteger(compressionLevel) || compressionLevel < -7 || compressionLevel > 22) {
      throw new Error(`Invalid Zstandard compression level: ${compressionLevel}.`);
    }
    const objectRelative = objectRelativePath(contentHash, "zstd");
    const objectFile = resolveInside(this.root, objectRelative);
    const objectCreated = await this.installObject(
      objectFile,
      payload,
      "zstd",
      compressionLevel,
    );
    const compressedBytes = (await fs.stat(objectFile)).size;
    const reference: SequentialShardReference = {
      version: 1,
      kind: "trading-sequential-shard",
      namespace,
      key,
      createdAt: new Date().toISOString(),
      object: {
        algorithm: "sha256",
        contentHash,
        file: slash(objectRelative),
        compression: "zstd",
        compressionLevel,
        uncompressedBytes: payload.byteLength,
        compressedBytes,
      },
      sequence: { ...request.sequence },
      layout: request.layout,
      ...(request.metadata ? { metadata: request.metadata } : {}),
    };
    const referenceFile = this.referenceFile(namespace, key);
    const existing = await optionalReference(referenceFile);
    if (existing && existing.object.contentHash !== contentHash) {
      throw new Error(
        `Immutable storage reference ${namespace}/${key} already points to `
        + `${existing.object.contentHash}; requested ${contentHash}.`,
      );
    }
    if (!existing) await writeJsonAtomic(referenceFile, reference);

    return {
      reference: existing ?? reference,
      referenceFile,
      objectFile,
      objectCreated,
    };
  }

  async putArtifact(
    request: PutImmutableArtifactRequest,
  ): Promise<PutImmutableArtifactResult> {
    const namespace = storageIdentifier(request.namespace);
    const key = storageIdentifier(request.key);
    if (!request.mediaType) throw new Error("Immutable artifact media type is required.");
    const payload = Buffer.from(
      request.payload.buffer,
      request.payload.byteOffset,
      request.payload.byteLength,
    );
    const contentHash = createHash("sha256").update(payload).digest("hex");
    const compression = request.compression ?? "zstd";
    const compressionLevel = request.compressionLevel ?? 9;
    const objectRelative = objectRelativePath(contentHash, compression);
    const objectFile = resolveInside(this.root, objectRelative);
    const objectCreated = await this.installObject(
      objectFile,
      payload,
      compression,
      compressionLevel,
    );
    const compressedBytes = (await fs.stat(objectFile)).size;
    const reference: ImmutableArtifactReference = {
      version: 1,
      kind: "trading-immutable-artifact",
      namespace,
      key,
      createdAt: new Date().toISOString(),
      object: {
        algorithm: "sha256",
        contentHash,
        file: slash(objectRelative),
        compression,
        ...(compression === "zstd" ? { compressionLevel } : {}),
        uncompressedBytes: payload.byteLength,
        compressedBytes,
      },
      mediaType: request.mediaType,
      ...(request.metadata ? { metadata: request.metadata } : {}),
    };
    const referenceFile = this.referenceFile(namespace, key);
    const existing = await optionalStoredReference(referenceFile);
    if (existing && existing.object.contentHash !== contentHash) {
      throw new Error(
        `Immutable storage reference ${namespace}/${key} already points to `
        + `${existing.object.contentHash}; requested ${contentHash}.`,
      );
    }
    if (existing && existing.kind !== reference.kind) {
      throw new Error(`Storage reference kind mismatch at ${namespace}/${key}.`);
    }
    if (!existing) await writeJsonAtomic(referenceFile, reference);
    return {
      reference: (existing ?? reference) as ImmutableArtifactReference,
      referenceFile,
      objectFile,
      objectCreated,
    };
  }

  async putArtifactFile(
    request: PutImmutableArtifactFileRequest,
  ): Promise<PutImmutableArtifactResult> {
    const namespace = storageIdentifier(request.namespace);
    const key = storageIdentifier(request.key);
    if (!request.mediaType) throw new Error("Immutable artifact media type is required.");
    const sourceFile = path.resolve(request.sourceFile);
    const sourceStat = await fs.stat(sourceFile);
    if (!sourceStat.isFile()) throw new Error(`Artifact source is not a file: ${sourceFile}.`);
    const contentHash = await hashFile(sourceFile);
    const objectRelative = objectRelativePath(contentHash, "none");
    const objectFile = resolveInside(this.root, objectRelative);
    const objectCreated = await this.installExistingFile(objectFile, sourceFile);
    const storedBytes = (await fs.stat(objectFile)).size;
    if (storedBytes !== sourceStat.size) {
      throw new Error(`Artifact object size changed while adopting ${sourceFile}.`);
    }
    const reference: ImmutableArtifactReference = {
      version: 1,
      kind: "trading-immutable-artifact",
      namespace,
      key,
      createdAt: new Date().toISOString(),
      object: {
        algorithm: "sha256",
        contentHash,
        file: slash(objectRelative),
        compression: "none",
        uncompressedBytes: storedBytes,
        compressedBytes: storedBytes,
      },
      mediaType: request.mediaType,
      ...(request.metadata ? { metadata: request.metadata } : {}),
    };
    const referenceFile = this.referenceFile(namespace, key);
    const existing = await optionalStoredReference(referenceFile);
    if (existing && (existing.kind !== reference.kind
      || existing.object.contentHash !== contentHash)) {
      throw new Error(`Immutable artifact reference changed: ${namespace}/${key}.`);
    }
    if (!existing) await writeJsonAtomic(referenceFile, reference);
    return {
      reference: (existing ?? reference) as ImmutableArtifactReference,
      referenceFile,
      objectFile,
      objectCreated,
    };
  }

  referenceFile(namespace: string, key: string): string {
    const parts = [
      ...storageIdentifier(namespace).split("/"),
      ...storageIdentifier(key).split("/"),
    ];
    const final = parts.pop()!;
    return resolveInside(this.root, path.join("refs", ...parts, `${final}.json`));
  }

  async readReference(namespace: string, key: string): Promise<SequentialShardReference> {
    return parseReference(
      JSON.parse(await fs.readFile(this.referenceFile(namespace, key), "utf8")),
    );
  }

  async readPayload(
    reference: StorageReference,
    verify = true,
  ): Promise<Buffer> {
    validateStoredReference(reference);
    const objectFile = resolveInside(this.root, reference.object.file);
    const compressed = await fs.readFile(objectFile);
    const payload = reference.object.compression === "zstd"
      ? reference.object.uncompressedBytes === 0
        ? zstdDecompressSync(compressed)
        : zstdDecompressSync(compressed, {
            maxOutputLength: reference.object.uncompressedBytes,
          })
      : compressed;
    if (payload.byteLength !== reference.object.uncompressedBytes) {
      throw new Error(
        `${objectFile} decoded to ${payload.byteLength} bytes; `
        + `expected ${reference.object.uncompressedBytes}.`,
      );
    }
    if (verify) {
      const actual = createHash("sha256").update(payload).digest("hex");
      if (actual !== reference.object.contentHash) {
        throw new Error(`${objectFile} failed its SHA-256 content check.`);
      }
    }
    return payload;
  }

  async audit(options: { externalReferenceRoots?: string[] } = {}): Promise<StorageAudit> {
    const referenceFiles = [...new Set((await Promise.all([
      path.join(this.root, "refs"),
      ...(options.externalReferenceRoots ?? []),
    ].map((root) => walkFiles(root, ".json")))).flat().map((file) => path.resolve(file)))];
    const objectFiles = await walkObjectFiles(path.join(this.root, "objects"));
    const temporaryFiles = await walkTemporaryFiles(this.root);
    const referenced = new Set<string>();
    const invalidReferences: Array<{ file: string; error: string }> = [];
    let referenceCount = 0;

    for (const file of referenceFiles) {
      try {
        const value = JSON.parse(await fs.readFile(file, "utf8")) as { kind?: unknown };
        if (value.kind !== "trading-sequential-shard"
          && value.kind !== "trading-immutable-artifact") continue;
        referenceCount += 1;
        const reference = parseStoredReference(value);
        const objectFile = resolveInside(this.root, reference.object.file);
        const stat = await optionalStat(objectFile);
        if (!stat) throw new Error(`Referenced object is missing: ${objectFile}`);
        if (stat.size !== reference.object.compressedBytes) {
          throw new Error(
            `Referenced object has ${stat.size} bytes; `
            + `expected ${reference.object.compressedBytes}: ${objectFile}`,
          );
        }
        referenced.add(objectFile.toLowerCase());
      } catch (error) {
        invalidReferences.push({
          file,
          error: error instanceof Error ? error.message : String(error),
        });
      }
    }

    let referencedBytes = 0;
    let orphanBytes = 0;
    let referencedObjects = 0;
    const orphanObjects: StorageAudit["orphanObjects"] = [];
    for (const file of objectFiles) {
      const stat = await fs.stat(file);
      if (referenced.has(path.resolve(file).toLowerCase())) {
        referencedObjects += 1;
        referencedBytes += stat.size;
      } else {
        orphanBytes += stat.size;
        orphanObjects.push({
          file,
          bytes: stat.size,
          modifiedAtMs: stat.mtimeMs,
        });
      }
    }
    return {
      references: referenceCount,
      objects: objectFiles.length,
      referencedObjects,
      orphanObjects,
      invalidReferences,
      referencedBytes,
      orphanBytes,
      temporaryFiles,
    };
  }

  async collectGarbage(options: {
    apply?: boolean;
    minimumAgeMs?: number;
    externalReferenceRoots?: string[];
  } = {}): Promise<StorageAudit & { removedObjects: number; removedBytes: number }> {
    const audit = await this.audit({
      externalReferenceRoots: options.externalReferenceRoots,
    });
    const cutoff = Date.now() - (options.minimumAgeMs ?? 24 * 60 * 60 * 1_000);
    const removable = audit.orphanObjects.filter((item) => item.modifiedAtMs <= cutoff);
    if (options.apply) {
      for (const item of removable) await fs.rm(item.file);
    }
    return {
      ...audit,
      removedObjects: options.apply ? removable.length : 0,
      removedBytes: options.apply
        ? removable.reduce((total, item) => total + item.bytes, 0)
        : 0,
    };
  }

  private async installObject(
    objectFile: string,
    payload: Buffer,
    compression: "zstd" | "none",
    compressionLevel: number,
  ): Promise<boolean> {
    if (await optionalStat(objectFile)) return false;
    await fs.mkdir(path.dirname(objectFile), { recursive: true });
    const compressed = compression === "zstd"
      ? await zstdCompressAsync(payload, {
          params: {
            [zlibConstants.ZSTD_c_compressionLevel]: compressionLevel,
          },
        })
      : payload;
    const temporary = `${objectFile}.${process.pid}-${randomUUID()}.tmp`;
    try {
      await fs.writeFile(temporary, compressed, { flag: "wx" });
      try {
        await fs.rename(temporary, objectFile);
        return true;
      } catch (error) {
        const code = (error as NodeJS.ErrnoException).code;
        if (code === "EEXIST" || (code === "EPERM" && await optionalStat(objectFile))) return false;
        throw error;
      }
    } finally {
      await fs.rm(temporary, { force: true });
    }
  }

  private async installExistingFile(objectFile: string, sourceFile: string): Promise<boolean> {
    if (await optionalStat(objectFile)) return false;
    await fs.mkdir(path.dirname(objectFile), { recursive: true });
    try {
      await fs.link(sourceFile, objectFile);
      return true;
    } catch (error) {
      const code = (error as NodeJS.ErrnoException).code;
      if (code === "EEXIST") return false;
      if (code !== "EXDEV" && code !== "EPERM") throw error;
    }
    const temporary = `${objectFile}.${process.pid}-${randomUUID()}.tmp`;
    try {
      await fs.copyFile(sourceFile, temporary, fsConstants.COPYFILE_EXCL);
      try {
        await fs.rename(temporary, objectFile);
        return true;
      } catch (error) {
        const code = (error as NodeJS.ErrnoException).code;
        if (code === "EEXIST" || (code === "EPERM" && await optionalStat(objectFile))) return false;
        throw error;
      }
    } finally {
      await fs.rm(temporary, { force: true });
    }
  }
}

async function hashFile(file: string): Promise<string> {
  const digest = createHash("sha256");
  for await (const chunk of createReadStream(file)) digest.update(chunk as Buffer);
  return digest.digest("hex");
}

export async function resolveReferenceFile(referenceFile: string): Promise<{
  reference: SequentialShardReference;
  referenceFile: string;
  storageRoot: string;
}> {
  const current = path.resolve(referenceFile);
  const reference = parseReference(JSON.parse(await fs.readFile(current, "utf8")));
  return {
    reference,
    referenceFile: current,
    storageRoot: storageRootForReference(current),
  };
}

export async function readReferencedPayload(
  referenceFile: string,
  verify = true,
): Promise<{ reference: SequentialShardReference; payload: Buffer }> {
  const resolved = await resolveReferenceFile(referenceFile);
  const store = new SequentialShardStore(resolved.storageRoot);
  return {
    reference: resolved.reference,
    payload: await store.readPayload(resolved.reference, verify),
  };
}

export function readReferencedPayloadSync(
  referenceFile: string,
  verify = true,
): { reference: SequentialShardReference; payload: Buffer } {
  const current = path.resolve(referenceFile);
  const reference = parseReference(JSON.parse(readFileSync(current, "utf8")));
  const storageRoot = storageRootForReference(current);
  const objectFile = resolveInside(storageRoot, reference.object.file);
  const stored = readFileSync(objectFile);
  const payload = reference.object.uncompressedBytes === 0
    ? zstdDecompressSync(stored)
    : zstdDecompressSync(stored, {
        maxOutputLength: reference.object.uncompressedBytes,
      });
  if (payload.byteLength !== reference.object.uncompressedBytes) {
    throw new Error(`${objectFile} decoded to an unexpected size.`);
  }
  if (verify && createHash("sha256").update(payload).digest("hex")
    !== reference.object.contentHash) {
    throw new Error(`${objectFile} failed its SHA-256 content check.`);
  }
  return { reference, payload };
}

function storageRootForReference(referenceFile: string): string {
  let cursor = path.dirname(path.resolve(referenceFile));
  while (path.dirname(cursor) !== cursor) {
    if (path.basename(cursor).toLowerCase() === "refs") return path.dirname(cursor);
    if (path.basename(cursor).toLowerCase() === "datasets"
      && path.basename(path.dirname(cursor)).toLowerCase() === "training") {
      return path.join(path.dirname(cursor), "immutable");
    }
    cursor = path.dirname(cursor);
  }
  throw new Error(`Canonical storage reference is not inside a refs directory: ${referenceFile}`);
}

function objectRelativePath(hash: string, compression: "zstd" | "none"): string {
  if (!HASH_PATTERN.test(hash)) throw new Error(`Invalid SHA-256 hash: ${hash}.`);
  return path.join(
    "objects",
    "sha256",
    hash.slice(0, 2),
    `${hash}.${compression === "zstd" ? "zst" : "bin"}`,
  );
}

function storageIdentifier(value: string): string {
  const normalized = value.replaceAll("\\", "/").replace(/^\/+|\/+$/g, "");
  const segments = normalized.split("/");
  if (segments.length === 0 || segments.some((segment) => !SAFE_SEGMENT.test(segment))) {
    throw new Error(`Unsafe storage identifier: ${value}.`);
  }
  return segments.join("/");
}

function validateSequence(sequence: PutSequentialShardRequest["sequence"]): void {
  if (!Number.isSafeInteger(sequence.start)
    || !Number.isSafeInteger(sequence.step)
    || sequence.step < 1
    || !Number.isSafeInteger(sequence.count)
    || sequence.count < 0
    || (sequence.unit !== "unix-ms" && sequence.unit !== "index")) {
    throw new Error("Sequential shard axis is invalid.");
  }
  if (sequence.count === 0) return;
  const end = sequence.start + (sequence.count - 1) * sequence.step;
  if (!Number.isSafeInteger(end)) throw new Error("Sequential shard axis exceeds safe integers.");
}

function validateLayout(layout: PutSequentialShardRequest["layout"]): void {
  if (!layout || typeof layout.encoding !== "string" || !layout.encoding) {
    throw new Error("Sequential shard layout requires an encoding.");
  }
}

function validateReference(reference: SequentialShardReference): void {
  if (reference.version !== 1
    || reference.kind !== "trading-sequential-shard"
    || reference.object.algorithm !== "sha256"
    || !HASH_PATTERN.test(reference.object.contentHash)
    || reference.object.compression !== "zstd"
    || reference.object.uncompressedBytes < 0
    || reference.object.compressedBytes < 0) {
    throw new Error("Invalid sequential shard reference.");
  }
  if (slash(reference.object.file) !== slash(objectRelativePath(
    reference.object.contentHash,
    reference.object.compression,
  ))) {
    throw new Error("Sequential shard object path does not match its content hash.");
  }
  storageIdentifier(reference.namespace);
  storageIdentifier(reference.key);
  validateSequence(reference.sequence);
  validateLayout(reference.layout);
}

function validateStoredReference(reference: StorageReference): void {
  if (reference.kind === "trading-sequential-shard") {
    validateReference(reference);
    return;
  }
  if (reference.version !== 1
    || reference.kind !== "trading-immutable-artifact"
    || reference.object.algorithm !== "sha256"
    || !HASH_PATTERN.test(reference.object.contentHash)
    || (reference.object.compression !== "zstd" && reference.object.compression !== "none")
    || reference.object.uncompressedBytes < 0
    || reference.object.compressedBytes < 0
    || typeof reference.mediaType !== "string"
    || !reference.mediaType) {
    throw new Error("Invalid immutable artifact reference.");
  }
  if (slash(reference.object.file) !== slash(objectRelativePath(
    reference.object.contentHash,
    reference.object.compression,
  ))) {
    throw new Error("Immutable artifact object path does not match its content hash.");
  }
  storageIdentifier(reference.namespace);
  storageIdentifier(reference.key);
}

function parseReference(value: unknown): SequentialShardReference {
  if (!value || typeof value !== "object") throw new Error("Storage reference is not an object.");
  const reference = value as SequentialShardReference;
  validateReference(reference);
  return reference;
}

function parseStoredReference(value: unknown): StorageReference {
  if (!value || typeof value !== "object") throw new Error("Storage reference is not an object.");
  const reference = value as StorageReference;
  validateStoredReference(reference);
  return reference;
}

function resolveInside(root: string, relative: string): string {
  const resolvedRoot = path.resolve(root);
  const resolved = path.resolve(resolvedRoot, relative);
  const prefix = resolvedRoot.endsWith(path.sep) ? resolvedRoot : `${resolvedRoot}${path.sep}`;
  if (resolved !== resolvedRoot && !resolved.toLowerCase().startsWith(prefix.toLowerCase())) {
    throw new Error(`Storage path escapes its root: ${relative}.`);
  }
  return resolved;
}

async function optionalReference(file: string): Promise<SequentialShardReference | undefined> {
  try {
    return parseReference(JSON.parse(await fs.readFile(file, "utf8")));
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return undefined;
    throw error;
  }
}

async function optionalStoredReference(file: string): Promise<StorageReference | undefined> {
  try {
    return parseStoredReference(JSON.parse(await fs.readFile(file, "utf8")));
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return undefined;
    throw error;
  }
}

async function optionalStat(file: string): Promise<import("node:fs").Stats | undefined> {
  try {
    return await fs.stat(file);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return undefined;
    throw error;
  }
}

async function writeJsonAtomic(file: string, value: unknown): Promise<void> {
  await fs.mkdir(path.dirname(file), { recursive: true });
  const temporary = `${file}.${process.pid}-${randomUUID()}.tmp`;
  try {
    await fs.writeFile(temporary, `${JSON.stringify(value, null, 2)}\n`, { flag: "wx" });
    await fs.rename(temporary, file);
  } finally {
    await fs.rm(temporary, { force: true });
  }
}

async function walkFiles(root: string, suffix: string): Promise<string[]> {
  const files: string[] = [];
  async function walk(directory: string): Promise<void> {
    let entries: Array<import("node:fs").Dirent>;
    try {
      entries = await fs.readdir(directory, { withFileTypes: true });
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") return;
      throw error;
    }
    for (const entry of entries) {
      const file = path.join(directory, entry.name);
      if (entry.isDirectory()) await walk(file);
      else if (entry.isFile() && entry.name.endsWith(suffix)) files.push(file);
    }
  }
  await walk(root);
  return files;
}

async function walkObjectFiles(root: string): Promise<string[]> {
  const files = await walkFiles(root, "");
  return files.filter((file) => file.endsWith(".zst") || file.endsWith(".bin"));
}

async function walkTemporaryFiles(root: string): Promise<StorageAudit["temporaryFiles"]> {
  const files: StorageAudit["temporaryFiles"] = [];
  async function walk(directory: string): Promise<void> {
    let entries: Array<import("node:fs").Dirent>;
    try {
      entries = await fs.readdir(directory, { withFileTypes: true });
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") return;
      throw error;
    }
    for (const entry of entries) {
      const file = path.join(directory, entry.name);
      if (entry.isDirectory()) await walk(file);
      else if (entry.isFile() && /\.(?:tmp|old)$/.test(entry.name)) {
        const stat = await fs.stat(file);
        files.push({ file, bytes: stat.size, modifiedAtMs: stat.mtimeMs });
      }
    }
  }
  await walk(root);
  return files;
}

function slash(value: string): string {
  return value.replaceAll(path.sep, "/");
}
