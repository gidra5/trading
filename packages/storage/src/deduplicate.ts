import { createHash, randomUUID } from "node:crypto";
import { createReadStream } from "node:fs";
import fs from "node:fs/promises";
import path from "node:path";

export interface HardLinkDeduplicationOptions {
  root: string;
  apply?: boolean;
  minimumBytes?: number;
  hashConcurrency?: number;
  onProgress?: (progress: { hashedObjects: number; candidateObjects: number }) => void;
}

export interface HardLinkDeduplicationReport {
  mode: "dry-run" | "applied";
  root: string;
  scannedFiles: number;
  scannedBytes: number;
  uniqueObjects: number;
  candidateObjects: number;
  hashedObjects: number;
  duplicateObjects: number;
  duplicatePaths: number;
  duplicateBytes: number;
  linkedPaths: number;
  reclaimedBytes: number;
  skippedObjects: number;
  failures: Array<{ file: string; error: string }>;
}

interface FileRecord {
  file: string;
  basename: string;
  identity: string;
  size: number;
  links: number;
  modifiedNs: bigint;
}

interface FileObject {
  identity: string;
  size: number;
  links: number;
  paths: FileRecord[];
  digest?: string;
}

/**
 * Replaces byte-identical files with hard links after a full SHA-256 check.
 * Candidate matching uses basename and size so unrelated equal-sized tensors
 * do not need to be read. Existing hard links are recognized by file identity.
 */
export async function deduplicateFilesWithHardLinks(
  options: HardLinkDeduplicationOptions,
): Promise<HardLinkDeduplicationReport> {
  const root = path.resolve(options.root);
  const minimumBytes = options.minimumBytes ?? 64 * 1_024;
  const hashConcurrency = options.hashConcurrency ?? 2;
  if (!Number.isSafeInteger(minimumBytes) || minimumBytes < 0) {
    throw new Error("Deduplication minimum bytes must be a non-negative integer.");
  }
  if (!Number.isSafeInteger(hashConcurrency) || hashConcurrency < 1 || hashConcurrency > 16) {
    throw new Error("Deduplication hash concurrency must be between 1 and 16.");
  }

  const files = await scanFiles(root);
  const objects = fileObjects(files);
  const candidates = candidateGroups(objects, minimumBytes);
  const candidateObjects = [...new Map(
    candidates.flat().map((object) => [object.identity, object]),
  ).values()];
  let hashedObjects = 0;
  await mapLimit(candidateObjects, hashConcurrency, async (object) => {
    object.digest = await stableHash(object.paths[0]!);
    hashedObjects += 1;
    options.onProgress?.({ hashedObjects, candidateObjects: candidateObjects.length });
  });

  const duplicates = exactDuplicateGroups(candidateObjects);
  const report: HardLinkDeduplicationReport = {
    mode: options.apply ? "applied" : "dry-run",
    root,
    scannedFiles: files.length,
    scannedBytes: files.reduce((total, file) => total + file.size, 0),
    uniqueObjects: objects.length,
    candidateObjects: candidateObjects.length,
    hashedObjects,
    duplicateObjects: 0,
    duplicatePaths: 0,
    duplicateBytes: 0,
    linkedPaths: 0,
    reclaimedBytes: 0,
    skippedObjects: 0,
    failures: [],
  };

  for (const group of duplicates) {
    const canonical = [...group].sort(compareCanonical)[0]!;
    for (const duplicate of group) {
      if (duplicate === canonical) continue;
      report.duplicateObjects += 1;
      report.duplicatePaths += duplicate.paths.length;
      report.duplicateBytes += duplicate.size;
      if (!options.apply) continue;

      let linked = 0;
      for (const target of duplicate.paths) {
        try {
          await replaceWithHardLink(canonical.paths[0]!, target);
          linked += 1;
          report.linkedPaths += 1;
        } catch (error) {
          report.failures.push({
            file: target.file,
            error: error instanceof Error ? error.message : String(error),
          });
        }
      }
      if (linked === duplicate.paths.length && duplicate.links <= duplicate.paths.length) {
        report.reclaimedBytes += duplicate.size;
      } else {
        report.skippedObjects += 1;
      }
    }
  }
  return report;
}

async function scanFiles(root: string): Promise<FileRecord[]> {
  const records: FileRecord[] = [];
  const pending = [root];
  while (pending.length > 0) {
    const directory = pending.pop()!;
    let entries: Array<import("node:fs").Dirent>;
    try {
      entries = await fs.readdir(directory, { withFileTypes: true });
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT" && directory === root) return [];
      throw error;
    }
    for (const entry of entries) {
      const file = path.join(directory, entry.name);
      if (entry.isDirectory()) {
        pending.push(file);
      } else if (entry.isFile() && !entry.name.includes(".dedupe-link-")) {
        const stat = await fs.stat(file, { bigint: true });
        records.push({
          file,
          basename: entry.name.toLowerCase(),
          identity: fileIdentity(stat),
          size: safeNumber(stat.size, `file size for ${file}`),
          links: safeNumber(stat.nlink, `link count for ${file}`),
          modifiedNs: stat.mtimeNs,
        });
      }
    }
  }
  return records;
}

function fileObjects(files: readonly FileRecord[]): FileObject[] {
  const byIdentity = new Map<string, FileObject>();
  for (const file of files) {
    const existing = byIdentity.get(file.identity);
    if (existing) {
      existing.paths.push(file);
    } else {
      byIdentity.set(file.identity, {
        identity: file.identity,
        size: file.size,
        links: file.links,
        paths: [file],
      });
    }
  }
  return [...byIdentity.values()];
}

function candidateGroups(
  objects: readonly FileObject[],
  minimumBytes: number,
): FileObject[][] {
  const byNameAndSize = new Map<string, FileObject[]>();
  for (const object of objects) {
    if (object.size < minimumBytes) continue;
    for (const basename of new Set(object.paths.map((record) => record.basename))) {
      const key = `${basename}\0${object.size}`;
      const group = byNameAndSize.get(key) ?? [];
      group.push(object);
      byNameAndSize.set(key, group);
    }
  }
  return [...byNameAndSize.values()].filter((group) => group.length > 1);
}

function exactDuplicateGroups(objects: readonly FileObject[]): FileObject[][] {
  const bySizeAndDigest = new Map<string, FileObject[]>();
  for (const object of objects) {
    const key = `${object.size}\0${object.digest!}`;
    const group = bySizeAndDigest.get(key) ?? [];
    group.push(object);
    bySizeAndDigest.set(key, group);
  }
  return [...bySizeAndDigest.values()].filter((group) => group.length > 1);
}

function compareCanonical(left: FileObject, right: FileObject): number {
  return right.paths.length - left.paths.length
    || right.links - left.links
    || left.paths[0]!.file.localeCompare(right.paths[0]!.file);
}

async function stableHash(record: FileRecord): Promise<string> {
  const digest = createHash("sha256");
  for await (const chunk of createReadStream(record.file)) digest.update(chunk as Buffer);
  const after = await fs.stat(record.file, { bigint: true });
  if (fileIdentity(after) !== record.identity
    || safeNumber(after.size, `file size for ${record.file}`) !== record.size
    || after.mtimeNs !== record.modifiedNs) {
    throw new Error(`File changed while hashing: ${record.file}.`);
  }
  return digest.digest("hex");
}

async function replaceWithHardLink(canonical: FileRecord, target: FileRecord): Promise<void> {
  const [canonicalStat, targetStat] = await Promise.all([
    fs.stat(canonical.file, { bigint: true }),
    fs.stat(target.file, { bigint: true }),
  ]);
  if (fileIdentity(canonicalStat) !== canonical.identity
    || fileIdentity(targetStat) !== target.identity
    || canonicalStat.size !== targetStat.size
    || targetStat.mtimeNs !== target.modifiedNs) {
    throw new Error("Canonical or duplicate file changed after verification.");
  }
  const temporary = path.join(
    path.dirname(target.file),
    `.${path.basename(target.file)}.dedupe-link-${process.pid}-${randomUUID()}`,
  );
  try {
    await fs.link(canonical.file, temporary);
    await fs.rename(temporary, target.file);
    const installed = await fs.stat(target.file, { bigint: true });
    if (fileIdentity(installed) !== canonical.identity) {
      throw new Error("Hard-link replacement did not retain the canonical file identity.");
    }
  } finally {
    await fs.rm(temporary, { force: true });
  }
}

function fileIdentity(stat: import("node:fs").BigIntStats): string {
  return `${stat.dev}:${stat.ino}`;
}

function safeNumber(value: bigint, label: string): number {
  const result = Number(value);
  if (!Number.isSafeInteger(result)) throw new Error(`Unsafe ${label}: ${value}.`);
  return result;
}

async function mapLimit<T>(
  values: readonly T[],
  concurrency: number,
  worker: (value: T) => Promise<void>,
): Promise<void> {
  let next = 0;
  await Promise.all(Array.from({ length: Math.min(concurrency, values.length) }, async () => {
    while (next < values.length) {
      const index = next;
      next += 1;
      await worker(values[index]!);
    }
  }));
}
