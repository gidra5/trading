import fs from "node:fs";
import path from "node:path";

const GIB = 1024 ** 3;

export function trainingStorageLayout(repoRoot) {
  const root = path.join(repoRoot, "data", "training");
  return {
    root,
    immutable: path.join(root, "immutable"),
    datasets: path.join(root, "datasets"),
    runs: path.join(root, "runs"),
    cache: path.join(root, "cache"),
    analysis: path.join(root, "analysis"),
  };
}

export function prepareTrainingCache(repoRoot) {
  const layout = trainingStorageLayout(repoRoot);
  const directories = {
    temporary: path.join(layout.cache, "tmp"),
    triton: path.join(layout.cache, "triton"),
    torchinductor: path.join(layout.cache, "torchinductor"),
    cuda: path.join(layout.cache, "cuda"),
  };
  for (const directory of Object.values(directories)) {
    fs.mkdirSync(directory, { recursive: true });
  }
  const cachePruned = pruneCache(layout.cache, {
    maximumBytes: positiveGiB("TRADING_ML_CACHE_MAX_GIB", 12) * GIB,
    reserveBytes: positiveGiB("TRADING_DISK_RESERVE_GIB", 32) * GIB,
  });
  const orphanPruned = pruneTrainingOrphans(layout, {
    minimumAgeMs: positiveHours("TRADING_STORAGE_ORPHAN_GRACE_HOURS", 1)
      * 60 * 60 * 1_000,
  });
  return { ...directories, maintenance: { cachePruned, orphanPruned } };
}

export function assertInside(candidate, root, label) {
  const resolved = path.resolve(candidate);
  const resolvedRoot = path.resolve(root);
  const relative = path.relative(resolvedRoot, resolved);
  if (!relative || relative.startsWith("..") || path.isAbsolute(relative)) {
    throw new Error(`${label} must be a child of ${resolvedRoot}: ${resolved}`);
  }
  return resolved;
}

function pruneCache(root, { maximumBytes, reserveBytes }) {
  const files = walkFiles(root);
  let totalBytes = files.reduce((total, file) => total + file.bytes, 0);
  const availableBytes = Number(fs.statfsSync(root).bavail) * Number(fs.statfsSync(root).bsize);
  const requiredBytes = Math.max(
    0,
    totalBytes - maximumBytes,
    reserveBytes - availableBytes,
  );
  if (requiredBytes === 0) return { files: 0, bytes: 0 };
  let removedFiles = 0;
  let removedBytes = 0;
  for (const file of files.sort((left, right) => left.modifiedAtMs - right.modifiedAtMs)) {
    try {
      fs.rmSync(file.path, { force: true });
      removedFiles += 1;
      removedBytes += file.bytes;
      totalBytes -= file.bytes;
    } catch (error) {
      if (error?.code !== "ENOENT") throw error;
    }
    if (removedBytes >= requiredBytes && totalBytes <= maximumBytes) break;
  }
  return { files: removedFiles, bytes: removedBytes };
}

export function pruneTrainingOrphans(layout, { minimumAgeMs }) {
  const referenced = new Set();
  let invalidReferences = 0;
  for (const root of [
    path.join(layout.immutable, "refs"),
    layout.datasets,
    layout.runs,
  ]) {
    for (const file of walkFilesIfPresent(root)) {
      if (!file.path.endsWith(".json")) continue;
      let value;
      try {
        value = JSON.parse(fs.readFileSync(file.path, "utf8"));
      } catch {
        continue;
      }
      if (value?.kind !== "trading-sequential-shard"
        && value?.kind !== "trading-immutable-artifact") continue;
      const relative = value?.object?.file;
      if (typeof relative !== "string") {
        invalidReferences += 1;
        continue;
      }
      const object = path.resolve(layout.immutable, relative);
      const inside = path.relative(layout.immutable, object);
      if (!inside || inside.startsWith("..") || path.isAbsolute(inside)) {
        invalidReferences += 1;
        continue;
      }
      referenced.add(object.toLowerCase());
    }
  }
  if (invalidReferences > 0) {
    return { files: 0, bytes: 0, invalidReferences, skipped: true };
  }
  const cutoff = Date.now() - minimumAgeMs;
  let removedFiles = 0;
  let removedBytes = 0;
  for (const file of walkFilesIfPresent(path.join(layout.immutable, "objects"))) {
    if (referenced.has(path.resolve(file.path).toLowerCase())
      || file.modifiedAtMs > cutoff) continue;
    try {
      fs.rmSync(file.path, { force: true });
      removedFiles += 1;
      removedBytes += file.bytes;
    } catch (error) {
      if (error?.code !== "ENOENT") throw error;
    }
  }
  return { files: removedFiles, bytes: removedBytes, invalidReferences, skipped: false };
}

function walkFiles(root) {
  const result = [];
  const pending = [root];
  while (pending.length > 0) {
    const directory = pending.pop();
    for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
      const file = path.join(directory, entry.name);
      if (entry.isDirectory()) pending.push(file);
      else if (entry.isFile()) {
        const stat = fs.statSync(file);
        result.push({ path: file, bytes: stat.size, modifiedAtMs: stat.mtimeMs });
      }
    }
  }
  return result;
}

function walkFilesIfPresent(root) {
  return fs.existsSync(root) ? walkFiles(root) : [];
}

function positiveGiB(name, fallback) {
  const value = Number(process.env[name] ?? fallback);
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`${name} must be a positive GiB value.`);
  }
  return value;
}

function positiveHours(name, fallback) {
  const value = Number(process.env[name] ?? fallback);
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`${name} must be a positive hour value.`);
  }
  return value;
}
