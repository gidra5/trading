import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import {
  pruneTrainingOrphans,
  trainingStorageLayout,
} from "./training-storage.mjs";

const HOUR_MS = 60 * 60 * 1_000;

test("checkpoint GC scans only canonical pointers and checkpoint objects", () => {
  const { layout, cleanup } = fixture();
  try {
    const protectedHash = "11".repeat(32);
    const orphanHash = "22".repeat(32);
    const freshHash = "33".repeat(32);
    const sequentialHash = "44".repeat(32);
    const protectedObject = writeObject(layout, protectedHash, ".bin");
    const orphanObject = writeObject(layout, orphanHash, ".bin");
    const freshObject = writeObject(layout, freshHash, ".bin");
    const sequentialObject = writeObject(layout, sequentialHash, ".zst");
    const old = new Date(Date.now() - 2 * HOUR_MS);
    const fresh = new Date(Date.now() - HOUR_MS / 2);
    for (const file of [protectedObject, orphanObject, sequentialObject]) {
      fs.utimesSync(file, old, old);
    }
    fs.utimesSync(freshObject, fresh, fresh);

    const pointer = path.join(
      layout.runs,
      "disposable-smoke",
      "nested-run",
      "checkpoints",
      "best.json",
    );
    writeJson(pointer, checkpointReference(protectedHash));

    const unrelatedFiles = [
      path.join(layout.immutable, "refs", "features", "day.json"),
      path.join(layout.datasets, "dataset.json"),
      path.join(layout.runs, "disposable-smoke", "status.json"),
    ];
    for (const file of unrelatedFiles) {
      // If any arbitrary JSON tree were still treated as a reference, this
      // otherwise valid pointer would incorrectly retain the stale orphan.
      writeJson(file, checkpointReference(orphanHash));
    }

    const reads = [];
    const originalReadFileSync = fs.readFileSync;
    fs.readFileSync = function monitoredReadFileSync(file, ...args) {
      reads.push(path.resolve(file));
      return originalReadFileSync.call(this, file, ...args);
    };
    let result;
    try {
      result = pruneTrainingOrphans(layout, { minimumAgeMs: HOUR_MS });
    } finally {
      fs.readFileSync = originalReadFileSync;
    }

    assert.deepEqual(reads, [path.resolve(pointer)]);
    assert.deepEqual(result, {
      files: 1,
      bytes: 1,
      invalidReferences: 0,
      skipped: false,
    });
    assert.equal(fs.existsSync(protectedObject), true);
    assert.equal(fs.existsSync(orphanObject), false);
    assert.equal(fs.existsSync(freshObject), true);
    assert.equal(fs.existsSync(sequentialObject), true);
  } finally {
    cleanup();
  }
});

test("checkpoint GC aborts before deletion on an invalid object path", () => {
  const { layout, cleanup } = fixture();
  try {
    const orphanHash = "55".repeat(32);
    const orphanObject = writeObject(layout, orphanHash, ".bin");
    const old = new Date(Date.now() - 2 * HOUR_MS);
    fs.utimesSync(orphanObject, old, old);

    const invalid = checkpointReference("66".repeat(32));
    invalid.object.file = "../../escape.bin";
    writeJson(
      path.join(layout.runs, "nested", "checkpoints", "last.json"),
      invalid,
    );

    assert.deepEqual(
      pruneTrainingOrphans(layout, { minimumAgeMs: HOUR_MS }),
      { files: 0, bytes: 0, invalidReferences: 1, skipped: true },
    );
    assert.equal(fs.existsSync(orphanObject), true);
  } finally {
    cleanup();
  }
});

test("checkpoint GC retains nested selection checkpoint objects", () => {
  const { layout, cleanup } = fixture();
  try {
    const selectionHash = "77".repeat(32);
    const selectionObject = writeObject(layout, selectionHash, ".bin");
    const old = new Date(Date.now() - 2 * HOUR_MS);
    fs.utimesSync(selectionObject, old, old);
    writeJson(
      path.join(
        layout.runs,
        "density-run",
        "checkpoints",
        "selections",
        "validation-mse.json",
      ),
      checkpointReference(selectionHash),
    );

    assert.deepEqual(
      pruneTrainingOrphans(layout, { minimumAgeMs: HOUR_MS }),
      { files: 0, bytes: 0, invalidReferences: 0, skipped: false },
    );
    assert.equal(fs.existsSync(selectionObject), true);
  } finally {
    cleanup();
  }
});

function fixture() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "training-storage-"));
  return {
    layout: trainingStorageLayout(root),
    cleanup: () => fs.rmSync(root, { recursive: true, force: true }),
  };
}

function writeObject(layout, contentHash, extension) {
  const file = path.join(
    layout.immutable,
    "objects",
    "sha256",
    contentHash.slice(0, 2),
    `${contentHash}${extension}`,
  );
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, "x");
  return file;
}

function writeJson(file, value) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, JSON.stringify(value), "utf8");
}

function checkpointReference(contentHash) {
  return {
    version: 1,
    kind: "trading-immutable-artifact",
    namespace: "training/checkpoints",
    key: "nested/checkpoints/best.json",
    createdAt: "2026-01-01T00:00:00.000Z",
    object: {
      algorithm: "sha256",
      contentHash,
      file: `objects/sha256/${contentHash.slice(0, 2)}/${contentHash}.bin`,
      compression: "none",
      uncompressedBytes: 1,
      compressedBytes: 1,
    },
    mediaType: "application/x-pytorch-checkpoint",
  };
}
