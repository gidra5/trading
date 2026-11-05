import assert from "node:assert/strict";
import test from "node:test";
import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { gunzipSync } from "node:zlib";
import { RotatingJsonlSink } from "./live-jsonl-sink.ts";

test("gzip sink flushes complete members with session/order metadata and durable status", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "trading-sink-"));
  const sink = new RotatingJsonlSink(root, { fsync: true, sessionId: "test", sessionFiles: true });
  try {
    const at = Date.now();
    sink.write("venue/BTC", { sequence: 1 }, at); await sink.flush();
    sink.write("venue/BTC", { sequence: 2 }, at); await sink.close();
    const file = path.join(root, "venue/BTC", `${new Date(at).toISOString().slice(0, 10)}.test.jsonl.gz`);
    const rows = gunzipSync(await fs.readFile(file)).toString().trim().split("\n").map((line) => JSON.parse(line));
    assert.deepEqual(rows.map((row) => row.recordSequence), [1, 2]);
    assert.equal(rows[0].sessionId, "test");
    assert.equal(sink.health().pendingBytes, 0);
    assert.equal(sink.health().fsync, true);
    assert.ok(sink.health().lastCommittedAt);
  } finally { await fs.rm(root, { recursive: true, force: true }); }
});
test("write failure is surfaced once and the sink refuses further coverage", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "trading-sink-fail-"));
  let errors = 0;
  const sink = new RotatingJsonlSink(root, { onError: () => { errors++; } });
  try {
    sink.write("source", { value: 1 });
    await fs.rm(path.join(root, "source"), { recursive: true });
    await fs.writeFile(path.join(root, "source"), "not a directory");
    await assert.rejects(sink.flush());
    assert.equal(errors, 1);
    assert.throws(() => sink.write("source", { value: 2 }));
    assert.ok(sink.health().failure);
    await assert.rejects(sink.close());
  } finally { await fs.rm(root, { recursive: true, force: true }); }
});
test("backlog limits fail rather than silently dropping records", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "trading-sink-limit-"));
  const sink = new RotatingJsonlSink(root, { maxPendingBytes: 1 });
  try {
    assert.throws(() => sink.write("source", { value: 1 }), /backlog/);
    await assert.rejects(sink.close(), /backlog/);
  } finally { await fs.rm(root, { recursive: true, force: true }); }
});
