import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import fs from "node:fs/promises";
import { createServer } from "node:http";
import os from "node:os";
import path from "node:path";
import { test } from "node:test";
import AdmZip from "adm-zip";
import { fetchBinanceSpotDailyShard } from "../src/binance-history-cache.js";

const DAY_MS = 86_400_000;
const DATE = "2026-01-01";
const DAY = Date.parse(`${DATE}T00:00:00.000Z`);
const INTERVAL = "6h";
const INTERVAL_MS = 21_600_000;

test("daily shard recovery installs a complete checksum-verified archive", async () => {
  const archive = archiveWithTimes([
    DAY,
    DAY + INTERVAL_MS,
    DAY + 2 * INTERVAL_MS,
    DAY + 3 * INTERVAL_MS,
  ]);
  const source = await serveArchive(archive);
  const dataDir = await fs.mkdtemp(path.join(os.tmpdir(), "trading-history-recovery-"));
  try {
    await fetchBinanceSpotDailyShard({
      dataDir,
      date: DATE,
      day: DAY,
      interval: INTERVAL,
      intervalMs: INTERVAL_MS,
      archiveRoot: source.root,
    });
    const target = path.join(
      dataDir,
      "historical/spot-btcusdt/btcusdt",
      INTERVAL,
      `${DATE}.jsonl.gz`,
    );
    const compressed = await fs.readFile(target);
    assert.ok(compressed.length > 0);
  } finally {
    await source.close();
    await fs.rm(dataDir, { recursive: true, force: true });
  }
});

test("daily shard recovery rejects an archive with an internal candle gap", async () => {
  const archive = archiveWithTimes([
    DAY,
    DAY + INTERVAL_MS,
    DAY + 3 * INTERVAL_MS,
  ]);
  const source = await serveArchive(archive);
  const dataDir = await fs.mkdtemp(path.join(os.tmpdir(), "trading-history-gap-"));
  try {
    await assert.rejects(
      fetchBinanceSpotDailyShard({
        dataDir,
        date: DATE,
        day: DAY,
        interval: INTERVAL,
        intervalMs: INTERVAL_MS,
        archiveRoot: source.root,
      }),
      /missing, duplicated, or out of order/,
    );
  } finally {
    await source.close();
    await fs.rm(dataDir, { recursive: true, force: true });
  }
});

function archiveWithTimes(times: number[]): Buffer {
  const rows = times.map((openTime) => [
    openTime,
    100,
    101,
    99,
    100.5,
    1,
    openTime + INTERVAL_MS - 1,
    0,
    0,
    0,
    0,
    0,
  ].join(",")).join("\n");
  const zip = new AdmZip();
  zip.addFile(`BTCUSDT-${INTERVAL}-${DATE}.csv`, Buffer.from(`${rows}\n`));
  return zip.toBuffer();
}

async function serveArchive(archive: Buffer): Promise<{ root: string; close: () => Promise<void> }> {
  const checksum = createHash("sha256").update(archive).digest("hex");
  const server = createServer((request, response) => {
    if (request.url?.endsWith(".CHECKSUM")) {
      response.end(`${checksum}  archive.zip\n`);
      return;
    }
    response.end(archive);
  });
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  const address = server.address();
  assert.ok(address && typeof address !== "string");
  return {
    root: `http://127.0.0.1:${address.port}`,
    close: () => new Promise((resolve, reject) => server.close((error) =>
      error ? reject(error) : resolve())),
  };
}

assert.equal(DAY_MS / INTERVAL_MS, 4);
