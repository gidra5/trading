import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { TradingStorageLayout } from "@trading/storage";
import type { Candle } from "@trading/bot-algo";
import { TradingStorage } from "../src/storage.js";

function candle(openTime: number): Candle {
  return {
    symbol: "BTCUSDT",
    interval: "12h",
    openTime,
    closeTime: openTime + 43_200_000 - 1,
    open: 100,
    high: 103,
    low: 99,
    close: 102,
    volume: 5,
    closed: true,
  };
}

test("live server candles use mutable daily staging and canonical immutable shards", async () => {
  const dataDir = await fs.mkdtemp(path.join(os.tmpdir(), "trading-server-storage-"));
  try {
    const storage = new TradingStorage(dataDir, "spot-btcusdt", "BTCUSDT", "12h");
    const layout = new TradingStorageLayout(dataDir);
    const day = Date.parse("2026-01-01T00:00:00.000Z");
    await storage.ensureReady();
    await storage.appendCandle(candle(day));

    const stagingFile = path.join(
      layout.marketMutable,
      "candles",
      "spot-btcusdt",
      "btcusdt",
      "12h",
      "2026-01-01.jsonl",
    );
    assert.ok((await fs.stat(stagingFile)).isFile());

    await storage.appendCandle(candle(day + 43_200_000));
    await assert.rejects(fs.stat(stagingFile), { code: "ENOENT" });
    assert.ok((await fs.stat(path.join(
      layout.candleReferences("spot-btcusdt", "btcusdt", "12h"),
      "2026-01-01.json",
    ))).isFile());
    assert.deepEqual(await storage.loadCandles(2), [
      candle(day),
      candle(day + 43_200_000),
    ]);

    await assert.rejects(
      fs.stat(path.join(
        layout.marketMutable,
        "streams",
        "spot-btcusdt",
        "btcusdt-12h-candles.jsonl",
      )),
      { code: "ENOENT" },
    );
  } finally {
    await fs.rm(dataDir, { recursive: true, force: true });
  }
});
