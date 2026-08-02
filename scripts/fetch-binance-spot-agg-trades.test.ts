import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import AdmZip from "adm-zip";
import {
  oracleScopedTradeFlowDates,
  parseBinanceAggregateTradeCsvRow,
  type OracleCorpusSplitContract,
} from "./lib/binance-agg-trades.js";
import { parseArchive } from "./fetch-binance-spot-agg-trades.js";
import { openSingleZipEntry } from "./lib/single-entry-zip.js";

test("parses pre-2025 milliseconds and post-2025 microseconds with aggressor direction", () => {
  const millisecond = parseBinanceAggregateTradeCsvRow(
    "10,100.5,2.25,20,22,1735689599999,true,true",
  );
  assert.equal(millisecond.kind, "trade");
  if (millisecond.kind !== "trade") return;
  assert.equal(millisecond.value.timestampUnit, "millisecond");
  assert.equal(millisecond.value.aggregate.timeMicros, 1_735_689_599_999_000);
  assert.equal(millisecond.value.aggregate.tradeCount, 3);
  assert.equal(millisecond.value.aggregate.aggressorSide, -1);

  const microsecond = parseBinanceAggregateTradeCsvRow(
    "11,101,1.5,23,23,1735689600010866,false,True",
  );
  assert.equal(microsecond.kind, "trade");
  if (microsecond.kind !== "trade") return;
  assert.equal(microsecond.value.timestampUnit, "microsecond");
  assert.equal(microsecond.value.aggregate.timeMicros, 1_735_689_600_010_866);
  assert.equal(microsecond.value.aggregate.aggressorSide, 1);
});

test("accepts a header and filters only Binance's invalid zero sentinel", () => {
  assert.equal(
    parseBinanceAggregateTradeCsvRow(
      "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker,is_best_match",
    ).kind,
    "header",
  );
  assert.equal(
    parseBinanceAggregateTradeCsvRow("10,0,0,-1,-1,1650000000000,true,true").kind,
    "invalid-sentinel",
  );
  assert.throws(
    () => parseBinanceAggregateTradeCsvRow("10,0,1,20,20,1650000000000,true,true"),
    /Invalid aggTrade/,
  );
  assert.throws(
    () => parseBinanceAggregateTradeCsvRow(
      "id,price,quantity,first_trade_id,last_trade_id,time,is_buyer_maker,is_best_match",
    ),
    /ordered Binance schema/,
  );
});

test("archive parser rejects aggregate and constituent raw-trade ID gaps", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "trading-agg-gaps-"));
  try {
    const cases = [
      {
        name: "aggregate",
        second: "12,101,1,21,21,1704067200001,false,true",
        pattern: /aggregate trade ID gap/,
      },
      {
        name: "raw",
        second: "11,101,1,22,22,1704067200001,false,true",
        pattern: /raw-trade IDs have a gap or overlap/,
      },
    ];
    for (const item of cases) {
      const archive = path.join(root, `${item.name}.zip`);
      const csv = `${item.name}.csv`;
      const zip = new AdmZip();
      zip.addFile(csv, Buffer.from([
        "10,100,1,20,20,1704067200000,false,true",
        item.second,
        "",
      ].join("\n")));
      zip.writeZip(archive);
      await assert.rejects(
        parseArchive(archive, csv, "2024-01-01"),
        item.pattern,
      );
    }
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("oracle-derived allowlist includes predecessor context and rejects sealed test dates", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "trading-agg-scope-"));
  try {
    const start = Date.parse("2026-01-01T00:00:00.000Z");
    for (let index = 0; index < 65; index += 1) {
      const date = new Date(start + index * 86_400_000).toISOString().slice(0, 10);
      await fs.writeFile(path.join(root, `${date}.json`), "not opened");
    }
    const names = (await fs.readdir(root)).sort();
    const { createHash } = await import("node:crypto");
    const contract: OracleCorpusSplitContract = {
      schemaVersion: 1,
      targetContract: path.basename(root),
      referenceCount: 65,
      referenceFilenameSha256: createHash("sha256")
        .update(`${names.join("\n")}\n`).digest("hex"),
      filenameHashEncoding: "utf8-lf-with-trailing-lf",
      train: { count: 5, first: "2026-01-01", last: "2026-01-05" },
      validation: { count: 30, first: "2026-01-06", last: "2026-02-04" },
      test: {
        count: 30,
        first: "2026-02-05",
        last: "2026-03-06",
        policy: "sealed-never-load",
      },
    };
    const scope = await oracleScopedTradeFlowDates(root, undefined, contract);
    assert.equal(scope.sealedTestStart, "2026-02-05");
    assert.ok(scope.dates.includes("2025-12-31"));
    assert.ok(scope.dates.includes("2026-02-04"));
    assert.ok(!scope.dates.includes("2026-02-05"));
    await assert.rejects(
      oracleScopedTradeFlowDates(root, ["2026-02-05"], contract),
      /sealed oracle test split/,
    );
    assert.equal(await fs.readFile(path.join(root, "2026-01-01.json"), "utf8"), "not opened");
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("oracle scope rejects appended filenames before opening any target", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "trading-agg-seal-"));
  try {
    const original = ["2026-01-01.json", "2026-01-02.json", "2026-01-03.json"];
    for (const name of original) {
      await fs.writeFile(path.join(root, name), "must stay unopened");
    }
    const { createHash } = await import("node:crypto");
    const contract: OracleCorpusSplitContract = {
      schemaVersion: 1,
      targetContract: path.basename(root),
      referenceCount: 3,
      referenceFilenameSha256: createHash("sha256")
        .update(`${original.join("\n")}\n`).digest("hex"),
      filenameHashEncoding: "utf8-lf-with-trailing-lf",
      train: { count: 1, first: "2026-01-01", last: "2026-01-01" },
      validation: { count: 1, first: "2026-01-02", last: "2026-01-02" },
      test: {
        count: 1,
        first: "2026-01-03",
        last: "2026-01-03",
        policy: "sealed-never-load",
      },
    };
    await fs.writeFile(path.join(root, "2026-01-04.json"), "appended");
    await assert.rejects(
      oracleScopedTradeFlowDates(root, undefined, contract),
      /filenames differ from the immutable split contract/,
    );
    assert.equal(
      await fs.readFile(path.join(root, "2026-01-01.json"), "utf8"),
      "must stay unopened",
    );
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("single-entry ZIP reader streams and validates the expected entry", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "trading-single-zip-"));
  try {
    const file = path.join(root, "sample.zip");
    const payload = Buffer.from("one\ntwo\nthree\n".repeat(10_000));
    const zip = new AdmZip();
    zip.addFile("sample.csv", payload);
    zip.writeZip(file);
    const entry = await openSingleZipEntry(file, "sample.csv");
    const chunks: Buffer[] = [];
    for await (const chunk of entry.stream) chunks.push(chunk as Buffer);
    assert.equal(await entry.completed, payload.byteLength);
    assert.deepEqual(Buffer.concat(chunks), payload);
    await assert.rejects(openSingleZipEntry(file, "wrong.csv"), /expected wrong.csv/);
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});
