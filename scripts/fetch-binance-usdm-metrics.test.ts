import assert from "node:assert/strict";
import test from "node:test";
import AdmZip from "adm-zip";
import { parseMetricsArchive } from "./fetch-binance-usdm-metrics.js";

const HEADER = [
  "create_time",
  "symbol",
  "sum_open_interest",
  "sum_open_interest_value",
  "count_toptrader_long_short_ratio",
  "sum_toptrader_long_short_ratio",
  "count_long_short_ratio",
  "sum_taker_long_short_vol_ratio",
].join(",");

function archive(
  mutate?: (lines: string[]) => void,
): Buffer {
  const start = Date.parse("2026-01-01T00:00:00.000Z");
  const lines = [HEADER];
  for (let index = 0; index < 288; index += 1) {
    const timestamp = new Date(start + index * 300_000)
      .toISOString().slice(0, 19).replace("T", " ");
    lines.push([
      timestamp,
      "BTCUSDT",
      100_000 + index,
      6_000_000_000 + index,
      1.8,
      1.1,
      1.7,
      0.9,
    ].join(","));
  }
  mutate?.(lines);
  const zip = new AdmZip();
  zip.addFile("BTCUSDT-metrics-2026-01-01.csv", Buffer.from(`${lines.join("\n")}\n`));
  return zip.toBuffer();
}

test("parses an exact dense UTC day of USD-M futures metrics", () => {
  const parsed = parseMetricsArchive(
    archive(),
    "BTCUSDT-metrics-2026-01-01.csv",
    "2026-01-01",
  );
  assert.equal(parsed.rows.length, 288);
  assert.equal(parsed.rows[0]!.openTime, Date.parse("2026-01-01T00:00:00.000Z"));
  assert.equal(parsed.rows.at(-1)!.openTime, Date.parse("2026-01-01T23:55:00.000Z"));
  assert.equal(parsed.rows[2]!.topTraderPositionLongShortRatio, 1.1);
  assert.equal(parsed.rows[2]!.takerBuySellVolumeRatio, 0.9);
});

test("keeps shifted archive timestamps causal at the UTC boundary", () => {
  const parsed = parseMetricsArchive(
    archive((lines) => {
      for (let index = 1; index < lines.length; index += 1) {
        const columns = lines[index]!.split(",");
        const shifted = Date.parse(`${columns[0]!.replace(" ", "T")}Z`) + 300_000;
        columns[0] = new Date(shifted).toISOString().slice(0, 19).replace("T", " ");
        lines[index] = columns.join(",");
      }
    }),
    "BTCUSDT-metrics-2026-01-01.csv",
    "2026-01-01",
  );
  assert.equal(parsed.sourceCsvRows, 288);
  assert.equal(parsed.observedGridRows, 287);
  assert.equal(parsed.missingGridRows, 1);
  assert.equal(parsed.outsideUtcDayRows, 1);
  assert.equal(parsed.rows[0]!.sumOpenInterest, null);
  assert.equal(parsed.rows[1]!.sumOpenInterest, 100_000);
});

test("preserves missing values and quarantines unsafe timestamps", () => {
  const missing = parseMetricsArchive(
    archive((lines) => lines.pop()),
    "BTCUSDT-metrics-2026-01-01.csv",
    "2026-01-01",
  );
  assert.equal(missing.missingGridRows, 1);
  assert.equal(missing.rows.at(-1)!.sumOpenInterest, null);
  const offGrid = parseMetricsArchive(
    archive((lines) => {
      lines[2] = lines[2]!.replace("00:05:00", "00:06:01");
    }),
    "BTCUSDT-metrics-2026-01-01.csv",
    "2026-01-01",
  );
  assert.equal(offGrid.offGridRows, 1);
  assert.equal(offGrid.missingGridRows, 1);
  assert.equal(offGrid.rows[1]!.sumOpenInterest, null);
  const twoSecondsLate = parseMetricsArchive(
    archive((lines) => {
      lines[2] = lines[2]!.replace("00:05:00", "00:05:02");
    }),
    "BTCUSDT-metrics-2026-01-01.csv",
    "2026-01-01",
  );
  assert.equal(twoSecondsLate.offGridRows, 1);
  assert.equal(twoSecondsLate.timestampAdjustedRows, 0);
  assert.equal(twoSecondsLate.rows[1]!.sumOpenInterest, null);
  const partial = parseMetricsArchive(
    archive((lines) => {
      const columns = lines[3]!.split(",");
      columns[4] = "";
      lines[3] = columns.join(",");
    }),
    "BTCUSDT-metrics-2026-01-01.csv",
    "2026-01-01",
  );
  assert.equal(partial.rows[2]!.topTraderAccountLongShortRatio, null);
  assert.equal(partial.missingValueCounts.topTraderAccountLongShortRatio, 1);
  assert.throws(
    () => parseMetricsArchive(
      archive((lines) => {
        const columns = lines[3]!.split(",");
        columns[4] = "-1";
        lines[3] = columns.join(",");
      }),
      "BTCUSDT-metrics-2026-01-01.csv",
      "2026-01-01",
    ),
    /invalid futures-metrics value/,
  );
});
