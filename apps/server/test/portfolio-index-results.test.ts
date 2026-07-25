import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { gzipSync } from "node:zlib";
import { PortfolioIndexResultsReader } from "../src/portfolio-index-results.js";

test("reports durable sleeve build progress", async (context) => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "portfolio-index-progress-"));
  context.after(() => fs.rm(root, { recursive: true, force: true }));
  const sleeve = path.join(
    root,
    "walk-forward-index",
    "sleeves",
    "2025-07-01_2026-06-30-1m-test",
  );
  await fs.mkdir(sleeve, { recursive: true });
  await fs.writeFile(
    path.join(sleeve, "progress.json"),
    JSON.stringify({
      complete: false,
      events: 525_600,
      completedEvents: 262_400,
      batchSize: 128,
    }),
  );

  const overview = await new PortfolioIndexResultsReader(root).overview();

  assert.equal(overview.status, "building");
  assert.equal(overview.progress?.scale, "1m");
  assert.equal(overview.progress?.completedEvents, 262_400);
  assert.equal(overview.progress?.ratio, 262_400 / 525_600);
});

test("reads and downsamples all portfolio paths from minute candles", async (context) => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "portfolio-index-series-"));
  context.after(() => fs.rm(root, { recursive: true, force: true }));
  const history = path.join(root, "index-history");
  await fs.mkdir(history, { recursive: true });
  await fs.writeFile(
    path.join(history, "latest.json"),
    JSON.stringify({
      window: {
        start: "2026-01-01",
        end: "2026-01-01",
        minuteCandles: 2,
      },
    }),
  );
  const header = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "gross_close",
    "fee_only_close",
    "conservative_close",
    "target_exposure",
    "cash_weight",
    "active_constituents",
    "gross_traded_notional_ratio",
    "baseline_transaction_cost",
    "baseline_funding_cashflow",
    "missing_next_return_weight",
  ].join(",");
  const rows = [
    [
      "2026-01-01T00:00:00.000Z",
      999,
      1_003,
      998,
      1_001,
      1_002,
      1_001.5,
      1_000,
      1,
      0,
      100,
      0.2,
      0.3,
      0.02,
      0,
    ].join(","),
    [
      "2026-01-01T00:01:00.000Z",
      1_001,
      1_004,
      1_000,
      1_003,
      1_005,
      1_004,
      1_001,
      0.98,
      0.02,
      98,
      0.1,
      0.2,
      -0.01,
      0,
    ].join(","),
  ];
  await fs.writeFile(
    path.join(history, "latest.candles.csv.gz"),
    gzipSync(`${header}\n${rows.join("\n")}\n`),
  );

  const reader = new PortfolioIndexResultsReader(root);
  const overview = await reader.overview();
  const series = await reader.series({ maxPoints: 100 });

  assert.equal(overview.status, "complete");
  assert.equal(series.sourceMinutes, 2);
  assert.equal(series.points.length, 2);
  assert.deepEqual(
    {
      baseline: series.points[1]?.baseline,
      gross: series.points[1]?.gross,
      feeOnly: series.points[1]?.feeOnly,
      conservative: series.points[1]?.conservative,
      funding: series.points[1]?.fundingCashflow,
    },
    {
      baseline: 1_003,
      gross: 1_005,
      feeOnly: 1_004,
      conservative: 1_001,
      funding: -0.01,
    },
  );
});
