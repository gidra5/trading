import assert from "node:assert/strict";
import test from "node:test";
import AdmZip from "adm-zip";
import {
  officialKlineSource,
  parseKlinesArchive,
  parseOfficialChecksum,
} from "./fetch-binance-usdm-klines.js";

const DAY = "2026-01-01";
const START = Date.parse(`${DAY}T00:00:00.000Z`);
const CSV_NAME = `BTCUSDT-1m-${DAY}.csv`;
const HEADER = [
  "open_time",
  "open",
  "high",
  "low",
  "close",
  "volume",
  "close_time",
  "quote_volume",
  "count",
  "taker_buy_volume",
  "taker_buy_quote_volume",
  "ignore",
].join(",");

interface RowOptions {
  openTime?: number;
  closeTime?: number;
  open?: number;
  high?: number;
  low?: number;
  close?: number;
  baseVolume?: number;
  quoteVolume?: number;
  tradeCount?: string | number;
  takerBuyBaseVolume?: number;
  takerBuyQuoteVolume?: number;
  ignore?: number;
  microseconds?: boolean;
}

function csvRow(index: number, options: RowOptions = {}): string {
  const openTime = options.openTime ?? START + index * 60_000;
  const closeTime = options.closeTime ?? openTime + 59_999;
  const noTrade = options.tradeCount === 0;
  const open = options.open ?? 100 + index / 100;
  const close = options.close ?? (noTrade ? open : open + 0.01);
  const baseVolume = options.baseVolume ?? (noTrade ? 0 : 10);
  const quoteVolume = options.quoteVolume ?? (noTrade ? 0 : 1_000);
  const timestampScale = options.microseconds ? 1_000 : 1;
  return [
    openTime * timestampScale,
    open,
    options.high ?? Math.max(open, close) + (noTrade ? 0 : 0.02),
    options.low ?? Math.min(open, close) - (noTrade ? 0 : 0.02),
    close,
    baseVolume,
    closeTime * timestampScale,
    quoteVolume,
    options.tradeCount ?? 25,
    options.takerBuyBaseVolume ?? (noTrade ? 0 : 4),
    options.takerBuyQuoteVolume ?? (noTrade ? 0 : 400),
    options.ignore ?? 0,
  ].join(",");
}

function archive(lines: string[], options: { header?: boolean; name?: string } = {}): Buffer {
  const zip = new AdmZip();
  const content = [
    ...(options.header === false ? [] : [HEADER]),
    ...lines,
    "",
  ].join("\n");
  zip.addFile(options.name ?? CSV_NAME, Buffer.from(content));
  return zip.toBuffer();
}

test("pins the official USD-M daily source path and checksum filename", () => {
  const source = officialKlineSource(DAY);
  assert.deepEqual(source, {
    archiveName: "BTCUSDT-1m-2026-01-01.zip",
    csvName: CSV_NAME,
    url: "https://data.binance.vision/data/futures/um/daily/klines/"
      + "BTCUSDT/1m/BTCUSDT-1m-2026-01-01.zip",
    checksumUrl: "https://data.binance.vision/data/futures/um/daily/klines/"
      + "BTCUSDT/1m/BTCUSDT-1m-2026-01-01.zip.CHECKSUM",
  });
  const hash = "Ab".repeat(32);
  assert.deepEqual(
    parseOfficialChecksum(`${hash}  ${source.archiveName}\n`, source.url),
    { sha256: hash.toLowerCase(), filename: source.archiveName },
  );
  assert.throws(
    () => parseOfficialChecksum(`${hash}  another.zip`, source.url),
    /invalid checksum response/,
  );
});

test("parses an exact headered dense UTC day without fabricating rows", () => {
  const parsed = parseKlinesArchive(
    archive(Array.from({ length: 1_440 }, (_, index) => csvRow(index))),
    CSV_NAME,
    DAY,
  );
  assert.equal(parsed.headerRows, 1);
  assert.equal(parsed.sourceTimestampUnit, "millisecond");
  assert.equal(parsed.observedGridRows, 1_440);
  assert.equal(parsed.liveGridRows, 1_440);
  assert.equal(parsed.noTradeGridRows, 0);
  assert.equal(parsed.missingGridRows, 0);
  assert.equal(parsed.timestampAdjustedRows, 0);
  assert.equal(parsed.rows[0]!.open, 100);
  assert.equal(parsed.rows.at(-1)!.openTime, START + 1_439 * 60_000);
});

test("accepts headerless archives and distinguishes missing from no-trade rows", () => {
  const parsed = parseKlinesArchive(
    archive([
      csvRow(0),
      csvRow(2, { tradeCount: 0 }),
    ], { header: false }),
    CSV_NAME,
    DAY,
  );
  assert.equal(parsed.headerRows, 0);
  assert.equal(parsed.observedGridRows, 2);
  assert.equal(parsed.liveGridRows, 1);
  assert.equal(parsed.noTradeGridRows, 1);
  assert.equal(parsed.missingGridRows, 1_438);
  assert.deepEqual(parsed.rows[1], {
    openTime: START + 60_000,
    open: null,
    high: null,
    low: null,
    close: null,
    baseVolume: null,
    quoteVolume: null,
    tradeCount: null,
    takerBuyBaseVolume: null,
    takerBuyQuoteVolume: null,
  });
  assert.equal(parsed.rows[2]!.tradeCount, 0);
  assert.equal(parsed.rows[2]!.close, parsed.rows[2]!.open);
});

test("preserves exact timing and quarantines rather than shifting off-grid rows", () => {
  const parsed = parseKlinesArchive(
    archive([
      csvRow(0, { microseconds: true }),
      csvRow(1, {
        openTime: START + 61_000,
        closeTime: START + 120_999,
        microseconds: true,
      }),
    ]),
    CSV_NAME,
    DAY,
  );
  assert.equal(parsed.sourceTimestampUnit, "microsecond");
  assert.equal(parsed.observedGridRows, 1);
  assert.equal(parsed.offGridRows, 1);
  assert.equal(parsed.timestampAdjustedRows, 0);
  assert.equal(parsed.rows[1]!.open, null);
});

test("rejects schema, close-time, OHLC, taker, count, and duplicate violations", () => {
  const cases: Array<{ archive: Buffer; pattern: RegExp }> = [
    {
      archive: archive([csvRow(0)], { name: "wrong.csv" }),
      pattern: /expected one ZIP entry/,
    },
    {
      archive: (() => {
        const zip = new AdmZip();
        zip.addFile(CSV_NAME, Buffer.from(`bad,header,with,12,fields,x,x,x,x,x,x,x\n`));
        return zip.toBuffer();
      })(),
      pattern: /header differs/,
    },
    {
      archive: archive([csvRow(0, { closeTime: START + 60_000 })]),
      pattern: /close time is not open\+59999ms/,
    },
    {
      archive: archive([csvRow(0, { high: 99 })]),
      pattern: /inconsistent USD-M kline row/,
    },
    {
      archive: archive([csvRow(0, { takerBuyBaseVolume: 11 })]),
      pattern: /inconsistent USD-M kline row/,
    },
    {
      archive: archive([csvRow(0, { tradeCount: "1.5" })]),
      pattern: /invalid trade count/,
    },
    {
      archive: archive([csvRow(0, { tradeCount: 0, baseVolume: 1 })]),
      pattern: /inconsistent USD-M kline row/,
    },
    {
      archive: archive([csvRow(0, {
        tradeCount: 0,
        high: 101,
      })]),
      pattern: /inconsistent USD-M kline row/,
    },
    {
      archive: archive([csvRow(0), csvRow(0)]),
      pattern: /duplicate USD-M kline grid row/,
    },
  ];
  for (const item of cases) {
    assert.throws(() => parseKlinesArchive(item.archive, CSV_NAME, DAY), item.pattern);
  }
});
