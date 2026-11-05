import assert from "node:assert/strict";
import test from "node:test";
import fs from "node:fs";
import { BinanceEventCollector, SequencedDepthBook, selectCaptureMarkets, createCaptureBatches, type CaptureUniverse, type DepthEvent } from "./binance-event-collection.ts";

const snapshot = { lastUpdateId: 10, bids: [["99", "2"]], asks: [["101", "4"]] } as const;
const event = (U: number, u = U, pu?: number): DepthEvent => ({ U, u, pu, E: 1000 + u, b: [["99", "3"]], a: [] });
test("spot snapshots bridge buffered IDs; state is unavailable until the bridge", () => {
  const book = new SequencedDepthBook("spot");
  book.consume(event(9, 10));
  book.consume(event(11));
  assert.equal(book.isValid(), false);
  const results = book.snapshot(snapshot);
  assert.equal(results[0]!.status, "ignored");
  assert.equal(results[1]!.bridge, true);
  assert.equal(book.current()!.bidQty, "3");
  assert.equal(book.current()!.updateId, 11);
  assert.equal(book.checkpoint()!.knownLevelLimitPerSide, 10_000);
});
test("spot gap invalidates state and a fresh snapshot can recover without mutating inputs", () => {
  const book = new SequencedDepthBook("spot");
  book.snapshot(snapshot);
  assert.equal(book.current(), undefined);
  book.consume(event(11));
  assert.equal(book.consume(event(13)).reason, "depth_sequence_gap");
  assert.equal(book.current(), undefined);
  book.snapshot({ ...snapshot, lastUpdateId: 12 });
  assert.equal(book.current()!.updateId, 13);
  assert.equal(snapshot.bids[0][1], "2");
});
test("futures uses inclusive first snapshot bridge and pu thereafter", () => {
  const book = new SequencedDepthBook("usdm-futures");
  book.consume(event(9, 11, 8));
  assert.equal(book.snapshot(snapshot)[0]!.bridge, true);
  assert.equal(book.consume(event(12, 20, 11)).status, "applied");
  assert.equal(book.consume(event(21, 22, 19)).reason, "depth_sequence_gap");
  assert.equal(book.isValid(), false);
});
test("futures cannot bridge a snapshot with a first update that starts after its ID", () => {
  const book = new SequencedDepthBook("usdm-futures");
  book.consume(event(11, 12, 10));
  assert.equal(book.snapshot(snapshot)[0]!.reason, "snapshot_does_not_bridge");
  assert.equal(book.needsSnapshot(), true);
});
test("buffer overflow is explicit and bounded; crossed books are invalid", () => {
  const book = new SequencedDepthBook("spot", 2);
  book.consume(event(11)); book.consume(event(12));
  assert.equal(book.consume(event(13)).reason, "reconstruction_buffer_overflow");
  assert.equal(book.bufferedCount(), 1);
  book.snapshot({ ...snapshot, lastUpdateId: 12 });
  assert.equal(book.consume({ ...event(14), b: [["102", "1"]] }).reason, "crossed_or_empty_book");
  assert.equal(book.isValid(), false);
});
test("unsafe numeric IDs cannot silently certify a reconstructed book", () => {
  const book = new SequencedDepthBook("spot");
  assert.throws(() => book.consume(event(Number.MAX_SAFE_INTEGER + 1)), /Unsafe/);
});
test("economic identity and inverse physical orientation survive route selection", () => {
  const universe: CaptureUniverse = { entries: [
    { assetId: "crypto:CAT", baseAsset: "CAT", routes: [{ venue: "usdm-futures", symbol: "1000CATUSDT", baseAsset: "1000CAT", quoteAsset: "USDT", priceOrientation: "direct" }] },
    { assetId: "tradfi:CAT", baseAsset: "CAT", routes: [{ venue: "usdm-futures", symbol: "CATUSDT", baseAsset: "CAT", quoteAsset: "USDT", priceOrientation: "direct", contractType: "TRADIFI_PERPETUAL" }] },
    { assetId: "fx:BRL", baseAsset: "BRL", routes: [{ venue: "spot", symbol: "USDTBRL", baseAsset: "USDT", quoteAsset: "BRL", priceOrientation: "inverse" }] },
  ] };
  const selected = selectCaptureMarkets(universe);
  assert.equal(selected.length, 3);
  assert.equal(selected[2]!.physicalBaseAsset, "USDT");
  assert.equal(selected[2]!.priceOrientation, "inverse");
  assert.equal(selected[1]!.assetId, "tradfi:CAT");
  assert.equal(selectCaptureMarkets(universe, ["CATUSDT"], ["usdm-futures"]).length, 1);
  assert.throws(() => selectCaptureMarkets(universe, ["MISSING"]), /absent/);
  assert.throws(() => selectCaptureMarkets({ entries: [universe.entries[0]!, { ...universe.entries[1]!, routes: universe.entries[0]!.routes }] }), /Conflicting identities/);
});
test("public depth and futures market feeds use separate bounded subscriptions", () => {
  const markets = selectCaptureMarkets({ entries: [{ assetId: "crypto:BTC", baseAsset: "BTC", routes: [
    { venue: "spot", symbol: "BTCUSDT", baseAsset: "BTC", quoteAsset: "USDT", priceOrientation: "direct" },
    { venue: "usdm-futures", symbol: "BTCUSDT", baseAsset: "BTC", quoteAsset: "USDT", priceOrientation: "direct" },
  ] }] });
  const batches = createCaptureBatches(markets, 1);
  assert.equal(batches.length, 3);
  assert.ok(batches[0]!.streams.includes("btcusdt@trade"));
  assert.ok(batches[1]!.url.includes("/public/"));
  assert.ok(batches[2]!.url.includes("/market/"));
  assert.deepEqual(batches[2]!.streams, ["btcusdt@aggTrade", "btcusdt@markPrice@1s"]);
  assert.throws(() => createCaptureBatches(markets, 101), /1..100/);
});

test("Binance Unicode market identifiers are preserved without permitting URL separators", () => {
  const symbols = ["龙虾USDT", "币安人生USDT", "我踏马来了USDT"];
  const universe: CaptureUniverse = { entries: symbols.map((symbol) => ({ assetId: `crypto:${symbol}`, baseAsset: symbol,
    routes: [{ venue: "usdm-futures" as const, symbol, baseAsset: symbol, quoteAsset: "USDT", priceOrientation: "direct" as const }] })) };
  const selected = selectCaptureMarkets(universe);
  assert.deepEqual(selected.map((market) => market.symbol), symbols);
  assert.ok(createCaptureBatches(selected)[0]!.streams.includes("龙虾usdt@depth@100ms"));
  assert.throws(() => selectCaptureMarkets({ entries: [{ ...universe.entries[0]!, routes: [{ ...universe.entries[0]!.routes[0]!, symbol: "BTC/USDT?x=1" }] }] }), /Invalid Binance symbol/);
});

test("a delayed snapshot queue does not consume reconstruction buffers before its HTTP request", () => {
  const book = new SequencedDepthBook("spot", 2);
  for (let id = 1; id <= 12_000; id++) assert.equal(book.observe(event(id), false).status, "ignored");
  assert.equal(book.bufferedCount(), 0);
  book.observe(event(12_001), true);
  book.snapshot({ ...snapshot, lastUpdateId: 12_000 });
  assert.equal(book.current()!.updateId, 12_001);
});

test("stopping a clock sweep aborts the first venue and never requests the second", async () => {
  const markets = selectCaptureMarkets({ entries: [{ assetId: "crypto:BTC", baseAsset: "BTC", routes: [
    { venue: "spot", symbol: "BTCUSDT", baseAsset: "BTC", quoteAsset: "USDT", priceOrientation: "direct" },
    { venue: "usdm-futures", symbol: "BTCUSDT", baseAsset: "BTC", quoteAsset: "USDT", priceOrientation: "direct" },
  ] }] });
  const previousFetch = globalThis.fetch;
  let calls = 0; let aborted = 0;
  globalThis.fetch = ((_input: unknown, init?: RequestInit) => {
    calls++;
    return new Promise((_resolve, reject) => {
      init?.signal?.addEventListener("abort", () => { aborted++; reject(init.signal!.reason); }, { once: true });
    });
  }) as typeof fetch;
  const capture = new BinanceEventCollector({ markets, sessionId: "test", sink: { write() {} }, onFatal(error) { throw error; } });
  try {
    const sweep = Reflect.apply(Reflect.get(capture, "clockLoop"), capture, []) as Promise<void>;
    capture.stop();
    await sweep;
    assert.equal(calls, 1);
    assert.equal(aborted, 1);
  } finally { capture.stop(); globalThis.fetch = previousFetch; }
});

test("the saved full universe selects every physical route with explicit orientation", () => {
  const universe = JSON.parse(fs.readFileSync(new URL("../../docs/portfolio/restored-market-universe.json", import.meta.url), "utf8")) as CaptureUniverse;
  const selected = selectCaptureMarkets(universe);
  const physicalRoutes = new Set(universe.entries.flatMap((entry) => entry.routes.map((route) => `${route.venue}:${route.symbol}`)));
  assert.equal(selected.length, physicalRoutes.size);
  assert.deepEqual(new Set(selected.map((market) => `${market.venue}:${market.symbol}`)), physicalRoutes);
  assert.ok(selected.some((market) => market.symbol === "龙虾USDT"));
  const inverse = selected.find((market) => market.symbol === "USDTBRL")!;
  assert.equal(inverse.physicalBaseAsset, "USDT");
  assert.equal(inverse.quoteAsset, "BRL");
  assert.equal(inverse.priceOrientation, "inverse");
});
