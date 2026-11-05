import assert from "node:assert/strict";
import test from "node:test";
import { createHyperliquidCaptureSpecs } from "./hyperliquid-collection.ts";

test("market observations need no watched account and retain exact coin identifiers", () => {
  const specs = createHyperliquidCaptureSpecs(["BTC", "xyz:TSLA", "BTC"]);
  assert.equal(specs.length, 1);
  assert.equal(specs[0]!.subscriptions.length, 6);
  assert.deepEqual(specs[0]!.subscriptions[3], {
    method: "subscribe", subscription: { type: "trades", coin: "xyz:TSLA" },
  });
});

test("only explicitly selected accounts receive state and order subscriptions", () => {
  const address = `0x${"aB".repeat(20)}`;
  const specs = createHyperliquidCaptureSpecs([], [address, address.toLowerCase()]);
  assert.equal(specs.length, 1);
  assert.equal(specs[0]!.subscriptions.length, 6);
  const subscriptions = specs[0]!.subscriptions.map((row) => row.subscription as Record<string, unknown>);
  assert(subscriptions.every((row) => row.user === address.toLowerCase()));
  assert.deepEqual(subscriptions.find((row) => row.type === "userFills"), {
    type: "userFills", user: address.toLowerCase(), aggregateByTime: false,
  });
});

test("invalid addresses and excessive user subscriptions fail before connecting", () => {
  assert.throws(() => createHyperliquidCaptureSpecs(["BTC"], ["0x1234"]), /20-byte/);
  const users = Array.from({ length: 11 }, (_, index) => `0x${index.toString(16).padStart(40, "0")}`);
  assert.throws(() => createHyperliquidCaptureSpecs(["BTC"], users), /10 unique users/);
  assert.throws(() => createHyperliquidCaptureSpecs(["BTC\nETH"]), /without whitespace/);
});
