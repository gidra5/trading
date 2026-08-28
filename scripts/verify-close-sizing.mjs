import assert from "node:assert/strict";
import {
  SimulatedExecutionEngine,
  closeQuantityWithoutMinimumNotionalRemainder,
} from "../packages/bot-algo/dist/index.js";

assert.equal(
  closeQuantityWithoutMinimumNotionalRemainder({
    requestedQuantity: 5,
    availableQuantity: 20,
    executionPrice: 1,
    minNotional: 5,
  }),
  5,
  "an executable remainder must preserve the requested partial close",
);

assert.equal(
  closeQuantityWithoutMinimumNotionalRemainder({
    requestedQuantity: 5,
    availableQuantity: 8,
    executionPrice: 1,
    minNotional: 5,
  }),
  8,
  "a partial close that would leave sub-minimum dust must close the full position",
);

assert.equal(
  closeQuantityWithoutMinimumNotionalRemainder({
    requestedQuantity: 0.05,
    availableQuantity: 0.051,
    executionPrice: 100,
    minNotional: 5,
  }),
  0.051,
  "the rule must work with fractional asset quantities",
);

const config = {
  symbol: "TESTUSDT",
  baseAsset: "TEST",
  quoteAsset: "USDT",
  startingQuote: 1_000,
  maxLeverage: 100,
  feeBps: 0,
  minOrderQuote: 5,
};

const longBot = new SimulatedExecutionEngine(undefined, config);
longBot.recordManualTrade(
  { side: "buy", price: 100, quantity: 0.08, positionEffect: "open" },
  1,
);
const longClose = longBot.createSellOrder(100, 2, "verification", 0.05);
assert.equal(longClose?.quantity, 0.08, "the strategy must fully close a long before leaving $3");

const shortBot = new SimulatedExecutionEngine(undefined, config);
shortBot.recordManualTrade(
  { side: "sell", price: 100, quantity: 0.08, positionEffect: "open" },
  1,
);
const shortClose = shortBot.createBuyToCoverOrder(100, 2, "verification", 0.05);
assert.equal(shortClose?.quantity, 0.08, "the strategy must fully close a short before leaving $3");

const largerLongBot = new SimulatedExecutionEngine(undefined, config);
largerLongBot.recordManualTrade(
  { side: "buy", price: 100, quantity: 0.2, positionEffect: "open" },
  1,
);
const partialClose = largerLongBot.createSellOrder(100, 2, "verification", 0.05);
assert.equal(partialClose?.quantity, 0.05, "the strategy must preserve executable remainders");

console.log("ok - minimum-notional close sizing");
