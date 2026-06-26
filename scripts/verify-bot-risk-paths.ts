import assert from "node:assert/strict";
import {
  analyzePositions,
  createStrategyConfig,
  SimulatedTradingBot,
  type BotEvent,
  type PartialStrategyConfig,
  type PositionLedger,
} from "../packages/bot-algo/src/index.js";

const SYMBOL = "BTCUSDT";

function createBot(overrides: PartialStrategyConfig = {}): SimulatedTradingBot {
  const bot = new SimulatedTradingBot(undefined, {
    symbol: SYMBOL,
    baseAsset: "BTC",
    quoteAsset: "USDT",
    startingQuote: 1_000,
    maxLeverage: 100,
    shortMarginModel: "spot-borrow",
    longBorrowDepth: 0,
    shortBorrowDepth: 0,
    feeBps: 0,
    maxPositionQuote: 100_000,
    minOrderQuote: 1,
    legacyValleyPeak: {
      buyConfirmationOffsets: [],
      sellConfirmationOffsets: [],
    },
    ...overrides,
  });
  bot.setStatus("stopped", 0);
  return bot;
}

function expectApprox(
  actual: number,
  expected: number,
  label: string,
  tolerance = 0.000_001,
): void {
  assert.ok(
    Math.abs(actual - expected) <= tolerance,
    `${label}: expected ${expected}, got ${actual}`,
  );
}

function activeLedger(bot: SimulatedTradingBot, currentPrice: number): PositionLedger {
  return analyzePositions(bot.snapshot(), { currentPrice });
}

function filledReason(events: BotEvent[], reason: string): BotEvent | undefined {
  return events.find(
    (event) =>
      event.type === "order_filled" &&
      event.order?.reason === reason &&
      event.fill?.reason === reason,
  );
}

function verifyInternalBaseBorrow(): string {
  const bot = createBot({ longBorrowDepth: 1, shortBorrowDepth: 1 });
  bot.recordManualTrade(
    { side: "buy", price: 10_000, quantity: 0.05, positionEffect: "open" },
    1,
  );
  bot.recordManualTrade(
    { side: "sell", price: 11_000, quantity: 0.03, positionEffect: "open" },
    2,
  );

  const ledger = activeLedger(bot, 10_500);
  const short = ledger.shorts.find((lot) => lot.remainingQuantity > 0);
  assert.ok(short, "Expected an active short lot.");
  expectApprox(ledger.summary.internalBorrowedBaseQuantity, 0.03, "internal short base borrow");
  expectApprox(ledger.summary.externalBorrowedBaseQuantity, 0, "external short base borrow");
  expectApprox(short.internalBorrowedQuantity, 0.03, "short lot internal borrow");
  expectApprox(short.externalBorrowedQuantity, 0, "short lot external borrow");
  assert.equal(short.borrowedFromPositionCount, 1);
  return "internal base borrow";
}

function verifyInternalQuoteBorrow(): string {
  const bot = createBot({ longBorrowDepth: 1, shortBorrowDepth: 1 });
  bot.recordManualTrade(
    { side: "sell", price: 11_000, quantity: 0.05, positionEffect: "open" },
    1,
  );
  bot.recordManualTrade(
    { side: "buy", price: 10_000, quantity: 0.03, positionEffect: "open" },
    2,
  );

  const ledger = activeLedger(bot, 10_500);
  const long = ledger.longs.find((lot) => lot.remainingQuantity > 0);
  assert.ok(long, "Expected an active long lot.");
  expectApprox(long.internalBorrowedQuote, 300, "long lot internal quote borrow");
  expectApprox(long.externalBorrowedQuote, 0, "long lot external quote borrow");
  assert.equal(long.borrowedFromPositionCount, 1);
  assert.ok(
    ledger.summary.internalBorrowedQuote >= 300,
    `Expected summary internal borrow value to include at least 300 quote, got ${ledger.summary.internalBorrowedQuote}.`,
  );
  return "internal quote borrow";
}

function verifyExternalBorrow(): string {
  const shortBot = createBot();
  shortBot.recordManualTrade(
    { side: "sell", price: 10_000, quantity: 0.05, positionEffect: "open" },
    1,
  );
  const shortLedger = activeLedger(shortBot, 10_000);
  expectApprox(shortLedger.summary.internalBorrowedBaseQuantity, 0, "external short internal borrow");
  expectApprox(shortLedger.summary.externalBorrowedBaseQuantity, 0.05, "external short base borrow");
  expectApprox(shortLedger.summary.externalBorrowedQuote, 500, "external short quote value");

  const longBot = createBot();
  longBot.recordManualTrade(
    { side: "buy", price: 10_000, quantity: 0.2, positionEffect: "open" },
    1,
  );
  const longLedger = activeLedger(longBot, 10_000);
  expectApprox(longLedger.summary.internalBorrowedQuote, 0, "external long internal quote borrow");
  expectApprox(longLedger.summary.externalBorrowedQuote, 1_000, "external long quote borrow");
  return "external borrow";
}

function verifyLeverageCapReduction(): string {
  const bot = createBot({
    shortMarginModel: "futures-margin",
    maxLeverage: 100,
  });
  bot.recordManualTrade(
    { side: "buy", price: 10_000, quantity: 0.2, positionEffect: "open" },
    1,
  );
  expectApprox(activeLedger(bot, 10_000).summary.effectiveLeverage, 2, "initial futures leverage");

  bot.setConfig(createStrategyConfig({ ...bot.view().config, maxLeverage: 1 }), 2);
  const events = bot.onTick(
    { symbol: SYMBOL, eventTime: 3, price: 10_000 },
    { collectEvents: true, processOpenOrders: false, simulateLiquidation: false },
  );
  const reduction = filledReason(events, "leverage cap reduction");
  assert.ok(reduction, "Expected a filled leverage-cap reduction order.");
  assert.equal(reduction.order?.side, "sell");
  assert.ok((reduction.fill?.quantity ?? 0) > 0, "Expected leverage reduction to close quantity.");
  assert.ok(
    activeLedger(bot, 10_000).summary.effectiveLeverage <= 1.000_1,
    "Expected leverage reduction to bring effective leverage back to the cap.",
  );
  return "leverage cap reduction";
}

function verifyLiquidation(): string {
  const bot = createBot({
    shortMarginModel: "futures-margin",
    maxLeverage: 999,
  });
  bot.recordManualTrade(
    { side: "buy", price: 10_000, quantity: 0.2, positionEffect: "open" },
    1,
  );
  expectApprox(bot.accountLiquidationPrice() ?? 0, 5_000, "account liquidation price");

  const events = bot.onTick(
    { symbol: SYMBOL, eventTime: 2, price: 4_999 },
    { collectEvents: true, processOpenOrders: false, simulateLiquidation: true },
  );
  const liquidation = filledReason(events, "account liquidation");
  assert.ok(liquidation, "Expected a filled liquidation order.");
  assert.equal(liquidation.order?.side, "sell");
  assert.equal(liquidation.fill?.liquidation, true);
  assert.equal(bot.liquidatedPositionCount(), 1);

  const ledger = activeLedger(bot, 4_999);
  expectApprox(ledger.summary.longQuantity, 0, "liquidated long quantity");
  expectApprox(ledger.summary.shortQuantity, 0, "liquidated short quantity");
  assert.equal(bot.accountLiquidationPrice(), undefined);
  assert.ok(bot.view().metrics.realizedPnl < 0, "Expected liquidation to realize a loss.");
  return "liquidation";
}

const checks = [
  verifyInternalBaseBorrow,
  verifyInternalQuoteBorrow,
  verifyExternalBorrow,
  verifyLeverageCapReduction,
  verifyLiquidation,
];

for (const check of checks) {
  console.log(`ok - ${check()}`);
}
