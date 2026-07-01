import assert from "node:assert/strict";
import {
  canFillOrderAtTick,
  orderExecutionPrice,
} from "../packages/bot-algo/src/execution.js";
import {
  analyzePositions,
  createStrategyConfig,
  createLeveragedBalanceModel,
  SimulatedExecutionEngine,
  summarizeClosedPositions,
  type BotEvent,
  type PartialStrategyConfig,
  type PositionLedger,
} from "../packages/bot-algo/src/index.js";

const SYMBOL = "BTCUSDT";

function createBot(overrides: PartialStrategyConfig = {}): SimulatedExecutionEngine {
  const bot = new SimulatedExecutionEngine(undefined, {
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

function activeLedger(bot: SimulatedExecutionEngine, currentPrice: number): PositionLedger {
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
  const bot = createBot({
    longBorrowDepth: 1,
    shortBorrowDepth: 1,
  });
  bot.recordManualTrade(
    { side: "buy", price: 10_000, quantity: 0.05, positionEffect: "open" },
    1,
  );
  bot.recordManualTrade(
    { side: "sell", price: 11_000, quantity: 0.03, positionEffect: "open" },
    2,
  );

  const ledger = activeLedger(bot, 10_500);
  const long = ledger.longs.find((lot) => lot.remainingQuantity > 0 || lot.lentQuantity > 0);
  const short = ledger.shorts.find((lot) => lot.remainingQuantity > 0);
  assert.ok(long, "Expected the lending long lot.");
  assert.ok(short, "Expected an active short lot.");
  expectApprox(long.remainingQuantity, 0.02, "lender long remaining quantity");
  expectApprox(long.lentQuantity ?? 0, 0.03, "lender long lent quantity");
  expectApprox(long.remainingCostQuote, 170, "lender long remaining cost after lend");
  expectApprox(ledger.summary.internalBorrowedBaseQuantity, 0.03, "internal short base borrow");
  expectApprox(ledger.summary.externalBorrowedBaseQuantity, 0, "external short base borrow");
  expectApprox(short.internalBorrowedQuantity, 0.03, "short lot internal borrow");
  expectApprox(short.externalBorrowedQuantity, 0, "short lot external borrow");
  assert.equal(short.borrowedFromPositionCount, 1);
  return "internal base borrow";
}

function verifyInternalQuoteBorrow(): string {
  const bot = createBot({
    longBorrowDepth: 1,
    shortBorrowDepth: 1,
  });
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

function verifyLeverageCapAdmissionControl(): string {
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
  assert.equal(
    filledReason(events, "leverage cap reduction"),
    undefined,
    "Expected no synthetic leverage-cap reduction fill.",
  );
  expectApprox(
    activeLedger(bot, 10_000).summary.effectiveLeverage,
    2,
    "over-cap futures leverage persists",
  );

  assert.throws(
    () =>
      bot.recordManualTrade(
        { side: "buy", price: 10_000, quantity: 0.01, positionEffect: "open" },
        4,
      ),
    /Leverage limit exceeded/,
    "Expected new exposure-increasing fills over the cap to be rejected.",
  );

  bot.recordManualTrade(
    { side: "sell", price: 10_000, quantity: 0.05, positionEffect: "close" },
    5,
  );
  expectApprox(
    activeLedger(bot, 10_000).summary.effectiveLeverage,
    1.5,
    "close fills are allowed while over cap",
  );
  return "leverage cap admission control";
}

function verifyFuturesOneXLongHasNoPositiveLiquidation(): string {
  const bot = createBot({
    shortMarginModel: "futures-margin",
    maxLeverage: 1,
  });
  bot.recordManualTrade(
    { side: "buy", price: 10_000, quantity: 0.1, positionEffect: "open" },
    1,
  );

  expectApprox(bot.view().quoteFree, 0, "1x futures long quote balance");
  expectApprox(bot.view().baseFree, 0.1, "1x futures long base quantity");
  expectApprox(activeLedger(bot, 10_000).summary.effectiveLeverage, 1, "1x futures long leverage");
  assert.equal(
    bot.accountLiquidationPrice(),
    undefined,
    "Expected no positive liquidation price for a fully funded 1x futures long.",
  );
  assert.throws(
    () =>
      bot.recordManualTrade(
        { side: "buy", price: 10_000, quantity: 0.0001, positionEffect: "open" },
        2,
      ),
    /Leverage limit exceeded/,
    "Expected a 1x futures long to reject quote-borrowing expansion.",
  );
  return "futures 1x long has no positive liquidation";
}

function verifyBalanceModelsUseExchangeLevelCapacity(): string {
  const futuresBot = createBot({
    shortMarginModel: "futures-margin",
    maxLeverage: 1,
    longBorrowDepth: 1,
    shortBorrowDepth: 1,
  });
  futuresBot.recordManualTrade(
    { side: "buy", price: 10_000, quantity: 0.05, positionEffect: "open" },
    1,
  );
  futuresBot.recordManualTrade(
    { side: "sell", price: 11_000, quantity: 0.05, positionEffect: "open" },
    2,
  );
  const futuresCapacity = createLeveragedBalanceModel("futures-margin").entryCapacityQuote({
    state: futuresBot.snapshot(),
    marketPrice: 10_000,
    side: "buy",
    targetLeverage: 1,
    feeRate: 0,
    sidePositionCapacityQuote: Number.POSITIVE_INFINITY,
    pendingLongEntryQuote: 0,
    pendingShortEntryQuote: 0,
  });
  expectApprox(
    futuresCapacity,
    1_050,
    "futures capacity after internally borrowed flat book",
  );

  const spotBot = createBot({
    shortMarginModel: "spot-borrow",
    maxLeverage: 1,
  });
  const spotShortCapacity = createLeveragedBalanceModel("spot-borrow").entryCapacityQuote({
    state: spotBot.snapshot(),
    marketPrice: 10_000,
    side: "sell",
    targetLeverage: 1,
    feeRate: 0,
    sidePositionCapacityQuote: Number.POSITIVE_INFINITY,
    pendingLongEntryQuote: 0,
    pendingShortEntryQuote: 0,
  });
  assert.ok(
    spotShortCapacity < spotBot.view().config.minOrderQuote,
    `Expected spot-borrow 1x short capacity from flat to be below min order, got ${spotShortCapacity}.`,
  );
  return "balance models use exchange-level capacity";
}

function verifyLiquidation(): string {
  const bot = createBot({
    shortMarginModel: "futures-margin",
    maxLeverage: 999,
    positionRisk: {
      marketSlippageBps: 0,
    },
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

function verifyFeeAwareLiquidationSettlement(): string {
  const bot = createBot({
    shortMarginModel: "futures-margin",
    maxLeverage: 999,
    feeBps: 10,
  });
  bot.recordManualTrade(
    { side: "buy", price: 10_000, quantity: 0.2, positionEffect: "open" },
    1,
  );
  const liquidationPrice = bot.accountLiquidationPrice();
  assert.ok(liquidationPrice, "Expected a fee-aware account liquidation price.");
  assert.ok(
    liquidationPrice > 5_000,
    `Expected liquidation price to include sell fee, got ${liquidationPrice}.`,
  );

  const events = bot.onTick(
    { symbol: SYMBOL, eventTime: 2, price: liquidationPrice - 1 },
    { collectEvents: true, processOpenOrders: false, simulateLiquidation: true },
  );
  const liquidation = filledReason(events, "account liquidation");
  assert.ok(liquidation, "Expected a filled fee-aware liquidation order.");
  assert.equal(liquidation.fill?.liquidation, true);
  assert.equal(bot.liquidatedPositionCount(), 1);

  const metrics = bot.view().metrics;
  expectApprox(metrics.equity, 0, "liquidated equity");
  expectApprox(metrics.netPnl, -1_000, "liquidated net pnl");
  expectApprox(metrics.returnPct, -100, "liquidated return");
  expectApprox(metrics.realizedPnl, -1_000, "liquidated realized pnl");
  return "fee-aware liquidation settlement";
}

function verifyMarketOrderSlippage(): string {
  const bot = createBot({
    positionRisk: {
      marketSlippageBps: 100,
    },
  });
  bot.recordManualTrade(
    { side: "buy", price: 100, quantity: 1, positionEffect: "open" },
    1,
  );
  const created = bot.createPositionCloseOrders({ includeUnprofitable: true }, 2);
  assert.equal(created[0]?.order?.type, "market");

  const events = bot.onTick(
    { symbol: SYMBOL, eventTime: 3, price: 100 },
    { collectEvents: true, processOpenOrders: true, simulateLiquidation: false },
  );
  const fill = filledReason(events, "forced close position")?.fill;
  assert.ok(fill, "Expected a filled market close order.");
  expectApprox(fill.price, 99, "sell market slippage fill price");
  return "market order slippage";
}

function verifyStopMarketOrderSlippage(): string {
  const order = {
    side: "sell" as const,
    type: "stop-market" as const,
    trigger: "below" as const,
    price: 100,
  };
  assert.equal(canFillOrderAtTick(order, 99), true);
  expectApprox(
    orderExecutionPrice(order, 99, 100),
    98.01,
    "sell stop-market slippage fill price",
  );
  return "stop-market order slippage";
}

function verifyClosedPositionRemainderProfit(): string {
  const bot = createBot({
    longBorrowDepth: 1,
    shortBorrowDepth: 1,
    borrowerProfitShareToLender: 1,
  });
  bot.recordManualTrade(
    { side: "buy", price: 100, quantity: 1, positionEffect: "open" },
    1,
  );
  bot.recordManualTrade(
    { side: "sell", price: 80, quantity: 1, positionEffect: "open" },
    2,
  );
  const afterBorrow = activeLedger(bot, 80);
  const borrowedLong = afterBorrow.longs.find((lot) => lot.lentQuantity > 0);
  assert.ok(borrowedLong, "Expected the long lot to lend base to the short.");
  expectApprox(borrowedLong.remainingQuantity, 0, "borrowed long remaining quantity");
  expectApprox(borrowedLong.lentQuantity ?? 0, 1, "borrowed long lent quantity");
  expectApprox(borrowedLong.remainingCostQuote, 20, "borrowed long remaining cost");

  bot.recordManualTrade(
    { side: "buy", price: 70, quantity: 1, positionEffect: "close" },
    3,
  );
  const afterReturn = activeLedger(bot, 70);
  const returnedLong = afterReturn.longs.find((lot) => lot.remainingQuantity > 0);
  assert.ok(returnedLong, "Expected the borrowed base to return to the long lot.");
  expectApprox(returnedLong.remainingQuantity, 1, "returned long available quantity");
  expectApprox(returnedLong.lentQuantity ?? 0, 0, "returned long lent quantity");
  expectApprox(returnedLong.remainingCostQuote, 90, "returned long remaining cost");

  bot.recordManualTrade(
    { side: "sell", price: 95, quantity: 1, positionEffect: "close" },
    4,
  );

  const ledger = activeLedger(bot, 95);
  const long = ledger.longs.find((lot) => lot.status === "closed");
  assert.ok(long, "Expected the linked long lot to be closed.");
  expectApprox(long.remainingCostQuote, -5, "closed long remainder profit");

  const stats = summarizeClosedPositions(bot.snapshot());
  assert.equal(stats.closedPositionCount, 2);
  assert.equal(stats.profitableClosedPositionCount, 2);
  return "closed position remainder profit";
}

function verifyShortLenderStaysOpenUntilQuoteReturns(): string {
  const bot = createBot({
    longBorrowDepth: 1,
    shortBorrowDepth: 1,
    borrowerProfitShareToLender: 1,
  });
  bot.recordManualTrade(
    { side: "sell", price: 100, quantity: 1, positionEffect: "open" },
    1,
  );
  bot.recordManualTrade(
    { side: "buy", price: 90, quantity: 1, positionEffect: "open" },
    2,
  );

  const afterBorrow = activeLedger(bot, 90);
  const lendingShort = afterBorrow.shorts.find((lot) => lot.lentQuote > 0);
  assert.ok(lendingShort, "Expected the short lot to lend quote to the long.");
  expectApprox(lendingShort.remainingQuantity, 0, "lending short quantity before cover");
  expectApprox(lendingShort.remainingProceedsQuote, 10, "lending short remaining proceeds");
  expectApprox(lendingShort.lentQuote, 90, "lending short lent quote");

  bot.recordManualTrade(
    { side: "sell", price: 95, quantity: 1, positionEffect: "close" },
    3,
  );
  const afterReturn = activeLedger(bot, 95);
  const settledShort = afterReturn.shorts.find((lot) => lot.id === lendingShort.id);
  assert.ok(settledShort, "Expected the short lot to remain in the ledger.");
  assert.equal(settledShort.status, "open");
  expectApprox(settledShort.remainingQuantity, 1, "settled short returned quantity");
  expectApprox(settledShort.lentQuote, 0, "settled short lent quote");

  bot.recordManualTrade(
    { side: "buy", price: 80, quantity: 1, positionEffect: "close" },
    4,
  );

  const stats = summarizeClosedPositions(bot.snapshot());
  assert.equal(stats.closedPositionCount, 2);
  assert.equal(stats.profitableClosedPositionCount, 2);
  return "short lender stays open until quote returns";
}

function verifyBorrowerProfitDistributedAcrossLenders(): string {
  const bot = createBot({
    longBorrowDepth: 1,
    shortBorrowDepth: 1,
    borrowerProfitShareToLender: 1,
  });
  bot.recordManualTrade(
    { side: "sell", price: 100, quantity: 1, positionEffect: "open" },
    1,
  );
  bot.recordManualTrade(
    { side: "sell", price: 200, quantity: 1, positionEffect: "open" },
    2,
  );
  bot.recordManualTrade(
    { side: "buy", price: 100, quantity: 2, positionEffect: "open" },
    3,
  );

  const afterBorrow = activeLedger(bot, 100);
  const firstShort = afterBorrow.shorts.find((lot) => lot.averagePrice === 100);
  const secondShort = afterBorrow.shorts.find((lot) => lot.averagePrice === 200);
  assert.ok(firstShort, "Expected the first quote-lending short lot.");
  assert.ok(secondShort, "Expected the second quote-lending short lot.");
  expectApprox(firstShort.lentQuote, 100, "first short lent quote");
  expectApprox(secondShort.lentQuote, 100, "second short lent quote");

  bot.recordManualTrade(
    { side: "sell", price: 120, quantity: 1, positionEffect: "close" },
    4,
  );

  const afterPartialClose = activeLedger(bot, 120);
  const firstSettled = afterPartialClose.shorts.find((lot) => lot.id === firstShort.id);
  const secondSettled = afterPartialClose.shorts.find((lot) => lot.id === secondShort.id);
  assert.ok(firstSettled, "Expected the first short to remain after partial settlement.");
  assert.ok(secondSettled, "Expected the second short to remain after partial settlement.");
  expectApprox(firstSettled.lentQuote, 50, "first short proportional lent quote");
  expectApprox(secondSettled.lentQuote, 50, "second short proportional lent quote");
  expectApprox(firstSettled.remainingProceedsQuote, 60, "first short proportional proceeds");
  expectApprox(secondSettled.remainingProceedsQuote, 160, "second short proportional proceeds");
  return "borrower profit distributed across lenders";
}

function verifyInactiveInternalBorrowDoesNotMutateLender(): string {
  const bot = createBot({
    longBorrowDepth: 1,
    shortBorrowDepth: 1,
    internalBorrowAccounting: "inactive",
    borrowerProfitShareToLender: 1,
  });
  bot.recordManualTrade(
    { side: "buy", price: 100, quantity: 1, positionEffect: "open" },
    1,
  );
  bot.recordManualTrade(
    { side: "sell", price: 80, quantity: 1, positionEffect: "open" },
    2,
  );

  const afterBorrow = activeLedger(bot, 80);
  const long = afterBorrow.longs.find((lot) => lot.remainingQuantity > 0);
  const short = afterBorrow.shorts.find((lot) => lot.remainingQuantity > 0);
  assert.ok(long, "Expected the base-lending long lot.");
  assert.ok(short, "Expected the short lot.");
  expectApprox(long.remainingQuantity, 1, "inactive lender available quantity");
  expectApprox(long.lentQuantity ?? 0, 0, "inactive lender lent quantity");
  expectApprox(long.remainingCostQuote, 100, "inactive lender cost before settlement");
  expectApprox(short.internalBorrowedQuantity, 0, "inactive short internal borrow");
  expectApprox(short.externalBorrowedQuantity, 1, "inactive short external borrow");

  bot.recordManualTrade(
    { side: "buy", price: 70, quantity: 1, positionEffect: "close" },
    3,
  );
  const afterShortClose = activeLedger(bot, 70);
  const settledLong = afterShortClose.longs.find((lot) => lot.id === long.id);
  assert.ok(settledLong, "Expected the long lot to remain unchanged.");
  expectApprox(settledLong.remainingQuantity, 1, "inactive settled lender quantity");
  expectApprox(settledLong.remainingCostQuote, 100, "inactive lender cost after short profit");

  bot.recordManualTrade(
    { side: "sell", price: 95, quantity: 1, positionEffect: "close" },
    4,
  );
  const closedLong = activeLedger(bot, 95).longs.find((lot) => lot.id === long.id);
  assert.ok(closedLong, "Expected the long lot to remain in the ledger.");
  assert.equal(closedLong.status, "closed");
  expectApprox(closedLong.remainingCostQuote, 5, "inactive closed long remainder loss");
  return "inactive internal borrow does not mutate lender";
}

const checks = [
  verifyInternalBaseBorrow,
  verifyInternalQuoteBorrow,
  verifyExternalBorrow,
  verifyLeverageCapAdmissionControl,
  verifyFuturesOneXLongHasNoPositiveLiquidation,
  verifyBalanceModelsUseExchangeLevelCapacity,
  verifyLiquidation,
  verifyFeeAwareLiquidationSettlement,
  verifyMarketOrderSlippage,
  verifyStopMarketOrderSlippage,
  verifyClosedPositionRemainderProfit,
  verifyShortLenderStaysOpenUntilQuoteReturns,
  verifyBorrowerProfitDistributedAcrossLenders,
  verifyInactiveInternalBorrowDoesNotMutateLender,
];

for (const check of checks) {
  console.log(`ok - ${check()}`);
}
