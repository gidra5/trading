import type {
  InternalBorrowAccounting,
  LongPositionLot,
  PaperBotState,
  PositionLedger,
  PositionRiskConfig,
  ShortPositionLot,
  TradeFill,
  TradingOrder,
} from "./types.js";
import { createLeveragedBalanceModel } from "./leveraged-balance.js";

const EPSILON = 1e-10;

interface LedgerContext {
  currentPrice: number;
  lowerPriceExpectation: number;
  lowerBaselinePrice: number;
  upperPriceExpectation: number;
  upperBaselinePrice: number;
  maxLossPct: number;
  feeAndSlippageRate: number;
  netMarketSellPrice: number;
  grossMarketBuyPrice: number;
  longBorrowDepth: number;
  shortBorrowDepth: number;
  internalBorrowAccounting: InternalBorrowAccounting;
  borrowerProfitShareToLender: number;
}

interface BorrowProfile {
  borrowedQuantity: number;
  borrowedQuote: number;
  internalBorrowedQuantity: number;
  internalBorrowedQuote: number;
  externalBorrowedQuantity: number;
  externalBorrowedQuote: number;
  borrowedFromPositionCount: number;
}

interface BorrowProfiles {
  longs: Map<string, BorrowProfile>;
  shorts: Map<string, BorrowProfile>;
}

interface ShortBorrowAllocation {
  longLotId: string;
  quantity: number;
  quote: number;
  depthRemaining: number;
}

interface LongBorrowAllocation {
  shortLotId: string;
  quantity: number;
  quote: number;
  depthRemaining: number;
}

export interface ClosedPositionStats {
  closedPositionCount: number;
  profitableClosedPositionCount: number;
  profitableClosedPositionRate: number;
  liquidatedPositionCount: number;
}

type MutableLongLot = Omit<
  LongPositionLot,
  | "status"
  | "breakEvenSellPrice"
  | "maxLossSellPrice"
  | "recommendedSellQuote"
  | "recommendedSellQuantity"
  | "projectedRemainingQuantity"
  | "projectedRemainingCostQuote"
  | "projectedBreakEvenSellPrice"
  | "canReachLowerBaseline"
  | "exposureQuote"
  | "leverage"
  | "borrowedQuantity"
  | "borrowedQuote"
  | "internalBorrowedQuantity"
  | "internalBorrowedQuote"
  | "externalBorrowedQuantity"
  | "externalBorrowedQuote"
  | "borrowedFromPositionCount"
  | "borrowLocked"
> & {
  lentQuantity: number;
  borrowDepthRemaining: number;
  borrowAllocations: LongBorrowAllocation[];
};

type MutableShortLot = Omit<
  ShortPositionLot,
  | "status"
  | "breakEvenBuyPrice"
  | "maxLossBuyPrice"
  | "recommendedBuyQuote"
  | "recommendedBuyQuantity"
  | "projectedRemainingQuantity"
  | "projectedRemainingProceedsQuote"
  | "projectedBreakEvenBuyPrice"
  | "canReachUpperBaseline"
  | "exposureQuote"
  | "leverage"
  | "borrowedQuantity"
  | "borrowedQuote"
  | "internalBorrowedQuantity"
  | "internalBorrowedQuote"
  | "externalBorrowedQuantity"
  | "externalBorrowedQuote"
  | "borrowedFromPositionCount"
  | "borrowLocked"
> & {
  lentQuote: number;
  borrowDepthRemaining: number;
  borrowAllocations: ShortBorrowAllocation[];
};

export const defaultPositionRiskConfig: PositionRiskConfig = {
  lowerPriceExpectation: 0,
  lowerBaselinePrice: 0,
  upperPriceExpectation: 0,
  upperBaselinePrice: 0,
  maxLossPct: 0.02,
  marketSlippageBps: 10,
};

export function createPositionRiskConfig(
  overrides: Partial<PositionRiskConfig> = {},
): PositionRiskConfig {
  const config = {
    ...defaultPositionRiskConfig,
    ...overrides,
  };

  config.lowerPriceExpectation = cleanNonNegative(config.lowerPriceExpectation);
  config.lowerBaselinePrice = cleanNonNegative(config.lowerBaselinePrice);
  config.upperPriceExpectation = cleanNonNegative(config.upperPriceExpectation);
  config.upperBaselinePrice = cleanNonNegative(config.upperBaselinePrice);
  config.maxLossPct = clamp(cleanNumber(config.maxLossPct), 0, 1);
  config.marketSlippageBps = cleanNonNegative(config.marketSlippageBps);

  return config;
}

export function analyzePositions(
  state: PaperBotState,
  options: { currentPrice?: number } = {},
): PositionLedger {
  const context = buildContext(state, options.currentPrice);
  const longs: MutableLongLot[] = [];
  const shorts: MutableShortLot[] = [];
  const orderById = new Map(state.orders.map((order) => [order.id, order]));

  for (const fill of chronologicalFills(state.fills)) {
    if (fill.side === "buy") {
      applyBuyFill(fill, orderById.get(fill.orderId), shorts, longs, context);
    } else {
      applySellFill(fill, orderById.get(fill.orderId), longs, shorts, context);
    }
  }

  for (const order of state.orders) {
    if (order.status === "open" && order.positionEffect !== "close") {
      appendPendingOrderLot(order, state.config.feeBps / 10_000, context, longs, shorts);
    }
  }

  const borrowProfiles = buildBorrowProfiles(longs, shorts, context, state);
  const finalizedLongs = longs.map((lot) =>
    finalizeLongLot(lot, context, borrowProfiles.longs.get(lot.id) ?? emptyBorrowProfile()),
  );
  const finalizedShorts = shorts.map((lot) =>
    finalizeShortLot(lot, context, borrowProfiles.shorts.get(lot.id) ?? emptyBorrowProfile()),
  );
  const activeLongs = finalizedLongs.filter((lot) => lot.status !== "pending");
  const activeShorts = finalizedShorts.filter((lot) => lot.status !== "pending");
  const longQuantity = roundAsset(sum(activeLongs, "remainingQuantity"));
  const longExposureQuote = roundQuote(longQuantity * context.currentPrice);
  const shortExposureQuote = roundQuote(sum(activeShorts, "exposureQuote"));
  const grossExposureQuote = roundQuote(longExposureQuote + shortExposureQuote);
  const externalBorrowedQuote = roundQuote(
    sum(activeLongs, "externalBorrowedQuote") + sum(activeShorts, "externalBorrowedQuote"),
  );
  const effectiveLeverage = createLeveragedBalanceModel(
    state.config.shortMarginModel,
  ).effectiveLeverage(state, context.currentPrice);

  return {
    summary: {
      currentPrice: context.currentPrice,
      netMarketSellPrice: context.netMarketSellPrice,
      grossMarketBuyPrice: context.grossMarketBuyPrice,
      lowerBaselinePrice: context.lowerBaselinePrice,
      upperBaselinePrice: context.upperBaselinePrice,
      lowerPriceExpectation: context.lowerPriceExpectation,
      upperPriceExpectation: context.upperPriceExpectation,
      maxLossPct: context.maxLossPct,
      feeAndSlippageRate: context.feeAndSlippageRate,
      longQuantity,
      shortQuantity: roundAsset(sum(activeShorts, "remainingQuantity")),
      longRemainingCostQuote: roundQuote(sum(activeLongs, "remainingCostQuote")),
      shortRemainingProceedsQuote: roundQuote(sum(activeShorts, "remainingProceedsQuote")),
      grossExposureQuote,
      netExposureQuote: roundQuote(longExposureQuote - shortExposureQuote),
      longExposureQuote,
      shortExposureQuote,
      internalBorrowedBaseQuantity: roundAsset(sum(activeShorts, "internalBorrowedQuantity")),
      externalBorrowedBaseQuantity: roundAsset(sum(activeShorts, "externalBorrowedQuantity")),
      internalBorrowedQuote: roundQuote(
        sum(activeLongs, "internalBorrowedQuote") + sum(activeShorts, "internalBorrowedQuote"),
      ),
      externalBorrowedQuote,
      effectiveLeverage,
      pendingLongQuantity: roundAsset(sum(finalizedLongs, "pendingQuantity")),
      pendingShortQuantity: roundAsset(sum(finalizedShorts, "pendingQuantity")),
      pendingLongQuote: roundQuote(sum(finalizedLongs, "pendingQuote")),
      pendingShortQuote: roundQuote(sum(finalizedShorts, "pendingQuote")),
      realizedQuotePnl: roundQuote(state.realizedPnl),
    },
    longs: finalizedLongs,
    shorts: finalizedShorts,
  };
}

export function summarizeClosedPositions(state: PaperBotState): ClosedPositionStats {
  const ledger = analyzePositions(state);
  let closedPositionCount = 0;
  let profitableClosedPositionCount = 0;
  let liquidatedPositionCount = 0;

  for (const lot of [...ledger.longs, ...ledger.shorts]) {
    if (lot.status !== "closed" || lot.filledQuantity <= EPSILON) {
      continue;
    }

    closedPositionCount += 1;
    if (closedLotProfitQuote(lot) > EPSILON) {
      profitableClosedPositionCount += 1;
    }
  }

  for (const fill of state.fills) {
    if (fill.liquidation) {
      liquidatedPositionCount += Math.max(1, fill.liquidatedPositionCount ?? 1);
    }
  }

  return {
    closedPositionCount,
    profitableClosedPositionCount,
    profitableClosedPositionRate:
      closedPositionCount > 0 ? (profitableClosedPositionCount / closedPositionCount) * 100 : 0,
    liquidatedPositionCount,
  };
}

function closedLotProfitQuote(lot: LongPositionLot | ShortPositionLot): number {
  if (lot.side === "long") {
    return roundQuote(-lot.remainingCostQuote);
  }

  return roundQuote(lot.remainingProceedsQuote);
}

function buildContext(state: PaperBotState, currentPriceOverride?: number): LedgerContext {
  const risk = createPositionRiskConfig(state.config.positionRisk);
  const currentPrice =
    cleanPositive(currentPriceOverride) || cleanPositive(state.lastPrice) || cleanPositive(state.avgEntryPrice);
  const feeAndSlippageRate =
    cleanNonNegative(state.config.feeBps) / 10_000 + risk.marketSlippageBps / 10_000;

  return {
    currentPrice,
    lowerPriceExpectation:
      risk.lowerPriceExpectation > 0 ? risk.lowerPriceExpectation : roundQuote(currentPrice * 0.85),
    lowerBaselinePrice:
      risk.lowerBaselinePrice > 0 ? risk.lowerBaselinePrice : roundQuote(currentPrice * 0.9),
    upperPriceExpectation:
      risk.upperPriceExpectation > 0 ? risk.upperPriceExpectation : roundQuote(currentPrice * 1.15),
    upperBaselinePrice:
      risk.upperBaselinePrice > 0 ? risk.upperBaselinePrice : roundQuote(currentPrice * 1.1),
    maxLossPct: risk.maxLossPct,
    feeAndSlippageRate,
    netMarketSellPrice: currentPrice > 0 ? currentPrice / (1 + feeAndSlippageRate) : 0,
    grossMarketBuyPrice: currentPrice > 0 ? currentPrice * (1 + feeAndSlippageRate) : 0,
    longBorrowDepth: normalizeBorrowDepth(state.config.longBorrowDepth),
    shortBorrowDepth: normalizeBorrowDepth(state.config.shortBorrowDepth),
    internalBorrowAccounting:
      state.config.internalBorrowAccounting === "inactive" ? "inactive" : "active",
    borrowerProfitShareToLender: clamp(
      cleanNumber(state.config.borrowerProfitShareToLender ?? 1),
      0,
      1,
    ),
  };
}

function applyBuyFill(
  fill: TradeFill,
  sourceOrder: TradingOrder | undefined,
  shorts: MutableShortLot[],
  longs: MutableLongLot[],
  context: LedgerContext,
): void {
  const unitCost = (fill.quoteQuantity + fill.feeQuote) / fill.quantity;
  let quantityLeft = fill.quantity;
  let costLeft = fill.quoteQuantity + fill.feeQuote;

  if (fill.targetPositionId) {
    const target = shorts.find((short) => short.id === fill.targetPositionId);
    if (target) {
      const closed = closeShortLot(target, quantityLeft, unitCost, longs, context);
      quantityLeft = roundAsset(quantityLeft - closed.quantity);
      costLeft = roundQuote(costLeft - closed.quote);
    }
  }

  if (fill.positionEffect !== "open") {
    for (const short of shorts) {
      if (quantityLeft <= EPSILON) {
        break;
      }
      if (short.remainingQuantity <= EPSILON) {
        continue;
      }
      const closed = closeShortLot(short, quantityLeft, unitCost, longs, context);
      quantityLeft = roundAsset(quantityLeft - closed.quantity);
      costLeft = roundQuote(costLeft - closed.quote);
    }
  }

  if (fill.positionEffect !== "close" && quantityLeft > EPSILON && costLeft > EPSILON) {
    const borrowAllocations = allocateLongBorrowFromShortLots(
      shorts,
      quantityLeft,
      unitCost,
      context,
    );
    longs.push({
      id: `long_${fill.id}`,
      side: "long",
      sourceOrderId: fill.orderId,
      openedAt: sourceOrder?.createdAt ?? fill.filledAt,
      originalQuantity: roundAsset(quantityLeft),
      filledQuantity: roundAsset(quantityLeft),
      pendingQuantity: 0,
      pendingQuote: 0,
      pendingLimitPrice: 0,
      closedQuantity: 0,
      closedQuote: 0,
      averagePrice: roundQuote(costLeft / quantityLeft),
      costQuote: roundQuote(costLeft),
      remainingQuantity: roundAsset(quantityLeft),
      remainingCostQuote: roundQuote(costLeft),
      ...lotLifecycleFields(sourceOrder ?? fill),
      lentQuantity: 0,
      borrowDepthRemaining: inheritedLongBorrowDepth(
        borrowAllocations,
        context.longBorrowDepth,
      ),
      borrowAllocations,
    });
  }
}

function applySellFill(
  fill: TradeFill,
  sourceOrder: TradingOrder | undefined,
  longs: MutableLongLot[],
  shorts: MutableShortLot[],
  context: LedgerContext,
): void {
  const unitProceeds = (fill.quoteQuantity - fill.feeQuote) / fill.quantity;
  let quantityLeft = fill.quantity;
  let proceedsLeft = fill.quoteQuantity - fill.feeQuote;

  if (fill.targetPositionId) {
    const target = longs.find((long) => long.id === fill.targetPositionId);
    if (target) {
      const closed = closeLongLot(target, quantityLeft, unitProceeds, shorts, context);
      quantityLeft = roundAsset(quantityLeft - closed.quantity);
      proceedsLeft = roundQuote(proceedsLeft - closed.quote);
    }
  }

  if (fill.positionEffect !== "open") {
    for (const long of longs) {
      if (quantityLeft <= EPSILON) {
        break;
      }
      if (long.remainingQuantity <= EPSILON) {
        continue;
      }
      const closed = closeLongLot(long, quantityLeft, unitProceeds, shorts, context);
      quantityLeft = roundAsset(quantityLeft - closed.quantity);
      proceedsLeft = roundQuote(proceedsLeft - closed.quote);
    }
  }

  if (fill.positionEffect !== "close" && quantityLeft > EPSILON && proceedsLeft > EPSILON) {
    const borrowAllocations = allocateShortBorrowFromLongLots(
      longs,
      quantityLeft,
      unitProceeds,
      context,
    );
    shorts.push({
      id: `short_${fill.id}`,
      side: "short",
      sourceOrderId: fill.orderId,
      openedAt: sourceOrder?.createdAt ?? fill.filledAt,
      originalQuantity: roundAsset(quantityLeft),
      filledQuantity: roundAsset(quantityLeft),
      pendingQuantity: 0,
      pendingQuote: 0,
      pendingLimitPrice: 0,
      closedQuantity: 0,
      closedQuote: 0,
      averagePrice: roundQuote(proceedsLeft / quantityLeft),
      proceedsQuote: roundQuote(proceedsLeft),
      remainingQuantity: roundAsset(quantityLeft),
      remainingProceedsQuote: roundQuote(proceedsLeft),
      ...lotLifecycleFields(sourceOrder ?? fill),
      lentQuote: 0,
      borrowDepthRemaining: inheritedShortBorrowDepth(
        borrowAllocations,
        context.shortBorrowDepth,
      ),
      borrowAllocations,
    });
  }
}

function closeShortLot(
  short: MutableShortLot,
  requestedQuantity: number,
  unitCost: number,
  longs: MutableLongLot[],
  context: LedgerContext,
): { quantity: number; quote: number } {
  const quantity = Math.min(requestedQuantity, short.remainingQuantity);
  const quote = quantity * unitCost;
  settleShortBorrowAllocations(short, quantity, unitCost, longs, context);
  short.remainingQuantity = roundAsset(short.remainingQuantity - quantity);
  short.remainingProceedsQuote = roundQuote(short.remainingProceedsQuote - quote);
  short.closedQuantity = roundAsset(short.closedQuantity + quantity);
  short.closedQuote = roundQuote(short.closedQuote + quote);
  return {
    quantity: roundAsset(quantity),
    quote: roundQuote(quote),
  };
}

function closeLongLot(
  long: MutableLongLot,
  requestedQuantity: number,
  unitProceeds: number,
  shorts: MutableShortLot[],
  context: LedgerContext,
): { quantity: number; quote: number } {
  const quantity = Math.min(requestedQuantity, long.remainingQuantity);
  const quote = quantity * unitProceeds;
  settleLongBorrowAllocations(long, quantity, unitProceeds, shorts, context);
  long.remainingQuantity = roundAsset(long.remainingQuantity - quantity);
  long.remainingCostQuote = roundQuote(long.remainingCostQuote - quote);
  long.closedQuantity = roundAsset(long.closedQuantity + quantity);
  long.closedQuote = roundQuote(long.closedQuote + quote);
  return {
    quantity: roundAsset(quantity),
    quote: roundQuote(quote),
  };
}

function allocateShortBorrowFromLongLots(
  longs: MutableLongLot[],
  quantity: number,
  unitProceeds: number,
  context: LedgerContext,
): ShortBorrowAllocation[] {
  if (context.internalBorrowAccounting === "inactive") {
    return [];
  }

  let quantityLeft = roundAsset(quantity);
  const allocations: ShortBorrowAllocation[] = [];
  const sources = [...longs]
    .filter(
      (lot) =>
        lot.borrowDepthRemaining > 0 &&
        !hasLotLifecycleControls(lot) &&
        lot.remainingQuantity > EPSILON,
    )
    .sort((left, right) => {
      const leftBreakEven = longLotBreakEvenBeforeFees(left);
      const rightBreakEven = longLotBreakEvenBeforeFees(right);
      const leftIsBad = unitProceeds < leftBreakEven;
      const rightIsBad = unitProceeds < rightBreakEven;
      if (leftIsBad !== rightIsBad) {
        return leftIsBad ? -1 : 1;
      }
      return rightBreakEven - leftBreakEven || left.openedAt - right.openedAt;
    });

  for (const source of sources) {
    if (quantityLeft <= EPSILON) {
      break;
    }

    const available = Math.max(0, source.remainingQuantity);
    const borrowedQuantity = roundAsset(Math.min(quantityLeft, available));
    if (borrowedQuantity <= EPSILON) {
      continue;
    }

    const borrowedQuote = roundQuote(borrowedQuantity * unitProceeds);
    source.lentQuantity = roundAsset(source.lentQuantity + borrowedQuantity);
    source.remainingQuantity = roundAsset(source.remainingQuantity - borrowedQuantity);
    source.remainingCostQuote = roundQuote(source.remainingCostQuote - borrowedQuote);
    quantityLeft = roundAsset(quantityLeft - borrowedQuantity);
    allocations.push({
      longLotId: source.id,
      quantity: borrowedQuantity,
      quote: borrowedQuote,
      depthRemaining: Math.max(0, source.borrowDepthRemaining - 1),
    });
  }

  return allocations;
}

function allocateLongBorrowFromShortLots(
  shorts: MutableShortLot[],
  quantity: number,
  unitCost: number,
  context: LedgerContext,
): LongBorrowAllocation[] {
  if (context.internalBorrowAccounting === "inactive") {
    return [];
  }

  let quantityLeft = roundAsset(quantity);
  const allocations: LongBorrowAllocation[] = [];
  const sources = [...shorts]
    .filter(
      (lot) =>
        lot.borrowDepthRemaining > 0 &&
        !hasLotLifecycleControls(lot) &&
        lot.remainingQuantity > EPSILON &&
        lot.remainingProceedsQuote > EPSILON,
    )
    .sort((left, right) => {
      const leftBreakEven = shortLotBreakEvenBeforeFees(left);
      const rightBreakEven = shortLotBreakEvenBeforeFees(right);
      const leftIsBad = unitCost > leftBreakEven;
      const rightIsBad = unitCost > rightBreakEven;
      if (leftIsBad !== rightIsBad) {
        return leftIsBad ? -1 : 1;
      }
      return leftBreakEven - rightBreakEven || left.openedAt - right.openedAt;
    });

  for (const source of sources) {
    if (quantityLeft <= EPSILON) {
      break;
    }

    const affordableQuantity = Math.min(
      source.remainingQuantity,
      source.remainingProceedsQuote / unitCost,
    );
    const borrowedQuantity = roundAsset(Math.min(quantityLeft, affordableQuantity));
    if (borrowedQuantity <= EPSILON) {
      continue;
    }

    const borrowedQuote = roundQuote(borrowedQuantity * unitCost);
    source.lentQuote = roundQuote(source.lentQuote + borrowedQuote);
    source.remainingQuantity = roundAsset(source.remainingQuantity - borrowedQuantity);
    source.remainingProceedsQuote = roundQuote(
      source.remainingProceedsQuote - borrowedQuote,
    );
    quantityLeft = roundAsset(quantityLeft - borrowedQuantity);
    allocations.push({
      shortLotId: source.id,
      quantity: borrowedQuantity,
      quote: borrowedQuote,
      depthRemaining: Math.max(0, source.borrowDepthRemaining - 1),
    });
  }

  return allocations;
}

function settleShortBorrowAllocations(
  short: MutableShortLot,
  closedQuantity: number,
  unitCost: number,
  longs: MutableLongLot[],
  context: LedgerContext,
): void {
  const allocations = short.borrowAllocations.filter(
    (allocation) => allocation.quantity > EPSILON && allocation.quote > EPSILON,
  );
  const totalBorrowedQuantity = roundAsset(
    allocations.reduce((total, allocation) => total + allocation.quantity, 0),
  );
  const quantityToSettle = roundAsset(Math.min(closedQuantity, totalBorrowedQuantity));
  if (quantityToSettle <= EPSILON || totalBorrowedQuantity <= EPSILON) {
    return;
  }

  const settlementRatio = Math.min(1, quantityToSettle / totalBorrowedQuantity);
  let quantityLeft = quantityToSettle;
  let principalLeft = roundQuote(
    allocations.reduce((total, allocation) => total + allocation.quote * settlementRatio, 0),
  );

  for (let index = 0; index < allocations.length; index += 1) {
    const allocation = allocations[index];
    if (quantityLeft <= EPSILON || principalLeft <= EPSILON) {
      break;
    }
    const lastAllocation = index === allocations.length - 1;
    const quantity = roundAsset(
      Math.min(
        allocation.quantity,
        lastAllocation ? quantityLeft : allocation.quantity * settlementRatio,
        quantityLeft,
      ),
    );
    const principalQuote = roundQuote(
      Math.min(
        allocation.quote,
        lastAllocation ? principalLeft : allocation.quote * settlementRatio,
        principalLeft,
      ),
    );
    if (quantity <= EPSILON || principalQuote <= EPSILON) {
      continue;
    }
    const coverQuote = roundQuote(quantity * unitCost);
    const long = longs.find((lot) => lot.id === allocation.longLotId);
    if (long) {
      const profitQuote = principalQuote - coverQuote;
      long.lentQuantity = roundAsset(Math.max(0, long.lentQuantity - quantity));
      long.remainingQuantity = roundAsset(long.remainingQuantity + quantity);
      const returnedQuote =
        profitQuote > 0
          ? roundQuote(
              principalQuote - profitQuote * context.borrowerProfitShareToLender,
            )
          : coverQuote;
      long.remainingCostQuote = roundQuote(long.remainingCostQuote + returnedQuote);
    }

    allocation.quantity = roundAsset(allocation.quantity - quantity);
    allocation.quote = roundQuote(allocation.quote - principalQuote);
    quantityLeft = roundAsset(quantityLeft - quantity);
    principalLeft = roundQuote(principalLeft - principalQuote);
  }

  short.borrowAllocations = short.borrowAllocations.filter(
    (allocation) => allocation.quantity > EPSILON && allocation.quote > EPSILON,
  );
}

function settleLongBorrowAllocations(
  long: MutableLongLot,
  closedQuantity: number,
  unitProceeds: number,
  shorts: MutableShortLot[],
  context: LedgerContext,
): void {
  const allocations = long.borrowAllocations.filter(
    (allocation) => allocation.quantity > EPSILON && allocation.quote > EPSILON,
  );
  const totalBorrowedQuantity = roundAsset(
    allocations.reduce((total, allocation) => total + allocation.quantity, 0),
  );
  const quantityToSettle = roundAsset(Math.min(closedQuantity, totalBorrowedQuantity));
  if (quantityToSettle <= EPSILON || totalBorrowedQuantity <= EPSILON) {
    return;
  }

  const settlementRatio = Math.min(1, quantityToSettle / totalBorrowedQuantity);
  let quantityLeft = quantityToSettle;
  let principalLeft = roundQuote(
    allocations.reduce((total, allocation) => total + allocation.quote * settlementRatio, 0),
  );

  for (let index = 0; index < allocations.length; index += 1) {
    const allocation = allocations[index];
    if (quantityLeft <= EPSILON || principalLeft <= EPSILON) {
      break;
    }
    const lastAllocation = index === allocations.length - 1;
    const quantity = roundAsset(
      Math.min(
        allocation.quantity,
        lastAllocation ? quantityLeft : allocation.quantity * settlementRatio,
        quantityLeft,
      ),
    );
    const principalQuote = roundQuote(
      Math.min(
        allocation.quote,
        lastAllocation ? principalLeft : allocation.quote * settlementRatio,
        principalLeft,
      ),
    );
    if (quantity <= EPSILON || principalQuote <= EPSILON) {
      continue;
    }
    const returnedQuote = roundQuote(quantity * unitProceeds);
    const short = shorts.find((lot) => lot.id === allocation.shortLotId);
    if (short) {
      const profitQuote = returnedQuote - principalQuote;
      short.lentQuote = roundQuote(Math.max(0, short.lentQuote - principalQuote));
      short.remainingQuantity = roundAsset(short.remainingQuantity + quantity);
      const lenderQuote =
        profitQuote > 0
          ? roundQuote(
              principalQuote + profitQuote * context.borrowerProfitShareToLender,
            )
          : returnedQuote;
      short.remainingProceedsQuote = roundQuote(
        short.remainingProceedsQuote + lenderQuote,
      );
    }

    allocation.quantity = roundAsset(allocation.quantity - quantity);
    allocation.quote = roundQuote(allocation.quote - principalQuote);
    quantityLeft = roundAsset(quantityLeft - quantity);
    principalLeft = roundQuote(principalLeft - principalQuote);
  }

  long.borrowAllocations = long.borrowAllocations.filter(
    (allocation) => allocation.quantity > EPSILON && allocation.quote > EPSILON,
  );
}

function longLotBreakEvenBeforeFees(long: MutableLongLot): number {
  return long.remainingQuantity > EPSILON
    ? long.remainingCostQuote / long.remainingQuantity
    : 0;
}

function shortLotBreakEvenBeforeFees(short: MutableShortLot): number {
  return short.remainingQuantity > EPSILON
    ? short.remainingProceedsQuote / short.remainingQuantity
    : 0;
}

function inheritedLongBorrowDepth(
  allocations: LongBorrowAllocation[],
  originDepth: number,
): number {
  return allocations.length > 0
    ? Math.max(...allocations.map((allocation) => allocation.depthRemaining))
    : originDepth;
}

function inheritedShortBorrowDepth(
  allocations: ShortBorrowAllocation[],
  originDepth: number,
): number {
  return allocations.length > 0
    ? Math.max(...allocations.map((allocation) => allocation.depthRemaining))
    : originDepth;
}

function appendPendingOrderLot(
  order: TradingOrder,
  feeRate: number,
  context: LedgerContext,
  longs: MutableLongLot[],
  shorts: MutableShortLot[],
): void {
  const pendingQuantity = roundAsset(Math.max(0, order.quantity - order.filledQuantity));
  if (pendingQuantity <= EPSILON) {
    return;
  }

  if (order.side === "buy") {
    const pendingQuote =
      order.estimatedQuoteCost > 0
        ? (order.estimatedQuoteCost * pendingQuantity) / order.quantity
        : pendingQuantity * order.price * (1 + feeRate);
    longs.push({
      id: `pending_long_${order.id}`,
      side: "long",
      sourceOrderId: order.id,
      openedAt: order.createdAt,
      originalQuantity: pendingQuantity,
      filledQuantity: 0,
      pendingQuantity,
      pendingQuote: roundQuote(pendingQuote),
      pendingLimitPrice: order.price,
      closedQuantity: 0,
      closedQuote: 0,
      averagePrice: roundQuote(pendingQuote / pendingQuantity),
      costQuote: roundQuote(pendingQuote),
      remainingQuantity: pendingQuantity,
      remainingCostQuote: roundQuote(pendingQuote),
      ...lotLifecycleFields(order),
      lentQuantity: 0,
      borrowDepthRemaining: context.longBorrowDepth,
      borrowAllocations: [],
    });
    return;
  }

  const pendingQuote = pendingQuantity * order.price * (1 - feeRate);
  shorts.push({
    id: `pending_short_${order.id}`,
    side: "short",
    sourceOrderId: order.id,
    openedAt: order.createdAt,
    originalQuantity: pendingQuantity,
    filledQuantity: 0,
    pendingQuantity,
    pendingQuote: roundQuote(pendingQuote),
    pendingLimitPrice: order.price,
    closedQuantity: 0,
    closedQuote: 0,
    averagePrice: roundQuote(pendingQuote / pendingQuantity),
    proceedsQuote: roundQuote(pendingQuote),
    remainingQuantity: pendingQuantity,
    remainingProceedsQuote: roundQuote(pendingQuote),
    ...lotLifecycleFields(order),
    lentQuote: 0,
    borrowDepthRemaining: context.shortBorrowDepth,
    borrowAllocations: [],
  });
}

function buildBorrowProfiles(
  longs: MutableLongLot[],
  shorts: MutableShortLot[],
  context: LedgerContext,
  state: PaperBotState,
): BorrowProfiles {
  const profiles: BorrowProfiles = {
    longs: new Map(longs.map((lot) => [lot.id, emptyBorrowProfile()])),
    shorts: new Map(shorts.map((lot) => [lot.id, emptyBorrowProfile()])),
  };

  allocateShortBaseBorrow(shorts, profiles, context);
  allocateLongQuoteBorrow(longs, profiles, state);

  return profiles;
}

function allocateShortBaseBorrow(
  shorts: MutableShortLot[],
  profiles: BorrowProfiles,
  context: LedgerContext,
): void {
  for (const short of shorts) {
    const profile = profiles.shorts.get(short.id);
    if (!profile || short.remainingQuantity <= EPSILON) {
      continue;
    }

    const borrowedFrom = new Set<string>();
    for (const allocation of short.borrowAllocations) {
      if (allocation.quantity <= EPSILON) {
        continue;
      }
      profile.internalBorrowedQuantity = roundAsset(
        profile.internalBorrowedQuantity + allocation.quantity,
      );
      borrowedFrom.add(allocation.longLotId);
    }

    const needed = Math.max(0, short.remainingQuantity - profile.internalBorrowedQuantity);
    profile.externalBorrowedQuantity = roundAsset(Math.max(0, needed));
    profile.borrowedQuantity = roundAsset(short.remainingQuantity);
    profile.internalBorrowedQuote = roundQuote(
      profile.internalBorrowedQuantity * context.currentPrice,
    );
    profile.externalBorrowedQuote = roundQuote(
      profile.externalBorrowedQuantity * context.currentPrice,
    );
    profile.borrowedQuote = roundQuote(profile.borrowedQuantity * context.currentPrice);
    profile.borrowedFromPositionCount = borrowedFrom.size;
  }
}

function allocateLongQuoteBorrow(
  longs: MutableLongLot[],
  profiles: BorrowProfiles,
  state: PaperBotState,
): void {
  let externalQuoteBudget = Math.max(0, -(state.quoteFree + state.quoteReserved));

  for (const long of longs) {
    const profile = profiles.longs.get(long.id);
    if (!profile || long.remainingQuantity <= EPSILON) {
      continue;
    }

    const borrowedFrom = new Set<string>();
    for (const allocation of long.borrowAllocations) {
      if (allocation.quote <= EPSILON || allocation.quantity <= EPSILON) {
        continue;
      }
      profile.internalBorrowedQuote = roundQuote(
        profile.internalBorrowedQuote + allocation.quote,
      );
      profile.internalBorrowedQuantity = roundAsset(
        profile.internalBorrowedQuantity + allocation.quantity,
      );
      borrowedFrom.add(allocation.shortLotId);
    }

    const externalQuote = Math.min(
      Math.max(0, long.remainingCostQuote - profile.internalBorrowedQuote),
      externalQuoteBudget,
    );
    if (externalQuote > EPSILON) {
      profile.externalBorrowedQuote = roundQuote(externalQuote);
      externalQuoteBudget = roundQuote(externalQuoteBudget - externalQuote);
    }

    profile.borrowedQuote = roundQuote(
      profile.internalBorrowedQuote + profile.externalBorrowedQuote,
    );
    profile.externalBorrowedQuantity =
      long.averagePrice > EPSILON
        ? roundAsset(profile.externalBorrowedQuote / long.averagePrice)
        : 0;
    profile.borrowedQuantity = roundAsset(
      profile.internalBorrowedQuantity + profile.externalBorrowedQuantity,
    );
    profile.borrowedFromPositionCount = borrowedFrom.size;
  }
}

function finalizeLongLot(
  lot: MutableLongLot,
  context: LedgerContext,
  borrow: BorrowProfile,
): LongPositionLot {
  const hasRemainingQuantity = lot.remainingQuantity > EPSILON;
  const denominator = Math.max(lot.remainingQuantity, EPSILON);
  const exposureQuote = roundQuote(lot.remainingQuantity * context.currentPrice);
  const breakEvenSellPrice = hasRemainingQuantity
    ? (lot.remainingCostQuote / denominator) * (1 + context.feeAndSlippageRate)
    : 0;
  const maxLossSellPrice =
    hasRemainingQuantity
      ? (Math.max(0, lot.remainingCostQuote - context.maxLossPct * lot.costQuote) / denominator) *
        (1 + context.feeAndSlippageRate)
      : 0;
  const sellDenominator = context.netMarketSellPrice - context.lowerBaselinePrice;
  const recommendedSellQuote =
    sellDenominator > EPSILON && context.netMarketSellPrice > 0
      ? Math.max(
          0,
          ((lot.remainingCostQuote - context.lowerBaselinePrice * lot.remainingQuantity) /
            sellDenominator) *
            context.netMarketSellPrice,
        )
      : 0;
  const recommendedSellQuantity =
    context.netMarketSellPrice > 0 ? recommendedSellQuote / context.netMarketSellPrice : 0;
  const canReachLowerBaseline =
    recommendedSellQuote > EPSILON &&
    lot.remainingCostQuote > 0 &&
    recommendedSellQuote < lot.remainingCostQuote &&
    recommendedSellQuantity <= lot.remainingQuantity + EPSILON;
  const projectedSellQuote = canReachLowerBaseline ? recommendedSellQuote : 0;
  const projectedSellQuantity = canReachLowerBaseline ? recommendedSellQuantity : 0;
  const projectedRemainingQuantity = roundAsset(
    Math.max(0, lot.remainingQuantity - projectedSellQuantity),
  );
  const projectedRemainingCostQuote = roundQuote(lot.remainingCostQuote - projectedSellQuote);

  return {
    ...lot,
    status: lotStatus(
      lot.filledQuantity,
      lot.pendingQuantity,
      lot.remainingQuantity,
      lot.lentQuantity,
    ),
    closedQuantity: roundAsset(lot.closedQuantity),
    closedQuote: roundQuote(lot.closedQuote),
    costQuote: roundQuote(lot.costQuote),
    remainingCostQuote: roundQuote(lot.remainingCostQuote),
    exposureQuote,
    leverage: calculateLeverage(exposureQuote, borrow.externalBorrowedQuote),
    ...roundBorrowProfile(borrow),
    expiresAt: lotExpiresAt(lot),
    borrowLocked: hasLotLifecycleControls(lot),
    breakEvenSellPrice: roundQuote(breakEvenSellPrice),
    maxLossSellPrice: roundQuote(maxLossSellPrice),
    recommendedSellQuote: roundQuote(recommendedSellQuote),
    recommendedSellQuantity: roundAsset(recommendedSellQuantity),
    projectedRemainingQuantity,
    projectedRemainingCostQuote,
    projectedBreakEvenSellPrice:
      projectedRemainingQuantity > EPSILON
        ? roundQuote(
            (projectedRemainingCostQuote / projectedRemainingQuantity) *
              (1 + context.feeAndSlippageRate),
          )
        : 0,
    canReachLowerBaseline,
  };
}

function finalizeShortLot(
  lot: MutableShortLot,
  context: LedgerContext,
  borrow: BorrowProfile,
): ShortPositionLot {
  const denominator = Math.max(lot.remainingQuantity, EPSILON);
  const exposureQuote = roundQuote(lot.remainingQuantity * context.currentPrice);
  const feeFactor = (1 + context.feeAndSlippageRate) ** 2;
  const breakEvenBuyPrice = lot.remainingProceedsQuote / denominator / feeFactor;
  const maxLossDenominator = lot.remainingQuantity - context.maxLossPct * lot.originalQuantity;
  const maxLossBuyPrice =
    maxLossDenominator > EPSILON
      ? Math.max(0, lot.remainingProceedsQuote / maxLossDenominator) / feeFactor
      : 0;
  const buyDenominator = context.grossMarketBuyPrice - context.upperBaselinePrice;
  const rawRecommendedBuyQuote =
    Math.abs(buyDenominator) > EPSILON && context.grossMarketBuyPrice > 0
      ? ((lot.remainingProceedsQuote - context.upperBaselinePrice * lot.remainingQuantity) /
          buyDenominator) *
        context.grossMarketBuyPrice
      : 0;
  const recommendedBuyQuote = Math.max(0, rawRecommendedBuyQuote);
  const recommendedBuyQuantity =
    context.grossMarketBuyPrice > 0 ? recommendedBuyQuote / context.grossMarketBuyPrice : 0;
  const canReachUpperBaseline =
    recommendedBuyQuote > EPSILON &&
    lot.remainingQuantity > EPSILON &&
    maxLossBuyPrice > 0 &&
    recommendedBuyQuote < lot.remainingProceedsQuote &&
    recommendedBuyQuantity <= lot.remainingQuantity + EPSILON;
  const projectedBuyQuote = canReachUpperBaseline ? recommendedBuyQuote : 0;
  const projectedBuyQuantity = canReachUpperBaseline ? recommendedBuyQuantity : 0;
  const projectedRemainingQuantity = roundAsset(
    Math.max(0, lot.remainingQuantity - projectedBuyQuantity),
  );
  const projectedRemainingProceedsQuote = roundQuote(
    lot.remainingProceedsQuote - projectedBuyQuote,
  );

  return {
    ...lot,
    status: lotStatus(
      lot.filledQuantity,
      lot.pendingQuantity,
      lot.remainingQuantity,
      lot.lentQuote,
    ),
    closedQuantity: roundAsset(lot.closedQuantity),
    closedQuote: roundQuote(lot.closedQuote),
    proceedsQuote: roundQuote(lot.proceedsQuote),
    remainingProceedsQuote: roundQuote(lot.remainingProceedsQuote),
    exposureQuote,
    leverage: calculateLeverage(exposureQuote, borrow.externalBorrowedQuote),
    ...roundBorrowProfile(borrow),
    expiresAt: lotExpiresAt(lot),
    borrowLocked: hasLotLifecycleControls(lot),
    breakEvenBuyPrice: roundQuote(breakEvenBuyPrice),
    maxLossBuyPrice: roundQuote(maxLossBuyPrice),
    recommendedBuyQuote: roundQuote(recommendedBuyQuote),
    recommendedBuyQuantity: roundAsset(recommendedBuyQuantity),
    projectedRemainingQuantity,
    projectedRemainingProceedsQuote,
    projectedBreakEvenBuyPrice:
      projectedRemainingQuantity > EPSILON
        ? roundQuote(projectedRemainingProceedsQuote / projectedRemainingQuantity / feeFactor)
        : 0,
    canReachUpperBaseline,
  };
}

function emptyBorrowProfile(): BorrowProfile {
  return {
    borrowedQuantity: 0,
    borrowedQuote: 0,
    internalBorrowedQuantity: 0,
    internalBorrowedQuote: 0,
    externalBorrowedQuantity: 0,
    externalBorrowedQuote: 0,
    borrowedFromPositionCount: 0,
  };
}

function roundBorrowProfile(profile: BorrowProfile): BorrowProfile {
  return {
    borrowedQuantity: roundAsset(profile.borrowedQuantity),
    borrowedQuote: roundQuote(profile.borrowedQuote),
    internalBorrowedQuantity: roundAsset(profile.internalBorrowedQuantity),
    internalBorrowedQuote: roundQuote(profile.internalBorrowedQuote),
    externalBorrowedQuantity: roundAsset(profile.externalBorrowedQuantity),
    externalBorrowedQuote: roundQuote(profile.externalBorrowedQuote),
    borrowedFromPositionCount: profile.borrowedFromPositionCount,
  };
}

function calculateLeverage(exposureQuote: number, externalBorrowedQuote: number): number {
  if (exposureQuote <= EPSILON || externalBorrowedQuote <= EPSILON) {
    return 1;
  }

  const ownExposureQuote = exposureQuote - externalBorrowedQuote;
  if (ownExposureQuote <= EPSILON) {
    return 999;
  }

  return roundLeverage(clamp(exposureQuote / ownExposureQuote, 1, 999));
}

function roundLeverage(value: number): number {
  return Number((Number.isFinite(value) ? value : 999).toFixed(4));
}

function lotStatus(
  filledQuantity: number,
  pendingQuantity: number,
  remainingQuantity: number,
  outstandingLent = 0,
): "pending" | "open" | "partially-closed" | "closed" {
  if (pendingQuantity > EPSILON && filledQuantity <= EPSILON) {
    return "pending";
  }
  if (
    remainingQuantity <= EPSILON &&
    pendingQuantity <= EPSILON &&
    outstandingLent <= EPSILON
  ) {
    return "closed";
  }
  if (filledQuantity > EPSILON && remainingQuantity < filledQuantity - EPSILON) {
    return "partially-closed";
  }
  return "open";
}

function chronologicalFills(fills: TradeFill[]): TradeFill[] {
  return fills.slice().sort((a, b) => {
    if (a.filledAt !== b.filledAt) {
      return a.filledAt - b.filledAt;
    }

    return a.id.localeCompare(b.id);
  });
}

function lotLifecycleFields(
  source: Pick<TradingOrder | TradeFill, "lifetimeMs" | "stopLossPrice" | "takeProfitPrice">,
): Pick<
  LongPositionLot,
  "lifetimeMs" | "stopLossPrice" | "takeProfitPrice"
> {
  const lifetimeMs = cleanPositive(source.lifetimeMs);
  const stopLossPrice = cleanPositive(source.stopLossPrice);
  const takeProfitPrice = cleanPositive(source.takeProfitPrice);

  return {
    ...(lifetimeMs > 0 ? { lifetimeMs } : {}),
    ...(stopLossPrice > 0 ? { stopLossPrice } : {}),
    ...(takeProfitPrice > 0 ? { takeProfitPrice } : {}),
  };
}

function hasLotLifecycleControls(
  lot: Pick<LongPositionLot | ShortPositionLot, "lifetimeMs" | "stopLossPrice" | "takeProfitPrice">,
): boolean {
  return (
    cleanPositive(lot.lifetimeMs) > 0 ||
    cleanPositive(lot.stopLossPrice) > 0 ||
    cleanPositive(lot.takeProfitPrice) > 0
  );
}

function lotExpiresAt(
  lot: Pick<LongPositionLot | ShortPositionLot, "openedAt" | "lifetimeMs">,
): number | undefined {
  const lifetimeMs = cleanPositive(lot.lifetimeMs);
  return lifetimeMs > 0 ? lot.openedAt + lifetimeMs : undefined;
}

function cleanNumber(value: number): number {
  return Number.isFinite(value) ? value : 0;
}

function cleanNonNegative(value: number): number {
  return Math.max(0, cleanNumber(value));
}

function cleanPositive(value: number | undefined): number {
  return Number.isFinite(value) && (value as number) > 0 ? (value as number) : 0;
}

function normalizeBorrowDepth(value: number | undefined): number {
  if (!Number.isFinite(value)) {
    return 0;
  }
  return Math.max(0, Math.round(value ?? 0));
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function roundAsset(value: number): number {
  return Number((Number.isFinite(value) ? value : 0).toFixed(8));
}

function roundQuote(value: number): number {
  return Number((Number.isFinite(value) ? value : 0).toFixed(6));
}

function sum<T>(items: T[], key: keyof T): number {
  return items.reduce((total, item) => {
    const value = item[key];
    return total + (typeof value === "number" ? value : 0);
  }, 0);
}
