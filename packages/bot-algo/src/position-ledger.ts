import type {
  LongPositionLot,
  PaperBotState,
  PositionLedger,
  PositionRiskConfig,
  ShortPositionLot,
  TradeFill,
  TradingOrder,
} from "./types.js";

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
  quantityFloor: number;
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
>;

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
>;

export const defaultPositionRiskConfig: PositionRiskConfig = {
  lowerPriceExpectation: 0,
  lowerBaselinePrice: 0,
  upperPriceExpectation: 0,
  upperBaselinePrice: 0,
  maxLossPct: 0.02,
  marketSlippageBps: 10,
  quantityFloor: 0.01,
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
  config.quantityFloor = Math.max(EPSILON, cleanNonNegative(config.quantityFloor));

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
      applyBuyFill(fill, orderById.get(fill.orderId), shorts, longs);
    } else {
      applySellFill(fill, orderById.get(fill.orderId), longs, shorts);
    }
  }

  for (const order of state.orders) {
    if (order.status === "open") {
      appendPendingOrderLot(order, state.config.feeBps / 10_000, longs, shorts);
    }
  }

  const finalizedLongs = longs.map((lot) => finalizeLongLot(lot, context));
  const finalizedShorts = shorts.map((lot) => finalizeShortLot(lot, context));
  const activeLongs = finalizedLongs.filter((lot) => lot.status !== "pending");
  const activeShorts = finalizedShorts.filter((lot) => lot.status !== "pending");

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
      longQuantity: roundAsset(sum(activeLongs, "remainingQuantity")),
      shortQuantity: roundAsset(sum(activeShorts, "remainingQuantity")),
      longRemainingCostQuote: roundQuote(sum(activeLongs, "remainingCostQuote")),
      shortRemainingProceedsQuote: roundQuote(sum(activeShorts, "remainingProceedsQuote")),
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
    quantityFloor: risk.quantityFloor,
  };
}

function applyBuyFill(
  fill: TradeFill,
  sourceOrder: TradingOrder | undefined,
  shorts: MutableShortLot[],
  longs: MutableLongLot[],
): void {
  const unitCost = (fill.quoteQuantity + fill.feeQuote) / fill.quantity;
  let quantityLeft = fill.quantity;
  let costLeft = fill.quoteQuantity + fill.feeQuote;

  if (fill.targetPositionId) {
    const target = shorts.find((short) => short.id === fill.targetPositionId);
    if (target) {
      const closed = closeShortLot(target, quantityLeft, unitCost);
      quantityLeft = roundAsset(quantityLeft - closed.quantity);
      costLeft = roundQuote(costLeft - closed.quote);
    }
  }

  for (const short of shorts) {
    if (quantityLeft <= EPSILON) {
      break;
    }
    if (short.remainingQuantity <= EPSILON) {
      continue;
    }
    const closed = closeShortLot(short, quantityLeft, unitCost);
    quantityLeft = roundAsset(quantityLeft - closed.quantity);
    costLeft = roundQuote(costLeft - closed.quote);
  }

  if (quantityLeft > EPSILON && costLeft > EPSILON) {
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
    });
  }
}

function applySellFill(
  fill: TradeFill,
  sourceOrder: TradingOrder | undefined,
  longs: MutableLongLot[],
  shorts: MutableShortLot[],
): void {
  const unitProceeds = (fill.quoteQuantity - fill.feeQuote) / fill.quantity;
  let quantityLeft = fill.quantity;
  let proceedsLeft = fill.quoteQuantity - fill.feeQuote;

  if (fill.targetPositionId) {
    const target = longs.find((long) => long.id === fill.targetPositionId);
    if (target) {
      const closed = closeLongLot(target, quantityLeft, unitProceeds);
      quantityLeft = roundAsset(quantityLeft - closed.quantity);
      proceedsLeft = roundQuote(proceedsLeft - closed.quote);
    }
  }

  for (const long of longs) {
    if (quantityLeft <= EPSILON) {
      break;
    }
    if (long.remainingQuantity <= EPSILON) {
      continue;
    }
    const closed = closeLongLot(long, quantityLeft, unitProceeds);
    quantityLeft = roundAsset(quantityLeft - closed.quantity);
    proceedsLeft = roundQuote(proceedsLeft - closed.quote);
  }

  if (quantityLeft > EPSILON && proceedsLeft > EPSILON) {
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
    });
  }
}

function closeShortLot(
  short: MutableShortLot,
  requestedQuantity: number,
  unitCost: number,
): { quantity: number; quote: number } {
  const quantity = Math.min(requestedQuantity, short.remainingQuantity);
  const quote = quantity * unitCost;
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
): { quantity: number; quote: number } {
  const quantity = Math.min(requestedQuantity, long.remainingQuantity);
  const quote = quantity * unitProceeds;
  long.remainingQuantity = roundAsset(long.remainingQuantity - quantity);
  long.remainingCostQuote = roundQuote(long.remainingCostQuote - quote);
  long.closedQuantity = roundAsset(long.closedQuantity + quantity);
  long.closedQuote = roundQuote(long.closedQuote + quote);
  return {
    quantity: roundAsset(quantity),
    quote: roundQuote(quote),
  };
}

function appendPendingOrderLot(
  order: TradingOrder,
  feeRate: number,
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
  });
}

function finalizeLongLot(lot: MutableLongLot, context: LedgerContext): LongPositionLot {
  const denominator = Math.max(lot.remainingQuantity, context.quantityFloor);
  const breakEvenSellPrice = (lot.remainingCostQuote / denominator) * (1 + context.feeAndSlippageRate);
  const maxLossSellPrice =
    (Math.max(0, lot.remainingCostQuote - context.maxLossPct * lot.costQuote) / denominator) *
    (1 + context.feeAndSlippageRate);
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
    status: lotStatus(lot.filledQuantity, lot.pendingQuantity, lot.remainingQuantity),
    closedQuantity: roundAsset(lot.closedQuantity),
    closedQuote: roundQuote(lot.closedQuote),
    costQuote: roundQuote(lot.costQuote),
    remainingCostQuote: roundQuote(lot.remainingCostQuote),
    breakEvenSellPrice: roundQuote(breakEvenSellPrice),
    maxLossSellPrice: roundQuote(maxLossSellPrice),
    recommendedSellQuote: roundQuote(recommendedSellQuote),
    recommendedSellQuantity: roundAsset(recommendedSellQuantity),
    projectedRemainingQuantity,
    projectedRemainingCostQuote,
    projectedBreakEvenSellPrice: roundQuote(
      (projectedRemainingCostQuote / Math.max(projectedRemainingQuantity, context.quantityFloor)) *
        (1 + context.feeAndSlippageRate),
    ),
    canReachLowerBaseline,
  };
}

function finalizeShortLot(lot: MutableShortLot, context: LedgerContext): ShortPositionLot {
  const denominator = Math.max(lot.remainingQuantity, context.quantityFloor);
  const feeFactor = (1 + context.feeAndSlippageRate) ** 2;
  const breakEvenBuyPrice = lot.remainingProceedsQuote / denominator / feeFactor;
  const maxLossBuyPrice =
    Math.max(
      0,
      lot.remainingProceedsQuote /
        Math.max(
          lot.remainingQuantity - context.maxLossPct * lot.originalQuantity,
          context.quantityFloor,
        ),
    ) / feeFactor;
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
    status: lotStatus(lot.filledQuantity, lot.pendingQuantity, lot.remainingQuantity),
    closedQuantity: roundAsset(lot.closedQuantity),
    closedQuote: roundQuote(lot.closedQuote),
    proceedsQuote: roundQuote(lot.proceedsQuote),
    remainingProceedsQuote: roundQuote(lot.remainingProceedsQuote),
    breakEvenBuyPrice: roundQuote(breakEvenBuyPrice),
    maxLossBuyPrice: roundQuote(maxLossBuyPrice),
    recommendedBuyQuote: roundQuote(recommendedBuyQuote),
    recommendedBuyQuantity: roundAsset(recommendedBuyQuantity),
    projectedRemainingQuantity,
    projectedRemainingProceedsQuote,
    projectedBreakEvenBuyPrice: roundQuote(
      projectedRemainingProceedsQuote /
        Math.max(projectedRemainingQuantity, context.quantityFloor) /
        feeFactor,
    ),
    canReachUpperBaseline,
  };
}

function lotStatus(
  filledQuantity: number,
  pendingQuantity: number,
  remainingQuantity: number,
): "pending" | "open" | "partially-closed" | "closed" {
  if (pendingQuantity > EPSILON && filledQuantity <= EPSILON) {
    return "pending";
  }
  if (remainingQuantity <= EPSILON && pendingQuantity <= EPSILON) {
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

function cleanNumber(value: number): number {
  return Number.isFinite(value) ? value : 0;
}

function cleanNonNegative(value: number): number {
  return Math.max(0, cleanNumber(value));
}

function cleanPositive(value: number | undefined): number {
  return Number.isFinite(value) && (value as number) > 0 ? (value as number) : 0;
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
