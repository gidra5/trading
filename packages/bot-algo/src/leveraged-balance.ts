import type { PaperBotState, ShortMarginModel } from "./types.js";

export type BalanceEntrySide = "buy" | "sell";

export interface BalanceEntryCapacityInput {
  state: PaperBotState;
  marketPrice: number;
  side: BalanceEntrySide;
  targetLeverage: number;
  feeRate: number;
  sidePositionCapacityQuote: number;
  pendingLongEntryQuote: number;
  pendingShortEntryQuote: number;
}

export interface BalanceLiquidationInput {
  state: PaperBotState;
  feeRate: number;
  slippageRate: number;
}

export interface BalanceProjection {
  quoteBalance: number;
  baseQuantity: number;
  equity: number;
  netExposureQuote: number;
  debtQuote: number;
  effectiveLeverage: number;
}

interface BalanceEntry {
  side: BalanceEntrySide;
  quote: number;
}

const EPSILON = 1e-10;
const MAX_LEVERAGE = 999;

const roundQuote = (value: number) => Number(value.toFixed(6));

export abstract class LeveragedBalanceModel {
  abstract readonly model: ShortMarginModel;

  entryCapacityQuote(input: BalanceEntryCapacityInput): number {
    if (
      input.marketPrice <= 0 ||
      input.targetLeverage <= 0 ||
      input.sidePositionCapacityQuote <= EPSILON
    ) {
      return 0;
    }

    const cappedLeverage = clamp(input.targetLeverage, 1, MAX_LEVERAGE);
    const positionCapacity = Number.isFinite(input.sidePositionCapacityQuote)
      ? Math.max(0, input.sidePositionCapacityQuote)
      : this.defaultCapacityUpperBound(input, cappedLeverage);
    if (positionCapacity <= EPSILON) {
      return 0;
    }

    const pendingEntries = this.pendingEntries(input);
    const pendingProjection = this.project(input.state, input.marketPrice, input.feeRate, pendingEntries);
    if (pendingProjection.equity <= EPSILON) {
      return 0;
    }

    const upper = Math.min(
      positionCapacity,
      Math.max(positionCapacity, this.defaultCapacityUpperBound(input, cappedLeverage)),
    );
    if (upper <= EPSILON) {
      return 0;
    }

    const canAdmit = (quote: number) => {
      const projection = this.project(input.state, input.marketPrice, input.feeRate, [
        ...pendingEntries,
        { side: input.side, quote },
      ]);
      return (
        projection.equity > EPSILON &&
        projection.effectiveLeverage <= cappedLeverage + 0.0001
      );
    };

    if (canAdmit(upper)) {
      return roundQuote(upper);
    }

    let low = 0;
    let high = upper;
    for (let index = 0; index < 48; index += 1) {
      const mid = (low + high) / 2;
      if (canAdmit(mid)) {
        low = mid;
      } else {
        high = mid;
      }
    }

    return roundQuote(low);
  }

  liquidationPrice(input: BalanceLiquidationInput): number | undefined {
    const quoteBalance = totalQuoteBalance(input.state);
    const baseQuantity = totalBaseQuantity(input.state);
    if (baseQuantity > EPSILON && quoteBalance < -EPSILON) {
      const proceedsRate = Math.max(
        0.000001,
        (1 - Math.max(0, input.slippageRate)) * (1 - Math.max(0, input.feeRate)),
      );
      return roundQuote(-quoteBalance / (baseQuantity * proceedsRate));
    }
    if (baseQuantity < -EPSILON && quoteBalance > EPSILON) {
      const costRate = (1 + Math.max(0, input.slippageRate)) * (1 + Math.max(0, input.feeRate));
      return roundQuote(quoteBalance / (-baseQuantity * costRate));
    }
    return undefined;
  }

  project(
    state: PaperBotState,
    marketPrice: number,
    feeRate: number,
    entries: readonly BalanceEntry[] = [],
  ): BalanceProjection {
    let quoteBalance = totalQuoteBalance(state);
    let baseQuantity = totalBaseQuantity(state);

    for (const entry of entries) {
      const quote = Math.max(0, entry.quote);
      if (quote <= EPSILON) {
        continue;
      }
      if (entry.side === "buy") {
        quoteBalance -= quote * (1 + feeRate);
        baseQuantity += quote / marketPrice;
      } else {
        quoteBalance += quote * (1 - feeRate);
        baseQuantity -= quote / marketPrice;
      }
    }

    quoteBalance = roundQuote(quoteBalance);
    const netExposureQuote = roundQuote(baseQuantity * marketPrice);
    const equity = roundQuote(quoteBalance + netExposureQuote);
    const debtQuote = this.debtQuote(quoteBalance, baseQuantity, marketPrice);
    const projection = {
      quoteBalance,
      baseQuantity,
      equity,
      netExposureQuote,
      debtQuote,
    };
    return {
      ...projection,
      effectiveLeverage: this.effectiveLeverageForProjection(projection),
    };
  }

  effectiveLeverage(state: PaperBotState, marketPrice: number): number {
    if (marketPrice <= 0) {
      return 1;
    }
    return this.project(state, marketPrice, 0).effectiveLeverage;
  }

  riskExposureQuote(state: PaperBotState, marketPrice: number): number {
    if (marketPrice <= 0) {
      return 0;
    }
    const projection = this.project(state, marketPrice, 0);
    return roundQuote(Math.abs(projection.netExposureQuote));
  }

  projectedEffectiveLeverage(
    state: PaperBotState,
    marketPrice: number,
    feeRate: number,
    pendingLongEntryQuote: number,
    pendingShortEntryQuote: number,
  ): number {
    return this.project(state, marketPrice, feeRate, [
      ...entryIfPositive("buy", pendingLongEntryQuote),
      ...entryIfPositive("sell", pendingShortEntryQuote),
    ]).effectiveLeverage;
  }

  protected abstract debtQuote(
    quoteBalance: number,
    baseQuantity: number,
    marketPrice: number,
  ): number;

  protected abstract effectiveLeverageForProjection(
    projection: Omit<BalanceProjection, "effectiveLeverage">,
  ): number;

  private pendingEntries(input: BalanceEntryCapacityInput): BalanceEntry[] {
    return [
      ...entryIfPositive("buy", input.pendingLongEntryQuote),
      ...entryIfPositive("sell", input.pendingShortEntryQuote),
    ];
  }

  private defaultCapacityUpperBound(
    input: BalanceEntryCapacityInput,
    leverage: number,
  ): number {
    const pendingProjection = this.project(input.state, input.marketPrice, input.feeRate, [
      ...entryIfPositive("buy", input.pendingLongEntryQuote),
      ...entryIfPositive("sell", input.pendingShortEntryQuote),
    ]);
    if (pendingProjection.equity <= EPSILON) {
      return 0;
    }
    return roundQuote(
      Math.max(
        input.state.config.minOrderQuote,
        leverage * pendingProjection.equity + Math.abs(pendingProjection.netExposureQuote),
      ),
    );
  }
}

export class FuturesMarginBalanceModel extends LeveragedBalanceModel {
  readonly model = "futures-margin" as const;

  override entryCapacityQuote(input: BalanceEntryCapacityInput): number {
    if (
      input.marketPrice <= 0 ||
      input.targetLeverage <= 0 ||
      input.sidePositionCapacityQuote <= EPSILON
    ) {
      return 0;
    }

    const leverage = clamp(input.targetLeverage, 1, MAX_LEVERAGE);
    const pendingProjection = this.project(input.state, input.marketPrice, input.feeRate, [
      ...entryIfPositive("buy", input.pendingLongEntryQuote),
      ...entryIfPositive("sell", input.pendingShortEntryQuote),
    ]);
    if (pendingProjection.equity <= EPSILON) {
      return 0;
    }

    const sideSign = input.side === "buy" ? 1 : -1;
    const leverageCapacity = Math.max(
      0,
      (leverage * pendingProjection.equity -
        sideSign * pendingProjection.netExposureQuote) /
        (1 + leverage * input.feeRate),
    );
    const feeCapacity =
      input.feeRate > EPSILON
        ? (pendingProjection.equity / input.feeRate) * 0.999999
        : Number.POSITIVE_INFINITY;

    return roundQuote(
      Math.min(
        Number.isFinite(input.sidePositionCapacityQuote)
          ? Math.max(0, input.sidePositionCapacityQuote)
          : Number.POSITIVE_INFINITY,
        leverageCapacity,
        feeCapacity,
      ),
    );
  }

  protected debtQuote(): number {
    return 0;
  }

  protected effectiveLeverageForProjection(
    projection: Omit<BalanceProjection, "effectiveLeverage">,
  ): number {
    const exposureQuote = Math.abs(projection.netExposureQuote);
    if (exposureQuote <= EPSILON) {
      return 1;
    }
    if (projection.equity <= EPSILON) {
      return MAX_LEVERAGE;
    }
    return roundLeverage(clamp(exposureQuote / projection.equity, 1, MAX_LEVERAGE));
  }
}

export class SpotBorrowBalanceModel extends LeveragedBalanceModel {
  readonly model = "spot-borrow" as const;

  protected debtQuote(
    quoteBalance: number,
    baseQuantity: number,
    marketPrice: number,
  ): number {
    return roundQuote(
      Math.max(0, -quoteBalance) +
        Math.max(0, -baseQuantity * marketPrice),
    );
  }

  protected effectiveLeverageForProjection(
    projection: Omit<BalanceProjection, "effectiveLeverage">,
  ): number {
    if (projection.debtQuote <= EPSILON) {
      return 1;
    }
    if (projection.equity <= EPSILON) {
      return MAX_LEVERAGE;
    }
    return roundLeverage(clamp(1 + projection.debtQuote / projection.equity, 1, MAX_LEVERAGE));
  }
}

export function createLeveragedBalanceModel(model: ShortMarginModel): LeveragedBalanceModel {
  return model === "spot-borrow"
    ? new SpotBorrowBalanceModel()
    : new FuturesMarginBalanceModel();
}

function totalQuoteBalance(state: PaperBotState): number {
  return state.quoteFree + state.quoteReserved;
}

function totalBaseQuantity(state: PaperBotState): number {
  return state.baseFree + state.baseReserved;
}

function entryIfPositive(side: BalanceEntrySide, quote: number): BalanceEntry[] {
  return quote > EPSILON ? [{ side, quote }] : [];
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function roundLeverage(value: number): number {
  return Number((Number.isFinite(value) ? value : MAX_LEVERAGE).toFixed(4));
}
