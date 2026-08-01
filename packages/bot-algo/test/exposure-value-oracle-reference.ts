import type { ExposureValueOracleOptions } from "../src/exposure-value-distillation.js";

/**
 * Deliberately slow, literal implementation of the rolling exposure-value
 * oracle contract. Keep this independent of the production Bellman helpers:
 * optimized CPU and CUDA implementations are tested against it.
 *
 * The canonical recurrence separates the initially forced H-step hold from a
 * continuation policy whose decisions are delayed by D candles:
 *
 *   V(t, 0, x) = log rebalance(x -> 0)
 *   d = min(D, k)
 *   V(t, k, x) = max_b [
 *     log rebalance(x -> b)
 *     + hold(t, t + d, b)
 *     + V(t + d, k - d, drift(t, t + d, b))
 *   ]
 *   F(t, H, T, a) = hold(t, t + H, a)
 *     + V(t + H, T - H, drift(t, t + H, a))
 *
 * With C = T - H continuation candles, the policy makes ceil(C / D)
 * continuation decisions. The literal search costs
 * O(timestamps * ceil(C / D) * actionGrid²), plus the candle-by-candle holding
 * simulations, so this implementation is intended only for small correctness
 * fixtures.
 */
export interface BruteForceExposureValueOracle {
  grid: Float64Array;
  actionValues: Float64Array;
  probabilities: Float64Array;
}

interface HoldingResult {
  portfolio: PortfolioState;
}

interface PortfolioState {
  /** Quote-asset quantity; negative values are borrowed quote debt. */
  quote: number;
  /** Base-asset quantity; negative values are borrowed asset debt. */
  asset: number;
}

export function prepareBruteForceExposureValueOracle(
  prices: ArrayLike<number>,
  options: ExposureValueOracleOptions,
): BruteForceExposureValueOracle {
  if (prices.length < 2) throw new Error("Reference oracle requires at least two prices.");
  const holdingPeriodSteps = options.holdingPeriodSteps ?? 1;
  const decisionDelaySteps = options.decisionDelaySteps ?? 1;
  const valueHorizonSteps = options.valueHorizonSteps ?? holdingPeriodSteps;
  const minimumExposure = options.minExposure ?? -1;
  const maximumExposure = options.maxExposure ?? 1;
  const execution = {
    friction: options.friction,
    maxEffectiveExposure: options.maxEffectiveExposure ?? 250,
    quoteBorrowRate: options.quoteBorrowRate ?? 0,
    assetBorrowRate: options.assetBorrowRate ?? 0,
  };
  const grid = Float64Array.from(
    { length: options.gridSize },
    (_, index) => minimumExposure
      + index / (options.gridSize - 1) * (maximumExposure - minimumExposure),
  );
  const actionValues = new Float64Array(prices.length * grid.length);
  const probabilities = new Float64Array(actionValues.length);
  const finalTime = prices.length - 1;
  const memoizedDecisionForcedValues = new Map<string, number>();

  const continuationValue = (
    time: number,
    portfolio: PortfolioState,
    remainingSteps: number,
  ): number => {
    if (time >= finalTime || remainingSteps <= 0) {
      const closeout = rebalancePortfolio(
        portfolio,
        prices[time]!,
        0,
        execution.friction,
      );
      const finalEquity = closeout
        ? portfolioEquity(closeout, prices[time]!)
        : Number.NEGATIVE_INFINITY;
      return finalEquity > 0 ? Math.log(finalEquity) : Number.NEGATIVE_INFINITY;
    }
    let best = Number.NEGATIVE_INFINITY;
    for (let targetIndex = 0; targetIndex < grid.length; targetIndex += 1) {
      const rebalanced = rebalancePortfolio(
        portfolio,
        prices[time]!,
        grid[targetIndex]!,
        execution.friction,
      );
      if (!rebalanced) continue;
      const rebalancedEquity = portfolioEquity(rebalanced, prices[time]!);
      if (!(rebalancedEquity > 0)) continue;
      const forced = decisionForcedValue(time, targetIndex, remainingSteps);
      if (!Number.isFinite(forced)) continue;
      best = Math.max(best, Math.log(rebalancedEquity) + forced);
    }
    return best;
  };

  const decisionForcedValue = (
    time: number,
    targetIndex: number,
    remainingSteps: number,
  ): number => {
    const duration = Math.min(decisionDelaySteps, remainingSteps, finalTime - time);
    const key = `${time}:${targetIndex}:${remainingSteps}:${duration}`;
    const memoized = memoizedDecisionForcedValues.get(key);
    if (memoized !== undefined) return memoized;
    const holding = holdPortfolio(
      prices,
      time,
      time + duration,
      portfolioAtExposure(1, grid[targetIndex]!, prices[time]!),
      execution,
    );
    if (!holding) {
      memoizedDecisionForcedValues.set(key, Number.NEGATIVE_INFINITY);
      return Number.NEGATIVE_INFINITY;
    }
    const result = continuationValue(
      time + duration,
      holding.portfolio,
      remainingSteps - duration,
    );
    memoizedDecisionForcedValues.set(key, result);
    return result;
  };

  for (let time = options.scoreStartIndex; time < prices.length; time += 1) {
    const row = time * grid.length;
    const availableSteps = Math.min(valueHorizonSteps, finalTime - time);
    const initialHoldingSteps = Math.min(holdingPeriodSteps, availableSteps);
    const endpointTime = time + initialHoldingSteps;
    const continuationSteps = availableSteps - initialHoldingSteps;
    let maximum = Number.NEGATIVE_INFINITY;
    for (let targetIndex = 0; targetIndex < grid.length; targetIndex += 1) {
      const holding = holdPortfolio(
        prices,
        time,
        endpointTime,
        portfolioAtExposure(1, grid[targetIndex]!, prices[time]!),
        execution,
      );
      const value = holding
        ? continuationValue(
            endpointTime,
            holding.portfolio,
            continuationSteps,
          )
        : Number.NEGATIVE_INFINITY;
      actionValues[row + targetIndex] = value;
      maximum = Math.max(maximum, value);
    }
    if (!Number.isFinite(maximum)) {
      throw new Error(`Reference oracle found no feasible action at time ${time}.`);
    }
    let total = 0;
    for (let targetIndex = 0; targetIndex < grid.length; targetIndex += 1) {
      const value = actionValues[row + targetIndex]!;
      const weight = Number.isFinite(value)
        ? Math.exp((value - maximum) / options.temperature)
        : 0;
      probabilities[row + targetIndex] = weight;
      total += weight;
    }
    for (let targetIndex = 0; targetIndex < grid.length; targetIndex += 1) {
      probabilities[row + targetIndex] /= total;
    }
  }

  return { grid, actionValues, probabilities };
}

/**
 * Construct exact quote and base-asset quantities at a marked exposure.
 */
function portfolioAtExposure(
  equity: number,
  exposure: number,
  price: number,
): PortfolioState {
  return {
    quote: equity * (1 - exposure),
    asset: equity * exposure / price,
  };
}

function portfolioEquity(portfolio: PortfolioState, price: number): number {
  return portfolio.quote + portfolio.asset * price;
}

/**
 * Execute the fee-aware trade in amount space. A buy spends gross quote and
 * receives `(1 - friction)` times that notional in base asset; a sell removes
 * gross base notional and credits `(1 - friction)` times it in quote.
 */
function rebalancePortfolio(
  portfolio: PortfolioState,
  price: number,
  targetExposure: number,
  friction: number,
): PortfolioState | null {
  const equity = portfolioEquity(portfolio, price);
  if (!(equity > 0) || !Number.isFinite(equity)) return null;
  const assetValue = portfolio.asset * price;
  const currentExposure = assetValue / equity;
  const difference = targetExposure - currentExposure;
  if (Math.abs(difference) <= Number.EPSILON) return { ...portfolio };
  if (difference > 0) {
    const denominator = 1 - friction + friction * targetExposure;
    if (!(denominator > 0)) return null;
    const grossQuoteSpent = equity * difference / denominator;
    return {
      quote: portfolio.quote - grossQuoteSpent,
      asset: portfolio.asset + (1 - friction) * grossQuoteSpent / price,
    };
  }
  const denominator = 1 - friction * targetExposure;
  if (!(denominator > 0)) return null;
  const grossAssetValueSold = equity * -difference / denominator;
  return {
    quote: portfolio.quote + (1 - friction) * grossAssetValueSold,
    asset: portfolio.asset - grossAssetValueSold / price,
  };
}

/**
 * Literal amount-space holding simulation. Positive quote and asset quantities
 * remain constant. Only borrowed (negative) quantities accrue maintenance.
 * The fixed base-asset quantity is marked at every price for liquidation.
 */
function holdPortfolio(
  prices: ArrayLike<number>,
  startTime: number,
  endpointTime: number,
  initialPortfolio: PortfolioState,
  execution: {
    friction: number;
    maxEffectiveExposure: number;
    quoteBorrowRate: number;
    assetBorrowRate: number;
  },
): HoldingResult | null {
  const portfolio = { ...initialPortfolio };
  for (let time = startTime; time < endpointTime; time += 1) {
    if (portfolio.quote < 0) {
      portfolio.quote *= 1 + execution.quoteBorrowRate;
    }
    if (portfolio.asset < 0) {
      portfolio.asset *= 1 + execution.assetBorrowRate;
    }
    const assetValue = portfolio.asset * prices[time + 1]!;
    const markedEquity = portfolio.quote + assetValue;
    const liquidatedAssetValue = assetValue >= 0
      ? assetValue * (1 - execution.friction)
      : assetValue / Math.max(Number.EPSILON, 1 - execution.friction);
    const liquidationEquity = portfolio.quote + liquidatedAssetValue;
    const liquidated = !(liquidationEquity > 0)
      || !Number.isFinite(liquidationEquity)
      || Math.abs(liquidatedAssetValue / liquidationEquity)
        > execution.maxEffectiveExposure
      || !(markedEquity > 0)
      || !Number.isFinite(markedEquity);
    if (liquidated) return null;
  }
  return { portfolio };
}
