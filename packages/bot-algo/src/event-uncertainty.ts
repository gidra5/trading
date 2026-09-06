export interface EventForecastResidual {
  predicted: number;
  realized: number;
}

export interface EventUncertaintyRiskInput {
  initialEquity: number;
  liquidatableHighWater: number;
  maximumInitialRiskBps: number;
  minimumProtectedProfitFraction: number;
  confidence: number;
}

/** Portion of a directional point forecast which remains after its interval is
 * moved toward the neutral value. An interval touching/crossing neutral has no
 * robust directional confidence; a point interval has full confidence. */
export function eventIntervalConfidence(
  center: number,
  low: number,
  high: number,
  neutral = 0,
): number {
  if (![center, low, high, neutral].every(Number.isFinite)
    || low > center || center > high)
    throw new Error("Invalid forecast interval");
  const distance = center - neutral;
  if (!distance) return 0;
  const robustDistance = distance > 0 ? low - neutral : neutral - high;
  return Math.max(0, Math.min(1, robustDistance / Math.abs(distance)));
}

/** Smooth signal-to-uncertainty ratio for position sizing. Unlike the robust
 * directional confidence above, an interval crossing neutral retains a small
 * budget when its center is informative relative to its radius. A neutral
 * center has no budget and a non-neutral point interval has a full budget. */
export function eventRelativeUncertaintyConfidence(
  center: number,
  low: number,
  high: number,
  neutral = 0,
): number {
  if (![center, low, high, neutral].every(Number.isFinite)
    || low > center || center > high)
    throw new Error("Invalid forecast interval");
  const signal = Math.abs(center - neutral);
  const radius = Math.max(center - low, high - center);
  return signal ? signal / (signal + radius) : 0;
}

/** Convert forecast confidence into a liquidatable-equity floor. At zero
 * confidence the whole high-water balance is protected. As confidence rises,
 * the initial risk allowance and unprotected share of accumulated profit grow
 * linearly to their declared maxima. */
export function eventUncertaintyRiskFloor(input: EventUncertaintyRiskInput) {
  const { initialEquity, liquidatableHighWater, maximumInitialRiskBps,
    minimumProtectedProfitFraction, confidence } = input;
  if (!(initialEquity > 0) || !(liquidatableHighWater > 0)
    || ![maximumInitialRiskBps, minimumProtectedProfitFraction, confidence].every(Number.isFinite)
    || maximumInitialRiskBps < 0 || minimumProtectedProfitFraction < 0
    || minimumProtectedProfitFraction > 1 || confidence < 0 || confidence > 1)
    throw new Error("Invalid uncertainty risk state");
  const peakProfit = Math.max(0, liquidatableHighWater - initialEquity);
  const effectiveInitialRiskBps = maximumInitialRiskBps * confidence;
  const protectedProfitFraction = 1 - confidence * (1 - minimumProtectedProfitFraction);
  const floor = peakProfit > 0
    ? initialEquity + protectedProfitFraction * peakProfit
    : initialEquity * (1 - effectiveInitialRiskBps / 10_000);
  return { floor, peakProfit, confidence, effectiveInitialRiskBps, protectedProfitFraction };
}

/** Finite-sample split-conformal radius for a two-sided absolute-residual set. */
export function eventConformalResidualRadius(
  rows: readonly EventForecastResidual[],
  coverage: number,
): number {
  if (!rows.length || !(coverage > 0 && coverage < 1))
    throw new Error("Conformal residuals and coverage must be nonempty and finite");
  const residuals = rows.map(({ predicted, realized }) => {
    if (!Number.isFinite(predicted) || !Number.isFinite(realized))
      throw new Error("Conformal forecasts and outcomes must be finite");
    return Math.abs(realized - predicted);
  }).sort((left, right) => left - right);
  const rank = Math.ceil((rows.length + 1) * coverage) - 1;
  return rank < residuals.length ? residuals[rank]! : Infinity;
}
