import type { EquityPoint } from "./types.js";

export interface RiskAdjustedMetrics {
  riskAdjustedReturn?: number;
  sharpeRatio?: number;
  backtestSharpeRatio?: number;
}

const YEAR_MS = 365 * 24 * 60 * 60 * 1000;

export function calculateRiskAdjustedMetrics(
  equityCurve: EquityPoint[],
  returnPct: number,
  maxDrawdownPct: number,
): RiskAdjustedMetrics {
  const sampledSharpeRatio = calculateSampledSharpeRatio(equityCurve);

  return {
    riskAdjustedReturn:
      Number.isFinite(maxDrawdownPct) && maxDrawdownPct > 0
        ? returnPct / maxDrawdownPct
        : undefined,
    sharpeRatio: sampledSharpeRatio?.annualized,
    backtestSharpeRatio: sampledSharpeRatio?.backtest,
  };
}

function calculateSampledSharpeRatio(
  equityCurve: EquityPoint[],
): { annualized: number; backtest: number } | undefined {
  if (equityCurve.length < 3) {
    return undefined;
  }

  const returns: number[] = [];
  for (let index = 1; index < equityCurve.length; index += 1) {
    const previous = equityCurve[index - 1];
    const current = equityCurve[index];
    if (previous.equity > 0 && current.time > previous.time) {
      returns.push((current.equity - previous.equity) / previous.equity);
    }
  }

  if (returns.length < 2) {
    return undefined;
  }

  const mean = average(returns);
  const deviation = sampleStandardDeviation(returns, mean);
  if (deviation <= 0) {
    return undefined;
  }

  const sharpe = mean / deviation;

  const durationMs = equityCurve[equityCurve.length - 1].time - equityCurve[0].time;
  const durationYears = durationMs > 0 ? durationMs / YEAR_MS : 0;
  const periodsPerYear = durationYears > 0 ? returns.length / durationYears : returns.length;

  return {
    annualized: sharpe * Math.sqrt(periodsPerYear),
    backtest: sharpe,
  };
}

function average(values: number[]): number {
  return values.reduce((total, value) => total + value, 0) / values.length;
}

function sampleStandardDeviation(values: number[], mean: number): number {
  if (values.length < 2) {
    return 0;
  }

  const variance =
    values.reduce((total, value) => total + (value - mean) ** 2, 0) / (values.length - 1);
  return Math.sqrt(variance);
}
