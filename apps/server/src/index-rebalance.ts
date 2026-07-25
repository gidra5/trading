const WEIGHT_TOLERANCE = 1e-10;
const MAXIMUM_SOLVER_ITERATIONS = 80;

export interface RebalancePosition {
  currentValue: number;
  targetWeight: number;
  costRate: number;
}

export interface RebalanceResult {
  preTradeValue: number;
  postTradeValue: number;
  transactionCost: number;
  grossTradedNotional: number;
  grossTurnover: number;
  cashValue: number;
  positions: number[];
}

/**
 * Rebalance a long-only portfolio without borrowing.
 *
 * Target weights are fractions of post-cost net asset value. This makes the
 * trade self-financing: post-trade risky positions plus cash equal post-cost
 * value exactly, and transaction costs are charged on every dollar bought or
 * sold. The scalar post-cost value is solved from
 *
 *   V' = V - sum(costRate_i * abs(targetWeight_i * V' - currentValue_i)).
 */
export function rebalanceLongOnly(
  cashValue: number,
  positions: readonly RebalancePosition[],
): RebalanceResult {
  finiteNonNegative(cashValue, "Cash value");
  let targetExposure = 0;
  let currentRiskyValue = 0;
  for (const position of positions) {
    finiteNonNegative(position.currentValue, "Current position value");
    finiteNonNegative(position.targetWeight, "Target weight");
    finiteNonNegative(position.costRate, "Transaction-cost rate");
    if (position.costRate >= 1) {
      throw new Error("Transaction-cost rates must be below one.");
    }
    targetExposure += position.targetWeight;
    currentRiskyValue += position.currentValue;
  }
  if (targetExposure > 1 + WEIGHT_TOLERANCE) {
    throw new Error(
      `Long-only target exposure must not exceed one; received ${targetExposure}.`,
    );
  }

  const preTradeValue = cashValue + currentRiskyValue;
  if (preTradeValue === 0) {
    return {
      preTradeValue: 0,
      postTradeValue: 0,
      transactionCost: 0,
      grossTradedNotional: 0,
      grossTurnover: 0,
      cashValue: 0,
      positions: positions.map(() => 0),
    };
  }

  const postTradeValue = solvePostTradeValue(preTradeValue, positions);
  const rebalancedPositions = positions.map(
    (position) => position.targetWeight * postTradeValue,
  );
  let transactionCost = 0;
  let grossTradedNotional = 0;
  for (let index = 0; index < positions.length; index += 1) {
    const traded = Math.abs(
      rebalancedPositions[index] - positions[index].currentValue,
    );
    grossTradedNotional += traded;
    transactionCost += traded * positions[index].costRate;
  }
  const postTradeCash = Math.max(
    0,
    postTradeValue -
      rebalancedPositions.reduce((total, value) => total + value, 0),
  );
  const accountingError =
    Math.abs(preTradeValue - transactionCost - postTradeValue);
  if (accountingError > Math.max(1e-9, preTradeValue * 1e-10)) {
    throw new Error(
      `Self-financing rebalance failed by ${accountingError}.`,
    );
  }

  return {
    preTradeValue,
    postTradeValue,
    transactionCost,
    grossTradedNotional,
    grossTurnover: grossTradedNotional / preTradeValue,
    cashValue: postTradeCash,
    positions: rebalancedPositions,
  };
}

function solvePostTradeValue(
  preTradeValue: number,
  positions: readonly RebalancePosition[],
): number {
  let lower = 0;
  let upper = preTradeValue;
  for (
    let iteration = 0;
    iteration < MAXIMUM_SOLVER_ITERATIONS;
    iteration += 1
  ) {
    const candidate = (lower + upper) / 2;
    const afterCosts =
      preTradeValue -
      positions.reduce(
        (total, position) =>
          total +
          position.costRate *
            Math.abs(
              position.targetWeight * candidate -
                position.currentValue,
            ),
        0,
      );
    if (afterCosts > candidate) {
      lower = candidate;
    } else {
      upper = candidate;
    }
  }
  return (lower + upper) / 2;
}

function finiteNonNegative(value: number, label: string): void {
  if (!Number.isFinite(value) || value < 0) {
    throw new Error(`${label} must be finite and non-negative.`);
  }
}
