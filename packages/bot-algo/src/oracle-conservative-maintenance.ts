import { type OracleState } from "./oracle-execution.js";
import { ZERO, abs, add, eq, gt, lt, max, mul, neg, sub, type Rational } from "./oracle-rational.js";

type Account = Pick<OracleState, "quote" | "asset" | "price">;

export interface MaintenanceAssessment {
  readonly actualChange: Rational;
  /** Extra evaluation debit, accumulated separately from the real account. */
  readonly debit: Rational;
  /** Actual maintenance change less the debit; strictly negative for a
   * nonzero position when extraRate > 0. Price marking follows this phase. */
  readonly assessedChange: Rational;
}

const markedWealth = (s: Account): Rational => add(s.quote, mul(s.asset, s.price));

/** ConservativeMaintenance.debit in Lean. Assess the selected actual
 * maintenance rule at one mark, without changing executions or feasibility. */
export function assessMaintenance(
  before: Account, after: Account, extraRate: Rational,
): MaintenanceAssessment {
  if (lt(extraRate, ZERO)) throw new Error("Extra maintenance rate must be nonnegative.");
  if (!gt(before.price, ZERO) || !eq(before.price, after.price))
    throw new Error("Maintenance must be assessed before price marking, at the same positive price.");
  const actualChange = sub(markedWealth(after), markedWealth(before));
  const debit = add(max(actualChange, ZERO), mul(extraRate, abs(mul(before.asset, before.price))));
  return { actualChange, debit, assessedChange: sub(actualChange, debit) };
}

/** Funding alternative only. Call on the account AFTER arrival fills and
 * market execution, BEFORE funding/marking/liquidation in Oracle.advance. */
export function assessFunding(
  account: Account, factor: Rational, extraRate: Rational,
): MaintenanceAssessment {
  if (!gt(factor, ZERO)) throw new Error("Funding factor must be positive.");
  return assessMaintenance(account, { ...account, asset: mul(factor, account.asset) }, extraRate);
}

/** Separate borrowing alternative from Oracle.md. This evaluates its
 * documented maintenance phase; it does not add a borrowing execution engine
 * or combine borrowing with signed-asset funding. */
export function assessBorrowing(
  account: Account, quoteRate: Rational, assetRate: Rational, extraRate: Rational,
): MaintenanceAssessment {
  if (lt(quoteRate, ZERO) || lt(assetRate, ZERO)) throw new Error("Borrowing rates must be nonnegative.");
  return assessMaintenance(account, {
    ...account,
    quote: sub(account.quote, mul(quoteRate, max(neg(account.quote), ZERO))),
    asset: sub(account.asset, mul(assetRate, max(neg(account.asset), ZERO))),
  }, extraRate);
}

/** lowerValue's terminal payoff. Completed-policy callers must first validate
 * the exact flat terminal close. This arithmetic alone does not establish
 * terminal feasibility, and the assessment never drives execution. */
export function conservativeTerminalWealth(actualWealth: Rational, debit: Rational): Rational {
  if (lt(actualWealth, ZERO) || lt(debit, ZERO))
    throw new Error("Terminal wealth and accumulated assessment debit must be nonnegative.");
  return max(sub(actualWealth, debit), ZERO);
}
