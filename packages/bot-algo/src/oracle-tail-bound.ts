import { advance, executeAction, prepareArrival, type OracleAction, type OracleState } from './oracle-execution.js';
import { expectedUtility, type PriceTree, type SolverConfig, type Utility } from './oracle-crra-policy.js';
import { ZERO, ONE, add, sub, mul, div, gt, lt, rat, eq, sum, type Rational } from './oracle-rational.js';

/** Quote-envelope Bellman score for the supplied finite law and action menu.
 * It is an upper bound only for policies using those prefix menus/laws.
 * It is not a continuous-action or distribution-approximation certificate.
 * The optional first action fixes the same action selected by the short solve.
 * Frontier quote is valued before arrival/closing; their shared cap is included.
 */
export function quotePrefixUpper(state: OracleState, tree: PriceTree, config: SolverConfig,
  firstAction?: OracleAction): Utility {
  if (config.riskAversion !== 0) throw new Error('Quote tail envelope requires gamma=0');
  if (!tree.children.length) return add(gt(state.quote, ZERO) ? state.quote : ZERO, config.rules.quoteMaximum);
  if (!eq(sum(tree.children.map(b => b.probability)), ONE) || tree.children.some(b => !gt(b.probability, ZERO)))
    throw new Error('Prefix law must have positive normalized probabilities');
  const arrived = prepareArrival(state, config.rules);
  if (!arrived.ok) return null;
  const actions = firstAction ? [firstAction] : config.menu(arrived.value, tree, config.rules).map(x => x.action);
  let best: Utility = null;
  for (const action of actions) {
    const executed = executeAction(arrived.value, action, config.rules);
    if (!executed.ok) continue;
    const branches = tree.children.map(branch => ({ probability: branch.probability,
      utility: quotePrefixUpper(advance(executed.value, branch.price, config.funding(state.time), config.rules), branch, config) }));
    const value = expectedUtility(branches);
    if (value !== null && (best === null || gt(value, best))) best = value;
  }
  return best;
}

export function relativeTailEnvelope(prefixValue: Rational, prefixUpper: Rational, remainingTransitions: number,
  quoteCap: Rational): Rational {
  if (!gt(prefixValue, ZERO) || lt(prefixUpper, prefixValue) || lt(quoteCap, ZERO) ||
    !Number.isSafeInteger(remainingTransitions) || remainingTransitions < 0)
    throw new Error('Require positive prefix value, valid upper bound and nonnegative integer tail length');
  return div(add(sub(prefixUpper, prefixValue), mul(rat(remainingTransitions), quoteCap)), prefixValue);
}

/** Necessary depth even if the prefix relaxation gap were zero. This does
 * not assert sufficiency: Q_H is replaced by its maximal cash-cap upper bound.
 * Horizon/depth count physical decisions; initial wealth is all positive cash.
 */
export function necessaryPerTickDepth(horizon: number, initialCash: Rational, quoteCap: Rational,
  tolerance: Rational): number {
  if (!Number.isSafeInteger(horizon) || horizon < 1 || !gt(initialCash, ZERO) ||
    !gt(quoteCap, ZERO) || !gt(tolerance, ZERO)) throw new Error('Invalid depth-bound domain');
  let lo = 1, hi = horizon;
  while (lo < hi) {
    const h = Math.floor((lo + hi) / 2);
    const maximalPrefixValue = add(initialCash, mul(rat(h + 1), quoteCap));
    if (lt(mul(rat(horizon - h), quoteCap), mul(tolerance, maximalPrefixValue))) hi = h;
    else lo = h + 1;
  }
  return lo;
}
