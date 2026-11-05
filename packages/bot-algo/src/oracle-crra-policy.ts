import {
  advance, equity, exposure, executeAction, prepareArrival, requestForTarget, terminalCloseChecked,
  type OracleAction, type OracleOrder, type OracleRules, type OracleState, type PreparedArrival,
} from "./oracle-execution.js";
import {
  ONE, ZERO, abs, add, cmp, div, eq, gt, isZero, lt, min, mul, nearestStep, neg, rat,
  serializeRat, sub, sum, type Rational, type RationalInput,
} from "./oracle-rational.js";

/** Probabilities and transitions are conditional on the observable price history. */
export interface PriceTree { readonly children: readonly PriceBranch[] }
export interface PriceBranch extends PriceTree {
  readonly probability: Rational;
  readonly price: Rational;
  /** Additional observable market information, never a latent/scenario ID. */
  readonly observationKey?: string;
}
export interface ForecastTree {
  readonly children: readonly {
    readonly probability: RationalInput;
    readonly price: RationalInput;
    readonly observationKey?: string;
    readonly children?: ForecastTree["children"];
  }[];
}

/**
 * Quantize prices, exactly normalize decimal weights, and merge observationally
 * identical branches BEFORE optimizing their conditional continuation. Otherwise
 * the optimizer could observe a hidden scenario label and become clairvoyant.
 */
export function observableTree(input: ForecastTree, priceStep: Rational): PriceTree {
  function canonical(children: ForecastTree["children"]): PriceTree {
    if (!children.length) return { children: [] };
    const total = sum(children.map((branch) => rat(branch.probability)));
    if (!gt(total, ZERO)) throw new Error("Forecast probabilities must have positive total mass");
    const groups = new Map<string, { price: Rational; observationKey?: string; weight: Rational; tails: { weight: Rational; tree: PriceTree }[] }>();
    for (const branch of children) {
      const weight = rat(branch.probability);
      if (lt(weight, ZERO)) throw new Error("Negative forecast probability");
      if (isZero(weight)) continue;
      const price = nearestStep(rat(branch.price), priceStep);
      if (!gt(price, ZERO)) throw new Error("Forecast price rounds outside the positive price grid");
      const key = JSON.stringify([serializeRat(price), branch.observationKey ?? ""]);
      const group = groups.get(key) ?? { price, observationKey: branch.observationKey, weight: ZERO, tails: [] };
      group.weight = add(group.weight, weight);
      group.tails.push({ weight, tree: canonical(branch.children ?? []) });
      groups.set(key, group);
    }
    const output: PriceBranch[] = [];
    for (const group of groups.values()) {
      const lengths = group.tails.map(({ tree }) => tree.children.length > 0);
      if (lengths.some(Boolean) && !lengths.every(Boolean))
        throw new Error("Indistinguishable scenarios must have the same decision horizon");
      // Recanonicalize the mixture, conditioning on the observed (rounded) price.
      const mixed = group.tails.flatMap(({ weight, tree }) => tree.children.map((child) => ({
        probability: mul(div(weight, group.weight), child.probability),
        price: child.price, children: child.children,
        observationKey: child.observationKey,
      })));
      output.push({ price: group.price, observationKey: group.observationKey, probability: div(group.weight, total), ...canonical(mixed) });
    }
    return { children: output.sort((a, b) => cmp(a.price, b.price)) };
  }
  const result = canonical(input.children);
  const depths = (tree: PriceTree): number[] => tree.children.length
    ? tree.children.flatMap((child) => depths(child).map((depth) => depth + 1)) : [0];
  if (new Set(depths(result)).size !== 1) throw new Error("Forecast leaves must share a fixed horizon");
  return result;
}

/** Bottom represents U_2(wealth <= 0) = -infinity. Positive wealth has U_2 = -1/wealth. */
export type Utility = Rational | null;
export type RiskAversion = 0 | 2;
export const defaultRiskAversion: RiskAversion = 0;
export function crra2(wealth: Rational): Utility { return gt(wealth, ZERO) ? neg(div(ONE, wealth)) : null; }
/** Gamma zero has finite utility at ruin: U_0(0) = -1. Negative wealth is outside the CRRA domain. */
export function terminalUtility(wealth: Rational, riskAversion: RiskAversion): Utility {
  if (riskAversion === 0) return lt(wealth, ZERO) ? null : sub(wealth, ONE);
  if (riskAversion === 2) return crra2(wealth);
  throw new Error("Exact CRRA supports only risk aversion 0 or 2");
}
export function compareUtility(a: Utility, b: Utility): number {
  return a === null ? (b === null ? 0 : -1) : b === null ? 1 : cmp(a, b);
}
export function expectedUtility(branches: readonly { readonly probability: Rational; readonly utility: Utility }[]): Utility {
  if (branches.some(({ probability }) => lt(probability, ZERO)) ||
      !eq(sum(branches.map(({ probability }) => probability)), ONE))
    throw new Error("Expected utility requires a normalized nonnegative probability law");
  let value = ZERO;
  for (const branch of branches) {
    if (isZero(branch.probability)) continue;
    if (branch.utility === null) return null;
    value = add(value, mul(branch.probability, branch.utility));
  }
  return value;
}

export interface NamedAction { readonly id: string; readonly action: OracleAction }
export type ActionMenu = (arrival: PreparedArrival, tree: PriceTree, rules: OracleRules) => readonly NamedAction[];
export interface MenuConfig {
  readonly exposureTargets: readonly Rational[];
  readonly limitNotionalFractions: readonly Rational[];
  readonly limitOffsetsInTicks: readonly number[];
  /** Explicit finite-menu restriction, independently of the account's maximum book size. */
  readonly maximumNewOrders: number;
}
export const defaultMenu: MenuConfig = {
  exposureTargets: [-2, -1, 0, 1, 2].map(rat),
  limitNotionalFractions: [rat("1/2")], limitOffsetsInTicks: [1], maximumNewOrders: 1,
};

/** Cartesian finite market/cancel/place menu. Invalid actions are removed by the execution model. */
export function finiteActionMenu(config: MenuConfig): ActionMenu {
  if (!Number.isSafeInteger(config.maximumNewOrders) || config.maximumNewOrders < 0 ||
      config.limitOffsetsInTicks.some((n) => !Number.isSafeInteger(n) || n <= 0) ||
      config.limitNotionalFractions.some((n) => !gt(n, ZERO))) throw new Error("Invalid finite action menu");
  if (new Set(config.limitOffsetsInTicks).size !== config.limitOffsetsInTicks.length ||
      new Set(config.limitNotionalFractions.map(serializeRat)).size !== config.limitNotionalFractions.length)
    throw new Error("Duplicate limit menu entries");
  return (arrival, _tree, rules) => {
    const state = arrival.state;
    const wealth = equity(state);
    if (!gt(wealth, ZERO)) return [{ id: "hold", action: { marketQuote: ZERO, cancelIds: [], place: [] } }];
    const availableQuote = sub(rules.quoteMaximum, arrival.consumedQuote);
    const market = new Map<string, Rational>([["0", ZERO]]);
    for (const target of config.exposureTargets) {
      if (lt(target, rules.entryLower) || gt(target, rules.entryUpper)) continue;
      const denominator = lt(exposure(state), target)
        ? add(sub(ONE, rules.takerFee), mul(rules.takerFee, target))
        : sub(ONE, mul(rules.takerFee, target));
      if (!eq(exposure(state), target) && !gt(denominator, ZERO)) continue;
      const requested = requestForTarget(state, target, rules);
      const capped = mul(lt(requested, ZERO) ? neg(ONE) : ONE, min(abs(requested), availableQuote));
      market.set(serializeRat(capped), capped);
    }
    // All subsets of existing orders may be canceled. IDs are execution labels,
    // never forecast information; new IDs are determined by observable state.
    const cancellations: string[][] = [[]];
    for (const order of state.orders) for (const subset of [...cancellations]) cancellations.push([...subset, order.id]);
    const newOrders: OracleOrder[] = [];
    for (const side of ["buy", "sell"] as const) for (const offset of config.limitOffsetsInTicks)
      for (const size of config.limitNotionalFractions) {
        const price = add(state.price, mul(rat(side === "buy" ? -offset : offset), rules.priceStep));
        if (!gt(price, ZERO)) continue;
        const notional = min(mul(wealth, size), side === "buy" ? rules.quoteMaximum : div(rules.quoteMaximum, sub(ONE, rules.makerFee)));
        const id = `t${state.time}:${side}:${offset}:${serializeRat(size)}`;
        newOrders.push({ id, side, price, notional, placedAt: state.time });
      }
    const placements: OracleOrder[][] = [[]];
    for (const order of newOrders) for (const subset of [...placements])
      if (subset.length < config.maximumNewOrders) placements.push([...subset, order]);
    const result: NamedAction[] = [];
    for (const cancelIds of cancellations) for (const place of placements) {
      if (state.orders.length - cancelIds.length + place.length > rules.maxOrders) continue;
      for (const marketQuote of market.values()) result.push({
        id: JSON.stringify([serializeRat(marketQuote), cancelIds, place.map(({ id }) => id)]),
        action: { marketQuote, cancelIds, place },
      });
    }
    return result;
  };
}

export interface CertificateBranch {
  readonly weight: string;
  readonly wealth?: string;
  readonly child?: DecisionCertificate;
}
export interface DecisionCertificate {
  readonly terminalRequirement: "zero-asset-and-no-orders";
  readonly riskAversion: RiskAversion;
  readonly chosenId: string;
  readonly candidates: readonly { readonly id: string; readonly feasible: true; readonly branches: readonly CertificateBranch[] }[];
}
export interface Decision {
  readonly id: string;
  readonly action: OracleAction;
  readonly state: OracleState;
  readonly value: Utility;
  readonly feasibleCandidates: number;
  readonly certificate?: DecisionCertificate;
}
export interface SolverConfig {
  readonly riskAversion: RiskAversion;
  readonly rules: OracleRules;
  readonly menu: ActionMenu;
  /** Known funding schedule; must not inspect the realized future price. */
  readonly funding: (time: number) => Rational;
}
export class NoFeasiblePolicy extends Error {}

/**
 * Backward induction, gamma=0 or 2, finite observable tree and declared finite menus.
 * Each action is shared by every continuation before the next price is observed.
 * Exact arithmetic is the executable counterpart of ExactCRRAKernel.lean.
 */
export function solveCrra(state: OracleState, tree: PriceTree, config: SolverConfig, certificate = false): Decision {
  if (config.riskAversion !== 0 && config.riskAversion !== 2) throw new Error("Exact CRRA supports only risk aversion 0 or 2");
  if (!tree.children.length) throw new Error("A decision requires at least one future tick");
  if (new Set(tree.children.map((branch) => JSON.stringify([serializeRat(branch.price), branch.observationKey ?? ""]))).size !== tree.children.length)
    throw new Error("Indistinguishable prices must be merged with observableTree before solving");
  if (!eq(sum(tree.children.map((b) => b.probability)), ONE) || tree.children.some((b) => !gt(b.probability, ZERO)))
    throw new Error("Use a canonical observable tree with positive normalized probabilities");
  const prepared = prepareArrival(state, config.rules);
  if (!prepared.ok) throw new NoFeasiblePolicy(`${prepared.error.code}: ${prepared.error.message}`);
  const candidates: DecisionCertificate["candidates"][number][] = [];
  const ids = new Set<string>();
  let best: Decision | undefined;
  let feasibleCandidates = 0;
  for (const named of config.menu(prepared.value, tree, config.rules)) {
    if (ids.has(named.id)) throw new Error("Action IDs must be unique within a decision menu");
    ids.add(named.id);
    const executed = executeAction(prepared.value, named.action, config.rules);
    if (!executed.ok) continue;
    const outcomes: { probability: Rational; utility: Utility }[] = [];
    const branches: CertificateBranch[] = [];
    let feasible = true;
    for (const branch of tree.children) {
      const moved = advance(executed.value, branch.price, config.funding(state.time), config.rules);
      if (!branch.children.length) {
        const terminal = terminalCloseChecked(moved, config.rules);
        if (!terminal.ok) { feasible = false; break; }
        outcomes.push({ probability: branch.probability, utility: terminalUtility(terminal.value.wealth, config.riskAversion) });
        if (certificate) branches.push({ weight: serializeRat(branch.probability), wealth: serializeRat(terminal.value.wealth) });
      } else {
        try {
          const child = solveCrra(moved, branch, config, certificate);
          outcomes.push({ probability: branch.probability, utility: child.value });
          if (certificate) branches.push({ weight: serializeRat(branch.probability), child: child.certificate! });
        } catch (error) {
          if (!(error instanceof NoFeasiblePolicy)) throw error;
          feasible = false; break;
        }
      }
    }
    if (!feasible) continue;
    feasibleCandidates++;
    const value = expectedUtility(outcomes);
    if (certificate) candidates.push({ id: named.id, feasible: true, branches });
    if (!best || compareUtility(value, best.value) > 0)
      best = { ...named, state: executed.value, value, feasibleCandidates: 0 };
  }
  if (!best) throw new NoFeasiblePolicy("No feasible continuation in the declared action menu");
  return { ...best, feasibleCandidates, ...(certificate ? { certificate: { terminalRequirement: "zero-asset-and-no-orders" as const, riskAversion: config.riskAversion, chosenId: best.id, candidates } } : {}) };
}
