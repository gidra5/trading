import { validateEventDistribution, type EventDistribution } from "./event-distribution.js";
import { eventCapAllowsTrade, eventHolding, eventTrade, eventTradeDecision, validateEventCosts, type EventAccount, type EventCosts, type EventTrade } from "./event-log-policy.js";
import { prepareEventTwoStep } from "./event-two-step.js";
import { prepareEventOneStep } from "./event-one-step-prepared.js";
import { prepareEventMultiStepUpper } from "./event-multi-step-upper.js";

type Branch = { next: number; probability: number; logHolding: number; account: EventAccount;
  result: ReturnType<ReturnType<typeof prepareEventTwoStep>> };

function quantityTrade(account: EventAccount, quantity: number, costs: EventCosts): EventTrade | null {
  const turnover = Math.abs(quantity) * account.price, cost = turnover * (costs.feeBps + costs.slippageBps) / 10000;
  const equity = account.equity - cost;
  const exposure = quantity ? (account.exposure * account.equity + quantity * account.price) / equity : account.exposure;
  const feasible = quantity === 0 ? Math.abs(exposure) <= costs.maxLeverage + 1e-9
    : equity > 0 && Math.abs(quantity / costs.quantityStep - Math.round(quantity / costs.quantityStep)) <= 1e-7
      && Math.abs(quantity) >= costs.minQuantity - 1e-12 && turnover >= costs.minNotional - 1e-8
      && turnover <= costs.maxNotional + 1e-8 && eventCapAllowsTrade(account.exposure, exposure, turnover, account.price, costs);
  return feasible ? { equity, price: account.price, exposure, quantity, turnover, cost } : null;
}

/** Bound the THREE-event value of a specified root order. Every successor
 * solves its own stochastic H2 problem after observing that state. This does
 * not certify a globally optimal H3 root action. A profiling limit leaves the
 * complete action value unresolved, never substitutes a sampled expectation. */
export function prepareEventThreeStepActions(input: EventDistribution, inputCosts: EventCosts, terminal: "marked" | "friction",
  preparation: { globalUpper?: boolean; shadowPoints?: number } = {}) {
  validateEventDistribution(input); validateEventCosts(inputCosts);
  const model = structuredClone(input), costs = { ...inputCosts }, two = prepareEventTwoStep(model, costs, terminal, preparation);
  const seedTargets = new Map<number, Map<number, number>>();
  return (leaf: number, account: EventAccount, quantity: number,
    options: { maxEvaluations?: number; tolerance?: number; limitOutcomes?: number; offsetOutcomes?: number; warmStart?: boolean;
      onBranch?: (row: Branch, index: number, total: number) => void } = {}) => {
    if (!model.kernels[leaf] || ![account.equity, account.price, account.exposure, quantity].every(Number.isFinite)
      || !(account.equity > 0 && account.price > 0)
      || options.maxEvaluations !== undefined && (!Number.isInteger(options.maxEvaluations) || options.maxEvaluations < 2)
      || options.tolerance !== undefined && (!Number.isFinite(options.tolerance) || options.tolerance < 4e-11)
      || options.limitOutcomes !== undefined
        && (!Number.isInteger(options.limitOutcomes) || options.limitOutcomes < 1)
      || options.offsetOutcomes !== undefined
        && (!Number.isInteger(options.offsetOutcomes) || options.offsetOutcomes < 0)) throw new Error("Invalid three-event action probe");
    const trade = quantityTrade(account, quantity, costs), feasible = Boolean(trade);
    const empty = { quantity, feasible, complete: true, lowerValue: -Infinity, upperValue: -Infinity, gap: 0,
      evaluatedOutcomes: 0, totalOutcomes: 0, evaluatedProbability: 0, certifiedContinuations: 0, maximumContinuationGap: 0 };
    if (!trade) return empty;
    const { equity, exposure } = trade;
    const groups = new Map<string, { next: number; probability: number; logHolding: number; account: EventAccount }>();
    for (const atom of model.kernels[leaf]) if (atom.probability > 0) {
      const held = eventHolding(exposure, atom, costs);
      if (held.liquidated) return empty; // Any positive ruin mass makes the whole log expectation -Infinity.
      const future = { equity: equity * held.factor, price: account.price * (1 + atom.return), exposure: held.exposure };
      const key = [atom.next, future.equity, future.price, future.exposure, held.factor].join(":");
      const old = groups.get(key);
      if (old) old.probability += atom.probability;
      else groups.set(key, { next: atom.next, probability: atom.probability, logHolding: Math.log(held.factor), account: future });
    }
    // This is lossless coalescing of identical successor accounts, after every
    // original outcome's liquidation check. Adjacent account queries can seed
    // each other's full H2 searches; sorting affects computation only.
    const branches = [...groups.values()].sort((a, b) => a.next - b.next || a.account.exposure - b.account.exposure
      || a.account.price - b.account.price || a.account.equity - b.account.equity);
    let lowerValue = Math.log(equity / account.equity), upperValue = lowerValue, evaluatedProbability = 0;
    let certifiedContinuations = 0, maximumContinuationGap = 0;
    const offset = options.offsetOutcomes ?? 0;
    if (offset >= branches.length && branches.length) throw new Error("Three-event profile offset exceeds the successor count");
    const selected = branches.slice(offset, offset + (options.limitOutcomes ?? branches.length));
    for (let i = 0; i < selected.length; i++) {
      const b = selected[i], targets = seedTargets.get(b.next) ?? new Map<number, number>();
      const seedQuantities = options.warmStart === false ? [] : [...targets.values()]
        .map(target => eventTrade(b.account, Math.max(-costs.maxLeverage, Math.min(costs.maxLeverage, target)), costs)?.quantity)
        .filter((q): q is number => q !== undefined);
      const result = two(b.next, b.account, { ...options, seedQuantities });
      if (result.quantity && Number.isFinite(result.value)) {
        targets.set(Math.sign(result.quantity), result.exposure); seedTargets.set(b.next, targets);
      }
      lowerValue += b.probability * (b.logHolding + result.lowerValue);
      upperValue += b.probability * (b.logHolding + result.upperValue);
      evaluatedProbability += b.probability;
      certifiedContinuations += Number(result.converged && result.feasible);
      maximumContinuationGap = Math.max(maximumContinuationGap, result.gap);
      options.onBranch?.({ ...b, result }, i, branches.length);
      if (result.upperValue === -Infinity) return { ...empty, evaluatedOutcomes: i + 1, totalOutcomes: branches.length,
        evaluatedProbability, certifiedContinuations, maximumContinuationGap };
    }
    const complete = offset === 0 && selected.length === branches.length;
    return { quantity, feasible, complete, lowerValue: complete ? lowerValue : -Infinity, upperValue: complete ? upperValue : Infinity,
      gap: !complete ? Infinity : upperValue === lowerValue ? 0 : upperValue - lowerValue,
      evaluatedOutcomes: selected.length, totalOutcomes: branches.length, evaluatedProbability,
      certifiedContinuations, maximumContinuationGap };
  };
}

/** Feasible H3 policy with a GLOBAL numerical performance bound. The candidate
 * budget limits expensive lower-policy evaluations, not the bound's domain.
 * A gap above tolerance remains unresolved even if one listed order wins.
 * This is finite receding planning, not a stationary-policy certificate. */
export function prepareEventThreeStep(input: EventDistribution, inputCosts: EventCosts, terminal: "marked" | "friction",
  preparation: { shadowPoints?: number } = {}) {
  const model = structuredClone(input), costs = { ...inputCosts }, shadowPoints = preparation.shadowPoints ?? 129;
  const upper = prepareEventMultiStepUpper(model, costs, terminal, { depth: 3, shadowPoints, method: "marginal" });
  const evaluate = prepareEventThreeStepActions(model, costs, terminal, { globalUpper: true, shadowPoints });
  const one = new Map<number, ReturnType<typeof prepareEventOneStep>>();
  const fee = (costs.feeBps + costs.slippageBps) / 10000;
  return (leaf: number, account: EventAccount,
    options: { tolerance?: number; maxEvaluations?: number; maxRootEvaluations?: number } = {}) => {
    const tolerance = options.tolerance ?? 1e-7, maxEvaluations = options.maxEvaluations ?? 64;
    const maxRootEvaluations = options.maxRootEvaluations ?? 4;
    if (!(tolerance >= 4e-10) || !Number.isFinite(tolerance) || !Number.isInteger(maxEvaluations) || maxEvaluations < 2
      || !Number.isInteger(maxRootEvaluations) || maxRootEvaluations < 1) throw new Error("Invalid three-event policy budget");
    const bound = upper.query(leaf, account), sellBound = upper.query(leaf, account, 3, -1, "exchange-relaxation");
    const buyBound = upper.query(leaf, account, 3, 1, "exchange-relaxation");
    // Every root order is a buy or a sell (holding belongs to both closed
    // halves). Their maximum is another whole-lattice upper bound and can be
    // tighter than the unrestricted multi-shadow envelope.
    const directionalUpperValue = Math.max(sellBound.upperValue, buyBound.upperValue);
    let globalUpperValue = Math.min(bound.upperValue, directionalUpperValue);
    let best = quantityTrade(account, 0, costs), lowerValue = account.exposure === 0 ? 0 : -Infinity;
    let feasible = account.exposure === 0, cashLowerPolicy = feasible;
    const evaluated: Array<ReturnType<typeof evaluate> & { continuationTolerance: number }> = [];
    const candidates: Array<{ trade: EventTrade; upperValue: number }> = [];
    const finish = () => {
      if (globalUpperValue < lowerValue - 2e-10) throw new Error("Global H3 upper bound below a feasible policy");
      const upperValue = Math.max(lowerValue, globalUpperValue), gap = upperValue === lowerValue ? 0 : upperValue - lowerValue;
      return { ...eventTradeDecision(account, best ?? { ...account, quantity: 0, turnover: 0, cost: 0 }, lowerValue),
        feasible, lowerValue, upperValue, globalUpperValue, recursiveUpperValue: bound.upperValue,
        directionalUpperValues: [sellBound.upperValue, buyBound.upperValue], gap,
        nonzeroUpperValues: [sellBound.nonzeroUpperValue, buyBound.nonzeroUpperValue],
        converged: feasible && gap <= tolerance, tolerance, terminal,
        cashLowerPolicy, rootEvaluations: evaluated.length,
        candidates: candidates.map(c => ({ quantity: c.trade.quantity, upperValue: c.upperValue })), evaluated };
    };
    // Staying in cash for every remaining event is an executable lower policy.
    // When the global bound is already within tolerance, no path expansion is
    // needed. This is not the assumption that future optimal value is zero.
    if (feasible && globalUpperValue <= lowerValue + tolerance) return finish();
    let h1 = one.get(leaf);
    if (!h1) { h1 = prepareEventOneStep(model.kernels[leaf], costs, terminal); one.set(leaf, h1); }
    const quantities = new Set<number>([0, h1(account).quantity]);
    for (const target0 of bound.seedExposures) {
      const target = Math.max(-costs.maxLeverage, Math.min(costs.maxLeverage, target0)), side = Math.sign(target - account.exposure);
      const lots = (target - account.exposure) * account.equity
        / (account.price * costs.quantityStep * (1 + target * fee * side));
      if (Number.isFinite(lots)) for (const k of [Math.floor(lots), Math.ceil(lots)]) quantities.add(k * costs.quantityStep);
    }
    const maximum = Math.floor(costs.maxNotional / account.price / costs.quantityStep + 1e-8);
    const minimum = Math.max(1, Math.ceil(Math.max((costs.minNotional - 1e-8) / account.price / costs.quantityStep,
      (costs.minQuantity - 1e-12) / costs.quantityStep)));
    for (const k of [minimum, -minimum, maximum, -maximum]) quantities.add(k * costs.quantityStep);
    for (const quantity of quantities) {
      const trade = quantityTrade(account, quantity, costs); if (!trade) continue;
      let value = Math.log(trade.equity / account.equity);
      for (const atom of model.kernels[leaf]) if (atom.probability > 0) {
        const h = eventHolding(trade.exposure, atom, costs);
        if (h.liquidated) { value = -Infinity; break; }
        const next = upper.query(atom.next, { equity: trade.equity * h.factor,
          price: account.price * (1 + atom.return), exposure: h.exposure }, 2).upperValue;
        if (!Number.isFinite(next)) { value = next; break; }
        value += atom.probability * (Math.log(h.factor) + next);
      }
      candidates.push({ trade, upperValue: value + 1e-10 });
    }
    candidates.sort((a, b) => b.upperValue - a.upperValue || Math.abs(a.trade.quantity) - Math.abs(b.trade.quantity));
    for (const candidate of candidates) {
      if (evaluated.length >= maxRootEvaluations || globalUpperValue <= lowerValue + tolerance) break;
      if (candidate.upperValue < lowerValue) continue;
      // A coarse continuation policy often already closes the GLOBAL gap.
      // Refine only when this action's remaining interval could close it;
      // both full passes count against the same expensive root budget.
      // The outer certificate always retains the requested tolerance.
      for (const continuationTolerance of [10 * tolerance, tolerance]) {
        if (evaluated.length >= maxRootEvaluations || globalUpperValue <= lowerValue + tolerance) break;
        const result = evaluate(leaf, account, candidate.trade.quantity, { maxEvaluations, tolerance: continuationTolerance });
        evaluated.push({ ...result, continuationTolerance });
        // Partition the whole root action set into hold, nonzero buys and
        // nonzero sells. Intersect complete hold bounds across refinements.
        if (result.quantity === 0 && result.complete) globalUpperValue = Math.min(globalUpperValue,
          Math.max(result.upperValue, sellBound.nonzeroUpperValue!, buyBound.nonzeroUpperValue!));
        if (result.feasible && (!feasible || result.lowerValue > lowerValue
          || result.lowerValue === lowerValue && Math.abs(candidate.trade.quantity) < Math.abs(best!.quantity))) {
          best = candidate.trade; lowerValue = result.lowerValue; feasible = true; cashLowerPolicy = false;
        }
        if (!result.complete || result.upperValue <= lowerValue
          || globalUpperValue - result.upperValue > tolerance) break;
      }
    }
    return finish();
  };
}
