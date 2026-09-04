import { validateEventDistribution, type EventDistribution, type MoveAtom } from "./event-distribution.js";
import { eventCapAllowsTrade, eventHolding, eventTradeDecision, validateEventCosts,
  type EventAccount, type EventCosts, type EventTrade } from "./event-log-policy.js";
import type { EventTerminal } from "./event-one-step.js";
import { eventOneStepUpper } from "./event-one-step-upper.js";
import { prepareEventOneStep } from "./event-one-step-prepared.js";
import { prepareEventMultiStepUpper } from "./event-multi-step-upper.js";

type Terminal = Exclude<EventTerminal, "market">;
type Bound = { low: number; high: number; upper: number };
type Options = { tolerance?: number; maxEvaluations?: number; seedQuantities?: readonly number[] };
type Compiled = { relaxations: Map<number, ReturnType<typeof eventOneStepUpper>>;
  continuations: Map<number, ReturnType<typeof prepareEventOneStep>>;
  globalUpper?: ReturnType<typeof prepareEventMultiStepUpper> };
const compiledLaw = (): Compiled => ({ relaxations: new Map(), continuations: new Map() });

/** Reuse only law-dependent work across accounts. Own a snapshot so later
 * forecast/cost mutations cannot invalidate previously compiled quantities. */
export function prepareEventTwoStep(model: EventDistribution, costs: EventCosts, terminal: Terminal,
  options: { globalUpper?: boolean; shadowPoints?: number } = {}) {
  validateEventDistribution(model); validateEventCosts(costs);
  if (!["marked", "friction"].includes(terminal)) throw new Error("Invalid two-event terminal");
  const snapshot = structuredClone(model), frozenCosts = { ...costs }, compiled = compiledLaw();
  if (options.globalUpper) compiled.globalUpper = prepareEventMultiStepUpper(snapshot, frozenCosts, terminal,
    { depth: 2, shadowPoints: options.shadowPoints ?? 129, method: "marginal" });
  return (leaf: number, account: EventAccount, options: Options = {}) => searchEventTwoStep(snapshot, leaf, account, frozenCosts, terminal, options, compiled);
}

/** Two-event Bellman search on the full order lattice, with exact one-event
 * continuation. A root interval is bounded by shadow-price dual tangents of
 * continuous one-event relaxations. Cap-recovery exceptions use dominating
 * cash/asset portfolios: the feasible-action set is not globally monotone.
 * Marked wealth and proportional liquidation are monotone in both holdings;
 * dust-dependent terminal liquidation is not supported by this bound.
 * A budget stop retains the unresolved upper/lower gap. */
export function decideEventTwoStep(model: EventDistribution, leaf: number, account: EventAccount, costs: EventCosts,
  terminal: Terminal, options: Options = {}) {
  validateEventDistribution(model); validateEventCosts(costs);
  return searchEventTwoStep(model, leaf, account, costs, terminal, options, compiledLaw());
}

function searchEventTwoStep(model: EventDistribution, leaf: number, account: EventAccount, costs: EventCosts,
  terminal: Terminal, options: Options, { relaxations, continuations, globalUpper }: Compiled) {
  const tolerance = options.tolerance ?? 1e-7, maxEvaluations = options.maxEvaluations ?? 128, padding = 1e-11;
  if (!(account.equity > 0 && account.price > 0) || !Object.values(account).every(Number.isFinite) || !model.kernels[leaf]
    || !["marked", "friction"].includes(terminal) || !(tolerance >= 4 * padding) || !Number.isFinite(tolerance)
    || !Number.isInteger(maxEvaluations) || maxEvaluations < 2) throw new Error("Invalid two-event search problem");
  const { equity: E, price: P, exposure: x } = account, f = (costs.feeBps + costs.slippageBps) / 10000;
  const step = costs.quantityStep, unit = P * step, Q = x * E / P, C = E - Q * P;
  const maximum = Math.floor(costs.maxNotional / unit + 1e-8), atoms = model.kernels[leaf].filter(a => a.probability > 0);
  if (!Number.isSafeInteger(maximum)) throw new Error("Two-event lattice exceeds integer precision");
  const stats = { evaluations: 0, actionEvaluations: 0, boundEvaluations: 0, continuationCalls: 0,
    exceptionEvaluations: 0, prunedIntervals: 0 };
  const cache = new Map<number, { trade: EventTrade | null; value: number }>();
  let best: EventTrade = { ...account, quantity: 0, turnover: 0, cost: 0 }, lower = -Infinity, hasAction = false;
  let prunedUpper = -Infinity;
  const trade = (k: number): EventTrade | null => {
    if (!k) return Math.abs(x) <= costs.maxLeverage + 1e-9 ? { ...account, quantity: 0, turnover: 0, cost: 0 } : null;
    const quantity = k * step, turnover = Math.abs(quantity) * P, cost = turnover * f, equity = E - cost;
    if (Math.abs(quantity) < costs.minQuantity - 1e-12 || turnover < costs.minNotional - 1e-8
      || turnover > costs.maxNotional + 1e-8 || equity <= 0) return null;
    const exposure = (Q + quantity) * P / equity;
    return eventCapAllowsTrade(x, exposure, turnover, P, costs) ? { equity, price: P, exposure, quantity, turnover, cost } : null;
  };
  const oneStep = (next: number) => {
    let solve = continuations.get(next);
    if (!solve) { solve = prepareEventOneStep(model.kernels[next], costs, terminal); continuations.set(next, solve); }
    return solve;
  };
  const continuation = (next: number, a: EventAccount) => {
    stats.continuationCalls++;
    return oneStep(next).value(a);
  };
  const action = (k: number): number | undefined => {
    if (!Number.isSafeInteger(k) || Math.abs(k) > maximum) return -Infinity;
    const cached = cache.get(k); if (cached) return cached.value;
    const next = trade(k);
    if (!next) { cache.set(k, { trade: null, value: -Infinity }); return -Infinity; }
    if (stats.evaluations >= maxEvaluations) return undefined;
    stats.evaluations++; stats.actionEvaluations++;
    let value = Math.log(next.equity / E);
    for (const a of atoms) {
      const h = eventHolding(next.exposure, a, costs);
      if (h.liquidated) { value = -Infinity; break; }
      const future = { equity: next.equity * h.factor, price: P * (1 + a.return), exposure: h.exposure };
      const v = continuation(a.next, future);
      if (!Number.isFinite(v)) { value = -Infinity; break; }
      value += a.probability * (Math.log(h.factor) + v);
    }
    cache.set(k, { trade: next, value });
    if (!hasAction || value > lower + 1e-14 || (value === lower && Math.abs(next.quantity) < Math.abs(best.quantity))) {
      best = next; lower = value; hasAction = true;
    }
    return value;
  };
  const cashAfter = (k: number, a: MoveAtom) => {
    const q = Q + k * step, cash = C - k * unit - Math.abs(k) * unit * f;
    const debt = q < 0 ? -q * P : Math.max(0, -cash);
    return cash - debt * (q < 0 ? costs.shortBorrowBpsPerDay : costs.longBorrowBpsPerDay) / 10000 * a.duration / 1440;
  };
  const freeCapValue = (a: EventAccount, quantity: number, kernel: readonly MoveAtom[]) => {
    // Used only for an upper bound on exceptional maximum-notional recovery
    // orders that might be feasible somewhere in the original root interval.
    stats.exceptionEvaluations++;
    const equity = a.equity - Math.abs(quantity) * a.price * f;
    if (!(equity > 0)) return -Infinity;
    const exposure = (a.exposure * a.equity + quantity * a.price) / equity;
    let value = Math.log(equity / a.equity);
    for (const atom of kernel) if (atom.probability > 0) {
      const held = eventHolding(exposure, atom, costs);
      const factor = held.factor - (terminal === "friction" ? f * Math.abs(exposure) * (1 + atom.return) : 0);
      if (held.liquidated || !(factor > 0)) return -Infinity;
      value += atom.probability * Math.log(factor);
    }
    return value;
  };
  const upper = (low: number, high: number): number => {
    if (low === high) return action(low) ?? Infinity;
    if (stats.evaluations >= maxEvaluations) return Infinity;
    stats.evaluations++; stats.boundEvaluations++;
    const critical = [low, high, 0, -Q / step, C / (unit * (1 + f)), C / (unit * (1 - f))]
      .filter(k => k >= low && k <= high);
    // A tangent at the incumbent is tight where it matters if that action is
    // already optimal. Midpoint tangents waste subdivision near a no-trade kink.
    const qMin = Q + low * step, qMax = Q + high * step;
    const middle = Number.isFinite(lower) ? Math.min(high, Math.max(low, best.quantity / step)) : low + (high - low) / 2;
    const values = critical.map(() => 0);
    for (const a of atoms) {
      const nextPrice = P * (1 + a.return), cashes = critical.map(k => cashAfter(k, a));
      const cashMax = Math.max(...cashes), equity = cashMax + qMax * nextPrice;
      if (!(equity > 0)) return -Infinity; // No surviving account can exceed this upper portfolio.
      const dominant = { equity, price: nextPrice, exposure: qMax * nextPrice / equity };
      let relax = relaxations.get(a.next);
      if (!relax) { relax = eventOneStepUpper(model.kernels[a.next], costs, terminal); relaxations.set(a.next, relax); }
      let atCash = cashAfter(middle, a), atQuantity = Q + middle * step, tangent = relax(atCash, atQuantity, nextPrice);
      if (!tangent) { atCash = cashMax; atQuantity = qMax; tangent = relax(atCash, atQuantity, nextPrice); }
      if (!tangent) return Infinity;
      let exceptional = -Infinity;
      // Bound possible pre/post exposure ratios conservatively over the
      // cash/quantity box. Only recovery clips outside the ordinary cap need
      // this exception; ordinary cap-feasible actions preserve dominance.
      const equities = critical.map((k, i) => cashes[i] + (Q + k * step) * nextPrice);
      const preAbsMax = Math.min(...equities) > 0
        ? Math.max(...critical.map((k, i) => Math.abs(Q + k * step) * nextPrice / equities[i])) : Infinity;
      if (preAbsMax > costs.maxLeverage) {
        const maxLots = Math.floor(costs.maxNotional / nextPrice / step + 1e-8);
        const firstLot = Math.max(1, Math.ceil((costs.maxNotional - step * nextPrice - 1e-8) / nextPrice / step));
        if (maxLots - firstLot > 4) return Infinity; // Extremely fine tolerance band: keep the bound honest.
        for (let lots = firstLot; lots <= maxLots; lots++) for (const side of [-1, 1]) {
          const quantity = side * lots * step, lo = qMin + quantity, hi = qMax + quantity;
          // If this entire clip restores the ordinary cap, the normal bound
          // already covers it. Between critical knots cash/equity are affine;
          // absolute notional / positive equity has its maximum at an endpoint.
          const postEquities = equities.map(e => e - Math.abs(quantity) * nextPrice * f);
          if (postEquities.every(e => e > 0) && critical.every((k, i) =>
            Math.abs(Q + k * step + quantity) * nextPrice <= costs.maxLeverage * postEquities[i])) continue;
          const minAbs = lo <= 0 && hi >= 0 ? 0 : Math.min(Math.abs(lo), Math.abs(hi));
          const eMax = cashMax + qMax * nextPrice - Math.abs(quantity) * nextPrice * f;
          if (!(eMax > 0) || minAbs * nextPrice / eMax >= preAbsMax - 1e-9) continue;
          exceptional = Math.max(exceptional, Math.log(equity) + freeCapValue(dominant, quantity, model.kernels[a.next]));
        }
      }
      for (let i = 0; i < critical.length; i++) {
        const normal = tangent.value + tangent.dCash * (cashes[i] - atCash)
          + tangent.dQuantity * (Q + critical[i] * step - atQuantity);
        values[i] += a.probability * Math.max(normal, exceptional);
      }
    }
    // Between cash/sign/borrowing knots each tangent is affine. The maximum
    // with exceptional-action constants is convex, hence its maximum over
    // each piece occurs at an endpoint. Use ONE root quantity across outcomes.
    return Math.max(...values) - Math.log(E) + padding;
  };
  // Ordinary cap-feasible orders form an interval. Maximum-notional recovery
  // clips can lie outside it and are included as separate intervals.
  const ranges: Array<[number, number]> = [];
  const inverse = (target: number) => (target * E - Q * P) / (unit * (1 + target * f * Math.sign(target - x)));
  if (f * Math.abs(x) < 1) {
    const low = Math.max(-maximum, Math.ceil(inverse(-costs.maxLeverage - 1e-8)) - 2);
    const high = Math.min(maximum, Math.floor(inverse(costs.maxLeverage + 1e-8)) + 2);
    if (low <= high) ranges.push([low, high]);
  } else ranges.push([-maximum, maximum]); // Degenerate exposure: retain the full lattice conservatively.
  if (Math.abs(x) > costs.maxLeverage) {
    const clip = Math.max(1, Math.ceil((costs.maxNotional - unit - 1e-8) / unit));
    if (clip <= maximum) ranges.push([-maximum, -clip], [clip, maximum]);
  }
  if (Math.abs(x) <= costs.maxLeverage + 1e-9) ranges.push([0, 0]);
  ranges.sort((a, b) => a[0] - b[0]);
  const merged: Array<[number, number]> = [];
  for (const range of ranges) {
    const last = merged.at(-1);
    if (last && range[0] <= last[1] + 1) last[1] = Math.max(last[1], range[1]); else merged.push([...range]);
  }
  const global = globalUpper?.query(leaf, account, 2), globalBound = global?.upperValue ?? Infinity;
  const needsCandidate = () => globalBound > lower + tolerance;
  action(0);
  // The relaxation supplies targets, never executable values. Validate and
  // evaluate neighboring integer orders under the full original law. A tight
  // bound can then avoid both poorer seeds and interval subdivision.
  if (global && Number.isFinite(globalBound)) for (const [i, side] of [-1, 1].entries()) {
    const target = Math.max(-costs.maxLeverage, Math.min(costs.maxLeverage, global.seedExposures[i]));
    if (Math.sign(target - x) !== side || !needsCandidate()) continue;
    const lots = (target * E - Q * P) / (unit * (1 + target * f * side));
    action(Math.floor(lots));
    if (needsCandidate()) action(Math.ceil(lots));
  }
  if (needsCandidate()) {
    const h1 = oneStep(leaf)(account); action(Math.round(h1.quantity / step));
  }
  for (const quantity of options.seedQuantities ?? []) if (needsCandidate()) action(Math.round(quantity / step));
  const initial = { quantity: best.quantity, value: lower,
    candidates: [...cache].map(([k, r]) => ({ quantity: k * step, feasible: Boolean(r.trade), value: r.value })) };
  if (globalBound < lower - 2e-10) throw new Error("Recursive upper bound below a feasible two-event value");
  const finish = (bound: number, remainingIntervals: number) => {
    const upperValue = Math.max(lower, Math.min(bound, globalBound)), gap = upperValue === lower ? 0 : upperValue - lower;
    return { ...eventTradeDecision(account, best, lower), feasible: hasAction, lowerValue: lower, upperValue,
      gap, converged: gap <= tolerance, tolerance, terminal, initial, remainingIntervals, search: stats,
      ...(globalUpper ? { globalUpperValue: globalBound } : {}) };
  };
  if (globalBound <= lower + tolerance) return finish(globalBound, 0);
  const frontier: Bound[] = merged.map(([low, high]) => ({ low, high, upper: upper(low, high) }));
  while (frontier.length && stats.evaluations < maxEvaluations) {
    frontier.sort((a, b) => b.upper - a.upper);
    const node = frontier.shift()!;
    if (node.upper <= lower + tolerance) {
      stats.prunedIntervals++; prunedUpper = Math.max(prunedUpper, node.upper); continue;
    }
    if (node.low === node.high) { action(node.low); continue; }
    const middle = Math.floor((node.low + node.high) / 2);
    action(middle);
    if (globalBound <= lower + tolerance) return finish(globalBound, 0);
    for (const [low, high] of [[node.low, middle - 1], [middle + 1, node.high]]) if (low <= high)
      frontier.push({ low, high, upper: Math.min(node.upper, upper(low, high)) });
  }
  return finish(Math.max(lower, prunedUpper, ...frontier.map(n => n.upper)), frontier.length);
}
