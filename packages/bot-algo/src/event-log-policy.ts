import { eventLeaf, validateEventDistribution, type EventDistribution, type MoveAtom } from "./event-distribution.js";

export interface EventAccount { equity: number; price: number; exposure: number; }
export interface EventCosts {
  feeBps: number; slippageBps: number; maxLeverage: number; maintenanceMargin: number;
  minNotional: number; maxNotional: number; minQuantity: number; quantityStep: number;
  longBorrowBpsPerDay: number; shortBorrowBpsPerDay: number;
}
export const DEFAULT_EVENT_COSTS: EventCosts = {
  feeBps: 10, slippageBps: 2, maxLeverage: 5, maintenanceMargin: 0.005,
  minNotional: 5, maxNotional: 50_000, minQuantity: 0.00001, quantityStep: 0.00001,
  longBorrowBpsPerDay: 1, shortBorrowBpsPerDay: 1,
};
export interface EventTrade extends EventAccount { quantity: number; turnover: number; cost: number; }
export function eventCapAllowsTrade(previous: number, next: number, turnover: number, price: number, c: EventCosts): boolean {
  return Math.abs(next) <= c.maxLeverage + 1e-8
    || (Math.abs(previous) > c.maxLeverage && Math.abs(next) < Math.abs(previous) - 1e-9
      && turnover >= c.maxNotional - c.quantityStep * price - 1e-8);
}
export function validateEventCosts(c: EventCosts) {
  if (Object.values(c).some(v => !Number.isFinite(v) || v < 0) || c.maxLeverage < 1
    || c.maxNotional < c.minNotional || c.quantityStep <= 0 || c.maintenanceMargin >= 1
    || (c.feeBps + c.slippageBps) * c.maxLeverage / 1e4 >= 1) throw new Error("Invalid event account costs");
}

/** Solve post-fee target exposure, then round the ORDER (not exposure) to lots. */
export function eventTrade(account: EventAccount, target: number, c: EventCosts): EventTrade | null {
  const { equity, price, exposure } = account;
  if (!(equity > 0 && price > 0) || !Number.isFinite(target)) return null;
  if (Math.abs(target - exposure) < 1e-12) return { ...account, quantity: 0, turnover: 0, cost: 0 };
  const riskReducing = Math.abs(account.exposure) > c.maxLeverage + 1e-9
    && Math.abs(target) < Math.abs(account.exposure) - 1e-9;
  if (Math.abs(target) > c.maxLeverage + 1e-9 && !riskReducing) return null;
  const f = (c.feeBps + c.slippageBps) / 1e4, sign = Math.sign(target - exposure);
  const factor = (1 + f * sign * exposure) / (1 + f * sign * target);
  const intended = (target * equity * factor - exposure * equity) / price;
  let quantity = Math.sign(intended) * Math.floor(Math.abs(intended) / c.quantityStep + 1e-8) * c.quantityStep;
  // When marked exposure has drifted above the target cap, rounding toward zero
  // must not leave an otherwise feasible restoring order a fraction above it.
  const projected = (q: number) => {
    const turnover = Math.abs(q) * price, nextEquity = equity - turnover * (c.feeBps + c.slippageBps) / 1e4;
    return nextEquity > 0 ? (exposure * equity + q * price) / nextEquity : Infinity;
  };
  if (riskReducing && Math.abs(target) <= c.maxLeverage && Math.abs(projected(quantity)) > c.maxLeverage + 1e-8) {
    quantity += Math.sign(intended) * c.quantityStep;
  }
  const turnover = Math.abs(quantity) * price;
  if (Math.abs(quantity) < c.minQuantity - 1e-12 || turnover < c.minNotional - 1e-8
    || turnover > c.maxNotional + 1e-8 || !quantity) return null;
  const cost = turnover * f, nextEquity = equity - cost;
  if (nextEquity <= 0) return null;
  const nextExposure = (exposure * equity + quantity * price) / nextEquity;
  if (!eventCapAllowsTrade(exposure, nextExposure, turnover, price, c)) return null;
  return { equity: nextEquity, price, exposure: nextExposure, quantity, turnover, cost };
}

export function eventHolding(exposure: number, atom: Pick<MoveAtom, "return" | "low" | "high" | "duration">, c: EventCosts) {
  const borrow = (exposure > 0 ? Math.max(0, exposure - 1) * c.longBorrowBpsPerDay
    : -exposure * c.shortBorrowBpsPerDay) / 1e4 * atom.duration / 1440;
  const worst = exposure >= 0 ? atom.low : atom.high;
  const minimum = 1 + exposure * worst - borrow;
  const factor = 1 + exposure * atom.return - borrow;
  const liquidated = minimum <= c.maintenanceMargin * Math.abs(exposure) * (1 + worst) || factor <= 0;
  return { factor, exposure: exposure * (1 + atom.return) / factor, liquidated };
}

export interface EventConvergence { changedActionFraction: number | null; maxActionChange: number | null; valueIncrementSpan: number | null; }
export interface EventValueTable { depth: number; holdValues: Float64Array; convergence?: EventConvergence; }
export interface EventPolicy {
  model: EventDistribution; costs: EventCosts; equities: number[]; prices: number[];
  exposures: number[]; targets: number[]; tables: EventValueTable[];
}
export interface SerializedEventPolicy extends Omit<EventPolicy, "tables"> {
  contract: "event-log-policy-v1";
  tables: { depth: number; holdValues: (number | null)[]; convergence?: EventConvergence }[];
}
export function serializeEventPolicy(p: EventPolicy): SerializedEventPolicy {
  return { ...p, contract: "event-log-policy-v1", tables: p.tables.map(t => ({ ...t,
    holdValues: Array.from(t.holdValues, v => Number.isFinite(v) ? v : null) })) };
}
export function restoreEventPolicy(p: SerializedEventPolicy): EventPolicy {
  if (p.contract !== "event-log-policy-v1") throw new Error("Unknown event policy artifact");
  validateEventDistribution(p.model);
  validateEventCosts(p.costs);
  const size = p.model.kernels.length * p.equities.length * p.prices.length * p.exposures.length;
  for (const axis of [p.equities, p.prices, p.exposures]) {
    if (axis.length < 2 || axis.some((v, i) => !Number.isFinite(v) || (i > 0 && v <= axis[i - 1]))) throw new Error("Invalid policy grid");
  }
  if (p.tables.some((t, i) => t.depth !== i + 1 || t.holdValues.length !== size
    || t.holdValues.some(v => v !== null && !Number.isFinite(v)))) throw new Error("Invalid policy values");
  return { ...p, tables: p.tables.map(t => ({ ...t,
    holdValues: Float64Array.from(t.holdValues, v => v === null ? -Infinity : v) })) };
}

function bracket(axis: number[], value: number): [number, number, number] {
  if (value <= axis[0]) return [0, 0, 0];
  const last = axis.length - 1;
  if (value >= axis[last]) return [last, last, 0];
  let low = 0, high = last;
  while (high - low > 1) { const m = (low + high) >> 1; if (axis[m] <= value) low = m; else high = m; }
  return [low, high, (value - axis[low]) / (axis[high] - axis[low])];
}
function offset(p: EventPolicy, leaf: number, w: number, price: number, exposure: number) {
  return (((leaf * p.equities.length + w) * p.prices.length + price) * p.exposures.length + exposure);
}
function lookup(p: EventPolicy, values: Float64Array, leaf: number, account: EventAccount): number {
  const [wl, wh, wf] = bracket(p.equities, account.equity);
  const [pl, ph, pf] = bracket(p.prices, account.price);
  const [xl, xh, xf] = bracket(p.exposures, account.exposure);
  let result = 0;
  for (let w = 0; w < 2; w++) for (let price = 0; price < 2; price++) for (let x = 0; x < 2; x++) {
    const weight = (w ? wf : 1 - wf) * (price ? pf : 1 - pf) * (x ? xf : 1 - xf);
    if (weight) result += weight * values[offset(p, leaf, w ? wh : wl, price ? ph : pl, x ? xh : xl)];
  }
  return result;
}

function candidates(p: Pick<EventPolicy, "costs" | "targets">, account: EventAccount): number[] {
  const f = (p.costs.feeBps + p.costs.slippageBps) / 1e4;
  const values = [...p.targets, account.exposure];
  // Compare minimum and maximum feasible orders to doing nothing. Never blindly round up.
  const minimum = Math.ceil(Math.max(p.costs.minQuantity, p.costs.minNotional / account.price) / p.costs.quantityStep) * p.costs.quantityStep;
  const maximum = Math.floor(p.costs.maxNotional / account.price / p.costs.quantityStep) * p.costs.quantityStep;
  for (const quantity of [minimum, maximum]) for (const side of [-1, 1]) {
    const notional = quantity * account.price;
    if (account.equity > f * notional) values.push((account.exposure * account.equity + side * notional) / (account.equity - f * notional));
  }
  return values;
}

export function decideEvent(p: EventPolicy, leaf: number, account: EventAccount, depth: number) {
  const table = p.tables[depth - 1];
  if (!table) throw new Error(`Bellman depth ${depth} is unavailable`);
  return chooseEventTrade(p, account, held => lookup(p, table.holdValues, leaf, held));
}

/** Common feasible-action optimizer for distribution and fitted-value policies. */
export function chooseEventTrade(p: Pick<EventPolicy, "costs" | "targets">, account: EventAccount, holdValue: (account: EventAccount) => number) {
  let best: EventTrade = { ...account, quantity: 0, turnover: 0, cost: 0 };
  // The target cap is checked at every event. Price drift may carry inventory
  // outside it during a move, in which case doing nothing is no longer feasible.
  let value = Math.abs(account.exposure) <= p.costs.maxLeverage + 1e-9
    ? holdValue(account) : -Infinity;
  for (const target of candidates(p, account)) {
    const trade = eventTrade(account, target, p.costs);
    if (!trade) continue;
    if (!trade.quantity && Math.abs(account.exposure) > p.costs.maxLeverage + 1e-9) continue;
    const score = Math.log(trade.equity / account.equity) + holdValue(trade);
    if (score > value + 1e-12 || (score === value && !Number.isFinite(value)
      && Math.abs(account.exposure) > p.costs.maxLeverage
      && Math.abs(trade.exposure) < Math.abs(best.exposure))) { value = score; best = trade; }
  }
  return eventTradeDecision(account, best, value);
}

/** Classify an aggregate exposure change. These flags do not implement
 * lifecycle position decomposition or prove an entry/exit Bellman symmetry. */
export function eventTradeDecision(account: EventAccount, best: EventTrade, value: number) {
  const previous = account.exposure, next = best.exposure;
  return { ...best, value, longEntry: Math.max(0, next) > Math.max(0, previous) + 1e-9,
    longExit: Math.max(0, next) < Math.max(0, previous) - 1e-9,
    shortEntry: Math.min(0, next) < Math.min(0, previous) - 1e-9,
    shortExit: Math.min(0, next) > Math.min(0, previous) + 1e-9 };
}

export interface EventOutcomeLookahead {
  policy: EventPolicy;
  masses: number[][];
  /** Outer index is depth minus one; inner index is outcome group. */
  holds: Float64Array[][];
}

/** One Bellman backup with the new current sign law and the frozen base value
 * V_(depth-1) thereafter. At depth one this is the full terminal-settlement
 * problem. At larger depths it is a policy-improvement approximation, NOT
 * recursive propagation of the new sign head through hypothetical features.
 * Conditional operators preserve every joint atom and its original successor. */
export function buildEventOutcomeLookahead(p: EventPolicy, groupCount: number, group: (atom: MoveAtom) => number): EventOutcomeLookahead {
  if (!Number.isInteger(groupCount) || groupCount < 1 || groupCount > 16) throw new Error("Invalid outcome-group count");
  const groups = p.model.kernels.map(kernel => kernel.map(group));
  if (groups.some(row => row.some(g => !Number.isInteger(g) || g < 0 || g >= groupCount))) throw new Error("Invalid outcome group");
  const masses = p.model.kernels.map((kernel, leaf) => {
    const m = new Array<number>(groupCount).fill(0);
    for (let i = 0; i < kernel.length; i++) m[groups[leaf][i]] += kernel[i].probability;
    return m;
  });
  const operators = Array.from({ length: groupCount }, (_, sign) => compileEventOperator({ ...p, model: { ...p.model,
    kernels: p.model.kernels.map((kernel, leaf) => masses[leaf][sign]
      ? kernel.filter((_, i) => groups[leaf][i] === sign).map(a => ({ ...a, probability: a.probability / masses[leaf][sign] }))
      : []) } }));
  const size = p.tables[0].holdValues.length, holds: Float64Array[][] = [];
  for (let depth = 1; depth <= p.tables.length; depth++) {
    const previous = new Float64Array(size);
    for (let leaf = 0; leaf < p.model.kernels.length; leaf++) for (let w = 0; w < p.equities.length; w++)
      for (let price = 0; price < p.prices.length; price++) for (let x = 0; x < p.exposures.length; x++) {
        const account = { equity: p.equities[w], price: p.prices[price], exposure: p.exposures[x] };
        const settlement = 1 - Math.abs(account.exposure) * (p.costs.feeBps + p.costs.slippageBps) / 1e4;
        previous[offset(p, leaf, w, price, x)] = depth > 1 ? decideEvent(p, leaf, account, depth - 1).value
          : settlement > 0 ? Math.log(settlement) : -Infinity;
      }
    holds.push(operators.map(operator => Float64Array.from(operator, row => {
      let value = row.reward;
      for (let j = 0; j < row.targets.length; j++) value += row.weights[j] * previous[row.targets[j]];
      return value;
    })));
  }
  return { policy: p, masses, holds };
}

export function buildEventSignLookahead(p: EventPolicy): EventOutcomeLookahead {
  return buildEventOutcomeLookahead(p, 3, atom => Math.sign(atom.return) + 1);
}

/** Conditional holding values before mixing outcome probabilities. Useful for
 * fitting the forecast to the planner's fixed value function. */
export function eventOutcomeHoldingValues(lookahead: EventOutcomeLookahead, leaf: number, account: EventAccount, depth: number) {
  const tables = lookahead.holds[depth - 1];
  if (!tables || !lookahead.masses[leaf]) throw new Error("Unavailable conditional holding values");
  return tables.map(table => lookup(lookahead.policy, table, leaf, account));
}

export interface EventKernelLookahead {
  policy: EventPolicy;
  previous: Float64Array[];
  /** Lazy per-grid-cell, per-joint-atom holding values. No forecast weights. */
  cells: Map<number, Float64Array>;
}

/** Fixed continuation tables for an exact first backup with arbitrary current
 * probabilities on the SAME joint atoms. Lazy cell values avoid rebuilding
 * Bellman policies for each continuously varying severity forecast. */
export function buildEventKernelLookahead(p: EventPolicy): EventKernelLookahead {
  const size = p.tables[0].holdValues.length;
  const previous = p.tables.map(({ depth }) => {
    const values = new Float64Array(size);
    for (let leaf = 0; leaf < p.model.kernels.length; leaf++) for (let w = 0; w < p.equities.length; w++)
      for (let price = 0; price < p.prices.length; price++) for (let x = 0; x < p.exposures.length; x++) {
        const account = { equity: p.equities[w], price: p.prices[price], exposure: p.exposures[x] };
        const settlement = 1 - Math.abs(account.exposure) * (p.costs.feeBps + p.costs.slippageBps) / 1e4;
        values[offset(p, leaf, w, price, x)] = depth > 1 ? decideEvent(p, leaf, account, depth - 1).value
          : settlement > 0 ? Math.log(settlement) : -Infinity;
      }
    return values;
  });
  return { policy: p, previous, cells: new Map() };
}

export function decideEventKernel(lookahead: EventKernelLookahead, leaf: number, account: EventAccount, depth: number, kernel: readonly MoveAtom[]) {
  const p = lookahead.policy, original = p.model.kernels[leaf], previous = lookahead.previous[depth - 1];
  if (!original || !previous || kernel.length !== original.length || kernel.some((a, i) => {
    const b = original[i];
    return !Number.isFinite(a.probability) || (b.probability > 0 ? a.probability <= 0 : a.probability !== 0)
      || a.return !== b.return || a.low !== b.low || a.high !== b.high || a.duration !== b.duration || a.next !== b.next;
  }) || Math.abs(kernel.reduce((s, a) => s + a.probability, 0) - 1) > 1e-8) throw new Error("Current kernel must preserve joint atoms and supported tails");
  const size = previous.length, values = new Map<number, number>();
  const cellValue = (w: number, price: number, x: number) => {
    const index = offset(p, leaf, w, price, x), key = (depth - 1) * size + index;
    if (values.has(index)) return values.get(index)!;
    let cells = lookahead.cells.get(key);
    if (!cells) {
      cells = Float64Array.from(original, atom => {
        const held = eventHolding(p.exposures[x], atom, p.costs);
        return held.liquidated ? -Infinity : Math.log(held.factor) + lookup(p, previous, atom.next,
          { equity: p.equities[w] * held.factor, price: p.prices[price] * (1 + atom.return), exposure: held.exposure });
      });
      lookahead.cells.set(key, cells);
    }
    let result = 0;
    for (let i = 0; i < kernel.length; i++) if (kernel[i].probability) result += kernel[i].probability * cells[i];
    values.set(index, result);
    return result;
  };
  return chooseEventTrade(p, account, held => {
    const [wl, wh, wf] = bracket(p.equities, held.equity), [pl, ph, pf] = bracket(p.prices, held.price), [xl, xh, xf] = bracket(p.exposures, held.exposure);
    let value = 0;
    for (let w = 0; w < 2; w++) for (let price = 0; price < 2; price++) for (let x = 0; x < 2; x++) {
      const weight = (w ? wf : 1 - wf) * (price ? pf : 1 - pf) * (x ? xf : 1 - xf);
      if (weight) value += weight * cellValue(w ? wh : wl, price ? ph : pl, x ? xh : xl);
    }
    return value;
  });
}

export function decideEventSign(lookahead: EventOutcomeLookahead, leaf: number, account: EventAccount, depth: number, probability: number) {
  if (!Number.isFinite(probability) || probability < 0 || probability > 1) throw new Error("Invalid sign probability");
  const { policy: p, masses, holds } = lookahead, base = masses[leaf], tables = holds[depth - 1];
  if (!base || !tables) throw new Error("Unavailable sign lookahead state or depth");
  if (!base[0] || !base[2]) return decideEvent(p, leaf, account, depth);
  if (probability === 0 || probability === 1) throw new Error("Sign lookahead cannot remove an existing tail sign");
  const active = base[0] + base[2], mass = [active * (1 - probability), base[1], active * probability];
  return decideEventOutcomes(lookahead, leaf, account, depth, mass);
}

export function decideEventOutcomes(lookahead: EventOutcomeLookahead, leaf: number, account: EventAccount, depth: number, mass: readonly number[]) {
  return chooseEventTrade(lookahead.policy, account, outcomeHoldingValue(lookahead, leaf, depth, mass));
}

function outcomeHoldingValue(lookahead: EventOutcomeLookahead, leaf: number, depth: number, mass: readonly number[]) {
  const { policy: p, masses, holds } = lookahead, base = masses[leaf], tables = holds[depth - 1];
  if (!base || !tables || mass.length !== base.length || mass.some((v, i) => !Number.isFinite(v) || v < 0
    || (base[i] > 0 ? v <= 0 : v > 0)) || Math.abs(mass.reduce((s, v) => s + v, 0) - base.reduce((s, v) => s + v, 0)) > 1e-8)
    throw new Error("Outcome probabilities must preserve total mass and supported tail groups");
  return (held: EventAccount) => {
    let value = 0;
    for (let sign = 0; sign < mass.length; sign++) if (mass[sign]) value += mass[sign] * lookup(p, tables[sign], leaf, held);
    return value;
  };
}

/** Inspect the same feasible, lot-rounded candidates and interpolated values
 * used by the policy. Values include the immediate order cost. This is a
 * diagnostic, not a separate optimizer or a realized-future action selector. */
export function eventActionValues(p: EventPolicy, leaf: number, account: EventAccount, depth: number,
  outcomes?: { lookahead: EventOutcomeLookahead; mass: readonly number[] }) {
  const table = p.tables[depth - 1];
  if (!table || !p.model.kernels[leaf]) throw new Error("Unavailable action-value state or depth");
  if (outcomes && outcomes.lookahead.policy !== p) throw new Error("Mismatched action-value lookahead");
  const holdValue = outcomes ? outcomeHoldingValue(outcomes.lookahead, leaf, depth, outcomes.mass)
    : (held: EventAccount) => lookup(p, table.holdValues, leaf, held);
  const seen = new Set<number>();
  return [account.exposure, ...candidates(p, account)].flatMap(target => {
    const trade = eventTrade(account, target, p.costs);
    if (!trade || (!trade.quantity && Math.abs(account.exposure) > p.costs.maxLeverage + 1e-9)
      || seen.has(trade.quantity)) return [];
    seen.add(trade.quantity);
    const orderLogCost = Math.log(trade.equity / account.equity);
    return [{ ...trade, target, orderLogCost, value: orderLogCost + holdValue(trade) }];
  });
}

interface EventOperatorRow { reward: number; targets: Uint32Array; weights: Float64Array; }
/** The hold Bellman operator is affine in the previous value table. Aggregate
 * identical interpolation destinations once instead of rebuilding brackets for
 * every atom, account cell and recursion depth. No outcome is discarded. */
function compileEventOperator(p: EventPolicy): EventOperatorRow[] {
  const rows = new Array<EventOperatorRow>(p.model.kernels.length * p.equities.length * p.prices.length * p.exposures.length);
  for (let leaf = 0; leaf < p.model.kernels.length; leaf++) for (let x = 0; x < p.exposures.length; x++) {
    const transitions = p.model.kernels[leaf].filter(a => a.probability > 0).map(atom =>
      ({ atom, ...eventHolding(p.exposures[x], atom, p.costs) }));
    const ruined = transitions.some(t => t.liquidated);
    const reward = ruined ? -Infinity : transitions.reduce((sum, t) => sum + t.atom.probability * Math.log(t.factor), 0);
    for (let w = 0; w < p.equities.length; w++) for (let price = 0; price < p.prices.length; price++) {
      const coefficients = new Map<number, number>();
      if (!ruined) for (const t of transitions) {
        const [wl, wh, wf] = bracket(p.equities, p.equities[w] * t.factor);
        const [pl, ph, pf] = bracket(p.prices, p.prices[price] * (1 + t.atom.return));
        const [xl, xh, xf] = bracket(p.exposures, t.exposure);
        for (let wi = 0; wi < 2; wi++) for (let pi = 0; pi < 2; pi++) for (let xi = 0; xi < 2; xi++) {
          const weight = (wi ? wf : 1 - wf) * (pi ? pf : 1 - pf) * (xi ? xf : 1 - xf);
          if (!weight) continue;
          const index = offset(p, t.atom.next, wi ? wh : wl, pi ? ph : pl, xi ? xh : xl);
          // A positive but underflowed coefficient must still propagate ruin.
          const mass = t.atom.probability * weight || Number.MIN_VALUE;
          coefficients.set(index, (coefficients.get(index) ?? 0) + mass);
        }
      }
      rows[offset(p, leaf, w, price, x)] = { reward,
        targets: Uint32Array.from(coefficients.keys()), weights: Float64Array.from(coefficients.values()) };
    }
  }
  return rows;
}

/** Fitted finite-state Bellman iteration over complete next-move distributions.
 * max is INSIDE the conditional next-state expectation, never over a realized
 * future path. Inventory, equity and price are retained on an interpolation grid.
 * Every depth includes cash settlement at its terminal boundary. */
export function buildEventPolicy(model: EventDistribution, costs: EventCosts,
  options: { depths: number; referenceEquity: number; referencePrice: number; actionSteps?: number }): EventPolicy {
  validateEventCosts(costs);
  validateEventDistribution(model);
  const steps = options.actionSteps ?? 10, L = costs.maxLeverage;
  if (!Number.isInteger(options.depths) || options.depths < 0 || options.referenceEquity <= 0 || options.referencePrice <= 0 || steps < 1) throw new Error("Invalid Bellman grid");
  const targets = Array.from({ length: 2 * steps + 1 }, (_, i) => (i - steps) * L / steps);
  const p: EventPolicy = { model, costs, targets,
    equities: [0.25, 0.5, 1, 2, 4].map(v => v * options.referenceEquity),
    prices: [0.5, 1, 2].map(v => v * options.referencePrice),
    exposures: [-2 * L, ...targets, 2 * L], tables: [] };
  // Exact H1/H2/H3 solvers use the joint law directly, without grid tables.
  if (options.depths === 0) return p;
  const size = model.kernels.length * p.equities.length * p.prices.length * p.exposures.length;
  const operator = compileEventOperator(p);
  let previous = new Float64Array(size);
  let previousActions: Float64Array | undefined;
  for (let leaf = 0; leaf < model.kernels.length; leaf++) for (let w = 0; w < p.equities.length; w++)
    for (let price = 0; price < p.prices.length; price++) for (let x = 0; x < p.exposures.length; x++) {
      const settlement = 1 - Math.abs(p.exposures[x]) * (costs.feeBps + costs.slippageBps) / 1e4;
      previous[offset(p, leaf, w, price, x)] = settlement > 0 ? Math.log(settlement) : -Infinity;
    }
  for (let depth = 1; depth <= options.depths; depth++) {
    const holds = new Float64Array(size);
    for (let i = 0; i < size; i++) {
      const row = operator[i]; let value = row.reward;
      for (let j = 0; j < row.targets.length; j++) value += row.weights[j] * previous[row.targets[j]];
      holds[i] = value;
    }
    p.tables.push({ depth, holdValues: holds });
    const values = new Float64Array(size), actions = new Float64Array(size);
    let changed = 0, maxActionChange = 0, minIncrement = Infinity, maxIncrement = -Infinity;
    for (let leaf = 0; leaf < model.kernels.length; leaf++) for (let w = 0; w < p.equities.length; w++)
      for (let price = 0; price < p.prices.length; price++) for (let x = 0; x < p.exposures.length; x++) {
        const i = offset(p, leaf, w, price, x), decision = decideEvent(p, leaf,
          { equity: p.equities[w], price: p.prices[price], exposure: p.exposures[x] }, depth);
        values[i] = decision.value; actions[i] = decision.exposure;
        if (previousActions) {
          const difference = Math.abs(actions[i] - previousActions[i]);
          if (difference > 1e-8) changed++;
          maxActionChange = Math.max(maxActionChange, difference);
        }
        if (Number.isFinite(values[i]) && Number.isFinite(previous[i])) {
          const increment = values[i] - previous[i];
          minIncrement = Math.min(minIncrement, increment); maxIncrement = Math.max(maxIncrement, increment);
        }
      }
    p.tables.at(-1)!.convergence = { changedActionFraction: previousActions ? changed / size : null,
      maxActionChange: previousActions ? maxActionChange : null,
      valueIncrementSpan: minIncrement < Infinity ? maxIncrement - minIncrement : null };
    previous = values; previousActions = actions;
  }
  return p;
}

export function predictEventAction(p: EventPolicy, features: readonly number[], account: EventAccount, depth: number) {
  return decideEvent(p, eventLeaf(p.model, features), account, depth);
}
