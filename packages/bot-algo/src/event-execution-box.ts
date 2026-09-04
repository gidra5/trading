import { cloneEventExecutionAtoms, type EventExecutionAtom } from "./event-execution-one-step.js";
import { maximizeEventAcceptanceSequence } from "./event-execution-acceptance.js";
import { eventAffineSegmentSupports } from "./event-affine-segment.js";

export interface EventExecutionBoxOptions {
  /** Preserve the interval/prefix/suffix structure of ordinary acceptance. */
  ordered?: boolean;
  /** Optimize financial wealth over a supplied two-vertex balance segment. */
  coupledWealth?: boolean;
  /** Applies to subdivisions after the initial guarded intervals. */
  maxRefinements?: number;
  /** Numerical gap for the relaxed problem, in absolute log-equity units. */
  valueTolerance?: number;
}

export interface EventExecutionAccountBox {
  price: number;
  cash: readonly [number, number];
  quantity: readonly [number, number];
  /** When supplied, cover only the convex hull of these [cash, quantity]
   * vertices inside the box. Preserve trade-induced balance dependence. */
  balanceVertices?: readonly (readonly [number, number])[];
  /** Restrict the covered inventories to integer quantityStep multiples. */
  quantityLattice?: boolean;
}

/** Upper bound on expected marked LOG TERMINAL EQUITY for every live account
 * in a balance box, optimized over a common committed quantity request.
 * Cash/inventory maxima dominate financial payoffs, but order acceptance is
 * bounded using the whole box: uncertain opening-price groups may accept or
 * reject in their own favor. Outcomes with identical available openings share
 * one acceptance decision, as in the real executor. This preserves an upper
 * when extra wealth changes rejection masks.
 * The returned request belongs to the relaxation, not an executable policy. */
export function prepareEventExecutionBoxUpper(input: readonly EventExecutionAtom[], options: EventExecutionBoxOptions = {}) {
  const atoms = cloneEventExecutionAtoms(input), c = atoms[0].path.costs;
  if (Math.abs(atoms.reduce((s, a) => s + a.probability, 0) - 1) > 1e-12)
    throw new Error("Execution box requires normalized probability within 1e-12");
  const step = c.quantityStep, fee = (c.feeBps + c.slippageBps) / 10000, L = c.maxLeverage + 1e-8;
  const maxRefinements = options.maxRefinements ?? 10000, tolerance = options.valueTolerance ?? 1e-8;
  if (!(maxRefinements >= 0) || maxRefinements !== Infinity && !Number.isInteger(maxRefinements)
    || !Number.isFinite(tolerance) || tolerance < 0) throw new Error("Invalid execution box search options");
  return (box: EventExecutionAccountBox) => {
    const P = box.price, [rawClo, rawChi] = box.cash, [rawQlo, rawQhi] = box.quantity;
    if (!(P > 0) || ![P, rawClo, rawChi, rawQlo, rawQhi].every(Number.isFinite)
      || rawClo > rawChi || rawQlo > rawQhi) throw new Error("Invalid execution account box");
    const vertices = box.balanceVertices ?? [[rawClo, rawQlo], [rawClo, rawQhi], [rawChi, rawQlo], [rawChi, rawQhi]];
    if (!vertices.length || vertices.some(([C, Q]) => ![C, Q].every(Number.isFinite)
      || C < rawClo || C > rawChi || Q < rawQlo || Q > rawQhi)) throw new Error("Invalid execution balance vertices");
    const available = atoms.filter(a => a.path.openingAvailable);
    const maximum = available.length ? Math.max(...available.map(a => Math.floor((c.maxNotional + 1e-8)
      / (P * a.path.openRatio * step)))) + 2 : 0;
    const search = { maximumLots: maximum, evaluatedOrders: 0, intervals: 0, derivativeEvaluations: 0, refinements: 0 };
    const unknown = () => ({ upperLogEquity: Infinity, optimisticRequest: 0, complete: false, search });
    if (!Number.isSafeInteger(maximum) || Math.max(Math.abs(rawQlo / step), Math.abs(rawQhi / step)) + maximum > 1e12)
      return unknown();
    if (box.quantityLattice && [rawQlo, rawQhi].some(q => Math.abs(q / step - Math.round(q / step)) > 1e-7))
      throw new Error("Execution box lattice endpoints are not aligned");
    const qScale = Math.max(1, Math.abs(rawQlo), Math.abs(rawQhi), maximum * step);
    const qPad = 64 * Number.EPSILON * qScale;
    const cashPad = 64 * Number.EPSILON * Math.max(1, Math.abs(rawClo), Math.abs(rawChi), P * qScale, c.maxNotional);
    const Clo = rawClo - cashPad, Chi = rawChi + cashPad, Qlo = rawQlo - qPad, Qhi = rawQhi + qPad;
    if (![Clo, Chi, Qlo, Qhi].every(Number.isFinite)) return unknown();
    const halfway = Math.ceil(rawQlo / step - .5) <= Math.floor(rawQhi / step - .5);
    const rounding = qPad + (box.quantityLattice ? 0 : halfway ? step / 2
      : Math.max(Math.abs(rawQlo - Math.round(rawQlo / step) * step), Math.abs(rawQhi - Math.round(rawQhi / step) * step)));
    type Affine = { a: number; b: number };
    const rows = atoms.map(atom => {
      const p = atom.path, G = P * p.openRatio, close = P * p.closeRatio;
      const openings = vertices.map(([C, Q]) => ({ E: C + Q * G, N: Q * G }));
      const equityPad = cashPad + qPad * G;
      const plusPad = L * cashPad + (L - 1) * qPad * G, minusPad = L * cashPad + (L + 1) * qPad * G;
      const heldBorrowed = Qhi > 0 && Chi < 0;
      const heldRisk = Chi + Qhi * P * (Qhi < 0 ? p.maximumShortMaintenancePrice
        : (1 - c.maintenanceMargin) * (heldBorrowed ? p.minimumDiscountedLongLow : p.lowRatio));
      let heldWealth = (heldBorrowed ? Chi * p.longDebtGrowth : Chi)
        + Qhi * (close + (Qhi < 0 ? P * p.shortBorrowPriceIntegral : 0));
      let wealthSupports: Affine[][] | undefined;
      if (options.coupledWealth && vertices.length === 2) {
        const [[C0, Q0], [C1, Q1]] = vertices;
        const supports = (side: number, roundingCredit: number) => eventAffineSegmentSupports([1, p.longDebtGrowth]
          .flatMap(cashFactor => [close, close + P * p.shortBorrowPriceIntegral].map(assetPrice => ({
            a: (C0 + cashPad + roundingCredit * G) * cashFactor + (Q0 + qPad + roundingCredit) * assetPrice,
            b: -G * step * (1 + fee * side) * cashFactor + step * assetPrice,
            slope: (C1 - C0) * cashFactor + (Q1 - Q0) * assetPrice }))));
        heldWealth = Math.min(...supports(0, 0).map(line => line.a));
        wealthSupports = [-1, 1].map(side => supports(side, rounding));
      }
      return { ...atom, G, unit: G * step, close,
        Emin: Math.min(...openings.map(o => o.E)) - equityPad, Emax: Math.max(...openings.map(o => o.E)) + equityPad,
        beforeMax: openings.some(o => o.E <= equityPad) ? Infinity
          : Math.max(...openings.map(o => (Math.abs(o.N) + qPad * G) / (o.E - equityPad))),
        plusMin: Math.min(...openings.map(o => L * o.E - o.N)) - plusPad,
        plusMax: Math.max(...openings.map(o => L * o.E - o.N)) + plusPad,
        minusMin: Math.min(...openings.map(o => L * o.E + o.N)) - minusPad,
        minusMax: Math.max(...openings.map(o => L * o.E + o.N)) + minusPad,
        q0: Qhi + rounding, cash0: Chi + rounding * G,
        heldRisk, heldWealth, wealthSupports,
        held: heldRisk > 0 && heldWealth > 0 ? heldWealth : -Infinity,
        openingRuin: Chi + Qhi * G <= c.maintenanceMargin * Math.abs(Qhi) * G };
    });
    // Opening maintenance is increasing in cash and inventory. If even the
    // dominating balance fails, every account in the box fails before trading.
    if (rows.some(r => ![r.G, r.unit, r.close, r.Emin, r.Emax, r.plusMin, r.plusMax, r.minusMin, r.minusMax,
      r.q0, r.cash0, r.heldRisk, r.heldWealth,
      r.cash0 * r.path.longDebtGrowth, r.q0 * P * r.path.maximumShortMaintenancePrice,
      maximum * r.unit * r.path.longDebtGrowth,
      maximum * step * P * (r.path.closeRatio + r.path.shortBorrowPriceIntegral)].every(Number.isFinite))) return unknown();
    if (rows.some(r => r.wealthSupports?.some(supports => supports.some(line => !Number.isFinite(line.a) || !Number.isFinite(line.b)))))
      return unknown();
    if (rows.some(r => r.openingRuin)) return { upperLogEquity: -Infinity, optimisticRequest: 0, complete: true, search };
    type Row = typeof rows[number];
    const geometry = (row: Row, k: number) => {
      const p = row.path, cash1 = -row.unit * (1 + fee * Math.sign(k));
      const q = row.q0 + step * k, cash = row.cash0 + cash1 * k, borrowed = q > 0 && cash < 0;
      const riskPrice = P * (q < 0 ? p.maximumShortMaintenancePrice
        : (1 - c.maintenanceMargin) * (borrowed ? p.minimumDiscountedLongLow : p.lowRatio));
      const cashFactor = borrowed ? p.longDebtGrowth : 1;
      const assetPrice = row.close + (q < 0 ? P * p.shortBorrowPriceIntegral : 0);
      const supports = row.wealthSupports?.[k < 0 ? 0 : 1];
      const wealth = supports ? supports.reduce((best, line) => line.a + line.b * k < best.a + best.b * k ? line : best)
        : { a: row.cash0 * cashFactor + row.q0 * assetPrice, b: cash1 * cashFactor + step * assetPrice };
      return { wealth,
        risk: { a: row.cash0 + row.q0 * riskPrice, b: cash1 + step * riskPrice },
        equity: { a: row.cash0 + row.q0 * row.G, b: cash1 + step * row.G } };
    };
    const absoluteRange = (lo: number, hi: number) => [lo <= 0 && hi >= 0 ? 0 : Math.min(Math.abs(lo), Math.abs(hi)),
      Math.max(Math.abs(lo), Math.abs(hi))];
    const acceptance = (row: Row, k: number) => {
      const request = k * step, turnover = Math.abs(request) * row.G, cost = turnover * fee;
      if (!k || !row.path.openingAvailable || Math.abs(request) < c.minQuantity - 1e-12
        || turnover < c.minNotional - 1e-8 || turnover > c.maxNotional + 1e-8) return { possible: false, certain: false };
      const lo = row.Emin - cost, hi = row.Emax - cost;
      // Bound each affine cap constraint over the actual balance hull. Their
      // maxima need not share a vertex: allowing that is an upper relaxation.
      const plus = (1 + L * fee * Math.sign(k)) * row.unit * k;
      const minus = (L * fee * Math.sign(k) - 1) * row.unit * k;
      const certain = lo > 0 && row.plusMin >= plus && row.minusMin >= minus;
      const possible = hi > 0 && row.plusMax >= plus && row.minusMax >= minus;
      if (certain || turnover < c.maxNotional - row.unit - 1e-8 || !(hi > 0)) return { certain, possible };
      const previous = absoluteRange(Qlo, Qhi), after = absoluteRange(Qlo + request, Qhi + request);
      const beforeMin = previous[0] * row.G / row.Emax;
      const beforeMax = row.beforeMax;
      const afterMin = after[0] * row.G / hi, afterMax = lo > 0 ? after[1] * row.G / lo : Infinity;
      return { certain: certain || beforeMin > c.maxLeverage && afterMax < beforeMin - 1e-9 && lo > 0,
        possible: possible || beforeMax > c.maxLeverage && afterMin < beforeMax - 1e-9 };
    };
    const affineAt = (row: Row, k: number): Affine | null => {
      const accepted = acceptance(row, k), g = geometry(row, k);
      const wealth = accepted.possible && g.equity.a + g.equity.b * k > 0 && g.risk.a + g.risk.b * k > 0
        && g.wealth.a + g.wealth.b * k > 0 ? g.wealth.a + g.wealth.b * k : -Infinity;
      const held = accepted.certain ? -Infinity : row.held;
      return wealth === -Infinity && held === -Infinity ? null : wealth >= held ? g.wealth : { a: held, b: 0 };
    };
    const guards = new Set<number>([-maximum, 0, maximum]), visited = new Set<number>();
    const grouped = new Map<number | "unavailable", Row[]>();
    for (const row of rows) {
      const key = row.path.openingAvailable ? row.path.openRatio : "unavailable";
      if (!grouped.has(key)) grouped.set(key, []);
      grouped.get(key)!.push(row);
    }
    const groups = [...grouped.values()].map(members => ({ rows: members,
      guards: new Set<number>([-maximum, 0, maximum]),
      heldValue: members.some(r => r.held === -Infinity) ? -Infinity
        : members.reduce((s, r) => s + r.probability * Math.log(r.held), 0),
      heldAffines: members.map(r => ({ p: r.probability, affine: { a: r.held, b: 0 } })) }));
    const groupOf = new Map(groups.flatMap(group => group.rows.map(row => [row, group] as const)));
    const tradeAffines = (group: typeof groups[number], k: number) => {
      const out: Array<{ p: number; affine: Affine }> = [];
      for (const row of group.rows) {
        const g = geometry(row, k);
        if (!(g.equity.a + g.equity.b * k > 0 && g.risk.a + g.risk.b * k > 0 && g.wealth.a + g.wealth.b * k > 0)) return null;
        out.push({ p: row.probability, affine: g.wealth });
      }
      return out;
    };
    const scoreAffines = (affines: Array<{ p: number; affine: Affine }>, k: number) =>
      affines.reduce((s, r) => s + r.p * Math.log(r.affine.a + r.affine.b * k), 0);
    const selectGroup = (group: typeof groups[number], k: number) => {
      if (group.rows.length === 1) {
        const row = group.rows[0], affine = affineAt(row, k);
        return affine ? { value: row.probability * Math.log(affine.a + affine.b * k), affines: [{ p: row.probability, affine }] } : null;
      }
      const accepted = acceptance(group.rows[0], k), trade = accepted.possible ? tradeAffines(group, k) : null;
      const tradedValue = trade ? scoreAffines(trade, k) : -Infinity;
      const heldValue = accepted.certain ? -Infinity : group.heldValue;
      if (tradedValue === -Infinity && heldValue === -Infinity) return null;
      return tradedValue >= heldValue ? { value: tradedValue, affines: trade! } : { value: heldValue, affines: group.heldAffines };
    };
    const boundary = (set: Set<number>, k: number) => {
      if (!Number.isFinite(k) || k < -maximum - 2 || k > maximum + 2) return;
      for (const n of [Math.floor(k) - 1, Math.floor(k), Math.ceil(k), Math.ceil(k) + 1]) if (Math.abs(n) <= maximum) set.add(n);
    };
    for (const row of rows) {
      if (!row.path.openingAvailable) continue;
      const local = new Set<number>([-maximum, 0, maximum]);
      const minimum = Math.max((c.minQuantity - 1e-12) / step, (c.minNotional - 1e-8) / row.unit);
      boundary(local, minimum); boundary(local, -minimum);
      boundary(local, (c.maxNotional + 1e-8) / row.unit); boundary(local, -(c.maxNotional + 1e-8) / row.unit);
      // Recovery is exceptional only in this narrow maximum-order band.
      // Make every such integer a guard, so interval interiors need only the
      // ordinary linear cap checks. Very fine unsupported bands stay unbounded.
      const recoveryLo = Math.max(1, Math.ceil((c.maxNotional - row.unit - 1e-8) / row.unit));
      const recoveryHi = Math.floor((c.maxNotional + 1e-8) / row.unit);
      if (recoveryHi - recoveryLo > 16) return unknown();
      for (let k = recoveryLo; k <= recoveryHi; k++) { boundary(local, k); boundary(local, -k); }
      boundary(local, -row.q0 / step);
      // Positive minimum-of-affines wealth requires every support to be
      // positive. An active support's log tangent upper-bounds that minimum
      // even across support crossings, so ordered tangent search needs only
      // these solvency guards. The derivative search still needs crossings.
      for (const supports of row.wealthSupports ?? []) {
        for (const line of supports) boundary(local, -line.a / line.b);
        if (!options.ordered) for (let i = 0; i < supports.length; i++) for (let j = i + 1; j < supports.length; j++)
          boundary(local, (supports[j].a - supports[i].a) / (supports[i].b - supports[j].b));
      }
      for (const side of [-1, 1]) {
        boundary(local, row.cash0 / (row.unit * (1 + fee * side)));
        for (const E of [row.Emin, row.Emax]) boundary(local, E / (fee * row.unit * side));
        for (const a of [row.plusMin, row.plusMax]) boundary(local, a / (row.unit * (1 + L * fee * side)));
        for (const a of [row.minusMin, row.minusMax]) boundary(local, a / (row.unit * (L * fee * side - 1)));
      }
      const initial = [...local].sort((a, b) => a - b);
      for (let i = 1; i < initial.length; i++) {
        if (initial[i] - initial[i - 1] <= 1) continue;
        const g = geometry(row, Math.floor((initial[i] + initial[i - 1]) / 2));
        for (const a of [g.wealth, g.risk, g.equity]) boundary(local, -a.a / a.b);
        if (Number.isFinite(row.held)) boundary(local, (row.held - g.wealth.a) / g.wealth.b);
      }
      for (const k of local) { guards.add(k); groupOf.get(row)!.guards.add(k); }
    }
    if (options.ordered) {
      const ordered = [...groups].sort((a, b) => (a.rows[0].path.openingAvailable ? a.rows[0].G : Infinity)
        - (b.rows[0].path.openingAvailable ? b.rows[0].G : Infinity));
      const direction = Clo > 0 ? "prefix" : Chi < 0 ? "suffix" : "interval";
      const choicesAt = (k: number) => ordered.map(group => {
        const row = group.rows[0], request = k * step, turnover = Math.abs(request) * row.G;
        const eligible = !!k && row.path.openingAvailable && Math.abs(request) >= c.minQuantity - 1e-12
          && turnover >= c.minNotional - 1e-8 && turnover <= c.maxNotional + 1e-8;
        const beforeMax = row.beforeMax;
        // A possible maximum-order recovery may sit outside the ordinary cap
        // interval. Leave those groups independent, even if this admits extra
        // choices. All such request integers are explicit guards above.
        const recovery = eligible && beforeMax > c.maxLeverage && turnover >= c.maxNotional - row.unit - 1e-8;
        const accepted = acceptance(row, k), affines = accepted.possible ? tradeAffines(group, k) : null;
        return { accepted: affines ? scoreAffines(affines, k) : -Infinity,
          rejected: accepted.certain ? -Infinity : group.heldValue,
          free: !eligible || recovery, affines };
      });
      let best = -Infinity, quantity = 0;
      const consider = (k: number, choices = choicesAt(k)) => {
        if (visited.has(k)) return;
        visited.add(k); search.evaluatedOrders++;
        const value = maximizeEventAcceptanceSequence(choices, direction);
        if (value > best || value === best && Math.abs(k) < Math.abs(quantity)) { best = value; quantity = k; }
      };
      type Interval = { lo: number; hi: number; upper: number };
      const heap: Interval[] = [];
      const push = (row: Interval) => {
        heap.push(row); let i = heap.length - 1;
        while (i) { const parent = (i - 1) >> 1; if (heap[parent].upper >= row.upper) break;
          heap[i] = heap[parent]; i = parent; } heap[i] = row;
      };
      const pop = () => {
        const top = heap[0], last = heap.pop()!;
        if (heap.length) {
          let i = 0;
          while (2 * i + 1 < heap.length) {
            let next = 2 * i + 1; if (next + 1 < heap.length && heap[next + 1].upper > heap[next].upper) next++;
            if (heap[next].upper <= last.upper) break; heap[i] = heap[next]; i = next;
          }
          heap[i] = last;
        }
        return top;
      };
      const addInterval = (lo: number, hi: number) => {
        if (lo > hi) return;
        if (lo === hi) { consider(lo); return; }
        const middle = Math.floor((lo + hi) / 2), choices = choicesAt(middle);
        consider(middle, choices); search.intervals++;
        const tangents = choices.map(row => ({ ...row, slope: row.affines
          ? row.affines.reduce((s, a) => s + a.p * a.affine.b / (a.affine.a + a.affine.b * middle), 0) : 0 }));
        // Every accepted log-affine branch lies below its tangent. An allowed
        // sequence sums affine tangents; maximizing over sequences is convex
        // in k, so its interval maximum is at one of these two endpoints.
        const at = (k: number) => maximizeEventAcceptanceSequence(tangents.map(row => ({
          accepted: row.accepted === -Infinity ? -Infinity : row.accepted + row.slope * (k - middle),
          rejected: row.rejected, free: row.free })), direction);
        const upper = Math.max(at(lo), at(hi)) + 1e-9;
        if (Number.isNaN(upper)) push({ lo, hi, upper: Infinity });
        else if (upper > best) push({ lo, hi, upper });
      };
      const points = [...guards].sort((a, b) => a - b);
      for (const k of points) consider(k);
      for (let i = 1; i < points.length; i++) addInterval(points[i - 1] + 1, points[i] - 1);
      while (heap.length && heap[0].upper > best + tolerance && search.refinements < maxRefinements) {
        const interval = pop(), middle = Math.floor((interval.lo + interval.hi) / 2); search.refinements++;
        addInterval(interval.lo, middle); addInterval(middle + 1, interval.hi);
      }
      const upper = Math.max(best + 1e-9, heap[0]?.upper ?? -Infinity), complete = !heap.length || heap[0].upper <= best;
      return { upperLogEquity: upper, optimisticRequest: quantity * step,
        complete, certified: complete || upper - best <= tolerance, search,
        relaxedLowerLogEquity: best, relaxedGap: upper === best ? 0 : upper - best,
        ordering: direction, remainingIntervals: heap.length };
    }
    // Between financial/acceptance guards, a group's all-accepted log value is
    // concave and its all-rejected value is constant. Their nonnegative
    // difference is a single interval. Find both integer crossings before
    // using concavity in the final common-request search.
    for (const group of groups) {
      if (group.rows.length === 1 || !Number.isFinite(group.heldValue)) continue;
      const local = [...group.guards].sort((a, b) => a - b);
      for (let i = 1; i < local.length; i++) {
        const lo = local[i - 1] + 1, hi = local[i] - 1; if (lo > hi) continue;
        const middle = Math.floor((lo + hi) / 2), accepted = acceptance(group.rows[0], middle);
        if (!accepted.possible || accepted.certain) continue;
        const affines = tradeAffines(group, middle); if (!affines) continue;
        const difference = (k: number) => scoreAffines(affines, k) - group.heldValue;
        let left = lo, right = hi;
        while (left < right) {
          const k = Math.floor((left + right) / 2);
          const delta = affines.reduce((s, r) => s + r.p * Math.log1p(r.affine.b / (r.affine.a + r.affine.b * k)), 0);
          if (delta > 0) left = k + 1; else right = k;
        }
        const peak = left; if (difference(peak) < 0) continue;
        left = lo; right = peak;
        while (left < right) {
          const k = Math.floor((left + right) / 2);
          if (difference(k) >= 0) right = k; else left = k + 1;
        }
        boundary(guards, left);
        left = peak; right = hi;
        while (left < right) {
          const k = Math.ceil((left + right) / 2);
          if (difference(k) >= 0) left = k; else right = k - 1;
        }
        boundary(guards, left);
      }
    }
    let best = -Infinity, quantity = 0;
    const consider = (k: number) => {
      if (!Number.isSafeInteger(k) || Math.abs(k) > maximum || visited.has(k)) return;
      visited.add(k); search.evaluatedOrders++;
      let value = 0;
      for (const group of groups) {
        const selected = selectGroup(group, k); if (!selected) { value = -Infinity; break; }
        value += selected.value;
      }
      if (value > best || value === best && Math.abs(k) < Math.abs(quantity)) { best = value; quantity = k; }
    };
    const points = [...guards].sort((a, b) => a - b);
    for (const k of points) consider(k);
    for (let i = 1; i < points.length; i++) {
      let lo = points[i - 1] + 1, hi = points[i] - 1; if (lo > hi) continue;
      const middle = Math.floor((lo + hi) / 2), selected = groups.map(group => selectGroup(group, middle));
      if (selected.some(g => !g)) continue;
      const affine = selected.flatMap(g => g!.affines);
      for (const row of affine) {
        const a = row.affine!;
        if (a.b > 0) lo = Math.max(lo, Math.floor(-a.a / a.b) + 1);
        else if (a.b < 0) hi = Math.min(hi, Math.ceil(-a.a / a.b) - 1);
      }
      if (lo > hi) continue;
      search.intervals++; consider(lo); consider(hi);
      while (lo < hi) {
        const k = Math.floor((lo + hi) / 2); let delta = 0; search.derivativeEvaluations++;
        for (const row of affine) { const a = row.affine!; delta += row.p * Math.log1p(a.b / (a.a + a.b * k)); }
        if (delta > 0) lo = k + 1; else hi = k;
      }
      for (let k = lo - 1; k <= lo + 1; k++) consider(k);
    }
    return { upperLogEquity: best + 1e-9, optimisticRequest: quantity * step, complete: true, search };
  };
}
