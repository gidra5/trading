import { validateEventDistribution, type EventDistribution } from "./event-distribution.js";
import { validateEventCosts, type EventAccount, type EventCosts } from "./event-log-policy.js";

type Coefficient = { d: number; upper: number; exposure: number; optimizationGap: number };
type Line = { slope: number; intercept: number };

/** Convex hull of ordinary executable lots on one root side. Undefined keeps
 * the wider continuous relaxation when endpoint arithmetic is uncertain. */
function rootLotInterval(account: EventAccount, costs: EventCosts, side: -1 | 1): [number, number] | undefined {
  const E = account.equity, P = account.price, N = E * account.exposure;
  const step = costs.quantityStep, unit = P * step, f = (costs.feeBps + costs.slippageBps) / 10000, L = costs.maxLeverage + 1e-8;
  const maximum = Math.ceil((costs.maxNotional + 1e-8) / unit) + 2;
  if (!Number.isSafeInteger(maximum) || maximum > 1e12) return undefined;
  const minimum = Math.max(1, Math.ceil(Math.max((costs.minNotional - 1e-8) / unit, (costs.minQuantity - 1e-12) / step)) - 2);
  let low = Math.max(side > 0 ? minimum : -maximum, Math.ceil((-L * E - N) / (unit * (1 - L * f * side))) - 2);
  let high = Math.min(side > 0 ? maximum : -minimum, Math.floor((L * E - N) / (unit * (1 + L * f * side))) + 2);
  const valid = (k: number) => {
    const q = k * step, notional = Math.abs(q) * P, equity = E - f * notional;
    return Math.abs(q) >= costs.minQuantity - 1e-12 && notional >= costs.minNotional - 1e-8
      && notional <= costs.maxNotional + 1e-8 && equity > 0 && Math.abs((N + q * P) / equity) <= L;
  };
  let adjusted = 0;
  while (low <= high && !valid(low) && adjusted++ < 8) low++;
  while (low <= high && !valid(high) && adjusted++ < 16) high--;
  if (low > high) return [Infinity, -Infinity];
  if (!valid(low) || !valid(high)) return undefined;
  return [low * step * P, high * step * P];
}

/** Recursive continuous upper relaxation. Shadow execution prices undercharge
 * actual transaction fees. Minimum/maximum orders, lots and maintenance are
 * relaxed; the ordinary post-trade leverage cap and borrowing remain. Each
 * stored log(C+(1+d)N)+K_h(d) bounds the finite-horizon value globally for
 * ordinary cap-feasible trades. Query-time reachability must exclude the
 * exceptional maximum-order recovery trades before applying this bound. */
export function prepareEventMultiStepUpper(input: EventDistribution, inputCosts: EventCosts, terminal: "marked" | "friction",
  options: { depth: number; shadowPoints?: number; tolerance?: number; method?: "uniform" | "marginal" }) {
  validateEventDistribution(input); validateEventCosts(inputCosts);
  const model = structuredClone(input), costs = { ...inputCosts }, depth = options.depth;
  const points = options.shadowPoints ?? 9, tolerance = options.tolerance ?? 1e-10, pad = 1e-10;
  const method = options.method ?? "uniform";
  const f = (costs.feeBps + costs.slippageBps) / 10000, L = costs.maxLeverage + 1e-8;
  if (!Number.isInteger(depth) || depth < 1 || !Number.isInteger(points) || points < 3 || points > 513
    || !Number.isFinite(tolerance) || tolerance < pad || f * L >= .99 || !["marked", "friction"].includes(terminal)
    || !["uniform", "marginal"].includes(method))
    throw new Error("Unsupported multi-event upper relaxation");
  const kernels = model.kernels.map(k => k.filter(a => a.probability > 0));
  if (kernels.some(k => Math.abs(k.reduce((s, a) => s + a.probability, 0) - 1) > 1e-12))
    throw new Error("Upper relaxation requires probability normalization within numerical padding");
  const grid = f ? Array.from({ length: points }, (_, i) => -f + 2 * f * i / (points - 1)) : [0];
  const terminalGrid = terminal === "marked" ? [0] : [...new Set([-f, f])];
  const tables: Coefficient[][][] = [kernels.map(() => terminalGrid.map(d => ({ d, upper: 0, exposure: 0, optimizationGap: 0 })))];
  const lineValue = (l: Line, t: number) => l.intercept + l.slope * t;
  const pairMaximum = (a: Line, b: Line, lo: number, hi: number) => {
    const cross = a.slope === b.slope ? lo : (b.intercept - a.intercept) / (a.slope - b.slope);
    return Math.max(...[lo, hi, Math.min(hi, Math.max(lo, cross))].map(t => Math.min(lineValue(a, t), lineValue(b, t))));
  };
  const coefficient = (h: number, leaf: number, d: number, restriction?: { lower?: number; upper?: number }): Coefficient => {
    const bottom = Math.max(-L / (1 - d * L), restriction?.lower ?? -Infinity);
    const top = Math.min(L / (1 + d * L), restriction?.upper ?? Infinity);
    // A one-lot hull is a point, not an empty action set. Evaluate that
    // portfolio directly because interval bisection has no width to search.
    if (bottom === top) {
      const N = bottom, C = 1 - (1 + d) * N;
      let value = 0;
      for (const atom of kernels[leaf]) {
        const borrowing = (costs.longBorrowBpsPerDay * Math.max(0, -C)
          + costs.shortBorrowBpsPerDay * Math.max(0, -N)) / 10000 * atom.duration / 1440;
        const values = tables[h - 1][atom.next].map(k => {
          const budget = C - borrowing + (1 + k.d) * N * (1 + atom.return);
          return budget > 0 ? Math.log(budget) + k.upper : Infinity;
        });
        value += atom.probability * Math.min(...values);
      }
      return { d, upper: value + pad, exposure: bottom / (1 - d * bottom), optimizationGap: 0 };
    }
    const cuts = [...new Set([bottom, top, 0, 1 / (1 + d)].filter(t => t >= bottom && t <= top))].sort((a, b) => a - b);
    let globalUpper = -Infinity, bestSample = -Infinity, bestExposure = 0;
    for (let piece = 1; piece < cuts.length; piece++) {
      let lo = cuts[piece - 1], hi = cuts[piece];
      const middle = (lo + hi) / 2, borrowed = middle > 1 / (1 + d);
      const rows = kernels[leaf].map(atom => {
        const beta = (middle < 0 ? costs.shortBorrowBpsPerDay : borrowed ? costs.longBorrowBpsPerDay : 0)
          / 10000 * atom.duration / 1440;
        const a = 1 + (borrowed ? beta : 0), cashSlope = -(1 + d) * a + (middle < 0 ? beta : 0);
        const shadows = tables[h - 1][atom.next].map(prev => ({ b: cashSlope + (1 + prev.d) * (1 + atom.return), k: prev.upper }));
        for (const s of shadows) {
          if (s.b > 0) lo = Math.max(lo, -a / s.b);
          else if (s.b < 0) hi = Math.min(hi, -a / s.b);
        }
        return { p: atom.probability, a, shadows };
      });
      if (!(lo < hi)) continue;
      const score = (t: number) => {
        let value = 0, derivative = 0;
        for (const a of rows) {
          let best = Infinity, slope = 0;
          for (const s of a.shadows) {
            const wealth = a.a + s.b * t;
            if (!(wealth > 0)) return { value: -Infinity, derivative: s.b > 0 ? Infinity : -Infinity };
            const v = Math.log(wealth) + s.k;
            if (v < best) { best = v; slope = s.b / wealth; }
          }
          value += a.p * best; derivative += a.p * slope;
        }
        return { value, derivative };
      };
      let left = lo, right = hi, upper = Infinity, lower = -Infinity, leftLine: Line | undefined, rightLine: Line | undefined;
      for (let i = 0; i < 64; i++) {
        const t = left + (right - left) / 2; if (t === left || t === right) break;
        const s = score(t);
        if (!Number.isFinite(s.value)) { if (s.derivative > 0) left = t; else right = t; continue; }
        lower = Math.max(lower, s.value);
        if (s.value > bestSample) { bestSample = s.value; bestExposure = t / (1 - d * t); }
        const line = { slope: s.derivative, intercept: s.value - s.derivative * t };
        upper = Math.min(upper, Math.max(lineValue(line, lo), lineValue(line, hi)));
        if (s.derivative >= 0) { left = t; leftLine = line; } else { right = t; rightLine = line; }
        if (leftLine && rightLine) upper = Math.min(upper, pairMaximum(leftLine, rightLine, lo, hi));
        if (upper - lower <= tolerance) break;
      }
      globalUpper = Math.max(globalUpper, upper);
    }
    return { d, upper: globalUpper + pad, exposure: bestExposure,
      optimizationGap: globalUpper === bestSample ? 0 : Math.max(0, globalUpper - bestSample) };
  };
  const marginal = (h: number, leaf: number, x: number): Coefficient | null => {
    const C = 1 - x, N = x;
    let value = 0, dCash = 0, dNotional = 0;
    for (const atom of kernels[leaf]) {
      const long = costs.longBorrowBpsPerDay / 10000 * atom.duration / 1440;
      const short = costs.shortBorrowBpsPerDay / 10000 * atom.duration / 1440;
      const cash = C - long * Math.max(0, -C) - short * Math.max(0, -N), nextN = N * (1 + atom.return);
      let best = Infinity, cashDerivative = 0, notionalDerivative = 0;
      for (const k of tables[h - 1][atom.next]) {
        const budget = cash + (1 + k.d) * nextN;
        if (!(budget > 0)) return null;
        const v = Math.log(budget) + k.upper;
        if (v < best) {
          best = v; cashDerivative = (1 + (C < 0 ? long : 0)) / budget;
          notionalDerivative = ((N < 0 ? short : 0) + (1 + k.d) * (1 + atom.return)) / budget;
        }
      }
      value += atom.probability * best; dCash += atom.probability * cashDerivative; dNotional += atom.probability * notionalDerivative;
    }
    const d = dNotional / dCash - 1, budget = C + (1 + d) * N;
    if (!(dCash > 0 && budget > 0) || d <= -f || d >= f || !Number.isFinite(value)) return null;
    // Concavity: the holding portfolio maximizes the previous envelope under
    // its marginal shadow price. Homogeneity then gives K directly, without
    // another 64-step optimization. This targets the narrow useful price band.
    return { d, upper: value - Math.log(budget) + pad, exposure: x, optimizationGap: 0 };
  };
  for (let h = 1; h <= depth; h++) tables.push(kernels.map((_, leaf) => {
    if (method === "uniform") return grid.map(d => coefficient(h, leaf, d));
    const values = [...new Set([-f, f])].map(d => coefficient(h, leaf, d));
    const exposures = [...new Set([...Array.from({ length: points }, (_, i) => -L + 2 * L * i / (points - 1)), 0, 1].filter(x => x >= -L && x <= L))];
    for (const x of exposures) { const k = marginal(h, leaf, x); if (k) values.push(k); }
    return values.sort((a, b) => a.d - b.d);
  }));
  const maxMove = Math.max(...kernels.flatMap(k => k.map(a => Math.abs(a.return))));
  const maxPriceFactor = Math.max(...kernels.flatMap(k => k.map(a => 1 + a.return)));
  const maxBorrowFraction = Math.max(...kernels.flatMap(k => k.map(a =>
    Math.max(Math.max(0, L - 1) * costs.longBorrowBpsPerDay, L * costs.shortBorrowBpsPerDay) / 10000 * a.duration / 1440)));
  const recoveryExcluded = (account: EventAccount, h: number) => {
    let equityBound = account.equity, equityLower = account.equity, priceBound = account.price;
    let notionalBound = Math.abs(account.exposure) * account.equity;
    const maximumOrder = costs.maxNotional + 1e-8;
    for (let i = 0; i < h; i++) {
      const minimumClip = Math.max(0, costs.maxNotional - costs.quantityStep * priceBound - 1e-8);
      if (notionalBound > L * equityLower && minimumClip < 2 * notionalBound) {
        // A recovery order must oppose an over-cap holding. Its pre-order
        // absolute notional lies in [L*equityLower, notionalBound]. Bound the
        // residual after every admissible maximum-order clip. If every such
        // clip restores the ordinary cap, it creates no exceptional action.
        // The older 2*N test covers clips too large to reduce exposure at all.
        const residual = Math.max(Math.abs(L * equityLower - maximumOrder), Math.abs(notionalBound - minimumClip));
        const postEquityLower = equityLower - f * maximumOrder;
        if (!(postEquityLower > 0) || residual > L * postEquityLower) return false;
      }
      // By induction, all admissible post-trade accounts are now ordinary
      // cap-feasible. Nonnegative costs give uniform next-state wealth bounds.
      notionalBound = L * equityBound * maxPriceFactor;
      equityLower = Math.max(0, equityLower - f * maximumOrder) * Math.max(0, 1 - L * maxMove - maxBorrowFraction);
      if (equityLower <= 0 && i + 1 < h) return false;
      equityBound *= 1 + L * maxMove; priceBound *= maxPriceFactor;
    }
    return true;
  };
  const holdingUpper = (leaf: number, account: EventAccount, h: number) => {
    const N = account.exposure * account.equity, C = account.equity - N;
    let value = 0;
    for (const atom of kernels[leaf]) {
      const long = costs.longBorrowBpsPerDay / 10000 * atom.duration / 1440;
      const short = costs.shortBorrowBpsPerDay / 10000 * atom.duration / 1440;
      const cash = C - long * Math.max(0, -C) - short * Math.max(0, -N), nextN = N * (1 + atom.return);
      const candidates = tables[h - 1][atom.next].map(k => ({ k, budget: cash + (1 + k.d) * nextN }))
        .filter(c => c.budget > 0);
      if (!candidates.length) return Infinity;
      value += atom.probability * Math.min(...candidates.map(c => Math.log(c.budget) + c.k.upper));
    }
    return value - Math.log(account.equity) + pad;
  };
  return {
    coefficients: structuredClone(tables), method,
    query(leaf: number, account: EventAccount, h = depth, side?: -1 | 1,
      orderLimits: "exchange-relaxation" | undefined = undefined) {
      if (!Number.isInteger(h) || h < 1 || h > depth || !kernels[leaf] || !(account.equity > 0 && account.price > 0)
        || !Object.values(account).every(Number.isFinite) || side !== undefined && side !== -1 && side !== 1
        || orderLimits !== undefined && orderLimits !== "exchange-relaxation")
        throw new Error("Invalid multi-event upper query");
      const excluded = recoveryExcluded(account, h), N = account.exposure * account.equity, C = account.equity - N;
      const rootBudget = side === undefined ? 0 : C + (1 + side * f) * N;
      // With the actual directional fee, q>=0 means t>=N/A and q<=0 means
      // t<=N/A. This bounds a whole side of the root lattice, including hold.
      let restriction: { lower?: number; upper?: number } = side === 1
        ? { lower: N / rootBudget } : { upper: N / rootBudget };
      if (side !== undefined && orderLimits && rootBudget > 0) {
        // Relax lots to a continuous interval, retaining the exchange's
        // notional/quantity tolerances as conservative outer limits.
        // Holding is bounded separately because the minimum disconnects it
        // from every nonzero order.
        const minimumNotional = Math.max(0, costs.minNotional - 1e-8,
          (costs.minQuantity - 1e-12) * account.price);
        const maximumNotional = costs.maxNotional + 1e-8;
        restriction = side === 1
          ? { lower: (N + minimumNotional) / rootBudget, upper: (N + maximumNotional) / rootBudget }
          : { lower: (N - maximumNotional) / rootBudget, upper: (N - minimumNotional) / rootBudget };
        const lots = rootLotInterval(account, costs, side);
        if (lots) restriction = { lower: (N + lots[0]) / rootBudget, upper: (N + lots[1]) / rootBudget };
      }
      const row = side === undefined || !excluded || !(rootBudget > 0) ? tables[h][leaf]
        : [coefficient(h, leaf, side * f, restriction)];
      const candidates = row.map(k => ({ k, budget: C + (1 + k.d) * N }));
      let upperValue = excluded && candidates.every(c => c.budget > 0)
        ? Math.min(...candidates.map(c => Math.log(c.budget / account.equity) + c.k.upper)) : Infinity;
      const nonzeroUpperValue = side !== undefined && orderLimits ? upperValue : undefined;
      if (side !== undefined && orderLimits && excluded && Math.abs(account.exposure) <= costs.maxLeverage + 1e-9)
        upperValue = Math.max(upperValue, holdingUpper(leaf, account, h));
      return { upperValue, nonzeroUpperValue, recoveryExcluded: excluded, seedExposures: [row[0].exposure, row.at(-1)!.exposure] };
    },
  };
}
