import type { EventExecutionPath } from "./event-execution-path.js";
import { eventCapAllowsTrade, validateEventCosts, type EventAccount } from "./event-log-policy.js";

export interface EventExecutionAtom { probability: number; path: EventExecutionPath; }

/** Validate and own a frozen execution mixture before compiling an evaluator. */
export function cloneEventExecutionAtoms(input: readonly EventExecutionAtom[]) {
  if (!input.length) throw new Error("Invalid execution law");
  const atoms = input.filter(a => a.probability > 0).map(a => ({ probability: a.probability, path: structuredClone(a.path) }));
  const costs = input[0].path.costs;
  validateEventCosts(costs);
  if (input.some(a => !Number.isFinite(a.probability) || a.probability < 0)
    || Math.abs(input.reduce((s, a) => s + a.probability, 0) - 1) > 1e-8
    || atoms.some(({ path: p }) => p.version !== 1 || !Number.isInteger(p.seconds) || p.seconds < 1
      || ![p.openRatio, p.closeRatio, p.lowRatio, p.highRatio, p.longDebtGrowth, p.minimumDiscountedLongLow,
        p.shortBorrowPriceIntegral, p.maximumShortMaintenancePrice].every(Number.isFinite)
      || p.lowRatio <= 0 || p.lowRatio > Math.min(p.openRatio, p.closeRatio)
      || p.highRatio < Math.max(p.openRatio, p.closeRatio) || p.longDebtGrowth < 1
      || p.minimumDiscountedLongLow <= 0 || p.minimumDiscountedLongLow > p.lowRatio
      || p.shortBorrowPriceIntegral < 0 || p.maximumShortMaintenancePrice <= 0
      || typeof p.openingAvailable !== "boolean" || typeof p.terminalAvailable !== "boolean"
      || (Object.keys(costs) as Array<keyof typeof costs>).some(key => p.costs[key] !== costs[key])))
    throw new Error("Invalid or inconsistent execution mixture");
  return atoms;
}

/** Global one-event BASE-QUANTITY REQUEST search. Each outcome may accept or
 * reject the request at its own next open. Split at every acceptance, fee,
 * inventory, funding and terminal-dust boundary. On each remaining interval,
 * survival is an intersection of linear inequalities and expected log wealth
 * is concave. No-action remains admissible above the entry leverage cap, as in
 * the replay; it still bears maintenance/liquidation risk. */
export function prepareEventExecutionOneStep(input: readonly EventExecutionAtom[], terminal: "marked" | "market" = "marked",
  options: { captureRegions?: boolean; objective?: "log" | "mean";
    alternativeProbabilities?: readonly (readonly number[])[] } = {}) {
  if (!["marked", "market"].includes(terminal)) throw new Error("Invalid execution terminal");
  if (options.objective && !["log", "mean"].includes(options.objective)) throw new Error("Invalid execution objective");
  const atoms = cloneEventExecutionAtoms(input), costs = input[0].path.costs;
  const c = { ...costs }, step = c.quantityStep, fee = (c.feeBps + c.slippageBps) / 10000;
  const objective = options.objective ?? "log";
  const retained = input.map((atom, index) => atom.probability > 0 ? index : -1).filter(index => index >= 0);
  const probabilityModels = [atoms.map(atom => atom.probability)];
  for (const probabilities of options.alternativeProbabilities ?? []) {
    if (probabilities.length !== input.length || probabilities.some(probability => !Number.isFinite(probability) || probability < 0)
      || Math.abs(probabilities.reduce((sum, probability) => sum + probability, 0) - 1) > 1e-8
      || input.some((atom, index) => atom.probability <= 0 && probabilities[index] > 0))
      throw new Error("Invalid alternative execution probabilities");
    probabilityModels.push(retained.map(index => probabilities[index]));
  }
  if (objective === "mean" && probabilityModels.length > 1)
    throw new Error("Robust mean execution objective is not supported");
  return (account: EventAccount, solveOptions: { minimumLiquidatableEquity?: number } = {}) => {
    const { equity: E, price: P, exposure: x } = account;
    if (!(E > 0 && P > 0) || ![E, P, x].every(Number.isFinite)) throw new Error("Invalid execution account");
    const minimumLiquidatableEquity = solveOptions.minimumLiquidatableEquity;
    if (minimumLiquidatableEquity !== undefined
      && (!Number.isFinite(minimumLiquidatableEquity) || minimumLiquidatableEquity < 0))
      throw new Error("Invalid minimum liquidatable equity");
    const Q = x * E / P, roundedQ = Math.round(Q / step) * step;
    const rows = atoms.map(atom => {
      const p = atom.path, open = P * p.openRatio, close = P * p.closeRatio;
      const equity = E + Q * P * (p.openRatio - 1);
      return { ...atom, open, close, equity, before: Q * open / equity,
        openingRuin: equity <= c.maintenanceMargin * Math.abs(Q) * open };
    });
    const available = atoms.filter(a => a.path.openingAvailable);
    // Larger requests fail the maximum-notional check in every outcome and
    // are equivalent to zero. Include numerical neighbors of the limiting lot.
    const maximum = available.length ? Math.max(...available.map(a => Math.floor((c.maxNotional + 1e-8)
      / (P * a.path.openRatio * step)))) + 2 : 0;
    if (!Number.isSafeInteger(maximum) || !Number.isSafeInteger(Math.round(Q / step))
      || !Number.isSafeInteger(Math.round(Q / step) + maximum) || !Number.isSafeInteger(Math.round(Q / step) - maximum))
      throw new Error("Execution request lattice exceeds integer precision");
    const search = { evaluatedOrders: 0, regions: 0, searchedIntervals: 0, derivativeEvaluations: 0, maximumLots: maximum };
    const visited = new Set<number>(), scores = new Map<number, number>(), guards = new Set<number>([-maximum, 0, maximum]);
    const feasiblePoints = new Set<number>(), feasibleIntervals: Array<readonly [number, number]> = [];
    let best = 0, value = -Infinity;
    const consider = (k: number) => {
      if (!Number.isSafeInteger(k) || Math.abs(k) > maximum) return -Infinity;
      if (visited.has(k)) return scores.get(k) ?? -Infinity;
      visited.add(k); search.evaluatedOrders++;
      const request = k * step;
      if (Math.abs(request / step - Math.round(request / step)) > 1e-7) throw new Error("Execution request lost lot precision");
      const modelScores = probabilityModels.map(() => 0);
      let score = -Infinity;
      // The same scalar transition as evaluateEventExecutionPath, with the
      // account/opening constants prepared once and no diagnostic objects per
      // candidate/outcome. Final scores are checked against that reference.
      for (let rowIndex = 0; rowIndex < rows.length; rowIndex++) {
        const row = rows[rowIndex];
        const p = row.path, G = row.open;
        if (row.openingRuin) break;
        let quantity = Q, equity = row.equity;
        if (request && p.openingAvailable) {
          const turnover = Math.abs(request) * G, cost = turnover * fee;
          if (Math.abs(request) >= c.minQuantity - 1e-12 && turnover >= c.minNotional - 1e-8 && turnover <= c.maxNotional + 1e-8
            && equity > cost && eventCapAllowsTrade(row.before, (Q + request) * G / (equity - cost), turnover, G, c)) {
            equity -= cost; quantity = Math.round((Q + request) / step) * step;
          }
        }
        const cash = equity - quantity * G, borrowedLong = quantity > 0 && cash < 0;
        const riskCash = quantity >= 0 ? cash + quantity * P * (1 - c.maintenanceMargin)
          * (borrowedLong ? p.minimumDiscountedLongLow : p.lowRatio) : cash + quantity * P * p.maximumShortMaintenancePrice;
        if (!(riskCash > 0)) break;
        const borrowing = borrowedLong ? -cash * (p.longDebtGrowth - 1) : quantity < 0 ? -quantity * P * p.shortBorrowPriceIntegral : 0;
        equity = cash + quantity * row.close - borrowing;
        if (!(equity > 0)) break;
        const notional = Math.abs(quantity) * row.close;
        let remainingQuantity = quantity;
        if (terminal === "market" && p.terminalAvailable && quantity && Math.abs(quantity) >= c.minQuantity - 1e-12 && notional >= c.minNotional - 1e-8) {
          equity -= notional * fee; remainingQuantity = 0;
        }
        if (!(equity > 0)) break;
        // Reserve one executable close at the terminal mark. This makes the
        // floor comparable across marked and already-settled accounts without
        // treating unrealized PnL as bankable equity.
        const liquidatableEquity = equity - Math.abs(remainingQuantity) * row.close * fee;
        if (minimumLiquidatableEquity !== undefined
          && liquidatableEquity < minimumLiquidatableEquity - 1e-8) break;
        const payoff = objective === "log" ? Math.log(equity / E) : equity / E;
        for (let model = 0; model < probabilityModels.length; model++)
          modelScores[model] += probabilityModels[model][rowIndex] * payoff;
        if (rowIndex === rows.length - 1) score = Math.min(...modelScores);
      }
      scores.set(k, score);
      if (options.captureRegions && Number.isFinite(score)) feasiblePoints.add(k);
      if (score > value || score === value && Math.abs(k) < Math.abs(best)) { best = k; value = score; }
      return score;
    };
    const finish = () => {
      const result = { quantity: best * step, value, feasible: Number.isFinite(value), terminal, complete: true, search,
        ...(objective === "mean" ? { objective: "mean-terminal-equity-ratio" as const } : {}) };
      if (!options.captureRegions) return result;
      // Every non-singleton interval retains a fixed acceptance and funding
      // branch for every outcome. Do not merge adjacent intervals across a
      // guard. Outside +/-maximum all requests are equivalent to zero.
      const candidates = [...feasibleIntervals, ...[...feasiblePoints].map(k => [k, k] as const)]
        .sort((a, b) => a[0] - b[0] || b[1] - a[1]);
      const requestRegions: Array<readonly [number, number]> = [];
      for (const interval of candidates) {
        const last = requestRegions.at(-1);
        if (last && interval[1] <= last[1]) continue;
        requestRegions.push(last && interval[0] <= last[1] ? [last[1] + 1, interval[1]] : interval);
      }
      return { ...result, requestRegions };
    };
    consider(0);
    if (!maximum) return finish();
    // Ruin before the requested trade cannot be repaired by that trade.
    if (rows.some(row => row.openingRuin)) return finish();
    const boundary = (k: number) => {
      if (!Number.isFinite(k) || k < -maximum - 2 || k > maximum + 2) return;
      for (const n of [Math.floor(k) - 1, Math.floor(k), Math.ceil(k), Math.ceil(k) + 1])
        if (Math.abs(n) <= maximum) guards.add(n);
    };
    boundary(-roundedQ / step);
    for (const { path: p } of available) {
      const G = P * p.openRatio, unit = G * step, Eg = E + Q * (G - P), N = Q * G, before = N / Eg;
      const minimum = Math.max((c.minQuantity - 1e-12) / step, (c.minNotional - 1e-8) / unit);
      boundary(minimum); boundary(-minimum);
      boundary((c.maxNotional + 1e-8) / unit); boundary(-(c.maxNotional + 1e-8) / unit);
      for (const side of [-1, 1]) {
        boundary(Eg / (fee * unit * side));
        boundary((Eg - roundedQ * G) / (unit * (1 + fee * side)));
        for (const cap of [c.maxLeverage + 1e-8, -c.maxLeverage - 1e-8,
          Math.abs(before) - 1e-9, -Math.abs(before) + 1e-9])
          boundary((cap * Eg - N) / (unit * (1 + cap * fee * side)));
        boundary(side * (c.maxNotional - unit - 1e-8) / unit);
      }
      if (terminal === "market" && p.terminalAvailable) {
        const minimumPosition = Math.max(c.minQuantity - 1e-12, (c.minNotional - 1e-8) / (P * p.closeRatio));
        boundary((minimumPosition - roundedQ) / step); boundary((-minimumPosition - roundedQ) / step);
      }
    }
    const points = [...guards].sort((a, b) => a - b);
    for (const k of points) consider(k);
    for (let index = 1; index < points.length; index++) {
      let low = points[index - 1] + 1, high = points[index] - 1;
      if (low > high) continue;
      search.regions++;
      const middle = Math.floor((low + high) / 2), side = Math.sign(middle);
      const affine: Array<{ a: number; b: number; probability: number }> = [];
      const positive = (a: number, b: number) => {
        if (!b) { if (!(a > 0)) high = low - 1; return; }
        const crossing = -a / b;
        if (crossing >= low - 1 && crossing <= high + 1)
          for (const k of [Math.floor(crossing) - 1, Math.floor(crossing), Math.ceil(crossing), Math.ceil(crossing) + 1]) consider(k);
        if (b > 0) low = Math.max(low, Math.floor(crossing) + 1);
        else high = Math.min(high, Math.ceil(crossing) - 1);
      };
      for (const atom of atoms) {
        const p = atom.path, G = P * p.openRatio, unit = G * step, Eg = E + Q * (G - P);
        const turnover = Math.abs(middle) * unit, cost = turnover * fee, request = middle * step;
        const accepted = p.openingAvailable && Math.abs(request) >= c.minQuantity - 1e-12
          && turnover >= c.minNotional - 1e-8 && turnover <= c.maxNotional + 1e-8 && Eg > cost
          && eventCapAllowsTrade(Q * G / Eg, (Q + request) * G / (Eg - cost), turnover, G, c);
        const q0 = accepted ? roundedQ : Q, q1 = accepted ? step : 0;
        const cash0 = Eg - q0 * G, cash1 = accepted ? -unit * (1 + side * fee) : 0;
        const q = q0 + q1 * middle, cash = cash0 + cash1 * middle;
        const borrowedLong = q > 0 && cash < 0;
        const riskPrice = q >= 0 ? P * (1 - c.maintenanceMargin)
          * (borrowedLong ? p.minimumDiscountedLongLow : p.lowRatio) : P * p.maximumShortMaintenancePrice;
        positive(cash0 + q0 * riskPrice, cash1 + q1 * riskPrice);
        if (low > high) break;
        const cashFactor = borrowedLong ? p.longDebtGrowth : 1;
        const assetPrice = P * (p.closeRatio + (q < 0 ? p.shortBorrowPriceIntegral : 0));
        let a = cash0 * cashFactor + q0 * assetPrice, b = cash1 * cashFactor + q1 * assetPrice;
        positive(a, b);
        const terminalCloses = terminal === "market" && p.terminalAvailable && Math.abs(q) >= c.minQuantity - 1e-12
          && Math.abs(q) * P * p.closeRatio >= c.minNotional - 1e-8;
        if (terminalCloses) {
          const closeCost = Math.sign(q) * P * p.closeRatio * fee;
          a -= q0 * closeCost; b -= q1 * closeCost; positive(a, b);
        }
        if (minimumLiquidatableEquity !== undefined) {
          const reservedClose = terminalCloses ? 0 : Math.sign(q) * P * p.closeRatio * fee;
          positive(a - q0 * reservedClose - minimumLiquidatableEquity + 1e-8, b - q1 * reservedClose);
        }
        if (low > high) break;
        affine.push({ a, b, probability: atom.probability });
      }
      if (low > high) continue;
      if (options.captureRegions) feasibleIntervals.push([low, high]);
      search.searchedIntervals++; consider(low); consider(high);
      if (objective === "mean") continue;
      if (probabilityModels.length > 1) {
        // The lower envelope of concave expected-log objectives is concave.
        // Compare adjacent lattice values to find its (possibly kinked) peak.
        while (low < high) {
          const k = Math.floor((low + high) / 2);
          search.derivativeEvaluations++;
          if (consider(k) <= consider(k + 1)) low = k + 1;
          else high = k;
        }
        for (let k = low - 1; k <= low + 1; k++) consider(k);
        continue;
      }
      while (low < high) {
        const k = Math.floor((low + high) / 2); let delta = 0;
        search.derivativeEvaluations++;
        for (const row of affine) delta += row.probability * Math.log1p(row.b / (row.a + row.b * k));
        if (delta > 0) low = k + 1; else high = k;
      }
      for (let k = low - 1; k <= low + 1; k++) consider(k);
    }
    return finish();
  };
}
