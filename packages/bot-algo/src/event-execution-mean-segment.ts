import { cloneEventExecutionAtoms, type EventExecutionAtom } from "./event-execution-one-step.js";
import { maximizeEventAcceptanceSequence } from "./event-execution-acceptance.js";

export interface EventExecutionBalanceSegment {
  price: number;
  vertices: readonly [readonly [number, number], readonly [number, number]];
}

/**
 * Upper-bound absolute expected marked terminal equity over a balance segment.
 * Child request, opening-size rules and ordered acceptance remain linked.
 * Acceptance that changes somewhere along the segment may choose its favorable
 * branch, while borrowing and maintenance are relaxed upward. With these
 * segment-wide choices fixed, every policy is affine in the balance coordinate;
 * their maximum is convex and therefore reaches its maximum at an endpoint.
 */
export function prepareEventExecutionMeanSegmentUpper(input: readonly EventExecutionAtom[]) {
  const atoms = cloneEventExecutionAtoms(input), c = atoms[0].path.costs;
  if (Math.abs(atoms.reduce((sum, atom) => sum + atom.probability, 0) - 1) > 1e-12)
    throw new Error("Execution mean segment requires normalized probability within 1e-12");
  const step = c.quantityStep, fee = (c.feeBps + c.slippageBps) / 10000, L = c.maxLeverage + 1e-8;
  return (segment: EventExecutionBalanceSegment) => {
    const P = segment.price, vertices = segment.vertices;
    if (!(P > 0) || vertices.some(vertex => vertex.length !== 2 || vertex.some(value => !Number.isFinite(value))))
      throw new Error("Invalid execution balance segment");
    const rawClo = Math.min(vertices[0][0], vertices[1][0]), rawChi = Math.max(vertices[0][0], vertices[1][0]);
    const rawQlo = Math.min(vertices[0][1], vertices[1][1]), rawQhi = Math.max(vertices[0][1], vertices[1][1]);
    const available = atoms.filter(atom => atom.path.openingAvailable);
    const maximum = available.length ? Math.max(...available.map(atom => Math.floor((c.maxNotional + 1e-8)
      / (P * atom.path.openRatio * step)))) + 2 : 0;
    if (!Number.isSafeInteger(maximum) || Math.max(Math.abs(rawQlo / step), Math.abs(rawQhi / step)) + maximum > 1e12)
      return { upperTerminalEquity: Infinity, endpointUpperTerminalEquities: [Infinity, Infinity] as const,
        complete: false, search: { maximumLots: maximum, evaluatedOrders: 0, intervals: 0 } };
    const qScale = Math.max(1, Math.abs(rawQlo), Math.abs(rawQhi), maximum * step);
    const qPad = 64 * Number.EPSILON * qScale;
    const cashPad = 64 * Number.EPSILON * Math.max(1, Math.abs(rawClo), Math.abs(rawChi), P * qScale, c.maxNotional);
    const Clo = rawClo - cashPad, Chi = rawChi + cashPad, Qlo = rawQlo - qPad, Qhi = rawQhi + qPad;
    const rounding = step / 2 + qPad;
    const rows = atoms.map(atom => {
      const p = atom.path, G = P * p.openRatio, close = P * p.closeRatio;
      const openings = vertices.map(([C, Q]) => ({ E: C + Q * G, N: Q * G }));
      const equityPad = cashPad + qPad * G;
      const plusPad = L * cashPad + (L - 1) * qPad * G, minusPad = L * cashPad + (L + 1) * qPad * G;
      return { ...atom, G, unit: G * step, close,
        Emin: Math.min(...openings.map(row => row.E)) - equityPad,
        Emax: Math.max(...openings.map(row => row.E)) + equityPad,
        beforeMax: openings.some(row => row.E <= equityPad) ? Infinity
          : Math.max(...openings.map(row => (Math.abs(row.N) + qPad * G) / (row.E - equityPad))),
        plusMin: Math.min(...openings.map(row => L * row.E - row.N)) - plusPad,
        plusMax: Math.max(...openings.map(row => L * row.E - row.N)) + plusPad,
        minusMin: Math.min(...openings.map(row => L * row.E + row.N)) - minusPad,
        minusMax: Math.max(...openings.map(row => L * row.E + row.N)) + minusPad };
    });
    type Row = typeof rows[number];
    const absoluteRange = (lo: number, hi: number) => [lo <= 0 && hi >= 0 ? 0 : Math.min(Math.abs(lo), Math.abs(hi)),
      Math.max(Math.abs(lo), Math.abs(hi))];
    const acceptance = (row: Row, k: number) => {
      const request = k * step, turnover = Math.abs(request) * row.G, cost = turnover * fee;
      if (!k || !row.path.openingAvailable || Math.abs(request) < c.minQuantity - 1e-12
        || turnover < c.minNotional - 1e-8 || turnover > c.maxNotional + 1e-8) return { possible: false, certain: false };
      const lo = row.Emin - cost, hi = row.Emax - cost;
      const plus = (1 + L * fee * Math.sign(k)) * row.unit * k;
      const minus = (L * fee * Math.sign(k) - 1) * row.unit * k;
      const certain = lo > 0 && row.plusMin >= plus && row.minusMin >= minus;
      const possible = hi > 0 && row.plusMax >= plus && row.minusMax >= minus;
      if (certain || turnover < c.maxNotional - row.unit - 1e-8 || !(hi > 0)) return { certain, possible };
      const previous = absoluteRange(Qlo, Qhi), after = absoluteRange(Qlo + request, Qhi + request);
      const beforeMin = previous[0] * row.G / row.Emax, beforeMax = row.beforeMax;
      const afterMin = after[0] * row.G / hi, afterMax = lo > 0 ? after[1] * row.G / lo : Infinity;
      return { certain: certain || beforeMin > c.maxLeverage && afterMax < beforeMin - 1e-9 && lo > 0,
        possible: possible || beforeMax > c.maxLeverage && afterMin < beforeMax - 1e-9 };
    };
    const grouped = new Map<number | "unavailable", Row[]>();
    for (const row of rows) {
      const key = row.path.openingAvailable ? row.path.openRatio : "unavailable", members = grouped.get(key) ?? [];
      members.push(row); grouped.set(key, members);
    }
    const groups = [...grouped.values()].sort((a, b) => (a[0].path.openingAvailable ? a[0].G : Infinity)
      - (b[0].path.openingAvailable ? b[0].G : Infinity));
    const guards = new Set<number>([-maximum, 0, maximum]);
    const boundary = (value: number) => {
      if (!Number.isFinite(value) || value < -maximum - 2 || value > maximum + 2) return;
      for (const k of [Math.floor(value) - 1, Math.floor(value), Math.ceil(value), Math.ceil(value) + 1])
        if (Math.abs(k) <= maximum) guards.add(k);
    };
    for (const row of rows) {
      if (!row.path.openingAvailable) continue;
      const minimum = Math.max((c.minQuantity - 1e-12) / step, (c.minNotional - 1e-8) / row.unit);
      boundary(minimum); boundary(-minimum);
      boundary((c.maxNotional + 1e-8) / row.unit); boundary(-(c.maxNotional + 1e-8) / row.unit);
      const recoveryLo = Math.max(1, Math.ceil((c.maxNotional - row.unit - 1e-8) / row.unit));
      const recoveryHi = Math.floor((c.maxNotional + 1e-8) / row.unit);
      if (recoveryHi - recoveryLo > 16) return { upperTerminalEquity: Infinity,
        endpointUpperTerminalEquities: [Infinity, Infinity] as const, complete: false,
        search: { maximumLots: maximum, evaluatedOrders: 0, intervals: 0 } };
      for (let k = recoveryLo; k <= recoveryHi; k++) { boundary(k); boundary(-k); }
      boundary(-Qlo / step); boundary(-Qhi / step);
      for (const side of [-1, 1]) {
        for (const E of [row.Emin, row.Emax]) boundary(E / (fee * row.unit * side));
        for (const value of [row.plusMin, row.plusMax]) boundary(value / (row.unit * (1 + L * fee * side)));
        for (const value of [row.minusMin, row.minusMax]) boundary(value / (row.unit * (L * fee * side - 1)));
      }
    }
    const endpointBest = [-Infinity, -Infinity], endpointRequest = [0, 0], visited = new Set<number>();
    const consider = (k: number) => {
      if (!Number.isSafeInteger(k) || Math.abs(k) > maximum || visited.has(k)) return;
      visited.add(k);
      for (let endpoint = 0; endpoint < 2; endpoint++) {
        const [C, Q] = vertices[endpoint];
        const choices = groups.map(members => {
          const row = members[0], request = k * step, turnover = Math.abs(request) * row.G;
          const eligible = !!k && row.path.openingAvailable && Math.abs(request) >= c.minQuantity - 1e-12
            && turnover >= c.minNotional - 1e-8 && turnover <= c.maxNotional + 1e-8;
          const recovery = eligible && row.beforeMax > c.maxLeverage && turnover >= c.maxNotional - row.unit - 1e-8;
          const accepted = acceptance(row, k);
          let traded = 0, held = 0;
          for (const member of members) {
            const base = C + Q * member.close, slope = member.close - member.G;
            held += member.probability * Math.max(0, base + cashPad + qPad * member.close);
            traded += member.probability * Math.max(0, base + step * k * slope - Math.abs(k) * member.unit * fee
              + cashPad + qPad * member.close + rounding * Math.abs(slope));
          }
          return { accepted: accepted.possible ? traded : -Infinity,
            rejected: accepted.certain ? -Infinity : held, free: !eligible || recovery };
        });
        const value = maximizeEventAcceptanceSequence(choices, "interval");
        if (value > endpointBest[endpoint] || value === endpointBest[endpoint] && Math.abs(k) < Math.abs(endpointRequest[endpoint])) {
          endpointBest[endpoint] = value; endpointRequest[endpoint] = k;
        }
      }
    };
    const points = [...guards].sort((a, b) => a - b);
    for (const k of points) consider(k);
    let intervals = 0;
    for (let i = 1; i < points.length; i++) {
      const lo = points[i - 1] + 1, hi = points[i] - 1; if (lo > hi) continue;
      // Possible/certain acceptance is fixed here. For either endpoint every
      // accepted/rejected sequence is affine in k; their maximum is convex.
      intervals++; consider(lo); consider(hi);
    }
    const pad = 1e-10 * Math.max(1, ...endpointBest.map(Math.abs));
    const endpointUpperTerminalEquities = endpointBest.map(value => value + pad) as [number, number];
    return { upperTerminalEquity: Math.max(...endpointUpperTerminalEquities), endpointUpperTerminalEquities,
      optimisticRequests: endpointRequest.map(k => k * step) as [number, number], complete: true,
      search: { maximumLots: maximum, evaluatedOrders: visited.size, intervals } };
  };
}
