import { EVENT_RUN_VOLATILITY_FEATURES, eventLeaf, eventMoveLabel, validateEventDistribution,
  type EventDistribution, type MoveSample, type MoveAtom } from "./event-distribution.js";

/** Keep the fitted canonical partition, but learn physical joint paths at two
 * observed volatility levels. Low-volatility paths use recent history; the
 * high band adds older completed episodes. Every successor uses its own RV5.
 * The prior mostly pools within the current volatility band, with a small
 * global component so a quiet state does not rule out an unseen shock. */
export function trainEventVolatilityLaw(base: EventDistribution, recent: readonly MoveSample[], older: readonly MoveSample[],
  options: { quantile: number; prior: number; globalPriorShare?: number; compressPrior?: boolean }): EventDistribution {
  const globalPriorShare = options.globalPriorShare ?? 0.1;
  if (!base.runConditioned || base.runSymmetry || base.runVolatility || base.hidden || base.forest || base.projection || base.boost
    || !(options.quantile > 0 && options.quantile < 1) || !Number.isFinite(options.prior) || options.prior < 0 || !recent.length
    || !(globalPriorShare >= 0 && globalPriorShare <= 1))
    throw new Error("Invalid volatility law source or settings");
  const rows = [...recent, ...older], width = EVENT_RUN_VOLATILITY_FEATURES.length;
  if (rows.some(r => r.features.length !== width || r.nextFeatures.length !== width
    || [...r.features, ...r.nextFeatures].some(v => !Number.isFinite(v)) || r.features[width - 1] < 0 || r.nextFeatures[width - 1] < 0))
    throw new Error("Volatility law requires observed current and successor RV5");
  const sorted = recent.map(r => r.features[width - 1]).sort((a, b) => a - b), cut = sorted[Math.floor(sorted.length * options.quantile)];
  const high = (r: MoveSample) => r.features[width - 1] > cut;
  const populations = [recent.filter(r => !high(r)), [...recent.filter(high), ...older.filter(high)]];
  if (populations.some(p => !p.length)) throw new Error("Missing volatility-band training population");
  const model: EventDistribution = { version: 1, clock: base.clock, featureNames: EVENT_RUN_VOLATILITY_FEATURES, nodes: base.nodes.map(n => ({ ...n })),
    runVolatility: { cut, prior: options.prior, globalPriorShare, recentSamples: recent.length, pooledHighSamples: older.filter(high).length },
    kernels: [], counts: [], classProbabilities: [], priorClasses: new Array<number>(15).fill(0), trainingSamples: populations[0].length + populations[1].length };
  const bands = populations.map(rows => rows.map(r => ({ state: eventLeaf(model, r.features),
    atom: { return: r.return, low: r.low, high: r.high, duration: r.duration, next: eventLeaf(model, r.nextFeatures), probability: 0 } })));
  // Compress only the global prior, within joint class/successor strata.
  // Never erase an observed rare event class or successor. Each stratum keeps
  // its worst excursion paths and a median representative for remaining mass.
  const priors = [...bands, bands.flat()].map(rows => {
    if (options.compressPrior === false) return rows.map(({ atom }) => ({ ...atom, probability: 1 / rows.length }));
    const strata = new Map<string, MoveAtom[]>();
    for (const { atom } of rows) {
      const key = `${atom.next}:${eventMoveLabel(atom.return, atom.duration, base.clock)}`;
      if (!strata.has(key)) strata.set(key, []);
      strata.get(key)!.push(atom);
    }
    const prior: MoveAtom[] = [];
    for (const group of strata.values()) {
      const ordered = group.sort((a, b) => a.return - b.return);
      const extreme = new Set([ordered.reduce((a, b) => a.low < b.low ? a : b), ordered.reduce((a, b) => a.high > b.high ? a : b)]);
      prior.push(...[...extreme].map(a => ({ ...a, probability: 1 / rows.length })));
      const ordinary = ordered.filter(a => !extreme.has(a));
      if (ordinary.length) prior.push({ ...ordinary[Math.floor(ordinary.length / 2)], probability: ordinary.length / rows.length });
    }
    return prior;
  });
  for (let state = 0; state < base.kernels.length * 2; state++) {
    const band = state % 2, observed = bands[band].filter(r => r.state === state), n = observed.length;
    const strength = n ? options.prior : Math.max(1, options.prior), total = n + strength;
    const prior = [...priors[band].map(a => ({ ...a, probability: a.probability * (1 - globalPriorShare) })),
      ...priors[2].map(a => ({ ...a, probability: a.probability * globalPriorShare }))];
    const kernel = [...observed.map(r => ({ ...r.atom, probability: 1 / total })),
      ...(strength ? prior.map(a => ({ ...a, probability: a.probability * strength / total })) : [])];
    const classes = new Array<number>(15).fill(0);
    for (const a of kernel) classes[eventMoveLabel(a.return, a.duration, base.clock)] += a.probability;
    model.kernels.push(kernel); model.counts.push(n); model.classProbabilities.push(classes);
  }
  for (const r of populations.flat()) model.priorClasses[r.label] += 1 / model.trainingSamples;
  validateEventDistribution(model);
  return model;
}

/** Lift the original quiet-state kernel into the refined state space without
 * changing its return/extrema/duration/parent-successor marginal. Resolve next
 * RV5 from the training path that generated each original atom (including its
 * reciprocal counterpart). Ambiguous identical paths split by observed counts.
 * No evaluated price or guessed next-volatility transition is permitted. */
export function retainQuietEventLaw(base: EventDistribution, refined: EventDistribution, recent: readonly MoveSample[]): EventDistribution {
  if (!base.runConditioned || !refined.runVolatility || JSON.stringify(base.nodes) !== JSON.stringify(refined.nodes)
    || JSON.stringify(base.clock) !== JSON.stringify(refined.clock) || refined.kernels.length !== 2 * base.kernels.length
    || recent.some(r => r.features.length !== 21 || r.nextFeatures.length !== 21 || [...r.features, ...r.nextFeatures].some(v => !Number.isFinite(v))))
    throw new Error("Incompatible quiet-law refinement");
  const key = (a: Pick<MoveAtom, "return" | "low" | "high" | "duration" | "next">) =>
    [a.return, a.low, a.high, a.duration, a.next].join(":");
  const provenance = new Map<string, number[]>();
  const bySuccessor = new Map<string, Array<{ atom: Pick<MoveAtom, "return" | "low" | "high">; band: number }>>();
  for (const r of recent) {
    const next = eventLeaf(base, r.nextFeatures.slice(0, base.featureNames.length)), band = Number(r.nextFeatures[20] > refined.runVolatility.cut), side = next % 3;
    const paths = [{ ...r, next }, { return: -r.return / (1 + r.return), low: -r.high / (1 + r.high), high: -r.low / (1 + r.low),
      duration: r.duration, next: next - side + (side === 2 ? 2 : 1 - side) }];
    for (const a of paths) {
      const k = key(a); if (!provenance.has(k)) provenance.set(k, [0, 0]);
      provenance.get(k)![band]++;
      const bucket = `${a.duration}:${a.next}`;
      if (!bySuccessor.has(bucket)) bySuccessor.set(bucket, []);
      bySuccessor.get(bucket)!.push({ atom: a, band });
    }
  }
  const model: EventDistribution = { ...refined, runVolatility: { ...refined.runVolatility, quietBase: true },
    kernels: refined.kernels.map((kernel, state) => {
      if (state % 2) return kernel.map(a => ({ ...a }));
      return base.kernels[state / 2].flatMap(a => {
        let bands = provenance.get(key(a));
        if (!bands) {
          // Reciprocal round trips can differ by an ulp, including exactly on
          // a decimal-rounding boundary. Search only matching duration/next
          // buckets and verify all three prices at machine precision.
          const matches = (bySuccessor.get(`${a.duration}:${a.next}`) ?? []).filter(({ atom }) =>
            (["return", "low", "high"] as const).every(k => Math.abs(a[k] - atom[k])
              <= 32 * Number.EPSILON * Math.max(Math.abs(a[k]), Math.abs(atom[k]), 1e-8)));
          if (matches.length) {
            bands = [0, 0]; for (const match of matches) bands[match.band]++;
            provenance.set(key(a), bands);
          }
        }
        if (!bands) throw new Error("Original quiet atom has no observed successor-volatility provenance");
        const count = bands[0] + bands[1];
        return bands.flatMap((n, band) => n ? [{ ...a, next: 2 * a.next + band, probability: a.probability * n / count }] : []);
      });
    }), classProbabilities: [] };
  model.classProbabilities = model.kernels.map(kernel => {
    const classes = new Array<number>(15).fill(0);
    for (const a of kernel) classes[eventMoveLabel(a.return, a.duration, model.clock)] += a.probability;
    return classes;
  });
  validateEventDistribution(model); return model;
}
