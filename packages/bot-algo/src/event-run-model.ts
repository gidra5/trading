import { canonicalEventRunFeatures, eventRunOrientation, eventLeaf, eventMoveLabel, EVENT_RUN_FEATURES,
  trainEventDistribution, validateEventDistribution, type EventDistribution, type EventClock, type MoveSample, type MoveAtom } from "./event-distribution.js";

/** Learn shared run partitions in canonical coordinates. Without directionPrior,
 * physical laws are exact reflections. A finite directionPrior instead shrinks
 * each physical-direction empirical law toward that pooled law, in sample-count
 * units. Neither assumption makes account actions symmetric. */
export function trainEventRunDistribution(samples: readonly MoveSample[], clock: EventClock,
  options: { maxDepth: number; minLeaf: number; prior: number; criterion?: "mean" | "distribution";
    honestyFraction?: number; directionPrior?: number; priorLimit?: number }): EventDistribution {
  const priorLimit = options.priorLimit ?? 128;
  if (!clock.reversalClock || !Number.isFinite(options.prior) || options.prior < 0 || !Number.isInteger(options.minLeaf) || options.minLeaf < 1)
    throw new Error("Invalid canonical run training options");
  if (options.directionPrior !== undefined && (!Number.isFinite(options.directionPrior) || options.directionPrior < 0))
    throw new Error("Invalid run direction prior");
  if (!Number.isSafeInteger(priorLimit) || priorLimit < 1) throw new Error("Invalid run prior sample limit");
  const orientations = samples.map(s => eventRunOrientation(s.features));
  const nextOrientations = samples.map(s => eventRunOrientation(s.nextFeatures));
  const canonical = samples.map((s, i) => {
    const inverted = orientations[i] < 0, value = inverted ? -s.return / (1 + s.return) : s.return;
    return { ...s, features: canonicalEventRunFeatures(s.features), nextFeatures: canonicalEventRunFeatures(s.nextFeatures),
      return: value, low: inverted ? -s.high / (1 + s.high) : s.low, high: inverted ? -s.low / (1 + s.low) : s.high,
      label: eventMoveLabel(value, s.duration, clock) };
  });
  const tree = trainEventDistribution(canonical, clock, { ...options, featureNames: EVENT_RUN_FEATURES });
  const split = options.honestyFraction ? Math.floor(samples.length * (1 - options.honestyFraction)) : 0;
  const rows = samples.slice(split).map((_, i) => i + split);
  const groups: number[][] = tree.kernels.map(() => []);
  for (const i of rows) groups[eventLeaf(tree, canonical[i].features)].push(i);
  const sorted = [...rows].sort((a, b) => canonical[a].return - canonical[b].return);
  const prior: Array<{ index: number; weight: number }> = [], step = Math.max(1, Math.ceil(sorted.length / priorLimit));
  for (let i = 0; i < sorted.length; i += step) {
    const end = Math.min(sorted.length, i + step);
    prior.push({ index: sorted[Math.floor((i + end - 1) / 2)], weight: (end - i) / sorted.length });
  }
  const model: EventDistribution = { version: 1, clock, featureNames: EVENT_RUN_FEATURES, nodes: tree.nodes,
    ...(options.directionPrior === undefined ? { runSymmetry: true } : { runConditioned: { directionPrior: options.directionPrior } }),
    kernels: [], counts: [], classProbabilities: [], priorClasses: new Array<number>(15).fill(0), trainingSamples: rows.length };
  for (const i of rows) model.priorClasses[samples[i].label] += 1 / rows.length;
  for (const group of groups) {
    const strength = group.length ? options.prior : Math.max(1, options.prior), total = group.length + strength;
    const atom = (i: number, probability: number): MoveAtom => {
      const s = canonical[i], relativeDirection = nextOrientations[i] * (orientations[i] || 1);
      const side = relativeDirection > 0 ? 0 : relativeDirection < 0 ? 1 : 2;
      return { probability, return: s.return, low: s.low, high: s.high, duration: s.duration,
        next: 3 * eventLeaf(tree, s.nextFeatures) + side };
    };
    const up = [...group.map(i => atom(i, 1 / total)),
      ...(strength ? prior.map(p => atom(p.index, strength * p.weight / total)) : [])];
    const down = up.map(a => {
      const side = a.next % 3;
      return { ...a, return: -a.return / (1 + a.return), low: -a.high / (1 + a.high), high: -a.low / (1 + a.low),
        next: a.next - side + (side === 2 ? 2 : 1 - side) };
    });
    const neutral = [...up, ...down].map(a => ({ ...a, probability: a.probability / 2 }));
    for (const [side, pooled] of [up, down, neutral].entries()) {
      let kernel = pooled, n = group.length;
      if (options.directionPrior !== undefined) {
        const physical = group.filter(i => (orientations[i] > 0 ? 0 : orientations[i] < 0 ? 1 : 2) === side);
        n = physical.length;
        // Empty directions retain the pooled fallback even when requested
        // shrinkage is zero. No future direction or outcome enters this group.
        const strength = n ? options.directionPrior : Math.max(1, options.directionPrior), total = n + strength;
        kernel = physical.map(i => {
          const s = samples[i], direction = nextOrientations[i];
          return { probability: 1 / total, return: s.return, low: s.low, high: s.high, duration: s.duration,
            next: 3 * eventLeaf(tree, canonical[i].nextFeatures) + (direction > 0 ? 0 : direction < 0 ? 1 : 2) };
        });
        if (strength) kernel.push(...pooled.map(a => ({ ...a, probability: a.probability * strength / total })));
      }
      const classes = new Array<number>(15).fill(0.5 / (n + 7.5));
      for (const a of kernel) classes[eventMoveLabel(a.return, a.duration, clock)] += a.probability * n / (n + 7.5);
      model.kernels.push(kernel); model.counts.push(n); model.classProbabilities.push(classes);
    }
  }
  validateEventDistribution(model);
  return model;
}
