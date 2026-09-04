import { EVENT_FEATURES, validateEventDistribution, type EventClock, type EventDistribution,
  type MoveAtom, type MoveSample } from "./event-distribution.js";

const normalize = (values: number[]) => {
  const total = values.reduce((sum, v) => sum + v, 0);
  if (!(total > 0) || !Number.isFinite(total)) throw new Error("Degenerate event belief");
  return values.map(v => v / total);
};
const nearest = (beliefs: number[][], value: number[]) => {
  let best = 0, distance = Infinity;
  for (let i = 0; i < beliefs.length; i++) {
    const d = value.reduce((sum, p, j) => sum + (p - beliefs[i][j]) ** 2, 0);
    if (d < distance) { best = i; distance = d; }
  }
  return best;
};

/** Baum-Welch on contiguous, non-overlapping observed event sequences.
 * The policy state is a quantized posterior, never the generating hidden state.
 * Class-conditioned empirical atoms pool tails across regimes, deliberately
 * limiting the model's degrees of freedom. */
export function trainEventHidden(samples: readonly MoveSample[], clock: EventClock,
  options: { states: number; resolution: number; iterations: number; smoothing: number; atomsPerClass?: number }): EventDistribution {
  const { states, resolution, iterations, smoothing } = options, atomsPerClass = options.atomsPerClass ?? 8;
  if (![2, 3].includes(states) || !Number.isInteger(resolution) || resolution < 2 || resolution > 16
    || !Number.isInteger(iterations) || iterations < 1 || iterations > 500 || !Number.isFinite(smoothing) || smoothing <= 0
    || !Number.isInteger(atomsPerClass) || atomsPerClass < 1 || samples.length < 30) throw new Error("Invalid hidden event model options");
  const sequences: number[][] = [], groups: MoveSample[][] = Array.from({ length: 15 }, () => []);
  let previousEnd = -1, previousSeries = samples[0]?.series;
  for (const s of samples) {
    if (!Number.isInteger(s.label) || s.label < 0 || s.label >= 15 || s.end <= s.start
      || (s.series === previousSeries && s.start < previousEnd) || !Number.isFinite(s.return) || !Number.isFinite(s.duration))
      throw new Error("Hidden event training requires ordered non-overlapping completed events");
    if (s.start !== previousEnd || s.series !== previousSeries) sequences.push([]);
    sequences.at(-1)!.push(s.label); groups[s.label].push(s); previousEnd = s.end; previousSeries = s.series;
  }
  const priorClasses = groups.map(g => g.length / samples.length);
  let transition = Array.from({ length: states }, (_, i) => Array.from({ length: states }, (_, j) => i === j ? 0.8 : 0.2 / (states - 1)));
  let emission = Array.from({ length: states }, (_, i) => normalize(priorClasses.map((p, label) =>
    p * Math.exp(1.5 * (2 * i / (states - 1) - 1) * (Math.floor(label / 3) - 2)))));
  let initial = new Array<number>(states).fill(1 / states), logLikelihood = -Infinity;
  for (let iteration = 0; iteration <= iterations; iteration++) {
    const transitionCounts = Array.from({ length: states }, () => new Array<number>(states).fill(smoothing / states));
    const emissionCounts = Array.from({ length: states }, () => priorClasses.map(p => smoothing * p));
    const initialCounts = new Array<number>(states).fill(smoothing / states);
    logLikelihood = 0;
    for (const labels of sequences) {
      const n = labels.length, forward: number[][] = [], factors: number[] = [];
      for (let t = 0; t < n; t++) {
        const predicted = t ? Array.from({ length: states }, (_, j) => forward[t - 1].reduce((s, p, i) => s + p * transition[i][j], 0)) : initial;
        const raw = predicted.map((p, j) => p * emission[j][labels[t]]), factor = raw.reduce((s, p) => s + p, 0);
        factors.push(factor); forward.push(raw.map(p => p / factor)); logLikelihood += Math.log(factor);
      }
      let backward = new Array<number>(states).fill(1);
      for (let t = n - 1; t >= 0; t--) {
        const gamma = normalize(forward[t].map((p, j) => p * backward[j]));
        gamma.forEach((p, j) => { emissionCounts[j][labels[t]] += p; if (!t) initialCounts[j] += p; });
        if (t) {
          const nextBackward = new Array<number>(states).fill(0);
          for (let i = 0; i < states; i++) for (let j = 0; j < states; j++) {
            const conditional = transition[i][j] * emission[j][labels[t]] * backward[j] / factors[t];
            transitionCounts[i][j] += forward[t - 1][i] * conditional;
            nextBackward[i] += conditional;
          }
          backward = nextBackward;
        }
      }
    }
    if (iteration === iterations) break;
    transition = transitionCounts.map(normalize); emission = emissionCounts.map(normalize); initial = normalize(initialCounts);
  }
  // A fresh episode starts with no hidden-state knowledge. Use the fitted
  // stationary prior instead of inferring the final training regime in hindsight.
  let stationary = new Array<number>(states).fill(1 / states);
  for (let i = 0; i < 500; i++) stationary = Array.from({ length: states }, (_, j) => stationary.reduce((s, p, k) => s + p * transition[k][j], 0));
  const beliefs: number[][] = [];
  const grid = (prefix: number[], remaining: number) => {
    if (prefix.length === states - 1) { beliefs.push([...prefix, remaining].map(v => v / resolution)); return; }
    for (let i = 0; i <= remaining; i++) grid([...prefix, i], remaining - i);
  };
  grid([], resolution);
  const initialState = nearest(beliefs, stationary), nextByClass: number[][] = [], classProbabilities: number[][] = [];
  // Compress within class and return bins using endpoint quadrature: preserve
  // class probability and mean return/duration, retain worst observed excursions.
  // This is a finite quadrature, not an exact empirical path distribution.
  const templates = groups.map(group => {
    const sorted = [...group].sort((a, b) => a.return - b.return), atoms: Omit<MoveAtom, "next">[] = [];
    const bins = Math.min(atomsPerClass, sorted.length);
    for (let bin = 0; bin < bins; bin++) {
      const rows = sorted.slice(Math.floor(bin * sorted.length / bins), Math.floor((bin + 1) * sorted.length / bins));
      const mass = rows.length / group.length, minimum = rows[0].return, maximum = rows.at(-1)!.return;
      const mean = rows.reduce((s, r) => s + r.return, 0) / rows.length;
      const base = { low: Math.min(...rows.map(r => r.low)), high: Math.max(...rows.map(r => r.high)),
        duration: rows.reduce((s, r) => s + r.duration, 0) / rows.length };
      if (maximum - minimum < 1e-12) atoms.push({ ...base, return: mean, probability: mass });
      else {
        const upper = Math.max(0, Math.min(1, (mean - minimum) / (maximum - minimum)));
        if (upper < 1) atoms.push({ ...base, return: minimum, probability: mass * (1 - upper) });
        if (upper > 0) atoms.push({ ...base, return: maximum, probability: mass * upper });
      }
    }
    return atoms;
  });
  const kernels = beliefs.map(belief => {
    const predicted = Array.from({ length: states }, (_, j) => belief.reduce((s, p, i) => s + p * transition[i][j], 0));
    const probabilities = priorClasses.map((_, label) => predicted.reduce((s, p, j) => s + p * emission[j][label], 0));
    classProbabilities.push(probabilities);
    const next = probabilities.map((p, label) => p > 0 ? nearest(beliefs, predicted.map((v, j) => v * emission[j][label] / p)) : initialState);
    nextByClass.push(next);
    return templates.flatMap((atoms, label) => atoms.map(a => ({ ...a, probability: a.probability * probabilities[label], next: next[label] })));
  });
  const model: EventDistribution = { version: 1, clock, featureNames: EVENT_FEATURES, nodes: [], kernels,
    counts: beliefs.map(() => samples.length), classProbabilities, priorClasses, trainingSamples: samples.length,
    hidden: { transition, emission, beliefs, initial: initialState, nextByClass, iterations, sequences: sequences.length, logLikelihood, resolution } };
  validateEventDistribution(model);
  return model;
}
