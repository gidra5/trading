import { eventLeaf, type EventDistribution, type MoveSample } from "./event-distribution.js";

/** Weighted empirical CRPS = E|X-y| - 0.5 E|X-X'|.
 * Sorted prefix sums prepare a distribution once and score in O(log n).
 * This compares unchanged physical targets across different class grids. */
export function prepareEventCrps(values: readonly number[], probabilities: readonly number[]): (actual: number) => number {
  if (!values.length || values.length !== probabilities.length || values.some(v => !Number.isFinite(v))
    || probabilities.some(p => !Number.isFinite(p) || p < 0)) throw new Error("Invalid CRPS distribution");
  const total = probabilities.reduce((s, p) => s + p, 0);
  if (!(total > 0) || !Number.isFinite(total)) throw new Error("Invalid CRPS probability mass");
  const rows = values.map((value, i) => ({ value, weight: probabilities[i] / total })).sort((a, b) => a.value - b.value);
  const mass = [0], moment = [0]; let halfPairDistance = 0;
  for (let i = 0; i < rows.length; i++) {
    const { value, weight } = rows[i];
    halfPairDistance += weight * (value * mass[i] - moment[i]);
    mass.push(mass[i] + weight); moment.push(moment[i] + weight * value);
  }
  return actual => {
    if (!Number.isFinite(actual)) throw new Error("Invalid CRPS observation");
    let low = 0, high = rows.length;
    while (low < high) { const mid = (low + high) >>> 1; if (rows[mid].value <= actual) low = mid + 1; else high = mid; }
    return actual * (2 * mass[low] - mass[rows.length]) + moment[rows.length] - 2 * moment[low] - halfPairDistance;
  };
}

export function eventMarginalCrps(model: EventDistribution, samples: readonly MoveSample[]) {
  if (!samples.length) throw new Error("CRPS requires scored observations");
  const scores = model.kernels.map(k => ({
    returns: prepareEventCrps(k.map(a => a.return * 10000), k.map(a => a.probability)),
    durations: prepareEventCrps(k.map(a => a.duration * 60), k.map(a => a.probability)),
  }));
  let returns = 0, durations = 0;
  for (const s of samples) {
    const score = scores[eventLeaf(model, s.features)];
    returns += score.returns(s.return * 10000); durations += score.durations(s.duration * 60);
  }
  return { returnCrpsBps: returns / samples.length, durationCrpsSeconds: durations / samples.length };
}
