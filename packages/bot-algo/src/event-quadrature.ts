import { eventMoveLabel, validateEventClock, validateEventDistribution, type EventClock, type EventDistribution, type MoveAtom } from "./event-distribution.js";

/** Positive support reduction on observed paths. The basis preserves mass,
 * arithmetic/reciprocal return, log return, second return moment and both
 * duration moments. It does not make the entire Bellman integrand exact.
 * See https://arxiv.org/html/1611.02065, discrete Tchakaloff theorem. */
export function eventQuadratureFeatures(a: MoveAtom): number[] {
  return [1, a.return, -a.return / (1 + a.return), Math.log1p(a.return), a.return ** 2, a.duration, Math.log1p(a.duration)];
}

/** Null direction of a short, row-scaled rectangular moment matrix. */
function nullDirection(columns: readonly number[][]): number[] | null {
  const width = columns.length, height = columns[0].length;
  const matrix = Array.from({ length: height }, (_, r) => columns.map(c => c[r]));
  const pivots: number[] = [];
  for (let col = 0; col < width && pivots.length < height; col++) {
    const row = pivots.length;
    let pivot = row;
    for (let r = row + 1; r < height; r++) if (Math.abs(matrix[r][col]) > Math.abs(matrix[pivot][col])) pivot = r;
    if (Math.abs(matrix[pivot][col]) < 1e-12) continue;
    [matrix[row], matrix[pivot]] = [matrix[pivot], matrix[row]];
    const value = matrix[row][col];
    for (let j = col; j < width; j++) matrix[row][j] /= value;
    for (let r = 0; r < height; r++) if (r !== row) {
      const scale = matrix[r][col];
      for (let j = col; j < width; j++) matrix[r][j] -= scale * matrix[row][j];
    }
    pivots.push(col);
  }
  const free = Array.from({ length: width }, (_, i) => i).find(i => !pivots.includes(i));
  if (free === undefined) return null;
  const v = new Array<number>(width).fill(0); v[free] = 1;
  for (let r = 0; r < pivots.length; r++) v[pivots[r]] = -matrix[r][free];
  if (v.some(x => !Number.isFinite(x))) return null;
  const max = Math.max(...v.map(Math.abs));
  return v.map(x => x / max);
}

function reduceGroup(group: readonly MoveAtom[]): MoveAtom[] {
  const width = eventQuadratureFeatures(group[0]).length;
  if (group.length <= width + 2) return group.map(a => ({ ...a }));
  // Preserve both adverse excursion/duration Pareto frontiers. For any
  // nonnegative borrowing rate, a dominated path cannot uniquely cause ruin:
  // another observed path has at least as adverse an excursion and duration.
  // Worst excursion alone is insufficient when borrowing accrues over time.
  const pinned = new Set<MoveAtom>();
  for (const side of ["low", "high"] as const) {
    const sorted = [...group].sort((a, b) => (side === "low" ? a.low - b.low : b.high - a.high) || b.duration - a.duration);
    let longest = -Infinity;
    for (const a of sorted) if (a.duration > longest) { pinned.add(a); longest = a.duration; }
  }
  if (group.length - pinned.size <= width) return group.map(a => ({ ...a }));
  const remaining = group.filter(a => !pinned.has(a));
  const features = remaining.map(eventQuadratureFeatures);
  const lower = Array.from({ length: width }, (_, j) => Math.min(...features.map(f => f[j])));
  const upper = Array.from({ length: width }, (_, j) => Math.max(...features.map(f => f[j])));
  const scaled = features.map(f => f.map((v, j) => j === 0 ? 1 : upper[j] > lower[j] ? (v - lower[j]) / (upper[j] - lower[j]) : 0));
  const active: Array<{ i: number; weight: number }> = [];
  const fallback = () => group.map(a => ({ ...a }));
  for (let i = 0; i < remaining.length; i++) {
    active.push({ i, weight: remaining[i].probability });
    if (active.length <= width) continue;
    const direction = nullDirection(active.map(a => scaled[a.i]));
    if (!direction) return fallback();
    let step = Infinity, remove = -1;
    for (let j = 0; j < direction.length; j++) if (direction[j] > 0 && active[j].weight / direction[j] < step) {
      step = active[j].weight / direction[j]; remove = j;
    }
    if (!Number.isFinite(step) || remove < 0) return fallback();
    for (let j = 0; j < active.length; j++) {
      const next = active[j].weight - step * direction[j];
      if (!Number.isFinite(next) || next < -1e-14) return fallback();
      active[j].weight = Math.max(0, next);
    }
    active.splice(remove, 1);
  }
  // Numerical safety is an accuracy gate, not an assumed theorem in floats.
  // If rank handling or accumulated error is poor, retain the whole group.
  for (let j = 0; j < width; j++) {
    const expected = remaining.reduce((s, a, i) => s + a.probability * scaled[i][j], 0);
    const actual = active.reduce((s, a) => s + a.weight * scaled[a.i][j], 0);
    const mass = remaining.reduce((s, a) => s + a.probability, 0);
    if (Math.abs(actual - expected) > 1e-10 * mass) return fallback();
  }
  return [...pinned].map(a => ({ ...a })).concat(active.filter(a => a.weight > 0).map(a => ({ ...remaining[a.i], probability: a.weight })));
}

/** Match moments separately inside each actual class/successor stratum.
 * The reciprocal class is also retained for a subsequent chart reflection.
 * Every returned path was present in the input; only masses are changed. */
export function compressEventKernel(kernel: readonly MoveAtom[], clock: EventClock): MoveAtom[] {
  validateEventClock(clock);
  if (!kernel.length
    || kernel.some(a => !Number.isFinite(a.probability) || a.probability < 0 || !Number.isFinite(a.return) || a.return <= -1
      || !Number.isFinite(a.low) || !Number.isFinite(a.high) || !Number.isFinite(a.duration) || a.duration <= 0
      || !Number.isSafeInteger(a.next) || a.next < 0)) throw new Error("Invalid observed kernel for quadrature");
  const groups = new Map<string, MoveAtom[]>();
  for (const a of kernel) {
    if (!a.probability) continue;
    const key = `${a.next}:${eventMoveLabel(a.return, a.duration, clock)}:${eventMoveLabel(-a.return / (1 + a.return), a.duration, clock)}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key)!.push(a);
  }
  return [...groups.values()].flatMap(reduceGroup);
}

export function compressEventDistribution(model: EventDistribution): EventDistribution {
  validateEventDistribution(model);
  const compressed = { ...model, kernels: model.kernels.map(k => compressEventKernel(k, model.clock)) };
  validateEventDistribution(compressed);
  return compressed;
}
