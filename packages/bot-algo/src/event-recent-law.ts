import { eventLeaf, eventMoveLabel, validateEventDistribution,
  type EventDistribution, type MoveAtom } from "./event-distribution.js";

export interface CompletedEventMove {
  originTime: number; availableAt: number; features: number[]; nextFeatures: number[];
  return: number; low: number; high: number; duration: number;
}
export interface RecentEventOptions { window: number; prior: number; after: number; scope?: "leaf" | "parent" | "direction"; }
interface RecentAtom { originTime: number; availableAt: number; leaf: number; label: number; atom: MoveAtom; }

/** A chronological empirical update of whole joint outcomes in fixed observable
 * states. The positive base mass retains historical tail support. The caller
 * supplies only completed, nonoverlapping events after the base fitting period. */
export class EventRecentLaw {
  private readonly history: RecentAtom[] = [];
  private readonly means: number[];
  private readonly classes: number[][];
  private readonly related: number[][];
  private now: number;
  constructor(readonly base: EventDistribution, readonly options: RecentEventOptions) {
    validateEventDistribution(base);
    if (base.hidden || base.runSymmetry || base.meanCalibration) throw new Error("Recent laws require unrestricted observable states");
    if (!Number.isInteger(options.window) || options.window < 1 || !Number.isFinite(options.prior) || options.prior <= 0
      || !Number.isFinite(options.after) || !["leaf", "parent", "direction"].includes(options.scope ?? "leaf"))
      throw new Error("Invalid recent-event options");
    if (options.scope && options.scope !== "leaf" && !base.runConditioned)
      throw new Error("Shared recent laws require direction-conditioned canonical run states");
    this.now = options.after;
    const parents = new Map<number, number>(), descendants = new Map<number, number[]>();
    if (options.scope === "parent") {
      const visit = (index: number, parent: number): number[] => {
        const node = base.nodes[index];
        if (node.leaf >= 0) { parents.set(node.leaf, parent); descendants.set(index, [node.leaf]); return [node.leaf]; }
        const leaves = [...visit(node.left, index), ...visit(node.right, index)];
        descendants.set(index, leaves); return leaves;
      };
      visit(0, 0);
    }
    this.related = base.kernels.map((_, leaf) => options.scope === "direction"
      ? base.kernels.map((_, i) => i).filter(i => i % 3 === leaf % 3)
      : options.scope === "parent"
        ? descendants.get(parents.get(Math.floor(leaf / 3))!)!.map(i => i * 3 + leaf % 3) : [leaf]);
    this.means = base.kernels.map(k => k.reduce((s, a) => s + a.probability * a.return, 0));
    this.classes = base.kernels.map(kernel => {
      const classes = new Array<number>(15).fill(0);
      for (const a of kernel) classes[eventMoveLabel(a.return, a.duration, base.clock)] += a.probability;
      return classes;
    });
  }
  private advance(now: number): void {
    if (!Number.isFinite(now) || now < this.now) throw new Error("Recent-event clock cannot move backwards");
    this.now = now;
  }
  private observations(leaf: number): RecentAtom[] {
    return this.history.filter(row => this.related[leaf].includes(row.leaf));
  }
  forecast(features: readonly number[], now: number) {
    this.advance(now);
    const leaf = eventLeaf(this.base, features), rows = this.observations(leaf);
    const total = this.options.prior + rows.length;
    const classes = this.classes[leaf].map(p => this.options.prior * p / total);
    for (const row of rows) classes[row.label] += 1 / total;
    return { leaf, recentCount: rows.length, localCount: rows.filter(row => row.leaf === leaf).length,
      mean: (this.options.prior * this.means[leaf]
      + rows.reduce((s, row) => s + row.atom.return, 0)) / total, classes };
  }
  observe(move: CompletedEventMove, now: number): void {
    if (!Number.isFinite(move.originTime) || !Number.isFinite(move.availableAt) || !Number.isFinite(move.duration)
      || move.duration <= 0 || move.availableAt !== move.originTime + move.duration * 60_000
      || move.originTime < this.options.after || move.availableAt > now
      || (this.history.length && move.originTime < this.history.at(-1)!.availableAt))
      throw new Error("Recent updates require completed, nonoverlapping post-fit events");
    if (![move.return, move.low, move.high].every(Number.isFinite) || move.low <= -1
      || move.low > Math.min(0, move.return) + 1e-12 || move.high < Math.max(0, move.return) - 1e-12)
      throw new Error("Invalid recent joint event support");
    const leaf = eventLeaf(this.base, move.features), next = eventLeaf(this.base, move.nextFeatures);
    this.advance(now);
    this.history.push({ originTime: move.originTime, availableAt: move.availableAt, leaf,
      label: eventMoveLabel(move.return, move.duration, this.base.clock),
      atom: { return: move.return, low: move.low, high: move.high, duration: move.duration, next, probability: 1 } });
    if (this.history.length > this.options.window) this.history.shift();
  }
  /** Freeze the current law for a receding-horizon solve. This does not model
   * future belief updates inside Bellman branches. */
  snapshot(now: number): EventDistribution {
    this.advance(now);
    const groups = this.base.kernels.map((_, leaf) => this.observations(leaf));
    const model = { ...this.base, kernels: this.base.kernels.map((kernel, leaf) => {
      const rows = groups[leaf], total = this.options.prior + rows.length;
      return rows.length ? [...kernel.map(a => ({ ...a, probability: a.probability > 0
        ? Math.max(Number.MIN_VALUE, a.probability * this.options.prior / total) : 0 })),
        ...rows.map(row => ({ ...row.atom, probability: 1 / total }))] : kernel;
    }), counts: this.base.counts.map((n, leaf) => n + groups[leaf].length), trainingSamples: this.base.trainingSamples + this.history.length,
    classProbabilities: this.classes.map((classes, leaf) => {
      const total = this.options.prior + groups[leaf].length, next = classes.map(p => this.options.prior * p / total);
      for (const row of groups[leaf]) next[row.label] += 1 / total;
      return next;
    }) };
    validateEventDistribution(model);
    return model;
  }
}
