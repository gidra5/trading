import { cloneEventExecutionAtoms, prepareEventExecutionOneStep, type EventExecutionAtom } from "./event-execution-one-step.js";
import { evaluateEventExecutionPath } from "./event-execution-path.js";
import type { EventAccount } from "./event-log-policy.js";

/** Nested information bounds for the SAME committed-order H1 problem. A
 * relaxed controller observes which opening-price group occurs before choosing
 * its request. Coarsening to one group restores nonanticipativity exactly.
 * Group-optimal requests also supply executable candidates under the full law,
 * giving a lower bound without assuming their independent choices are feasible
 * as a single real request. No return probabilities or execution rules change. */
export function prepareEventExecutionPartitions(input: readonly EventExecutionAtom[], maximumDepth = 3) {
  if (!Number.isInteger(maximumDepth) || maximumDepth < 0 || maximumDepth > 8) throw new Error("Invalid execution partition depth");
  const atoms = cloneEventExecutionAtoms(input);
  const ordered = atoms.map((atom, index) => ({ atom, index })).sort((a, b) => a.atom.path.openRatio - b.atom.path.openRatio || a.index - b.index);
  type Node = { rows: typeof ordered; mass: number; solve: ReturnType<typeof prepareEventExecutionOneStep> };
  const node = (rows: typeof ordered, root = false): Node => {
    const mass = rows.reduce((s, a) => s + a.atom.probability, 0);
    return { rows, mass: root ? 1 : mass, solve: prepareEventExecutionOneStep(root ? atoms
      : rows.map(a => ({ ...a.atom, probability: a.atom.probability / mass }))) };
  };
  const levels: Node[][] = [[node(ordered, true)]];
  for (let depth = 1; depth <= maximumDepth; depth++) {
    const next: Node[] = [];
    for (const parent of levels.at(-1)!) {
      const total = parent.rows.reduce((s, a) => s + a.atom.probability, 0);
      let sum = 0, cut = 0, distance = Infinity;
      for (let i = 1; i < parent.rows.length; i++) {
        sum += parent.rows[i - 1].atom.probability;
        if (parent.rows[i - 1].atom.path.openRatio === parent.rows[i].atom.path.openRatio) continue;
        if (Math.abs(sum - total / 2) < distance) { cut = i; distance = Math.abs(sum - total / 2); }
      }
      if (!cut) next.push(parent);
      else next.push(node(parent.rows.slice(0, cut)), node(parent.rows.slice(cut)));
    }
    levels.push(next);
  }
  return Object.assign((account: EventAccount, depth: number) => {
    if (!Number.isInteger(depth) || depth < 0 || depth > maximumDepth) throw new Error("Invalid execution partition query");
    const groups = levels[depth].map((part, index) => ({ index, mass: part.mass, ...part.solve(account) }));
    if (!depth || groups.length === 1) return { lowerValue: groups[0].value, upperValue: groups[0].value, quantity: groups[0].quantity,
      complete: true, gap: 0, groups, candidates: [groups[0].quantity] };
    const upper = groups.some(g => g.value === -Infinity) ? -Infinity : groups.reduce((s, g) => s + g.mass * g.value, 0);
    const candidates = [...new Set([0, ...groups.map(g => g.quantity)])];
    let lower = -Infinity, quantity = 0;
    for (const request of candidates) {
      let value = 0;
      for (const atom of atoms) value += atom.probability * evaluateEventExecutionPath(atom.path, account, request).logGrowth;
      if (value > lower || value === lower && Math.abs(request) < Math.abs(quantity)) { lower = value; quantity = request; }
    }
    if (upper < lower - 1e-10) throw new Error("Conditional execution partition optimum below a common request");
    return { lowerValue: lower, upperValue: upper === -Infinity ? upper : upper + 1e-10, quantity, complete: upper === -Infinity,
      gap: upper === lower && upper === -Infinity ? 0 : Math.max(0, upper + 1e-10 - lower), groups, candidates };
  }, { groupCounts: levels.map(level => level.length) });
}
