import { prepareEventExecutionOneStep, type EventExecutionAtom } from "./event-execution-one-step.js";
import { prepareEventExecutionUpper } from "./event-execution-upper.js";
import { prepareEventExecutionPartitions } from "./event-execution-partitions.js";
import { evaluateEventExecutionPath } from "./event-execution-path.js";
import type { EventAccount } from "./event-log-policy.js";

export interface EventExecutionSuccessor extends EventExecutionAtom { next: number; }

/** Bound/evaluate ONE committed H2 root request. The second-stage action is
 * globally optimized under the same execution law. Exact-state memoization and
 * opening-information upper bounds avoid unnecessary continuation solves.
 * A pruned request is proven unable to beat the supplied achievable incumbent;
 * An explicit positive valueTolerance permits a certified fixed-request value
 * interval using nested information partitions. Its lower endpoint retains an
 * executable continuation policy. This routine does not search or certify the
 * full set of root requests. */
export function prepareEventExecutionBackup(input: readonly (readonly EventExecutionSuccessor[])[]) {
  if (!input.length || input.some(k => k.some(a => !Number.isInteger(a.next) || a.next < 0 || a.next >= input.length)))
    throw new Error("Invalid execution successor law");
  const kernels = structuredClone(input), solvers = kernels.map(k => prepareEventExecutionOneStep(k));
  const bounds = kernels.map(k => prepareEventExecutionUpper(k));
  const partitions: Array<ReturnType<typeof prepareEventExecutionPartitions> | undefined> = kernels.map(() => undefined);
  const costs = kernels[0][0].path.costs;
  if (kernels.some(k => k.some(a => (Object.keys(costs) as Array<keyof typeof costs>).some(key => a.path.costs[key] !== costs[key]))))
    throw new Error("Execution Bellman states require identical costs");
  type Continuation = { leaf: number; account: EventAccount; lower: number; upper: number; quantity: number; exact: boolean; partitioned: boolean };
  const cache = new Map<string, Continuation>();
  const prepare = (leaf: number, account: EventAccount) => {
    const key = JSON.stringify([leaf, account.equity, account.price, account.exposure]);
    let row = cache.get(key);
    if (!row) {
      let lower = 0;
      for (const a of kernels[leaf]) if (a.probability) lower += a.probability * evaluateEventExecutionPath(a.path, account, 0).logGrowth;
      const upper = bounds[leaf](account);
      if (upper < lower - 1e-10) throw new Error("Execution continuation upper below holding policy");
      row = { leaf, account, lower, upper, quantity: 0, exact: upper === -Infinity, partitioned: false };
      cache.set(key, row);
    }
    return { key, row };
  };
  return (leaf: number, account: EventAccount, quantity: number,
    options: { incumbent?: number; maxSolves?: number; maxSeconds?: number; valueTolerance?: number } = {}) => {
    const incumbent = options.incumbent ?? -Infinity, maxSolves = options.maxSolves ?? Infinity, maxSeconds = options.maxSeconds ?? Infinity;
    const tolerance = options.valueTolerance ?? 0;
    if (!Number.isInteger(leaf) || leaf < 0 || leaf >= kernels.length || Number.isNaN(incumbent) || incumbent === Infinity
      || !(maxSolves >= 0) || !(maxSeconds >= 0) || !Number.isFinite(tolerance) || tolerance < 0) throw new Error("Invalid execution backup options");
    const started = performance.now(), branches = new Map<string, { probability: number; immediate: number; row: Continuation }>();
    let solves = 0, partitionSolves = 0, rootRuin = false;
    for (const atom of kernels[leaf]) {
      if (!atom.probability) continue;
      const next = evaluateEventExecutionPath(atom.path, account, quantity);
      if (!Number.isFinite(next.logGrowth)) { rootRuin = true; break; }
      const prepared = prepare(atom.next, { equity: next.equity, price: next.price, exposure: next.exposure });
      const existing = branches.get(prepared.key);
      if (existing) existing.probability += atom.probability;
      else branches.set(prepared.key, { probability: atom.probability, immediate: next.logGrowth, row: prepared.row });
    }
    const rows = [...branches.values()];
    const total = (side: "lower" | "upper") => {
      if (rootRuin || rows.some(r => r.row[side] === -Infinity)) return -Infinity;
      if (rows.some(r => r.row[side] === Infinity)) return Infinity;
      return rows.reduce((s, r) => s + r.probability * (r.immediate + r.row[side]), 0);
    };
    const finish = (status: "complete" | "certified" | "pruned" | "budget") => {
      const lowerValue = total("lower"), exact = status === "complete";
      return { quantity, scope: "fixed-root-two-event-value" as const, valueTolerance: tolerance,
        status, complete: exact, certified: exact || status === "certified", value: exact ? lowerValue : null, lowerValue,
        upperValue: exact ? lowerValue : total("upper") + 1e-10, continuationSolves: solves, partitionSolves,
        continuationStates: rows.length, cacheSize: cache.size, seconds: (performance.now() - started) / 1000,
        continuationPolicy: rows.map(r => ({ probability: r.probability, leaf: r.row.leaf, account: r.row.account,
          quantity: r.row.quantity, exact: r.row.exact, lower: r.row.lower, upper: r.row.upper })) };
    };
    if (rootRuin || rows.some(r => r.row.exact && r.row.upper === -Infinity)) return finish("complete");
    while (rows.some(r => !r.row.exact)) {
      if (total("upper") + 1e-10 <= incumbent) return finish("pruned");
      if (tolerance > 0 && total("upper") + 1e-10 - total("lower") <= tolerance) return finish("certified");
      if (solves + partitionSolves >= maxSolves || performance.now() - started >= maxSeconds * 1000) return finish("budget");
      // Re-rank after every tightening: a solved small group should not force
      // a full continuation while other branches still have wider bounds.
      const branch = rows.filter(r => !r.row.exact).reduce((best, row) =>
        row.probability * (row.row.upper - row.row.lower) > best.probability * (best.row.upper - best.row.lower) ? row : best);
      // Calibration profiling shows that the extra partition solves are useful
      // on large kernels; on small ones a direct exact solve is often cheaper.
      // This gate changes work only, never the lower/upper value contract.
      if (tolerance > 0 && kernels[branch.row.leaf].length >= 1024 && !branch.row.partitioned) {
        const row = branch.row, partition = partitions[row.leaf] ??= prepareEventExecutionPartitions(kernels[row.leaf], 1);
        const result = partition(row.account, 1); partitionSolves++;
        if (result.lowerValue > row.upper + 1e-10 || result.upperValue < row.lower - 1e-10)
          throw new Error("Execution information partitions violate continuation bounds");
        if (result.lowerValue >= row.lower) { row.lower = result.lowerValue; row.quantity = result.quantity; }
        row.upper = Math.max(row.lower, Math.min(row.upper, result.upperValue)); row.partitioned = true;
        row.exact = result.complete || row.upper === -Infinity;
        if (row.upper === -Infinity) return finish("complete");
        continue;
      }
      const row = branch.row, result = solvers[row.leaf](row.account); solves++;
      if (!result.complete || result.value > row.upper + 1e-10 || result.value < row.lower - 1e-10)
        throw new Error("Execution exact continuation violates its bounds");
      row.lower = result.value; row.upper = result.value; row.quantity = result.quantity; row.exact = true;
      if (!Number.isFinite(result.value)) return finish("complete");
    }
    return finish("complete");
  };
}
