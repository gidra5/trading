import type { MoveSample } from "../packages/bot-algo/src/event-distribution.js";
import { chooseEventTrade, eventHolding, type EventAccount, type EventCosts } from "../packages/bot-algo/src/event-log-policy.js";
import { fittedEventHolding, type FittedEventValue } from "../packages/bot-algo/src/event-fitted-value.js";
type PathMove = Pick<MoveSample, "start" | "end" | "return" | "duration">;
type HoldingMove = Pick<MoveSample, "return" | "low" | "high" | "duration">;

/** Two-event observed payoff from an already chosen post-order account.
 * With a next H1 grid, choose its feasible action after event one; otherwise
 * hold or make the mandatory cap reduction. Event two never chooses its order.
 * Terminal settlement uses the fitted-target proportional-cost convention. */
export function eventControllerHolding(p: FittedEventValue, account: EventAccount, first: HoldingMove, second: HoldingMove,
  nextLeaf: number, nextGrid?: readonly number[]) {
  if (![account.equity, account.price, account.exposure].every(Number.isFinite) || account.equity <= 0 || account.price <= 0
    || [first, second].some(r => ![r.return, r.low, r.high, r.duration].every(Number.isFinite)
      || r.return <= -1 || r.low <= -1 || r.low > Math.min(0, r.return) || r.high < Math.max(0, r.return) || r.duration <= 0))
    throw new Error("Invalid controller holding path");
  const a = eventHolding(account.exposure, first, p.costs);
  if (a.liquidated) return { value: -Infinity, liquidated: true, trade: undefined };
  const middle = { equity: account.equity * a.factor, price: account.price * (1 + first.return), exposure: a.exposure };
  const trade = chooseEventTrade(p, middle, nextGrid ? held => fittedEventHolding(p, nextGrid, held, nextLeaf) : () => 0);
  const b = eventHolding(trade.exposure, second, p.costs), settlement = 1 - Math.abs(b.exposure) * (p.costs.feeBps + p.costs.slippageBps) / 10000;
  return { value: b.liquidated || !(settlement > 0) || !Number.isFinite(trade.value) ? -Infinity
    : Math.log(a.factor) + Math.log(trade.equity / middle.equity) + Math.log(b.factor) + Math.log(settlement),
    liquidated: b.liquidated, trade };
}

/** Counterfactual fixed-inventory holding values at each complete event prefix.
 * Exposure drifts with equity; terminal costs are charged once per prefix.
 * No entry or intermediate orders: cap breaches are explicitly reported and
 * these values alone do not define a feasible multi-event trading policy. */
export function eventPathHolding(exposure: number,
  steps: readonly Pick<MoveSample, "return" | "low" | "high" | "duration">[], costs: EventCosts) {
  if (!Number.isFinite(exposure) || !steps.length || steps.some(r => ![r.return, r.low, r.high, r.duration].every(Number.isFinite)
    || r.return <= -1 || r.low <= -1 || r.low > Math.min(0, r.return) || r.high < Math.max(0, r.return) || r.duration <= 0))
    throw new Error("Invalid holding path");
  let factor = 1, current = exposure, liquidated = false, capBreaches = 0;
  const fee = (costs.feeBps + costs.slippageBps) / 10000;
  return steps.map((step, i) => {
    if (!liquidated) {
      if (i > 0 && Math.abs(current) > costs.maxLeverage + 1e-9) capBreaches++;
      const held = eventHolding(current, step, costs);
      liquidated = held.liquidated;
      factor *= held.factor; current = held.exposure;
    }
    const settlement = 1 - Math.abs(current) * fee;
    return { factor, exposure: current, liquidated, capBreaches,
      value: liquidated || !(settlement > 0) ? -Infinity : Math.log(factor) + Math.log(settlement) };
  });
}

/** Require a complete contiguous path, never restart across omitted events. */
export function eventSignHorizonPaths<T extends PathMove>(samples: readonly T[], horizon: number) {
  if (!Number.isInteger(horizon) || horizon < 1 || samples.some(r => r.end <= r.start || !Number.isFinite(r.return)
    || !(r.return > -1) || !Number.isFinite(r.duration) || !(r.duration > 0))) throw new Error("Invalid horizon path inputs");
  const result: Array<{ start: number; end: number; steps: T[]; cumulativeReturns: number[]; cumulativeMinutes: number[] }> = [];
  for (let i = 0; i + horizon <= samples.length; i++) {
    const steps = samples.slice(i, i + horizon);
    if (steps.some((r, j) => j > 0 && r.start !== steps[j - 1].end)) continue;
    let multiplier = 1, minutes = 0;
    const cumulativeReturns: number[] = [], cumulativeMinutes: number[] = [];
    for (const r of steps) {
      multiplier *= 1 + r.return; minutes += r.duration;
      cumulativeReturns.push(multiplier - 1); cumulativeMinutes.push(minutes);
    }
    result.push({ start: steps[0].start, end: steps.at(-1)!.end, steps, cumulativeReturns, cumulativeMinutes });
  }
  return result;
}

/** Block-held-out H1 diagnostics. Purge complete feature/label support from
 * complementary training; past-only indices additionally forbid later rows. */
export function eventPolicyEvaluationFolds(training: readonly { start: number; end: number }[],
  evaluation: readonly { start: number; end: number }[], start: number, end: number, parts: number, history: number) {
  if (![start, end, history].every(Number.isFinite) || end <= start || history < 0 || !Number.isInteger(parts) || parts < 2
    || [...training, ...evaluation].some(r => !Number.isFinite(r.start) || !Number.isFinite(r.end)
      || r.start < start || r.end >= end || r.end <= r.start)) throw new Error("Invalid policy-evaluation support");
  return Array.from({ length: parts }, (_, index) => {
    const from = start + (end - start) * index / parts, to = start + (end - start) * (index + 1) / parts;
    const test = evaluation.flatMap((r, i) => r.start >= from && r.start < to ? [i] : []);
    const supportStart = from - history, supportEnd = test.reduce((v, i) => Math.max(v, evaluation[i].end), to);
    const past = training.flatMap((r, i) => r.end < supportStart ? [i] : []);
    const complement = training.flatMap((r, i) => r.end < supportStart || r.start - history > supportEnd ? [i] : []);
    return { index, from, to, supportStart, supportEnd, test, past, complement };
  });
}
