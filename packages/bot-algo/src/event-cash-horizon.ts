import { validateEventDistribution, type EventDistribution } from "./event-distribution.js";
import { validateEventCosts, type EventCosts } from "./event-log-policy.js";

/** Sufficient cash-optimality check under the SAME forecast probability law.
 * m_0(i)=1, m_h(i)=E[(1+R)m_{h-1}(next)|i]. If every m through H lies
 * inside [1-f,1+f], P_t*m_{H-t}(state_t) is a forecast-measure martingale
 * inside the execution spread, with the actual marked price at the terminal.
 * Trading cannot increase shadow wealth; nonnegative borrowing only reduces
 * it. Jensen then bounds expected log growth from cash by zero, attained by
 * staying cash. Lot/order/leverage constraints only shrink the strategy set.
 * Strategies with positive-probability liquidation have log value -Infinity.
 * Failure to find this sufficient construction does not justify an entry.
 * This is a numerically padded finite-horizon check, not stationary proof. */
export function eventCashHorizon(model: EventDistribution, costs: EventCosts, maximumDepth: number) {
  validateEventDistribution(model); validateEventCosts(costs);
  if (!Number.isSafeInteger(maximumDepth) || maximumDepth < 1 || maximumDepth > 100000)
    throw new Error("Invalid cash horizon");
  const fee = (costs.feeBps + costs.slippageBps) / 10000, n = model.kernels.length;
  let maximumMassError = 0;
  const transition = model.kernels.map(kernel => {
    const mass = kernel.reduce((s, a) => s + a.probability, 0);
    maximumMassError = Math.max(maximumMassError, Math.abs(mass - 1));
    if (Math.abs(mass - 1) > 1e-12) throw new Error("Cash bound requires normalized probability rows");
    const row = new Array<number>(n).fill(0);
    // Interpret the stored floating-point weights as a probability law.
    for (const atom of kernel) row[atom.next] += atom.probability / mass * (1 + atom.return);
    return row;
  });
  const rows: Array<{ depth: number; minimumShadowBps: number; maximumShadowBps: number;
    numericalPad: number; insideSpread: boolean }> = [];
  let previous = new Array<number>(n).fill(1), verifiedDepth = 0;
  for (let h = 1; h <= maximumDepth; h++) {
    const next = transition.map(row => row.reduce((sum, weight, j) => sum + weight * previous[j], 0));
    const minimum = Math.min(...next), maximum = Math.max(...next);
    // Deliberately wider than summation noise for the small finite state laws.
    const numericalPad = h * 1e-10 * Math.max(1, Math.abs(minimum), Math.abs(maximum));
    const insideSpread = Number.isFinite(minimum) && Number.isFinite(maximum)
      && minimum - numericalPad >= 1 - fee && maximum + numericalPad <= 1 + fee;
    rows.push({ depth: h, minimumShadowBps: (minimum - 1) * 10000, maximumShadowBps: (maximum - 1) * 10000,
      numericalPad, insideSpread });
    if (!insideSpread) break;
    verifiedDepth = h; previous = next;
  }
  return { verifiedDepth, maximumDepth, stoppedAtFirstFailure: verifiedDepth < maximumDepth, feeBps: fee * 10000,
    maximumMassError, transition, rows,
    scope: "Flat initial account, all model leaves, all depths up to verifiedDepth, fixed forecast, decision-price execution, marked terminal; sufficient numerical check only." };
}
