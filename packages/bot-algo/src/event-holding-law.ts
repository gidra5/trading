import type { MoveAtom } from "./event-distribution.js";
import { eventHolding, type EventCosts } from "./event-log-policy.js";

/** Repeated evaluation of the SAME empirical holding law. No atoms are
 * removed. A log series is used only when its absolute remainder and that
 * of its derivative are below 1e-17; otherwise evaluate the original sum.
 * Survival constraints include every positive mass, independently of moments.
 * Arithmetic roundoff remains subject to the caller's numerical padding. */
export function prepareEventHoldingLaw(kernel: readonly MoveAtom[], costs: EventCosts, terminal: "marked" | "friction") {
  const atoms = kernel.filter(a => a.probability > 0), degree = 16;
  const f = terminal === "friction" ? (costs.feeBps + costs.slippageBps) / 10000 : 0;
  const mass = atoms.reduce((s, a) => s + a.probability, 0);
  const mean = atoms.reduce((s, a) => s + a.probability * a.return, 0);
  const pieces = [-1, 0, 1].map(side => {
    const anchor = side === 1 ? 1 : 0, moments = new Float64Array(degree + 1);
    let constant = 0, radius = 0, series = true, low = -Infinity, high = Infinity;
    const positive = (a: number, b: number) => {
      if (b > 0) low = Math.max(low, -a / b);
      else if (b < 0) high = Math.min(high, -a / b);
      else if (!(a > 0)) low = Infinity;
    };
    const rows = atoms.map(atom => {
      const long = side === 1 ? costs.longBorrowBpsPerDay / 10000 * atom.duration / 1440 : 0;
      const short = side === -1 ? costs.shortBorrowBpsPerDay / 10000 * atom.duration / 1440 : 0;
      const a = 1 + long, b = atom.return + short - long - (side === -1 ? -1 : 1) * f * (1 + atom.return);
      const worst = side === -1 ? atom.high : atom.low;
      positive(a, atom.return + short - long);
      positive(a, worst + short - long - costs.maintenanceMargin * (side === -1 ? -1 : 1) * (1 + worst));
      const base = a + b * anchor;
      if (!(base > 0)) series = false;
      else {
        const c = b / base;
        constant += atom.probability * Math.log(base); radius = Math.max(radius, Math.abs(c));
        let power = c;
        for (let j = 1; j <= degree; j++) { moments[j] += atom.probability * (j % 2 ? 1 : -1) * power / j; power *= c; }
      }
      return { p: atom.probability, a, b };
    });
    return { anchor, moments, constant, radius, series, low, high, rows };
  });
  const piece = (x: number) => pieces[x < 0 ? 0 : x <= 1 ? 1 : 2];
  return {
    mass, mean,
    survives(x: number) {
      const p = piece(x), near = 1e-10 * (1 + Math.abs(x));
      if (Math.abs(x - p.low) <= near || Math.abs(x - p.high) <= near)
        return atoms.every(a => !eventHolding(x, a, costs).liquidated);
      return x > p.low && x < p.high;
    },
    score(x: number) {
      const p = piece(x), delta = x - p.anchor, rho = Math.abs(delta) * p.radius;
      const tail = rho < 1 ? rho ** degree / (1 - rho) : Infinity;
      if (p.series && mass * tail * Math.max(rho / (degree + 1), p.radius) <= 1e-17) {
        let value = p.moments[degree], derivative = degree * p.moments[degree];
        for (let j = degree - 1; j >= 1; j--) { value = value * delta + p.moments[j]; derivative = derivative * delta + j * p.moments[j]; }
        return { value: p.constant + value * delta, derivative, series: true };
      }
      let value = 0, derivative = 0;
      for (const a of p.rows) {
        const wealth = a.a + a.b * x;
        if (!(wealth > 0)) return { value: -Infinity, derivative: NaN, series: false };
        value += a.p * Math.log(wealth); derivative += a.p * a.b / wealth;
      }
      return { value, derivative, series: false };
    },
  };
}
