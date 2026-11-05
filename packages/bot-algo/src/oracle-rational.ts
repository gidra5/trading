/** Exact, canonical rational arithmetic for the Oracle execution and CRRA solver. */
export interface Rational { readonly n: bigint; readonly d: bigint }
export type RationalInput = Rational | string | number | bigint;
export const ZERO: Rational = Object.freeze({ n: 0n, d: 1n });
export const ONE: Rational = Object.freeze({ n: 1n, d: 1n });

function gcd(a: bigint, b: bigint): bigint {
  a = a < 0n ? -a : a; b = b < 0n ? -b : b;
  while (b) { const rest = a % b; a = b; b = rest; }
  return a;
}
export function fraction(n: bigint, d: bigint = 1n): Rational {
  if (!d) throw new Error("Rational denominator is zero");
  if (!n) return ZERO;
  if (d < 0n) { n = -n; d = -d; }
  const g = gcd(n, d);
  return { n: n / g, d: d / g };
}
export function rat(value: RationalInput): Rational {
  if (typeof value === "object") return value;
  if (typeof value === "bigint") return fraction(value);
  if (typeof value === "number" && !Number.isFinite(value)) throw new Error("Nonfinite rational input");
  const text = String(value).trim();
  const ratio = /^([+-]?\d+)\/([+-]?\d+)$/.exec(text);
  if (ratio) return fraction(BigInt(ratio[1]), BigInt(ratio[2]));
  const decimal = /^([+-]?)(\d+)(?:\.(\d*))?(?:[eE]([+-]?\d+))?$/.exec(text);
  if (!decimal) throw new Error(`Invalid rational: ${text}`);
  const exponent = Number(decimal[4] ?? 0) - (decimal[3]?.length ?? 0);
  if (!Number.isSafeInteger(exponent) || Math.abs(exponent) > 10000) throw new Error("Rational exponent out of range");
  const n = BigInt(`${decimal[1]}${decimal[2]}${decimal[3] ?? ""}`);
  return exponent >= 0 ? fraction(n * 10n ** BigInt(exponent)) : fraction(n, 10n ** BigInt(-exponent));
}
export function add(a: Rational, b: Rational): Rational {
  if (!a.n) return b; if (!b.n) return a;
  const g = gcd(a.d, b.d);
  return fraction(a.n * (b.d / g) + b.n * (a.d / g), a.d * (b.d / g));
}
export function neg(a: Rational): Rational { return a.n ? { n: -a.n, d: a.d } : ZERO; }
export function sub(a: Rational, b: Rational): Rational { return add(a, neg(b)); }
export function mul(a: Rational, b: Rational): Rational {
  if (!a.n || !b.n) return ZERO;
  const g = gcd(a.n, b.d), h = gcd(b.n, a.d);
  return { n: (a.n / g) * (b.n / h), d: (a.d / h) * (b.d / g) };
}
export function div(a: Rational, b: Rational): Rational {
  if (!b.n) throw new Error("Rational division by zero");
  return mul(a, fraction(b.d, b.n));
}
export function cmp(a: Rational, b: Rational): -1 | 0 | 1 {
  const delta = a.n * b.d - b.n * a.d;
  return delta < 0n ? -1 : delta > 0n ? 1 : 0;
}
export function eq(a: Rational, b: Rational): boolean { return cmp(a, b) === 0; }
export function lt(a: Rational, b: Rational): boolean { return cmp(a, b) < 0; }
export function lte(a: Rational, b: Rational): boolean { return cmp(a, b) <= 0; }
export function gt(a: Rational, b: Rational): boolean { return cmp(a, b) > 0; }
export function gte(a: Rational, b: Rational): boolean { return cmp(a, b) >= 0; }
export function abs(a: Rational): Rational { return a.n < 0n ? neg(a) : a; }
export function min(a: Rational, b: Rational): Rational { return lte(a, b) ? a : b; }
export function max(a: Rational, b: Rational): Rational { return gte(a, b) ? a : b; }
export function isZero(a: Rational): boolean { return a.n === 0n; }
export function sum(values: readonly Rational[]): Rational { return values.reduce(add, ZERO); }
export function floor(a: Rational): bigint { return a.n >= 0n ? a.n / a.d : -((-a.n + a.d - 1n) / a.d); }
export function ceil(a: Rational): bigint { return -floor(neg(a)); }
export function floorToStep(value: Rational, step: Rational): Rational {
  if (!gt(step, ZERO)) throw new Error("Step must be positive");
  return mul(fraction(floor(div(value, step))), step);
}
export function ceilToStep(value: Rational, step: Rational): Rational {
  if (!gt(step, ZERO)) throw new Error("Step must be positive");
  return mul(fraction(ceil(div(value, step))), step);
}
export function nearestStep(value: Rational, step: Rational): Rational {
  if (!gt(step, ZERO)) throw new Error("Step must be positive");
  return mul(fraction(floor(add(div(value, step), fraction(1n, 2n)))), step);
}
export function serializeRat(a: Rational): string { return a.d === 1n ? String(a.n) : `${a.n}/${a.d}`; }
/** Display/metrics only; execution and optimizer comparisons must stay rational. */
export function toNumber(a: Rational): number {
  const numeratorNumber = Number(a.n), denominatorNumber = Number(a.d);
  if (Number.isFinite(numeratorNumber) && Number.isFinite(denominatorNumber)) return numeratorNumber / denominatorNumber;
  const numerator = abs(a).n.toString(), denominator = a.d.toString();
  const digits = 16;
  return (a.n < 0n ? -1 : 1) * Number(numerator.slice(0, digits)) /
    Number(denominator.slice(0, digits)) *
    10 ** (Math.max(0, numerator.length - digits) - Math.max(0, denominator.length - digits));
}
