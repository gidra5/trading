export interface EventAffineSegmentFacet { a: number; b: number; slope: number; }

/** Return affine supports for max_{0<=t<=1} min_i(a_i+b_i*k+slope_i*t).
 * Its dual mixes facets with total weight one. Extreme dual choices are a
 * single facet or two opposite slopes mixed to zero; the result is their
 * pointwise minimum. This preserves dependence along a balance segment. */
export function eventAffineSegmentSupports(facets: readonly EventAffineSegmentFacet[]) {
  const lines = facets.map(f => ({ a: f.a + Math.max(0, f.slope), b: f.b }));
  for (const positive of facets) if (positive.slope > 0) for (const negative of facets) if (negative.slope < 0) {
    const weight = -negative.slope / (positive.slope - negative.slope);
    lines.push({ a: weight * positive.a + (1 - weight) * negative.a,
      b: weight * positive.b + (1 - weight) * negative.b });
  }
  return lines;
}
