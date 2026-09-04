/** Exact prediction weights for fixed ridge coordinates with an unpenalized intercept. */
export function eventRidgeInfluence(design: readonly (readonly number[])[], penalty: number) {
  const n = design[0]?.length ?? 0;
  if (!n || design.length < 2 || !(penalty > 0) || !Number.isFinite(penalty)
    || design.some(r => r.length !== n || r[0] !== 1 || r.some(v => !Number.isFinite(v)))) throw new Error("Invalid ridge influence design");
  const matrix = Array.from({ length: n }, () => new Float64Array(n));
  for (const row of design) for (let j = 0; j < n; j++) for (let k = 0; k <= j; k++) matrix[j][k] += row[j] * row[k] / design.length;
  for (let j = 0; j < n; j++) {
    matrix[j][j] += j ? penalty : 0;
    for (let k = 0; k <= j; k++) {
      let value = matrix[j][k]; for (let m = 0; m < k; m++) value -= matrix[j][m] * matrix[k][m];
      if (j === k) { if (!(value > 0)) throw new Error("Singular ridge influence design"); matrix[j][k] = Math.sqrt(value); }
      else matrix[j][k] = value / matrix[k][k];
    }
  }
  return (query: readonly number[]) => {
    if (query.length !== n || query[0] !== 1 || query.some(v => !Number.isFinite(v))) throw new Error("Invalid ridge influence query");
    const solution = [...query];
    for (let j = 0; j < n; j++) { for (let k = 0; k < j; k++) solution[j] -= matrix[j][k] * solution[k]; solution[j] /= matrix[j][j]; }
    for (let j = n - 1; j >= 0; j--) { for (let k = j + 1; k < n; k++) solution[j] -= matrix[k][j] * solution[k]; solution[j] /= matrix[j][j]; }
    return design.map(row => row.reduce((s, v, j) => s + v * solution[j], 0) / design.length);
  };
}
