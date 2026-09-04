/** Uniform nearest-event averaging. Geometry never depends on the target values. */
export function eventLocalValuePredictor(design: readonly (readonly number[])[], targets: readonly (readonly number[])[]) {
  const width = design[0]?.length ?? 0, outputs = targets[0]?.length ?? 0;
  if (!design.length || !width || !outputs || targets.length !== design.length
    || design.some(r => r.length !== width || r.some(v => !Number.isFinite(v)))
    || targets.some(r => r.length !== outputs || r.some(v => !Number.isFinite(v)))) throw new Error("Invalid local value training data");
  const x = design.map(r => Float64Array.from(r)), y = targets.map(r => [...r]);
  return (query: readonly number[], counts: readonly number[]) => {
    if (query.length !== width || query.some(v => !Number.isFinite(v)) || !counts.length
      || counts.some(k => !Number.isInteger(k) || k < 1 || k > x.length) || new Set(counts).size !== counts.length)
      throw new Error("Invalid local value query/counts");
    const neighbors = x.map((row, index) => {
      let squared = 0; for (let f = 0; f < width; f++) squared += (row[f] - query[f]) ** 2;
      return { index, squared };
    }).sort((a, b) => a.squared - b.squared || a.index - b.index);
    const requested = new Set(counts), maximum = Math.max(...counts), sum = new Array<number>(outputs).fill(0);
    const result = new Map<number, { count: number; values: number[]; radius: number; indices: number[] }>();
    for (let i = 0; i < maximum; i++) {
      for (let f = 0; f < outputs; f++) sum[f] += y[neighbors[i].index][f];
      if (requested.has(i + 1)) result.set(i + 1, { count: i + 1, values: sum.map(v => v / (i + 1)),
        radius: Math.sqrt(neighbors[i].squared), indices: neighbors.slice(0, i + 1).map(r => r.index) });
    }
    return counts.map(k => result.get(k)!);
  };
}
