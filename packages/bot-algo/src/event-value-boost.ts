/** Small shared-partition, vector-valued squared-error boosting for value grids. */
export interface EventValueBoostOptions { trees: number; depth: number; minLeaf: number; rate: number; }
interface ValueNode { feature: number; cut: number; left: number; right: number; values: number[]; samples: number; }
export interface EventValueBoost { options: EventValueBoostOptions; width: number; outputs: number; trees: ValueNode[][]; }

export function predictEventValueBoost(model: EventValueBoost, features: readonly number[]): number[] {
  if (features.length !== model.width || features.some(v => !Number.isFinite(v))) throw new Error("Invalid boosted value inputs");
  const result = new Array<number>(model.outputs).fill(0);
  for (const tree of model.trees) {
    let index = 0;
    while (tree[index].feature >= 0) { const node = tree[index]; index = features[node.feature] <= node.cut ? node.left : node.right; }
    for (let j = 0; j < result.length; j++) result[j] += tree[index].values[j];
  }
  return result;
}

export function trainEventValueBoost(features: readonly number[][], targets: readonly number[][], options: EventValueBoostOptions): EventValueBoost {
  const width = features[0]?.length ?? 0, outputs = targets[0]?.length ?? 0, count = features.length;
  if (!width || !outputs || count !== targets.length || count < 2 || features.some(r => r.length !== width || r.some(v => !Number.isFinite(v)))
    || targets.some(r => r.length !== outputs || r.some(v => !Number.isFinite(v)))
    || !Number.isInteger(options.trees) || options.trees < 1 || options.trees > 256 || !Number.isInteger(options.depth) || options.depth < 1 || options.depth > 3
    || !Number.isInteger(options.minLeaf) || options.minLeaf < 2 || !(options.rate > 0 && options.rate <= 1)) throw new Error("Invalid value-boost training inputs");
  // Quantile cutpoints are fitted on training inputs only and shared across trees.
  const cuts = Array.from({ length: width }, (_, f) => {
    const sorted = features.map(r => r[f]).sort((a, b) => a - b), values: number[] = [];
    const boundaries: number[] = [];
    for (let i = options.minLeaf; i <= count - options.minLeaf; i++) if (sorted[i - 1] < sorted[i]) boundaries.push(i);
    let at = 0;
    for (let q = 0; q <= 16 && boundaries.length; q++) {
      const target = options.minLeaf + q * (count - 2 * options.minLeaf) / 16;
      while (at + 1 < boundaries.length && Math.abs(boundaries[at + 1] - target) < Math.abs(boundaries[at] - target)) at++;
      const i = boundaries[at]; values.push((sorted[i - 1] + sorted[i]) / 2);
    }
    return [...new Set(values)];
  });
  const bins = cuts.map((values, f) => features.map(r => { let b = 0; while (b < values.length && r[f] > values[b]) b++; return b; }));
  const residual = targets.map(r => [...r]), model: EventValueBoost = { options: { ...options }, width, outputs, trees: [] };
  for (let iteration = 0; iteration < options.trees; iteration++) {
    const nodes: ValueNode[] = [];
    const grow = (indices: number[], depth: number): number => {
      const total = new Float64Array(outputs);
      for (const i of indices) for (let j = 0; j < outputs; j++) total[j] += residual[i][j];
      const node: ValueNode = { feature: -1, cut: 0, left: -1, right: -1,
        values: Array.from(total, v => options.rate * v / indices.length), samples: indices.length };
      const index = nodes.push(node) - 1;
      if (depth >= options.depth || indices.length < 2 * options.minLeaf) return index;
      let bestGain = 1e-20, bestFeature = -1, bestCut = 0;
      const parent = total.reduce((s, v) => s + v * v / indices.length, 0);
      for (let f = 0; f < width; f++) {
        if (!cuts[f].length) continue;
        const counts = new Uint32Array(cuts[f].length + 1), sums = Array.from(counts, () => new Float64Array(outputs));
        for (const i of indices) {
          const b = bins[f][i]; counts[b]++;
          for (let j = 0; j < outputs; j++) sums[b][j] += residual[i][j];
        }
        const left = new Float64Array(outputs); let leftCount = 0;
        for (let b = 0; b < cuts[f].length; b++) {
          leftCount += counts[b]; for (let j = 0; j < outputs; j++) left[j] += sums[b][j];
          const rightCount = indices.length - leftCount;
          if (leftCount < options.minLeaf || rightCount < options.minLeaf) continue;
          let gain = -parent;
          for (let j = 0; j < outputs; j++) gain += left[j] ** 2 / leftCount + (total[j] - left[j]) ** 2 / rightCount;
          if (gain > bestGain) { bestGain = gain; bestFeature = f; bestCut = cuts[f][b]; }
        }
      }
      if (bestFeature < 0) return index;
      node.feature = bestFeature; node.cut = bestCut; node.values = [];
      node.left = grow(indices.filter(i => features[i][bestFeature] <= bestCut), depth + 1);
      node.right = grow(indices.filter(i => features[i][bestFeature] > bestCut), depth + 1);
      return index;
    };
    grow(Array.from({ length: count }, (_, i) => i), 0);
    model.trees.push(nodes);
    for (let i = 0; i < count; i++) {
      let index = 0;
      while (nodes[index].feature >= 0) { const node = nodes[index]; index = features[i][node.feature] <= node.cut ? node.left : node.right; }
      for (let j = 0; j < outputs; j++) residual[i][j] -= nodes[index].values[j];
    }
  }
  return model;
}
