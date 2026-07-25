export type BasisCorrelationMethod = "pearson" | "spearman";

export interface AssetReturnSeries {
  symbol: string;
  baseAsset: string;
  quoteAsset: string;
  returns: readonly number[];
}

export interface PortfolioBasisOptions {
  size?: number;
  minSize?: number;
  maxSize?: number;
  targetMedianRSquared?: number;
  targetP10RSquared?: number;
  correlationMethod?: BasisCorrelationMethod;
  anchorSymbol?: string;
  tolerance?: number;
  pivotPriorityScores?: readonly number[];
  residualEquivalenceBand?: number;
}

export interface PortfolioBasisEntry {
  rank: number;
  symbol: string;
  baseAsset: string;
  quoteAsset: string;
  residualRatio: number;
  maxAbsCorrelationToEarlier: number;
  meanAbsCorrelationToEarlier: number;
  closestEarlierSymbol?: string;
  closestEarlierCorrelation?: number;
}

export interface MarketBasisCoverage {
  symbol: string;
  baseAsset: string;
  rSquared: number;
  residualRatio: number;
  closestBasisSymbol: string;
  closestBasisCorrelation: number;
}

export interface BasisCoveragePoint {
  size: number;
  meanRSquared: number;
  medianRSquared: number;
  p10RSquared: number;
  minRSquared: number;
}

export interface PortfolioBasisResult {
  correlationMethod: BasisCorrelationMethod;
  pivotPriorityMode: "none" | "score-within-residual-band";
  residualEquivalenceBand: number;
  sizingMode: "fixed" | "coverage";
  sampleCount: number;
  universeSize: number;
  requestedSize: number;
  minSize: number;
  maxSize: number;
  targetMedianRSquared?: number;
  targetP10RSquared?: number;
  targetReached: boolean;
  entries: PortfolioBasisEntry[];
  coverage: MarketBasisCoverage[];
  coverageCurve: BasisCoveragePoint[];
  pairwiseMeanAbsCorrelation: number;
  pairwiseMaxAbsCorrelation: number;
  marketMeanRSquared: number;
  marketMedianRSquared: number;
  marketP10RSquared: number;
  marketMinRSquared: number;
  correlationMatrix: number[][];
}

interface NormalizedReturnMatrix {
  vectors: number[][];
  correlations: number[][];
  sampleCount: number;
}

/**
 * Select actual asset columns using column-pivoted modified Gram-Schmidt.
 *
 * Every input series is centered and normalized first, so dot products are
 * Pearson correlations. Spearman mode rank-transforms each series before the
 * same operation. At each step, the asset with the largest component
 * orthogonal to the already selected span is chosen.
 */
export function selectPortfolioBasis(
  series: readonly AssetReturnSeries[],
  options: PortfolioBasisOptions,
): PortfolioBasisResult {
  if (series.length === 0) {
    throw new Error("Portfolio basis requires at least one return series.");
  }

  const sizing = normalizeSizingOptions(options, series.length);
  const correlationMethod = options.correlationMethod ?? "pearson";
  const tolerance = positiveFinite(options.tolerance, 1e-10);
  const pivotPriorityScores = normalizePivotPriorityScores(
    options.pivotPriorityScores,
    series.length,
  );
  const residualEquivalenceBand = unitRatio(
    options.residualEquivalenceBand ?? 0,
    "Residual equivalence band",
  );
  const normalized = normalizeReturnMatrix(series, correlationMethod);
  const residuals = normalized.vectors.map((vector) => [...vector]);
  const selected: number[] = [];
  const orthonormalBasis: number[][] = [];
  const residualRatioAtSelection: number[] = [];
  const cumulativeRSquared = new Array<number>(series.length).fill(0);
  const coverageCurve: BasisCoveragePoint[] = [];
  const anchorIndex = findAnchorIndex(series, options.anchorSymbol);

  while (selected.length < sizing.maxSize) {
    const pivot = choosePivot(
      residuals,
      new Set(selected),
      selected.length === 0 ? anchorIndex : undefined,
      pivotPriorityScores,
      residualEquivalenceBand,
    );
    if (!pivot || pivot.normSquared <= tolerance * tolerance) {
      break;
    }

    const norm = Math.sqrt(pivot.normSquared);
    const q = residuals[pivot.index].map((value) => value / norm);
    selected.push(pivot.index);
    orthonormalBasis.push(q);
    residualRatioAtSelection.push(clampUnit(norm));

    for (let index = 0; index < normalized.vectors.length; index += 1) {
      const projection = dot(normalized.vectors[index], q);
      cumulativeRSquared[index] = clampUnit(
        cumulativeRSquared[index] + projection * projection,
      );
    }
    coverageCurve.push(
      coveragePoint(selected.length, cumulativeRSquared),
    );

    for (let index = 0; index < residuals.length; index += 1) {
      if (selected.includes(index)) {
        continue;
      }
      removeProjection(residuals[index], q);
      // A second pass keeps the basis stable when columns are nearly collinear.
      removeProjection(residuals[index], q);
    }

    if (
      sizing.mode === "coverage" &&
      selected.length >= sizing.minSize &&
      coverageTargetsReached(coverageCurve.at(-1)!, sizing)
    ) {
      break;
    }
  }

  const entries = selected.map((seriesIndex, rank) =>
    basisEntry(
      series,
      normalized.correlations,
      selected,
      rank,
      seriesIndex,
      residualRatioAtSelection[rank],
    ),
  );
  const coverage = series.map((asset, index) =>
    coverageEntry(
      series,
      asset,
      index,
      selected,
      normalized.vectors,
      normalized.correlations,
      orthonormalBasis,
    ),
  );
  coverage.sort(
    (a, b) =>
      a.rSquared - b.rSquared ||
      a.symbol.localeCompare(b.symbol),
  );

  const pairwise = selectedPairwiseAbsoluteCorrelations(
    selected,
    normalized.correlations,
  );
  const rSquaredValues = coverage.map((entry) => entry.rSquared);
  const finalCoverage = coverageCurve.at(-1) ?? coveragePoint(0, []);
  const targetReached =
    sizing.mode === "fixed" || coverageTargetsReached(finalCoverage, sizing);

  return {
    correlationMethod,
    pivotPriorityMode:
      pivotPriorityScores === undefined
        ? "none"
        : "score-within-residual-band",
    residualEquivalenceBand,
    sizingMode: sizing.mode,
    sampleCount: normalized.sampleCount,
    universeSize: series.length,
    requestedSize: sizing.requestedSize,
    minSize: sizing.minSize,
    maxSize: sizing.maxSize,
    targetMedianRSquared: sizing.targetMedianRSquared,
    targetP10RSquared: sizing.targetP10RSquared,
    targetReached,
    entries,
    coverage,
    coverageCurve,
    pairwiseMeanAbsCorrelation: mean(pairwise),
    pairwiseMaxAbsCorrelation: maximum(pairwise),
    marketMeanRSquared: mean(rSquaredValues),
    marketMedianRSquared: quantile(rSquaredValues, 0.5),
    marketP10RSquared: quantile(rSquaredValues, 0.1),
    marketMinRSquared: rSquaredValues.length > 0 ? Math.min(...rSquaredValues) : 0,
    correlationMatrix: normalized.correlations,
  };
}

export function pearsonCorrelation(
  left: readonly number[],
  right: readonly number[],
): number {
  if (left.length !== right.length || left.length < 2) {
    throw new Error("Correlation inputs must have equal lengths of at least two.");
  }
  return dot(normalizeVector(left, "left"), normalizeVector(right, "right"));
}

export function rankValues(values: readonly number[]): number[] {
  const ordered = values
    .map((value, index) => ({ value, index }))
    .sort((a, b) => a.value - b.value || a.index - b.index);
  const ranks = new Array<number>(values.length);

  for (let start = 0; start < ordered.length;) {
    let end = start + 1;
    while (end < ordered.length && ordered[end].value === ordered[start].value) {
      end += 1;
    }
    const averageRank = (start + end - 1) / 2 + 1;
    for (let index = start; index < end; index += 1) {
      ranks[ordered[index].index] = averageRank;
    }
    start = end;
  }

  return ranks;
}

function normalizeReturnMatrix(
  series: readonly AssetReturnSeries[],
  method: BasisCorrelationMethod,
): NormalizedReturnMatrix {
  const sampleCount = series[0].returns.length;
  if (sampleCount < 3) {
    throw new Error("Portfolio basis requires at least three aligned return samples.");
  }

  const seenSymbols = new Set<string>();
  const vectors = series.map((asset) => {
    if (seenSymbols.has(asset.symbol)) {
      throw new Error(`Duplicate return series for ${asset.symbol}.`);
    }
    seenSymbols.add(asset.symbol);
    if (asset.returns.length !== sampleCount) {
      throw new Error(
        `${asset.symbol} has ${asset.returns.length} returns; expected ${sampleCount}.`,
      );
    }

    const values = method === "spearman" ? rankValues(asset.returns) : asset.returns;
    return normalizeVector(values, asset.symbol);
  });

  const correlations = vectors.map((left, leftIndex) =>
    vectors.map((right, rightIndex) => {
      if (leftIndex === rightIndex) {
        return 1;
      }
      return clampCorrelation(dot(left, right));
    }),
  );

  return { vectors, correlations, sampleCount };
}

function normalizeVector(values: readonly number[], label: string): number[] {
  let sum = 0;
  for (const value of values) {
    if (!Number.isFinite(value)) {
      throw new Error(`${label} contains a non-finite return.`);
    }
    sum += value;
  }

  const average = sum / values.length;
  const centered = values.map((value) => value - average);
  const norm = Math.sqrt(dot(centered, centered));
  if (!Number.isFinite(norm) || norm <= 1e-14) {
    throw new Error(`${label} has zero return variance.`);
  }
  return centered.map((value) => value / norm);
}

function findAnchorIndex(
  series: readonly AssetReturnSeries[],
  anchorSymbol: string | undefined,
): number | undefined {
  if (!anchorSymbol) {
    return undefined;
  }
  const normalizedAnchor = anchorSymbol.toUpperCase();
  const index = series.findIndex(
    (asset) =>
      asset.symbol.toUpperCase() === normalizedAnchor ||
      asset.baseAsset.toUpperCase() === normalizedAnchor,
  );
  return index >= 0 ? index : undefined;
}

function choosePivot(
  residuals: readonly number[][],
  selected: ReadonlySet<number>,
  forcedIndex: number | undefined,
  priorityScores: readonly number[] | undefined,
  residualEquivalenceBand: number,
): { index: number; normSquared: number } | undefined {
  if (forcedIndex !== undefined && !selected.has(forcedIndex)) {
    return {
      index: forcedIndex,
      normSquared: dot(residuals[forcedIndex], residuals[forcedIndex]),
    };
  }

  const candidates: Array<{ index: number; normSquared: number }> = [];
  let maximumNormSquared = -1;
  for (let index = 0; index < residuals.length; index += 1) {
    if (selected.has(index)) {
      continue;
    }
    const normSquared = Math.max(0, dot(residuals[index], residuals[index]));
    candidates.push({ index, normSquared });
    maximumNormSquared = Math.max(maximumNormSquared, normSquared);
  }
  if (candidates.length === 0) {
    return undefined;
  }

  const minimumEquivalentResidual =
    maximumNormSquared * (1 - residualEquivalenceBand);
  let best: { index: number; normSquared: number } | undefined;
  for (const candidate of candidates) {
    if (candidate.normSquared + 1e-15 < minimumEquivalentResidual) {
      continue;
    }
    const candidatePriority = priorityScores?.[candidate.index] ?? 0;
    const bestPriority = best ? (priorityScores?.[best.index] ?? 0) : -Infinity;
    if (
      !best ||
      candidatePriority > bestPriority + 1e-15 ||
      (Math.abs(candidatePriority - bestPriority) <= 1e-15 &&
        (candidate.normSquared > best.normSquared + 1e-15 ||
          (Math.abs(candidate.normSquared - best.normSquared) <= 1e-15 &&
            candidate.index < best.index)))
    ) {
      best = candidate;
    }
  }
  return best;
}

function normalizePivotPriorityScores(
  values: readonly number[] | undefined,
  expectedLength: number,
): readonly number[] | undefined {
  if (values === undefined) {
    return undefined;
  }
  if (values.length !== expectedLength) {
    throw new Error(
      `pivotPriorityScores has ${values.length} values; expected ${expectedLength}.`,
    );
  }
  for (const value of values) {
    if (!Number.isFinite(value)) {
      throw new Error("pivotPriorityScores must contain only finite values.");
    }
  }
  return values;
}

function removeProjection(vector: number[], q: readonly number[]): void {
  const projection = dot(vector, q);
  for (let index = 0; index < vector.length; index += 1) {
    vector[index] -= projection * q[index];
  }
}

function basisEntry(
  series: readonly AssetReturnSeries[],
  correlations: readonly number[][],
  selected: readonly number[],
  rank: number,
  seriesIndex: number,
  residualRatio: number,
): PortfolioBasisEntry {
  const earlier = selected.slice(0, rank);
  const relationships = earlier.map((earlierIndex) => ({
    symbol: series[earlierIndex].symbol,
    correlation: correlations[seriesIndex][earlierIndex],
  }));
  relationships.sort(
    (a, b) =>
      Math.abs(b.correlation) - Math.abs(a.correlation) ||
      a.symbol.localeCompare(b.symbol),
  );
  const closest = relationships[0];

  return {
    rank: rank + 1,
    symbol: series[seriesIndex].symbol,
    baseAsset: series[seriesIndex].baseAsset,
    quoteAsset: series[seriesIndex].quoteAsset,
    residualRatio,
    maxAbsCorrelationToEarlier: closest ? Math.abs(closest.correlation) : 0,
    meanAbsCorrelationToEarlier: mean(
      relationships.map((relationship) => Math.abs(relationship.correlation)),
    ),
    closestEarlierSymbol: closest?.symbol,
    closestEarlierCorrelation: closest?.correlation,
  };
}

function coverageEntry(
  series: readonly AssetReturnSeries[],
  asset: AssetReturnSeries,
  assetIndex: number,
  selected: readonly number[],
  normalizedVectors: readonly number[][],
  correlations: readonly number[][],
  orthonormalBasis: readonly number[][],
): MarketBasisCoverage {
  let rSquared = 0;
  for (const q of orthonormalBasis) {
    const projection = dot(normalizedVectors[assetIndex], q);
    rSquared += projection * projection;
  }
  rSquared = clampUnit(rSquared);

  let closestBasisIndex = selected[0];
  for (const candidate of selected.slice(1)) {
    if (
      Math.abs(correlations[assetIndex][candidate]) >
      Math.abs(correlations[assetIndex][closestBasisIndex])
    ) {
      closestBasisIndex = candidate;
    }
  }

  return {
    symbol: asset.symbol,
    baseAsset: asset.baseAsset,
    rSquared,
    residualRatio: Math.sqrt(Math.max(0, 1 - rSquared)),
    closestBasisSymbol:
      selected.length > 0 ? series[closestBasisIndex].symbol : "",
    closestBasisCorrelation:
      selected.length > 0 ? correlations[assetIndex][closestBasisIndex] : 0,
  };
}

function selectedPairwiseAbsoluteCorrelations(
  selected: readonly number[],
  correlations: readonly number[][],
): number[] {
  const values: number[] = [];
  for (let left = 0; left < selected.length; left += 1) {
    for (let right = left + 1; right < selected.length; right += 1) {
      values.push(Math.abs(correlations[selected[left]][selected[right]]));
    }
  }
  return values;
}

function normalizeSizingOptions(
  options: PortfolioBasisOptions,
  universeSize: number,
): {
  mode: "fixed" | "coverage";
  requestedSize: number;
  minSize: number;
  maxSize: number;
  targetMedianRSquared?: number;
  targetP10RSquared?: number;
} {
  if (options.size !== undefined) {
    const size = positiveInteger(options.size, "Portfolio basis size");
    const boundedSize = Math.min(universeSize, size);
    return {
      mode: "fixed",
      requestedSize: boundedSize,
      minSize: boundedSize,
      maxSize: boundedSize,
    };
  }

  const minSize = Math.min(
    universeSize,
    positiveInteger(options.minSize ?? 8, "Minimum portfolio basis size"),
  );
  const maxSize = Math.min(
    universeSize,
    positiveInteger(options.maxSize ?? 128, "Maximum portfolio basis size"),
  );
  if (maxSize < minSize) {
    throw new Error("Maximum portfolio basis size must be at least the minimum size.");
  }
  const targetMedianRSquared = unitRatio(
    options.targetMedianRSquared ?? 0.8,
    "Median R-squared target",
  );
  const targetP10RSquared = unitRatio(
    options.targetP10RSquared ?? 0.5,
    "10th-percentile R-squared target",
  );
  return {
    mode: "coverage",
    requestedSize: maxSize,
    minSize,
    maxSize,
    targetMedianRSquared,
    targetP10RSquared,
  };
}

function coveragePoint(
  size: number,
  rSquaredValues: readonly number[],
): BasisCoveragePoint {
  return {
    size,
    meanRSquared: mean(rSquaredValues),
    medianRSquared: quantile(rSquaredValues, 0.5),
    p10RSquared: quantile(rSquaredValues, 0.1),
    minRSquared:
      rSquaredValues.length > 0 ? Math.min(...rSquaredValues) : 0,
  };
}

function coverageTargetsReached(
  point: BasisCoveragePoint,
  sizing: {
    targetMedianRSquared?: number;
    targetP10RSquared?: number;
  },
): boolean {
  return (
    (sizing.targetMedianRSquared === undefined ||
      point.medianRSquared >= sizing.targetMedianRSquared) &&
    (sizing.targetP10RSquared === undefined ||
      point.p10RSquared >= sizing.targetP10RSquared)
  );
}

function positiveInteger(value: number, label: string): number {
  if (!Number.isFinite(value) || value < 1) {
    throw new Error(`${label} must be a positive integer.`);
  }
  return Math.floor(value);
}

function unitRatio(value: number, label: string): number {
  if (!Number.isFinite(value) || value < 0 || value > 1) {
    throw new Error(`${label} must be between zero and one.`);
  }
  return value;
}

function positiveFinite(value: number | undefined, fallback: number): number {
  return value !== undefined && Number.isFinite(value) && value > 0 ? value : fallback;
}

function dot(left: readonly number[], right: readonly number[]): number {
  let result = 0;
  for (let index = 0; index < left.length; index += 1) {
    result += left[index] * right[index];
  }
  return result;
}

function clampCorrelation(value: number): number {
  return Math.max(-1, Math.min(1, value));
}

function clampUnit(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function mean(values: readonly number[]): number {
  return values.length > 0
    ? values.reduce((total, value) => total + value, 0) / values.length
    : 0;
}

function maximum(values: readonly number[]): number {
  let result = 0;
  for (const value of values) {
    result = Math.max(result, value);
  }
  return result;
}

function quantile(values: readonly number[], probability: number): number {
  if (values.length === 0) {
    return 0;
  }
  const sorted = [...values].sort((a, b) => a - b);
  const position = (sorted.length - 1) * probability;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) {
    return sorted[lower];
  }
  const weight = position - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}
