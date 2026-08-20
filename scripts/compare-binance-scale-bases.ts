import fs from "node:fs/promises";
import path from "node:path";
import { cappedProportionalWeights } from "../apps/server/src/index-weighting.js";

const REPORT_VERSION = 2;
const SAMPLE_COUNT = 360;
const MAXIMUM_CONSTITUENT_WEIGHT = 0.05;
const TOLERANCE = 1e-10;
const SCALE_DEFINITIONS = [
  { id: "1d", label: "360d × 1d" },
  { id: "4h", label: "60d × 4h" },
  { id: "1h", label: "15d × 1h" },
  { id: "15m", label: "3.75d × 15m" },
  { id: "1m", label: "6h × 1m" },
] as const;

interface SingleScaleReport {
  version: number;
  generatedAt: string;
  parameters: {
    products: string;
    quoteAsset: string;
    interval: string;
    sampleCount: number;
    returnStartTime: string;
    returnEndTime: string;
    correlationMethod: string;
    residualEquivalenceBand: number;
  };
  universe: {
    eligibleSymbols: number;
    discoveryWarnings: string[];
  };
  basis: {
    entries: Array<{
      rank: number;
      symbol: string;
      baseAsset: string;
      residualRatio: number;
    }>;
    correlationMatrix: number[][];
    pairwiseMeanAbsCorrelation: number;
    pairwiseMaxAbsCorrelation: number;
    marketMeanRSquared: number;
    marketMedianRSquared: number;
    marketP10RSquared: number;
    marketMinRSquared: number;
  };
  assets: Array<{
    symbol: string;
    baseAsset: string;
    meanAbsoluteReturn: number;
    medianDailyQuoteVolume: number;
  }>;
  matrixSymbols: string[];
}

interface LoadedScale {
  id: string;
  label: string;
  file: string;
  report: SingleScaleReport;
  assets: string[];
  assetIndex: Map<string, number>;
  meanAbsoluteReturns: Float64Array;
  medianDailyQuoteVolumes: Float64Array;
  selected: number[];
}

interface GramState {
  coordinates: number[][];
  cumulativeProjection: Float64Array;
}

interface SelectionDiagnostics {
  size: number;
  meanAbsoluteReturn: number;
  medianAbsoluteReturn: number;
  meanResidualAtSelection: number;
  pairwiseMeanAbsCorrelation: number;
  pairwiseMaxAbsCorrelation: number;
  coverage: CoverageStats;
}

interface CoverageStats {
  mean: number;
  median: number;
  p10: number;
  min: number;
}

interface ScaleComparison {
  id: string;
  label: string;
  file: string;
  window: {
    start: string;
    end: string;
    samples: number;
  };
  eligibleAssets: number;
  basisSize: number;
  residualEquivalenceBand: number;
  amplitudePriority: SelectionDiagnostics;
  pureOrthogonality: SelectionDiagnostics;
  weighting: {
    sizeMeasure: "median-daily-quote-volume";
    maximumConstituentWeight: number;
    effectiveConstituents: number;
    top10Weight: number;
  };
  versusPureOrthogonality: {
    commonAssets: number;
    overlapOfBasis: number;
    jaccard: number;
    meanAbsoluteReturnUplift: number;
    medianAbsoluteReturnUplift: number;
    meanResidualChange: number;
    marketMedianRSquaredChange: number;
    marketP10RSquaredChange: number;
  };
  selectedAssets: Array<{
    rank: number;
    asset: string;
    meanAbsoluteReturn: number;
    medianDailyQuoteVolume: number;
    liquidityRank: number;
    uncappedWeight: number;
    cappedWeight: number;
  }>;
  pureOrthogonalityAssets: string[];
}

const args = process.argv.slice(2);
if (args.includes("--help")) {
  printHelp();
  process.exit(0);
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.stack ?? error.message : error);
  process.exitCode = 1;
});

async function main(): Promise<void> {
  const reportDir = path.resolve(
    optionValue(args, "--report-dir") ?? "docs/portfolio",
  );
  const scales = await Promise.all(
    SCALE_DEFINITIONS.map((definition) => loadScale(definition, reportDir)),
  );
  validateScales(scales);

  const comparisons = scales.map((scale) => {
    const baseline = selectFromGram(
      scale,
      scale.selected.length,
      0,
      undefined,
      "BTC",
    );
    const reconstructed = selectFromGram(
      scale,
      scale.selected.length,
      scale.report.parameters.residualEquivalenceBand,
      scale.meanAbsoluteReturns,
      "BTC",
    );
    const reconstructionMatches =
      reconstructed.indices.length === scale.selected.length &&
      reconstructed.indices.every(
        (index, rank) => index === scale.selected[rank],
      );
    if (!reconstructionMatches) {
      throw new Error(
        `${scale.file}: Gram reconstruction does not match the recorded selection.`,
      );
    }

    const amplitude = diagnostics(
      scale,
      scale.selected,
      reconstructed.residuals,
    );
    const pureOrthogonality = diagnostics(
      scale,
      baseline.indices,
      baseline.residuals,
    );
    const amplitudeAssets = new Set(
      scale.selected.map((index) => scale.assets[index]),
    );
    const overlap = baseline.indices.filter((index) =>
      amplitudeAssets.has(scale.assets[index]),
    ).length;
    const selectedLiquidity = scale.selected.map(
      (index) => scale.medianDailyQuoteVolumes[index],
    );
    const uncappedWeights = proportionalWeights(selectedLiquidity);
    const cappedWeights = cappedProportionalWeights(
      selectedLiquidity,
      MAXIMUM_CONSTITUENT_WEIGHT,
    );
    const liquidityRank = new Map(
      scale.selected
        .map((index, selectedIndex) => ({
          index,
          selectedIndex,
          asset: scale.assets[index],
          liquidity: scale.medianDailyQuoteVolumes[index],
        }))
        .sort(
          (left, right) =>
            right.liquidity - left.liquidity ||
            left.asset.localeCompare(right.asset),
        )
        .map((entry, rank) => [entry.selectedIndex, rank + 1]),
    );

    return {
      id: scale.id,
      label: scale.label,
      file: path.relative(process.cwd(), scale.file),
      window: {
        start: scale.report.parameters.returnStartTime,
        end: scale.report.parameters.returnEndTime,
        samples: scale.report.parameters.sampleCount,
      },
      eligibleAssets: scale.assets.length,
      basisSize: scale.selected.length,
      residualEquivalenceBand:
        scale.report.parameters.residualEquivalenceBand,
      amplitudePriority: amplitude,
      pureOrthogonality,
      weighting: {
        sizeMeasure: "median-daily-quote-volume" as const,
        maximumConstituentWeight: MAXIMUM_CONSTITUENT_WEIGHT,
        effectiveConstituents: effectiveConstituents(cappedWeights),
        top10Weight: topWeight(cappedWeights, 10),
      },
      versusPureOrthogonality: {
        commonAssets: overlap,
        overlapOfBasis: overlap / scale.selected.length,
        jaccard:
          overlap / (scale.selected.length + baseline.indices.length - overlap),
        meanAbsoluteReturnUplift:
          ratioChange(
            amplitude.meanAbsoluteReturn,
            pureOrthogonality.meanAbsoluteReturn,
          ),
        medianAbsoluteReturnUplift:
          ratioChange(
            amplitude.medianAbsoluteReturn,
            pureOrthogonality.medianAbsoluteReturn,
          ),
        meanResidualChange:
          amplitude.meanResidualAtSelection -
          pureOrthogonality.meanResidualAtSelection,
        marketMedianRSquaredChange:
          amplitude.coverage.median - pureOrthogonality.coverage.median,
        marketP10RSquaredChange:
          amplitude.coverage.p10 - pureOrthogonality.coverage.p10,
      },
      selectedAssets: scale.selected.map((index, rank) => ({
        rank: rank + 1,
        asset: scale.assets[index],
        meanAbsoluteReturn: scale.meanAbsoluteReturns[index],
        medianDailyQuoteVolume: scale.medianDailyQuoteVolumes[index],
        liquidityRank: liquidityRank.get(rank)!,
        uncappedWeight: uncappedWeights[rank],
        cappedWeight: cappedWeights[rank],
      })),
      pureOrthogonalityAssets: baseline.indices.map(
        (index) => scale.assets[index],
      ),
    };
  });

  const selectedSets = scales.map(
    (scale) => new Set(scale.selected.map((index) => scale.assets[index])),
  );
  const union = [
    ...new Set(
      scales.flatMap((scale) =>
        scale.selected.map((index) => scale.assets[index]),
      ),
    ),
  ];
  const recurrenceEntries = union
    .map((asset) => {
      const scaleSelections = scales.flatMap((scale, scaleIndex) => {
        const rank = scale.selected.findIndex(
          (index) => scale.assets[index] === asset,
        );
        return rank < 0
          ? []
          : [
              {
                id: scale.id,
                rank: rank + 1,
                basisSize: scale.selected.length,
                meanAbsoluteReturn:
                  scale.meanAbsoluteReturns[scale.assetIndex.get(asset)!],
              },
            ];
      });
      return {
        asset,
        selectedScaleCount: scaleSelections.length,
        eligibleScaleCount: scales.filter((scale) =>
          scale.assetIndex.has(asset),
        ).length,
        scales: scaleSelections,
        meanNormalizedRank:
          scaleSelections.reduce(
            (total, selection) =>
              total + selection.rank / selection.basisSize,
            0,
          ) / scaleSelections.length,
      };
    })
    .sort(
      (left, right) =>
        right.selectedScaleCount - left.selectedScaleCount ||
        left.meanNormalizedRank - right.meanNormalizedRank ||
        left.asset.localeCompare(right.asset),
    );
  const majorityAssets = recurrenceEntries
    .filter((entry) => entry.selectedScaleCount >= 3)
    .map((entry) => entry.asset);
  const allFiveAssets = recurrenceEntries
    .filter((entry) => entry.selectedScaleCount === scales.length)
    .map((entry) => entry.asset);

  const overlap = scales.flatMap((left, leftIndex) =>
    scales.slice(leftIndex + 1).map((right, offset) => {
      const rightIndex = leftIndex + offset + 1;
      const intersection = [...selectedSets[leftIndex]].filter((asset) =>
        selectedSets[rightIndex].has(asset),
      ).length;
      const unionSize = new Set([
        ...selectedSets[leftIndex],
        ...selectedSets[rightIndex],
      ]).size;
      return {
        left: left.id,
        right: right.id,
        commonAssets: intersection,
        jaccard: intersection / unionSize,
        overlapOfSmallerBasis:
          intersection /
          Math.min(
            selectedSets[leftIndex].size,
            selectedSets[rightIndex].size,
          ),
      };
    }),
  );
  const recurrenceCounts = Object.fromEntries(
    Array.from({ length: scales.length }, (_, offset) => {
      const count = scales.length - offset;
      return [
        count,
        recurrenceEntries.filter(
          (entry) => entry.selectedScaleCount === count,
        ).length,
      ];
    }),
  );
  const majorityCoverage = Object.fromEntries(
    scales.map((scale) => [
      scale.id,
      coverageForAssets(scale, majorityAssets),
    ]),
  );
  const scaleSleeveWeight = 1 / comparisons.length;
  const aggregateByAsset = new Map<
    string,
    { weight: number; scaleWeights: Record<string, number> }
  >();
  for (const comparison of comparisons) {
    for (const asset of comparison.selectedAssets) {
      const aggregate = aggregateByAsset.get(asset.asset) ?? {
        weight: 0,
        scaleWeights: {},
      };
      aggregate.scaleWeights[comparison.id] = asset.cappedWeight;
      aggregate.weight += scaleSleeveWeight * asset.cappedWeight;
      aggregateByAsset.set(asset.asset, aggregate);
    }
  }
  const aggregateConstituents = [...aggregateByAsset.entries()]
    .map(([asset, value]) => ({ asset, ...value }))
    .sort(
      (left, right) =>
        right.weight - left.weight ||
        left.asset.localeCompare(right.asset),
    )
    .map((entry, rank) => ({ rank: rank + 1, ...entry }));
  const aggregateWeights = aggregateConstituents.map(
    (entry) => entry.weight,
  );
  const aggregateWeightTotal = aggregateWeights.reduce(
    (total, weight) => total + weight,
    0,
  );
  if (Math.abs(aggregateWeightTotal - 1) > 1e-10) {
    throw new Error(
      `Multiscale aggregate weights sum to ${aggregateWeightTotal}.`,
    );
  }

  const report = {
    version: REPORT_VERSION,
    generatedAt: new Date().toISOString(),
    endDate: scales[0].report.parameters.returnEndTime.slice(0, 10),
    methodology: {
      name: "independent per-scale pivoted QR comparison",
      sampleCountPerScale: SAMPLE_COUNT,
      residualEquivalenceBand:
        scales[0].report.parameters.residualEquivalenceBand,
      amplitudeRule:
        "within each scale only, prefer the largest mean absolute candle return among pivots within the residual-equivalence band",
      crossScaleRule:
        "compare membership recurrence only; candle-return amplitudes are never averaged across intervals",
      majorityThreshold: 3,
    },
    indexWeighting: {
      name: "equal-weight multiscale index of capped liquidity-weighted scale sleeves",
      marketCapAvailable: false,
      sizeProxy: "median daily quote volume within each scale window",
      maximumConstituentWeightPerScale: MAXIMUM_CONSTITUENT_WEIGHT,
      scaleSleeveWeight,
      aggregateConstituentCount: aggregateConstituents.length,
      aggregateEffectiveConstituents:
        effectiveConstituents(aggregateWeights),
      aggregateTop10Weight: topWeight(aggregateWeights, 10),
      constituents: aggregateConstituents,
    },
    scales: comparisons,
    pairwiseBasisOverlap: overlap,
    recurrence: {
      unionSize: recurrenceEntries.length,
      exactScaleCounts: recurrenceCounts,
      allFiveAssets,
      majorityAssets,
      majorityCoverage,
      entries: recurrenceEntries,
    },
  };
  const output = await writeReport(report, reportDir);

  console.log(
    `Independent 360-return bases: ${comparisons
      .map((scale) => `${scale.id} k${scale.basisSize}`)
      .join(", ")}`,
  );
  console.log(
    `Selected at all five scales (${allFiveAssets.length}): ${allFiveAssets.join(", ")}`,
  );
  console.log(
    `Selected at three or more scales: ${majorityAssets.length}; union: ${recurrenceEntries.length}`,
  );
  console.log(
    `Top aggregate weights: ${aggregateConstituents
      .slice(0, 10)
      .map((entry) => `${entry.asset} ${formatPercent(entry.weight)}`)
      .join(", ")}`,
  );
  for (const scale of comparisons) {
    console.log(
      `${scale.id}: median |return| uplift ${formatSignedPercent(
        scale.versusPureOrthogonality.medianAbsoluteReturnUplift,
      )}; basis overlap ${formatPercent(
        scale.versusPureOrthogonality.overlapOfBasis,
      )}`,
    );
  }
  console.log(`JSON: ${output.json}`);
  console.log(`Markdown: ${output.markdown}`);
  console.log(`Weights CSV: ${output.weights}`);
}

async function loadScale(
  definition: (typeof SCALE_DEFINITIONS)[number],
  reportDir: string,
): Promise<LoadedScale> {
  const runDir = path.join(reportDir, "runs");
  const expression = new RegExp(
    `^\\d{4}-\\d{2}-\\d{2}-all-usdt-${escapeRegExp(
      definition.id,
    )}-${SAMPLE_COUNT}c-q4(?:-amp[^-]+|-pure)?-pearson-k\\d+\\.json$`,
  );
  const candidates = (await fs.readdir(runDir))
    .filter((file) => expression.test(file))
    .map((file) => path.join(runDir, file));
  const reports = await Promise.all(
    candidates.map(async (file) => ({
      file,
      report: JSON.parse(
        await fs.readFile(file, "utf8"),
      ) as SingleScaleReport,
    })),
  );
  const compatible = reports
    .filter(
      ({ report }) =>
        report.version === 4 &&
        report.parameters.sampleCount === SAMPLE_COUNT &&
        report.parameters.interval === definition.id &&
        report.universe.discoveryWarnings.length === 0,
    )
    .sort((left, right) =>
      right.report.generatedAt.localeCompare(left.report.generatedAt),
    );
  if (compatible.length === 0) {
    throw new Error(
      `No complete q4 ${SAMPLE_COUNT}-candle report found for ${definition.id}.`,
    );
  }
  const { file, report } = compatible[0];
  const assetBySymbol = new Map(
    report.assets.map((asset) => [asset.symbol, asset]),
  );
  const assets = report.matrixSymbols.map((symbol) => {
    const asset = assetBySymbol.get(symbol);
    if (!asset) {
      throw new Error(`${file}: missing metadata for ${symbol}.`);
    }
    return asset.baseAsset;
  });
  const assetIndex = new Map(
    assets.map((asset, index) => [asset, index]),
  );
  if (assetIndex.size !== assets.length) {
    throw new Error(`${file}: duplicate economic assets in return matrix.`);
  }
  const selected = report.basis.entries.map((entry) => {
    const index = report.matrixSymbols.indexOf(entry.symbol);
    if (index < 0) {
      throw new Error(`${file}: selected symbol ${entry.symbol} is absent.`);
    }
    return index;
  });
  return {
    ...definition,
    file,
    report,
    assets,
    assetIndex,
    meanAbsoluteReturns: Float64Array.from(
      report.matrixSymbols.map(
        (symbol) => assetBySymbol.get(symbol)!.meanAbsoluteReturn,
      ),
    ),
    medianDailyQuoteVolumes: Float64Array.from(
      report.matrixSymbols.map(
        (symbol) => assetBySymbol.get(symbol)!.medianDailyQuoteVolume,
      ),
    ),
    selected,
  };
}

function validateScales(scales: readonly LoadedScale[]): void {
  const values = (read: (scale: LoadedScale) => string | number) =>
    new Set(scales.map(read));
  if (
    values((scale) => scale.report.parameters.products).size !== 1 ||
    values((scale) => scale.report.parameters.quoteAsset).size !== 1 ||
    values((scale) => scale.report.parameters.correlationMethod).size !== 1 ||
    values(
      (scale) => scale.report.parameters.residualEquivalenceBand,
    ).size !== 1 ||
    values((scale) =>
      scale.report.parameters.returnEndTime.slice(0, 10),
    ).size !== 1
  ) {
    throw new Error(
      "Scale reports must share product universe, quote, method, and end day.",
    );
  }
}

function selectFromGram(
  scale: LoadedScale,
  size: number,
  residualEquivalenceBand: number,
  priorityScores: Float64Array | undefined,
  anchorAsset: string | undefined,
): { indices: number[]; residuals: number[] } {
  const state = createGramState(scale.assets.length);
  const selected: number[] = [];
  const selectedSet = new Set<number>();
  const residuals: number[] = [];
  const anchorIndex =
    anchorAsset === undefined ? undefined : scale.assetIndex.get(anchorAsset);

  while (selected.length < size) {
    let pivot: number | undefined;
    if (
      selected.length === 0 &&
      anchorIndex !== undefined &&
      !selectedSet.has(anchorIndex)
    ) {
      pivot = anchorIndex;
    } else {
      let maximumResidual = -1;
      for (let index = 0; index < scale.assets.length; index += 1) {
        if (!selectedSet.has(index)) {
          maximumResidual = Math.max(
            maximumResidual,
            1 - state.cumulativeProjection[index],
          );
        }
      }
      const minimumEquivalent =
        maximumResidual * (1 - residualEquivalenceBand);
      let bestPriority = -Infinity;
      let bestResidual = -1;
      for (let index = 0; index < scale.assets.length; index += 1) {
        if (selectedSet.has(index)) {
          continue;
        }
        const residual = 1 - state.cumulativeProjection[index];
        if (residual + 1e-15 < minimumEquivalent) {
          continue;
        }
        const priority = priorityScores?.[index] ?? 0;
        if (
          pivot === undefined ||
          priority > bestPriority + 1e-15 ||
          (Math.abs(priority - bestPriority) <= 1e-15 &&
            (residual > bestResidual + 1e-15 ||
              (Math.abs(residual - bestResidual) <= 1e-15 &&
                index < pivot)))
        ) {
          pivot = index;
          bestPriority = priority;
          bestResidual = residual;
        }
      }
    }
    if (pivot === undefined) {
      break;
    }
    const residual = 1 - state.cumulativeProjection[pivot];
    if (residual <= TOLERANCE) {
      break;
    }
    updateGramState(
      state,
      scale.report.basis.correlationMatrix,
      pivot,
    );
    selected.push(pivot);
    selectedSet.add(pivot);
    residuals.push(Math.sqrt(Math.max(0, residual)));
  }
  return { indices: selected, residuals };
}

function createGramState(size: number): GramState {
  return {
    coordinates: Array.from({ length: size }, () => []),
    cumulativeProjection: new Float64Array(size),
  };
}

function updateGramState(
  state: GramState,
  gram: readonly number[][],
  pivot: number,
): void {
  const residual = 1 - state.cumulativeProjection[pivot];
  const denominator = Math.sqrt(residual);
  const pivotCoordinates = [...state.coordinates[pivot]];
  for (let index = 0; index < state.coordinates.length; index += 1) {
    let residualInnerProduct = gram[index][pivot];
    for (
      let coordinate = 0;
      coordinate < pivotCoordinates.length;
      coordinate += 1
    ) {
      residualInnerProduct -=
        state.coordinates[index][coordinate] *
        pivotCoordinates[coordinate];
    }
    const projection = residualInnerProduct / denominator;
    if (!Number.isFinite(projection)) {
      throw new Error("Non-finite Gram projection.");
    }
    state.coordinates[index].push(projection);
    state.cumulativeProjection[index] = Math.min(
      1,
      Math.max(
        0,
        state.cumulativeProjection[index] + projection * projection,
      ),
    );
  }
}

function diagnostics(
  scale: LoadedScale,
  selected: readonly number[],
  residuals: readonly number[],
): SelectionDiagnostics {
  const amplitudes = selected.map(
    (index) => scale.meanAbsoluteReturns[index],
  );
  const pairwise = pairwiseDiagnostics(
    selected,
    scale.report.basis.correlationMatrix,
  );
  return {
    size: selected.length,
    meanAbsoluteReturn: mean(amplitudes),
    medianAbsoluteReturn: quantile(amplitudes, 0.5),
    meanResidualAtSelection: mean(residuals),
    pairwiseMeanAbsCorrelation: pairwise.mean,
    pairwiseMaxAbsCorrelation: pairwise.max,
    coverage: coverageForIndices(scale, selected),
  };
}

function coverageForAssets(
  scale: LoadedScale,
  assets: readonly string[],
): {
  requestedAssets: number;
  availableAssets: number;
  independentAssets: number;
  coverage: CoverageStats;
} {
  const indices = assets.flatMap((asset) => {
    const index = scale.assetIndex.get(asset);
    return index === undefined ? [] : [index];
  });
  const result = coverageForIndices(scale, indices);
  return {
    requestedAssets: assets.length,
    availableAssets: indices.length,
    independentAssets: result.independentAssets,
    coverage: result,
  };
}

function coverageForIndices(
  scale: LoadedScale,
  selected: readonly number[],
): CoverageStats & { independentAssets: number } {
  const state = createGramState(scale.assets.length);
  let independentAssets = 0;
  for (const pivot of selected) {
    if (1 - state.cumulativeProjection[pivot] <= TOLERANCE) {
      continue;
    }
    updateGramState(
      state,
      scale.report.basis.correlationMatrix,
      pivot,
    );
    independentAssets += 1;
  }
  const coverage = [...state.cumulativeProjection];
  return {
    mean: mean(coverage),
    median: quantile(coverage, 0.5),
    p10: quantile(coverage, 0.1),
    min: minimum(coverage),
    independentAssets,
  };
}

function pairwiseDiagnostics(
  selected: readonly number[],
  correlations: readonly number[][],
): { mean: number; max: number } {
  let total = 0;
  let maximum = 0;
  let count = 0;
  for (let left = 0; left < selected.length; left += 1) {
    for (let right = left + 1; right < selected.length; right += 1) {
      const value = Math.abs(
        correlations[selected[left]][selected[right]],
      );
      total += value;
      maximum = Math.max(maximum, value);
      count += 1;
    }
  }
  return { mean: count > 0 ? total / count : 0, max: maximum };
}

async function writeReport(
  report: Record<string, unknown> & {
    endDate: string;
    indexWeighting: {
      name: string;
      marketCapAvailable: boolean;
      sizeProxy: string;
      maximumConstituentWeightPerScale: number;
      scaleSleeveWeight: number;
      aggregateConstituentCount: number;
      aggregateEffectiveConstituents: number;
      aggregateTop10Weight: number;
      constituents: Array<{
        rank: number;
        asset: string;
        weight: number;
        scaleWeights: Record<string, number>;
      }>;
    };
    scales: ScaleComparison[];
    pairwiseBasisOverlap: Array<{
      left: string;
      right: string;
      commonAssets: number;
      jaccard: number;
      overlapOfSmallerBasis: number;
    }>;
    recurrence: {
      unionSize: number;
      exactScaleCounts: Record<string, number>;
      allFiveAssets: string[];
      majorityAssets: string[];
      majorityCoverage: Record<
        string,
        {
          requestedAssets: number;
          availableAssets: number;
          independentAssets: number;
          coverage: CoverageStats;
        }
      >;
      entries: Array<{
        asset: string;
        selectedScaleCount: number;
        eligibleScaleCount: number;
        scales: Array<{
          id: string;
          rank: number;
          basisSize: number;
          meanAbsoluteReturn: number;
        }>;
      }>;
    };
  },
  reportDir: string,
): Promise<{ json: string; markdown: string; weights: string }> {
  const root = path.join(reportDir, "scale-comparison");
  const stem =
    `${report.endDate}-5scale-360c-q4-` +
    `${residualBandTag(
      report.scales[0].residualEquivalenceBand,
    )}-liqcap5pct`;
  const json = path.join(root, `${stem}.json`);
  const markdown = path.join(root, `${stem}.md`);
  const weights = path.join(root, `${stem}.weights.csv`);
  const jsonContent = `${JSON.stringify(report, null, 2)}\n`;
  const markdownContent = renderMarkdown(report);
  const weightsContent = renderWeightsCsv(report);
  await Promise.all([
    writeTextAtomic(json, jsonContent),
    writeTextAtomic(markdown, markdownContent),
    writeTextAtomic(weights, weightsContent),
    writeTextAtomic(path.join(root, "latest.json"), jsonContent),
    writeTextAtomic(path.join(root, "latest.md"), markdownContent),
    writeTextAtomic(
      path.join(root, "latest.weights.csv"),
      weightsContent,
    ),
  ]);
  return { json, markdown, weights };
}

function renderMarkdown(
  report: Parameters<typeof writeReport>[0],
): string {
  const lines = [
    "# Binance independent-scale basis comparison",
    "",
    `Generated ${String(report.generatedAt)}.`,
    "",
    "## Method",
    "",
    "- Five bases are selected independently from exactly 360 returns each.",
    `- At each pivot, orthogonality defines the candidate pool. Within ${formatPercent(
      report.scales[0].residualEquivalenceBand,
    )} of the maximum unexplained variance, the largest mean absolute return at that same interval wins.`,
    "- Return amplitudes are never averaged across intervals. Cross-scale comparison uses basis membership recurrence only.",
    "- BTC remains the explicit first anchor at every scale.",
    "- Index ranking/weighting is a separate layer and does not alter basis membership.",
    "",
    "## Per-scale results",
    "",
    "| Scale | Eligible | Basis | Median abs return | Versus pure QR | Common assets | Median R² | P10 R² | Mean abs corr |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
  ];
  for (const scale of report.scales) {
    lines.push(
      `| ${scale.label} | ${scale.eligibleAssets} | ${scale.basisSize} | ` +
        `${formatBasisPoints(
          scale.amplitudePriority.medianAbsoluteReturn,
        )} | ${formatSignedPercent(
          scale.versusPureOrthogonality.medianAbsoluteReturnUplift,
        )} | ${scale.versusPureOrthogonality.commonAssets} ` +
        `(${formatPercent(
          scale.versusPureOrthogonality.overlapOfBasis,
        )}) | ${formatPercent(
          scale.amplitudePriority.coverage.median,
        )} | ${formatPercent(
          scale.amplitudePriority.coverage.p10,
        )} | ${formatDecimal(
          scale.amplitudePriority.pairwiseMeanAbsCorrelation,
        )} |`,
    );
  }
  lines.push(
    "",
    "The uplift column compares the return-prioritized basis with pure maximum-residual QR at the same basis size. A positive value means the secondary rule selected a higher-movement set. Coverage changes remain available in the JSON report.",
    "",
    "## S&P-like ranking and weighting",
    "",
    "A true float-adjusted market-cap weight is unavailable from Binance market data because circulating/free-float supply is not provided consistently across the cross-product universe. This report therefore uses median quote volume as an explicit investable-size proxy.",
    "",
    `- Each scale sleeve is proportional to its constituents' median quote volume and capped at ${formatPercent(
      report.indexWeighting.maximumConstituentWeightPerScale,
    )} per asset.`,
    `- Each of the five independently constructed scale sleeves contributes ${formatPercent(
      report.indexWeighting.scaleSleeveWeight,
    )} to the final multiscale index.`,
    `- Final constituents: ${report.indexWeighting.aggregateConstituentCount}`,
    `- Effective constituent count: ${report.indexWeighting.aggregateEffectiveConstituents.toFixed(
      1,
    )}`,
    `- Aggregate top-10 weight: ${formatPercent(
      report.indexWeighting.aggregateTop10Weight,
    )}`,
    "",
    "### Top aggregate weights",
    "",
    "| # | Asset | Final weight | Per-scale sleeve weights |",
    "| -: | --- | ---: | --- |",
  );
  for (const constituent of report.indexWeighting.constituents.slice(0, 50)) {
    lines.push(
      `| ${constituent.rank} | ${constituent.asset} | ` +
        `${formatPercent(constituent.weight)} | ` +
        `${report.scales
          .map(
            (scale) =>
              `${scale.id} ${formatPercent(
                constituent.scaleWeights[scale.id] ?? 0,
              )}`,
          )
          .join(", ")} |`,
    );
  }
  lines.push(
    "",
    "### Per-scale concentration",
    "",
    "| Scale | Effective constituents | Top-10 weight | Largest constituents |",
    "| --- | ---: | ---: | --- |",
  );
  for (const scale of report.scales) {
    const largest = [...scale.selectedAssets]
      .sort(
        (left, right) =>
          right.cappedWeight - left.cappedWeight ||
          left.liquidityRank - right.liquidityRank,
      )
      .slice(0, 5);
    lines.push(
      `| ${scale.label} | ${scale.weighting.effectiveConstituents.toFixed(
        1,
      )} | ${formatPercent(scale.weighting.top10Weight)} | ` +
        `${largest
          .map(
            (asset) =>
              `${asset.asset} ${formatPercent(asset.cappedWeight)}`,
          )
          .join(", ")} |`,
    );
  }
  lines.push(
    "",
    "## Cross-scale recurrence",
    "",
    `- Unique assets selected by at least one scale: ${report.recurrence.unionSize}`,
    `- Selected at all five scales: ${report.recurrence.allFiveAssets.length}`,
    `- Selected at three or more scales: ${report.recurrence.majorityAssets.length}`,
    `- Exact recurrence counts: ${Object.entries(
      report.recurrence.exactScaleCounts,
    )
      .map(([count, assets]) => `${count} scales: ${assets}`)
      .join(", ")}`,
    "",
    `All-five core: ${report.recurrence.allFiveAssets.join(", ")}`,
    "",
    "The all-five intersection is a stability core, not a sufficient spanning basis. The majority set is also diagnostic: it is the simple membership consensus requested here, not a new jointly optimized basis.",
    "",
    "### Coverage of the 3-of-5 membership consensus",
    "",
    "| Scale | Available consensus assets | Median R² | P10 R² | Minimum R² |",
    "| --- | ---: | ---: | ---: | ---: |",
  );
  for (const scale of report.scales) {
    const coverage = report.recurrence.majorityCoverage[scale.id];
    lines.push(
      `| ${scale.label} | ${coverage.availableAssets} | ` +
        `${formatPercent(coverage.coverage.median)} | ` +
        `${formatPercent(coverage.coverage.p10)} | ` +
        `${formatPercent(coverage.coverage.min)} |`,
    );
  }
  lines.push(
    "",
    "### Pairwise basis overlap",
    "",
    "| Left | Right | Common assets | Jaccard | Share of smaller basis |",
    "| --- | --- | ---: | ---: | ---: |",
  );
  for (const pair of report.pairwiseBasisOverlap) {
    lines.push(
      `| ${pair.left} | ${pair.right} | ${pair.commonAssets} | ` +
        `${formatPercent(pair.jaccard)} | ` +
        `${formatPercent(pair.overlapOfSmallerBasis)} |`,
    );
  }
  lines.push(
    "",
    "### Assets selected at three or more scales",
    "",
    "| Asset | Scales | Per-scale rank and mean abs return |",
    "| --- | ---: | --- |",
  );
  for (const entry of report.recurrence.entries.filter(
    (entry) => entry.selectedScaleCount >= 3,
  )) {
    lines.push(
      `| ${entry.asset} | ${entry.selectedScaleCount}/5 | ` +
        `${entry.scales
          .map(
            (scale) =>
              `${scale.id} #${scale.rank} (${formatBasisPoints(
                scale.meanAbsoluteReturn,
              )})`,
          )
          .join(", ")} |`,
    );
  }
  lines.push(
    "",
    "## Interpretation",
    "",
    "Mean absolute candle return is a movement/opportunity proxy, not expected profit. It can favor jumpy or difficult-to-execute assets, so a practical portfolio still needs liquidity, spread, slippage, leverage, and out-of-sample stability constraints.",
    "",
  );
  return `${lines.join("\n")}\n`;
}

function renderWeightsCsv(
  report: Parameters<typeof writeReport>[0],
): string {
  const scaleIds = report.scales.map((scale) => scale.id);
  const lines = [
    ["rank", "asset", "weight", ...scaleIds.map((id) => `${id}_weight`)]
      .map(csvValue)
      .join(","),
    ...report.indexWeighting.constituents.map((constituent) =>
      [
        constituent.rank,
        constituent.asset,
        constituent.weight,
        ...scaleIds.map(
          (id) => constituent.scaleWeights[id] ?? 0,
        ),
      ]
        .map(csvValue)
        .join(","),
    ),
  ];
  return `${lines.join("\n")}\n`;
}

async function writeTextAtomic(
  file: string,
  content: string,
): Promise<void> {
  await fs.mkdir(path.dirname(file), { recursive: true });
  const temporary = `${file}.${process.pid}.tmp`;
  await fs.writeFile(temporary, content);
  await fs.rename(temporary, file);
}

function optionValue(values: readonly string[], name: string): string | undefined {
  const index = values.indexOf(name);
  return index < 0 ? undefined : values[index + 1];
}

function csvValue(value: string | number): string {
  const text = String(value);
  return /[",\r\n]/.test(text)
    ? `"${text.replaceAll('"', '""')}"`
    : text;
}

function proportionalWeights(values: readonly number[]): number[] {
  const total = values.reduce((sum, value) => sum + value, 0);
  return total > 0
    ? values.map((value) => value / total)
    : values.map(() => 1 / values.length);
}

function effectiveConstituents(weights: readonly number[]): number {
  const concentration = weights.reduce(
    (sum, weight) => sum + weight * weight,
    0,
  );
  return concentration > 0 ? 1 / concentration : 0;
}

function topWeight(weights: readonly number[], count: number): number {
  return [...weights]
    .sort((left, right) => right - left)
    .slice(0, count)
    .reduce((sum, weight) => sum + weight, 0);
}

function ratioChange(value: number, baseline: number): number {
  return baseline === 0 ? 0 : value / baseline - 1;
}

function mean(values: readonly number[]): number {
  return values.length === 0
    ? 0
    : values.reduce((total, value) => total + value, 0) / values.length;
}

function minimum(values: readonly number[]): number {
  let result = Number.POSITIVE_INFINITY;
  for (const value of values) {
    result = Math.min(result, value);
  }
  return values.length === 0 ? 0 : result;
}

function quantile(values: readonly number[], probability: number): number {
  if (values.length === 0) {
    return 0;
  }
  const sorted = [...values].sort((left, right) => left - right);
  const position = (sorted.length - 1) * probability;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) {
    return sorted[lower];
  }
  const weight = position - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function formatSignedPercent(value: number): string {
  return `${value >= 0 ? "+" : ""}${(value * 100).toFixed(1)}%`;
}

function formatBasisPoints(value: number): string {
  return `${(value * 10_000).toFixed(2)} bp`;
}

function residualBandTag(value: number): string {
  if (value === 0) {
    return "pure";
  }
  return `amp${(value * 100).toFixed(4).replace(/0+$/, "").replace(/\.$/, "").replace(".", "p")}pct`;
}

function formatDecimal(value: number): string {
  return value.toFixed(3);
}

function printHelp(): void {
  console.log(`Usage: npm run basis:compare-scales -- [options]

Compare the latest independent q4 Binance bases for exactly 360 returns at
1d, 4h, 1h, 15m, and 1m. The report compares return-prioritized selection with
pure orthogonality at each scale and then measures cross-scale recurrence.

  --report-dir docs/portfolio    Source-report and output root`);
}
