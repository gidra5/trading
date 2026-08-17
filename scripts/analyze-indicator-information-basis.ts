import fs from "node:fs";
import path from "node:path";
import { deserialize, serialize } from "node:v8";
import { fileURLToPath, pathToFileURL } from "node:url";
import { readCandleShardReferenceSync } from "@trading/storage";
import {
  buildSignalDefinitions,
  IndicatorEngine,
} from "./analyze-technical-indicator-predictiveness.ts";

const DAY_MS = 86_400_000;
const HISTORY_STATES = 33;
const FEATURE_BINS = 4;
const SOURCE_FEATURE_BINS = 16;
const MAGNITUDE_BINS = 16;
const TARGET_CLASSES = 33;
const WARMUP_RETURNS = 12_000;
const EVALUATION_STRIDE = 4;
const SMOOTHING = 0.5;
const DEFAULT_ANALYSIS = "data/benchmarks/log-return-distributions.json";
const DEFAULT_CALIBRATION = "data/runtime-cache/technical-indicator-calibration.bin";
const DEFAULT_COUNTS = "data/runtime-cache/technical-indicator-predictiveness-counts.bin";
const DEFAULT_INDICATOR_RESULTS = "data/benchmarks/technical-indicator-predictiveness.json";
const DEFAULT_STEP_CACHE_DIRECTORY = "data/runtime-cache/indicator-information-basis";
const DEFAULT_OUTPUT = "data/benchmarks/indicator-information-basis.json";
const DEFAULT_REPORT = "docs/experiments/indicator-information-basis-2026-08-16.md";

interface Options {
  analysisPath: string;
  calibrationPath: string;
  countsPath: string;
  indicatorResultsPath: string;
  stepCacheDirectory: string;
  outputPath: string;
  reportPath: string;
  maximumBasisSize: number;
  rebuildSteps: boolean;
}

interface AnalysisReport {
  generatedAt: string;
  source: {
    symbol: string;
    oneSecond: { referenceDirectory: string };
  };
  fullHistory: {
    id: string;
    label: string;
    startTime: string;
    endTime: string;
  };
}

interface Calibration {
  featureEdges: number[][];
  magnitudeEdges: number[];
}

interface CandidateAnnualRow {
  year: number;
  observations: number;
  cumulativeGainBits: number;
  marginalGainBits: number;
}

interface CandidateResult {
  id: string;
  label: string;
  family: string;
  observations: number;
  cumulativeGainBits: number;
  marginalGainBits: number;
  positiveMarginalYears: number;
  annual: CandidateAnnualRow[];
}

interface StepResult {
  selectedBefore: string[];
  featureDimensions: number;
  candidates: CandidateResult[];
}

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function main(): void {
  const options = parseOptions(process.argv.slice(2));
  const analysis = readJson<AnalysisReport>(options.analysisPath);
  const indicatorResults = readJson<any>(options.indicatorResultsPath);
  const definitions = buildSignalDefinitions();
  const definitionById = new Map(definitions.map((definition) => [definition.id, definition]));
  const calibrationEnvelope = deserialize(fs.readFileSync(options.calibrationPath)) as {
    calibration: Calibration;
  };
  const calibration = calibrationEnvelope.calibration;
  if (calibration.featureEdges.length !== definitions.length) {
    throw new Error("Indicator calibration does not match the current signal grid.");
  }
  const countsEnvelope = deserialize(fs.readFileSync(options.countsPath)) as {
    counts: {
      signals: Array<Array<{ jointCounts: Float64Array }>>;
      returnsEvaluated: number;
      activeEvaluated: number;
    };
  };
  if (countsEnvelope.counts.signals.length !== definitions.length) {
    throw new Error("Indicator counts do not match the current signal grid.");
  }
  const singleStep = evaluateSingles(definitions, countsEnvelope.counts.signals);
  const selected: string[] = [];
  const steps: StepResult[] = [singleStep];
  const first = chooseReliableWinner(singleStep.candidates);
  selected.push(first.id);
  const files = selectedFiles(
    path.resolve(repoRoot, analysis.source.oneSecond.referenceDirectory),
    Date.parse(analysis.fullHistory.startTime),
    Date.parse(analysis.fullHistory.endTime),
  );
  while (selected.length < options.maximumBasisSize) {
    const previousCandidates = steps.at(-1)!.candidates;
    const candidateLimit = selected.length <= 2 ? Number.POSITIVE_INFINITY
      : selected.length === 3 ? 32
      : 12;
    const candidatePool = [...previousCandidates]
      .filter((candidate) => !selected.includes(candidate.id))
      .sort(compareCandidates)
      .slice(0, candidateLimit)
      .map((candidate) => candidate.id);
    const step = loadOrScanStep(
      options,
      analysis,
      files,
      definitions,
      calibration,
      selected,
      candidatePool,
    );
    steps.push(step);
    const winner = chooseReliableWinner(step.candidates);
    if (winner.marginalGainBits <= 0) break;
    selected.push(winner.id);
  }
  const singleById = new Map(singleStep.candidates.map((candidate) => [candidate.id, candidate]));
  const basis = selected.map((id, index) => {
    const step = steps[index]!;
    const result = step.candidates.find((candidate) => candidate.id === id)!;
    const individual = singleById.get(id)!;
    return {
      order: index + 1,
      ...definitionById.get(id),
      individualGainBits: individual.cumulativeGainBits,
      cumulativeGainBits: result.cumulativeGainBits,
      marginalGainBits: result.marginalGainBits,
      retainedIndividualInformationFraction:
        result.marginalGainBits / individual.cumulativeGainBits,
      positiveMarginalYears: result.positiveMarginalYears,
      annual: result.annual,
    };
  });
  const artifact = {
    version: 1,
    generatedAt: new Date().toISOString(),
    symbol: analysis.source.symbol,
    scale: "1s → next 1s",
    window: analysis.fullHistory,
    objective: "Greedy maximum rolling-held-out conditional log information about the 33-cell next-return distribution.",
    definition: "At each step maximize E_test[log2 P_train(R|H,B,F)-log2 P_train(R|H,B)].",
    historyBaseline: "Previous return: zero or sign crossed with 16 first-year active-magnitude cells.",
    featureQuantization: `${FEATURE_BINS} first-year equal-mass cells per indicator.`,
    targetClasses: TARGET_CLASSES,
    maximumBasisSize: options.maximumBasisSize,
    returnsEvaluated: countsEnvelope.counts.returnsEvaluated,
    activeEvaluated: countsEnvelope.counts.activeEvaluated,
    historyBaselineGainBits: indicatorResults.historyBaselineByTarget.signedDistribution,
    basis,
    finalIncrementalGainBits: basis.at(-1)!.cumulativeGainBits,
    combinedGainBits:
      indicatorResults.historyBaselineByTarget.signedDistribution
      + basis.at(-1)!.cumulativeGainBits,
    steps: steps.map((step) => ({
      selectedBefore: step.selectedBefore,
      featureDimensions: step.featureDimensions,
      candidatesEvaluated: step.candidates.length,
      topCandidates: [...step.candidates]
        .sort(compareCandidates)
        .slice(0, 20),
    })),
    limitations: [
      "This is greedy forward selection, not an exhaustive search over all indicator subsets.",
      "Quartile cells make exact interactions statistically and computationally tractable; a continuous model can retain more within-cell information.",
      `The search is capped at ${options.maximumBasisSize} indicators because exact state count grows as 4^d. Steps 1–3 scan all indicators; step 4 carries the strongest 32 prior conditional candidates and step 5 carries the strongest 12.`,
      "Every indicator is a deterministic transform of price history, so this is a compact predictive representation rather than new market information beyond the raw history.",
      "The objective is distributional log score, not trading PnL after spread, fees, latency, and market impact.",
    ],
  };
  fs.mkdirSync(path.dirname(options.outputPath), { recursive: true });
  fs.writeFileSync(options.outputPath, `${JSON.stringify(artifact, null, 2)}\n`, "utf8");
  fs.mkdirSync(path.dirname(options.reportPath), { recursive: true });
  fs.writeFileSync(options.reportPath, renderReport(artifact), "utf8");
  console.log(`Wrote ${path.relative(repoRoot, options.outputPath)}`);
  console.log(`Wrote ${path.relative(repoRoot, options.reportPath)}`);
}

function parseOptions(args: string[]): Options {
  const values = new Map<string, string>();
  let rebuildSteps = false;
  for (let index = 0; index < args.length; index += 1) {
    const key = args[index]!;
    if (key === "--rebuild-steps") {
      rebuildSteps = true;
      continue;
    }
    const value = args[index + 1];
    if (!key.startsWith("--") || !value || value.startsWith("--")) {
      throw new Error(`Invalid argument near ${key}.`);
    }
    values.set(key.slice(2), value);
    index += 1;
  }
  const maximumBasisSize = Number(values.get("max-size") ?? 5);
  if (!Number.isInteger(maximumBasisSize) || maximumBasisSize < 1 || maximumBasisSize > 5) {
    throw new Error("--max-size must be an integer from 1 through 5.");
  }
  return {
    analysisPath: path.resolve(repoRoot, values.get("analysis") ?? DEFAULT_ANALYSIS),
    calibrationPath: path.resolve(repoRoot, values.get("calibration") ?? DEFAULT_CALIBRATION),
    countsPath: path.resolve(repoRoot, values.get("counts") ?? DEFAULT_COUNTS),
    indicatorResultsPath: path.resolve(
      repoRoot,
      values.get("indicator-results") ?? DEFAULT_INDICATOR_RESULTS,
    ),
    stepCacheDirectory: path.resolve(
      repoRoot,
      values.get("step-cache-directory") ?? DEFAULT_STEP_CACHE_DIRECTORY,
    ),
    outputPath: path.resolve(repoRoot, values.get("output") ?? DEFAULT_OUTPUT),
    reportPath: path.resolve(repoRoot, values.get("report") ?? DEFAULT_REPORT),
    maximumBasisSize,
    rebuildSteps,
  };
}

function readJson<T>(file: string): T {
  return JSON.parse(fs.readFileSync(file, "utf8")) as T;
}

function evaluateSingles(
  definitions: ReturnType<typeof buildSignalDefinitions>,
  signals: Array<Array<{ jointCounts: Float64Array }>>,
): StepResult {
  console.error("Evaluating quartile single-indicator information...");
  return {
    selectedBefore: [],
    featureDimensions: 1,
    candidates: definitions.map((definition, index) => {
      const annual = signals[index]!.map((row) => coarsenFeatureCounts(row.jointCounts));
      return {
        id: definition.id,
        label: definition.label,
        family: definition.family,
        ...rollingEvaluation(annual, 0),
      };
    }),
  };
}

export function coarsenFeatureCounts(source: Float64Array): Float64Array {
  const result = new Float64Array(HISTORY_STATES * FEATURE_BINS * TARGET_CLASSES);
  for (let history = 0; history < HISTORY_STATES; history += 1) {
    for (let feature = 0; feature < SOURCE_FEATURE_BINS; feature += 1) {
      const coarseFeature = Math.floor(feature / (SOURCE_FEATURE_BINS / FEATURE_BINS));
      for (let target = 0; target < TARGET_CLASSES; target += 1) {
        result[(history * FEATURE_BINS + coarseFeature) * TARGET_CLASSES + target] +=
          source[(history * SOURCE_FEATURE_BINS + feature) * TARGET_CLASSES + target]!;
      }
    }
  }
  return result;
}

function rollingEvaluation(annual: Float64Array[], selectedDimensions: number) {
  const train = new Float64Array(annual[0]!.length);
  const rows: CandidateAnnualRow[] = [];
  for (let year = 0; year < annual.length; year += 1) {
    if (year > 0) {
      rows.push(evaluateCandidateYear(train, annual[year]!, selectedDimensions, year));
    }
    addInPlace(train, annual[year]!);
  }
  return summarizeAnnual(rows);
}

function loadOrScanStep(
  options: Options,
  analysis: AnalysisReport,
  files: Array<{ file: string; dayStart: number }>,
  definitions: ReturnType<typeof buildSignalDefinitions>,
  calibration: Calibration,
  selected: string[],
  candidatePool: string[],
): StepResult {
  const cacheKey = `after-${selected.join("_")}.bin`;
  const cachePath = path.join(options.stepCacheDirectory, cacheKey);
  const metadata = JSON.stringify({
    version: 1,
    analysisGeneratedAt: analysis.generatedAt,
    selected,
    candidatePool,
    definitions,
    calibration,
    warmupReturns: WARMUP_RETURNS,
    evaluationStride: EVALUATION_STRIDE,
  });
  if (!options.rebuildSteps && fs.existsSync(cachePath)) {
    const cached = deserialize(fs.readFileSync(cachePath)) as {
      metadata: string;
      step: StepResult;
    };
    const completeLegacyStep = selected.length <= 2
      && cached.step.selectedBefore.join("|") === selected.join("|")
      && cached.step.candidates.length === definitions.length - selected.length;
    if (cached.metadata === metadata || completeLegacyStep) {
      console.log(`Loading basis step from ${path.relative(repoRoot, cachePath)}`);
      return cached.step;
    }
  }
  const step = scanCandidateAdditions(
    files,
    analysis,
    definitions,
    calibration,
    selected,
    candidatePool,
  );
  fs.mkdirSync(path.dirname(cachePath), { recursive: true });
  fs.writeFileSync(cachePath, serialize({ metadata, step }));
  return step;
}

function scanCandidateAdditions(
  files: Array<{ file: string; dayStart: number }>,
  analysis: AnalysisReport,
  definitions: ReturnType<typeof buildSignalDefinitions>,
  calibration: Calibration,
  selected: string[],
  candidatePool: string[],
): StepResult {
  const candidatePoolSet = new Set(candidatePool);
  const candidateDefinitions = definitions.filter((definition) =>
    !selected.includes(definition.id) && candidatePoolSet.has(definition.id));
  const definitionIndex = new Map(definitions.map((definition, index) => [definition.id, index]));
  const selectedIndices = selected.map((id) => definitionIndex.get(id)!);
  const candidateIndices = candidateDefinitions.map((definition) => definitionIndex.get(definition.id)!);
  const featureStates = FEATURE_BINS ** (selected.length + 1);
  const tableLength = HISTORY_STATES * featureStates * TARGET_CLASSES;
  console.error(
    `Scanning ${candidateDefinitions.length} additions after [${selected.join(", ")}], ${tableLength.toLocaleString("en-US")} cells each...`,
  );
  const train = candidateDefinitions.map(() => new Float64Array(tableLength));
  const current = candidateDefinitions.map(() => new Float64Array(tableLength));
  const annualRows = candidateDefinitions.map((): CandidateAnnualRow[] => []);
  const featureQuartileEdges = calibration.featureEdges.map(quartileEdges);
  const values = new Float64Array(definitions.length);
  const start = Date.parse(analysis.fullHistory.startTime);
  let engine: IndicatorEngine | undefined;
  let previousClose = Number.NaN;
  let previousReturnState = 0;
  let returnsSeen = 0;
  let currentYear = 0;
  for (const [fileIndex, entry] of files.entries()) {
    const year = anniversaryIndex(start, entry.dayStart);
    if (year !== currentYear) {
      finalizeYear(train, current, annualRows, selected.length, currentYear);
      currentYear = year;
    }
    if (fileIndex % 100 === 0) {
      console.error(`Basis [${selected.join(", ")}] ${fileIndex}/${files.length}...`);
    }
    const candles = readCandleShardReferenceSync(entry.file);
    if (candles.length !== 86_400) throw new Error(`${path.basename(entry.file)} is incomplete.`);
    for (const candle of candles) {
      if (!Number.isFinite(candle.close) || candle.close <= 0) {
        throw new Error(`${path.basename(entry.file)} contains an invalid close.`);
      }
      if (!engine) {
        engine = new IndicatorEngine(definitions, candle.close);
      } else {
        const returnBps = candle.close === previousClose
          ? 0
          : Math.log(candle.close / previousClose) * 10_000;
        const active = returnBps !== 0;
        const magnitude = Math.abs(returnBps);
        const sign = returnBps > 0 ? 1 : 0;
        const magnitudeBin = active ? upperBound(calibration.magnitudeEdges, magnitude) : -1;
        const targetClass = active ? 1 + sign * MAGNITUDE_BINS + magnitudeBin : 0;
        if (returnsSeen >= WARMUP_RETURNS && isEvaluationTarget(returnsSeen)) {
          engine.values(values);
          let selectedState = 0;
          for (const selectedIndex of selectedIndices) {
            selectedState = selectedState * FEATURE_BINS
              + upperBound(featureQuartileEdges[selectedIndex]!, values[selectedIndex]!);
          }
          for (let candidate = 0; candidate < candidateIndices.length; candidate += 1) {
            const candidateIndex = candidateIndices[candidate]!;
            const featureState = selectedState * FEATURE_BINS
              + upperBound(featureQuartileEdges[candidateIndex]!, values[candidateIndex]!);
            const position = (
              previousReturnState * featureStates + featureState
            ) * TARGET_CLASSES + targetClass;
            current[candidate]![position] += 1;
          }
        }
        previousReturnState = active
          ? 1 + sign * MAGNITUDE_BINS + magnitudeBin
          : 0;
        engine.update(candle.close);
        returnsSeen += 1;
      }
      previousClose = candle.close;
    }
  }
  finalizeYear(train, current, annualRows, selected.length, currentYear);
  return {
    selectedBefore: [...selected],
    featureDimensions: selected.length + 1,
    candidates: candidateDefinitions.map((definition, index) => ({
      id: definition.id,
      label: definition.label,
      family: definition.family,
      ...summarizeAnnual(annualRows[index]!),
    })),
  };
}

function finalizeYear(
  train: Float64Array[],
  current: Float64Array[],
  annualRows: CandidateAnnualRow[][],
  selectedDimensions: number,
  year: number,
): void {
  console.error(`Finalizing basis year ${year}...`);
  for (let candidate = 0; candidate < train.length; candidate += 1) {
    if (year > 0) {
      annualRows[candidate]!.push(evaluateCandidateYear(
        train[candidate]!,
        current[candidate]!,
        selectedDimensions,
        year,
      ));
    }
    addInPlace(train[candidate]!, current[candidate]!);
    current[candidate]!.fill(0);
  }
}

export function evaluateCandidateYear(
  train: Float64Array,
  test: Float64Array,
  selectedDimensions: number,
  year: number,
): CandidateAnnualRow {
  const basisStates = FEATURE_BINS ** selectedDimensions;
  const fullFeatureStates = basisStates * FEATURE_BINS;
  const historyTrain = new Float64Array(HISTORY_STATES * TARGET_CLASSES);
  const historyTotals = new Float64Array(HISTORY_STATES);
  const basisTrain = new Float64Array(HISTORY_STATES * basisStates * TARGET_CLASSES);
  const basisTotals = new Float64Array(HISTORY_STATES * basisStates);
  const fullTotals = new Float64Array(HISTORY_STATES * fullFeatureStates);
  for (let history = 0; history < HISTORY_STATES; history += 1) {
    for (let feature = 0; feature < fullFeatureStates; feature += 1) {
      const basis = Math.floor(feature / FEATURE_BINS);
      const fullState = history * fullFeatureStates + feature;
      const basisState = history * basisStates + basis;
      for (let target = 0; target < TARGET_CLASSES; target += 1) {
        const count = train[fullState * TARGET_CLASSES + target]!;
        historyTrain[history * TARGET_CLASSES + target] += count;
        historyTotals[history] += count;
        basisTrain[basisState * TARGET_CLASSES + target] += count;
        basisTotals[basisState] += count;
        fullTotals[fullState] += count;
      }
    }
  }
  let observations = 0;
  let cumulative = 0;
  let marginal = 0;
  for (let history = 0; history < HISTORY_STATES; history += 1) {
    for (let feature = 0; feature < fullFeatureStates; feature += 1) {
      const basis = Math.floor(feature / FEATURE_BINS);
      const fullState = history * fullFeatureStates + feature;
      const basisState = history * basisStates + basis;
      for (let target = 0; target < TARGET_CLASSES; target += 1) {
        const count = test[fullState * TARGET_CLASSES + target]!;
        if (count === 0) continue;
        observations += count;
        const fullProbability = smoothedProbability(
          train[fullState * TARGET_CLASSES + target]!,
          fullTotals[fullState]!,
        );
        const basisProbability = smoothedProbability(
          basisTrain[basisState * TARGET_CLASSES + target]!,
          basisTotals[basisState]!,
        );
        const historyProbability = smoothedProbability(
          historyTrain[history * TARGET_CLASSES + target]!,
          historyTotals[history]!,
        );
        cumulative += count * Math.log2(fullProbability / historyProbability);
        marginal += count * Math.log2(fullProbability / basisProbability);
      }
    }
  }
  return {
    year,
    observations,
    cumulativeGainBits: cumulative / observations,
    marginalGainBits: marginal / observations,
  };
}

function summarizeAnnual(rows: CandidateAnnualRow[]) {
  const observations = rows.reduce((sum, row) => sum + row.observations, 0);
  return {
    observations,
    cumulativeGainBits: rows.reduce(
      (sum, row) => sum + row.cumulativeGainBits * row.observations,
      0,
    ) / observations,
    marginalGainBits: rows.reduce(
      (sum, row) => sum + row.marginalGainBits * row.observations,
      0,
    ) / observations,
    positiveMarginalYears: rows.filter((row) => row.marginalGainBits > 0).length,
    annual: rows,
  };
}

function chooseReliableWinner(candidates: CandidateResult[]): CandidateResult {
  const stable = candidates.filter((candidate) => candidate.positiveMarginalYears === 4);
  const pool = stable.length > 0 ? stable : candidates;
  return [...pool].sort(compareCandidates)[0]!;
}

function compareCandidates(left: CandidateResult, right: CandidateResult): number {
  return right.marginalGainBits - left.marginalGainBits;
}

function smoothedProbability(count: number, total: number): number {
  return (count + SMOOTHING) / (total + TARGET_CLASSES * SMOOTHING);
}

function addInPlace(target: Float64Array, source: Float64Array): void {
  for (let index = 0; index < target.length; index += 1) target[index] += source[index]!;
}

export function quartileEdges(edges: number[]): number[] {
  if (edges.length !== SOURCE_FEATURE_BINS - 1) {
    throw new Error("Quartile coarsening expects 15 internal edges.");
  }
  return [edges[3]!, edges[7]!, edges[11]!];
}

function upperBound(sorted: number[], value: number): number {
  let low = 0;
  let high = sorted.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (value < sorted[middle]!) high = middle;
    else low = middle + 1;
  }
  return low;
}

function isEvaluationTarget(index: number): boolean {
  let mixed = index ^ (index >>> 16);
  mixed = Math.imul(mixed, 0x45d9f3b);
  mixed ^= mixed >>> 16;
  return (mixed >>> 0) % EVALUATION_STRIDE === 0;
}

function selectedFiles(directory: string, start: number, end: number) {
  const result = fs.readdirSync(directory, { withFileTypes: true })
    .filter((entry) => entry.isFile() && /^\d{4}-\d{2}-\d{2}\.json$/.test(entry.name))
    .map((entry) => ({
      file: path.join(directory, entry.name),
      dayStart: Date.parse(`${entry.name.slice(0, 10)}T00:00:00.000Z`),
    }))
    .filter((entry) => entry.dayStart >= start && entry.dayStart < end)
    .sort((left, right) => left.dayStart - right.dayStart);
  for (let index = 1; index < result.length; index += 1) {
    if (result[index]!.dayStart !== result[index - 1]!.dayStart + DAY_MS) {
      throw new Error(`Missing one-second shard before ${new Date(result[index]!.dayStart).toISOString()}.`);
    }
  }
  return result;
}

function anniversary(startMs: number, years: number): number {
  const start = new Date(startMs);
  return Date.UTC(start.getUTCFullYear() + years, start.getUTCMonth(), start.getUTCDate());
}

function anniversaryIndex(startMs: number, dayMs: number): number {
  const start = new Date(startMs);
  const day = new Date(dayMs);
  let index = day.getUTCFullYear() - start.getUTCFullYear();
  if (dayMs < anniversary(startMs, index)) index -= 1;
  return Math.max(0, index);
}

function renderReport(artifact: any): string {
  const coreThreeGain = artifact.basis[Math.min(2, artifact.basis.length - 1)].cumulativeGainBits;
  const coreFourGain = artifact.basis[Math.min(3, artifact.basis.length - 1)].cumulativeGainBits;
  const lines = [
    "# Greedy information basis for the next one-second return",
    "",
    `Generated ${artifact.generatedAt}. The exact quartile interaction search uses ${artifact.returnsEvaluated.toLocaleString("en-US")} held-out-sampled targets from the same five-year ${artifact.symbol} one-second history.`,
    "",
    "## Result",
    "",
    `The selected ${artifact.basis.length}-indicator basis adds ${formatMetric(artifact.finalIncrementalGainBits)} bits/target beyond the latest-return state. Together with that baseline, the cumulative held-out gain over the unconditional distribution is ${formatMetric(artifact.combinedGainBits)} bits/target.`,
    "",
    "| order | indicator | family | individual quartile gain | marginal gain when added | cumulative basis gain | retained individual information | positive years |",
    "|---:|---|---|---:|---:|---:|---:|---:|",
  ];
  for (const entry of artifact.basis) {
    lines.push(
      `| ${entry.order} | ${entry.label} | ${entry.family} | ${formatMetric(entry.individualGainBits)} | ${formatMetric(entry.marginalGainBits)} | ${formatMetric(entry.cumulativeGainBits)} | ${(entry.retainedIndividualInformationFraction * 100).toFixed(2)}% | ${entry.positiveMarginalYears}/4 |`,
    );
  }
  lines.push(
    "",
    "The retained-information column is `conditional marginal gain / standalone indicator gain`. Values far below 100% reveal redundancy with features already in the basis; values above 100% indicate complementary interaction.",
    "",
    "## Practical stopping points",
    "",
    `- The first three indicators capture ${(coreThreeGain / artifact.finalIncrementalGainBits * 100).toFixed(2)}% of the maximum tested five-feature information.`,
    `- The first four capture ${(coreFourGain / artifact.finalIncrementalGainBits * 100).toFixed(2)}%.`,
    `- The full basis improves geometric assigned probability by ${((2 ** artifact.finalIncrementalGainBits - 1) * 100).toFixed(3)}% beyond the latest-return state; latest return plus basis improves it by ${((2 ** artifact.combinedGainBits - 1) * 100).toFixed(3)}% over the unconditional distribution.`,
    "- Use the first three as the compact basis, the first four when a small extra interaction cost is acceptable, and all five only when maximizing held-out distributional score matters more than minimality.",
    "",
    "## Selection alternatives",
    "",
  );
  for (let index = 0; index < artifact.steps.length; index += 1) {
    const step = artifact.steps[index]!;
    lines.push(
      `### Step ${index + 1}: after ${step.selectedBefore.length === 0 ? "history only" : step.selectedBefore.join(" + ")} (${step.candidatesEvaluated} candidates)`,
      "",
      "| candidate | family | marginal bits | cumulative bits | positive years |",
      "|---|---|---:|---:|---:|",
    );
    for (const candidate of step.topCandidates.slice(0, 12)) {
      lines.push(
        `| ${candidate.label} | ${candidate.family} | ${formatMetric(candidate.marginalGainBits)} | ${formatMetric(candidate.cumulativeGainBits)} | ${candidate.positiveMarginalYears}/4 |`,
      );
    }
    lines.push("");
  }
  lines.push(
    "## Objective and method",
    "",
    "For history state `H`, current basis `B`, candidate `F`, and next-return cell `R`, the selected feature maximizes:",
    "",
    "```text",
    "mean_test[log2 P_train(R | H, B, F) - log2 P_train(R | H, B)]",
    "```",
    "",
    `Each indicator is reduced to ${FEATURE_BINS} equal-mass year-0 cells. Interactions are counted exactly; annual test windows use only earlier years for their probability tables. A candidate must be positive in all four years when such candidates exist.`,
    "",
    "## Limitations",
    "",
    ...artifact.limitations.map((limitation: string) => `- ${limitation}`),
    "",
    "## Reproducibility",
    "",
    "```text",
    "node --conditions=development --import tsx scripts/analyze-indicator-information-basis.ts",
    "```",
    "",
    "Complete rankings are stored in `data/benchmarks/indicator-information-basis.json`.",
  );
  return `${lines.join("\n")}\n`;
}

function formatMetric(value: number): string {
  return value === 0 ? "0" : value.toPrecision(8).replace(/\.0+(?=e|$)/, "");
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  try {
    main();
  } catch (error: unknown) {
    console.error(error instanceof Error ? error.stack ?? error.message : String(error));
    process.exitCode = 1;
  }
}
