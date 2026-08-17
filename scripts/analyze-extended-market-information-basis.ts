import fs from "node:fs";
import path from "node:path";
import { deserialize, serialize } from "node:v8";
import { fileURLToPath, pathToFileURL } from "node:url";
import { readCandleShardReferenceSync } from "@trading/storage";
import {
  buildSignalDefinitions,
  IndicatorEngine,
} from "./analyze-technical-indicator-predictiveness.ts";
import { evaluateCandidateYear } from "./analyze-indicator-information-basis.ts";
import {
  buildMarketFeatureDefinitions,
  CausalMarketFeatureEngine,
} from "./analyze-volume-multiscale-information.ts";

const DAY_MS = 86_400_000;
const HISTORY_STATES = 33;
const FEATURE_BINS = 4;
const MAGNITUDE_BINS = 16;
const TARGET_CLASSES = 33;
const WARMUP_RETURNS = 12_000;
const EVALUATION_STRIDE = 4;
const CANDIDATE_LIMIT = 24;
const PRICE_BASIS_IDS = ["rsi-2", "ema-acceleration-2-1", "ema-slope-8-8"] as const;
const PRICE_BENCHMARK_IDS = ["ema-acceleration-2-2", "ema-gap-8"] as const;
const DEFAULT_ANALYSIS = "data/benchmarks/log-return-distributions.json";
const DEFAULT_PRICE_CALIBRATION = "data/runtime-cache/technical-indicator-calibration.bin";
const DEFAULT_MARKET_CALIBRATION = "data/runtime-cache/volume-multiscale-calibration.bin";
const DEFAULT_SCREEN = "data/benchmarks/volume-multiscale-information.json";
const DEFAULT_COUNTS = "data/runtime-cache/extended-market-information-basis-counts.bin";
const DEFAULT_OUTPUT = "data/benchmarks/extended-market-information-basis.json";
const DEFAULT_REPORT = "docs/experiments/extended-market-information-basis-2026-08-16.md";

interface AnalysisReport {
  generatedAt: string;
  source: { symbol: string; oneSecond: { referenceDirectory: string } };
  fullHistory: { startTime: string; endTime: string };
}

interface Calibration {
  featureEdges: number[][];
  magnitudeEdges: number[];
}

interface RankedCandidate {
  id: string;
  label: string;
  family: string;
  resolution: string;
  individualGainBits: number;
  marginalGainBits: number;
  cumulativeGainBits: number;
  positiveMarginalYears: number;
}

interface ScreenArtifact {
  generatedAt: string;
  bestExternal: RankedCandidate;
  rankings: RankedCandidate[];
  coreThreeGainBits: number;
}

interface CandidateSource extends RankedCandidate {
  source: "market" | "price";
  sourceIndex: number;
  edges: number[];
}

interface AnnualRow {
  year: number;
  observations: number;
  cumulativeGainBits: number;
  marginalGainBits: number;
}

interface CandidateResult extends RankedCandidate {
  observations: number;
  annual: AnnualRow[];
}

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function main(): void {
  const values = parseArguments(process.argv.slice(2));
  const analysisPath = resolve(values.get("analysis") ?? DEFAULT_ANALYSIS);
  const priceCalibrationPath = resolve(values.get("price-calibration") ?? DEFAULT_PRICE_CALIBRATION);
  const marketCalibrationPath = resolve(values.get("market-calibration") ?? DEFAULT_MARKET_CALIBRATION);
  const screenPath = resolve(values.get("screen") ?? DEFAULT_SCREEN);
  const countsPath = resolve(values.get("counts") ?? DEFAULT_COUNTS);
  const outputPath = resolve(values.get("output") ?? DEFAULT_OUTPUT);
  const reportPath = resolve(values.get("report") ?? DEFAULT_REPORT);
  const rebuild = values.get("rebuild-counts") === "true";
  const analysis = readJson<AnalysisReport>(analysisPath);
  const screen = readJson<ScreenArtifact>(screenPath);
  const priceCalibration = (deserialize(fs.readFileSync(priceCalibrationPath)) as {
    calibration: Calibration;
  }).calibration;
  const marketEdges = (deserialize(fs.readFileSync(marketCalibrationPath)) as {
    marketEdges: number[][];
  }).marketEdges;
  const allPriceDefinitions = buildSignalDefinitions();
  const priceDefinitionIndex = new Map(
    allPriceDefinitions.map((definition, index) => [definition.id, index]),
  );
  const marketDefinitions = buildMarketFeatureDefinitions();
  const marketDefinitionById = new Map(marketDefinitions.map((definition) => [definition.id, definition]));
  const selected = candidateSource(
    screen.bestExternal,
    marketDefinitionById,
    marketEdges,
    priceDefinitionIndex,
    priceCalibration,
  );
  const poolRows = screen.rankings
    .filter((row) => row.id !== selected.id)
    .filter((row) => row.positiveMarginalYears === 4)
    .slice(0, CANDIDATE_LIMIT);
  for (const priceId of PRICE_BENCHMARK_IDS.map((id) => `price-${id}`)) {
    if (!poolRows.some((row) => row.id === priceId)) {
      poolRows.push(screen.rankings.find((row) => row.id === priceId)!);
    }
  }
  const pool = poolRows.map((row) => candidateSource(
    row,
    marketDefinitionById,
    marketEdges,
    priceDefinitionIndex,
    priceCalibration,
  ));
  const files = selectedFiles(
    path.resolve(repoRoot, analysis.source.oneSecond.referenceDirectory),
    Date.parse(analysis.fullHistory.startTime),
    Date.parse(analysis.fullHistory.endTime),
  );
  const metadata = JSON.stringify({
    version: 1,
    analysisGeneratedAt: analysis.generatedAt,
    screenGeneratedAt: screen.generatedAt,
    selected: selected.id,
    pool: pool.map((candidate) => candidate.id),
    evaluationStride: EVALUATION_STRIDE,
  });
  let annualRows: AnnualRow[][];
  if (!rebuild && fs.existsSync(countsPath)) {
    const cached = deserialize(fs.readFileSync(countsPath)) as {
      metadata: string;
      annualRows: AnnualRow[][];
    };
    if (cached.metadata === metadata) {
      console.log(`Loading extended-basis counts from ${path.relative(repoRoot, countsPath)}`);
      annualRows = cached.annualRows;
    } else {
      annualRows = scan(files, analysis, priceCalibration, priceDefinitionIndex, selected, pool);
      writeBinary(countsPath, { metadata, annualRows });
    }
  } else {
    annualRows = scan(files, analysis, priceCalibration, priceDefinitionIndex, selected, pool);
    writeBinary(countsPath, { metadata, annualRows });
  }
  const results = pool.map((candidate, index): CandidateResult => {
    const annual = annualRows[index]!;
    const observations = annual.reduce((sum, row) => sum + row.observations, 0);
    return {
      ...candidate,
      observations,
      cumulativeGainBits: weightedMean(annual, (row) => row.cumulativeGainBits),
      marginalGainBits: weightedMean(annual, (row) => row.marginalGainBits),
      positiveMarginalYears: annual.filter((row) => row.marginalGainBits > 0).length,
      annual,
    };
  }).sort(compareResults);
  const winner = results[0]!;
  const artifact = {
    version: 1,
    generatedAt: new Date().toISOString(),
    symbol: analysis.source.symbol,
    scale: "causal history through t → next 1s return",
    selectionMethod: "Greedy exact quartile interaction search with four rolling annual holdouts.",
    evaluationStride: EVALUATION_STRIDE,
    candidateShortlist: pool.length,
    basisBefore: [
      ...PRICE_BASIS_IDS,
      selected.id,
    ],
    selectedFourth: selected,
    selectedFifth: winner,
    coreThreeGainBits: screen.coreThreeGainBits,
    fourFeatureGainBits: weightedMean(
      winner.annual,
      (row) => row.cumulativeGainBits - row.marginalGainBits,
    ),
    fiveFeatureGainBits: winner.cumulativeGainBits,
    rankings: results,
    limitations: [
      `The fifth-step scan carries the strongest ${CANDIDATE_LIMIT} stable candidates from the prior exhaustive external-feature screen, plus the two prior price benchmarks.`,
      "This is greedy forward selection, not a proof of the globally optimal nonlinear feature subset.",
      "Inputs are quartile-quantized for exact interaction counts; a learned continuous model can retain more detail.",
      "The spot archive supplies base volume but not taker imbalance, trade count, order book, spread, or derivatives flow.",
    ],
  };
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, `${JSON.stringify(artifact, null, 2)}\n`, "utf8");
  fs.mkdirSync(path.dirname(reportPath), { recursive: true });
  fs.writeFileSync(reportPath, renderReport(artifact), "utf8");
  console.log(`Wrote ${path.relative(repoRoot, outputPath)}`);
  console.log(`Wrote ${path.relative(repoRoot, reportPath)}`);
}

function scan(
  files: Array<{ file: string; dayStart: number }>,
  analysis: AnalysisReport,
  calibration: Calibration,
  priceDefinitionIndex: Map<string, number>,
  selected: CandidateSource,
  pool: CandidateSource[],
): AnnualRow[][] {
  const selectedDimensions = PRICE_BASIS_IDS.length + 1;
  const featureStates = FEATURE_BINS ** (selectedDimensions + 1);
  const tableLength = HISTORY_STATES * featureStates * TARGET_CLASSES;
  console.error(
    `Scanning ${pool.length} fifth-coordinate candidates after ${selected.id}; `
      + `${tableLength.toLocaleString("en-US")} cells each...`,
  );
  const train = pool.map(() => new Float64Array(tableLength));
  const current = pool.map(() => new Float64Array(tableLength));
  const annualRows = pool.map((): AnnualRow[] => []);
  const marketEngine = new CausalMarketFeatureEngine();
  const priceIds = [...PRICE_BASIS_IDS, ...PRICE_BENCHMARK_IDS];
  const priceDefinitions = selectedPriceDefinitions(priceIds);
  const priceEngineIndex = new Map(priceDefinitions.map((definition, index) => [definition.id, index]));
  const priceValues = new Float64Array(priceDefinitions.length);
  const priceBasisEdges = PRICE_BASIS_IDS.map((id) => quartileEdges(
    calibration.featureEdges[priceDefinitionIndex.get(id)!]!,
  ));
  let priceEngine: IndicatorEngine | undefined;
  const start = Date.parse(analysis.fullHistory.startTime);
  let previousClose = Number.NaN;
  let previousReturnState = 0;
  let returnsSeen = 0;
  let currentYear = 0;
  for (const [fileIndex, entry] of files.entries()) {
    const year = anniversaryIndex(start, entry.dayStart);
    if (year !== currentYear) {
      finalizeYear(train, current, annualRows, selectedDimensions, currentYear);
      currentYear = year;
    }
    if (fileIndex % 50 === 0) console.error(`Extended basis ${fileIndex}/${files.length}...`);
    const candles = readCandleShardReferenceSync(entry.file);
    if (candles.length !== 86_400) throw new Error(`${path.basename(entry.file)} is incomplete.`);
    for (const candle of candles) {
      if (!priceEngine) {
        priceEngine = new IndicatorEngine(priceDefinitions, candle.close);
        marketEngine.update(candle);
      } else {
        const returnBps = candle.close === previousClose
          ? 0
          : Math.log(candle.close / previousClose) * 10_000;
        const active = returnBps !== 0;
        const sign = returnBps > 0 ? 1 : 0;
        const magnitudeBin = active
          ? upperBound(calibration.magnitudeEdges, Math.abs(returnBps))
          : -1;
        const targetClass = active ? 1 + sign * MAGNITUDE_BINS + magnitudeBin : 0;
        if (returnsSeen >= WARMUP_RETURNS && isEvaluationTarget(returnsSeen)) {
          priceEngine.values(priceValues);
          const marketValues = marketEngine.values();
          const marketValid = marketEngine.valid();
          if (isValid(selected, marketValid)) {
            let selectedState = 0;
            for (let basis = 0; basis < PRICE_BASIS_IDS.length; basis += 1) {
              selectedState = selectedState * FEATURE_BINS + upperBound(
                priceBasisEdges[basis]!,
                priceValues[priceEngineIndex.get(PRICE_BASIS_IDS[basis]!)!]!,
              );
            }
            selectedState = selectedState * FEATURE_BINS + upperBound(
              selected.edges,
              sourceValue(selected, marketValues, priceValues, priceEngineIndex),
            );
            for (let candidate = 0; candidate < pool.length; candidate += 1) {
              const definition = pool[candidate]!;
              if (!isValid(definition, marketValid)) continue;
              const featureState = selectedState * FEATURE_BINS + upperBound(
                definition.edges,
                sourceValue(definition, marketValues, priceValues, priceEngineIndex),
              );
              current[candidate]![
                (previousReturnState * featureStates + featureState) * TARGET_CLASSES + targetClass
              ] += 1;
            }
          }
        }
        previousReturnState = active ? 1 + sign * MAGNITUDE_BINS + magnitudeBin : 0;
        priceEngine.update(candle.close);
        marketEngine.update(candle);
        returnsSeen += 1;
      }
      previousClose = candle.close;
    }
  }
  finalizeYear(train, current, annualRows, selectedDimensions, currentYear);
  return annualRows;
}

function finalizeYear(
  train: Float64Array[],
  current: Float64Array[],
  annualRows: AnnualRow[][],
  selectedDimensions: number,
  year: number,
): void {
  console.error(`Finalizing extended-basis year ${year}...`);
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

function candidateSource(
  row: RankedCandidate,
  marketById: Map<string, ReturnType<typeof buildMarketFeatureDefinitions>[number]>,
  marketEdges: number[][],
  priceIndex: Map<string, number>,
  priceCalibration: Calibration,
): CandidateSource {
  if (row.id.startsWith("price-")) {
    const id = row.id.slice("price-".length);
    const sourceIndex = priceIndex.get(id);
    if (sourceIndex === undefined) throw new Error(`Missing price candidate ${row.id}.`);
    return {
      ...row,
      source: "price",
      sourceIndex,
      edges: quartileEdges(priceCalibration.featureEdges[sourceIndex]!),
    };
  }
  const definition = marketById.get(row.id);
  if (!definition) throw new Error(`Missing market candidate ${row.id}.`);
  return {
    ...row,
    source: "market",
    sourceIndex: definition.sourceIndex,
    edges: marketEdges[definition.sourceIndex]!,
  };
}

function sourceValue(
  source: CandidateSource,
  marketValues: Float64Array,
  priceValues: Float64Array,
  priceEngineIndex: Map<string, number>,
): number {
  return source.source === "market"
    ? marketValues[source.sourceIndex]!
    : priceValues[priceEngineIndex.get(source.id.slice("price-".length))!]!;
}

function isValid(source: CandidateSource, marketValid: Uint8Array): boolean {
  return source.source === "price" || marketValid[source.sourceIndex] === 1;
}

function renderReport(artifact: any): string {
  const fourth = artifact.selectedFourth as CandidateSource;
  const fifth = artifact.selectedFifth as CandidateResult;
  const lines = [
    "# Extended price, volume, and multiscale information basis",
    "",
    `Generated ${artifact.generatedAt} for ${artifact.symbol} next-one-second return distributions.`,
    "",
    "## Result",
    "",
    `Adding **${fourth.label}** to the prior price core-three raises held-out information from ${metric(artifact.coreThreeGainBits)} to ${metric(artifact.fourFeatureGainBits)} bits/target beyond the latest-return state.`,
    "",
    `After conditioning on that volume regime, the best remaining fifth coordinate is **${fifth.label}**, adding ${metric(fifth.marginalGainBits)} bits/target for a cumulative ${metric(artifact.fiveFeatureGainBits)} bits/target. It is positive in ${fifth.positiveMarginalYears}/4 annual holdouts.`,
    "",
    "| order | coordinate | marginal bits | cumulative bits |",
    "|---:|---|---:|---:|",
    `| 1–3 | price core: RSI(2s), EMA acceleration(2,1), EMA slope(8,8) | ${metric(artifact.coreThreeGainBits)} | ${metric(artifact.coreThreeGainBits)} |`,
    `| 4 | ${fourth.label} | ${metric(artifact.fourFeatureGainBits - artifact.coreThreeGainBits)} | ${metric(artifact.fourFeatureGainBits)} |`,
    `| 5 | ${fifth.label} | ${metric(fifth.marginalGainBits)} | ${metric(artifact.fiveFeatureGainBits)} |`,
    "",
    "## Fifth-coordinate alternatives",
    "",
    "| rank | candidate | scale | marginal bits | cumulative bits | positive years |",
    "|---:|---|---|---:|---:|---:|",
  ];
  artifact.rankings.forEach((row: CandidateResult, index: number) => {
    lines.push(`| ${index + 1} | ${row.label} | ${row.resolution} | ${metric(row.marginalGainBits)} | ${metric(row.cumulativeGainBits)} | ${row.positiveMarginalYears}/4 |`);
  });
  lines.push(
    "",
    "The fifth-step counts condition on the exact quartile cross of the previous-return state, all three price coordinates, the selected hourly-volume coordinate, and each candidate. All probabilities for a test year are estimated only from earlier years.",
    "",
    "## Limits",
    "",
    ...artifact.limitations.map((limitation: string) => `- ${limitation}`),
    "",
    "Complete values are stored in `data/benchmarks/extended-market-information-basis.json`.",
  );
  return `${lines.join("\n")}\n`;
}

function compareResults(left: CandidateResult, right: CandidateResult): number {
  const stable = right.positiveMarginalYears - left.positiveMarginalYears;
  return stable !== 0 ? stable : right.marginalGainBits - left.marginalGainBits;
}

function weightedMean(rows: AnnualRow[], value: (row: AnnualRow) => number): number {
  const observations = rows.reduce((sum, row) => sum + row.observations, 0);
  return rows.reduce((sum, row) => sum + value(row) * row.observations, 0) / observations;
}

function selectedPriceDefinitions(ids: readonly string[]) {
  const byId = new Map(buildSignalDefinitions().map((definition) => [definition.id, definition]));
  return ids.map((id) => {
    const definition = byId.get(id);
    if (!definition) throw new Error(`Missing price definition ${id}.`);
    return definition;
  });
}

function quartileEdges(edges: number[]): number[] {
  if (edges.length !== 15) throw new Error("Expected 15 calibration edges.");
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

function addInPlace(target: Float64Array, source: Float64Array): void {
  for (let index = 0; index < target.length; index += 1) target[index] += source[index]!;
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
      throw new Error(`Missing shard before ${new Date(result[index]!.dayStart).toISOString()}.`);
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

function parseArguments(args: string[]): Map<string, string> {
  const values = new Map<string, string>();
  for (let index = 0; index < args.length; index += 1) {
    const key = args[index]!;
    if (key === "--rebuild-counts") {
      values.set("rebuild-counts", "true");
      continue;
    }
    const value = args[index + 1];
    if (!key.startsWith("--") || !value || value.startsWith("--")) {
      throw new Error(`Invalid argument near ${key}.`);
    }
    values.set(key.slice(2), value);
    index += 1;
  }
  return values;
}

function writeBinary(file: string, value: unknown): void {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, serialize(value));
}

function readJson<T>(file: string): T {
  return JSON.parse(fs.readFileSync(file, "utf8")) as T;
}

function resolve(file: string): string {
  return path.resolve(repoRoot, file);
}

function metric(value: number): string {
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
