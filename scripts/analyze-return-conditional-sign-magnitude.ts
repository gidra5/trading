import fs from "node:fs";
import path from "node:path";
import { deserialize, serialize } from "node:v8";
import { fileURLToPath, pathToFileURL } from "node:url";
import { readCandleShardReferenceSync } from "@trading/storage";

const DAY_MS = 86_400_000;
const TARGET_RESOLUTIONS = [16, 32, 64] as const;
const PRIMARY_RESOLUTION = 32;
const SMOOTHING = 0.5;
const DEFAULT_ANALYSIS = "data/benchmarks/log-return-distributions.json";
const DEFAULT_MAGNITUDE = "data/benchmarks/one-second-sign-magnitude-dependence.json";
const DEFAULT_CACHE = "data/runtime-cache/one-second-conditional-sign-magnitude-counts.bin";
const DEFAULT_OUTPUT = "data/benchmarks/one-second-conditional-sign-magnitude-dependence.json";
const DEFAULT_REPORT = "docs/experiments/one-second-conditional-sign-magnitude-dependence-2026-08-16.md";

interface Options {
  analysisPath: string;
  magnitudePath: string;
  cachePath: string;
  outputPath: string;
  reportPath: string;
  rebuildCache: boolean;
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

interface MagnitudeResult {
  generatedAt: string;
  byResolution: Record<string, { edgesBps: Array<number | null> }>;
}

interface HistorySpec {
  id: string;
  label: string;
  description: string;
  states: number;
  minimumPastReturns: number;
}

const HISTORY_SPECS: HistorySpec[] = [
  {
    id: "none",
    label: "No history",
    description: "Unconditional sign-magnitude dependence baseline.",
    states: 1,
    minimumPastReturns: 0,
  },
  {
    id: "previousSign",
    label: "Previous sign",
    description: "Previous return is negative, exactly zero, or positive.",
    states: 3,
    minimumPastReturns: 1,
  },
  {
    id: "last3Signs",
    label: "Last 3 signs",
    description: "Ordered ternary sign/zero history of the previous three returns.",
    states: 3 ** 3,
    minimumPastReturns: 3,
  },
  {
    id: "last5Signs",
    label: "Last 5 signs",
    description: "Ordered ternary sign/zero history of the previous five returns.",
    states: 3 ** 5,
    minimumPastReturns: 5,
  },
  {
    id: "previousReturn17",
    label: "Previous return (17 states)",
    description: "Previous return: zero or sign crossed with eight magnitude cells.",
    states: 17,
    minimumPastReturns: 1,
  },
  {
    id: "last2Returns9",
    label: "Last 2 returns (9 states each)",
    description: "Ordered two-return history: zero or sign crossed with four magnitude cells.",
    states: 9 ** 2,
    minimumPastReturns: 2,
  },
  {
    id: "last3Returns9",
    label: "Last 3 returns (9 states each)",
    description: "Ordered three-return history: zero or sign crossed with four magnitude cells.",
    states: 9 ** 3,
    minimumPastReturns: 3,
  },
];

interface CountTable {
  byResolution: Record<string, Float64Array>;
  annualByResolution: Record<string, Float64Array[]>;
}

interface ConditionalCounts {
  tables: Record<string, CountTable>;
  returns: number;
  activeTargets: number;
  zeroTargets: number;
}

export interface ConditionalMetrics {
  observations: number;
  historyStates: number;
  occupiedHistoryStates: number;
  magnitudeBins: number;
  conditionalMutualInformationBits: number;
  conditionalMutualInformationNullBiasBits: number;
  conditionalMutualInformationAboveBiasBits: number;
  conditionalSignEntropyBits: number;
  fractionConditionalSignEntropyExplainedAboveBias: number;
  jsFromConditionalIndependentProductBits: number;
  totalVariationFromConditionalIndependentProduct: number;
  inSampleHistoryOnlySignAccuracy: number;
  inSampleHistoryAndMagnitudeSignAccuracy: number;
  inSampleSignAccuracyGain: number;
}

interface RollingResult {
  testObservations: number;
  signLogLossGainBitsPerReturn: number;
  magnitudeLogLossGainBitsPerReturn: number;
  historyOnlySignAccuracy: number;
  historyAndMagnitudeSignAccuracy: number;
  signAccuracyGain: number;
  annual: Array<{
    index: number;
    testObservations: number;
    signLogLossGainBitsPerReturn: number;
    magnitudeLogLossGainBitsPerReturn: number;
    historyOnlySignAccuracy: number;
    historyAndMagnitudeSignAccuracy: number;
    signAccuracyGain: number;
  }>;
}

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function main(): void {
  const options = parseOptions(process.argv.slice(2));
  const analysis = readJson<AnalysisReport>(options.analysisPath);
  const magnitude = readJson<MagnitudeResult>(options.magnitudePath);
  const targetEdges = Object.fromEntries(TARGET_RESOLUTIONS.map((resolution) => [
    String(resolution),
    finiteInternalEdges(magnitude.byResolution[String(resolution)]!.edgesBps),
  ])) as Record<string, number[]>;
  const return9Edges = coarsenEdges(
    finiteInternalEdges(magnitude.byResolution["8"]!.edgesBps),
    2,
  );
  const return17Edges = finiteInternalEdges(magnitude.byResolution["8"]!.edgesBps);
  const start = Date.parse(analysis.fullHistory.startTime);
  const end = Date.parse(analysis.fullHistory.endTime);
  const files = selectedFiles(
    path.resolve(repoRoot, analysis.source.oneSecond.referenceDirectory),
    start,
    end,
  );
  const metadata = JSON.stringify({
    version: 1,
    analysisGeneratedAt: analysis.generatedAt,
    magnitudeGeneratedAt: magnitude.generatedAt,
    files: files.map((entry) => path.basename(entry.file)),
    histories: HISTORY_SPECS,
    targetEdges,
    return9Edges,
    return17Edges,
  });
  const counts = loadOrBuildCounts(
    options,
    metadata,
    files,
    start,
    targetEdges,
    return9Edges,
    return17Edges,
  );
  const artifact = analyzeCounts(analysis, counts);
  fs.mkdirSync(path.dirname(options.outputPath), { recursive: true });
  fs.writeFileSync(options.outputPath, `${JSON.stringify(artifact, null, 2)}\n`, "utf8");
  fs.mkdirSync(path.dirname(options.reportPath), { recursive: true });
  fs.writeFileSync(options.reportPath, renderReport(artifact), "utf8");
  console.log(`Wrote ${path.relative(repoRoot, options.outputPath)}`);
  console.log(`Wrote ${path.relative(repoRoot, options.reportPath)}`);
}

function parseOptions(args: string[]): Options {
  const values = new Map<string, string>();
  let rebuildCache = false;
  for (let index = 0; index < args.length; index += 1) {
    const key = args[index]!;
    if (key === "--rebuild-cache") {
      rebuildCache = true;
      continue;
    }
    const value = args[index + 1];
    if (!key.startsWith("--") || !value || value.startsWith("--")) {
      throw new Error(`Invalid argument near ${key}.`);
    }
    values.set(key.slice(2), value);
    index += 1;
  }
  return {
    analysisPath: path.resolve(repoRoot, values.get("analysis") ?? DEFAULT_ANALYSIS),
    magnitudePath: path.resolve(repoRoot, values.get("magnitude") ?? DEFAULT_MAGNITUDE),
    cachePath: path.resolve(repoRoot, values.get("cache") ?? DEFAULT_CACHE),
    outputPath: path.resolve(repoRoot, values.get("output") ?? DEFAULT_OUTPUT),
    reportPath: path.resolve(repoRoot, values.get("report") ?? DEFAULT_REPORT),
    rebuildCache,
  };
}

function readJson<T>(file: string): T {
  return JSON.parse(fs.readFileSync(file, "utf8")) as T;
}

function finiteInternalEdges(edges: Array<number | null>): number[] {
  return edges.slice(1, -1).map((value) => {
    if (value === null || !Number.isFinite(value)) throw new Error("Internal magnitude edge is not finite.");
    return value;
  });
}

function coarsenEdges(edges: number[], stride: number): number[] {
  return edges.filter((_, index) => (index + 1) % stride === 0);
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
  if (result.length === 0) throw new Error("No complete one-second shards selected.");
  for (let index = 1; index < result.length; index += 1) {
    if (result[index]!.dayStart !== result[index - 1]!.dayStart + DAY_MS) {
      throw new Error(`Missing one-second shard before ${new Date(result[index]!.dayStart).toISOString()}.`);
    }
  }
  return result;
}

function loadOrBuildCounts(
  options: Options,
  metadata: string,
  files: Array<{ file: string; dayStart: number }>,
  start: number,
  targetEdges: Record<string, number[]>,
  return9Edges: number[],
  return17Edges: number[],
): ConditionalCounts {
  if (!options.rebuildCache && fs.existsSync(options.cachePath)) {
    const cached = deserialize(fs.readFileSync(options.cachePath)) as {
      metadata: string;
      counts: ConditionalCounts;
    };
    if (cached.metadata === metadata) {
      console.log(`Loading cached counts from ${path.relative(repoRoot, options.cachePath)}`);
      return cached.counts;
    }
  }
  const counts = streamCounts(files, start, targetEdges, return9Edges, return17Edges);
  fs.mkdirSync(path.dirname(options.cachePath), { recursive: true });
  fs.writeFileSync(options.cachePath, serialize({ metadata, counts }));
  return counts;
}

function createTables(): Record<string, CountTable> {
  return Object.fromEntries(HISTORY_SPECS.map((spec) => [spec.id, {
    byResolution: Object.fromEntries(TARGET_RESOLUTIONS.map((resolution) => [
      String(resolution),
      new Float64Array(spec.states * 2 * resolution),
    ])),
    annualByResolution: Object.fromEntries(TARGET_RESOLUTIONS.map((resolution) => [
      String(resolution),
      [],
    ])),
  }])) as Record<string, CountTable>;
}

function streamCounts(
  files: Array<{ file: string; dayStart: number }>,
  start: number,
  targetEdges: Record<string, number[]>,
  return9Edges: number[],
  return17Edges: number[],
): ConditionalCounts {
  const tables = createTables();
  let previousClose = Number.NaN;
  let historyCount = 0;
  let lastSign = 0;
  let lastReturn17 = 0;
  let signCode3 = 0;
  let signCode5 = 0;
  let return9Code2 = 0;
  let return9Code3 = 0;
  let returns = 0;
  let activeTargets = 0;
  let zeroTargets = 0;

  for (const [fileIndex, entry] of files.entries()) {
    if (fileIndex % 100 === 0) {
      console.error(`Reading conditional sign-magnitude ${fileIndex}/${files.length}...`);
    }
    const year = anniversaryIndex(start, entry.dayStart);
    ensureAnnualTables(tables, year);
    const candles = readCandleShardReferenceSync(entry.file);
    if (candles.length !== 86_400) throw new Error(`${path.basename(entry.file)} is incomplete.`);
    for (const candle of candles) {
      if (!Number.isFinite(candle.close) || candle.close <= 0) {
        throw new Error(`${path.basename(entry.file)} contains an invalid close.`);
      }
      if (Number.isFinite(previousClose)) {
        returns += 1;
        let signState: number;
        let return9State: number;
        let return17State: number;
        if (candle.close === previousClose) {
          zeroTargets += 1;
          signState = 1;
          return9State = 0;
          return17State = 0;
        } else {
          activeTargets += 1;
          const returnBps = Math.log(candle.close / previousClose) * 10_000;
          const magnitude = Math.abs(returnBps);
          const targetSign = returnBps > 0 ? 1 : 0;
          signState = returnBps > 0 ? 2 : 0;
          return9State = 1 + targetSign * 4 + upperBound(return9Edges, magnitude);
          return17State = 1 + targetSign * 8 + upperBound(return17Edges, magnitude);
          const histories: Record<string, number> = {
            none: 0,
            previousSign: lastSign,
            last3Signs: signCode3,
            last5Signs: signCode5,
            previousReturn17: lastReturn17,
            last2Returns9: return9Code2,
            last3Returns9: return9Code3,
          };
          for (const spec of HISTORY_SPECS) {
            if (historyCount < spec.minimumPastReturns) continue;
            const history = histories[spec.id]!;
            const table = tables[spec.id]!;
            for (const resolution of TARGET_RESOLUTIONS) {
              const magnitudeBin = upperBound(targetEdges[String(resolution)]!, magnitude);
              const index = (history * 2 + targetSign) * resolution + magnitudeBin;
              table.byResolution[String(resolution)]![index] += 1;
              table.annualByResolution[String(resolution)]![year]![index] += 1;
            }
          }
        }
        signCode3 = appendRolling(signCode3, signState, 3, 3);
        signCode5 = appendRolling(signCode5, signState, 3, 5);
        return9Code2 = appendRolling(return9Code2, return9State, 9, 2);
        return9Code3 = appendRolling(return9Code3, return9State, 9, 3);
        lastSign = signState;
        lastReturn17 = return17State;
        historyCount += 1;
      }
      previousClose = candle.close;
    }
  }
  return { tables, returns, activeTargets, zeroTargets };
}

function ensureAnnualTables(tables: Record<string, CountTable>, year: number): void {
  for (const spec of HISTORY_SPECS) {
    const table = tables[spec.id]!;
    for (const resolution of TARGET_RESOLUTIONS) {
      const annual = table.annualByResolution[String(resolution)]!;
      while (annual.length <= year) {
        annual.push(new Float64Array(spec.states * 2 * resolution));
      }
    }
  }
}

function appendRolling(code: number, state: number, base: number, length: number): number {
  return (code % (base ** (length - 1))) * base + state;
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

function anniversaryIndex(startMs: number, dayMs: number): number {
  const start = new Date(startMs);
  const day = new Date(dayMs);
  let index = day.getUTCFullYear() - start.getUTCFullYear();
  const anniversary = Date.UTC(
    start.getUTCFullYear() + index,
    start.getUTCMonth(),
    start.getUTCDate(),
  );
  if (dayMs < anniversary) index -= 1;
  return Math.max(0, index);
}

export function conditionalMetrics(
  counts: Float64Array,
  historyStates: number,
  magnitudeBins: number,
): ConditionalMetrics {
  let total = 0;
  for (const count of counts) total += count;
  let cmi = 0;
  let js = 0;
  let tv = 0;
  let nullDegrees = 0;
  let conditionalSignEntropy = 0;
  let historyOnlyCorrect = 0;
  let enhancedCorrect = 0;
  let occupiedHistoryStates = 0;
  for (let history = 0; history < historyStates; history += 1) {
    const signTotals = [0, 0];
    const magnitudeTotals = new Float64Array(magnitudeBins);
    let historyTotal = 0;
    for (let sign = 0; sign < 2; sign += 1) {
      for (let magnitude = 0; magnitude < magnitudeBins; magnitude += 1) {
        const count = counts[(history * 2 + sign) * magnitudeBins + magnitude]!;
        signTotals[sign]! += count;
        magnitudeTotals[magnitude] += count;
        historyTotal += count;
      }
    }
    if (historyTotal === 0) continue;
    occupiedHistoryStates += 1;
    historyOnlyCorrect += Math.max(signTotals[0]!, signTotals[1]!);
    const occupiedSigns = signTotals.filter((value) => value > 0).length;
    let occupiedMagnitudes = 0;
    for (const value of magnitudeTotals) if (value > 0) occupiedMagnitudes += 1;
    nullDegrees += Math.max(0, occupiedSigns - 1) * Math.max(0, occupiedMagnitudes - 1);
    for (const signTotal of signTotals) {
      if (signTotal > 0) {
        const probability = signTotal / historyTotal;
        conditionalSignEntropy -= historyTotal / total * probability * Math.log2(probability);
      }
    }
    for (let magnitude = 0; magnitude < magnitudeBins; magnitude += 1) {
      enhancedCorrect += Math.max(
        counts[(history * 2) * magnitudeBins + magnitude]!,
        counts[(history * 2 + 1) * magnitudeBins + magnitude]!,
      );
      for (let sign = 0; sign < 2; sign += 1) {
        const count = counts[(history * 2 + sign) * magnitudeBins + magnitude]!;
        const observed = count / total;
        const product = signTotals[sign]! * magnitudeTotals[magnitude]! / (historyTotal * total);
        if (observed > 0) {
          cmi += observed * Math.log2(count * historyTotal
            / (signTotals[sign]! * magnitudeTotals[magnitude]!));
        }
        const mixture = (observed + product) / 2;
        if (observed > 0) js += 0.5 * observed * Math.log2(observed / mixture);
        if (product > 0) js += 0.5 * product * Math.log2(product / mixture);
        tv += 0.5 * Math.abs(observed - product);
      }
    }
  }
  const bias = nullDegrees / (2 * total * Math.LN2);
  const corrected = Math.max(0, cmi - bias);
  return {
    observations: total,
    historyStates,
    occupiedHistoryStates,
    magnitudeBins,
    conditionalMutualInformationBits: cmi,
    conditionalMutualInformationNullBiasBits: bias,
    conditionalMutualInformationAboveBiasBits: corrected,
    conditionalSignEntropyBits: conditionalSignEntropy,
    fractionConditionalSignEntropyExplainedAboveBias: corrected / conditionalSignEntropy,
    jsFromConditionalIndependentProductBits: js,
    totalVariationFromConditionalIndependentProduct: tv,
    inSampleHistoryOnlySignAccuracy: historyOnlyCorrect / total,
    inSampleHistoryAndMagnitudeSignAccuracy: enhancedCorrect / total,
    inSampleSignAccuracyGain: (enhancedCorrect - historyOnlyCorrect) / total,
  };
}

export function rollingEvaluation(
  annual: Float64Array[],
  historyStates: number,
  magnitudeBins: number,
): RollingResult {
  const train = new Float64Array(historyStates * 2 * magnitudeBins);
  const yearly: RollingResult["annual"] = [];
  for (let year = 0; year < annual.length; year += 1) {
    const test = annual[year]!;
    if (year > 0) yearly.push(evaluateHoldout(train, test, historyStates, magnitudeBins, year));
    for (let index = 0; index < train.length; index += 1) train[index] += test[index]!;
  }
  const testObservations = yearly.reduce((sum, row) => sum + row.testObservations, 0);
  return {
    testObservations,
    signLogLossGainBitsPerReturn: weightedMean(yearly, "signLogLossGainBitsPerReturn"),
    magnitudeLogLossGainBitsPerReturn: weightedMean(yearly, "magnitudeLogLossGainBitsPerReturn"),
    historyOnlySignAccuracy: weightedMean(yearly, "historyOnlySignAccuracy"),
    historyAndMagnitudeSignAccuracy: weightedMean(yearly, "historyAndMagnitudeSignAccuracy"),
    signAccuracyGain: weightedMean(yearly, "signAccuracyGain"),
    annual: yearly,
  };
}

function evaluateHoldout(
  train: Float64Array,
  test: Float64Array,
  historyStates: number,
  magnitudeBins: number,
  index: number,
) {
  let testObservations = 0;
  let signGain = 0;
  let magnitudeGain = 0;
  let historyOnlyCorrect = 0;
  let enhancedCorrect = 0;
  for (let history = 0; history < historyStates; history += 1) {
    const trainSign = [0, 0];
    const trainMagnitude = new Float64Array(magnitudeBins);
    let trainHistory = 0;
    for (let sign = 0; sign < 2; sign += 1) {
      for (let magnitude = 0; magnitude < magnitudeBins; magnitude += 1) {
        const count = train[(history * 2 + sign) * magnitudeBins + magnitude]!;
        trainSign[sign]! += count;
        trainMagnitude[magnitude] += count;
        trainHistory += count;
      }
    }
    // Smooth the joint H x S x M table, then derive every marginal from that
    // one coherent distribution. This preserves the Bayes identity
    // P(S|H,M)/P(S|H) = P(M|H,S)/P(M|H) exactly, including empty cells.
    const historyPositive = (
      trainSign[1]! + magnitudeBins * SMOOTHING
    ) / (trainHistory + 2 * magnitudeBins * SMOOTHING);
    const historyChoice = historyPositive >= 0.5 ? 1 : 0;
    for (let magnitude = 0; magnitude < magnitudeBins; magnitude += 1) {
      const trainMagnitudeTotal = trainMagnitude[magnitude]!;
      const enhancedPositive = (
        train[(history * 2 + 1) * magnitudeBins + magnitude]! + SMOOTHING
      ) / (trainMagnitudeTotal + 2 * SMOOTHING);
      const enhancedChoice = enhancedPositive >= 0.5 ? 1 : 0;
      for (let sign = 0; sign < 2; sign += 1) {
        const position = (history * 2 + sign) * magnitudeBins + magnitude;
        const testCount = test[position]!;
        if (testCount === 0) continue;
        testObservations += testCount;
        if (historyChoice === sign) historyOnlyCorrect += testCount;
        if (enhancedChoice === sign) enhancedCorrect += testCount;
        const historySignProbability = (
          trainSign[sign]! + magnitudeBins * SMOOTHING
        ) / (trainHistory + 2 * magnitudeBins * SMOOTHING);
        const enhancedSignProbability = (
          train[position]! + SMOOTHING
        ) / (trainMagnitudeTotal + 2 * SMOOTHING);
        const historyMagnitudeProbability = (
          trainMagnitudeTotal + 2 * SMOOTHING
        ) / (trainHistory + 2 * magnitudeBins * SMOOTHING);
        const enhancedMagnitudeProbability = (
          train[position]! + SMOOTHING
        ) / (trainSign[sign]! + magnitudeBins * SMOOTHING);
        signGain += testCount * Math.log2(enhancedSignProbability / historySignProbability);
        magnitudeGain += testCount * Math.log2(
          enhancedMagnitudeProbability / historyMagnitudeProbability,
        );
      }
    }
  }
  return {
    index,
    testObservations,
    signLogLossGainBitsPerReturn: signGain / testObservations,
    magnitudeLogLossGainBitsPerReturn: magnitudeGain / testObservations,
    historyOnlySignAccuracy: historyOnlyCorrect / testObservations,
    historyAndMagnitudeSignAccuracy: enhancedCorrect / testObservations,
    signAccuracyGain: (enhancedCorrect - historyOnlyCorrect) / testObservations,
  };
}

function weightedMean(
  rows: RollingResult["annual"],
  key: "signLogLossGainBitsPerReturn" | "magnitudeLogLossGainBitsPerReturn"
    | "historyOnlySignAccuracy" | "historyAndMagnitudeSignAccuracy" | "signAccuracyGain",
): number {
  const observations = rows.reduce((sum, row) => sum + row.testObservations, 0);
  return rows.reduce((sum, row) => sum + row[key] * row.testObservations, 0) / observations;
}

function analyzeCounts(analysis: AnalysisReport, counts: ConditionalCounts) {
  const histories = Object.fromEntries(HISTORY_SPECS.map((spec) => {
    const table = counts.tables[spec.id]!;
    const byResolution = Object.fromEntries(TARGET_RESOLUTIONS.map((resolution) => [
      String(resolution),
      conditionalMetrics(table.byResolution[String(resolution)]!, spec.states, resolution),
    ]));
    const rolling = rollingEvaluation(
      table.annualByResolution[String(PRIMARY_RESOLUTION)]!,
      spec.states,
      PRIMARY_RESOLUTION,
    );
    return [spec.id, { ...spec, byResolution, primary: byResolution[String(PRIMARY_RESOLUTION)], rolling }];
  }));
  const strongest = HISTORY_SPECS
    .filter((spec) => spec.id !== "none")
    .map((spec) => histories[spec.id])
    .sort((left, right) =>
      right.primary.conditionalMutualInformationAboveBiasBits
      - left.primary.conditionalMutualInformationAboveBiasBits)[0];
  const previousSignTable = counts.tables.previousSign!;
  const previousSignStateBreakdown = ["Previous negative", "Previous zero", "Previous positive"]
    .map((label, state) => {
      const global = sliceHistoryState(
        previousSignTable.byResolution[String(PRIMARY_RESOLUTION)]!,
        state,
        PRIMARY_RESOLUTION,
      );
      const annual = previousSignTable.annualByResolution[String(PRIMARY_RESOLUTION)]!
        .map((table) => sliceHistoryState(table, state, PRIMARY_RESOLUTION));
      return {
        state,
        label,
        metrics: conditionalMetrics(global, 1, PRIMARY_RESOLUTION),
        direct: singleStateDiagnostics(global, PRIMARY_RESOLUTION),
        rolling: rollingEvaluation(annual, 1, PRIMARY_RESOLUTION),
      };
    });
  return {
    version: 1,
    generatedAt: new Date().toISOString(),
    symbol: analysis.source.symbol,
    scale: "1s",
    window: analysis.fullHistory,
    source: {
      analysis: DEFAULT_ANALYSIS,
      unconditionalSignMagnitude: DEFAULT_MAGNITUDE,
      referenceDirectory: analysis.source.oneSecond.referenceDirectory,
    },
    returns: counts.returns,
    activeTargets: counts.activeTargets,
    zeroTargets: counts.zeroTargets,
    targetMagnitudeResolutions: TARGET_RESOLUTIONS,
    primaryMagnitudeResolution: PRIMARY_RESOLUTION,
    definitions: {
      conditionalIndependence: "P(sign,magnitude|history,active)=P(sign|history,active)P(magnitude|history,active).",
      conditionalMutualInformation: "I(sign;magnitude|history,active), in bits per active target return.",
      rollingSignGain: "Held-out log2 P(sign|history,magnitude)-log2 P(sign|history), fitted only on prior annual windows.",
      rollingMagnitudeGain: "Held-out log2 P(magnitude|history,sign)-log2 P(magnitude|history), fitted only on prior annual windows.",
    },
    histories,
    previousSignStateBreakdown,
    strongestInSampleHistory: strongest.id,
    conclusion: classifyResult(histories),
    limitations: [
      "Conditional independence is relative to the finite past-only state representations tested; unmeasured history can change the result.",
      "The in-sample conditional-mutual-information bias correction is first-order; rolling prior-year holdouts are the primary stability control.",
      "All target and history magnitude edges were fitted on the full-history descriptive distribution. Rolling scores fit conditional probabilities only on prior years, but a production evaluation should also refit edges on each prefix.",
      "This measures same-return sign-magnitude coupling after conditioning on history, not predictability of the realized return itself.",
    ],
  };
}

function sliceHistoryState(
  counts: Float64Array,
  history: number,
  magnitudeBins: number,
): Float64Array {
  const start = history * 2 * magnitudeBins;
  return counts.slice(start, start + 2 * magnitudeBins);
}

function singleStateDiagnostics(counts: Float64Array, magnitudeBins: number) {
  const signTotals = [0, 0];
  for (let sign = 0; sign < 2; sign += 1) {
    for (let magnitude = 0; magnitude < magnitudeBins; magnitude += 1) {
      signTotals[sign]! += counts[sign * magnitudeBins + magnitude]!;
    }
  }
  let magnitudeLawTvBetweenSigns = 0;
  let minimumPositiveProbabilityByMagnitude = 1;
  let maximumPositiveProbabilityByMagnitude = 0;
  for (let magnitude = 0; magnitude < magnitudeBins; magnitude += 1) {
    const negative = counts[magnitude]!;
    const positive = counts[magnitudeBins + magnitude]!;
    magnitudeLawTvBetweenSigns += 0.5 * Math.abs(
      negative / signTotals[0]! - positive / signTotals[1]!,
    );
    const total = negative + positive;
    if (total > 0) {
      const probability = positive / total;
      minimumPositiveProbabilityByMagnitude = Math.min(
        minimumPositiveProbabilityByMagnitude,
        probability,
      );
      maximumPositiveProbabilityByMagnitude = Math.max(
        maximumPositiveProbabilityByMagnitude,
        probability,
      );
    }
  }
  return {
    positiveProbability: signTotals[1]! / (signTotals[0]! + signTotals[1]!),
    magnitudeLawTvBetweenSigns,
    minimumPositiveProbabilityByMagnitude,
    maximumPositiveProbabilityByMagnitude,
  };
}

function classifyResult(histories: Record<string, any>): string {
  const rows = HISTORY_SPECS.map((spec) => histories[spec.id]);
  const maximumCmi = Math.max(...rows.map((row) =>
    row.primary.conditionalMutualInformationAboveBiasBits));
  const maximumRollingGain = Math.max(...rows.map((row) =>
    row.rolling.signLogLossGainBitsPerReturn));
  if (maximumCmi < 1e-5 && maximumRollingGain <= 0) {
    return "No stable conditional sign-magnitude dependence is resolved by the tested histories.";
  }
  if (maximumCmi < 1e-4 && maximumRollingGain < 1e-4) {
    return "The tested histories leave a measurable but very weak sign-magnitude coupling.";
  }
  return "At least one tested history reveals material conditional sign-magnitude coupling.";
}

function anniversaryIso(startValue: string, offset: number): string {
  const start = new Date(startValue);
  return new Date(Date.UTC(
    start.getUTCFullYear() + offset,
    start.getUTCMonth(),
    start.getUTCDate(),
  )).toISOString();
}

function renderReport(artifact: ReturnType<typeof analyzeCounts>): string {
  const lines = [
    "# Conditional one-second sign–magnitude dependence",
    "",
    `Generated ${artifact.generatedAt}. The analysis uses ${artifact.activeTargets.toLocaleString("en-US")} active BTCUSDT one-second target returns from ${artifact.window.startTime} through ${artifact.window.endTime}. Every history state uses only returns strictly before its target.`,
    "",
    "## Result",
    "",
    artifact.conclusion,
    "",
    "Conditional independence requires both `P(sign | magnitude, history) = P(sign | history)` and `P(magnitude | sign, history) = P(magnitude | history)`. Conditional mutual information measures these equivalent failures in sample; rolling gains test whether they persist into later years.",
    "",
    "## History comparison at 32 magnitude cells",
    "",
    "| history | states | CMI (bits) | CMI above bias | conditional TV | sign entropy explained | rolling sign gain (bits) | rolling magnitude gain (bits) | rolling sign-accuracy gain |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
  ];
  for (const spec of HISTORY_SPECS) {
    const row = artifact.histories[spec.id]!;
    const metrics = row.primary;
    const rolling = row.rolling;
    lines.push(
      `| ${spec.label} | ${spec.states} | ${formatMetric(metrics.conditionalMutualInformationBits)} | ${formatMetric(metrics.conditionalMutualInformationAboveBiasBits)} | ${formatMetric(metrics.totalVariationFromConditionalIndependentProduct)} | ${(metrics.fractionConditionalSignEntropyExplainedAboveBias * 100).toFixed(6)}% | ${formatMetric(rolling.signLogLossGainBitsPerReturn)} | ${formatMetric(rolling.magnitudeLogLossGainBitsPerReturn)} | ${(rolling.signAccuracyGain * 100).toFixed(6)} pp |`,
    );
  }
  const strongest = artifact.histories[artifact.strongestInSampleHistory]!;
  lines.push(
    "",
    `## Resolution robustness for ${strongest.label}`,
    "",
    "| magnitude cells | CMI (bits) | null bias | CMI above bias | conditional TV | sign entropy explained |",
    "|---:|---:|---:|---:|---:|---:|",
  );
  for (const [resolution, metrics] of Object.entries(strongest.byResolution)) {
    lines.push(
      `| ${resolution} | ${formatMetric(metrics.conditionalMutualInformationBits)} | ${formatMetric(metrics.conditionalMutualInformationNullBiasBits)} | ${formatMetric(metrics.conditionalMutualInformationAboveBiasBits)} | ${formatMetric(metrics.totalVariationFromConditionalIndependentProduct)} | ${(metrics.fractionConditionalSignEntropyExplainedAboveBias * 100).toFixed(6)}% |`,
    );
  }
  lines.push(
    "",
    "## Previous-sign state breakdown",
    "",
    "This separates the three states inside the simplest nontrivial history. The target itself is always active; `Previous zero` means only that the immediately preceding one-second return was zero.",
    "",
    "| history state | active targets | P(positive) | P(positive \u007c magnitude) range | TV of magnitude laws by sign | CMI above bias | rolling log-score gain |",
    "|---|---:|---:|---:|---:|---:|---:|",
  );
  for (const row of artifact.previousSignStateBreakdown) {
    lines.push(
      `| ${row.label} | ${row.metrics.observations.toLocaleString("en-US")} | ${(row.direct.positiveProbability * 100).toFixed(4)}% | ${(row.direct.minimumPositiveProbabilityByMagnitude * 100).toFixed(3)}%–${(row.direct.maximumPositiveProbabilityByMagnitude * 100).toFixed(3)}% | ${(row.direct.magnitudeLawTvBetweenSigns * 100).toFixed(4)}% | ${formatMetric(row.metrics.conditionalMutualInformationAboveBiasBits)} | ${formatMetric(row.rolling.signLogLossGainBitsPerReturn)} |`,
    );
  }
  lines.push(
    "",
    "## Rolling annual holdouts",
    "",
    "Each annual target window uses all preceding annual windows to estimate its conditional tables. Positive gains favor coupling sign and magnitude; negative gains favor conditional factorization.",
    "",
  );
  for (const spec of HISTORY_SPECS) {
    const row = artifact.histories[spec.id]!;
    lines.push(
      `### ${spec.label}`,
      "",
      "| test year index | active targets | sign log-loss gain (bits) | magnitude log-loss gain (bits) | sign-accuracy gain |",
      "|---:|---:|---:|---:|---:|",
    );
    for (const annual of row.rolling.annual) {
      lines.push(
        `| ${annual.index} | ${annual.testObservations.toLocaleString("en-US")} | ${formatMetric(annual.signLogLossGainBitsPerReturn)} | ${formatMetric(annual.magnitudeLogLossGainBitsPerReturn)} | ${(annual.signAccuracyGain * 100).toFixed(6)} pp |`,
      );
    }
    lines.push("");
  }
  lines.push(
    "## History-state definitions",
    "",
  );
  for (const spec of HISTORY_SPECS) lines.push(`- **${spec.label}:** ${spec.description}`);
  lines.push(
    "",
    "## Reproducibility",
    "",
    "```text",
    "node --conditions=development --import tsx scripts/analyze-return-conditional-sign-magnitude.ts",
    "```",
    "",
    "The complete history/resolution metrics and annual rolling evaluations are stored in `data/benchmarks/one-second-conditional-sign-magnitude-dependence.json`.",
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
