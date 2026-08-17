import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  readCandleShardReferenceSync,
  TradingStorageLayout,
  type SequentialCandle,
} from "@trading/storage";
import { aggregateLogReturns, type TimedReturn } from "./lib/log-return-distribution.js";
import { STANDARDIZED_PAIR_EDGES } from "./lib/consecutive-return-pairs.js";
import {
  fitTripleCandidates,
  fitTripleGeneralizedGaussianPower,
  projectionJensenShannonBits,
  summarizeTripleShape,
  TripleMoments,
  type TripleCandidateFit,
  type TripleMomentsSnapshot,
  type TripleSample,
  type TripleShapeStatistics,
} from "./lib/consecutive-return-triples.js";

const DAY_MS = 86_400_000;
const YEAR_MS = 365 * DAY_MS;
const SCALES = [
  { id: "1s", label: "1 second", intervalMs: 1_000 },
  { id: "1m", label: "1 minute", intervalMs: 60_000 },
  { id: "15m", label: "15 minutes", intervalMs: 15 * 60_000 },
  { id: "1h", label: "1 hour", intervalMs: 60 * 60_000 },
  { id: "4h", label: "4 hours", intervalMs: 4 * 60 * 60_000 },
  { id: "1d", label: "1 day", intervalMs: DAY_MS },
] as const;

interface Options {
  dataDir: string;
  outputPath: string;
  reportPath: string;
  requestedEndTime?: number;
  sampleTarget: number;
  fitSampleLimit: number;
  visualizationSampleLimit: number;
}

interface WindowDefinition {
  id: string;
  label: string;
  kind: "full" | "annual-epoch" | "trailing";
  startTime: number;
  endTime: number;
}

interface WindowAccumulator {
  definition: WindowDefinition;
  moments: TripleMoments;
  sample: TripleSample;
  samplingProbability: number;
}

interface WindowScaleResult {
  id: string;
  label: string;
  kind: WindowDefinition["kind"];
  startTime: string;
  endTime: string;
  durationDays: number;
  observations: number;
  meansBps: [number, number, number];
  standardDeviationsBps: [number, number, number];
  lag1ReturnCorrelation: number;
  lag2ReturnCorrelation: number;
  lag1AbsoluteReturnCorrelation: number;
  lag2AbsoluteReturnCorrelation: number;
  lag1SquaredReturnCorrelation: number;
  lag2SquaredReturnCorrelation: number;
  allZeroFraction: number;
  anyZeroFraction: number;
  nonzeroAllSameSignFraction: number | null;
  signPatterns: number[];
  continuousObservations: number;
  sampleObservations: number;
  samplingProbability: number;
  shape: TripleShapeStatistics;
  generalizedGaussianApproximation: {
    family: "trivariate-generalized-gaussian";
    power: number;
    scale: number;
    observations: number;
  };
  similarityToFull?: {
    meanProjectionJensenShannonBits: number;
  };
  sampleWarning: string | null;
}

interface ScaleResult {
  id: string;
  label: string;
  intervalMs: number;
  fullHistory: WindowScaleResult;
  annualEpochs: WindowScaleResult[];
  trailingWindows: WindowScaleResult[];
  fullHistoryModelFits: TripleCandidateFit[];
  selectedFullHistoryModel: TripleCandidateFit;
}

interface AnalysisReport {
  version: number;
  generatedAt: string;
  source: {
    market: "spot-btcusdt";
    symbol: "BTCUSDT";
    analysisStartTime: string;
    analysisEndTime: string;
    durationDays: number;
    oneSecond: SourceDescription;
    oneMinute: SourceDescription;
  };
  methodology: Record<string, string | number | string[]>;
  histogram: {
    standardizedEdges: number[];
    layout: string;
  };
  scales: ScaleResult[];
  crossScaleSimilarity: Array<{
    leftScale: string;
    rightScale: string;
    meanProjectionJensenShannonBits: number;
  }>;
  annualWindowStability: Array<{
    scale: string;
    lag1AbsoluteCorrelationRange: [number, number] | null;
    lag2AbsoluteCorrelationRange: [number, number] | null;
    generalizedGaussianPowerRange: [number, number] | null;
    projectionJensenShannonBitsMedian: number;
    projectionJensenShannonBitsMaximum: number;
  }>;
}

interface SourceDescription {
  interval: "1s" | "1m";
  referenceDirectory: string;
  files: number;
  candles: number;
  firstCandleOpenTime: string;
  lastCandleEndTime: string;
}

interface OneSecondSource {
  referenceDirectory: string;
  files: string[];
  firstCandleOpenTime: number;
  lastCandleEndTime: number;
}

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.stack ?? error.message : String(error));
  process.exitCode = 1;
});

async function main(): Promise<void> {
  const options = parseOptions(process.argv.slice(2));
  const minute = loadMinuteCandles(options.dataDir);
  const minuteCandleCount = minute.candles.length;
  const oneSecond = discoverOneSecondSource(options.dataDir);
  const latestCommonDay = Math.floor(
    Math.min(minute.lastCandleEndTime, oneSecond.lastCandleEndTime) / DAY_MS,
  ) * DAY_MS;
  const analysisEnd = options.requestedEndTime ?? latestCommonDay;
  if (analysisEnd > latestCommonDay || analysisEnd % DAY_MS !== 0) {
    throw new Error("Analysis end must be an available complete UTC day boundary.");
  }
  const analysisStart = subtractUtcYears(analysisEnd, 5);
  if (analysisStart < minute.firstCandleOpenTime
    || analysisStart < oneSecond.firstCandleOpenTime) {
    throw new Error("The local candle store does not cover the requested five-year window.");
  }
  const definitions = buildWindowDefinitions(analysisStart, analysisEnd);
  const scales: ScaleResult[] = [];
  console.error(`Analyzing triples over ${dateOnly(analysisStart)}..${dateOnly(analysisEnd)}.`);
  for (const scale of SCALES) {
    console.error(`Preparing consecutive ${scale.id} return triples...`);
    const accumulators = createAccumulators(definitions, scale.intervalMs, options.sampleTarget);
    if (scale.id === "1s") {
      processOneSecondTriples(oneSecond, accumulators, analysisStart, analysisEnd);
    } else {
      processTimedReturnTriples(
        aggregateLogReturns(minute.candles, scale.intervalMs),
        accumulators,
        scale.intervalMs,
      );
    }
    const windows = accumulators.map((accumulator) => finalizeWindow(
      accumulator,
      options.fitSampleLimit,
      accumulator.definition.kind === "full" ? options.visualizationSampleLimit : 0,
    ));
    const fullHistory = windows.find((window) => window.kind === "full")!;
    for (const window of windows) {
      if (window === fullHistory) continue;
      window.similarityToFull = {
        meanProjectionJensenShannonBits: projectionJensenShannonBits(
          window.shape.projectionHistograms,
          fullHistory.shape.projectionHistograms,
        ),
      };
    }
    const fullAccumulator = accumulators.find((item) => item.definition.kind === "full")!;
    console.error(`Fitting candidate ${scale.id} trivariate distributions...`);
    const fits = fitTripleCandidates(
      fullAccumulator.sample,
      fullAccumulator.moments.snapshot(),
      options.fitSampleLimit,
    );
    scales.push({
      id: scale.id,
      label: scale.label,
      intervalMs: scale.intervalMs,
      fullHistory,
      annualEpochs: windows.filter((window) => window.kind === "annual-epoch"),
      trailingWindows: windows.filter((window) => window.kind === "trailing"),
      fullHistoryModelFits: fits,
      selectedFullHistoryModel: fits[0]!,
    });
  }
  minute.candles.length = 0;
  const crossScaleSimilarity = scales.slice(1).map((right, index) => ({
    leftScale: scales[index]!.id,
    rightScale: right.id,
    meanProjectionJensenShannonBits: projectionJensenShannonBits(
      scales[index]!.fullHistory.shape.projectionHistograms,
      right.fullHistory.shape.projectionHistograms,
    ),
  }));
  const annualWindowStability = scales.map((scale) => {
    const distances = scale.annualEpochs
      .map((window) => window.similarityToFull!.meanProjectionJensenShannonBits)
      .sort((left, right) => left - right);
    return {
      scale: scale.id,
      lag1AbsoluteCorrelationRange: finiteRange(
        scale.annualEpochs.map((window) => window.lag1AbsoluteReturnCorrelation),
      ),
      lag2AbsoluteCorrelationRange: finiteRange(
        scale.annualEpochs.map((window) => window.lag2AbsoluteReturnCorrelation),
      ),
      generalizedGaussianPowerRange: finiteRange(
        scale.annualEpochs.map((window) => window.generalizedGaussianApproximation.power),
      ),
      projectionJensenShannonBitsMedian: quantile(distances, 0.5),
      projectionJensenShannonBitsMaximum: distances.at(-1)!,
    };
  });
  const report: AnalysisReport = {
    version: 2,
    generatedAt: new Date().toISOString(),
    source: {
      market: "spot-btcusdt",
      symbol: "BTCUSDT",
      analysisStartTime: iso(analysisStart),
      analysisEndTime: iso(analysisEnd),
      durationDays: (analysisEnd - analysisStart) / DAY_MS,
      oneSecond: sourceDescription(oneSecond),
      oneMinute: {
        interval: "1m",
        referenceDirectory: relative(minute.referenceDirectory),
        files: minute.files,
        candles: minuteCandleCount,
        firstCandleOpenTime: iso(minute.firstCandleOpenTime),
        lastCandleEndTime: iso(minute.lastCandleEndTime),
      },
    },
    methodology: {
      target: "Ordered triple (r_t, r_{t+1}, r_{t+2}) of three consecutive close-to-close natural-log returns.",
      scales: SCALES.map((scale) => scale.id),
      alignment: "Native one-second closes and non-overlapping UTC-aligned 1m, 15m, 1h, 4h, and 1d closes.",
      gapPolicy: "All three returns must be adjacent and their endpoints strictly inside the window; gaps are never bridged.",
      windows: "One exact five-calendar-year window, five non-overlapping one-year epochs, and trailing 30d, 90d, and 365d windows.",
      zeroPolicy: "All-zero and any-zero masses are reported separately. Continuous densities are fitted only to triples where every return is nonzero.",
      standardization: "Each continuous component is centered and covariance-whitened within its window before radial fitting.",
      sampling: `All covariance, magnitude, squared-return, sign, and zero statistics are exact. Nonlinear tails, projection histograms, and fits use a deterministic uniform sample targeting ${options.sampleTarget.toLocaleString("en-US")} triples per window.`,
      fitting: `Maximum likelihood on up to ${options.fitSampleLimit.toLocaleString("en-US")} sampled continuous triples; AIC compares trivariate Gaussian, Student t, generalized Gaussian, generalized t, and radial-lognormal families.`,
      generalizedGaussianFormula: "f(z)=p/[4*pi*s^3*Gamma(3/p)]*exp(-(rho(z)/s)^p).",
      generalizedTFormula: "f(z)=p/[4*pi*s^3*B(3/p,q-3/p)]*[1+(rho(z)/s)^p]^(-q).",
      visualization: "Three pairwise standardized projections retain all coordinate pairs; the embedded point sample supports a rotatable 3D view.",
      modelLimit: "These are symmetric stationary unconditional families, not a volatility-state or predictive transition model.",
    },
    histogram: {
      standardizedEdges: [...STANDARDIZED_PAIR_EDGES],
      layout: "Each projection is row-major second-coordinate-by-first-coordinate probabilities with overflow bins.",
    },
    scales,
    crossScaleSimilarity,
    annualWindowStability,
  };
  fs.mkdirSync(path.dirname(options.outputPath), { recursive: true });
  fs.writeFileSync(options.outputPath, `${JSON.stringify(report, numberReplacer, 2)}\n`, "utf8");
  fs.mkdirSync(path.dirname(options.reportPath), { recursive: true });
  fs.writeFileSync(options.reportPath, renderMarkdown(report, options.outputPath), "utf8");
  console.log(`Wrote ${relative(options.outputPath)}`);
  console.log(`Wrote ${relative(options.reportPath)}`);
  printSummary(report);
}

function processTimedReturnTriples(
  returns: readonly TimedReturn[],
  accumulators: readonly WindowAccumulator[],
  intervalMs: number,
): void {
  for (let index = 2; index < returns.length; index += 1) {
    const first = returns[index - 2]!;
    const second = returns[index - 1]!;
    const third = returns[index]!;
    if (second.endTime !== first.endTime + intervalMs
      || third.endTime !== second.endTime + intervalMs) continue;
    const sampleHash = hashUnit(Math.floor(third.endTime / intervalMs), intervalMs);
    for (const accumulator of accumulators) {
      const definition = accumulator.definition;
      if (first.endTime <= definition.startTime || third.endTime > definition.endTime) continue;
      accumulator.moments.add(first.value, second.value, third.value);
      if (sampleHash < accumulator.samplingProbability) {
        accumulator.sample.x.push(first.value);
        accumulator.sample.y.push(second.value);
        accumulator.sample.z.push(third.value);
      }
    }
  }
}

function processOneSecondTriples(
  source: OneSecondSource,
  accumulators: readonly WindowAccumulator[],
  analysisStart: number,
  analysisEnd: number,
): void {
  let previousClose = Number.NaN;
  let previousCandleTime = Number.NaN;
  let firstReturn = Number.NaN;
  let firstReturnEnd = Number.NaN;
  let secondReturn = Number.NaN;
  let secondReturnEnd = Number.NaN;
  const firstReadDay = analysisStart - DAY_MS;
  for (const [fileIndex, file] of source.files.entries()) {
    const dayStart = parseUtcDay(path.basename(file, ".json"));
    const dayEnd = dayStart + DAY_MS;
    if (dayStart < firstReadDay || dayEnd > analysisEnd) continue;
    if (fileIndex % 100 === 0) {
      console.error(`Streaming one-second triple history ${fileIndex}/${source.files.length}...`);
    }
    const candles = readCandleShardReferenceSync(file);
    validateOneSecondDay(candles, dayStart, file);
    const active = accumulators.filter((accumulator) => (
      dayStart >= accumulator.definition.startTime
      && dayEnd <= accumulator.definition.endTime
    ));
    const allDay = new TripleMoments();
    const insideDay = new TripleMoments();
    for (const candle of candles) {
      if (previousCandleTime === candle.openTime - 1_000 && previousClose > 0) {
        const thirdReturn = Math.log(candle.close / previousClose);
        const thirdReturnEnd = candle.openTime + 1_000;
        if (secondReturnEnd === thirdReturnEnd - 1_000
          && firstReturnEnd === secondReturnEnd - 1_000
          && Number.isFinite(firstReturn)
          && Number.isFinite(secondReturn)) {
          allDay.add(firstReturn, secondReturn, thirdReturn);
          if (firstReturnEnd > dayStart) insideDay.add(firstReturn, secondReturn, thirdReturn);
          if (active.length > 0) {
            const sampleHash = hashUnit(Math.floor(thirdReturnEnd / 1_000), 1_000);
            for (const accumulator of active) {
              if (firstReturnEnd <= accumulator.definition.startTime) continue;
              if (sampleHash < accumulator.samplingProbability) {
                accumulator.sample.x.push(firstReturn);
                accumulator.sample.y.push(secondReturn);
                accumulator.sample.z.push(thirdReturn);
              }
            }
          }
        }
        firstReturn = secondReturn;
        firstReturnEnd = secondReturnEnd;
        secondReturn = thirdReturn;
        secondReturnEnd = thirdReturnEnd;
      } else {
        firstReturn = Number.NaN;
        firstReturnEnd = Number.NaN;
        secondReturn = Number.NaN;
        secondReturnEnd = Number.NaN;
      }
      previousClose = candle.close;
      previousCandleTime = candle.openTime;
    }
    for (const accumulator of active) {
      accumulator.moments.merge(
        accumulator.definition.startTime === dayStart ? insideDay : allDay,
      );
    }
  }
}

function finalizeWindow(
  accumulator: WindowAccumulator,
  fitSampleLimit: number,
  visualizationLimit: number,
): WindowScaleResult {
  const moments = accumulator.moments.snapshot();
  const shape = summarizeTripleShape(
    accumulator.sample,
    moments,
    visualizationLimit,
  );
  const generalizedGaussian = fitTripleGeneralizedGaussianPower(
    accumulator.sample,
    moments,
    fitSampleLimit,
  );
  return {
    id: accumulator.definition.id,
    label: accumulator.definition.label,
    kind: accumulator.definition.kind,
    startTime: iso(accumulator.definition.startTime),
    endTime: iso(accumulator.definition.endTime),
    durationDays: (accumulator.definition.endTime - accumulator.definition.startTime) / DAY_MS,
    observations: moments.observations,
    meansBps: moments.means.map((value) => value * 10_000) as [number, number, number],
    standardDeviationsBps: moments.standardDeviations.map((value) => value * 10_000) as [number, number, number],
    lag1ReturnCorrelation: mean(moments.correlations.slice(0, 2)),
    lag2ReturnCorrelation: moments.correlations[2],
    lag1AbsoluteReturnCorrelation: mean(moments.absoluteCorrelations.slice(0, 2)),
    lag2AbsoluteReturnCorrelation: moments.absoluteCorrelations[2],
    lag1SquaredReturnCorrelation: mean(moments.squaredCorrelations.slice(0, 2)),
    lag2SquaredReturnCorrelation: moments.squaredCorrelations[2],
    allZeroFraction: moments.allZeroFraction,
    anyZeroFraction: moments.anyZeroFraction,
    nonzeroAllSameSignFraction: moments.nonzeroAllSameSignFraction,
    signPatterns: moments.signPatterns,
    continuousObservations: moments.continuous.observations,
    sampleObservations: accumulator.sample.x.length,
    samplingProbability: accumulator.samplingProbability,
    shape,
    generalizedGaussianApproximation: {
      family: "trivariate-generalized-gaussian",
      ...generalizedGaussian,
    },
    sampleWarning: moments.observations < 100
      ? "Fewer than 100 triples; all shape and fit estimates are unstable."
      : moments.observations < 1_000
        ? "Fewer than 1,000 triples; tail and model-selection estimates are noisy."
        : null,
  };
}

function renderMarkdown(report: AnalysisReport, outputPath: string): string {
  const fullRows = report.scales.map((scale) => {
    const full = scale.fullHistory;
    return [
      scale.id,
      integer(full.observations),
      fixed(full.lag1ReturnCorrelation, 4),
      fixed(full.lag2ReturnCorrelation, 4),
      fixed(full.lag1AbsoluteReturnCorrelation, 3),
      fixed(full.lag2AbsoluteReturnCorrelation, 3),
      fixed(full.shape.allThreeAbsoluteTail2SigmaLift, 1),
      fixed(100 * full.allZeroFraction, 2),
      fixed(full.generalizedGaussianApproximation.power, 3),
      familyLabel(scale.selectedFullHistoryModel.family),
    ];
  });
  const candidateRows = report.scales.flatMap((scale) => scale.fullHistoryModelFits.map((fit) => [
    scale.id,
    familyLabel(fit.family),
    fixed(fit.parameters.power, 3),
    fixed(fit.parameters.tail, 3),
    fixed(fit.parameters.degreesFreedom, 2),
    fixed(fit.parameters.logRadiusMean, 3),
    fixed(fit.parameters.logRadiusStandardDeviation, 3),
    fixed(fit.nllPerObservation, 5),
    deltaAicText(fit.deltaAic),
  ]));
  const similarityRows = report.crossScaleSimilarity.map((item) => [
    `${item.leftScale} → ${item.rightScale}`,
    fixed(item.meanProjectionJensenShannonBits, 4),
  ]);
  const stabilityRows = report.annualWindowStability.map((item) => [
    item.scale,
    rangeText(item.lag1AbsoluteCorrelationRange, 3),
    rangeText(item.lag2AbsoluteCorrelationRange, 3),
    rangeText(item.generalizedGaussianPowerRange, 3),
    fixed(item.projectionJensenShannonBitsMedian, 4),
    fixed(item.projectionJensenShannonBitsMaximum, 4),
  ]);
  const winnerSummary = [...new Set(report.scales.map(
    (scale) => scale.selectedFullHistoryModel.family,
  ))].map((family) => `${familyLabel(family)} at ${report.scales
    .filter((scale) => scale.selectedFullHistoryModel.family === family)
    .map((scale) => scale.id)
    .join(", ")}`).join("; ");
  const daily = report.scales.find((scale) => scale.id === "1d")!;
  const dailyBest = daily.selectedFullHistoryModel;
  const dailySecond = daily.fullHistoryModelFits[1]!;
  const dailyTwoSigmaTripleCount = Math.round(
    (daily.fullHistory.shape.allThreeAbsoluteTail2Sigma ?? 0)
      * daily.fullHistory.shape.sampleObservations,
  );
  return `# Three consecutive BTCUSDT log returns across scales and windows

Generated ${report.generatedAt}. Exact common history: **${report.source.analysisStartTime} to ${report.source.analysisEndTime}** (exclusive end).

## Result

- Signed dependence remains small beyond one step: lag-2 return correlations range from ${fixed(Math.min(...report.scales.map((scale) => scale.fullHistory.lag2ReturnCorrelation)), 4)} to ${fixed(Math.max(...report.scales.map((scale) => scale.fullHistory.lag2ReturnCorrelation)), 4)}. Magnitude dependence persists, with lag-2 absolute-return correlations from ${fixed(Math.min(...report.scales.map((scale) => scale.fullHistory.lag2AbsoluteReturnCorrelation)), 3)} to ${fixed(Math.max(...report.scales.map((scale) => scale.fullHistory.lag2AbsoluteReturnCorrelation)), 3)}.
- Three adjacent $>2\\sigma$ magnitudes occur ${fixed(Math.min(...report.scales.map((scale) => scale.fullHistory.shape.allThreeAbsoluteTail2SigmaLift ?? Number.NaN)), 1)}–${fixed(Math.max(...report.scales.map((scale) => scale.fullHistory.shape.allThreeAbsoluteTail2SigmaLift ?? Number.NaN)), 1)} times more often than independent marginals predict. The main three-step structure is a persistent common volatility state, not directional continuation.
- AIC winners by scale are ${winnerSummary}.
- At 1d the closest tested family is **${familyLabel(dailyBest.family)}**${dailyBest.parameters.power === null ? "" : ` with $p=${fixed(dailyBest.parameters.power, 3)}$`}; ${familyLabel(dailySecond.family)} follows at $\\Delta$AIC ${fixed(dailySecond.deltaAic, 2)}.
- At 1s, ${fixed(100 * report.scales[0]!.fullHistory.allZeroFraction, 2)}% of triples are exactly $(0,0,0)$ and ${fixed(100 * report.scales[0]!.fullHistory.anyZeroFraction, 2)}% touch at least one zero plane. Continuous fits exclude those singular components.

## Full-history triple statistics

${markdownTable(
    ["Scale", "Triples", "Corr lag 1", "Corr lag 2", "Magnitude corr lag 1", "Magnitude corr lag 2", "All-3 >2σ lift", "(0,0,0) %", "GGD p", "AIC winner"],
    fullRows,
  )}

\`All-3 >2σ lift\` is the observed probability that all three standardized magnitudes exceed two, divided by the product of the three marginal exceedance probabilities.

## Closest continuous distribution

For covariance-whitened $z\\in\\mathbb{R}^3$, the trivariate generalized Gaussian is

$$
f(z)=\\frac{p}{4\\pi s^3\\Gamma(3/p)}
\\exp\\!\\left[-\\left(\\frac{\\lVert z\\rVert_2}{s}\\right)^p\\right],
$$

and the generalized t is

$$
f(z)=\\frac{p}{4\\pi s^3B(3/p,q-3/p)}
\\left[1+\\left(\\frac{\\lVert z\\rVert_2}{s}\\right)^p\\right]^{-q}.
$$

For the radial-lognormal candidate, $\\log\\rho\\sim\\mathcal N(m,\\tau^2)$ for $\\rho=\\lVert z\\rVert_2$, so

$$
f(z)=\\frac{1}{4\\pi\\sqrt{2\\pi}\\tau\\rho^3}
\\exp\\!\\left[-\\frac{(\\log\\rho-m)^2}{2\\tau^2}\\right].
$$

${markdownTable(
    ["Scale", "Candidate", "p", "q", "ν", "mean log ρ", "sd log ρ", "NLL / triple", "ΔAIC"],
    candidateRows,
  )}

## Similarity across scales

The comparison averages Jensen–Shannon divergence across all three standardized pairwise projections: $(r_t,r_{t+1})$, $(r_{t+1},r_{t+2})$, and $(r_t,r_{t+2})$.

${markdownTable(["Adjacent scales", "Mean projection JS bits"], similarityRows)}

## Stability across one-year epochs

${markdownTable(
    ["Scale", "Lag-1 magnitude range", "Lag-2 magnitude range", "GGD p range", "Median JS to 5y", "Max JS to 5y"],
    stabilityRows,
  )}

At 4h and 1d, annual projection distances have a large finite-sample noise floor. Daily model selection uses only about 1,824 triples, annual daily estimates use about 363, and the daily all-three $>2\\sigma$ lift is based on only ${dailyTwoSigmaTripleCount} observed triples.

## Interpretation

- The three-dimensional law strengthens the pairwise conclusion: a shared radial distribution is much closer than a Gaussian, while direct signed correlations remain close to zero.
- The generalized t captures a sharp center and polynomial joint tails. A generalized Gaussian becomes competitive only when aggregation has softened the power-law component.
- The 1s fit remains dominated by tick/no-trade microstructure and reaches the allowed lower generalized-Gaussian power bound; its power estimate is not a literal continuous-law parameter.
- Close-to-close triples are descriptive and omit intrabar extremes, spread, fees, slippage, and liquidation paths.

## Reproduction

Run \`npm run analysis:return-triples\`, then \`npm run analysis:return-triples:render\`. Machine-readable results are in \`${relative(outputPath)}\`.
`;
}

function printSummary(report: AnalysisReport): void {
  console.log("\nFull-history consecutive-return triples");
  for (const scale of report.scales) {
    const full = scale.fullHistory;
    console.log(
      `${scale.id.padEnd(3)} n=${integer(full.observations).padStart(11)} `
      + `lag1|r|=${fixed(full.lag1AbsoluteReturnCorrelation, 3).padStart(6)} `
      + `lag2|r|=${fixed(full.lag2AbsoluteReturnCorrelation, 3).padStart(6)} `
      + `GGD-p=${fixed(full.generalizedGaussianApproximation.power, 3).padStart(6)} `
      + `best=${report.scales.find((item) => item.id === scale.id)!.selectedFullHistoryModel.family}`,
    );
  }
}

function loadMinuteCandles(dataDir: string): {
  referenceDirectory: string;
  files: number;
  candles: SequentialCandle[];
  firstCandleOpenTime: number;
  lastCandleEndTime: number;
} {
  const referenceDirectory = new TradingStorageLayout(dataDir)
    .candleReferences("spot-btcusdt", "btcusdt", "1m");
  const files = fs.readdirSync(referenceDirectory, { withFileTypes: true })
    .filter((entry) => entry.isFile() && /^\d{4}-\d{2}-\d{2}\.json$/.test(entry.name))
    .map((entry) => path.join(referenceDirectory, entry.name))
    .sort();
  const candles: SequentialCandle[] = [];
  for (const [index, file] of files.entries()) {
    if (index % 250 === 0) console.error(`Loading minute history ${index}/${files.length}...`);
    candles.push(...readCandleShardReferenceSync(file));
  }
  candles.sort((left, right) => left.openTime - right.openTime);
  const unique: SequentialCandle[] = [];
  for (const candle of candles) {
    if (unique.at(-1)?.openTime === candle.openTime) unique[unique.length - 1] = candle;
    else unique.push(candle);
  }
  if (unique.length === 0) throw new Error("No BTCUSDT minute candles were found.");
  return {
    referenceDirectory,
    files: files.length,
    candles: unique,
    firstCandleOpenTime: unique[0]!.openTime,
    lastCandleEndTime: unique.at(-1)!.openTime + 60_000,
  };
}

function discoverOneSecondSource(dataDir: string): OneSecondSource {
  const referenceDirectory = new TradingStorageLayout(dataDir)
    .candleReferences("spot-btcusdt", "btcusdt", "1s");
  const files = fs.readdirSync(referenceDirectory, { withFileTypes: true })
    .filter((entry) => entry.isFile() && /^\d{4}-\d{2}-\d{2}\.json$/.test(entry.name))
    .map((entry) => path.join(referenceDirectory, entry.name))
    .sort();
  if (files.length === 0) throw new Error("No BTCUSDT one-second candles were found.");
  for (let index = 1; index < files.length; index += 1) {
    const previous = parseUtcDay(path.basename(files[index - 1]!, ".json"));
    const current = parseUtcDay(path.basename(files[index]!, ".json"));
    if (current !== previous + DAY_MS) throw new Error("One-second history contains a day gap.");
  }
  return {
    referenceDirectory,
    files,
    firstCandleOpenTime: parseUtcDay(path.basename(files[0]!, ".json")),
    lastCandleEndTime: parseUtcDay(path.basename(files.at(-1)!, ".json")) + DAY_MS,
  };
}

function sourceDescription(source: OneSecondSource): SourceDescription {
  return {
    interval: "1s",
    referenceDirectory: relative(source.referenceDirectory),
    files: source.files.length,
    candles: source.files.length * 86_400,
    firstCandleOpenTime: iso(source.firstCandleOpenTime),
    lastCandleEndTime: iso(source.lastCandleEndTime),
  };
}

function validateOneSecondDay(
  candles: readonly SequentialCandle[],
  dayStart: number,
  file: string,
): void {
  if (candles.length !== 86_400
    || candles[0]?.openTime !== dayStart
    || candles.at(-1)?.openTime !== dayStart + DAY_MS - 1_000) {
    throw new Error(`${path.basename(file)} is not a complete one-second day.`);
  }
  for (let index = 0; index < candles.length; index += 1) {
    const candle = candles[index]!;
    if (candle.openTime !== dayStart + index * 1_000
      || candle.closed === false
      || !(candle.close > 0)) {
      throw new Error(`${path.basename(file)} has an invalid candle at ${index}.`);
    }
  }
}

function buildWindowDefinitions(startTime: number, endTime: number): WindowDefinition[] {
  return [
    { id: "full-5y", label: "Full five-year history", kind: "full", startTime, endTime },
    ...Array.from({ length: 5 }, (_, index): WindowDefinition => ({
      id: `year-${index + 1}-${dateOnly(subtractUtcYears(endTime, 5 - index))}`,
      label: `${dateOnly(subtractUtcYears(endTime, 5 - index))} to ${dateOnly(subtractUtcYears(endTime, 4 - index))}`,
      kind: "annual-epoch",
      startTime: subtractUtcYears(endTime, 5 - index),
      endTime: subtractUtcYears(endTime, 4 - index),
    })),
    { id: "trailing-30d", label: "Trailing 30 days", kind: "trailing", startTime: endTime - 30 * DAY_MS, endTime },
    { id: "trailing-90d", label: "Trailing 90 days", kind: "trailing", startTime: endTime - 90 * DAY_MS, endTime },
    { id: "trailing-365d", label: "Trailing 365 days", kind: "trailing", startTime: endTime - YEAR_MS, endTime },
  ];
}

function createAccumulators(
  definitions: readonly WindowDefinition[],
  intervalMs: number,
  sampleTarget: number,
): WindowAccumulator[] {
  return definitions.map((definition) => ({
    definition,
    moments: new TripleMoments(),
    sample: { x: [], y: [], z: [] },
    samplingProbability: Math.min(
      1,
      sampleTarget / Math.max(1, (definition.endTime - definition.startTime) / intervalMs - 2),
    ),
  }));
}

function parseOptions(args: string[]): Options {
  if (args.includes("--help")) {
    console.log(`Usage: tsx scripts/analyze-consecutive-return-triples.ts [options]

  --data-dir PATH
  --output PATH
  --report PATH
  --end YYYY-MM-DD
  --sample-target N
  --fit-sample-limit N
  --visualization-sample-limit N`);
    process.exit(0);
  }
  const values = new Map<string, string>();
  for (let index = 0; index < args.length; index += 1) {
    const key = args[index];
    const value = args[index + 1];
    if (!key?.startsWith("--") || !value || value.startsWith("--")) {
      throw new Error(`Invalid argument near ${key ?? "end"}.`);
    }
    values.set(key.slice(2), value);
    index += 1;
  }
  const requestedEnd = values.get("end");
  return {
    dataDir: path.resolve(repoRoot, values.get("data-dir") ?? "data"),
    outputPath: path.resolve(
      repoRoot,
      values.get("output") ?? "data/benchmarks/consecutive-log-return-triples.json",
    ),
    reportPath: path.resolve(
      repoRoot,
      values.get("report")
        ?? `docs/experiments/consecutive-log-return-triples-${dateOnly(Date.now())}.md`,
    ),
    ...(requestedEnd === undefined ? {} : { requestedEndTime: parseUtcDay(requestedEnd) }),
    sampleTarget: positiveInteger(values.get("sample-target") ?? "250000", "sample-target"),
    fitSampleLimit: positiveInteger(values.get("fit-sample-limit") ?? "30000", "fit-sample-limit"),
    visualizationSampleLimit: positiveInteger(
      values.get("visualization-sample-limit") ?? "4000",
      "visualization-sample-limit",
    ),
  };
}

function familyLabel(family: string): string {
  return ({
    "trivariate-gaussian": "Trivariate Gaussian",
    "trivariate-student-t": "Trivariate Student t",
    "trivariate-generalized-gaussian": "Trivariate generalized Gaussian",
    "trivariate-radial-lognormal": "Trivariate radial lognormal",
    "trivariate-generalized-t": "Trivariate generalized t",
  } as Record<string, string>)[family] ?? family;
}

function hashUnit(value: number, salt: number): number {
  let hash = (value ^ salt) >>> 0;
  hash = Math.imul(hash ^ (hash >>> 16), 0x45d9f3b);
  hash = Math.imul(hash ^ (hash >>> 16), 0x45d9f3b);
  return ((hash ^ (hash >>> 16)) >>> 0) / 0x1_0000_0000;
}

function finiteRange(values: readonly number[]): [number, number] | null {
  const finite = values.filter(Number.isFinite);
  return finite.length ? [Math.min(...finite), Math.max(...finite)] : null;
}

function quantile(sorted: readonly number[], probability: number): number {
  const position = probability * (sorted.length - 1);
  const lower = Math.floor(position);
  const weight = position - lower;
  return sorted[lower]! * (1 - weight) + sorted[Math.min(lower + 1, sorted.length - 1)]! * weight;
}

function mean(values: readonly number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function markdownTable(headers: string[], rows: string[][]): string {
  return [
    `| ${headers.join(" | ")} |`,
    `| ${headers.map(() => "---").join(" | ")} |`,
    ...rows.map((row) => `| ${row.join(" | ")} |`),
  ].join("\n");
}

function rangeText(range: [number, number] | null, digits: number): string {
  return range === null ? "n/a" : `${range[0].toFixed(digits)}–${range[1].toFixed(digits)}`;
}

function deltaAicText(value: number): string {
  if (value < 10) return value.toFixed(2);
  if (value < 1_000) return value.toFixed(1);
  return value.toFixed(0);
}

function fixed(value: number | null, digits: number): string {
  return value === null || !Number.isFinite(value) ? "n/a" : value.toFixed(digits);
}

function integer(value: number): string {
  return value.toLocaleString("en-US", { maximumFractionDigits: 0 });
}

function subtractUtcYears(time: number, years: number): number {
  const date = new Date(time);
  return Date.UTC(date.getUTCFullYear() - years, date.getUTCMonth(), date.getUTCDate());
}

function parseUtcDay(value: string): number {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) throw new Error(`Invalid UTC date: ${value}`);
  const result = Date.parse(`${value}T00:00:00.000Z`);
  if (!Number.isFinite(result) || dateOnly(result) !== value) throw new Error(`Invalid UTC date: ${value}`);
  return result;
}

function positiveInteger(value: string, label: string): number {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < 1) throw new Error(`${label} must be positive.`);
  return parsed;
}

function iso(time: number): string {
  return new Date(time).toISOString();
}

function dateOnly(time: number): string {
  return iso(time).slice(0, 10);
}

function relative(file: string): string {
  return path.relative(repoRoot, file).replaceAll("\\", "/");
}

function numberReplacer(_key: string, value: unknown): unknown {
  if (typeof value !== "number" || !Number.isFinite(value) || Number.isInteger(value)) return value;
  return Number(value.toPrecision(10));
}
