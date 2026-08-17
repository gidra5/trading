import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

interface HistogramWindow {
  id: string;
  zeroProbability: number;
  histogram: {
    observations: number;
    binWidthBps: number;
    lowerBps: number;
    binCount: number;
    nonzeroBins: Array<[index: number, probability: number]>;
  };
}

interface HistogramReport {
  version: number;
  symbol: string;
  commonEndTime: string;
  scales: Array<{ id: string; label: string; windows: HistogramWindow[] }>;
}

interface PreparedHistogram {
  points: Array<[x: number, weight: number]>;
  observationsOutside: number;
  empiricalOutsideMass: number;
  excludeCentralBin: boolean;
  excludedLower: number;
  excludedUpper: number;
  sigmaBps: number;
}

type CandidateId = "laplace" | "student-t" | "generalized-normal" | "generalized-t";

interface CandidateFit {
  id: CandidateId;
  parameters: number[];
  parameterCount: number;
  nllPerObservation: number;
  aic: number;
}

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

try {
  main();
} catch (error: unknown) {
  console.error(error instanceof Error ? error.stack ?? error.message : String(error));
  process.exitCode = 1;
}

function main(): void {
  const args = parseArgs(process.argv.slice(2));
  const inputPath = path.resolve(
    repoRoot,
    args.get("input") ?? "data/benchmarks/log-return-histograms.json",
  );
  const outputPath = path.resolve(
    repoRoot,
    args.get("output") ?? "data/benchmarks/log-return-distribution-fits.json",
  );
  const report = JSON.parse(fs.readFileSync(inputPath, "utf8")) as HistogramReport;
  if (report.version !== 1 || report.symbol !== "BTCUSDT" || report.scales.length !== 6) {
    throw new Error("Histogram input is not the expected six-scale BTCUSDT report.");
  }

  const scales = report.scales.map((scale) => {
    const full = scale.windows.find((window) => window.id === "full");
    if (!full) throw new Error(`Scale ${scale.id} has no full-history window.`);
    const excludeCentralBin = scale.id === "1s" || scale.id === "1m";
    const prepared = prepareHistogram(full, excludeCentralBin);
    const candidates = fitCandidates(prepared);
    const selected = candidates.reduce((best, candidate) => (
      candidate.aic < best.aic ? candidate : best
    ));
    const selectedModel = serializeCandidate(selected);
    const excludedProbability = prepared.excludeCentralBin
      ? integrateSimpson(
        (value) => Math.exp(selectedModel.logPdf(value)),
        prepared.excludedLower,
        prepared.excludedUpper,
      )
      : 0;
    const bestAic = Math.min(...candidates.map((candidate) => candidate.aic));
    return {
      id: scale.id,
      label: scale.label,
      window: "full",
      observationsOutsideExcludedBin: Math.round(prepared.observationsOutside),
      empiricalOutsideMass: round(prepared.empiricalOutsideMass),
      sigmaBps: round(prepared.sigmaBps),
      excludedBinSigma: prepared.excludeCentralBin
        ? [round(prepared.excludedLower), round(prepared.excludedUpper)]
        : null,
      family: selectedModel.family,
      parameters: Object.fromEntries(Object.entries(selectedModel.parameters)
        .map(([key, value]) => [key, value === null ? null : round(value)])),
      modelOutsideProbability: round(1 - excludedProbability),
      asymptoticPdfExponent: selectedModel.asymptoticPdfExponent === null
        ? null
        : round(selectedModel.asymptoticPdfExponent),
      asymptoticSurvivalExponent: selectedModel.asymptoticSurvivalExponent === null
        ? null
        : round(selectedModel.asymptoticSurvivalExponent),
      comparison: candidates.map((candidate) => ({
        family: candidate.id,
        parameterCount: candidate.parameterCount,
        nllPerObservation: round(candidate.nllPerObservation),
        deltaAic: round(candidate.aic - bestAic),
      })),
    };
  });

  const output = {
    version: 1,
    generatedAt: new Date().toISOString(),
    symbol: report.symbol,
    commonEndTime: report.commonEndTime,
    methodology: {
      target: "Full-history binned log-return distribution at each scale.",
      excludedBin: "For 1s and 1m, the single histogram bin containing zero (width 0.1 full-history sigma) is excluded so the exact-zero and microstructure spike cannot dominate. The 15m, 1h, 4h, and 1d fits include the zero-containing bin.",
      objective: "Maximum conditional binned-midpoint likelihood; conditional outside the excluded bin for 1s and 1m, unconditional over the full histogram for 15m and slower.",
      selection: "Generalized t compared with Laplace, Student t, and generalized normal using AIC.",
      normalization: "The 1s and 1m curves are rescaled to empirical mass outside the excluded bin; 15m and slower curves integrate to one and cover zero.",
    },
    formulas: {
      generalizedT: "f(x)=p/[2*s*B(1/p,q-1/p)]*[1+(abs(x-mu)/s)^p]^(-q), q>1/p",
      generalizedNormal: "f(x)=p/[2*s*Gamma(1/p)]*exp(-(abs(x-mu)/s)^p)",
      normalization: "x=return/full-history-sigma",
    },
    scales,
  };
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, `${JSON.stringify(output, null, 2)}\n`, "utf8");
  console.log(outputPath);
  for (const scale of scales) {
    console.log(`${scale.id}: ${scale.family} ${JSON.stringify(scale.parameters)}`);
  }
}

function serializeCandidate(candidate: CandidateFit): {
  family: string;
  parameters: Record<string, number | null>;
  logPdf: (value: number) => number;
  asymptoticPdfExponent: number | null;
  asymptoticSurvivalExponent: number | null;
} {
  if (candidate.id === "generalized-t") {
    const [location, logScale, logPower, logTailMargin] = candidate.parameters;
    const scale = Math.exp(logScale!);
    const power = Math.exp(logPower!);
    const tail = 1 / power + Math.exp(logTailMargin!);
    const model = generalizedTModel(location!, scale, power, tail);
    return {
      family: "symmetric-generalized-t",
      parameters: {
        locationSigma: location!, scaleSigma: scale, power, tail,
        degreesFreedom: null,
        logNormalizer: Math.log(power) - Math.log(2 * scale)
          - logBeta(1 / power, tail - 1 / power),
      },
      logPdf: model.logPdf,
      asymptoticPdfExponent: power * tail,
      asymptoticSurvivalExponent: power * tail - 1,
    };
  }
  if (candidate.id === "generalized-normal") {
    const [location, logScale, logPower] = candidate.parameters;
    const scale = Math.exp(logScale!);
    const power = Math.exp(logPower!);
    const logNormalizer = Math.log(power) - Math.log(2 * scale) - logGamma(1 / power);
    return {
      family: "symmetric-generalized-normal",
      parameters: {
        locationSigma: location!, scaleSigma: scale, power, tail: null,
        degreesFreedom: null, logNormalizer,
      },
      logPdf: (value) => logNormalizer - (Math.abs(value - location!) / scale) ** power,
      asymptoticPdfExponent: null,
      asymptoticSurvivalExponent: null,
    };
  }
  if (candidate.id === "student-t") {
    const [location, logScale, logDegreesMargin] = candidate.parameters;
    const scale = Math.exp(logScale!);
    const degreesFreedom = 0.5 + Math.exp(logDegreesMargin!);
    const logNormalizer = logGamma((degreesFreedom + 1) / 2)
      - logGamma(degreesFreedom / 2) - 0.5 * Math.log(degreesFreedom * Math.PI)
      - Math.log(scale);
    return {
      family: "symmetric-student-t",
      parameters: {
        locationSigma: location!, scaleSigma: scale, power: null, tail: null,
        degreesFreedom, logNormalizer,
      },
      logPdf: (value) => logNormalizer - (degreesFreedom + 1) / 2
        * Math.log1p(((value - location!) / scale) ** 2 / degreesFreedom),
      asymptoticPdfExponent: degreesFreedom + 1,
      asymptoticSurvivalExponent: degreesFreedom,
    };
  }
  const [location, logScale] = candidate.parameters;
  const scale = Math.exp(logScale!);
  const logNormalizer = -Math.log(2 * scale);
  return {
    family: "symmetric-laplace",
    parameters: {
      locationSigma: location!, scaleSigma: scale, power: null, tail: null,
      degreesFreedom: null, logNormalizer,
    },
    logPdf: (value) => logNormalizer - Math.abs(value - location!) / scale,
    asymptoticPdfExponent: null,
    asymptoticSurvivalExponent: null,
  };
}

function prepareHistogram(
  window: HistogramWindow,
  excludeCentralBin: boolean,
): PreparedHistogram {
  const histogram = window.histogram;
  const sigmaBps = histogram.binWidthBps / 0.1;
  const excludedIndex = Math.floor((0 - histogram.lowerBps) / histogram.binWidthBps);
  const excludedLower = (histogram.lowerBps + excludedIndex * histogram.binWidthBps)
    / sigmaBps;
  const excludedUpper = excludedLower + histogram.binWidthBps / sigmaBps;
  const rawPoints = histogram.nonzeroBins.flatMap(([index, probability]) => {
    if ((excludeCentralBin && index === excludedIndex) || probability <= 0) return [];
    const midpointBps = histogram.lowerBps + (index + 0.5) * histogram.binWidthBps;
    return [[midpointBps / sigmaBps, probability] as [number, number]];
  });
  const empiricalOutsideMass = rawPoints.reduce((sum, point) => sum + point[1], 0);
  return {
    points: rawPoints.map(([value, probability]) => [value, probability / empiricalOutsideMass]),
    observationsOutside: histogram.observations * empiricalOutsideMass,
    empiricalOutsideMass,
    excludeCentralBin,
    excludedLower,
    excludedUpper,
    sigmaBps,
  };
}

function fitCandidates(histogram: PreparedHistogram): CandidateFit[] {
  const definitions: Array<{
    id: CandidateId;
    start: number[];
    parameterCount: number;
    model: (parameters: number[]) => { logPdf: (value: number) => number } | undefined;
  }> = [
    {
      id: "laplace",
      start: [0, -0.4],
      parameterCount: 2,
      model: ([location, logScale]) => {
        const scale = Math.exp(logScale!);
        return {
          logPdf: (value) => -Math.log(2 * scale) - Math.abs(value - location!) / scale,
        };
      },
    },
    {
      id: "student-t",
      start: [0, -0.5, 1],
      parameterCount: 3,
      model: ([location, logScale, logDegreesMargin]) => {
        const scale = Math.exp(logScale!);
        const degrees = 0.5 + Math.exp(logDegreesMargin!);
        const constant = logGamma((degrees + 1) / 2) - logGamma(degrees / 2)
          - 0.5 * Math.log(degrees * Math.PI) - Math.log(scale);
        return {
          logPdf: (value) => constant - (degrees + 1) / 2
            * Math.log1p(((value - location!) / scale) ** 2 / degrees),
        };
      },
    },
    {
      id: "generalized-normal",
      start: [0, -0.5, -0.2],
      parameterCount: 3,
      model: ([location, logScale, logPower]) => {
        const scale = Math.exp(logScale!);
        const power = Math.exp(logPower!);
        if (power > 20) return undefined;
        const constant = Math.log(power) - Math.log(2 * scale) - logGamma(1 / power);
        return {
          logPdf: (value) => constant - (Math.abs(value - location!) / scale) ** power,
        };
      },
    },
    {
      id: "generalized-t",
      start: [0, -0.5, Math.log(1.5), Math.log(2)],
      parameterCount: 4,
      model: ([location, logScale, logPower, logTailMargin]) => {
        const scale = Math.exp(logScale!);
        const power = Math.exp(logPower!);
        const tail = 1 / power + Math.exp(logTailMargin!);
        if (power > 20 || tail > 100) return undefined;
        return generalizedTModel(location!, scale, power, tail);
      },
    },
  ];

  return definitions.map((definition) => {
    const objective = (parameters: number[]): number => {
      if (parameters.some((value) => !Number.isFinite(value) || Math.abs(value) > 12)) {
        return 1e6;
      }
      const model = definition.model(parameters);
      if (!model) return 1e6;
      const excludedProbability = histogram.excludeCentralBin
        ? integrateSimpson(
          (value) => Math.exp(model.logPdf(value)),
          histogram.excludedLower,
          histogram.excludedUpper,
        )
        : 0;
      const outsideProbability = 1 - excludedProbability;
      if (!(outsideProbability > 0 && outsideProbability <= 1)) return 1e6;
      let nll = 0;
      for (const [value, weight] of histogram.points) {
        const logProbability = model.logPdf(value) - Math.log(outsideProbability);
        if (!Number.isFinite(logProbability)) return 1e6;
        nll -= weight * logProbability;
      }
      return nll;
    };
    const result = minimizeNelderMead(objective, definition.start, 0.3);
    return {
      id: definition.id,
      parameters: result.parameters,
      parameterCount: definition.parameterCount,
      nllPerObservation: result.value,
      aic: 2 * definition.parameterCount
        + 2 * histogram.observationsOutside * result.value,
    };
  });
}

function generalizedTModel(
  location: number,
  scale: number,
  power: number,
  tail: number,
): { logPdf: (value: number) => number } {
  const constant = Math.log(power) - Math.log(2 * scale)
    - logBeta(1 / power, tail - 1 / power);
  return {
    logPdf: (value) => constant - tail
      * Math.log1p((Math.abs(value - location) / scale) ** power),
  };
}

function integrateSimpson(
  evaluate: (value: number) => number,
  lower: number,
  upper: number,
  subdivisions = 64,
): number {
  const width = (upper - lower) / subdivisions;
  let sum = evaluate(lower) + evaluate(upper);
  for (let index = 1; index < subdivisions; index += 1) {
    sum += (index % 2 === 0 ? 2 : 4) * evaluate(lower + index * width);
  }
  return sum * width / 3;
}

function minimizeNelderMead(
  objective: (parameters: number[]) => number,
  start: number[],
  step: number,
): { parameters: number[]; value: number } {
  const dimensions = start.length;
  let simplex = [start.slice()];
  for (let dimension = 0; dimension < dimensions; dimension += 1) {
    const point = start.slice();
    point[dimension] += step;
    simplex.push(point);
  }
  let values = simplex.map(objective);
  for (let iteration = 0; iteration < 20_000; iteration += 1) {
    const order = values.map((_, index) => index).sort((left, right) => values[left] - values[right]);
    simplex = order.map((index) => simplex[index]);
    values = order.map((index) => values[index]);
    const valueSpread = Math.max(...values.map((value) => Math.abs(value - values[0]!)));
    const pointSpread = Math.max(...simplex.map((point) => Math.max(
      ...point.map((value, index) => Math.abs(value - simplex[0]![index]!)),
    )));
    if (valueSpread < 1e-11 && pointSpread < 1e-7) break;
    const centroid = Array.from({ length: dimensions }, (_, dimension) => (
      simplex.slice(0, dimensions)
        .reduce((sum, point) => sum + point[dimension]!, 0) / dimensions
    ));
    const reflected = centroid.map((value, index) => value + value - simplex.at(-1)![index]!);
    const reflectedValue = objective(reflected);
    if (values[0]! <= reflectedValue && reflectedValue < values[dimensions - 1]!) {
      simplex[dimensions] = reflected;
      values[dimensions] = reflectedValue;
      continue;
    }
    if (reflectedValue < values[0]!) {
      const expanded = centroid.map((value, index) => value + 2 * (reflected[index]! - value));
      const expandedValue = objective(expanded);
      if (expandedValue < reflectedValue) {
        simplex[dimensions] = expanded;
        values[dimensions] = expandedValue;
      } else {
        simplex[dimensions] = reflected;
        values[dimensions] = reflectedValue;
      }
      continue;
    }
    const contracted = centroid.map((value, index) => (
      value + 0.5 * (simplex[dimensions]![index]! - value)
    ));
    const contractedValue = objective(contracted);
    if (contractedValue < values[dimensions]!) {
      simplex[dimensions] = contracted;
      values[dimensions] = contractedValue;
      continue;
    }
    for (let index = 1; index <= dimensions; index += 1) {
      simplex[index] = simplex[index]!.map((value, dimension) => (
        simplex[0]![dimension]! + 0.5 * (value - simplex[0]![dimension]!)
      ));
      values[index] = objective(simplex[index]!);
    }
  }
  const bestIndex = values.indexOf(Math.min(...values));
  return { parameters: simplex[bestIndex]!, value: values[bestIndex]! };
}

function logBeta(left: number, right: number): number {
  return logGamma(left) + logGamma(right) - logGamma(left + right);
}

function logGamma(value: number): number {
  const coefficients = [
    676.5203681218851,
    -1259.1392167224028,
    771.3234287776531,
    -176.6150291621406,
    12.507343278686905,
    -0.13857109526572012,
    9.984369578019572e-6,
    1.5056327351493116e-7,
  ];
  if (value < 0.5) {
    return Math.log(Math.PI) - Math.log(Math.sin(Math.PI * value)) - logGamma(1 - value);
  }
  const shifted = value - 1;
  let series = 0.9999999999998099;
  for (let index = 0; index < coefficients.length; index += 1) {
    series += coefficients[index]! / (shifted + index + 1);
  }
  const t = shifted + coefficients.length - 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (shifted + 0.5) * Math.log(t)
    - t + Math.log(series);
}

function parseArgs(args: string[]): Map<string, string> {
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
  return values;
}

function round(value: number): number {
  return Number(value.toPrecision(10));
}
