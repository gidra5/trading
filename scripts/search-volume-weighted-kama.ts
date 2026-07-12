import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { fork } from "node:child_process";
import {
  evaluateVwKamaOracle,
  vwKamaScore,
  type VwKamaPreset,
} from "../packages/bot-algo/src/kama-signal-evaluator.js";
import { volumeWeightedKamaWarmupSamples } from "../packages/bot-algo/src/indicators.js";
import { perfectMarginOracle } from "../packages/bot-algo/src/perfect-margin-oracle.js";
import type { PerfectMarginOracleResult } from "../packages/bot-algo/src/perfect-margin-oracle.js";
import type { TradingCandle } from "../packages/bot-algo/src/trading-api.js";

type Family = "volume" | "canonical";
type Stage = "fit" | "validation" | "test";
type DeadbandMode = "flat" | "hold";
type ThresholdMode = "static" | "adaptive";
type SearchAlgorithm = "random" | "de";
type SearchMode = "standard" | "per-window";

interface Window { label: string; start: number; end: number }
interface Range { min: number; max: number }
interface Args {
  sourceDir: string;
  sourceIntervalMs: number;
  scales: number[];
  fitWindows?: Window[];
  validationWindows?: Window[];
  testWindows?: Window[];
  trials: number;
  algorithm: SearchAlgorithm;
  mode: SearchMode;
  generations: number;
  differentialWeight: number;
  crossoverRate: number;
  immigrantRate: number;
  screenWindows: number;
  screenScales: number[];
  workers: number;
  presetWindowIds?: string[];
  presetOutputPath: string;
  top: number;
  seed: number;
  oracleFriction: number;
  matchWindowMs: number;
  timingHalfLifeMs: number;
  warmupMultiple: number;
  caseWarmupMs: number;
  efficiency: Range;
  fast: Range;
  slow: Range;
  volume: Range;
  power: Range;
  volumeCap: Range;
  volumePower: Range;
  deadbandBpsHour: Range;
  thresholdLookback: Range;
  thresholdNoiseMultiplier: Range;
  outputPath: string;
  reportPath: string;
}

interface Candidate {
  id: string;
  family: Family;
  efficiencyMs: number;
  fastMs: number;
  slowMs: number;
  power: number;
  volumeMs: number;
  volumeCap: number;
  volumePower: number;
  deadbandBpsHour: number;
  deadbandMode: DeadbandMode;
  thresholdMode: ThresholdMode;
  thresholdLookbackMs: number;
  thresholdNoiseMultiplier: number;
}

type Genome = Omit<Candidate, "id" | "family" | "volumePower"> & { volumePower: number };

interface CaseData {
  id: string;
  stage: Stage;
  scaleMs: number;
  window: Window;
  candles: TradingCandle[];
  scoreStart: number;
  oracle: PerfectMarginOracleResult;
  days: number;
}

interface CaseScore {
  caseId: string;
  scale: string;
  window: string;
  score: number;
  precision: number;
  recall: number;
  f1: number;
  rawPrecision: number;
  rawRecall: number;
  exposureAgreement: number;
  noiseSignalRatio: number | null;
  signalCleanliness: number;
  signalsPerDay: number;
  lagP50Ms: number | null;
  lagP90Ms: number | null;
  lagP95Ms: number | null;
  lagMedianSignedMs: number | null;
}

interface CaseStats {
  caseId: string;
  scale: string;
  window: string;
  timingCredit: number;
  matchedCount: number;
  signalCount: number;
  oracleCount: number;
  stateCredit: number;
  stateCount: number;
  days: number;
  absoluteLags: number[];
  signedLags: number[];
}

interface AggregateScore {
  objective: number;
  median: number;
  p10: number;
  precision: number;
  recall: number;
  f1: number;
  exposureAgreement: number;
  noiseSignalRatio: number | null;
  signalCleanliness: number;
  signalsPerDay: number;
  lagP50Ms: number | null;
  lagP90Ms: number | null;
}

interface CandidateResult {
  candidate: Candidate;
  stage: Stage;
  aggregate: AggregateScore;
  cases: CaseScore[];
}

const DAY = 86_400_000;
const HOUR = 3_600_000;
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const stamp = new Date().toISOString().replace(/[:.]/g, "").replace("T", "-").slice(0, 17);
const windowJob = process.env.VW_KAMA_WINDOW_JOB;

if (!windowJob) {
  void run(parseArgs(process.argv.slice(2))).catch((error) => {
    console.error(error instanceof Error ? error.stack ?? error.message : error);
    process.exitCode = 1;
  });
} else {
  try {
    process.send?.(optimizeWindow(JSON.parse(windowJob) as WindowJob));
  } catch (error) {
    process.send?.({ error: error instanceof Error ? error.stack ?? error.message : String(error) });
  }
}

async function run(config: Args): Promise<void> {
  const startedAt = Date.now();
  validateScales(config.scales, config.sourceIntervalMs);
  validateScales(config.screenScales, config.sourceIntervalMs);
  if (config.algorithm === "de" && config.trials < 4) {
    throw new Error("Differential evolution requires at least four genomes.");
  }
  const bounds = sourceBounds(config.sourceDir);
  const defaults = defaultWindows(bounds.start, bounds.end, config.sourceIntervalMs);
  const windows = {
    fit: config.fitWindows ?? defaults.fit,
    validation: config.validationWindows ?? defaults.validation,
    test: config.testWindows ?? defaults.test,
  };
  if (config.mode === "per-window") {
    await runPerWindow(config, windows.fit, bounds, startedAt);
    return;
  }
  assertChronological(windows.fit, windows.validation, windows.test);
  const maxWarmupMs = maximumWarmupMs(config);
  const candidates = searchCandidates(config, windows.fit, bounds, maxWarmupMs);

  fs.mkdirSync(path.dirname(config.outputPath), { recursive: true });
  fs.mkdirSync(path.dirname(config.reportPath), { recursive: true });
  fs.writeFileSync(config.outputPath, "");
  appendJson(config.outputPath, {
    type: "meta",
    generatedAt: new Date(startedAt).toISOString(),
    sourceDir: path.relative(repoRoot, config.sourceDir),
    sourceInterval: formatDuration(config.sourceIntervalMs),
    scales: config.scales.map(formatDuration),
    windows,
    trials: config.trials,
    algorithm: config.algorithm,
    generations: config.generations,
    screening: {
      windows: config.screenWindows,
      scales: config.screenScales.map(formatDuration),
    },
    seed: config.seed,
    oracle: {
      friction: config.oracleFriction,
      matchWindow: formatDuration(config.matchWindowMs),
      timingHalfLife: formatDuration(config.timingHalfLifeMs),
      warmupMultiple: config.warmupMultiple,
      caseWarmup: formatDuration(maxWarmupMs),
    },
    signalMemory: {
      friction: config.oracleFriction,
      rule: "absolute close-to-last-signal return must be greater than friction",
    },
    ranges: {
      efficiency: formatRange(config.efficiency, formatDuration),
      fast: formatRange(config.fast, formatDuration),
      slow: formatRange(config.slow, formatDuration),
      volume: formatRange(config.volume, formatDuration),
      power: formatRange(config.power),
      volumeCap: formatRange(config.volumeCap),
      volumePower: formatRange(config.volumePower),
      deadbandBpsHour: formatRange(config.deadbandBpsHour),
      thresholdLookback: formatRange(config.thresholdLookback, formatDuration),
      thresholdNoiseMultiplier: formatRange(config.thresholdNoiseMultiplier),
    },
    selection: "fit rank -> validation finalist per family -> finalists-only holdout",
    matching: "chronological one-to-one target-state alignment maximizing timing credit",
  });

  const fit = evaluateStage("fit", windows.fit, candidates, bounds, maxWarmupMs, config);
  for (const result of fit) appendResult(config.outputPath, result, false);
  const fitTop = (["volume", "canonical"] as const).flatMap((family) =>
    rank(fit.filter((result) => result.candidate.family === family)).slice(0, config.top));
  const validation = evaluateStage(
    "validation",
    windows.validation,
    fitTop.map((result) => result.candidate),
    bounds,
    maxWarmupMs,
    config,
  );
  for (const result of validation) appendResult(config.outputPath, result, true);
  const finalists = (["volume", "canonical"] as const).map((family) => {
    const selected = rank(validation.filter((result) => result.candidate.family === family))[0];
    if (!selected) throw new Error(`No ${family} validation finalist.`);
    return selected;
  });
  const test = evaluateStage(
    "test",
    windows.test,
    finalists.map((result) => result.candidate),
    bounds,
    maxWarmupMs,
    config,
  );
  for (const result of test) appendResult(config.outputPath, result, true);
  appendJson(config.outputPath, {
    type: "selection",
    finalists: finalists.map((result) => ({
      family: result.candidate.family,
      id: result.candidate.id,
      validationObjective: round(result.aggregate.objective, 6),
    })),
  });
  appendJson(config.outputPath, {
    type: "status",
    status: "completed",
    completedAt: new Date().toISOString(),
    elapsedMs: Date.now() - startedAt,
  });
  fs.writeFileSync(config.reportPath, report(config, windows, fit, validation, test, maxWarmupMs));
  console.error(`Results: ${path.relative(repoRoot, config.outputPath)}`);
  console.error(`Report: ${path.relative(repoRoot, config.reportPath)}`);
}

function searchCandidates(
  config: Args,
  windows: Window[],
  bounds: { start: number; end: number },
  warmupMs: number,
  seeds: Candidate[] = [],
): Candidate[] {
  if (config.algorithm === "random") return [...globalCandidates(), ...seeds, ...generateCandidates(config)];
  const random = mulberry32(config.seed);
  let population = Array.from({ length: Math.max(4, config.trials) }, () => randomGenome(config, random));
  for (let index = 0; index < Math.min(seeds.length, population.length); index += 1) {
    population[index] = candidateGenome(seeds[index]!);
  }
  const screenConfig = {
    ...config,
    scales: config.screenScales.filter((scale) => config.scales.includes(scale)),
  };
  if (screenConfig.scales.length === 0) screenConfig.scales = [config.scales[0]!];
  const screenWindows = windows.slice(0, Math.max(1, Math.min(config.screenWindows, windows.length)));
  let scores = populationFitness(population, screenWindows, bounds, warmupMs, screenConfig, "initial", seeds);
  for (let generation = 0; generation < config.generations; generation += 1) {
    const trials = population.map((target, index) => random() < config.immigrantRate
      ? randomGenome(config, random)
      : differentialTrial(population, target, index, config, random));
    const trialScores = populationFitness(
      trials,
      screenWindows,
      bounds,
      warmupMs,
      screenConfig,
      `generation ${generation + 1}`,
      seeds,
    );
    for (let index = 0; index < population.length; index += 1) {
      if (trialScores[index]! > scores[index]!) {
        population[index] = trials[index]!;
        scores[index] = trialScores[index]!;
      }
    }
    console.error(`DE generation ${generation + 1}/${config.generations}: best ${(Math.max(...scores) * 100).toFixed(2)}%.`);
  }
  return [...globalCandidates(), ...seeds, ...candidatesFromGenomes(population)];
}

function populationFitness(
  genomes: Genome[],
  windows: Window[],
  bounds: { start: number; end: number },
  warmupMs: number,
  config: Args,
  label: string,
  seeds: Candidate[],
): number[] {
  console.error(`DE ${label}: screening ${genomes.length} genomes on ${windows.length} window(s) × ${config.scales.length} scale(s).`);
  const mandatory = [...globalCandidates(), ...seeds];
  const evaluated = evaluateStage(
    "fit",
    windows,
    [...mandatory, ...candidatesFromGenomes(genomes)],
    bounds,
    warmupMs,
    config,
  );
  return genomes.map((_, index) => Math.max(
    evaluated[mandatory.length + index * 2]?.aggregate.objective ?? 0,
    evaluated[mandatory.length + index * 2 + 1]?.aggregate.objective ?? 0,
  ));
}

function differentialTrial(
  population: Genome[],
  target: Genome,
  targetIndex: number,
  config: Args,
  random: () => number,
): Genome {
  const available = population.map((_, index) => index).filter((index) => index !== targetIndex);
  shuffle(available, random);
  const [a, b, c] = available.slice(0, 3).map((index) => genomeVector(population[index]!, config));
  const base = genomeVector(target, config);
  const forced = Math.floor(random() * base.length);
  const vector = base.map((value, index) => index === forced || random() < config.crossoverRate
    ? clamp01(a![index]! + config.differentialWeight * (b![index]! - c![index]!))
    : value);
  return vectorGenome(vector, config);
}

function genomeVector(genome: Genome, config: Args): number[] {
  return [
    logUnit(genome.efficiencyMs, config.efficiency),
    logUnit(genome.fastMs, config.fast),
    logUnit(genome.slowMs, config.slow),
    unit(genome.power, config.power),
    logUnit(genome.volumeMs, config.volume),
    unit(genome.volumeCap, config.volumeCap),
    unit(genome.volumePower, config.volumePower),
    logUnit(genome.deadbandBpsHour, config.deadbandBpsHour),
    genome.deadbandMode === "hold" ? 1 : 0,
    genome.thresholdMode === "adaptive" ? 1 : 0,
    logUnit(genome.thresholdLookbackMs, config.thresholdLookback),
    unit(genome.thresholdNoiseMultiplier, config.thresholdNoiseMultiplier),
  ];
}

function vectorGenome(vector: number[], config: Args): Genome {
  const fastMs = logRange(vector[1]!, config.fast);
  return {
    efficiencyMs: logRange(vector[0]!, config.efficiency),
    fastMs,
    slowMs: Math.max(fastMs, logRange(vector[2]!, config.slow)),
    power: fromUnit(vector[3]!, config.power),
    volumeMs: logRange(vector[4]!, config.volume),
    volumeCap: fromUnit(vector[5]!, config.volumeCap),
    volumePower: fromUnit(vector[6]!, config.volumePower),
    deadbandBpsHour: logRange(vector[7]!, config.deadbandBpsHour),
    deadbandMode: vector[8]! >= 0.5 ? "hold" : "flat",
    thresholdMode: vector[9]! >= 0.5 ? "adaptive" : "static",
    thresholdLookbackMs: logRange(vector[10]!, config.thresholdLookback),
    thresholdNoiseMultiplier: fromUnit(vector[11]!, config.thresholdNoiseMultiplier),
  };
}

interface WindowJob {
  config: Args;
  window: Window;
  windowId: string;
  index: number;
  bounds: { start: number; end: number };
  warmupMs: number;
  seeds: Candidate[];
}

async function runPerWindow(
  config: Args,
  windows: Window[],
  bounds: { start: number; end: number },
  startedAt: number,
): Promise<void> {
  if (config.presetWindowIds && config.presetWindowIds.length !== windows.length) {
    throw new Error("--preset-window-ids must contain one id per fit window.");
  }
  const warmupMs = maximumWarmupMs(config);
  const existing = loadPresets(config.presetOutputPath);
  const jobs = windows.map((window, index): WindowJob => ({
    config: { ...config, seed: config.seed + index * 10_007 },
    window,
    windowId: config.presetWindowIds?.[index] ?? window.label,
    index,
    bounds,
    warmupMs,
    seeds: existing
      .filter((preset) => preset.scope === "window" && preset.windowId === (config.presetWindowIds?.[index] ?? window.label))
      .map(presetCandidate),
  }));
  const grouped = config.workers === 1
    ? jobs.map(optimizeWindow)
    : await parallelMap(jobs, Math.min(config.workers, jobs.length), runWindowWorker);
  const generated = grouped.flat();
  const replaced = new Set(generated.map((preset) => `${preset.windowId}:${preset.intervalMs ?? "all"}`));
  const presets = [
    ...existing.filter((preset) => !replaced.has(`${preset.windowId}:${preset.intervalMs ?? "all"}`)),
    ...generated,
  ].sort((left, right) => `${left.windowId}:${left.intervalMs ?? 0}`.localeCompare(
    `${right.windowId}:${right.intervalMs ?? 0}`,
  ));
  fs.mkdirSync(path.dirname(config.presetOutputPath), { recursive: true });
  fs.writeFileSync(config.presetOutputPath, `${JSON.stringify(presets, null, 2)}\n`);
  fs.mkdirSync(path.dirname(config.reportPath), { recursive: true });
  fs.writeFileSync(config.reportPath, [
    "# VW-KAMA per-window optimization",
    "",
    `Generated: ${new Date().toISOString()}`,
    `Algorithm: ${config.algorithm}; ${config.trials} population/trials; ${config.generations} generations; ${config.scales.map(formatDuration).join(", ")}.`,
    "",
    "These are hindsight upper-bound configurations for comparison, not deployable validation results.",
    "",
    "Every selected preset is scored against the global candidates at the same window and candle scale; the global score is therefore a hard lower bound.",
    "",
    "| window | scale | score | preset |",
    "|---|---:|---:|---|",
    ...generated.map((preset) => `| ${preset.windowId} | ${formatDuration(preset.intervalMs)} | ${pct(preset.score)} | ${preset.id} |`),
    "",
  ].join("\n"));
  console.error(`Presets: ${path.relative(repoRoot, config.presetOutputPath)}`);
  console.error(`Completed in ${formatDuration(Date.now() - startedAt)}.`);
}

function optimizeWindow(job: WindowJob) {
  const { config, window, bounds, warmupMs } = job;
  console.error(`Optimizing window ${job.index + 1}: ${formatWindow(window)}.`);
  const candidates = searchCandidates(config, [window], bounds, warmupMs, job.seeds);
  const evaluated = evaluateStage("fit", [window], candidates, bounds, warmupMs, config);
  return config.scales.map((intervalMs) => {
    const scale = formatDuration(intervalMs);
    const scored = evaluated.flatMap((result) => {
      const score = result.cases.find((item) => item.scale === scale);
      return score ? [{ result, score }] : [];
    }).sort((left, right) => right.score.score - left.score.score
      || left.result.candidate.id.localeCompare(right.result.candidate.id));
    const best = scored[0];
    const baseline = scored.find((item) => item.result.candidate.id.startsWith("global-"));
    if (!best || !baseline || best.score.score + Number.EPSILON < baseline.score.score) {
      throw new Error(`Global lower bound failed for ${window.label} at ${scale}.`);
    }
    return {
      id: `window-${job.windowId}-${scale}`,
      label: `Window best found · ${window.label} · ${scale}`,
      scope: "window" as const,
      windowId: job.windowId,
      intervalMs,
      parameters: candidateParameters(best.result.candidate),
      score: round(best.score.score, 6),
      source: `${config.algorithm.toUpperCase()} hindsight best; global baseline retained`,
    };
  });
}

function runWindowWorker(job: WindowJob): Promise<ReturnType<typeof optimizeWindow>> {
  return new Promise((resolve, reject) => {
    const worker = fork(fileURLToPath(import.meta.url), [], {
      execArgv: ["--import", "tsx"],
      env: { ...process.env, VW_KAMA_WINDOW_JOB: JSON.stringify(job) },
      stdio: ["ignore", "ignore", "inherit", "ipc"],
    });
    worker.once("message", (message: ReturnType<typeof optimizeWindow> | { error: string }) => {
      worker.disconnect();
      if (!Array.isArray(message)) reject(new Error(message.error));
      else resolve(message);
    });
    worker.once("error", reject);
    worker.once("exit", (code) => {
      if (code !== 0) reject(new Error(`VW-KAMA optimizer worker exited with code ${code}.`));
    });
  });
}

async function parallelMap<T, R>(
  values: T[],
  workers: number,
  operation: (value: T) => Promise<R>,
): Promise<R[]> {
  const results = new Array<R>(values.length);
  let next = 0;
  await Promise.all(Array.from({ length: workers }, async () => {
    while (next < values.length) {
      const index = next++;
      results[index] = await operation(values[index]!);
    }
  }));
  return results;
}

function loadPresets(file: string): VwKamaPreset[] {
  if (!fs.existsSync(file)) return [];
  const parsed = JSON.parse(fs.readFileSync(file, "utf8")) as unknown;
  return Array.isArray(parsed) ? parsed as VwKamaPreset[] : [];
}

function evaluateStage(
  stage: Stage,
  windows: Window[],
  candidates: Candidate[],
  bounds: { start: number; end: number },
  warmupMs: number,
  config: Args,
): CandidateResult[] {
  const collected = candidates.map((): CaseScore[] => []);
  for (let windowIndex = 0; windowIndex < windows.length; windowIndex += 1) {
    const window = windows[windowIndex]!;
    const parts = candidates.map(() => new Map<string, CaseStats[]>());
    let sourceCount = 0;
    let segmentCount = 0;
    for (const source of loadSourceSegments(config.sourceDir, {
      start: Math.max(bounds.start, window.start - warmupMs),
      end: window.end,
    }, config.sourceIntervalMs)) {
      sourceCount += source.length;
      const cases = buildCases(stage, [window], config.scales, source, warmupMs, config, false);
      if (cases.length === 0) continue;
      segmentCount += 1;
      console.error(
        `${capitalize(stage)} window ${windowIndex + 1}/${windows.length}, segment ${segmentCount}: `
        + `${source.length.toLocaleString()} source candles, ${candidates.length} candidates.`,
      );
      const progressEvery = Math.max(10, Math.ceil(candidates.length / 10));
      for (let candidateIndex = 0; candidateIndex < candidates.length; candidateIndex += 1) {
        const grouped = parts[candidateIndex]!;
        for (const testCase of cases) {
          const list = grouped.get(testCase.id) ?? [];
          list.push(evaluateCase(candidates[candidateIndex]!, testCase, config));
          grouped.set(testCase.id, list);
        }
        if ((candidateIndex + 1) % progressEvery === 0 || candidateIndex + 1 === candidates.length) {
          console.error(
            `${capitalize(stage)} window ${windowIndex + 1}/${windows.length}, segment ${segmentCount}: `
            + `${candidateIndex + 1}/${candidates.length} candidates.`,
          );
        }
      }
    }
    for (let candidateIndex = 0; candidateIndex < candidates.length; candidateIndex += 1) {
      const grouped = parts[candidateIndex]!;
      for (const scale of config.scales) {
        const id = `${formatDuration(scale)}:${window.label}`;
        const caseParts = grouped.get(id);
        if (!caseParts?.length) {
          throw new Error(`Insufficient continuous candles for ${window.label} at ${formatDuration(scale)}.`);
        }
        collected[candidateIndex]!.push(scoreCase(caseParts));
      }
    }
    console.error(
      `${capitalize(stage)} window ${windowIndex + 1}/${windows.length}: `
      + `${sourceCount.toLocaleString()} source candles across ${segmentCount} scored segments.`,
    );
  }
  return candidates.map((candidate, index) => {
    const cases = orderCases(collected[index]!, windows, config.scales);
    return { candidate, stage, aggregate: aggregate(cases), cases };
  });
}

function orderCases(cases: CaseScore[], windows: Window[], scales: number[]): CaseScore[] {
  const order = new Map(scales.flatMap((scale) => windows.map((window) =>
    `${formatDuration(scale)}:${window.label}`)).map((id, index) => [id, index]));
  return cases.slice().sort((left, right) =>
    (order.get(left.caseId) ?? Number.MAX_SAFE_INTEGER)
    - (order.get(right.caseId) ?? Number.MAX_SAFE_INTEGER));
}

function buildCases(
  stage: Stage,
  windows: Window[],
  scales: number[],
  source: TradingCandle[],
  warmupMs: number,
  config: Args,
  required = true,
): CaseData[] {
  return scales.flatMap((scaleMs) => windows.flatMap((window) => {
    const relevant = source.filter((candle) =>
      candle.openTime >= window.start - warmupMs && candle.openTime < window.end);
    const segments = continuousSegments(
      aggregateCandles(relevant, config.sourceIntervalMs, scaleMs),
      scaleMs,
    );
    const cases = segments.flatMap((candles, segment) => {
      const scoreAt = Math.max(window.start, (candles[0]?.openTime ?? window.end) + warmupMs);
      const scoreStart = candles.findIndex((candle) => candle.openTime >= scoreAt);
      if (scoreStart < 0 || candles.length - scoreStart < 3) return [];
      const scored = candles.slice(scoreStart).filter((candle) => candle.openTime < window.end);
      if (scored.length < 3) return [];
      const caseCandles = candles.slice(0, scoreStart).concat(scored);
      const oracle = perfectMarginOracle(caseCandles, {
        startingQuote: 1,
        leverage: 1,
        friction: config.oracleFriction,
        eventMode: "close",
        maxPathCandles: 1,
      });
      return [{
        id: `${formatDuration(scaleMs)}:${window.label}`,
        stage,
        scaleMs,
        window,
        candles: caseCandles,
        scoreStart,
        oracle,
        days: Math.max(scaleMs / DAY, scored.length * scaleMs / DAY),
      }];
    });
    if (required && cases.length === 0) {
      throw new Error(`Insufficient continuous candles for ${window.label} at ${formatDuration(scaleMs)}.`);
    }
    return cases;
  }));
}

function evaluateCase(candidate: Candidate, testCase: CaseData, config: Args): CaseStats {
  const result = evaluateVwKamaOracle(testCase.candles, {
    intervalMs: testCase.scaleMs,
    scoreStartTime: testCase.candles[testCase.scoreStart]!.openTime,
    parameters: {
      efficiencyMs: candidate.efficiencyMs,
      fastMs: candidate.fastMs,
      slowMs: candidate.slowMs,
      power: candidate.power,
      volumeMs: candidate.volumeMs,
      volumeCap: candidate.volumeCap,
      volumePower: candidate.volumePower,
      deadbandBpsHour: candidate.deadbandBpsHour,
      deadbandMode: candidate.deadbandMode,
      thresholdMode: candidate.thresholdMode,
      thresholdLookbackMs: candidate.thresholdLookbackMs,
      thresholdNoiseMultiplier: candidate.thresholdNoiseMultiplier,
    },
    oracleFriction: config.oracleFriction,
    matchWindowMs: config.matchWindowMs,
    timingHalfLifeMs: config.timingHalfLifeMs,
    warmupMultiple: config.warmupMultiple,
    oracleResult: testCase.oracle,
    includeTrace: false,
  });
  const metrics = result.metrics;
  return {
    caseId: testCase.id,
    scale: formatDuration(testCase.scaleMs),
    window: testCase.window.label,
    timingCredit: result.candidateTransitions.reduce((sum, item) => sum + item.timingCredit, 0),
    matchedCount: metrics.matchedCount,
    signalCount: metrics.signalCount,
    oracleCount: metrics.oracleCount,
    absoluteLags: result.candidateTransitions.flatMap((item) =>
      item.lagMs === null ? [] : [Math.abs(item.lagMs)]),
    signedLags: result.candidateTransitions.flatMap((item) =>
      item.lagMs === null ? [] : [item.lagMs]),
    stateCredit: metrics.exposureAgreement * result.candleCount,
    stateCount: result.candleCount,
    days: testCase.days,
  };
}

function scoreCase(parts: CaseStats[]): CaseScore {
  const first = parts[0]!;
  const signalCount = sum(parts.map((part) => part.signalCount));
  const oracleCount = sum(parts.map((part) => part.oracleCount));
  const timingCredit = sum(parts.map((part) => part.timingCredit));
  const matchedCount = sum(parts.map((part) => part.matchedCount));
  const precision = eventRatio(timingCredit, signalCount, oracleCount);
  const recall = eventRatio(timingCredit, oracleCount, signalCount);
  const f1 = harmonic(precision, recall);
  const stateCount = sum(parts.map((part) => part.stateCount));
  const exposureAgreement = ratio(sum(parts.map((part) => part.stateCredit)), stateCount);
  const extraSignalCount = signalCount - matchedCount;
  const signalCleanliness = signalCount > 0 ? matchedCount / signalCount : 1;
  const days = sum(parts.map((part) => part.days));
  const absoluteLags = parts.flatMap((part) => part.absoluteLags);
  const signedLags = parts.flatMap((part) => part.signedLags);
  return {
    caseId: first.caseId,
    scale: first.scale,
    window: first.window,
    score: vwKamaScore(f1, exposureAgreement, signalCleanliness),
    precision,
    recall,
    f1,
    rawPrecision: eventRatio(matchedCount, signalCount, oracleCount),
    rawRecall: eventRatio(matchedCount, oracleCount, signalCount),
    exposureAgreement,
    noiseSignalRatio: matchedCount > 0 ? extraSignalCount / matchedCount : extraSignalCount > 0 ? null : 0,
    signalCleanliness,
    signalsPerDay: ratio(signalCount, days),
    lagP50Ms: nullablePercentile(absoluteLags, 0.5),
    lagP90Ms: nullablePercentile(absoluteLags, 0.9),
    lagP95Ms: nullablePercentile(absoluteLags, 0.95),
    lagMedianSignedMs: nullablePercentile(signedLags, 0.5),
  };
}

function aggregate(scores: CaseScore[]): AggregateScore {
  const values = scores.map((score) => score.score);
  const medianScore = percentile(values, 0.5);
  const p10 = percentile(values, 0.1);
  return {
    objective: (medianScore + p10) / 2,
    median: medianScore,
    p10,
    precision: median(scores.map((score) => score.precision)),
    recall: median(scores.map((score) => score.recall)),
    f1: median(scores.map((score) => score.f1)),
    exposureAgreement: median(scores.map((score) => score.exposureAgreement)),
    noiseSignalRatio: nullableMedian(scores.map((score) => score.noiseSignalRatio)),
    signalCleanliness: median(scores.map((score) => score.signalCleanliness)),
    signalsPerDay: median(scores.map((score) => score.signalsPerDay)),
    lagP50Ms: nullableMedian(scores.map((score) => score.lagP50Ms)),
    lagP90Ms: nullableMedian(scores.map((score) => score.lagP90Ms)),
  };
}

function generateCandidates(config: Args): Candidate[] {
  const random = mulberry32(config.seed);
  return candidatesFromGenomes(Array.from({ length: config.trials }, () => randomGenome(config, random)));
}

function globalCandidates(): Candidate[] {
  const threshold = {
    thresholdMode: "static" as const,
    thresholdLookbackMs: HOUR,
    thresholdNoiseMultiplier: 0,
  };
  const refined = {
    efficiencyMs: 1_856_036,
    fastMs: 1_873_481,
    slowMs: 3_701_806,
    power: 0.61357,
    volumeMs: 440_028,
    volumeCap: 1.97271,
    deadbandBpsHour: 0.44351,
    deadbandMode: "hold" as const,
    ...threshold,
  };
  return [
    {
      id: "global-clean-v0182",
      family: "volume",
      efficiencyMs: 265_846,
      fastMs: 1_113_418,
      slowMs: 64_800_000,
      power: 0.62547,
      volumeMs: 283_525,
      volumeCap: 5.08117,
      volumePower: 0.48543,
      deadbandBpsHour: 30,
      deadbandMode: "hold",
      thresholdMode: "adaptive",
      thresholdLookbackMs: 2_457_073,
      thresholdNoiseMultiplier: 0,
    },
    {
      id: "global-clean-k0050",
      family: "canonical",
      efficiencyMs: 1_420_250,
      fastMs: 976_600,
      slowMs: 976_600,
      power: 0.79612,
      volumeMs: 82_113,
      volumeCap: 4.27354,
      volumePower: 0,
      deadbandBpsHour: 21.21468,
      deadbandMode: "hold",
      thresholdMode: "static",
      thresholdLookbackMs: 900_000,
      thresholdNoiseMultiplier: 5.60731,
    },
    {
      id: "global-runtime-k0021",
      family: "canonical",
      efficiencyMs: 14 * 60_000,
      fastMs: 28 * 60_000,
      slowMs: 153 * 60_000,
      power: 0.49045,
      volumeMs: 130 * 60_000,
      volumeCap: 2.65003,
      volumePower: 0,
      deadbandBpsHour: 67.56654,
      deadbandMode: "flat",
      ...threshold,
    },
    { id: "global-refined-v0016", family: "volume", volumePower: 1.72738, ...refined },
    { id: "global-refined-k0016", family: "canonical", volumePower: 0, ...refined },
  ];
}

function randomGenome(config: Args, random: () => number): Genome {
  const fastMs = logRandom(random, config.fast);
  return {
    efficiencyMs: logRandom(random, config.efficiency),
    fastMs,
    slowMs: Math.max(fastMs, logRandom(random, config.slow)),
    power: linearRandom(random, config.power),
    volumeMs: logRandom(random, config.volume),
    volumeCap: linearRandom(random, config.volumeCap),
    volumePower: linearRandom(random, config.volumePower),
    deadbandBpsHour: logRandom(random, config.deadbandBpsHour),
    deadbandMode: random() < 0.5 ? "flat" : "hold",
    thresholdMode: random() < 0.5 ? "static" : "adaptive",
    thresholdLookbackMs: logRandom(random, config.thresholdLookback),
    thresholdNoiseMultiplier: linearRandom(random, config.thresholdNoiseMultiplier),
  };
}

function candidatesFromGenomes(genomes: Genome[]): Candidate[] {
  return genomes.flatMap((genome, index) => {
    const suffix = String(index + 1).padStart(4, "0");
    return [
      { ...genome, id: `v${suffix}`, family: "volume" as const },
      { ...genome, id: `k${suffix}`, family: "canonical" as const, volumePower: 0 },
    ];
  });
}

function candidateParameters(candidate: Candidate) {
  return {
    efficiencyMs: candidate.efficiencyMs,
    fastMs: candidate.fastMs,
    slowMs: candidate.slowMs,
    power: candidate.power,
    volumeMs: candidate.volumeMs,
    volumeCap: candidate.volumeCap,
    volumePower: candidate.volumePower,
    deadbandBpsHour: candidate.deadbandBpsHour,
    deadbandMode: candidate.deadbandMode,
    thresholdMode: candidate.thresholdMode,
    thresholdLookbackMs: candidate.thresholdLookbackMs,
    thresholdNoiseMultiplier: candidate.thresholdNoiseMultiplier,
  };
}

function candidateGenome(candidate: Candidate): Genome {
  const { id: _id, family: _family, ...genome } = candidate;
  return genome;
}

function presetCandidate(preset: VwKamaPreset, index: number): Candidate {
  return {
    id: `seed-${preset.intervalMs ?? index}-${index}`,
    family: preset.parameters.volumePower > 0 ? "volume" : "canonical",
    ...preset.parameters,
    thresholdMode: preset.parameters.thresholdMode ?? "static",
    thresholdLookbackMs: preset.parameters.thresholdLookbackMs ?? HOUR,
    thresholdNoiseMultiplier: preset.parameters.thresholdNoiseMultiplier ?? 0,
  };
}

function rank(results: CandidateResult[]): CandidateResult[] {
  return results.slice().sort((left, right) =>
    right.aggregate.objective - left.aggregate.objective
    || right.aggregate.p10 - left.aggregate.p10
    || left.candidate.id.localeCompare(right.candidate.id));
}

function aggregateCandles(candles: TradingCandle[], sourceMs: number, targetMs: number): TradingCandle[] {
  if (targetMs === sourceMs) return candles.slice();
  const result: TradingCandle[] = [];
  const expected = targetMs / sourceMs;
  let current: TradingCandle | undefined;
  let bucket = -1;
  let count = 0;
  let contiguous = true;
  let lastOpen = -1;
  const flush = () => {
    if (current && contiguous && count === expected) result.push(current);
  };
  for (const candle of candles) {
    const nextBucket = Math.floor(candle.openTime / targetMs) * targetMs;
    if (nextBucket !== bucket) {
      flush();
      bucket = nextBucket;
      count = 1;
      contiguous = candle.openTime === bucket;
      lastOpen = candle.openTime;
      current = { ...candle, openTime: bucket, closeTime: bucket + targetMs - 1 };
      continue;
    }
    if (!current) continue;
    contiguous &&= candle.openTime === lastOpen + sourceMs;
    lastOpen = candle.openTime;
    current.high = Math.max(current.high, candle.high);
    current.low = Math.min(current.low, candle.low);
    current.close = candle.close;
    current.volume += candle.volume;
    count += 1;
  }
  flush();
  return result;
}

function continuousSegments(candles: TradingCandle[], intervalMs: number): TradingCandle[][] {
  const segments: TradingCandle[][] = [];
  let current: TradingCandle[] = [];
  for (const candle of candles) {
    const previous = current.at(-1);
    if (previous && candle.openTime !== previous.openTime + intervalMs) {
      if (current.length > 0) segments.push(current);
      current = [];
    }
    current.push(candle);
  }
  if (current.length > 0) segments.push(current);
  return segments;
}

function* loadSourceSegments(
  sourceDir: string,
  range: { start: number; end: number },
  intervalMs: number,
): Generator<TradingCandle[]> {
  let candles: TradingCandle[] = [];
  let found = false;
  for (let time = utcDay(range.start); time < range.end; time += DAY) {
    const date = new Date(time).toISOString().slice(0, 10);
    const file = path.join(sourceDir, `${date}.jsonl`);
    if (!fs.existsSync(file)) continue;
    for (const line of fs.readFileSync(file, "utf8").split("\n")) {
      if (!line.trim()) continue;
      const parsed = JSON.parse(line) as TradingCandle & { interval?: string; closed?: boolean };
      if (!validCandle(parsed) || parsed.closed === false) continue;
      if (parsed.openTime < range.start || parsed.openTime >= range.end) continue;
      if (parsed.interval && parseDuration(parsed.interval) !== intervalMs) {
        throw new Error(`Candle interval ${parsed.interval} does not match --source-interval ${formatDuration(intervalMs)}.`);
      }
      const candle: TradingCandle = {
        openTime: parsed.openTime,
        closeTime: parsed.closeTime,
        open: parsed.open,
        high: parsed.high,
        low: parsed.low,
        close: parsed.close,
        volume: parsed.volume,
      };
      const previous = candles.at(-1);
      if (previous && candle.openTime < previous.openTime) {
        throw new Error(`Source candles are not ordered at ${new Date(candle.openTime).toISOString()}.`);
      }
      if (previous?.openTime === candle.openTime) {
        candles[candles.length - 1] = candle;
      } else {
        if (previous && candle.openTime !== previous.openTime + intervalMs) {
          yield candles;
          candles = [];
        }
        candles.push(candle);
        found = true;
      }
    }
  }
  if (candles.length > 0) yield candles;
  if (!found) throw new Error("No source candles found for the selected window.");
}

function sourceBounds(sourceDir: string): { start: number; end: number } {
  const dates = fs.readdirSync(sourceDir)
    .map((file) => /^(\d{4}-\d{2}-\d{2})\.jsonl$/.exec(file)?.[1])
    .filter((date): date is string => Boolean(date))
    .sort();
  if (dates.length === 0) throw new Error(`No daily JSONL shards in ${sourceDir}.`);
  const start = Date.parse(`${dates[0]}T00:00:00Z`);
  const shardEnd = Date.parse(`${dates.at(-1)}T00:00:00Z`) + DAY;
  const lastCompletedDay = utcDay(Date.now());
  return { start, end: Math.max(start + 1, Math.min(shardEnd, lastCompletedDay)) };
}

function defaultWindows(start: number, end: number, sourceMs: number): Record<Stage, Window[]> {
  const unit = Math.min(60 * DAY, 250_000 * sourceMs, Math.floor((end - start) / 8));
  if (unit < 1_000) throw new Error("Source history is too short for automatic chronological windows.");
  const base = end - unit * 8;
  const make = (label: string, index: number): Window => ({ label, start: base + index * unit, end: base + (index + 1) * unit });
  return {
    fit: [0, 1, 2, 3].map((index) => make(`fit-${index + 1}`, index)),
    validation: [4, 5].map((index) => make(`validation-${index - 3}`, index)),
    test: [6, 7].map((index) => make(`test-${index - 5}`, index)),
  };
}

function assertChronological(fit: Window[], validation: Window[], test: Window[]): void {
  for (const [label, windows] of [["fit", fit], ["validation", validation], ["test", test]] as const) {
    if (windows.length === 0) throw new Error(`${label} windows cannot be empty.`);
    if (windows.some((window) => window.end <= window.start)) throw new Error(`Invalid ${label} window.`);
  }
  if (Math.max(...fit.map((window) => window.end)) > Math.min(...validation.map((window) => window.start))
    || Math.max(...validation.map((window) => window.end)) > Math.min(...test.map((window) => window.start))) {
    throw new Error("Fit, validation, and test windows must be chronological and non-overlapping.");
  }
}

function parseArgs(argv: string[]): Args {
  const allowed = new Set([
    "source-dir", "source-interval", "scales", "fit-windows", "validation-windows",
    "test-windows", "trials", "top", "seed", "oracle-friction", "match-window",
    "timing-half-life", "warmup-multiple", "case-warmup", "efficiency-range", "fast-range",
    "slow-range", "volume-range", "power-range", "volume-cap-range",
    "volume-power-range", "deadband-bps-hour-range", "threshold-lookback-range",
    "threshold-noise-multiplier-range", "algorithm", "mode", "generations",
    "differential-weight", "crossover-rate", "immigrant-rate", "screen-windows",
    "screen-scales", "workers", "preset-window-ids", "preset-output", "output", "report",
  ]);
  const values = new Map<string, string>();
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index]!;
    if (token === "--help") { help(); process.exit(0); }
    if (!token.startsWith("--")) throw new Error(`Unexpected argument: ${token}`);
    const [rawKey, inline] = token.slice(2).split("=", 2);
    const value = inline ?? argv[++index];
    if (!rawKey || !value || value.startsWith("--")) throw new Error(`Missing value for --${rawKey}.`);
    if (!allowed.has(rawKey)) throw new Error(`Unknown option: --${rawKey}.`);
    values.set(rawKey, value);
  }
  const get = (key: string, fallback: string) => values.get(key) ?? fallback;
  const sourceDir = path.resolve(repoRoot, get("source-dir", "data/historical/spot-btcusdt/btcusdt/1m"));
  const outputPath = path.resolve(repoRoot, get("output", `data/benchmarks/volume-weighted-kama-${stamp}.jsonl`));
  const reportPath = path.resolve(repoRoot, get("report", `docs/volume-weighted-kama-${stamp}.md`));
  const algorithm = get("algorithm", "de");
  const mode = get("mode", "standard");
  if (algorithm !== "random" && algorithm !== "de") throw new Error("--algorithm must be random or de.");
  if (mode !== "standard" && mode !== "per-window") throw new Error("--mode must be standard or per-window.");
  return {
    sourceDir,
    sourceIntervalMs: parseDuration(get("source-interval", "1m")),
    scales: unique(get("scales", "1m,5m,15m,1h").split(",").map(parseDuration)).sort((a, b) => a - b),
    fitWindows: values.has("fit-windows") ? parseWindows(values.get("fit-windows")!, "fit") : undefined,
    validationWindows: values.has("validation-windows") ? parseWindows(values.get("validation-windows")!, "validation") : undefined,
    testWindows: values.has("test-windows") ? parseWindows(values.get("test-windows")!, "test") : undefined,
    trials: integer(get("trials", "48"), "trials"),
    algorithm,
    mode,
    generations: integer(get("generations", "4"), "generations", 0),
    differentialWeight: bounded(get("differential-weight", "0.75"), "differential-weight", 0, 2),
    crossoverRate: bounded(get("crossover-rate", "0.8"), "crossover-rate", 0, 1),
    immigrantRate: bounded(get("immigrant-rate", "0.08"), "immigrant-rate", 0, 1),
    screenWindows: integer(get("screen-windows", "1"), "screen-windows"),
    screenScales: unique(get("screen-scales", "1m,15m").split(",").map(parseDuration)).sort((a, b) => a - b),
    workers: integer(get("workers", String(Math.max(1, Math.min(4, Math.floor((Number(process.env.UV_THREADPOOL_SIZE) || 4) / 2))))), "workers"),
    presetWindowIds: values.has("preset-window-ids")
      ? values.get("preset-window-ids")!.split(",").filter(Boolean)
      : undefined,
    presetOutputPath: path.resolve(repoRoot, get("preset-output", "data/benchmarks/vw-kama-window-presets.json")),
    top: integer(get("top", "6"), "top"),
    seed: integer(get("seed", "17"), "seed", 0),
    oracleFriction: nonNegative(get("oracle-friction", "0.00175"), "oracle-friction"),
    matchWindowMs: parseDuration(get("match-window", "2h")),
    timingHalfLifeMs: parseDuration(get("timing-half-life", "10m")),
    warmupMultiple: positive(get("warmup-multiple", "3"), "warmup-multiple"),
    caseWarmupMs: values.has("case-warmup") ? parseDuration(values.get("case-warmup")!) : 0,
    efficiency: durationRange(get("efficiency-range", "1m..2h"), "efficiency-range"),
    fast: durationRange(get("fast-range", "1s..30m"), "fast-range"),
    slow: durationRange(get("slow-range", "5m..24h"), "slow-range"),
    volume: durationRange(get("volume-range", "1m..12h"), "volume-range"),
    power: numberRange(get("power-range", "0.3..5"), "power-range", 0.1),
    volumeCap: numberRange(get("volume-cap-range", "1..10"), "volume-cap-range", 1),
    volumePower: numberRange(get("volume-power-range", "0.01..3"), "volume-power-range", 0),
    deadbandBpsHour: numberRange(get("deadband-bps-hour-range", "0.05..2000"), "deadband-bps-hour-range", 0),
    thresholdLookback: durationRange(get("threshold-lookback-range", "5m..24h"), "threshold-lookback-range"),
    thresholdNoiseMultiplier: numberRange(
      get("threshold-noise-multiplier-range", "0..8"),
      "threshold-noise-multiplier-range",
      0,
    ),
    outputPath,
    reportPath,
  };
}

function parseWindows(value: string, prefix: string): Window[] {
  return value.split(",").filter(Boolean).map((part, index) => {
    const pieces = part.split("..");
    if (pieces.length !== 2) throw new Error(`--${prefix}-windows entries must be start..end.`);
    const start = endpoint(pieces[0]!, false);
    const end = endpoint(pieces[1]!, true);
    return { label: `${prefix}-${index + 1}`, start, end };
  });
}

function endpoint(value: string, end: boolean): number {
  const dateOnly = /^\d{4}-\d{2}-\d{2}$/.test(value);
  const parsed = Date.parse(dateOnly ? `${value}T00:00:00Z` : value);
  if (!Number.isFinite(parsed)) throw new Error(`Invalid date/time: ${value}`);
  return parsed + (end && dateOnly ? DAY : 0);
}

function durationRange(value: string, label: string): Range {
  const [min, max, ...extra] = value.split("..").map(parseDuration);
  if (extra.length || min === undefined || max === undefined || max < min) throw new Error(`Invalid --${label}.`);
  return { min, max };
}

function numberRange(value: string, label: string, minimum: number): Range {
  const [min, max, ...extra] = value.split("..").map(Number);
  if (extra.length || min === undefined || max === undefined || !Number.isFinite(min) || !Number.isFinite(max)
    || min < minimum || max < min) throw new Error(`Invalid --${label}.`);
  return { min, max };
}

function parseDuration(value: string): number {
  const match = /^(\d+(?:\.\d+)?)(ms|s|m|h|d|w)?$/.exec(value.trim());
  if (!match) throw new Error(`Invalid duration: ${value}`);
  const factors: Record<string, number> = { ms: 1, s: 1_000, m: 60_000, h: HOUR, d: DAY, w: 7 * DAY };
  const result = Number(match[1]) * factors[match[2] ?? "ms"]!;
  if (!Number.isFinite(result) || result <= 0) throw new Error(`Duration must be positive: ${value}`);
  return Math.round(result);
}

function validateScales(scales: number[], sourceMs: number): void {
  for (const scale of scales) {
    if (scale < sourceMs) throw new Error(`Target ${formatDuration(scale)} is finer than source ${formatDuration(sourceMs)}.`);
    if (scale % sourceMs !== 0) throw new Error(`Target ${formatDuration(scale)} is not divisible by source ${formatDuration(sourceMs)}.`);
  }
}

function candidateWarmupSamples(candidate: Candidate, scaleMs: number, multiple: number): number {
  const kama = volumeWeightedKamaWarmupSamples({
    efficiencyPeriod: samples(candidate.efficiencyMs, scaleMs),
    slowPeriod: samples(candidate.slowMs, scaleMs),
    volumePeriod: samples(candidate.volumeMs, scaleMs),
  }, multiple);
  const threshold = candidate.thresholdMode === "adaptive"
    ? samples(candidate.thresholdLookbackMs, scaleMs) * multiple
    : 0;
  return Math.max(kama, threshold);
}

function maximumWarmupMs(config: Args): number {
  return Math.max(
    config.caseWarmupMs,
    config.efficiency.max * config.warmupMultiple,
    config.slow.max * config.warmupMultiple,
    config.volume.max * config.warmupMultiple,
    config.thresholdLookback.max * config.warmupMultiple,
  );
}

function samples(durationMs: number, scaleMs: number): number {
  return Math.max(1, Math.round(durationMs / scaleMs));
}

function appendResult(file: string, result: CandidateResult, includeCases: boolean): void {
  appendJson(file, {
    type: "candidate",
    stage: result.stage,
    candidate: compactCandidate(result.candidate),
    aggregate: roundObject(result.aggregate),
    ...(includeCases ? { cases: result.cases.map(roundObject) } : {}),
  });
}

function report(
  config: Args,
  windows: Record<Stage, Window[]>,
  fit: CandidateResult[],
  validation: CandidateResult[],
  test: CandidateResult[],
  caseWarmupMs: number,
): string {
  const lines = [
    "# Volume-weighted KAMA oracle approximation search",
    "",
    `Generated: ${new Date().toISOString()}`,
    `Raw results: \`${path.relative(repoRoot, config.outputPath)}\``,
    "",
    "The fit ranking never reads validation or holdout scores. The best validated volume-aware and canonical (`volumePower=0`) candidates are the only candidates evaluated on holdout.",
    "",
    "## Data and objective",
    "",
    `- Source: \`${path.relative(repoRoot, config.sourceDir)}\` (${formatDuration(config.sourceIntervalMs)}); target scales: ${config.scales.map(formatDuration).join(", ")}.`,
    `- Windows: ${Object.entries(windows).map(([stage, list]) => `${stage} ${list.map(formatWindow).join(", ")}`).join("; ")}.`,
    `- Each continuous segment reserves ${formatDuration(caseWarmupMs)} before scoring.`,
    "- Signal: completed-candle volume-weighted KAMA derivative rate; candidates either go flat inside the deadband or hold their prior exposure until the opposite threshold.",
    `- Signal memory: after the first signal, a candidate state change emits only when the current close is strictly more than ${round(config.oracleFriction * 10_000, 3)} bps from the last emitted signal price; rejected changes retain the prior state. This is the same friction used by the oracle.`,
    "- Matching is one chronological one-to-one alignment by resulting state. It maximizes total timing credit, so extra candidate transitions reduce precision and uncovered oracle transitions reduce recall.",
    "- Case score weights timing-credited transition F1 60%, graded oracle-state agreement 30%, and signal cleanliness 10%. Cleanliness is matched / (matched + extra); the displayed noise/signal ratio is extra / matched. Trading returns and execution prices are neither computed nor ranked.",
    "- Candidate objective equally weights the median and P10 case score; every scale/window case has equal weight.",
    "",
    "## Holdout finalists",
    "",
    resultTable(test),
    "",
    "## Validation finalists",
    "",
    resultTable(rank(validation)),
    "",
    "## Best fit candidates",
    "",
    resultTable(rank(fit).slice(0, 12)),
    "",
    "## Finalist parameters",
    "",
    "| family | id | efficiency | fast | slow | power | volume | cap | volume power | base threshold | state mode | threshold mode | noise lookback | noise multiplier |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|",
    ...test.map(({ candidate }) => `| ${candidate.family} | ${candidate.id} | ${formatDuration(candidate.efficiencyMs)} | ${formatDuration(candidate.fastMs)} | ${formatDuration(candidate.slowMs)} | ${round(candidate.power, 3)} | ${formatDuration(candidate.volumeMs)} | ${round(candidate.volumeCap, 3)} | ${round(candidate.volumePower, 3)} | ${round(candidate.deadbandBpsHour, 3)} bps/hour | ${candidate.deadbandMode} | ${candidate.thresholdMode} | ${formatDuration(candidate.thresholdLookbackMs)} | ${round(candidate.thresholdNoiseMultiplier, 3)} |`),
    "",
    "## Holdout cases",
    "",
    "| candidate | scale/window | score | precision | recall | F1 | agreement | cleanliness | noise/signal | signals/day | timing error P50 | timing error P90 |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ...test.flatMap((result) => result.cases.map((item) => `| ${result.candidate.id} | ${item.caseId} | ${pct(item.score)} | ${pct(item.precision)} | ${pct(item.recall)} | ${pct(item.f1)} | ${pct(item.exposureAgreement)} | ${pct(item.signalCleanliness)} | ${formatNullableRatio(item.noiseSignalRatio)} | ${round(item.signalsPerDay, 2)} | ${formatNullableDuration(item.lagP50Ms)} | ${formatNullableDuration(item.lagP90Ms)} |`)),
    "",
  ];
  return lines.join("\n");
}

function resultTable(results: CandidateResult[]): string {
  return [
    "| family | id | objective | median | P10 | precision | recall | F1 | agreement | cleanliness | noise/signal | signals/day | timing error P50 |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ...results.map((result) => {
      const score = result.aggregate;
      return `| ${result.candidate.family} | ${result.candidate.id} | ${pct(score.objective)} | ${pct(score.median)} | ${pct(score.p10)} | ${pct(score.precision)} | ${pct(score.recall)} | ${pct(score.f1)} | ${pct(score.exposureAgreement)} | ${pct(score.signalCleanliness)} | ${formatNullableRatio(score.noiseSignalRatio)} | ${round(score.signalsPerDay, 2)} | ${formatNullableDuration(score.lagP50Ms)} |`;
    }),
  ].join("\n");
}

function compactCandidate(candidate: Candidate): Record<string, string | number> {
  return {
    id: candidate.id,
    family: candidate.family,
    efficiency: formatDuration(candidate.efficiencyMs),
    fast: formatDuration(candidate.fastMs),
    slow: formatDuration(candidate.slowMs),
    power: round(candidate.power, 5),
    volume: formatDuration(candidate.volumeMs),
    volumeCap: round(candidate.volumeCap, 5),
    volumePower: round(candidate.volumePower, 5),
    deadbandBpsHour: round(candidate.deadbandBpsHour, 5),
    deadbandMode: candidate.deadbandMode,
    thresholdMode: candidate.thresholdMode,
    thresholdLookback: formatDuration(candidate.thresholdLookbackMs),
    thresholdNoiseMultiplier: round(candidate.thresholdNoiseMultiplier, 5),
  };
}

function help(): void {
  console.log(`Usage: npm run search:kama -- [options]

  --source-dir PATH                 Daily Binance JSONL shard directory
  --source-interval 1m             Source candle scale (ms, s, m, h, d, w)
  --scales 1m,5m,15m,1h            Causal target aggregations; never finer than source
  --fit-windows START..END,...      Date-only ends are inclusive; timestamps are exclusive
  --validation-windows ...          Must follow fit windows chronologically
  --test-windows ...                Untouched until two family finalists are selected
  --algorithm de --trials 48       Differential evolution (or random) population
  --generations 4 --seed 17        Deterministic evolution controls
  --differential-weight 0.75 --crossover-rate 0.8 --immigrant-rate 0.08
  --screen-windows 1 --screen-scales 1m,15m   Cheap early-pruning stage
  --efficiency-range 1m..2h         Duration ranges become sample counts per scale
  --fast-range 1s..30m --slow-range 5m..24h --volume-range 1m..12h
  --power-range 0.3..5 --volume-cap-range 1..10 --volume-power-range 0.01..3
  --deadband-bps-hour-range 0.05..2000
  --threshold-lookback-range 5m..24h --threshold-noise-multiplier-range 0..8
  --mode per-window --preset-window-ids ID,... --preset-output PATH
  --match-window 2h --timing-half-life 10m
  --oracle-friction 0.00175 --warmup-multiple 3 --case-warmup 72h
  --output PATH --report PATH`);
}

function validCandle(candle: TradingCandle): boolean {
  return Number.isFinite(candle.openTime) && Number.isFinite(candle.closeTime)
    && candle.open > 0 && candle.high > 0 && candle.low > 0 && candle.close > 0
    && Number.isFinite(candle.volume) && candle.volume >= 0;
}

function appendJson(file: string, value: unknown): void { fs.appendFileSync(file, `${JSON.stringify(value)}\n`); }
function utcDay(time: number): number { return Math.floor(time / DAY) * DAY; }
function ratio(value: number, total: number): number { return total > 0 ? value / total : 0; }
function eventRatio(value: number, total: number, oppositeTotal: number): number {
  return total > 0 ? value / total : oppositeTotal === 0 ? 1 : 0;
}
function harmonic(a: number, b: number): number { return a + b > 0 ? 2 * a * b / (a + b) : 0; }
function sum(values: number[]): number { return values.reduce((total, value) => total + value, 0); }
function mean(values: number[]): number { return values.length ? sum(values) / values.length : 0; }
function unique(values: number[]): number[] { return [...new Set(values)]; }
function median(values: number[]): number { return percentile(values, 0.5); }
function nullableMedian(values: Array<number | null>): number | null { return nullablePercentile(values.filter((value): value is number => value !== null), 0.5); }
function nullablePercentile(values: number[], quantile: number): number | null { return values.length ? percentile(values, quantile) : null; }
function percentile(values: number[], quantile: number): number {
  if (!values.length) return 0;
  const sorted = values.slice().sort((a, b) => a - b);
  const index = (sorted.length - 1) * quantile;
  const lower = Math.floor(index);
  const rate = index - lower;
  return (sorted[lower] ?? 0) * (1 - rate) + (sorted[lower + 1] ?? sorted[lower] ?? 0) * rate;
}
function round(value: number, digits = 4): number { return Number(value.toFixed(digits)); }
function roundObject<T extends object>(value: T): T { return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, typeof item === "number" ? round(item, 6) : item])) as T; }
function pct(value: number): string { return `${round(value * 100, 2)}%`; }
function capitalize(value: string): string { return value[0]!.toUpperCase() + value.slice(1); }
function formatWindow(window: Window): string { return `${new Date(window.start).toISOString().slice(0, 10)}..${new Date(window.end - 1).toISOString().slice(0, 10)}`; }
function formatNullableDuration(value: number | null): string { return value === null ? "-" : formatDuration(value); }
function formatNullableRatio(value: number | null): string { return value === null ? "∞" : round(value, 3).toString(); }
function formatDuration(ms: number): string {
  for (const [suffix, size] of [["w", 7 * DAY], ["d", DAY], ["h", HOUR], ["m", 60_000], ["s", 1_000]] as const) {
    if (ms >= size && ms % size === 0) return `${ms / size}${suffix}`;
  }
  return `${round(ms, 0)}ms`;
}
function formatRange(range: Range, format: (value: number) => string = String): string { return `${format(range.min)}..${format(range.max)}`; }
function positive(value: string, label: string): number { const parsed = Number(value); if (!Number.isFinite(parsed) || parsed <= 0) throw new Error(`--${label} must be positive.`); return parsed; }
function nonNegative(value: string, label: string): number { const parsed = Number(value); if (!Number.isFinite(parsed) || parsed < 0) throw new Error(`--${label} must be non-negative.`); return parsed; }
function bounded(value: string, label: string, minimum: number, maximum: number): number { const parsed = Number(value); if (!Number.isFinite(parsed) || parsed < minimum || parsed > maximum) throw new Error(`--${label} must be between ${minimum} and ${maximum}.`); return parsed; }
function integer(value: string, label: string, minimum = 1): number { const parsed = Number(value); if (!Number.isSafeInteger(parsed) || parsed < minimum) throw new Error(`--${label} must be an integer >= ${minimum}.`); return parsed; }
function linearRandom(random: () => number, range: Range): number { return range.min + random() * (range.max - range.min); }
function logRandom(random: () => number, range: Range): number { return range.min <= 0 ? linearRandom(random, range) : Math.exp(Math.log(range.min) + random() * (Math.log(range.max) - Math.log(range.min))); }
function unit(value: number, range: Range): number { return range.max === range.min ? 0 : clamp01((value - range.min) / (range.max - range.min)); }
function fromUnit(value: number, range: Range): number { return range.min + clamp01(value) * (range.max - range.min); }
function logUnit(value: number, range: Range): number { return range.min <= 0 ? unit(value, range) : range.max === range.min ? 0 : clamp01((Math.log(value) - Math.log(range.min)) / (Math.log(range.max) - Math.log(range.min))); }
function logRange(value: number, range: Range): number { return range.min <= 0 ? fromUnit(value, range) : Math.exp(Math.log(range.min) + clamp01(value) * (Math.log(range.max) - Math.log(range.min))); }
function clamp01(value: number): number { return Math.max(0, Math.min(1, value)); }
function shuffle(values: number[], random: () => number): void { for (let index = values.length - 1; index > 0; index -= 1) { const other = Math.floor(random() * (index + 1)); [values[index], values[other]] = [values[other]!, values[index]!]; } }
function mulberry32(seed: number): () => number { let state = seed >>> 0; return () => { state += 0x6d2b79f5; let value = state; value = Math.imul(value ^ value >>> 15, value | 1); value ^= value + Math.imul(value ^ value >>> 7, value | 61); return ((value ^ value >>> 14) >>> 0) / 4_294_967_296; }; }
