import fs from "node:fs";
import path from "node:path";
import { spawn } from "node:child_process";
import { createInterface } from "node:readline";

interface SuiteResult {
  id: string;
  startTime: number;
  endTime: number;
  summary: {
    returnPct: number;
    maxInitialBalanceDrawdownPct: number;
    maxDrawdownPct: number;
    tradeCount: number;
    liquidatedPositionCount?: number;
  };
}

interface SuiteReport {
  oracle: {
    intervalMs: number;
    valueHorizonMs: number;
    returnCorrelation?: number;
    returnNoiseSeed?: number;
    distributionConfidenceGate?: "disabled";
    returnCorrelationBasis?: "rolling-value-horizon-log-return";
    confidenceLeverageFloorScaling?: "quadratic-static-confidence";
    expansionConfirmationMass?: number;
    expansionConfirmationBasis?: "distribution-confidence-mass";
    expansionDeltaCapFraction?: number;
  };
  results: SuiteResult[];
}

interface BenchmarkTask {
  horizonMinutes: number;
  correlation: number;
  seed: number;
  reportFile: string;
  logFile: string;
}

const DAY_MS = 86_400_000;
const expectedWindows = 28;
const horizons = listArgument("horizons", [15, 60]);
const correlations = listArgument("correlations", [0.95, 0.8, 0.5, 0.2, 0.05]);
const samples = integerArgument("samples", 5);
const concurrency = integerArgument("concurrency", 4);
const backend = argument("oracle-backend") ?? "cuda";
const expansionConfirmationMass = numberArgument("expansion-confirmation-mass", 1);
const expansionDeltaCapFraction = numberArgument("expansion-delta-cap-fraction", 0.75);
const outputDirectory = path.resolve(
  argument("output-directory") ?? "data/benchmarks/hindsight-oracle-1m-return-noise-v7-rolling-correlation",
);
const summaryFile = path.join(outputDirectory, "summary.json");

async function main(): Promise<void> {
  validateArguments();
  fs.mkdirSync(outputDirectory, { recursive: true });
  const tasks = benchmarkTasks();
  const pending = tasks.filter((task) => !completeReport(task.reportFile, task));
  console.log(
    `NOISE-BENCHMARK tasks=${tasks.length} complete=${tasks.length - pending.length} `
    + `pending=${pending.length} concurrency=${concurrency}`,
  );
  writeSummary(tasks);

  let next = 0;
  let failure: unknown;
  await Promise.all(Array.from({ length: Math.min(concurrency, pending.length) }, async () => {
    while (!failure) {
      const task = pending[next++];
      if (!task) return;
      try {
        await runTask(task);
        writeSummary(tasks);
      } catch (error) {
        failure ??= error;
      }
    }
  }));
  if (failure) throw failure;
  writeSummary(tasks);
  console.log(`NOISE-BENCHMARK-COMPLETE ${summaryFile}`);
}

function benchmarkTasks(): BenchmarkTask[] {
  const tasks: BenchmarkTask[] = [];
  for (const correlation of correlations) {
    const sampleCount = correlation === 1 ? 1 : samples;
    for (let seed = 1; seed <= sampleCount; seed += 1) {
      for (const horizonMinutes of horizons) {
        const stem = `h${padded(horizonMinutes)}-rho${correlationTag(correlation)}-seed${padded(seed)}`;
        tasks.push({
          horizonMinutes,
          correlation,
          seed,
          reportFile: path.join(outputDirectory, `${stem}.json`),
          logFile: path.join(outputDirectory, `${stem}.log`),
        });
      }
    }
  }
  return tasks;
}

async function runTask(task: BenchmarkTask): Promise<void> {
  console.log(
    `SAMPLE-START horizon=${task.horizonMinutes} correlation=${task.correlation} seed=${task.seed}`,
  );
  const child = spawn(process.execPath, [
    "--max-old-space-size=8192",
    "--expose-gc",
    "--conditions=development",
    "--import",
    "tsx",
    "scripts/backtest-hindsight-oracle-suite.ts",
    "--interval=1m",
    `--value-horizon-minutes=${task.horizonMinutes}`,
    "--exclude-prefix=fit-,latest-",
    "--quiet-replay",
    `--oracle-backend=${backend}`,
    "--oracle-workers=1",
    `--oracle-return-correlation=${task.correlation}`,
    `--oracle-noise-seed=${task.seed}`,
    `--oracle-expansion-confirmation-mass=${expansionConfirmationMass}`,
    `--oracle-expansion-delta-cap-fraction=${expansionDeltaCapFraction}`,
    `--output=${task.reportFile}`,
  ], {
    cwd: process.cwd(),
    stdio: ["ignore", "pipe", "pipe"],
  });
  const log = fs.createWriteStream(task.logFile, { flags: "a" });
  child.stdout.pipe(log, { end: false });
  child.stderr.pipe(log, { end: false });
  const lines = createInterface({ input: child.stdout });
  lines.on("line", (line) => {
    if (!line.startsWith("RESULT ")) return;
    const result = JSON.parse(line.slice("RESULT ".length)) as { id: string };
    console.log(
      `SAMPLE-PROGRESS horizon=${task.horizonMinutes} correlation=${task.correlation} `
      + `seed=${task.seed} window=${result.id}`,
    );
  });
  const exitCode = await new Promise<number | null>((resolve, reject) => {
    child.once("error", reject);
    child.once("exit", resolve);
  });
  lines.close();
  log.end();
  if (exitCode !== 0) {
    throw new Error(
      `Noise sample failed with exit code ${exitCode}: horizon=${task.horizonMinutes} `
      + `correlation=${task.correlation} seed=${task.seed}; see ${task.logFile}`,
    );
  }
  if (!completeReport(task.reportFile, task)) {
    throw new Error(`Noise sample did not produce ${expectedWindows} windows: ${task.reportFile}`);
  }
  console.log(
    `SAMPLE-RESULT horizon=${task.horizonMinutes} correlation=${task.correlation} seed=${task.seed}`,
  );
}

function writeSummary(tasks: readonly BenchmarkTask[]): void {
  const samplesByLevel = new Map<string, SuiteReport[]>();
  for (const task of tasks) {
    if (!completeReport(task.reportFile, task)) continue;
    const key = levelKey(task.horizonMinutes, task.correlation);
    const reports = samplesByLevel.get(key) ?? [];
    reports.push(readReport(task.reportFile));
    samplesByLevel.set(key, reports);
  }

  const levels = [...samplesByLevel.entries()]
    .map(([key, reports]) => summarizeLevel(key, reports))
    .sort((left, right) => left.horizonMinutes - right.horizonMinutes
      || right.returnCorrelation - left.returnCorrelation);
  const output = {
    generatedAt: new Date().toISOString(),
    method: {
      candleIntervalMs: 60_000,
      noise: "orthogonalized mixture calibrated over rolling value-horizon log returns",
      noiseInterpretation: "rho is the realized Pearson correlation between true and noisy rolling returns at the oracle value horizon",
      staticConfidenceScale: "equal to rho for noisy samples; 1 for the clean baseline",
      distributionConfidenceGate: "disabled",
      leverageRule: "maxLeverage * rho * (floor*rho + (1-floor*rho) * distributionConfidence)",
      expansionConfirmationMass,
      expansionConfirmationBasis: "distribution-confidence-mass",
      expansionDeltaCapRule: "maxLeverage * expansionDeltaCapFraction * rho^2",
      expansionDeltaCapFraction,
      reversalRule: "close to zero before entering the opposite side",
      executionCandles: "unmodified real OHLC",
      expectedWindows,
      requestedSamplesPerNoisyLevel: samples,
    },
    horizonsMinutes: horizons,
    returnCorrelations: correlations,
    levels,
  };
  const temporary = `${summaryFile}.${process.pid}.tmp`;
  fs.writeFileSync(temporary, `${JSON.stringify(output, null, 2)}\n`, "utf8");
  fs.renameSync(temporary, summaryFile);
}

function summarizeLevel(key: string, reports: readonly SuiteReport[]) {
  const [horizonMinutes, returnCorrelation] = key.split(":").map(Number) as [number, number];
  const cases = reports.flatMap((report) => report.results.map((result) => ({
    dailyReturnPct: compoundedDailyReturnPct(result),
    dailyLogGrowth: dailyLogGrowth(result),
    maxInitialBalanceDrawdownPct: result.summary.maxInitialBalanceDrawdownPct,
    maxDrawdownPct: result.summary.maxDrawdownPct,
    trades: result.summary.tradeCount,
    liquidated: (result.summary.liquidatedPositionCount ?? 0) > 0,
  })));
  const dailyReturns = cases.map((row) => row.dailyReturnPct).sort((a, b) => a - b);
  const initialDrawdowns = cases
    .map((row) => row.maxInitialBalanceDrawdownPct)
    .sort((a, b) => a - b);
  const drawdowns = cases.map((row) => row.maxDrawdownPct).sort((a, b) => a - b);
  const perSample = reports.map((report) => ({
    profitableWindows: report.results.filter((result) => result.summary.returnPct > 0).length,
    geometricMeanDailyReturnPct: 100 * Math.expm1(mean(report.results.map(dailyLogGrowth))),
    worstDailyReturnPct: Math.min(...report.results.map(compoundedDailyReturnPct)),
  }));
  return {
    horizonMinutes,
    returnCorrelation,
    completedSamples: reports.length,
    completedCases: cases.length,
    allWindowsProfitableSamples: perSample.filter((sample) =>
      sample.profitableWindows === expectedWindows).length,
    profitableCasePct: 100 * cases.filter((row) => row.dailyReturnPct > 0).length / cases.length,
    geometricMeanDailyReturnPct: 100 * Math.expm1(mean(cases.map((row) => row.dailyLogGrowth))),
    medianDailyReturnPct: percentile(dailyReturns, 0.5),
    p10DailyReturnPct: percentile(dailyReturns, 0.1),
    worstDailyReturnPct: dailyReturns[0],
    meanSampleWorstDailyReturnPct: mean(perSample.map((sample) => sample.worstDailyReturnPct)),
    meanMaxInitialBalanceDrawdownPct: mean(initialDrawdowns),
    p90MaxInitialBalanceDrawdownPct: percentile(initialDrawdowns, 0.9),
    maximumInitialBalanceDrawdownPct: initialDrawdowns.at(-1),
    meanMaxDrawdownPct: mean(drawdowns),
    p90MaxDrawdownPct: percentile(drawdowns, 0.9),
    maximumDrawdownPct: drawdowns.at(-1),
    liquidationCasePct: 100 * cases.filter((row) => row.liquidated).length / cases.length,
    meanTradesPerWindow: mean(cases.map((row) => row.trades)),
  };
}

function completeReport(file: string, task: BenchmarkTask): boolean {
  if (!fs.existsSync(file)) return false;
  try {
    const report = readReport(file);
    assertReportContract(
      report,
      task.horizonMinutes,
      task.correlation,
      task.seed,
      file,
    );
    return report.results.length === expectedWindows;
  } catch {
    return false;
  }
}

function assertReportContract(
  report: SuiteReport,
  horizonMinutes: number,
  correlation: number,
  seed: number,
  file: string,
): void {
  if (
    report.oracle.intervalMs !== 60_000
    || report.oracle.valueHorizonMs !== horizonMinutes * 60_000
    || (report.oracle.returnCorrelation ?? 1) !== correlation
    || (report.oracle.returnNoiseSeed ?? 0) !== seed
    || correlation < 1 && (
      report.oracle.distributionConfidenceGate !== "disabled"
      || report.oracle.returnCorrelationBasis !== "rolling-value-horizon-log-return"
      || report.oracle.confidenceLeverageFloorScaling !== "quadratic-static-confidence"
    )
    || report.oracle.expansionConfirmationMass !== expansionConfirmationMass
    || report.oracle.expansionConfirmationBasis !== "distribution-confidence-mass"
    || report.oracle.expansionDeltaCapFraction !== expansionDeltaCapFraction
    || !Array.isArray(report.results)
  ) throw new Error(`Incompatible noise benchmark report: ${file}`);
}

function readReport(file: string): SuiteReport {
  return JSON.parse(fs.readFileSync(file, "utf8")) as SuiteReport;
}

function compoundedDailyReturnPct(result: SuiteResult): number {
  return 100 * Math.expm1(dailyLogGrowth(result));
}

function dailyLogGrowth(result: SuiteResult): number {
  const factor = Math.max(Number.MIN_VALUE, 1 + result.summary.returnPct / 100);
  const days = (result.endTime - result.startTime) / DAY_MS;
  return Math.log(factor) / days;
}

function percentile(sorted: readonly number[], fraction: number): number | undefined {
  if (sorted.length === 0) return undefined;
  const position = (sorted.length - 1) * fraction;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  const weight = position - lower;
  return sorted[lower]! * (1 - weight) + sorted[upper]! * weight;
}

function mean(values: readonly number[]): number {
  return values.length === 0
    ? Number.NaN
    : values.reduce((sum, value) => sum + value, 0) / values.length;
}

function validateArguments(): void {
  if (samples < 1) throw new Error("--samples must be positive.");
  if (concurrency < 1) throw new Error("--concurrency must be positive.");
  if (!['cpu', 'cuda', 'auto'].includes(backend)) {
    throw new Error("--oracle-backend must be cpu, cuda, or auto.");
  }
  if (horizons.some((value) => !Number.isSafeInteger(value) || value < 1)) {
    throw new Error("--horizons must contain positive integers.");
  }
  if (correlations.some((value) => !(value >= 0 && value <= 1))) {
    throw new Error("--correlations must be in [0, 1].");
  }
  if (expansionConfirmationMass < 0) {
    throw new Error("--expansion-confirmation-mass must be non-negative.");
  }
  if (!(expansionDeltaCapFraction >= 0)) {
    throw new Error("--expansion-delta-cap-fraction must be non-negative.");
  }
}

function levelKey(horizonMinutes: number, correlation: number): string {
  return `${horizonMinutes}:${correlation}`;
}

function correlationTag(value: number): string {
  return value.toFixed(3).replace(".", "");
}

function padded(value: number): string {
  return value.toString().padStart(3, "0");
}

function argument(name: string): string | undefined {
  const prefix = `--${name}=`;
  return process.argv.find((value) => value.startsWith(prefix))?.slice(prefix.length);
}

function integerArgument(name: string, fallback: number): number {
  const raw = argument(name);
  if (raw === undefined) return fallback;
  if (!/^\d+$/.test(raw)) throw new Error(`--${name} must be an integer.`);
  const value = Number(raw);
  if (!Number.isSafeInteger(value)) throw new Error(`--${name} is out of range.`);
  return value;
}

function numberArgument(name: string, fallback: number): number {
  const raw = argument(name);
  if (raw === undefined) return fallback;
  const value = Number(raw);
  if (!Number.isFinite(value)) throw new Error(`--${name} must be finite.`);
  return value;
}

function listArgument(name: string, fallback: readonly number[]): number[] {
  const raw = argument(name);
  if (raw === undefined) return [...fallback];
  const values = raw.split(",").map(Number);
  if (values.length === 0 || values.some((value) => !Number.isFinite(value))) {
    throw new Error(`--${name} must be a comma-separated number list.`);
  }
  return values;
}

void main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : String(error));
  process.exitCode = 1;
});
