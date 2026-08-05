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
    maxDrawdownPct: number;
    tradeCount: number;
  };
}

interface SuiteReport {
  oracle: { valueHorizonMs: number };
  results: SuiteResult[];
}

interface HorizonSummary {
  horizonMinutes: number;
  completedWindows: number;
  passingWindows: number;
  thresholdDailyReturnPct: number;
  minimumDailyReturnPct: number;
  medianDailyReturnPct: number;
  maximumDailyReturnPct: number;
  failingWindows: Array<{
    id: string;
    dailyReturnPct: number;
    totalReturnPct: number;
    maxDrawdownPct: number;
  }>;
}

const minHorizon = integerArgument("min-horizon-minutes", 2);
const maxHorizon = integerArgument("max-horizon-minutes", 60);
const concurrency = integerArgument("concurrency", 2);
const oracleProducers = integerArgument("oracle-producers", 2);
const thresholdDailyReturnPct = finiteArgument("threshold-daily-return-pct", 100);
const backend = argument("oracle-backend") ?? "cuda";
const outputDirectory = path.resolve(
  argument("output-directory")
    ?? "data/benchmarks/hindsight-oracle-horizon-sweep-v1",
);
const screenDirectory = path.resolve(
  argument("screen-directory")
    ?? "data/benchmarks/hindsight-oracle-horizon-search-v1",
);
const summaryFile = path.join(outputDirectory, "horizon-counts.json");

async function main(): Promise<void> {
  if (minHorizon < 1 || maxHorizon < minHorizon) {
    throw new Error("The horizon range is invalid.");
  }
  if (concurrency < 1) throw new Error("--concurrency must be positive.");
  if (oracleProducers < 1) throw new Error("--oracle-producers must be positive.");
  if (!Number.isFinite(thresholdDailyReturnPct)) {
    throw new Error("--threshold-daily-return-pct must be finite.");
  }
  if (!["cpu", "cuda", "auto"].includes(backend)) {
    throw new Error("--oracle-backend must be cpu, cuda, or auto.");
  }

  fs.mkdirSync(outputDirectory, { recursive: true });
  const horizons = Array.from(
    { length: maxHorizon - minHorizon + 1 },
    (_, index) => minHorizon + index,
  );
  const pending = [...horizons];
  const ready: number[] = [];
  const activeReplays = new Set<Promise<void>>();
  let replayFailure: unknown;

  const pumpReplays = (): void => {
    while (ready.length > 0 && activeReplays.size < concurrency && !replayFailure) {
      const horizonMinutes = ready.shift()!;
      let replay!: Promise<void>;
      replay = runHorizon(horizonMinutes)
        .then(() => writeSummary())
        .catch((error: unknown) => {
          replayFailure ??= error;
        })
        .finally(() => {
          activeReplays.delete(replay);
          pumpReplays();
        });
      activeReplays.add(replay);
    }
  };

  await Promise.all(Array.from(
    { length: Math.min(oracleProducers, pending.length) },
    async () => {
      while (pending.length > 0) {
        if (replayFailure) throw replayFailure;
        const horizonMinutes = pending.shift();
        if (horizonMinutes === undefined) return;
        await precomputeHorizon(horizonMinutes);
        ready.push(horizonMinutes);
        pumpReplays();
      }
    },
  ));

  while ((ready.length > 0 || activeReplays.size > 0) && !replayFailure) {
    pumpReplays();
    await delay(250);
  }
  if (replayFailure) throw replayFailure;

  writeSummary();
  console.log(`SWEEP-COMPLETE ${summaryFile}`);
}

async function precomputeHorizon(horizonMinutes: number): Promise<void> {
  const reportFile = horizonReportFile(horizonMinutes);
  seedFromScreen(horizonMinutes, reportFile);
  const logFile = path.join(
    outputDirectory,
    `horizon-${padded(horizonMinutes)}m-oracle.log`,
  );
  console.log(`ORACLE-START minutes=${horizonMinutes}`);
  const exitCode = await runSuiteProcess([
    "--oracle-only",
    "--exclude-prefix=fit-,latest-",
    `--value-horizon-minutes=${horizonMinutes}`,
    `--output=${reportFile}`,
    `--oracle-backend=${backend}`,
    "--oracle-workers=1",
  ], logFile);
  if (exitCode !== 0) {
    throw new Error(
      `Oracle horizon ${horizonMinutes} failed with exit code ${exitCode}; see ${logFile}`,
    );
  }
  console.log(`ORACLE-RESULT minutes=${horizonMinutes}`);
}

async function runHorizon(horizonMinutes: number): Promise<void> {
  const reportFile = horizonReportFile(horizonMinutes);
  seedFromScreen(horizonMinutes, reportFile);
  const logFile = path.join(outputDirectory, `horizon-${padded(horizonMinutes)}m.log`);
  console.log(`HORIZON-START minutes=${horizonMinutes}`);
  const exitCode = await runSuiteProcess([
    "--exclude-prefix=fit-,latest-",
    `--value-horizon-minutes=${horizonMinutes}`,
    `--output=${reportFile}`,
    `--oracle-backend=${backend}`,
    "--oracle-workers=1",
  ], logFile, (line) => {
    if (line.startsWith("RESULT ")) {
      const result = JSON.parse(line.slice("RESULT ".length)) as { id: string };
      console.log(`HORIZON-PROGRESS minutes=${horizonMinutes} window=${result.id}`);
    }
  });
  if (exitCode !== 0) {
    throw new Error(
      `Horizon ${horizonMinutes} failed with exit code ${exitCode}; see ${logFile}`,
    );
  }
  const summary = summarizeHorizon(readReport(reportFile), thresholdDailyReturnPct);
  console.log(
    `HORIZON-RESULT minutes=${horizonMinutes} passing=${summary.passingWindows}`
    + `/${summary.completedWindows} minimumDailyPct=${summary.minimumDailyReturnPct}`,
  );
}

async function runSuiteProcess(
  arguments_: readonly string[],
  logFile: string,
  onLine?: (line: string) => void,
): Promise<number | null> {
  const child = spawn(process.execPath, [
    "--max-old-space-size=8192",
    "--expose-gc",
    "--conditions=development",
    "--import",
    "tsx",
    "scripts/backtest-hindsight-oracle-suite.ts",
    ...arguments_,
  ], {
    cwd: process.cwd(),
    stdio: ["ignore", "pipe", "pipe"],
  });
  const log = fs.createWriteStream(logFile, { flags: "a" });
  child.stdout.pipe(log, { end: false });
  child.stderr.pipe(log, { end: false });
  const lines = createInterface({ input: child.stdout });
  if (onLine) lines.on("line", onLine);
  const exitCode = await new Promise<number | null>((resolve, reject) => {
    child.once("error", reject);
    child.once("exit", resolve);
  });
  lines.close();
  log.end();
  return exitCode;
}

function seedFromScreen(horizonMinutes: number, reportFile: string): void {
  if (fs.existsSync(reportFile)) return;
  const screenFile = path.join(
    screenDirectory,
    `horizon-${padded(horizonMinutes)}m-screen.json`,
  );
  if (!fs.existsSync(screenFile)) return;
  const report = readReport(screenFile);
  if (
    report.oracle.valueHorizonMs !== horizonMinutes * 60_000
    || report.results.some((result) => result.id !== "regime-flat-2026-04")
  ) return;
  fs.copyFileSync(screenFile, reportFile);
}

function writeSummary(): void {
  const horizons = fs.readdirSync(outputDirectory)
    .map((file) => /^horizon-(\d{3})m\.json$/.exec(file)?.[1])
    .filter((value): value is string => value !== undefined)
    .map(Number)
    .sort((a, b) => a - b)
    .map((horizonMinutes) => summarizeHorizon(
      readReport(horizonReportFile(horizonMinutes)),
      thresholdDailyReturnPct,
    ));
  const output = {
    generatedAt: new Date().toISOString(),
    thresholdDailyReturnPct,
    excludedWindowPrefixes: ["fit-", "latest-"],
    expectedWindows: 28,
    horizons,
  };
  const temporary = `${summaryFile}.${process.pid}.tmp`;
  fs.writeFileSync(temporary, `${JSON.stringify(output, null, 2)}\n`, "utf8");
  fs.renameSync(temporary, summaryFile);
}

function summarizeHorizon(
  report: SuiteReport,
  threshold: number,
): HorizonSummary {
  const rows = report.results.map((result) => ({
    result,
    dailyReturnPct: compoundedDailyReturnPct(result),
  }));
  const dailyReturns = rows.map((row) => row.dailyReturnPct).sort((a, b) => a - b);
  return {
    horizonMinutes: report.oracle.valueHorizonMs / 60_000,
    completedWindows: rows.length,
    passingWindows: rows.filter((row) => row.dailyReturnPct >= threshold).length,
    thresholdDailyReturnPct: threshold,
    minimumDailyReturnPct: dailyReturns[0] ?? Number.NaN,
    medianDailyReturnPct: median(dailyReturns),
    maximumDailyReturnPct: dailyReturns.at(-1) ?? Number.NaN,
    failingWindows: rows
      .filter((row) => row.dailyReturnPct < threshold)
      .sort((a, b) => a.dailyReturnPct - b.dailyReturnPct)
      .map(({ result, dailyReturnPct }) => ({
        id: result.id,
        dailyReturnPct,
        totalReturnPct: result.summary.returnPct,
        maxDrawdownPct: result.summary.maxDrawdownPct,
      })),
  };
}

function compoundedDailyReturnPct(result: SuiteResult): number {
  const equityFactor = 1 + result.summary.returnPct / 100;
  if (!(equityFactor > 0)) return -100;
  const days = (result.endTime - result.startTime) / 86_400_000;
  return 100 * (equityFactor ** (1 / days) - 1);
}

function median(values: readonly number[]): number {
  if (values.length === 0) return Number.NaN;
  const middle = Math.floor(values.length / 2);
  return values.length % 2 === 1
    ? values[middle]!
    : (values[middle - 1]! + values[middle]!) / 2;
}

function readReport(file: string): SuiteReport {
  return JSON.parse(fs.readFileSync(file, "utf8")) as SuiteReport;
}

function horizonReportFile(horizonMinutes: number): string {
  return path.join(outputDirectory, `horizon-${padded(horizonMinutes)}m.json`);
}

function padded(value: number): string {
  return value.toString().padStart(3, "0");
}

function delay(milliseconds: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
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

function finiteArgument(name: string, fallback: number): number {
  const raw = argument(name);
  if (raw === undefined) return fallback;
  const value = Number(raw);
  if (!Number.isFinite(value)) throw new Error(`--${name} must be finite.`);
  return value;
}

void main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : String(error));
  process.exitCode = 1;
});
