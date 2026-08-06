import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { Worker, isMainThread, parentPort, workerData } from "node:worker_threads";
import { readReferencedPayload } from "@trading/storage";

const CACHE_ROOT = path.resolve("data/training/immutable/refs/oracle/1m");
const OUTPUT_FILE = path.resolve(
  "data/benchmarks/hindsight-oracle-noise-kl-v1/summary.json",
);
const NOISE_KIND = "orthogonalized-rolling-horizon-return-correlation-v4";
const EPSILON = 1e-12;
const HISTOGRAM_MIN_EXPONENT = -12;
const HISTOGRAM_MAX_EXPONENT = 2;
const HISTOGRAM_BINS_PER_DECADE = 40;
const HISTOGRAM_BIN_COUNT =
  (HISTOGRAM_MAX_EXPONENT - HISTOGRAM_MIN_EXPONENT) * HISTOGRAM_BINS_PER_DECADE;

interface OracleContract {
  numericBackend: string;
  intervalMs: number;
  decisionIntervalMs: number;
  options: { valueHorizonSteps: number } & Record<string, unknown>;
  usableGrid: number[];
  returnNoise?: {
    kind: string;
    correlation: number;
    seed: number;
    rollingHorizonSteps: number;
  };
}

interface NamespaceDefinition {
  directory: string;
  name: string;
  contract: OracleContract;
  baseContractKey: string;
  days: string[];
}

interface WorkerTask {
  cleanDirectory: string;
  noisyDirectory: string;
  horizonMinutes: number;
  rho: number;
  seed: number;
  days: string[];
}

interface Histogram {
  zeroCount: number;
  bins: number[];
}

interface ScalarAggregate {
  count: number;
  sum: number;
  sumSquares: number;
  max: number;
  histogram: Histogram;
}

interface DivergenceAggregate {
  rows: number;
  cells: number;
  alignedDays: number;
  skippedRows: number;
  forwardInfiniteRows: number;
  reverseInfiniteRows: number;
  modeAgreementRows: number;
  exactForwardFinite: ScalarAggregate;
  exactReverseFinite: ScalarAggregate;
  clippedForward: ScalarAggregate;
  clippedReverse: ScalarAggregate;
  jensenShannon: ScalarAggregate;
  totalVariation: ScalarAggregate;
}

interface WorkerResult {
  horizonMinutes: number;
  rho: number;
  seed: number;
  cleanNamespace: string;
  noisyNamespace: string;
  days: string[];
  aggregate: DivergenceAggregate;
}

interface OracleShard {
  start: number;
  step: number;
  rows: number;
  columns: number;
  probabilities: Float32Array;
}

void (isMainThread
  ? main()
  : analyzeTask(workerData as WorkerTask).then((result) => parentPort!.postMessage(result))
).catch((error: unknown) => {
  console.error(error);
  process.exitCode = 1;
});

async function main(): Promise<void> {
  const cacheRoot = path.resolve(argument("cache-root") ?? CACHE_ROOT);
  const outputFile = path.resolve(argument("output") ?? OUTPUT_FILE);
  const namespaces = discoverNamespaces(cacheRoot);
  const cleanByHorizon = new Map<number, NamespaceDefinition>();
  for (const namespace of namespaces) {
    if (namespace.contract.returnNoise) continue;
    const horizon = namespace.contract.options.valueHorizonSteps;
    if (cleanByHorizon.has(horizon)) {
      throw new Error(`Multiple clean oracle namespaces found for ${horizon}m.`);
    }
    cleanByHorizon.set(horizon, namespace);
  }

  const tasks: WorkerTask[] = [];
  for (const noisy of namespaces) {
    const noise = noisy.contract.returnNoise;
    if (!noise || noise.kind !== NOISE_KIND) continue;
    const horizon = noisy.contract.options.valueHorizonSteps;
    const clean = cleanByHorizon.get(horizon);
    if (!clean) throw new Error(`No clean oracle namespace found for ${horizon}m.`);
    if (noise.rollingHorizonSteps !== horizon) {
      throw new Error(`${noisy.name} uses a mismatched rolling noise horizon.`);
    }
    if (noisy.baseContractKey !== clean.baseContractKey) {
      throw new Error(`${noisy.name} does not match the ${horizon}m clean contract.`);
    }
    const cleanDays = new Set(clean.days);
    const days = noisy.days.filter((day) => cleanDays.has(day));
    if (days.length === 0) throw new Error(`${noisy.name} has no clean aligned days.`);
    tasks.push({
      cleanDirectory: clean.directory,
      noisyDirectory: noisy.directory,
      horizonMinutes: horizon,
      rho: noise.correlation,
      seed: noise.seed,
      days,
    });
  }
  tasks.sort((left, right) =>
    left.horizonMinutes - right.horizonMinutes
    || right.rho - left.rho
    || left.seed - right.seed,
  );
  if (tasks.length === 0) throw new Error(`No ${NOISE_KIND} caches found in ${cacheRoot}.`);

  const requestedWorkers = integerArgument("workers", Math.max(1, os.availableParallelism() - 1));
  const concurrency = Math.min(tasks.length, requestedWorkers);
  console.log(
    `KL tasks=${tasks.length} workers=${concurrency} epsilon=${EPSILON} `
    + `noise=${NOISE_KIND}`,
  );
  const results = await runWorkerPool(tasks, concurrency);
  results.sort((left, right) =>
    left.horizonMinutes - right.horizonMinutes
    || right.rho - left.rho
    || left.seed - right.seed,
  );

  const pooled = new Map<string, {
    horizonMinutes: number;
    rho: number;
    seeds: number[];
    aggregate: DivergenceAggregate;
  }>();
  for (const result of results) {
    const key = `${result.horizonMinutes}:${result.rho}`;
    const entry = pooled.get(key) ?? {
      horizonMinutes: result.horizonMinutes,
      rho: result.rho,
      seeds: [],
      aggregate: emptyDivergenceAggregate(),
    };
    entry.seeds.push(result.seed);
    mergeDivergence(entry.aggregate, result.aggregate);
    pooled.set(key, entry);
  }

  const report = {
    version: 1,
    generatedAt: new Date().toISOString(),
    source: {
      cacheRoot,
      noiseKind: NOISE_KIND,
      cleanReference: "oracle distribution generated from unperturbed future candles",
      noisyComparison: "oracle distribution generated from perturbed future candles",
      alignment: "same horizon, UTC day, one-minute timestamp, and exposure grid",
      overlapHandling: "each stored UTC day and timestamp is counted once per noise seed",
      distribution: "unconditioned initial-action exposure distribution stored by the oracle cache",
    },
    units: "nats",
    epsilon: {
      value: EPSILON,
      definition:
        "For D(clean||noisy), noisy probabilities below epsilon are floored and renormalized; "
        + "the reverse metric floors and renormalizes the clean distribution.",
    },
    quantiles: {
      method: "logarithmic histogram approximation",
      binsPerDecade: HISTOGRAM_BINS_PER_DECADE,
      range: [10 ** HISTOGRAM_MIN_EXPONENT, 10 ** HISTOGRAM_MAX_EXPONENT],
    },
    pooled: Array.from(pooled.values())
      .sort((left, right) => left.horizonMinutes - right.horizonMinutes || right.rho - left.rho)
      .map((entry) => ({
        horizonMinutes: entry.horizonMinutes,
        rho: entry.rho,
        seeds: entry.seeds.sort((left, right) => left - right),
        ...summarizeDivergence(entry.aggregate),
      })),
    samples: results.map((result) => ({
      horizonMinutes: result.horizonMinutes,
      rho: result.rho,
      seed: result.seed,
      cleanNamespace: result.cleanNamespace,
      noisyNamespace: result.noisyNamespace,
      firstDay: result.days[0],
      lastDay: result.days.at(-1),
      ...summarizeDivergence(result.aggregate),
    })),
  };
  fs.mkdirSync(path.dirname(outputFile), { recursive: true });
  fs.writeFileSync(outputFile, `${JSON.stringify(report, null, 2)}\n`);
  console.table(report.pooled.map((entry) => ({
    horizon: `${entry.horizonMinutes}m`,
    rho: entry.rho,
    rows: entry.rows,
    forwardKl: entry.forwardKlEpsilon.mean,
    forwardKlP90: entry.forwardKlEpsilon.p90,
    jsd: entry.jensenShannon.mean,
    tv: entry.totalVariation.mean,
    modeAgreement: entry.modeAgreementFraction,
    forwardInfinite: entry.forwardInfiniteFraction,
  })));
  console.log(`REPORT ${outputFile}`);
}

function discoverNamespaces(cacheRoot: string): NamespaceDefinition[] {
  return fs.readdirSync(cacheRoot, { withFileTypes: true })
    .filter((entry) => entry.isDirectory() && entry.name.startsWith("hindsight-bot-"))
    .map((entry) => {
      const directory = path.join(cacheRoot, entry.name);
      const days = fs.readdirSync(directory)
        .filter((file) => /^\d{4}-\d{2}-\d{2}\.json$/.test(file))
        .map((file) => file.slice(0, -5))
        .sort();
      if (days.length === 0) return null;
      const reference = JSON.parse(
        fs.readFileSync(path.join(directory, `${days[0]}.json`), "utf8"),
      ) as { metadata?: { contract?: OracleContract } };
      const contract = reference.metadata?.contract;
      if (!contract || contract.intervalMs !== 60_000 || contract.decisionIntervalMs !== 60_000) {
        return null;
      }
      const { returnNoise: _returnNoise, ...baseContract } = contract;
      return {
        directory,
        name: entry.name,
        contract,
        baseContractKey: JSON.stringify(baseContract),
        days,
      } satisfies NamespaceDefinition;
    })
    .filter((value): value is NamespaceDefinition => value !== null);
}

async function runWorkerPool(tasks: WorkerTask[], concurrency: number): Promise<WorkerResult[]> {
  const results: WorkerResult[] = [];
  let nextTask = 0;
  let completed = 0;
  await Promise.all(Array.from({ length: concurrency }, async () => {
    while (nextTask < tasks.length) {
      const taskIndex = nextTask;
      nextTask += 1;
      const task = tasks[taskIndex]!;
      const result = await runWorker(task);
      results.push(result);
      completed += 1;
      console.log(
        `KL ${completed}/${tasks.length} h=${task.horizonMinutes}m rho=${task.rho} `
        + `seed=${task.seed} rows=${result.aggregate.rows}`,
      );
    }
  }));
  return results;
}

function runWorker(task: WorkerTask): Promise<WorkerResult> {
  return new Promise((resolve, reject) => {
    const worker = new Worker(__filename, { workerData: task });
    worker.once("message", (result: WorkerResult) => resolve(result));
    worker.once("error", reject);
    worker.once("exit", (code) => {
      if (code !== 0) reject(new Error(`KL worker exited with code ${code}.`));
    });
  });
}

async function analyzeTask(task: WorkerTask): Promise<WorkerResult> {
  const aggregate = emptyDivergenceAggregate();
  for (const day of task.days) {
    const clean = await loadShard(path.join(task.cleanDirectory, `${day}.json`));
    const noisy = await loadShard(path.join(task.noisyDirectory, `${day}.json`));
    validateAlignedShards(clean, noisy, day);
    analyzeShardPair(clean, noisy, aggregate);
    aggregate.alignedDays += 1;
  }
  return {
    horizonMinutes: task.horizonMinutes,
    rho: task.rho,
    seed: task.seed,
    cleanNamespace: path.basename(task.cleanDirectory),
    noisyNamespace: path.basename(task.noisyDirectory),
    days: task.days,
    aggregate,
  };
}

async function loadShard(referenceFile: string): Promise<OracleShard> {
  const { reference, payload } = await readReferencedPayload(referenceFile, false);
  const rows = reference.layout.rows;
  const columns = Number(reference.layout.columns);
  if (
    reference.layout.encoding !== "raw-row-major"
    || reference.layout.dtype !== "float32-le"
    || rows !== reference.sequence.count
    || !Number.isInteger(columns)
    || columns <= 0
  ) throw new Error(`Unsupported oracle shard layout: ${referenceFile}`);
  const probabilities = payload.byteOffset % Float32Array.BYTES_PER_ELEMENT === 0
    ? new Float32Array(
        payload.buffer,
        payload.byteOffset,
        payload.byteLength / Float32Array.BYTES_PER_ELEMENT,
      )
    : new Float32Array(payload.buffer.slice(
        payload.byteOffset,
        payload.byteOffset + payload.byteLength,
      ));
  if (probabilities.length !== rows * columns) {
    throw new Error(`Oracle shard has an unexpected payload length: ${referenceFile}`);
  }
  return {
    start: reference.sequence.start,
    step: reference.sequence.step,
    rows,
    columns,
    probabilities,
  };
}

function validateAlignedShards(clean: OracleShard, noisy: OracleShard, day: string): void {
  if (
    clean.start !== noisy.start
    || clean.step !== noisy.step
    || clean.rows !== noisy.rows
    || clean.columns !== noisy.columns
  ) throw new Error(`Clean and noisy oracle shards are not aligned on ${day}.`);
}

function analyzeShardPair(
  clean: OracleShard,
  noisy: OracleShard,
  aggregate: DivergenceAggregate,
): void {
  const columns = clean.columns;
  for (let row = 0; row < clean.rows; row += 1) {
    const offset = row * columns;
    let cleanSum = 0;
    let noisySum = 0;
    let cleanMode = 0;
    let noisyMode = 0;
    for (let column = 0; column < columns; column += 1) {
      const cleanValue = clean.probabilities[offset + column]!;
      const noisyValue = noisy.probabilities[offset + column]!;
      cleanSum += cleanValue;
      noisySum += noisyValue;
      if (cleanValue > clean.probabilities[offset + cleanMode]!) cleanMode = column;
      if (noisyValue > noisy.probabilities[offset + noisyMode]!) noisyMode = column;
    }
    if (!(cleanSum > 0) || !(noisySum > 0)) {
      aggregate.skippedRows += 1;
      continue;
    }

    let exactForward = 0;
    let exactReverse = 0;
    let clippedForward = 0;
    let clippedReverse = 0;
    let noisyClipSum = 0;
    let cleanClipSum = 0;
    let jensenShannon = 0;
    let totalVariation = 0;
    let forwardInfinite = false;
    let reverseInfinite = false;
    for (let column = 0; column < columns; column += 1) {
      const p = clean.probabilities[offset + column]! / cleanSum;
      const q = noisy.probabilities[offset + column]! / noisySum;
      const pClip = Math.max(p, EPSILON);
      const qClip = Math.max(q, EPSILON);
      cleanClipSum += pClip;
      noisyClipSum += qClip;
      if (p > 0) clippedForward += p * Math.log(p / qClip);
      if (q > 0) clippedReverse += q * Math.log(q / pClip);
      if (p > 0 && q > 0) {
        const logRatio = Math.log(p / q);
        exactForward += p * logRatio;
        exactReverse -= q * logRatio;
      } else {
        if (p > 0) forwardInfinite = true;
        if (q > 0) reverseInfinite = true;
      }
      const midpoint = (p + q) / 2;
      if (p > 0) jensenShannon += 0.5 * p * Math.log(p / midpoint);
      if (q > 0) jensenShannon += 0.5 * q * Math.log(q / midpoint);
      totalVariation += Math.abs(p - q) / 2;
    }
    clippedForward += Math.log(noisyClipSum);
    clippedReverse += Math.log(cleanClipSum);
    exactForward = Math.max(0, exactForward);
    exactReverse = Math.max(0, exactReverse);
    clippedForward = Math.max(0, clippedForward);
    clippedReverse = Math.max(0, clippedReverse);
    jensenShannon = Math.max(0, jensenShannon);

    aggregate.rows += 1;
    aggregate.cells += columns;
    if (cleanMode === noisyMode) aggregate.modeAgreementRows += 1;
    if (forwardInfinite) aggregate.forwardInfiniteRows += 1;
    else addScalar(aggregate.exactForwardFinite, exactForward);
    if (reverseInfinite) aggregate.reverseInfiniteRows += 1;
    else addScalar(aggregate.exactReverseFinite, exactReverse);
    addScalar(aggregate.clippedForward, clippedForward);
    addScalar(aggregate.clippedReverse, clippedReverse);
    addScalar(aggregate.jensenShannon, jensenShannon);
    addScalar(aggregate.totalVariation, totalVariation);
  }
}

function emptyDivergenceAggregate(): DivergenceAggregate {
  return {
    rows: 0,
    cells: 0,
    alignedDays: 0,
    skippedRows: 0,
    forwardInfiniteRows: 0,
    reverseInfiniteRows: 0,
    modeAgreementRows: 0,
    exactForwardFinite: emptyScalarAggregate(),
    exactReverseFinite: emptyScalarAggregate(),
    clippedForward: emptyScalarAggregate(),
    clippedReverse: emptyScalarAggregate(),
    jensenShannon: emptyScalarAggregate(),
    totalVariation: emptyScalarAggregate(),
  };
}

function emptyScalarAggregate(): ScalarAggregate {
  return {
    count: 0,
    sum: 0,
    sumSquares: 0,
    max: 0,
    histogram: { zeroCount: 0, bins: Array(HISTOGRAM_BIN_COUNT).fill(0) as number[] },
  };
}

function addScalar(aggregate: ScalarAggregate, value: number): void {
  if (!Number.isFinite(value) || value < 0) throw new Error(`Invalid divergence value ${value}.`);
  aggregate.count += 1;
  aggregate.sum += value;
  aggregate.sumSquares += value * value;
  aggregate.max = Math.max(aggregate.max, value);
  if (value === 0) {
    aggregate.histogram.zeroCount += 1;
    return;
  }
  const bin = Math.max(0, Math.min(
    HISTOGRAM_BIN_COUNT - 1,
    Math.floor(
      (Math.log10(value) - HISTOGRAM_MIN_EXPONENT) * HISTOGRAM_BINS_PER_DECADE,
    ),
  ));
  aggregate.histogram.bins[bin] += 1;
}

function mergeDivergence(target: DivergenceAggregate, source: DivergenceAggregate): void {
  target.rows += source.rows;
  target.cells += source.cells;
  target.alignedDays += source.alignedDays;
  target.skippedRows += source.skippedRows;
  target.forwardInfiniteRows += source.forwardInfiniteRows;
  target.reverseInfiniteRows += source.reverseInfiniteRows;
  target.modeAgreementRows += source.modeAgreementRows;
  mergeScalar(target.exactForwardFinite, source.exactForwardFinite);
  mergeScalar(target.exactReverseFinite, source.exactReverseFinite);
  mergeScalar(target.clippedForward, source.clippedForward);
  mergeScalar(target.clippedReverse, source.clippedReverse);
  mergeScalar(target.jensenShannon, source.jensenShannon);
  mergeScalar(target.totalVariation, source.totalVariation);
}

function mergeScalar(target: ScalarAggregate, source: ScalarAggregate): void {
  target.count += source.count;
  target.sum += source.sum;
  target.sumSquares += source.sumSquares;
  target.max = Math.max(target.max, source.max);
  target.histogram.zeroCount += source.histogram.zeroCount;
  for (let index = 0; index < target.histogram.bins.length; index += 1) {
    target.histogram.bins[index] += source.histogram.bins[index]!;
  }
}

function summarizeDivergence(aggregate: DivergenceAggregate) {
  return {
    alignedDays: aggregate.alignedDays,
    rows: aggregate.rows,
    cells: aggregate.cells,
    skippedRows: aggregate.skippedRows,
    forwardInfiniteFraction: fraction(aggregate.forwardInfiniteRows, aggregate.rows),
    reverseInfiniteFraction: fraction(aggregate.reverseInfiniteRows, aggregate.rows),
    modeAgreementFraction: fraction(aggregate.modeAgreementRows, aggregate.rows),
    exactForwardKlFiniteRows: summarizeScalar(aggregate.exactForwardFinite),
    exactReverseKlFiniteRows: summarizeScalar(aggregate.exactReverseFinite),
    forwardKlEpsilon: summarizeScalar(aggregate.clippedForward),
    reverseKlEpsilon: summarizeScalar(aggregate.clippedReverse),
    jensenShannon: summarizeScalar(aggregate.jensenShannon),
    totalVariation: summarizeScalar(aggregate.totalVariation),
  };
}

function summarizeScalar(aggregate: ScalarAggregate) {
  const mean = aggregate.count > 0 ? aggregate.sum / aggregate.count : null;
  const variance = aggregate.count > 0 && mean !== null
    ? Math.max(0, aggregate.sumSquares / aggregate.count - mean * mean)
    : null;
  return {
    count: aggregate.count,
    mean,
    standardDeviation: variance === null ? null : Math.sqrt(variance),
    p50: histogramQuantile(aggregate.histogram, aggregate.count, 0.5),
    p90: histogramQuantile(aggregate.histogram, aggregate.count, 0.9),
    p99: histogramQuantile(aggregate.histogram, aggregate.count, 0.99),
    max: aggregate.count > 0 ? aggregate.max : null,
  };
}

function histogramQuantile(histogram: Histogram, count: number, quantile: number): number | null {
  if (count === 0) return null;
  const target = Math.max(1, Math.ceil(count * quantile));
  if (target <= histogram.zeroCount) return 0;
  let cumulative = histogram.zeroCount;
  for (let index = 0; index < histogram.bins.length; index += 1) {
    cumulative += histogram.bins[index]!;
    if (cumulative >= target) {
      return 10 ** (
        HISTOGRAM_MIN_EXPONENT + (index + 0.5) / HISTOGRAM_BINS_PER_DECADE
      );
    }
  }
  return 10 ** HISTOGRAM_MAX_EXPONENT;
}

function fraction(numerator: number, denominator: number): number | null {
  return denominator > 0 ? numerator / denominator : null;
}

function argument(name: string): string | undefined {
  const prefix = `--${name}=`;
  return process.argv.find((value) => value.startsWith(prefix))?.slice(prefix.length);
}

function integerArgument(name: string, fallback: number): number {
  const raw = argument(name);
  if (raw === undefined) return fallback;
  const parsed = Number(raw);
  if (!Number.isInteger(parsed) || parsed <= 0) throw new Error(`--${name} must be positive.`);
  return parsed;
}
