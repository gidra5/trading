import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { gunzipSync } from "node:zlib";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const DEFAULT_INPUT = "data/market/mutable/external-live";
const DEFAULT_OUTPUT = "data/benchmarks/live-external-early-screen.json";
const DEFAULT_REPORT = "docs/experiments/live-external-early-screen-2026-08-17.md";
const DEFAULT_COLLECTOR_STATUS = "data/runtime-cache/external-live-collector/collector-status.json";
const HORIZONS = [1, 5, 15, 60, 300, 900, 1_800, 3_600];
const CANDIDATE_SOURCES = [
  "binance-usdm-premium-index",
  "deribit-btc-option-summary",
  "mempool-live",
  "gdelt-crypto-news",
] as const;

interface LiveRecord {
  recordedAt: number;
  payload: any;
}

interface FeatureObservation {
  second: number;
  values: Record<string, number>;
}

interface FeatureSeries {
  family: string;
  source: string;
  nominalCadenceSeconds: number;
  observations: FeatureObservation[];
}

interface PriceGrid {
  startSecond: number;
  endSecond: number;
  logPrice: Float64Array;
  absoluteReturnPrefix: Float64Array;
  invalidPrefix: Uint32Array;
}

interface ScoreRow {
  family: string;
  source: string;
  feature: string;
  horizonSeconds: number;
  observations: number;
  trainingObservations: number;
  evaluationObservations: number;
  effectiveEvaluationOutcomes: number;
  bitsPerTarget: number;
  blockBits: number[];
  positiveBlocks: number;
  evidence: "smoke-only" | "early" | "provisional";
  decision: "promising" | "inconclusive" | "weak-in-this-window";
}

export interface ReadinessRow {
  horizonSeconds: number;
  independentOutcomesAt1Hour: number;
  independentOutcomesAt1Day: number;
  earliestUse: string;
  provisionalHistory: string;
}

export function buildReadinessTable(): ReadinessRow[] {
  return HORIZONS.map((horizonSeconds) => ({
    horizonSeconds,
    independentOutcomesAt1Hour: Math.floor(3_600 / horizonSeconds),
    independentOutcomesAt1Day: Math.floor(86_400 / horizonSeconds),
    earliestUse: horizonSeconds <= 15
      ? "1h feed/large-effect smoke test; 1d early screen"
      : horizonSeconds <= 60
        ? "1d early screen"
        : horizonSeconds <= 900
          ? "1d is weak; use at least 7d"
          : "1d is not a predictive validation window",
    provisionalHistory: horizonSeconds <= 15
      ? "3-7d"
      : horizonSeconds <= 60
        ? "7-14d"
        : horizonSeconds <= 900
          ? "14-30d"
          : "30-90d",
  }));
}

export function effectiveOutcomeCount(durationSeconds: number, horizonSeconds: number, observations: number) {
  return Math.max(0, Math.min(observations, Math.floor(durationSeconds / Math.max(1, horizonSeconds))));
}

export function observedCoverage(records: Array<{ recordedAt: number }>, maximumCarrySeconds = 5) {
  const seconds = [...new Set(records
    .map((row) => Math.floor(row.recordedAt / 1_000))
    .filter(Number.isFinite))].sort((left, right) => left - right);
  if (seconds.length === 0) return { seconds: 0, firstObservedAt: null, lastObservedAt: null, wallSpanSeconds: 0 };
  let coveredSeconds = 1;
  for (let index = 1; index < seconds.length; index += 1) {
    coveredSeconds += Math.min(maximumCarrySeconds, seconds[index]! - seconds[index - 1]!);
  }
  return {
    seconds: coveredSeconds,
    firstObservedAt: seconds[0]! * 1_000,
    lastObservedAt: seconds.at(-1)! * 1_000,
    wallSpanSeconds: seconds.at(-1)! - seconds[0]! + 1,
  };
}

export function classifyEarlyScore(
  durationHours: number,
  bitsPerTarget: number,
  blockBits: number[],
): Pick<ScoreRow, "evidence" | "decision"> {
  const evidence = durationHours < 24 ? "smoke-only" : durationHours < 24 * 7 ? "early" : "provisional";
  const positiveBlocks = blockBits.filter((value) => value > 0).length;
  const requiredPositive = Math.max(1, Math.ceil(blockBits.length * 0.75));
  const decision = bitsPerTarget > 0 && positiveBlocks >= requiredPositive
    ? "promising"
    : bitsPerTarget <= 0 && positiveBlocks <= Math.floor(blockBits.length / 4)
      ? "weak-in-this-window"
      : "inconclusive";
  return { evidence, decision };
}

export function run(args = process.argv.slice(2)) {
  const value = (name: string) => {
    const index = args.indexOf(name);
    return index < 0 ? undefined : args[index + 1];
  };
  const input = resolve(value("--input-dir") ?? DEFAULT_INPUT);
  const output = resolve(value("--output") ?? DEFAULT_OUTPUT);
  const report = resolve(value("--report") ?? DEFAULT_REPORT);
  const checkpoint = value("--checkpoint");
  const artifact = analyze(input);
  writeArtifact(artifact, output, report);
  if (checkpoint) {
    const outputExt = path.extname(output);
    const reportExt = path.extname(report);
    writeArtifact(
      artifact,
      output.slice(0, -outputExt.length) + `-${checkpoint}${outputExt}`,
      report.slice(0, -reportExt.length) + `-${checkpoint}${reportExt}`,
    );
  }
  console.log(`Wrote ${path.relative(repoRoot, output)}`);
  console.log(`Wrote ${path.relative(repoRoot, report)}`);
  return artifact;
}

function analyze(input: string) {
  if (!fs.existsSync(input)) throw new Error(`Live input directory does not exist: ${input}`);
  const sessionRecords = loadRecords(input, "collector-session");
  const startEvents = sessionRecords.filter((row) => row.payload?.event === "start");
  const collectorStartedAt = startEvents.length > 0
    ? Date.parse(String(startEvents.at(-1)!.payload.startedAt))
    : oldestMtime(input);
  const generatedAt = Date.now();
  const targetRecords = loadRecords(input, "binance-spot-aggtrade");
  const coverage = observedCoverage(targetRecords);
  const durationSeconds = coverage.seconds;
  const durationHours = durationSeconds / 3_600;
  const wallDurationSeconds = Math.max(0, (generatedAt - collectorStartedAt) / 1_000);
  const liveStatus = readJson(path.resolve(repoRoot, DEFAULT_COLLECTOR_STATUS));
  const liveSessionBytes = liveStatus && path.resolve(String(liveStatus.output ?? "")) === input
    ? liveStatus.compressedBytesBySource as Record<string, number> | undefined
    : undefined;
  const sessionWallDurationHours = wallDurationSeconds / 3_600;
  const inventory = inventorySources(input, sessionWallDurationHours, collectorStartedAt, liveSessionBytes);
  const priceGrid = buildPriceGrid(targetRecords);
  const featureSeries = CANDIDATE_SOURCES
    .map((source) => extractFeatureSeries(source, loadRecords(input, source)))
    .filter((series) => series.observations.length > 0);
  const featureHealth = featureSeries.flatMap(summarizeFeatureHealth);
  const scores = priceGrid && durationHours >= 1
    ? featureSeries.flatMap((series) => scoreSeries(series, priceGrid, durationHours))
    : [];
  const candidateBytes = inventory
    .filter((row) => CANDIDATE_SOURCES.includes(row.source as typeof CANDIDATE_SOURCES[number]) || row.source === "deribit-btc-option-surface-raw")
    .reduce((sum, row) => sum + row.bytes, 0);
  const candidateSessionBytes = inventory
    .filter((row) => CANDIDATE_SOURCES.includes(row.source as typeof CANDIDATE_SOURCES[number]) || row.source === "deribit-btc-option-surface-raw")
    .reduce((sum, row) => sum + row.sessionBytes, 0);
  const totalBytes = inventory.reduce((sum, row) => sum + row.bytes, 0);
  const sessionBytes = inventory.reduce((sum, row) => sum + row.sessionBytes, 0);
  return {
    version: 2,
    generatedAt: new Date(generatedAt).toISOString(),
    input: path.relative(repoRoot, input).replaceAll("\\", "/"),
    collectorStartedAt: new Date(collectorStartedAt).toISOString(),
    firstObservedAt: coverage.firstObservedAt === null ? null : new Date(coverage.firstObservedAt).toISOString(),
    lastObservedAt: coverage.lastObservedAt === null ? null : new Date(coverage.lastObservedAt).toISOString(),
    stalenessSeconds: coverage.lastObservedAt === null ? null : Math.max(0, (generatedAt - coverage.lastObservedAt) / 1_000),
    wallDurationSeconds,
    wallSpanSeconds: coverage.wallSpanSeconds,
    durationSeconds,
    durationHours,
    checkpoint: durationHours < 1 ? "pre-1h" : durationHours < 24 ? "1h-smoke" : durationHours < 24 * 7 ? "1d-early" : "multi-day",
    interpretation: durationHours < 1
      ? "Feed-health preview only; predictive scores are deliberately disabled before one hour."
      : durationHours < 24
        ? "Pipeline and very-large-effect smoke test only. Negative results cannot reject a feature."
        : durationHours < 24 * 7
          ? "Early 1s-1m screen. Retain only as preliminary evidence; this window contains too few independent slow-horizon outcomes."
          : "Provisional multi-regime evidence for fast targets; longer targets still require their horizon-specific history.",
    storage: {
      totalBytes,
      sessionBytes,
      candidateBytes,
      candidateSessionBytes,
      projectedTotalBytesPerDay: sessionWallDurationHours > 0 ? sessionBytes * 24 / sessionWallDurationHours : null,
      projectedCandidateBytesPerDay: sessionWallDurationHours > 0 ? candidateSessionBytes * 24 / sessionWallDurationHours : null,
      note: "Evidence duration is observed target coverage, with gaps capped at five carried seconds; wall-clock age is never treated as data. Storage rate alone uses latest-session wall time because bytes accrue with elapsed time. Normal collection now stores causal 1s book summaries instead of full high-frequency books; --raw-books is diagnostic-only.",
    },
    readiness: buildReadinessTable(),
    inventory,
    featureHealth,
    scores,
    bestScores: scores
      .filter((row) => row.effectiveEvaluationOutcomes >= 16)
      .slice()
      .sort((left, right) => right.bitsPerTarget - left.bitsPerTarget)
      .slice(0, 30),
    rules: {
      oneHour: "Verify timestamps, missingness, cadence, feature variation, target alignment, and catch only very large 1s-15s effects.",
      oneDay: "Run the first held-out distribution-likelihood screen for 1s-1m targets. Do not reject a varying feature solely from one market day.",
      earlyStop: "Stop immediately for malformed, stale, or near-constant feeds. For predictive usefulness, stop only after the upper uncertainty bound is immaterial across separated days; keep when improvements repeat in at least 3/4 chronological blocks.",
      slowFeeds: "GDELT yields about 96 observations/day and needs at least 7d for an initial screen. A 1h target has only 24 non-overlapping outcomes/day and needs 30-90d.",
    },
  };
}

function writeArtifact(artifact: ReturnType<typeof analyze>, output: string, report: string) {
  fs.mkdirSync(path.dirname(output), { recursive: true });
  fs.mkdirSync(path.dirname(report), { recursive: true });
  fs.writeFileSync(output, `${JSON.stringify(artifact, null, 2)}\n`, "utf8");
  fs.writeFileSync(report, renderReport(artifact), "utf8");
}

function inventorySources(
  input: string,
  durationHours: number,
  collectorStartedAt: number,
  liveSessionBytes?: Record<string, number>,
) {
  return fs.readdirSync(input, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => {
      const root = path.join(input, entry.name);
      const files = listFiles(root);
      const bytes = files.reduce((sum, file) => sum + fs.statSync(file).size, 0);
      const sessionFiles = files.filter((file) => fs.statSync(file).birthtimeMs >= collectorStartedAt - 5_000);
      const sessionBytes = liveSessionBytes?.[entry.name]
        ?? sessionFiles.reduce((sum, file) => sum + fs.statSync(file).size, 0);
      return {
        source: entry.name,
        files: files.length,
        bytes,
        sessionFiles: sessionFiles.length,
        sessionBytes,
        projectedBytesPerDay: durationHours > 0 ? sessionBytes * 24 / durationHours : null,
      };
    })
    .sort((left, right) => right.bytes - left.bytes);
}

function readJson(file: string): any {
  try { return JSON.parse(fs.readFileSync(file, "utf8")); } catch { return undefined; }
}

function loadRecords(input: string, source: string): LiveRecord[] {
  const root = path.join(input, source);
  if (!fs.existsSync(root)) return [];
  const records: LiveRecord[] = [];
  for (const file of listFiles(root).filter((candidate) => candidate.endsWith(".jsonl.gz"))) {
    let text: string;
    try {
      text = gunzipSync(fs.readFileSync(file)).toString("utf8");
    } catch (error) {
      console.warn(`Skipping a live file snapshot that changed while reading: ${path.relative(repoRoot, file)} (${error instanceof Error ? error.message : error})`);
      continue;
    }
    for (const line of text.split("\n")) {
      if (!line) continue;
      const parsed = JSON.parse(line) as LiveRecord;
      if (Number.isFinite(parsed.recordedAt)) records.push(parsed);
    }
  }
  return records.sort((left, right) => left.recordedAt - right.recordedAt);
}

function listFiles(root: string): string[] {
  const files: string[] = [];
  const stack = [root];
  while (stack.length > 0) {
    const current = stack.pop()!;
    for (const entry of fs.readdirSync(current, { withFileTypes: true })) {
      const item = path.join(current, entry.name);
      if (entry.isDirectory()) stack.push(item);
      else if (entry.isFile()) files.push(item);
    }
  }
  return files.sort();
}

function oldestMtime(root: string) {
  const times = listFiles(root).map((file) => fs.statSync(file).birthtimeMs).filter(Number.isFinite);
  return times.length > 0 ? Math.min(...times) : Date.now();
}

function buildPriceGrid(records: LiveRecord[]): PriceGrid | undefined {
  const prices = new Map<number, number>();
  for (const row of records) {
    const message = row.payload?.message ?? row.payload;
    const price = Number(message?.p);
    const timestamp = Number(message?.T ?? message?.E ?? row.recordedAt);
    if (price > 0 && Number.isFinite(timestamp)) prices.set(Math.floor(timestamp / 1_000), Math.log(price));
  }
  if (prices.size < 2) return undefined;
  const seconds = [...prices.keys()].sort((left, right) => left - right);
  const startSecond = seconds[0]!;
  const endSecond = seconds.at(-1)!;
  const logPrice = new Float64Array(endSecond - startSecond + 1);
  const absoluteReturnPrefix = new Float64Array(logPrice.length + 1);
  const invalidPrefix = new Uint32Array(logPrice.length + 1);
  let previous = prices.get(startSecond)!;
  let previousObservedSecond = startSecond;
  for (let second = startSecond; second <= endSecond; second += 1) {
    const observed = prices.get(second);
    if (observed !== undefined) {
      previous = observed;
      previousObservedSecond = second;
    }
    const index = second - startSecond;
    logPrice[index] = previous;
    const change = index === 0 ? 0 : Math.abs(previous - logPrice[index - 1]!);
    absoluteReturnPrefix[index + 1] = absoluteReturnPrefix[index]! + change;
    invalidPrefix[index + 1] = invalidPrefix[index]! + (second - previousObservedSecond > 5 ? 1 : 0);
  }
  return { startSecond, endSecond, logPrice, absoluteReturnPrefix, invalidPrefix };
}

function extractFeatureSeries(source: typeof CANDIDATE_SOURCES[number], records: LiveRecord[]): FeatureSeries {
  const observations = records.map((row) => ({
    second: Math.floor(row.recordedAt / 1_000),
    values: source === "binance-usdm-premium-index"
      ? premiumFeatures(row.payload)
      : source === "deribit-btc-option-summary"
        ? optionFeatures(row.payload)
        : source === "mempool-live"
          ? mempoolFeatures(row.payload)
          : gdeltFeatures(row.payload),
  })).filter((row) => Object.keys(row.values).length > 0);
  return {
    family: source === "binance-usdm-premium-index"
      ? "futures-premium"
      : source === "deribit-btc-option-summary"
        ? "options"
        : source === "mempool-live"
          ? "mempool"
          : "news-gdelt",
    source,
    nominalCadenceSeconds: source === "gdelt-crypto-news" ? 900 : 60,
    observations,
  };
}

function premiumFeatures(payload: any): Record<string, number> {
  const mark = Number(payload?.markPrice);
  const index = Number(payload?.indexPrice);
  return finiteRecord({
    premium_bps: mark > 0 && index > 0 ? 10_000 * Math.log(mark / index) : Number.NaN,
    funding_rate_bps: 10_000 * Number(payload?.lastFundingRate),
    mark_index_abs_gap_bps: mark > 0 && index > 0 ? 10_000 * Math.abs(Math.log(mark / index)) : Number.NaN,
  });
}

function optionFeatures(payload: any): Record<string, number> {
  const expiries = Array.isArray(payload?.expiries) ? payload.expiries : [];
  const nearest = (days: number) => expiries.reduce((best: any, row: any) => {
    if (!Number.isFinite(Number(row?.daysToExpiry))) return best;
    return !best || Math.abs(Number(row.daysToExpiry) - days) < Math.abs(Number(best.daysToExpiry) - days) ? row : best;
  }, undefined);
  const one = nearest(1);
  const seven = nearest(7);
  const thirty = nearest(30);
  const atm1 = Number(one?.atmIv);
  const atm7 = Number(seven?.atmIv);
  const atm30 = Number(thirty?.atmIv);
  return finiteRecord({
    atm_iv_1d: atm1,
    atm_iv_7d: atm7,
    atm_iv_30d: atm30,
    atm_iv_term_7d_minus_1d: atm7 - atm1,
    atm_iv_term_30d_minus_7d: atm30 - atm7,
    put_call_25d_skew_1d: Number(one?.putCall25Skew),
    put_call_25d_skew_7d: Number(seven?.putCall25Skew),
    put_call_25d_skew_30d: Number(thirty?.putCall25Skew),
    total_call_put_oi_imbalance: Number(payload?.callPutOpenInterestImbalance),
    one_day_call_put_oi_imbalance: Number(one?.callPutOpenInterestImbalance),
    log_distance_to_major_strike: Number(payload?.logDistanceToNearestMajorStrike),
    hours_to_next_expiry: Number(payload?.hoursToNextExpiry),
  });
}

function mempoolFeatures(payload: any): Record<string, number> {
  const mempool = payload?.mempool ?? {};
  const fees = payload?.recommendedFees ?? {};
  const blocks = Array.isArray(payload?.projectedBlocks) ? payload.projectedBlocks : [];
  const firstBlock = blocks[0] ?? {};
  const vsize = Number(mempool?.vsize);
  const totalFee = Number(mempool?.total_fee);
  return finiteRecord({
    transaction_count: Number(mempool?.count),
    virtual_size: vsize,
    total_fee_btc: totalFee / 1e8,
    mean_fee_sat_vbyte: vsize > 0 ? totalFee / vsize : Number.NaN,
    fastest_fee: Number(fees?.fastestFee),
    half_hour_fee: Number(fees?.halfHourFee),
    hour_fee: Number(fees?.hourFee),
    economy_fee: Number(fees?.economyFee),
    minimum_fee: Number(fees?.minimumFee),
    projected_first_block_vsize: Number(firstBlock?.blockVSize),
    projected_first_block_fee_range_high: Number(Array.isArray(firstBlock?.feeRange) ? firstBlock.feeRange.at(-1) : Number.NaN),
  });
}

function gdeltFeatures(payload: any): Record<string, number> {
  return finiteRecord({
    crypto_terms_per_million: Number(payload?.ngrams?.cryptoTermsPerMillion),
    story_count: Number(payload?.gkg?.storyCount),
    source_count: Number(payload?.gkg?.sourceCount),
    mean_tone: Number(payload?.gkg?.meanTone),
    mean_positive: Number(payload?.gkg?.meanPositive),
    mean_negative: Number(payload?.gkg?.meanNegative),
    mean_polarity: Number(payload?.gkg?.meanPolarity),
  });
}

function finiteRecord(input: Record<string, number>) {
  return Object.fromEntries(Object.entries(input).filter(([, value]) => Number.isFinite(value)));
}

function summarizeFeatureHealth(series: FeatureSeries) {
  const names = [...new Set(series.observations.flatMap((row) => Object.keys(row.values)))];
  return names.map((feature) => {
    const values = series.observations.map((row) => row.values[feature]).filter(Number.isFinite) as number[];
    const sorted = values.slice().sort((left, right) => left - right);
    const changes = values.slice(1).reduce((sum, value, index) => sum + (value !== values[index] ? 1 : 0), 0);
    const mean = values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
    const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / Math.max(1, values.length - 1);
    const distinct = new Set(values.map((value) => value.toPrecision(12))).size;
    return {
      family: series.family,
      source: series.source,
      feature,
      observations: values.length,
      changes,
      distinct,
      mean,
      standardDeviation: Math.sqrt(variance),
      p05: quantile(sorted, 0.05),
      p95: quantile(sorted, 0.95),
      health: values.length < 4 ? "too-few-observations" : distinct < 3 || changes < 2 ? "near-constant" : "varying",
    };
  });
}

function scoreSeries(series: FeatureSeries, price: PriceGrid, durationHours: number): ScoreRow[] {
  const names = [...new Set(series.observations.flatMap((row) => Object.keys(row.values)))];
  const rows: ScoreRow[] = [];
  for (const feature of names) {
    for (const horizonSeconds of HORIZONS) {
      const samples = series.observations.flatMap((row) => {
        const index = row.second - price.startSecond;
        const end = index + horizonSeconds;
        const previous = index - horizonSeconds;
        if (previous < 0 || end >= price.logPrice.length) return [];
        if (price.invalidPrefix[end + 1]! !== price.invalidPrefix[previous]!) return [];
        const value = row.values[feature];
        if (!Number.isFinite(value)) return [];
        const volatilityStart = Math.max(0, index - Math.max(60, horizonSeconds));
        return [{
          second: row.second,
          feature: value,
          target: price.logPrice[end]! - price.logPrice[index]!,
          previous: price.logPrice[index]! - price.logPrice[previous]!,
          volatility: price.absoluteReturnPrefix[index + 1]! - price.absoluteReturnPrefix[volatilityStart]!,
        }];
      });
      if (samples.length < 48) continue;
      const trainingCount = Math.floor(samples.length * 0.6);
      const evaluationCount = samples.length - trainingCount;
      if (trainingCount < 24 || evaluationCount < 16) continue;
      const training = samples.slice(0, trainingCount);
      const evaluation = samples.slice(trainingCount);
      const targetEdges = quantileEdges(training.map((row) => row.target), 4);
      const previousEdges = quantileEdges(training.map((row) => row.previous), 3);
      const volatilityEdges = quantileEdges(training.map((row) => row.volatility), 3);
      const featureEdges = quantileEdges(training.map((row) => row.feature), 3);
      const baselineStates = training.map((row) => bin(row.previous, previousEdges) * 3 + bin(row.volatility, volatilityEdges));
      const candidateStates = training.map((row, index) => baselineStates[index]! * 3 + bin(row.feature, featureEdges));
      const targets = training.map((row) => bin(row.target, targetEdges));
      const baselineModel = fitCategorical(baselineStates, targets, 9, 4);
      const candidateModel = fitCategorical(candidateStates, targets, 27, 4);
      const logRatios = evaluation.map((row) => {
        const target = bin(row.target, targetEdges);
        const base = bin(row.previous, previousEdges) * 3 + bin(row.volatility, volatilityEdges);
        const candidate = base * 3 + bin(row.feature, featureEdges);
        return Math.log2(candidateModel.probability(candidate, target) / baselineModel.probability(base, target));
      });
      const blockBits = blockMeans(logRatios, 6);
      const bitsPerTarget = mean(logRatios);
      const classified = classifyEarlyScore(durationHours, bitsPerTarget, blockBits);
      const evaluationDuration = Math.max(0, evaluation.at(-1)!.second - evaluation[0]!.second);
      const effectiveEvaluationOutcomes = effectiveOutcomeCount(evaluationDuration, horizonSeconds, evaluationCount);
      rows.push({
        family: series.family,
        source: series.source,
        feature,
        horizonSeconds,
        observations: samples.length,
        trainingObservations: trainingCount,
        evaluationObservations: evaluationCount,
        effectiveEvaluationOutcomes,
        bitsPerTarget,
        blockBits,
        positiveBlocks: blockBits.filter((value) => value > 0).length,
        evidence: classified.evidence,
        decision: effectiveEvaluationOutcomes < 16 ? "inconclusive" : classified.decision,
      });
    }
  }
  return rows;
}

function fitCategorical(states: number[], targets: number[], stateCount: number, classCount: number) {
  const counts = new Float64Array(stateCount * classCount);
  const totals = new Float64Array(stateCount);
  for (let index = 0; index < states.length; index += 1) {
    counts[states[index]! * classCount + targets[index]!] += 1;
    totals[states[index]!] += 1;
  }
  const alpha = 0.5;
  return {
    probability(state: number, target: number) {
      return (counts[state * classCount + target]! + alpha) / (totals[state]! + alpha * classCount);
    },
  };
}

function quantileEdges(values: number[], bins: number) {
  const sorted = values.slice().sort((left, right) => left - right);
  return Array.from({ length: bins - 1 }, (_, index) => quantile(sorted, (index + 1) / bins));
}

function quantile(sorted: number[], probability: number) {
  if (sorted.length === 0) return Number.NaN;
  const location = (sorted.length - 1) * probability;
  const lower = Math.floor(location);
  const upper = Math.ceil(location);
  const weight = location - lower;
  return sorted[lower]! * (1 - weight) + sorted[upper]! * weight;
}

function bin(value: number, edges: number[]) {
  let index = 0;
  while (index < edges.length && value > edges[index]!) index += 1;
  return index;
}

function blockMeans(values: number[], maximumBlocks: number) {
  const blockCount = Math.min(maximumBlocks, Math.max(1, Math.floor(values.length / 8)));
  return Array.from({ length: blockCount }, (_, block) => {
    const start = Math.floor(block * values.length / blockCount);
    const end = Math.floor((block + 1) * values.length / blockCount);
    return mean(values.slice(start, end));
  });
}

function mean(values: number[]) {
  return values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
}

function renderReport(artifact: ReturnType<typeof analyze>) {
  const mib = (bytes: number | null) => bytes === null ? "n/a" : `${(bytes / 1024 ** 2).toFixed(2)} MiB`;
  const seconds = (value: number) => value < 60 ? `${value}s` : value < 3_600 ? `${value / 60}m` : `${value / 3_600}h`;
  const lines = [
    "# Live external-feature early screen — 2026-08-17",
    "",
    `Generated at ${artifact.generatedAt} after **${artifact.durationHours.toFixed(3)} hours of observed target coverage**. The latest collector session is ${artifact.wallDurationSeconds.toFixed(0)} wall-clock seconds old; the archive spans ${(artifact.wallSpanSeconds / 3_600).toFixed(3)} wall-clock hours including gaps.`,
    "",
    `Latest target observation: ${artifact.lastObservedAt ?? "none"}; staleness: ${artifact.stalenessSeconds === null ? "n/a" : `${artifact.stalenessSeconds.toFixed(1)}s`}.`,
    "",
    `**Current interpretation:** ${artifact.interpretation}`,
    "",
    "## What one hour and one day can establish",
    "",
    "- **1 hour:** data-quality and alignment validation, plus a smoke test for unusually large 1s–15s effects. It is not a rejection test.",
    "- **1 day:** an early held-out screen for 1s–1m targets. It still represents one market regime, so negative results are not enough to discard a varying feature.",
    "- **7+ days:** first useful separated-day evidence for minute-cadence options/mempool and 15-minute GDELT measurements.",
    "- **30–90 days:** needed for a credible 1h target test because a day contains only 24 non-overlapping 1h outcomes.",
    "",
    "| target | independent outcomes in 1h | independent outcomes in 1d | earliest use | provisional history |",
    "|---:|---:|---:|---|---:|",
    ...artifact.readiness.map((row) => `| ${seconds(row.horizonSeconds)} | ${row.independentOutcomesAt1Hour} | ${row.independentOutcomesAt1Day} | ${row.earliestUse} | ${row.provisionalHistory} |`),
    "",
    "## Storage",
    "",
    `- Stored across all trials: ${mib(artifact.storage.totalBytes)}; active session: ${mib(artifact.storage.sessionBytes)}; projected active rate: ${mib(artifact.storage.projectedTotalBytesPerDay)}/day.`,
    "- Treat a projection from the first 15 minutes as an upper-biased startup estimate: one immediate options surface, GDELT pull, and mempool snapshot have not yet been amortized over their normal cadences.",
    `- Unresolved slow candidate feeds in the active session: ${mib(artifact.storage.candidateSessionBytes)}; projected ${mib(artifact.storage.projectedCandidateBytesPerDay)}/day.`,
    `- ${artifact.storage.note}`,
    "",
    "## Feed inventory",
    "",
    "| source | files | stored | active-session stored | projected/day |",
    "|---|---:|---:|---:|---:|",
    ...artifact.inventory.map((row) => `| ${row.source} | ${row.files} | ${mib(row.bytes)} | ${mib(row.sessionBytes)} | ${mib(row.projectedBytesPerDay)} |`),
    "",
    "## Candidate feature health",
    "",
    "| family | feature | observations | changes | distinct | health |",
    "|---|---|---:|---:|---:|---|",
    ...artifact.featureHealth.map((row) => `| ${row.family} | ${row.feature} | ${row.observations} | ${row.changes} | ${row.distinct} | ${row.health} |`),
    "",
    "## Predictive smoke/early screen",
    "",
  ];
  if (artifact.bestScores.length === 0) {
    lines.push("No predictive score is reported yet. The one-hour minimum is intentional; rows collected before it are only a feed-health preview.");
  } else {
    lines.push(
      "Scores are out-of-sample gain over trailing-return and trailing-volatility bins. Positive bits/target is better. The first 60% of this live window fits all quantile edges and categorical probabilities; the last 40% is evaluated in six chronological blocks.",
      "This is a single chronological split. During the smoke stage, a few appended observations can move the split boundary and materially reorder sparse categorical scores; do not treat the ranking as a selected model basis until it survives separated-day evaluation.",
      "",
      "| family | feature | target | eval rows | effective outcomes | bits/target | positive blocks | decision | evidence |",
      "|---|---|---:|---:|---:|---:|---:|---|---|",
      ...artifact.bestScores.map((row) => `| ${row.family} | ${row.feature} | ${seconds(row.horizonSeconds)} | ${row.evaluationObservations} | ${row.effectiveEvaluationOutcomes} | ${row.bitsPerTarget.toFixed(6)} | ${row.positiveBlocks}/${row.blockBits.length} | ${row.decision} | ${row.evidence} |`),
    );
  }
  lines.push(
    "",
    "## Retention decision",
    "",
    "1. Stop a feed immediately only if it is malformed, timestamp-misaligned, stale, or near-constant.",
    "2. At 24h, promote repeated positive results to `early`; do not reject a varying feature from a single day.",
    "3. Starting with separated days, stop a feature family only when its upper uncertainty bound is below the chosen material gain threshold across target horizons.",
    "4. Compact high-volume raw books into causal 1s derived features after their reconstruction tests; the unresolved slow feeds themselves are cheap enough to retain through the required evidence window.",
    "",
    "Machine-readable results are in `data/benchmarks/live-external-early-screen.json`.",
    "",
  );
  return lines.join("\n");
}

function resolve(relativeOrAbsolute: string) {
  return path.isAbsolute(relativeOrAbsolute) ? relativeOrAbsolute : path.resolve(repoRoot, relativeOrAbsolute);
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) run();
