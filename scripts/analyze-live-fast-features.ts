import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { gunzipSync } from "node:zlib";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const DEFAULT_INPUT = "data/market/mutable/external-live";
const DEFAULT_OUTPUT = "data/benchmarks/live-fast-feature-early-screen.json";
const DEFAULT_REPORT = "docs/experiments/live-fast-feature-early-screen-2026-08-19.md";
const HORIZONS = [1, 5, 15, 60];

interface LiveRecord { recordedAt: number; payload: any }
export interface FeatureObservation { second: number; values: Record<string, number> }
export interface FeatureSeries { family: string; source: string; observations: FeatureObservation[] }
export interface PriceGrid {
  startSecond: number;
  logPrice: Float64Array;
  absoluteReturnPrefix: Float64Array;
  invalidPrefix: Uint32Array;
}

interface Sample {
  second: number;
  feature: number;
  target: number;
  previous: number;
  volatility: number;
}

interface ComponentSpec {
  id: string;
  family: string;
  label: string;
  condition: (row: Sample, thresholds: number[]) => boolean;
  target: (row: Sample, thresholds: number[]) => number;
  classCount: number;
  thresholdQuantile?: number;
  thresholdIndex?: number;
}

const MAGNITUDE_QUANTILES = [0.25, 0.5, 0.75, 0.9];

export function run(args = process.argv.slice(2)) {
  const value = (name: string) => {
    const index = args.indexOf(name);
    return index < 0 ? undefined : args[index + 1];
  };
  const input = resolve(value("--input-dir") ?? DEFAULT_INPUT);
  const output = resolve(value("--output") ?? DEFAULT_OUTPUT);
  const report = resolve(value("--report") ?? DEFAULT_REPORT);
  const artifact = analyze(input);
  fs.mkdirSync(path.dirname(output), { recursive: true });
  fs.mkdirSync(path.dirname(report), { recursive: true });
  fs.writeFileSync(output, `${JSON.stringify(artifact, null, 2)}\n`, "utf8");
  fs.writeFileSync(report, renderReport(artifact), "utf8");
  console.log(`Wrote ${path.relative(repoRoot, output)}`);
  console.log(`Wrote ${path.relative(repoRoot, report)}`);
  return artifact;
}

function analyze(input: string) {
  const { price, series, diagnostics } = loadFastAnalysisInputs(input);
  const scores = series.flatMap((row) => scoreSeries(row, price));
  const componentScores = series.flatMap((row) => scoreComponentSeries(row, price));
  const contaminatedPrefixes = Object.entries(diagnostics.book.venues)
    .filter(([, row]: [string, any]) => row.status !== "healthy")
    .map(([name]) => camelToSnake(name));
  for (const row of scores) {
    row.contaminatedByFeedHealth = contaminatedPrefixes.some((prefix) => row.feature.startsWith(prefix));
    if (row.contaminatedByFeedHealth) row.classification = "feed-contaminated";
  }
  for (const row of componentScores) {
    row.contaminatedByFeedHealth = contaminatedPrefixes.some((prefix) => row.feature.startsWith(prefix));
    if (row.contaminatedByFeedHealth) row.classification = "feed-contaminated";
  }
  const ranked = scores
    .filter((row) => row.effectiveEvaluationOutcomes >= 64)
    .filter((row) => !row.contaminatedByFeedHealth)
    .sort((left, right) => right.bitsPerTarget - left.bitsPerTarget);
  const strongEarly = ranked.filter((row) => row.classification === "large-early-effect");
  const weakEarly = ranked
    .filter((row) => row.classification === "weak-in-this-window")
    .sort((left, right) => left.bitsPerTarget - right.bitsPerTarget);
  const rankedComponents = componentScores
    .filter((row) => row.effectiveEvaluationOutcomes >= 64)
    .filter((row) => !row.contaminatedByFeedHealth)
    .sort((left, right) => right.bitsPerTarget - left.bitsPerTarget);
  const strongComponents = rankedComponents.filter((row) => row.classification === "large-early-effect");
  return {
    version: 2,
    generatedAt: new Date().toISOString(),
    input: path.relative(repoRoot, input).replaceAll("\\", "/"),
    methodology: {
      target: "BTCUSDT end-of-completed-second log return at 1s, 5s, 15s, and 60s, aligned by local receive time",
      causality: "Every fast feature bucket from second t first becomes available at t+1; targets begin from the completed price at t.",
      baseline: "Categorical conditional density from prior same-horizon return and trailing absolute-return state.",
      candidate: "Baseline plus one feature discretized into training-only tertiles; target quartiles are also training-only.",
      componentTargets: "Inactivity; sign conditional on an active return; active-magnitude threshold events at training-only 25/50/75/90% quantiles; sign inside small/large magnitude subsets; magnitude-tail probability within each sign; and a joint zero/sign/magnitude state.",
      validation: "First 60% trains; final 40% evaluates in six chronological blocks. Three deterministic feature permutations estimate finite-sample/extra-state bias.",
      limitation: "One live day is an early effect/broken-feed screen, not promotion or rejection evidence. Scores are exploratory and not multiple-testing adjusted.",
    },
    diagnostics,
    series: series.map(seriesDiagnostics),
    scores,
    componentScores,
    componentWinners: summarizeComponentWinners(rankedComponents),
    strongComponents: strongComponents.slice(0, 120),
    strongEarly,
    weakEarly: weakEarly.slice(0, 40),
    bestScores: ranked.slice(0, 60),
    familySummary: summarizeFamilies(scores),
    conclusions: deriveConclusions(diagnostics, strongEarly, weakEarly),
  };
}

export function loadFastAnalysisInputs(input: string) {
  const targetRecords = loadRecords(input, "binance-spot-aggtrade");
  const price = buildPriceGrid(targetRecords);
  if (!price) throw new Error("No usable Binance spot aggregate-trade target grid.");
  const bookRecords = loadRecords(input, "cross-exchange-book-1s");
  const liquidationRecords = loadRecords(input, "binance-usdm-liquidations");
  const perpetualTradeRecords = loadRecords(input, "deribit-btc-perpetual-trades");
  const optionTradeRecords = loadRecords(input, "deribit-btc-option-trades");
  const series = [
    extractBookSeries(bookRecords),
    extractLiquidationSeries(liquidationRecords),
    extractDeribitTradeSeries(perpetualTradeRecords, false),
    extractDeribitTradeSeries(optionTradeRecords, true),
  ].filter((row) => row.observations.length > 0);
  return {
    price,
    series,
    diagnostics: {
      target: observedCoverage(targetRecords),
      book: bookDiagnostics(bookRecords),
      liquidations: liquidationDiagnostics(liquidationRecords),
      deribitPerpetualTrades: tradeDiagnostics(perpetualTradeRecords),
      deribitOptionTrades: tradeDiagnostics(optionTradeRecords),
    },
  };
}

function scoreComponentSeries(series: FeatureSeries, price: PriceGrid) {
  const names = [...new Set(series.observations.flatMap((row) => Object.keys(row.values)))];
  const rows: any[] = [];
  for (const horizonSeconds of HORIZONS) {
    const prepared = new Map<number, Omit<Sample, "feature">>();
    for (const row of series.observations) {
      const index = row.second - price.startSecond - 1;
      const end = index + horizonSeconds;
      const previous = index - horizonSeconds;
      if (previous < 0 || end >= price.logPrice.length) continue;
      if (price.invalidPrefix[end + 1]! !== price.invalidPrefix[previous]!) continue;
      const volatilityStart = Math.max(0, index - Math.max(60, horizonSeconds) + 1);
      prepared.set(row.second, {
        second: row.second,
        target: price.logPrice[end]! - price.logPrice[index]!,
        previous: price.logPrice[index]! - price.logPrice[previous]!,
        volatility: price.absoluteReturnPrefix[index + 1]! - price.absoluteReturnPrefix[volatilityStart]!,
      });
    }
    for (const feature of names) {
      const samples: Sample[] = [];
      for (const row of series.observations) {
        const common = prepared.get(row.second);
        const value = row.values[feature];
        if (common && Number.isFinite(value)) samples.push({ ...common, feature: value });
      }
      if (samples.length < 320) continue;
      rows.push(...scoreFeatureComponents(series, feature, horizonSeconds, samples));
    }
  }
  return rows;
}

function scoreFeatureComponents(series: FeatureSeries, feature: string, horizonSeconds: number, samples: Sample[]) {
  const trainingCount = Math.floor(samples.length * 0.6);
  const fullTraining = samples.slice(0, trainingCount);
  const fullEvaluation = samples.slice(trainingCount);
  const activeMagnitudes = fullTraining
    .filter((row) => row.target !== 0)
    .map((row) => Math.abs(row.target))
    .sort((left, right) => left - right);
  if (activeMagnitudes.length < 64) return [];
  const thresholds = MAGNITUDE_QUANTILES.map((probability) => quantile(activeMagnitudes, probability));
  return componentSpecs().flatMap((spec) => {
    const training = fullTraining.filter((row) => spec.condition(row, thresholds));
    const evaluation = fullEvaluation.filter((row) => spec.condition(row, thresholds));
    if (training.length < 128 || evaluation.length < 64) return [];
    const targets = training.map((row) => spec.target(row, thresholds));
    if (new Set(targets).size < 2) return [];
    return [scoreComponent(series, feature, horizonSeconds, training, evaluation, thresholds, spec)];
  });
}

export function componentSpecs(): ComponentSpec[] {
  const specs: ComponentSpec[] = [
    {
      id: "inactive",
      family: "activity",
      label: "P(inactive)",
      condition: () => true,
      target: (row) => Number(row.target === 0),
      classCount: 2,
    },
    {
      id: "sign_given_active",
      family: "direction",
      label: "P(positive | active)",
      condition: (row) => row.target !== 0,
      target: (row) => Number(row.target > 0),
      classCount: 2,
    },
    {
      id: "joint_zero_sign_magnitude",
      family: "joint",
      label: "P(zero/sign/magnitude quartile)",
      condition: () => true,
      target: (row, thresholds) => {
        if (row.target === 0) return 0;
        const magnitudeBin = bin(Math.abs(row.target), thresholds.slice(0, 3));
        return 1 + magnitudeBin + Number(row.target > 0) * 4;
      },
      classCount: 9,
    },
  ];
  for (let index = 0; index < MAGNITUDE_QUANTILES.length; index += 1) {
    const q = MAGNITUDE_QUANTILES[index]!;
    specs.push({
      id: `large_q${Math.round(q * 100)}_given_active`,
      family: "magnitude",
      label: `P(|R| >= Q${Math.round(q * 100)} | active)`,
      condition: (row) => row.target !== 0,
      target: (row, thresholds) => Number(Math.abs(row.target) >= thresholds[index]!),
      classCount: 2,
      thresholdQuantile: q,
      thresholdIndex: index,
    });
  }
  for (const index of [1, 2, 3]) {
    const q = MAGNITUDE_QUANTILES[index]!;
    specs.push({
      id: `sign_given_large_q${Math.round(q * 100)}`,
      family: "direction-large",
      label: `P(positive | active, |R| >= Q${Math.round(q * 100)})`,
      condition: (row, thresholds) => row.target !== 0 && Math.abs(row.target) >= thresholds[index]!,
      target: (row) => Number(row.target > 0),
      classCount: 2,
      thresholdQuantile: q,
      thresholdIndex: index,
    });
  }
  for (const index of [0, 1, 2]) {
    const q = MAGNITUDE_QUANTILES[index]!;
    specs.push({
      id: `sign_given_small_q${Math.round(q * 100)}`,
      family: "direction-small",
      label: `P(positive | active, |R| < Q${Math.round(q * 100)})`,
      condition: (row, thresholds) => row.target !== 0 && Math.abs(row.target) < thresholds[index]!,
      target: (row) => Number(row.target > 0),
      classCount: 2,
      thresholdQuantile: q,
      thresholdIndex: index,
    });
  }
  for (const sign of [-1, 1]) {
    for (const index of [1, 2, 3]) {
      const q = MAGNITUDE_QUANTILES[index]!;
      specs.push({
        id: `large_q${Math.round(q * 100)}_given_${sign > 0 ? "positive" : "negative"}`,
        family: "magnitude-given-sign",
        label: `P(|R| >= Q${Math.round(q * 100)} | ${sign > 0 ? "positive" : "negative"})`,
        condition: (row) => row.target * sign > 0,
        target: (row, thresholds) => Number(Math.abs(row.target) >= thresholds[index]!),
        classCount: 2,
        thresholdQuantile: q,
        thresholdIndex: index,
      });
    }
  }
  return specs;
}

function scoreComponent(
  series: FeatureSeries,
  feature: string,
  horizonSeconds: number,
  training: Sample[],
  evaluation: Sample[],
  thresholds: number[],
  spec: ComponentSpec,
) {
  const previousEdges = quantileEdges(training.map((row) => row.previous), 3);
  const volatilityEdges = quantileEdges(training.map((row) => row.volatility), 3);
  const featureEdges = quantileEdges(training.map((row) => row.feature), 3);
  const targetOf = (row: Sample) => spec.target(row, thresholds);
  const baselineState = (row: Sample) => bin(row.previous, previousEdges) * 3 + bin(row.volatility, volatilityEdges);
  const baseline = fitCategorical(training.map(baselineState), training.map(targetOf), 9, spec.classCount);
  const candidate = fitCategorical(
    training.map((row) => baselineState(row) * 3 + bin(row.feature, featureEdges)),
    training.map(targetOf),
    27,
    spec.classCount,
  );
  const logRatios = evaluation.map((row) => {
    const base = baselineState(row);
    const target = targetOf(row);
    return Math.log2(candidate.probability(base * 3 + bin(row.feature, featureEdges), target) / baseline.probability(base, target));
  });
  const blocks = blockMeans(logRatios, 6);
  const bitsPerTarget = mean(logRatios);
  const lower95 = bitsPerTarget - 2.571 * sampleStandardDeviation(blocks) / Math.sqrt(blocks.length);
  const positiveBlocks = blocks.filter((value) => value > 0).length;
  const passesBlockScreen = lower95 > 0 && positiveBlocks >= 5 && bitsPerTarget > 0.001;
  const permutationBits = passesBlockScreen
    ? [1, 2, 3].map((seed) => scoreComponentPermutation(
      training, evaluation, previousEdges, volatilityEdges, spec.classCount, targetOf, seed,
    ))
    : [];
  const permutationMean = permutationBits.length > 0 ? mean(permutationBits) : null;
  const effectiveEvaluationOutcomes = Math.min(
    evaluation.length,
    Math.floor(Math.max(0, evaluation.at(-1)!.second - evaluation[0]!.second) / horizonSeconds),
  );
  return {
    family: series.family,
    source: series.source,
    feature,
    horizonSeconds,
    componentId: spec.id,
    componentFamily: spec.family,
    componentLabel: spec.label,
    thresholdQuantile: spec.thresholdQuantile ?? null,
    thresholdBps: spec.thresholdIndex === undefined ? null : thresholds[spec.thresholdIndex]!*10_000,
    trainingObservations: training.length,
    evaluationObservations: evaluation.length,
    effectiveEvaluationOutcomes,
    bitsPerTarget,
    lower95BlockBound: lower95,
    permutationBitsMean: permutationMean,
    excessOverPermutation: permutationMean === null ? null : bitsPerTarget - permutationMean,
    blockBits: blocks,
    positiveBlocks,
    classification: passesBlockScreen && permutationMean !== null && bitsPerTarget > permutationMean + 0.001
      ? "large-early-effect"
      : bitsPerTarget < 0 && positiveBlocks <= 1
        ? "weak-in-this-window"
        : "inconclusive",
  };
}

function scoreComponentPermutation(
  training: Sample[],
  evaluation: Sample[],
  previousEdges: number[],
  volatilityEdges: number[],
  classCount: number,
  targetOf: (row: Sample) => number,
  seed: number,
) {
  const trainFeatures = shuffled(training.map((row) => row.feature), seed);
  const evaluationFeatures = shuffled(evaluation.map((row) => row.feature), seed + 101);
  const featureEdges = quantileEdges(trainFeatures, 3);
  const baselineState = (row: Sample) => bin(row.previous, previousEdges) * 3 + bin(row.volatility, volatilityEdges);
  const targets = training.map(targetOf);
  const baseline = fitCategorical(training.map(baselineState), targets, 9, classCount);
  const candidate = fitCategorical(
    training.map((row, index) => baselineState(row) * 3 + bin(trainFeatures[index]!, featureEdges)),
    targets,
    27,
    classCount,
  );
  return mean(evaluation.map((row, index) => {
    const base = baselineState(row);
    const target = targetOf(row);
    return Math.log2(candidate.probability(base * 3 + bin(evaluationFeatures[index]!, featureEdges), target) / baseline.probability(base, target));
  }));
}

function summarizeComponentWinners(rows: any[]) {
  const groups = new Map<string, any[]>();
  for (const row of rows) {
    const key = `${row.horizonSeconds}:${row.componentId}`;
    const group = groups.get(key) ?? [];
    group.push(row);
    groups.set(key, group);
  }
  return [...groups.values()].map((group) => {
    const strict = group.filter((row) => row.classification === "large-early-effect");
    const best = (strict.length > 0 ? strict : group).sort((left, right) => right.bitsPerTarget - left.bitsPerTarget)[0]!;
    return {
      horizonSeconds: best.horizonSeconds,
      componentId: best.componentId,
      componentFamily: best.componentFamily,
      componentLabel: best.componentLabel,
      thresholdBps: best.thresholdBps,
      evaluatedFeatures: group.length,
      strictEffects: strict.length,
      bestFamily: best.family,
      bestFeature: best.feature,
      bitsPerTarget: best.bitsPerTarget,
      lower95BlockBound: best.lower95BlockBound,
      positiveBlocks: best.positiveBlocks,
      blockCount: best.blockBits.length,
      classification: best.classification,
    };
  }).sort((left, right) => left.horizonSeconds - right.horizonSeconds || left.componentId.localeCompare(right.componentId));
}

function extractBookSeries(records: LiveRecord[]): FeatureSeries {
  const observations = records.map((row) => ({
    second: Math.floor(row.recordedAt / 1_000) + 1,
    values: bookFeatures(row.payload),
  }));
  addRollingMeans(observations, [
    "binance_spot_l1_imbalance",
    "coinbase_spot_l1_imbalance",
    "kraken_spot_l1_imbalance",
    "deribit_perpetual_l1_imbalance",
    "binance_perpetual_l1_imbalance",
    "binance_depth_pressure",
    "spot_mid_dispersion_bps",
    "binance_perpetual_basis_bps",
  ], [5, 15, 60]);
  return { family: "cross-exchange-book", source: "cross-exchange-book-1s", observations };
}

export function bookFeatures(payload: any): Record<string, number> {
  const venues = payload?.venues ?? {};
  const values: Record<string, number> = {};
  const venueNames: Array<[string, string]> = [
    ["binanceSpot", "binance_spot"],
    ["coinbaseSpot", "coinbase_spot"],
    ["krakenSpot", "kraken_spot"],
    ["deribitPerpetual", "deribit_perpetual"],
    ["binancePerpetual", "binance_perpetual"],
  ];
  for (const [source, prefix] of venueNames) {
    const venue = venues[source];
    if (!venue?.valid || Number(venue.ageMs) > 5_000) continue;
    assignFinite(values, `${prefix}_spread_bps`, venue.spreadBps);
    assignFinite(values, `${prefix}_l1_imbalance`, venue.l1Imbalance);
    assignFinite(values, `${prefix}_top5_imbalance`, venue.top5Imbalance);
  }
  const flow = payload?.binanceSpotFlow ?? {};
  const bidAdded = finiteOrZero(flow.bidAddedQuote);
  const bidRemoved = finiteOrZero(flow.bidRemovedQuote);
  const askAdded = finiteOrZero(flow.askAddedQuote);
  const askRemoved = finiteOrZero(flow.askRemovedQuote);
  const gross = bidAdded + bidRemoved + askAdded + askRemoved;
  values.binance_depth_churn_log_quote = Math.log1p(gross);
  values.binance_depth_pressure = gross > 0 ? (bidAdded + askRemoved - askAdded - bidRemoved) / gross : 0;
  values.binance_add_imbalance = bidAdded + askAdded > 0 ? (bidAdded - askAdded) / (bidAdded + askAdded) : 0;
  values.binance_remove_imbalance = bidRemoved + askRemoved > 0 ? (askRemoved - bidRemoved) / (bidRemoved + askRemoved) : 0;
  assignFinite(values, "spot_mid_dispersion_bps", payload?.crossVenue?.spotMidDispersionBps);
  assignFinite(values, "best_executable_spread_bps", payload?.crossVenue?.bestExecutableSpreadBps);
  assignFinite(values, "binance_perpetual_basis_bps", payload?.crossVenue?.binancePerpetualBasisBps);
  return values;
}

function extractLiquidationSeries(records: LiveRecord[]): FeatureSeries {
  if (records.length === 0) return { family: "btc-liquidations", source: "binance-usdm-liquidations", observations: [] };
  const grouped = new Map<number, { count: number; quote: number; signed: number }>();
  for (const row of records) {
    const message = row.payload?.message ?? row.payload;
    const order = message?.o;
    if (order?.s !== "BTCUSDT") continue;
    const second = Math.floor(row.recordedAt / 1_000);
    const quote = Math.max(0, Number(order.ap ?? order.p) * Number(order.z ?? order.q));
    if (!Number.isFinite(quote)) continue;
    const current = grouped.get(second) ?? { count: 0, quote: 0, signed: 0 };
    current.count += 1;
    current.quote += quote;
    current.signed += order.S === "BUY" ? quote : -quote;
    grouped.set(second, current);
  }
  return {
    family: "btc-liquidations",
    source: "binance-usdm-liquidations",
    observations: denseRollingEvents(records, grouped, "btc_liquidation"),
  };
}

function extractDeribitTradeSeries(records: LiveRecord[], option: boolean): FeatureSeries {
  if (records.length === 0) return { family: option ? "deribit-option-flow" : "deribit-perpetual-flow", source: "", observations: [] };
  const grouped = new Map<number, { count: number; quote: number; signed: number }>();
  for (const row of records) {
    const trades = row.payload?.message?.params?.data;
    if (!Array.isArray(trades)) continue;
    const second = Math.floor(row.recordedAt / 1_000);
    const current = grouped.get(second) ?? { count: 0, quote: 0, signed: 0 };
    for (const trade of trades) {
      const amount = Math.max(0, Number(trade.amount ?? trade.contracts));
      if (!Number.isFinite(amount)) continue;
      current.count += 1;
      current.quote += amount;
      current.signed += trade.direction === "buy" ? amount : -amount;
    }
    grouped.set(second, current);
  }
  const prefix = option ? "deribit_option_trade" : "deribit_perpetual_trade";
  return {
    family: option ? "deribit-option-flow" : "deribit-perpetual-flow",
    source: option ? "deribit-btc-option-trades" : "deribit-btc-perpetual-trades",
    observations: denseRollingEvents(records, grouped, prefix),
  };
}

function denseRollingEvents(
  records: LiveRecord[],
  grouped: Map<number, { count: number; quote: number; signed: number }>,
  prefix: string,
) {
  const start = Math.floor(records[0]!.recordedAt / 1_000);
  const end = Math.floor(records.at(-1)!.recordedAt / 1_000);
  const length = end - start + 1;
  const count = new Float64Array(length);
  const quote = new Float64Array(length);
  const signed = new Float64Array(length);
  for (const [second, value] of grouped) {
    const index = second - start;
    if (index < 0 || index >= length) continue;
    count[index] = value.count;
    quote[index] = value.quote;
    signed[index] = value.signed;
  }
  const countPrefix = prefixSum(count);
  const quotePrefix = prefixSum(quote);
  const signedPrefix = prefixSum(signed);
  return Array.from({ length }, (_, index) => {
    const values: Record<string, number> = {};
    for (const window of [1, 5, 15, 60]) {
      const from = Math.max(0, index + 1 - window);
      const eventCount = rangeSum(countPrefix, from, index + 1);
      const total = rangeSum(quotePrefix, from, index + 1);
      const net = rangeSum(signedPrefix, from, index + 1);
      values[`${prefix}_count_${window}s`] = Math.log1p(eventCount);
      values[`${prefix}_amount_${window}s`] = Math.log1p(total);
      values[`${prefix}_imbalance_${window}s`] = total > 0 ? net / total : 0;
    }
    return { second: start + index + 1, values };
  });
}

function addRollingMeans(observations: FeatureObservation[], names: string[], windows: number[]) {
  for (const name of names) {
    for (const window of windows) {
      const queue: Array<{ second: number; value: number }> = [];
      let sum = 0;
      for (const row of observations) {
        const value = row.values[name];
        if (Number.isFinite(value)) {
          queue.push({ second: row.second, value });
          sum += value;
        }
        while (queue.length > 0 && queue[0]!.second <= row.second - window) sum -= queue.shift()!.value;
        if (queue.length >= Math.max(1, Math.floor(window * 0.8))) row.values[`${name}_mean_${window}s`] = sum / queue.length;
      }
    }
  }
}

function scoreSeries(series: FeatureSeries, price: PriceGrid) {
  const names = [...new Set(series.observations.flatMap((row) => Object.keys(row.values)))];
  const rows: any[] = [];
  for (const horizonSeconds of HORIZONS) {
    const prepared = new Map<number, Omit<Sample, "feature">>();
    for (const row of series.observations) {
      const index = row.second - price.startSecond - 1;
      const end = index + horizonSeconds;
      const previous = index - horizonSeconds;
      if (previous < 0 || end >= price.logPrice.length) continue;
      if (price.invalidPrefix[end + 1]! !== price.invalidPrefix[previous]!) continue;
      const volatilityStart = Math.max(0, index - Math.max(60, horizonSeconds) + 1);
      prepared.set(row.second, {
        second: row.second,
        target: price.logPrice[end]! - price.logPrice[index]!,
        previous: price.logPrice[index]! - price.logPrice[previous]!,
        volatility: price.absoluteReturnPrefix[index + 1]! - price.absoluteReturnPrefix[volatilityStart]!,
      });
    }
    for (const feature of names) {
      const samples: Sample[] = [];
      for (const row of series.observations) {
        const common = prepared.get(row.second);
        const value = row.values[feature];
        if (common && Number.isFinite(value)) samples.push({ ...common, feature: value });
      }
      if (samples.length < 160) continue;
      rows.push(scoreFeature(series, feature, horizonSeconds, samples));
    }
  }
  return rows;
}

function scoreFeature(series: FeatureSeries, feature: string, horizonSeconds: number, samples: Sample[]) {
  const trainingCount = Math.floor(samples.length * 0.6);
  const training = samples.slice(0, trainingCount);
  const evaluation = samples.slice(trainingCount);
  const targetEdges = quantileEdges(training.map((row) => row.target), 4);
  const previousEdges = quantileEdges(training.map((row) => row.previous), 3);
  const volatilityEdges = quantileEdges(training.map((row) => row.volatility), 3);
  const featureEdges = quantileEdges(training.map((row) => row.feature), 3);
  const baselineStates = training.map((row) => bin(row.previous, previousEdges) * 3 + bin(row.volatility, volatilityEdges));
  const targetBins = training.map((row) => bin(row.target, targetEdges));
  const baseline = fitCategorical(baselineStates, targetBins, 9, 4);
  const candidateStates = training.map((row, index) => baselineStates[index]! * 3 + bin(row.feature, featureEdges));
  const candidate = fitCategorical(candidateStates, targetBins, 27, 4);
  const logRatios = evaluation.map((row) => {
    const target = bin(row.target, targetEdges);
    const base = bin(row.previous, previousEdges) * 3 + bin(row.volatility, volatilityEdges);
    const state = base * 3 + bin(row.feature, featureEdges);
    return Math.log2(candidate.probability(state, target) / baseline.probability(base, target));
  });
  const permutationBits = [1, 2, 3].map((seed) => scorePermutation(training, evaluation, targetEdges, previousEdges, volatilityEdges, seed));
  const blocks = blockMeans(logRatios, 6);
  const bitsPerTarget = mean(logRatios);
  const standardDeviation = sampleStandardDeviation(blocks);
  const lower95 = bitsPerTarget - 2.571 * standardDeviation / Math.sqrt(blocks.length);
  const permutationMean = mean(permutationBits);
  const effectiveEvaluationOutcomes = Math.min(
    evaluation.length,
    Math.floor(Math.max(0, evaluation.at(-1)!.second - evaluation[0]!.second) / horizonSeconds),
  );
  const positiveBlocks = blocks.filter((value) => value > 0).length;
  const classification = lower95 > 0 && positiveBlocks >= 5 && bitsPerTarget > permutationMean + 0.001
    ? "large-early-effect"
    : bitsPerTarget < 0 && positiveBlocks <= 1
      ? "weak-in-this-window"
      : "inconclusive";
  return {
    family: series.family,
    source: series.source,
    feature,
    horizonSeconds,
    observations: samples.length,
    trainingObservations: training.length,
    evaluationObservations: evaluation.length,
    effectiveEvaluationOutcomes,
    bitsPerTarget,
    lower95BlockBound: lower95,
    permutationBitsMean: permutationMean,
    excessOverPermutation: bitsPerTarget - permutationMean,
    blockBits: blocks,
    positiveBlocks,
    classification,
  };
}

function scorePermutation(
  training: Sample[], evaluation: Sample[], targetEdges: number[], previousEdges: number[], volatilityEdges: number[], seed: number,
) {
  const trainFeatures = shuffled(training.map((row) => row.feature), seed);
  const evaluationFeatures = shuffled(evaluation.map((row) => row.feature), seed + 101);
  const edges = quantileEdges(trainFeatures, 3);
  const trainStates = training.map((row, index) => {
    const base = bin(row.previous, previousEdges) * 3 + bin(row.volatility, volatilityEdges);
    return base * 3 + bin(trainFeatures[index]!, edges);
  });
  const targets = training.map((row) => bin(row.target, targetEdges));
  const model = fitCategorical(trainStates, targets, 27, 4);
  const baseline = fitCategorical(
    training.map((row) => bin(row.previous, previousEdges) * 3 + bin(row.volatility, volatilityEdges)),
    targets,
    9,
    4,
  );
  return mean(evaluation.map((row, index) => {
    const target = bin(row.target, targetEdges);
    const base = bin(row.previous, previousEdges) * 3 + bin(row.volatility, volatilityEdges);
    return Math.log2(model.probability(base * 3 + bin(evaluationFeatures[index]!, edges), target) / baseline.probability(base, target));
  }));
}

function bookDiagnostics(records: LiveRecord[]) {
  const venueNames = ["binanceSpot", "coinbaseSpot", "krakenSpot", "deribitPerpetual", "binancePerpetual"];
  const venues = Object.fromEntries(venueNames.map((name) => {
    const rows = records.map((row) => row.payload?.venues?.[name]).filter(Boolean);
    const valid = rows.filter((row) => row.valid && Number(row.ageMs) <= 5_000);
    const ages = valid.map((row) => Number(row.ageMs)).filter(Number.isFinite).sort((a, b) => a - b);
    return [name, {
      rows: rows.length,
      validRows: valid.length,
      validFraction: rows.length > 0 ? valid.length / rows.length : 0,
      medianAgeMs: quantile(ages, 0.5),
      p99AgeMs: quantile(ages, 0.99),
      status: rows.length > 0 && valid.length / rows.length >= 0.95 ? "healthy" : "degraded",
    }];
  }));
  return { ...observedCoverage(records), rows: records.length, venues };
}

function liquidationDiagnostics(records: LiveRecord[]) {
  const symbols = new Map<string, number>();
  let btcMessages = 0;
  for (const row of records) {
    const symbol = String((row.payload?.message ?? row.payload)?.o?.s ?? "unknown");
    symbols.set(symbol, (symbols.get(symbol) ?? 0) + 1);
    if (symbol === "BTCUSDT") btcMessages += 1;
  }
  return {
    ...observedCoverage(records),
    messages: records.length,
    btcMessages,
    btcFraction: records.length > 0 ? btcMessages / records.length : 0,
    topSymbols: [...symbols].sort((left, right) => right[1] - left[1]).slice(0, 10).map(([symbol, count]) => ({ symbol, count })),
  };
}

function tradeDiagnostics(records: LiveRecord[]) {
  let trades = 0;
  for (const row of records) {
    const data = row.payload?.message?.params?.data;
    if (Array.isArray(data)) trades += data.length;
  }
  return { ...observedCoverage(records), messages: records.length, trades };
}

function seriesDiagnostics(series: FeatureSeries) {
  const coverage = observedCoverage(series.observations.map((row) => ({ recordedAt: row.second * 1_000 })));
  const names = [...new Set(series.observations.flatMap((row) => Object.keys(row.values)))];
  return {
    family: series.family,
    source: series.source,
    observations: series.observations.length,
    featureCount: names.length,
    coverage,
  };
}

function deriveConclusions(diagnostics: any, strong: any[], weak: any[]) {
  const degradedVenues = Object.entries(diagnostics.book.venues)
    .filter(([, value]: any) => value.status !== "healthy")
    .map(([name, value]: any) => `${name} (${(100 * value.validFraction).toFixed(1)}% valid)`);
  return {
    feedHealth: degradedVenues.length === 0
      ? "All compact venue books are at least 95% valid."
      : `Degraded compact books: ${degradedVenues.join(", ")}. Treat their features as missing behind observed/age masks.`,
    largeEffects: strong.length > 0
      ? `${strong.length} feature/horizon pairs clear the strict early-effect rule; they are candidates for the 3-7 day confirmation, not production promotion.`
      : "No feature/horizon pair clears the strict early-effect rule yet.",
    weakEffects: weak.length > 0
      ? `${weak.length} feature/horizon pairs are consistently negative in this window, but one day is not enough for permanent rejection.`
      : "No feature is stable enough to call weak across this window.",
    nextDecision: "Repeat after 3 and 7 observed days. Promote only effects that retain positive held-out gain across separated days and in a joint ablation against the established basis.",
  };
}

function summarizeFamilies(scores: any[]) {
  const families = [...new Set(scores.map((row) => row.family))];
  return families.map((family) => {
    const rows = scores.filter((row) => row.family === family && !row.contaminatedByFeedHealth);
    const ordered = rows.slice().sort((left, right) => right.bitsPerTarget - left.bitsPerTarget);
    const best = ordered[0];
    return {
      family,
      pairs: rows.length,
      strictEarlyEffects: rows.filter((row) => row.classification === "large-early-effect").length,
      weakInWindow: rows.filter((row) => row.classification === "weak-in-this-window").length,
      bestFeature: best?.feature ?? null,
      bestHorizonSeconds: best?.horizonSeconds ?? null,
      bestBitsPerTarget: best?.bitsPerTarget ?? null,
      bestLower95BlockBound: best?.lower95BlockBound ?? null,
    };
  });
}

function camelToSnake(value: string) {
  return value.replace(/[A-Z]/g, (letter) => `_${letter.toLowerCase()}`);
}

function buildPriceGrid(records: LiveRecord[]): PriceGrid | undefined {
  const prices = new Map<number, number>();
  for (const row of records) {
    const message = row.payload?.message ?? row.payload;
    const price = Number(message?.p);
    if (price > 0) prices.set(Math.floor(row.recordedAt / 1_000), Math.log(price));
  }
  if (prices.size < 2) return undefined;
  const seconds = [...prices.keys()].sort((left, right) => left - right);
  const startSecond = seconds[0]!;
  const endSecond = seconds.at(-1)!;
  const logPrice = new Float64Array(endSecond - startSecond + 1);
  const absoluteReturnPrefix = new Float64Array(logPrice.length + 1);
  const invalidPrefix = new Uint32Array(logPrice.length + 1);
  let previous = prices.get(startSecond)!;
  let previousObserved = startSecond;
  for (let second = startSecond; second <= endSecond; second += 1) {
    const observed = prices.get(second);
    if (observed !== undefined) {
      previous = observed;
      previousObserved = second;
    }
    const index = second - startSecond;
    logPrice[index] = previous;
    absoluteReturnPrefix[index + 1] = absoluteReturnPrefix[index]! + (index === 0 ? 0 : Math.abs(previous - logPrice[index - 1]!));
    invalidPrefix[index + 1] = invalidPrefix[index]! + (second - previousObserved > 5 ? 1 : 0);
  }
  return { startSecond, logPrice, absoluteReturnPrefix, invalidPrefix };
}

function observedCoverage(records: Array<{ recordedAt: number }>) {
  const seconds = [...new Set(records.map((row) => Math.floor(row.recordedAt / 1_000)))].sort((a, b) => a - b);
  if (seconds.length === 0) return { observedHours: 0, wallHours: 0, firstObservedAt: null, lastObservedAt: null, maximumGapSeconds: null };
  let covered = 1;
  let maximumGap = 0;
  for (let index = 1; index < seconds.length; index += 1) {
    const gap = seconds[index]! - seconds[index - 1]!;
    covered += Math.min(5, gap);
    maximumGap = Math.max(maximumGap, gap);
  }
  return {
    observedHours: covered / 3_600,
    wallHours: (seconds.at(-1)! - seconds[0]! + 1) / 3_600,
    firstObservedAt: new Date(seconds[0]! * 1_000).toISOString(),
    lastObservedAt: new Date(seconds.at(-1)! * 1_000).toISOString(),
    maximumGapSeconds: maximumGap,
  };
}

function loadRecords(input: string, source: string): LiveRecord[] {
  const root = path.join(input, source);
  if (!fs.existsSync(root)) return [];
  const records: LiveRecord[] = [];
  for (const file of listFiles(root).filter((candidate) => candidate.endsWith(".jsonl.gz"))) {
    let text: string;
    try { text = gunzipSync(fs.readFileSync(file)).toString("utf8"); }
    catch (error) {
      console.warn(`Skipping changing file ${path.relative(repoRoot, file)}: ${error instanceof Error ? error.message : error}`);
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

function listFiles(root: string) {
  return fs.readdirSync(root, { withFileTypes: true })
    .filter((entry) => entry.isFile())
    .map((entry) => path.join(root, entry.name))
    .sort();
}

function prefixSum(values: Float64Array) {
  const prefix = new Float64Array(values.length + 1);
  for (let index = 0; index < values.length; index += 1) prefix[index + 1] = prefix[index]! + values[index]!;
  return prefix;
}

function rangeSum(prefix: Float64Array, start: number, end: number) {
  return prefix[end]! - prefix[start]!;
}

function assignFinite(target: Record<string, number>, name: string, value: unknown) {
  const numeric = Number(value);
  if (Number.isFinite(numeric)) target[name] = numeric;
}

function finiteOrZero(value: unknown) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : 0;
}

function fitCategorical(states: number[], targets: number[], stateCount: number, classCount: number) {
  const counts = new Float64Array(stateCount * classCount);
  const totals = new Float64Array(stateCount);
  for (let index = 0; index < states.length; index += 1) {
    counts[states[index]! * classCount + targets[index]!] += 1;
    totals[states[index]!] += 1;
  }
  return {
    probability(state: number, target: number) {
      return (counts[state * classCount + target]! + 0.5) / (totals[state]! + 0.5 * classCount);
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
  const blockCount = Math.min(maximumBlocks, Math.max(1, Math.floor(values.length / 32)));
  return Array.from({ length: blockCount }, (_, block) => {
    const start = Math.floor(block * values.length / blockCount);
    const end = Math.floor((block + 1) * values.length / blockCount);
    return mean(values.slice(start, end));
  });
}

function mean(values: number[]) {
  return values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
}

function sampleStandardDeviation(values: number[]) {
  const center = mean(values);
  return Math.sqrt(values.reduce((sum, value) => sum + (value - center) ** 2, 0) / Math.max(1, values.length - 1));
}

function shuffled(values: number[], seed: number) {
  const output = values.slice();
  let state = seed >>> 0;
  const random = () => {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 2 ** 32;
  };
  for (let index = output.length - 1; index > 0; index -= 1) {
    const target = Math.floor(random() * (index + 1));
    [output[index], output[target]] = [output[target]!, output[index]!];
  }
  return output;
}

function renderReport(artifact: ReturnType<typeof analyze>) {
  const percent = (value: number) => `${(100 * value).toFixed(2)}%`;
  const bits = (value: number) => value.toFixed(6);
  const lines = [
    "# Live fast-feature early screen — 2026-08-19",
    "",
    `Generated at ${artifact.generatedAt}. Target coverage is **${artifact.diagnostics.target.observedHours.toFixed(3)} observed hours**.`,
    "",
    "## Interpretation",
    "",
    `- ${artifact.conclusions.feedHealth}`,
    `- ${artifact.conclusions.largeEffects}`,
    `- ${artifact.conclusions.weakEffects}`,
    `- ${artifact.conclusions.nextDecision}`,
    "- This is explicitly an early effect and broken-feed screen. It does not promote or permanently reject model inputs.",
    "",
    "## Causal validation",
    "",
    `- ${artifact.methodology.causality}`,
    `- ${artifact.methodology.target}`,
    `- ${artifact.methodology.baseline}`,
    `- ${artifact.methodology.candidate}`,
    `- ${artifact.methodology.componentTargets}`,
    `- ${artifact.methodology.validation}`,
    `- ${artifact.methodology.limitation}`,
    "",
    "## Feed health",
    "",
    `Compact book rows: ${artifact.diagnostics.book.rows.toLocaleString()}; observed coverage: ${artifact.diagnostics.book.observedHours.toFixed(3)}h; maximum gap: ${artifact.diagnostics.book.maximumGapSeconds}s.`,
    "",
    "| venue | valid rows | valid fraction | median age | p99 age | status |",
    "|---|---:|---:|---:|---:|---|",
    ...Object.entries(artifact.diagnostics.book.venues).map(([name, row]: [string, any]) => `| ${name} | ${row.validRows.toLocaleString()} | ${percent(row.validFraction)} | ${row.medianAgeMs.toFixed(1)}ms | ${row.p99AgeMs.toFixed(1)}ms | ${row.status} |`),
    "",
    "Kraken-derived scores from this archive are marked `feed-contaminated` and excluded from all rankings. The collector failed to truncate the reconstructed book to the subscribed depth; [Kraken's reconstruction rules](https://docs.kraken.com/exchange/guides/websockets/book-checksum-v2) explicitly say that zero-quantity removals are not sent for levels that merely fall out of scope. The collector now truncates to depth 100 and reconnects for a fresh snapshot whenever the reconstructed book crosses. Historical compact rows are not rewritten.",
    "",
    `All-market liquidation messages: ${artifact.diagnostics.liquidations.messages.toLocaleString()}; BTCUSDT messages: ${artifact.diagnostics.liquidations.btcMessages.toLocaleString()} (${percent(artifact.diagnostics.liquidations.btcFraction)}). Absence is encoded as zero rather than dropping non-event seconds.`,
    "",
    `Deribit perpetual trades: ${artifact.diagnostics.deribitPerpetualTrades.trades.toLocaleString()}; option trades: ${artifact.diagnostics.deribitOptionTrades.trades.toLocaleString()}.`,
    "",
    "## Family summary",
    "",
    "| family | evaluated pairs | strict early effects | weak in window | best clean feature | target | bits/target | lower block bound |",
    "|---|---:|---:|---:|---|---:|---:|---:|",
    ...artifact.familySummary.map((row) => `| ${row.family} | ${row.pairs} | ${row.strictEarlyEffects} | ${row.weakInWindow} | ${row.bestFeature ?? "n/a"} | ${row.bestHorizonSeconds ?? "n/a"}s | ${row.bestBitsPerTarget === null ? "n/a" : bits(row.bestBitsPerTarget)} | ${row.bestLower95BlockBound === null ? "n/a" : bits(row.bestLower95BlockBound)} |`),
    "",
    `Whole-return strict rows by family: ${artifact.familySummary.map((row) => `${row.family} ${row.strictEarlyEffects}/${row.pairs}`).join("; ")}. The main pattern is still magnitude/activity-state information rather than stable direction; the component audit below makes that distinction explicit.`,
    "",
    "## Separate future-component targets",
    "",
    "Each row below selects the best individually tested input for that target component. `P(|R| >= threshold)` and `P(|R| < threshold)` are complementary binary targets and therefore have the same information score at the same threshold; only the `>=` form is listed. Thresholds are active-return quantiles fitted on the training interval and are shown in bps. A conditional row is scored only on outcomes satisfying its condition.",
    "",
    "| horizon | component | threshold | tested inputs | strict effects | best family / input | bits/eligible target | lower block bound | status |",
    "|---:|---|---:|---:|---:|---|---:|---:|---|",
    ...artifact.componentWinners.map((row) => `| ${row.horizonSeconds}s | ${row.componentLabel} | ${row.thresholdBps === null ? "n/a" : `${row.thresholdBps.toFixed(4)} bps`} | ${row.evaluatedFeatures} | ${row.strictEffects} | ${row.bestFamily} / ${row.bestFeature} | ${bits(row.bitsPerTarget)} | ${bits(row.lower95BlockBound)} | ${row.classification} |`),
    "",
    "The component score is conditional information per eligible outcome, so values from differently conditioned rows are not additive and should not be compared as though they used the same sample population. The joint zero/sign/magnitude row is the closest component audit to a single complete-distribution target.",
    "At 1s, Q25/Q50/Q75 active magnitudes cluster around one BTC price tick (about 0.0015–0.0016 bps in this window), so those are not three economically distinct regimes. The Q90 row is the first clearly separated 1s tail threshold.",
    "",
    "## Strict early-effect candidates",
    "",
  ];
  if (artifact.strongEarly.length === 0) lines.push("No feature/horizon pair cleared the strict early-effect rule.");
  else lines.push(
    "A row must have a positive 95% block bound, at least 5/6 positive chronological blocks, and exceed its shuffled-feature control. These remain confirmation candidates only.",
    "",
    "| family | feature | target | bits/target | lower block bound | shuffled control | positive blocks | effective outcomes |",
    "|---|---|---:|---:|---:|---:|---:|---:|",
    ...artifact.strongEarly.map((row) => `| ${row.family} | ${row.feature} | ${row.horizonSeconds}s | ${bits(row.bitsPerTarget)} | ${bits(row.lower95BlockBound)} | ${bits(row.permutationBitsMean)} | ${row.positiveBlocks}/${row.blockBits.length} | ${row.effectiveEvaluationOutcomes.toLocaleString()} |`),
  );
  lines.push(
    "",
    "## Best exploratory scores",
    "",
    "| family | feature | target | bits/target | excess over shuffle | blocks | classification |",
    "|---|---|---:|---:|---:|---:|---|",
    ...artifact.bestScores.slice(0, 40).map((row) => `| ${row.family} | ${row.feature} | ${row.horizonSeconds}s | ${bits(row.bitsPerTarget)} | ${bits(row.excessOverPermutation)} | ${row.positiveBlocks}/${row.blockBits.length} | ${row.classification} |`),
    "",
    "## Consistently negative rows in this window",
    "",
    artifact.weakEarly.length === 0 ? "None." : "These are candidates for later rejection, not rejected features. A single live day cannot distinguish a genuinely useless input from a regime-specific failure.",
    "",
    ...(artifact.weakEarly.length === 0 ? [] : [
      "| family | feature | target | bits/target | blocks |",
      "|---|---|---:|---:|---:|",
      ...artifact.weakEarly.slice(0, 30).map((row) => `| ${row.family} | ${row.feature} | ${row.horizonSeconds}s | ${bits(row.bitsPerTarget)} | ${row.positiveBlocks}/${row.blockBits.length} |`),
    ]),
    "",
    "Machine-readable results: `data/benchmarks/live-fast-feature-early-screen.json`.",
    "",
  );
  return lines.join("\n");
}

function resolve(relativeOrAbsolute: string) {
  return path.isAbsolute(relativeOrAbsolute) ? relativeOrAbsolute : path.resolve(repoRoot, relativeOrAbsolute);
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) run();
