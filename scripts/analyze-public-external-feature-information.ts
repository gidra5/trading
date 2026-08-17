import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { readCandleShardReferenceSync, type SequentialCandle } from "@trading/storage";
import type { CoinMetricsRow, CommunityCryptoDailyRow, DvolRow, MempoolMiningRow, VixRow } from "./lib/external-public-data.ts";

const MINUTE_MS = 60_000;
const DAY_MS = 86_400_000;
const PRIMARY_START = "2025-03-18";
const PRIMARY_END = "2025-11-13";
const TRANSFER_START = "2026-04-21";
const TRANSFER_END = "2026-06-23";
const TRAIN_DAYS = 180;
const TARGET_BINS = 8;
const FEATURE_BINS = 4;
const SMOOTHING = 0.5;
const MISSING_BIN = 255;
const SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT"] as const;
const ALT_SYMBOLS = SYMBOLS.slice(1);
const DEFAULT_EXTERNAL_DIRECTORY = "data/market/mutable/external";
const DEFAULT_OUTPUT = "data/benchmarks/public-external-feature-information.json";
const DEFAULT_REPORT = "docs/experiments/public-external-feature-information-2026-08-17.md";
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

interface StoredArtifact<T> {
  retrievedAt: string;
  source: string;
  rows: T[];
}

interface Segment {
  firstTime: number;
  targetStart: number;
  targetEnd: number;
  minuteCount: number;
  returnPrefix: Map<string, Float64Array>;
  squareReturnPrefix: Map<string, Float64Array>;
  validReturnPrefix: Map<string, Uint32Array>;
  volumePrefix: Map<string, Float64Array>;
  validVolumePrefix: Map<string, Uint32Array>;
}

interface CandidateDefinition {
  id: string;
  label: string;
  family: string;
  source: "cross-market" | "deribit-dvol" | "cboe-vix" | "coinmetrics" | "community-daily" | "mempool-proxy";
  lookback: string;
  value: (segment: Segment, index: number, time: number) => number;
}

interface ObservationSet {
  target: Uint8Array;
  sign: Uint8Array;
  magnitude: Uint8Array;
  base: Uint8Array;
  split: Uint8Array;
  times: Float64Array;
  rawTarget: Float64Array;
  featureBins: Uint8Array[];
  targetEdges: number[];
  magnitudeEdges: number[];
  baseEdges: { previousReturn: number[]; realizedVolatility: number[] };
  candidateEdges: number[][];
}

interface GainScore {
  observations: number;
  bits: number;
  signBits: number;
  magnitudeBits: number;
}

interface CandidateScore {
  id: string;
  label: string;
  family: string;
  source: CandidateDefinition["source"];
  lookback: string;
  edges: number[];
  primary: GainScore;
  firstHalf: GainScore;
  secondHalf: GainScore;
  transfer: GainScore;
  stable: boolean;
}

interface SelectedStep extends CandidateScore {
  step: number;
  conditionedOn: string[];
}

interface GroupResult {
  id: string;
  horizonMinutes: number;
  cadenceMinutes: number;
  candidateCount: number;
  observations: Record<string, number>;
  ranked: CandidateScore[];
  selected: SelectedStep[];
}

export async function run(args = process.argv.slice(2)) {
  const value = (name: string) => {
    const index = args.indexOf(name);
    return index < 0 ? undefined : args[index + 1];
  };
  const externalDirectory = resolve(value("--external-dir") ?? DEFAULT_EXTERNAL_DIRECTORY);
  const output = resolve(value("--output") ?? DEFAULT_OUTPUT);
  const report = resolve(value("--report") ?? DEFAULT_REPORT);
  const dvol = readArtifact<DvolRow>(path.join(externalDirectory, "deribit-btc-dvol-1h.json"));
  const vix = readArtifact<VixRow>(path.join(externalDirectory, "fred-cboe-vix-1d.json"));
  const coinMetrics = readArtifact<CoinMetricsRow>(path.join(externalDirectory, "coinmetrics-btc-network-flows-1d.json"));
  const mempool = readArtifact<MempoolMiningRow>(path.join(externalDirectory, "mempool-btc-mining-proxies-3y.json"));
  const community = readArtifact<CommunityCryptoDailyRow>(path.join(externalDirectory, "community-crypto-market-daily.json"));
  dvol.rows.sort((left, right) => left.availableAt - right.availableAt);
  vix.rows.sort((left, right) => left.availableAt - right.availableAt);
  coinMetrics.rows.sort((left, right) => left.availableAt - right.availableAt);
  mempool.rows.sort((left, right) => left.availableAt - right.availableAt);
  community.rows.sort((left, right) => left.availableAt - right.availableAt);

  console.error("Loading aligned primary candle segment...");
  const primary = loadSegment(PRIMARY_START, PRIMARY_END, 31);
  console.error("Loading aligned transfer candle segment...");
  const transfer = loadSegment(TRANSFER_START, TRANSFER_END, 31);
  const candidates = buildCandidates(dvol.rows, vix.rows, coinMetrics.rows, community.rows, mempool.rows);
  const groups: GroupResult[] = [];
  const horizons = [1, 5, 15, 30, 60];
  const sourceGroups: Array<{ id: string; source: CandidateDefinition["source"]; horizons: number[] }> = [
    { id: "cross-market", source: "cross-market", horizons },
    { id: "deribit-dvol", source: "deribit-dvol", horizons: [5, 15, 30, 60] },
    { id: "cboe-vix", source: "cboe-vix", horizons },
    { id: "coinmetrics", source: "coinmetrics", horizons: [15, 30, 60] },
    { id: "community-daily", source: "community-daily", horizons: [15, 30, 60] },
    { id: "mempool-proxy", source: "mempool-proxy", horizons: [15, 30, 60] },
  ];
  for (const group of sourceGroups) for (const horizon of group.horizons) {
    const definitions = candidates.filter((candidate) => candidate.source === group.source);
    const cadence = group.source === "cross-market" ? Math.min(5, horizon) : 60;
    console.error(`Scoring ${group.id} at ${horizon}m (${definitions.length} candidates)...`);
    groups.push(analyzeGroup(`${group.id}-${horizon}m`, horizon, cadence, definitions, primary, transfer));
  }
  console.error("Scoring the joint hourly external basis...");
  groups.push(analyzeGroup("joint-external-60m", 60, 60, candidates, primary, transfer));

  const artifact = {
    version: 1,
    generatedAt: new Date().toISOString(),
    objective: "Incremental information about future BTCUSDT return distributions at 1m through 1h",
    sources: {
      target: "Binance spot BTCUSDT completed 1m candles",
      crossMarket: ALT_SYMBOLS.map((symbol) => `Binance spot ${symbol} completed 1m candles`),
      dvol: { source: dvol.source, retrievedAt: dvol.retrievedAt, rows: dvol.rows.length, first: dvol.rows[0]?.time, last: dvol.rows.at(-1)?.time },
      vix: { source: vix.source, retrievedAt: vix.retrievedAt, rows: vix.rows.length, first: vix.rows[0]?.time, last: vix.rows.at(-1)?.time },
      coinMetrics: { source: coinMetrics.source, retrievedAt: coinMetrics.retrievedAt, rows: coinMetrics.rows.length, first: coinMetrics.rows[0]?.time, last: coinMetrics.rows.at(-1)?.time },
      community: { source: community.source, retrievedAt: community.retrievedAt, rows: community.rows.length, first: community.rows[0]?.time, last: community.rows.at(-1)?.time },
      mempool: { source: mempool.source, retrievedAt: mempool.retrievedAt, rows: mempool.rows.length, first: mempool.rows[0]?.time, last: mempool.rows.at(-1)?.time },
    },
    split: {
      primary: { start: PRIMARY_START, end: PRIMARY_END, trainDays: TRAIN_DAYS, testDays: dayCount(PRIMARY_START, PRIMARY_END) - TRAIN_DAYS },
      transfer: { start: TRANSFER_START, end: TRANSFER_END, days: dayCount(TRANSFER_START, TRANSFER_END) },
    },
    method: {
      target: "Eight training-quantile cells of the forward BTC log return",
      baseline: "Quartiles of the same-horizon trailing BTC return crossed with trailing realized volatility",
      candidate: "Training-only quartiles; missing observations are omitted on a matched basis",
      score: "Held-out candidate-minus-baseline log likelihood in bits per target, plus sign and magnitude components",
      stability: "Positive full-distribution gain in both chronological primary halves and the separated transfer block",
      greedySelection: "At each step choose the candidate with the largest positive worst-block gain conditional on the already selected quartile coordinates",
      slowSourceCadence: "Hourly evaluation to avoid treating repeated daily or half-daily values as independent minute observations",
    },
    groups,
    limitations: [
      "DVOL is an hourly volatility-index history, not a historical full option surface; ATM, 25-delta skew, term structure, strike OI, and IV/skew changes remain live-forward measurements.",
      "VIX is a daily US-equity option-implied volatility index. The backtest delays each close until the next UTC day and therefore tests it as a slow macro regime feature, not a live intraday VIX feed.",
      "mempool.space history contains mined-block aggregates rather than the earlier unconfirmed transaction backlog; it is labeled as a mempool proxy and not as historical live mempool state.",
      "Coin Metrics Community exchange-flow history is downloaded retrospectively and can be revised. The screen assumes next-day availability and stores each row's latest revision time separately; results are provisional and not a true point-in-time test.",
      "The community whale/miner/liquidation/derivatives archive is CC-BY but upstream-derived and retrospectively revised. Its next-day screen is exploratory, not point-in-time production evidence.",
      "The cross-market screen covers liquid crypto spot markets. CME macro futures still require a licensed intraday point-in-time source.",
      "Information gain measures distribution forecast value before fees, latency, market impact, or a trading decision rule.",
    ],
  };
  fs.mkdirSync(path.dirname(output), { recursive: true });
  fs.writeFileSync(output, `${JSON.stringify(artifact)}\n`, "utf8");
  fs.mkdirSync(path.dirname(report), { recursive: true });
  fs.writeFileSync(report, renderReport(artifact), "utf8");
  console.log(`Wrote ${path.relative(repoRoot, output)}`);
  console.log(`Wrote ${path.relative(repoRoot, report)}`);
  return artifact;
}

function analyzeGroup(
  id: string,
  horizonMinutes: number,
  cadenceMinutes: number,
  candidates: CandidateDefinition[],
  primary: Segment,
  transfer: Segment,
): GroupResult {
  const observations = buildObservationSet(primary, transfer, horizonMinutes, cadenceMinutes, candidates);
  const ranked = candidates.map((candidate, index) => scoreCandidate(candidate, index, observations, []))
    .sort(compareCandidateScores);
  const selected: SelectedStep[] = [];
  const selectedIndices: number[] = [];
  for (let step = 0; step < 4; step += 1) {
    const remaining = candidates.map((candidate, index) => ({ candidate, index }))
      .filter(({ index }) => !selectedIndices.includes(index))
      .map(({ candidate, index }) => scoreCandidate(candidate, index, observations, selectedIndices))
      .filter((score) => score.stable)
      .sort(compareCandidateScores);
    const best = remaining[0];
    if (!best || worstBlock(best) <= 0 || best.primary.bits <= 0.0001) break;
    const index = candidates.findIndex((candidate) => candidate.id === best.id);
    selected.push({ ...best, step: step + 1, conditionedOn: selected.map((item) => item.id) });
    selectedIndices.push(index);
  }
  return {
    id,
    horizonMinutes,
    cadenceMinutes,
    candidateCount: candidates.length,
    observations: countSplits(observations.split),
    ranked,
    selected,
  };
}

function buildObservationSet(
  primary: Segment,
  transfer: Segment,
  horizon: number,
  cadence: number,
  candidates: CandidateDefinition[],
): ObservationSet {
  const raw = collectRawObservations(primary, transfer, horizon, cadence, candidates);
  const trainingIndices = raw.split.flatMap((split, index) => split === 0 ? [index] : []);
  const targetEdges = quantileEdges(trainingIndices.map((index) => raw.target[index]!), TARGET_BINS);
  const magnitudeEdges = quantileEdges(trainingIndices.map((index) => Math.abs(raw.target[index]!)), 4);
  const previousEdges = quantileEdges(trainingIndices.map((index) => raw.previous[index]!), FEATURE_BINS);
  const volatilityEdges = quantileEdges(trainingIndices.map((index) => raw.volatility[index]!), FEATURE_BINS);
  const target = new Uint8Array(raw.target.length);
  const sign = new Uint8Array(raw.target.length);
  const magnitude = new Uint8Array(raw.target.length);
  const base = new Uint8Array(raw.target.length);
  for (let index = 0; index < raw.target.length; index += 1) {
    target[index] = bin(raw.target[index]!, targetEdges);
    sign[index] = raw.target[index]! >= 0 ? 1 : 0;
    magnitude[index] = bin(Math.abs(raw.target[index]!), magnitudeEdges);
    base[index] = bin(raw.previous[index]!, previousEdges) * FEATURE_BINS
      + bin(raw.volatility[index]!, volatilityEdges);
  }
  const candidateEdges = raw.features.map((values) => quantileEdges(
    trainingIndices.map((index) => values[index]!).filter(Number.isFinite),
    FEATURE_BINS,
  ));
  const featureBins = raw.features.map((values, candidateIndex) => {
    const encoded = new Uint8Array(values.length).fill(MISSING_BIN);
    const edges = candidateEdges[candidateIndex]!;
    if (edges.length !== FEATURE_BINS - 1) return encoded;
    for (let index = 0; index < values.length; index += 1) {
      if (Number.isFinite(values[index]!)) encoded[index] = bin(values[index]!, edges);
    }
    return encoded;
  });
  return {
    target,
    sign,
    magnitude,
    base,
    split: Uint8Array.from(raw.split),
    times: Float64Array.from(raw.times),
    rawTarget: Float64Array.from(raw.target),
    featureBins,
    targetEdges,
    magnitudeEdges,
    baseEdges: { previousReturn: previousEdges, realizedVolatility: volatilityEdges },
    candidateEdges,
  };
}

function collectRawObservations(
  primary: Segment,
  transfer: Segment,
  horizon: number,
  cadence: number,
  candidates: CandidateDefinition[],
) {
  const times: number[] = [];
  const target: number[] = [];
  const previous: number[] = [];
  const volatility: number[] = [];
  const split: number[] = [];
  const features = candidates.map(() => [] as number[]);
  const append = (segment: Segment, transferSegment: boolean) => {
    const startIndex = Math.round((segment.targetStart - segment.firstTime) / MINUTE_MS);
    const endIndex = Math.round((segment.targetEnd - segment.firstTime) / MINUTE_MS);
    const btcReturns = segment.returnPrefix.get("BTCUSDT")!;
    const btcSquares = segment.squareReturnPrefix.get("BTCUSDT")!;
    const btcValid = segment.validReturnPrefix.get("BTCUSDT")!;
    for (let index = startIndex; index + horizon <= endIndex; index += cadence) {
      if (!validWindow(btcValid, index, index + horizon) || !validWindow(btcValid, index - horizon, index)) continue;
      const forward = btcReturns[index + horizon]! - btcReturns[index]!;
      const trailing = btcReturns[index]! - btcReturns[index - horizon]!;
      const trailingVolatility = Math.sqrt(Math.max(0, btcSquares[index]! - btcSquares[index - horizon]!));
      if (![forward, trailing, trailingVolatility].every(Number.isFinite)) continue;
      const time = segment.firstTime + index * MINUTE_MS;
      const dayIndex = Math.floor((time - segment.targetStart) / DAY_MS);
      const splitId = transferSegment ? 3 : dayIndex < TRAIN_DAYS ? 0 : dayIndex < TRAIN_DAYS + 30 ? 1 : 2;
      times.push(time);
      target.push(forward);
      previous.push(trailing);
      volatility.push(trailingVolatility);
      split.push(splitId);
      for (let candidate = 0; candidate < candidates.length; candidate += 1) {
        features[candidate]!.push(candidates[candidate]!.value(segment, index, time));
      }
    }
  };
  append(primary, false);
  append(transfer, true);
  return { times, target, previous, volatility, split, features };
}

function scoreCandidate(
  candidate: CandidateDefinition,
  candidateIndex: number,
  observations: ObservationSet,
  selected: number[],
): CandidateScore {
  const primary = scoreAllTargets(observations, [0], [1, 2], selected, candidateIndex);
  const firstHalf = scoreAllTargets(observations, [0], [1], selected, candidateIndex);
  const secondHalf = scoreAllTargets(observations, [0], [2], selected, candidateIndex);
  const transfer = scoreAllTargets(observations, [0, 1, 2], [3], selected, candidateIndex);
  return {
    id: candidate.id,
    label: candidate.label,
    family: candidate.family,
    source: candidate.source,
    lookback: candidate.lookback,
    edges: observations.candidateEdges[candidateIndex]!,
    primary,
    firstHalf,
    secondHalf,
    transfer,
    stable: primary.observations > 0 && firstHalf.bits > 0 && secondHalf.bits > 0 && transfer.bits > 0,
  };
}

function scoreAllTargets(
  observations: ObservationSet,
  trainSplits: number[],
  evaluationSplits: number[],
  selected: number[],
  candidate: number,
): GainScore {
  const matched = (index: number) => observations.featureBins[candidate]![index] !== MISSING_BIN
    && selected.every((item) => observations.featureBins[item]![index] !== MISSING_BIN);
  const trainIndices: number[] = [];
  const evaluationIndices: number[] = [];
  for (let index = 0; index < observations.split.length; index += 1) {
    if (!matched(index)) continue;
    if (trainSplits.includes(observations.split[index]!)) trainIndices.push(index);
    if (evaluationSplits.includes(observations.split[index]!)) evaluationIndices.push(index);
  }
  return {
    observations: evaluationIndices.length,
    bits: conditionalGain(observations, trainIndices, evaluationIndices, selected, candidate, observations.target, TARGET_BINS),
    signBits: conditionalGain(observations, trainIndices, evaluationIndices, selected, candidate, observations.sign, 2),
    magnitudeBits: conditionalGain(observations, trainIndices, evaluationIndices, selected, candidate, observations.magnitude, 4),
  };
}

export function conditionalGain(
  observations: Pick<ObservationSet, "base" | "featureBins">,
  training: number[],
  evaluation: number[],
  selected: number[],
  candidate: number,
  targets: Uint8Array,
  targetClasses: number,
): number {
  const baseContexts = 16 * FEATURE_BINS ** selected.length;
  const fullContexts = baseContexts * FEATURE_BINS;
  const baseCounts = new Float64Array(baseContexts * targetClasses);
  const baseTotals = new Float64Array(baseContexts);
  const fullCounts = new Float64Array(fullContexts * targetClasses);
  const fullTotals = new Float64Array(fullContexts);
  for (const index of training) {
    const base = contextAt(observations, index, selected);
    const full = base * FEATURE_BINS + observations.featureBins[candidate]![index]!;
    const target = targets[index]!;
    baseCounts[base * targetClasses + target] += 1;
    baseTotals[base] += 1;
    fullCounts[full * targetClasses + target] += 1;
    fullTotals[full] += 1;
  }
  let gain = 0;
  let count = 0;
  for (const index of evaluation) {
    const base = contextAt(observations, index, selected);
    const full = base * FEATURE_BINS + observations.featureBins[candidate]![index]!;
    const target = targets[index]!;
    const baseProbability = (baseCounts[base * targetClasses + target]! + SMOOTHING)
      / (baseTotals[base]! + SMOOTHING * targetClasses);
    const fullProbability = (fullCounts[full * targetClasses + target]! + SMOOTHING)
      / (fullTotals[full]! + SMOOTHING * targetClasses);
    gain += Math.log2(fullProbability / baseProbability);
    count += 1;
  }
  return count > 0 ? gain / count : Number.NaN;
}

function contextAt(
  observations: Pick<ObservationSet, "base" | "featureBins">,
  index: number,
  selected: number[],
) {
  let context = observations.base[index]!;
  for (const feature of selected) context = context * FEATURE_BINS + observations.featureBins[feature]![index]!;
  return context;
}

function buildCandidates(
  dvol: DvolRow[],
  vix: VixRow[],
  coinMetrics: CoinMetricsRow[],
  community: CommunityCryptoDailyRow[],
  mempool: MempoolMiningRow[],
): CandidateDefinition[] {
  return [
    ...crossMarketCandidates(),
    ...dvolCandidates(dvol),
    ...vixCandidates(vix),
    ...coinMetricsCandidates(coinMetrics),
    ...communityDailyCandidates(community),
    ...mempoolCandidates(mempool),
  ];
}

function vixCandidates(rows: VixRow[]): CandidateDefinition[] {
  const rowChange = (time: number, observations: number) => {
    const index = latestIndex(rows, time);
    return index >= observations ? rows[index]!.close - rows[index - observations]!.close : Number.NaN;
  };
  const candidates: CandidateDefinition[] = [candidate(
    "vix-level", "Cboe VIX level", "VIX level", "cboe-vix", "latest completed US session",
    (_segment, _index, time) => latest(rows, time)?.close ?? Number.NaN,
  )];
  for (const observations of [1, 5, 21]) {
    candidates.push(candidate(
      `vix-change-${observations}d`, "VIX change", "VIX change", "cboe-vix", `${observations} trading day(s)`,
      (_segment, _index, time) => rowChange(time, observations),
    ));
    candidates.push(candidate(
      `vix-absolute-change-${observations}d`, "Absolute VIX change", "VIX shock", "cboe-vix", `${observations} trading day(s)`,
      (_segment, _index, time) => Math.abs(rowChange(time, observations)),
    ));
  }
  for (const days of [7, 30]) candidates.push(candidate(
    `vix-btc-realized-spread-${days}d`, "VIX minus BTC realized volatility", "Cross-asset implied-realized spread", "cboe-vix", `${days}d BTC realized`,
    (segment, index, time) => {
      const row = latest(rows, time);
      return row ? row.close - annualizedVolatility(segment, "BTCUSDT", index, days * 1_440) : Number.NaN;
    },
  ));
  return candidates;
}

function communityDailyCandidates(rows: CommunityCryptoDailyRow[]): CandidateDefinition[] {
  const metricFamily: Record<string, string> = {
    btc_exchange_inflow_total: "Exchange flows",
    btc_exchange_netflow: "Exchange flows",
    btc_exchange_outflow_total: "Exchange flows",
    btc_exchange_reserve: "Exchange reserves",
    btc_exchange_reserve_usd: "Exchange reserves",
    btc_exchange_supply_ratio: "Exchange reserves",
    btc_exchange_whale_ratio: "Whale flows",
    btc_fund_flow_ratio: "Fund flows",
    btc_miner_netflow_total: "Miner flows",
    btc_miners_position_index: "Miner flows",
    btc_long_liquidations: "Daily liquidations",
    btc_long_liquidations_usd: "Daily liquidations",
    btc_short_liquidations: "Daily liquidations",
    btc_short_liquidations_usd: "Daily liquidations",
    btc_funding_rates: "Futures state",
    btc_open_interest: "Futures state",
    btc_taker_buy_sell_ratio: "Futures state",
    btc_coinbase_premium_gap: "Cross-exchange premium",
    btc_coinbase_premium_index: "Cross-exchange premium",
    btc_korea_premium_index: "Cross-exchange premium",
    stablecoin_exchange_inflow_total: "Stablecoin flows",
    stablecoin_exchange_netflow: "Stablecoin flows",
    stablecoin_exchange_outflow_total: "Stablecoin flows",
    stablecoin_exchange_reserve: "Stablecoin flows",
    stablecoin_exchange_supply_ratio: "Stablecoin flows",
    btc_exchange_stablecoins_ratio: "Stablecoin flows",
    btc_exchange_stablecoins_ratio_usd: "Stablecoin flows",
    btc_mvrv_ratio: "On-chain valuation",
    btc_puell_multiple: "Miner valuation",
  };
  const metrics = [...new Set(rows.flatMap((row) => Object.keys(row.values)))].sort();
  const candidates: CandidateDefinition[] = [];
  for (const metric of metrics) {
    const label = metric.replace(/^btc_/, "").replaceAll("_", " ");
    const family = metricFamily[metric] ?? "Community daily state";
    candidates.push(candidate(
      `community-${metric}-level`, label, family, "community-daily", "latest daily value",
      (_segment, _index, time) => signedLog(latest(rows, time)?.values[metric] ?? Number.NaN),
    ));
    for (const days of [1, 3, 7, 30]) candidates.push(candidate(
      `community-${metric}-change-${days}d`, `${label} change`, family, "community-daily", `${days}d`,
      (_segment, _index, time) => {
        const index = latestIndex(rows, time);
        if (index < days) return Number.NaN;
        const current = rows[index]!.values[metric];
        const previous = rows[index - days]!.values[metric];
        return current === undefined || previous === undefined ? Number.NaN : signedLog(current) - signedLog(previous);
      },
    ));
  }
  return candidates;
}

function crossMarketCandidates(): CandidateDefinition[] {
  const candidates: CandidateDefinition[] = [];
  for (const symbol of ALT_SYMBOLS) {
    const asset = symbol.replace("USDT", "");
    for (const lookback of [1, 2, 5, 15, 30, 60]) {
      candidates.push(candidate(
        `cross-${asset.toLowerCase()}-return-${lookback}m`, `${asset} trailing return`, `${asset} return`, "cross-market", `${lookback}m`,
        (segment, index) => windowReturn(segment, symbol, index, lookback),
      ));
      candidates.push(candidate(
        `cross-${asset.toLowerCase()}-relative-${lookback}m`, `${asset} minus BTC return`, `${asset}/BTC relative return`, "cross-market", `${lookback}m`,
        (segment, index) => windowReturn(segment, symbol, index, lookback) - windowReturn(segment, "BTCUSDT", index, lookback),
      ));
    }
    for (const lookback of [5, 15, 30, 60]) {
      candidates.push(candidate(
        `cross-${asset.toLowerCase()}-volatility-${lookback}m`, `${asset} realized volatility`, `${asset} volatility`, "cross-market", `${lookback}m`,
        (segment, index) => windowVolatility(segment, symbol, index, lookback),
      ));
      candidates.push(candidate(
        `cross-${asset.toLowerCase()}-vol-ratio-${lookback}m`, `${asset}/BTC volatility ratio`, `${asset}/BTC volatility ratio`, "cross-market", `${lookback}m`,
        (segment, index) => Math.log((windowVolatility(segment, symbol, index, lookback) + 1e-12)
          / (windowVolatility(segment, "BTCUSDT", index, lookback) + 1e-12)),
      ));
    }
    for (const lookback of [5, 15, 60]) candidates.push(candidate(
      `cross-${asset.toLowerCase()}-log-volume-${lookback}m`, `${asset} log volume`, `${asset} volume`, "cross-market", `${lookback}m`,
      (segment, index) => Math.log1p(windowVolume(segment, symbol, index, lookback)),
    ));
  }
  for (const lookback of [1, 2, 5, 15, 30, 60]) {
    candidates.push(candidate(
      `cross-alt-factor-return-${lookback}m`, "Alt-market mean return", "Alt-market factor", "cross-market", `${lookback}m`,
      (segment, index) => mean(ALT_SYMBOLS.map((symbol) => windowReturn(segment, symbol, index, lookback))),
    ));
    candidates.push(candidate(
      `cross-alt-dispersion-${lookback}m`, "Alt-market return dispersion", "Cross-market dispersion", "cross-market", `${lookback}m`,
      (segment, index) => standardDeviation(ALT_SYMBOLS.map((symbol) => windowReturn(segment, symbol, index, lookback))),
    ));
  }
  return candidates;
}

function dvolCandidates(rows: DvolRow[]): CandidateDefinition[] {
  const candidates: CandidateDefinition[] = [candidate(
    "dvol-level", "BTC DVOL level", "DVOL level", "deribit-dvol", "latest completed 1h",
    (_segment, _index, time) => latest(rows, time)?.close ?? Number.NaN,
  ), candidate(
    "dvol-age", "DVOL observation age", "DVOL age", "deribit-dvol", "latest completed 1h",
    (_segment, _index, time) => {
      const row = latest(rows, time);
      return row ? (time - row.availableAt) / MINUTE_MS : Number.NaN;
    },
  )];
  for (const hours of [1, 2, 4, 24, 72, 168]) {
    candidates.push(candidate(
      `dvol-change-${hours}h`, "DVOL change", "DVOL change", "deribit-dvol", `${hours}h`,
      (_segment, _index, time) => eventChange(rows, time, hours * 60, (row) => row.close),
    ));
    candidates.push(candidate(
      `dvol-absolute-change-${hours}h`, "Absolute DVOL change", "DVOL absolute change", "deribit-dvol", `${hours}h`,
      (_segment, _index, time) => Math.abs(eventChange(rows, time, hours * 60, (row) => row.close)),
    ));
  }
  for (const days of [1, 7, 30]) candidates.push(candidate(
    `dvol-realized-spread-${days}d`, "DVOL minus trailing realized volatility", "Implied-realized volatility spread", "deribit-dvol", `${days}d`,
    (segment, index, time) => {
      const row = latest(rows, time);
      if (!row) return Number.NaN;
      const realized = annualizedVolatility(segment, "BTCUSDT", index, days * 1_440);
      return row.close - realized;
    },
  ));
  return candidates;
}

function coinMetricsCandidates(rows: CoinMetricsRow[]): CandidateDefinition[] {
  const candidates: CandidateDefinition[] = [];
  const flow = (row: CoinMetricsRow, name: string) => row.values[name] ?? Number.NaN;
  for (const days of [1, 3, 7, 30]) {
    candidates.push(candidate(
      `coinmetrics-flow-in-${days}d`, "Exchange inflow", "Exchange flows", "coinmetrics", `${days}d sum`,
      (_segment, _index, time) => signedLog(eventSum(rows, time, days, (row) => flow(row, "FlowInExUSD"))),
    ));
    candidates.push(candidate(
      `coinmetrics-flow-out-${days}d`, "Exchange outflow", "Exchange flows", "coinmetrics", `${days}d sum`,
      (_segment, _index, time) => signedLog(eventSum(rows, time, days, (row) => flow(row, "FlowOutExUSD"))),
    ));
    candidates.push(candidate(
      `coinmetrics-net-flow-${days}d`, "Exchange net inflow", "Exchange net flow", "coinmetrics", `${days}d sum`,
      (_segment, _index, time) => signedLog(eventSum(rows, time, days, (row) => flow(row, "FlowInExUSD") - flow(row, "FlowOutExUSD"))),
    ));
    candidates.push(candidate(
      `coinmetrics-flow-imbalance-${days}d`, "Exchange flow imbalance", "Exchange flow imbalance", "coinmetrics", `${days}d sum`,
      (_segment, _index, time) => {
        const inflow = eventSum(rows, time, days, (row) => flow(row, "FlowInExUSD"));
        const outflow = eventSum(rows, time, days, (row) => flow(row, "FlowOutExUSD"));
        return (inflow - outflow) / Math.max(1, inflow + outflow);
      },
    ));
  }
  for (const days of [1, 3, 7, 30]) for (const [field, label, family] of [
    ["SplyExNtv", "Exchange-held BTC change", "Exchange reserves"],
    ["TxCnt", "Transaction-count change", "On-chain activity"],
    ["AdrActCnt", "Active-address change", "On-chain activity"],
    ["HashRate", "Hash-rate change", "Miner/network state"],
    ["FeeTotNtv", "Total-fee change", "On-chain fees"],
  ] as const) candidates.push(candidate(
    `coinmetrics-${field.toLowerCase()}-change-${days}d`, label, family, "coinmetrics", `${days}d`,
    (_segment, _index, time) => eventLogChange(rows, time, days, (row) => flow(row, field)),
  ));
  candidates.push(candidate(
    "coinmetrics-age", "Coin Metrics release age", "Source age", "coinmetrics", "latest daily release",
    (_segment, _index, time) => {
      const row = latest(rows, time);
      return row ? (time - row.availableAt) / 3_600_000 : Number.NaN;
    },
  ));
  return candidates;
}

function mempoolCandidates(rows: MempoolMiningRow[]): CandidateDefinition[] {
  const fields = [
    ["fees.avgFees", "Average block fees", "Block fees"],
    ["feeRates.avgFee_50", "Median block fee rate", "Block fee rates"],
    ["feeRates.avgFee_90", "90th-percentile block fee rate", "Block fee rates"],
    ["feeRates.avgFee_100", "Maximum block fee rate", "Block fee rates"],
    ["sizesWeights.sizes.avgSize", "Average block size", "Block utilization"],
    ["sizesWeights.weights.avgWeight", "Average block weight", "Block utilization"],
    ["rewards.avgRewards", "Average block reward", "Block rewards"],
  ] as const;
  const candidates: CandidateDefinition[] = [];
  for (const [field, label, family] of fields) {
    candidates.push(candidate(
      `mempool-${field.replaceAll(".", "-").toLowerCase()}-level`, label, family, "mempool-proxy", "latest trailing bucket",
      (_segment, _index, time) => signedLog(latest(rows, time)?.values[field] ?? Number.NaN),
    ));
    for (const buckets of [1, 2, 6, 14]) candidates.push(candidate(
      `mempool-${field.replaceAll(".", "-").toLowerCase()}-change-${buckets}`, `${label} change`, family, "mempool-proxy", `${buckets} buckets`,
      (_segment, _index, time) => eventLogChange(rows, time, buckets, (row) => row.values[field] ?? Number.NaN),
    ));
  }
  candidates.push(candidate(
    "mempool-proxy-age", "Mempool-proxy observation age", "Source age", "mempool-proxy", "latest trailing bucket",
    (_segment, _index, time) => {
      const row = latest(rows, time);
      return row ? (time - row.availableAt) / 3_600_000 : Number.NaN;
    },
  ));
  return candidates;
}

function candidate(
  id: string,
  label: string,
  family: string,
  source: CandidateDefinition["source"],
  lookback: string,
  value: CandidateDefinition["value"],
): CandidateDefinition {
  return { id, label, family, source, lookback, value };
}

function loadSegment(start: string, end: string, warmupDays: number): Segment {
  const targetStart = parseDay(start);
  const targetEnd = parseDay(end) + DAY_MS;
  const firstTime = targetStart - warmupDays * DAY_MS;
  const finalTime = targetEnd + DAY_MS;
  const minuteCount = (finalTime - firstTime) / MINUTE_MS;
  const returnPrefix = new Map<string, Float64Array>();
  const squareReturnPrefix = new Map<string, Float64Array>();
  const validReturnPrefix = new Map<string, Uint32Array>();
  const volumePrefix = new Map<string, Float64Array>();
  const validVolumePrefix = new Map<string, Uint32Array>();
  for (const symbol of SYMBOLS) {
    const closes = new Float64Array(minuteCount).fill(Number.NaN);
    const volumes = new Float64Array(minuteCount).fill(Number.NaN);
    for (let day = firstTime; day < finalTime; day += DAY_MS) {
      const date = new Date(day).toISOString().slice(0, 10);
      const file = candleReference(symbol, date);
      if (!fs.existsSync(file)) continue;
      const rows = readCandleShardReferenceSync(file) as SequentialCandle[];
      for (const row of rows) {
        const index = Math.round((row.openTime - firstTime) / MINUTE_MS);
        if (index < 0 || index >= minuteCount) continue;
        closes[index] = row.close;
        volumes[index] = row.volume;
      }
    }
    const returns = new Float64Array(minuteCount + 1);
    const squares = new Float64Array(minuteCount + 1);
    const validReturns = new Uint32Array(minuteCount + 1);
    const volumeSums = new Float64Array(minuteCount + 1);
    const validVolumes = new Uint32Array(minuteCount + 1);
    for (let index = 0; index < minuteCount; index += 1) {
      const validReturn = index > 0 && Number.isFinite(closes[index]) && Number.isFinite(closes[index - 1])
        && closes[index]! > 0 && closes[index - 1]! > 0;
      const value = validReturn ? Math.log(closes[index]! / closes[index - 1]!) : 0;
      returns[index + 1] = returns[index]! + value;
      squares[index + 1] = squares[index]! + value * value;
      validReturns[index + 1] = validReturns[index]! + (validReturn ? 1 : 0);
      const validVolume = Number.isFinite(volumes[index]) && volumes[index]! >= 0;
      volumeSums[index + 1] = volumeSums[index]! + (validVolume ? volumes[index]! : 0);
      validVolumes[index + 1] = validVolumes[index]! + (validVolume ? 1 : 0);
    }
    returnPrefix.set(symbol, returns);
    squareReturnPrefix.set(symbol, squares);
    validReturnPrefix.set(symbol, validReturns);
    volumePrefix.set(symbol, volumeSums);
    validVolumePrefix.set(symbol, validVolumes);
  }
  return { firstTime, targetStart, targetEnd, minuteCount, returnPrefix, squareReturnPrefix, validReturnPrefix, volumePrefix, validVolumePrefix };
}

function windowReturn(segment: Segment, symbol: string, end: number, minutes: number) {
  const prefix = segment.returnPrefix.get(symbol)!;
  const valid = segment.validReturnPrefix.get(symbol)!;
  return validWindow(valid, end - minutes, end) ? prefix[end]! - prefix[end - minutes]! : Number.NaN;
}

function windowVolatility(segment: Segment, symbol: string, end: number, minutes: number) {
  const prefix = segment.squareReturnPrefix.get(symbol)!;
  const valid = segment.validReturnPrefix.get(symbol)!;
  return validWindow(valid, end - minutes, end)
    ? Math.sqrt(Math.max(0, prefix[end]! - prefix[end - minutes]!))
    : Number.NaN;
}

function annualizedVolatility(segment: Segment, symbol: string, end: number, minutes: number) {
  const volatility = windowVolatility(segment, symbol, end, minutes);
  return Number.isFinite(volatility) ? volatility * Math.sqrt(525_600 / minutes) * 100 : Number.NaN;
}

function windowVolume(segment: Segment, symbol: string, end: number, minutes: number) {
  const prefix = segment.volumePrefix.get(symbol)!;
  const valid = segment.validVolumePrefix.get(symbol)!;
  return validWindow(valid, end - minutes, end) ? prefix[end]! - prefix[end - minutes]! : Number.NaN;
}

function validWindow(prefix: Uint32Array, start: number, end: number) {
  return start >= 0 && end < prefix.length && prefix[end]! - prefix[start]! === end - start;
}

function latest<T extends { availableAt: number }>(rows: T[], time: number): T | undefined {
  const index = latestIndex(rows, time);
  return index >= 0 ? rows[index] : undefined;
}

function latestIndex<T extends { availableAt: number }>(rows: T[], time: number) {
  let low = 0;
  let high = rows.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (rows[middle]!.availableAt <= time) low = middle + 1;
    else high = middle;
  }
  return low - 1;
}

function eventChange<T extends { availableAt: number }>(
  rows: T[], time: number, minutes: number, value: (row: T) => number,
) {
  const current = latest(rows, time);
  const previous = latest(rows, time - minutes * MINUTE_MS);
  return current && previous ? value(current) - value(previous) : Number.NaN;
}

function eventSum<T extends { availableAt: number }>(
  rows: T[], time: number, count: number, value: (row: T) => number,
) {
  const index = latestIndex(rows, time);
  if (index < count - 1) return Number.NaN;
  let total = 0;
  for (let offset = 0; offset < count; offset += 1) {
    const item = value(rows[index - offset]!);
    if (!Number.isFinite(item)) return Number.NaN;
    total += item;
  }
  return total;
}

function eventLogChange<T extends { availableAt: number }>(
  rows: T[], time: number, lag: number, value: (row: T) => number,
) {
  const index = latestIndex(rows, time);
  if (index < lag) return Number.NaN;
  const current = value(rows[index]!);
  const previous = value(rows[index - lag]!);
  return current > 0 && previous > 0 ? Math.log(current / previous) : Number.NaN;
}

function quantileEdges(values: number[], bins: number) {
  const sorted = values.filter(Number.isFinite).sort((left, right) => left - right);
  if (sorted.length < bins * 10) return [];
  const edges = Array.from({ length: bins - 1 }, (_, index) => {
    const position = ((index + 1) / bins) * (sorted.length - 1);
    const lower = Math.floor(position);
    const weight = position - lower;
    return sorted[lower]! * (1 - weight) + sorted[Math.min(sorted.length - 1, lower + 1)]! * weight;
  });
  return new Set(edges).size === edges.length ? edges : [];
}

function bin(value: number, edges: number[]) {
  let result = 0;
  while (result < edges.length && value > edges[result]!) result += 1;
  return result;
}

function compareCandidateScores(left: CandidateScore, right: CandidateScore) {
  return Number(right.stable) - Number(left.stable)
    || worstBlock(right) - worstBlock(left)
    || right.primary.bits - left.primary.bits;
}

function worstBlock(score: CandidateScore) {
  return Math.min(score.firstHalf.bits, score.secondHalf.bits, score.transfer.bits);
}

function countSplits(splits: Uint8Array) {
  const result: Record<string, number> = { train: 0, primaryFirst: 0, primarySecond: 0, transfer: 0 };
  for (const split of splits) {
    if (split === 0) result.train += 1;
    else if (split === 1) result.primaryFirst += 1;
    else if (split === 2) result.primarySecond += 1;
    else result.transfer += 1;
  }
  return result;
}

function renderReport(artifact: Awaited<ReturnType<typeof run>>) {
  const lines = [
    "# Public external feature information audit",
    "",
    `Generated ${artifact.generatedAt}. All scores are held-out increments beyond BTC's own trailing return and realized-volatility state.`,
    "",
    "## Outcome",
    "",
  ];
  const selected = artifact.groups.flatMap((group) => group.selected.map((item) => ({ group, item })));
  lines.push(
    `The public backfill produced ${artifact.sources.dvol.rows.toLocaleString()} hourly DVOL rows, ${artifact.sources.vix.rows.toLocaleString()} daily VIX rows, ${artifact.sources.coinMetrics.rows.toLocaleString()} Coin Metrics rows, ${artifact.sources.community.rows.toLocaleString()} community whale/miner/derivatives rows, and ${artifact.sources.mempool.rows.toLocaleString()} mined-block proxy rows. The matched cross-market corpus contains BTC plus ${ALT_SYMBOLS.join(", ")} minute bars.`,
    "",
    `${selected.length} conditional coordinates passed the three-block stability rule across all source/horizon screens. A selected coordinate is a distribution feature, not automatically a profitable direction signal.`,
    "",
    "## Selected conditional basis",
    "",
    "| screen | horizon | step | feature | lookback | primary bits | transfer bits | sign bits | magnitude bits |",
    "|---|---:|---:|---|---|---:|---:|---:|---:|",
  );
  for (const { group, item } of selected) lines.push(
    `| ${group.id} | ${group.horizonMinutes}m | ${item.step} | ${item.label} | ${item.lookback} | ${fixed(item.primary.bits)} | ${fixed(item.transfer.bits)} | ${fixed(item.primary.signBits)} | ${fixed(item.primary.magnitudeBits)} |`,
  );
  lines.push("", "## Best marginal candidate by screen", "", "| screen | candidates | best candidate | stable | lookback | first half | second half | transfer |", "|---|---:|---|:---:|---|---:|---:|---:|");
  for (const group of artifact.groups) {
    const best = group.ranked.find((item) => item.stable) ?? group.ranked[0];
    lines.push(`| ${group.id} | ${group.candidateCount} | ${best?.label ?? "none"} | ${best?.stable ? "yes" : "no"} | ${best?.lookback ?? "—"} | ${fixed(best?.firstHalf.bits)} | ${fixed(best?.secondHalf.bits)} | ${fixed(best?.transfer.bits)} |`);
  }
  lines.push(
    "",
    "## Interpretation constraints",
    "",
    ...artifact.limitations.map((item) => `- ${item}`),
    "",
    "## Method",
    "",
    `- Target: ${artifact.method.target}.`,
    `- Baseline: ${artifact.method.baseline}.`,
    `- Candidate: ${artifact.method.candidate}.`,
    `- Score: ${artifact.method.score}.`,
    `- Stability: ${artifact.method.stability}.`,
    `- Selection: ${artifact.method.greedySelection}.`,
    `- Slow sources: ${artifact.method.slowSourceCadence}.`,
    "",
    "Complete candidate rankings, frozen quantile edges, observation counts, sign decomposition, and magnitude decomposition are in `data/benchmarks/public-external-feature-information.json`.",
    "",
  );
  return lines.join("\n");
}

function readArtifact<T>(file: string): StoredArtifact<T> {
  return JSON.parse(fs.readFileSync(file, "utf8")) as StoredArtifact<T>;
}

function candleReference(symbol: string, date: string) {
  const lower = symbol.toLowerCase();
  return resolve(`data/market/immutable/refs/candles/spot-${lower}/${lower}/1m/${date}.json`);
}

function parseDay(value: string) {
  return Date.parse(`${value}T00:00:00.000Z`);
}

function dayCount(start: string, end: string) {
  return Math.round((parseDay(end) - parseDay(start)) / DAY_MS) + 1;
}

function resolve(value: string) {
  return path.resolve(repoRoot, value);
}

function signedLog(value: number) {
  return Number.isFinite(value) ? Math.sign(value) * Math.log1p(Math.abs(value)) : Number.NaN;
}

function mean(values: number[]) {
  return values.every(Number.isFinite) ? values.reduce((total, value) => total + value, 0) / values.length : Number.NaN;
}

function standardDeviation(values: number[]) {
  if (!values.every(Number.isFinite)) return Number.NaN;
  const average = mean(values);
  return Math.sqrt(values.reduce((total, value) => total + (value - average) ** 2, 0) / values.length);
}

function fixed(value: number | undefined, digits = 6) {
  return value === undefined || !Number.isFinite(value) ? "—" : value.toFixed(digits);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  run().catch((error: unknown) => {
    console.error(error instanceof Error ? error.stack ?? error.message : error);
    process.exitCode = 1;
  });
}
