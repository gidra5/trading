import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { gunzipSync } from "node:zlib";
import { conditionalGain } from "./analyze-public-external-feature-information.ts";
import type { LiquidationEvent, QuoteSecond } from "./lib/tardis-monthly-samples.ts";

const SECOND_MS = 1_000;
const SECONDS_PER_DAY = 86_400;
const FEATURE_BINS = 4;
const TARGET_BINS = 8;
const MISSING_BIN = 255;
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const DEFAULT_INPUT = "data/market/mutable/external/tardis-monthly-samples";
const DEFAULT_OUTPUT = "data/benchmarks/tardis-monthly-sample-information.json";
const DEFAULT_REPORT = "docs/experiments/tardis-monthly-sample-information-2026-08-17.md";
const VENUES = ["binance-spot-btcusdt", "coinbase-spot-btcusd", "kraken-spot-xbtusd", "deribit-btc-perpetual"] as const;
const LOOKBACKS = [1, 2, 5, 15, 30, 60, 300, 900, 3_600];
const HORIZONS = [1, 5, 15, 60, 300, 900, 1_800, 3_600];

interface QuoteArtifact { date: string; rawRows: number; rows: QuoteSecond[] }
interface LiquidationArtifact { date: string; rawRows: number; rows: LiquidationEvent[] }
interface DayData {
  date: string;
  split: number;
  quote: Map<string, VenueDay>;
  signedLiquidationPrefix: Float64Array;
  absoluteLiquidationPrefix: Float64Array;
  liquidationCountPrefix: Float64Array;
  liquidationMax: Float64Array;
}
interface VenueDay {
  logMid: Float64Array;
  bid: Float64Array;
  ask: Float64Array;
  bidAmount: Float64Array;
  askAmount: Float64Array;
  observedAt: Float64Array;
  squareReturnPrefix: Float64Array;
}
interface Candidate {
  id: string;
  label: string;
  family: string;
  lookback: string;
  value(day: DayData, second: number): number;
}

export function run(args = process.argv.slice(2)) {
  const value = (name: string) => {
    const index = args.indexOf(name);
    return index < 0 ? undefined : args[index + 1];
  };
  const input = resolve(value("--input-dir") ?? DEFAULT_INPUT);
  const output = resolve(value("--output") ?? DEFAULT_OUTPUT);
  const report = resolve(value("--report") ?? DEFAULT_REPORT);
  const days = loadDays(input);
  const candidates = buildCandidates();
  const groups = HORIZONS.map((horizon) => {
    console.error(`Scoring Tardis monthly samples at ${horizon}s (${candidates.length} candidates)...`);
    return analyzeHorizon(days, candidates, horizon);
  });
  const artifact = {
    version: 1,
    generatedAt: new Date().toISOString(),
    objective: "Incremental information about future BTC return distributions from synchronized monthly public samples",
    source: "Tardis first-day-of-month downloadable CSV samples",
    dates: days.map(({ date, split }) => ({ date, split: ["train", "first-half", "second-half", "transfer"][split] })),
    method: {
      timing: "Tardis collector local_timestamp; the last quote observed in each second predicts later Binance quote mids",
      baseline: "quartiles of same-horizon trailing Binance return and realized volatility",
      candidate: "training-only quartiles; missing observations matched before scoring",
      stability: "positive held-out bits in Aug-Sep 2025, Oct-Nov 2025, and May-Jun 2026",
      caveat: "first UTC day of each month only; this is independent multi-regime sample evidence, not continuous-history evidence",
    },
    groups,
  };
  fs.mkdirSync(path.dirname(output), { recursive: true });
  fs.mkdirSync(path.dirname(report), { recursive: true });
  fs.writeFileSync(output, JSON.stringify(artifact, null, 2));
  fs.writeFileSync(report, renderReport(artifact));
  console.log(`Wrote ${path.relative(repoRoot, output)}`);
  console.log(`Wrote ${path.relative(repoRoot, report)}`);
  return artifact;
}

function analyzeHorizon(days: DayData[], candidates: Candidate[], horizon: number) {
  const cadence = Math.min(5, horizon);
  const rawTarget: number[] = [];
  const rawMagnitude: number[] = [];
  const rawPrevious: number[] = [];
  const rawVolatility: number[] = [];
  const split: number[] = [];
  const sampleDays: DayData[] = [];
  const sampleSeconds: number[] = [];
  for (const day of days) {
    const binance = day.quote.get(VENUES[0])!;
    for (let second = 3_600; second + horizon < SECONDS_PER_DAY; second += cadence) {
      const target = binance.logMid[second + horizon]! - binance.logMid[second]!;
      const previous = windowReturn(binance, second, horizon);
      const volatility = windowVolatility(binance, second, horizon);
      if (![target, previous, volatility].every(Number.isFinite)) continue;
      rawTarget.push(target);
      rawMagnitude.push(Math.abs(target));
      rawPrevious.push(previous);
      rawVolatility.push(volatility);
      split.push(day.split);
      sampleDays.push(day);
      sampleSeconds.push(second);
    }
  }
  const training = indicesWhere(split, 0);
  const targetEdges = quantileEdges(training.map((index) => rawTarget[index]!), TARGET_BINS);
  const magnitudeEdges = quantileEdges(training.map((index) => rawMagnitude[index]!), 4);
  const previousEdges = quantileEdges(training.map((index) => rawPrevious[index]!), FEATURE_BINS);
  const volatilityEdges = quantileEdges(training.map((index) => rawVolatility[index]!), FEATURE_BINS);
  const target = Uint8Array.from(rawTarget.map((item) => binValue(item, targetEdges)));
  const sign = Uint8Array.from(rawTarget.map((item) => item >= 0 ? 1 : 0));
  const magnitude = Uint8Array.from(rawMagnitude.map((item) => binValue(item, magnitudeEdges)));
  const base = Uint8Array.from(rawPrevious.map((item, index) => binValue(item, previousEdges) * FEATURE_BINS + binValue(rawVolatility[index]!, volatilityEdges)));
  const ranked = candidates.map((candidate) => {
    const values = sampleDays.map((day, index) => candidate.value(day, sampleSeconds[index]!));
    const { edges, bins } = quantizeCandidate(candidate, values, training);
    const observations = { base, featureBins: [bins] };
    const matched = (index: number) => bins[index] !== MISSING_BIN;
    const trainMatched = training.filter(matched);
    const score = (evaluationSplits: number[], targets: Uint8Array, classes: number) => {
      const evaluation = indicesWhereMany(split, evaluationSplits).filter(matched);
      return {
        observations: evaluation.length,
        bits: conditionalGain(observations, trainMatched, evaluation, [], 0, targets, classes),
      };
    };
    const first = score([1], target, TARGET_BINS);
    const second = score([2], target, TARGET_BINS);
    const transfer = score([3], target, TARGET_BINS);
    const primary = score([1, 2], target, TARGET_BINS);
    return {
      id: candidate.id,
      label: candidate.label,
      family: candidate.family,
      lookback: candidate.lookback,
      edges,
      primaryBits: primary.bits,
      firstHalfBits: first.bits,
      secondHalfBits: second.bits,
      transferBits: transfer.bits,
      signBits: score([1, 2], sign, 2).bits,
      magnitudeBits: score([1, 2], magnitude, 4).bits,
      observations: { primary: primary.observations, transfer: transfer.observations },
      stable: first.bits > 0 && second.bits > 0 && transfer.bits > 0,
    };
  }).sort((left, right) => Math.min(right.firstHalfBits, right.secondHalfBits, right.transferBits) - Math.min(left.firstHalfBits, left.secondHalfBits, left.transferBits));
  return {
    horizonSeconds: horizon,
    cadenceSeconds: cadence,
    candidateCount: candidates.length,
    observations: split.reduce((counts, item) => { counts[item] = (counts[item] ?? 0) + 1; return counts; }, {} as Record<number, number>),
    stableCount: ranked.filter(({ stable }) => stable).length,
    best: ranked.find(({ stable }) => stable) ?? ranked[0],
    ranked,
  };
}

function buildCandidates(): Candidate[] {
  const candidates: Candidate[] = [];
  for (const venue of VENUES) {
    candidates.push(candidate(`${venue}-imbalance`, `${venue} L1 quantity imbalance`, "cross-exchange-books", "latest", (day, second) => {
      const item = day.quote.get(venue)!;
      return (item.bidAmount[second]! - item.askAmount[second]!) / (item.bidAmount[second]! + item.askAmount[second]! + 1e-12);
    }));
    candidates.push(candidate(`${venue}-spread`, `${venue} spread`, "cross-exchange-books", "latest", (day, second) => {
      const item = day.quote.get(venue)!;
      return 20_000 * (item.ask[second]! - item.bid[second]!) / (item.ask[second]! + item.bid[second]!);
    }));
    candidates.push(candidate(`${venue}-age`, `${venue} quote age`, "cross-exchange-books", "latest", (day, second) => Date.parse(`${day.date}T00:00:00.000Z`) + (second + 1) * SECOND_MS - itemAt(day, venue).observedAt[second]!));
    for (const lookback of LOOKBACKS.filter((item) => item <= 300)) {
      candidates.push(candidate(`${venue}-return-${lookback}s`, `${venue} trailing return`, "cross-exchange-lead-lag", `${lookback}s`, (day, second) => windowReturn(day.quote.get(venue)!, second, lookback)));
      candidates.push(candidate(`${venue}-vol-${lookback}s`, `${venue} realized volatility`, "cross-exchange-volatility", `${lookback}s`, (day, second) => windowVolatility(day.quote.get(venue)!, second, lookback)));
    }
  }
  for (const venue of VENUES.slice(1)) for (const lookback of LOOKBACKS.filter((item) => item <= 300)) {
    candidates.push(candidate(`${venue}-lead-${lookback}s`, `${venue} minus Binance trailing return`, "cross-exchange-lead-lag", `${lookback}s`, (day, second) => windowReturn(day.quote.get(venue)!, second, lookback) - windowReturn(day.quote.get(VENUES[0])!, second, lookback)));
  }
  candidates.push(candidate("cross-venue-mid-dispersion", "Cross-venue mid dispersion", "cross-exchange-books", "latest", (day, second) => standardDeviation(VENUES.map((venue) => day.quote.get(venue)!.logMid[second]!))));
  candidates.push(candidate("best-cross-venue-spread", "Best executable cross-venue spread", "cross-exchange-books", "latest", (day, second) => {
    const bestBid = Math.max(...VENUES.map((venue) => day.quote.get(venue)!.bid[second]!));
    const bestAsk = Math.min(...VENUES.map((venue) => day.quote.get(venue)!.ask[second]!));
    const mid = Math.exp(day.quote.get(VENUES[0])!.logMid[second]!);
    return 10_000 * (bestAsk - bestBid) / mid;
  }));
  for (const lookback of LOOKBACKS) {
    candidates.push(candidate(`liquidation-signed-${lookback}s`, "Signed BTC liquidation notional", "liquidations", `${lookback}s`, (day, second) => signedLog(windowPrefix(day.signedLiquidationPrefix, second, lookback))));
    candidates.push(candidate(`liquidation-absolute-${lookback}s`, "BTC liquidation notional", "liquidations", `${lookback}s`, (day, second) => Math.log1p(windowPrefix(day.absoluteLiquidationPrefix, second, lookback))));
    candidates.push(candidate(`liquidation-count-${lookback}s`, "BTC liquidation count", "liquidations", `${lookback}s`, (day, second) => Math.log1p(windowPrefix(day.liquidationCountPrefix, second, lookback))));
  }
  return candidates;
}

function loadDays(input: string): DayData[] {
  const dates = fs.readdirSync(path.join(input, VENUES[0])).filter((name) => name.endsWith(".json.gz")).map((name) => name.slice(0, -8)).sort();
  return dates.filter((date) => VENUES.every((venue) => fs.existsSync(path.join(input, venue, `${date}.json.gz`))))
    .map((date) => {
      const quote = new Map<string, VenueDay>();
      for (const venue of VENUES) quote.set(venue, buildVenue(readGzip<QuoteArtifact>(path.join(input, venue, `${date}.json.gz`)).rows));
      const liquidationEvents = ["binance-usdm-liquidations", "deribit-liquidations"]
        .flatMap((source) => fs.existsSync(path.join(input, source, `${date}.json.gz`)) ? readGzip<LiquidationArtifact>(path.join(input, source, `${date}.json.gz`)).rows : []);
      const signed = new Float64Array(SECONDS_PER_DAY);
      const absolute = new Float64Array(SECONDS_PER_DAY);
      const count = new Float64Array(SECONDS_PER_DAY);
      const largest = new Float64Array(SECONDS_PER_DAY);
      const dayStart = Date.parse(`${date}T00:00:00.000Z`);
      for (const [observedAt, side, price, amount] of liquidationEvents) {
        const second = Math.floor((observedAt - dayStart) / SECOND_MS);
        if (second < 0 || second >= SECONDS_PER_DAY) continue;
        const notional = price * amount;
        signed[second] += side * notional;
        absolute[second] += notional;
        count[second] += 1;
        largest[second] = Math.max(largest[second]!, notional);
      }
      return {
        date,
        split: splitForDate(date),
        quote,
        signedLiquidationPrefix: prefix(signed),
        absoluteLiquidationPrefix: prefix(absolute),
        liquidationCountPrefix: prefix(count),
        liquidationMax: largest,
      };
    });
}

function buildVenue(rows: QuoteSecond[]): VenueDay {
  const bid = new Float64Array(SECONDS_PER_DAY).fill(Number.NaN);
  const ask = new Float64Array(SECONDS_PER_DAY).fill(Number.NaN);
  const bidAmount = new Float64Array(SECONDS_PER_DAY).fill(Number.NaN);
  const askAmount = new Float64Array(SECONDS_PER_DAY).fill(Number.NaN);
  const observedAt = new Float64Array(SECONDS_PER_DAY).fill(Number.NaN);
  const dayStart = rows[0]?.[0] ?? 0;
  const midnight = Math.floor(dayStart / 86_400_000) * 86_400_000;
  for (const [bucket, observed, bidPrice, askPrice, bidQuantity, askQuantity] of rows) {
    const second = Math.floor((bucket - midnight) / SECOND_MS);
    if (second < 0 || second >= SECONDS_PER_DAY) continue;
    bid[second] = bidPrice; ask[second] = askPrice; bidAmount[second] = bidQuantity; askAmount[second] = askQuantity; observedAt[second] = observed;
  }
  for (let second = 1; second < SECONDS_PER_DAY; second += 1) if (!Number.isFinite(bid[second])) {
    bid[second] = bid[second - 1]!; ask[second] = ask[second - 1]!; bidAmount[second] = bidAmount[second - 1]!; askAmount[second] = askAmount[second - 1]!; observedAt[second] = observedAt[second - 1]!;
  }
  const logMid = Float64Array.from(bid, (value, index) => value > 0 && ask[index]! > 0 ? Math.log((value + ask[index]!) / 2) : Number.NaN);
  const squares = new Float64Array(SECONDS_PER_DAY);
  for (let second = 1; second < SECONDS_PER_DAY; second += 1) {
    const change = logMid[second]! - logMid[second - 1]!;
    squares[second] = Number.isFinite(change) ? change * change : Number.NaN;
  }
  return { logMid, bid, ask, bidAmount, askAmount, observedAt, squareReturnPrefix: prefix(squares) };
}

function renderReport(artifact: ReturnType<typeof run>): string {
  const lines = [
    "# Tardis monthly-sample external information audit", "",
    `Generated ${artifact.generatedAt}. Free Tardis samples cover ${artifact.dates.length} independent UTC days.`, "",
    "## Outcome", "",
    "These results replace an unqualified `awaiting-data` label with sparse monthly-sample evidence. A pass here is still weaker than a continuous point-in-time archive because only the first UTC day of each month is public.", "",
    "| target | candidates | stable | best feature | lookback | primary bits | transfer bits | sign bits | magnitude bits |", "|---:|---:|---:|---|---:|---:|---:|---:|---:|",
  ];
  for (const group of artifact.groups) {
    const best = group.best;
    lines.push(`| ${formatHorizon(group.horizonSeconds)} | ${group.candidateCount} | ${group.stableCount} | ${best?.label ?? "none"} | ${best?.lookback ?? "—"} | ${fixed(best?.primaryBits)} | ${fixed(best?.transferBits)} | ${fixed(best?.signBits)} | ${fixed(best?.magnitudeBits)} |`);
  }
  lines.push("", "## Causal and validation constraints", "", ...Object.values(artifact.method).map((item) => `- ${item}.`), "", "Complete rankings and frozen quantile edges are in `data/benchmarks/tardis-monthly-sample-information.json`.", "");
  return lines.join("\n");
}

function candidate(id: string, label: string, family: string, lookback: string, value: Candidate["value"]): Candidate { return { id, label, family, lookback, value }; }
function itemAt(day: DayData, venue: string) { return day.quote.get(venue)!; }
function windowReturn(venue: VenueDay, end: number, seconds: number) { return end >= seconds ? venue.logMid[end]! - venue.logMid[end - seconds]! : Number.NaN; }
function windowVolatility(venue: VenueDay, end: number, seconds: number) { return end >= seconds ? Math.sqrt(Math.max(0, venue.squareReturnPrefix[end + 1]! - venue.squareReturnPrefix[end - seconds + 1]!)) : Number.NaN; }
function windowPrefix(values: Float64Array, end: number, seconds: number) { return end >= seconds ? values[end + 1]! - values[end - seconds + 1]! : Number.NaN; }
function prefix(values: Float64Array) { const output = new Float64Array(values.length + 1); for (let index = 0; index < values.length; index += 1) output[index + 1] = output[index]! + (Number.isFinite(values[index]) ? values[index]! : 0); return output; }
function splitForDate(date: string) { return date <= "2025-07-01" ? 0 : date <= "2025-09-01" ? 1 : date <= "2025-11-01" ? 2 : 3; }
function readGzip<T>(file: string): T { return JSON.parse(gunzipSync(fs.readFileSync(file)).toString("utf8")) as T; }
function indicesWhere(values: number[], target: number) { const output: number[] = []; for (let index = 0; index < values.length; index += 1) if (values[index] === target) output.push(index); return output; }
function indicesWhereMany(values: number[], targets: number[]) { const set = new Set(targets); const output: number[] = []; for (let index = 0; index < values.length; index += 1) if (set.has(values[index]!)) output.push(index); return output; }
function quantileEdges(values: number[], bins: number) { const sorted = values.filter(Number.isFinite).sort((a, b) => a - b); if (sorted.length === 0) return Array(bins - 1).fill(0); return Array.from({ length: bins - 1 }, (_, index) => sorted[Math.min(sorted.length - 1, Math.floor((index + 1) * sorted.length / bins))]!); }
function quantizeCandidate(candidate: Candidate, values: number[], training: number[]) {
  if (candidate.family !== "liquidations") {
    const edges = quantileEdges(training.map((index) => values[index]!).filter(Number.isFinite), FEATURE_BINS);
    return { edges, bins: Uint8Array.from(values.map((item) => Number.isFinite(item) ? binValue(item, edges) : MISSING_BIN)) };
  }
  const signed = candidate.id.includes("signed");
  const positive = training.map((index) => values[index]!).filter((item) => Number.isFinite(item) && item > 0).sort((a, b) => a - b);
  const cut1 = positive[Math.floor(positive.length / 3)] ?? 0;
  const cut2 = positive[Math.floor(2 * positive.length / 3)] ?? cut1;
  const bins = Uint8Array.from(values.map((item) => {
    if (!Number.isFinite(item)) return MISSING_BIN;
    if (signed) return item < 0 ? 0 : item === 0 ? 1 : item <= (positive[Math.floor(positive.length / 2)] ?? 0) ? 2 : 3;
    return item === 0 ? 0 : item <= cut1 ? 1 : item <= cut2 ? 2 : 3;
  }));
  return { edges: signed ? ["negative", "zero", positive[Math.floor(positive.length / 2)] ?? 0] : [0, cut1, cut2], bins };
}
function binValue(value: number, edges: number[]) { let bin = 0; while (bin < edges.length && value > edges[bin]!) bin += 1; return bin; }
function standardDeviation(values: number[]) { if (!values.every(Number.isFinite)) return Number.NaN; const mean = values.reduce((sum, item) => sum + item, 0) / values.length; return Math.sqrt(values.reduce((sum, item) => sum + (item - mean) ** 2, 0) / values.length); }
function signedLog(value: number) { return Math.sign(value) * Math.log1p(Math.abs(value)); }
function fixed(value: number | undefined) { return value === undefined || !Number.isFinite(value) ? "—" : value.toFixed(6); }
function formatHorizon(seconds: number) { return seconds < 60 ? `${seconds}s` : seconds < 3_600 ? `${seconds / 60}m` : "1h"; }
function resolve(value: string) { return path.resolve(repoRoot, value); }

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try { run(); } catch (error) { console.error(error instanceof Error ? error.stack ?? error.message : error); process.exitCode = 1; }
}
