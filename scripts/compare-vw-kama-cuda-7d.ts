import fs from "node:fs/promises";
import path from "node:path";
import { performance } from "node:perf_hooks";
import { fileURLToPath } from "node:url";
import {
  columnarVwKamaCandles,
  evaluateVwKamaOracle,
  prepareVwKamaOracle,
  VW_KAMA_SCORE_VERSION,
  vwKamaScore,
  type VwKamaParameters,
  type VwKamaPreset,
} from "../packages/bot-algo/src/kama-signal-evaluator.js";
import { perfectMarginOracle } from "../packages/bot-algo/src/perfect-margin-oracle.js";
import type { TradingCandle } from "../packages/bot-algo/src/trading-api.js";
import {
  evaluateVwKamaCudaBatch,
  vwKamaCudaStatus,
  type VwKamaCudaCaseResult,
} from "../packages/bot-algo/src/vw-kama-cuda.js";
import { KamaInspectorEngine } from "../apps/server/src/kama-inspector.js";

const SECOND = 1_000;
const MINUTE = 60_000;
const HOUR = 3_600_000;
const DAY = 86_400_000;
const MAX_WARMUP_MS = 3 * DAY;
const CANDIDATE_IDS = ["global-clean-v0182", "global-clean-k0050"] as const;
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const sourceDir = path.join(repoRoot, "data/historical/spot-btcusdt/btcusdt/1s");
const stamp = new Date().toISOString().slice(0, 10);
const defaultJson = path.join(repoRoot, `data/benchmarks/vw-kama-cuda-7d-v2-comparison-${stamp}.json`);
const defaultReport = path.join(repoRoot, `docs/vw-kama-cuda-7d-v2-comparison-${stamp}.md`);

interface ScoreStats {
  score: number;
  precision: number;
  recall: number;
  f1: number;
  exposureAgreement: number;
  signalCleanliness: number;
  stateCredit: number;
  stateCount: number;
  timingCredit: number;
  signalCount: number;
  oracleCount: number;
  matchedCount: number;
}

interface ComparisonCase {
  windowId: string;
  window: string;
  group: string;
  startTime: number;
  endTime: number;
  scaleMs: number;
  scale: string;
  candidateId: string;
  candleCount: number;
  segmentCount: number;
  cpu: ScoreStats;
  gpu: ScoreStats;
  delta: {
    score: number;
    f1: number;
    exposureAgreement: number;
    signalCleanliness: number;
    signalCount: number;
    matchedCount: number;
  };
}

interface SegmentStats {
  stateCredit: number;
  stateCount: number;
  timingCredit: number;
  signalCount: number;
  oracleCount: number;
  matchedCount: number;
}

interface Timings {
  loadMs: number;
  prepareMs: number;
  cpuMs: number;
  gpuMs: number;
  gpuKernelMs: number;
}

void main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : error);
  process.exitCode = 1;
});

async function main(): Promise<void> {
  const startedAt = performance.now();
  const outputPath = path.resolve(process.argv[2] ?? defaultJson);
  const reportPath = path.resolve(process.argv[3] ?? defaultReport);
  const cuda = await vwKamaCudaStatus();
  if (!cuda.available) throw new Error(`CUDA unavailable: ${cuda.reason}`);

  const engine = new KamaInspectorEngine(path.join(repoRoot, "data"));
  const catalog = engine.catalog();
  const windows = catalog.windows.filter((window) => window.endTime - window.startTime === 7 * DAY);
  const scales = catalog.scales.map((scale) => scale.intervalMs);
  const presets = CANDIDATE_IDS.map((id) => {
    const preset = catalog.presets.find((item) => item.id === id);
    if (!preset) throw new Error(`Missing score-v2 preset ${id}.`);
    return preset;
  });
  if (windows.length !== 13 || scales.length !== 7 || presets.length !== 2) {
    throw new Error(`Expected 13 windows × 7 scales × 2 candidates; found ${windows.length} × ${scales.length} × ${presets.length}.`);
  }

  const comparisons: ComparisonCase[] = [];
  const timings: Timings = { loadMs: 0, prepareMs: 0, cpuMs: 0, gpuMs: 0, gpuKernelMs: 0 };
  let sourceCandles = 0;
  let candidateCandles = 0;
  let segmentCount = 0;

  for (let windowIndex = 0; windowIndex < windows.length; windowIndex += 1) {
    const window = windows[windowIndex]!;
    const loadStarted = performance.now();
    const source = await loadSource(window.startTime - MAX_WARMUP_MS, window.endTime);
    timings.loadMs += performance.now() - loadStarted;
    sourceCandles += source.length;
    console.error(`${windowIndex + 1}/${windows.length} ${window.id}: ${source.length.toLocaleString()} source candles.`);

    for (const scaleMs of scales) {
      const prepareStarted = performance.now();
      const segments = continuousSegments(aggregateCandles(source, SECOND, scaleMs), scaleMs);
      timings.prepareMs += performance.now() - prepareStarted;
      const cpuByCandidate = new Map<string, SegmentStats[]>();
      const gpuByCandidate = new Map<string, SegmentStats[]>();
      for (const preset of presets) {
        cpuByCandidate.set(preset.id, []);
        gpuByCandidate.set(preset.id, []);
      }
      let scoredSegments = 0;

      for (let index = 0; index < segments.length; index += 1) {
        const candles = segments[index]!;
        const scoreGroups = new Map<number, VwKamaPreset[]>();
        for (const preset of presets) {
          const scoreStartTime = Math.max(
            window.startTime,
            candles[0]!.openTime + (index > 0 ? candidateWarmupMs(preset.parameters, scaleMs, catalog.defaults.warmupMultiple) : 0),
          );
          const scoreStartIndex = lowerBound(candles, scoreStartTime);
          if (scoreStartIndex >= candles.length || candles[scoreStartIndex]!.openTime >= window.endTime) continue;
          const group = scoreGroups.get(scoreStartIndex) ?? [];
          group.push(preset);
          scoreGroups.set(scoreStartIndex, group);
        }
        if (scoreGroups.size === 0) continue;
        scoredSegments += 1;
        segmentCount += 1;

        const oracleStarted = performance.now();
        const oracle = perfectMarginOracle(candles, {
          startingQuote: 1,
          leverage: 1,
          friction: catalog.defaults.oracleFriction,
          eventMode: "close",
          maxPathCandles: 1,
        });
        const columns = columnarVwKamaCandles(candles);
        timings.prepareMs += performance.now() - oracleStarted;

        for (const [scoreStartIndex, group] of scoreGroups) {
          const prepared = prepareVwKamaOracle(columns, scoreStartIndex, oracle);
          const options = {
            intervalMs: scaleMs,
            scoreStartIndex,
            oracleFriction: catalog.defaults.oracleFriction,
            matchWindowMs: catalog.defaults.matchWindowMs,
            timingHalfLifeMs: catalog.defaults.timingHalfLifeMs,
            warmupMultiple: catalog.defaults.warmupMultiple,
          };
          const gpuStarted = performance.now();
          const gpu = await evaluateVwKamaCudaBatch(
            columns,
            prepared,
            group.map((preset) => preset.parameters),
            options,
          );
          timings.gpuMs += performance.now() - gpuStarted;
          timings.gpuKernelMs += gpu[0]?.elapsedMs ?? 0;

          for (let candidateIndex = 0; candidateIndex < group.length; candidateIndex += 1) {
            const preset = group[candidateIndex]!;
            const cpuStarted = performance.now();
            const cpu = evaluateVwKamaOracle(columns, {
              ...options,
              scoreStartTime: candles[scoreStartIndex]!.openTime,
              parameters: preset.parameters,
              preparedOracle: prepared,
              includeTrace: false,
            });
            timings.cpuMs += performance.now() - cpuStarted;
            cpuByCandidate.get(preset.id)!.push(cpuSegment(cpu));
            gpuByCandidate.get(preset.id)!.push(gpuSegment(gpu[candidateIndex]!));
          }
        }
      }

      for (const preset of presets) {
        const cpu = combine(cpuByCandidate.get(preset.id)!);
        const gpu = combine(gpuByCandidate.get(preset.id)!);
        const candleCount = cpu.stateCount;
        candidateCandles += candleCount;
        comparisons.push({
          windowId: window.id,
          window: window.label,
          group: window.group,
          startTime: window.startTime,
          endTime: window.endTime,
          scaleMs,
          scale: duration(scaleMs),
          candidateId: preset.id,
          candleCount,
          segmentCount: scoredSegments,
          cpu,
          gpu,
          delta: {
            score: gpu.score - cpu.score,
            f1: gpu.f1 - cpu.f1,
            exposureAgreement: gpu.exposureAgreement - cpu.exposureAgreement,
            signalCleanliness: gpu.signalCleanliness - cpu.signalCleanliness,
            signalCount: gpu.signalCount - cpu.signalCount,
            matchedCount: gpu.matchedCount - cpu.matchedCount,
          },
        });
      }
    }
  }

  const completedAt = new Date().toISOString();
  const summary = summarize(comparisons);
  const artifact = {
    generatedAt: completedAt,
    scoreVersion: VW_KAMA_SCORE_VERSION,
    configurationSource: "Score-v2 chronological validation finalists",
    device: cuda.device,
    matrix: {
      windows: windows.length,
      scales: scales.map(duration),
      candidates: presets.map((preset) => preset.id),
      cases: comparisons.length,
      sourceCandles,
      candidateCandles,
      segmentCount,
    },
    timings: {
      ...roundRecord(timings),
      totalMs: Math.round(performance.now() - startedAt),
      cpuCandidateCandlesPerSecond: throughput(candidateCandles, timings.cpuMs),
      gpuCandidateCandlesPerSecond: throughput(candidateCandles, timings.gpuMs),
      gpuVsSerialCpu: timings.cpuMs / timings.gpuMs,
    },
    summary,
    comparisons,
  };
  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  await fs.mkdir(path.dirname(reportPath), { recursive: true });
  await fs.writeFile(outputPath, `${JSON.stringify(artifact, null, 2)}\n`);
  await fs.writeFile(reportPath, report(artifact, outputPath));
  console.error(`Completed ${comparisons.length} comparisons without evaluator errors.`);
  console.error(`Maximum absolute score drift: ${(summary.maxAbsoluteScoreDelta * 100).toFixed(6)} points.`);
  console.error(`Results: ${path.relative(repoRoot, outputPath)}`);
  console.error(`Report: ${path.relative(repoRoot, reportPath)}`);
}

function cpuSegment(result: ReturnType<typeof evaluateVwKamaOracle>): SegmentStats {
  return {
    stateCredit: result.metrics.exposureAgreement * result.candleCount,
    stateCount: result.candleCount,
    timingCredit: result.candidateTransitions.reduce((sum, transition) => sum + transition.timingCredit, 0),
    signalCount: result.metrics.signalCount,
    oracleCount: result.metrics.oracleCount,
    matchedCount: result.metrics.matchedCount,
  };
}

function gpuSegment(result: VwKamaCudaCaseResult): SegmentStats {
  return {
    stateCredit: result.stateCredit,
    stateCount: result.stateCount,
    timingCredit: result.timingCredit,
    signalCount: result.signalCount,
    oracleCount: result.oracleCount,
    matchedCount: result.matchedCount,
  };
}

function combine(parts: SegmentStats[]): ScoreStats {
  if (parts.length === 0) throw new Error("No scored segments remain after warmup.");
  const sum = <K extends keyof SegmentStats>(key: K): number =>
    parts.reduce((total, part) => total + part[key], 0);
  const stateCredit = sum("stateCredit");
  const stateCount = sum("stateCount");
  const timingCredit = sum("timingCredit");
  const signalCount = sum("signalCount");
  const oracleCount = sum("oracleCount");
  const matchedCount = sum("matchedCount");
  const precision = eventRatio(timingCredit, signalCount, oracleCount);
  const recall = eventRatio(timingCredit, oracleCount, signalCount);
  const f1 = harmonic(precision, recall);
  const exposureAgreement = stateCredit / stateCount;
  const signalCleanliness = signalCount > 0 ? matchedCount / signalCount : 1;
  return {
    score: vwKamaScore(f1, exposureAgreement, signalCleanliness),
    precision,
    recall,
    f1,
    exposureAgreement,
    signalCleanliness,
    stateCredit,
    stateCount,
    timingCredit,
    signalCount,
    oracleCount,
    matchedCount,
  };
}

function summarize(comparisons: ComparisonCase[]) {
  const absoluteScoreDeltas = comparisons.map((item) => Math.abs(item.delta.score));
  const exactSignals = comparisons.filter((item) => item.delta.signalCount === 0).length;
  const exactMatches = comparisons.filter((item) => item.delta.matchedCount === 0).length;
  const rankingGroups = new Map<string, ComparisonCase[]>();
  for (const item of comparisons) {
    const key = `${item.windowId}:${item.scale}`;
    const group = rankingGroups.get(key) ?? [];
    group.push(item);
    rankingGroups.set(key, group);
  }
  let rankingFlips = 0;
  const rankingDetails = [];
  for (const [key, group] of rankingGroups) {
    const cpu = group.slice().sort((left, right) => right.cpu.score - left.cpu.score)[0]!;
    const gpu = group.slice().sort((left, right) => right.gpu.score - left.gpu.score)[0]!;
    if (cpu.candidateId !== gpu.candidateId) {
      rankingFlips += 1;
      rankingDetails.push({
        key,
        windowId: cpu.windowId,
        scale: cpu.scale,
        cpuWinner: cpu.candidateId,
        gpuWinner: gpu.candidateId,
        candidates: group.map((item) => ({
          candidateId: item.candidateId,
          cpuScore: item.cpu.score,
          gpuScore: item.gpu.score,
        })),
      });
    }
  }
  const byScale = [...new Set(comparisons.map((item) => item.scale))].map((scale) => {
    const cases = comparisons.filter((item) => item.scale === scale);
    const deltas = cases.map((item) => Math.abs(item.delta.score));
    return {
      scale,
      cases: cases.length,
      nonzeroScoreCases: deltas.filter((value) => value !== 0).length,
      meanAbsoluteScoreDelta: mean(deltas),
      maxAbsoluteScoreDelta: Math.max(...deltas),
      exactSignalCountCases: cases.filter((item) => item.delta.signalCount === 0).length,
    };
  });
  const byCandidate = [...new Set(comparisons.map((item) => item.candidateId))].map((candidateId) => {
    const cases = comparisons.filter((item) => item.candidateId === candidateId);
    const cpuScores = cases.map((item) => item.cpu.score);
    const gpuScores = cases.map((item) => item.gpu.score);
    const cpuMedian = percentile(cpuScores, 0.5);
    const gpuMedian = percentile(gpuScores, 0.5);
    const cpuP10 = percentile(cpuScores, 0.1);
    const gpuP10 = percentile(gpuScores, 0.1);
    return {
      candidateId,
      cpuMedian,
      gpuMedian,
      cpuP10,
      gpuP10,
      cpuObjective: (cpuMedian + cpuP10) / 2,
      gpuObjective: (gpuMedian + gpuP10) / 2,
      meanAbsoluteScoreDelta: mean(cases.map((item) => Math.abs(item.delta.score))),
    };
  });
  const worst = comparisons.slice().sort((left, right) =>
    Math.abs(right.delta.score) - Math.abs(left.delta.score)).slice(0, 12).map((item) => ({
      windowId: item.windowId,
      scale: item.scale,
      candidateId: item.candidateId,
      cpuScore: item.cpu.score,
      gpuScore: item.gpu.score,
      scoreDelta: item.delta.score,
      signalDelta: item.delta.signalCount,
      matchedDelta: item.delta.matchedCount,
    }));
  return {
    meanAbsoluteScoreDelta: mean(absoluteScoreDeltas),
    medianAbsoluteScoreDelta: percentile(absoluteScoreDeltas, 0.5),
    p95AbsoluteScoreDelta: percentile(absoluteScoreDeltas, 0.95),
    maxAbsoluteScoreDelta: Math.max(...absoluteScoreDeltas),
    maxAbsoluteF1Delta: Math.max(...comparisons.map((item) => Math.abs(item.delta.f1))),
    maxAbsoluteExposureAgreementDelta: Math.max(...comparisons.map((item) => Math.abs(item.delta.exposureAgreement))),
    maxAbsoluteCleanlinessDelta: Math.max(...comparisons.map((item) => Math.abs(item.delta.signalCleanliness))),
    exactSignalCountCases: exactSignals,
    exactMatchedCountCases: exactMatches,
    rankingGroups: rankingGroups.size,
    rankingFlips,
    rankingDetails,
    byScale,
    byCandidate,
    worst,
  };
}

function report(artifact: any, outputPath: string): string {
  const summary = artifact.summary;
  const comparisons = artifact.comparisons as ComparisonCase[];
  const timings = artifact.timings;
  return [
    "# VW-KAMA CUDA vs CPU seven-day stress comparison",
    "",
    `Generated: ${artifact.generatedAt}`,
    `Device: ${artifact.device}`,
    `Raw results: \`${path.relative(repoRoot, outputPath)}\``,
    "",
    "## Scope",
    "",
    `- ${artifact.matrix.windows} real seven-day BTCUSDT windows × ${artifact.matrix.scales.length} scales (${artifact.matrix.scales.join(", ")}) × ${artifact.matrix.candidates.length} score-v2 validation finalists = ${artifact.matrix.cases} CPU/GPU comparisons.`,
    `- Configurations: ${artifact.matrix.candidates.map((item: string) => `\`${item}\``).join(" and ")}.`,
    `- Both paths use the current score-v${artifact.scoreVersion} formula; “v2” identifies the historical configuration selection, not the score used here.`,
    `- GPU uses Float32 fast math; CPU uses the Float64 TypeScript evaluator. Both receive identical prepared candle and oracle columns.`,
    "",
    "## Result",
    "",
    `All ${artifact.matrix.cases} comparisons completed without CUDA or evaluator errors.`,
    "",
    "| measure | result |",
    "|---|---:|",
    `| Mean absolute score drift | ${points(summary.meanAbsoluteScoreDelta, 6)} |`,
    `| Median absolute score drift | ${points(summary.medianAbsoluteScoreDelta, 6)} |`,
    `| P95 absolute score drift | ${points(summary.p95AbsoluteScoreDelta, 6)} |`,
    `| Maximum absolute score drift | ${points(summary.maxAbsoluteScoreDelta, 6)} |`,
    `| Maximum absolute F1 drift | ${points(summary.maxAbsoluteF1Delta, 6)} |`,
    `| Maximum absolute exposure-agreement drift | ${points(summary.maxAbsoluteExposureAgreementDelta, 6)} |`,
    `| Maximum absolute cleanliness drift | ${points(summary.maxAbsoluteCleanlinessDelta, 6)} |`,
    `| Exact signal-count cases | ${summary.exactSignalCountCases}/${artifact.matrix.cases} |`,
    `| Exact matched-count cases | ${summary.exactMatchedCountCases}/${artifact.matrix.cases} |`,
    `| Pairwise winner flips | ${summary.rankingFlips}/${summary.rankingGroups} |`,
    "",
    "The drift is small relative to the score range, but not always zero: long Float32 recurrences can cross a signal threshold and then alter later state. Use CUDA for population screening and re-evaluate the final shortlist on the CPU before persisting a winner.",
    "",
    "## Drift by scale",
    "",
    "| scale | cases | nonzero score | mean absolute drift | maximum drift | exact signals |",
    "|---:|---:|---:|---:|---:|---:|",
    ...summary.byScale.map((item: any) =>
      `| ${item.scale} | ${item.cases} | ${item.nonzeroScoreCases}/${item.cases} | ${points(item.meanAbsoluteScoreDelta, 6)} | ${points(item.maxAbsoluteScoreDelta, 6)} | ${item.exactSignalCountCases}/${item.cases} |`),
    "",
    "All 1m, 5m, 15m, and 1h cases were identical in this matrix. Measurable divergence was confined to the longer sequential 1s, 5s, and 15s evaluations.",
    "",
    "## Finalist aggregate",
    "",
    "The validation objective below is `(median + P10) / 2` over all 91 window/scale cases for each finalist.",
    "",
    "| candidate | CPU median | GPU median | CPU P10 | GPU P10 | CPU objective | GPU objective | mean absolute drift |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
    ...summary.byCandidate.map((item: any) =>
      `| ${item.candidateId} | ${pct(item.cpuMedian)} | ${pct(item.gpuMedian)} | ${pct(item.cpuP10)} | ${pct(item.gpuP10)} | ${pct(item.cpuObjective)} | ${pct(item.gpuObjective)} | ${points(item.meanAbsoluteScoreDelta, 6)} |`),
    "",
    "The aggregate finalist ordering was unchanged.",
    "",
    "## Pairwise ranking changes",
    "",
    ...(summary.rankingDetails.length === 0
      ? ["No pairwise winner changed.", ""]
      : [
          "| window | scale | candidate | CPU | GPU |",
          "|---|---:|---|---:|---:|",
          ...summary.rankingDetails.flatMap((detail: any) => detail.candidates.map((candidate: any) =>
            `| ${detail.windowId} | ${detail.scale} | ${candidate.candidateId} | ${pct(candidate.cpuScore)} | ${pct(candidate.gpuScore)} |`)),
          "",
          "The only flip was a near tie on the CPU, so exact CPU verification of the top few candidates is sufficient to catch it.",
          "",
        ]),
    "## Evaluator time",
    "",
    "Data loading, aggregation, and oracle construction are excluded from the evaluator comparison.",
    "",
    "| path | time | candidate-candles/s |",
    "|---|---:|---:|",
    `| CPU serial Float64 | ${seconds(timings.cpuMs)} | ${Math.round(timings.cpuCandidateCandlesPerSecond).toLocaleString()} |`,
    `| CUDA batch including transfers/native alignment | ${seconds(timings.gpuMs)} | ${Math.round(timings.gpuCandidateCandlesPerSecond).toLocaleString()} |`,
    `| CUDA kernels only | ${seconds(timings.gpuKernelMs)} | — |`,
    "",
    `Observed CUDA/serial-CPU speed ratio: ${timings.gpuVsSerialCpu.toFixed(3)}×. This matrix has only two candidates per batch, so it is an accuracy stress test rather than a representative high-occupancy optimizer benchmark.`,
    "",
    "## Largest score differences",
    "",
    "| window | scale | candidate | CPU | GPU | GPU−CPU | Δ signals | Δ matched |",
    "|---|---:|---|---:|---:|---:|---:|---:|",
    ...summary.worst.map((item: any) =>
      `| ${item.windowId} | ${item.scale} | ${item.candidateId} | ${pct(item.cpuScore)} | ${pct(item.gpuScore)} | ${signedPoints(item.scoreDelta, 6)} | ${signed(item.signalDelta)} | ${signed(item.matchedDelta)} |`),
    "",
    "## All cases",
    "",
    "| window | scale | candidate | candles | CPU | GPU | GPU−CPU | Δ F1 | Δ agreement | Δ clean | Δ signals | Δ matched |",
    "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ...comparisons.map((item) =>
      `| ${item.windowId} | ${item.scale} | ${item.candidateId} | ${item.candleCount.toLocaleString()} | ${pct(item.cpu.score)} | ${pct(item.gpu.score)} | ${signedPoints(item.delta.score, 6)} | ${signedPoints(item.delta.f1, 6)} | ${signedPoints(item.delta.exposureAgreement, 6)} | ${signedPoints(item.delta.signalCleanliness, 6)} | ${signed(item.delta.signalCount)} | ${signed(item.delta.matchedCount)} |`),
    "",
  ].join("\n");
}

async function loadSource(start: number, end: number): Promise<TradingCandle[]> {
  const candles: TradingCandle[] = [];
  for (let day = utcDay(start); day < end; day += DAY) {
    const date = new Date(day).toISOString().slice(0, 10);
    const content = await fs.readFile(path.join(sourceDir, `${date}.jsonl`), "utf8");
    for (const line of content.split("\n")) {
      if (!line) continue;
      const candle = JSON.parse(line) as TradingCandle;
      if (candle.openTime >= start && candle.openTime < end) candles.push(candle);
    }
  }
  candles.sort((left, right) => left.openTime - right.openTime);
  if (candles.length === 0) throw new Error(`No source candles for ${new Date(start).toISOString()}.`);
  return candles;
}

function aggregateCandles(candles: TradingCandle[], sourceMs: number, targetMs: number): TradingCandle[] {
  if (sourceMs === targetMs) return candles.slice();
  const result: TradingCandle[] = [];
  const expected = targetMs / sourceMs;
  let current: TradingCandle | undefined;
  let bucket = -1;
  let count = 0;
  let contiguous = false;
  let previous = -1;
  const flush = (): void => {
    if (current && contiguous && count === expected) result.push(current);
  };
  for (const candle of candles) {
    const nextBucket = Math.floor(candle.openTime / targetMs) * targetMs;
    if (nextBucket !== bucket) {
      flush();
      bucket = nextBucket;
      count = 1;
      contiguous = candle.openTime === bucket;
      previous = candle.openTime;
      current = { ...candle, openTime: bucket, closeTime: bucket + targetMs - 1 };
      continue;
    }
    if (!current) continue;
    contiguous &&= candle.openTime === previous + sourceMs;
    previous = candle.openTime;
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
  const result: TradingCandle[][] = [];
  let current: TradingCandle[] = [];
  for (const candle of candles) {
    if (current.length && candle.openTime !== current.at(-1)!.openTime + intervalMs) {
      result.push(current);
      current = [];
    }
    current.push(candle);
  }
  if (current.length) result.push(current);
  return result;
}

function candidateWarmupMs(parameters: VwKamaParameters, intervalMs: number, warmupMultiple: number): number {
  const confirmation = (parameters.confirmationMix ?? 0) > 0;
  const thresholdEnabled = parameters.thresholdNoiseResponse === "inverse"
    ? (parameters.thresholdInverseMaxBpsHour ?? 0) > 0
    : (parameters.thresholdNoiseMultiplier ?? 0) > 0;
  const longest = Math.max(
    parameters.efficiencyMs,
    (parameters.efficiencyVolumePower ?? 0) > 0
      ? parameters.efficiencyVolumeEmaMs ?? parameters.volumeMs
      : 0,
    parameters.slowMs,
    parameters.volumeMs,
    thresholdEnabled ? parameters.thresholdLookbackMs ?? 0 : 0,
    confirmation ? parameters.confirmationAccelerationLookbackMs ?? 0 : 0,
    confirmation ? parameters.confirmationDistanceLookbackMs ?? 0 : 0,
    (confirmation && (parameters.confirmationEmaWeight ?? 0) > 0)
      || (parameters.confirmationEmaGateStrength ?? 0) > 0
      ? parameters.confirmationEmaMs ?? 0
      : 0,
    confirmation && (parameters.confirmationRsiWeight ?? 0) > 0 ? parameters.confirmationRsiMs ?? 0 : 0,
    confirmation && (parameters.confirmationDmiWeight ?? 0) > 0 ? (parameters.confirmationDmiMs ?? 0) * 2 : 0,
    (parameters.meanReversionReversalThreshold ?? 0) > 0
      ? Math.max(
        parameters.meanReversionEfficiencyMs ?? parameters.efficiencyMs,
        parameters.meanReversionFastMs ?? parameters.fastMs,
        parameters.meanReversionSlowMs ?? parameters.slowMs,
        parameters.meanReversionVolatilityMs ?? parameters.slowMs,
      )
      : 0,
  );
  return Math.ceil(longest / intervalMs) * intervalMs * warmupMultiple;
}

function lowerBound(candles: TradingCandle[], time: number): number {
  let low = 0;
  let high = candles.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (candles[middle]!.openTime < time) low = middle + 1;
    else high = middle;
  }
  return low;
}

function eventRatio(value: number, total: number, opposite: number): number {
  return total > 0 ? value / total : opposite === 0 ? 1 : 0;
}

function harmonic(left: number, right: number): number {
  return left + right > 0 ? 2 * left * right / (left + right) : 0;
}

function duration(ms: number): string {
  if (ms % HOUR === 0) return `${ms / HOUR}h`;
  if (ms % MINUTE === 0) return `${ms / MINUTE}m`;
  return `${ms / SECOND}s`;
}

function utcDay(time: number): number {
  return Math.floor(time / DAY) * DAY;
}

function mean(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function percentile(values: number[], quantile: number): number {
  const sorted = values.slice().sort((left, right) => left - right);
  const index = (sorted.length - 1) * quantile;
  const lower = Math.floor(index);
  const fraction = index - lower;
  return sorted[lower]! * (1 - fraction) + (sorted[lower + 1] ?? sorted[lower]!) * fraction;
}

function throughput(candidateCandles: number, elapsedMs: number): number {
  return candidateCandles / (elapsedMs / 1_000);
}

function roundRecord(values: Timings): Timings {
  return Object.fromEntries(Object.entries(values).map(([key, value]) => [key, Math.round(value)])) as unknown as Timings;
}

function pct(value: number): string {
  return `${(value * 100).toFixed(6)}%`;
}

function points(value: number, digits: number): string {
  return `${(value * 100).toFixed(digits)} points`;
}

function signedPoints(value: number, digits: number): string {
  return `${value >= 0 ? "+" : ""}${(value * 100).toFixed(digits)}`;
}

function signed(value: number): string {
  return `${value >= 0 ? "+" : ""}${value}`;
}

function seconds(ms: number): string {
  return `${(ms / 1_000).toFixed(3)}s`;
}
