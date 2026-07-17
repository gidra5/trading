import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  VW_KAMA_SCORE_VERSION,
  vwKamaScore,
  type VwKamaParameters,
  type VwKamaPreset,
} from "../packages/bot-algo/src/index.js";

type Stage = "fit" | "validation" | "test";
type Family = "volume" | "canonical";

interface Window { label: string; start: number; end: number }
interface CandidateEntry {
  type: "candidate";
  stage: Stage;
  candidate: Record<string, unknown> & { id: string; family: Family };
  aggregate: Record<string, number | null>;
  cases?: Array<Record<string, number | string | null>>;
}

const aggregateMetrics = [
  "precision",
  "recall",
  "f1",
  "exposureAgreement",
  "signalCleanliness",
  "signalsPerDay",
] as const;
const requiredCandidateFields = [
  "efficiency", "efficiencyVolumeEma", "efficiencyVolumePower", "fast", "slow", "power",
  "volume", "volumeCap", "volumePower", "deadbandBpsHour", "deadbandMode",
  "hysteresisReleaseRatio", "thresholdLookback", "thresholdNoiseMultiplier",
  "buyMaxFraction", "sellMaxFraction", "buySizingSigmaBpsHour", "sellSizingSigmaBpsHour",
  "agreementMode", "confirmationMix", "confirmationMinQuality",
  "confirmationAccelerationLookback", "confirmationDistanceLookback",
  "confirmationAccelerationWeight", "confirmationDistanceWeight", "confirmationBias",
  "confirmationEma", "confirmationEmaThresholdBpsHour", "confirmationEmaWeight",
  "confirmationEmaGateStrength", "confirmationRsi", "confirmationRsiThreshold",
  "confirmationRsiWeight", "confirmationDmi", "confirmationDmiWeight",
  "confirmationAdxThreshold", "meanReversionSuppressionThreshold", "meanReversionEfficiency",
  "meanReversionFast", "meanReversionSlow", "meanReversionVolatility",
  "meanReversionReversalThreshold",
] as const;

async function main(): Promise<void> {
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const input = path.resolve(root, process.argv[2]
  ?? "data/benchmarks/vw-kama-1s-mean-reversion-agreement60-de-2026-07-13.jsonl");
const presetOutput = path.resolve(root, process.argv[3]
  ?? "data/benchmarks/vw-kama-global-presets.json");
const entries = (await fs.readFile(input, "utf8")).trim().split("\n").map((line) => JSON.parse(line));
const meta = entries.find((entry) => entry.type === "meta");
const status = entries.findLast((entry) => entry.type === "status");
const selection = entries.findLast((entry) => entry.type === "selection");
assert(meta, "Missing meta record.");
assert(status?.status === "completed", "Run did not complete.");
assert(meta.scoreVersion === VW_KAMA_SCORE_VERSION, `Expected score v${VW_KAMA_SCORE_VERSION}.`);
assert(meta.sourceInterval === "1s", "Expected one-second source candles.");
assert(JSON.stringify(meta.scales) === '["1s"]', "Expected one-second evaluation only.");
assert(selection?.finalists?.length === 2, "Expected two validation-selected finalists.");
assert(meta.algorithm === "de" && Number.isFinite(meta.trials) && Number.isFinite(meta.generations),
  "Missing optimizer metadata.");
assert(Number.isFinite(meta.optimization?.restarts)
  && Number.isFinite(meta.optimization?.refinementRounds), "Missing DE optimization metadata.");
assert(meta.warmStart?.genomes > 0 && Array.isArray(meta.warmStart.files)
  && meta.warmStart.files.length > 0, "Global run was not warm-started.");
assert(meta.warmStart.quasiRandomGenomes === meta.trials
  && meta.warmStart.selectedPopulation === meta.trials, "Warm-start population metadata mismatch.");
const telemetry = meta.optimization.telemetry as Array<{
  restart: number;
  generation: number;
  fullFit: boolean;
  unique: number;
}>;
assert(Array.isArray(telemetry)
  && telemetry.length === meta.optimization.restarts * meta.generations,
"Incomplete differential-evolution telemetry.");
const telemetryKeys = new Set<string>();
for (const item of telemetry) {
  assert(item.restart >= 1 && item.restart <= meta.optimization.restarts
    && item.generation >= 1 && item.generation <= meta.generations,
  "Invalid differential-evolution telemetry coordinates.");
  telemetryKeys.add(`${item.restart}:${item.generation}`);
  assert(item.fullFit === (item.generation % meta.optimization.fullEvaluationInterval === 0),
    `Unexpected full-fit schedule at restart ${item.restart}, generation ${item.generation}.`);
  assert(item.unique > 0 && item.unique <= meta.trials, "Invalid telemetry population size.");
}
assert(telemetryKeys.size === telemetry.length, "Duplicate differential-evolution telemetry records.");

const windows = meta.windows as Record<Stage, Window[]>;
for (const stage of ["fit", "validation", "test"] as const) {
  assert(Array.isArray(windows[stage]) && windows[stage].length > 0, `Missing ${stage} windows.`);
  for (const window of windows[stage]) assert(window.start < window.end, `Invalid ${stage} window ${window.label}.`);
}
assert(Math.max(...windows.fit.map((window) => window.end))
  <= Math.min(...windows.validation.map((window) => window.start)), "Fit overlaps validation.");
assert(Math.max(...windows.validation.map((window) => window.end))
  <= Math.min(...windows.test.map((window) => window.start)), "Validation overlaps holdout.");

const candidates = entries.filter((entry): entry is CandidateEntry => entry.type === "candidate");
for (const entry of candidates) {
  assert(entry.candidate?.id && ["volume", "canonical"].includes(entry.candidate.family), "Invalid candidate.");
  if (entry.candidate.family === "canonical") {
    assert(entry.candidate.volumePower === 0 && entry.candidate.efficiencyVolumePower === 0,
      `Canonical candidate ${entry.candidate.id} has nonzero volume powers.`);
  }
  if (entry.stage !== "fit") {
    assert(entry.cases?.length === windows[entry.stage].length, `${entry.stage}/${entry.candidate.id} case count mismatch.`);
    const expected = new Set(windows[entry.stage].map((window) => window.label));
    for (const testCase of entry.cases ?? []) {
      assert(testCase.scale === "1s" && expected.delete(String(testCase.window)),
        `Unexpected ${entry.stage} case for ${entry.candidate.id}.`);
      const score = vwKamaScore(
        Number(testCase.f1),
        Number(testCase.exposureAgreement),
        Number(testCase.signalCleanliness),
      );
      assert(Math.abs(score - Number(testCase.score)) <= 1.5e-6,
        `Score formula mismatch for ${entry.candidate.id}/${testCase.window}.`);
      assertNear(Number(testCase.f1), harmonic(Number(testCase.precision), Number(testCase.recall)),
        `${entry.stage}/${entry.candidate.id}/${testCase.window} F1`);
      assertNear(Number(testCase.signalCleanliness), Number(testCase.rawPrecision),
        `${entry.stage}/${entry.candidate.id}/${testCase.window} cleanliness`);
      const noise = testCase.noiseSignalRatio;
      if (noise === null) {
        assertNear(Number(testCase.signalCleanliness), 0,
          `${entry.stage}/${entry.candidate.id}/${testCase.window} null-noise cleanliness`);
      } else {
        assertNear(Number(testCase.signalCleanliness), 1 / (1 + Number(noise)),
          `${entry.stage}/${entry.candidate.id}/${testCase.window} noise/cleanliness`);
      }
    }
    assert(expected.size === 0, `Missing ${entry.stage} cases for ${entry.candidate.id}.`);
    const cases = entry.cases ?? [];
    const scores = cases.map((item) => Number(item.score));
    const median = percentile(scores, 0.5);
    const p10 = percentile(scores, 0.1);
    assertNear(entry.aggregate.median, median, `${entry.stage}/${entry.candidate.id} median`);
    assertNear(entry.aggregate.p10, p10, `${entry.stage}/${entry.candidate.id} P10`);
    assertNear(entry.aggregate.objective, (median + p10) / 2,
      `${entry.stage}/${entry.candidate.id} objective`);
    for (const metric of aggregateMetrics) {
      assertNear(entry.aggregate[metric], percentile(cases.map((item) => Number(item[metric])), 0.5),
        `${entry.stage}/${entry.candidate.id} ${metric}`);
    }
    for (const metric of ["noiseSignalRatio", "lagP50Ms", "lagP90Ms"] as const) {
      assertNullableNear(entry.aggregate[metric], nullableMedian(cases.map((item) => item[metric])),
        `${entry.stage}/${entry.candidate.id} ${metric}`);
    }
  }
}

const finalistIds = new Set<string>();
for (const family of ["volume", "canonical"] as const) {
  const ranked = candidates.filter((entry) => entry.stage === "validation" && entry.candidate.family === family)
    .sort((left, right) => right.aggregate.objective - left.aggregate.objective
      || right.aggregate.p10 - left.aggregate.p10
      || left.candidate.id.localeCompare(right.candidate.id));
  const selected = selection.finalists.find((item: { family: Family }) => item.family === family);
  assert(ranked[0] && selected?.id === ranked[0].candidate.id, `${family} finalist is not validation-best.`);
  assert(Math.abs(selected.validationObjective - ranked[0].aggregate.objective) <= 1e-6,
    `${family} validation objective mismatch.`);
  finalistIds.add(selected.id);
}
const holdout = candidates.filter((entry) => entry.stage === "test");
assert(holdout.length === 2, "Holdout must contain only two finalists.");
assert(holdout.every((entry) => finalistIds.delete(entry.candidate.id)) && finalistIds.size === 0,
  "Holdout candidates do not match validation finalists.");

const presets: VwKamaPreset[] = selection.finalists.map((selected: {
  family: Family;
  id: string;
  validationObjective: number;
}) => {
  const entry = candidates.find((candidate) =>
    candidate.stage === "validation" && candidate.candidate.id === selected.id)!;
  for (const field of requiredCandidateFields) {
    assert(field in entry.candidate, `Finalist ${selected.id} is missing ${field}.`);
  }
  return {
    id: `global-score-v${VW_KAMA_SCORE_VERSION}-${selected.family}`,
    label: `Global · score-v${VW_KAMA_SCORE_VERSION} ${selected.family} finalist`,
    scope: "global",
    windowId: null,
    intervalMs: 1_000,
    parameters: storedParameters(entry.candidate),
    score: selected.validationObjective,
    scoreVersion: VW_KAMA_SCORE_VERSION,
    source: "Chronological validation-selected global finalist; holdout was not used for selection",
    generatedAt: meta.generatedAt,
    optimization: {
      algorithm: meta.algorithm,
      population: meta.trials,
      generations: meta.generations,
      restarts: meta.optimization.restarts,
      refinementRounds: meta.optimization.refinementRounds,
      elapsedMs: status.elapsedMs,
      hindsight: false,
    },
  };
});
await fs.mkdir(path.dirname(presetOutput), { recursive: true });
await fs.writeFile(presetOutput, `${JSON.stringify(presets, null, 2)}\n`);

console.log(JSON.stringify({
  input: path.relative(root, input),
  scoreVersion: meta.scoreVersion,
  warmStart: meta.warmStart,
  telemetryRecords: telemetry.length,
  stages: Object.fromEntries((["fit", "validation", "test"] as const).map((stage) => [
    stage,
    candidates.filter((entry) => entry.stage === stage).length,
  ])),
  finalists: selection.finalists.map((selected: { family: Family; id: string }) => {
    const validation = candidates.find((entry) => entry.stage === "validation" && entry.candidate.id === selected.id)!;
    const test = candidates.find((entry) => entry.stage === "test" && entry.candidate.id === selected.id)!;
    return { family: selected.family, id: selected.id, validation: validation.aggregate, test: test.aggregate };
  }),
  presetOutput: path.relative(root, presetOutput),
  elapsedMs: status.elapsedMs,
}, null, 2));
}

void main();

function storedParameters(value: Record<string, unknown>): VwKamaParameters {
  const number = (key: string, fallback: number): number =>
    typeof value[key] === "number" && Number.isFinite(value[key]) ? value[key] as number : fallback;
  const duration = (key: string, fallback: number): number =>
    typeof value[key] === "string" ? parseDuration(value[key] as string) : fallback;
  return {
    efficiencyMs: duration("efficiency", 60_000),
    efficiencyVolumeEmaMs: duration("efficiencyVolumeEma", duration("volume", 60_000)),
    efficiencyVolumePower: number("efficiencyVolumePower", 0),
    fastMs: duration("fast", 60_000),
    slowMs: duration("slow", 60_000),
    power: number("power", 1),
    volumeMs: duration("volume", 60_000),
    volumeCap: number("volumeCap", 1),
    volumePower: number("volumePower", 0),
    deadbandBpsHour: number("deadbandBpsHour", 0),
    deadbandMode: text(value.deadbandMode, ["flat", "hold", "hysteresis"], "hold"),
    hysteresisReleaseRatio: number("hysteresisReleaseRatio", 0.25),
    thresholdLookbackMs: duration("thresholdLookback", 3_600_000),
    thresholdNoiseMultiplier: number("thresholdNoiseMultiplier", 0),
    buyMaxFraction: number("buyMaxFraction", 1),
    sellMaxFraction: number("sellMaxFraction", 1),
    buySizingSigmaBpsHour: number("buySizingSigmaBpsHour", 1e12),
    sellSizingSigmaBpsHour: number("sellSizingSigmaBpsHour", 1e12),
    agreementMode: text(value.agreementMode, ["sizing", "confidence"], "sizing"),
    confirmationMix: number("confirmationMix", 0),
    confirmationMinQuality: number("confirmationMinQuality", 0),
    confirmationAccelerationLookbackMs: duration("confirmationAccelerationLookback", 3_600_000),
    confirmationDistanceLookbackMs: duration("confirmationDistanceLookback", 3_600_000),
    confirmationAccelerationWeight: number("confirmationAccelerationWeight", 1),
    confirmationDistanceWeight: number("confirmationDistanceWeight", 1),
    confirmationBias: number("confirmationBias", 0),
    confirmationEmaMs: duration("confirmationEma", 3_600_000),
    confirmationEmaThresholdBpsHour: number("confirmationEmaThresholdBpsHour", 0),
    confirmationEmaWeight: number("confirmationEmaWeight", 0),
    confirmationEmaGateStrength: number("confirmationEmaGateStrength", 0),
    confirmationRsiMs: duration("confirmationRsi", 840_000),
    confirmationRsiThreshold: number("confirmationRsiThreshold", 0),
    confirmationRsiWeight: number("confirmationRsiWeight", 0),
    confirmationDmiMs: duration("confirmationDmi", 840_000),
    confirmationDmiWeight: number("confirmationDmiWeight", 0),
    confirmationAdxThreshold: number("confirmationAdxThreshold", 20),
    meanReversionSuppressionThreshold: number("meanReversionSuppressionThreshold", 1),
    meanReversionEfficiencyMs: duration("meanReversionEfficiency", 3_600_000),
    meanReversionFastMs: duration("meanReversionFast", 900_000),
    meanReversionSlowMs: duration("meanReversionSlow", 3_600_000),
    meanReversionVolatilityMs: duration("meanReversionVolatility", 3_600_000),
    meanReversionReversalThreshold: number("meanReversionReversalThreshold", 0),
  };
}

function parseDuration(value: string): number {
  const match = /^(\d+(?:\.\d+)?)(ms|s|m|h|d|w)?$/.exec(value.trim());
  assert(match, `Invalid duration ${value}.`);
  return Math.round(Number(match[1]) * ({ ms: 1, s: 1_000, m: 60_000, h: 3_600_000,
    d: 86_400_000, w: 604_800_000 }[match[2] ?? "ms"]!));
}

function text<T extends string>(value: unknown, allowed: readonly T[], fallback: T): T {
  return typeof value === "string" && allowed.includes(value as T) ? value as T : fallback;
}

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

function assertNear(actual: number, expected: number, label: string): void {
  assert(Number.isFinite(actual) && Math.abs(actual - expected) <= 2e-6,
    `${label} mismatch: stored ${actual}, recomputed ${expected}.`);
}

function assertNullableNear(actual: number | null, expected: number | null, label: string): void {
  if (actual === null || expected === null) {
    assert(actual === expected, `${label} mismatch: stored ${actual}, recomputed ${expected}.`);
    return;
  }
  assertNear(actual, expected, label);
}

function nullableMedian(values: Array<number | string | null | undefined>): number | null {
  const present = values.filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  return present.length > 0 ? percentile(present, 0.5) : null;
}

function harmonic(left: number, right: number): number {
  return left + right > 0 ? 2 * left * right / (left + right) : 0;
}

function percentile(values: number[], quantile: number): number {
  assert(values.length > 0 && values.every(Number.isFinite), "Cannot aggregate invalid case metrics.");
  const sorted = values.slice().sort((left, right) => left - right);
  const position = (sorted.length - 1) * quantile;
  const lower = Math.floor(position);
  const fraction = position - lower;
  return sorted[lower]! * (1 - fraction) + (sorted[lower + 1] ?? sorted[lower]!) * fraction;
}
